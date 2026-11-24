"""Summaries of the scored tables: one series per method and arm, compared on shared queries with a paired bootstrap.

    python -m general_models_vs_plms.report --tables DIR --out DIR [--validation DIR ...]

A series is a method under an arm, such as Claude Fable 5.2 at low effort under `k256_cipher`. Every arm of a split
asks the same queries, and a split's bootstrap resamples are drawn once from its name, so any two series of a split
are compared on the same resampled queries. A series with several seeds is the mean of its seeds.

A route that asked only the first prompt of each task is read on those queries alone (`queries` is `first 100` in
each table), and its controls ran on one split per dataset, the `probe` group. `--validation` names the scored tables
of the validation prompts, on which a route's effort level was chosen.

The tables this writes are what the figures and the write-up read: `means.csv` (a series' mean over a group of
splits), `per_split.csv`, `paired.csv` (named comparisons with the paired difference), `split_differences.csv`,
`answered_only.csv` (each language model against the baselines on the queries it answered), `matches.csv` (answers
equal to the published value, beside a random deal of the same answers), `effort.csv`, `effort_validation.csv`,
`refusals.csv`, and the calibration, drift, completeness and cost tables.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from scipy.stats import rankdata

from . import plan
from .baselines import CONTROL_NAMES, PLM_NAMES
from .scoring import bootstrap_spearman, spearman
from .tasks import CHUNK_SIZE, name_code


DRAWS = 2000
POSTCUTOFF_DATASET = "fgfr2"
EMBEDDING_RIDGES = tuple(f"ridge_{name}" for name in PLM_NAMES)
AUGMENTED_RIDGES = tuple(f"ridge_onehot_plus_{name}" for name in PLM_NAMES)
ZERO_SHOT_METHODS = tuple(f"zero_shot_{name}" for name in PLM_NAMES)
RIDGE_METHODS = ("ridge_onehot", *EMBEDDING_RIDGES, *AUGMENTED_RIDGES)
CONTROL_RIDGES = tuple(f"ridge_{name}" for name in CONTROL_NAMES)
TRIVIAL_METHODS = ("nearest_neighbor", "mutation_count", "stratum_mean")
BASELINE_METHODS = (*RIDGE_METHODS, *ZERO_SHOT_METHODS, *CONTROL_RIDGES, *TRIVIAL_METHODS)
CONTROL_ARMS = ("k256_cipher", "k256_described", "k0_described", "k256_shuffled", "k256_flipped", "klarge")
# What a preference-tuned protein language model is read against, beside the best ridge of its group of splits.
TUNED_REFERENCES = ("ridge_esmc_600m", "ridge_onehot_plus_esmc_6b", "ridge_onehot", "zero_shot_esmc_600m", "zero_shot_esmc_6b")
# The arms of a tuned model that differ only in the order of their batches.
ORDER_ARMS = ("dpo", "dpo_seed1", "dpo_seed2")
# On the held-out rows a tuned model is read at these epochs of one setting, the learning rate and the weight of a
# ratio that the reported arm chose on most splits, so that no row it is scored on chose the checkpoint.
INSIDE_SETTING = (1e-5, 2.0)
INSIDE_EPOCHS = (1, 2, 5, 10, 20)
# A language model is compared on the queries it answered only where it answered at least this many of a split.
MIN_ANSWERED = 30
# An answer is near a published value if it is within this many units of the last digit the prompts show.
NEAR_UNITS = 10
# A split's values are discrete if fewer than this share of its query values are distinct at the shown precision.
DISTINCT_SHARE = 0.5


@dataclass(frozen=True)
class Series:
    """A method under an arm. `sign` is -1 to score a series against the reversed ranking."""

    label: str
    method: str
    arm: str
    sign: float = 1.0


def named(method: str, arm: str) -> Series:
    """A series labeled by its method and arm."""
    return Series(f"{method} {arm}", method, arm)


def load_predictions(tables: Path) -> pd.DataFrame:
    """Predictions joined with each task's split, arm and seed."""
    predictions = pd.read_csv(tables / "predictions.csv.gz", dtype={"stratum": str})
    metrics = pd.read_csv(tables / "metrics.csv")
    keys = metrics[["task_id", "split", "arm", "seed", "k"]].drop_duplicates()
    return predictions.merge(keys, on="task_id", how="left")


def first_rows(predictions: pd.DataFrame, count: int) -> pd.DataFrame:
    """Each split's first `count` queries in query order: the ones a route that asked one prompt a task was given."""
    kept = {(split, query) for split, rows in predictions.groupby("split") for query in sorted(rows["query_id"].unique())[:count]}
    return predictions[[key in kept for key in zip(predictions["split"], predictions["query_id"], strict=True)]]


def rest_rows(predictions: pd.DataFrame, count: int) -> pd.DataFrame:
    """Each split's queries after its first `count` in query order: the ones a route that asked one prompt a task never read."""
    kept = {(split, query) for split, rows in predictions.groupby("split") for query in sorted(rows["query_id"].unique())[count:]}
    return predictions[[key in kept for key in zip(predictions["split"], predictions["query_id"], strict=True)]]


def split_samples(split: str, queries: int, draws: int = DRAWS) -> np.ndarray:
    """A split's bootstrap resamples of its queries, the same for every series so that differences are paired."""
    return np.random.default_rng(name_code(split) % (2 ** 31)).integers(0, queries, size=(draws, queries))  # (r, q)


class Scores:
    """Predictions grouped by split, method and arm, with each series' bootstrap draws kept after first use."""

    def __init__(self, predictions: pd.DataFrame) -> None:
        self.groups = dict(tuple(predictions.groupby(["split", "method", "arm"], sort=False)))
        self.methods = set(predictions["method"])
        self.splits = sorted(predictions["split"].unique())
        self.kept: dict[tuple[object, ...], tuple[float, np.ndarray]] = {}

    def add(self, predictions: pd.DataFrame) -> None:
        """Take in more series, such as a blend of two that are already here."""
        self.groups.update(dict(tuple(predictions.groupby(["split", "method", "arm"], sort=False))))
        self.methods |= set(predictions["method"])

    def matrix(self, split: str, series: Series, first_queries: int | None = None) -> tuple[np.ndarray, np.ndarray] | None:
        """Measured values and one row of predictions per seed, in query order; None if the series did not run.

        `first_queries` keeps the first that many queries, for a comparison with an arm that asked only those.
        """
        rows = self.groups.get((split, series.method, series.arm))
        if rows is None:
            return None

        wide = rows.pivot(index="query_id", columns="seed", values="prediction").sort_index()
        truth = rows.drop_duplicates("query_id").set_index("query_id")["truth"].sort_index()
        assert wide.index.equals(truth.index) and not wide.isna().any().any(), f"{series.label} on {split} lacks queries"
        if first_queries is not None:
            wide, truth = wide.iloc[:first_queries], truth.iloc[:first_queries]
        return truth.to_numpy(), series.sign * wide.to_numpy().T  # (q,), (s, q)

    def sampled(self, split: str, series: Series, first_queries: int | None = None) -> tuple[float, np.ndarray] | None:
        """A series' Spearman on a split as the mean of its seeds, and that mean on each bootstrap resample."""
        key = (split, series.method, series.arm, series.sign, first_queries)
        if key not in self.kept:
            found = self.matrix(split, series, first_queries)
            if found is None:
                return None
            truth, values = found  # (q,), (s, q)
            draws = bootstrap_spearman(truth, values, DRAWS, seed=name_code(split) % (2 ** 31)).mean(axis=1)  # (r,)
            self.kept[key] = (float(np.mean([spearman(truth, row) for row in values])), draws)
        return self.kept[key]  # the score, and its draws (r,)


def compare(scores: Scores, splits: list[str], series: list[Series], first_queries: int | None = None) -> dict[str, object]:
    """Per-split Spearman of each series, the mean over splits, and bootstrap draws of that mean.

    A split enters only if every series has predictions for it, so the means compare like with like.
    """
    usable, per_split, draws = [], {}, []
    for split in splits:
        found = [scores.sampled(split, one, first_queries) for one in series]
        if any(item is None for item in found):
            continue
        per_split[split] = [item[0] for item in found]
        draws.append(np.stack([item[1] for item in found], axis=1))  # (r, n_series)
        usable.append(split)

    labels = [one.label for one in series]
    if not usable:
        return {"splits": [], "series": labels}

    macro_draws = np.mean(draws, axis=0)  # (r, n_series)
    table = pd.DataFrame(per_split, index=labels).T
    return {
        "splits": usable, "series": labels, "per_split": table,
        "mean": table.mean(axis=0).to_numpy(), "draws": macro_draws,
        "low": np.quantile(macro_draws, 0.025, axis=0), "high": np.quantile(macro_draws, 0.975, axis=0),
    }


def difference(comparison: dict[str, object], left: str, right: str) -> dict[str, float]:
    """Mean of `left` minus `right` over the shared splits, with its paired 95% interval and the share of draws above 0."""
    labels = comparison["series"]
    i, j = labels.index(left), labels.index(right)
    gap = comparison["draws"][:, i] - comparison["draws"][:, j]  # (r,)
    return {"difference": float(comparison["mean"][i] - comparison["mean"][j]),
            "low": float(np.quantile(gap, 0.025)), "high": float(np.quantile(gap, 0.975)),
            "share_above_zero": float((gap > 0).mean())}


def paired(scores: Scores, splits: list[str], left: Series, right: Series,
           first_queries: int | None = None) -> dict[str, object] | None:
    """Both series' means over the splits they share, and the paired difference of left minus right."""
    comparison = compare(scores, splits, [left, right], first_queries)
    if not comparison["splits"]:
        return None
    return {"left": left.label, "right": right.label, "splits": len(comparison["splits"]),
            "left_mean": float(comparison["mean"][0]), "left_low": float(comparison["low"][0]),
            "left_high": float(comparison["high"][0]), "right_mean": float(comparison["mean"][1]),
            "right_low": float(comparison["low"][1]), "right_high": float(comparison["high"][1]),
            **difference(comparison, left.label, right.label)}


def series_table(scores: Scores) -> pd.DataFrame:
    """Every method under every arm on every split it ran: Spearman as the mean of its seeds, with its interval."""
    rows = []
    for split, method, arm in sorted(scores.groups):
        value, draws = scores.sampled(split, Series(method, method, arm))
        rows.append({"split": split, "arm": arm, "method": method, "spearman": value,
                     "low": float(np.quantile(draws, 0.025)), "high": float(np.quantile(draws, 0.975))})
    return pd.DataFrame(rows)


def means_table(scores: Scores, groups: dict[str, list[str]], first_queries: int | None = None) -> pd.DataFrame:
    """Every method under every arm: its mean Spearman over each group's splits that it ran on, with its interval."""
    rows = []
    for method, arm in sorted({(method, arm) for _, method, arm in scores.groups}):
        for group, splits in groups.items():
            comparison = compare(scores, splits, [Series(method, method, arm)], first_queries)
            if comparison["splits"]:
                rows.append({"group": group, "arm": arm, "method": method, "splits": len(comparison["splits"]),
                             "group_splits": len(splits), "mean": float(comparison["mean"][0]),
                             "low": float(comparison["low"][0]), "high": float(comparison["high"][0])})
    return pd.DataFrame(rows)


def split_differences(scores: Scores, splits: list[str], left: Series, right: Series) -> list[dict[str, object]]:
    """Per split that both series ran: their scores and the paired difference of left minus right with its interval."""
    rows = []
    for split in splits:
        one, other = scores.sampled(split, left), scores.sampled(split, right)
        if one is None or other is None:
            continue
        gap = one[1] - other[1]  # (r,)
        rows.append({"split": split, "left": left.label, "right": right.label, "left_spearman": one[0],
                     "right_spearman": other[0], "difference": one[0] - other[0],
                     "low": float(np.quantile(gap, 0.025)), "high": float(np.quantile(gap, 0.975))})
    return rows


def rank_average(predictions: pd.DataFrame, left: str, right: str, name: str) -> pd.DataFrame:
    """Rows of a method that averages two methods' ranks within each task, on the tasks both ran."""
    keys = ["task_id", "query_id"]
    other = predictions.loc[predictions["method"] == right, [*keys, "prediction"]].rename(columns={"prediction": "other"})
    merged = predictions[predictions["method"] == left].merge(other, on=keys)
    ranks = merged.groupby("task_id")[["prediction", "other"]].rank()
    return merged.assign(prediction=ranks.mean(axis=1), method=name).drop(columns="other")


def row_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pearson correlation of matching rows, 0 where a row is constant."""
    # left, right: (r, q)
    left = left - left.mean(axis=1, keepdims=True)  # (r, q)
    right = right - right.mean(axis=1, keepdims=True)  # (r, q)
    norms = np.sqrt((left ** 2).sum(axis=1) * (right ** 2).sum(axis=1))  # (r,)
    return np.divide((left * right).sum(axis=1), norms, out=np.zeros(len(norms)), where=norms > 0)  # (r,)


def partial_spearman(truth: np.ndarray, predictions: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Spearman of measured values and predictions once a control method's ranks are taken out of both, per row."""
    # truth, predictions, control: (r, q), one resample of the queries per row
    ranks = [rankdata(values, axis=1) for values in (truth, predictions, control)]  # each (r, q)
    with_truth = row_correlation(ranks[0], ranks[1])  # (r,)
    truth_control = row_correlation(ranks[0], ranks[2])  # (r,)
    with_control = row_correlation(ranks[1], ranks[2])  # (r,)
    scale = np.sqrt((1.0 - truth_control ** 2) * (1.0 - with_control ** 2))  # (r,)
    return np.divide(with_truth - truth_control * with_control, scale, out=np.zeros(len(scale)), where=scale > 0)  # (r,)


def partial_comparison(scores: Scores, splits: list[str], series: Series, control: Series) -> dict[str, object] | None:
    """Mean over splits of a series' partial Spearman given a control, on seed 0, with a 95% bootstrap interval."""
    per_split, sampled = [], []
    for split in splits:
        found, held = scores.matrix(split, series), scores.matrix(split, control)
        if found is None or held is None:
            continue
        truth, values, other = found[0], found[1][0], held[1][0]  # each (q,)
        per_split.append(float(partial_spearman(truth[None, :], values[None, :], other[None, :])[0]))
        samples = split_samples(split, len(truth))  # (r, q)
        sampled.append(partial_spearman(truth[samples], values[samples], other[samples]))
    if not per_split:
        return None
    means = np.mean(sampled, axis=0)  # (r,)
    return {"series": series.label, "given": control.label, "partial_spearman": float(np.mean(per_split)),
            "low": float(np.quantile(means, 0.025)), "high": float(np.quantile(means, 0.975)), "splits": len(per_split)}


def answered_only(predictions: pd.DataFrame, splits: list[str], model: Series, others: list[Series]) -> list[dict[str, object]]:
    """A language model against other series on the queries it answered, over the splits where it answered enough.

    The table's main scores give an unanswered query the median of the shown values, so a refused prompt lowers a
    score. Here the model and each other series are scored on the answered queries alone, which asks how good the
    answers are and leaves out how many there are.
    """
    seed_zero = predictions[(predictions["seed"] == 0) & predictions["split"].isin(splits)]
    own = seed_zero[(seed_zero["method"] == model.method) & (seed_zero["arm"] == model.arm)]
    kept = own.loc[own["answered"], ["split", "query_id"]]
    counts = kept.groupby("split").size()
    usable = [split for split in splits if counts.get(split, 0) >= MIN_ANSWERED]
    # A model that answered every query has nothing to add here: its main scores already are its answered-only scores.
    if not usable or own["answered"].all():
        return []

    scores = Scores(seed_zero.merge(kept, on=["split", "query_id"]))
    rows = []
    for other in others:
        found = paired(scores, usable, model, other)
        if found is not None:
            rows.append({"splits_of": len(splits), "queries": int(counts[usable].sum()), "queries_asked": len(own), **found})
    return rows


def dealt_matches(truth: np.ndarray, answer: np.ndarray, deals: np.ndarray, within: int) -> tuple[int, np.ndarray]:
    """Answers within `within` units of their own query's value, and that count on each random deal of the answers."""
    # truth, answer: (q,) in units of the last shown digit; deals: (r, q), each row an order of the answers
    return int((np.abs(truth - answer) <= within).sum()), (np.abs(truth[None, :] - answer[deals]) <= within).sum(axis=1)  # (r,)


def match_tables(predictions: pd.DataFrame, decimals: dict[str, int], groups: dict[str, list[str]],
                 draws: int = DRAWS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Answers that equal the published value at the precision the prompts show, per split and pooled over each group.

    A model that has memorized a dataset's values can write them down digit for digit. `exact` counts answers equal
    to their query's value, and `near` those within `NEAR_UNITS` of the last shown digit. Each `_dealt` column is the
    mean of the same count when the answers are dealt to the queries at random, which is what the spread of the
    values alone gives, and each `_p` is the share of deals that reach the observed count.
    """
    groups_of = {split: [name for name, splits in groups.items() if split in splits] for split in predictions["split"].unique()}
    deals_of: dict[tuple[str, int], np.ndarray] = {}
    per_split, pooled = [], {}
    for (split, method, arm), rows in predictions[predictions["answered"]].groupby(["split", "method", "arm"]):
        scale = 10.0 ** decimals[split]
        truth = np.rint(rows["truth"].to_numpy() * scale).astype(np.int64)  # (q,)
        answer = np.rint(np.clip(rows["prediction"].to_numpy(), -1e12, 1e12) * scale).astype(np.int64)  # (q,)
        if (split, len(truth)) not in deals_of:
            generator = np.random.default_rng(name_code(split) % (2 ** 31))
            deals_of[split, len(truth)] = generator.random((draws, len(truth))).argsort(axis=1)  # (r, q)
        deals = deals_of[split, len(truth)]
        exact, exact_dealt = dealt_matches(truth, answer, deals, 0)
        near, near_dealt = dealt_matches(truth, answer, deals, NEAR_UNITS)
        per_split.append({"split": split, "method": method, "arm": arm, "answered": len(truth), "exact": exact,
                          "exact_dealt": float(exact_dealt.mean()), "near": near, "near_dealt": float(near_dealt.mean())})
        for group in groups_of[split]:
            total = pooled.setdefault((group, method, arm), {"splits": 0, "answered": 0, "exact": 0, "near": 0,
                                                             "exact_draws": np.zeros(draws), "near_draws": np.zeros(draws)})
            total["splits"] += 1
            total["answered"] += len(truth)
            total["exact"] += exact
            total["near"] += near
            total["exact_draws"] += exact_dealt
            total["near_draws"] += near_dealt

    pooled_rows = []
    for (group, method, arm), total in pooled.items():
        row = {"group": group, "method": method, "arm": arm, "splits": total["splits"], "answered": total["answered"]}
        for kind in ("exact", "near"):
            dealt = total[f"{kind}_draws"]  # (r,)
            row.update({kind: total[kind], f"{kind}_dealt": float(dealt.mean()),
                        f"{kind}_p": float((1 + (dealt >= total[kind]).sum()) / (draws + 1))})
        pooled_rows.append(row)
    return pd.DataFrame(per_split), pd.DataFrame(pooled_rows)


def match_groups(predictions: pd.DataFrame, decimals: dict[str, int], groups: dict[str, list[str]]) -> dict[str, list[str]]:
    """The groups of splits that matches are pooled over, with the benchmark divided by how its values repeat.

    Where fewer than half of a split's query values are distinct at the shown precision (activity in three levels, a
    wavelength in whole nanometers), an answer equals a value often by chance and a method that predicts the level
    matches it without having read it. Those splits are pooled apart from the splits with continuous values.
    """
    truths = predictions.drop_duplicates(["split", "query_id"])
    discrete = []
    for split, rows in truths.groupby("split"):
        shown = np.rint(rows["truth"].to_numpy() * 10.0 ** decimals[split])  # (q,)
        if len(np.unique(shown)) < DISTINCT_SHARE * len(shown):
            discrete.append(split)
    divided = dict(groups)
    for name in ("flip2", "probe"):
        if name in groups:
            divided[f"{name}_continuous"] = [split for split in groups[name] if split not in discrete]
            divided[f"{name}_discrete"] = [split for split in groups[name] if split in discrete]
    return divided


def effort_table(calls: pd.DataFrame, groups: dict[str, list[str]], arm: str = "k256") -> pd.DataFrame:
    """Per language model and group of splits: what the first prompt of each task cost under an arm, in tokens and seconds."""
    first = with_method(calls)
    first = first[(first["arm"] == arm) & (first["seed"] == 0) & first["record_id"].str.endswith("__c0")]
    tables = []
    for group, splits in groups.items():
        spent = first[first["split"].isin(splits)].groupby("method").agg(
            prompts=("record_id", "count"), refused=("refused", "sum"), answered=("answered", "sum"), queries=("queries", "sum"),
            median_output_tokens=("output_tokens", "median"), median_thinking_tokens=("thinking_tokens", "median"),
            median_seconds=("elapsed_seconds", "median")).reset_index()
        tables.append(spent.assign(group=group))
    return pd.concat(tables, ignore_index=True)


def validation_table(folders: list[Path]) -> pd.DataFrame:
    """Per language model and validation prompt: its Spearman and what the prompt cost in tokens and in seconds.

    A route's effort level is chosen on these rows, so the table holds every level that was tried, a level that
    answered only some of the prompts included.
    """
    tables = []
    for folder in folders:
        metrics = pd.read_csv(folder / "metrics.csv")
        spent = with_method(pd.read_csv(folder / "calls.csv")).groupby(["method", "task_id"]).agg(
            output_tokens=("output_tokens", "sum"), thinking_tokens=("thinking_tokens", "sum"),
            seconds=("elapsed_seconds", "sum"), answered=("answered", "sum")).reset_index()
        tables.append(metrics[["method", "task_id", "split", "arm", "queries", "spearman"]].merge(spent, on=["method", "task_id"]))
    return pd.concat(tables, ignore_index=True)


def calibration(predictions: pd.DataFrame, arm: str) -> pd.DataFrame:
    """Per method under an arm, the mean over tasks of Pearson correlation and of error in units of the label spread."""
    rows = []
    for (method, task_id), group in predictions[predictions["arm"] == arm].groupby(["method", "task_id"]):
        truth, values = group["truth"].to_numpy(), group["prediction"].to_numpy()  # (q,), (q,)
        spread = truth.std()
        pearson = float(np.corrcoef(truth, values)[0, 1]) if spread > 0 and values.std() > 0 else 0.0
        rows.append({"method": method, "task_id": task_id, "pearson": pearson,
                     "rmse_over_sd": float(np.sqrt(np.mean((truth - values) ** 2)) / spread),
                     "bias_over_sd": float((values.mean() - truth.mean()) / spread),
                     "spread_ratio": float(values.std() / spread)})
    return pd.DataFrame(rows).groupby("method")[["pearson", "rmse_over_sd", "bias_over_sd", "spread_ratio"]].mean().reset_index()


def chunk_drift(predictions: pd.DataFrame, arm: str, chunk_size: int = CHUNK_SIZE) -> pd.DataFrame:
    """Pooled Spearman beside the mean Spearman inside each prompt's 100 queries, per method under an arm.

    A language model answers 100 queries a prompt, so a shift in its scale between prompts lowers the pooled score and
    leaves the score inside a prompt alone. A fitted baseline has no such shift.
    """
    rows = []
    subset = predictions[predictions["arm"] == arm]
    chunk = (subset["query_id"].str[1:].astype(int) - 1) // chunk_size
    for (method, task_id), group in subset.assign(chunk=chunk).groupby(["method", "task_id"]):
        inside = [spearman(part["truth"].to_numpy(), part["prediction"].to_numpy()) for _, part in group.groupby("chunk")
                  if len(part) >= 20]
        rows.append({"method": method, "task_id": task_id, "pooled": spearman(group["truth"].to_numpy(), group["prediction"].to_numpy()),
                     "inside_prompts": float(np.mean(inside))})
    return pd.DataFrame(rows).groupby("method")[["pooled", "inside_prompts"]].mean().reset_index()


def with_method(calls: pd.DataFrame) -> pd.DataFrame:
    """Call records with the method name the other tables use, and flags for what went wrong."""
    return calls.assign(method=calls["model_requested"] + "@" + calls["effort"],
                        refused=calls["stop_reason"] == "refusal", truncated=calls["stop_reason"] == "max_tokens",
                        other_model=calls["model_served"] != calls["model_requested"],
                        nothing_read=(calls["stop_reason"] != "refusal") & (calls["answered"] == 0))


def completeness(calls: pd.DataFrame, metrics: pd.DataFrame, chunk_size: int = CHUNK_SIZE) -> pd.DataFrame:
    """Per language model and arm: tasks run, prompts sent against prompts owed, refusals, and queries answered."""
    owed = metrics[metrics["method"].str.contains("@")].assign(owed=lambda frame: np.ceil(frame["queries"] / chunk_size))
    owed = owed.groupby(["method", "arm"]).agg(tasks=("task_id", "nunique"), prompts_owed=("owed", "sum"),
                                               answered_fraction=("answered_fraction", "mean"))
    sent = with_method(calls).groupby(["method", "arm"]).agg(
        prompts_sent=("record_id", "count"), refused=("refused", "sum"), truncated=("truncated", "sum"),
        other_model=("other_model", "sum"), nothing_read=("nothing_read", "sum"))
    return owed.join(sent).reset_index()


def refusal_table(calls: pd.DataFrame) -> pd.DataFrame:
    """Per language model, arm and split: prompts sent and prompts the model refused, with the stated category."""
    flagged = with_method(calls)
    table = flagged.groupby(["method", "arm", "split"]).agg(prompts=("record_id", "count"), refused=("refused", "sum"),
                                                            nothing_read=("nothing_read", "sum")).reset_index()
    categories = (flagged[flagged["refused"]].groupby(["method", "arm", "split"])["stop_category"]
                  .agg(lambda values: ",".join(sorted(set(values.astype(str))))).rename("categories").reset_index())
    return table.merge(categories, on=["method", "arm", "split"], how="left")


def cost_table(calls: pd.DataFrame) -> pd.DataFrame:
    """Calls, tokens, refusals and spend by model and effort."""
    grouped = with_method(calls).groupby("method").agg(
        calls=("record_id", "count"), queries=("queries", "sum"), answered=("answered", "sum"),
        refused=("refused", "sum"), truncated=("truncated", "sum"), other_model=("other_model", "sum"),
        input_tokens=("input_tokens", "sum"), cache_write_tokens=("cache_write_tokens", "sum"),
        cache_read_tokens=("cache_read_tokens", "sum"), output_tokens=("output_tokens", "sum"),
        thinking_tokens=("thinking_tokens", "sum"), cost_usd=("cost_usd", "sum"),
        median_seconds=("elapsed_seconds", "median"))
    grouped["usd_per_1000_predictions"] = 1000 * grouped["cost_usd"] / grouped["queries"]
    return grouped.reset_index()


def group_splits(splits: list[str]) -> dict[str, list[str]]:
    """The benchmark's splits and the splits of the dataset published after the models' training data ends."""
    groups = {"flip2": [split for split in splits if not split.startswith(POSTCUTOFF_DATASET)],
              "postcutoff": [split for split in splits if split.startswith(POSTCUTOFF_DATASET)]}
    return {name: members for name, members in groups.items() if members}


def best_ridge(scores: Scores, splits: list[str], arm: str = "k256") -> str:
    """The ridge baseline with the highest mean Spearman over the splits, a choice that favors the baselines."""
    ridges = [method for method in RIDGE_METHODS if method in scores.methods]
    comparison = compare(scores, splits, [Series(method, method, arm) for method in ridges])
    return ridges[int(np.argmax(comparison["mean"]))]


def questions(scores: Scores, models: list[str], best: str) -> list[tuple[str, Series, Series]]:
    """Every paired comparison the write-up may state, as a question with its two series."""
    baselines = [method for method in BASELINE_METHODS if method in scores.methods]
    asked = []
    for model in models:
        plain = named(model, "k256")
        asked += [("256 examples against a baseline on the same examples", plain, named(method, "k256")) for method in baselines]
        asked += [(f"256 examples against the {arm} arm of the same model", plain, named(model, arm)) for arm in CONTROL_ARMS]
        asked += [("enciphered residues against a baseline on the same examples", named(model, "k256_cipher"), named(method, "k256"))
                  for method in ("ridge_onehot", "ridge_composition", "nearest_neighbor", best) if method in scores.methods]
        asked += [("no examples against a zero-shot score", named(model, "k0_described"), named(method, "k256"))
                  for method in ZERO_SHOT_METHODS if method in scores.methods]
        asked += [("asked for the published values against asked for a score, no examples", named(model, "k0_recall"),
                   named(model, "k0_described"))]
        asked += [("asked for the published values against a zero-shot score", named(model, "k0_recall"), named(method, "k256"))
                  for method in ZERO_SHOT_METHODS if method in scores.methods]
        asked += [("up to 2,048 examples against a baseline on the same examples", named(model, "klarge"), named(method, "klarge"))
                  for method in baselines]
        asked += [("up to 2,048 examples against a ridge fit on every training row", named(model, "klarge"), named(method, "klarge"))
                  for method in sorted(scores.methods) if method.endswith("_full")]
        asked += [(f"{arm}: the model against a baseline on the same examples", named(model, arm), named(method, arm))
                  for arm in ("k16", "k64") for method in baselines]
        blend = f"{model}+{best}"
        asked += [("the rank average of the model and the best ridge against one alone", named(blend, "k256"), named(other, "k256"))
                  for other in (model, best)]
        asked += [(f"{arm}: one language model against another", named(model, arm), named(other, arm))
                  for arm in ("k256", "k256_cipher", "k0_described", "klarge") for other in models if other != model]
    return asked


def baseline_questions(scores: Scores) -> list[tuple[str, Series, Series]]:
    """Paired comparisons among the baselines at 256 examples: protein models with each other, features with controls."""
    uses = (("ridge on the embedding", "ridge_{}"), ("zero-shot score", "zero_shot_{}"),
            ("ridge on one-hot with the score", "ridge_onehot_plus_{}"))
    asked = []
    for index, left in enumerate(PLM_NAMES):
        asked += [(f"{use}: one protein language model against another", named(pattern.format(left), "k256"),
                   named(pattern.format(right), "k256")) for right in PLM_NAMES[index + 1:] for use, pattern in uses]
        asked.append(("ridge on an embedding against its model's zero-shot score", named(f"ridge_{left}", "k256"),
                      named(f"zero_shot_{left}", "k256")))
    for control in CONTROL_RIDGES:
        # A control made from one model's embedding is compared with that model; the others with both ESMC models.
        owners = [f"ridge_{name}" for name in PLM_NAMES if control.startswith(f"ridge_{name}_")] or ["ridge_esmc_300m", "ridge_esmc_600m"]
        asked += [("features against a feature control", named(other, "k256"), named(control, "k256"))
                  for other in (*owners, "ridge_onehot")]
    asked += [("ridge on an embedding against ridge on one-hot", named(method, "k256"), named("ridge_onehot", "k256"))
              for method in EMBEDDING_RIDGES]
    return [(question, left, right) for question, left, right in asked
            if left.method in scores.methods and right.method in scores.methods]


def tuned_models(scores: Scores) -> list[str]:
    """The preference-tuned protein language models that have a scored primary arm."""
    return [model for model in plan.PREFERENCE_MODELS if f"{model}_dpo" in scores.methods]


def tuned_arm_of(tuned: str) -> str:
    """The arm of the tasks a tuned model or a variant of its tuning ran on."""
    return plan.PREFERENCE_TASK_ARMS.get(tuned, "k256")


def tuned_blends(predictions: pd.DataFrame, tuned: str, others: list[str]) -> list[pd.DataFrame]:
    """The rank averages of a tuned model's primary arm with each of the other methods that the predictions hold."""
    held = set(predictions["method"])
    return [rank_average(predictions, f"{tuned}_dpo", other, f"{tuned}_dpo+{other}") for other in others if other in held]


def tuned_orders(predictions: pd.DataFrame, tuned: str) -> pd.DataFrame:
    """Rows of a method that averages the ranks of a tuned model's primary arm over its batch orders, within each task.

    Empty unless every batch order ran. A task one of them missed is left out.
    """
    names = [f"{tuned}_{arm}" for arm in ORDER_ARMS]
    keys = ["task_id", "query_id"]
    rows = predictions[predictions["method"].isin(names)]
    if set(rows["method"]) != set(names):
        return predictions.iloc[:0]
    ranks = rows.assign(rank=rows.groupby(["task_id", "method"])["prediction"].rank())
    mean = ranks.groupby(keys)["rank"].agg(["mean", "size"]).reset_index()
    mean = mean[mean["size"] == len(names)]  # the queries every order scored
    merged = rows[rows["method"] == names[0]].merge(mean[[*keys, "mean"]], on=keys)
    return merged.assign(prediction=merged["mean"], method=f"{tuned}_dpo+orders").drop(columns="mean")


def tuned_questions(scores: Scores, models: list[str], best: str) -> list[tuple[str, Series, Series]]:
    """Paired comparisons of a preference-tuned protein language model on the examples of its tasks.

    Each tuned arm is set against the same model untuned, the same model under a ridge, the other loss, the same
    loss on permuted values, the fitted baselines on the same examples, and each language model. A variant of the
    tuning is also set against the primary arm of the first tuned model, and that arm's rank average with another
    method against each of its two parts. The average of its batch orders is set against one order, the best ridge
    and each language model.
    """
    asked = []
    every = tuned_models(scores)
    for tuned in every[:1]:
        for blend in sorted(method for method in scores.methods if method.startswith(f"{tuned}_dpo+")):
            asked += [("the rank average of a tuned model and another method against one alone", named(blend, "k256"), named(part, "k256"))
                      for part in (f"{tuned}_dpo", blend.split("+", 1)[1])]
        asked += [("the rank average of a tuned model's batch orders against a method on the same examples",
                   named(f"{tuned}_dpo+orders", "k256"), named(other, "k256")) for other in (best, *models)]
    for tuned in every:
        rows = tuned_arm_of(tuned)
        arm = {name: named(f"{tuned}_{name}", rows) for name in plan.PREFERENCE_ARMS}
        if tuned != every[0]:
            asked += [("a variant of the tuning, or another model, against the primary arm", arm[name], named(f"{every[0]}_dpo", "k256"))
                      for name in ("dpo", "zero", "ridge")]
            asked.append(("the same with no examples", arm["zero"], named(f"{every[0]}_zero", "k256")))
            asked.append(("a variant of the tuning, or another model, against the first model with no examples",
                          arm["dpo"], named(f"{every[0]}_zero", "k256")))
        asked += [("preference tuning against the same model with no examples", arm[name], arm["zero"]) for name in ("dpo", "bt")]
        asked += [("preference tuning against a ridge on the same model's hidden states", arm[name], arm["ridge"]) for name in ("dpo", "bt")]
        asked.append(("the ranked preference loss against the Bradley-Terry loss", arm["dpo"], arm["bt"]))
        asked.append(("preference tuning against the same loss on permuted values", arm["dpo"], arm["dpo_shuffled"]))
        # Tuning on permuted values starts from a model whose score may already rank the queries, so its null is that
        # score and not zero: permuted values can lower it and cannot raise it except by chance.
        asked.append(("preference tuning on permuted values against the same model with no examples", arm["dpo_shuffled"], arm["zero"]))
        asked += [("preference tuning under another batch order", arm["dpo"], arm[name]) for name in ("dpo_seed1", "dpo_seed2")]
        asked.append(("a ridge on the model's hidden states against the same model with no examples", arm["ridge"], arm["zero"]))
        for other in dict.fromkeys((best, *TUNED_REFERENCES)):
            asked += [("a preference-tuned protein language model against a baseline on the same examples", arm[name], named(other, rows))
                      for name in plan.PREFERENCE_ARMS]
        # The tuned arm is the left side of every question, as it is in the split differences.
        asked += [("a protein language model tuned on 256 examples against a language model that reads them in its prompt",
                   arm[name], named(model, "k256")) for model in models for name in plan.PREFERENCE_ARMS]
    return [(question, left, right) for question, left, right in asked
            if left.method in scores.methods and right.method in scores.methods]


def tuned_name(record: dict[str, object]) -> str:
    """The model a tuning record's methods are named after: its checkpoint, or the named variant of its tuning."""
    repository, _ = record["model"]
    return record.get("method_prefix") or repository.rsplit("/", 1)[1].replace("-", "_")


def tuning_table(records: list[dict[str, object]]) -> pd.DataFrame:
    """Per tuned model, split and arm: the recipe and the epoch chosen on the held-out rows, and the score there.

    `records` are the ones `preference_tuning` writes, one a split. An arm that failed has its reason and no recipe.
    """
    rows = []
    for record in records:
        shared = {"model": tuned_name(record), "revision": record["model"][1], "split": record["split"],
                  "fit_rows": record["fit_rows"], "held_out_rows": record["held_out_rows"],
                  "held_out_spearman_untuned": record["held_out_spearman_untuned"]}
        for arm, found in record["arms"].items():
            rows.append({**shared, "arm": arm, **found["recipe"], "epoch": found["epoch"],
                         "held_out_spearman": found["held_out_spearman"], "seconds": found["seconds"], "failed": ""})
        rows += [{**shared, "arm": arm, "failed": reason} for arm, reason in record["failed"].items()]
    return pd.DataFrame(rows)


def trace_table(records: list[dict[str, object]], predictions: pd.DataFrame) -> pd.DataFrame:
    """Per tuned model, split, setting and epoch of a traced arm: the held-out Spearman and the Spearman on the queries.

    `records` are the ones `preference_tuning --trace` writes, whose curves hold the scores of a task's leading
    queries after every epoch; a record that traced no query is passed over. Their measured values come from the
    scored `predictions`, in query order. `chosen` marks the checkpoint the held-out rows chose, the one a run
    reports. Every other row can only say what another choice would have scored, since comparing rows reads the
    measured values of the queries.
    """
    truths = predictions.drop_duplicates(["task_id", "query_id"]).sort_values("query_id")
    rows = []
    for record in records:
        traced = record.get("traced_queries", 0)
        if not traced:
            continue
        truth = truths.loc[truths["task_id"] == record["task_id"], "truth"].to_numpy()[:traced]  # (q_traced,)
        assert len(truth) == traced, f"{record['task_id']} traced {traced} queries and the tables hold {len(truth)}"
        for arm, found in record["arms"].items():
            for curve in found["curves"]:
                kept = all(curve[key] == found["recipe"][key] for key in ("learning_rate", "beta"))
                for epoch, (held, scores) in enumerate(zip(curve["held_out_spearman"], curve["query_scores"], strict=True), start=1):
                    rows.append({"model": tuned_name(record), "split": record["split"], "arm": arm,
                                 "learning_rate": curve["learning_rate"], "beta": curve["beta"], "epoch": epoch,
                                 "held_out_spearman": held, "spearman": spearman(truth, np.array(scores)), "queries": traced,
                                 "chosen": kept and epoch == found["epoch"]})
    return pd.DataFrame(rows)


def inside_table(inside: list[dict[str, object]], records: list[dict[str, object]]) -> pd.DataFrame:
    """Per split, every method's Spearman on the rows the tuning holds out for selection.

    `inside` is what `preference_tuning --inside` writes: each baseline fit on the rows the tuning fits on. From the
    tuning `records` come the same rows under a tuned model: untuned (`<model>_zero`), and after each epoch of each
    setting of its ranked-loss arm, with `chosen` on the checkpoint the run kept. A record of another arm of the
    tasks holds out other rows and is passed over.
    """
    rows = [{"split": found["split"], "method": method, "held_out_spearman": score, "chosen": False,
             "fit_rows": found["fit_rows"], "held_out_rows": found["held_out_rows"]}
            for found in inside for method, score in found["held_out_spearman"].items()]
    for record in records:
        if record.get("arm", "k256") != "k256" or "dpo" not in record["arms"]:
            continue
        shared = {"split": record["split"], "fit_rows": record["fit_rows"], "held_out_rows": record["held_out_rows"]}
        name, found = tuned_name(record), record["arms"]["dpo"]
        rows.append({**shared, "method": f"{name}_zero", "held_out_spearman": record["held_out_spearman_untuned"], "chosen": False})
        for curve in found["curves"]:
            kept = all(curve[key] == found["recipe"][key] for key in ("learning_rate", "beta"))
            rows += [{**shared, "method": f"{name}_dpo", "learning_rate": curve["learning_rate"], "beta": curve["beta"], "epoch": epoch,
                      "held_out_spearman": held, "chosen": kept and epoch == found["epoch"]}
                     for epoch, held in enumerate(curve["held_out_spearman"], start=1)]
    return pd.DataFrame(rows)


def inside_differences(table: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    """Over a group's splits, a tuned model on its held-out rows less each baseline fit on the rows it was tuned on.

    The tuned model is taken untuned, after `INSIDE_EPOCHS` of the setting `INSIDE_SETTING`, which no row chose, and
    at the checkpoint the held-out rows chose, whose score those rows inflate. The interval resamples the splits,
    the same resamples for every row of a group: a run keeps its Spearman on the held-out rows and not its score of
    each row, so the rows cannot be resampled.
    """
    tuned = table[table["method"].str.endswith("_dpo") & table["epoch"].notna()]
    baselines = table[table["epoch"].isna() & ~table["method"].isin({f"{model}_zero" for model in plan.PREFERENCE_MODELS})]
    wide = baselines.pivot(index="split", columns="method", values="held_out_spearman")  # (splits, baselines)
    rows = []
    for model, part in tuned.groupby(tuned["method"].str.removesuffix("_dpo")):
        rate, beta = INSIDE_SETTING
        fixed = part[np.isclose(part["learning_rate"], rate) & np.isclose(part["beta"], beta)]
        untuned = table[table["method"] == f"{model}_zero"].set_index("split")["held_out_spearman"]
        series = {"untuned": untuned, "chosen": part[part["chosen"]].set_index("split")["held_out_spearman"],
                  **{f"epoch {epoch}": fixed[fixed["epoch"] == epoch].set_index("split")["held_out_spearman"] for epoch in INSIDE_EPOCHS}}
        for group, splits in groups.items():
            for label, scores in series.items():
                for baseline in wide.columns:
                    # A baseline that needs a parent sequence has no score on a split without one.
                    shared = [split for split in splits if split in scores.index and split in wide.index and pd.notna(wide.at[split, baseline])]
                    if not shared:
                        continue
                    draws = np.random.default_rng(name_code(group) % (2 ** 31)).integers(0, len(shared), size=(DRAWS, len(shared)))  # (r, n)
                    gap = scores[shared].to_numpy() - wide.loc[shared, baseline].to_numpy()  # (n,)
                    sampled = gap[draws].mean(axis=1)  # (r,)
                    rows.append({"group": group, "model": model, "tuned": label, "baseline": baseline, "splits": len(shared),
                                 "ahead": int((gap > 0).sum()), "tuned_mean": float(scores[shared].mean()),
                                 "baseline_mean": float(wide.loc[shared, baseline].mean()), "difference": float(gap.mean()),
                                 "low": float(np.quantile(sampled, 0.025)), "high": float(np.quantile(sampled, 0.975))})
    return pd.DataFrame(rows)


def paired_rows(scores: Scores, splits: list[str], asked: list[tuple[str, Series, Series]],
                first_queries: int | None = None) -> list[dict[str, object]]:
    """The paired comparison of each question whose two series both ran on some split."""
    rows = []
    for question, left, right in asked:
        found = paired(scores, splits, left, right, first_queries)
        if found is not None:
            rows.append({"question": question, "left_method": left.method, "left_arm": left.arm,
                         "right_method": right.method, "right_arm": right.arm, **found})
    return rows


def partly_asked(calls: pd.DataFrame, metrics: pd.DataFrame) -> set[tuple[str, str]]:
    """The method and arm pairs for which fewer prompts were sent than the scored tasks hold.

    The subagent arm asks only the first prompt of each task under some arms. Such a series is compared on those
    queries alone, and it is left out of every table over all queries.
    """
    table = completeness(calls, metrics)
    short = table[table["prompts_sent"] < table["prompts_owed"]]
    return set(zip(short["method"], short["arm"], strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validation", type=Path, nargs="*", default=[], help="scored tables of the validation prompts")
    parser.add_argument("--tuning", type=Path, help="the records.json of preference_tuning, for the table of chosen recipes")
    parser.add_argument("--trace", type=Path, nargs="+",
                        help="records.json files of preference_tuning runs that traced: write preference_trace.csv from them and nothing else")
    parser.add_argument("--inside", type=Path, nargs="+",
                        help="inside files of preference_tuning: with --tuning, write the two tables of the held-out rows and nothing else")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions(args.tables)
    if args.inside:
        assert args.tuning is not None, "--inside needs the records of --tuning"
        inside = [found for path in args.inside for found in json.loads(path.read_text(encoding="utf-8"))]
        table = inside_table(inside, json.loads(args.tuning.read_text(encoding="utf-8")))
        table.to_csv(args.out / "preference_inside.csv", index=False)
        differences = inside_differences(table, group_splits(sorted(predictions["split"].unique())))
        differences.to_csv(args.out / "preference_inside_differences.csv", index=False)
        print(f"wrote the scores on the held-out rows to {args.out}")
    if args.trace:
        records = [record for path in args.trace for record in json.loads(path.read_text(encoding="utf-8"))]
        trace_table(records, predictions).to_csv(args.out / "preference_trace.csv", index=False)
        print(f"wrote the trace of every epoch to {args.out}")
    if args.trace or args.inside:
        return
    metrics = pd.read_csv(args.tables / "metrics.csv")
    calls = pd.read_csv(args.tables / "calls.csv")
    partial = partly_asked(calls, metrics)
    whole = predictions[~pd.MultiIndex.from_frame(predictions[["method", "arm"]]).isin(partial)]
    seed_zero = whole[whole["seed"] == 0]
    groups = group_splits(sorted(predictions["split"].unique()))
    models = sorted(method for method in set(predictions["method"]) if "@" in method)
    scores = Scores(seed_zero)
    best = {name: best_ridge(scores, splits) for name, splits in groups.items()}
    for model in models:
        for ridge in sorted(set(best.values())):
            scores.add(rank_average(seed_zero, model, ridge, f"{model}+{ridge}"))
    every_seed = Scores(whole[whole["arm"] == "k256"])
    # Every series of seed 0, the partly asked ones included, on the first prompt's queries of each split. The best
    # ridge is chosen again on those queries, so a route that asked one prompt a task meets the ridge that does best there.
    first_predictions = first_rows(predictions[predictions["seed"] == 0], CHUNK_SIZE)
    first = Scores(first_predictions)
    first_label = f"first {CHUNK_SIZE}"
    probes = [split for split in plan.PROBE_SPLITS if split in groups.get("flip2", [])]
    best_first = {name: best_ridge(first, splits) for name, splits in groups.items()} if partial else {}
    for model in models if partial else []:
        for ridge in sorted(set(best_first.values())):
            first.add(rank_average(first_predictions, model, ridge, f"{model}+{ridge}"))
    # The primary tuned model's ranking averaged with its own untuned ranking, with a masked model's score, with the
    # best ridges, and, on the first queries, with each language model: whether tuning holds something the other
    # method does not.
    for tuned in tuned_models(scores)[:1]:
        partners = [f"{tuned}_zero", "zero_shot_esmc_600m"]
        for blend in (*tuned_blends(seed_zero, tuned, [*partners, *sorted(set(best.values()))]), tuned_orders(seed_zero, tuned)):
            scores.add(blend)
        # Of the language models, only the ones asked one prompt a task are read on these queries.
        asked_first = [model for model in models if (model, "k256") in partial]
        blends = [*tuned_blends(first_predictions, tuned, [*partners, *sorted(set(best_first.values())), *asked_first]),
                  tuned_orders(first_predictions, tuned)]
        for blend in blends if partial else []:
            first.add(blend)

    # The queries after the first prompt's, for a tuned model whose recipe was picked on a trace of the first ones:
    # no trace and no language model of the one-prompt route read them. Only the tuned models, the ridges, and the
    # protein models' own scores are scored there.
    rest_label = f"after the first {CHUNK_SIZE}"
    beside_tuned = {*RIDGE_METHODS, *ZERO_SHOT_METHODS}
    is_tuned = seed_zero["method"].str.startswith(tuple(f"{model}_" for model in plan.PREFERENCE_MODELS))
    rest = Scores(rest_rows(seed_zero[is_tuned | seed_zero["method"].isin(beside_tuned)], CHUNK_SIZE)) if partial and is_tuned.any() else None
    best_rest = {name: best_ridge(rest, splits) for name, splits in groups.items()} if rest is not None else {}

    paired_tables, differences, answered, partials = [], [], [], []
    chosen = list(dict.fromkeys((*best.values(), *best_first.values(), "ridge_esmc_600m", "ridge_onehot", "zero_shot_esmc_600m")))
    for name, splits in groups.items() if rest is not None else []:
        rows = paired_rows(rest, splits, tuned_questions(rest, [], best_rest[name]))
        paired_tables.append(pd.DataFrame(rows).assign(group=name, queries=rest_label))
    # The benchmark's probe splits, one a dataset, are where such a route ran its controls: a group of their own.
    for name, splits in ({**groups, **({"probe": probes} if probes else {})} if partial else {}).items():
        ridge = best_first[name] if name in best_first else best_first["flip2"]
        asked = questions(first, models, ridge) + baseline_questions(first) + tuned_questions(first, models, ridge)
        rows = paired_rows(first, splits, asked, first_queries=CHUNK_SIZE)
        paired_tables.append(pd.DataFrame(rows).assign(group=name, queries=first_label))
    for name, splits in groups.items():
        asked = questions(scores, models, best[name]) + baseline_questions(scores) + tuned_questions(scores, models, best[name])
        paired_tables.append(pd.DataFrame(paired_rows(scores, splits, asked)).assign(group=name, queries="all"))
        # Split by split, a tuned model's primary arm against the best ridge and against each language model.
        for tuned in tuned_models(scores):
            primary_arm = named(f"{tuned}_dpo", tuned_arm_of(tuned))
            for scope, held, ridges in (("all", scores, best), *([(first_label, first, best_first)] if partial else [])):
                rows = split_differences(held, splits, primary_arm, named(ridges[name], tuned_arm_of(tuned)))
                for model in models if scope != "all" or not partial else []:
                    rows += split_differences(held, splits, primary_arm, named(model, "k256"))
                differences += [{"group": name, "queries": scope, **row} for row in rows]
        for model in models:
            for other in chosen:
                for scope, held in (("all", scores), *([(first_label, first)] if partial else [])):
                    rows = split_differences(held, splits, named(model, "k256"), named(other, "k256"))
                    differences += [{"group": name, "queries": scope, **row} for row in rows]
            for arm in ("k256", "k256_cipher", "klarge"):
                others = [named(method, "k256") for method in chosen] + [named(other, arm) for other in models if other != model]
                scope, rows = (first_label, first_predictions) if (model, arm) in partial else ("all", seed_zero)
                found = answered_only(rows, splits, named(model, arm), others)
                answered += [{"group": name, "arm": arm, "scope": scope, **row} for row in found]
            controls = [named(method, "k256") for method in (best[name], "mutation_count", "zero_shot_esmc_600m", "ridge_onehot")
                        if method in scores.methods]
            for control in controls:
                for series, given in ((named(model, "k256"), control), (control, named(model, "k256"))):
                    found = partial_comparison(scores, splits, series, given)
                    if found is not None:
                        partials.append({"group": name, **found})

    # The splits that ran every context size are a group of their own, so a learning curve compares like with like.
    curve = [split for split in plan.CURVE_SPLITS if split in scores.splits]
    mean_groups = {**groups, **({"curve": curve} if curve else {}), **({"probe": probes} if probes else {})}
    mean_tables = [means_table(scores, mean_groups).assign(seeds="seed 0", queries="all"),
                   means_table(every_seed, mean_groups).assign(seeds="every seed", queries="all")]
    if partial:
        mean_tables.append(means_table(first, mean_groups, CHUNK_SIZE).assign(seeds="seed 0", queries=first_label))
    if rest is not None:
        mean_tables.append(means_table(rest, groups).assign(seeds="seed 0", queries=rest_label))
    per_split = pd.concat([series_table(scores).assign(seeds="seed 0", queries="all"),
                           series_table(every_seed).assign(seeds="every seed", queries="all"),
                           *([series_table(first).assign(seeds="seed 0", queries=first_label)] if partial else []),
                           *([series_table(rest).assign(seeds="seed 0", queries=rest_label)] if rest is not None else [])],
                          ignore_index=True)
    first_seed = metrics[(metrics["seed"] == 0) & ~pd.MultiIndex.from_frame(metrics[["method", "arm"]]).isin(partial)]
    first_seed = first_seed.assign(group=np.where(first_seed["split"].str.startswith(POSTCUTOFF_DATASET), "postcutoff", "flip2"))
    metric_means = first_seed.groupby(["group", "arm", "method"]).agg(
        splits=("split", "nunique"), spearman=("spearman", "mean"), within_strata_spearman=("within_strata_spearman", "mean"),
        within_strata_queries=("within_strata_queries", "sum"), ndcg=("ndcg", "mean"), roc_auc=("roc_auc", "mean"),
        pr_auc=("pr_auc", "mean"), balanced_mcc=("balanced_mcc", "mean"),
        answered_fraction=("answered_fraction", "mean")).reset_index()

    pd.concat(mean_tables, ignore_index=True).to_csv(args.out / "means.csv", index=False)
    per_split.to_csv(args.out / "per_split.csv", index=False)
    pd.concat(paired_tables, ignore_index=True).to_csv(args.out / "paired.csv", index=False)
    pd.DataFrame(differences).to_csv(args.out / "split_differences.csv", index=False)
    pd.DataFrame(answered).to_csv(args.out / "answered_only.csv", index=False)
    pd.DataFrame(partials).to_csv(args.out / "partial_spearman.csv", index=False)
    metric_means.to_csv(args.out / "metric_means.csv", index=False)
    decimals = metrics.drop_duplicates("split").set_index("split")["decimals"].to_dict()
    seed_zero_all = predictions[predictions["seed"] == 0]
    matched_groups = match_groups(seed_zero_all, decimals, {**groups, **({"probe": probes} if probes else {})})
    scopes = [("all", seed_zero_all)] + ([(first_label, first_predictions)] if partial else [])
    matched = [match_tables(rows, decimals, matched_groups) for _, rows in scopes]
    for position, name in enumerate(("matches_per_split.csv", "matches.csv")):
        tables = [found[position].assign(queries=scope) for found, (scope, _) in zip(matched, scopes, strict=True)]
        pd.concat(tables, ignore_index=True).to_csv(args.out / name, index=False)
    effort_table(calls, mean_groups).to_csv(args.out / "effort.csv", index=False)
    if args.validation:
        validation_table(args.validation).to_csv(args.out / "effort_validation.csv", index=False)
    if args.tuning:
        records = json.loads(args.tuning.read_text(encoding="utf-8"))
        tuning_table(records).to_csv(args.out / "preference_tuning.csv", index=False)
    calibration(seed_zero, "k256").to_csv(args.out / "calibration_k256.csv", index=False)
    chunk_drift(seed_zero, "k256").to_csv(args.out / "chunk_drift_k256.csv", index=False)
    completeness(calls, metrics).to_csv(args.out / "completeness.csv", index=False)
    refusal_table(calls).to_csv(args.out / "refusals.csv", index=False)
    cost_table(calls).to_csv(args.out / "cost.csv", index=False)
    found = {"groups": mean_groups, "match_groups": matched_groups, "language_models": models,
             "tuned_models": tuned_models(scores), "best_ridge_at_k256": best,
             "best_ridge_first": best_first, "best_ridge_rest": best_rest, "rest_queries": rest_label, "draws": DRAWS,
             "partly_asked": sorted(f"{method} {arm}" for method, arm in partial), "partly_asked_queries": CHUNK_SIZE,
             "min_answered": MIN_ANSWERED}
    (args.out / "findings.json").write_text(json.dumps(found, indent=1), encoding="utf-8")
    print(f"wrote the summaries to {args.out}")


if __name__ == "__main__":
    main()
