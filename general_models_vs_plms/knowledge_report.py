"""Knowledge of a sequence against performance on it: tables, correlations, and the figure.

    python -m general_models_vs_plms.knowledge_report --set FILE --answers FILE --masked DIR --predictions FILE \
        --summary DIR --out DIR

Reads the set `sequence_knowledge select` drew, the scored answers of a language model (`score`), the tables of each
masked language model (`esmc`), the scored predictions of the in-context experiment, and its summary folder. Writes
`recovery.csv` (share of masked residues right by model, kind of sequence, and level), `rhodopsins.csv` (one row a
Rhomax query with every knowledge measure and every absolute error), `relations.csv` (the Spearman correlation of
knowledge with absolute error, with its interval and its partial given how unusual the pigment is), `parents.csv`,
`recited.csv` (error by how many levels came back whole), and `sequence_knowledge.png`.
"""

from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from matplotlib.axes import Axes
from scipy.stats import rankdata

from foundry.plotting.panels import letter_panels
from .figures import ESM2, ESMC, GRAY, INK, ORANGE, PLM_LABELS, Row, dot_chart, quiet, save, setup
from .report import partial_spearman, row_correlation
from .sequence_knowledge import LEVELS, RHOMAX_TASK, Entry, read_set


LANGUAGE_MODEL = "claude-mythos-5-2@harness-low"
LANGUAGE_LABEL = "Claude Mythos 5.2"
MASKED_MODELS = ("esm2_650m", "esmc_300m", "esmc_600m", "esmc_6b")
NATURAL, SHUFFLED = ("parent", "rhodopsin"), ("shuffled_parent", "shuffled_rhodopsin")
ARMS = {"k0_described": "no examples", "k256": "256 examples"}
DRAWS = 2000
SEED = 0
RECITED = 0.9
MARKERS = {"esm2_650m": "D", "esmc_300m": "v", "esmc_600m": "s", "esmc_6b": "^"}


def recovery_table(answers: pd.DataFrame, masked: dict[str, pd.DataFrame], asked: set[str]) -> pd.DataFrame:
    """Share of masked residues right by model, kind of sequence, and level, on the sequences the language model read."""
    tables = [answers.assign(model=LANGUAGE_MODEL),
              answers.assign(model="most common visible residue", correct=answers["unigram_correct"])]
    tables += [table[(table["level"] > 0) & table["name"].isin(asked)] for table in masked.values()]
    rows = []
    for table in tables:
        natural = table["kind"].isin(NATURAL)
        groups = [("natural", natural), ("shuffled", ~natural)] + [(kind, table["kind"] == kind) for kind in NATURAL]
        for kind, chosen in groups:
            for level, group in table[chosen].groupby("level"):
                rows.append({"model": group["model"].iloc[0], "sequences": kind, "level": level, "count": len(group),
                             "masked": int(group["masked"].sum()), "share_right": group["correct"].sum() / group["masked"].sum(),
                             "recited": int((group["correct"] >= RECITED * group["masked"]).sum())})
    return pd.DataFrame(rows)


def rhodopsin_table(entries: list[Entry], answers: pd.DataFrame, masked: dict[str, pd.DataFrame], predictions: pd.DataFrame) -> pd.DataFrame:
    """One row a Rhomax query: each model's knowledge of the sequence and each method's absolute error on it."""
    table = pd.DataFrame([{"name": entry.name, "row": entry.row, "asked": entry.asked} for entry in entries if entry.kind == "rhodopsin"])
    totals = answers.groupby("name")[["correct", "masked"]].sum()
    table["recovery_language_model"] = table["name"].map(totals["correct"] / totals["masked"])
    recited = (answers["correct"] >= RECITED * answers["masked"]).groupby(answers["name"]).sum()
    table["levels_recited"] = table["name"].map(recited)
    for model, scores in masked.items():
        one_at_a_time = scores[scores["level"] == 0].set_index("name")
        table[f"ppll_{model}"] = table["name"].map(one_at_a_time["mean_log_probability"])
        at_levels = scores[scores["level"] > 0].groupby("name")[["correct", "masked"]].sum()
        table[f"recovery_{model}"] = table["name"].map(at_levels["correct"] / at_levels["masked"])
    truth = predictions[predictions["task_id"] == RHOMAX_TASK].drop_duplicates("row").set_index("row")["truth"]
    table["truth"] = table["row"].map(truth)
    table["unusual"] = (table["truth"] - table["truth"].median()).abs()
    series = {f"error_language_model_{arm}": (LANGUAGE_MODEL, arm) for arm in ARMS}
    series |= {f"error_ridge_{model}": (f"ridge_{model}", "k256") for model in masked}
    series |= {"error_ridge_onehot": ("ridge_onehot", "k256")}
    for column, (method, arm) in series.items():
        task = RHOMAX_TASK.replace("__k256__", f"__{arm}__")
        found = predictions[(predictions["task_id"] == task) & (predictions["method"] == method)].set_index("row")["prediction"]
        table[column] = (table["row"].map(found) - table["truth"]).abs()
    return table


def relation(table: pd.DataFrame, knowledge: str, error: str, label: str) -> dict[str, object]:
    """Spearman of a knowledge measure with an absolute error over sequences, its bootstrap interval, and its partial."""
    kept = table.dropna(subset=[knowledge, error])
    x, y, z = (kept[column].to_numpy(dtype=float) for column in (knowledge, error, "unusual"))  # each (n,)
    samples = np.random.default_rng(SEED).integers(0, len(kept), size=(DRAWS, len(kept)))  # (DRAWS, n)
    drawn = row_correlation(rankdata(x[samples], axis=1), rankdata(y[samples], axis=1))  # (DRAWS,)
    partial = partial_spearman(y[None, :], x[None, :], z[None, :])[0]
    partial_drawn = partial_spearman(y[samples], x[samples], z[samples])  # (DRAWS,)
    return {"relation": label, "knowledge": knowledge, "error": error, "sequences": len(kept),
            "spearman": float(row_correlation(rankdata(x)[None, :], rankdata(y)[None, :])[0]),
            "low": float(np.percentile(drawn, 2.5)), "high": float(np.percentile(drawn, 97.5)),
            "partial_given_unusual": float(partial), "partial_low": float(np.percentile(partial_drawn, 2.5)),
            "partial_high": float(np.percentile(partial_drawn, 97.5)), "median_error": float(np.median(y))}


def relations_table(rhodopsins: pd.DataFrame, masked: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = [relation(rhodopsins, "recovery_language_model", f"error_language_model_{arm}", f"{LANGUAGE_LABEL}, {words}")
            for arm, words in ARMS.items()]
    for model in masked:
        rows.append(relation(rhodopsins, f"ppll_{model}", f"error_ridge_{model}", f"Ridge on {PLM_LABELS[model]} embedding, PPLL"))
        rows.append(relation(rhodopsins, f"recovery_{model}", f"error_ridge_{model}", f"Ridge on {PLM_LABELS[model]} embedding, recovery"))
    return pd.DataFrame(rows)


def recited_table(rhodopsins: pd.DataFrame) -> pd.DataFrame:
    """The language model's error by how many of the three levels it returned a rhodopsin at RECITED or better.

    Added after the answers were seen, because a sequence came back either nearly whole or at the rate of a guess.
    """
    rows = []
    for count, group in rhodopsins.dropna(subset=["levels_recited"]).groupby("levels_recited"):
        errors = {f"median_error_{arm}": float(group[f"error_language_model_{arm}"].median()) for arm in ARMS}
        rows.append({"levels_recited": int(count), "sequences": len(group), **errors})
    return pd.DataFrame(rows)


def parents_table(entries: list[Entry], answers: pd.DataFrame, masked: dict[str, pd.DataFrame], per_split: pd.DataFrame) -> pd.DataFrame:
    """Each dataset parent: knowledge of it, and the Spearman each model reaches on that dataset's probe split."""
    chosen = per_split[(per_split["seeds"] == "seed 0") & (per_split["queries"] == "first 100")]
    rows = []
    for entry in entries:
        if entry.kind != "parent":
            continue
        dataset = entry.name.split("/")[1]
        split = chosen[chosen["split"].str.startswith(f"{dataset}/")]
        totals = answers[answers["name"] == entry.name][["correct", "masked"]].sum()
        row = {"dataset": dataset, "length": len(entry.sequence), "recovery_language_model": totals["correct"] / totals["masked"]}
        for arm in ARMS:
            found = split[(split["method"] == LANGUAGE_MODEL) & (split["arm"] == arm)]
            row[f"spearman_language_model_{arm}"] = found["spearman"].mean() if len(found) else np.nan
        for model, scores in masked.items():
            mine = scores[scores["name"] == entry.name]
            row[f"ppll_{model}"] = mine[mine["level"] == 0]["mean_log_probability"].iloc[0]
            found = split[(split["method"] == f"zero_shot_{model}") & (split["arm"] == "k256")]
            row[f"spearman_zero_shot_{model}"] = found["spearman"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def draw_recovery(axis: Axes, recovery: pd.DataFrame, sequences: str, masked: list[str]) -> None:
    """Share of masked residues right against the share masked, one line a model."""
    styles = {LANGUAGE_MODEL: (LANGUAGE_LABEL, ORANGE, "o"), "most common visible residue": ("Most common visible residue", GRAY, "x")}
    styles |= {model: (PLM_LABELS[model], ESM2 if model.startswith("esm2") else ESMC, MARKERS[model]) for model in masked}
    for model, (label, color, marker) in styles.items():
        rows = recovery[(recovery["model"] == model) & (recovery["sequences"] == sequences)].sort_values("level")
        axis.plot(100 * rows["level"], 100 * rows["share_right"], color=color, marker=marker, markersize=6, linewidth=1.6,
                  linestyle="--" if color == GRAY else "-", label=label)
    axis.set_xticks([100 * level for level in LEVELS])
    axis.set_xlabel(f"Residues masked (%), {sequences} sequences")
    axis.set_ylim(0, 100)
    quiet(axis, grid="y")


def draw_relation(axis: Axes, rhodopsins: pd.DataFrame, knowledge: str, error: str, color: str, xlabel: str, ylabel: str) -> None:
    """One dot a rhodopsin: what the model knows of the sequence against its error on the absorption maximum."""
    kept = rhodopsins.dropna(subset=[knowledge, error])
    axis.scatter(kept[knowledge], kept[error], s=22, color=color, alpha=0.75, linewidths=0)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    quiet(axis, grid="y")


def relation_rows(relations: pd.DataFrame, masked: list[str]) -> list[Row]:
    """The rows of the last panel: each model's correlation of knowledge with its own error, under two headings."""
    found = relations.set_index("relation")

    def one(relation: str, label: str, color: str, marker: str) -> Row:
        row = found.loc[relation]
        return Row(label, row["spearman"], row["low"], row["high"], color, marker)

    rows = [Row(f"{LANGUAGE_LABEL}: share filled in, against its own error", heading=True)]
    rows += [one(f"{LANGUAGE_LABEL}, {words}", words.capitalize(), ORANGE, "o") for words in ARMS.values()]
    rows.append(Row("PPLL, against the error of the ridge on its embedding", heading=True))
    rows += [one(f"Ridge on {PLM_LABELS[model]} embedding, PPLL", PLM_LABELS[model],
                 ESM2 if model.startswith("esm2") else ESMC, MARKERS[model]) for model in masked]
    return rows


def figure(recovery: pd.DataFrame, rhodopsins: pd.DataFrame, relations: pd.DataFrame, masked: list[str], out: Path) -> None:
    """A and B, recovery on natural and on shuffled sequences. C, one dot a rhodopsin. D, every model's correlation."""
    setup()
    drawing, axes = plt.subplots(2, 2, figsize=(10.8, 7.6))
    draw_recovery(axes[0, 0], recovery, "natural", masked)
    draw_recovery(axes[0, 1], recovery, "shuffled", masked)
    axes[0, 0].set_ylabel("Masked residues filled in correctly (%)")
    axes[0, 1].legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right", ncol=1, labelcolor=INK)
    draw_relation(axes[1, 0], rhodopsins.assign(share=100 * rhodopsins["recovery_language_model"]), "share",
                  "error_language_model_k0_described", ORANGE, f"Masked residues {LANGUAGE_LABEL} fills in correctly (%)",
                  "Absolute error of its absorption maximum,\nno examples (nm)")
    dot_chart(axes[1, 1], relation_rows(relations, masked), "Spearman of knowledge with absolute error")
    letter_panels(axes)
    drawing.subplots_adjust(wspace=0.36, hspace=0.34)
    save(drawing, out, "sequence_knowledge")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--masked", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    entries = read_set(args.set)
    answers = pd.read_csv(args.answers)
    masked = {model: pd.read_csv(args.masked / f"knowledge_{model}.csv") for model in MASKED_MODELS
              if (args.masked / f"knowledge_{model}.csv").exists()}
    assert masked, f"no knowledge_<model>.csv in {args.masked}"
    predictions = pd.read_csv(args.predictions, usecols=["task_id", "method", "row", "truth", "prediction"])
    predictions = predictions[predictions["task_id"].str.startswith("rhomax")]

    recovery = recovery_table(answers, masked, {entry.name for entry in entries if entry.asked})
    rhodopsins = rhodopsin_table(entries, answers, masked, predictions)
    relations = relations_table(rhodopsins, masked)
    parents = parents_table(entries, answers, masked, pd.read_csv(args.summary / "per_split.csv"))
    args.out.mkdir(parents=True, exist_ok=True)
    tables = {"recovery": recovery, "rhodopsins": rhodopsins, "relations": relations, "parents": parents,
              "recited": recited_table(rhodopsins)}
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False, lineterminator="\n")
    figure(recovery, rhodopsins, relations, list(masked), args.out)
    print(recovery.pivot_table(index=["model", "sequences"], columns="level", values="share_right").round(3).to_string())
    print(relations[["relation", "sequences", "spearman", "low", "high", "partial_given_unusual"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
