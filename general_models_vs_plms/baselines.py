"""Baselines on the rows each task shows: ridge regressions, zero-shot likelihood scores and trivial predictors.

    python -m general_models_vs_plms.baselines --data DIR --tasks DIR --features DIR --out DIR [--models A,B] [--add]

A supervised baseline is fit on a task's context rows with the values the prompt shows, and its ridge penalty is
chosen by leave-one-out error inside those rows, so it sees what the language model sees and nothing more.

The controls are ridges of the same kind on features that hold less than a trained model's embedding: the amino acid
composition of each chain, a random vector per sequence, the embedding of an untrained model, and the embeddings a
trained model gives a reordered sequence (`plm_features` says how each is made).
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from tqdm import tqdm

from . import flip2, plan
from .encoding import AMINO_ACIDS, CHAIN_BREAK, GAP_CODE, Layout, composition, mutation_counts, one_hot, random_vectors
from .flip2 import SplitInfo
from .strata import strata_of
from .tasks import Task, label_decimals, load_layout, read_tasks


ALPHAS = np.logspace(-3, 4, 15)
PLM_NAMES = ("esm2_650m", "esmc_300m", "esmc_600m", "esmc_6b")
CONTROL_EMBEDDINGS = ("esmc_300m_random", "esmc_300m_shuffled", "esmc_300m_permuted", "esmc_600m_shuffled", "esmc_600m_permuted")
FEATURE_CONTROLS = ("composition", "random_vector")
CONTROL_NAMES = FEATURE_CONTROLS + CONTROL_EMBEDDINGS
# The width of the smallest ESMC model.
RANDOM_VECTOR_WIDTH = 960
LIKELIHOOD_WEIGHT = 100.0
BASELINE_ARMS = ("k16", "k64", "k256", "klarge")


def ridge(fit_features: np.ndarray, fit_values: np.ndarray, query_features: np.ndarray) -> np.ndarray:
    """Ridge predictions with the penalty chosen by leave-one-out error on the fit rows."""
    # fit_features: (k, f); fit_values: (k,); query_features: (q, f)
    model = RidgeCV(alphas=ALPHAS).fit(fit_features, fit_values)
    return model.predict(query_features)  # (q,)


def standardize(fit_features: np.ndarray, query_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center and scale both blocks by the fit rows' mean and standard deviation."""
    # fit_features: (k, f); query_features: (q, f)
    mean = fit_features.mean(axis=0)  # (f,)
    scale = fit_features.std(axis=0)  # (f,)
    scale = np.where(scale > 0, scale, 1.0)  # (f,)
    return (fit_features - mean) / scale, (query_features - mean) / scale  # (k, f), (q, f)


def nearest_neighbor(layout: Layout, context_rows: np.ndarray, context_values: np.ndarray, query_rows: np.ndarray) -> np.ndarray:
    """The mean value of the context rows at the smallest Hamming distance from each query."""
    # context_rows: (k,); context_values: (k,); query_rows: (q,)
    context_codes = layout.codes[context_rows]  # (k, l)
    predictions = np.empty(len(query_rows))  # (q,)
    for index, row in enumerate(query_rows):
        distances = (context_codes != layout.codes[row]).sum(axis=1)  # (k,)
        predictions[index] = context_values[distances == distances.min()].mean()
    return predictions  # (q,)


def stratum_mean(strata: np.ndarray, context_rows: np.ndarray, context_values: np.ndarray, query_rows: np.ndarray) -> np.ndarray:
    """The mean context value of each query's stratum, or of the whole context when its stratum is absent."""
    # strata: (n,); context_rows: (k,); context_values: (k,); query_rows: (q,)
    by_stratum: dict[str, list[float]] = {}
    for stratum, value in zip(strata[context_rows], context_values, strict=True):
        by_stratum.setdefault(str(stratum), []).append(float(value))
    overall = float(context_values.mean())
    return np.array([np.mean(by_stratum[str(stratum)]) if str(stratum) in by_stratum else overall
                     for stratum in strata[query_rows]])  # (q,)


def marginal_scores(layout: Layout, marginals: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Sum over a row's substitutions of the masked log-probability of the new residue less that of the parent's."""
    # marginals: (l_residues, 20), one row per parent residue with the chain break left out; rows: (n,)
    assert layout.parent is not None, "masked marginals need a parent"
    residue_index = np.full(256, -1, dtype=np.int64)  # (256,)
    residue_index[np.frombuffer(AMINO_ACIDS.encode("ascii"), dtype=np.uint8)] = np.arange(len(AMINO_ACIDS))
    is_residue_column = np.array([label != CHAIN_BREAK for label in layout.labels])  # (l,)
    table_row = np.cumsum(is_residue_column) - 1  # (l,)
    scores = np.zeros(len(rows))  # (n,)
    for index, row in enumerate(rows):
        for column in np.flatnonzero(layout.codes[row] != layout.parent):
            new_code, old_code = layout.codes[row, column], layout.parent[column]
            if new_code == GAP_CODE or old_code == GAP_CODE:
                continue
            scores[index] += (marginals[table_row[column], residue_index[new_code]]
                              - marginals[table_row[column], residue_index[old_code]])
    return scores  # (n,)


class SplitFeatures:
    """A split's protein language model features, addressed by row, for the named models and control embeddings."""

    def __init__(self, folder: Path, split_info: SplitInfo, names: tuple[str, ...] = PLM_NAMES) -> None:
        self.by_model = {name: np.load(folder / name / f"{split_info.dataset}-{split_info.split}.npz") for name in names}
        self.position = {name: {int(row): index for index, row in enumerate(arrays["rows"])}
                         for name, arrays in self.by_model.items()}

    def take(self, name: str, key: str, rows: np.ndarray) -> np.ndarray:
        """Rows of one stored array, in the order asked."""
        # rows: (n,)
        indices = [self.position[name][int(row)] for row in rows]
        return self.by_model[name][key][indices]  # (n, ...): (n, d) for embedding, (n,) for likelihood

    def zero_shot(self, name: str, layout: Layout, rows: np.ndarray) -> tuple[str, np.ndarray]:
        """The masked-marginal score where the split has a parent, else the pseudo-log-likelihood."""
        # rows: (n,)
        if layout.parent is not None:
            return "masked_marginal", marginal_scores(layout, self.by_model[name]["parent_marginals"], rows)  # (n,)
        return "likelihood", self.take(name, "likelihood", rows)  # (n,)

    def embedding_ridge(self, name: str, fit_rows: np.ndarray, values: np.ndarray, query_rows: np.ndarray) -> np.ndarray:
        """Ridge on one stored embedding, standardized on the fit rows."""
        # fit_rows: (k,); values: (k,); query_rows: (q,)
        fit_embedding, query_embedding = standardize(self.take(name, "embedding", fit_rows),
                                                     self.take(name, "embedding", query_rows))  # (k, d), (q, d)
        return ridge(fit_embedding, values, query_embedding)  # (q,)


def plm_ridges(
    features: SplitFeatures, layout: Layout, fit_rows: np.ndarray, values: np.ndarray, query_rows: np.ndarray,
    fit_onehot: np.ndarray, query_onehot: np.ndarray, models: tuple[str, ...] = PLM_NAMES, suffix: str = "",
) -> dict[str, np.ndarray]:
    """Per protein language model: ridge on its embedding, and ridge on one-hot features with its zero-shot score appended.

    The appended score is standardized on the fit rows and weighted so the penalty leaves it nearly free, as in the
    augmented models of Hsu et al. (2022).
    """
    # fit_rows: (k,); values: (k,); query_rows: (q,); fit_onehot: (k, f); query_onehot: (q, f)
    predictions = {}
    for name in models:
        predictions[f"ridge_{name}{suffix}"] = features.embedding_ridge(name, fit_rows, values, query_rows)
        _, query_score = features.zero_shot(name, layout, query_rows)  # (q,)
        _, fit_score = features.zero_shot(name, layout, fit_rows)  # (k,)
        fit_z, query_z = standardize(fit_score[:, None], query_score[:, None])  # (k, 1), (q, 1)
        augmented_fit = np.hstack([fit_onehot, LIKELIHOOD_WEIGHT * fit_z])  # (k, f + 1)
        augmented_query = np.hstack([query_onehot, LIKELIHOOD_WEIGHT * query_z])  # (q, f + 1)
        predictions[f"ridge_onehot_plus_{name}{suffix}"] = ridge(augmented_fit, values, augmented_query)
    return predictions  # each (q,)


def control_ridges(
    features: SplitFeatures, layout: Layout, fit_rows: np.ndarray, values: np.ndarray, query_rows: np.ndarray,
    controls: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Ridge on each control's features: composition, a random vector, or a stored control embedding."""
    # fit_rows: (k,); values: (k,); query_rows: (q,)
    predictions = {}
    for name in controls:
        if name == "composition":
            fit, query = standardize(composition(layout, fit_rows), composition(layout, query_rows))  # (k, 20 c), (q, 20 c)
            predictions["ridge_composition"] = ridge(fit, values, query)
        elif name == "random_vector":
            fit = random_vectors(layout, fit_rows, RANDOM_VECTOR_WIDTH)  # (k, w)
            query = random_vectors(layout, query_rows, RANDOM_VECTOR_WIDTH)  # (q, w)
            predictions["ridge_random_vector"] = ridge(fit, values, query)
        else:
            predictions[f"ridge_{name}"] = features.embedding_ridge(name, fit_rows, values, query_rows)
    return predictions  # each (q,)


def task_baselines(
    task: Task, layout: Layout, strata: np.ndarray, features: SplitFeatures,
    models: tuple[str, ...] = PLM_NAMES, controls: tuple[str, ...] = CONTROL_NAMES, model_free: bool = True,
) -> dict[str, list[float]]:
    """The baselines' predictions for a task's queries, in query order.

    `model_free` adds the baselines that need no protein language model; a later run that adds a model leaves them out.
    """
    # strata: (n,), one name per row of the split
    context_rows, query_rows = np.array(task.context_rows), np.array(task.query_rows)  # (k,), (q,)
    values = np.array(task.context_values)  # (k,)
    fit_onehot, query_onehot = one_hot(layout, context_rows, query_rows)  # (k, f), (q, f)
    predictions = {}
    if model_free:
        predictions["ridge_onehot"] = ridge(fit_onehot, values, query_onehot)
        predictions["nearest_neighbor"] = nearest_neighbor(layout, context_rows, values, query_rows)
        predictions["stratum_mean"] = stratum_mean(strata, context_rows, values, query_rows)
        if layout.parent is not None:
            predictions["mutation_count"] = -mutation_counts(layout)[query_rows].astype(float)  # (q,)

    predictions.update(plm_ridges(features, layout, context_rows, values, query_rows, fit_onehot, query_onehot, models))
    predictions.update(control_ridges(features, layout, context_rows, values, query_rows, controls))
    for name in models:
        kind, query_score = features.zero_shot(name, layout, query_rows)  # (q,)
        assert kind in ("masked_marginal", "likelihood")
        predictions[f"zero_shot_{name}"] = query_score
        predictions[f"likelihood_{name}"] = features.take(name, "likelihood", query_rows)  # (q,)
    return {method: np.asarray(scores, dtype=float).tolist() for method, scores in predictions.items()}


def full_train_plm_reference(
    frame: pd.DataFrame, layout: Layout, query_rows: np.ndarray, features: SplitFeatures,
    models: tuple[str, ...] = PLM_NAMES,
) -> dict[str, list[float]]:
    """The protein language model ridges fit on every training row, for the queries; methods end in `_full`."""
    # query_rows: (q,)
    targets = frame["target"].to_numpy()  # (n,)
    train_rows = np.flatnonzero(frame["part"].to_numpy() == "train")  # (n_train,)
    values = np.round(targets[train_rows], label_decimals(targets[train_rows]))  # (n_train,)
    fit_onehot, query_onehot = one_hot(layout, train_rows, query_rows)  # (n_train, f), (q, f)
    predictions = plm_ridges(features, layout, train_rows, values, query_rows, fit_onehot, query_onehot, models, suffix="_full")
    return {method: np.asarray(scores, dtype=float).tolist() for method, scores in predictions.items()}


def full_train_reference(
    split_info: SplitInfo, frame: pd.DataFrame, layout: Layout, query_rows: np.ndarray,
) -> dict[str, object]:
    """One-hot ridge fit on every training row: predictions for the queries and its score on the whole test part."""
    # query_rows: (q,)
    parts = frame["part"].to_numpy()  # (n,)
    targets = frame["target"].to_numpy()  # (n,)
    train_rows, test_rows = np.flatnonzero(parts == "train"), np.flatnonzero(parts == "test")  # (n_train,), (n_test,)
    values = np.round(targets[train_rows], label_decimals(targets[train_rows]))  # (n_train,)
    fit_features, test_features = one_hot(layout, train_rows, test_rows)  # (n_train, f), (n_test, f)
    model = RidgeCV(alphas=ALPHAS).fit(fit_features, values)
    test_predictions = model.predict(test_features)  # (n_test,)
    position = {int(row): index for index, row in enumerate(test_rows)}
    return {
        "ridge_onehot_full": [float(test_predictions[position[int(row)]]) for row in query_rows],
        "full_test_spearman": float(spearmanr(test_predictions, targets[test_rows]).statistic),
        "train_rows": int(len(train_rows)), "test_rows": int(len(test_rows)), "alpha": float(model.alpha_),
    }


def write_methods(path: Path, methods: dict[str, list[float]], add: bool) -> None:
    """Write a file of predictions by method, or add the methods to the ones its file already holds."""
    if add:
        assert path.exists(), f"--add needs the file an earlier run wrote: {path}"
        methods = {**json.loads(path.read_text(encoding="utf-8")), **methods}
    path.write_text(json.dumps(methods), encoding="utf-8")


def names_of(text: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    """A comma-separated list of names, each one of the allowed."""
    names = tuple(name for name in text.split(",") if name)
    unknown = sorted(set(names) - set(allowed))
    if unknown:
        raise SystemExit(f"unknown names {unknown}; choose from {list(allowed)}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", default=",".join(PLM_NAMES), help="protein language models, comma-separated")
    parser.add_argument("--controls", default=",".join(CONTROL_NAMES), help="controls, comma-separated; empty for none")
    parser.add_argument("--add", action="store_true",
                        help="add these models and controls to the files an earlier run wrote, and leave the rest as it is")
    parser.add_argument("--full-train", action="store_true",
                        help="write only the full-data reference of each split, from features of every training row")
    args = parser.parse_args()

    models, controls = names_of(args.models, PLM_NAMES), names_of(args.controls, CONTROL_NAMES)
    stored = models + tuple(name for name in controls if name in CONTROL_EMBEDDINGS)
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = [task for task in read_tasks(args.tasks) if task.arm in BASELINE_ARMS]
    infos = {split_info.name: split_info for split_info in plan.ALL_SPLITS}
    if args.full_train:
        for task in tqdm([task for task in tasks if task.arm == "k256" and task.seed == 0], desc="full train", mininterval=20):
            split_info = infos[task.split]
            frame = flip2.load_split(flip2.split_path(args.data, split_info))
            reference = full_train_plm_reference(frame, load_layout(split_info, frame), np.array(task.query_rows),
                                                 SplitFeatures(args.features, split_info, models), models)
            write_methods(args.out / f"full__{split_info.dataset}-{split_info.split}.json", reference, args.add)
        print(f"wrote the full-data references of {list(models)} to {args.out}")
        return

    loaded: dict[str, tuple[pd.DataFrame, Layout, np.ndarray, SplitFeatures]] = {}
    for task in tqdm(tasks, desc="baselines", mininterval=20):
        split_info = infos[task.split]
        if task.split not in loaded:
            frame = flip2.load_split(flip2.split_path(args.data, split_info))
            layout = load_layout(split_info, frame)
            loaded[task.split] = (frame, layout, strata_of(split_info, layout), SplitFeatures(args.features, split_info, stored))
            if not args.add:
                reference = full_train_reference(split_info, frame, layout, np.array(task.query_rows))
                (args.out / f"reference__{split_info.dataset}-{split_info.split}.json").write_text(json.dumps(reference), encoding="utf-8")
        _, layout, strata, features = loaded[task.split]
        methods = task_baselines(task, layout, strata, features, models, controls, model_free=not args.add)
        write_methods(args.out / f"{task.task_id}.json", methods, args.add)
    print(f"wrote baselines of {list(models)} and {list(controls)} for {len(tasks)} tasks to {args.out}")


if __name__ == "__main__":
    main()
