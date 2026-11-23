"""Facts about each split that could let a method score without reading the sequence: duplicates, ties and shifts.

    python -m general_models_vs_plms.data_checks --data DIR --tasks DIR --out FILE

One row per split: its sizes, how many test rows repeat a training sequence, how many of the 300 queries repeat a row
the 256-row prompt shows, how many distinct label values the training part has, how far the test labels sit from the
training labels, and how well the substitution count or the stratum alone ranks the test rows.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from pathlib import Path

from . import flip2, plan
from .encoding import mutation_counts
from .scoring import spearman
from .strata import strata_of
from .tasks import load_layout, read_tasks


def split_facts(split_info: flip2.SplitInfo, frame: pd.DataFrame, context_rows: np.ndarray, query_rows: np.ndarray) -> dict[str, object]:
    """The facts for one split, given the rows its 256-row seed-0 task shows and asks."""
    # context_rows: (k,); query_rows: (q,)
    layout = load_layout(split_info, frame)
    parts = frame["part"].to_numpy()  # (n,)
    targets = frame["target"].to_numpy()  # (n,)
    sequences = frame["sequence"].to_numpy()  # (n,)
    train, test = parts == "train", parts == "test"  # (n,), (n,)
    train_sequences = set(sequences[train].tolist())
    shown_sequences = set(sequences[context_rows].tolist())
    strata = strata_of(split_info, layout)  # (n,)
    stratum_means = pd.Series(targets[train]).groupby(strata[train]).mean()
    test_stratum_mean = pd.Series(strata[test]).map(stratum_means).fillna(targets[train].mean()).to_numpy()  # (n_test,)
    lengths = np.array([len(sequence) for sequence in sequences])  # (n,)

    facts = {
        "split": split_info.name, "layout": split_info.layout,
        "train_rows": int(train.sum()), "validation_rows": int((parts == "validation").sum()), "test_rows": int(test.sum()),
        "shortest": int(lengths.min()), "longest": int(lengths.max()),
        "train_repeated_sequences": int(train.sum() - len(train_sequences)),
        "test_rows_in_train": int(np.isin(sequences[test], list(train_sequences)).sum()),
        "queries_in_train": int(sum(sequence in train_sequences for sequence in sequences[query_rows])),
        "queries_shown_in_prompt": int(sum(sequence in shown_sequences for sequence in sequences[query_rows])),
        "train_label_values": int(len(np.unique(targets[train]))),
        "train_label_mean": float(targets[train].mean()), "train_label_sd": float(targets[train].std()),
        "test_label_mean": float(targets[test].mean()), "test_label_sd": float(targets[test].std()),
        "test_above_train_maximum": float((targets[test] > targets[train].max()).mean()),
        "test_below_train_minimum": float((targets[test] < targets[train].min()).mean()),
        "strata_in_train": int(len(np.unique(strata[train]))), "strata_in_test": int(len(np.unique(strata[test]))),
        "test_strata_absent_from_train": int(len(set(strata[test].tolist()) - set(strata[train].tolist()))),
        "stratum_mean_test_spearman": spearman(targets[test], test_stratum_mean),
    }
    if layout.parent is not None:
        counts = mutation_counts(layout)  # (n,)
        changed = layout.codes != layout.parent[None, :]  # (n, l)
        seen_columns = changed[train].any(axis=0)  # (l,)
        # A substitution is a column and the residue put there; 256 codes a column makes one integer of the pair.
        pair_codes = np.arange(layout.codes.shape[1])[None, :] * 256 + layout.codes  # (n, l)
        seen_pairs = np.unique(pair_codes[train][changed[train]])  # (p,)
        unseen_pair = changed[test] & ~np.isin(pair_codes[test], seen_pairs)  # (n_test, l)
        facts.update({
            "train_mean_substitutions": float(counts[train].mean()), "test_mean_substitutions": float(counts[test].mean()),
            "train_most_substitutions": int(counts[train].max()), "test_most_substitutions": int(counts[test].max()),
            "substitution_count_test_spearman": spearman(targets[test], -counts[test].astype(float)),
            "positions_changed_in_train": int(seen_columns.sum()),
            "test_rows_with_a_position_unseen_in_train": float((changed[test] & ~seen_columns[None, :]).any(axis=1).mean()),
            "test_rows_with_a_substitution_unseen_in_train": float(unseen_pair.any(axis=1).mean()),
            "test_rows_with_every_substitution_unseen_in_train": float(
                (unseen_pair.sum(axis=1) == changed[test].sum(axis=1))[changed[test].any(axis=1)].mean()),
        })
    return facts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    infos = {split_info.name: split_info for split_info in plan.ALL_SPLITS}
    rows = []
    for task in read_tasks(args.tasks):
        if task.arm == "k256" and task.seed == 0:
            split_info = infos[task.split]
            frame = flip2.load_split(flip2.split_path(args.data, split_info))
            rows.append(split_facts(split_info, frame, np.array(task.context_rows), np.array(task.query_rows)))
    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
