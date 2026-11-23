"""Analyses added after the results were read, and the counts the write-up states: written as the explainer's variables.

    py -X utf8 posthoc.py TABLES_DIR SUMMARY_DIR OUT_JSON

Each analysis here was prompted by a result and is labeled so where it is reported. Leaving Rhomax out asks whether the
main comparison rests on the one dataset where the language model answers from what it already holds. Run with the
package and the folder that holds `foundry` on PYTHONPATH.
"""

from __future__ import annotations

import json
import sys
import numpy as np
import pandas as pd

from pathlib import Path

from general_models_vs_plms import report
from general_models_vs_plms.report import Scores, named
from general_models_vs_plms.tasks import CHUNK_SIZE


LOW, HIGH = "claude-mythos-5-2@harness-low", "claude-mythos-5-2@harness-high"
RHOMAX = "rhomax/by_wild_type"


def number(value: float, signed: bool = False) -> str:
    """A number as page text, with a real minus sign and no negative zero."""
    text = f"{abs(value):.2f}"
    if float(text) == 0:
        return text
    return ("&minus;" if value < 0 else "+" if signed else "") + text


def difference(found: dict[str, object]) -> str:
    """A paired difference with its interval, as page text."""
    return f"{number(found['difference'], True)} ({number(found['low'], True)} to {number(found['high'], True)})"


def main() -> None:
    tables, summary, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    found = json.loads((summary / "findings.json").read_text(encoding="utf-8"))
    predictions = report.load_predictions(tables)
    first_predictions = report.first_rows(predictions[predictions["seed"] == 0], CHUNK_SIZE)
    first = Scores(first_predictions)
    variables: dict[str, object] = {}

    # The main comparison with Rhomax left out, against the same ridge and against the ridge that is best without it.
    others = [split for split in found["groups"]["flip2"] if split != RHOMAX]
    best = found["best_ridge_first"]["flip2"]
    best_without = report.best_ridge(first, others)
    for name, ridge in (("same", best), ("best", best_without)):
        row = report.paired(first, others, named(LOW, "k256"), named(ridge, "k256"))
        variables[f"no_rhomax_{name}_model"], variables[f"no_rhomax_{name}_ridge"] = number(row["left_mean"]), number(row["right_mean"])
        variables[f"no_rhomax_{name}_d"] = difference(row)
    variables["no_rhomax_best_is_same"] = best_without == best
    variables["no_rhomax_best_name"] = best_without
    # High against low effort, and each against the ridge, on the probe splits with Rhomax left out.
    probes = [split for split in found["groups"]["probe"] if split != RHOMAX]
    row = report.paired(first, probes, named(HIGH, "k256"), named(LOW, "k256"))
    variables["probe_no_rhomax_high"], variables["probe_no_rhomax_low"] = number(row["left_mean"]), number(row["right_mean"])
    variables["probe_no_rhomax_high_low_d"] = difference(row)
    for name, model in (("low", LOW), ("high", HIGH)):
        row = report.paired(first, probes, named(model, "k256"), named(best, "k256"))
        variables[f"probe_no_rhomax_{name}_ridge_d"] = difference(row)
        variables["probe_no_rhomax_ridge"] = number(row["right_mean"])
    # With no examples, against the best zero-shot score, with Rhomax left out.
    zero_shots = [method for method in report.ZERO_SHOT_METHODS if method in first.methods]
    comparison = report.compare(first, probes, [named(method, "k256") for method in zero_shots])
    top = zero_shots[int(np.argmax(comparison["mean"]))]
    row = report.paired(first, probes, named(LOW, "k0_described"), named(top, "k256"))
    variables["probe_no_rhomax_k0"], variables["probe_no_rhomax_zero_shot"] = number(row["left_mean"]), number(row["right_mean"])
    variables["probe_no_rhomax_k0_d"] = difference(row)
    variables["probe_no_rhomax_zero_shot_name"] = top

    # How the values of each split repeat, which decides where an exact match means anything.
    groups = found["match_groups"]
    variables["continuous_flip2"], variables["discrete_flip2"] = len(groups["flip2_continuous"]), len(groups["flip2_discrete"])
    variables["continuous_probe"], variables["discrete_probe"] = len(groups["probe_continuous"]), len(groups["probe_discrete"])
    metrics = pd.read_csv(tables / "metrics.csv")
    decimals = metrics.drop_duplicates("split").set_index("split")["decimals"]
    truths = predictions.drop_duplicates(["split", "query_id"])
    for split, rows in truths.groupby("split"):
        shown = np.rint(rows["truth"].to_numpy() * 10.0 ** decimals[split])  # (q,)
        variables[f"distinct_{split.replace('/', '_')}"] = f"{len(np.unique(shown))} of {len(shown)}"

    # What the subagent route asked and got back, from the records.
    calls = report.with_method(pd.read_csv(tables / "calls.csv"))
    route = calls[calls["method"].str.startswith("claude-mythos")]
    variables["route_test_prompts"] = int(len(route))
    variables["route_refused"] = int(route["refused"].sum())
    variables["route_answered_queries"] = f"{int(route['answered'].sum()):,} of {int(route['queries'].sum()):,}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(variables, indent=1), encoding="utf-8")
    print(json.dumps(variables, indent=1))


if __name__ == "__main__":
    main()
