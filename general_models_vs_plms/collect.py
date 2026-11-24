"""Join tasks, model replies and baselines with the measured values, and write the tidy tables.

    python -m general_models_vs_plms.collect --data DIR --tasks DIR --responses DIR --out DIR [--baselines DIR]
    python -m general_models_vs_plms.collect --tables DIR --add FILE --out DIR

`predictions.csv.gz` has one row per query and method, `metrics.csv` one row per task and method, and `calls.csv`
one row per model request. The second form copies scored tables with more methods joined on: FILE holds, by task and
method, one prediction per query in the task's query order, as `preference_tuning` bundles them.
"""

from __future__ import annotations

import argparse
import json
import shutil
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

from . import flip2, plan
from .prompts import parse_reply
from .scoring import score
from .strata import strata_of
from .subagent_arm import is_harness
from .tasks import CHUNK_SIZE, Task, load_layout, read_tasks


def reply_predictions(task: Task, folder: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], bool] | None:
    """A model's predictions for a task's queries, which it answered, its call records, and whether every prompt ran.

    None if no prompt of the task ran. A query the model was asked and did not answer takes the fill value, and so
    does a query of a prompt that never ran, which is why the caller scores only tasks whose prompts all ran.
    """
    predictions = np.full(len(task.query_ids), np.nan)  # (q,)
    calls = []
    for index in range(len(task.chunks)):
        path = folder / f"{task.task_id}__c{index}.json"
        if not path.exists():
            continue

        record = json.loads(path.read_text(encoding="utf-8"))
        chunk_ids = task.query_ids[index * CHUNK_SIZE:(index + 1) * CHUNK_SIZE]
        parsed = parse_reply(record["text"], chunk_ids)
        for offset, identifier in enumerate(chunk_ids):
            if identifier in parsed:
                predictions[index * CHUNK_SIZE + offset] = parsed[identifier]
        usage = record["usage"]
        calls.append({
            "record_id": record["record_id"], "task_id": task.task_id, "split": task.split, "arm": task.arm,
            "seed": task.seed, "model_requested": record["model_requested"], "model_served": record["model_served"],
            "effort": record["effort"], "stop_reason": record["stop_reason"], "stop_category": record["stop_category"],
            "input_tokens": usage.get("input_tokens"), "cache_write_tokens": usage.get("cache_creation_input_tokens"),
            "cache_read_tokens": usage.get("cache_read_input_tokens"), "output_tokens": usage.get("output_tokens"),
            "thinking_tokens": (usage.get("output_tokens_details") or {}).get("thinking_tokens"),
            "cost_usd": record["cost_usd"], "elapsed_seconds": record["elapsed_seconds"],
            "queries": len(chunk_ids), "answered": len(parsed),
        })
    if not calls:
        return None

    answered = ~np.isnan(predictions)  # (q,)
    if task.context_values:
        fill = float(np.median(task.context_values))
    else:
        fill = float(np.median(predictions[answered])) if answered.any() else 0.0
    predictions[~answered] = fill
    return predictions, answered, calls, len(calls) == len(task.chunks)  # (q,), (q,), one record per prompt that ran


def add_methods(tables: Path, methods_by_task: dict[str, dict[str, list[float]]], out: Path) -> None:
    """Write scored tables that hold every row of `tables` and, scored the same way, the methods given by task.

    The measured values and the strata of a task's queries are read from the rows already scored, so no data file
    is needed. A method the tables already hold for a task is refused: nothing scored is replaced.
    """
    predictions = pd.read_csv(tables / "predictions.csv.gz", dtype={"stratum": str})
    metrics = pd.read_csv(tables / "metrics.csv")
    by_task = dict(tuple(predictions.groupby("task_id", sort=False)))
    prediction_rows, metric_rows = [predictions], []
    for task_id, methods in methods_by_task.items():
        assert task_id in by_task, f"the tables hold no task {task_id}"
        scored = by_task[task_id]
        queries = scored[scored["method"] == scored["method"].iloc[0]].sort_values("query_id")
        keys = metrics[metrics["task_id"] == task_id].iloc[0]
        truth, strata = queries["truth"].to_numpy(), queries["stratum"].to_numpy()  # (q,), (q,)
        for method, values in methods.items():
            assert method not in set(scored["method"]), f"{method} is already scored for {task_id}"
            assert len(values) == len(queries), f"{method} has {len(values)} predictions for the {len(queries)} queries of {task_id}"
            method_predictions = np.array(values, dtype=float)  # (q,)
            assert np.isfinite(method_predictions).all(), f"{method} has a prediction for {task_id} that is not a number"
            metric_rows.append({"task_id": task_id, "split": keys["split"], "arm": keys["arm"], "seed": keys["seed"], "k": keys["k"],
                                "decimals": keys["decimals"], "method": method, "queries": len(truth), "answered_fraction": 1.0,
                                **score(truth, method_predictions, strata)})
            prediction_rows.append(queries.assign(method=method, prediction=method_predictions, answered=True))

    out.mkdir(parents=True, exist_ok=True)
    pd.concat(prediction_rows, ignore_index=True).to_csv(out / "predictions.csv.gz", index=False)
    pd.concat([metrics, pd.DataFrame(metric_rows)], ignore_index=True).to_csv(out / "metrics.csv", index=False)
    for name in ("calls.csv", "incomplete_tasks.txt"):
        shutil.copyfile(tables / name, out / name)
    print(f"added {len(metric_rows)} method scores over {len(methods_by_task)} tasks to the tables of {tables}, in {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--responses", type=Path)
    parser.add_argument("--baselines", type=Path)
    parser.add_argument("--tables", type=Path, help="scored tables to copy with the methods of --add joined on")
    parser.add_argument("--add", type=Path, help="a file of predictions by task and method, in each task's query order")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.tables is not None or args.add is not None:
        assert args.tables is not None and args.add is not None, "--tables and --add go together"
        add_methods(args.tables, json.loads(args.add.read_text(encoding="utf-8")), args.out)
        return

    assert args.data is not None and args.tasks is not None and args.responses is not None, "--data, --tasks and --responses are required"
    args.out.mkdir(parents=True, exist_ok=True)
    infos = {split_info.name: split_info for split_info in plan.ALL_SPLITS}
    model_folders = sorted(path for path in args.responses.iterdir() if path.is_dir() and "__" in path.name)
    loaded: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    prediction_rows, metric_rows, call_rows = [], [], []
    skipped: list[str] = []
    for task in tqdm(read_tasks(args.tasks), desc="collect", mininterval=20):
        if task.split not in loaded:
            split_info = infos[task.split]
            frame = flip2.load_split(flip2.split_path(args.data, split_info))
            loaded[task.split] = (frame["target"].to_numpy(), strata_of(split_info, load_layout(split_info, frame)))
        targets, strata = loaded[task.split]  # (n,), (n,)
        query_rows = np.array(task.query_rows)  # (q,)
        truth, query_strata = targets[query_rows], strata[query_rows]  # (q,), (q,)

        by_method: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for folder in model_folders:
            found = reply_predictions(task, folder)
            if found is None:
                continue
            predictions, answered, calls, complete = found
            call_rows.extend(calls)
            # A task the budget cut short is left out, so a prompt that never ran is not scored as a wrong answer.
            # The subagent arm asks only the first prompt of some tasks by design, and is compared on those queries.
            if complete or is_harness(folder.name.split("__")[-1]):
                by_method[folder.name.replace("__", "@")] = (predictions, answered)
            else:
                skipped.append(f"{folder.name} {task.task_id}")
        if args.baselines is not None and (args.baselines / f"{task.task_id}.json").exists():
            baselines = json.loads((args.baselines / f"{task.task_id}.json").read_text(encoding="utf-8"))
            reference = json.loads((args.baselines / f"reference__{task.split.replace('/', '-')}.json").read_text(encoding="utf-8"))
            baselines["ridge_onehot_full"] = reference["ridge_onehot_full"]
            full_path = args.baselines / f"full__{task.split.replace('/', '-')}.json"
            if full_path.exists():
                baselines.update(json.loads(full_path.read_text(encoding="utf-8")))
            for method, values in baselines.items():
                by_method[method] = (np.array(values), np.ones(len(values), dtype=bool))

        for method, (predictions, answered) in by_method.items():
            metric_rows.append({"task_id": task.task_id, "split": task.split, "arm": task.arm, "seed": task.seed,
                                "k": task.k, "decimals": task.decimals, "method": method, "queries": len(truth),
                                "answered_fraction": float(answered.mean()), **score(truth, predictions, query_strata)})
            prediction_rows.append(pd.DataFrame({
                "task_id": task.task_id, "method": method, "query_id": task.query_ids, "row": query_rows,
                "stratum": query_strata, "truth": truth, "prediction": predictions, "answered": answered}))

    pd.concat(prediction_rows).to_csv(args.out / "predictions.csv.gz", index=False)
    pd.DataFrame(metric_rows).to_csv(args.out / "metrics.csv", index=False)
    pd.DataFrame(call_rows).to_csv(args.out / "calls.csv", index=False)
    (args.out / "incomplete_tasks.txt").write_text("\n".join(skipped) + ("\n" if skipped else ""), encoding="utf-8")
    print(f"wrote {len(metric_rows)} metric rows and {len(call_rows)} call rows to {args.out}; "
          f"{len(skipped)} tasks left out because not every prompt ran")


if __name__ == "__main__":
    main()
