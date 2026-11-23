"""Join tasks, model replies and baselines with the measured values, and write the tidy tables.

    python -m general_models_vs_plms.collect --data DIR --tasks DIR --responses DIR --out DIR [--baselines DIR]

`predictions.csv.gz` has one row per query and method, `metrics.csv` one row per task and method, and `calls.csv`
one row per model request.
"""

from __future__ import annotations

import argparse
import json
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--baselines", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

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
