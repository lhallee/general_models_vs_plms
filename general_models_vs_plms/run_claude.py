"""Send a stage's tasks to one Claude model at one effort.

    python -m general_models_vs_plms.run_claude --tasks DIR --out DIR --model claude-fable-5-2 --effort low \
        --budget 150 [--arms k256,k256_cipher] [--splits amylase/by_mutation] [--seeds 0] [--workers 8] [--dry-run]

Tasks run in parallel and a task's chunks run in order, so the chunks after the first read the cached prefix.
"""

from __future__ import annotations

import argparse
import sys
import anthropic

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm

from .claude_calls import PRICES, BudgetExhausted, Prompt, SpendLedger, run_prompt, worst_case_cost
from .tasks import Task, read_tasks


EXPECTED_OUTPUT_TOKENS = {"low": 3000, "medium": 12000, "high": 30000, "xhigh": 50000, "max": 64000}


def select_tasks(tasks: list[Task], arms: list[str] | None, splits: list[str] | None, seeds: list[int] | None) -> list[Task]:
    """Tasks that pass the filters, ordered by the arm list when one is given so earlier arms are sent first."""
    chosen = [task for task in tasks
              if (arms is None or task.arm in arms) and (splits is None or task.split in splits)
              and (seeds is None or task.seed in seeds)]
    if arms is not None:
        chosen.sort(key=lambda task: (arms.index(task.arm), task.seed, task.split))
    return chosen


def task_prompts(task: Task) -> list[Prompt]:
    """A task's requests, one per chunk."""
    return [Prompt(f"{task.task_id}__c{index}", task.prefix, chunk) for index, chunk in enumerate(task.chunks)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(PRICES), required=True)
    parser.add_argument("--effort", choices=sorted(EXPECTED_OUTPUT_TOKENS), required=True)
    parser.add_argument("--budget", type=float, required=True, help="USD, over every ledger in the output folder")
    parser.add_argument("--arms", type=lambda text: text.split(","))
    parser.add_argument("--splits", type=lambda text: text.split(","))
    parser.add_argument("--seeds", type=lambda text: [int(seed) for seed in text.split(",")])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tasks = select_tasks(read_tasks(args.tasks), args.arms, args.splits, args.seeds)
    expected = EXPECTED_OUTPUT_TOKENS[args.effort]
    prompts = [prompt for task in tasks for prompt in task_prompts(task)]
    ceiling = sum(worst_case_cost(args.model, prompt, expected) for prompt in prompts)
    print(f"{len(tasks)} tasks, {len(prompts)} prompts, worst case {ceiling:.2f} USD for {args.model} at {args.effort}")
    if args.dry_run:
        return

    folder = args.out / f"{args.model}__{args.effort}"
    folder.mkdir(parents=True, exist_ok=True)
    ledger = SpendLedger(args.out, f"{args.model}__{args.effort}", args.budget)
    client = anthropic.Anthropic(max_retries=8, timeout=1800.0)
    progress = tqdm(total=len(prompts), desc=f"{args.model} {args.effort}", mininterval=20, file=sys.stderr)

    def run_task(task: Task) -> list[str]:
        outcomes = []
        for prompt in task_prompts(task):
            outcomes.append(run_prompt(client, args.model, args.effort, prompt, folder, ledger, expected))
            progress.update(1)
        return outcomes

    outcomes: list[str] = []
    exhausted = False
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in [pool.submit(run_task, task) for task in tasks]:
            try:
                outcomes.extend(future.result())
            except BudgetExhausted as stop:
                exhausted = True
                print(f"BUDGET {stop}", flush=True)
    progress.close()

    counts = {outcome: outcomes.count(outcome) for outcome in ("sent", "kept", "failed")}
    print(f"done {args.model} {args.effort}: {counts}, ledger {ledger.state['usd']:.2f} USD, folder total {ledger.total_usd():.2f} USD")
    if exhausted or counts["failed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
