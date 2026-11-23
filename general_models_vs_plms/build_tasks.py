"""Write a stage's tasks and a manifest of their prompt digests.

    python -m general_models_vs_plms.build_tasks --data DIR --out DIR --stage pilot
    python -m general_models_vs_plms.build_tasks --data DIR --out DIR --stage main --representation substitutions
    python -m general_models_vs_plms.build_tasks --data DIR --out DIR --stage memory
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path
from tqdm import tqdm

from . import flip2, plan
from .tasks import build_task, load_layout, prompt_digests, write_task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "main", "memory"), required=True)
    parser.add_argument("--representation", choices=("substitutions", "sequence"), default="substitutions",
                        help="main stage: the text form for splits that have a parent, chosen by the pilot")
    args = parser.parse_args()

    cells = {"pilot": plan.pilot_cells, "memory": plan.memory_cells,
             "main": lambda: plan.main_cells(args.representation)}[args.stage]()
    folder = args.out / args.stage
    loaded = {}
    manifest = {}
    for cell in tqdm(cells, desc=f"{args.stage} tasks"):
        if cell.split_info.name not in loaded:
            frame = flip2.load_split(flip2.split_path(args.data, cell.split_info))
            loaded[cell.split_info.name] = (frame, load_layout(cell.split_info, frame))
        frame, layout = loaded[cell.split_info.name]
        task = build_task(cell.split_info, frame, layout, cell.arm, cell.seed, cell.part, cell.query_count)
        write_task(task, folder)
        manifest[task.task_id] = {"split": task.split, "arm": task.arm, "seed": task.seed, "k": task.k,
                                  "queries": len(task.query_rows), "representation": task.representation,
                                  "prompt_characters": len(task.prefix) + sum(len(chunk) for chunk in task.chunks),
                                  "prompt_sha256": prompt_digests(task)}

    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"wrote {len(manifest)} tasks to {folder}")


if __name__ == "__main__":
    main()
