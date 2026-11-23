"""Does a model recognize a benchmark dataset from the unnamed rows a prompt shows? One question per dataset.

    python -m general_models_vs_plms.recognition_probe export --data DIR --tasks DIR --out DIR --index FILE

A question shows what the main 256-example prompt shows of a dataset, cut to its first rows, and asks which protein
it is and where the measurements were published. The prompt names neither. The answers are read by a person against
each dataset's description, since a protein or a study can be named in many ways. A model that names the dataset has
seen it, so leaving the name out of a prompt hides nothing from that model.
"""

from __future__ import annotations

import argparse
import hashlib
import json

from pathlib import Path

from . import flip2, plan
from .encoding import parent_text, variant_text
from .prompts import build_prefix, format_value
from .tasks import Task, load_layout, read_tasks


SHOWN_ROWS = 30
QUESTION = (
    "Which protein do these measurements describe, what was measured, and in which study or dataset were they "
    "published? Reply with three lines that start with Protein:, Measurement: and Source:. Where you do not know, "
    "write unknown."
)


def probe_text(task: Task, data_folder: Path) -> str:
    """The first measured rows of a task as its prompt writes them, followed by the question."""
    split_info = plan.split_infos((task.split,))[0]
    frame = flip2.load_split(flip2.split_path(data_folder, split_info))
    layout = load_layout(split_info, frame)
    rows, values = task.context_rows[:SHOWN_ROWS], task.context_values[:SHOWN_ROWS]
    measured = [(variant_text(layout, int(row), task.representation), format_value(value, task.decimals))
                for row, value in zip(rows, values, strict=True)]
    parent = parent_text(layout) if task.representation == "substitutions" else None
    return build_prefix(task.representation, layout.kind == "pair", None, parent, measured) + "\n\n" + QUESTION


def export(data_folder: Path, tasks_folder: Path, out: Path, index_path: Path) -> list[dict[str, object]]:
    """One question per dataset, from its first split's 256-example task, as numbered prompt files with their index."""
    entries, seen = [], set()
    for task in read_tasks(tasks_folder):
        dataset = task.split.split("/")[0]
        if task.arm != "k256" or task.seed != 0 or dataset in seen:
            continue
        seen.add(dataset)
        text = probe_text(task, data_folder)
        folder = out / f"p{len(entries) + 1:03d}"
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "prompt.txt").write_text(text, encoding="utf-8", newline="\n")
        entries.append({"folder": folder.name, "task_id": f"recognize__{dataset}", "chunk": 0, "record_id": f"recognize__{dataset}",
                        "split": task.split, "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps({"prompts": str(out), "entries": entries}, indent=1), encoding="utf-8")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    exporter = commands.add_parser("export")
    exporter.add_argument("--data", type=Path, required=True)
    exporter.add_argument("--tasks", type=Path, required=True)
    exporter.add_argument("--out", type=Path, required=True)
    exporter.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()

    entries = export(args.data, args.tasks, args.out, args.index)
    print(f"wrote {len(entries)} questions to {args.out} and their index to {args.index}")


if __name__ == "__main__":
    main()
