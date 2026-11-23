"""The prompts the reduced plan asks, in order of priority, with each prompt's instruction and effort.

    py -X utf8 launch_order.py --work WORK SELECTED_LEVEL

Reads each set's index under `WORK/w_private/` and writes `order.json` there. The reduced plan, set after the usage
limit was hit, asks the first prompt of each task only and puts the controls and probes on one split per dataset
(`plan.PROBE_SPLITS`). The sweep and the first k256 prompts at low effort were answered before the limit and stay in
the order for the record. One fresh subagent is started per item with the item's `prompt` as its only instruction and
its `effort` as its effort level. Run with the package and the folder that holds `foundry` on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path

from general_models_vs_plms import plan


LEVELS = list(plan.EFFORT_LEVELS)
INSTRUCTION = (
    "Read the file {path}\\prompt.txt in full. It holds a task and says how to reply. Write your reply, exactly in the "
    "form the task asks for and with nothing else, to {path}\\answer.txt with the Write tool. Use no other tools: do not "
    "run code or commands, search, or open any other file. If the Read tool does not return the whole file, read the "
    "rest before you answer. When the file is written, reply with the single word done.")
# Stage, set, arm (None for any), effort (None for the selected level), and whether only the probe splits are asked.
STAGES = [("sweep_low", "d1", None, "low", False), ("sweep_medium", "d2", None, "medium", False),
          ("sweep_high", "d3", None, "high", False), ("k256_low", "e1", None, "low", False),
          ("cipher", "a", "k256_cipher", None, True), ("recall", "g", None, None, True),
          ("recognition", "h", None, None, False), ("flipped", "b", "k256_flipped", None, True),
          ("k0_described", "b", "k0_described", None, True), ("k256_high", "e3", None, "high", True)]


def split_of(task_id: str) -> str:
    """A task's split, from its id."""
    return task_id.split("__")[0].replace("-", "/", 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("selected", choices=LEVELS)
    arguments = parser.parse_args()
    private, selected = arguments.work / "w_private", arguments.selected

    items = []
    for stage, set_name, arm, effort, probes_only in STAGES:
        index = json.loads((private / f"index_{set_name}.json").read_text(encoding="utf-8"))
        folder = Path(index["prompts"])
        for entry in index["entries"]:
            parts = entry["task_id"].split("__")
            if entry["chunk"] != 0 or (arm is not None and parts[1] != arm):
                continue
            if probes_only and split_of(entry["task_id"]) not in plan.PROBE_SPLITS:
                continue
            items.append({"key": f"{folder.name}/{entry['folder']}", "stage": stage, "effort": effort or selected,
                          "task": entry["task_id"], "description": f"Answer prompt {folder.name}-{entry['folder']}",
                          "prompt": INSTRUCTION.format(path=str(folder / entry["folder"]))})
    (private / "order.json").write_text(json.dumps(items, indent=1), encoding="utf-8")
    counts: dict[str, int] = {}
    for item in items:
        counts[f"{item['stage']}@{item['effort']}"] = counts.get(f"{item['stage']}@{item['effort']}", 0) + 1
    print(len(items), counts)


if __name__ == "__main__":
    main()
