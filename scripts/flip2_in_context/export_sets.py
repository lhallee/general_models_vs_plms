"""Export the prompt sets of the subagent route that came after the first export.

    py -X utf8 export_sets.py --state STATE --work WORK

STATE is the run's local folder, with `flip2/` and `tasks/` as the project's `build_tasks` wrote them. WORK receives the
prompt folders under `w/` and their indexes under `w_private/`. Sets `a` (k256 and k256_cipher, every prompt), `b`
(the other arms, first prompt) and `c` (sequence recall) come from `subagent_arm`'s own export. This adds `d1` to `d5`
(the three validation prompts, one folder per effort level), `e1` to `e5` (the first k256 prompt of each task, one
folder per effort level), `g` (value recall) and `h` (dataset recognition), and writes `w_private/sets.json`, which
says which effort each set is asked at. Run with the package and the folder that holds `foundry` on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path

from general_models_vs_plms import plan, recognition_probe, subagent_arm


PILOT_TASKS = {"amylase-by_mutation__k256_substitutions__s0", "ired-two_to_many__k256_substitutions__s0",
               "hydro-three_to_many__k256_sequence__s0"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    arguments = parser.parse_args()
    state, prompts, private = arguments.state, arguments.work / "w", arguments.work / "w_private"

    sets = [{"set": "a", "effort": "selected"}, {"set": "b", "effort": "selected"}, {"set": "c", "effort": "selected"}]
    for number, level in enumerate(plan.EFFORT_LEVELS, start=1):
        name = f"d{number}"
        entries = subagent_arm.export(state / "tasks" / "pilot", prompts / name, private / f"index_{name}.json",
                                      {"k256_substitutions", "k256_sequence"}, None, {0}, 1, PILOT_TASKS)
        assert len(entries) == 3, name
        sets.append({"set": name, "effort": level})
    for number, level in enumerate(plan.EFFORT_LEVELS, start=1):
        name = f"e{number}"
        entries = subagent_arm.export(state / "tasks" / "main", prompts / name, private / f"index_{name}.json",
                                      {"k256"}, None, {0}, 1)
        assert len(entries) == 18, name
        sets.append({"set": name, "effort": level})
    entries = subagent_arm.export(state / "tasks" / "memory", prompts / "g", private / "index_g.json", {"k0_recall"}, None, {0}, 1)
    assert len(entries) == 18
    sets.append({"set": "g", "effort": "selected"})
    entries = recognition_probe.export(state / "flip2", state / "tasks" / "main", prompts / "h", private / "index_h.json")
    assert len(entries) == 8
    sets.append({"set": "h", "effort": "selected"})
    (private / "sets.json").write_text(json.dumps(sets, indent=1), encoding="utf-8")
    print(f"wrote {len(sets)} sets to {private / 'sets.json'}")


if __name__ == "__main__":
    main()
