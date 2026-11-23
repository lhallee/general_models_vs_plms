"""Audit the subagents, turn their answer files into records, and score them: rerun at any point of the launch.

    py -X utf8 score_route.py --state STATE --work WORK --transcripts TRANSCRIPTS [--route ROUTE] [--collect]

STATE is the run's local folder: `flip2/`, `tasks/`, and `final/hub/` as pulled from the run's Hub dataset. WORK holds
the prompt sets (`w/`) and their indexes (`w_private/`, with `sets.json` and, once the level is chosen on validation
rows, `selected.txt`). TRANSCRIPTS is the coding agent's folder of subagent transcripts for the session that asked the
prompts. One audit per set is written under `WORK/w_private/audits/`, the records under ROUTE (`STATE/route` unless
given), and with `--collect` the scored tables. A prompt no subagent was given yet gets no record, so a set that is
still being asked is read as far as it has got. Run with the package and the folder that holds `foundry` on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import pandas as pd

from collections import Counter
from pathlib import Path

from general_models_vs_plms import plan, subagent_arm


MODEL = "claude-mythos-5-2"
API_FOLDERS = ("claude-fable-5-2__high", "claude-fable-5-2__low", "claude-haiku-5-5__low", "claude-opus-5-5__high",
               "claude-opus-5-5__low", "claude-opus-5__low")
# Where a set's records go: the validation sweep, the scored arms, or the probes a person or a summarizer reads.
DESTINATION = {"d": "responses_pilot", "c": "probes", "h": "probes"}
# The reduced plan asks no test prompt at medium, extra high or maximum effort: the one medium answer written before
# the usage limit stays unscored.
NOT_SCORED = ("e2", "e4", "e5")


def module(name: str, *arguments: str) -> None:
    """Run one of the project's modules and stop on a failure."""
    subprocess.run([sys.executable, "-X", "utf8", "-m", f"general_models_vs_plms.{name}", *arguments], check=True)


def audit_and_ingest(private: Path, route: Path, transcripts: Path) -> None:
    """Audit every set that has a level, write its audit, and turn its standing answers into records."""
    selected_path = private / "selected.txt"
    selected = selected_path.read_text(encoding="utf-8").strip() if selected_path.exists() else None
    subagents = subagent_arm.read_transcripts(transcripts)
    (private / "audits").mkdir(exist_ok=True)
    for one in json.loads((private / "sets.json").read_text(encoding="utf-8")):
        name = one["set"]
        level = selected if one["effort"] == "selected" else one["effort"]
        if level is None or name in NOT_SCORED:
            continue
        index_path, audit_path = private / f"index_{name}.json", private / "audits" / f"audit_{name}.json"
        rows = subagent_arm.audit(index_path, subagents)
        if not rows:
            continue
        audit_path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        latest = {row["prompt"]: row for row in rows}
        models, efforts = Counter(row["models"] for row in rows), Counter(row["efforts"] for row in rows)
        breaches = [row for row in latest.values() if row["breaches"]]
        unwritten = [row["prompt"] for row in latest.values() if not row["answer_file"]]
        responses = route / DESTINATION.get(name[0], "responses")
        taken, missing = subagent_arm.ingest(index_path, responses, MODEL, level, audit_path, asked_only=True)
        print(f"{name} at {level}: {len(rows)} subagents over {len(latest)} prompts, {taken} answers stand, {missing} do not; "
              f"models {dict(models)}; efforts {dict(efforts)}; breaches {len(breaches)}; no answer file {unwritten}")
        for row in breaches:
            print(f"  breach {row['prompt']} {row['agent']}: {row['breaches'][:200]}")
        wrong = [row["prompt"] for row in rows if not row["api_error"] and (row["models"] != MODEL or row["efforts"] != level)]
        stopped = [row["prompt"] for row in latest.values() if row["api_error"] and not row["answer_file"]]
        if wrong:
            print(f"  NOT {MODEL} at {level}: {wrong}")
        if stopped:
            print(f"  stopped by a failed request, no record: {stopped}")


def collect(state: Path, route: Path) -> None:
    """Score the records: the validation sweep by itself, and the test arms beside the API models and the baselines."""
    if (route / "responses_pilot").exists():
        module("collect", "--data", str(state / "flip2"), "--tasks", str(state / "tasks" / "pilot"),
               "--responses", str(route / "responses_pilot"), "--out", str(route / "tables" / "pilot"))
    if (route / "responses").exists():
        for folder in API_FOLDERS:
            if not (route / "responses" / folder).exists():
                shutil.copytree(state / "final" / "hub" / "responses" / folder, route / "responses" / folder)
        module("collect", "--data", str(state / "flip2"), "--tasks", str(state / "tasks" / "all"),
               "--responses", str(route / "responses"), "--baselines", str(state / "final" / "hub" / "baselines"),
               "--out", str(route / "tables" / "main"))


def report_sweep(private: Path, route: Path) -> None:
    """Each level's Spearman on the validation prompts, and the level the rule picks among the complete ones."""
    selected_path = private / "selected.txt"
    selected = selected_path.read_text(encoding="utf-8").strip() if selected_path.exists() else None
    pilot = pd.read_csv(route / "tables" / "pilot" / "metrics.csv")
    pilot = pilot.assign(level=pilot["method"].str.split("harness-").str[1], split=pilot["task_id"].str.split("__").str[0])
    asked = [level for level in plan.EFFORT_LEVELS if level in set(pilot["level"])]
    wide = pilot.pivot(index="level", columns="split", values="spearman").reindex(asked)
    wide = wide.assign(mean=wide.mean(axis=1), prompts=wide.notna().sum(axis=1))
    print(wide.round(3).to_string())
    complete = wide.drop(columns=["mean", "prompts"]).dropna().mean(axis=1).to_dict()
    print("complete levels:", {level: round(value, 3) for level, value in complete.items()},
          "rule picks:", plan.select_effort(complete), "recorded selection:", selected)
    wide.to_csv(route / "tables" / "effort_validation.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--route", type=Path)
    parser.add_argument("--collect", action="store_true")
    arguments = parser.parse_args()
    private, route = arguments.work / "w_private", arguments.route or arguments.state / "route"

    audit_and_ingest(private, route, arguments.transcripts)
    if arguments.collect:
        collect(arguments.state, route)
        report_sweep(private, route)


if __name__ == "__main__":
    main()
