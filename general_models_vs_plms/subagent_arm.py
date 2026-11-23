"""A model reached only through a coding agent's subagents: write its prompts as files, then read its answers back.

    python -m general_models_vs_plms.subagent_arm export --tasks DIR --out DIR --index FILE --arms k256 [--splits ...]
    python -m general_models_vs_plms.subagent_arm audit --index FILE --transcripts DIR --out FILE
    python -m general_models_vs_plms.subagent_arm ingest --index FILE --responses DIR --model NAME [--effort LEVEL] [--audit FILE]

`export` writes one folder per prompt, named by a number so the path says nothing about the protein or the arm. The
folder holds `prompt.txt`, the text the API arms send, and nothing else. A fresh subagent is given that one path, told
to read it and to write `answer.txt` beside it, and told to use no other tool. The index, which maps each number to
its task, is kept outside the folders the subagent is pointed at. A set of prompts is exported once per effort level
it is asked at, each time to a folder of its own, so the answers of two levels never share a file.

`audit` reads the subagents' transcripts as Claude Code keeps them (`agent-<id>.jsonl` with `agent-<id>.meta.json`)
and lists, for every prompt, which model answered, at which effort, and every tool call it made. A subagent kept the
protocol if it only read its own prompt and wrote its own answer.

`ingest` turns each prompt into a record of the form `run_claude` writes, under `<responses>/<model>__harness-<level>/`
(`<model>__harness` when no level was given), so `collect` scores it like any other method. A prompt with no answer
file gets a record with no text, which scores as unanswered. Run it once every prompt of the index has been tried.
Where only the first `--chunks` chunks of a task were exported, a comparison with that arm keeps those queries for
every method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re

from datetime import datetime
from pathlib import Path

from .tasks import read_tasks


EFFORT = "harness"
REPLY_CALL = "SubagentHandback"
HARNESS_ERROR = "<synthetic>"
PROMPT_PATH = re.compile(r"[\\/]([A-Za-z0-9_]+)[\\/](p\d{3})[\\/]prompt\.txt")


def effort_name(level: str | None) -> str:
    """The effort a record of this route carries: `harness`, or `harness-<level>` when the subagent was given a level."""
    return EFFORT if level is None else f"{EFFORT}-{level}"


def is_harness(effort: str) -> bool:
    """Whether an effort name belongs to this route."""
    return effort == EFFORT or effort.startswith(f"{EFFORT}-")


def export(tasks_folder: Path, out: Path, index_path: Path, arms: set[str], splits: set[str] | None, seeds: set[int],
           chunks: int, task_ids: set[str] | None = None) -> list[dict[str, object]]:
    """Write the prompt files and the index, and return the index."""
    entries = []
    for task in read_tasks(tasks_folder):
        if task.arm not in arms or task.seed not in seeds or (splits is not None and task.split not in splits):
            continue
        if task_ids is not None and task.task_id not in task_ids:
            continue
        for chunk_index, chunk in enumerate(task.chunks[:chunks]):
            text = task.prefix + chunk
            folder = out / f"p{len(entries) + 1:03d}"
            folder.mkdir(parents=True, exist_ok=False)
            (folder / "prompt.txt").write_text(text, encoding="utf-8", newline="\n")
            entries.append({"folder": folder.name, "task_id": task.task_id, "chunk": chunk_index,
                            "record_id": f"{task.task_id}__c{chunk_index}",
                            "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps({"prompts": str(out), "entries": entries}, indent=1), encoding="utf-8")
    return entries


def prompt_key(text: str) -> str | None:
    """The prompt a text points at, as `<set>/<number>`, or None if it names no prompt file."""
    found = PROMPT_PATH.search(text)
    return f"{found.group(1)}/{found.group(2)}" if found else None


def message_text(message: dict[str, object]) -> str:
    """The text of a transcript message, whose content is a string or a list of blocks."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    return "\n".join(block.get("text", "") for block in content or [] if isinstance(block, dict) and block.get("type") == "text")


def audit_transcript(path: Path) -> dict[str, object] | None:
    """One subagent's prompt, model, tool calls and breaches of the protocol; None if it was given no prompt file."""
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    messages = [entry for entry in entries if isinstance(entry.get("message"), dict)]
    asked = next((entry for entry in messages if entry.get("type") == "user"), None)
    key = prompt_key(message_text(asked["message"])) if asked else None
    if key is None:
        return None

    models, efforts, calls, breaches, last_text, api_error = set(), set(), [], [], "", ""
    usage_by_reply: dict[str, dict[str, object]] = {}
    stop_reason, stop_category, handed_back = None, None, False
    for entry in messages:
        if entry.get("type") != "assistant":
            continue
        message = entry["message"]
        # When a request fails, as on a usage limit, the harness writes the error as a reply of a model named so.
        if message.get("model") == HARNESS_ERROR:
            api_error = message_text(message)[:200] or "error"
            continue
        if message.get("model"):
            models.add(message["model"])
        if entry.get("effort"):
            efforts.add(str(entry["effort"]))
        # One reply is stored as several entries, one per content block, each with the reply's whole usage.
        usage_by_reply[str(message.get("id", len(usage_by_reply)))] = message.get("usage") or {}
        if message.get("stop_reason"):
            stop_reason = message["stop_reason"]
            stop_category = (message.get("stop_details") or {}).get("category")
        last_text = message_text(message) or last_text
        for block in message.get("content") or []:
            if not (isinstance(block, dict) and block.get("type") == "tool_use"):
                continue
            # The harness has a subagent hand its final reply back through a call of its own, which is no tool use.
            if block["name"] == REPLY_CALL:
                last_text = str((block.get("input") or {}).get("message", "")) or last_text
                handed_back = True
                continue
            target = str((block.get("input") or {}).get("file_path", ""))
            calls.append(block["name"])
            own_prompt = block["name"] == "Read" and prompt_key(target) == key
            own_answer = block["name"] == "Write" and prompt_key(target.replace("answer.txt", "prompt.txt")) == key \
                and target.endswith("answer.txt")
            if not (own_prompt or own_answer):
                breaches.append(f"{block['name']} {target}".strip())
    stamps = [datetime.fromisoformat(str(entry["timestamp"]).replace("Z", "+00:00")) for entry in entries if entry.get("timestamp")]
    output_tokens = sum(int(usage.get("output_tokens") or 0) for usage in usage_by_reply.values())
    thinking_tokens = sum(int((usage.get("output_tokens_details") or {}).get("thinking_tokens") or 0)
                          for usage in usage_by_reply.values())
    return {"prompt": key, "agent": path.stem, "started": entries[0].get("timestamp"), "models": ",".join(sorted(models)),
            "efforts": ",".join(sorted(efforts)), "tool_calls": ",".join(calls), "breaches": "; ".join(breaches),
            "wrote_answer": "Write" in calls, "replies": len(usage_by_reply), "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens, "seconds": (max(stamps) - min(stamps)).total_seconds() if stamps else None,
            "stop_reason": stop_reason, "stop_category": stop_category, "last_text": last_text[:300], "api_error": api_error,
            "handed_back": handed_back}


def read_transcripts(transcripts: Path) -> list[dict[str, object]]:
    """Every subagent in a folder of transcripts that was given a prompt file, read once for every index."""
    return [found for path in sorted(transcripts.glob("agent-*.jsonl")) if (found := audit_transcript(path)) is not None]


def audit(index_path: Path, subagents: list[dict[str, object]]) -> list[dict[str, object]]:
    """Every subagent that was given a prompt of the index, oldest first, with its model and its tool calls."""
    index = json.loads(index_path.read_text(encoding="utf-8"))
    name = Path(index["prompts"]).name
    by_folder = {f"{name}/{entry['folder']}": entry for entry in index["entries"]}
    rows = []
    for found in subagents:
        if found["prompt"] in by_folder:
            entry = by_folder[found["prompt"]]
            answered = (Path(index["prompts"]) / entry["folder"] / "answer.txt").exists()
            rows.append({**found, "task_id": entry["task_id"], "chunk": entry["chunk"], "answer_file": answered})
    return sorted(rows, key=lambda row: (row["prompt"], str(row["started"])))


def ingest(index_path: Path, responses: Path, model: str, level: str | None = None,
           audit_path: Path | None = None, asked_only: bool = False) -> tuple[int, int]:
    """Write one record per prompt of the index; return how many answers were taken and how many prompts have none.

    With an audit, an answer is taken only if the last subagent given its prompt kept the protocol, and the record
    carries that subagent's token counts and time. `asked_only` leaves out the prompts that no subagent has been
    given yet, for a read of a set that is still being asked.
    """
    assert audit_path or not asked_only, "which prompts were asked is known only from an audit"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    name = Path(index["prompts"]).name
    folder = responses / f"{model}__{effort_name(level)}"
    folder.mkdir(parents=True, exist_ok=True)
    # The audit lists a prompt's subagents oldest first, so the last row of a prompt is the one whose answer stands.
    audited = {row["prompt"]: row for row in json.loads(audit_path.read_text(encoding="utf-8"))} if audit_path else {}
    taken, missing = 0, 0
    for entry in index["entries"]:
        prompt_folder = Path(index["prompts"]) / entry["folder"]
        prompt = (prompt_folder / "prompt.txt").read_text(encoding="utf-8")
        assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == entry["prompt_sha256"], (
            f"{prompt_folder} holds a prompt that changed after export")
        answer_path = prompt_folder / "answer.txt"
        row = audited.get(f"{name}/{entry['folder']}")
        assert not (audit_path and answer_path.exists() and row is None), f"{prompt_folder} has an answer and no transcript"
        if asked_only and row is None:
            continue
        # A subagent stopped by a failed request never answered, and one that has neither handed back nor refused is
        # still at work: neither says anything about the model, so neither leaves a record.
        at_work = bool(row) and not (row["handed_back"] or row["stop_reason"] == "refusal")
        if row and not answer_path.exists() and (row["api_error"] or at_work):
            continue
        accepted = answer_path.exists() and not (row and row["breaches"])
        if accepted:
            stop_reason = "end_turn"
        elif answer_path.exists():
            stop_reason = "protocol_breach"
        else:
            stop_reason = "refusal" if row and row["stop_reason"] == "refusal" else "no_answer"
        usage = {"output_tokens": row["output_tokens"], "output_tokens_details": {"thinking_tokens": row["thinking_tokens"]}} if row else {}
        record = {"record_id": entry["record_id"], "model_requested": model, "model_served": row["models"] if row else model,
                  "effort": effort_name(level), "stop_reason": stop_reason, "stop_category": row["stop_category"] if row else None,
                  "block_types": ["text"], "text": answer_path.read_text(encoding="utf-8") if accepted else "", "usage": usage,
                  "request_id": row["agent"] if row else None, "elapsed_seconds": row["seconds"] if row else None, "cost_usd": 0.0}
        (folder / f"{entry['record_id']}.json").write_text(json.dumps(record), encoding="utf-8")
        taken += accepted
        missing += not accepted
    return taken, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    exporter = commands.add_parser("export")
    exporter.add_argument("--tasks", type=Path, required=True)
    exporter.add_argument("--out", type=Path, required=True)
    exporter.add_argument("--index", type=Path, required=True)
    exporter.add_argument("--arms", required=True, help="comma-separated arm names")
    exporter.add_argument("--splits", help="comma-separated split names; every split by default")
    exporter.add_argument("--seeds", default="0", help="comma-separated seeds")
    exporter.add_argument("--chunks", type=int, default=1, help="how many chunks of each task to ask")
    exporter.add_argument("--task-ids", help="comma-separated task ids; every task of the arms by default")
    auditor = commands.add_parser("audit")
    auditor.add_argument("--index", type=Path, required=True)
    auditor.add_argument("--transcripts", type=Path, required=True)
    auditor.add_argument("--out", type=Path, required=True)
    reader = commands.add_parser("ingest")
    reader.add_argument("--index", type=Path, required=True)
    reader.add_argument("--responses", type=Path, required=True)
    reader.add_argument("--model", required=True)
    reader.add_argument("--effort", help="the effort level the subagents were given, if any")
    reader.add_argument("--audit", type=Path, help="the audit of this index; an answer of a subagent that broke the protocol is left out")
    reader.add_argument("--asked-only", action="store_true", help="leave out prompts no subagent has been given yet")
    args = parser.parse_args()

    if args.command == "export":
        entries = export(args.tasks, args.out, args.index, set(args.arms.split(",")),
                         set(args.splits.split(",")) if args.splits else None,
                         {int(seed) for seed in args.seeds.split(",")}, args.chunks,
                         set(args.task_ids.split(",")) if args.task_ids else None)
        print(f"wrote {len(entries)} prompts to {args.out} and their index to {args.index}")
    elif args.command == "audit":
        rows = audit(args.index, read_transcripts(args.transcripts))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        prompts = {row["prompt"] for row in rows}
        breaches = [row for row in rows if row["breaches"]]
        models = sorted({row["models"] for row in rows})
        print(f"{len(rows)} subagents over {len(prompts)} prompts; models {models}; {len(breaches)} broke the protocol")
        for row in breaches:
            print(f"  {row['prompt']} {row['agent']}: {row['breaches']}")
    else:
        taken, missing = ingest(args.index, args.responses, args.model, args.effort, args.audit, args.asked_only)
        print(f"wrote {taken + missing} records; {missing} prompts have no answer that stands")


if __name__ == "__main__":
    main()
