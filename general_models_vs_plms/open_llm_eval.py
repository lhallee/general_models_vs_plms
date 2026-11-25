"""Zero-API-spend open-weight general LLM evaluation on FLIP2 and expanded post-cutoff tasks.

Runs an open-weight instruction/reasoning CausalLM (e.g., Qwen2.5-14B-Instruct or Qwen3-14B) locally inside Docker on
a GPU machine using the exact prompt format (`Task.prefix` + `Task.chunks[0]`, first 100 queries) and reply parser
(`prompts.parse_reply`) used for Claude Mythos 5.2. Applies the exact same median fill rule for unanswered queries
and writes a `*_methods.json` file compatible with `esmc_adaptation.evaluate` for paired bootstrap comparisons.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import numpy as np
import torch

from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts import parse_reply
from .tasks import CHUNK_SIZE, Task, read_tasks


INLINE_PAIR_RE = re.compile(r"\b(Q\d+)\s*[\t ,:;|=]\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")
THREE_COL_LINE_RE = re.compile(r"^(Q\d+)\s+[^\s]+\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$")


def strip_thinking_blocks(text: str) -> str:
    """Remove `<think>...</think>` reasoning spans before parsing final `Q###\\tvalue` predictions."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1].strip()
    return text


def predictions_from_reply(task: Task, reply_text: str, chunk_index: int = 0) -> tuple[list[float], int, int]:
    """Parse one chunk's predictions with the exact median-fill rule of `collect.reply_predictions`."""
    chunk_ids = task.query_ids[chunk_index * CHUNK_SIZE : (chunk_index + 1) * CHUNK_SIZE]
    cleaned = strip_thinking_blocks(reply_text)
    parsed = parse_reply(cleaned, chunk_ids)
    # Fallback to full reply text if model wrote answers inside or before a tag
    if len(parsed) < len(chunk_ids) and cleaned != reply_text:
        parsed_full = parse_reply(reply_text, chunk_ids)
        if len(parsed_full) > len(parsed):
            parsed = parsed_full
    # Fallback when an open LLM echoes the variant column (Q###\tVARIANT\tVALUE) or writes inline pairs
    if not parsed:
        expected = set(chunk_ids)
        for raw_line in cleaned.splitlines():
            match = THREE_COL_LINE_RE.match(raw_line.strip().strip("`*|").strip())
            if match is not None:
                identifier, value = match.group(1), float(match.group(2))
                if identifier in expected and identifier not in parsed and math.isfinite(value):
                    parsed[identifier] = value
    if not parsed:
        expected = set(chunk_ids)
        for identifier, raw_val in INLINE_PAIR_RE.findall(cleaned):
            value = float(raw_val)
            if identifier in expected and identifier not in parsed and math.isfinite(value):
                parsed[identifier] = value

    values = np.full(len(chunk_ids), np.nan, dtype=float)  # (q,)
    for offset, identifier in enumerate(chunk_ids):
        if identifier in parsed:
            values[offset] = parsed[identifier]

    answered_mask = ~np.isnan(values)  # (q,)
    if task.context_values:
        fill = float(np.median(task.context_values))
    else:
        fill = float(np.median(values[answered_mask])) if answered_mask.any() else 0.0
    values[~answered_mask] = fill
    return values.tolist(), int(answered_mask.sum()), len(chunk_ids)


def format_chat_prompt(tokenizer: object, user_text: str, enable_thinking: bool = False) -> str:
    """Apply the model's chat template cleanly, passing `enable_thinking` only when supported."""
    messages = [{"role": "user", "content": user_text}]
    try:
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        )
    except TypeError:
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )


@torch.inference_mode()
def run_open_llm_tasks(
    model_repo: str,
    revision: str | None,
    method_tag: str,
    tasks_dir: Path,
    out_dir: Path,
    arms: set[str],
    splits_filter: set[str] | None,
    max_new_tokens: int = 2048,
    enable_thinking: bool = False,
) -> None:
    """Run greedy decoding of an open-weight CausalLM on selected tasks (first 100-query chunk)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / f"raw_{method_tag}"
    raw_dir.mkdir(parents=True, exist_ok=True)

    tasks = [t for t in read_tasks(tasks_dir) if t.arm in arms and t.seed == 0]
    if splits_filter:
        tasks = [t for t in tasks if t.split in splits_filter]
    assert tasks, f"no matching tasks found in {tasks_dir} for arms={arms}, splits={splits_filter}"

    rev_kw = {"revision": revision} if revision else {}
    tokenizer = AutoTokenizer.from_pretrained(model_repo, **rev_kw)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_repo,
        dtype=torch.bfloat16,
        **rev_kw,
    ).to(device)
    model.eval()
    resolved_rev = str(getattr(model.config, "_commit_hash", revision or "unknown"))

    methods_path = out_dir / f"open_llm_{method_tag}_methods.json"
    records_path = out_dir / f"open_llm_{method_tag}_records.json"
    methods_by_task: dict[str, dict[str, list[float]]] = (
        json.loads(methods_path.read_text(encoding="utf-8")) if methods_path.exists() else {}
    )
    records: dict[str, dict[str, object]] = (
        json.loads(records_path.read_text(encoding="utf-8")) if records_path.exists() else {}
    )

    for task in tqdm(tasks, desc=f"open_llm:{method_tag}"):
        if task.task_id in records:
            continue
        user_prompt = task.prefix + task.chunks[0]
        formatted = format_chat_prompt(tokenizer, user_prompt, enable_thinking=enable_thinking)
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        elapsed = time.perf_counter() - t0
        new_ids = generated[0, prompt_tokens:]
        reply_text = tokenizer.decode(new_ids, skip_special_tokens=True)
        peak_gb = float(torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0

        preds, answered, total_q = predictions_from_reply(task, reply_text, chunk_index=0)
        methods_by_task.setdefault(task.task_id, {})[method_tag] = preds
        records[task.task_id] = {
            "split": task.split,
            "arm": task.arm,
            "model_repo": model_repo,
            "revision": resolved_rev,
            "enable_thinking": enable_thinking,
            "prompt_tokens": prompt_tokens,
            "output_tokens": int(len(new_ids)),
            "answered": answered,
            "queries": total_q,
            "answered_fraction": round(answered / max(1, total_q), 4),
            "elapsed_seconds": round(elapsed, 2),
            "peak_gpu_gb": round(peak_gb, 3),
        }
        (raw_dir / f"{task.task_id}__c0.json").write_text(
            json.dumps(
                {
                    "task_id": task.task_id,
                    "split": task.split,
                    "arm": task.arm,
                    "model_repo": model_repo,
                    "revision": resolved_rev,
                    "reply_text": reply_text,
                    "answered": answered,
                    "queries": total_q,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        methods_path.write_text(json.dumps(methods_by_task), encoding="utf-8")
        records_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    print(f"wrote {len(records)} open-LLM evaluations to {methods_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="HuggingFace repo id (e.g. Qwen/Qwen2.5-14B-Instruct)")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--tag", required=True, help="Short method name (e.g. qwen25_14b)")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", default="k256,k256_cipher,k256_flipped,k0_described")
    parser.add_argument("--splits", default="")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--enable-thinking", action="store_true")
    args = parser.parse_args()

    arms = {a for a in args.arms.split(",") if a}
    splits = {s for s in args.splits.split(",") if s} or None
    run_open_llm_tasks(
        model_repo=args.model,
        revision=args.revision,
        method_tag=args.tag,
        tasks_dir=args.tasks,
        out_dir=args.out,
        arms=arms,
        splits_filter=splits,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=args.enable_thinking,
    )


if __name__ == "__main__":
    main()
