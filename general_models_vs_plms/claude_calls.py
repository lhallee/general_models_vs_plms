"""Send prompts to a Claude model through the Messages API and keep a record of every reply.

A record is written once per prompt and never overwritten, so a rerun resumes. No fallback model is configured: a
refusal is recorded as a refusal, and the record names the model that served the request. Spending is metered from
the usage each reply reports, in one ledger file per model and effort, and a call that would pass the budget summed
over every ledger in the folder is not sent.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import anthropic

from dataclasses import dataclass
from pathlib import Path


MAX_TOKENS = 64000
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.1
# USD per million input and output tokens, from the claude-api reference cached on 2026-11-03.
PRICES = {
    "claude-fable-5-2": (8.0, 40.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.5, 2.5),
}


class BudgetExhausted(RuntimeError):
    """The next call could take the spend past the budget."""


@dataclass(frozen=True)
class Prompt:
    """One request: a cacheable prefix, the text after it, and where its record goes."""

    record_id: str
    prefix: str
    suffix: str


def call_cost(model: str, usage: dict[str, object]) -> float:
    """USD for one reply, from its reported token counts."""
    input_price, output_price = PRICES[model]
    fresh = int(usage.get("input_tokens") or 0)
    written = int(usage.get("cache_creation_input_tokens") or 0)
    read = int(usage.get("cache_read_input_tokens") or 0)
    generated = int(usage.get("output_tokens") or 0)
    input_cost = (fresh + written * CACHE_WRITE_MULTIPLIER + read * CACHE_READ_MULTIPLIER) * input_price
    return (input_cost + generated * output_price) / 1e6


def worst_case_cost(model: str, prompt: Prompt, expected_output_tokens: int) -> float:
    """An upper estimate for a call before it is sent: one token a character, written to the cache."""
    input_price, output_price = PRICES[model]
    characters = len(prompt.prefix) + len(prompt.suffix)
    return (characters * CACHE_WRITE_MULTIPLIER * input_price + expected_output_tokens * output_price) / 1e6


class SpendLedger:
    """Running spend of one model and effort, saved after every call, with a budget shared across ledgers."""

    def __init__(self, folder: Path, name: str, budget_usd: float) -> None:
        self.folder = folder
        self.path = folder / f"spend__{name}.json"
        self.budget_usd = budget_usd
        self.lock = threading.Lock()
        self.state = {"usd": 0.0, "calls": 0, "pending_usd": 0.0}
        if self.path.exists():
            self.state = {**json.loads(self.path.read_text(encoding="utf-8")), "pending_usd": 0.0}

    def total_usd(self) -> float:
        """Spend over every ledger in the folder, with this ledger's calls in flight counted at their estimate."""
        others = 0.0
        for path in self.folder.glob("spend__*.json"):
            if path != self.path:
                others += float(json.loads(path.read_text(encoding="utf-8"))["usd"])
        return others + self.state["usd"] + self.state["pending_usd"]

    def reserve(self, estimate_usd: float) -> None:
        """Hold an estimate for a call about to be sent, or refuse it."""
        with self.lock:
            if self.total_usd() + estimate_usd > self.budget_usd:
                raise BudgetExhausted(f"spend {self.total_usd():.2f} USD plus {estimate_usd:.2f} passes {self.budget_usd:.2f}")
            self.state["pending_usd"] += estimate_usd

    def settle(self, estimate_usd: float, actual_usd: float) -> None:
        """Replace a held estimate by what the call cost."""
        with self.lock:
            self.state["pending_usd"] -= estimate_usd
            self.state["usd"] += actual_usd
            self.state["calls"] += int(actual_usd > 0)
            saved = {"usd": self.state["usd"], "calls": self.state["calls"]}
            self.path.write_text(json.dumps(saved), encoding="utf-8")


def send(client: anthropic.Anthropic, model: str, effort: str, prompt: Prompt) -> dict[str, object]:
    """One streamed request, returned as a plain record."""
    content = [{"type": "text", "text": prompt.prefix, "cache_control": {"type": "ephemeral"}},
               {"type": "text", "text": prompt.suffix}]
    started = time.monotonic()
    with client.messages.stream(model=model, max_tokens=MAX_TOKENS, messages=[{"role": "user", "content": content}],
                                extra_body={"output_config": {"effort": effort}}) as stream:
        message = stream.get_final_message()
        # The request id is a convenience for support, and where a stream keeps it has moved between SDK versions.
        request_id = getattr(stream, "request_id", None)

    details = getattr(message, "stop_details", None)
    return {
        "record_id": prompt.record_id, "model_requested": model, "model_served": message.model, "effort": effort,
        "stop_reason": message.stop_reason, "stop_category": getattr(details, "category", None),
        "block_types": [block.type for block in message.content],
        "text": "".join(block.text for block in message.content if block.type == "text"),
        "usage": message.usage.model_dump(mode="json"), "request_id": request_id,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }


def run_prompt(
    client: anthropic.Anthropic, model: str, effort: str, prompt: Prompt, folder: Path, ledger: SpendLedger,
    expected_output_tokens: int,
) -> str:
    """Send a prompt unless its record exists; returns `kept`, `sent` or `failed`."""
    path = folder / f"{prompt.record_id}.json"
    if path.exists():
        return "kept"

    estimate = worst_case_cost(model, prompt, expected_output_tokens)
    ledger.reserve(estimate)
    try:
        record = send(client, model, effort, prompt)
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as error:
        ledger.settle(estimate, 0.0)
        print(f"FAILED {prompt.record_id}: {type(error).__name__}: {str(error)[:300]}", flush=True)
        return "failed"

    record["cost_usd"] = call_cost(model, record["usage"])
    ledger.settle(estimate, record["cost_usd"])
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(record), encoding="utf-8")
    partial.replace(path)
    return "sent"


def main() -> None:
    """Send one short prompt and print its record, to check the request path against the installed SDK before a stage."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--model", choices=sorted(PRICES), default="claude-haiku-5-5")
    parser.add_argument("--effort", default="low")
    args = parser.parse_args()

    client = anthropic.Anthropic(max_retries=2, timeout=120.0)
    record = send(client, args.model, args.effort, Prompt("smoke", "This is a connection check.", " Reply with the single word ready."))
    record["cost_usd"] = call_cost(args.model, record["usage"])
    print(json.dumps(record, indent=1))
    assert record["model_served"].startswith(args.model), f"{record['model_served']} served a request for {args.model}"
    assert record["stop_reason"] == "end_turn" and record["text"].strip(), "the check prompt got no plain reply"
    print(f"SMOKE OK anthropic {anthropic.__version__}")


if __name__ == "__main__":
    main()
