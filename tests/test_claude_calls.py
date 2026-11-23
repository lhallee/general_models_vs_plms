"""Cost metering, the shared budget and the write-once record, against a stand-in client."""

from __future__ import annotations

import json
import pytest

from contextlib import contextmanager
from types import SimpleNamespace

from general_models_vs_plms.claude_calls import BudgetExhausted, Prompt, SpendLedger, call_cost, run_prompt, worst_case_cost
from general_models_vs_plms.run_claude import select_tasks, task_prompts
from general_models_vs_plms.tasks import Task


class StandInUsage:
    def model_dump(self, mode: str) -> dict[str, object]:
        return {"input_tokens": 1000, "cache_creation_input_tokens": 2000, "cache_read_input_tokens": 4000,
                "output_tokens": 500, "output_tokens_details": {"thinking_tokens": 0}}


class StandInClient:
    """Answers every stream with one text block and counts the requests."""

    def __init__(self, stop_reason: str = "end_turn") -> None:
        self.requests: list[dict[str, object]] = []
        self.stop_reason = stop_reason
        self.messages = SimpleNamespace(stream=self.stream)

    @contextmanager
    def stream(self, **request: object):
        self.requests.append(request)
        message = SimpleNamespace(
            model=request["model"], stop_reason=self.stop_reason, stop_details=None, usage=StandInUsage(),
            content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text="Q001\t0.5")],
            _request_id="req_stand_in")
        yield SimpleNamespace(get_final_message=lambda: message)


def test_call_cost_prices_fresh_cached_and_generated_tokens() -> None:
    usage = StandInUsage().model_dump(mode="json")

    assert call_cost("claude-fable-5-2", usage) == pytest.approx((1000 * 8 + 2000 * 8 * 1.25 + 4000 * 8 * 0.1 + 500 * 40) / 1e6)
    assert call_cost("claude-opus-5-5", usage) == pytest.approx(call_cost("claude-fable-5-2", usage) / 2)
    assert worst_case_cost("claude-fable-5-2", Prompt("a", "x" * 1000, "y" * 1000), 100) == pytest.approx((2000 * 10 + 100 * 40) / 1e6)


def test_run_prompt_writes_one_record_and_never_resends(tmp_path) -> None:
    client = StandInClient()
    ledger = SpendLedger(tmp_path, "claude-fable-5-2__low", budget_usd=1.0)
    prompt = Prompt("task__c0", "prefix", "suffix")

    assert run_prompt(client, "claude-fable-5-2", "low", prompt, tmp_path, ledger, 100) == "sent"
    assert run_prompt(client, "claude-fable-5-2", "low", prompt, tmp_path, ledger, 100) == "kept"

    assert len(client.requests) == 1
    request = client.requests[0]
    assert request["extra_body"] == {"output_config": {"effort": "low"}}
    assert "fallbacks" not in request and "system" not in request and "tools" not in request
    assert request["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    record = json.loads((tmp_path / "task__c0.json").read_text(encoding="utf-8"))
    assert record["text"] == "Q001\t0.5" and record["model_served"] == "claude-fable-5-2"
    assert record["cost_usd"] == pytest.approx(ledger.state["usd"]) and ledger.state["calls"] == 1
    assert ledger.state["pending_usd"] == pytest.approx(0.0)


def test_ledgers_in_one_folder_share_the_budget(tmp_path) -> None:
    client = StandInClient()
    first = SpendLedger(tmp_path, "claude-fable-5-2__low", budget_usd=0.06)
    assert run_prompt(client, "claude-fable-5-2", "low", Prompt("a__c0", "p", "s"), tmp_path, first, 100) == "sent"

    second = SpendLedger(tmp_path, "claude-opus-5-5__low", budget_usd=0.06)
    assert second.total_usd() == pytest.approx(first.state["usd"])
    with pytest.raises(BudgetExhausted):
        run_prompt(client, "claude-opus-5-5", "low", Prompt("b__c0", "p", "s"), tmp_path, second, 1000)
    assert len(client.requests) == 1

    resumed = SpendLedger(tmp_path, "claude-fable-5-2__low", budget_usd=0.06)
    assert resumed.state["usd"] == pytest.approx(first.state["usd"]) and resumed.state["calls"] == 1


def make_task(arm: str, split: str, seed: int) -> Task:
    return Task(f"{split.replace('/', '-')}__{arm}__s{seed}", split, arm, seed, "test", 4, "substitutions", False, False,
                "true", 2, [0], [0.1], [1, 2], ["Q001", "Q002"], "prefix", ["chunk a", "chunk b"])


def test_select_tasks_filters_and_orders_by_the_arm_list() -> None:
    tasks = [make_task("k256", "a/x", 0), make_task("k256_cipher", "a/x", 0), make_task("k256", "b/y", 1), make_task("k16", "a/x", 0)]

    chosen = select_tasks(tasks, ["k256_cipher", "k256"], None, [0])

    assert [task.arm for task in chosen] == ["k256_cipher", "k256"]
    assert [prompt.record_id for prompt in task_prompts(chosen[0])] == ["a-x__k256_cipher__s0__c0", "a-x__k256_cipher__s0__c1"]
    assert len(select_tasks(tasks, None, ["b/y"], None)) == 1
