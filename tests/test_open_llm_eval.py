"""Unit tests for open-weight LLM reply parsing and median fill logic."""

from __future__ import annotations

from general_models_vs_plms.open_llm_eval import predictions_from_reply, strip_thinking_blocks
from general_models_vs_plms.tasks import Task


def _dummy_task(context_values: list[float]) -> Task:
    return Task(
        task_id="t0",
        split="fgfr2/random",
        arm="k256",
        seed=0,
        part="test",
        k=len(context_values),
        representation="substitutions",
        described=False,
        cipher=False,
        labels="true",
        decimals=2,
        context_rows=list(range(len(context_values))),
        context_values=context_values,
        query_rows=[10, 11, 12],
        query_ids=["Q001", "Q002", "Q003"],
        prefix="",
        chunks=[""],
    )


def test_strip_thinking_and_median_fill() -> None:
    raw = "<think>Let me reason: Q001\t99.0 is wrong.</think>\nQ001\t1.5\nQ003\t-2.0\n"
    assert strip_thinking_blocks(raw).startswith("Q001\t1.5")
    task = _dummy_task([1.0, 2.0, 5.0])  # median = 2.0
    preds, answered, total = predictions_from_reply(task, raw, chunk_index=0)
    assert (answered, total) == (2, 3)
    assert preds == [1.5, 2.0, -2.0]


def test_zero_shot_fill_uses_answered_median() -> None:
    task = _dummy_task([])
    raw = "Q001\t10.0\nQ002\t30.0\n"
    preds, answered, total = predictions_from_reply(task, raw, chunk_index=0)
    assert (answered, total) == (2, 3)
    assert preds == [10.0, 30.0, 20.0]


def test_three_column_and_inline_fallbacks() -> None:
    task = _dummy_task([1.0, 2.0, 5.0])
    raw_3col = "Q001\tY126H\t-0.396\nQ002\tF185S\t0.512\n"
    preds, answered, total = predictions_from_reply(task, raw_3col, chunk_index=0)
    assert (answered, total) == (2, 3)
    assert preds == [-0.396, 0.512, 2.0]

    raw_inline = "Q001\t5\tQ002\t3\tQ003\t1"
    preds_in, ans_in, _ = predictions_from_reply(task, raw_inline, chunk_index=0)
    assert ans_in == 3
    assert preds_in == [5.0, 3.0, 1.0]
