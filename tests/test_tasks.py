"""Task construction: draws, label transformations, prompt text and the reply parser."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pathlib import Path

from general_models_vs_plms import plan
from general_models_vs_plms.encoding import substitution_text
from general_models_vs_plms.flip2 import SplitInfo
from general_models_vs_plms.prompts import build_chunk, build_prefix, format_value, parse_reply
from general_models_vs_plms.tasks import (
    Arm, build_task, draw_context_rows, draw_query_rows, label_decimals, load_layout, prompt_digests, read_tasks,
    shown_values, write_task,
)


INFO = SplitInfo("toy", "split", "number", "single_parent", "The protein is a toy. Higher is better.")
LETTERS = np.array(list("ACDEFGHIKLMNPQRSTVWY"))


def toy_frame(rows: int = 400) -> pd.DataFrame:
    generator = np.random.default_rng(1)
    parent = np.array(list("MKTAYIAKQRQISFVKSHFSRQ"))
    sequences = []
    for _ in range(rows):
        variant = parent.copy()
        for column in generator.choice(len(parent), size=generator.integers(1, 4), replace=False):
            variant[column] = generator.choice(LETTERS[parent[column] != LETTERS])
        sequences.append("".join(variant))
    part = np.where(np.arange(rows) < 250, "train", np.where(np.arange(rows) < 300, "validation", "test"))
    return pd.DataFrame({"sequence": sequences, "target": generator.normal(size=rows).round(5), "part": part})


def test_contexts_nest_and_queries_come_from_the_requested_part() -> None:
    frame = toy_frame()
    small = draw_context_rows(frame, INFO.name, 16, seed=0)
    large = draw_context_rows(frame, INFO.name, 64, seed=0)
    other_seed = draw_context_rows(frame, INFO.name, 64, seed=1)
    queries = draw_query_rows(frame, INFO.name, "test", 50)

    assert large[:16].tolist() == small.tolist()
    assert set(large.tolist()) != set(other_seed.tolist())
    assert (frame["part"].to_numpy()[large] == "train").all()
    assert (frame["part"].to_numpy()[queries] == "test").all()
    assert len(set(queries.tolist())) == 50
    assert len(draw_context_rows(frame, INFO.name, 9999, seed=0)) == 250


def test_label_decimals_follow_the_interquartile_range() -> None:
    assert label_decimals(np.array([0.0, 1.0, 2.0, 3.0])) == 0
    assert label_decimals(np.linspace(0.0, 0.21, 50)) == 3
    assert label_decimals(np.linspace(-0.034, 0.042, 50)) == 4
    assert label_decimals(np.linspace(-3.0, 1.0, 50)) == 2


def test_label_transformations_keep_the_shown_values_as_a_set_or_reflect_them() -> None:
    targets = np.array([0.12, 0.5, 0.91, 0.33])
    generator = np.random.default_rng(0)

    assert shown_values(targets, 2, "true", generator) == ["0.12", "0.50", "0.91", "0.33"]
    assert sorted(shown_values(targets, 2, "shuffled", generator)) == ["0.12", "0.33", "0.50", "0.91"]
    assert shown_values(targets, 2, "flipped", generator) == ["0.91", "0.53", "0.12", "0.70"]
    with pytest.raises(ValueError, match="unknown label"):
        shown_values(targets, 2, "reversed", generator)


def test_a_task_is_reproducible_holds_no_query_label_and_shares_queries_across_arms(tmp_path: Path) -> None:
    frame = toy_frame()
    layout = load_layout(INFO, frame)
    plain = build_task(INFO, frame, layout, Arm("k64", 64), 0, "test", 100)
    again = build_task(INFO, frame, layout, Arm("k64", 64), 0, "test", 100)
    hidden = build_task(INFO, frame, layout, Arm("k64_cipher", 64, cipher=True), 0, "test", 100)
    described = build_task(INFO, frame, layout, Arm("k64_described", 64, described=True), 0, "test", 100)

    assert prompt_digests(plain) == prompt_digests(again)
    assert plain.query_rows == hidden.query_rows == described.query_rows
    assert plain.context_rows == hidden.context_rows
    assert plain.prefix != hidden.prefix
    assert INFO.description in described.prefix and INFO.description not in plain.prefix
    assert len(plain.chunks) == 1 and plain.k == 64
    query_lines = [line for line in plain.chunks[0].splitlines() if line.startswith("Q")]
    assert len(query_lines) == 100
    for identifier, row, line in zip(plain.query_ids, plain.query_rows, query_lines, strict=True):
        assert line.split("\t") == [identifier, substitution_text(layout, row)]
    assert not set(plain.context_rows) & set(plain.query_rows)

    write_task(plain, tmp_path)
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    assert read_tasks(tmp_path) == [plain]


def test_an_arm_cannot_name_the_protein_and_encipher_it() -> None:
    frame = toy_frame()
    with pytest.raises(ValueError, match="cannot both"):
        build_task(INFO, frame, load_layout(INFO, frame), Arm("bad", 8, described=True, cipher=True), 0, "test", 10)


def test_zero_shot_prompt_says_the_values_are_withheld_and_asks_for_scores() -> None:
    frame = toy_frame()
    task = build_task(INFO, frame, load_layout(INFO, frame), Arm("k0_described", 0, described=True), 0, "test", 10)

    assert task.k == 0 and task.context_values == []
    assert "values are withheld" in task.prefix and "Measured variants" not in task.prefix
    assert "Score each variant" in task.chunks[0] and "your score as a plain number" in task.chunks[0]


def test_recall_prompt_names_where_the_values_are_published_and_asks_for_them() -> None:
    frame = toy_frame()
    layout = load_layout(INFO, frame)
    recall = build_task(INFO, frame, layout, Arm("k0_recall", 0, described=True, recall=True), 0, "test", 10)
    scored = build_task(INFO, frame, layout, Arm("k0_described", 0, described=True), 0, "test", 10)

    assert recall.query_rows == scored.query_rows and recall.context_values == []
    assert "from a published dataset" in recall.prefix
    assert "The measurements are published in the FLIP2 benchmark, as the split file toy/split.csv.gz." in recall.prefix
    assert "Write the measured value that the dataset reports" in recall.chunks[0] and "Score each" not in recall.chunks[0]
    assert "published" not in scored.prefix
    elsewhere = SplitInfo("toy", "split", "number", "single_parent", "A toy.", "a registry, as entry 7")
    assert "The measurements are published in a registry, as entry 7." in build_task(
        elsewhere, frame, layout, Arm("k0_recall", 0, described=True, recall=True), 0, "test", 10).prefix
    with pytest.raises(ValueError, match="recall arm"):
        build_task(INFO, frame, layout, Arm("bad", 8, described=True, recall=True), 0, "test", 10)


def test_the_selected_effort_is_the_lowest_within_the_tolerance_of_the_best() -> None:
    assert plan.select_effort({"low": 0.50, "medium": 0.53, "high": 0.56, "xhigh": 0.575, "max": 0.57}) == "high"
    assert plan.select_effort({"low": 0.56, "medium": 0.50, "high": 0.575}) == "low"
    assert plan.select_effort({"low": 0.40, "max": 0.50}) == "max"


def test_prompt_text_for_sequences_and_pairs() -> None:
    prefix = build_prefix("sequence", True, None, None, [("GEED:KNYK", "0.5")])
    assert prefix.startswith("Below are measurements for protein sequences. Each sequence has two chains")
    assert prefix.endswith("GEED:KNYK\t0.5")
    chunk = build_chunk("sequence", [("Q001", "GEAD:KNYK")], has_measurements=True)
    assert "Q001\tGEAD:KNYK" in chunk and "Predict the measured value of each sequence below." in chunk


def test_format_value_has_no_negative_zero() -> None:
    assert format_value(-0.0001, 3) == "0.000"
    assert format_value(-0.0006, 3) == "-0.001"
    assert format_value(525.0, 0) == "525"


def test_parse_reply_reads_separators_and_ignores_noise() -> None:
    reply = "Here you go\n```\nQ001\t0.25\nQ002 -1.5e-1\nQ003, 3\n**Q004: .5**\nQ005\tnan\nQ001\t9\nQ099\t1\nQ006 = 7 units\n```"

    parsed = parse_reply(reply, ["Q001", "Q002", "Q003", "Q004", "Q005", "Q006"])

    assert parsed == {"Q001": 0.25, "Q002": -0.15, "Q003": 3.0, "Q004": 0.5}
