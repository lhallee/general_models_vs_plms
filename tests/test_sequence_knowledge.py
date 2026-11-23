"""Masks, prompt files, and the scoring of filled-in sequences."""

from __future__ import annotations

import json
import numpy as np

from pathlib import Path

from general_models_vs_plms import sequence_knowledge
from general_models_vs_plms.encoding import AMINO_ACIDS
from general_models_vs_plms.sequence_knowledge import LEVELS, MASK, Entry


def toy_entries(count: int = 7) -> list[Entry]:
    """Sequences of different lengths, the last one not asked of a language model."""
    generator = np.random.default_rng(3)
    sequences = ["".join(generator.choice(list(AMINO_ACIDS), size=70 + 11 * index)) for index in range(count)]
    return [Entry(f"toy/{index}", "parent", sequence, asked=index < count - 1) for index, sequence in enumerate(sequences)]


def test_masks_are_reproducible_and_hide_the_stated_share() -> None:
    entry = Entry("toy/x", "parent", "ACDEFGHIKLMNPQRSTVWY" * 5 + "X", asked=True)

    for level in LEVELS:
        positions = sequence_knowledge.masked_positions(entry, level)  # (m,)
        assert np.array_equal(positions, sequence_knowledge.masked_positions(entry, level))
        assert len(positions) == round(level * 100) and len(set(positions.tolist())) == len(positions)
        assert all(entry.sequence[position] != "X" for position in positions)
        hidden = sequence_knowledge.corrupted(entry.sequence, positions)
        assert hidden.count(MASK) == len(positions) and len(hidden) == len(entry.sequence)


def test_shuffling_keeps_composition_and_changes_the_order() -> None:
    sequence = toy_entries(1)[0].sequence

    assert sorted(sequence_knowledge.shuffled(sequence)) == sorted(sequence)
    assert sequence_knowledge.shuffled(sequence) != sequence
    assert sequence_knowledge.shuffled(sequence) == sequence_knowledge.shuffled(sequence)


def test_each_asked_sequence_meets_each_level_once_and_once_a_file(tmp_path: Path) -> None:
    entries = toy_entries()
    index = sequence_knowledge.write_prompts(entries, tmp_path)

    assert len(index) == sequence_knowledge.BATCHES * len(LEVELS)
    seen: dict[str, list[float]] = {}
    for item in index:
        names = [meaning["name"] for meaning in item["labels"].values()]
        assert len(names) == len(set(names))
        for meaning in item["labels"].values():
            seen.setdefault(meaning["name"], []).append(meaning["level"])
        text = (tmp_path / item["folder"] / "prompt.txt").read_text(encoding="utf-8")
        assert "toy" not in text
    assert set(seen) == {entry.name for entry in entries if entry.asked}
    assert all(sorted(levels) == sorted(LEVELS) for levels in seen.values())


def test_a_true_answer_scores_every_mask_and_a_piece_of_the_wrong_length_scores_none(tmp_path: Path) -> None:
    entries = toy_entries()
    by_name = {entry.name: entry for entry in entries}
    index = sequence_knowledge.write_prompts(entries, tmp_path)
    (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")
    first = index[0]
    lines = []
    for label, meaning in first["labels"].items():
        parts = sequence_knowledge.pieces(by_name[meaning["name"]].sequence)
        lines += [f"{label}.{number:02d}\t{piece}" for number, piece in enumerate(parts, start=1)]
    # The first piece of the answer comes back one residue short.
    lines[0] = lines[0][:-1]
    (tmp_path / first["folder"] / "answer.txt").write_text("\n".join(lines), encoding="utf-8")

    table = sequence_knowledge.score(entries, tmp_path)

    assert len(table) == len(first["labels"]) and set(table["prompt"]) == {first["folder"]}
    whole = table[table["misaligned_pieces"] == 0]
    assert (whole["correct"] == whole["masked"]).all() and (table["altered_pieces"] == 0).all()
    short = table[table["misaligned_pieces"] == 1]
    assert len(short) == 1 and int(short["correct"].iloc[0]) < int(short["masked"].iloc[0])
    assert (table["unigram_correct"] <= table["masked"]).all()
