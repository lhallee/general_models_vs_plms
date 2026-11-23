"""Prompt text for in-context fitness prediction, and the parser for a model's reply.

A prompt is a prefix (the task, the parent and the measured variants) and one chunk of queries. The prefix is the
same for every chunk of a task, so it can be cached. The text names no strategy and no hypothesis.
"""

from __future__ import annotations

import math
import re

from collections.abc import Sequence


SUBSTITUTION_INTRO = (
    "A variant is written as its substitutions relative to the parent sequence. For example, A12V means that "
    "position 12 holds A in the parent and V in the variant. Several substitutions are separated by spaces, and the "
    "word parent means the parent sequence itself."
)
PAIR_SUBSTITUTION_NOTE = (
    "The parent has two chains separated by a colon. Positions in the first chain are numbered from 1, positions in "
    "the second chain are written p1, p2 and so on, and a hyphen means that the residue is absent."
)
PAIR_SEQUENCE_NOTE = "Each sequence has two chains separated by a colon."
REPLY_LINE = re.compile(r"^(Q\d+)\s*[\t ,:;|=]\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)$")


def format_value(value: float, decimals: int) -> str:
    """A measured value at a fixed number of decimals, with no negative zero."""
    text = f"{value:.{decimals}f}"
    return f"{0.0:.{decimals}f}" if float(text) == 0.0 else text


def build_prefix(
    representation: str, is_pair: bool, description: str | None, parent: str | None,
    measured: Sequence[tuple[str, str]], source: str | None = None,
) -> str:
    """The stable part of a prompt: the task, an optional assay description, the parent and the measured variants.

    `source` names where the measurements are published, for the prompt that asks a model to recall them.
    """
    noun = "variant" if representation == "substitutions" else "sequence"
    subject = "variants of one protein" if representation == "substitutions" else "protein sequences"
    if measured:
        opening = f"Below are measurements for {subject}."
    elif source is not None:
        opening = f"Below are {subject} from a published dataset. Each has a measured value there, and the values are withheld."
    else:
        opening = f"Below are {subject}. Each has a measured value, and the values are withheld."
    if representation == "substitutions":
        assert parent is not None, "substitutions need a parent sequence"
        intro = [opening, SUBSTITUTION_INTRO] + ([PAIR_SUBSTITUTION_NOTE] if is_pair else [])
    else:
        intro = [opening] + ([PAIR_SEQUENCE_NOTE] if is_pair else [])

    blocks = [" ".join(intro)]
    if description is not None:
        blocks.append(description)
    if source is not None:
        blocks.append(f"The measurements are published in {source}.")
    if representation == "substitutions":
        blocks.append(f"Parent sequence:\n{parent}")
    if measured:
        lines = "\n".join(f"{text}\t{value}" for text, value in measured)
        blocks.append(f"Measured {noun}s, one per line, written as the {noun}, a tab, and the measured value:\n{lines}")
    return "\n\n".join(blocks)


def build_chunk(representation: str, queries: Sequence[tuple[str, str]], has_measurements: bool, recall: bool = False) -> str:
    """The part of a prompt that changes: one chunk of queries and the reply format.

    `recall` asks for the value the published dataset reports, where a prompt with no measurements otherwise asks
    only for a score that ranks the queries.
    """
    noun = "variant" if representation == "substitutions" else "sequence"
    lines = "\n".join(f"{identifier}\t{text}" for identifier, text in queries)
    if has_measurements:
        ask = f"Predict the measured value of each {noun} below."
        reply = "your predicted value"
    elif recall:
        ask = (f"Write the measured value that the dataset reports for each {noun} below. Where you do not remember "
               "the value, give your best estimate of it.")
        reply = "the measured value"
    else:
        ask = (f"Score each {noun} below so that a {noun} you expect to have a higher measured value gets a higher "
               "score.")
        reply = "your score"
    return (f"\n\n{ask} Each line is an identifier, a tab, and the {noun}:\n{lines}\n\n"
            f"Reply with one line for each identifier, in the same order, written as the identifier, a tab, and "
            f"{reply} as a plain number. Write nothing else.")


def parse_reply(text: str, query_ids: Sequence[str]) -> dict[str, float]:
    """The finite number a reply gives for each expected identifier; the first line naming an identifier wins."""
    expected = set(query_ids)
    predictions: dict[str, float] = {}
    for raw_line in text.splitlines():
        match = REPLY_LINE.match(raw_line.strip().strip("`*|").strip())
        if match is None:
            continue

        identifier, value = match.group(1), float(match.group(2))
        if identifier in expected and identifier not in predictions and math.isfinite(value):
            predictions[identifier] = value
    return predictions
