"""Tasks: which labeled rows a model sees, which rows it must predict, and the exact prompt text.

A task holds no label of a query row. Every draw is seeded by the split's name and the task's seed, so a rebuild on
another machine gives the same prompts, which `prompt_digests` lets a reader check.
"""

from __future__ import annotations

import hashlib
import json
import zlib
import numpy as np
import pandas as pd

from dataclasses import asdict, dataclass
from pathlib import Path

from .encoding import Layout, apply_cipher, build_layout, draw_cipher, parent_text, variant_text
from .flip2 import SplitInfo
from .prompts import build_chunk, build_prefix, format_value


CHUNK_SIZE = 100
PART_CODES = {"train": 0, "validation": 1, "test": 2}


@dataclass(frozen=True)
class Arm:
    """One experimental condition. `k` is the number of measured rows shown, 0 for none.

    `labels` is `true`, `shuffled` (the shown values permuted among the shown rows) or `flipped` (each shown value
    reflected about the middle of the shown range). `cipher` relabels the residues and reorders the columns. `recall`
    shows no rows, names the protein and where its measurements are published, and asks for the published values.
    """

    name: str
    k: int
    described: bool = False
    cipher: bool = False
    labels: str = "true"
    representation: str = "substitutions"
    recall: bool = False


@dataclass(frozen=True)
class Task:
    """One arm of one split under one seed, with its prompt as a cacheable prefix and query chunks."""

    task_id: str
    split: str
    arm: str
    seed: int
    part: str
    k: int
    representation: str
    described: bool
    cipher: bool
    labels: str
    decimals: int
    context_rows: list[int]
    context_values: list[float]
    query_rows: list[int]
    query_ids: list[str]
    prefix: str
    chunks: list[str]


def name_code(name: str) -> int:
    """A stable integer for a split's name, for seeding."""
    return zlib.crc32(name.encode("utf-8"))


def label_decimals(train_targets: np.ndarray) -> int:
    """Decimals that give a split's labels about three digits across their interquartile range; 0 for integers."""
    # train_targets: (n,)
    if np.all(train_targets == np.round(train_targets)):
        return 0

    low, high = np.quantile(train_targets, [0.25, 0.75])
    spread = high - low
    assert spread > 0, "a split's training labels have no interquartile spread"
    return int(np.clip(2 - np.floor(np.log10(spread)), 0, 4))


def draw_context_rows(frame: pd.DataFrame, name: str, k: int, seed: int) -> np.ndarray:
    """The first `k` training rows of a seeded order, so a smaller context is a subset of a larger one."""
    train_rows = np.flatnonzero(frame["part"].to_numpy() == "train")  # (n_train,)
    order = np.random.default_rng([seed, name_code(name), 1]).permutation(train_rows)  # (n_train,)
    return order[:min(k, len(order))]  # (k,)


def draw_query_rows(frame: pd.DataFrame, name: str, part: str, count: int) -> np.ndarray:
    """A seeded sample of one part's rows, in the random order the prompt lists them."""
    pool = np.flatnonzero(frame["part"].to_numpy() == part)  # (n_part,)
    order = np.random.default_rng([name_code(name), 2, PART_CODES[part]]).permutation(pool)  # (n_part,)
    return order[:min(count, len(order))]  # (q,)


def shown_values(targets: np.ndarray, decimals: int, labels: str, generator: np.random.Generator) -> list[str]:
    """The context values as text, after the arm's label transformation."""
    # targets: (k,)
    rounded = np.round(targets, decimals)  # (k,)
    if labels == "shuffled":
        rounded = generator.permutation(rounded)  # (k,)
    elif labels == "flipped":
        rounded = rounded.min() + rounded.max() - rounded  # (k,)
    elif labels != "true":
        raise ValueError(f"unknown label transformation {labels!r}")
    return [format_value(float(value), decimals) for value in rounded]


def build_task(
    split_info: SplitInfo, frame: pd.DataFrame, layout: Layout, arm: Arm, seed: int, part: str, query_count: int,
) -> Task:
    """Draw a task's rows and write its prompt."""
    if arm.described and arm.cipher:
        raise ValueError("an arm cannot both name the protein and encipher it")
    if arm.recall and (arm.k or not arm.described):
        raise ValueError("the recall arm names the protein and shows no measured row")

    representation = "sequence" if layout.kind == "sequences" else arm.representation
    context_rows = draw_context_rows(frame, split_info.name, arm.k, seed)  # (k,)
    query_rows = draw_query_rows(frame, split_info.name, part, query_count)  # (q,)
    assert not set(context_rows.tolist()) & set(query_rows.tolist()), "a query row is also a context row"

    targets = frame["target"].to_numpy()  # (n,)
    decimals = label_decimals(targets[frame["part"].to_numpy() == "train"])
    label_generator = np.random.default_rng([seed, name_code(split_info.name), 3])
    values = shown_values(targets[context_rows], decimals, arm.labels, label_generator) if arm.k else []

    shown = layout
    if arm.cipher:
        shown = apply_cipher(layout, *draw_cipher(layout, seed=name_code(split_info.name) + seed))

    measured = [(variant_text(shown, int(row), representation), value) for row, value in zip(context_rows, values, strict=True)]
    query_ids = [f"Q{index + 1:03d}" for index in range(len(query_rows))]
    queries = [(identifier, variant_text(shown, int(row), representation))
               for identifier, row in zip(query_ids, query_rows, strict=True)]
    parent = parent_text(shown) if representation == "substitutions" else None
    prefix = build_prefix(representation, layout.kind == "pair", split_info.description if arm.described else None,
                          parent, measured, split_info.source if arm.recall else None)
    chunks = [build_chunk(representation, queries[start:start + CHUNK_SIZE], has_measurements=bool(measured), recall=arm.recall)
              for start in range(0, len(queries), CHUNK_SIZE)]

    return Task(
        task_id=f"{split_info.dataset}-{split_info.split}__{arm.name}__s{seed}", split=split_info.name, arm=arm.name, seed=seed,
        part=part, k=len(context_rows), representation=representation, described=arm.described, cipher=arm.cipher,
        labels=arm.labels, decimals=decimals, context_rows=context_rows.tolist(),
        context_values=[float(value) for value in values], query_rows=query_rows.tolist(), query_ids=query_ids,
        prefix=prefix, chunks=chunks,
    )


def prompt_digests(task: Task) -> list[str]:
    """SHA-256 of each full prompt of a task, one per chunk."""
    return [hashlib.sha256((task.prefix + chunk).encode("utf-8")).hexdigest() for chunk in task.chunks]


def write_task(task: Task, folder: Path) -> Path:
    """Save a task as JSON and return its path."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{task.task_id}.json"
    path.write_text(json.dumps(asdict(task)), encoding="utf-8")
    return path


def read_task(path: Path) -> Task:
    """Load a task saved by `write_task`."""
    return Task(**json.loads(path.read_text(encoding="utf-8")))


def read_tasks(folder: Path) -> list[Task]:
    """Every task in a folder, by name; a task file is named by its identifier, which ends in a seed."""
    return [read_task(path) for path in sorted(folder.glob("*__s[0-9]*.json"))]


def load_layout(split_info: SplitInfo, frame: pd.DataFrame) -> Layout:
    """The padded layout of a split's sequences."""
    return build_layout(frame["sequence"].tolist(), split_info.layout)
