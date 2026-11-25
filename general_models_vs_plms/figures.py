"""Figures of the scored summaries: one message a figure, a plain label on every row, an interval on every mean.

    python -m general_models_vs_plms.figures --summary DIR --out DIR --primary METHOD [--only NAME ...]

Every figure reads the tables `report` wrote. A row or a point is a method, and its bar is the 95% interval of a
bootstrap over queries. One color system holds throughout: orange is the primary language model, plum the Claude
models reached through the public API, aqua a trained ESMC model, blue ESM-2, olive a generative protein language
model read or tuned on the examples, and gray anything with no trained protein model or trained on permuted values.
Rows run from the controls at the top to the result at the bottom.
"""

from __future__ import annotations

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataclasses import dataclass
from pathlib import Path
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from foundry.plotting.panels import letter_panels
from . import plan
from .baselines import PLM_NAMES


ORANGE, PLUM, ESM2, ESMC, OLIVE, OLIVE_PALE = "#eb6834", "#a03a8c", "#2a78d6", "#1baf7a", "#7a5a00", "#c9ad55"
GRAY, PALE, INK, GRID = "#6b6f76", "#b9bcc2", "#1f2328", "#e6e7ea"
MYTHOS = "claude-mythos-5-2@harness"
API_SELECTED, API_CHEAP = "claude-fable-5-2@high", "claude-fable-5-2@low"
DPI = 300
EFFORT_WORDS = {"low": "low", "medium": "medium", "high": "high", "xhigh": "extra high", "max": "maximum"}
MODEL_LABELS = {
    "claude-fable-5-2@high": "Claude Fable 5.2, API, high effort", "claude-fable-5-2@low": "Claude Fable 5.2, API, low effort",
    "claude-opus-5-5@high": "Claude Opus 5.5, API, high effort", "claude-opus-5-5@low": "Claude Opus 5.5, API, low effort",
    "claude-opus-5@low": "Claude Opus 5, API, low effort", "claude-haiku-5-5@low": "Claude Haiku 5.5, API, low effort",
}
PLM_LABELS = {"esm2_650m": "ESM-2 650M", "esmc_300m": "ESMC-300M", "esmc_600m": "ESMC-600M", "esmc_6b": "ESMC-6B"}
METHOD_LABELS = {
    "ridge_onehot": "Ridge on one-hot sequence", "nearest_neighbor": "Value of the nearest example",
    "mutation_count": "Number of substitutions", "stratum_mean": "Mean of the example's group",
    "ridge_composition": "Ridge on amino acid composition", "ridge_random_vector": "Ridge on a random vector",
    "ridge_esmc_300m_random": "Ridge on untrained ESMC-300M", "ridge_esmc_300m_shuffled": "Ridge on ESMC-300M, residues shuffled",
    "ridge_esmc_300m_permuted": "Ridge on ESMC-300M, positions scrambled", "ridge_esmc_600m_shuffled": "Ridge on ESMC-600M, residues shuffled",
    "ridge_esmc_600m_permuted": "Ridge on ESMC-600M, positions scrambled",
}
TUNED_LABELS = {"progen2_small": "ProGen2-small", "progen2_base": "ProGen2-base"}
TUNED_ARM_LABELS = {
    "zero": "{} score, no examples", "ridge": "Ridge on {} hidden states", "dpo": "{} tuned, ranked preference loss (DPO)",
    "bt": "{} tuned, Bradley-Terry loss", "dpo_shuffled": "{} tuned (DPO) on shuffled values",
    "dpo_seed1": "{} tuned (DPO), second batch order", "dpo_seed2": "{} tuned (DPO), third batch order",
}
# The same arms under a heading that already names the model; the control's heading names what it is, so its row
# names the model.
TUNED_ROW_LABELS = {
    "zero": "Score of the sequence", "ridge": "Ridge on hidden states", "dpo": "Tuned, ranked preference loss (DPO)",
    "bt": "Tuned, Bradley-Terry loss", "dpo_shuffled": "{}, ranked preference loss",
}
TUNED_MARKERS = {"zero": "^", "ridge": "o", "dpo": "h", "bt": "h", "dpo_shuffled": "h", "dpo_seed1": "h", "dpo_seed2": "h"}
# What each variant of the tuning changed, for a row under a heading that names the model. The variants ran after the
# reported scores were seen.
VARIANT_LABELS = {"progen2_small_slow": "Learning rate 0.000001, a tenth of the grid's lowest",
                  "progen2_small_lowrank": "Low-rank update, every released weight frozen",
                  "progen2_small_rows": "Every labeled row of the split, up to 2,048",
                  "progen2_small_long": "One setting, up to 40 epochs, checkpoint chosen",
                  "progen2_small_long20": "One setting, 20 epochs, nothing chosen",
                  "progen2_small_long40": "One setting, 40 epochs, nothing chosen",
                  "progen2_small_nll20": "One setting, 20 epochs, likelihood term of weight 1",
                  "progen2_small_nllmid20": "One setting, 20 epochs, likelihood term of weight 0.1",
                  "progen2_small_nlllight20": "One setting, 20 epochs, likelihood term of weight 0.01"}
DATASET_LABELS = {"amylase": "Amylase", "ired": "IRED", "nucb": "NucB", "trpb": "TrpB", "hydro": "Hydro", "rhomax": "RhoMax",
                  "pdz3": "PDZ3", "fgfr2": "FGFR2"}
ARM_LABELS = {"k256_flipped": "256 examples, values reversed", "k256_shuffled": "256 examples, values shuffled",
              "k0_described": "No examples, protein named", "k0_recall": "No examples, published values asked for",
              "k256_cipher": "256 examples, residues enciphered", "k256_described": "256 examples, protein named",
              "k256": "256 examples"}
SCORE = "Spearman correlation with the measured values"
MINUS = "−"
ESMC_NAMES = ("esmc_300m", "esmc_600m", "esmc_6b")
TIER_CONTROLS = ("ridge_random_vector", "ridge_esmc_600m_shuffled", "ridge_esmc_300m_random", "ridge_composition",
                 "ridge_esmc_600m_permuted", "nearest_neighbor", "ridge_onehot")
CONTROL_MARKERS = {"ridge_random_vector": "X", "ridge_composition": "P", "ridge_esmc_300m_shuffled": "v", "ridge_esmc_600m_shuffled": "v",
                   "ridge_esmc_300m_permuted": "D", "ridge_esmc_600m_permuted": "D", "ridge_esmc_300m_random": "*",
                   "ridge_onehot": "s", "nearest_neighbor": "p"}


@dataclass(frozen=True)
class Row:
    """One row of a dot chart: a label with a mean and its interval, or a group heading with neither."""

    label: str
    mean: float = float("nan")
    low: float = float("nan")
    high: float = float("nan")
    color: str = INK
    marker: str = "o"
    hollow: bool = False
    heading: bool = False


def effort_of(method: str) -> str | None:
    """The effort level a subagent series of Claude Mythos 5.2 ran at, or None for any other method."""
    return method.removeprefix(f"{MYTHOS}-") if method.startswith(f"{MYTHOS}-") else None


def tuned_arm(method: str) -> tuple[str, str] | None:
    """The model and the arm of a preference-tuning method, such as `progen2_small_dpo`, or None for any other method."""
    for model in plan.PREFERENCE_MODELS:
        if method.startswith(f"{model}_") and method.removeprefix(f"{model}_") in plan.PREFERENCE_ARMS:
            return model, method.removeprefix(f"{model}_")
    return None


def label_of(method: str) -> str:
    """A method's name in plain words."""
    if effort_of(method):
        return f"Claude Mythos 5.2, {EFFORT_WORDS[effort_of(method)]} effort"
    found = tuned_arm(method)
    if found:
        return TUNED_ARM_LABELS[found[1]].format(TUNED_LABELS[found[0]])
    if method in MODEL_LABELS:
        return MODEL_LABELS[method]
    if method in METHOD_LABELS:
        return METHOD_LABELS[method]
    for name, label in PLM_LABELS.items():
        if method == f"ridge_{name}":
            return f"Ridge on {label} embedding"
        if method == f"ridge_onehot_plus_{name}":
            return f"Ridge on one-hot + {label} score"
        if method == f"zero_shot_{name}":
            return f"{label} score, no examples"
    raise KeyError(f"no label for {method}")


def color_of(method: str, primary: str) -> str:
    """Orange for the primary language model, plum for other Claude models, aqua and blue for trained protein models."""
    if method == primary or method.startswith(f"{primary}+"):
        return ORANGE
    if method.startswith("claude-"):
        return PLUM
    if any(method.endswith(tag) for tag in ("_random", "_shuffled", "_permuted")) or method in METHOD_LABELS:
        return GRAY
    if tuned_arm(method):
        return OLIVE
    if "esm2" in method:
        return ESM2
    if "esmc" in method:
        return ESMC
    return GRAY


def marker_of(method: str) -> str:
    """Circle for a ridge on an embedding and a language model, square for an augmented ridge, triangle for zero-shot."""
    if method in CONTROL_MARKERS:
        return CONTROL_MARKERS[method]
    found = tuned_arm(method)
    if found:
        return TUNED_MARKERS[found[1]]
    if method.startswith("zero_shot_"):
        return "^"
    if method.startswith("ridge_onehot_plus_"):
        return "s"
    return "o"


def split_label(split: str) -> str:
    """A split as its dataset's name and the benchmark's split name."""
    dataset, name = split.split("/")
    return f"{DATASET_LABELS[dataset]}: {name}"


class Summary:
    """The tables `report` wrote, with the lookups the figures share."""

    def __init__(self, folder: Path, primary: str) -> None:
        self.folder = folder
        self.means = pd.read_csv(folder / "means.csv")
        self.per_split = pd.read_csv(folder / "per_split.csv")
        self.paired = pd.read_csv(folder / "paired.csv")
        self.differences = pd.read_csv(folder / "split_differences.csv")
        self.answered = pd.read_csv(folder / "answered_only.csv")
        self.cost = pd.read_csv(folder / "cost.csv")
        self.metric_means = pd.read_csv(folder / "metric_means.csv")
        self.matches = pd.read_csv(folder / "matches.csv")
        self.matches_per_split = pd.read_csv(folder / "matches_per_split.csv")
        self.effort = pd.read_csv(folder / "effort.csv")
        self.validation = pd.read_csv(folder / "effort_validation.csv") if (folder / "effort_validation.csv").exists() else None
        self.found = json.loads((folder / "findings.json").read_text(encoding="utf-8"))
        self.primary = primary
        self.first = f"first {self.found['partly_asked_queries']}"
        assert primary in self.found["language_models"], f"{primary} is not among {self.found['language_models']}"
        # A primary model that was asked one prompt a task is shown, with every row beside it, on those queries alone.
        self.scope = self.queries_of(primary, "k256")
        self.partial = self.scope != "all"

    def color(self, method: str) -> str:
        """A method's color, with orange kept for the primary language model."""
        return color_of(method, self.primary)

    def best_ridge(self, group: str = "flip2") -> str:
        """The ridge with the highest mean at 256 examples on a group of splits, on the queries the primary model is read on."""
        return self.found["best_ridge_first" if self.partial else "best_ridge_at_k256"][group]

    def queries_of(self, method: str, arm: str) -> str:
        """`all`, or the first queries alone where the series was asked only its first prompt."""
        if f"{method} {arm}" in self.found["partly_asked"]:
            return f"first {self.found['partly_asked_queries']}"
        return "all"

    def scope_words(self, queries: str | None = None) -> str:
        """What an axis label adds where the scores are of the first queries of each split."""
        queries = queries or self.scope
        return "" if queries == "all" else f", {queries} queries of each"

    def mean(self, method: str, arm: str = "k256", group: str = "flip2", seeds: str = "seed 0", queries: str | None = None) -> pd.Series:
        """A series' mean over a group's splits; the series must have run on every one of them."""
        table, queries = self.means, queries or self.scope
        found = table[(table["method"] == method) & (table["arm"] == arm) & (table["group"] == group) & (table["seeds"] == seeds)
                      & (table["queries"] == queries)]
        assert len(found) == 1, f"{method} under {arm} has {len(found)} rows of means for {group}, {queries} queries"
        row = found.iloc[0]
        assert row["splits"] == row["group_splits"], f"{method} under {arm} ran {row['splits']} of {row['group_splits']} {group} splits"
        return row

    def has(self, method: str, arm: str = "k256", group: str = "flip2", queries: str | None = None) -> bool:
        """Whether a series ran on every split of a group."""
        table = self.means
        found = table[(table["method"] == method) & (table["arm"] == arm) & (table["group"] == group)
                      & (table["queries"] == (queries or self.scope))]
        return bool((found["splits"] == found["group_splits"]).any())

    def row(self, method: str, arm: str = "k256", group: str = "flip2", label: str | None = None, hollow: bool = False,
            color: str | None = None, marker: str | None = None, queries: str | None = None) -> Row:
        """A dot-chart row of a series' mean over a group."""
        found = self.mean(method, arm, group, queries=queries)
        return Row(label or label_of(method), found["mean"], found["low"], found["high"], color or self.color(method),
                   marker or marker_of(method), hollow)

    def split_row(self, method: str, split: str, arm: str = "k256", label: str | None = None) -> Row:
        """A dot-chart row of a series' score on one split."""
        table = self.per_split
        found = table[(table["method"] == method) & (table["arm"] == arm) & (table["split"] == split) & (table["seeds"] == "seed 0")
                      & (table["queries"] == self.scope)]
        assert len(found) == 1, f"{method} under {arm} has {len(found)} rows for {split}"
        row = found.iloc[0]
        return Row(label or label_of(method), row["spearman"], row["low"], row["high"], self.color(method), marker_of(method))

    def splits(self, group: str = "flip2") -> int:
        """How many splits a group holds."""
        return len(self.found["groups"][group])


def setup() -> None:
    """One look for every figure."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 9, "legend.fontsize": 8.5, "axes.edgecolor": GRAY, "axes.labelcolor": INK,
        "xtick.color": GRAY, "ytick.color": INK, "text.color": INK, "axes.linewidth": 0.8, "legend.frameon": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def save(figure: Figure, out: Path, name: str) -> None:
    """Write a figure as a PNG at 300 dpi and close it."""
    out.mkdir(parents=True, exist_ok=True)
    figure.savefig(out / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(figure)


def quiet(axis: Axes, grid: str = "x") -> None:
    """A light grid behind the marks and no frame but the value axis."""
    axis.grid(axis=grid, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def draw_mark(axis: Axes, row: Row, y: float, size: float = 7.5) -> None:
    """A row's interval and its marker."""
    axis.plot([row.low, row.high], [y, y], color=row.color, linewidth=1.8, solid_capstyle="round", zorder=2)
    axis.plot([row.mean], [y], marker=row.marker, markersize=size, markerfacecolor="white" if row.hollow else row.color,
              markeredgecolor=row.color, markeredgewidth=1.8, linestyle="none", zorder=3)


def write_value(axis: Axes, row: Row, y: float) -> None:
    """A row's mean as text to the right of its interval."""
    text = f"{row.mean:.2f}"
    axis.annotate("0.00" if text == "-0.00" else text.replace("-", MINUS), (row.high, y), xytext=(6, 0), textcoords="offset points",
                  va="center", ha="left", fontsize=8.5, color=INK)


def write_heading(axis: Axes, label: str, y: float) -> None:
    """A group's name in bold at the left edge of the plot, on a white ground so no grid line runs through it."""
    axis.text(0.0, y - 0.08, label, transform=axis.get_yaxis_transform(), ha="left", va="center", fontsize=8.5,
              fontweight="bold", color=INK, zorder=5, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})


def finish_rows(axis: Axes, positions: list[float], labels: list[str], xlabel: str) -> None:
    """Row labels down the left, the value axis below, and nothing else."""
    axis.set_yticks(positions)
    axis.set_yticklabels(labels)
    axis.tick_params(axis="y", length=0)
    axis.set_ylim(min(positions) - 0.7, max(positions) + 0.7)
    axis.set_xlabel(xlabel)
    quiet(axis)
    axis.spines["left"].set_visible(False)


def dot_chart(axis: Axes, rows: list[Row], xlabel: str) -> None:
    """Rows from top to bottom: each mean as a marker on its interval, with the value written beside it."""
    positions = [-float(index) for index in range(len(rows))]
    marks = [row for row in rows if not row.heading]
    low, high = min(row.low for row in marks), max(row.high for row in marks)
    pad = 0.04 * (high - low)
    for y, row in zip(positions, rows, strict=True):
        if row.heading:
            write_heading(axis, row.label, y)
        else:
            draw_mark(axis, row, y)
            write_value(axis, row, y)
    finish_rows(axis, positions, ["" if row.heading else row.label for row in rows], xlabel)
    axis.set_xlim(min(low - pad, -pad), high + 6 * pad)
    if low < 0:
        axis.axvline(0.0, color=GRAY, linewidth=0.8, zorder=1)


def chart_height(rows: list[Row]) -> float:
    """A figure height that gives every row the same room."""
    return 0.75 + 0.3 * len(rows)


def by_mean(rows: list[Row]) -> list[Row]:
    """Rows from the lowest mean to the highest."""
    return sorted(rows, key=lambda row: row.mean)


def headline_rows(summary: Summary, group: str = "flip2", split: str | None = None) -> list[Row]:
    """The rows of the main comparison at 256 examples, for a group's mean or for one split."""
    def one(method: str) -> Row:
        return summary.split_row(method, split) if split else summary.row(method, group=group)

    zero_shot = max((f"zero_shot_{name}" for name in PLM_NAMES), key=lambda method: summary.mean(method)["mean"])
    ridges = [f"ridge_{name}" for name in PLM_NAMES]
    # The ridge the answer is read against, where it is not one of the ridges on an embedding.
    ridges += [summary.best_ridge()] if summary.best_ridge() not in (*ridges, "ridge_onehot") else []
    # One language model a figure: a model reached through subagents and a model reached through the API are scored on
    # different queries behind different surroundings, so they never share an axis.
    rows = [
        Row("No protein language model, 256 examples", heading=True), one("nearest_neighbor"), one("ridge_onehot"),
        Row("Protein language model, no examples", heading=True), one(zero_shot),
        Row("Protein language model + ridge, 256 examples", heading=True), *(one(method) for method in ridges),
        Row("Language model, 256 examples in its prompt", heading=True), one(summary.primary),
    ]
    if split:
        return rows
    # Within a group, worst first by the mean over the benchmark, the primary model last; a split's panel keeps that order.
    ordered, block = [], []
    for row in rows:
        if row.heading:
            ordered += by_mean(block) + [row]
            block = []
        else:
            block.append(row)
    return ordered + sorted(block, key=lambda row: (row.color == ORANGE, row.mean))


def headline(summary: Summary, out: Path) -> None:
    """The mean over the benchmark's splits at 256 examples: the language model, protein model ridges, and floors."""
    rows = headline_rows(summary)
    figure, axis = plt.subplots(figsize=(7.0, chart_height(rows)))
    dot_chart(axis, rows, f"{SCORE}, mean of {summary.splits()} FLIP2 splits{summary.scope_words()}")
    save(figure, out, "headline")


def postcutoff(summary: Summary, out: Path) -> None:
    """The headline's rows, in its order, on each split of the dataset published after the models' training data ends."""
    order = [row.label for row in headline_rows(summary)]
    splits = summary.found["groups"]["postcutoff"]
    figure, axes = plt.subplots(1, len(splits), figsize=(5.2 * len(splits) + 1.6, 0.75 + 0.3 * len(order)), sharey=True)
    for axis, split in zip(axes, splits, strict=True):
        rows = {row.label: row for row in headline_rows(summary, "postcutoff", split)}
        # Every panel sets the row labels: the shared axis shows them on the first panel only, and blanking them on a
        # later panel would blank them on all.
        dot_chart(axis, [rows[label] for label in order], f"{SCORE}, {split_label(split)}")
    letter_panels(axes)
    figure.subplots_adjust(wspace=0.12)
    save(figure, out, "postcutoff")


def tier_rows(summary: Summary, group: str = "flip2", split: str | None = None) -> list[Row]:
    """Controls, then ESMC with no examples, then ESMC under a ridge, then the language model: one block a tier.

    Inside a block the rows run from the lowest mean over the benchmark to the highest, and a split's panel keeps that
    order, so the same method sits on the same line in every panel.
    """
    def one(method: str) -> Row:
        return summary.split_row(method, split) if split else summary.row(method, group=group)

    blocks = [
        ("Controls, 256 examples", list(TIER_CONTROLS)),
        ("ESMC, no examples", [f"zero_shot_{name}" for name in ESMC_NAMES]),
        ("ESMC + ridge, 256 examples", [f"ridge_{name}" for name in ESMC_NAMES] + [f"ridge_onehot_plus_{name}" for name in ESMC_NAMES]),
        ("General language model, 256 examples in its prompt", [summary.primary]),
    ]
    rows = []
    for heading, methods in blocks:
        rows.append(Row(heading, heading=True))
        rows += [one(method) for method in sorted(methods, key=lambda method: summary.mean(method)["mean"])]
    return rows


def shade_primary(axis: Axes, rows: list[Row]) -> None:
    """A pale band over the primary model's interval, so every other row is read against it."""
    primary = next(row for row in rows if row.color == ORANGE)
    axis.axvspan(primary.low, primary.high, color=ORANGE, alpha=0.10, linewidth=0, zorder=0)


def tiers(summary: Summary, out: Path) -> None:
    """The one figure of the comparison: controls, ESMC three ways, and the language model, on the same queries."""
    rows = tier_rows(summary)
    figure, axis = plt.subplots(figsize=(7.4, chart_height(rows)))
    dot_chart(axis, rows, f"{SCORE}, mean of {summary.splits()} FLIP2 splits{summary.scope_words()}")
    shade_primary(axis, rows)
    save(figure, out, "tiers")


def tiers_both(summary: Summary, out: Path) -> None:
    """The tiers on the benchmark and, in the same order, on the dataset published after the models' training data ends."""
    panels = [("flip2", "FLIP2 splits"), ("postcutoff", "FGFR2 splits published after the training cutoff")]
    figure, axes = plt.subplots(1, 2, figsize=(12.6, chart_height(tier_rows(summary))), sharey=True)
    for axis, (group, words) in zip(axes, panels, strict=True):
        rows = tier_rows(summary, group)
        dot_chart(axis, rows, f"{SCORE},\nmean of {summary.splits(group)} {words}{summary.scope_words()}")
        shade_primary(axis, rows)
    letter_panels(axes)
    figure.subplots_adjust(wspace=0.10)
    save(figure, out, "tiers_both")


def tuned_rows(summary: Summary, group: str = "flip2") -> list[Row]:
    """A tuned protein language model beside what it is read against: its control, itself untuned, a ridge, and the model.

    Inside a block the rows run from the lowest mean over the benchmark to the highest, in every panel. An arm a
    model did not run has no row. Only the first tuned model is drawn, the one whose design was fixed before any
    score was seen; `scale_rows` holds what was run afterward.
    """
    tuned = summary.found["tuned_models"][:1]
    blocks = [("Control: tuned on shuffled values", [f"{model}_dpo_shuffled" for model in tuned])]
    for model in tuned:
        blocks.append((f"{TUNED_LABELS[model]}, no examples", [f"{model}_zero"]))
        blocks.append((f"{TUNED_LABELS[model]}, 256 examples", [f"{model}_ridge", f"{model}_bt", f"{model}_dpo"]))
    blocks.append(("ESMC + ridge, 256 examples", list(dict.fromkeys(summary.best_ridge(name) for name in ("flip2", "postcutoff")))))
    blocks.append(("General language model, 256 examples in its prompt", [summary.primary]))
    rows = []
    for heading, methods in blocks:
        methods = [method for method in methods if summary.has(method)]
        if not methods:
            continue
        rows.append(Row(heading, heading=True))
        for method in sorted(methods, key=lambda method: summary.mean(method)["mean"]):
            # Under a heading that names the model, a row says only what was done with it.
            found = tuned_arm(method)
            label = TUNED_ROW_LABELS[found[1]].format(TUNED_LABELS[found[0]]) if found else None
            rows.append(summary.row(method, group=group, label=label, hollow=method.endswith("_bt")))
    return rows


def preference_tuning(summary: Summary, out: Path) -> None:
    """A generative protein language model tuned on the 256 examples, on the benchmark and on the later dataset."""
    panels = [("flip2", "FLIP2 splits"), ("postcutoff", "FGFR2 splits published after the training cutoff")]
    figure, axes = plt.subplots(1, 2, figsize=(11.4, chart_height(tuned_rows(summary)) + 0.2), sharey=True)
    for axis, (group, words) in zip(axes, panels, strict=True):
        rows = tuned_rows(summary, group)
        # The scope goes on a line of its own, since a panel is narrower here than the label is long.
        scope = summary.scope_words().replace(", ", ",\n", 1)
        dot_chart(axis, rows, f"{SCORE},\nmean of {summary.splits(group)} {words}{scope}")
        shade_primary(axis, rows)
    letter_panels(axes)
    figure.subplots_adjust(wspace=0.10)
    save(figure, out, "preference_tuning")


def scale_rows(summary: Summary, group: str = "flip2") -> list[Row]:
    """What was run after the reported scores were seen, between the reported arm and what it is read against.

    The blocks: the first tuned model as reported, each variant of its tuning, a larger checkpoint, and the reported
    arm's ranks averaged with another method's. A block keeps the order of its rows in every panel, and a run that
    did not finish on every split of the benchmark has no row.
    """
    first, *others = summary.found["tuned_models"]
    ridges = list(dict.fromkeys(summary.best_ridge(name) for name in ("flip2", "postcutoff")))
    name = TUNED_LABELS[first]
    blocks: list[tuple[str, list[tuple[str, str, str, str, bool]]]] = [
        (f"{name}, as reported", [(f"{first}_zero", "k256", "Score of the sequence, no examples", "^", False),
                                  (f"{first}_dpo", "k256", "Tuned on 256 examples", "h", False)]),
        (f"{name} tuned another way", [(f"{model}_dpo", plan.PREFERENCE_TASK_ARMS.get(model, "k256"), VARIANT_LABELS[model], "h", True)
                                       for model in others if model in VARIANT_LABELS]),
    ]
    for model in others:
        if model in TUNED_LABELS:
            blocks.append((f"A larger checkpoint: {TUNED_LABELS[model]}", [
                (f"{model}_zero", "k256", "Score of the sequence, no examples", "^", False),
                (f"{model}_ridge", "k256", "Ridge on hidden states, 256 examples", "o", False),
                (f"{model}_dpo", "k256", "Tuned on 256 examples", "h", False),
                (f"{model}_long20_dpo", "k256", "Tuned, one setting, 20 epochs, nothing chosen", "h", True)]))
    partners = [("orders", "its two other batch orders"), (f"{first}_zero", "its own score with no examples"),
                *((ridge, label_of(ridge)) for ridge in ridges),
                ("zero_shot_esmc_600m", label_of("zero_shot_esmc_600m")), (summary.primary, label_of(summary.primary))]
    blocks.append((f"Tuned {name}, ranks averaged with", [(f"{first}_dpo+{other}", "k256", label[0].upper() + label[1:], "D", False)
                                                         for other, label in partners]))
    rows = []
    for heading, entries in blocks:
        entries = [entry for entry in entries if summary.has(entry[0], entry[1])]
        if not entries:
            continue
        rows.append(Row(heading, heading=True))
        rows += [summary.row(method, arm, group, label=label, hollow=hollow, color=OLIVE, marker=marker)
                 for method, arm, label, marker, hollow in entries]
    rows.append(Row("ESMC + ridge, 256 examples", heading=True))
    rows += [summary.row(ridge, group=group) for ridge in ridges]
    rows.append(Row("General language model, 256 examples in its prompt", heading=True))
    rows.append(summary.row(summary.primary, group=group))
    return rows


def preference_scale(summary: Summary, out: Path) -> None:
    """Other scales of the tuning, a larger checkpoint, and rank averages, beside the reported arm and its references."""
    panels = [("flip2", "FLIP2 splits"), ("postcutoff", "FGFR2 splits published after the training cutoff")]
    figure, axes = plt.subplots(1, 2, figsize=(12.0, chart_height(scale_rows(summary)) + 0.2), sharey=True)
    for axis, (group, words) in zip(axes, panels, strict=True):
        rows = scale_rows(summary, group)
        scope = summary.scope_words().replace(", ", ",\n", 1)
        dot_chart(axis, rows, f"{SCORE},\nmean of {summary.splits(group)} {words}{scope}")
        shade_primary(axis, rows)
    letter_panels(axes)
    figure.subplots_adjust(wspace=0.10)
    save(figure, out, "preference_scale")


def preference_by_split(summary: Summary, out: Path) -> None:
    """Split by split, the tuned protein language model beside the language model and the best ridge.

    Rows run from the split where tuning trails the language model most to the split where it leads most, the
    benchmark's splits first and the later dataset's below them.
    """
    tuned, best, model = f"{summary.found['tuned_models'][0]}_dpo", summary.best_ridge(), summary.primary
    table = summary.per_split[(summary.per_split["queries"] == summary.scope) & (summary.per_split["seeds"] == "seed 0")
                              & (summary.per_split["arm"] == "k256")]
    scores = table.set_index(["split", "method"])["spearman"]
    groups = summary.found["groups"]
    ordered = [sorted(groups[name], key=lambda split: scores[(split, tuned)] - scores[(split, model)])
               for name in ("flip2", "postcutoff")]
    splits = [*ordered[0], *ordered[1]]
    positions = [float(index) for index in range(len(ordered[0]))] + [len(ordered[0]) + 1.0 + index for index in range(len(ordered[1]))]
    marks = {best: ("o", summary.color(best)), tuned: ("h", OLIVE), model: ("o", ORANGE)}
    figure, axis = plt.subplots(figsize=(7.4, 1.3 + 0.3 * (len(splits) + 1)))
    for y, split in zip(positions, splits, strict=True):
        found = [scores[(split, method)] for method in marks]
        axis.plot([min(found), max(found)], [y, y], color=PALE, linewidth=1.4, zorder=1)
        for value, (marker, color) in zip(found, marks.values(), strict=True):
            axis.plot([value], [y], marker=marker, markersize=7.5, color=color, markeredgecolor="white", markeredgewidth=0.8,
                      linestyle="none", zorder=3)
    finish_rows(axis, positions, [split_label(split) for split in splits], f"{SCORE}, 256 examples{summary.scope_words()}")
    axis.axvline(0.0, color=GRAY, linewidth=0.8, zorder=1)
    axis.invert_yaxis()
    handles = [Line2D([], [], marker=marker, color=color, linestyle="none", markersize=7.5, label=label_of(method))
               for method, (marker, color) in marks.items()]
    axis.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, handletextpad=0.3)
    save(figure, out, "preference_by_split")


def preference_trace(summary: Summary, out: Path) -> None:
    """Every epoch of every setting of the ranked loss, on the held-out rows and on the queries.

    Reads `preference_trace.csv`, which `report --trace` writes, and `preference_tuning.csv`. A line is the mean over
    the benchmark's splits of one Spearman a split and has no interval. Comparing its points on the queries reads
    their measured values, which no reported score does.
    """
    model = summary.found["tuned_models"][0]
    splits = summary.found["groups"]["flip2"]
    trace = pd.read_csv(summary.folder / "preference_trace.csv")
    trace = trace[(trace["model"] == model) & (trace["arm"] == "dpo") & trace["split"].isin(splits)]
    chosen = pd.read_csv(summary.folder / "preference_tuning.csv")
    chosen = chosen[(chosen["model"] == model) & (chosen["arm"] == "dpo") & chosen["split"].isin(splits)]
    assert trace["split"].nunique() == len(splits) == len(chosen), f"the trace holds {trace['split'].nunique()} of {len(splits)} splits"
    queries = int(trace["queries"].iloc[0])
    reported, primary, ridge = summary.mean(f"{model}_dpo"), summary.mean(summary.primary), summary.mean(summary.best_ridge())
    starts = {"held_out_spearman": chosen["held_out_spearman_untuned"].mean(), "spearman": summary.mean(f"{model}_zero")["mean"]}
    panels = [("held_out_spearman", f"Epochs of tuning, scored on the {int(chosen['held_out_rows'].iloc[0])} held-out rows"),
              ("spearman", f"Epochs of tuning, scored on the first {queries} queries")]
    rates = sorted(trace["learning_rate"].unique())
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=True)
    for axis, (column, xlabel) in zip(axes, panels, strict=True):
        curves = trace.groupby(["learning_rate", "beta", "epoch"])[column].mean()
        for (rate, beta), curve in curves.groupby(level=["learning_rate", "beta"]):
            epochs = [0, *curve.index.get_level_values("epoch")]
            color, style = (OLIVE if rate == rates[0] else OLIVE_PALE), ("-" if beta == trace["beta"].max() else "--")
            axis.plot(epochs, [starts[column], *curve.to_numpy()], color=color, linestyle=style, linewidth=2.0,
                      label=f"Learning rate {f'{rate:.5f}'.rstrip('0')}, weight of a ratio {beta:g}")
        axis.plot([0], [starts[column]], marker="^", markersize=7, markerfacecolor="white", markeredgecolor=OLIVE, markeredgewidth=1.8,
                  linestyle="none")
        axis.set_xlabel(xlabel)
        axis.set_xlim(-0.6, trace["epoch"].max() + 0.4)
        axis.set_xticks(range(0, int(trace["epoch"].max()) + 1, 5))
        quiet(axis, "y")
    held, test = axes
    held.set_ylabel(f"{SCORE},\nmean of {len(splits)} FLIP2 splits")
    # The lines are the same in both panels. Their key sits above the queries' scores, where that panel is empty.
    test.legend(loc="upper left", ncol=1, handlelength=2.6)
    # On the queries: the two methods the tuned model is read against, and the score the run reports.
    test.axhspan(primary["low"], primary["high"], color=ORANGE, alpha=0.10, linewidth=0)
    marks = [(primary["mean"], label_of(summary.primary), ORANGE), (ridge["mean"], label_of(summary.best_ridge()), summary.color(summary.best_ridge())),
             (reported["mean"], "Reported: the checkpoint the\nheld-out rows chose", OLIVE)]
    placed: list[float] = []
    for value, label, color in sorted(marks):
        test.axhline(value, color=color, linewidth=1.3, linestyle=":" if color == OLIVE else "-")
        y = value if not placed or value - placed[-1] >= 0.03 else placed[-1] + 0.03
        placed.append(y)
        test.annotate(label, (trace["epoch"].max() + 0.4, y), xytext=(6, 0), textcoords="offset points", ha="left", va="center",
                      fontsize=8.5, color=color, annotation_clip=False)
    letter_panels(axes)
    figure.subplots_adjust(wspace=0.08)
    save(figure, out, "preference_trace")


def preference_long(summary: Summary, out: Path) -> None:
    """One setting of the ranked loss run for twice the epochs, on the held-out rows and on every query.

    Reads the rows of the `long` variant in `preference_trace.csv`. The run scores every query of a split after every
    epoch, so its points are read against the best ridge and the reported arm on all queries, which the primary
    language model did not answer. A line is the mean over the benchmark's splits and has no interval.
    """
    first = summary.found["tuned_models"][0]
    model, splits = f"{first}_long", summary.found["groups"]["flip2"]
    trace = pd.read_csv(summary.folder / "preference_trace.csv")
    trace = trace[(trace["model"] == model) & (trace["arm"] == "dpo") & trace["split"].isin(splits)]
    chosen = pd.read_csv(summary.folder / "preference_tuning.csv")
    chosen = chosen[(chosen["model"] == model) & (chosen["arm"] == "dpo") & chosen["split"].isin(splits)]
    assert trace["split"].nunique() == len(splits) == len(chosen), f"the longer run holds {trace['split'].nunique()} of {len(splits)} splits"
    assert len(trace[["learning_rate", "beta"]].drop_duplicates()) == 1, "the longer run has one setting"
    best = summary.found["best_ridge_at_k256"]["flip2"]
    ridge, reported = summary.mean(best, queries="all"), summary.mean(f"{first}_dpo", queries="all")
    starts = {"held_out_spearman": chosen["held_out_spearman_untuned"].mean(), "spearman": summary.mean(f"{first}_zero", queries="all")["mean"]}
    panels = [("held_out_spearman", f"Epochs of tuning, scored on the {int(chosen['held_out_rows'].iloc[0])} held-out rows"),
              ("spearman", "Epochs of tuning, scored on every query")]
    last = int(trace["epoch"].max())
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=True)
    for axis, (column, xlabel) in zip(axes, panels, strict=True):
        curve = trace.groupby("epoch")[column].mean()
        axis.plot([0, *curve.index], [starts[column], *curve.to_numpy()], color=OLIVE, linewidth=2.0)
        axis.plot([0], [starts[column]], marker="^", markersize=7, markerfacecolor="white", markeredgecolor=OLIVE, markeredgewidth=1.8,
                  linestyle="none")
        axis.set_xlabel(xlabel)
        axis.set_xlim(-0.8, last + 0.6)
        axis.set_xticks(range(0, last + 1, 5))
        quiet(axis, "y")
    held, test = axes
    held.set_ylabel(f"{SCORE},\nmean of {len(splits)} FLIP2 splits")
    # The two epochs read with nothing chosen, named before the run.
    curve = trace.groupby("epoch")["spearman"].mean()
    for epoch in (20, last):
        test.plot([epoch], [curve[epoch]], marker="h", markersize=9, color=OLIVE, linestyle="none")
        test.annotate(f"{curve[epoch]:.2f}", (epoch, curve[epoch]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5,
                      color=OLIVE)
    test.axhspan(ridge["low"], ridge["high"], color=summary.color(best), alpha=0.10, linewidth=0)
    for found, label, color, style in ((ridge, label_of(best), summary.color(best), "-"),
                                       (reported, "Reported: the checkpoint the\nheld-out rows chose", OLIVE, ":")):
        test.axhline(found["mean"], color=color, linewidth=1.3, linestyle=style)
        test.annotate(label, (last + 0.6, found["mean"]), xytext=(6, 0), textcoords="offset points", ha="left", va="center", fontsize=8.5,
                      color=color, annotation_clip=False)
    letter_panels(axes)
    figure.subplots_adjust(wspace=0.08)
    save(figure, out, "preference_long")


def per_split(summary: Summary, out: Path) -> None:
    """Split by split, the language model beside the best ridge: who is ahead, and by how much."""
    best = summary.best_ridge()
    table = summary.differences
    table = table[(table["left"] == f"{summary.primary} k256") & (table["right"] == f"{best} k256") & (table["queries"] == summary.scope)]
    main = table[table["group"] == "flip2"].sort_values("difference")
    late = table[table["group"] == "postcutoff"].sort_values("difference")
    assert len(main) == summary.splits(), f"{len(main)} FLIP2 split differences of {summary.primary} against {best}"
    ordered = pd.concat([main, late])
    positions = [-float(index) for index in range(len(main))] + [-float(index) - len(main) - 1.0 for index in range(len(late))]

    figure, axis = plt.subplots(figsize=(7.0, 0.9 + 0.3 * (len(ordered) + 1)))
    for y, (_, row) in zip(positions, ordered.iterrows(), strict=True):
        decided = row["low"] > 0 or row["high"] < 0
        ahead = ORANGE if row["difference"] > 0 else summary.color(best)
        axis.plot([row["left_spearman"], row["right_spearman"]], [y, y], color=ahead if decided else PALE,
                  linewidth=2.6 if decided else 1.4, solid_capstyle="round", zorder=2)
        axis.plot([row["right_spearman"]], [y], marker="o", markersize=7.5, color=summary.color(best), linestyle="none", zorder=3)
        axis.plot([row["left_spearman"]], [y], marker="o", markersize=7.5, color=ORANGE, linestyle="none", zorder=4)
        axis.text(1.02, y, f"{row['difference']:+.2f}".replace("-", MINUS), transform=axis.get_yaxis_transform(), ha="left",
                  va="center", fontsize=8.5, fontweight="bold" if decided else "normal", color=ahead if decided else GRAY)
    axis.text(1.02, 0.9, "difference", transform=axis.get_yaxis_transform(), ha="left", va="center", fontsize=8.5, color=GRAY)
    finish_rows(axis, positions, [split_label(split) for split in ordered["split"]], f"{SCORE}, 256 examples{summary.scope_words()}")
    axis.set_ylim(positions[-1] - 0.7, 1.3)
    axis.axvline(0.0, color=GRAY, linewidth=0.8, zorder=1)
    handles = [Line2D([], [], marker="o", color=ORANGE, linestyle="none", markersize=7.5, label=label_of(summary.primary)),
               Line2D([], [], marker="o", color=summary.color(best), linestyle="none", markersize=7.5, label=label_of(best))]
    axis.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.3, columnspacing=1.6)
    save(figure, out, "per_split")


def controls_of(summary: Summary, out: Path, model: str, name: str) -> None:
    """What a language model draws on: its prompt changed one way at a time, beside two ridges on the same examples."""
    # A model asked one prompt a task ran its controls on the probe splits, one a dataset, and is read on those queries.
    partial = summary.queries_of(model, "k256_cipher") != "all"
    group, queries = ("probe", summary.first) if partial else ("flip2", "all")
    best = summary.found["best_ridge_first" if partial else "best_ridge_at_k256"]["flip2"]
    rows = [Row("Ridge, 256 examples", heading=True), summary.row("ridge_onehot", group=group, queries=queries),
            summary.row(best, group=group, queries=queries), Row(label_of(model), heading=True)]
    # A model reached only through the API was not asked the arms added for the subagent route.
    rows += by_mean([summary.row(model, arm, group, label=label, hollow=arm != "k256", queries=queries)
                     for arm, label in ARM_LABELS.items() if summary.has(model, arm, group, queries=queries)])
    figure, axis = plt.subplots(figsize=(7.0, chart_height(rows)))
    held = ", one per dataset" if partial else ""
    dot_chart(axis, rows, f"{SCORE},\nmean of {summary.splits(group)} FLIP2 splits{held}{summary.scope_words(queries)}")
    save(figure, out, name)


def claude_controls(summary: Summary, out: Path) -> None:
    """The primary language model's prompt controls."""
    controls_of(summary, out, summary.primary, "claude_controls")


def api_controls(summary: Summary, out: Path) -> None:
    """The same controls for Claude Fable 5.2 through the API at low effort."""
    controls_of(summary, out, API_CHEAP, "api_controls")


def ridge_controls(summary: Summary, out: Path) -> None:
    """What a ridge on a protein model's embedding draws on: the same ridge on features that hold less."""
    controls = ["ridge_random_vector", "ridge_composition", "ridge_esmc_300m_shuffled", "ridge_esmc_600m_shuffled",
                "ridge_esmc_300m_permuted", "ridge_esmc_600m_permuted", "ridge_esmc_300m_random", "ridge_onehot"]
    rows = [Row("Controls", heading=True), *by_mean([summary.row(method) for method in controls]),
            Row("Trained protein language model", heading=True), *by_mean([summary.row(f"ridge_{name}") for name in PLM_NAMES])]
    figure, axis = plt.subplots(figsize=(7.0, chart_height(rows)))
    dot_chart(axis, rows, f"{SCORE}, 256 examples, mean of {summary.splits()} FLIP2 splits")
    save(figure, out, "ridge_controls")


def plm_models(summary: Summary, out: Path) -> None:
    """Each protein language model used three ways. No language model is drawn: this is a figure of the baselines."""
    uses = {"zero_shot_{}": ("^", "Score with no examples"), "ridge_{}": ("o", "Ridge on embedding"),
            "ridge_onehot_plus_{}": ("s", "Ridge on one-hot + score")}
    names = sorted(PLM_NAMES, key=lambda name: summary.mean(f"ridge_{name}")["mean"])
    figure, axis = plt.subplots(figsize=(7.0, 0.9 + 0.62 * len(names)))
    offsets = np.linspace(0.22, -0.22, len(uses))  # (3,)
    for index, name in enumerate(names):
        for offset, (pattern, (marker, _)) in zip(offsets, uses.items(), strict=True):
            found = summary.mean(pattern.format(name))
            row = Row(name, found["mean"], found["low"], found["high"], summary.color(f"ridge_{name}"), marker, hollow=marker == "^")
            draw_mark(axis, row, -index + offset, size=6.5)
    finish_rows(axis, [-float(index) for index in range(len(names))], [PLM_LABELS[name] for name in names],
                f"{SCORE}, 256 examples,\nmean of {summary.splits()} FLIP2 splits{summary.scope_words()}")
    axis.set_ylim(-len(names) + 0.4, 0.62)
    handles = [Line2D([], [], marker=marker, color=GRAY, markerfacecolor="white" if marker == "^" else GRAY, linestyle="none",
                      markersize=6.5, markeredgewidth=1.8, label=label) for marker, label in uses.values()]
    axis.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=3, handletextpad=0.3, columnspacing=1.6)
    save(figure, out, "plm_models")


def learning_curve(summary: Summary, out: Path) -> None:
    """Score against the number of examples, on the splits that ran every size."""
    arms = ["k16", "k64", "k256", "klarge"]
    model = summary.primary
    queries = max((summary.queries_of(model, arm) for arm in arms), key=len)
    best = summary.best_ridge()
    plm = best.removeprefix("ridge_onehot_plus_").removeprefix("ridge_")
    lines = [("ridge_onehot", "Ridge on one-hot"), (f"ridge_{plm}", f"Ridge on {PLM_LABELS[plm]} embedding"),
             (f"ridge_onehot_plus_{plm}", f"Ridge on one-hot + {PLM_LABELS[plm]} score"), (model, label_of(model))]
    figure, axis = plt.subplots(figsize=(7.0, 3.9))
    ends = []
    for method, label in lines:
        found = [summary.mean(method, arm, "curve", queries=queries) for arm in arms]
        means = np.array([row["mean"] for row in found])  # (4,)
        lows, highs = np.array([row["low"] for row in found]), np.array([row["high"] for row in found])  # (4,), (4,)
        color = summary.color(method)
        axis.fill_between(range(1, 5), lows, highs, color=color, alpha=0.10, linewidth=0)
        axis.plot(range(1, 5), means, color=color, marker=marker_of(method), markersize=6.5, linewidth=2.0,
                  linestyle="--" if method.startswith("ridge_onehot_plus_") else "-")
        ends.append((means[-1], label, color))
    # At zero examples: the protein model's score with no examples, and the language model told only what the protein is.
    zero_shot = summary.mean(f"zero_shot_{plm}", "k256", "curve", queries=queries)
    unshown = summary.mean(model, "k0_described", "curve", queries=queries)
    for found, method, label in ((zero_shot, f"zero_shot_{plm}", f"{PLM_LABELS[plm]} score"), (unshown, model, "protein named")):
        color = summary.color(method)
        axis.plot([0, 0], [found["low"], found["high"]], color=color, linewidth=1.8, solid_capstyle="round")
        axis.plot([0], [found["mean"]], marker=marker_of(method), markersize=6.5, markerfacecolor="white", markeredgecolor=color,
                  markeredgewidth=1.8, linestyle="none")
        axis.annotate(label, (0, found["mean"]), xytext=(8, 0), textcoords="offset points", ha="left", va="center", fontsize=8, color=color)
    # Labels at the right end of each line, moved apart where two lines end close together.
    placed: list[float] = []
    for value, label, color in sorted(ends):
        y = value if not placed or value - placed[-1] >= 0.024 else placed[-1] + 0.024
        placed.append(y)
        axis.annotate(label, (4, y), xytext=(8, 0), textcoords="offset points", ha="left", va="center", fontsize=8.5, color=color)
    axis.set_xticks(range(5))
    axis.set_xticklabels(["0", "16", "64", "256", "up to\n2,048"])
    axis.set_xlim(-0.3, 4.1)
    axis.set_xlabel("Examples shown")
    splits = int(summary.mean(model, "k16", "curve", queries=queries)["splits"])
    scope = "" if queries == "all" else f",\n{queries} queries of each"
    axis.set_ylabel(f"{SCORE},\nmean of {splits} FLIP2 splits{scope}")
    quiet(axis, "y")
    save(figure, out, "learning_curve")


def claude_models(summary: Summary, out: Path) -> None:
    """Each Claude model with 256 examples, as written and with the residues enciphered."""
    # A primary model asked one prompt a task ran its enciphered prompts on the probe splits, one a dataset.
    best, group = summary.best_ridge(), "probe" if summary.partial else "flip2"
    models = [method for method in summary.found["language_models"]
              if summary.has(method, group=group) and summary.has(method, "k256_cipher", group)]
    models = sorted(models, key=lambda method: (method == summary.primary, summary.mean(method, group=group)["mean"]))
    rows = [Row("Ridge, 256 examples", heading=True), summary.row("ridge_onehot", group=group), summary.row(best, group=group),
            Row("Language model, 256 examples in its prompt", heading=True)]
    positions = [-float(index) for index in range(len(rows))]
    figure, axis = plt.subplots(figsize=(7.0, 0.9 + 0.3 * len(rows) + 0.5 * len(models)))
    for y, row in zip(positions, rows, strict=True):
        if row.heading:
            write_heading(axis, row.label, y)
        else:
            draw_mark(axis, row, y)
            write_value(axis, row, y)
    labels = ["" if row.heading else row.label for row in rows]
    y = positions[-1] - 0.9
    for method in models:
        plain, cipher = summary.row(method, group=group), summary.row(method, "k256_cipher", group, hollow=True)
        for row, offset in ((plain, 0.14), (cipher, -0.14)):
            draw_mark(axis, row, y + offset)
            write_value(axis, row, y + offset)
        positions.append(y)
        labels.append(label_of(method))
        y -= 1.0
    held = ", one per dataset" if summary.partial else ""
    finish_rows(axis, positions, labels, f"{SCORE},\nmean of {summary.splits(group)} FLIP2 splits{held}{summary.scope_words()}")
    handles = [Line2D([], [], marker="o", color=GRAY, linestyle="none", markersize=7, label="Sequences as written"),
               Line2D([], [], marker="o", color=GRAY, markerfacecolor="white", markeredgewidth=1.8, linestyle="none", markersize=7,
                      label="Residues enciphered")]
    axis.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.3, columnspacing=1.6)
    save(figure, out, "claude_models")


def refusals(summary: Summary, out: Path) -> None:
    """How often each model refused a prompt, and its score on the queries it answered beside a ridge on the same queries."""
    # Of the subagent series, only the primary: the other effort levels of the same model would repeat its row.
    cost = summary.cost[[effort_of(method) is None or method == summary.primary for method in summary.cost["method"]]]
    cost = cost.assign(share=lambda frame: frame["refused"] / frame["calls"]).sort_values("share", ascending=False)
    best = summary.best_ridge()
    answered = summary.answered
    answered = answered[(answered["group"] == "flip2") & (answered["arm"] == "k256") & (answered["right"] == f"{best} k256")
                        & (answered["scope"] == summary.scope)]
    answered = answered.assign(method=answered["left"].str.split(" ").str[0]).set_index("method")
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 0.9 + 0.42 * len(cost)), gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.85})

    positions = [float(index) for index in range(len(cost))]
    axes[0].barh(positions, 100 * cost["share"], color=[summary.color(method) for method in cost["method"]], height=0.62)
    for y, (_, row) in zip(positions, cost.iterrows(), strict=True):
        axes[0].annotate(f"{100 * row['share']:.0f}%  ({int(row['refused'])} of {int(row['calls'])})", (100 * row["share"], y),
                         xytext=(5, 0), textcoords="offset points", va="center", fontsize=8.5)
    finish_rows(axes[0], positions, [label_of(method) for method in cost["method"]], "Prompts refused, all arms (%)")
    axes[0].set_xlim(0, 100 * cost["share"].max() * 1.75 + 1)

    shown = [method for method in cost["method"] if method in answered.index]
    for y, method in enumerate(shown):
        row, everything = answered.loc[method], summary.mean(method)
        ridge = Row("", row["right_mean"], row["right_low"], row["right_high"], summary.color(best))
        model = Row("", row["left_mean"], row["left_low"], row["left_high"], summary.color(method))
        draw_mark(axes[1], ridge, y + 0.16, size=7)
        draw_mark(axes[1], model, y - 0.16, size=7)
        axes[1].plot([everything["mean"]], [y - 0.16], marker="o", markersize=7, markerfacecolor="white",
                     markeredgecolor=summary.color(method), markeredgewidth=1.8, linestyle="none", zorder=3)
    finish_rows(axes[1], [float(index) for index in range(len(shown))], [label_of(method) for method in shown],
                f"{SCORE}, 256 examples, mean of FLIP2 splits")
    handles = [Line2D([], [], marker="o", color=GRAY, markerfacecolor="white", markeredgewidth=1.8, linestyle="none", markersize=7,
                      label="Model, every query"),
               Line2D([], [], marker="o", color=GRAY, linestyle="none", markersize=7, label="Model, answered queries"),
               Line2D([], [], marker="o", color=summary.color(best), linestyle="none", markersize=7, label=f"{label_of(best)}, same queries")]
    axes[1].legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, handletextpad=0.3)
    for axis in axes:
        axis.invert_yaxis()
    letter_panels(axes)
    save(figure, out, "refusals")


def api_effort(summary: Summary, out: Path) -> None:
    """Split by split, Claude Fable 5.2 through the API at high effort against low effort."""
    table = summary.per_split
    table = table[(table["arm"] == "k256") & (table["seeds"] == "seed 0") & (table["queries"] == "all")
                  & table["method"].isin([API_SELECTED, API_CHEAP])]
    wide = table.pivot(index="split", columns="method", values="spearman")
    figure, axis = plt.subplots(figsize=(4.6, 4.6))
    low, high = min(wide.min().min(), 0.0) - 0.05, wide.max().max() + 0.08
    axis.plot([low, high], [low, high], color=PALE, linewidth=1.2, zorder=1)
    axis.scatter(wide[API_CHEAP], wide[API_SELECTED], s=38, color=PLUM, zorder=3, linewidths=0)
    for split, row in wide.iterrows():
        if abs(row[API_SELECTED] - row[API_CHEAP]) > 0.12:
            axis.annotate(split_label(split), (row[API_CHEAP], row[API_SELECTED]), xytext=(7, -3), textcoords="offset points",
                          fontsize=8, color=GRAY)
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_aspect("equal")
    axis.set_xlabel("Spearman, Claude Fable 5.2, low effort")
    axis.set_ylabel("Spearman, Claude Fable 5.2, high effort")
    quiet(axis, "both")
    save(figure, out, "api_effort")


def complementarity(summary: Summary, out: Path) -> None:
    """The language model, the best ridge, and the average of their two rankings."""
    best = summary.best_ridge()
    rows = [summary.row(best), summary.row(summary.primary),
            summary.row(f"{summary.primary}+{best}", label="Both, ranks averaged", color=INK, marker="D")]
    figure, axis = plt.subplots(figsize=(7.0, chart_height(rows)))
    dot_chart(axis, rows, f"{SCORE}, 256 examples,\nmean of {summary.splits()} FLIP2 splits{summary.scope_words()}")
    save(figure, out, "complementarity")


def within_groups(summary: Summary, out: Path) -> None:
    """Each method's score over all queries beside its score inside groups of like variants."""
    table = summary.metric_means
    table = table[(table["group"] == "flip2") & (table["arm"] == "k256")].set_index("method")
    best = summary.best_ridge()
    zero_shot = max((f"zero_shot_{name}" for name in PLM_NAMES), key=lambda method: summary.mean(method)["mean"])
    methods = list(dict.fromkeys(("nearest_neighbor", "ridge_onehot", zero_shot, "ridge_esmc_600m", best, summary.primary)))
    methods = sorted(methods, key=lambda method: (method == summary.primary, table.loc[method, "spearman"]))
    figure, axis = plt.subplots(figsize=(7.0, 0.9 + 0.36 * len(methods)))
    for y, method in enumerate(methods):
        row, color = table.loc[method], summary.color(method)
        axis.plot([row["within_strata_spearman"], row["spearman"]], [y, y], color=PALE, linewidth=1.4, zorder=1)
        axis.plot([row["spearman"]], [y], marker="o", markersize=7.5, color=color, linestyle="none", zorder=3)
        axis.plot([row["within_strata_spearman"]], [y], marker="o", markersize=7.5, markerfacecolor="white", markeredgecolor=color,
                  markeredgewidth=1.8, linestyle="none", zorder=3)
    finish_rows(axis, [float(index) for index in range(len(methods))], [label_of(method) for method in methods],
                f"{SCORE}, 256 examples, mean of {summary.splits()} FLIP2 splits")
    axis.invert_yaxis()
    handles = [Line2D([], [], marker="o", color=GRAY, markerfacecolor="white", markeredgewidth=1.8, linestyle="none", markersize=7,
                      label="Inside groups of like variants"),
               Line2D([], [], marker="o", color=GRAY, linestyle="none", markersize=7, label="All queries")]
    axis.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.3, columnspacing=1.6)
    save(figure, out, "within_groups")


def effort(summary: Summary, out: Path) -> None:
    """Claude Mythos 5.2 at each effort level: on the validation prompts, on test rows, and what each level spent thinking."""
    assert summary.validation is not None, "the summary holds no table of the validation prompts"
    table = summary.validation.assign(level=[effort_of(method) for method in summary.validation["method"]]).dropna(subset=["level"])
    levels = [level for level in EFFORT_WORDS if level in set(table["level"])]
    wide = table.pivot(index="task_id", columns="level", values="spearman").reindex(columns=levels)  # prompts by levels
    splits = table.drop_duplicates("task_id").set_index("task_id")["split"]
    figure, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), gridspec_kw={"width_ratios": [1.35, 0.75, 1.0], "wspace": 0.42})

    # A: each validation prompt as a gray line, and the mean of each level that answered every prompt.
    for task_id, row in wide.iterrows():
        axes[0].plot(range(len(levels)), row.to_numpy(), color=PALE, linewidth=1.2, marker="o", markersize=4.5, zorder=2)
        last = row.dropna()
        axes[0].annotate(split_label(splits[task_id]), (levels.index(last.index[-1]), last.iloc[-1]), xytext=(7, 0),
                         textcoords="offset points", ha="left", va="center", fontsize=8, color=GRAY)
    means = wide.dropna(axis=1).mean(axis=0)
    places = [levels.index(level) for level in means.index]
    axes[0].plot(places, means.to_numpy(), color=ORANGE, linewidth=1.6, zorder=3)
    for x, (level, value) in zip(places, means.items(), strict=True):
        selected = f"{MYTHOS}-{level}" == summary.primary
        axes[0].plot([x], [value], marker="o", markersize=8, markerfacecolor=ORANGE if selected else "white",
                     markeredgecolor=ORANGE, markeredgewidth=1.8, linestyle="none", zorder=4)
        axes[0].annotate(f"{value:.2f}", (x, value), xytext=(0, 7), textcoords="offset points", ha="center", va="bottom",
                         fontsize=8.5, color=ORANGE)
    axes[0].set_ylabel(f"Spearman on {len(wide)} validation prompts,\n100 queries each")
    axes[0].set_xticks(range(len(levels)))
    axes[0].set_xticklabels([EFFORT_WORDS[level].capitalize() for level in levels])
    axes[0].set_xlim(-0.4, len(levels) + 0.95)
    handles = [Line2D([], [], marker="o", color=ORANGE, linestyle="none", markersize=8, label="Mean, level chosen"),
               Line2D([], [], marker="o", color=ORANGE, markerfacecolor="white", markeredgewidth=1.8, linestyle="none", markersize=8,
                      label="Mean, other levels"),
               Line2D([], [], marker="o", color=PALE, markersize=4.5, linewidth=1.2, label="One prompt")]
    axes[0].legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handletextpad=0.3, columnspacing=1.2)

    # B: the levels asked on test rows, on the probe splits, beside the best ridge on the same queries.
    best = summary.best_ridge()
    tested = [level for level in levels if summary.has(f"{MYTHOS}-{level}", group="probe", queries=summary.first)]
    ridge = summary.mean(best, group="probe", queries=summary.first)
    axes[1].axhspan(ridge["low"], ridge["high"], color=summary.color(best), alpha=0.13, linewidth=0, zorder=0)
    axes[1].axhline(ridge["mean"], color=summary.color(best), linewidth=1.2, zorder=1)
    axes[1].annotate("Best ridge, same queries", (len(tested) / 2 - 0.5, ridge["low"]), xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.5, color=summary.color(best))
    for x, level in enumerate(tested):
        row = summary.mean(f"{MYTHOS}-{level}", group="probe", queries=summary.first)
        axes[1].plot([x, x], [row["low"], row["high"]], color=ORANGE, linewidth=1.8, solid_capstyle="round", zorder=3)
        axes[1].plot([x], [row["mean"]], marker="o", markersize=8, markeredgecolor=ORANGE, markeredgewidth=1.8, linestyle="none",
                     markerfacecolor=ORANGE if f"{MYTHOS}-{level}" == summary.primary else "white", zorder=4)
        axes[1].annotate(f"{row['mean']:.2f}", (x, row["mean"]), xytext=(9, 0), textcoords="offset points", ha="left", va="center",
                         fontsize=8.5)
    axes[1].set_ylabel(f"Spearman on test rows, mean of {summary.splits('probe')}\nFLIP2 splits, {summary.first} queries of each")
    axes[1].set_xticks(range(len(tested)))
    axes[1].set_xticklabels([EFFORT_WORDS[level].capitalize() for level in tested])
    axes[1].set_xlim(-0.6, len(tested) - 0.3)

    # C: what a prompt cost in thinking at each level, on the validation prompts.
    tokens = table.groupby("level")["thinking_tokens"].median().reindex(levels).to_numpy()  # (levels,)
    axes[2].bar(range(len(levels)), tokens, color=ORANGE, width=0.62)
    for x, value in enumerate(tokens):
        axes[2].annotate(f"{value:,.0f}", (x, value), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    axes[2].set_ylabel("Thinking tokens per validation prompt,\nmedian")
    axes[2].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axes[2].set_xticks(range(len(levels)))
    axes[2].set_xticklabels([EFFORT_WORDS[level].capitalize() for level in levels])
    axes[2].set_xlim(-0.6, len(levels) - 0.4)
    for axis in axes:
        axis.set_xlabel("Effort level given to the subagent")
        quiet(axis, "y")
    letter_panels(axes)
    save(figure, out, "effort")


def exact_matches(summary: Summary, out: Path) -> None:
    """Answers equal to the published value beside a random deal of the same answers: by dataset, then by arm on one.

    The first panel is the arm that asks for the published values with no examples, one row a split. The second takes
    the split whose matches most exceed the deal and shows every version of the prompt there, with the best ridge.
    """
    best, model = summary.best_ridge(), summary.primary
    table = summary.matches_per_split[summary.matches_per_split["queries"] == summary.scope].set_index(["split", "method", "arm"])
    groups = summary.found["groups"]
    asked = [split for split in (*groups["probe" if summary.partial else "flip2"], *groups.get("postcutoff", []))
             if (split, model, "k0_recall") in table.index]
    excess = {split: (table.loc[(split, model, "k0_recall"), "exact"] + 1) / (table.loc[(split, model, "k0_recall"), "exact_dealt"] + 1)
              for split in asked}
    focus = max(excess, key=excess.get)
    by_split = [(split_label(split), split, model, "k0_recall") for split in asked]
    order = ("k0_described", "k0_recall", "k256", "k256_flipped", "k256_cipher")
    by_arm = [(ARM_LABELS[arm], focus, model, arm) for arm in order if (focus, model, arm) in table.index]
    by_arm.append((f"{label_of(best)}, 256 examples", focus, best, "k256"))

    figure, axes = plt.subplots(1, 2, figsize=(12.4, 0.95 + 0.4 * max(len(by_split), len(by_arm))), gridspec_kw={"wspace": 0.95})
    panels = ((by_split, "Answers equal to the published value (%),\nno examples, published values asked for"),
              (by_arm, f"Answers equal to the published value (%),\n{split_label(focus)}"))
    for axis, (rows, xlabel) in zip(axes, panels, strict=True):
        positions = [float(index) for index in range(len(rows))]
        for y, (_, split, method, arm) in zip(positions, rows, strict=True):
            row, color = table.loc[(split, method, arm)], summary.color(method)
            observed, dealt = 100 * row["exact"] / row["answered"], 100 * row["exact_dealt"] / row["answered"]
            axis.plot([dealt, observed], [y, y], color=PALE, linewidth=1.4, zorder=1)
            axis.plot([dealt], [y], marker="o", markersize=7.5, markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.8,
                      linestyle="none", zorder=3)
            axis.plot([observed], [y], marker="o", markersize=7.5, color=color, linestyle="none", zorder=4)
            axis.annotate(f"{int(row['exact']):,} of {int(row['answered']):,}", (max(observed, dealt), y), xytext=(8, 0),
                          textcoords="offset points", ha="left", va="center", fontsize=8.5)
        finish_rows(axis, positions, [label for label, *_ in rows], xlabel)
        upper = axis.get_xlim()[1] * 1.3 + 0.5
        axis.set_xlim(-0.03 * upper, upper)
        axis.invert_yaxis()
    handles = [Line2D([], [], marker="o", color=GRAY, linestyle="none", markersize=7, label="Answers as given"),
               Line2D([], [], marker="o", color=GRAY, markerfacecolor="white", markeredgewidth=1.8, linestyle="none", markersize=7,
                      label="The same answers dealt to the queries at random")]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, handletextpad=0.3, columnspacing=1.6)
    letter_panels(axes)
    save(figure, out, "exact_matches")


def arms_by_split(summary: Summary, out: Path) -> None:
    """Split by split, the primary model under each version of its prompt, with the best ridge on the same rows as a tick.

    Rows run from the split where 256 examples add least to the score with no examples, to the split where they add most.
    """
    model, best = summary.primary, summary.best_ridge()
    table = summary.per_split[(summary.per_split["queries"] == summary.scope) & (summary.per_split["seeds"] == "seed 0")]
    scores = table.set_index(["split", "method", "arm"])["spearman"]
    arms = {"k256": ("o", False, "256 examples"), "k0_described": ("^", True, "No examples, protein named"),
            "k256_cipher": ("s", True, "256 examples, residues enciphered"), "k256_flipped": ("D", True, "256 examples, values reversed")}
    groups = summary.found["groups"]
    splits = [split for split in (*groups["probe" if summary.partial else "flip2"], *groups.get("postcutoff", []))
              if all((split, model, arm) in scores.index for arm in arms)]
    splits = sorted(splits, key=lambda split: scores[(split, model, "k256")] - scores[(split, model, "k0_described")])
    figure, axis = plt.subplots(figsize=(8.6, 1.5 + 0.5 * len(splits)))
    positions = [float(index) for index in range(len(splits))]
    for y, split in zip(positions, splits, strict=True):
        found = [scores[(split, model, arm)] for arm in arms]
        axis.plot([min(found), max(found)], [y, y], color=PALE, linewidth=1.4, zorder=1)
        axis.plot([scores[(split, best, "k256")]], [y], marker="|", markersize=16, markeredgewidth=2.4, color=summary.color(best),
                  linestyle="none", zorder=2)
        for value, (marker, hollow, _) in zip(found, arms.values(), strict=True):
            axis.plot([value], [y], marker=marker, markersize=7.5, markerfacecolor="white" if hollow else ORANGE, markeredgecolor=ORANGE,
                      markeredgewidth=1.8, linestyle="none", zorder=3)
    finish_rows(axis, positions, [split_label(split) for split in splits], f"{SCORE}{summary.scope_words()}")
    axis.axvline(0.0, color=GRAY, linewidth=0.8, zorder=1)
    axis.invert_yaxis()
    handles = [Line2D([], [], marker=marker, color=ORANGE, markerfacecolor="white" if hollow else ORANGE, markeredgewidth=1.8,
                      linestyle="none", markersize=7.5, label=label) for marker, hollow, label in arms.values()]
    handles.append(Line2D([], [], marker="|", color=summary.color(best), linestyle="none", markersize=14, markeredgewidth=2.4,
                          label=f"{label_of(best)}, 256 examples"))
    axis.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.3, columnspacing=1.6)
    save(figure, out, "arms_by_split")


FIGURES = {"headline": headline, "tiers": tiers, "tiers_both": tiers_both, "per_split": per_split,
           "claude_controls": claude_controls, "ridge_controls": ridge_controls,
           "arms_by_split": arms_by_split,
           "plm_models": plm_models, "learning_curve": learning_curve, "claude_models": claude_models, "refusals": refusals,
           "complementarity": complementarity, "within_groups": within_groups, "postcutoff": postcutoff, "effort": effort,
           "exact_matches": exact_matches, "api_controls": api_controls, "api_effort": api_effort,
           "preference_tuning": preference_tuning, "preference_by_split": preference_by_split, "preference_trace": preference_trace,
           "preference_long": preference_long, "preference_scale": preference_scale}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--primary", required=True, help="the language model the main figures are about, as its method name")
    parser.add_argument("--only", nargs="*", choices=sorted(FIGURES), help="draw these figures; every figure by default")
    args = parser.parse_args()

    setup()
    summary = Summary(args.summary, args.primary)
    for name in args.only or FIGURES:
        FIGURES[name](summary, args.out)
        print(f"drew {name}")


if __name__ == "__main__":
    main()
