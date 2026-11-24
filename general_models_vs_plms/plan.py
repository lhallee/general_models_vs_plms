"""The experiment's stages: which arms run on which splits, with which seeds and how many queries.

The pilot reads only validation rows and chooses the text form and the effort. The main stage reads test rows once.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from . import flip2, postcutoff
from .flip2 import SplitInfo
from .tasks import Arm


K_SHOT = 256
K_LARGE = 2048
MAIN_QUERIES = 300
PILOT_QUERIES = 100
PILOT_SPLITS = ("amylase/by_mutation", "ired/two_to_many", "hydro/three_to_many")
CURVE_SPLITS = ("amylase/by_mutation", "amylase/close_to_far", "ired/two_to_many", "trpb/two_to_many",
                "hydro/low_to_high", "hydro/to_P06241")
ALL_SPLITS = (*flip2.SPLITS, *postcutoff.SPLITS)
# One split per dataset, for the controls and probes of a route too costly to ask on every split.
PROBE_SPLITS = ("amylase/by_mutation", "hydro/three_to_many", "ired/two_to_many", "nucb/two_to_many", "pdz3/single_to_double",
                "rhomax/by_wild_type", "trpb/two_to_many", "fgfr2/random")
# Preference tuning of a generative protein language model on a task's rows: the models, and the arm each method
# name ends in (`progen2_small_dpo`). `zero` is the untuned score, `ridge` a ridge on the untuned hidden states, `dpo`
# the ranked preference loss, `bt` the Bradley-Terry loss, and `dpo_shuffled` the ranked loss on permuted values.
# A model name with a third word is a variant of that model's tuning that changes one thing: `slow` a lower
# learning rate, `lowrank` a low-rank update of frozen weights, `rows` the largest example sets, and `long` one
# setting alone, run for 40 epochs on ProGen2-small and for 20 on ProGen2-base. `long20` and `long40` are that run
# after a fixed number of epochs, with no checkpoint chosen.
PREFERENCE_MODELS = ("progen2_small", "progen2_base", "progen2_small_slow", "progen2_small_lowrank", "progen2_small_rows",
                     "progen2_small_long", "progen2_small_long20", "progen2_small_long40", "progen2_base_long", "progen2_base_long20")
# The arm of the tasks a variant of the tuning runs on, where it is not the 256 examples of the others.
PREFERENCE_TASK_ARMS = {"progen2_small_rows": "klarge"}
PREFERENCE_ARMS = ("zero", "ridge", "dpo", "bt", "dpo_shuffled", "dpo_seed1", "dpo_seed2")
# The effort levels a subagent can be given, lowest first, and how close to the best a lower level must score to be chosen.
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
EFFORT_TOLERANCE = 0.02


@dataclass(frozen=True)
class Cell:
    """One arm on one split under one seed, with the part its queries come from."""

    split_info: SplitInfo
    arm: Arm
    seed: int
    part: str
    query_count: int


def split_infos(names: tuple[str, ...]) -> list[SplitInfo]:
    """Catalog entries for the named splits, in the order given."""
    by_name = {split_info.name: split_info for split_info in ALL_SPLITS}
    return [by_name[name] for name in names]


def select_effort(validation_means: dict[str, float], tolerance: float = EFFORT_TOLERANCE) -> str:
    """The lowest effort level whose mean validation Spearman is within the tolerance of the best level's."""
    best = max(validation_means.values())
    return next(level for level in EFFORT_LEVELS if level in validation_means and validation_means[level] >= best - tolerance)


def pilot_cells() -> list[Cell]:
    """Both text forms at 256 examples on three splits, with queries from the validation rows."""
    cells = []
    for split_info in split_infos(PILOT_SPLITS):
        forms = ("sequence",) if split_info.layout == "sequences" else ("substitutions", "sequence")
        for form in forms:
            arm = Arm(f"k{K_SHOT}_{form}", K_SHOT, representation=form)
            cells.append(Cell(split_info, arm, 0, "validation", PILOT_QUERIES))
    return cells


def main_arms(representation: str) -> list[tuple[Arm, tuple[int, ...], tuple[str, ...] | None]]:
    """Each main arm with its seeds and the splits it is limited to, None meaning every split."""
    base = Arm(f"k{K_SHOT}", K_SHOT, representation=representation)
    return [
        (base, (0, 1, 2), None),
        (replace(base, name=f"k{K_SHOT}_cipher", cipher=True), (0,), None),
        (replace(base, name=f"k{K_SHOT}_described", described=True), (0,), None),
        (replace(base, name="k0_described", k=0, described=True), (0,), None),
        (replace(base, name="klarge", k=K_LARGE), (0,), None),
        (replace(base, name=f"k{K_SHOT}_shuffled", labels="shuffled"), (0,), None),
        (replace(base, name=f"k{K_SHOT}_flipped", labels="flipped"), (0,), None),
        (replace(base, name="k16", k=16), (0,), CURVE_SPLITS),
        (replace(base, name="k64", k=64), (0,), CURVE_SPLITS),
    ]


def memory_cells() -> list[Cell]:
    """The recall arm on every split: no measured row, and the protein and the place of publication named.

    It asks the queries of the main arms, so what a model recalls is compared with what it predicts for the same rows.
    """
    arm = Arm("k0_recall", 0, described=True, recall=True)
    return [Cell(split_info, arm, 0, "test", MAIN_QUERIES) for split_info in ALL_SPLITS]


def main_cells(representation: str) -> list[Cell]:
    """Every main arm on its splits, with queries from the test rows."""
    cells = []
    for arm, seeds, limited_to in main_arms(representation):
        for split_info in ALL_SPLITS:
            if limited_to is not None and split_info.name not in limited_to:
                continue
            cells.extend(Cell(split_info, arm, seed, "test", MAIN_QUERIES) for seed in seeds)
    return cells
