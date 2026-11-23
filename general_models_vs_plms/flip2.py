"""FLIP2 splits as their files define them: the catalog, the download, and each row's part.

The catalog describes a split by what its file holds, which differs from the benchmark site for
`amylase/one_to_many` (the file trains on multi-mutants and tests on single mutants) and `nucb/two_to_many`
(the file trains on single mutants only).
"""

from __future__ import annotations

import hashlib
import urllib.request
import numpy as np
import pandas as pd

from dataclasses import dataclass
from pathlib import Path

from foundry.retry import retry


BASE_URL = "https://flip.protein.properties/assets/splits"
PARTS = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitInfo:
    """One benchmark split. `layout` says how its sequences relate to each other.

    `single_parent`: equal-length substitution variants of one parent. `pair`: `domain:peptide` strings, variants of
    one parent pair. `sequences`: several unrelated or variable-length parents, read as whole sequences.
    """

    dataset: str
    split: str
    split_type: str
    layout: str
    description: str
    published_in: str = ""

    @property
    def name(self) -> str:
        return f"{self.dataset}/{self.split}"

    @property
    def source(self) -> str:
        """Where the split's measurements are published, for a prompt that asks a model to recall them."""
        return self.published_in or f"the FLIP2 benchmark, as the split file {self.dataset}/{self.split}.csv.gz"

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.dataset}/{self.split}.csv.gz"


AMYLASE = ("The protein is the alpha-amylase of Bacillus subtilis. The value is its stain removal activity in a "
           "detergent assay, and a higher value means more activity.")
IRED = ("The protein is an imine reductase from Streptosporangium roseum. The value is its enzymatic activity "
        "measured in a microfluidic screen of an error-prone PCR library, and a higher value means more activity.")
NUCB = ("The protein is nuclease B (NucB) of Bacillus licheniformis. The value is its nuclease activity at pH 7, "
        "binned into the levels 0, 1, 2 and 3, and a higher level means more activity.")
TRPB = ("The protein is the beta subunit of tryptophan synthase (TrpB). The value is a growth-based fitness from "
        "combinatorial libraries at sets of interacting residues, and a higher value means better growth.")
HYDRO = ("The proteins are three small domains: the SH3 domain of human FYN (UniProt P06241), barley chymotrypsin "
         "inhibitor 2 (UniProt P01053) and the cold shock protein CspA of Escherichia coli (UniProt P0A9X9). In each, "
         "seven residues of the hydrophobic core were randomized to F, I, L, M or V. The value is a stability "
         "measurement, and a higher value means a more stable protein.")
RHOMAX = ("The proteins are microbial rhodopsins, with their variants and chimeras. The value is the wavelength of "
          "peak light absorption in nanometers.")
PDZ3 = ("The first chain is the third PDZ domain of PSD-95 (PDZ3) and the second chain is its peptide ligand from "
        "CRIPT. The value is a binding measurement for the pair, and a higher value means stronger binding.")

SPLITS = (
    SplitInfo("amylase", "one_to_many", "number", "single_parent", AMYLASE),
    SplitInfo("amylase", "close_to_far", "position", "single_parent", AMYLASE),
    SplitInfo("amylase", "far_to_close", "position", "single_parent", AMYLASE),
    SplitInfo("amylase", "by_mutation", "mutation", "single_parent", AMYLASE),
    SplitInfo("ired", "two_to_many", "number", "single_parent", IRED),
    SplitInfo("nucb", "two_to_many", "number", "single_parent", NUCB),
    SplitInfo("trpb", "one_to_many", "number", "single_parent", TRPB),
    SplitInfo("trpb", "two_to_many", "number", "single_parent", TRPB),
    SplitInfo("trpb", "by_position", "position", "single_parent", TRPB),
    SplitInfo("hydro", "three_to_many", "number", "sequences", HYDRO),
    SplitInfo("hydro", "low_to_high", "fitness", "sequences", HYDRO),
    SplitInfo("hydro", "to_P06241", "wild_type", "sequences", HYDRO),
    SplitInfo("hydro", "to_P0A9X9", "wild_type", "sequences", HYDRO),
    SplitInfo("hydro", "to_P01053", "wild_type", "sequences", HYDRO),
    SplitInfo("rhomax", "by_wild_type", "wild_type", "sequences", RHOMAX),
    SplitInfo("pdz3", "single_to_double", "number", "pair", PDZ3),
)


def split_by_name(name: str) -> SplitInfo:
    """The catalog entry called `dataset/split`."""
    matches = [split_info for split_info in SPLITS if split_info.name == name]
    if len(matches) != 1:
        raise KeyError(f"no FLIP2 split is named {name!r}")
    return matches[0]


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@retry(attempts=6, cap_seconds=60.0)
def fetch_bytes(url: str) -> bytes:
    """The body of an HTTP GET, asked again after a failure that passes."""
    request = urllib.request.Request(url, headers={"User-Agent": "general_models_vs_plms"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def split_path(folder: Path, split_info: SplitInfo) -> Path:
    """Where a split's file sits under the data folder."""
    return folder / split_info.dataset / f"{split_info.split}.csv.gz"


def download_split(split_info: SplitInfo, folder: Path) -> Path:
    """Fetch a split's file unless it is already there, and return its path."""
    path = split_path(folder, split_info)
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    partial.write_bytes(fetch_bytes(split_info.url))
    partial.replace(path)
    return path


def load_split(path: Path) -> pd.DataFrame:
    """A split's rows with `sequence`, `target` and `part`, in file order.

    `part` is `test` for a test row, `validation` for a training row the file flags for validation, else `train`.
    """
    frame = pd.read_csv(path)
    missing = {"sequence", "target", "set", "validation"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks the columns {sorted(missing)}")

    is_validation = frame["validation"].astype(str).str.lower() == "true"  # (n,)
    is_test = frame["set"] == "test"  # (n,)
    if not frame["set"].isin(["train", "test"]).all() or (is_validation & is_test).any():
        raise ValueError(f"{path} holds a set other than train and test, or a validation row in test")

    part = np.where(is_test, "test", np.where(is_validation, "validation", "train"))  # (n,)
    rows = pd.DataFrame({"sequence": frame["sequence"].astype(str), "target": frame["target"].astype(float), "part": part})
    if rows["target"].isna().any():
        raise ValueError(f"{path} holds a missing target")
    return rows
