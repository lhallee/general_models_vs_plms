"""A fitness dataset published after the language models' training cutoff, written in the FLIP2 file format.

The data are the erdafitinib resistance scores of the FGFR2 kinase domain saturation scan, MaveDB score set
`urn:mavedb:00001329-a-3`, which MaveDB published on 2026-11-20. Rows that encode the same protein change are
averaged, and the target is the base-2 logarithm of that mean, because the raw score is an enrichment ratio with a
long right tail. Rows that keep the wild-type residue are left out of the splits and kept as a measure of the
assay's noise. Two splits are drawn here: by variant at random, and by position.
"""

from __future__ import annotations

import io
import json
import re
import numpy as np
import pandas as pd

from datetime import UTC, datetime
from pathlib import Path

from .flip2 import SplitInfo, fetch_bytes, file_sha256


SCORE_SET = "urn:mavedb:00001329-a-3"
API = "https://api.mavedb.org/api/v1/score-sets"
EARLIEST_PUBLICATION = "2026-08-01"
THREE_TO_ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H",
    "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}
PROTEIN_CHANGE = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})")
FGFR2 = ("The protein is the kinase domain of human FGFR2, residues 461 to 780, so position 1 here is residue 461. "
         "The value is the base-2 logarithm of a resistance score from a pooled saturation mutagenesis screen in "
         "NCI-H1581 cells under the FGFR inhibitor erdafitinib, and a higher value means more resistance to the drug.")
PUBLISHED_IN = f"MaveDB, as the score set {SCORE_SET}"
SPLITS = (
    SplitInfo("fgfr2", "random", "mutation", "single_parent", FGFR2, PUBLISHED_IN),
    SplitInfo("fgfr2", "by_position", "position", "single_parent", FGFR2, PUBLISHED_IN),
)


def variant_table(scores_csv: bytes) -> tuple[str, pd.DataFrame]:
    """The parent segment and one row per protein change, with the mean score of the rows that encode it."""
    scores = pd.read_csv(io.BytesIO(scores_csv))
    parsed = scores["hgvs_pro"].str.extract(PROTEIN_CHANGE)
    if parsed.isna().any().any():
        raise ValueError("a score row is not a single amino acid substitution")

    scores["wild_type"] = parsed[0].map(THREE_TO_ONE)
    scores["position"] = parsed[1].astype(int)
    scores["mutant"] = parsed[2].map(THREE_TO_ONE)
    if scores[["wild_type", "mutant", "score"]].isna().any().any():
        raise ValueError("a score row names a residue outside the 20 amino acids or lacks a score")

    residues = scores.groupby("position")["wild_type"].agg(lambda column: "".join(sorted(set(column))))
    first, last = int(residues.index.min()), int(residues.index.max())
    if (residues.str.len() != 1).any() or len(residues) != last - first + 1:
        raise ValueError("the scores do not give one wild-type residue at every position of one segment")

    parent = "".join(residues.sort_index())
    variants = scores.groupby(["position", "wild_type", "mutant"], as_index=False)["score"].mean()
    variants["offset"] = variants["position"] - first
    return parent, variants


def split_synonymous(variants: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Protein changes apart from the rows that keep the wild-type residue, which measure the assay's noise."""
    unchanged = variants["mutant"] == variants["wild_type"]  # (n,)
    return variants[~unchanged].reset_index(drop=True), variants[unchanged].reset_index(drop=True)


def write_splits(folder: Path, seed: int = 0) -> dict[str, object]:
    """Download the score set, check its publication date, and write both splits; returns the provenance record."""
    metadata_bytes = fetch_bytes(f"{API}/{SCORE_SET}")
    scores_bytes = fetch_bytes(f"{API}/{SCORE_SET}/scores")
    metadata = json.loads(metadata_bytes)
    published = metadata["publishedDate"]
    if published < EARLIEST_PUBLICATION:
        raise ValueError(f"{SCORE_SET} was published on {published}, before {EARLIEST_PUBLICATION}")

    parent, every_change = variant_table(scores_bytes)
    variants, synonymous = split_synonymous(every_change)
    sequences =[parent[:offset] + mutant + parent[offset + 1:] for offset, mutant in zip(variants["offset"], variants["mutant"], strict=True)]
    target = np.log2(variants["score"].to_numpy())  # (n,)
    generator = np.random.default_rng([seed, 2026])
    by_variant = generator.random(len(variants)) < 0.5  # (n,)
    positions = np.sort(variants["position"].unique())  # (p,)
    test_positions = generator.permutation(positions)[: len(positions) // 2]  # (p / 2,)
    by_position = variants["position"].isin(test_positions).to_numpy()  # (n,)

    out = folder / "fgfr2"
    out.mkdir(parents=True, exist_ok=True)
    (out / "scores.csv").write_bytes(scores_bytes)
    (out / "metadata.json").write_bytes(metadata_bytes)
    for split, is_test in (("random", by_variant), ("by_position", by_position)):
        frame = pd.DataFrame({"sequence": sequences, "target": target, "set": np.where(is_test, "test", "train"),
                              "validation": False})
        frame.to_csv(out / f"{split}.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})

    provenance = {
        "score_set": SCORE_SET, "published_date": published, "creation_date": metadata["creationDate"],
        "license": metadata["license"]["shortName"], "title": metadata["title"],
        "retrieved_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "scores_sha256": file_sha256(out / "scores.csv"), "score_rows": int(len(pd.read_csv(io.BytesIO(scores_bytes)))),
        "protein_changes": int(len(variants)), "parent": parent, "first_residue": int(every_change["position"].min()),
        "synonymous_changes": int(len(synonymous)),
        "synonymous_log2_sd": float(np.log2(synonymous["score"].to_numpy()).std()),
        "missense_log2_sd": float(target.std()),
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance
