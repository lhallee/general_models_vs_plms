"""Tests for the expanded post-cutoff MaveDB benchmark suite and homology audit."""

from __future__ import annotations

import pytest

from pathlib import Path

from general_models_vs_plms.postcutoff_suite import (
    POSTCUTOFF_ASSAYS,
    REJECTED_ASSAYS,
    build_postcutoff_report,
    max_ungapped_identity,
    parse_mavedb_missense,
)


def test_parse_mavedb_missense_filters_prefix_indels_and_keeps_synonymous() -> None:
    target = "MAVKL"
    csv_bytes = (
        b"accession,hgvs_pro,score\n"
        b"a1,ARAF:p.Met1Arg,1.2\n"
        b"a2,ARAF:p.Ala2Val,0.8\n"
        b"a3,ARAF:p.Ala2=,1.0\n"
        b"a4,ARAF:p.Val3Ter,0.1\n"
        b"a5,ARAF:p.Val3del,0.2\n"
        b"a6,BRAF:p.Met1Arg,9.9\n"
        b"a7,ARAF:p.Lys4Glu,2.0\n"
        b"a8,ARAF:p.Lys4Glu,4.0\n"
    )

    missense, syn, skipped = parse_mavedb_missense(csv_bytes, target, gene_prefix="ARAF")
    assert len(missense) == 3
    assert len(syn) == 1
    assert skipped["other_gene_prefix"] == 1
    assert skipped["nonsense_or_indel"] == 2
    assert skipped["wt_mismatch"] == 0
    k4e = missense.loc[missense["position"] == 4, "score"].iloc[0]
    assert k4e == pytest.approx(3.0)


def test_parse_mavedb_missense_flags_wt_mismatch() -> None:
    csv_bytes = b"accession,hgvs_pro,score\na1,p.Gly1Arg,1.0\na2,p.Ala2Val,0.5\n"
    missense, _, skipped = parse_mavedb_missense(csv_bytes, "MAV")
    assert len(missense) == 1
    assert skipped["wt_mismatch"] == 1


def test_max_ungapped_identity_bounds() -> None:
    same = max_ungapped_identity("ACDEFGHIKLM", "ACDEFGHIKLM")
    assert same["kmer_jaccard"] == 1.0
    assert same["best_window_identity"] == 1.0
    diff = max_ungapped_identity("AAAAAAAAAA", "CCCCCCCCCC")
    assert diff["kmer_jaccard"] == 0.0
    assert diff["best_window_identity"] == 0.0


def test_build_postcutoff_report_records_verified_and_rejected(tmp_path: Path) -> None:
    payload = build_postcutoff_report(
        tmp_path,
        parent_sequences={"FGFR2": "ACDEFGHIKLMNPQRSTVWY", "GBA1": "YYYYYYYYYYYYYYYYYYYY"},
    )
    assert len(payload["assays"]) == len(POSTCUTOFF_ASSAYS)
    assert len(payload["rejected_assays"]) == len(REJECTED_ASSAYS)
    assert all(not h["api_spend_approved"] for h in payload["horizon_models"])
    assert (tmp_path / "catalog" / "postcutoff_benchmark.json").exists()
    assert (tmp_path / "POSTCUTOFF_BENCHMARK.md").exists()
