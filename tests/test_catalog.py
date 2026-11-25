"""Tests for the structured literature/X compilation catalog and confounder-audit schema."""

from __future__ import annotations

import json
import pytest

from pathlib import Path
from typing import Any

from general_models_vs_plms.catalog import (
    CONFOUNDER_AXES,
    build_catalog,
    filter_claims,
    flatten_claims,
    load_studies,
    parse_study,
)


def _sample_audited(study_id: str = "sample_study") -> dict[str, Any]:
    return {
        "id": study_id,
        "title": "Sample Controlled Study",
        "short_label": "Sample 2026",
        "authors": "A. Author and B. Author",
        "date": "2026-11-20",
        "updated_date": "",
        "source_tier": "preprint",
        "read_depth": "full_text",
        "urls": ["https://example.org/sample"],
        "tasks": ["variant_effect_few_shot"],
        "regimes": ["intrinsic_text_only"],
        "models_tested": ["Model-X"],
        "plm_baselines": ["ESMC-600M ridge"],
        "wet_lab_validated": False,
        "code_and_data_released": "Full code and data on GitHub",
        "methods": {
            "datasets": "Post-cutoff DMS assay",
            "evaluation_protocol": "256 labeled variants in prompt, 100 held-out queries",
            "controls_included": "Cipher, flipped labels, post-cutoff split",
            "compute_and_tools": "Single text prompt, no tools",
            "metrics": "Spearman correlation with 95% paired bootstrap",
            "wet_lab_protocol": "None (computational evaluation)",
        },
        "key_results": [
            {
                "claim_id": "sample_postcutoff",
                "task": "variant_effect_few_shot",
                "regime": "intrinsic_text_only",
                "benchmark": "Post-cutoff DMS (n=100)",
                "metric": "Spearman",
                "llm_model": "Model-X",
                "llm_score": "0.56",
                "baseline_model": "ESMC-600M ridge",
                "baseline_score": "0.57",
                "wet_lab": False,
                "evidence_strength": "supported_controlled",
                "notes": "Published after model cutoff; ties matched ridge on identical 256 rows.",
            }
        ],
        "confounder_audit": {
            axis: {"status": "controlled", "evidence": f"Verified controlled for {axis}."} for axis in CONFOUNDER_AXES
        },
        "critiques_and_commentary": [],
        "takeaway": "Ties a matched pLM ridge on post-cutoff data with zero tool advantage.",
    }


def test_parse_study_rejects_missing_axes_for_audited() -> None:
    raw = _sample_audited()
    del raw["confounder_audit"]["temporal_cutoff"]
    with pytest.raises(ValueError, match="axes mismatch"):
        parse_study(raw, expected_id="sample_study")


def test_parse_study_forbids_critiquing_queued_unread() -> None:
    raw = _sample_audited("queued_one")
    raw["read_depth"] = "queued_unread"
    raw["key_results"][0]["evidence_strength"] = "queued"
    with pytest.raises(ValueError, match="empty confounder_audit"):
        parse_study(raw, expected_id="queued_one")

    raw["confounder_audit"] = {}
    parsed = parse_study(raw, expected_id="queued_one")
    assert parsed.read_depth == "queued_unread"
    assert parsed.confounder_audit == {}


def test_build_catalog_and_query_roundtrip(tmp_path: Path) -> None:
    studies_dir = tmp_path / "catalog" / "studies"
    studies_dir.mkdir(parents=True)

    s1 = _sample_audited("controlled_study")
    s2 = _sample_audited("leaked_study")
    s2["short_label"] = "Leaked 2026"
    s2["key_results"][0]["claim_id"] = "leaked_claim"
    s2["key_results"][0]["evidence_strength"] = "unverified_leaked"
    s2["confounder_audit"]["temporal_cutoff"] = {"status": "evidenced_concern", "evidence": "Test set predates cutoff."}

    (studies_dir / "controlled_study.json").write_text(json.dumps(s1), encoding="utf-8")
    (studies_dir / "leaked_study.json").write_text(json.dumps(s2), encoding="utf-8")

    bundle = build_catalog(tmp_path)
    assert bundle["counts"]["studies_audited"] == 2
    assert bundle["counts"]["claims_supported_controlled"] == 1
    assert bundle["counts"]["claims_unverified_leaked"] == 1
    assert (tmp_path / "catalog" / "index.json").exists()
    assert (tmp_path / "catalog" / "claims.csv").exists()
    assert (tmp_path / "catalog" / "studies.jsonl").exists()
    assert (tmp_path / "studies" / "controlled_study.md").exists()
    assert (tmp_path / "COMPILATION.md").exists()
    assert (tmp_path / "docs" / "compilation.html").exists()
    assert not (tmp_path / "SURVEY.md").exists()

    claims = flatten_claims(load_studies(studies_dir))
    rigorous = filter_claims(claims, rigorous_only=True)
    assert [c["claim_id"] for c in rigorous] == ["sample_postcutoff"]

    flagged = filter_claims(claims, axis_concern="temporal_cutoff")
    assert [c["claim_id"] for c in flagged] == ["leaked_claim"]
