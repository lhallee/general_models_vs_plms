"""Structured compilation, confounder audit, and agent-queryable index for general LLMs vs protein language models.

Canonical study records live in ``catalog/studies/<id>.json``. Running ``build`` validates every record against the
8-axis confounder schema and generates:

- ``catalog/index.json``: single machine-readable bundle for agents and the interactive HTML page;
- ``catalog/studies.jsonl``: one study record per line;
- ``catalog/claims.csv``: flat claim-level table with confounder flags;
- ``studies/<id>.md``: full methods and confounder-audit dossier per audited study;
- ``COMPILATION.md``: master compilation index separating controlled conclusions from confounded or leaked claims;
- ``docs/compilation.html``: standalone interactive browser for humans (pass ``--write-survey`` to also overwrite ``SURVEY.md``).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


TASKS: dict[str, str] = {
    "sequence_recall": "Sequence recall and memorization",
    "variant_effect_zero_shot": "Zero-shot variant effect prediction",
    "variant_effect_few_shot": "Few-shot / in-context variant effect and optimization",
    "inverse_folding": "Inverse folding and sequence design",
    "structure_prediction": "Structure prediction and side-chain packing",
    "docking": "Protein-protein and small-molecule docking",
    "backbone_and_binder_design": "De novo backbone and binder design",
    "protein_qa_and_reasoning": "Protein QA and multimodal reasoning",
}

REGIMES: dict[str, str] = {
    "intrinsic_text_only": "Intrinsic LLM (text only, no tools)",
    "intrinsic_with_code": "Intrinsic LLM + code execution (no specialized protein models)",
    "hybrid_generation_plus_oracle_filter": "LLM generation + specialized oracle filter / sequence model",
    "tool_orchestrated_agent": "LLM agent orchestrating specialized protein tools",
    "finetuned_llm": "Protein-finetuned LLM",
}

SOURCE_TIERS: dict[str, str] = {
    "our_experiment": "Our controlled experiment",
    "peer_reviewed": "Peer-reviewed paper",
    "preprint": "Preprint",
    "technical_report": "Lab technical report",
    "blog_repo": "Blog post / repository",
    "x_thread": "X thread (empirical probe)",
}

READ_DEPTHS: dict[str, str] = {
    "full_text": "Full text verified",
    "full_post_and_code": "Full post and repository verified",
    "full_thread": "Full X thread verified",
    "queued_unread": "Queued (unread)",
}

EVIDENCE_STRENGTHS: dict[str, str] = {
    "supported_controlled": "Supported (controlled)",
    "suggestive_confounded": "Suggestive (confounded)",
    "unverified_leaked": "Unverified / likely leaked",
    "queued": "Queued (unread)",
}

CONFOUNDER_AXES: dict[str, str] = {
    "temporal_cutoff": "Temporal cutoff",
    "homology_overlap": "Homology overlap",
    "cross_modal_literature_recall": "Cross-modal / literature recall",
    "preprocessing_artifacts": "Preprocessing artifacts",
    "downstream_oracle_filtering": "Downstream oracle filtering",
    "compute_and_tool_asymmetry": "Compute & tool asymmetry",
    "baseline_fairness": "Baseline fairness",
    "data_and_wetlab_transparency": "Data & wet-lab transparency",
}

AXIS_STATUSES: tuple[str, ...] = (
    "controlled",
    "uncontrolled",
    "evidenced_concern",
    "not_applicable",
)

AXIS_SHORT: dict[str, str] = {
    "controlled": "Controlled",
    "uncontrolled": "Uncontrolled",
    "evidenced_concern": "Concern",
    "not_applicable": "N/A",
}

STATUS_PRIORITY: dict[str, int] = {
    "supported_controlled": 0,
    "suggestive_confounded": 1,
    "unverified_leaked": 2,
    "queued": 3,
}

SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class AxisAudit:
    """Audit result for one confounder or leakage axis."""

    status: str
    evidence: str


@dataclass(frozen=True)
class ClaimRecord:
    """One structured empirical comparison or claim from a study."""

    claim_id: str
    task: str
    regime: str
    benchmark: str
    metric: str
    llm_model: str
    llm_score: str
    baseline_model: str
    baseline_score: str
    wet_lab: bool
    evidence_strength: str
    notes: str


@dataclass(frozen=True)
class CritiqueLink:
    """External technical critique or commentary attached to a study."""

    author: str
    date: str
    source_type: str
    url: str
    summary: str


@dataclass(frozen=True)
class MethodsRecord:
    """Structured methods documentation for a study."""

    datasets: str
    evaluation_protocol: str
    controls_included: str
    compute_and_tools: str
    metrics: str
    wet_lab_protocol: str


@dataclass(frozen=True)
class StudyRecord:
    """Canonical record for one empirical study, report, or X probe."""

    id: str
    title: str
    short_label: str
    authors: str
    date: str
    updated_date: str
    source_tier: str
    read_depth: str
    urls: list[str]
    tasks: list[str]
    regimes: list[str]
    models_tested: list[str]
    plm_baselines: list[str]
    wet_lab_validated: bool
    code_and_data_released: str
    methods: MethodsRecord
    key_results: list[ClaimRecord]
    confounder_audit: dict[str, AxisAudit]
    critiques_and_commentary: list[CritiqueLink]
    takeaway: str


def project_root() -> Path:
    """Root folder of the general_models_vs_plms project."""
    return Path(__file__).resolve().parent.parent


def _require_str(mapping: dict[str, Any], key: str, context: str, allow_empty: bool = False) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{context}: '{key}' must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{context}: '{key}' must not be empty")
    return value.strip()


def _require_str_list(mapping: dict[str, Any], key: str, context: str, allow_empty: bool = False) -> list[str]:
    value = mapping.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{context}: '{key}' must be a list of non-empty strings")
    if not allow_empty and not value:
        raise ValueError(f"{context}: '{key}' must not be empty")
    return [item.strip() for item in value]


def parse_study(raw: dict[str, Any], expected_id: str | None = None) -> StudyRecord:
    """Validate and parse one raw study dictionary into a StudyRecord."""
    study_id = _require_str(raw, "id", "study")
    if expected_id is not None and study_id != expected_id:
        raise ValueError(f"{study_id}: filename stem '{expected_id}' does not match id")

    source_tier = _require_str(raw, "source_tier", study_id)
    if source_tier not in SOURCE_TIERS:
        raise ValueError(f"{study_id}: unknown source_tier '{source_tier}'")

    read_depth = _require_str(raw, "read_depth", study_id)
    if read_depth not in READ_DEPTHS:
        raise ValueError(f"{study_id}: unknown read_depth '{read_depth}'")

    tasks = _require_str_list(raw, "tasks", study_id)
    for task in tasks:
        if task not in TASKS:
            raise ValueError(f"{study_id}: unknown task '{task}'")

    regimes = _require_str_list(raw, "regimes", study_id)
    for regime in regimes:
        if regime not in REGIMES:
            raise ValueError(f"{study_id}: unknown regime '{regime}'")

    if not isinstance(raw.get("wet_lab_validated"), bool):
        raise ValueError(f"{study_id}: 'wet_lab_validated' must be a boolean")

    raw_methods = raw.get("methods")
    if not isinstance(raw_methods, dict):
        raise ValueError(f"{study_id}: 'methods' must be an object")
    methods = MethodsRecord(
        datasets=_require_str(raw_methods, "datasets", f"{study_id}.methods"),
        evaluation_protocol=_require_str(raw_methods, "evaluation_protocol", f"{study_id}.methods"),
        controls_included=_require_str(raw_methods, "controls_included", f"{study_id}.methods"),
        compute_and_tools=_require_str(raw_methods, "compute_and_tools", f"{study_id}.methods"),
        metrics=_require_str(raw_methods, "metrics", f"{study_id}.methods"),
        wet_lab_protocol=_require_str(raw_methods, "wet_lab_protocol", f"{study_id}.methods"),
    )

    raw_results = raw.get("key_results")
    if not isinstance(raw_results, list) or not raw_results:
        raise ValueError(f"{study_id}: 'key_results' must be a non-empty list")
    key_results: list[ClaimRecord] = []
    seen_claim_ids: set[str] = set()
    for index, item in enumerate(raw_results):
        if not isinstance(item, dict):
            raise ValueError(f"{study_id}.key_results[{index}]: must be an object")
        ctx = f"{study_id}.key_results[{index}]"
        claim_id = _require_str(item, "claim_id", ctx)
        if claim_id in seen_claim_ids:
            raise ValueError(f"{study_id}: duplicate claim_id '{claim_id}'")
        seen_claim_ids.add(claim_id)
        task = _require_str(item, "task", ctx)
        if task not in TASKS:
            raise ValueError(f"{ctx}: unknown task '{task}'")
        regime = _require_str(item, "regime", ctx)
        if regime not in REGIMES:
            raise ValueError(f"{ctx}: unknown regime '{regime}'")
        strength = _require_str(item, "evidence_strength", ctx)
        if strength not in EVIDENCE_STRENGTHS:
            raise ValueError(f"{ctx}: unknown evidence_strength '{strength}'")
        if read_depth == "queued_unread" and strength != "queued":
            raise ValueError(f"{ctx}: queued_unread study must mark claims as 'queued'")
        if read_depth != "queued_unread" and strength == "queued":
            raise ValueError(f"{ctx}: audited study cannot mark claims as 'queued'")
        if not isinstance(item.get("wet_lab"), bool):
            raise ValueError(f"{ctx}: 'wet_lab' must be a boolean")
        key_results.append(
            ClaimRecord(
                claim_id=claim_id,
                task=task,
                regime=regime,
                benchmark=_require_str(item, "benchmark", ctx),
                metric=_require_str(item, "metric", ctx),
                llm_model=_require_str(item, "llm_model", ctx),
                llm_score=_require_str(item, "llm_score", ctx),
                baseline_model=_require_str(item, "baseline_model", ctx),
                baseline_score=_require_str(item, "baseline_score", ctx),
                wet_lab=item["wet_lab"],
                evidence_strength=strength,
                notes=_require_str(item, "notes", ctx),
            )
        )

    raw_audit = raw.get("confounder_audit")
    if not isinstance(raw_audit, dict):
        raise ValueError(f"{study_id}: 'confounder_audit' must be an object")
    confounder_audit: dict[str, AxisAudit] = {}
    if read_depth == "queued_unread":
        if raw_audit:
            raise ValueError(f"{study_id}: queued_unread study must have empty confounder_audit")
    else:
        missing = [axis for axis in CONFOUNDER_AXES if axis not in raw_audit]
        extra = [axis for axis in raw_audit if axis not in CONFOUNDER_AXES]
        if missing or extra:
            raise ValueError(f"{study_id}: confounder_audit axes mismatch (missing={missing}, extra={extra})")
        for axis in CONFOUNDER_AXES:
            entry = raw_audit[axis]
            if not isinstance(entry, dict):
                raise ValueError(f"{study_id}.confounder_audit.{axis}: must be an object")
            status = _require_str(entry, "status", f"{study_id}.confounder_audit.{axis}")
            if status not in AXIS_STATUSES:
                raise ValueError(f"{study_id}.confounder_audit.{axis}: invalid status '{status}'")
            evidence = _require_str(entry, "evidence", f"{study_id}.confounder_audit.{axis}")
            confounder_audit[axis] = AxisAudit(status=status, evidence=evidence)

    raw_critiques = raw.get("critiques_and_commentary", [])
    if not isinstance(raw_critiques, list):
        raise ValueError(f"{study_id}: 'critiques_and_commentary' must be a list")
    critiques: list[CritiqueLink] = []
    for index, item in enumerate(raw_critiques):
        if not isinstance(item, dict):
            raise ValueError(f"{study_id}.critiques_and_commentary[{index}]: must be an object")
        ctx = f"{study_id}.critiques_and_commentary[{index}]"
        critiques.append(
            CritiqueLink(
                author=_require_str(item, "author", ctx),
                date=_require_str(item, "date", ctx),
                source_type=_require_str(item, "source_type", ctx),
                url=_require_str(item, "url", ctx),
                summary=_require_str(item, "summary", ctx),
            )
        )

    return StudyRecord(
        id=study_id,
        title=_require_str(raw, "title", study_id),
        short_label=_require_str(raw, "short_label", study_id),
        authors=_require_str(raw, "authors", study_id),
        date=_require_str(raw, "date", study_id),
        updated_date=_require_str(raw, "updated_date", study_id, allow_empty=True),
        source_tier=source_tier,
        read_depth=read_depth,
        urls=_require_str_list(raw, "urls", study_id),
        tasks=tasks,
        regimes=regimes,
        models_tested=_require_str_list(raw, "models_tested", study_id),
        plm_baselines=_require_str_list(raw, "plm_baselines", study_id, allow_empty=True),
        wet_lab_validated=raw["wet_lab_validated"],
        code_and_data_released=_require_str(raw, "code_and_data_released", study_id),
        methods=methods,
        key_results=key_results,
        confounder_audit=confounder_audit,
        critiques_and_commentary=critiques,
        takeaway=_require_str(raw, "takeaway", study_id),
    )


def load_studies(studies_dir: Path) -> list[StudyRecord]:
    """Load and validate all study JSON records sorted newest-first (ours first on ties)."""
    if not studies_dir.exists():
        raise FileNotFoundError(f"Studies directory does not exist: {studies_dir}")
    records: list[StudyRecord] = []
    for path in sorted(studies_dir.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        records.append(parse_study(raw, expected_id=path.stem))
    records.sort(key=lambda s: (s.read_depth == "queued_unread", s.source_tier != "our_experiment", s.date), reverse=False)
    records.sort(key=lambda s: s.date, reverse=True)
    records.sort(key=lambda s: (s.read_depth == "queued_unread", s.source_tier != "our_experiment"))
    return records


def flatten_claims(studies: list[StudyRecord]) -> list[dict[str, Any]]:
    """Expand studies into flat claim rows carrying study metadata and all 8 confounder statuses."""
    rows: list[dict[str, Any]] = []
    for study in studies:
        concern_axes = [axis for axis, audit in study.confounder_audit.items() if audit.status == "evidenced_concern"]
        uncontrolled_axes = [axis for axis, audit in study.confounder_audit.items() if audit.status == "uncontrolled"]
        controlled_axes = [axis for axis, audit in study.confounder_audit.items() if audit.status == "controlled"]
        for claim in study.key_results:
            row: dict[str, Any] = {
                "study_id": study.id,
                "study_label": study.short_label,
                "study_title": study.title,
                "authors": study.authors,
                "date": study.date,
                "source_tier": study.source_tier,
                "read_depth": study.read_depth,
                "url": study.urls[0],
                "claim_id": claim.claim_id,
                "task": claim.task,
                "task_label": TASKS[claim.task],
                "regime": claim.regime,
                "regime_label": REGIMES[claim.regime],
                "benchmark": claim.benchmark,
                "metric": claim.metric,
                "llm_model": claim.llm_model,
                "llm_score": claim.llm_score,
                "baseline_model": claim.baseline_model,
                "baseline_score": claim.baseline_score,
                "wet_lab": claim.wet_lab,
                "evidence_strength": claim.evidence_strength,
                "evidence_strength_label": EVIDENCE_STRENGTHS[claim.evidence_strength],
                "notes": claim.notes,
                "controlled_count": len(controlled_axes),
                "uncontrolled_count": len(uncontrolled_axes),
                "concern_count": len(concern_axes),
            }
            for axis in CONFOUNDER_AXES:
                row[f"axis_{axis}"] = study.confounder_audit[axis].status if axis in study.confounder_audit else "unread"
            rows.append(row)
    rows.sort(key=lambda r: (STATUS_PRIORITY[r["evidence_strength"]], r["task"], r["study_id"], r["claim_id"]))
    return rows


def task_synthesis(studies: list[StudyRecord]) -> list[dict[str, Any]]:
    """Build per-task summary counts and rigorous-only findings for humans and agents."""
    claims = flatten_claims(studies)
    summary: list[dict[str, Any]] = []
    for task_id, task_label in TASKS.items():
        task_claims = [c for c in claims if c["task"] == task_id]
        controlled = [c for c in task_claims if c["evidence_strength"] == "supported_controlled"]
        confounded = [c for c in task_claims if c["evidence_strength"] == "suggestive_confounded"]
        leaked = [c for c in task_claims if c["evidence_strength"] == "unverified_leaked"]
        queued = [c for c in task_claims if c["evidence_strength"] == "queued"]
        summary.append(
            {
                "task": task_id,
                "task_label": task_label,
                "total_claims": len(task_claims),
                "supported_controlled_count": len(controlled),
                "suggestive_confounded_count": len(confounded),
                "unverified_leaked_count": len(leaked),
                "queued_count": len(queued),
                "supported_claims": controlled,
                "confounded_claims": confounded,
                "leaked_claims": leaked,
            }
        )
    return summary


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def render_dossier(study: StudyRecord) -> str:
    """Render one study's full Markdown dossier."""
    lines: list[str] = [
        f"# {study.title}",
        "",
        f"- **Study ID:** `{study.id}`",
        f"- **Authors:** {study.authors}",
        f"- **Date:** {study.date}" + (f" (updated {study.updated_date})" if study.updated_date else ""),
        f"- **Source tier:** {SOURCE_TIERS[study.source_tier]} (`{study.source_tier}`)",
        f"- **Verification depth:** {READ_DEPTHS[study.read_depth]} (`{study.read_depth}`)",
        f"- **Tasks:** {', '.join(TASKS[t] for t in study.tasks)}",
        f"- **Regimes:** {', '.join(REGIMES[r] for r in study.regimes)}",
        f"- **Models tested:** {', '.join(study.models_tested)}",
        f"- **Protein-specific baselines:** {', '.join(study.plm_baselines) if study.plm_baselines else 'None'}",
        f"- **Wet-lab validation:** {'Yes' if study.wet_lab_validated else 'No'}",
        f"- **Code and data availability:** {study.code_and_data_released}",
        "- **Primary links:** " + ", ".join(f"[{u}]({u})" for u in study.urls),
        "",
        "## Takeaway",
        "",
        study.takeaway,
        "",
        "## Methods",
        "",
        f"- **Datasets and benchmarks:** {study.methods.datasets}",
        f"- **Evaluation protocol:** {study.methods.evaluation_protocol}",
        f"- **Controls included:** {study.methods.controls_included}",
        f"- **Compute and tools:** {study.methods.compute_and_tools}",
        f"- **Metrics:** {study.methods.metrics}",
        f"- **Wet-lab protocol:** {study.methods.wet_lab_protocol}",
        "",
        "## Key Results",
        "",
        "| Task | Regime | Benchmark | Metric | General LLM | Protein Baseline | Evidence Status | Notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in study.key_results:
        lines.append(
            f"| {_md_escape(TASKS[c.task])} | {_md_escape(REGIMES[c.regime])} | {_md_escape(c.benchmark)} | "
            f"{_md_escape(c.metric)} | {_md_escape(c.llm_model)}: {_md_escape(c.llm_score)} | "
            f"{_md_escape(c.baseline_model)}: {_md_escape(c.baseline_score)} | "
            f"**{EVIDENCE_STRENGTHS[c.evidence_strength]}** | {_md_escape(c.notes)} |"
        )

    lines.extend(["", "## Confounder and Leakage Audit (8 Axes)", ""])
    if study.read_depth == "queued_unread":
        lines.append("Queued (unread in full). No confounder critique is recorded until full primary text is verified.")
    else:
        lines.extend(["| Axis | Status | Evidence |", "|---|---|---|"])
        for axis, label in CONFOUNDER_AXES.items():
            audit = study.confounder_audit[axis]
            lines.append(f"| {label} | **{AXIS_SHORT[audit.status]}** (`{audit.status}`) | {_md_escape(audit.evidence)} |")

    if study.critiques_and_commentary:
        lines.extend(["", "## External Critiques and Commentary", ""])
        for item in study.critiques_and_commentary:
            lines.append(f"- **{item.author}** ({item.date}, {item.source_type}): [{item.url}]({item.url}) - {item.summary}")

    lines.append("")
    return "\n".join(lines)


def render_survey(studies: list[StudyRecord], generated_date: str) -> str:
    """Render SURVEY.md from validated study records."""
    audited = [s for s in studies if s.read_depth != "queued_unread"]
    queued = [s for s in studies if s.read_depth == "queued_unread"]
    claims = flatten_claims(studies)
    supported = [c for c in claims if c["evidence_strength"] == "supported_controlled"]
    confounded = [c for c in claims if c["evidence_strength"] == "suggestive_confounded"]
    leaked = [c for c in claims if c["evidence_strength"] == "unverified_leaked"]
    by_task = task_synthesis(studies)

    lines: list[str] = [
        "# Compiled Evidence: General-Purpose LLMs vs Protein Language Models",
        "",
        f"Generated from `catalog/studies/*.json` on {generated_date}. Do not edit by hand; run "
        "`py -X utf8 -m general_models_vs_plms.catalog build`.",
        "",
        f"- **Audited studies (full text / primary data verified):** {len(audited)}",
        f"- **Queued studies (unread; excluded from critiques):** {len(queued)}",
        f"- **Indexed claims:** {len(claims)} total ({len(supported)} supported/controlled, "
        f"{len(confounded)} suggestive/confounded, {len(leaked)} unverified/likely leaked)",
        "",
        "For interactive filtering across tasks, regimes, models, and confounder axes, open [docs/compilation.html](docs/compilation.html) "
        "(and [docs/index.html](docs/index.html) / [RESULTS.md](RESULTS.md) for our step-by-step FLIP2, FGFR2, and ProGen2 experiment cards). "
        "For agent or script access, read `catalog/index.json`, `catalog/claims.csv`, `catalog/studies.jsonl`, or run "
        "`py -X utf8 -m general_models_vs_plms.catalog query --rigorous-only`.",
        "",
        "## 1. What the Rigorous Evidence Currently Supports (By Task)",
        "",
        "Only claims marked **Supported (controlled)** appear in the first table below. Claims undermined by unaudited leakage, "
        "heavy downstream oracle filtering, or missing matched baselines are explicitly separated into Section 2.",
        "",
        "| Task | Controlled Claims | Suggestive / Confounded | Unverified / Likely Leaked | Strictly Supported Findings |",
        "|---|---|---|---|---|",
    ]
    for item in by_task:
        if item["supported_claims"]:
            bullets = "; ".join(
                f"**{c['study_label']}** ({c['llm_model']} {c['llm_score']} vs {c['baseline_model']} {c['baseline_score']} on {c['benchmark']})"
                for c in item["supported_claims"]
            )
        else:
            bullets = "No fully controlled head-to-head claim verified yet."
        lines.append(
            f"| {item['task_label']} | {item['supported_controlled_count']} | {item['suggestive_confounded_count']} | "
            f"{item['unverified_leaked_count']} | {_md_escape(bullets)} |"
        )

    lines.extend(
        [
            "",
            "## 2. Full Indexed Study Catalog (Audited in Full)",
            "",
            "| Study | Date | Tier | Tasks | Regimes | Wet Lab | Read Depth | Dossier |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for s in audited:
        task_str = ", ".join(TASKS[t] for t in s.tasks)
        regime_str = ", ".join(s.regimes)
        lines.append(
            f"| **{_md_escape(s.short_label)}** | {s.date} | `{s.source_tier}` | {_md_escape(task_str)} | "
            f"{_md_escape(regime_str)} | {'Yes' if s.wet_lab_validated else 'No'} | `{s.read_depth}` | "
            f"[studies/{s.id}.md](studies/{s.id}.md) |"
        )

    lines.extend(
        [
            "",
            "## 3. Eight-Axis Confounder and Leakage Audit Matrix",
            "",
            "Legend: **Controlled** (explicitly mitigated and tested), **Uncontrolled** (not tested or undisclosed), "
            "**Concern** (direct evidence in text/traces of leakage, artifact shortcut, or confounder), **N/A** (axis does not apply).",
            "",
            "| Study | Cutoff | Homology | Lit. Recall | Artifacts | Oracle Filter | Compute/Tools | Baselines | Data/Wet-Lab |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    axis_order = list(CONFOUNDER_AXES.keys())
    for s in audited:
        cells = [AXIS_SHORT[s.confounder_audit[a].status] for a in axis_order]
        lines.append(f"| [{_md_escape(s.short_label)}](studies/{s.id}.md) | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## 4. All Indexed Claims (Grouped by Evidence Status)",
            "",
            "| Status | Task | Study | Regime | Benchmark | General LLM | Protein Baseline | Notes |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for c in claims:
        if c["evidence_strength"] == "queued":
            continue
        lines.append(
            f"| **{c['evidence_strength_label']}** | {_md_escape(c['task_label'])} | "
            f"[{_md_escape(c['study_label'])}](studies/{c['study_id']}.md) | `{c['regime']}` | "
            f"{_md_escape(c['benchmark'])} | {_md_escape(c['llm_model'])}: {_md_escape(c['llm_score'])} | "
            f"{_md_escape(c['baseline_model'])}: {_md_escape(c['baseline_score'])} | {_md_escape(c['notes'])} |"
        )

    if queued:
        lines.extend(
            [
                "",
                "## 5. Queued / Unread Sources (Excluded from Critiques)",
                "",
                "These sources could not be fetched in full text (for example, publisher rate limits or paywalls). They are "
                "indexed here for completeness and receive no confounder tags until read in full.",
                "",
                "| Study | Date | Tier | Tasks | Link | Why Queued |",
                "|---|---|---|---|---|---|",
            ]
        )
        for s in queued:
            lines.append(
                f"| **{_md_escape(s.short_label)}** | {s.date} | `{s.source_tier}` | "
                f"{_md_escape(', '.join(TASKS[t] for t in s.tasks))} | [{s.urls[0]}]({s.urls[0]}) | {_md_escape(s.takeaway)} |"
            )

    lines.extend(
        [
            "",
            "## 6. Agent and Script Access",
            "",
            "```bash",
            "# Validate all study records and regenerate index.json, claims.csv, dossiers, and SURVEY.md",
            "py -X utf8 -m general_models_vs_plms.catalog build",
            "",
            "# Query strictly controlled claims across all tasks",
            "py -X utf8 -m general_models_vs_plms.catalog query --rigorous-only",
            "",
            "# Query a specific task as JSON for agent consumption",
            "py -X utf8 -m general_models_vs_plms.catalog query --task inverse_folding --json",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def build_catalog(root: Path | None = None, *, write_survey: bool = False, write_html: bool = True) -> dict[str, Any]:
    """Validate all studies and generate machine-readable, Markdown, and interactive HTML compilation outputs."""
    base = root or project_root()
    studies_dir = base / "catalog" / "studies"
    studies = load_studies(studies_dir)
    generated_utc = datetime.now(UTC).isoformat(timespec="seconds")
    generated_date = generated_utc[:10]

    claims = flatten_claims(studies)
    synthesis = task_synthesis(studies)

    bundle: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": generated_utc,
        "vocabularies": {
            "tasks": TASKS,
            "regimes": REGIMES,
            "source_tiers": SOURCE_TIERS,
            "read_depths": READ_DEPTHS,
            "evidence_strengths": EVIDENCE_STRENGTHS,
            "confounder_axes": CONFOUNDER_AXES,
            "axis_statuses": list(AXIS_STATUSES),
        },
        "counts": {
            "studies_total": len(studies),
            "studies_audited": sum(1 for s in studies if s.read_depth != "queued_unread"),
            "studies_queued": sum(1 for s in studies if s.read_depth == "queued_unread"),
            "claims_total": len(claims),
            "claims_supported_controlled": sum(1 for c in claims if c["evidence_strength"] == "supported_controlled"),
            "claims_suggestive_confounded": sum(1 for c in claims if c["evidence_strength"] == "suggestive_confounded"),
            "claims_unverified_leaked": sum(1 for c in claims if c["evidence_strength"] == "unverified_leaked"),
            "claims_queued": sum(1 for c in claims if c["evidence_strength"] == "queued"),
        },
        "task_synthesis": synthesis,
        "studies": [asdict(s) for s in studies],
        "claims": claims,
    }

    catalog_dir = base / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    (catalog_dir / "index.json").write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with (catalog_dir / "studies.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for study in studies:
            handle.write(json.dumps(asdict(study), ensure_ascii=False) + "\n")

    if claims:
        fieldnames = list(claims[0].keys())
        with (catalog_dir / "claims.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(claims)

    dossiers_dir = base / "studies"
    dossiers_dir.mkdir(parents=True, exist_ok=True)
    active_names: set[str] = set()
    for study in studies:
        filename = f"{study.id}.md"
        active_names.add(filename)
        (dossiers_dir / filename).write_text(render_dossier(study), encoding="utf-8")
    for stale in dossiers_dir.glob("*.md"):
        if stale.name not in active_names:
            stale.unlink()

    compiled_md = render_survey(studies, generated_date)
    (base / "COMPILATION.md").write_text(compiled_md, encoding="utf-8")
    if write_survey:
        (base / "SURVEY.md").write_text(compiled_md, encoding="utf-8")
    if write_html:
        from .catalog_html import write_compilation_html

        write_compilation_html(bundle, base / "docs" / "compilation.html")
    return bundle


def filter_claims(
    claims: list[dict[str, Any]],
    *,
    task: str | None = None,
    regime: str | None = None,
    strength: str | None = None,
    source_tier: str | None = None,
    model_substring: str | None = None,
    rigorous_only: bool = False,
    axis_concern: str | None = None,
) -> list[dict[str, Any]]:
    """Filter flattened claims by task, regime, evidence strength, source tier, model, or confounder concern."""
    out = claims
    if task:
        if task not in TASKS:
            raise ValueError(f"Unknown task '{task}'. Valid: {', '.join(TASKS)}")
        out = [c for c in out if c["task"] == task]
    if regime:
        if regime not in REGIMES:
            raise ValueError(f"Unknown regime '{regime}'. Valid: {', '.join(REGIMES)}")
        out = [c for c in out if c["regime"] == regime]
    if rigorous_only:
        out = [c for c in out if c["evidence_strength"] == "supported_controlled"]
    elif strength:
        if strength not in EVIDENCE_STRENGTHS:
            raise ValueError(f"Unknown strength '{strength}'. Valid: {', '.join(EVIDENCE_STRENGTHS)}")
        out = [c for c in out if c["evidence_strength"] == strength]
    if source_tier:
        if source_tier not in SOURCE_TIERS:
            raise ValueError(f"Unknown source_tier '{source_tier}'. Valid: {', '.join(SOURCE_TIERS)}")
        out = [c for c in out if c["source_tier"] == source_tier]
    if model_substring:
        needle = model_substring.lower()
        out = [c for c in out if needle in c["llm_model"].lower() or needle in c["baseline_model"].lower()]
    if axis_concern:
        if axis_concern not in CONFOUNDER_AXES:
            raise ValueError(f"Unknown confounder axis '{axis_concern}'. Valid: {', '.join(CONFOUNDER_AXES)}")
        out = [c for c in out if c.get(f"axis_{axis_concern}") == "evidenced_concern"]
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for building, validating, and querying the compilation catalog."""
    parser = argparse.ArgumentParser(description="Build, validate, and query the general_models_vs_plms catalog.")
    parser.add_argument("--root", type=Path, default=None, help="Project root directory (default: auto-detected).")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="Validate all study records and regenerate index.json, claims.csv, dossiers, COMPILATION.md, and docs/compilation.html.")
    b.add_argument("--write-survey", action="store_true", help="Also overwrite SURVEY.md with the generated compilation.")
    b.add_argument("--no-html", action="store_true", help="Skip generating docs/compilation.html.")
    sub.add_parser("validate", help="Validate all study records without writing outputs.")

    q = sub.add_parser("query", help="Filter claims for human or agent inspection.")
    q.add_argument("--task", default=None, help=f"Filter by task ({', '.join(TASKS)}).")
    q.add_argument("--regime", default=None, help=f"Filter by regime ({', '.join(REGIMES)}).")
    q.add_argument("--strength", default=None, help=f"Filter by evidence strength ({', '.join(EVIDENCE_STRENGTHS)}).")
    q.add_argument("--source-tier", default=None, help=f"Filter by source tier ({', '.join(SOURCE_TIERS)}).")
    q.add_argument("--model", default=None, help="Case-insensitive substring match on LLM or baseline model.")
    q.add_argument("--rigorous-only", action="store_true", help="Return only supported_controlled claims.")
    q.add_argument("--axis-concern", default=None, help="Return claims from studies flagged with evidenced_concern on an axis.")
    q.add_argument("--json", action="store_true", dest="as_json", help="Emit JSON instead of a Markdown table.")

    args = parser.parse_args(argv)
    base = args.root or project_root()

    if args.command == "validate":
        studies = load_studies(base / "catalog" / "studies")
        print(f"Validated {len(studies)} study records cleanly.")
        return 0

    if args.command == "build":
        bundle = build_catalog(base, write_survey=args.write_survey, write_html=not args.no_html)
        counts = bundle["counts"]
        print(
            f"Built catalog: {counts['studies_audited']} audited, {counts['studies_queued']} queued, "
            f"{counts['claims_total']} claims ({counts['claims_supported_controlled']} supported/controlled, "
            f"{counts['claims_suggestive_confounded']} suggestive/confounded, "
            f"{counts['claims_unverified_leaked']} unverified/leaked)."
        )
        return 0

    if args.command == "query":
        studies = load_studies(base / "catalog" / "studies")
        matched = filter_claims(
            flatten_claims(studies),
            task=args.task,
            regime=args.regime,
            strength=args.strength,
            source_tier=args.source_tier,
            model_substring=args.model,
            rigorous_only=args.rigorous_only,
            axis_concern=args.axis_concern,
        )
        if args.as_json:
            print(json.dumps(matched, indent=2, ensure_ascii=False))
        else:
            print("| Status | Task | Study | Benchmark | General LLM | Protein Baseline |")
            print("|---|---|---|---|---|---|")
            for c in matched:
                print(
                    f"| {c['evidence_strength_label']} | {c['task']} | {c['study_label']} | "
                    f"{_md_escape(c['benchmark'])} | {_md_escape(c['llm_model'])}: {_md_escape(c['llm_score'])} | "
                    f"{_md_escape(c['baseline_model'])}: {_md_escape(c['baseline_score'])} |"
                )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
