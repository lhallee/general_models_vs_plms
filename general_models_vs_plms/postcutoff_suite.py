"""Expanded post-cutoff MaveDB deep mutational scanning suite and zero-spend frontier horizon.

Extends ``postcutoff.py`` (FGFR2 alone) into a multi-protein, homology-audited post-June-2026 benchmark suite
covering three unrelated human protein families:

- **FGFR2** (``urn:mavedb:00001329-a-3``, published 2026-11-20, no prior publication, CC BY-SA 4.0): receptor
  tyrosine kinase domain erdafitinib drug resistance.
- **GBA1** (``urn:mavedb:00001327-a-1``, published 2026-11-11, no prior publication, CC0): lysosomal
  beta-glucocerebrosidase (GCase) enzymatic activity (536 aa).
- **ARAF** (``urn:mavedb:00001314-0-4`` activity and ``urn:mavedb:00001314-0-3`` abundance, published 2026-10-16;
  linked bioRxiv preprint 10.64898/2026.10.04.756099 posted 2026-10-04, CC0): Raf-family kinase (606 aa).

Also records rejected post-June-2026 MaveDB uploads whose underlying preprints predate the June 2026 model cutoff
(notably ``PRKN`` ``urn:mavedb:00001281-a-1/2``, linked to a 2026-02-09 bioRxiv preprint) and maintains the
zero-spend horizon registry of frontier LLMs and specialized baselines to evaluate next.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import numpy as np
import pandas as pd

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .flip2 import fetch_bytes, file_sha256
from .postcutoff import API, THREE_TO_ONE


MISSENSE_RE = re.compile(r"^(?:([A-Z0-9]+):)?p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$")
SYNONYMOUS_RE = re.compile(r"^(?:([A-Z0-9]+):)?p\.([A-Z][a-z]{2})(\d+)=$")
EARLIEST_PUBLICATION = "2026-07-01"


@dataclass(frozen=True)
class PostCutoffAssay:
    """Metadata and parsing specification for one verified post-cutoff MaveDB assay."""

    slug: str
    urn: str
    gene: str
    protein_family: str
    published_date: str
    earliest_preprint_date: str
    linked_doi: str
    license_name: str
    target_gene_name: str
    gene_prefix: str
    score_transform: str
    description: str


@dataclass(frozen=True)
class RejectedAssay:
    """Post-cutoff MaveDB upload excluded after provenance/leakage auditing."""

    urn: str
    gene: str
    mavedb_published_date: str
    linked_preprint_date: str
    linked_doi: str
    rejection_reason: str


@dataclass(frozen=True)
class HorizonModel:
    """Planned model or baseline on the expanded benchmark horizon (no paid API spend approved)."""

    model_id: str
    category: str
    route: str
    training_cutoff: str
    api_spend_approved: bool
    status: str
    notes: str


POSTCUTOFF_ASSAYS: tuple[PostCutoffAssay, ...] = (
    PostCutoffAssay(
        slug="fgfr2_erdafitinib",
        urn="urn:mavedb:00001329-a-3",
        gene="FGFR2",
        protein_family="Receptor tyrosine kinase domain (320 aa segment)",
        published_date="2026-11-20",
        earliest_preprint_date="",
        linked_doi="",
        license_name="CC BY-SA 4.0",
        target_gene_name="FGFR2",
        gene_prefix="",
        score_transform="log2",
        description="Human FGFR2 kinase domain (residues 461-780) erdafitinib resistance saturation screen; 9 companion FGFR inhibitor screens also published 2026-11-20.",
    ),
    PostCutoffAssay(
        slug="gba1_gcase_activity",
        urn="urn:mavedb:00001327-a-1",
        gene="GBA1",
        protein_family="Lysosomal beta-glucocerebrosidase (GH30 hydrolase, 536 aa)",
        published_date="2026-11-11",
        earliest_preprint_date="",
        linked_doi="",
        license_name="CC0",
        target_gene_name="GBA1",
        gene_prefix="",
        score_transform="identity",
        description="Full-length human GBA1 lysosomal GCase enzymatic activity measured by LysoFix-GBA flow-cytometry DMS (~9,886 missense variants, already log/signed scale).",
    ),
    PostCutoffAssay(
        slug="araf_activity_baseline",
        urn="urn:mavedb:00001314-0-4",
        gene="ARAF",
        protein_family="Raf serine/threonine kinase (606 aa)",
        published_date="2026-10-16",
        earliest_preprint_date="2026-10-04",
        linked_doi="10.64898/2026.10.04.756099",
        license_name="CC0",
        target_gene_name="ARAF",
        gene_prefix="ARAF",
        score_transform="log2",
        description="Human ARAF kinase activity baseline DMS (~10,960 ARAF missense variants after excluding 7 BRAF control rows, indels, and nonsense; 558 synonymous noise controls).",
    ),
    PostCutoffAssay(
        slug="araf_abundance_baseline",
        urn="urn:mavedb:00001314-0-3",
        gene="ARAF",
        protein_family="Raf serine/threonine kinase (606 aa)",
        published_date="2026-10-16",
        earliest_preprint_date="2026-10-04",
        linked_doi="10.64898/2026.10.04.756099",
        license_name="CC0",
        target_gene_name="ARAF",
        gene_prefix="ARAF",
        score_transform="log2",
        description="Human ARAF cellular abundance baseline DMS (~5,041 ARAF missense variants; 262 synonymous noise controls), enabling activity-vs-abundance specificity tests on the same protein.",
    ),
)

REJECTED_ASSAYS: tuple[RejectedAssay, ...] = (
    RejectedAssay(
        urn="urn:mavedb:00001281-a-1",
        gene="PRKN",
        mavedb_published_date="2026-06-28",
        linked_preprint_date="2026-02-09",
        linked_doi="10.64898/2026.02.09.704817",
        rejection_reason="MaveDB upload date (2026-06-28) sits on the June 2026 cutoff boundary and its linked bioRxiv preprint (2026-02-09) predates Claude 5.2 and GPT-6 cutoffs by four months.",
    ),
    RejectedAssay(
        urn="urn:mavedb:00001281-a-2",
        gene="PRKN",
        mavedb_published_date="2026-06-28",
        linked_preprint_date="2026-02-09",
        linked_doi="10.64898/2026.02.09.704817",
        rejection_reason="Same linked bioRxiv preprint (2026-02-09) predating June 2026 training cutoffs.",
    ),
)

HORIZON_MODELS: tuple[HorizonModel, ...] = (
    HorizonModel(
        model_id="esmc_600m_and_6b_ridge_and_lora",
        category="specialized_plm",
        route="Docker GPU on named workstation (zero API spend)",
        training_cutoff="pre-2025",
        api_spend_approved=False,
        status="ready_when_gpu_named",
        notes="Embed and fit ridge + LoRA/fine-tuned ESMC on GBA1, ARAF activity/abundance, and FGFR2 (random and by_position splits).",
    ),
    HorizonModel(
        model_id="open_weight_reasoning_llms",
        category="general_llm_open_weight",
        route="Local/remote Docker vLLM on named GPU machine (zero API spend)",
        training_cutoff="varies_by_checkpoint",
        api_spend_approved=False,
        status="horizon_zero_spend",
        notes="Evaluate open-weight reasoning models (e.g., Qwen / DeepSeek) under identical k256, cipher, flipped, and recognition controls.",
    ),
    HorizonModel(
        model_id="gpt_6_2_and_gpt_6_astra",
        category="general_llm_frontier_api",
        route="External API (not run; zero paid API spend approved)",
        training_cutoff="2026 (exact date undisclosed by OpenAI)",
        api_spend_approved=False,
        status="horizon_awaiting_approval",
        notes="High priority for controlled comparison against OpenAI's unaudited ProteinGym claims once non-paid access or budget approval is granted.",
    ),
    HorizonModel(
        model_id="gemini_4_frontier",
        category="general_llm_frontier_api",
        route="External API (not run; zero paid API spend approved)",
        training_cutoff="2026",
        api_spend_approved=False,
        status="horizon_awaiting_approval",
        notes="Evaluate on identical FLIP2 + expanded post-cutoff suite under cipher/flipped/recognition controls when approved.",
    ),
    HorizonModel(
        model_id="claude_mythos_5_2_expanded_postcutoff",
        category="general_llm_harness",
        route="Audited single-prompt subagents on GBA1 and ARAF (zero API spend)",
        training_cutoff="2026-06",
        api_spend_approved=False,
        status="ready_after_splits_built",
        notes="Extend our controlled Mythos 5.2 post-cutoff evaluation from 1 protein (FGFR2) to 3 unrelated human protein families (FGFR2, GBA1, ARAF).",
    ),
)


def max_ungapped_identity(seq_a: str, seq_b: str, kmer: int = 5) -> dict[str, float]:
    """Compute 5-mer Jaccard similarity and best sliding-window ungapped identity between two protein sequences."""
    if not seq_a or not seq_b:
        return {"kmer_jaccard": 0.0, "best_window_identity": 0.0}
    kmers_a = {seq_a[i : i + kmer] for i in range(max(0, len(seq_a) - kmer + 1))} if len(seq_a) >= kmer else {seq_a}
    kmers_b = {seq_b[i : i + kmer] for i in range(max(0, len(seq_b) - kmer + 1))} if len(seq_b) >= kmer else {seq_b}
    union = len(kmers_a | kmers_b)
    jaccard = len(kmers_a & kmers_b) / union if union else 0.0

    short, long = (seq_a, seq_b) if len(seq_a) <= len(seq_b) else (seq_b, seq_a)
    best = 0.0
    span = len(short)
    for offset in range(len(long) - span + 1):
        matches = sum(a == b for a, b in zip(short, long[offset : offset + span], strict=True))
        ident = matches / span
        if ident > best:
            best = ident
    return {"kmer_jaccard": round(jaccard, 4), "best_window_identity": round(best, 4)}


def parse_mavedb_missense(
    scores_csv: bytes,
    target_sequence: str,
    *,
    gene_prefix: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Parse single-AA missense substitutions and synonymous noise rows from a MaveDB scores CSV.

    Filters out nonsense (``Ter``), deletions/insertions, missing scores, and rows belonging to other gene prefixes.
    Verifies every wild-type residue against ``target_sequence`` (1-based numbering).
    """
    raw = pd.read_csv(io.BytesIO(scores_csv))
    if "hgvs_pro" not in raw.columns or "score" not in raw.columns:
        raise ValueError("MaveDB scores CSV must contain 'hgvs_pro' and 'score' columns")

    missense_rows: list[dict[str, Any]] = []
    syn_rows: list[dict[str, Any]] = []
    skipped = {"missing_hgvs_or_score": 0, "other_gene_prefix": 0, "nonsense_or_indel": 0, "wt_mismatch": 0}

    for hgvs, score in zip(raw["hgvs_pro"], raw["score"], strict=True):
        if pd.isna(hgvs) or pd.isna(score):
            skipped["missing_hgvs_or_score"] += 1
            continue
        text = str(hgvs).strip()
        m_syn = SYNONYMOUS_RE.match(text)
        if m_syn:
            prefix, wt3, pos_s = m_syn.groups()
            if gene_prefix and (prefix or "") != gene_prefix:
                skipped["other_gene_prefix"] += 1
                continue
            pos = int(pos_s)
            wt1 = THREE_TO_ONE.get(wt3)
            if wt1 is None or pos < 1 or pos > len(target_sequence) or target_sequence[pos - 1] != wt1:
                skipped["wt_mismatch"] += 1
                continue
            syn_rows.append({"position": pos, "wild_type": wt1, "mutant": wt1, "score": float(score)})
            continue

        m_sub = MISSENSE_RE.match(text)
        if not m_sub:
            skipped["nonsense_or_indel"] += 1
            continue
        prefix, wt3, pos_s, mut3 = m_sub.groups()
        if gene_prefix and (prefix or "") != gene_prefix:
            skipped["other_gene_prefix"] += 1
            continue
        if mut3 == "Ter" or wt3 not in THREE_TO_ONE or mut3 not in THREE_TO_ONE:
            skipped["nonsense_or_indel"] += 1
            continue
        pos = int(pos_s)
        wt1 = THREE_TO_ONE[wt3]
        mut1 = THREE_TO_ONE[mut3]
        if pos < 1 or pos > len(target_sequence) or target_sequence[pos - 1] != wt1:
            skipped["wt_mismatch"] += 1
            continue
        if wt1 == mut1:
            syn_rows.append({"position": pos, "wild_type": wt1, "mutant": mut1, "score": float(score)})
        else:
            missense_rows.append({"position": pos, "wild_type": wt1, "mutant": mut1, "score": float(score)})

    if not missense_rows:
        raise ValueError("No valid missense substitutions parsed from MaveDB CSV")

    missense_df = pd.DataFrame(missense_rows).groupby(["position", "wild_type", "mutant"], as_index=False)["score"].mean()
    syn_df = (
        pd.DataFrame(syn_rows).groupby(["position", "wild_type", "mutant"], as_index=False)["score"].mean()
        if syn_rows
        else pd.DataFrame(columns=["position", "wild_type", "mutant", "score"])
    )
    return missense_df, syn_df, skipped


def _extract_target_protein(metadata: dict[str, Any], target_gene_name: str) -> str:
    for target in metadata.get("targetGenes") or []:
        if target.get("name") == target_gene_name:
            seq_obj = target.get("targetSequence") or {}
            if seq_obj.get("sequenceType") == "protein" and seq_obj.get("sequence"):
                return str(seq_obj["sequence"]).strip()
    raise ValueError(f"Protein target sequence for '{target_gene_name}' not found in MaveDB metadata")


def write_assay_splits(assay: PostCutoffAssay, out_root: Path, *, seed: int = 0) -> dict[str, Any]:
    """Download one post-cutoff MaveDB assay, verify dates and WT sequence, and write FLIP2-format splits."""
    meta_bytes = fetch_bytes(f"{API}/{assay.urn}")
    scores_bytes = fetch_bytes(f"{API}/{assay.urn}/scores")
    meta = json.loads(meta_bytes)
    published = str(meta.get("publishedDate") or "")
    if published < EARLIEST_PUBLICATION:
        raise ValueError(f"{assay.urn} published {published} precedes {EARLIEST_PUBLICATION}")

    parent = _extract_target_protein(meta, assay.target_gene_name)
    missense, synonymous, skipped = parse_mavedb_missense(scores_bytes, parent, gene_prefix=assay.gene_prefix)
    raw_scores = missense["score"].to_numpy(dtype=float)  # (n,)
    if assay.score_transform == "log2":
        if (raw_scores <= 0).any():
            raise ValueError(f"{assay.urn}: non-positive scores found under log2 transform")
        target = np.log2(raw_scores)  # (n,)
    elif assay.score_transform == "identity":
        target = raw_scores
    else:
        raise ValueError(f"Unknown score_transform: {assay.score_transform}")

    sequences = [
        parent[: int(pos) - 1] + mut + parent[int(pos) :]
        for pos, mut in zip(missense["position"], missense["mutant"], strict=True)
    ]
    rng = np.random.default_rng([seed, 2026])
    by_variant = rng.random(len(missense)) < 0.5  # (n,)
    positions = np.sort(missense["position"].unique())  # (p,)
    test_positions = rng.permutation(positions)[: len(positions) // 2]  # (p / 2,)
    by_position = missense["position"].isin(test_positions).to_numpy()  # (n,)

    out = out_root / assay.slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "scores.csv").write_bytes(scores_bytes)
    (out / "metadata.json").write_bytes(meta_bytes)
    for split_name, is_test in (("random", by_variant), ("by_position", by_position)):
        frame = pd.DataFrame(
            {"sequence": sequences, "target": target, "set": np.where(is_test, "test", "train"), "validation": False}
        )
        frame.to_csv(out / f"{split_name}.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})

    provenance: dict[str, Any] = {
        "slug": assay.slug,
        "urn": assay.urn,
        "gene": assay.gene,
        "published_date": published,
        "earliest_preprint_date": assay.earliest_preprint_date,
        "linked_doi": assay.linked_doi,
        "license": assay.license_name,
        "score_transform": assay.score_transform,
        "retrieved_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "scores_sha256": file_sha256(out / "scores.csv"),
        "parent_length": len(parent),
        "missense_variants": int(len(missense)),
        "synonymous_variants": int(len(synonymous)),
        "skipped_counts": skipped,
        "target_sd": float(target.std()),
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def build_postcutoff_report(root: Path, parent_sequences: dict[str, str] | None = None) -> dict[str, Any]:
    """Write ``catalog/postcutoff_benchmark.json`` and ``POSTCUTOFF_BENCHMARK.md`` documenting the expanded suite."""
    pairwise: list[dict[str, Any]] = []
    if parent_sequences:
        names = sorted(parent_sequences)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                sim = max_ungapped_identity(parent_sequences[a], parent_sequences[b])
                pairwise.append({"pair": f"{a} vs {b}", **sim})

    payload: dict[str, Any] = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "earliest_publication_gate": EARLIEST_PUBLICATION,
        "assays": [asdict(a) for a in POSTCUTOFF_ASSAYS],
        "rejected_assays": [asdict(r) for r in REJECTED_ASSAYS],
        "pairwise_sequence_overlap": pairwise,
        "horizon_models": [asdict(h) for h in HORIZON_MODELS],
    }
    (root / "catalog").mkdir(parents=True, exist_ok=True)
    (root / "catalog" / "postcutoff_benchmark.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Post-Cutoff MaveDB Benchmark Expansion & Frontier Model Horizon",
        "",
        "Expands our leak-free evaluation beyond a single protein (`FGFR2`) to three unrelated human protein families "
        "verified from MaveDB (3,005 score sets scanned on 2026-11-25) with strict preprint-date auditing and zero paid API spend.",
        "",
        "## 1. Verified Post-June-2026 MaveDB Assays",
        "",
        "| Slug | URN | Gene / Family | MaveDB Date | Linked Preprint | License | Transform | Description |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for a in POSTCUTOFF_ASSAYS:
        pre = f"{a.earliest_preprint_date} (`{a.linked_doi}`)" if a.linked_doi else "None (first public on MaveDB)"
        lines.append(
            f"| `{a.slug}` | `{a.urn}` | {a.gene} ({a.protein_family}) | {a.published_date} | {pre} | "
            f"{a.license_name} | `{a.score_transform}` | {a.description} |"
        )

    lines.extend(
        [
            "",
            "## 2. Rejected Post-June-2026 Uploads (Pre-Cutoff Preprint Leakage Caught)",
            "",
            "| URN | Gene | MaveDB Upload | Linked Preprint | Why Rejected |",
            "|---|---|---|---|---|",
        ]
    )
    for r in REJECTED_ASSAYS:
        lines.append(
            f"| `{r.urn}` | {r.gene} | {r.mavedb_published_date} | {r.linked_preprint_date} (`{r.linked_doi}`) | {r.rejection_reason} |"
        )

    if pairwise:
        lines.extend(
            [
                "",
                "## 3. Sequence Homology Audit Between Post-Cutoff Parents",
                "",
                "| Pair | 5-mer Jaccard | Best Ungapped Window Identity |",
                "|---|---|---|",
            ]
        )
        for row in pairwise:
            lines.append(f"| {row['pair']} | {row['kmer_jaccard']:.4f} | {row['best_window_identity']:.4f} |")

    lines.extend(
        [
            "",
            "## 4. Frontier Model & Baseline Horizon (Zero Paid API Spend Approved)",
            "",
            "| Model / Baseline | Category | Route | API Spend Approved | Status | Plan |",
            "|---|---|---|---|---|---|",
        ]
    )
    for h in HORIZON_MODELS:
        lines.append(
            f"| `{h.model_id}` | `{h.category}` | {h.route} | {'Yes' if h.api_spend_approved else 'No'} | "
            f"`{h.status}` | {h.notes} |"
        )
    lines.append("")
    (root / "POSTCUTOFF_BENCHMARK.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI to write post-cutoff manifest/report or materialize splits for an assay."""
    parser = argparse.ArgumentParser(description="Post-cutoff MaveDB benchmark suite builder.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("report", help="Write catalog/postcutoff_benchmark.json and POSTCUTOFF_BENCHMARK.md.")
    fetch_cmd = sub.add_parser("fetch", help="Download and build FLIP2-format splits for one post-cutoff assay.")
    fetch_cmd.add_argument("--slug", required=True, choices=[a.slug for a in POSTCUTOFF_ASSAYS])
    fetch_cmd.add_argument("--out", type=Path, required=True)
    fetch_cmd.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    if args.cmd == "report":
        build_postcutoff_report(root)
        print("Wrote catalog/postcutoff_benchmark.json and POSTCUTOFF_BENCHMARK.md")
        return 0
    if args.cmd == "fetch":
        assay = next(a for a in POSTCUTOFF_ASSAYS if a.slug == args.slug)
        prov = write_assay_splits(assay, args.out, seed=args.seed)
        print(json.dumps(prov, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
