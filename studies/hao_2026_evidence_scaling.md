# Evidence Scaling for Zero-Shot Protein Reasoning with Large Language Models

- **Study ID:** `hao_2026_evidence_scaling`
- **Authors:** Z. Hao, C. Wang, D. Li, Y. Wang
- **Date:** 2026-10-01
- **Source tier:** Preprint (`preprint`)
- **Verification depth:** Queued (unread) (`queued_unread`)
- **Tasks:** Zero-shot variant effect prediction
- **Regimes:** LLM agent orchestrating specialized protein tools
- **Models tested:** Frozen general LLMs (full list pending PDF read)
- **Protein-specific baselines:** ProteinGym specialist predictors (pending PDF read)
- **Wet-lab validation:** No
- **Code and data availability:** Unverified (queued unread).
- **Primary links:** [https://doi.org/10.64898/2026.09.27.754822](https://doi.org/10.64898/2026.09.27.754822)

## Takeaway

Queued unread (bioRxiv HTTP 429 on 2026-11-25): adds structural/evolutionary tool evidence to frozen LLM prompts on ProteinGym; excluded from critiques until full text is verified.

## Methods

- **Datasets and benchmarks:** ProteinGym (exact subset pending full PDF read).
- **Evaluation protocol:** Adds structural and evolutionary evidence from standard tools to prompts of frozen LLMs (pending full PDF read).
- **Controls included:** Mismatched-variant evidence negative control mentioned in abstract (pending full PDF read).
- **Compute and tools:** Pending full PDF read.
- **Metrics:** Pending full PDF read.
- **Wet-lab protocol:** Pending full PDF read.

## Key Results

| Task | Regime | Benchmark | Metric | General LLM | Protein Baseline | Evidence Status | Notes |
|---|---|---|---|---|---|---|---|
| Zero-shot variant effect prediction | LLM agent orchestrating specialized protein tools | ProteinGym | Pending full PDF read | Frozen LLMs + structural/evolutionary evidence: Pending full PDF read | Specialist predictors: Pending full PDF read | **Queued (unread)** | bioRxiv full-text PDF/JATS returned HTTP 429; queued without critique until full text is downloaded. |

## Confounder and Leakage Audit (8 Axes)

Queued (unread in full). No confounder critique is recorded until full primary text is verified.
