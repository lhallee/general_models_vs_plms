# PG-LLM: Benchmarking General-Purpose Language Models for Protein Variant Ranking

- **Study ID:** `arora_2026_pg_llm`
- **Authors:** Rohit Krishan Arora, Leo Tianlai Chen, Melissa Du, Debora S. Marks, George M. Church (Harvard / Capable)
- **Date:** 2026-07-27 (updated 2026-08-21)
- **Source tier:** Preprint (`preprint`)
- **Verification depth:** Full post and repository verified (`full_post_and_code`)
- **Tasks:** Zero-shot variant effect prediction
- **Regimes:** Intrinsic LLM (text only, no tools)
- **Models tested:** Claude Opus 5, GPT-5.6 Sol (low/xhigh/max), Claude Opus 4.8 (low/xhigh/max), GPT-5.5 (low/xhigh), Gemini 3.5 Flash (low/high)
- **Protein-specific baselines:** 95 ProteinGym v1.3 predictors (46 sequence-only incl. ESM2-650M; 49 alignment/structure incl. VenusREM); ESM2-650M and VespaG on post-cutoff set
- **Wet-lab validation:** No
- **Code and data availability:** MIT-licensed benchmark runner, splits, leaderboard, and public reasoning traces on proteingymllm.com and GitHub (note: bioRxiv PDF v3 returned HTTP 429; verified from official website and GitHub repo).
- **Primary links:** [https://proteingymllm.com/](https://proteingymllm.com/), [https://github.com/rohitarorayyc/proteingym-llm](https://github.com/rohitarorayyc/proteingym-llm), [https://doi.org/10.64898/2026.07.27.741045](https://doi.org/10.64898/2026.07.27.741045)

## Takeaway

On 50-variant decile-stratified zero-shot ranking across 217 ProteinGym assays (with protein/organism names and assay context), Claude Opus 5 (0.406) and GPT-5.6 Sol (0.402) top the sequence-only specialist median (0.374) but trail alignment/structure models like VenusREM (0.523). On 59 post-Feb-2026 assays, GPT-5.6 Sol drops to 0.332 yet still exceeds ESM2-650M (0.202) and matches VespaG (0.323), verified from the official PG-LLM website and GitHub repo (bioRxiv PDF rate-limited).

## Methods

- **Datasets and benchmarks:** ProteinGym v1.3 substitutions (217 assays, 186 proteins) plus a post-knowledge-cutoff holdout of 59 assays from 19 studies with variant tables publicly retrievable >= 2026-02-01 (57/59 postdate Sol's 2026-02-16 cutoff).
- **Evaluation protocol:** Each episode supplies wild-type sequence, protein/organism name, phenotype description, fitness direction, and N=50 decile-stratified shuffled mutant sequences (3 seeds/assay); model returns JSON ranking with no tools, MSAs, or structures. Verified from proteingymllm.com and GitHub README (full bioRxiv PDF rate-limited).
- **Controls included:** 59-assay post-Feb-2026 holdout, LLM-judge reasoning-trace audit for source-study recognition (Gemini 3.5 Flash), difficulty-adjusted residual check on recognized vs unrecognized traces, and identical N=50 scoring of all 95 specialist baselines.
- **Compute and tools:** No external tools; sweeps low/high/xhigh/max reasoning effort (exact token/dollar totals not on landing summary).
- **Metrics:** Spearman rho (nested-macro assay->protein->category over 3 draws; study-group macro on 59 post-cutoff assays).
- **Wet-lab protocol:** None (retrospective ProteinGym + post-cutoff public DMS).

## Key Results

| Task | Regime | Benchmark | Metric | General LLM | Protein Baseline | Evidence Status | Notes |
|---|---|---|---|---|---|---|---|
| Zero-shot variant effect prediction | Intrinsic LLM (text only, no tools) | ProteinGym v1.3 (217 assays, N=50 decile-sampled sets x 3 seeds) | Nested-macro Spearman rho | Claude Opus 5 / GPT-5.6 Sol (max effort): Opus 5 0.406 (SE 0.002); GPT-5.6 Sol 0.402 (0.322 low -> 0.402 max); Opus 4.8 0.356 | 95 ProteinGym predictors (sequence-only median / alignment+struct median / VenusREM): Seq-only median 0.374 (Opus 5 beats 41/46); Alignment/struct median 0.435 (Opus 5 beats 8/49); VenusREM 0.523 | **Suggestive (confounded)** | Pre-cutoff benchmark with 88% (Sol) and 57% (Opus 4.8) source-study recognition in traces and N=50 decile subsampling (explicitly not interchangeable with full ProteinGym). |
| Zero-shot variant effect prediction | Intrinsic LLM (text only, no tools) | 59 post-Feb-2026 DMS assays (19 studies, N=50 decile-sampled sets) | Study-group macro Spearman rho | GPT-5.6 Sol (max) / Claude Opus 4.8 (xhigh) / Gemini 3.5 Flash (high): GPT-5.6 Sol 0.332 (0.271 low -> 0.332 max); Opus 4.8 0.290; Gemini 3.5 Flash 0.255 (Opus 5 omitted on site summary) | ESM2-650M / VespaG: ESM2-650M 0.202; VespaG 0.323 (no alignment/structure SOTA like VenusREM shown on site summary) | **Supported (controlled)** | Strong temporal holdout showing GPT-5.6 Sol drops from 0.402 to 0.332 post-cutoff yet still beats ESM2-650M (0.202) and matches VespaG (0.323) on N=50 sets; caveat: prompts include protein/organism names, homology is unreported on site, and bioRxiv PDF v3 could not be fetched due to HTTP 429. |

## Confounder and Leakage Audit (8 Axes)

| Axis | Status | Evidence |
|---|---|---|
| Temporal cutoff | **Controlled** (`controlled`) | Curates 59 assays from 19 studies publicly retrievable >= 2026-02-01 (57/59 after Sol's 2026-02-16 cutoff), exposing a drop from 0.402 -> 0.332 (Sol) and 0.356 -> 0.290 (Opus 4.8); Opus 5 absent from post-cutoff summary. |
| Homology overlap | **Uncontrolled** (`uncontrolled`) | No sequence-homology stratification of the 59 post-cutoff proteins against pretraining proteins is documented on the website or GitHub README. |
| Cross-modal / literature recall | **Concern** (`evidenced_concern`) | Prompts include protein and organism names; reasoning-trace audit finds 88% (Sol, 512/585) and 57% (Opus 4.8, 324/568) recognize the source study on ProteinGym, and authors warn visible traces cannot rule out silent recall. |
| Preprocessing artifacts | **Concern** (`evidenced_concern`) | Uses N=50 decile-stratified subsets to avoid long-context degradation, which the authors state makes scores not directly interchangeable with full ProteinGym assays. |
| Downstream oracle filtering | **N/A** (`not_applicable`) | Direct zero-shot ranking without external oracle filtering. |
| Compute & tool asymmetry | **Concern** (`evidenced_concern`) | Uses max/xhigh test-time reasoning effort sweeps (token/dollar costs unreported on site) against single-pass pLMs. |
| Baseline fairness | **Controlled** (`controlled`) | Scores all 95 ProteinGym predictors on identical N=50 draws with identical aggregation; on post-cutoff 59 assays only ESM2-650M and VespaG appear on the site summary (no VenusREM). |
| Data & wet-lab transparency | **Concern** (`evidenced_concern`) | Benchmark runner, splits, and traces are public on GitHub/proteingymllm.com, but full bioRxiv PDF v3 was HTTP 429 rate-limited so full supplementary tables remain unchecked. |

## External Critiques and Commentary

- **Anthony Gitter** (2026-11-25, x_thread): [https://x.com/anthonygitter/status/2125456205539844129](https://x.com/anthonygitter/status/2125456205539844129) - Highlighted PG-LLM's trace-recognition audit and post-cutoff dataset curation as essential controls missing from OpenAI's GPT-6 ProteinGym evaluation.
