# A Systematic Comparison of In-Context Learning Strategies for Peptide Property Predictions in Low-Data Regimes

- **Study ID:** `zhao_2026_peptide_icl`
- **Authors:** Jiantao Zhao, Andrei Voinea, Mathijs F. J. Mabesoone (Radboud University)
- **Date:** 2026-11-03
- **Source tier:** Preprint (`preprint`)
- **Verification depth:** Full text verified (`full_text`)
- **Tasks:** Few-shot / in-context variant effect and optimization, Zero-shot variant effect prediction
- **Regimes:** Intrinsic LLM (text only, no tools)
- **Models tested:** GPT-6.1 Sol, DeepSeek-V4.1 Flash (552B, low effort), Gemma-4 31B, Jev 1.14 (Typed Decision Model), GPT-6 Luna Decisions (Typed Decision Model)
- **Protein-specific baselines:** None (baselines are XGBoost on positional one-hot tokens, TabPFN v2, and TabFM)
- **Wet-lab validation:** No
- **Code and data availability:** Full code on GitHub (BigChemistry-RobotLab/icl-benchmark) and splits/predictions on Zenodo (10.5281/zenodo.23834634).
- **Primary links:** [https://arxiv.org/abs/2611.09994](https://arxiv.org/abs/2611.09994), [https://github.com/BigChemistry-RobotLab/icl-benchmark](https://github.com/BigChemistry-RobotLab/icl-benchmark), [https://doi.org/10.5281/zenodo.23834634](https://doi.org/10.5281/zenodo.23834634)

## Takeaway

On five low-data peptide classification tasks, frontier LLMs in context match but never statistically significantly beat even a simple positional one-hot XGBoost baseline (every #1 vs #2 bootstrap CI crosses zero), and no pLM baselines were tested. Strong zero-shot scores on the four pre-cutoff tasks vs chance-level zero-shot on the 2026 surface-activity task directly expose pretraining leakage alongside genuine in-context learning.

## Methods

- **Datasets and benchmarks:** 5 low-data peptide classification datasets (151-409 samples): Surface activity (2026, post-cutoff, 409 rows), Beta-sheet formation (2025, 268), Self-assembly (2024, 368), Breast-cancer activity (2018, 199, 3-class), Lung-cancer activity (2018, 151, 3-class).
- **Evaluation protocol:** Single fixed 80:20 stratified split (seed 42; group 5-fold by peptide for surface activity). LLMs/TDMs evaluated one-by-one via OpenRouter (5 replicates) vs local XGBoost/TabPFN/TabFM (100 replicates) with zero-shot and corrupted-label ablations.
- **Controls included:** Post-cutoff dataset (Surface activity 2026), zero-shot vs few-shot comparison, corrupted-label ablation (inverted/shifted labels), and one-by-one vs all-at-once inference check (p=0.2207).
- **Compute and tools:** No external tools; total API cost $69.43 across 36,562 requests / 146.9M tokens logged in Table S8.
- **Metrics:** Macro-F1 with 2,000-resample test bootstrap comparing #1 vs #2 models per task (Table S10).
- **Wet-lab protocol:** None (retrospective benchmark).

## Key Results

| Task | Regime | Benchmark | Metric | General LLM | Protein Baseline | Evidence Status | Notes |
|---|---|---|---|---|---|---|---|
| Few-shot / in-context variant effect and optimization | Intrinsic LLM (text only, no tools) | 5 low-data peptide classification datasets (151-409 rows) | Macro-F1 (bootstrap 95% CI for #1 vs #2) | GPT-6.1 Sol / DeepSeek-V4.1 Flash: Surface 0.762, Beta-sheet 0.791, Self-assembly 0.857, Breast 0.538, Lung 0.400 | XGBoost (positional one-hot) / TabFM: Surface XGB 0.714 (diff +0.048, -0.033 to +0.130); Beta-sheet XGB 0.800; Self-assembly XGB 0.750; Lung TabFM 0.540 | **Supported (controlled)** | No top model beats runner-up significantly on any of the 5 tasks (all bootstrap CIs include zero). Corrupted-label loss (0.432) exceeds zero-shot loss (0.223) in 22/25 pairs, proving in-context label reliance; pre-cutoff zero-shot inflation (0.546-0.722 vs 0.270-0.397 post-cutoff) flags pretraining exposure on older sets. |

## Confounder and Leakage Audit (8 Axes)

| Axis | Status | Evidence |
|---|---|---|
| Temporal cutoff | **Concern** (`evidenced_concern`) | 4 of 5 datasets (2018-2025) predate model cutoffs and show high zero-shot F1 (0.546-0.722 on beta-sheet, 0.665-0.696 on self-assembly) vs near-chance zero-shot (0.270-0.397) on post-cutoff 2026 Surface activity (Sec 3.3). |
| Homology overlap | **Uncontrolled** (`uncontrolled`) | Sec S1.1 states no sequence normalization, deduplication, or homology clustering was applied between train and test splits. |
| Cross-modal / literature recall | **Concern** (`evidenced_concern`) | Prompts name exact chemical systems (C16-tripeptides, buffers, surface tension <50 mN/m), enabling strong pretraining recall on the 4 pre-cutoff tasks. |
| Preprocessing artifacts | **Concern** (`evidenced_concern`) | Kept only minimum surface tension per (sequence, pH), dropped 16 published Houx sequences, and padded variable-length 3-38 aa peptides into positional one-hot vectors for XGBoost (weak encoding for unaligned peptides). |
| Downstream oracle filtering | **N/A** (`not_applicable`) | Direct classification task. |
| Compute & tool asymmetry | **Controlled** (`controlled`) | Zero tools; identical prompts across LLMs with complete token/cost accounting ($69.43 total, Table S8). |
| Baseline fairness | **Concern** (`evidenced_concern`) | No pLM embedding or physicochemical/k-mer baseline tested; single 80:20 split per task; positional one-hot XGBoost disadvantages local models on unaligned 5-38 aa cancer peptides. |
| Data & wet-lab transparency | **Controlled** (`controlled`) | Full code, splits, predictions, and confusion matrices published on GitHub and Zenodo. |
