# Large Scale Multi-Task Bayesian Optimization with Large Language Models (BOLT)

- **Study ID:** `zeng_2025_bolt_amp`
- **Authors:** Yimeng Zeng, Natalie Maus, Haydn Thomas Jones, Jeffrey Tao, Fangping Wan, Marcelo Der Torossian Torres, Cesar de la Fuente-Nunez, Ryan Marcus, Osbert Bastani, Jacob R. Gardner
- **Date:** 2025-03-11 (updated 2025-06-12)
- **Source tier:** Preprint (`preprint`)
- **Verification depth:** Full text verified (`full_text`)
- **Tasks:** Few-shot / in-context variant effect and optimization
- **Regimes:** Protein-finetuned LLM, LLM generation + specialized oracle filter / sequence model
- **Models tested:** GPT-4o-mini-0718 (iteratively fine-tuned)
- **Protein-specific baselines:** None (baselines are LOL-BO/STBO with peptide VAE, DKT/FSBO, POGPE, SGPE, Optformer, LLAMBO)
- **Wet-lab validation:** No
- **Code and data availability:** Unfilled 'GitHub: placeholder' (Sec 4); zero scalar numerical tables for AMP results (plots only).
- **Primary links:** [https://arxiv.org/abs/2503.08131](https://arxiv.org/abs/2503.08131)

## Takeaway

BOLT shows in plots that iteratively fine-tuning GPT-4o-mini on hundreds of 200,000-call LOL-BO trajectories enables strong few-shot AMP proposals on the APEX ML predictor. However, an ~18x training-task advantage over GP baselines, <=75% train/val sequence identity, missing pLM/AMP baselines, absent numerical tables, and purely surrogate-based evaluation confound the comparison.

## Methods

- **Datasets and benchmarks:** 1,000 extinct antimicrobial seed peptides (Wan et al. 2024; 100 held out for validation at >=25% Levenshtein distance, i.e., up to 75% sequence identity), minimizing APEX-predicted MIC against A. baumannii.
- **Evaluation protocol:** Outer loop iteratively fine-tunes GPT-4o-mini (7 rounds) on top trajectories from 200,000-call constrained latent-space LOL-BO runs plus self-augmented sequences (APEX MIC < 8); inner loop uses LLM initialization or few-shot sampling.
- **Controls included:** Untuned BOLT-0 check (fails), multi-task GP baselines (DKT/FSBO, POGPE, SGPE) trained on 5-50 STBO trajectories.
- **Compute and tools:** ~25,000 GPU hours + OpenAI fine-tuning across ~800-900 tasks (200k APEX oracle calls/task), vs LLAMBO capped at 10M tokens (<100 steps) and GP experts capped at <=50 trajectories.
- **Metrics:** Summed normalized APEX-predicted MIC and average predicted MIC vs sample count (Figs 1-2 right; no scalar tables).
- **Wet-lab protocol:** None (APEX ML predictor only).

## Key Results

| Task | Regime | Benchmark | Metric | General LLM | Protein Baseline | Evidence Status | Notes |
|---|---|---|---|---|---|---|---|
| Few-shot / in-context variant effect and optimization | Protein-finetuned LLM | 100 validation extinct AMP peptides (APEX predicted MIC, Figs 1-2) | Summed / mean APEX-predicted MIC (plots only, no scalar table) | BOLT (7-round fine-tuned GPT-4o-mini on ~800-900 LOL-BO trajectories): LLM initialization and 1-1,000 few-shot samples outperform 200,000-call STBO, DKT/FSBO, POGPE, SGPE, Optformer, LLAMBO in Figs 1-2 | STBO / DKT/FSBO (10-50 tasks) / POGPE & SGPE (5-20 experts); no pLM/AMP baseline: Plot-only (no scalar table) | **Suggestive (confounded)** | Severe training-task imbalance vs GP baselines (~800-900 vs 5-50 trajectories), <=75% train/val peptide identity, APEX ML surrogate without wet-lab MIC, no pLM/AMP baseline, and no scalar AMP tables. |

## Confounder and Leakage Audit (8 Axes)

| Axis | Status | Evidence |
|---|---|---|
| Temporal cutoff | **Concern** (`evidenced_concern`) | GPT-4o-mini-0718 coincides with/follows APEX (Wan et al. 2024) with no cutoff verification. |
| Homology overlap | **Concern** (`evidenced_concern`) | Sec 4 enforces only >=25% Levenshtein distance (allowing up to 75% sequence identity) between validation and training peptides. |
| Cross-modal / literature recall | **Concern** (`evidenced_concern`) | Prompt injects AMP domain priors ('positive charge, hydrophobicity, amphipathicity', Fig 5); no non-pretrained transformer control on peptides. |
| Preprocessing artifacts | **Concern** (`evidenced_concern`) | Strips non-AA characters silently, omits 75%-similarity violation rates, has 800 vs 900 train-set count discrepancy, and omits all scalar AMP tables. |
| Downstream oracle filtering | **Concern** (`evidenced_concern`) | Relies solely on APEX ML predictor for 200,000 BO calls/task, self-augmentation filtering (MIC < 8), and evaluation without wet-lab MIC assays. |
| Compute & tool asymmetry | **Concern** (`evidenced_concern`) | BOLT uses ~25,000 GPU hours + API fine-tuning across hundreds of tasks, while LLAMBO is capped at 10M tokens (<100 steps) and GP baselines at 5-50 trajectories. |
| Baseline fairness | **Concern** (`evidenced_concern`) | Massive training-data asymmetry against multi-task GP baselines (~800-900 vs 5-50 trajectories) and zero pLM or specialized AMP baselines. |
| Data & wet-lab transparency | **Concern** (`evidenced_concern`) | Unfilled 'GitHub: placeholder', no wet-lab validation, and no scalar AMP metrics table. |
