# Rank-and-Reason: Multi-Agent Collaboration Accelerates Zero-Shot Protein Mutation Prediction (VenusRAR)

- **Study ID:** `tan_2026_rank_and_reason`
- **Authors:** Yang Tan, Yuanxi Yu, Can Wu, Bozitao Zhong, Mingchen Li, Guisheng Fan, Jiankang Zhu, Yafeng Liang, Nanqing Dong, Liang Hong
- **Date:** 2026-02-03 (updated 2026-02-04)
- **Source tier:** Preprint (`preprint`)
- **Verification depth:** Full text verified (`full_text`)
- **Tasks:** Zero-shot variant effect prediction, Few-shot / in-context variant effect and optimization
- **Regimes:** LLM agent orchestrating specialized protein tools, Intrinsic LLM (text only, no tools)
- **Models tested:** DeepSeek-Reasoner (v3.2), DeepSeek-Chat (v3.2), GPT-5-mini, Qwen3-8B, GLM-4.5-Flash
- **Protein-specific baselines:** VenusRAR-Ensemble (unweighted 6-expert mean: VenusREM, ProSST-2048, SaProt-AF650M, ProtSSN-Ensemble, GEMME, ESM-IF1), VenusREM, AIDO-Protein-RAG (16B), ProSST-2048, ESM2-650M, ESMC, ProGen3-3B
- **Wet-lab validation:** Yes
- **Code and data availability:** GitHub link stated in abstract, but Appendix C/D system prompts, Cas12i3 agent logs, and exact Cas12i3 mutation identities/raw assay values are withheld ('Updated upon paper acceptance').
- **Primary links:** [https://arxiv.org/abs/2602.00197](https://arxiv.org/abs/2602.00197), [https://github.com/ai4protein/VenusRAR/](https://github.com/ai4protein/VenusRAR/)

## Takeaway

Standalone DeepSeek-Reasoner scores just 0.159 Spearman on ProteinGym (far behind ESM2-650M at 0.414), whereas an unweighted ensemble of six specialized models reaches 0.542 and LLM orchestration nudges it to 0.551 while improving low-N Top-K hit rates. Missing non-LLM heuristic filter controls, withheld prompts/mutant identities, and an uncontrolled Cas12i3 wet-lab arm leave the causal contribution of LLM reasoning unisolated.

## Methods

- **Datasets and benchmarks:** ProteinGym substitutions (217 assays, 2,465,767 mutations), ProteinGym-DMS99 (31 assays with >99% single-mutant coverage), and wet-lab Cas12i3 nuclease (1,045 aa, 30 single-point mutants).
- **Evaluation protocol:** Rank-Stage profiles protein metadata/MSA Neff/pLDDT and writes Python dynamic weighting over 6 specialized predictors; Reason-Stage pools top 200 variants per model (<=1,400) and audits ranks, pLDDT, RSA, charge, and position diversity. Standalone LLMs also tested zero-shot and 10/20/30-shot.
- **Controls included:** Unweighted 6-expert mean (VenusRAR-Ensemble), standalone LLM zero-shot/few-shot, and random selection. Missing: non-LLM learned meta-ensemble, deterministic biophysical/diversity filter, and wet-lab Cas12i3 control arm.
- **Compute and tools:** Orchestrates 6 specialized pLM/structure/MSA models plus AlphaFold2 pLDDT/RSA/charge and consumes ~249M LLM tokens across ProteinGym (~912k output tokens/protein, Table 7).
- **Metrics:** Spearman rho (ProteinGym 217); Normalized Max Score, Top 1/5/10% precision, and Top-K hit rate (DMS99); HEK293T EGxxFP fluorescence fold-change (Cas12i3).
- **Wet-lab protocol:** 30 single-point Cas12i3 mutants tested in HEK293T cells via EGxxFP fluorescence ratio normalized to WT=1.0; no control arm from unweighted ensemble or single pLM.

## Key Results

| Task | Regime | Benchmark | Metric | General LLM | Protein Baseline | Evidence Status | Notes |
|---|---|---|---|---|---|---|---|
| Zero-shot variant effect prediction | Intrinsic LLM (text only, no tools) | ProteinGym substitutions (217 assays) | Mean Spearman rho | DeepSeek-Reasoner (standalone, no tools): 0.159 | ESM2-650M / VenusREM / unweighted 6-expert ensemble: 0.414 (ESM2-650M) / 0.518 (VenusREM) / 0.542 (6-expert ensemble) | **Supported (controlled)** | Clean demonstration that a standalone reasoning LLM (0.159) sits far below single-sequence pLMs (0.414) and specialist ensembles (0.542); also below random on DMS99 Max at N=40. |
| Zero-shot variant effect prediction | LLM agent orchestrating specialized protein tools | ProteinGym 217, DMS99 (31 assays, N=30), and wet-lab Cas12i3 (N=30) | Spearman / Norm Max / Hit@5 / Wet-lab >WT | VenusRAR (DeepSeek-Reasoner + 6 specialized experts): 0.551 Spearman (+0.009 over 0.542 ensemble); DMS99 N=30 Max 68.42% & Hit@5 4.52; Cas12i3 14/30 (46.7%) >WT (up to 5.05x) | Unweighted 6-expert ensemble (no non-LLM diversity/biophysical filter tested): 0.542 Spearman; DMS99 N=30 Max 62.22% & Hit@5 1.29; Cas12i3 control not run | **Suggestive (confounded)** | Almost all global ranking power (0.542 of 0.551) comes from the 6 specialized models; no non-LLM diversity/pLDDT filter baseline or Cas12i3 experimental control arm is run, and prompts/mutations are withheld. |

## Confounder and Leakage Audit (8 Axes)

| Axis | Status | Evidence |
|---|---|---|
| Temporal cutoff | **Uncontrolled** (`uncontrolled`) | ProteinGym (2023/2024) predates all tested 2025/2026 LLMs with no post-cutoff holdout (though standalone LLMs score poorly at 0.159). |
| Homology overlap | **Uncontrolled** (`uncontrolled`) | No homology split against pLM or LLM training sets; Cas12i3 is called an orphan protein without reporting identity to engineered Cas12i variants cited in Appendix E.1. |
| Cross-modal / literature recall | **Concern** (`evidenced_concern`) | Rank-Stage supplies molecule name and taxon (Appendix C.1), adding text annotations boosts standalone LLM DMS99 precision (Table 14), and Cas12i3 engineering literature predates the study. |
| Preprocessing artifacts | **Concern** (`evidenced_concern`) | Text claims '367% improvement' on Top-5 Hit Rate at N=40 where 4.73/1.29 = 3.67x (+267%) and Rank-Stage (5.38) actually beats Reason-Stage (4.73); Table 15 repeats identical 20-shot and 30-shot numbers; 1-8 failed error files/run (Table 12). |
| Downstream oracle filtering | **Concern** (`evidenced_concern`) | System re-weights and filters 6 SOTA specialized protein predictors whose unweighted average already reaches 0.542 Spearman. |
| Compute & tool asymmetry | **Concern** (`evidenced_concern`) | Uses 6 specialized models + AF2 pLDDT/RSA + 249M LLM tokens (~912k output tokens/protein) compared against single models. |
| Baseline fairness | **Concern** (`evidenced_concern`) | No deterministic non-LLM diversity/biophysical heuristic filter or learned meta-ensemble baseline; Cas12i3 wet-lab assay has no pLM-only control arm. |
| Data & wet-lab transparency | **Concern** (`evidenced_concern`) | System prompts (App C-D), Cas12i3 logs (App E.2), exact mutations, and per-variant raw fluorescence measurements are withheld. |
