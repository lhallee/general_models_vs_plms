# Preference Tuning ProGen2 (DPO and Bradley-Terry) vs Claude Mythos 5.2 and ESMC Ridge on FLIP2

- **Study ID:** `our_2026_progen2_preference_tuning`
- **Authors:** Logan Hallee (Synthyra) and collaborators
- **Date:** 2026-11-24 (updated 2026-11-25)
- **Source tier:** Our controlled experiment (`our_experiment`)
- **Verification depth:** Full text verified (`full_text`)
- **Tasks:** Few-shot / in-context variant effect and optimization
- **Regimes:** Intrinsic LLM (text only, no tools)
- **Models tested:** Claude Mythos 5.2 (reference from FLIP2 ICL experiment)
- **Protein-specific baselines:** ProGen2-small (151M) zero-shot, hidden-state ridge, DPO, Bradley-Terry, LoRA, slow LR, klarge, and DPO + NLL likelihood term (weights 0.01, 0.1, 1.0), ProGen2-base (764M) zero-shot, hidden-state ridge, DPO (fixed 20 epochs), ESMC-600M / ESMC-6B ridge
- **Wet-lab validation:** No
- **Code and data availability:** Full code, Docker entrypoint, pinned Hub checkpoints, tuning logs, and tables publicly released.
- **Primary links:** [https://github.com/lhallee/general_models_vs_plms](https://github.com/lhallee/general_models_vs_plms)

## Takeaway

Preference tuning ProGen2 on the same 256 FLIP2 rows trails Claude Mythos 5.2 and ESMC ridge under pre-specified held-out checkpoint selection (0.263 vs 0.380 / 0.367) and reaches a tie without passing either under a fixed 20-epoch ProGen2-base recipe (0.360 on FLIP2; 0.339 vs 0.351 ridge on untouched queries 101-300) or ProGen2-small + light NLL anchor (w=0.1: 0.348 on q101-300, 0.378 on q1-100, fixing the 5.6-nat likelihood collapse), while trailing both on post-cutoff FGFR2.

## Methods

- **Datasets and benchmarks:** Identical 16 FLIP2 splits and 2 post-cutoff FGFR2 splits from our_2026_flip2_in_context (k=256 training rows, 205 train / 51 held-out validation; plus klarge up to 2,048 rows).
- **Evaluation protocol:** Pre-specified recipe tunes all weights of ProGen2-small with Plackett-Luce ranked DPO (lr in {1e-5, 1e-4}, beta in {0.1, 2}, up to 20 epochs) or Bradley-Terry loss, selecting checkpoint on 51 held-out rows. Follow-up diagnostic tests fixed recipe (lr=1e-5, beta=2, 20 epochs, no validation selection) evaluated cleanly on held-out queries 101-300 for ProGen2-small and ProGen2-base.
- **Controls included:** Shuffled-label DPO control (drops -0.109 below untuned), 3 batch-order seeds, untuned zero-shot likelihood, hidden-state ridge, in-distribution 51-row evaluation of all baselines, and untouched queries 101-300 for fixed-recipe verification.
- **Compute and tools:** Ran in Docker on oracle_gh200; bidirectional float32 mean log-likelihood verified against Salesforce's reference sanity cross-entropy (2.4 small, 1.9 base).
- **Metrics:** Mean Spearman correlation with 95% paired bootstrap intervals (2,000 draws).
- **Wet-lab protocol:** None (computational baseline comparison).

## Key Results

| Task | Regime | Benchmark | Metric | General LLM | Protein Baseline | Evidence Status | Notes |
|---|---|---|---|---|---|---|---|
| Few-shot / in-context variant effect and optimization | Intrinsic LLM (text only, no tools) | FLIP2 (16 splits, k=256, first 100 queries) | Mean Spearman (95% CI) | Claude Mythos 5.2 (in-context): 0.380 (0.333 to 0.421) | ProGen2-small DPO (pre-specified val-selected) / ProGen2-base DPO (fixed 20 ep): 0.263 (diff -0.117 vs Mythos) / 0.360 (diff -0.020, -0.067 to +0.026) | **Supported (controlled)** | Pre-specified ProGen2-small DPO loses by 0.117 because i.i.d. validation selection hurts under FLIP2 distribution shift; fixed 20-epoch ProGen2-base reaches 0.360 (tie) on FLIP2 and still trails Mythos on post-cutoff FGFR2 (0.379 vs 0.561, -0.183). |
| Few-shot / in-context variant effect and optimization | Intrinsic LLM (text only, no tools) | FLIP2 untouched queries 101-300 (16 splits, k=256) | Mean Spearman (95% CI) | Best ESMC ridge (reference): 0.351 | ProGen2-small fixed 20ep / ProGen2-base fixed 20ep: 0.324 (-0.026, -0.063 to +0.013) / 0.339 (-0.012, -0.048 to +0.025) | **Supported (controlled)** | Evaluated on queries 101-300 that had no role in selecting the fixed recipe; tuned ProGen2 ties but never surpasses ESMC ridge. |
| Few-shot / in-context variant effect and optimization | Intrinsic LLM (text only, no tools) | FLIP2 untouched queries 101-300 & first 100 (16 splits, k=256) | Mean Spearman (95% CI) & query NLL | Best ESMC ridge (0.351 on q101-300) / Claude Mythos 5.2 (0.380 on q1-100): 0.351 / 0.380 | ProGen2-small fixed 20ep + preferred-row NLL term (weights 1.0 primary, 0.1, 0.01): w=1.0: 0.295 (-0.056 vs ridge); w=0.1: 0.348 (-0.002 vs ridge; 0.378 on q1-100); w=0.01: 0.344 | **Supported (controlled)** | Without an NLL anchor, DPO collapses query log-likelihood by 5.6 nats/residue (-7.71 vs -2.12 released). Adding preferred-row NLL (Pang et al. / ProGen3 loss) stops the collapse (-1.46 at w=0.1) and adds ~0.02 at light weights (0.348 vs 0.324), reaching a tie with ESMC ridge and Mythos 5.2 without beating either. |

## Confounder and Leakage Audit (8 Axes)

| Axis | Status | Evidence |
|---|---|---|
| Temporal cutoff | **Controlled** (`controlled`) | Includes post-cutoff FGFR2 splits where ProGen2-base fixed DPO (0.379) trails both Mythos 5.2 (0.561) and ESMC ridge (0.572). |
| Homology overlap | **Controlled** (`controlled`) | Explicitly isolates distribution shift by scoring all baselines on both the 51 i.i.d. held-out rows (ESMC-600M ridge 0.637 vs ProGen2-small 0.522) and shifted FLIP2 test queries. |
| Cross-modal / literature recall | **Controlled** (`controlled`) | ProGen2 is a sequence-only pLM; Mythos 5.2 reference comes from our controlled FLIP2/FGFR2 experiment. |
| Preprocessing artifacts | **Controlled** (`controlled`) | Explicitly avoids test-set selection bias by validating the post-hoc fixed recipe on untouched queries 101-300 and flagging first-100 scores as optimistic. |
| Downstream oracle filtering | **N/A** (`not_applicable`) | Direct variant ranking without candidate filtering. |
| Compute & tool asymmetry | **Controlled** (`controlled`) | Same 256 labeled training rows; full weight tuning on GPU compared honestly against zero-weight-update ESMC ridge and Mythos ICL. |
| Baseline fairness | **Controlled** (`controlled`) | Tests full-weight DPO, Bradley-Terry, hidden-state ridge, LoRA, 10x lower LR, klarge up to 2,048 rows, and ProGen2-base; transparently notes ESMC fine-tuning remains untested. |
| Data & wet-lab transparency | **Controlled** (`controlled`) | All checkpoints pinned by SHA, sanity cross-entropy checked, all code and tables public. |
