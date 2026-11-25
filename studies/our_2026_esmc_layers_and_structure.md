# ESMC Intermediate Layers (23/27/60) + ESMFold2 Structural Stream and Penultimate Layer (-2) Evaluation

- **Study ID:** `our_2026_esmc_layers_and_structure`
- **Authors:** Logan Hallee (Synthyra) and collaborators
- **Date:** 2026-11-25 (updated 2026-11-25)
- **Source tier:** Our controlled experiment (`our_experiment`)
- **Verification depth:** Full text verified (`full_text`)
- **Tasks:** Few-shot / in-context variant effect and optimization
- **Regimes:** Intrinsic LLM (text only, no tools)
- **Models tested:** Claude Mythos 5.2 (reference from FLIP2/FGFR2 ICL experiment), Claude Fable 5.2 (reference from FLIP2/FGFR2 API experiment)
- **Protein-specific baselines:** ESMC-300M (depth L23, penultimate L29, final L30) + ESMFold2 width-256 structural stream, ESMC-600M (depth L27, penultimate L35, final L36) + ESMFold2 width-256 structural stream, ESMC-6B (depth L60, penultimate L79, final L80) + ESMFold2 width-256 structural stream
- **Wet-lab validation:** No
- **Code and data availability:** Full code (esmc_layers.py), Docker runner, SHA-256 verified canonical checkpoints, negative controls, and paired-bootstrap tables in repository.
- **Primary links:** [https://github.com/lhallee/general_models_vs_plms](https://github.com/lhallee/general_models_vs_plms)

## Takeaway

Combining ESMC's canonical intermediate depth layers (23/27/60) with the released ESMFold2 width-256 structural stream lifts post-cutoff GBA1/ARAF to 0.5858 (single 6B: 0.5783 vs 0.5501 final), post-cutoff FGFR2 q300 to 0.4910 (vs 0.4491 prior ridge / 0.4247 Fable 5.2), and single-model ESMC-600M FLIP2 to 0.4428 (+0.0628 strictly above Mythos 5.2), driven largely by the 256-dim structural stream. Independently, the penultimate layer (-2) beats final layer everywhere on ESMC-6B and on all post-cutoff suites at every scale, lifting pooled ESMC-6B alone to 0.3786/0.3828 on FLIP2, 0.5132/0.4048 on FGFR2, and 0.5300 on GBA1/ARAF.

## Methods

- **Datasets and benchmarks:** All 24 splits across 4 benchmark panels: 16 FLIP2 splits, 2 post-cutoff FGFR2 splits (Nov 2026), and 6 expanded post-cutoff MaveDB splits (GBA1 activity, ARAF activity, ARAF abundance; Oct-Nov 2026).
- **Evaluation protocol:** Exact canonical FastPLMs TokenTapExecutor extraction via foundry.embedding.canonical.build_runtime (BF16, all l+2 tokens including CLS and EOS, max 2,046 residues) tapping depth layer (23/27/60), penultimate layer (-2: 29/35/79), final layer (-1: 30/36/80), and the released ESMFold2 width-256 layer-mixture structural stream in one pass per batch. Reuses precomputed masked-marginal / likelihood zero-shot scores; concatenates per-chain slots on two-chain pdz3/single_to_double; fits pooled (mean, mean+var), mutated-site local, and 5-fold out-of-fold stacked multi-view ridges strictly on the 256 context rows.
- **Controls included:** Shuffled-label controls for depth_struct_mv, depth_struct_stacked, and penult_stacked on ESMC-600M (+0.041 / +0.031 / -0.020 on FLIP2 q100; -0.075 / -0.070 / +0.016 on GBA1/ARAF); non-degenerate layer separation verified on every split; exact final-layer parity verified against original ridge_esmc_600m.
- **Compute and tools:** Ran inside pinned Docker image general_models_vs_plms:cu128-20261121 on oracle_gh200 (1x GH200); peak VRAM 1.5 GB (300M), 2.1 GB (600M), 13.1 GB (6B); 22m 27s total across all 3 models x 24 splits; zero paid API spend.
- **Metrics:** Mean Spearman correlation with 95% paired bootstrap intervals (2,000 draws) on exact first 100 queries (vs Mythos 5.2) and all 300 queries (vs prior ridges and Fable 5.2).
- **Wet-lab protocol:** None (computational evaluation on primary FLIP2 and post-cutoff MaveDB experimental assays).

## Key Results

| Task | Regime | Benchmark | Metric | General LLM | Protein Baseline | Evidence Status | Notes |
|---|---|---|---|---|---|---|---|
| Few-shot / in-context variant effect and optimization | Intrinsic LLM (text only, no tools) | Post-cutoff MaveDB suites: GBA1+ARAF (6 splits, q300) and FGFR2 (2 splits, q300) | Mean Spearman (95% paired bootstrap CI) | Prior final-layer baselines (ridge_onehot_plus_esmc_6b 0.4606 GBA1/ARAF, 0.4491 FGFR2; Claude Fable 5.2 0.4247 FGFR2): 0.4606 (GBA1+ARAF) / 0.4491 (FGFR2 ridge) / 0.4247 (FGFR2 Fable 5.2) | ESMC depth (23/27/60) + ESMFold2 width-256 structural stream 5-fold stacked ridge (600M+6B rank average): GBA1+ARAF: 0.5858 (0.5487 to 0.6179; single 6B: 0.5783, +0.1177 over baseline, +0.0282 over final stacked); FGFR2 q300: 0.4865 (600M+6B) / 0.4910 (all-3 scales) | **Supported (controlled)** | Depth layer alone drops sharply (-0.13 to -0.16), while the 256-dim structural stream alone already beats final-layer mean pooling on every benchmark (0.5329 on GBA1+ARAF for 6B). Combining depth + structural stream drives large gains on ARAF activity (+0.099 by_position), ARAF abundance (+0.068 by_position), GBA1 (0.725-0.747), and FGFR2 unseen positions. |
| Few-shot / in-context variant effect and optimization | Intrinsic LLM (text only, no tools) | FLIP2 (16 splits, k=256, exact first 100 queries and all 300 queries) | Mean Spearman (95% paired bootstrap CI) | Claude Mythos 5.2 (@harness-low, q100) / Claude Fable 5.2 (@low, q300): 0.3800 (Mythos q100) / 0.3649 (Fable q300) | Single frozen ESMC-600M stacked with structural stream (L27 depth+struct: 0.4428; L35 penult+struct: 0.4439): 0.4428 (+0.0628, +0.0197 to +0.1043 vs Mythos) / 0.4439 (+0.0639, +0.0220 to +0.1050 vs Mythos); q300: 0.4250 / 0.4261 | **Supported (controlled)** | A single 600M parameter frozen ESMC model now strictly beats Claude Mythos 5.2 on FLIP2 by +0.063 to +0.064, with massive cross-protein structural transfer gains on hydro/to_P06241 (0.345 -> 0.696-0.703) and hydro/to_P01053 (0.212 -> 0.588). |
| Few-shot / in-context variant effect and optimization | Intrinsic LLM (text only, no tools) | All 24 splits (FLIP2 q100/q300, FGFR2 q100/q300, GBA1+ARAF q300) | Mean Spearman (paired against final layer) | Final-layer ESMC-6B pooled mean (L80: 0.3636 FLIP2 q100, 0.3496 q300, 0.3556 FGFR2 q100, 0.3283 q300, 0.4931 GBA1+ARAF): Final L80 pooled: 0.3636 / 0.3496 / 0.3556 / 0.3283 / 0.4931 | Independent penultimate layer (-2, L79 on ESMC-6B) pooled mean+var without structural stream or zero-shot: Penult L79 pooled: 0.3786 FLIP2 q100, 0.3828 FLIP2 q300 (+0.0332), 0.5132 FGFR2 q100 (+0.1576), 0.4048 FGFR2 q300 (+0.0765), 0.5300 GBA1+ARAF (+0.0369) | **Supported (controlled)** | Penultimate layer (-2) beats final layer everywhere on ESMC-6B and on both post-cutoff suites at all three scales (300M/600M/6B). Pure pooled penultimate ESMC-6B (0.3786 q100 / 0.3828 q300) already ties Mythos 5.2 and beats Fable 5.2 with no zero-shot scores, no mutsite pooling, and no structural stream; stacked penultimate 600M+6B reaches 0.5754 on GBA1+ARAF. |

## Confounder and Leakage Audit (8 Axes)

| Axis | Status | Evidence |
|---|---|---|
| Temporal cutoff | **Controlled** (`controlled`) | Evaluated identically on FLIP2 and on all 8 post-June-2026 MaveDB splits (FGFR2, GBA1, ARAF activity and abundance). |
| Homology overlap | **Controlled** (`controlled`) | Preserves exact homology-audited splits (<= 12.5% pairwise identity across post-cutoff families, <= 16.7% to FLIP2) and includes position-disjoint and cross-protein transfer splits. |
| Cross-modal / literature recall | **Controlled** (`controlled`) | Pure pLM methods read amino-acid sequences only; Mythos 5.2 and Fable 5.2 references come from identical controlled query slices. |
| Preprocessing artifacts | **Controlled** (`controlled`) | Uses exact SHA-256 verified canonical checkpoints and FastPLMs TokenTapExecutor; fixes multi-chain pdz3 concatenation; verifies non-degenerate layer separation on every split and final-layer parity with original ridges. |
| Downstream oracle filtering | **N/A** (`not_applicable`) | Direct continuous variant ranking across full held-out query sets. |
| Compute & tool asymmetry | **Controlled** (`controlled`) | Every feature family trained strictly on the exact same 256 context rows with no test peeking; all four streams extracted in one BF16 pass per batch (22m 27s total for 3 models x 24 splits). |
| Baseline fairness | **Controlled** (`controlled`) | Directly compares final (-1), penultimate (-2), intermediate depth (23/27/60), structural-only (width-256), depth+structural, and penultimate+structural across all three ESMC scales (300M, 600M, 6B) with shuffled-label negative controls. |
| Data & wet-lab transparency | **Controlled** (`controlled`) | All checkpoints pinned by SHA-256 via foundry.embedding.canonical, all timings, VRAM peaks, negative controls, and paired-bootstrap tables recorded. |
