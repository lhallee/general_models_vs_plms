# Mutation-Site ESMC Stacking, ConFit LoRA Preference Tuning, and Expanded Post-Cutoff MaveDB Suite

- **Study ID:** `our_2026_esmc_adaptation_and_postcutoff`
- **Authors:** Logan Hallee (Synthyra) and collaborators
- **Date:** 2026-11-25 (updated 2026-11-25)
- **Source tier:** Our controlled experiment (`our_experiment`)
- **Verification depth:** Full text verified (`full_text`)
- **Tasks:** Few-shot / in-context variant effect and optimization
- **Regimes:** Intrinsic LLM (text only, no tools)
- **Models tested:** Qwen/Qwen3-14B (zero-spend open-weight BF16 pilot, direct mode), Claude Mythos 5.2 (reference from FLIP2 ICL experiment), Claude Fable 5.2 (reference from FLIP2 API experiment)
- **Protein-specific baselines:** ESMC-600M & ESMC-6B mutation-site local pooling + zero-shot ridge, ESMC-600M & ESMC-6B 5-fold out-of-fold multi-view stacked ridge (mean + mutsite + one-hot + zero-shot), ConFit-style KL-anchored LoRA preference-tuned ESMC-300M & ESMC-600M, ESM-2 650M, ESMC-300M/600M/6B zero-shot and mean-pooled ridges
- **Wet-lab validation:** No
- **Code and data availability:** Full code, Docker entrypoint, pinned model revisions, negative controls, and paired-bootstrap tables in repository.
- **Primary links:** [https://github.com/lhallee/general_models_vs_plms](https://github.com/lhallee/general_models_vs_plms)

## Takeaway

The prior FLIP2 tie between Claude Mythos 5.2 (0.380) and ESMC ridge (0.367) was an artifact of global mean-pooling dilution: extracting mutation-site local embeddings plus zero-shot scores and 5-fold stacking lifts pure frozen ESMC (600M+6B) to 0.4567 on FLIP2 (+0.0768 strictly above Mythos 5.2, 95% CI +0.0279 to +0.1238) and 0.5485 across 6 new post-cutoff GBA1/ARAF splits, while hybridizing with Mythos 5.2 raises the overall peak to 0.4667. ConFit LoRA ESMC helps strongly on single-site post-cutoff DMS (+0.053 to +0.065) but trails frozen local/stacked ridges under FLIP2's heavier distribution shift.

## Methods

- **Datasets and benchmarks:** Identical 16 FLIP2 splits + 2 FGFR2 splits (k=256, seed 0) plus 6 new post-June-2026 MaveDB splits across two unrelated human protein families: GBA1 GCase activity (urn:mavedb:00001327-a-1, 2026-11-11, 536 aa, 9,886 missense), ARAF kinase activity (urn:mavedb:00001314-0-4, 2026-10-16, 606 aa, 10,955 missense), and ARAF abundance (urn:mavedb:00001314-0-3, 2026-10-16, 606 aa, 5,037 missense). Max ungapped window identity <= 12.5% between post-cutoff families and <= 16.7% against FLIP2.
- **Evaluation protocol:** All models fit strictly on the 256 context rows with no test peeking. Track 1A combines frozen mean embeddings with zero-shot scores; Track 1B extracts mutated-site hidden states (h_mut[M], h_mut[M] - h_wt[M]) and fits 5-fold out-of-fold stacked multi-view ridges; Track 1C attaches rank-8 zero-initialized LoRA adapters to ESMC attention layers and trains 15 epochs with Bradley-Terry ranking loss + KL=0.1 anchor to frozen logits.
- **Controls included:** Shuffled-label negative controls for fusion, mutsite, stacked ridge (+0.009 to +0.011 on FLIP2; -0.077 to -0.042 on post-cutoff), and ConFit LoRA (drops -0.13 to -0.24 below real labels); epoch-0 ConFit parity verified to 0.0002 against zero-shot masked marginals; PRKN pre-cutoff upload explicitly audited and rejected.
- **Compute and tools:** Ran inside pinned Docker image general_models_vs_plms:cu128-20261121 on oracle_gh200 (1x GH200); zero paid API spend.
- **Metrics:** Mean Spearman correlation with 95% paired bootstrap intervals (2,000 draws) on exact first 100 queries (vs Mythos 5.2) and all 300 queries (vs frozen ridges and Fable 5.2).
- **Wet-lab protocol:** None (computational evaluation on primary MaveDB / FLIP2 experimental assays).

## Key Results

| Task | Regime | Benchmark | Metric | General LLM | Protein Baseline | Evidence Status | Notes |
|---|---|---|---|---|---|---|---|
| Few-shot / in-context variant effect and optimization | Intrinsic LLM (text only, no tools) | FLIP2 (16 splits, k=256, exact first 100 queries) | Mean Spearman (95% paired bootstrap CI) | Claude Mythos 5.2 (@harness-low): 0.3800 (0.3327 to 0.4219) | Pure frozen ESMC-600M + ESMC-6B mutation-site+ZS ensemble (single ESMC-600M: 0.4288): 0.4567 (0.4103 to 0.4960); diff +0.0768 (+0.0279 to +0.1238) vs Mythos, +0.0900 vs old ridge | **Supported (controlled)** | Breaks the prior FLIP2 tie strictly in favor of pLMs without any LLM or fine-tuning. Shuffled-label controls score +0.0093 to +0.0107. On all 300 queries the pure frozen pLM ensemble reaches 0.4380 vs 0.3554 old ridge and 0.3649 Claude Fable 5.2. |
| Few-shot / in-context variant effect and optimization | Intrinsic LLM (text only, no tools) | FLIP2 (16 splits, k=256, exact first 100 queries) | Mean Spearman (95% paired bootstrap CI) | Equal-weight rank average (Claude Mythos 5.2 + mutsite+ZS ESMC-600M): 0.4667 (0.4219 to 0.5048) | Claude Mythos 5.2 alone (0.3800) / prior hybrid record (0.443): +0.0867 (+0.0580 to +0.1129) over Mythos alone; +0.1000 over old ridge | **Supported (controlled)** | Confirms strong complementarity between LLM reasoning and improved local pLM representations, raising the hybrid ceiling from 0.443 to 0.467. |
| Few-shot / in-context variant effect and optimization | Intrinsic LLM (text only, no tools) | Expanded post-June-2026 MaveDB suite (6 splits: GBA1, ARAF activity, ARAF abundance; q300) | Mean Spearman (95% paired bootstrap CI) | Prior FGFR2 best pLM recipe (ridge_onehot_plus_esmc_6b): 0.4606 (0.4177 to 0.4989) | ridge_stacked_esmc_6b / ridge_stacked_esmc_600m / ConFit LoRA ESMC-300M & 600M: Stacked 6B: 0.5485 (+0.0879, +0.0507 to +0.1240); Stacked 600M: 0.5430; ConFit 300M/600M: 0.5161 / 0.5119 | **Supported (controlled)** | On single-site post-cutoff DMS, ConFit LoRA ESMC gains +0.053 to +0.065 over zero-shot (0.451-0.459) and beats old ridges, while frozen stacked ESMC-6B reaches 0.5485 (0.728 on GBA1, 0.669 on ARAF activity). Under FLIP2's stronger extrapolation shift, ConFit LoRA ESMC (0.310-0.334) trails frozen mutation-site/stacked ridges (0.414-0.457). |
| Few-shot / in-context variant effect and optimization | Intrinsic LLM (text only, no tools) | Controlled 4-arm pilot on amylase/by_mutation (FLIP2) & fgfr2/random (post-cutoff), first 100 queries | Spearman correlation (all-100 median-filled) | Qwen/Qwen3-14B (BF16, direct mode, zero API spend): amylase: k0 -0.121, k256 +0.403, cipher +0.077, flipped -0.337; fgfr2: k0 +0.053, k256 +0.404, cipher +0.406, flipped -0.372 | ridge_onehot (0.500 / 0.462), Claude Mythos 5.2 (0.661 / 0.662), ESMC-600M mutsite+ZS (0.609 / 0.546): One-hot ridge +0.500 / +0.462; ESMC-600M+6B +0.665 / +0.605 | **Supported (controlled)** | Separates tabular ICL from protein prior: Qwen3-14B has zero biological prior (k0 near 0) and scores identically on plain vs enciphered post-cutoff FGFR2 (+0.404 vs +0.406, reversing cleanly to -0.372 under flipped labels), trailing one-hot ridge, ESMC, and Mythos 5.2. |

## Confounder and Leakage Audit (8 Axes)

| Axis | Status | Evidence |
|---|---|---|
| Temporal cutoff | **Controlled** (`controlled`) | Evaluates 6 newly published post-June-2026 MaveDB splits across GBA1 and ARAF (Oct-Nov 2026) alongside FGFR2 (Nov 2026), explicitly rejecting pre-cutoff PRKN (Feb 2026 bioRxiv). |
| Homology overlap | **Controlled** (`controlled`) | Audited pairwise ungapped window identity across all post-cutoff and FLIP2 parents (max 12.5% pairwise among post-cutoff families, <= 16.7% to FLIP2); evaluates both random and position-disjoint splits. |
| Cross-modal / literature recall | **Controlled** (`controlled`) | Pure pLM methods read protein sequences only; Mythos 5.2 comparison uses identical first-100 queries from our controlled FLIP2 experiment. |
| Preprocessing artifacts | **Controlled** (`controlled`) | Verified 0 wild-type mismatches across 25,878 missense rows, verified epoch-0 ConFit parity to 0.0002 against zero-shot masked marginals, guarded reference baseline tables against key overwrite, and reported two-chain pdz3 averaging as a conservative floor. |
| Downstream oracle filtering | **N/A** (`not_applicable`) | Direct continuous variant ranking without candidate filtering. |
| Compute & tool asymmetry | **Controlled** (`controlled`) | All methods trained strictly on the exact same 256 labeled variants with fixed recipes and no test peeking; frozen mutsite/stacked ridges run in ~2.4 min (600M) / ~4.4 min (6B) for all 18 splits. |
| Baseline fairness | **Controlled** (`controlled`) | Tests frozen mean-pool, one-hot+ZS, fusion, mutation-site local pooling, 5-fold multi-view stacking, and ConFit KL-anchored LoRA across ESM-2 650M and ESMC 300M/600M/6B with shuffled-label negative controls. |
| Data & wet-lab transparency | **Controlled** (`controlled`) | All MaveDB URNs, licenses (CC0 / CC BY-SA 4.0), model revisions, negative controls, and paired-bootstrap JSONs recorded. |
