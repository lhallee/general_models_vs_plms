# General-Purpose Language Models vs Ridge on Protein Language Models (FLIP2 and Post-Cutoff FGFR2)

- **Study ID:** `our_2026_flip2_in_context`
- **Authors:** Logan Hallee (Synthyra) and collaborators
- **Date:** 2026-11-22 (updated 2026-11-24)
- **Source tier:** Our controlled experiment (`our_experiment`)
- **Verification depth:** Full text verified (`full_text`)
- **Tasks:** Few-shot / in-context variant effect and optimization, Zero-shot variant effect prediction
- **Regimes:** Intrinsic LLM (text only, no tools)
- **Models tested:** Claude Mythos 5.2 (harness, low and high effort), Claude Fable 5.2 (API, low and high effort), Claude Opus 5.5 (API), Claude Opus 5 (API), Claude Haiku 5.5 (API)
- **Protein-specific baselines:** ESMC-600M embedding ridge, ESMC-6B embedding ridge, ESMC-300M embedding ridge, ESM-2 650M embedding ridge, One-hot + zero-shot ridge (ESMC-6B, ESMC-600M, ESMC-300M, ESM-2 650M), Zero-shot masked marginals / PPLL (ESMC-6B, ESMC-600M, ESMC-300M, ESM-2 650M), One-hot ridge, nearest-neighbor, composition, random vector, untrained ESMC-300M, shuffled/permuted ESMC controls
- **Wet-lab validation:** No
- **Code and data availability:** Full code, exact prompts, answers, transcript audit, summary tables, figures, and Hub dataset publicly released.
- **Primary links:** [https://github.com/lhallee/general_models_vs_plms](https://github.com/lhallee/general_models_vs_plms), [https://huggingface.co/datasets/Synthyra/general_models_vs_plms_flip2_20261121](https://huggingface.co/datasets/Synthyra/general_models_vs_plms_flip2_20261121)

## Takeaway

With 256 labeled variants and no tools, Claude Mythos 5.2 ties the best ESMC ridge on both FLIP2 (0.380 vs 0.367) and a post-cutoff FGFR2 screen (0.561 vs 0.572), and averaging their rankings beats both (0.443 / 0.611). Prompt controls show roughly half the score comes from genuine in-context learning and half from prior protein knowledge, with Rhomax driven entirely by prior knowledge.

## Methods

- **Datasets and benchmarks:** All 16 splits of the 7 FLIP2 datasets plus 2 splits (random and by_position) of a post-cutoff MaveDB screen (urn:mavedb:00001329-a-3, FGFR2 kinase domain erdafitinib resistance, published 2026-11-20 after the June 2026 Claude training cutoff).
- **Evaluation protocol:** Models receive 256 labeled training variants as plain text with the protein unnamed and rank 100 held-out queries per prompt (first 100 queries per split for Mythos 5.2 via audited single-prompt subagents; all 300 queries per split for public API models, kept strictly separate). Ridge baselines fit on the identical rounded 256 rows with leave-one-out penalty selection.
- **Controls included:** Residue/position cipher (k256_cipher), reflected labels (k256_flipped), shuffled labels (k256_shuffled), zero-shot named protein (k0_described), verbatim recall prompt (k0_recall) against 2,000 random deals, dataset recognition probe, post-cutoff FGFR2 splits, and 5 embedding-ridge controls (random, composition, untrained ESMC-300M, shuffled residues, permuted positions).
- **Compute and tools:** No external tools or code execution (subagent transcripts audited to verify Read prompt + Write answer only). Validation effort sweep selected low effort for Mythos 5.2 (median 2,156 thinking tokens/prompt); high effort also tested on probe splits.
- **Metrics:** Mean Spearman rank correlation across splits with 95% paired bootstrap intervals (2,000 draws), plus NDCG, top-decile ROC AUC, PR AUC, balanced MCC, and within-stratum Spearman.
- **Wet-lab protocol:** None (retrospective FLIP2 and prospective post-cutoff MaveDB evaluation).

## Key Results

| Task | Regime | Benchmark | Metric | General LLM | Protein Baseline | Evidence Status | Notes |
|---|---|---|---|---|---|---|---|
| Few-shot / in-context variant effect and optimization | Intrinsic LLM (text only, no tools) | FLIP2 (16 splits, k=256, first 100 queries) | Mean Spearman (95% paired bootstrap CI) | Claude Mythos 5.2 (low effort): 0.380 (0.333 to 0.421) | ESMC-600M embedding ridge (best ridge): 0.367 (0.319 to 0.409); diff +0.013 (-0.038 to +0.065) | **Supported (controlled)** | Pre-specified tie (3 wins, 3 losses, 10 undecided). Excluding Rhomax (driven by prior knowledge), Mythos scores 0.36 vs ridge 0.38 (-0.02, -0.07 to +0.03). Fable 5.2 API also ties best ridge on all 300 queries (0.365 vs 0.355). |
| Few-shot / in-context variant effect and optimization | Intrinsic LLM (text only, no tools) | Post-cutoff FGFR2 MaveDB (2 splits, k=256, first 100 queries) | Mean Spearman (95% CI) | Claude Mythos 5.2 (low effort): 0.561 (0.458 to 0.646) | One-hot + ESMC-6B zero-shot ridge (best ridge): 0.572 (0.469 to 0.663) | **Supported (controlled)** | Assay published 2026-11-20 after June 2026 model cutoff. Tie holds without measurement leakage (though FGFR2 sequence itself is known; cipher drops fgfr2/random from 0.662 to 0.342). |
| Few-shot / in-context variant effect and optimization | Intrinsic LLM (text only, no tools) | FLIP2 (16 splits) & FGFR2 (2 splits), k=256 | Mean Spearman (95% CI) | Equal-weight rank average (Mythos 5.2 + best ESMC ridge): 0.443 FLIP2 / 0.611 FGFR2 | Best individual method (Mythos 0.380 / ESMC ridge 0.367): +0.063 (+0.033 to +0.092) over Mythos; +0.077 over ridge on FLIP2 | **Supported (controlled)** | Untuned equal-weight rank average beats both models with intervals excluding zero, showing complementary error profiles. |
| Few-shot / in-context variant effect and optimization | Intrinsic LLM (text only, no tools) | 7 FLIP2 probe splits (k=256 controls) | Mean Spearman (95% CI) | Claude Mythos 5.2 (plain 0.496 -> cipher 0.278 / flipped -0.235 / k0 0.391): 0.278 cipher; -0.235 flipped; 0.391 k0_described | One-hot ridge 0.209 (vs cipher); ESMC-6B zero-shot 0.318 (vs k0): Cipher beats one-hot ridge by +0.070; k0_described 0.391 vs ESMC-6B 0.318 | **Supported (controlled)** | Separates true in-context learning (flipped reverses to -0.235 on 6/8 sets; cipher retains 0.278) from prior protein knowledge (Rhomax stays +0.583 when flipped and drops to -0.020 under cipher). |

## Confounder and Leakage Audit (8 Axes)

| Axis | Status | Evidence |
|---|---|---|
| Temporal cutoff | **Controlled** (`controlled`) | FLIP2 predates the June 2026 cutoff (Zenodo 2026-01-30), so the experiment explicitly pairs FLIP2 with post-cutoff MaveDB urn:mavedb:00001329-a-3 (published 2026-11-20) plus k0_recall exact-match tests (0/400 exact continuous matches). |
| Homology overlap | **Controlled** (`controlled`) | Evaluates FLIP2's extrapolation splits (by_position, close_to_far, low_to_high, unseen wild-types/proteins) and FGFR2 by_position; post-cutoff coverage is currently one protein (FGFR2), motivating our post-cutoff MaveDB expansion. |
| Cross-modal / literature recall | **Controlled** (`controlled`) | Explicitly quantified via dataset recognition probe (Mythos named all 7 FLIP2 proteins/studies, unnamed FGFR2 source), k256_cipher (drops probe mean 0.496 -> 0.278), k256_flipped (exposes Rhomax prior reliance at +0.583), and k0_recall. |
| Preprocessing artifacts | **Controlled** (`controlled`) | All methods train on identically rounded prompt labels; exact-match pools separate discrete (nucb, rhomax) from continuous splits; Mythos subagent harness route is kept strictly separate from public API models. |
| Downstream oracle filtering | **N/A** (`not_applicable`) | Direct variant ranking without generative candidate filtering. |
| Compute & tool asymmetry | **Controlled** (`controlled`) | Zero tools or code execution permitted (verified via transcript audit); validation-selected low effort reported alongside high-effort sweep (+0.027, CI includes zero); API and subagent routes never mixed. |
| Baseline fairness | **Controlled** (`controlled`) | Nine ridge baselines across ESM-2 650M and ESMC 300M/600M/6B fit on identical 256 rows plus five negative controls; note best ridge is chosen on test queries (favoring baselines) and no fine-tuned ESMC baseline was run yet. |
| Data & wet-lab transparency | **Controlled** (`controlled`) | All code, exact prompt files, model answers, audit scripts, and tables are publicly released; limitation: Mythos 5.2 ran 100 queries per split (1 draw, 1 answer) due to usage limits. |

## External Critiques and Commentary

- **Anthony Gitter** (2026-11-25, x_thread): [https://x.com/anthonygitter/status/2125456207464817118](https://x.com/anthonygitter/status/2125456207464817118) - Cited as an exemplar of rigorous in-context controls and post-cutoff DMS curation in contrast to OpenAI's unaudited ProteinGym benchmark.
