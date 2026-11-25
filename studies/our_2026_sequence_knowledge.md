# Sequence Knowledge vs Property Prediction Error in Claude Mythos 5.2 and Masked Protein Language Models

- **Study ID:** `our_2026_sequence_knowledge`
- **Authors:** Logan Hallee (Synthyra) and collaborators
- **Date:** 2026-11-23 (updated 2026-11-24)
- **Source tier:** Our controlled experiment (`our_experiment`)
- **Verification depth:** Full text verified (`full_text`)
- **Tasks:** Sequence recall and memorization, Zero-shot variant effect prediction
- **Regimes:** Intrinsic LLM (text only, no tools)
- **Models tested:** Claude Mythos 5.2 (harness, low effort)
- **Protein-specific baselines:** ESMC-6B, ESMC-600M, ESMC-300M, ESM-2 650M, Most common visible residue
- **Wet-lab validation:** No
- **Code and data availability:** All prompts, answers, tables, and scripts released in repository.
- **Primary links:** [https://github.com/lhallee/general_models_vs_plms](https://github.com/lhallee/general_models_vs_plms)

## Takeaway

Claude Mythos 5.2 reconstructs natural rhodopsins at 91%/76%/57% accuracy under 15%/50%/85% masking (far above ESMC-6B's 59%/54%/13%, and 100% on FGFR2 at 50% masking), confirming strong sequence memorization. Yet sequence recall does not predict Mythos's absorption-maximum error (Spearman -0.15, CI -0.41 to +0.13), showing sequence recall and property accuracy are distinct.

## Methods

- **Datasets and benchmarks:** 118 sequences from FLIP2 and FGFR2: 6 dataset parents (amylase, IRED, NucB, TrpB, PDZ3, FGFR2), 100 Rhomax queries (48 scored by Mythos 5.2), and 12 shuffled negative controls.
- **Evaluation protocol:** 15%, 50%, and 85% of residues masked at shared positions across all models. Masked pLMs scored by single-site pseudo-log-likelihood (PPLL) and masked recovery; Mythos 5.2 scored by share of masked residues restored across 6 rotated prompt files with no overlap across levels.
- **Controls included:** Shuffled sequence copies (12 sequences), most-common visible residue baseline, and partial Spearman controlling for distance of true absorption maximum from the median.
- **Compute and tools:** Single text prompt per batch via audited subagents (Read prompt, Write answer only) at low effort; ESMC/ESM-2 run in Docker on oracle_gh200.
- **Metrics:** Masked residue recovery accuracy at 15/50/85% masking; Spearman correlation (2,000 bootstrap draws) between sequence knowledge and absolute absorption-maximum prediction error (nm).
- **Wet-lab protocol:** None (computational memorization and error correlation audit).

## Key Results

| Task | Regime | Benchmark | Metric | General LLM | Protein Baseline | Evidence Status | Notes |
|---|---|---|---|---|---|---|---|
| Sequence recall and memorization | Intrinsic LLM (text only, no tools) | 48 natural rhodopsins at 15% / 50% / 85% masking (+12 shuffled controls) | Share of masked residues restored | Claude Mythos 5.2: 0.914 (15%), 0.758 (50%), 0.568 (85%); shuffled 0.096-0.107 | ESMC-6B (best pLM): 0.594 (15%), 0.540 (50%), 0.125 (85%); shuffled 0.052-0.113 | **Supported (controlled)** | All-or-nothing sequence recall: Mythos returned 34/48, 32/48, and 23/48 rhodopsins at >=90% accuracy (and FGFR2 100% at 50% mask), whereas no pLM reached 90% on any rhodopsin. |
| Sequence recall and memorization | Intrinsic LLM (text only, no tools) | 48 rhodopsins (Mythos) / 100 rhodopsins (pLMs) absorption-maximum error | Spearman (knowledge vs absolute error, 95% CI) | Claude Mythos 5.2 (k0 / k256): -0.147 (-0.407 to +0.131) k0; -0.044 (-0.331 to +0.237) k256 | ESMC-600M / ESM-2 650M / ESMC-300M / ESMC-6B PPLL: -0.502 (ESMC-600M), -0.198 (ESM-2), -0.077 (ESMC-300M), +0.033 (ESMC-6B) | **Supported (controlled)** | Memorizing a sequence does not predict Mythos's property-prediction accuracy (8 never-recited rhodopsins have 3.5 nm median error vs 4.0 nm for 19 always-recited). |

## Confounder and Leakage Audit (8 Axes)

| Axis | Status | Evidence |
|---|---|---|
| Temporal cutoff | **N/A** (`not_applicable`) | Explicitly designed as a direct sequence-memorization measurement on public pre-cutoff proteins (including confirming FGFR2 sequence is memorized even though its Nov 2026 assay is post-cutoff). |
| Homology overlap | **Concern** (`evidenced_concern`) | Experiment record explicitly notes Rhomax queries include variants of shared wild-types, so the 48 sequences are not fully independent and bootstrap CIs are slightly narrow; masked recovery mixes exact sequence recall with family knowledge. |
| Cross-modal / literature recall | **Controlled** (`controlled`) | Direct target of measurement; rotated prompt files prevent cross-level leakage within a batch. |
| Preprocessing artifacts | **Controlled** (`controlled`) | Shuffled controls confirm zero composition shortcut (0.096-0.107 vs 0.110 majority guess); length mismatches audited (1,005/1,005 right length, 1 visible residue altered). |
| Downstream oracle filtering | **N/A** (`not_applicable`) | Direct masked sequence completion without external oracle. |
| Compute & tool asymmetry | **Controlled** (`controlled`) | No tools allowed (transcripts audited); identical masked positions across Mythos 5.2 and all four pLMs. |
| Baseline fairness | **Controlled** (`controlled`) | Evaluates ESMC-300M/600M/6B and ESM-2 650M on both exact PPLL and identical multi-mask completion. |
| Data & wet-lab transparency | **Controlled** (`controlled`) | All prompts, answers, and per-sequence CSVs released; power limit transparently stated (n=48 detects \|r\| >= 0.4). |
