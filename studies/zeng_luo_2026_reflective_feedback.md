# Does Reflective LLM Feedback Survive a Wet-Lab-Scale Label Budget?

- **Study ID:** `zeng_luo_2026_reflective_feedback`
- **Authors:** Haoxuan Zeng and Xin Luo (University of Michigan)
- **Date:** 2026-11-01
- **Source tier:** Preprint (`preprint`)
- **Verification depth:** Full text verified (`full_text`)
- **Tasks:** Few-shot / in-context variant effect and optimization
- **Regimes:** LLM agent orchestrating specialized protein tools
- **Models tested:** Claude Opus 5, Claude Sonnet 5, Claude Haiku 4.5
- **Protein-specific baselines:** ESM-2 150M + ridge/MLP/GBR predictor with: No edits, Random edits, Local search (1x/5x wall time), Static ridge (Hsu et al. 2022), Fixed greedy/UCB/div-greedy
- **Wet-lab validation:** No
- **Code and data availability:** Summary snapshot and validation scripts only; raw assay labels, embeddings, full run records, and controller-call logs are not released (Appendix A).
- **Primary links:** [https://arxiv.org/abs/2611.01908](https://arxiv.org/abs/2611.01908)

## Takeaway

In a well-controlled, protein-anonymized replay across 36 ProteinGym assays at 96 labels/round, iterative Claude Opus 5 reflection over an ESM-2 150M pipeline fails to beat random configuration edits (Delta_z = -0.008) or a single round-1 call (-0.011), and trails wall-time-matched local CV search on position-disjoint splits (0.461 vs 0.506). Gains appear only in exploratory arms where acquisition is fixed externally.

## Methods

- **Datasets and benchmarks:** 36 ProteinGym single-substitution DMS assays (101,201 variants; 1,500-6,000 variants/assay, <=400 aa, 1 assay per UniProt accession).
- **Evaluation protocol:** Active-learning replay starting at 96 labels and adding 96/round for 6 rounds (384-672 labels in rounds 3-6) on 25% sealed random and position-disjoint test splits (3-5 seeds, 4,211 runs). LLM reads anonymized 5-fold CV diagnostics (no protein names or sequences) and proposes JSON edits to a 15-field ESM-2 150M modeling/acquisition pipeline.
- **Controls included:** Anonymized protein-free prompts, size-matched random edits, wall-time-matched local CV search (1x and 5x), static augmented ridge, one-call (round-1 only) and scalar-only reflection ablations, and Benjamini-Hochberg multiplicity correction.
- **Compute and tools:** LLM proposes config JSONs only (no code/search); local search explicitly matched to 1x and 5x LLM wall time.
- **Metrics:** Primary: sealed-test Fisher z = arctanh(rho) averaged over rounds 3-6 (TOST equivalence margin 0.02). Secondary: top-10% recovery R_96.
- **Wet-lab protocol:** None (offline ProteinGym replay at wet-lab label scale).

## Key Results

| Task | Regime | Benchmark | Metric | General LLM | Protein Baseline | Evidence Status | Notes |
|---|---|---|---|---|---|---|---|
| Few-shot / in-context variant effect and optimization | LLM agent orchestrating specialized protein tools | 36 ProteinGym assays, 96 labels/round (random & position-disjoint splits) | Fisher z (rounds 3-6) & R_96 | Claude Opus 5 reflective controller over ESM-2 150M: Random z=0.590 (R_96=0.391); Position-disjoint z=0.461 (R_96=0.319) | Random edits / No edits / Local search 5x / Static ridge: Random split: Random edits z=0.597 (Delta_z=-0.008, [-0.034,+0.018]), No edits 0.603; Position-disjoint: Local search 5x z=0.506, Static ridge 0.492 | **Supported (controlled)** | Rigorous negative result: joint LLM reflection gives no test Spearman gain over random edits (p=0.55) or one-shot call (-0.011), and trails local CV search (0.506) on position-disjoint splits. Exploratory fixed-acquisition arm (LLM + random acq.) reaches z=0.647 (+0.043). |

## Confounder and Leakage Audit (8 Axes)

| Axis | Status | Evidence |
|---|---|---|
| Temporal cutoff | **Controlled** (`controlled`) | Prompts strip all protein names, sequences, and assay identifiers, exposing only anonymized CV/stratum statistics (Appendix F.1). |
| Homology overlap | **Controlled** (`controlled`) | 36 distinct UniProt accessions tested under both random and 25% position-disjoint splits (Table 3). |
| Cross-modal / literature recall | **Controlled** (`controlled`) | Complete protein anonymization prevents biological literature recall; LLM acts solely on ML diagnostics. |
| Preprocessing artifacts | **Concern** (`evidenced_concern`) | Authors transparently report post-pilot search redesign, validator rejections (Opus no-menu rejected 640 entries across 509/648 calls yet reached z=0.638), and 1 missing Search-5x seed in the 5-seed grid. |
| Downstream oracle filtering | **N/A** (`not_applicable`) | Supervised active-learning replay over frozen ESM-2 150M features. |
| Compute & tool asymmetry | **Controlled** (`controlled`) | Matched against 1x and 5x wall-time local search and static augmented ridge. |
| Baseline fairness | **Controlled** (`controlled`) | 20 well-matched non-LLM and ablation arms; only noted gap is missing one-call control for exploratory fixed-acquisition arms. |
| Data & wet-lab transparency | **Concern** (`evidenced_concern`) | Offline replay only, and raw assay labels, embeddings, and full controller logs are excluded from the released snapshot (Appendix A p.8). |
