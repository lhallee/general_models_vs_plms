# Large Language Model is Secretly a Protein Sequence Optimizer

- **Study ID:** `wang_2025_secretly_optimizer`
- **Authors:** Yinkai Wang, Jiaxing He, Yuanqi Du, Xiaohui Chen, Jianan Canal Li, Li-Ping Liu, Xiaolin Xu, Soha Hassoun
- **Date:** 2025-01-16 (updated 2025-01-17)
- **Source tier:** Preprint (`preprint`)
- **Verification depth:** Full text verified (`full_text`)
- **Tasks:** Few-shot / in-context variant effect and optimization
- **Regimes:** LLM generation + specialized oracle filter / sequence model
- **Models tested:** Llama-3.1-8B-Instruct
- **Protein-specific baselines:** None (compared only against a random-mutation Evolutionary Algorithm)
- **Wet-lab validation:** No
- **Code and data availability:** No code or data repository URL provided; prompts omitted for 4 of 5 tasks.
- **Primary links:** [https://arxiv.org/abs/2501.09274](https://arxiv.org/abs/2501.09274)

## Takeaway

Prompting Llama-3.1-8B-Instruct as an evolutionary mutation operator outperforms a naive random-mutation EA on Potts-model and ML-surrogate landscapes (Syn-3bfo, AAV, GFP) while losing on GB1 exact DMS (4.80 vs 5.72). Missing pLM/MLDE/BO baselines, explicit protein naming, unverified ML surrogates, and unreported EA-fallback rates prevent concluding LLMs outperform specialized sequence optimizers.

## Methods

- **Datasets and benchmarks:** 5 fitness optimization landscapes: GB1 (4 sites, exact DMS), TrpB (4 sites, exact DMS), Syn-3bfo (Potts model), AAV (ML surrogate oracle), GFP (ML surrogate oracle).
- **Evaluation protocol:** Closed-loop evolutionary search (population 32/48/96 x 4 or 8 rounds, 3 seeds): Llama-3.1-8B-Instruct sees 2 parents + fitness and proposes 1 mutant offspring, falling back silently to EA crossover/mutation on invalid outputs.
- **Controls included:** Oracle-budget-matched random-mutation EA; Hamming-distance and multi-objective constraints.
- **Compute and tools:** Matched oracle evaluation counts vs EA (higher GPU inference compute for LLM).
- **Metrics:** Top-1, Top-10, and Top-50 mean oracle fitness across 3 seeds.
- **Wet-lab protocol:** None (DMS lookup and ML surrogate evaluation only).

## Key Results

| Task | Regime | Benchmark | Metric | General LLM | Protein Baseline | Evidence Status | Notes |
|---|---|---|---|---|---|---|---|
| Few-shot / in-context variant effect and optimization | LLM generation + specialized oracle filter / sequence model | GB1, TrpB, Syn-3bfo, AAV, GFP (96x4 / 96x8) | Top-1 oracle fitness (mean +/- SD, 3 seeds) | Llama-3.1-8B-Instruct as EA mutation operator: GB1 4.80, TrpB 0.78, Syn-3bfo 2.83, AAV 0.76, GFP 0.97 | Random-mutation EA (no pLM/BO/MLDE baseline): GB1 5.72 (EA wins), TrpB 0.74, Syn-3bfo 1.85, AAV 0.44, GFP 0.50 | **Suggestive (confounded)** | Beats naive random EA on Syn-3bfo and ML surrogates (AAV/GFP) while losing on exact GB1 DMS; zero pLM/BO baselines, explicit protein/position naming in GB1 prompt, and unreported EA-fallback rates. |

## Confounder and Leakage Audit (8 Axes)

| Axis | Status | Evidence |
|---|---|---|
| Temporal cutoff | **Uncontrolled** (`uncontrolled`) | Llama-3.1-8B (Jul 2024) post-dates GB1 (2016), GFP (2016), AAV (2021), 3bfo (2022), and overlaps TrpB (2024). |
| Homology overlap | **N/A** (`not_applicable`) | Single-landscape closed-loop optimization (though ML surrogates lack OOD validity checks, Sec 4.1). |
| Cross-modal / literature recall | **Concern** (`evidenced_concern`) | Appendix A.3 prompt explicitly names 'B1 domain of streptococcal protein G', WT sequence, exact positions 39/40/41/54, and instructs the model to use its own knowledge. |
| Preprocessing artifacts | **Concern** (`evidenced_concern`) | Algorithm 1 falls back silently to EA crossover/mutation when LLM returns None with zero fallback rate reported; prompts for 4/5 tasks omitted. |
| Downstream oracle filtering | **Concern** (`evidenced_concern`) | Largest wins (AAV 0.76 vs 0.44; GFP 0.97 vs 0.50) are evaluated solely on ML surrogate oracles without independent validation or wet lab. |
| Compute & tool asymmetry | **Controlled** (`controlled`) | Oracle call budgets and initial pools match between LLM and EA. |
| Baseline fairness | **Concern** (`evidenced_concern`) | Compares only against random-mutation EA with zero pLM, Bayesian optimization, MLDE, or AdaLead/PEX baselines, and only the LLM sees WT/task descriptions. |
| Data & wet-lab transparency | **Concern** (`evidenced_concern`) | No wet-lab validation, no code/data repository link, and incomplete prompt/hyperparameter reporting. |
