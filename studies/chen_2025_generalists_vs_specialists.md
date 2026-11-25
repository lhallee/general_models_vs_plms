# Generalists vs. Specialists: Evaluating LLMs on Highly-Constrained Biophysical Sequence Optimization Tasks

- **Study ID:** `chen_2025_generalists_vs_specialists`
- **Authors:** Angelica Chen, Samuel D. Stanton, Frances Ding, Robert G. Alberstein, Andrew M. Watkins, Richard Bonneau, Vladimir Gligorijevic, Kyunghyun Cho, Nathan C. Frey
- **Date:** 2024-10-29 (updated 2025-07-29)
- **Source tier:** Peer-reviewed paper (`peer_reviewed`)
- **Verification depth:** Full text verified (`full_text`)
- **Tasks:** Few-shot / in-context variant effect and optimization
- **Regimes:** Intrinsic LLM (text only, no tools), Protein-finetuned LLM, LLM generation + specialized oracle filter / sequence model
- **Models tested:** OpenAI o1 (prompting), Gemini 2.0 Flash Thinking (prompting), Pythia 2.8B (LLOME SFT/DPO/MargE/REINFORCE), From-scratch LLaMA 226K
- **Protein-specific baselines:** LaMBO-2 (314K-param 1D CNN discrete diffusion + ensemble guidance), Genetic Algorithm (GA)
- **Wet-lab validation:** No
- **Code and data availability:** No GitHub link provided for LLOME or Ehrlich code in paper.
- **Primary links:** [https://arxiv.org/abs/2410.22296](https://arxiv.org/abs/2410.22296)

## Takeaway

On procedurally generated, strictly leakage-free Ehrlich functions, off-the-shelf prompting of o1 and Gemini 2.0 Flash Thinking mostly fails, while online MargE fine-tuning of Pythia 2.8B (LLOME-MARGE) beats SFT/DPO and wins f2. Yet a 314K-parameter specialist (LaMBO-2, 0.01% the size) wins f3 with better feasibility and diversity, and LLOME remains untested on real proteins.

## Methods

- **Datasets and benchmarks:** 4 procedurally generated closed-form Ehrlich integer-sequence functions (f1 easy, f2/f3 medium, f4 hard; stylized after antibody affinity maturation; Appendix A.2.7 checks GA rank correlation against DhfR 0.86, TFBind8 0.89, TrpB 0.73).
- **Evaluation protocol:** Prompting test of o1 and Gemini 2.0 Flash Thinking on f2 (8 samples). Main comparison: 10 GA seed rounds (10k evals) + 10 outer rounds x 2,000 oracle evals (30,000 total) comparing LLOME (online SFT/DPO/MargE/REINFORCE on Pythia 2.8B) vs LaMBO-2 (314K) and GA. Hyperparameters tuned on f2 only.
- **Controls included:** Procedurally generated leakage-free functions, integer alphabet removing biological priors, size-matched from-scratch LLaMA 226K ablation (App A.9), exact closed-form oracle, and matched 30,000-eval budget.
- **Compute and tools:** Matched 30,000 oracle evaluations; Pythia 2.8B is ~8,900x larger than 314K LaMBO-2 (transparently acknowledged and tested with 226K LLaMA).
- **Metrics:** Minimum simple regret (1 - f(x)), % unique, % feasible, hypervolume (Figs 3-5, 10-19; no scalar regret table).
- **Wet-lab protocol:** None (synthetic closed-form benchmark).

## Key Results

| Task | Regime | Benchmark | Metric | General LLM | Protein Baseline | Evidence Status | Notes |
|---|---|---|---|---|---|---|---|
| Few-shot / in-context variant effect and optimization | Intrinsic LLM (text only, no tools) | Procedural Ehrlich function f2 (8 prompted samples, Sec 6.1) | Minimum simple regret (0 = optimal, 1 = worst) | OpenAI o1 / Gemini 2.0 Flash Thinking (prompting): o1 min regret 0.9375; Gemini min regret 0 once (most outputs infeasible or regret ~1) | LaMBO-2 / LLOME-MARGE (online optimization): Reaches low/zero regret systematically on f2 | **Supported (controlled)** | Zero-leakage procedural benchmark shows off-the-shelf prompting of reasoning LLMs fails to optimize constrained biophysical-like sequence functions reliably. |
| Few-shot / in-context variant effect and optimization | Protein-finetuned LLM | Procedural Ehrlich functions f1-f4 (30,000 exact oracle evals) | Regret & feasibility (Figs 3-5) | LLOME-MARGE (Pythia 2.8B online fine-tuning): Wins on f2 (reaches regret 0); ties on f1 (easy) & f4 (hard); DPO collapses; 226K from-scratch LLaMA matches Pythia 2.8B up to 22k evals | LaMBO-2 (314K discrete diffusion specialist): Wins on f3 with higher diversity/feasibility at 0.01% of Pythia 2.8B parameter count | **Supported (controlled)** | Clean leakage-free controlled result on synthetic functions (caveat: no real biological sequences optimized by LLOME/LaMBO-2). |

## Confounder and Leakage Audit (8 Axes)

| Axis | Status | Evidence |
|---|---|---|
| Temporal cutoff | **Controlled** (`controlled`) | Ehrlich functions are procedurally generated from random seeds, guaranteeing zero pretraining contamination. |
| Homology overlap | **N/A** (`not_applicable`) | Synthetic discrete integer-sequence benchmark rather than natural protein families. |
| Cross-modal / literature recall | **Controlled** (`controlled`) | Integer tokens remove biological identity; from-scratch 226K LLaMA matches pretrained Pythia 2.8B up to 22k evals. |
| Preprocessing artifacts | **Concern** (`evidenced_concern`) | Hyperparameters tuned only on f2 (disadvantaging LaMBO-2 on transfer); DPO premature collapse deflates hypervolume; main tests set a=0 (no non-monotonic epistasis). |
| Downstream oracle filtering | **Controlled** (`controlled`) | Exact closed-form oracle eliminates surrogate overfitting; likelihood-reward miscalibration explicitly audited (Fig 17). |
| Compute & tool asymmetry | **Concern** (`evidenced_concern`) | Oracle calls are matched (30k), but Pythia 2.8B is ~8,900x larger than 314K LaMBO-2 (partially mitigated by 226K LLaMA ablation). |
| Baseline fairness | **Controlled** (`controlled`) | Matched seeding, matched oracle budget, hyperparameter sweeps for both sides, and size-matched from-scratch control. |
| Data & wet-lab transparency | **Concern** (`evidenced_concern`) | Synthetic-only optimization benchmark (no real proteins or wet lab) and no public code URL in the paper. |
