# Binder Optimization with LLMs and Specialized Models (TNKS2 Small-Molecule Lead Optimization)

- **Study ID:** `rowan_2026_binder_optimization`
- **Authors:** Ishaan Ganti (Rowan Scientific)
- **Date:** 2026-08-17
- **Source tier:** Blog post / repository (`blog_repo`)
- **Verification depth:** Full text verified (`full_text`)
- **Tasks:** Protein-protein and small-molecule docking, De novo backbone and binder design
- **Regimes:** LLM generation + specialized oracle filter / sequence model, Intrinsic LLM + code execution (no specialized protein models)
- **Models tested:** GPT-5.6 Sol (High), Claude Opus 5 (High), Claude Opus 5 (High, no RDKit ablation)
- **Protein-specific baselines:** FLOWR.root (3D pocket flow matching), CReM, REINVENT 4 (LibInvent), DiffDec (3D diffusion)
- **Wet-lab validation:** No
- **Code and data availability:** Full tables, prompt appendix, and constraint lists in blog post; no public repo link or experimental validation.
- **Primary links:** [https://rowansci.com/blog/binder-optimization-llms](https://rowansci.com/blog/binder-optimization-llms)

## Takeaway

On TNKS2 small-molecule scaffold decoration, 1-shot CReM (-7.84 kcal/mol) and FLOWR (-6.00) beat 1-shot LLMs (-3.22 to -5.44), while 4 rounds of RBFE feedback lift GPT-5.6 Sol to -9.23 kcal/mol and an F85Q pocket-mutation control proves genuine structural responsiveness. However, specialized models were not run iteratively, and claims rest on noisy computational RBFE on one target without wet-lab verification.

## Methods

- **Datasets and benchmarks:** Tankyrase 2 (TNKS2) neutral phenylquinazolinone scaffold (Ligand 1a, experimental -8.55 kcal/mol; Rowan RBFE calibration R^2=0.411, rho=0.621, RMSE=0.768 kcal/mol); plus an F85Q pocket-mutation control.
- **Evaluation protocol:** 5 structural/chemical constraints + negative docking/MM-GBSA filter; 32 ligands/batch. Specialized baselines run 1-shot only with imposed decoration site/size priors; LLMs run in a no-network Python/RDKit sandbox choosing sites freely for 1-shot AND 4 iterative rounds with RBFE ddG, MM/GBSA, and docking feedback.
- **Controls included:** Wild-type vs F85Q pocket-mutation control (testing whether fluorinated para substituents adapt to pocket sterics), accidental Opus 5 no-RDKit ablation, and cross-method cycle-closure RBFE network check (RMS error 2.59 kcal/mol).
- **Compute and tools:** 23 RBFE jobs / 739 transformations (56.5 h). LLMs run 4 iterative RBFE-guided rounds with free site selection while specialized baselines run 1-shot only.
- **Metrics:** Hub-and-spoke RBFE ddG (kcal/mol vs parent) and ddG per added heavy atom.
- **Wet-lab protocol:** None (computational RBFE only).

## Key Results

| Task | Regime | Benchmark | Metric | General LLM | Protein Baseline | Evidence Status | Notes |
|---|---|---|---|---|---|---|---|
| Protein-protein and small-molecule docking | LLM generation + specialized oracle filter / sequence model | TNKS2 scaffold decoration (32 ligands/round, RBFE ddG kcal/mol) | Top-1 / Top-5 RBFE ddG (more negative = tighter) | GPT-5.6 Sol / Claude Opus 5 (1-shot -> 4 rounds RBFE feedback): 1-shot: GPT-5.6 Sol -5.44, Opus 5 -3.22 (-4.09 no RDKit); Round 4: GPT-5.6 Sol -9.23, Opus 5 -5.80 (-6.86 no RDKit); F85Q control drops fluorinated para site from 24/32 -> 1/32 (GPT) & 18/32 -> 4/32 (Opus) | CReM / FLOWR.root (mixed) / DiffDec / REINVENT 4 (1-shot only): 1-shot: CReM -7.84, FLOWR mixed -6.00, DiffDec -4.18, REINVENT 4 -3.99 | **Suggestive (confounded)** | F85Q pocket-mutation control cleanly proves LLMs respond to 3D pocket geometry rather than only reciting PDB TNKS2 inhibitors. However, specialized baselines were restricted to 1-shot with imposed site priors (where CReM and FLOWR beat LLMs), and all scores rest on computational RBFE (cycle-closure RMS error 2.59 kcal/mol) on one scaffold. |

## Confounder and Leakage Audit (8 Axes)

| Axis | Status | Evidence |
|---|---|---|
| Temporal cutoff | **Concern** (`evidenced_concern`) | TNKS2 and its PDB inhibitors predate model cutoffs (Opus 5 1-shot designs show higher Tanimoto similarity to PDB TNKS2 inhibitors), partially mitigated by the F85Q mutant pocket control. |
| Homology overlap | **N/A** (`not_applicable`) | Single-target small-molecule lead optimization on one scaffold. |
| Cross-modal / literature recall | **Controlled** (`controlled`) | TNKS2 is unnamed in prompt text (inferable from PDB coords), and F85Q pocket-mutation control drops fluorinated para substitutions from 24/32 -> 1/32 (GPT-5.6 Sol) and 18/32 -> 4/32 (Opus 5). |
| Preprocessing artifacts | **Concern** (`evidenced_concern`) | Appendix B discloses a structural non-binding Zn ion was accidentally stripped during PDBFixer prep (post-hoc check R^2=0.86, rho=0.90); DiffDec required OBabel + re-docking while FLOWR bypassed docking. |
| Downstream oracle filtering | **Concern** (`evidenced_concern`) | Graded solely by Rowan RBFE (Merck calibration R^2=0.411, rho=0.621; cross-method cycle-closure RMS error 2.59 kcal/mol, max 4.33 kcal/mol) with no wet-lab IC50/KD. |
| Compute & tool asymmetry | **Concern** (`evidenced_concern`) | LLMs ran 4 iterative rounds with RBFE feedback and free site choice while specialized baselines ran 1-shot only with imposed site/size distributions (explicitly acknowledged by author). |
| Baseline fairness | **Concern** (`evidenced_concern`) | Asymmetric 1-shot specialist vs 4-round iterative LLM comparison; single 32-ligand batch without multi-seed variance. |
| Data & wet-lab transparency | **Concern** (`evidenced_concern`) | Detailed numbers and prompts published in blog post, but no wet-lab validation and only one receptor/scaffold. |
