# Guided Inverse Folding with LLMs

- **Study ID:** `rowan_2026_guided_inverse_folding`
- **Authors:** Ishaan Ganti (Rowan Scientific)
- **Date:** 2026-10-07
- **Source tier:** Blog post / repository (`blog_repo`)
- **Verification depth:** Full text verified (`full_text`)
- **Tasks:** Inverse folding and sequence design, De novo backbone and binder design
- **Regimes:** Intrinsic LLM + code execution (no specialized protein models), LLM generation + specialized oracle filter / sequence model
- **Models tested:** GPT-6 Astra (High), GPT-5.6 Sol (High), GPT-5.6 Terra (High), Claude Opus 5 (Medium)
- **Protein-specific baselines:** ProteinMPNN (T=0.1, used to generate 3 of 4 in-context example sequences; not tabulated head-to-head)
- **Wet-lab validation:** No
- **Code and data availability:** Prompt templates and RBFE protocol in appendices; no public GitHub repo or numerical pass-rate table released.
- **Primary links:** [https://rowansci.com/blog/guided-inverse-folding-llms](https://rowansci.com/blog/guided-inverse-folding-llms)

## Takeaway

On four post-cutoff Gly-stripped PDB backbones, frontier LLMs fail zero-shot inverse folding but generate refoldable designs down to 40-50% identity when seeded with four compatible example sequences (native + ProteinMPNN) and a Python sandbox. Gains from iterative ESMFold2/Boltz-2 feedback are small and confounded by missing no-feedback controls, uncapped inter-round reuse, asymmetric Claude timeouts, and purely computational evaluation.

## Methods

- **Datasets and benchmarks:** 5 PDB backbones stripped to N/CA/C/O (all Gly): 4 post-cutoff (9VGJ, 9N30, 27AH, 9TLN, June-Aug 2026) + 1 pre-cutoff (9C14, Aug 2025). Plus a ligand-selectivity pocket redesign case study on 5WGD (240 aa) against estradiol (E2), estrone (E1), and testosterone (T).
- **Evaluation protocol:** Because zero-shot inverse folding failed in pilots, every prompt provides 4 compatible sequences (1 PDB reference + 3 ProteinMPNN T=0.1 sequences) and enforces <=70%, 50%, or 40% sequence identity caps against examples and within-batch (not across rounds). 3 rounds x 6 proposals in a closed-book Python 3.11 sandbox with ESMFold2/Boltz-2 refolding metrics fed back in rounds 2-3. 5WGD uses GPT-6 Astra to propose 8 sequences/ligand, co-folds with Boltz-2/ESMFold2, picks 1/ligand with fresh GPT-6 Astra, and evaluates via Rowan RBFE.
- **Controls included:** 4/5 post-cutoff Gly-stripped backbones, no-network Python sandbox, strict identity caps against input examples. Missing: no parallel no-feedback iteration control and no head-to-head ProteinMPNN benchmark.
- **Compute and tools:** Closed-book Python 3.11 sandbox (numpy/scipy/pandas) + 2 rounds of ESMFold2/Boltz-2 metrics. OpenAI models ran at High effort with no turn timeout; Claude Opus 5 downgraded to Medium effort with 600-900 s prompt timeouts.
- **Metrics:** Refolding pass rate (ESMFold2 & Boltz-2 Ca RMSD <= 2.5 A, TM-score >= 0.80, pLDDT >= 75); RBFE delta-delta-G (kcal/mol) on 5WGD.
- **Wet-lab protocol:** None (purely computational refolding and RBFE).

## Key Results

| Task | Regime | Benchmark | Metric | General LLM | Protein Baseline | Evidence Status | Notes |
|---|---|---|---|---|---|---|---|
| Inverse folding and sequence design | LLM generation + specialized oracle filter / sequence model | 5 PDB backbones (4 post-cutoff June-Aug 2026) at 70%/50%/40% identity caps | ESMFold2 + Boltz-2 refolding pass rate (plot-only, no scalar table) | GPT-6 Astra / GPT-5.6 Sol / GPT-5.6 Terra / Claude Opus 5 (with 4 example sequences): Zero-shot failed; with 4 examples GPT-6 Astra leads and produces passing designs at 50% (and 27AH 40% cap: 28/82 ref match); Opus 5 struggles even on 9TLN | ProteinMPNN (supplied 3/4 examples; no head-to-head pass rate reported): Not tabulated | **Suggestive (confounded)** | Post-cutoff Gly-stripped backbones control memorization well, showing zero-shot fails without examples while guided redesign succeeds down to 40-50% ID; confounded by ProteinMPNN seed examples, lack of inter-round ID caps, asymmetric Claude timeouts, and missing no-feedback control. |
| De novo backbone and binder design | LLM generation + specialized oracle filter / sequence model | 5WGD ligand-selectivity pocket redesign (N=1 candidate/ligand) | Rowan RBFE selectivity (kcal/mol) | GPT-6 Astra + Boltz-2/ESMFold2 co-folding + GPT-6 Astra selector: Testosterone design: +3.16 vs E1, +3.83 vs E2 (cycle disc 0.05); E1 design: +2.57 vs E2, +2.32 vs T (disc 2.01); E2 design failed (-0.71 vs E1) | Reference 5WGD/E2 (+4.10 vs E1, +2.43 vs T): No specialized ligand-design baseline | **Suggestive (confounded)** | Pre-cutoff 5WGD target, N=1 selected candidate per ligand, computational RBFE only (E1 cycle discrepancy 2.01 kcal/mol). |

## Confounder and Leakage Audit (8 Axes)

| Axis | Status | Evidence |
|---|---|---|
| Temporal cutoff | **Controlled** (`controlled`) | 4 of 5 main inverse-folding backbones (9VGJ, 9N30, 27AH, 9TLN) were released in PDB June-August 2026 after model cutoffs; pre-cutoff 9C14 and 5WGD are explicitly noted. |
| Homology overlap | **Concern** (`evidenced_concern`) | Zero-shot failed, so every prompt supplied 4 compatible sequences (1 native + 3 ProteinMPNN); identity caps were not enforced across rounds and Appendix D shows models hugging the identity ceiling. |
| Cross-modal / literature recall | **Controlled** (`controlled`) | 4/5 main backbones are post-cutoff, Gly-stripped, and run in a no-network sandbox (pre-cutoff 9C14 and 5WGD remain exposed). |
| Preprocessing artifacts | **Concern** (`evidenced_concern`) | Native reference exempted from refolding filter; inter-round sequence reuse uncapped; GPT-5.6 Terra violated the 70% cap on 3/6 final-round 9C14 sequences. |
| Downstream oracle filtering | **Concern** (`evidenced_concern`) | 3/4 input sequences come from ProteinMPNN, rounds 2-3 use ESMFold2/Boltz-2 feedback, and 5WGD selects 1/8 candidates via Boltz-2/ESMFold2 co-folds. |
| Compute & tool asymmetry | **Concern** (`evidenced_concern`) | OpenAI models ran at High effort without turn timeouts, whereas Claude Opus 5 was downgraded to Medium effort with 600-900 s timeouts. |
| Baseline fairness | **Concern** (`evidenced_concern`) | Authors admit no parallel no-feedback iteration control was run, and ProteinMPNN was not benchmarked head-to-head on the same identity-cap criteria. |
| Data & wet-lab transparency | **Concern** (`evidenced_concern`) | No wet-lab validation, no numerical pass-rate table in text, and no public repository link. |

## External Critiques and Commentary

- **Anthony Gitter** (2026-11-25, x_thread): [https://x.com/anthonygitter/status/2125456200464572481](https://x.com/anthonygitter/status/2125456200464572481) - Highlighted Rowan's choice of newly released post-cutoff PDB structures as a necessary control missing from OpenAI's GPT-6 inverse-folding benchmark.
