# RosettaSearch: Multi-Objective Inference-Time Search for Protein Sequence Design

- **Study ID:** `kshirsagar_2026_rosettasearch`
- **Authors:** Meghana Kshirsagar, Allen Nie, Ching-An Cheng, Fanglei Xue, Rahul Dodhia, Juan Lavista Ferres, Kevin K. Yang, Frank DiMaio
- **Date:** 2026-04-19 (updated 2026-04-30)
- **Source tier:** Preprint (`preprint`)
- **Verification depth:** Full text verified (`full_text`)
- **Tasks:** Inverse folding and sequence design, De novo backbone and binder design
- **Regimes:** LLM agent orchestrating specialized protein tools, LLM generation + specialized oracle filter / sequence model
- **Models tested:** o4-mini, Gemini-3, o3-mini, Unnamed VLM
- **Protein-specific baselines:** LigandMPNN (single-pass init), ProteinMPNN (single-pass init), BindCraft (init), Budget-matched random mutations + RF3, Chai-1 (independent refolding check)
- **Wet-lab validation:** No
- **Code and data availability:** Promised on GitHub upon release (Sec 7); no repository URL available in preprint.
- **Primary links:** [https://arxiv.org/abs/2604.17175](https://arxiv.org/abs/2604.17175)

## Takeaway

RosettaSearch shows inference-time LLM priority search over RosettaFold3 feedback improves single-pass ProteinMPNN/LigandMPNN sequences (reaching 89.5% vs 72.4% SR-1 on Dayhoff de novo backbones without native sequences) and beats budget-matched random mutations. However, PDB gains lean heavily on placing the native sequence in the prompt, RF3 overfitting drops PDB success from 20.5% to 5.3% on Chai-1, and no budget-matched MPNN best-of-75 baseline or wet-lab check is provided.

## Methods

- **Datasets and benchmarks:** ~400 (394 in Table 3 / 414 in Sec 6) low-fidelity PDB-D monomers, 275 de novo RFdiffusion backbones from Dayhoff Atlas (no native sequence context), and 48 (50 sampled) BindCraft binders.
- **Evaluation protocol:** Inference-time Priority Search / Sequential Revision over 75 RosettaFold3 (RF3, single-sequence no-MSA) calls per protein (3 textual step sizes x 25 RF3 calls; ~30.6 min/protein), guided by local residue-range pLDDT/TM-score/RMSD text feedback.
- **Controls included:** Budget-matched random mutations in flagged regions, Dayhoff no-native-sequence test, independent Chai-1 refolding evaluation, and context ablation table (Table 7). Missing: compute-matched LigandMPNN/ProteinMPNN best-of-75 + RF3 selection baseline.
- **Compute and tools:** 75 RF3 folding evaluations + LLM search (~30.6 min/protein) vs single-pass LigandMPNN/ProteinMPNN.
- **Metrics:** Single-sequence pLDDT, TM-score, Ca RMSD, SR-1 (pLDDT >= 0.8 & TM >= 0.8), SR-2 (+ RMSD <= 1.5 A) under RF3 and Chai-1.
- **Wet-lab protocol:** None (purely computational).

## Key Results

| Task | Regime | Benchmark | Metric | General LLM | Protein Baseline | Evidence Status | Notes |
|---|---|---|---|---|---|---|---|
| Inverse folding and sequence design | LLM generation + specialized oracle filter / sequence model | 275 Dayhoff de novo backbones (no native sequence context) | RF3 SR-1 / SR-2 | RosettaSearch (o4-mini + 75 RF3 calls, ProteinMPNN init): SR-1 89.5% (246/275), SR-2 67.3% (185/275), RMSD 1.78 A | Single-pass ProteinMPNN (no budget-matched MPNN best-of-75): SR-1 72.4% (199/275), SR-2 45.5% (125/275), RMSD 3.32 A | **Suggestive (confounded)** | No native sequence leakage on Dayhoff, showing genuine RF3-guided search gains, but lacks a compute-matched ProteinMPNN-75 + RF3 baseline and Chai-1 cross-check on Dayhoff. |
| Inverse folding and sequence design | LLM generation + specialized oracle filter / sequence model | PDB-D monomers (LigandMPNN init, Table 2 & Table 7) | SR-1 under RF3 / Independent Chai-1 | RosettaSearch (o4-mini + native sequence in prompt + 75 RF3 calls): 20.5% (RF3) -> 5.3% (Chai-1); without native seq in prompt: 11.4% RF3 (Table 7) | Single-pass LigandMPNN / Random mutations + 75 RF3: 7.9% (RF3) / 3.1% (Chai-1) for both LigandMPNN and Random mutations | **Suggestive (confounded)** | Native sequence in prompt drives much of the PDB gain (11.4% -> 17.2%, Sec 3.2.2 notes LLM returning the native sequence), and RF3 overfitting cuts absolute SR-1 from 20.5% to 5.3% on Chai-1. |

## Confounder and Leakage Audit (8 Axes)

| Axis | Status | Evidence |
|---|---|---|
| Temporal cutoff | **Uncontrolled** (`uncontrolled`) | PDB-D monomers and BindCraft binders predate o4-mini and Gemini-3 with no cutoff holdout. |
| Homology overlap | **Uncontrolled** (`uncontrolled`) | Randomly sampled from PDB-D without homology clustering (RF3 runs MSA-free). |
| Cross-modal / literature recall | **Concern** (`evidenced_concern`) | PDB prompts explicitly supply the native sequence (Fig 10), raising SR-1 from 11.4% to 17.2% (Table 7), and Sec 3.2.2 documents the LLM returning the native sequence itself (Dayhoff removes this). |
| Preprocessing artifacts | **Concern** (`evidenced_concern`) | Sec 3.2.2 and A.4-A.5 document proxy hacking (poly-Ala tails, repeats, safe-residue overuse) and 44% (22/50) trace validation failures (budget violations, indexing bugs), plus minor N mismatches (394 vs 414). |
| Downstream oracle filtering | **Concern** (`evidenced_concern`) | Direct 75-call RF3 reward optimization exhibits heavy RF3 overfitting: PDB SR-1 drops from 20.5% under RF3 to 5.3% under independent Chai-1 (Table 2). |
| Compute & tool asymmetry | **Concern** (`evidenced_concern`) | Compares 75 RF3-guided LLM search calls (~30.6 min/protein) against single-pass LigandMPNN/ProteinMPNN. |
| Baseline fairness | **Concern** (`evidenced_concern`) | No budget-matched LigandMPNN/ProteinMPNN best-of-75 + RF3 selection baseline or ResiDPO comparison (only random mutations are budget-matched). |
| Data & wet-lab transparency | **Concern** (`evidenced_concern`) | Purely in silico (no wet-lab validation) and code/data not yet released. |
