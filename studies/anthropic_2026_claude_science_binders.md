# Autonomous de Novo Protein Binder Design with Claude

- **Study ID:** `anthropic_2026_claude_science_binders`
- **Authors:** Claude Science and Amir Shanehsazzadeh (Anthropic)
- **Date:** 2026-08-18
- **Source tier:** Lab technical report (`technical_report`)
- **Verification depth:** Full text verified (`full_text`)
- **Tasks:** De novo backbone and binder design
- **Regimes:** LLM agent orchestrating specialized protein tools, LLM generation + specialized oracle filter / sequence model
- **Models tested:** Claude Mythos Preview (high effort), Claude Opus 4.8 (max effort)
- **Protein-specific baselines:** Orchestrated tools: PXDesign, RFdiffusion3, Genie 3, FreeBindCraft, BoltzGen, RFdiffusion, Proteina-Complexa, FoldCraft, BoltzDesign1, ProteinHunter, SolubleMPNN, ProteinMPNN, ESMC, ESMFold2, Protenix v2, Retrospective Adaptyv competition entries (not matched controls)
- **Wet-lab validation:** Yes
- **Code and data availability:** Full Hugging Face dataset (Anthropic/claude-protein-binder-design, CC BY 4.0 / MIT) with all 1,440 designs, raw CRO SPR sensorgrams, fits, integrated calls, provenance, and the 16,000-word protocol prompt.
- **Primary links:** [https://assets.claude.com/30bf50e22a01388bb29bf077ee3f244531594b7a.pdf](https://assets.claude.com/30bf50e22a01388bb29bf077ee3f244531594b7a.pdf), [https://huggingface.co/datasets/Anthropic/claude-protein-binder-design](https://huggingface.co/datasets/Anthropic/claude-protein-binder-design)

## Takeaway

Claude Mythos Preview and Opus 4.8 autonomously orchestrated open-source protein generators and a 3-model co-folding filter to achieve a 26.8% dual-CRO wet-lab SPR binder hit rate across 14 of 15 targets (49% at rank 1) with exemplary open data release. This validates agentic orchestration of specialized tools, not intrinsic LLM protein design, and lacks a compute-matched human-expert baseline.

## Methods

- **Datasets and benchmarks:** 16 protein targets given by name, UniProt ID, organism, oligomeric state, and cofactors (15 analyzed after excluding aggregating mature GDF-8: TNFa, VEGF-A, latent GDF-8, PD-L1, TREM2, IL-7Ra, TrkA, EGFR, Nipah G, BHRF1, RBX1, SpCas9 RNP, 15-PGDH, de novo BBF-14, E. coli MBP); 1,320 synthesized designs.
- **Evaluation protocol:** Autonomous agent runs a frozen ~16,000-word protocol with bioRxiv/offline literature corpus (including Adaptyv/ProteinBase competition reports), orchestrating >=7 backbone generators, SolubleMPNN/ProteinMPNN, novelty checks vs UniRef90/ProteinBase, and a 5-seed 3-model co-folding ensemble (ESMFold2, ESMFold2-Fast, Protenix v2; 4:1 ipSAE_min : sc-DockQ). Multi-target (48h, $50k Modal GPU budget) and single-target (24h, $10k/target).
- **Controls included:** Blinded dual-CRO synthesis and SPR validation (Adaptyv Bio + Twist Bioscience; Cohen's kappa = 0.71 across 1,235 shared designs), blinded dual-reader trace grading (96-98% agreement), off-target CLEC12A and mouse/cynomolgus ortholog specificity controls, binder novelty check (98% <30% PDB identity), and same-plate re-synthesis of RBX1 competition winner. No matched human-expert control campaign.
- **Compute and tools:** $10,000 per single-target 24h campaign (2.8x per-target compute of $50k/14-target 48h campaign) on Modal GPUs running 10+ specialized protein generation and co-folding models.
- **Metrics:** Integrated dual-CRO SPR binding hit rate, KD thresholds (<100 nM, <10 nM, <1 nM), rank-1/top-5/top-10 hit rates, and mean Average Precision (AP).
- **Wet-lab protocol:** Adaptyv Bio cell-free expression + immobilized single-cycle SPR (10 nM-1 uM duplicate, 1,320 designs) and Twist Bioscience HEK293 IgG1 Fc-fusion 6-point SPR (4.1 nM-1 uM, 1,260 designs) with blinded ordering and dual-reader curve grading.

## Key Results

| Task | Regime | Benchmark | Metric | General LLM | Protein Baseline | Evidence Status | Notes |
|---|---|---|---|---|---|---|---|
| De novo backbone and binder design | LLM agent orchestrating specialized protein tools | 15 targets, 1,320 designs across two blinded CROs (Adaptyv + Twist SPR) | Integrated wet-lab SPR hit rate & rank-1 calibration | Claude Mythos Preview & Opus 4.8 orchestrating open protein generators + 3-model ensemble: 354/1,320 (26.8%) across 14/15 targets (194 <100 nM, 90 <10 nM, 42 <1 nM); Rank-1 49% (20/41 campaigns), AP 0.48 vs 0.35 random (p<0.001) | No matched human-expert campaign (retrospective Adaptyv competitions only): RBX1 same-plate: Mythos rank-1 KD 3.9 nM vs competition winner 45 nM; de novo BBF-14 3/90; MBP 0/90 | **Supported (controlled)** | Rigorous dual-CRO wet-lab proof of autonomous tool orchestration (with exemplary open data release), explicitly scoped to agentic orchestration of specialized models rather than intrinsic LLM generation; lacks a matched human-expert control at equal GPU budget. |

## Confounder and Leakage Audit (8 Axes)

| Axis | Status | Evidence |
|---|---|---|
| Temporal cutoff | **Concern** (`evidenced_concern`) | Most targets are extensively characterized literature targets, and Adaptyv/ProteinBase competition reports for EGFR, TREM2, RBX1, and Nipah G were explicitly included in the agent's offline corpus. |
| Homology overlap | **Controlled** (`controlled`) | 98% (1,285/1,315) of tested binder designs have <30% sequence identity to any PDB chain (max 56%); however, targets themselves are mostly well-studied natural proteins (de novo target BBF-14 yielded 3/90 hits). |
| Cross-modal / literature recall | **Concern** (`evidenced_concern`) | Agents had bioRxiv access and competition corpora; 91-100% of designs targeted known natural-partner epitopes on 8/10 complexed targets (useful for engineering, but not de novo epitope discovery). |
| Preprocessing artifacts | **Controlled** (`controlled`) | All delivered designs synthesized unmodified; mature GDF-8 exclusion documented; blinded dual-reader SPR grading (96-98% agreement) and fixed integration rules published. |
| Downstream oracle filtering | **Concern** (`evidenced_concern`) | All backbones, sequences, and rankings come from specialized models (PXDesign, RFdiffusion3, BoltzGen, SolubleMPNN, ESMFold2, Protenix v2); this tests agentic orchestration, not intrinsic LLM protein modeling. |
| Compute & tool asymmetry | **Concern** (`evidenced_concern`) | Single-target runs spent $10,000/target in Modal GPU compute (2.8x per-target budget of multi-target runs) across 10+ specialized models, uncalibrated against retrospective competition entrants. |
| Baseline fairness | **Concern** (`evidenced_concern`) | Authors explicitly state they did not run a matched human-expert campaign using identical tools and GPU budgets; competition comparisons are retrospective. |
| Data & wet-lab transparency | **Controlled** (`controlled`) | Gold-standard transparency: two independent blinded CROs, off-target/ortholog controls, all 1,440 designs, raw SPR sensorgrams, fits, and full 16,000-word prompt released on Hugging Face. |

## External Critiques and Commentary

- **Anthony Gitter** (2026-08-19, x_thread): [https://x.com/anthonygitter/status/2089939463254188382](https://x.com/anthonygitter/status/2089939463254188382) - Praised as a model of scientific transparency for releasing full raw CRO data, designs, and methodology on Hugging Face.
