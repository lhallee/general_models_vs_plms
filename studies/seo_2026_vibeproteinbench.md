# VibeProteinBench: An Evaluation Benchmark for Language-Interfaced Vibe Protein Design

- **Study ID:** `seo_2026_vibeproteinbench`
- **Authors:** Hyunjin Seo, Hongjoon Ahn, Jimin Park, Sungjun Han, Gyubok Lee, Soojung Yang, Cheng-Hao Liu, Joseph S. Brown, Leo Chen, Gina El Nesr, Feyisayo Eweje, Sarah Gurev, Hyejin Lee, Junlang Liu, Zhihui Qi, Gyu Rie Lee, Sungsoo Ahn, Jamin Shin, Sangwon Jung
- **Date:** 2026-05-09 (updated 2026-09-26)
- **Source tier:** Preprint (`preprint`)
- **Verification depth:** Full text verified (`full_text`)
- **Tasks:** Protein QA and multimodal reasoning, Zero-shot variant effect prediction, De novo backbone and binder design
- **Regimes:** Intrinsic LLM (text only, no tools), LLM agent orchestrating specialized protein tools, Protein-finetuned LLM
- **Models tested:** GPT-5.4, Gemini3.1-Pro, Claude Opus-4.6, DeepSeek-V4-Pro, DeepSeek-V3.2, Kimi-K2.5, Qwen3.5-397B-A17B, Qwen3.5-9B, NatureLM-8x7B, SciReasoner-8B, TxGemma-9B/27B
- **Protein-specific baselines:** BoltzGen, BindCraft, ProteinDT-T5, PAAG, ProDVa
- **Wet-lab validation:** No
- **Code and data availability:** No public GitHub or Hugging Face URL provided in the preprint text.
- **Primary links:** [https://arxiv.org/abs/2605.10978](https://arxiv.org/abs/2605.10978)

## Takeaway

On 957 temporally and literature-filtered queries, frontier LLMs score well on recognition (82.6% tool-free, 84.8% with tools) but fail most engineering (56.7%/65.0%) and de novo generation tasks (14.8%/23.8%), trailing dedicated structure-conditioned binder tools like BoltzGen (36.7%). Even with specialized tools, agents frequently drop constraints or skip verification, though all design metrics remain computational rather than experimental.

## Methods

- **Datasets and benchmarks:** 957 core queries (627 Recognition, 120 Engineering [60 rationale-guided + 60 constraint-free], 210 Generation [90 GO-conditioned + 120 binders]) plus 90 auxiliary antibody/RNA/DNA binder queries.
- **Evaluation protocol:** Zero-shot temp=0 (tool-free) vs 30-iteration ReAct agent (with RFdiffusion/3, ProteinMPNN, LigandMPNN, ESM2, BoltzGen, PXDesign, Boltz-2, ESMFold2, AutoDock Vina, PyRosetta, UniProt search capped <=2025-09-01) and unbounded Codex CLI on 30 queries.
- **Controls included:** Strict 2025-09-01 temporal holdout on Recognition and Binders (verified by pre-2021 vs post-2025 binder drop: DeepSeek-V4-Pro 17.5% -> 6.7%), PubMed/Semantic Scholar literature exclusion on Engineering, <30% identity on fold recognition, and evaluator calibration on experimental benchmarks (61.5-80.0% BA).
- **Compute and tools:** Identical tool-free and 30-iteration ReAct setups across top 5 LLMs; Codex CLI token scaling audited (101x-163.6x tokens on 30 queries); Opus-4.6 and Qwen3.5-9B had thinking disabled due to empty replies.
- **Metrics:** Exact-match accuracy (Recognition); sequential constraint + rationale + in silico oracle pass rate (Protenix-v1, Boltz-2, MGnify SaProt stability, pyKVFinder, ESMC GO cosine).
- **Wet-lab protocol:** None (purely computational oracle scoring).

## Key Results

| Task | Regime | Benchmark | Metric | General LLM | Protein Baseline | Evidence Status | Notes |
|---|---|---|---|---|---|---|---|
| De novo backbone and binder design | Intrinsic LLM (text only, no tools) | VibeProteinBench 957 post-2025-09 queries (Recognition / Engineering / Generation) | Pass rate (%) | GPT-5.4 / DeepSeek-V4-Pro / Gemini3.1-Pro (tool-free -> 30-iter ReAct tools): Tool-free best: Rec 82.6% (GPT-5.4), Eng 56.7% (GPT-5.4), Gen 14.8% (Gemini3.1-Pro, binders 7.5%); With tools best: Rec 84.8% (DeepSeek-V4-Pro), Eng 65.0% (Qwen3.5-397B), Gen 23.8% (DeepSeek-V4-Pro) | BoltzGen / BindCraft (single-design binder baseline): BoltzGen 36.7%, BindCraft 20.0% on binders (exceeding all tool-free and tool-augmented LLMs) | **Supported (controlled)** | Strict post-2025-09-01 cutoff and literature-filtered engineering controls establish that both tool-free and ReAct-agent LLMs fail most de novo generation queries and trail dedicated BoltzGen (36.7%); caveat: computational oracle scoring only (61.5-80% calibrated BA). |

## Confounder and Leakage Audit (8 Axes)

| Axis | Status | Evidence |
|---|---|---|
| Temporal cutoff | **Controlled** (`controlled`) | Strict >=2025-09-01 temporal holdout for Recognition and Binders (Table 10 confirms pre-2021 -> post-2025 drop 17.5% -> 6.7%), and search_sequence capped <=2025-09-01. |
| Homology overlap | **Concern** (`evidenced_concern`) | Fold recognition enforces <30% sequence identity (App C.3), but Recognition, Engineering, and core binder targets do not enforce <30% identity against pre-cutoff UniProt/PDB. |
| Cross-modal / literature recall | **Controlled** (`controlled`) | Engineering excludes targets with prior engineering papers via PubMed + Semantic Scholar queries (Sec 4.2, App D.2), and Recognition/Binders postdate 2025-09-01. |
| Preprocessing artifacts | **Concern** (`evidenced_concern`) | Safety refusals count as failures (Opus-4.6 refuses 35.0% of Engineering, Table 18), and thinking was disabled for Opus-4.6 and Qwen3.5-9B due to empty responses. |
| Downstream oracle filtering | **Concern** (`evidenced_concern`) | Engineering and Generation pass rates rely exclusively on computational oracles (Protenix-v1, Boltz-2, MGnify SaProt, pyKVFinder, ESMC) with 61.5-80.0% calibrated BA rather than wet-lab assays. |
| Compute & tool asymmetry | **Controlled** (`controlled`) | Directly compares tool-free vs identical 30-step ReAct setups and documents Codex CLI token blowup (101x-163.6x, Table 6) and tool-shortcutting (24.1% lower diffusion steps, Table 8). |
| Baseline fairness | **Controlled** (`controlled`) | Includes general LLMs, domain-specialized LLMs (NatureLM, SciReasoner, TxGemma), text-protein models (ProteinDT, PAAG, ProDVa), and dedicated binder tools (BindCraft, BoltzGen). |
| Data & wet-lab transparency | **Concern** (`evidenced_concern`) | Zero wet-lab validation and no public repository URL in the manuscript. |
