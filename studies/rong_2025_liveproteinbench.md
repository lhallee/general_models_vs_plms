# LiveProteinBench: A Contamination-Free Benchmark for Assessing Models' Specialized Capabilities in Protein Science

- **Study ID:** `rong_2025_liveproteinbench`
- **Authors:** Dingyi Rong, Zijian Chen, Qi Jia, Kaiwei Zhang, Haotian Lu, Guangtao Zhai, Ning Liu
- **Date:** 2025-12-24
- **Source tier:** Preprint (`preprint`)
- **Verification depth:** Full text verified (`full_text`)
- **Tasks:** Protein QA and multimodal reasoning
- **Regimes:** Intrinsic LLM (text only, no tools), Protein-finetuned LLM
- **Models tested:** GPT-5, GPT-o3, Gemini-2.5-pro, Claude-3.7-sonnet, Gemini-2.0-flash, GPT-4o, Qwen3-32B, Deepseek-v3, Deepseek-r1, Llama3.3-70B, Qwen2.5-72B/32B, Qwen2.5-VL-32B, InternVL3-78B
- **Protein-specific baselines:** Domain-specialized SLLMs only: BioMedGPT-R1 (17B), EvoLlama (8B), Evolla (10B). Zero pLM or BLAST/InterProScan baselines.
- **Wet-lab validation:** No
- **Code and data availability:** GitHub repository released (Rongdingyi/LiveProteinBench); labels programmatically derived from UniProt/GOA.
- **Primary links:** [https://arxiv.org/abs/2512.22257](https://arxiv.org/abs/2512.22257), [https://github.com/Rongdingyi/LiveProteinBench](https://github.com/Rongdingyi/LiveProteinBench)

## Takeaway

On 1,797 post-Jan-2025 UniProt MCQs, GPT-5 (60.65%) outperforms small protein-adapted LLMs (23-32%) and drops steeply relative to 2020 entries (38.47% vs 66.76% on five hard tasks), while naive 2D structure images frequently degrade accuracy. Lack of homology clustering, partial 2025 cutoff overlap, strict format penalties on SLLMs, and missing pLM/BLAST baselines limit how strongly the gap reflects sequence understanding.

## Methods

- **Datasets and benchmarks:** 1,797 four-choice MCQs across 12 protein tasks from UniProt entries first publicly released after 2025-01-01, plus 195 CPI questions and a 2020 subset on 5 tasks for temporal comparison.
- **Evaluation protocol:** Standardized prompt requiring strict '1. reasoning ... 2. answer: [A/B/C/D]' format, tested sequence-only, with 6 orthogonal PyMOL 2D structure images, and with 3Di strings.
- **Controls included:** 2020 vs 2025 temporal drop ablation (Table 4), sequence-only vs 2D-image ablation (Table 5), thinking on/off ablation (Table 6), and 1-shot formatting rescue for SLLMs (Tables 9-10).
- **Compute and tools:** No external tools; frontier reasoning models use hidden CoT compute vs 8B-17B non-thinking SLLMs.
- **Metrics:** 4-choice MCQ Accuracy (%) and strict format Pass Rate (%).
- **Wet-lab protocol:** None (UniProt annotation QA).

## Key Results

| Task | Regime | Benchmark | Metric | General LLM | Protein Baseline | Evidence Status | Notes |
|---|---|---|---|---|---|---|---|
| Protein QA and multimodal reasoning | Intrinsic LLM (text only, no tools) | LiveProteinBench (1,797 post-Jan-2025 UniProt 4-choice MCQs, 12 tasks) | Mean accuracy (%) | GPT-5 / o3 / Claude-3.7-sonnet / Gemini-2.5-pro: GPT-5 60.65% (60.49% seq-only), o3 56.48%, Claude-3.7 51.47% seq-only (47.58% w/ images), Gemini-2.5-pro 47.97% | BioMedGPT-R1 (17B) / EvoLlama (8B) / Evolla (10B) [no pLM/BLAST baseline]: 31.83% / 24.26% (30.00% 1-shot) / 22.98% (0% strict format pass rate for BioMedGPT-R1 & Evolla) | **Suggestive (confounded)** | 2020->2025 temporal drop is real (GPT-5 66.76% -> 38.47% on 5 hard tasks), and 2D PyMOL images often hurt MLLMs; however, Jan 2025 overlaps mid-2025 model cutoffs, no homology clustering excludes close pre-2025 homologs, and strict regex parsing penalizes SLLMs. |

## Confounder and Leakage Audit (8 Axes)

| Axis | Status | Evidence |
|---|---|---|
| Temporal cutoff | **Concern** (`evidenced_concern`) | Uses a 2025-01-01 cutoff (and Table 4 confirms huge 2020->2025 drops), but Fig 1A itself shows GPT-5, Gemini-2.5-pro, and Qwen-3 training cutoffs extend well into 2025. |
| Homology overlap | **Uncontrolled** (`uncontrolled`) | No sequence-identity clustering against pre-2025 UniProt; Appendix E case studies confirm models solve items by matching conserved motifs and lengths to known families. |
| Cross-modal / literature recall | **Concern** (`evidenced_concern`) | Homologous 2025 entries allow conserved-motif family recall, and 2020 items exhibit heavy memorization inflation (Table 4). |
| Preprocessing artifacts | **Concern** (`evidenced_concern`) | Strict format parsing causes 0.00% Pass Rate for BioMedGPT-R1 and Evolla (Table 12), confounding instruction compliance with protein understanding; 6-view PyMOL images hurt several MLLMs. |
| Downstream oracle filtering | **N/A** (`not_applicable`) | 4-choice MCQ benchmark. |
| Compute & tool asymmetry | **Concern** (`evidenced_concern`) | Compares frontier reasoning models (GPT-5, o3) with hidden CoT against 8B-17B non-thinking SLLMs. |
| Baseline fairness | **Concern** (`evidenced_concern`) | Omits any pLM (ESM2/ESMC) or standard bioinformatics baseline (BLAST, InterProScan) and penalizes SLLMs via zero-shot formatting. |
| Data & wet-lab transparency | **Controlled** (`controlled`) | Experimentally validated UniProt/PDB labels, per-task counts reported, and public GitHub repository. |
