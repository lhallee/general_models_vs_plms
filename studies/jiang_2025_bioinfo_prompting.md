# Benchmarking Large Language Models on Multiple Tasks in Bioinformatics NLP with Prompting

- **Study ID:** `jiang_2025_bioinfo_prompting`
- **Authors:** Jiyue Jiang, Pengan Chen, Jiuming Wang, Dongchen He, Ziqin Wei, Liang Hong, Licheng Zong, Sheng Wang, Qinze Yu, Zixian Ma, Yanyu Chen, Yimin Fan, Xiangyu Shi, Jiawei Sun, Chuan Wu, Yu Li
- **Date:** 2025-03-06
- **Source tier:** Preprint (`preprint`)
- **Verification depth:** Full text verified (`full_text`)
- **Tasks:** Inverse folding and sequence design, Structure prediction and side-chain packing, Protein QA and multimodal reasoning
- **Regimes:** Intrinsic LLM (text only, no tools)
- **Models tested:** GPT-4o, Llama-3.1-70B, Mistral-large-2 (123B), Qwen-2.5-72B, Yi-1.5-34B, InternLM-2.5-20B
- **Protein-specific baselines:** None (zero pLM or bioinformatics baselines evaluated)
- **Wet-lab validation:** No
- **Code and data availability:** No public repository or dataset URL provided in the preprint; Pfam/Rfam numbers omitted from Table 2.
- **Primary links:** [https://arxiv.org/abs/2503.04013](https://arxiv.org/abs/2503.04013)

## Takeaway

Six vanilla 2024 LLMs under 0-shot and 5-shot prompting fail at protein inverse folding (4.6-7.0%, near 5% random chance) and reach only 24-34% secondary-structure recovery, while newline tokenization shifts sequence accuracy more than threefold. Missing pLM/bioinformatics baselines, homology controls, and public datasets limit interpretation of the 5-shot species and 10-shot Pfam gains.

## Methods

- **Datasets and benchmarks:** 30 bioinformatics tasks including PDB secondary structure sst8 (N=264), PDB inverse folding from secondary structure (N=264), protein species prediction (N=200), RBP binding (N=70), and Pfam family generation.
- **Evaluation protocol:** 0-shot and 5-shot CoT prompting (10-shot for Pfam) at temp=0.2 with BioFinder (QLoRA xFinder-llama38it, 93.5% extraction accuracy) for answer parsing; tests continuous vs space vs newline sequence tokenization.
- **Controls included:** 0-shot vs 5-shot comparison and sequence tokenization ablation (newline separation gives >3x alignment accuracy over continuous strings).
- **Compute and tools:** Identical text-only prompting across all 6 vanilla LLMs with no external tools.
- **Metrics:** Accuracy (species, RBP), sequence/sst8 recovery (%), bit score (Pfam).
- **Wet-lab protocol:** None.

## Key Results

| Task | Regime | Benchmark | Metric | General LLM | Protein Baseline | Evidence Status | Notes |
|---|---|---|---|---|---|---|---|
| Inverse folding and sequence design | Intrinsic LLM (text only, no tools) | PDB inverse folding from secondary structure (N=264) & sst8 prediction (N=264) | Sequence / sst8 recovery (%) at 0-shot -> 5-shot | 6 vanilla 2024 LLMs (GPT-4o, Llama-3.1-70B, Qwen-2.5-72B, Mistral-large-2, Yi-1.5-34B, InternLM-2.5-20B): Inverse folding 4.64%-6.95% (near 5% random 1/20 AA chance); sst8 recovery 23.99%-34.31% (best Llama-3.1-70B) | None (no ProteinMPNN / NetSurfP / pLM baseline): Random 1/20 AA = 5.0% | **Supported (controlled)** | Clean negative result showing vanilla 2024 LLMs fail text-prompted inverse folding (near 5% chance) and score weakly on secondary structure; 5-shot species jump (4-12% -> 75-82%) largely reflects revealing label format. |

## Confounder and Leakage Audit (8 Axes)

| Axis | Status | Evidence |
|---|---|---|
| Temporal cutoff | **Uncontrolled** (`uncontrolled`) | No temporal cutoff applied to PDB or species datasets. |
| Homology overlap | **Uncontrolled** (`uncontrolled`) | Protein PDB tasks deduplicate only exact matches with no homology clustering between 5-shot/10-shot exemplars and test items. |
| Cross-modal / literature recall | **Concern** (`evidenced_concern`) | RBP prompts give explicit protein name and species; 10-shot Pfam generation supplies 10 homologous sequences from the target family (App D), turning generation into in-context consensus copying. |
| Preprocessing artifacts | **Concern** (`evidenced_concern`) | BPE tokenization changes sequence accuracy >3x (Fig 4); 5-shot species jumps 6-20x from label-set formatting; BioFinder still mis-extracts 6.5% of sequences; Pfam omitted from Table 2 with text typos in Sec 5.4. |
| Downstream oracle filtering | **N/A** (`not_applicable`) | Direct prompted outputs without oracle filtering. |
| Compute & tool asymmetry | **Controlled** (`controlled`) | Identical 0-shot/5-shot CoT prompting across all 6 vanilla LLMs with no tools. |
| Baseline fairness | **Concern** (`evidenced_concern`) | Zero pLMs or bioinformatics baselines (BLAST, ProteinMPNN, NetSurfP) evaluated. |
| Data & wet-lab transparency | **Concern** (`evidenced_concern`) | No public code or dataset URL, no wet-lab data, and Pfam/Rfam omitted from Table 2. |
