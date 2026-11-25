# Protein as a Second Language for LLMs

- **Study ID:** `chen_2025_protein_second_language`
- **Authors:** Xinhui Chen, Zuchao Li, Mengqi Gao, Yufeng Zhang, Chak Tou Leong, Haoyang Li, Jiaqi Chen
- **Date:** 2025-10-13
- **Source tier:** Preprint (`preprint`)
- **Verification depth:** Full text verified (`full_text`)
- **Tasks:** Protein QA and multimodal reasoning
- **Regimes:** LLM agent orchestrating specialized protein tools, Intrinsic LLM (text only, no tools), Protein-finetuned LLM
- **Models tested:** GPT-4o, kimi-k2, Qwen3-14B, Mistral-7B-Instruct-v0.3, Qwen2.5-3B
- **Protein-specific baselines:** BioT5+ and ProLLaMA-7B evaluated WITHOUT retrieval; no MMseqs2 nearest-neighbor transfer or pLM baseline
- **Wet-lab validation:** No
- **Code and data availability:** No public link ('will be provided as supplementary material upon acceptance'); exact test-set sizes N omitted.
- **Primary links:** [https://arxiv.org/abs/2510.11188](https://arxiv.org/abs/2510.11188)

## Takeaway

Retrieving MMseqs2 sequence homologs and text-similar Swiss-Prot QA exemplars boosts GPT-4o ROUGE-L on ProtDescribe from 18.29 to 35.53 (with text-only retrieval already giving 32.59). Without homology holdouts between the Swiss-Prot retrieval base and Swiss-Prot benchmarks or a direct MMseqs2 transfer baseline, the gains reflect retrieval leakage rather than intrinsic protein sequence comprehension.

## Methods

- **Datasets and benchmarks:** Retrieval corpus of 79,926 DeepSeek-R1-generated Swiss-Prot QA pairs evaluated on ProtDescribe, Protein2Text-QA, and 4 Mol-Instructions protein tasks.
- **Evaluation protocol:** Retrieves k=4 to 11 exemplars matching MMseqs2 sequence homology + text/QA similarity and inserts them into frozen LLM prompts; k tuned on evaluation benchmarks.
- **Controls included:** Zero-shot vs SeqOnly vs QAOnly vs Dual retrieval ablation (Table 2); 48-case human evaluation (5 raters).
- **Compute and tools:** MMseqs2 + textual RAG over 79,926 Swiss-Prot entries for LLMs vs zero-retrieval BioT5+/ProLLaMA-7B.
- **Metrics:** ROUGE-L and 0-5 human rating on 48 cases.
- **Wet-lab protocol:** None.

## Key Results

| Task | Regime | Benchmark | Metric | General LLM | Protein Baseline | Evidence Status | Notes |
|---|---|---|---|---|---|---|---|
| Protein QA and multimodal reasoning | LLM agent orchestrating specialized protein tools | ProtDescribe / Protein2Text-QA / Mol-Instructions (Swiss-Prot/UniProt derived) | ROUGE-L | GPT-4o (zero-shot -> MMseqs2 + text RAG): ProtDescribe 18.29 -> 35.53 (QAOnly alone 32.59); Protein2Text 20.84 -> 26.86; Mol-Instructions 17.03 -> 19.89 | BioT5+ / ProLLaMA-7B (no retrieval; no MMseqs2 transfer baseline): ProtDescribe 9.97 / 12.77; Protein2Text 6.96 / 10.09; Mol-Instructions 3.60 / 16.85 | **Unverified / likely leaked** | Retrieves Swiss-Prot homologs and text-similar QA entries into Swiss-Prot/UniProt benchmarks without excluding test proteins or close homologs, tunes k on test sets, and omits MMseqs2 transfer and retrieval-matched baselines. |

## Confounder and Leakage Audit (8 Axes)

| Axis | Status | Evidence |
|---|---|---|
| Temporal cutoff | **Uncontrolled** (`uncontrolled`) | No temporal split on Swiss-Prot corpus or UniProt/PubMed evaluation benchmarks. |
| Homology overlap | **Concern** (`evidenced_concern`) | Explicitly uses MMseqs2 homology search over a Swiss-Prot corpus to answer Swiss-Prot/UniProt benchmarks without holding out exact matches or close homologs (Fig 6 cites exact UniProt homolog RlmN_AZOC5). |
| Cross-modal / literature recall | **Concern** (`evidenced_concern`) | Both retrieval DB and test sets derive from Swiss-Prot/PubMed; QAOnly text retrieval alone produces most of the ProtDescribe gain (18.29 -> 32.59 vs 35.51 Dual, Table 2). |
| Preprocessing artifacts | **Concern** (`evidenced_concern`) | ROUGE-L rewards copying fixed 4-field Swiss-Prot template wording from k=11 retrieved exemplars; k tuned per evaluation set; test N omitted. |
| Downstream oracle filtering | **N/A** (`not_applicable`) | Descriptive text generation. |
| Compute & tool asymmetry | **Concern** (`evidenced_concern`) | Frozen LLMs receive MMseqs2 + text RAG over 79,926 curated Swiss-Prot entries while BioT5+ and ProLLaMA-7B receive zero retrieval. |
| Baseline fairness | **Concern** (`evidenced_concern`) | No MMseqs2 nearest-neighbor annotation transfer baseline and no retrieval-augmented specialized baseline. |
| Data & wet-lab transparency | **Concern** (`evidenced_concern`) | No public code/data link, missing test-set sizes N, and unvalidated qualitative human cases. |
