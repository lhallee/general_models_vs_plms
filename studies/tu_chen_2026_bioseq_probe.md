# Claude Was Trained on Protein Sequences at Scale (LLM-bioseq-probe)

- **Study ID:** `tu_chen_2026_bioseq_probe`
- **Authors:** Xinming Tu and Leo Tianlai Chen
- **Date:** 2026-11-17 (updated 2026-11-20)
- **Source tier:** Blog post / repository (`blog_repo`)
- **Verification depth:** Full post and repository verified (`full_post_and_code`)
- **Tasks:** Sequence recall and memorization, Structure prediction and side-chain packing
- **Regimes:** Intrinsic LLM (text only, no tools)
- **Models tested:** Claude Fable 5.2, Claude Opus 5.5, Claude Fable 5.1, Claude Opus 5 / 4.8 / 4.7 / 4.6 / 4.5, GPT-6.2 Sol / Astra, GPT-6.1 Sol, GPT-6 Astra, GPT-5.5, Gemini 4 Carbon, Gemini 2.5 Pro, DeepSeek V4.1 Flash, Qwen 3.8, Kimi K3.1 Pro, GLM-5.3, MiniMax M3, gpt-oss-120b, Nemotron 3 Ultra
- **Protein-specific baselines:** ProGen3-3B, ProtGPT3-10B, ProtGPT3-1.3B, ProtGPT2, ESMC-600M, ESM3-open 1.4B, ESM-2 650M (prefix-only and bidirectional), Prot-xLSTM 102M
- **Wet-lab validation:** No
- **Code and data availability:** MIT-licensed repository with test sets, evaluation scripts, gzipped raw JSONL model replies, and figure code.
- **Primary links:** [https://xinmingtu.cn/blog/2026/claude-opus-5-5-protein-sequences/](https://xinmingtu.cn/blog/2026/claude-opus-5-5-protein-sequences/), [https://github.com/XinmingTu/LLM-bioseq-probe](https://github.com/XinmingTu/LLM-bioseq-probe)

## Takeaway

Claude Fable 5.2 (+45.1 pp main, +22.1 pp on post-cutoff <50%-identity RefSeq proteins) and Opus 5.5 (+27.4 / +13.4 pp) match or exceed open autoregressive pLMs (ProGen3-3B +32.5 / +12.0) on unstudied DUF proteins while GPT-6, Gemini 4, and open-weight LLMs stay near zero. On long-range contact maps, however, Fable 5.2 (~25%) trails ESM-2 650M (~67%) and only 2 of 46 targets are strictly free of pre-cutoff sequence/PDB relatives.

## Methods

- **Datasets and benchmarks:** Main set: 668 unreviewed TrEMBL bacterial/archaeal proteins from 167 Pfam Domains of Unknown Function (no PDB structure, <=3 Europe PMC hits, UniRef90 singletons, 5 cuts/protein = 3,340 cuts). Family-size set: +90 smaller families (257 total). Post-cutoff set (Appendix G): 56 RefSeq proteins first public 2-27 Aug 2026 with <50% UniRef50 identity to earlier clusters (34-49% closest relative). Contact maps (Appendix H): 46 post-Aug-2026 PDBs (only 2 strictly clean) + 6 controls.
- **Evaluation protocol:** Prefix-only next-amino-acid completion (model outputs 10 AAs; 1st AA scored) paired with accession-seeded shuffled copies at identical cut sites. Temperature 0 where accepted, lowest thinking, no tools, 1 sample/cut.
- **Controls included:** Shuffled composition-matched controls, 128 zero-paper Pfam families, family-size scaling across 257 families, strict post-cutoff <50% identity RefSeq holdout (Appendix G), and copy-then-continue check.
- **Compute and tools:** Zero external tools; thinking tokens appeared in only 9% of Fable 5.2 replies (non-thinking scored higher: 54.7% vs 48.2%). Total API cost ~$730.
- **Metrics:** Family-mean real-minus-shuffled next-AA accuracy gap (percentage points) with 10,000-resample 95% bootstrap CI and Spearman r vs log(family size); top-N long-range contact precision (|i-j| >= 24, Cb <= 8 A).
- **Wet-lab protocol:** None (computational sequence-modeling and contact-map probe).

## Key Results

| Task | Regime | Benchmark | Metric | General LLM | Protein Baseline | Evidence Status | Notes |
|---|---|---|---|---|---|---|---|
| Sequence recall and memorization | Intrinsic LLM (text only, no tools) | 668 TrEMBL DUF proteins (167 families, 3,340 cuts) | Real minus shuffled next-AA accuracy (pp, 95% CI) | Claude Fable 5.2 / Opus 5.5 (vs GPT/Gemini/open LLMs): Fable 5.2 +45.1 (42.5 to 47.6; 54.1% vs 9.0%); Opus 5.5 +27.4; GPT/Gemini/open LLMs -0.0 to +2.9 | ProGen3-3B / ProtGPT3-10B / ESM-2 650M (bidir): ProGen3-3B +32.5; ProtGPT3-10B +27.7; ESMC-600M +13.0 (prefix); ESM-2 650M +7.0 (prefix) / +34.6 (bidir) | **Supported (controlled)** | Holds in 128 zero-paper families (+45.9 Fable 5.2) and scales with family size (Spearman 0.38-0.50), providing strong controlled evidence of large-scale protein sequence pretraining in Claude 5.2/5.5. |
| Sequence recall and memorization | Intrinsic LLM (text only, no tools) | 56 post-cutoff Aug 2026 RefSeq proteins (<50% UniRef50 ID) | Real minus shuffled next-AA accuracy (pp) | Claude Fable 5.2 / Claude Opus 5.5 / GPT-6.2 Sol: Fable 5.2 +22.1; Opus 5.5 +13.4; GPT-6.2 Sol -0.4 | ProGen3-3B / ProtGPT3-10B: ProGen3-3B +12.0; ProtGPT3-10B +6.7 | **Supported (controlled)** | Rules out exact sequence memorization as the sole explanation; generalizes across <50% identity family relatives. |
| Structure prediction and side-chain packing | Intrinsic LLM (text only, no tools) | 46 post-Aug-2026 PDB structures (Appendix H) | Top-N long-range contact precision (\|i-j\| >= 24) | Claude Fable 5.2: ~25% (vs 3% chance) | ESM-2 650M (at Fable's N): ~67% | **Suggestive (confounded)** | 44/46 post-Aug-2026 PDBs had pre-cutoff sequences or earlier PDB homologs, leaving only 2 strictly clean targets; Fable trails ESM-2 650M substantially. |

## Confounder and Leakage Audit (8 Axes)

| Axis | Status | Evidence |
|---|---|---|
| Temporal cutoff | **Controlled** (`controlled`) | Appendix G specifically tests 56 RefSeq proteins first public 2-27 Aug 2026 after assumed July 2026 cutoff (+22.1 pp for Fable 5.2); Appendix H contact maps note only 2/46 targets are strictly clean. |
| Homology overlap | **Controlled** (`controlled`) | Main set uses UniRef90 singletons; Appendix G enforces <50% identity to earlier UniRef50 clusters (closest earlier relative 34-49%). |
| Cross-modal / literature recall | **Controlled** (`controlled`) | Unreviewed TrEMBL DUF families with no PDB structures and <=3 Europe PMC hits; 128 zero-paper families score +45.9 pp (Spearman 0.05 vs paper count). |
| Preprocessing artifacts | **Controlled** (`controlled`) | Composition-matched shuffled controls, repeat consistency check (57.5% vs 57.5%), 0.1% non-answer rate, and copy-then-continue control; Fable 5 excluded due to 44% policy blocks. |
| Downstream oracle filtering | **N/A** (`not_applicable`) | Pure intrinsic next-residue and contact-list outputs. |
| Compute & tool asymmetry | **Controlled** (`controlled`) | Zero tools, temp 0, minimal thinking; non-thinking Fable 5.2 replies outperformed thinking replies (54.7% vs 48.2%). |
| Baseline fairness | **Controlled** (`controlled`) | Autoregressive pLMs (ProGen3-3B, ProtGPT3-10B) receive identical prefix-only context; masked ESM-2 650M evaluated both prefix-only (+7.0) and bidirectional (+34.6). |
| Data & wet-lab transparency | **Controlled** (`controlled`) | All data, scripts, raw JSONL outputs, and figure code released under MIT license. |

## External Critiques and Commentary

- **Hacker News discussion** (2026-11-18, forum): [https://news.ycombinator.com/item?id=50541619](https://news.ycombinator.com/item?id=50541619) - Debated whether shuffled sequences are a sufficient baseline control vs Markov/di-residue controls; authors noted composition and copy-then-continue checks plus family-size scaling.
