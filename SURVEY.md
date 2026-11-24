# What others have reported

Public results on general-purpose language models and protein sequences, each set beside the protein-specific
comparison its authors made, with a summary of what they show together in the [README](README.md#what-the-field-has-found).
Sources were opened on 23 November 2026, and the numbers are the authors' own and were not rerun here. The **Read**
column says how much of each source was read, so a row marked "abstract" rests on less than one marked "full text".
Dates are the day a source was first posted, with later versions noted, and corrections are welcome as issues.

## Claude Fable 5.2, Claude Mythos 5.2, and Claude Opus 5.5

| Source | Who and when | What was tested | What they report | Read |
|---|---|---|---|---|
| [Claude was trained on protein sequences at scale](https://xinmingtu.cn/blog/2026/claude-opus-5-5-protein-sequences/), with [code](https://github.com/XinmingTu/LLM-bioseq-probe) (MIT) | Xinming Tu and Leo Tianlai Chen. Announced on 17 November 2026 and last updated on 20 November 2026 | Next-residue prediction on 668 bacterial and archaeal proteins from 167 Pfam families of unknown function, against shuffled copies of the same sequences, for 20 language models and several protein language models | Real against shuffled accuracy: Claude Fable 5.2 54% against 9%, a gap of 45.1 points (95% interval 42.5 to 47.6). Claude Opus 5.5 34.9% against 7.5%, a gap of 27.4. ProGen3-3B 32.5, ProtGPT3-10B 27.7. GPT, Gemini, and seven open-weight models near zero. On 56 proteins first public after the cutoff: Fable 5.2 +22.1, Opus 5.5 +13.4. The post gives ESM-2 a gap of 34.6 with context on both sides of the cut, and the repository lists ESM C 600M at +13.0 and ESM-2 650M at +7.0 | Full post and README |
| [CheckLLMfold](https://github.com/jkosinski/CheckLLMfold) (MIT) | Jan Kosinski, announced on 18 November 2026. The README credits Claude Opus 5.5 for the code and text, Sergey Ovchinnikov for the contact-map prompt and folding code, and Xinming Tu and Leo Chen for the next-residue probe | Contact maps and 3D models from Claude Fable 5.2 for ubiquitin, three well-studied proteins, and three bacterial proteins with no relative in the PDB | Long-range contact precision of 0.18 to 0.49 on the well-studied proteins and 0.04 to 0.33 on the proteins with no PDB relative, two of those three near chance. The README says it is AI-written and not thoroughly revised | Full README |
| [Introducing Claude Fable 5.2 and Claude Mythos 5.2](https://www.anthropic.com/claude-fable-and-mythos-5-2) | Anthropic, 9 November 2026, with an edit note of 11 November | Whole-capsid structure modeling with open folding models, and a 24-hour de novo binder design benchmark | Claude Mythos 5.2 modeled 28 of 34 capsids within 3 Å. Binder design: 85.1 for Mythos 5.2, 79.1 for Mythos 5.1, and 82.6 in a cell labeled "without safeguards". The page makes no claim about sequence recall or variant effects | Full page |

## No examples: variant ranking and protein knowledge

| Source | Who and when | What was tested | What they report | Read |
|---|---|---|---|---|
| [PG-LLM: Benchmarking General-Purpose Language Models for Protein Variant Ranking](https://doi.org/10.64898/2026.07.27.741045) | R. K. Arora, L. T. Chen, M. Du, D. Marks, and G. Church. bioRxiv, 28 July 2026 | Thirteen language models rank variants on 276 tasks, 217 from ProteinGym and 59 held out by date, with no tools, alignments, or structures | Claude Opus 5 at Spearman 0.406 and GPT 5.6 Sol at 0.402, where ESM2-650M scores 0.411 and VenusREM 0.523. Opus 5 is above 49 of 95 published predictors. Scores rise with test-time compute and level off below the best specialist models | Abstract |
| [Evidence Scaling for Zero-Shot Protein Reasoning with Large Language Models](https://doi.org/10.64898/2026.09.27.754822) | Z. Hao, C. Wang, D. Li, and Y. Wang. bioRxiv, 1 October 2026 | Structural and evolutionary evidence from standard tools added to the prompt of a frozen model, on ProteinGym | Richer evidence improves ranking, evidence matched to the wrong variants falls below no evidence, and with evidence the ranking reaches strong specialist predictors. No number in the abstract | Abstract |
| [Rank-and-Reason](https://arxiv.org/abs/2602.00197) | Yang Tan, Yuanxi Yu, Can Wu, Bozitao Zhong, Mingchen Li, Guisheng Fan, Jiankang Zhu, Yafeng Liang, Nanqing Dong, and Liang Hong. arXiv, 30 January 2026, v2 on 3 February 2026 | Ranking of ProteinGym substitution variants by a system of several language model agents | A language model alone, DeepSeek-Reasoner, scores Spearman 0.159. ESM2-650M scores 0.414 and VenusREM 0.518. The full system with tools scores 0.551 | Full text |
| [Emergence of Biological Structural Discovery in General-Purpose Language Models](https://doi.org/10.64898/2026.01.03.697478) | Liang Wang. bioRxiv, 7 January 2026 | Protein homology detection by a GPT-2 trained only on English | ROC AUC 0.79 with no protein training, 0.52 for a randomly initialized model, and 0.76 for a GPT-2 the author trained on English-only text | Abstract |
| [LiveProteinBench](https://arxiv.org/abs/2512.22257) | Dingyi Rong, Zijian Chen, Qi Jia, Kaiwei Zhang, Haotian Lu, Guangtao Zhai, and Ning Liu. arXiv, 24 December 2025 | Twelve tasks of protein property and function questions, with and without structure images | GPT-5 at 60.65% mean accuracy, the best protein-specific chat model, BioMedGPT-R1, at 31.83%. No protein language model baseline | Full text |
| [Protein as a Second Language for LLMs](https://arxiv.org/abs/2510.11188) | Xinhui Chen, Zuchao Li, Mengqi Gao, Yufeng Zhang, Chak Tou Leong, Haoyang Li, and Jiaqi Chen. arXiv, 13 October 2025 | Protein description and question answering by frozen chat models given retrieved context | GPT-4o at a mean ROUGE-L of 17.03 alone and 19.89 with the method | Full text |
| [ProteinLMBench](https://arxiv.org/abs/2406.05540) | Yiqing Shen, Zan Chen, Michail Mamalakis, Luhan He, Haiyang Xia, Tianbin Li, Yanzhou Su, Junjun He, and Yu Guang Wang. arXiv, 8 June 2024, v2 on 8 July 2024 | Multiple-choice questions on protein understanding | GPT-4.0-turbo at 57.94% correct, and a 7B model fine-tuned on the paper's dataset at 62.18% | Full text |
| [VibeProteinBench](https://arxiv.org/abs/2605.10978) | Hyunjin Seo, Hongjoon Ahn, Jimin Park, Sungjun Han, Gyubok Lee, Soojung Yang, and 13 more authors. arXiv, 9 May 2026 | One language-interfaced model on recognition, engineering, and generation of proteins | No model is strong on all three stages. No number in the abstract | Abstract |

## Labeled examples in the prompt, and optimization from feedback

| Source | Who and when | What was tested | What they report | Read |
|---|---|---|---|---|
| [A Systematic Comparison of In-Context Learning Strategies for Peptide Property Prediction in Low-Data Regimes](https://arxiv.org/abs/2611.09994) | Jiantao Zhao, Andrei Voinea, and Mathijs F.J. Mabesoone. arXiv, 3 November 2026 | In-context learning by several language models and tabular foundation models on four peptide classification tasks | In-context learning is competitive with XGBoost and not significantly above it. No number in the abstract | Abstract |
| [Does reflective LLM feedback survive a wet-lab-scale label budget?](https://arxiv.org/abs/2611.01908) | Haoxuan Zeng and Xin Luo. arXiv, 1 November 2026. NeurIPS 2026 workshop on Agentic AI for Biological Discovery | Edits to a fitness predictor proposed by a language model under 96 labels a round, replayed on 36 ProteinGym assays | No detectable advantage over random edits, a difference of -0.008 (95% interval -0.034 to +0.018) | Abstract |
| [RosettaSearch](https://arxiv.org/abs/2604.17175) | Meghana Kshirsagar, Allen Nie, Ching-An Cheng, Fanglei Xue, Rahul Dodhia, Juan Lavista Ferres, Kevin K. Yang, and Frank DiMaio. arXiv, 19 April 2026, v2 on 30 April 2026 | o4-mini and Gemini-3 as optimizers that refine designed sequences against structure scores | Structural fidelity gains of 18% to 68% over one pass of LigandMPNN | Abstract |
| [Large Language Model is Secretly a Protein Sequence Optimizer](https://arxiv.org/abs/2501.09274) | Yinkai Wang, Jiaxing He, Yuanqi Du, Xiaohui Chen, Jianan Canal Li, Li-Ping Liu, Xiaolin Xu, and Soha Hassoun. arXiv, 16 January 2025 | Llama-3.1-8B proposes mutations in a directed evolution loop, shown two parents and their fitness | Best fitness above an evolutionary algorithm on a synthetic landscape (2.83 against 1.85) and on GFP (0.97 against 0.50), and below it on GB1 (4.80 against 5.72). No protein language model baseline | Full text |
| [Large Scale Multi-Task Bayesian Optimization with Large Language Models](https://arxiv.org/abs/2503.08131) | Yimeng Zeng, Natalie Maus, Haydn Thomas Jones, Jeffrey Tao, Fangping Wan, Marcelo Der Torossian Torres, Cesar de la Fuente-Nunez, Ryan Marcus, Osbert Bastani, and Jacob R. Gardner. arXiv, 11 March 2025, v2 on 12 June 2025 | A fine-tuned language model proposes starting points for Bayesian optimization of antimicrobial peptides | No number in the abstract | Abstract |
| [Benchmarking Large Language Models on Multiple Tasks in Bioinformatics NLP with Prompting](https://arxiv.org/abs/2503.04013) | Jiyue Jiang, Pengan Chen, Jiuming Wang, Dongchen He, Ziqin Wei, Liang Hong, and 10 more authors. arXiv, 6 March 2025 | Six chat models with no examples and with five, on tasks that include protein species prediction and structure recovery | Five examples lift species prediction to 82.00% for the best model. No property prediction task and no protein language model baseline | Full text |
| [Generalists vs. Specialists](https://arxiv.org/abs/2410.22296) | Angelica Chen, Samuel D. Stanton, Frances Ding, Robert G. Alberstein, Andrew M. Watkins, Richard Bonneau, Vladimir Gligorijevic, Kyunghyun Cho, and Nathan C. Frey. arXiv, 29 October 2024, v5 on 29 July 2025 | o1 and Gemini 2.0 Flash Thinking, prompted with a few examples, on constrained sequence optimization over synthetic functions | Minimum regret of 0.9375 for o1 and 0 for Gemini on one function. No protein language model baseline | Part of the full text |

Among these, no study gives a general-purpose model labeled protein variants in its prompt and compares it with a
ridge on protein language model features fit on the same examples. That is the comparison of this repository.

## Data and methods this repository uses

- **The comparison.** Diego del Alamo proposed testing in-context learning against ridge regression on protein
  language models on FLIP2 on 21 November 2026, in a [public thread](https://x.com/sacdallago/status/2124189348434444295)
  with Christian Dallago, where the chance that FLIP2 is in the models' training data was also raised.
- **FLIP2.** Kieran Didi, Sarah Alamdari, Alex X. Lu, Bruce Wittmann, Kadina E. Johnston, Ava P. Amini, Ali Madani,
  Maya Czeneszew, Christian Dallago, and Kevin K. Yang. bioRxiv, 24 February 2026,
  [10.64898/2026.02.23.707496](https://doi.org/10.64898/2026.02.23.707496). The data have been on Zenodo since 30
  January 2026 ([10.5281/zenodo.18433203](https://doi.org/10.5281/zenodo.18433203), CC BY 4.0), which is before the
  training cutoff of the models tested here.
- **FGFR2 screen.** MaveDB score set `urn:mavedb:00001329-a-3`, "FGFR2 Erdafitinib resistance", created by Annika
  Maria Metzner with Sven Diederichs as contributor, published 20 November 2026, CC BY-SA 4.0. The record names no
  publication.
- **Ridge on one-hot features with a zero-shot score.** Chloe Hsu, Hunter Nisonoff, Clara Fannjiang, and Jennifer
  Listgarten. Learning protein fitness models from evolutionary and assay-labeled data. Nature Biotechnology, 2022.
- **The comparison with a tuned protein language model.** Diego del Alamo asked on 23 November 2026 for a comparison
  with Direct Preference Optimization (DPO) on ProGen2.
- **ProGen2.** Erik Nijkamp, Jeffrey A. Ruffolo, Eli N. Weinstein, Nikhil Naik, and Ali Madani. ProGen2: Exploring the
  boundaries of protein language models. Cell Systems, 2023, [arXiv 2206.13517](https://arxiv.org/abs/2206.13517). The
  weights are read from the mirror `hugohrban/progen2-small` on the Hugging Face Hub (BSD-3-Clause), pinned by
  revision and checked against the reference score of the authors' own scorer.
- **Preference tuning.** The ranked preference loss with a frozen reference model is that of Widatalla and colleagues,
  Aligning protein generative models with experimental fitness via Direct Preference Optimization, bioRxiv
  [10.1101/2024.05.20.595026](https://doi.org/10.1101/2024.05.20.595026), as Yang and colleagues applied it to
  ProGen2-small in Steering Generative Models with Experimental Data for Protein Fitness Optimization,
  [arXiv 2505.15093](https://arxiv.org/abs/2505.15093). The Bradley-Terry loss on likelihood differences is that of
  Hawkins-Hooker and colleagues, Likelihood-based fine-tuning of protein language models for few-shot fitness
  prediction and design, bioRxiv [10.1101/2024.05.28.596156](https://doi.org/10.1101/2024.05.28.596156).

## Not read

Anthropic's system cards for these models, and the full text of the rows marked "abstract".
