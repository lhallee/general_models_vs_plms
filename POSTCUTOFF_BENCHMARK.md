# Post-Cutoff MaveDB Benchmark Expansion & Frontier Model Horizon

Expands our leak-free evaluation beyond a single protein (`FGFR2`) to three unrelated human protein families verified from MaveDB (3,005 score sets scanned on 2026-11-25) with strict preprint-date auditing and zero paid API spend.

## 1. Verified Post-June-2026 MaveDB Assays

| Slug | URN | Gene / Family | MaveDB Date | Linked Preprint | License | Transform | Description |
|---|---|---|---|---|---|---|---|
| `fgfr2_erdafitinib` | `urn:mavedb:00001329-a-3` | FGFR2 (Receptor tyrosine kinase domain (320 aa segment)) | 2026-11-20 | None (first public on MaveDB) | CC BY-SA 4.0 | `log2` | Human FGFR2 kinase domain (residues 461-780) erdafitinib resistance saturation screen; 9 companion FGFR inhibitor screens also published 2026-11-20. |
| `gba1_gcase_activity` | `urn:mavedb:00001327-a-1` | GBA1 (Lysosomal beta-glucocerebrosidase (GH30 hydrolase, 536 aa)) | 2026-11-11 | None (first public on MaveDB) | CC0 | `identity` | Full-length human GBA1 lysosomal GCase enzymatic activity measured by LysoFix-GBA flow-cytometry DMS (~9,886 missense variants, already log/signed scale). |
| `araf_activity_baseline` | `urn:mavedb:00001314-0-4` | ARAF (Raf serine/threonine kinase (606 aa)) | 2026-10-16 | 2026-10-04 (`10.64898/2026.10.04.756099`) | CC0 | `log2` | Human ARAF kinase activity baseline DMS (~10,960 ARAF missense variants after excluding 7 BRAF control rows, indels, and nonsense; 558 synonymous noise controls). |
| `araf_abundance_baseline` | `urn:mavedb:00001314-0-3` | ARAF (Raf serine/threonine kinase (606 aa)) | 2026-10-16 | 2026-10-04 (`10.64898/2026.10.04.756099`) | CC0 | `log2` | Human ARAF cellular abundance baseline DMS (~5,041 ARAF missense variants; 262 synonymous noise controls), enabling activity-vs-abundance specificity tests on the same protein. |

## 2. Rejected Post-June-2026 Uploads (Pre-Cutoff Preprint Leakage Caught)

| URN | Gene | MaveDB Upload | Linked Preprint | Why Rejected |
|---|---|---|---|---|
| `urn:mavedb:00001281-a-1` | PRKN | 2026-06-28 | 2026-02-09 (`10.64898/2026.02.09.704817`) | MaveDB upload date (2026-06-28) sits on the June 2026 cutoff boundary and its linked bioRxiv preprint (2026-02-09) predates Claude 5.2 and GPT-6 cutoffs by four months. |
| `urn:mavedb:00001281-a-2` | PRKN | 2026-06-28 | 2026-02-09 (`10.64898/2026.02.09.704817`) | Same linked bioRxiv preprint (2026-02-09) predating June 2026 training cutoffs. |

## 3. Sequence Homology Audit Between Post-Cutoff Parents

| Pair | 5-mer Jaccard | Best Ungapped Window Identity |
|---|---|---|
| FLIP2_amylase vs FLIP2_ired | 0.0000 | 0.1034 |
| FLIP2_amylase vs FLIP2_nucb | 0.0000 | 0.1197 |
| FLIP2_amylase vs FLIP2_pdz3 | 0.0000 | 0.1296 |
| FLIP2_amylase vs FLIP2_trpb | 0.0000 | 0.0823 |
| FLIP2_amylase vs POST_ARAF | 0.0000 | 0.0941 |
| FLIP2_amylase vs POST_FGFR2 | 0.0000 | 0.0875 |
| FLIP2_amylase vs POST_GBA1 | 0.0000 | 0.1012 |
| FLIP2_ired vs FLIP2_nucb | 0.0000 | 0.1268 |
| FLIP2_ired vs FLIP2_pdz3 | 0.0000 | 0.1667 |
| FLIP2_ired vs FLIP2_trpb | 0.0015 | 0.1103 |
| FLIP2_ired vs POST_ARAF | 0.0000 | 0.1069 |
| FLIP2_ired vs POST_FGFR2 | 0.0000 | 0.1000 |
| FLIP2_ired vs POST_GBA1 | 0.0000 | 0.1000 |
| FLIP2_nucb vs FLIP2_pdz3 | 0.0000 | 0.1019 |
| FLIP2_nucb vs FLIP2_trpb | 0.0000 | 0.1127 |
| FLIP2_nucb vs POST_ARAF | 0.0000 | 0.1197 |
| FLIP2_nucb vs POST_FGFR2 | 0.0000 | 0.1338 |
| FLIP2_nucb vs POST_GBA1 | 0.0000 | 0.1127 |
| FLIP2_pdz3 vs FLIP2_trpb | 0.0000 | 0.1481 |
| FLIP2_pdz3 vs POST_ARAF | 0.0000 | 0.1296 |
| FLIP2_pdz3 vs POST_FGFR2 | 0.0000 | 0.1296 |
| FLIP2_pdz3 vs POST_GBA1 | 0.0000 | 0.1667 |
| FLIP2_trpb vs POST_ARAF | 0.0000 | 0.0823 |
| FLIP2_trpb vs POST_FGFR2 | 0.0000 | 0.0938 |
| FLIP2_trpb vs POST_GBA1 | 0.0000 | 0.1003 |
| POST_ARAF vs POST_FGFR2 | 0.0033 | 0.1250 |
| POST_ARAF vs POST_GBA1 | 0.0009 | 0.0784 |
| POST_FGFR2 vs POST_GBA1 | 0.0000 | 0.0875 |

## 4. Frontier Model & Baseline Horizon (Zero Paid API Spend Approved)

| Model / Baseline | Category | Route | API Spend Approved | Status | Plan |
|---|---|---|---|---|---|
| `esmc_600m_and_6b_ridge_and_lora` | `specialized_plm` | Docker GPU on named workstation (zero API spend) | No | `ready_when_gpu_named` | Embed and fit ridge + LoRA/fine-tuned ESMC on GBA1, ARAF activity/abundance, and FGFR2 (random and by_position splits). |
| `open_weight_reasoning_llms` | `general_llm_open_weight` | Local/remote Docker vLLM on named GPU machine (zero API spend) | No | `horizon_zero_spend` | Evaluate open-weight reasoning models (e.g., Qwen / DeepSeek) under identical k256, cipher, flipped, and recognition controls. |
| `gpt_6_2_and_gpt_6_astra` | `general_llm_frontier_api` | External API (not run; zero paid API spend approved) | No | `horizon_awaiting_approval` | High priority for controlled comparison against OpenAI's unaudited ProteinGym claims once non-paid access or budget approval is granted. |
| `gemini_4_frontier` | `general_llm_frontier_api` | External API (not run; zero paid API spend approved) | No | `horizon_awaiting_approval` | Evaluate on identical FLIP2 + expanded post-cutoff suite under cipher/flipped/recognition controls when approved. |
| `claude_mythos_5_2_expanded_postcutoff` | `general_llm_harness` | Audited single-prompt subagents on GBA1 and ARAF (zero API spend) | No | `ready_after_splits_built` | Extend our controlled Mythos 5.2 post-cutoff evaluation from 1 protein (FGFR2) to 3 unrelated human protein families (FGFR2, GBA1, ARAF). |
