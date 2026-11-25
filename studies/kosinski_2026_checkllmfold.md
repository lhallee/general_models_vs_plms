# CheckLLMfold: Contact Map, 3D Folding, Recognition, and Next-AA Audits of Claude Fable 5.2

- **Study ID:** `kosinski_2026_checkllmfold`
- **Authors:** Jan Kosinski (code/analysis drafted with Claude Opus 5.5; contact protocol adapted from Sergey Ovchinnikov)
- **Date:** 2026-11-18 (updated 2026-11-18)
- **Source tier:** Blog post / repository (`blog_repo`)
- **Verification depth:** Full post and repository verified (`full_post_and_code`)
- **Tasks:** Structure prediction and side-chain packing, Sequence recall and memorization, Protein QA and multimodal reasoning
- **Regimes:** Intrinsic LLM + code execution (no specialized protein models), Intrinsic LLM (text only, no tools)
- **Models tested:** Claude Fable 5.2 (headless claude -p, high effort for folding, low effort for oneshot/nextaa)
- **Protein-specific baselines:** ProGen3-3B, ProtGPT3-10B (re-analyzed from Tu & Chen nextaa outputs); AFDB used as ground truth for 3 bacterial DUFs
- **Wet-lab validation:** No
- **Code and data availability:** MIT-licensed GitHub repository with summary.tsv, transcripts, rebuilt PDBs, prompts, and nextaa scripts; README warns repo text/code was AI-drafted without thorough human audit.
- **Primary links:** [https://github.com/jkosinski/CheckLLMfold](https://github.com/jkosinski/CheckLLMfold), [https://x.com/jankosinski/status/2122997527192416440](https://x.com/jankosinski/status/2122997527192416440)

## Takeaway

CheckLLMfold demonstrates that Claude Fable 5.2's contact precision on famous PDBs (0.18-0.68, TM 0.33-0.52) is heavily confounded by exact PDB recognition (6/6 PDB ID recall and hardcoded FN3 strand topology), dropping near chance on two of three no-PDB-relative bacterial DUFs (0.04-0.07, TM 0.20-0.22). Its next-residue divergence test further proves Fable relies on remembered family relatives (predicting the old homolog 59% vs the new divergent residue 7%).

## Methods

- **Datasets and benchmarks:** Contact/3D folding: 4 famous PDBs (1UBQ, 256B, 1TEN, 2TRX) vs 3 bacterial DUF proteins with confident AFDB models and zero PDB relatives (A0A916UKZ0/DUF2489, A0A8J2YCU1/DUF2958, A0A5D0P002/DUF4288). Next-AA: 1,428 headless calls across 12 famous proteins, Tu & Chen UniRef50 cluster-size bins, and 48 new proteins (70-95% ID to old UniProtKB relatives, 192 divergent sites).
- **Evaluation protocol:** Headless OS-sandboxed claude -p (no network, randomized job directories) running Ovchinnikov's iterative contact-list + fold() script (4-6 rounds), compared against single-reply no-tools oneshot.py and direct recognition probes (6 calls/protein).
- **Controls included:** No-PDB-relative Foldseek/RCSB-filtered DUF targets, single-reply no-tools ablation, randomized directory names (after discarding a leaked ubiquitin path run), direct recognition probe, AFDB confidence re-scoring (pLDDT>=90, PAE<5 A), and old-vs-new homolog divergence test.
- **Compute and tools:** $1.14 to $3.91/run in iterative fold() mode (166-352 s) vs ~1,400 thinking tokens in single-reply no-tools mode (matching full protocol precision).
- **Metrics:** Long-range contact precision P_sep24_topL, medium+long P_sep6_topL, rebuilt CA TM_best_hand vs fold() ceiling on native contacts; next-AA real-minus-shuffled gap and old-vs-new divergent residue accuracy.
- **Wet-lab protocol:** None (computational contact/folding/memorization audit).

## Key Results

| Task | Regime | Benchmark | Metric | General LLM | Protein Baseline | Evidence Status | Notes |
|---|---|---|---|---|---|---|---|
| Structure prediction and side-chain packing | Intrinsic LLM + code execution (no specialized protein models) | 4 famous PDBs (1UBQ, 256B, 1TEN, 2TRX) vs 3 no-PDB-relative bacterial DUFs | Long-range top-L contact precision (\|i-j\|>=24) / TM_best_hand | Claude Fable 5.2 (full fold() protocol & single-reply oneshot): Famous PDBs: P_sep24_topL 0.179-0.684, TM 0.325-0.524 (6/6 exact PDB ID recognized); No-PDB DUFs: DUF2489 0.037-0.056 (TM 0.21-0.22), DUF2958 0.043-0.068 (TM 0.20), DUF4288 0.075-0.327 (TM 0.24-0.31) | Chance contact density (0.022-0.064) / fold() native-contact ceiling (TM 0.61-0.84): DUF2489 chance 0.022, DUF2958 chance 0.052, DUF4288 chance 0.032 | **Supported (controlled)** | Clean confounder audit: Fable's contact accuracy tracks PDB memorization (naming all 4 famous PDB IDs 6/6 and even hardcoding Tenascin FN3 strand labels A-G skipping D in scripts) and collapses near chance on 2 of 3 no-PDB-relative DUFs; single-reply no-tools matches iterative fold(). |
| Sequence recall and memorization | Intrinsic LLM (text only, no tools) | 48 new proteins (70-95% ID to old UniProtKB relatives, 192 sites) + UniRef50 cluster-size re-analysis | Old-relative vs new-protein residue prediction at divergent sites | Claude Fable 5.2: At divergent sites where old relative is known (n=29): predicts OLD relative 59% vs NEW residue 7%; Tu & Chen gap scales from +54.1 (UniRef50 >50) to +36.4 (cluster=1) to +22.1 (post-cutoff) | ProGen3-3B / ProtGPT3-10B (Tu & Chen re-analysis): ProGen3-3B +47.1 (>50) -> +23.8 (cluster=1) -> +12.0 (post-cutoff) | **Supported (controlled)** | Directly shows Fable 5.2 predicts from remembered UniProt family relatives (59% old relative vs 7% new divergence) while scaling like a protein language model across UniRef50 cluster sizes. |

## Confounder and Leakage Audit (8 Axes)

| Axis | Status | Evidence |
|---|---|---|
| Temporal cutoff | **Controlled** (`controlled`) | Explicitly contrasts pre-cutoff famous PDBs, no-PDB-relative DUFs, and post-cutoff new proteins diverging from old UniProtKB relatives (59% old vs 7% new). |
| Homology overlap | **Controlled** (`controlled`) | Uses RCSB seqsearch + Foldseek to verify zero PDB relatives for the 3 DUFs (best TM 0.52-0.60, E=0.52-1.1) and stratifies nextaa by UniRef50 cluster size. |
| Cross-modal / literature recall | **Controlled** (`controlled`) | Directly measured via recognition probes (4/4 famous PDBs named 6/6; DUFs 0/6 accession, 2/6-6/6 Pfam) and script inspection (catching Tenascin FN3 strand-label recall). |
| Preprocessing artifacts | **Concern** (`evidenced_concern`) | README transparently warns code and text were AI-drafted by Opus 5.5 without thorough human review, usage-limit-stopped runs were omitted, and bacterial ground truth uses AFDB models (checked at pLDDT>=90, PAE<5 A). |
| Downstream oracle filtering | **N/A** (`not_applicable`) | fold() reconstructs coordinates purely from the LLM's own contact list. |
| Compute & tool asymmetry | **Controlled** (`controlled`) | Directly ablates iterative fold() scripts against single-reply no-tools oneshot.py, showing equal contact precision. |
| Baseline fairness | **Concern** (`evidenced_concern`) | No head-to-head ESM-2 / AlphaFold contact precision baseline reported alongside Fable on the 7 targets (though native-contact ceilings and chance baselines are given). |
| Data & wet-lab transparency | **Controlled** (`controlled`) | All scripts, summary.tsv, transcripts, and rebuilt PDBs are MIT-licensed on GitHub (with AI-drafting caveat clearly stated). |
