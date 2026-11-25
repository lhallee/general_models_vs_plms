# Introducing Claude Fable 5.2 and Claude Mythos 5.2 (Capsid Modeling and 24-hr Binder Benchmark)

- **Study ID:** `anthropic_2026_fable_mythos_52`
- **Authors:** Anthropic
- **Date:** 2026-11-09 (updated 2026-11-11)
- **Source tier:** Lab technical report (`technical_report`)
- **Verification depth:** Full text verified (`full_text`)
- **Tasks:** Structure prediction and side-chain packing, De novo backbone and binder design
- **Regimes:** LLM agent orchestrating specialized protein tools
- **Models tested:** Claude Mythos 5.2, Claude Fable 5.2, Claude Mythos 5.1, Claude Opus 5.5
- **Protein-specific baselines:** OpenFold3-preview2 and ESMFold2 used as orchestrated folding engines (no standalone baseline table)
- **Wet-lab validation:** No
- **Code and data availability:** No capsid target list, per-target RMSDs, code, prompts, or binder metric definition released in the blog post (defers to System Card).
- **Primary links:** [https://www.anthropic.com/claude-fable-and-mythos-5-2](https://www.anthropic.com/claude-fable-and-mythos-5-2)

## Takeaway

Anthropic's release post shows Claude Mythos 5.2 orchestrating OpenFold3-preview2 and ESMFold2 via symmetry-aware scripts to model 28/34 capsids within 3 A, alongside an undocumented 85.1 binder benchmark score. Without target lists, baselines, or metric definitions, neither claim is independently verifiable as a controlled LLM-vs-pLM comparison.

## Methods

- **Datasets and benchmarks:** 34 natural and engineered viral capsids (including AAV2 PDB:1LP3); unspecified 24-hour de novo binder design benchmark.
- **Evaluation protocol:** Mythos 5.2 writes code running OpenFold3-preview2 and ESMFold2 on overlapping chain subsets and merges predictions using known capsid symmetries ('big bang' initialization referenced in prompt, Footnote 3). Counted accurate if either OpenFold3-preview2 OR ESMFold2 achieves Ca RMSD < 3 A.
- **Controls included:** None reported in the release post.
- **Compute and tools:** NVIDIA H200 GPU running OpenFold3-preview2 (~58 min on AAV2) and ESMFold2 (~7 min on AAV2); GPT-6 Astra and Gemini 4 Carbon omitted ('—') on the binder table.
- **Metrics:** Capsid count within 3 A Ca RMSD (best of two folding models); undefined numeric score (85.1) for 24-hr binder design.
- **Wet-lab protocol:** None.

## Key Results

| Task | Regime | Benchmark | Metric | General LLM | Protein Baseline | Evidence Status | Notes |
|---|---|---|---|---|---|---|---|
| Structure prediction and side-chain packing | LLM agent orchestrating specialized protein tools | 34 natural and engineered capsids (incl. AAV2 PDB:1LP3) | Count with Ca RMSD < 3 A (OpenFold3-preview2 or ESMFold2) | Claude Mythos 5.2 writing symmetry-merging wrapper around OpenFold3-preview2 / ESMFold2: 28/34 capsids < 3 A (AAV2: 1.2 A OpenFold3-preview2, 1.4 A ESMFold2) | Standalone OpenFold3 / ESMFold2 / SymProFold / CombFold: Not reported | **Suggestive (confounded)** | Folding is performed by OpenFold3-preview2 and ESMFold2 with best-of-two selection, known symmetries, and 'big bang' initialization hinted in prompt; no baseline or capsid list published. |
| De novo backbone and binder design | LLM agent orchestrating specialized protein tools | 24-hour de novo binder design (unspecified targets/metric) | Unspecified score (defers to System Card) | Claude Mythos 5.2 / Mythos 5.1 / Opus 5.5 (no safeguards): 85.1 (Mythos 5.2) / 79.1 (Mythos 5.1) / 82.6 (Opus 5.5) | GPT-6 Astra / Gemini 4 Carbon / specialized pipelines: Not reported ('—') | **Unverified / likely leaked** | No targets, metric definition, or external baselines appear in this release post. |

## Confounder and Leakage Audit (8 Axes)

| Axis | Status | Evidence |
|---|---|---|
| Temporal cutoff | **Uncontrolled** (`uncontrolled`) | AAV2 (PDB:1LP3) is a classic public structure and no temporal holdout is stated for the 34 capsids or binder targets. |
| Homology overlap | **Uncontrolled** (`uncontrolled`) | No sequence or structural split against Claude, OpenFold3-preview2, or ESMFold2 training data. |
| Cross-modal / literature recall | **Concern** (`evidenced_concern`) | Known capsid symmetries were leveraged and Footnote 3 explicitly notes 'big bang' initialization was referenced in the prompt. |
| Preprocessing artifacts | **Uncontrolled** (`uncontrolled`) | Best-of-two selection across OpenFold3-preview2 and ESMFold2 defines capsid success; binder score 85.1 lacks metric units or target list. |
| Downstream oracle filtering | **Concern** (`evidenced_concern`) | Structure prediction is executed entirely by OpenFold3-preview2 and ESMFold2 rather than intrinsic LLM folding. |
| Compute & tool asymmetry | **Uncontrolled** (`uncontrolled`) | Runs OpenFold3-preview2 and ESMFold2 on H200 GPUs; competitor rows are blank. |
| Baseline fairness | **Uncontrolled** (`uncontrolled`) | No comparison against existing symmetry-folding tools (SymProFold, CombFold, FoldaVirus) cited in Footnote 3. |
| Data & wet-lab transparency | **Concern** (`evidenced_concern`) | No target list, code, prompts, wet-lab data, or per-capsid RMSD table released in the post. |
