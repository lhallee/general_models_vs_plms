# General language models against protein language models

General-purpose language models can now read and write protein sequences, which raises the question of whether they
can stand in for the protein language models that the field uses to predict what a mutation does. This repository
holds an experiment that tests one sharp form of that question on the [FLIP2](https://flip.protein.properties/)
benchmark, a second that asks whether memory of the sequences explains the answer, a third that sets the general model
against a generative protein language model tuned on the same variants, and a dated survey of what others have
reported. A general model that reads 256 labeled variants in its prompt matches the standard recipe of ridge
regression on protein language model features and does not beat it, a generative protein language model tuned on
those variants falls short of both under the recipe set in advance and reaches their level, without passing it,
under its best recipe, and the best predictor measured here averages the general model with the ridge.

The [explainer](docs/index.html) walks through every experiment step by step for a reader outside the field, with a
primer on each piece and one card per experiment, and it is a single HTML file with its figures inside that opens in
any browser once the repository is downloaded or cloned. Every table and figure is in [RESULTS.md](RESULTS.md), the
other public results are in [SURVEY.md](SURVEY.md), and the commands are in [REPRODUCE.md](REPRODUCE.md).

## What we found

Claude Mythos 5.2 was shown 256 labeled variants of a protein as plain text, with the protein unnamed, and was asked
to rank 100 held-out variants. Ridge regression was fit on the same 256 rows with features from ESM-2 650M and from
ESMC at three sizes, the generative protein language model ProGen2-small was tuned on them, and every method was
scored on the same queries. Each number below is a mean Spearman correlation with the measured values, and each
interval is a 95% interval from 2,000 paired bootstrap draws of the queries.

![Controls, ESMC, and Claude Mythos 5.2 on FLIP2 and on FGFR2](report/figures/mythos/tiers_both.png)

*Mean Spearman with 95% intervals, on the first 100 queries of each split. **A**, the 16 FLIP2 splits. **B**, the 2
FGFR2 splits, published after the model's training data end. Gray rows are controls, green rows are ESMC at three
sizes, and orange is Claude Mythos 5.2, whose interval is the shaded band. Triangles use no examples, and circles and
squares are ridges fit on the same 256 rows the model reads.*

| 256 examples unless said, first 100 queries of each split | FLIP2, 16 splits | FGFR2 after the cutoff, 2 splits |
|---|---|---|
| Best control without a trained protein model (ridge on one-hot sequence) | 0.142 (0.097 to 0.182) | 0.295 (0.175 to 0.410) |
| ProGen2-small score with no examples | 0.239 (0.188 to 0.286) | 0.412 (0.278 to 0.521) |
| ProGen2-small tuned on the 256 rows with a preference loss (DPO), recipe set in advance | 0.263 (0.214 to 0.306) | 0.314 (0.187 to 0.429) |
| ProGen2-small tuned, one setting for 20 epochs, picked after the first scores | 0.339 (0.290 to 0.379) | 0.342 (0.209 to 0.463) |
| Best ESMC score with no examples (ESMC-6B) | 0.346 (0.298 to 0.388) | 0.481 (0.355 to 0.601) |
| ProGen2-base tuned, the same one setting for 20 epochs | 0.360 (0.314 to 0.401) | 0.379 (0.241 to 0.499) |
| Best ridge on ESMC features | 0.367 (0.319 to 0.409) | 0.572 (0.469 to 0.663) |
| Claude Mythos 5.2 reading the 256 rows in its prompt | 0.380 (0.333 to 0.421) | 0.561 (0.458 to 0.646) |
| Average of the model's ranking and a ridge's | 0.443 (0.399 to 0.482) | 0.611 (0.516 to 0.692) |

The best ridge is the one on the ESMC-600M embedding for FLIP2 and the one on one-hot features with the ESMC-6B score
for FGFR2, and the average in the last row pairs the model with that same ridge.

### In-context learning ties the best ridge on ESMC features

Over the 16 FLIP2 splits the model reaches 0.380 and the best ridge reaches 0.367, so the paired difference is +0.013
with an interval of -0.038 to +0.065 that includes zero. The rule was fixed before any model was asked, and it counts
in-context learning as better only if that interval lies above zero. The ridge is a strong opponent, because it is the
best of several and is chosen on the queries it is scored on, while no control without a trained protein model scores
above 0.14.

The tie is an average of large wins and large losses, since the model is ahead on 3 splits, behind on 3, and undecided
on 10. It falls furthest behind where every test value lies above the range of its examples, and it leads most on the
absorption maxima of rhodopsins, which it predicts at 0.658 with no examples at all. With that dataset left out, a
check added after the results were seen, the model scores 0.36 and the ridge 0.38.

### The tie holds on data published after the model's training

FLIP2 was public before the model's training data end, so a good score there could come from memory of the benchmark.
The same comparison was therefore repeated on an FGFR2 drug resistance screen that MaveDB published on 20 November
2026, after that date, where the model scores 0.561 and the best ridge 0.572.

### A generative protein language model tuned on the same rows reaches the level of both at its best recipe, and passes neither

A ridge never changes its protein model, so a third experiment tried the recipe that changes every weight. For each
split, ProGen2-small was tuned on the 256 rows with a preference loss (Direct Preference Optimization, DPO), which
teaches the model to give a variant with a higher measured value a higher likelihood, and the tuned model then ranked
the same 100 queries. Under the recipe set before the run, with the learning rate, the weight of the loss, and the
epoch chosen on 51 of the 256 rows, it reaches 0.263 on FLIP2, which is -0.117 (-0.163 to -0.071) from Claude Mythos
5.2 and -0.104 (-0.159 to -0.050) from the best ridge, and 0.314 on FGFR2, where it is behind both as well.

That recipe understates the method by about 0.07. Runs added after those scores were seen show that the 51 rows
choose a worse setting and stopping point than one setting run to a fixed length, so that setting (learning rate
0.00001, weight 2, 20 epochs, nothing chosen) was scored on queries 101 to 300 of each split, which had no part in
picking it. There ProGen2-small scores 0.324, +0.075 (+0.049 to +0.102) above the recipe set in advance and -0.026
(-0.063 to +0.013) from the best ridge on those queries, and ProGen2-base, a model five times larger, scores 0.339,
-0.012 (-0.048 to +0.025) from that ridge. On the first 100 queries ProGen2-base scores 0.360, -0.006 (-0.061 to
+0.043) from the best ridge and -0.020 (-0.067 to +0.026) from Claude Mythos 5.2. Every one of those intervals
includes zero and none lies above it, so at its best recipe preference tuning is level with the ridge and with the
general model on FLIP2 and beats neither. On the first 100 FGFR2 queries it stays behind both, at 0.379, -0.183
(-0.334 to -0.037) from Claude Mythos 5.2.

![ProGen2-small with no examples, under a ridge, and tuned, beside the best ridges and Claude Mythos 5.2](report/figures/mythos/preference_tuning.png)

*Mean Spearman with 95% intervals, on the first 100 queries of each split. **A**, the 16 FLIP2 splits. **B**, the 2
FGFR2 splits. Olive rows are ProGen2-small: its score with no examples (triangle), a ridge on its hidden states
(circle), and the model tuned on the 256 rows with a Bradley-Terry loss (hollow hexagon) and with the ranked preference
loss (filled hexagon). The gray row is the control, the same tuning on shuffled values. Green rows are the best
ridges on ESMC features, and orange is Claude Mythos 5.2, whose interval is the shaded band.*

![One setting for 40 epochs, on the held-out rows and on every query](report/figures/mythos/preference_long.png)

*Mean Spearman over the 16 FLIP2 splits after each epoch of the one setting, for ProGen2-small. **A**, on the 51 rows
held out from the examples. **B**, on every query of each split. The triangle is the released model, the hexagons
are the two epochs read with nothing chosen, the green line and band are the best ridge on those queries, and the
dotted line is the score under the recipe set in advance.*

The held-out rows mislead because they are drawn like the examples and FLIP2's test queries are not. On those rows
the tuning does what it is built to do, lifting the mean Spearman from 0.245 to 0.628 at the chosen checkpoint, and
their score keeps rising with more epochs while the score on the test queries is level from about epoch 15. About
half of the 0.07 comes from five splits on which the 51 rows chose another setting, and half from an earlier stop.
The control explains the level of the first recipe: tuning on shuffled values lowers the released model's score by
0.109, and tuning on the measured values scores 0.133 above that control.

Four more changes were tried on ProGen2-small with the checkpoint still chosen on held-out rows. A learning rate ten
times lower (0.205) and a low-rank update on frozen weights (0.200) score below the first recipe. Tuning on every
labeled row of a split, up to 2,048, scores 0.344 on all queries, 0.088 above tuning on 256, and stays -0.065
(-0.093 to -0.037) from the best ridge fit on the same rows. The ranks of the tuned model averaged with those of the
best ridge or of Claude Mythos 5.2 tie the other method alone. On rows with no shift a ridge on the ESMC-600M
embedding still leads (0.637 against 0.52 for the tuned model at a fixed recipe and 0.572 for a ridge on
ProGen2-small's own hidden states), so part of what remains is the protein model and not the way it is fit. The
fixed recipe was picked with the measured values of the first 100 queries of these splits, it ran under one batch
order, and no ESMC model was tuned.

The preference loss by itself also leaves the tuned model unfit as a model of sequences. It reads only how the rows
of a batch differ, and under the fixed recipe the mean log-likelihood per residue that ProGen2-small gives the test
queries falls from -2.12 as released to -7.71 after 20 epochs, on every FLIP2 split. Adding the negative
log-likelihood of each pair's preferred row to the loss, as Iterative Reasoning Preference Optimization does, slows
that fall at the lightest of three weights and stops it at the other two. It does not change the comparison. At that
method's own weight of 1, the run named in advance as the primary one, the ranking does not improve: 0.295 on queries
101 to 300, -0.030 (-0.063 to +0.005) from the recipe without the term. At a tenth and a hundredth of that weight it
scores 0.348 and 0.344, +0.024 (-0.002 to +0.051) and +0.020 (+0.001 to +0.039), which is level with the best ridge
on those queries (0.351) and not above it. Three weights were read under one batch order, so the gain of about 0.02
is a lead and not a finding.

### The model learns from its examples and also uses what it knows about the protein

Three changes to the prompt, tried on one split per dataset where the plain prompt scores 0.496, separate what the
model takes from the rows from what it brings with it. Reversing the measured values reverses its ranking, to -0.235,
which shows that the ranking follows the rows. Renaming the residues with a cipher, so that the protein cannot be
recognized, lowers the score to 0.278, a little above a ridge on one-hot features at 0.209, so about half of the score
depends on knowing the protein. With no examples at all the model scores 0.391, where the best score a protein
language model gives without examples, from ESMC-6B, is 0.318.

### It recognizes the benchmark and does not recite its values

Shown a parent sequence and 30 unlabeled rows, the model named the protein, the measurement, and the source study of
all seven FLIP2 datasets, and for FGFR2 it named the kinase domain and gave no source. Recognition is not recall of
the measurements, however, because when it was asked to write the published values of the four datasets with
continuous values it matched 0 of 400 exactly.

### It can write many of these proteins out, and that does not explain its accuracy

A second experiment hid 15%, 50%, and 85% of the residues of 48 rhodopsins and asked each model to restore them.
Claude Mythos 5.2 restored 91%, 76%, and 57%, while ESMC-6B restored 59%, 54%, and 13% from the same masks and no
model rose above the 11% of a guess on shuffled copies. The model also returned the FGFR2 kinase domain whole with half of
it hidden, so the FGFR2 screen is new to it and the protein is not.

How much of a rhodopsin the model can restore does not predict its error on that rhodopsin's absorption maximum, with
a Spearman of -0.15 (-0.41 to +0.13) across the 48. For the protein language models the matching measure is the
pseudo-log-likelihood of a sequence, and it tracks the error of the ridge on the embedding for ESMC-600M alone among
the four models (-0.50, -0.64 to -0.33), so it is no general guide to where such a ridge fails.

### Averaging the two rankings beats both

The model and the ridge err on different variants, which makes the plain average of their rankings, with equal weights
and no tuning, the best predictor measured here at 0.443 on FLIP2 and 0.611 on FGFR2.

### Through the public API, Claude Fable 5.2 also ties its best ridge

A separate experiment asked four Claude models through the public API and scored all 300 queries of each split, so its
numbers are compared only with baselines scored on those queries and are never set beside the ones above. Claude Fable
5.2 scores 0.365 where the best ridge scores 0.355, a difference of +0.009 (-0.015 to +0.036). Claude Opus 5 (0.264),
Claude Opus 5.5 (0.252), and Claude Haiku 5.5 (0.176) are below that ridge, and Claude Opus 5.5 refused 68 of its 265
prompts at low effort.

## What the field has found

Across the public results available on 23 November 2026, general language models have reached protein language models
that read a single sequence on ranking variants and have not passed them, while specialist predictors stay ahead. The
table sets each result for a general model beside its protein-specific comparison, and [SURVEY.md](SURVEY.md) lists
every source with its authors, its date, and how much of it was read, since several rows rest on an abstract alone.

| Question | General language models | Protein-specific models | Source |
|---|---|---|---|
| Do they hold natural protein sequences? | Claude Fable 5.2 predicts the next residue of proteins from families of unknown function at 54% against 9% on shuffled copies, a gap of 45 points. GPT, Gemini, and seven open-weight models are near zero | Autoregressive protein language models show smaller gaps on the same probe, 32.5 points for ProGen3-3B and 27.7 for ProtGPT3-10B | Tu and Chen, November 2026 |
| | Claude Mythos 5.2 restores 76% of a rhodopsin with half of its residues hidden | ESMC-6B restores 54% from the same masks | This repository |
| Do they know structure? | Contact maps from Claude Fable 5.2 reach a long-range precision of 0.18 to 0.49 on well-studied proteins and 0.04 to 0.33 on proteins with no relative in the PDB | No comparison in the source | Kosinski, November 2026 |
| Can they rank variants with no examples? | Claude Opus 5 at Spearman 0.406 and GPT 5.6 Sol at 0.402 on 276 tasks | ESM2-650M at 0.411, and the specialist VenusREM at 0.523 | Arora and colleagues, July 2026 |
| | DeepSeek-Reasoner alone at 0.159 on ProteinGym, and 0.551 as part of an agent system with tools | ESM2-650M at 0.414 and VenusREM at 0.518 | Tan and colleagues, January 2026 |
| | Claude Mythos 5.2 at 0.391 on 7 FLIP2 splits | ESMC-6B at 0.318 | This repository |
| Can they learn from labeled examples in the prompt? | Competitive with XGBoost on four peptide classification tasks and not significantly above it | No protein language model baseline in the abstract | Zhao and colleagues, November 2026 |
| | Claude Mythos 5.2 at 0.380 on FLIP2 with 256 examples | Ridge on the ESMC-600M embedding at 0.367 | This repository |
| Can they answer questions about proteins? | GPT-5 at 60.65% mean accuracy on twelve tasks | The best protein-specific chat model at 31.83% | Rong and colleagues, December 2025 |

Across these sources and the results here, memory of natural sequences is real in the newest Claude models and is
absent from the other general models that were probed, yet in our test it does not predict how well a property is
predicted. On ranking variants, a general model sits level with a single-sequence protein language
model whether it is given no examples or 256 of them, and below the specialist predictors. The gains reported so far
come from combination, either as a general model that is given the outputs of specialist tools or as its ranking
averaged with a ridge on protein language model features.

Among the sources surveyed, none gives a general model labeled protein variants in its prompt and compares it with a
ridge on protein language model features fit on the same examples, which is the comparison made here.

## How the comparison was made

| Piece | What it is |
|---|---|
| Data | The 16 splits of the 7 FLIP2 datasets, and 2 splits of an FGFR2 drug resistance screen from MaveDB that was published after the models' training data end |
| Task | 256 labeled rows drawn from a split's training part and up to 300 test queries, 100 to a prompt, as plain text with one variant and its value per line and the protein unnamed |
| Baselines | Ridge regression on one-hot features, on the mean-pooled embedding of ESM-2 650M, ESMC-300M, ESMC-600M, and ESMC-6B, and on one-hot features with a model's zero-shot score, all fit on the same 256 rows, and the zero-shot scores alone |
| Controls for the ridge | A random vector, amino acid composition, an untrained ESMC-300M, and trained ESMC given shuffled residues or scrambled positions |
| Controls for the language model | Enciphered residues, reversed values, shuffled values, no examples, and the protein named |
| Tuned protein language model | ProGen2-small, 151 million weights, with every weight tuned on 205 of the 256 rows by a ranked preference loss (Direct Preference Optimization, DPO) or a Bradley-Terry loss, and the learning rate, the weight of the loss, and the epoch chosen on the other 51 rows (73 and 51 on the one split with 124 rows) |
| Controls for the tuning | The same model with no examples, a ridge on its hidden states, the same tuning on shuffled values, and two more batch orders |
| Other scales of the tuning | One setting run for 20 and 40 epochs with nothing chosen, on ProGen2-small and on ProGen2-base (764 million weights), scored on every query; a learning rate ten times lower; a low-rank update; every labeled row of a split; the one setting with a likelihood term added to the loss at three weights; and every baseline fit on the 205 tuning rows and scored on the 51 held-out rows |
| Score | Spearman on each split, averaged over splits, with every comparison made on the same queries for both sides |

Claude Mythos 5.2 was asked through fresh subagents of a coding agent, one per prompt, at the effort level chosen on
validation rows, and each subagent's transcript was checked for the model that answered and for any tool use beyond
reading its prompt and writing its answer.

## Limits

Each Claude Mythos 5.2 score rests on one prompt of 100 queries per split, one draw of the 256 rows, and one answer per
prompt, so the intervals cover the draw of queries and not the spread between two answers to one prompt. The subagent
route also puts a coding agent's instructions in front of every prompt, and the prompt controls ran on one split per
dataset. The first 100 FGFR2 queries are easier than the rest for every method, with the best ridge at 0.45 on all 300,
which makes 0.56 a score on those queries and not the level of the dataset. The test of sequence memory covers one
dataset, and its 48 sequences detect only a correlation of about 0.4 or more. The tuned protein language models are
two generative models of one family. The recipe set in advance has three batch orders and a control, and the fixed
recipe that reaches the ridge has one batch order, no control, and a setting picked with the measured values of the
first 100 test queries, so its scores on those queries are optimistic and its scores on queries 101 to 300 are the
ones to read. The likelihood term was read at three weights on ProGen2-small alone, under one batch order. The result
does not say what a tuned ESMC would score.

## Repository

| Path | What it holds |
|---|---|
| [docs/index.html](docs/index.html) | The explainer, one self-contained page |
| [RESULTS.md](RESULTS.md) | Every table and figure |
| [SURVEY.md](SURVEY.md) | Other public results on general language models and protein sequences, with dates and authors |
| [REPRODUCE.md](REPRODUCE.md) | The commands for the tables, the figures, and the experiments |
| `report/figures/`, `report/tables/` | The figures, and the summary tables they are drawn from |
| [prompts/](prompts/README.md) | Every prompt Claude Mythos 5.2 was asked, as the file it read, with its answer |
| `general_models_vs_plms/`, `scripts/`, `tests/` | The package, the scripts of the subagent route, and the test suite |
| `Dockerfile`, `box.sh` | The image and the entrypoint the protein language models ran through |

## Sources

FLIP2 is the benchmark of Didi, Alamdari, Lu, Wittmann, Johnston, Amini, Madani, Czeneszew, Dallago, and Yang
([bioRxiv, February 2026](https://doi.org/10.64898/2026.02.23.707496)), and the FGFR2 screen is MaveDB score set
`urn:mavedb:00001329-a-3` by Annika Maria Metzner with Sven Diederichs. The ridge on one-hot features with a zero-shot
score follows Hsu, Nisonoff, Fannjiang, and Listgarten (Nature Biotechnology, 2022). ProGen2 is the model of Nijkamp,
Ruffolo, Weinstein, Naik, and Madani (Cell Systems, 2023), the ranked preference loss is that of Widatalla and
colleagues as Yang and colleagues applied it to ProGen2, and the Bradley-Terry loss is that of Hawkins-Hooker and
colleagues. The likelihood term is that of Pang and colleagues (Iterative Reasoning Preference Optimization, NeurIPS
2024, [arXiv 2404.19733](https://arxiv.org/abs/2404.19733)). Diego del Alamo proposed the comparison of in-context
learning with ridge regression on FLIP2, and the comparison with preference tuning of ProGen2. Full references are in
[SURVEY.md](SURVEY.md).
