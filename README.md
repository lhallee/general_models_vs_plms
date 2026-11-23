# General language models against protein language models

General-purpose language models can now read and write protein sequences, which raises the question of whether they
can stand in for the protein language models that the field uses to predict what a mutation does. This repository
holds an experiment that tests one sharp form of that question on the [FLIP2](https://flip.protein.properties/)
benchmark, a second experiment that asks whether memory of the sequences explains the answer, and a dated survey of
what others have reported. A general model that reads 256 labeled variants in its prompt matches the standard recipe
of ridge regression on protein language model features and does not beat it, and the best predictor measured here
averages the two.

The [explainer](docs/index.html) walks through every experiment step by step for a reader outside the field, with a
primer on each piece and one card per experiment, and it is a single HTML file with its figures inside that opens in
any browser once the repository is downloaded or cloned. Every table and figure is in [RESULTS.md](RESULTS.md), the
other public results are in [SURVEY.md](SURVEY.md), and the commands are in [REPRODUCE.md](REPRODUCE.md).

## What we found

Claude Mythos 5.2 was shown 256 labeled variants of a protein as plain text, with the protein unnamed, and was asked
to rank 100 held-out variants. Ridge regression was fit on the same 256 rows with features from ESM-2 650M and from
ESMC at three sizes, and every method was scored on the same queries. Each number below is a mean Spearman correlation
with the measured values, and each interval is a 95% interval from 2,000 paired bootstrap draws of the queries.

![Controls, ESMC, and Claude Mythos 5.2 on FLIP2 and on FGFR2](report/figures/mythos/tiers_both.png)

*Mean Spearman with 95% intervals, on the first 100 queries of each split. **A**, the 16 FLIP2 splits. **B**, the 2
FGFR2 splits, published after the model's training data end. Gray rows are controls, green rows are ESMC at three
sizes, and orange is Claude Mythos 5.2, whose interval is the shaded band. Triangles use no examples, and circles and
squares are ridges fit on the same 256 rows the model reads.*

| 256 examples unless said, first 100 queries of each split | FLIP2, 16 splits | FGFR2 after the cutoff, 2 splits |
|---|---|---|
| Best control without a trained protein model (ridge on one-hot sequence) | 0.142 (0.097 to 0.182) | 0.295 (0.175 to 0.410) |
| Best ESMC score with no examples (ESMC-6B) | 0.346 (0.298 to 0.388) | 0.481 (0.355 to 0.601) |
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
dataset, and its 48 sequences detect only a correlation of about 0.4 or more.

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
score follows Hsu, Nisonoff, Fannjiang, and Listgarten (Nature Biotechnology, 2022). Diego del Alamo proposed the
comparison of in-context learning with ridge regression on FLIP2. Full references are in [SURVEY.md](SURVEY.md).
