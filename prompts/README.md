# Prompts and answers

Every prompt that Claude Mythos 5.2 was asked, as the file the model read, with the answer it wrote beside it.
`asked.csv` lists the 93 prompts: the task, split, arm, effort level, start time, seconds, thinking tokens, and whether
an answer was written. 78 have an answer. The other 15 were cut off when the account's usage limit was reached and are
not scored. One answered test prompt at medium effort, in `e2`, is not scored either, because the final plan asks no
test prompt at that level.

## How a prompt was asked

One fresh subagent of a coding agent (Claude Code) was started per prompt, with the effort level of `asked.csv`, and
this as its only instruction, `{path}` being the prompt's folder:

```text
Read the file {path}\prompt.txt in full. It holds a task and says how to reply. Write your reply, exactly in the form
the task asks for and with nothing else, to {path}\answer.txt with the Write tool. Use no other tools: do not run code
or commands, search, or open any other file. If the Read tool does not return the whole file, read the rest before you
answer. When the file is written, reply with the single word done.
```

A subagent knows nothing of the conversation that started it. The folders carry neutral names so that a path tells it
nothing about the task. Each subagent's transcript was then audited: the model that answered, the effort level, and
the tools it called. `tool_calls` in `asked.csv` is that audit, and every answered prompt used `Read` and `Write` only.

## The sets

| Folder | What it holds |
|---|---|
| `e1`, `e2`, `e3` | The plain prompt, 256 labeled rows and 100 queries, at low, medium, and high effort |
| `a` | The same prompt with the residues enciphered (`k256_cipher`) |
| `b` | The shown values reversed (`k256_flipped`), and no examples with the protein named (`k0_described`) |
| `g` | No examples, and the published values asked for (`k0_recall`) |
| `h` | Dataset recognition: the parent and 30 unlabeled rows. In `asked.csv` its `split` reads `recognition` and its `arm` is the dataset |
| `d1` to `d5` | Validation prompts at five effort levels, used to choose the level |

The prompt text is built by `general_models_vs_plms/prompts.py`, and the rows of each task are drawn with a fixed seed
by `general_models_vs_plms/tasks.py`. The API models read prompts of the same form, three per task, built by the same
code.

## The data in these files

The variants and measured values in the prompts come from two sources, and both keep their own license.

- **FLIP2.** Kieran Didi, Sarah Alamdari, Alex X. Lu, Bruce Wittmann, Kadina E. Johnston, Ava P. Amini, Ali Madani,
  Maya Czeneszew, Christian Dallago, and Kevin K. Yang. FLIP2: Expanding Protein Fitness Landscape Benchmarks for
  Real-World Machine Learning Applications. bioRxiv, 24 February 2026,
  [10.64898/2026.02.23.707496](https://doi.org/10.64898/2026.02.23.707496). Data on Zenodo,
  [10.5281/zenodo.18433203](https://doi.org/10.5281/zenodo.18433203), CC BY 4.0. Each FLIP2 dataset comes from a study
  that the benchmark's site, [flip.protein.properties](https://flip.protein.properties/), names.
- **FGFR2.** MaveDB score set `urn:mavedb:00001329-a-3`, "FGFR2 Erdafitinib resistance", created by Annika Maria
  Metzner with Sven Diederichs as contributor, published 20 November 2026, CC BY-SA 4.0. The prompts and answers of
  the `fgfr2` splits are derived from it and are shared under the same license.
