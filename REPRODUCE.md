# Reproducing the experiments

The summary tables and figures can be rebuilt from the files in this repository, and the experiments themselves need
one GPU machine for the protein language models and a coding agent or an API key for the language models.

## From the tables in this repository

```bash
pip install -r requirements.txt
python -m pytest tests -q
# Summaries from a run's scored tables, then the three figure sets.
python -m general_models_vs_plms.report --tables <tables> --out <summary>
python -m general_models_vs_plms.figures --summary report/tables --out figures/mythos --primary claude-mythos-5-2@harness-low
python -m general_models_vs_plms.figures --summary report/tables --out figures/api --primary claude-fable-5-2@low
```

## The experiments themselves

The baselines and the API route run on one GPU machine, in the image, through `box.sh`. State is written under
`$STATE`.

```bash
bash box.sh prepare           # build the image, download FLIP2 and the FGFR2 score set, write the tasks
bash box.sh features          # ESM-2 650M and ESMC-600M features, then the baselines
bash box.sh more              # ESMC-300M, ESMC-6B, and the embedding controls
bash box.sh pilot             # API route: three effort levels on validation rows (needs ANTHROPIC_API_KEY)
bash box.sh main high low     # API route: every arm, the sequence recall probe, and the tables
```

The stages that call the Claude API spend money on every call, under a budget that `claude_calls.py` enforces.

The subagent route runs wherever the coding agent runs. It needs no API key and no GPU.

```bash
export PYTHONPATH="$PWD"
python scripts/flip2_in_context/export_sets.py --state <state> --work <work>   # prompt files under neutral names
python scripts/flip2_in_context/launch_order.py --work <work> low              # the order, and each instruction
# Start one fresh subagent per item of <work>/w_private/order.json, with the item's prompt and effort level.
STATE=<state> WORK=<work> TRANSCRIPTS=<subagent transcripts> bash scripts/flip2_in_context/run_all.sh
```

`run_all.sh` audits, scores, summarizes, and draws. It audits each subagent's transcript before it scores an answer: the model that answered, its effort level,
and that it called no tool beyond reading its prompt and writing its answer. It is written for Git Bash on Windows.
`posthoc.py` holds the analyses added after the results were seen, which are labeled so wherever they are reported.

The sequence-knowledge experiment has one stage on the GPU machine and six prompts on the subagent route.

```bash
bash box.sh knowledge prompts/k/sequences.json    # masked recovery and PPLL for the four protein language models
python -m general_models_vs_plms.sequence_knowledge prompts --set prompts/k/sequences.json --out <work>
# Start one fresh subagent per prompt folder of <work>, at low effort, with the instruction in prompts/README.md.
python -m general_models_vs_plms.sequence_knowledge score --set prompts/k/sequences.json --prompts <work> --out answers.csv
T=report/tables/sequence_knowledge
python -m general_models_vs_plms.knowledge_report --set prompts/k/sequences.json --answers $T/answers.csv --masked $T \
    --predictions $T/rhomax_predictions.csv --summary report/tables --out <out>
```

`rhomax_predictions.csv` holds the Rhomax rows of the in-context experiment's scored predictions that the report
reads. `sequence_knowledge select` draws the set from the splits and those predictions.

The preference-tuning experiment runs on the GPU machine from the tasks the first experiment wrote, and its scores
are joined onto that experiment's scored tables.

```bash
bash box.sh prefer check small                    # load ProGen2-small and compare it with its authors' reference score
bash box.sh prefer chain small 5                  # two splits, a projection of the rest, then every split within 5 hours
python -m general_models_vs_plms.collect --tables <tables> --add <state>/prefer/methods.json --out <new tables>
python -m general_models_vs_plms.report --tables <new tables> --out <summary> --tuning <state>/prefer/records.json
python -m general_models_vs_plms.figures --summary <summary> --out figures/mythos \
    --primary claude-mythos-5-2@harness-low --only preference_tuning preference_by_split
```

`prefer` waits for a GPU with no other client before it starts, and `preference_tuning.py` holds every setting of the
tuning: the checkpoints and their revisions, the two losses, the grid, and the rows held out for selection.

The trace repeats the ranked-loss arm and scores the first 100 queries after every epoch, so that every setting and
epoch can be compared on the test queries. That comparison reads their measured values, which the choice of a
checkpoint never does, so the trace is a diagnostic and no reported score comes from it.

```bash
bash box.sh prefer trace small 2                  # the same chain, the ranked-loss arm alone, into <state>/prefer_trace
python -m general_models_vs_plms.report --tables <new tables> --out <summary> \
    --trace <state>/prefer_trace/records.json <state>/prefer/records.json
python -m general_models_vs_plms.figures --summary <summary> --out figures/mythos \
    --primary claude-mythos-5-2@harness-low --only preference_trace
```

The runs added after the reported scores were seen use the same chain. Each writes its methods under a name of its
own into `<state>/prefer`, so the `collect` and `report` commands above take them in with the reported run.

```bash
bash box.sh prefer short base 2.4                 # ProGen2-base: no examples, the ridge, and the ranked loss at 0.00001
bash box.sh prefer chain small 1.2 long           # one setting for 40 epochs, every query scored after every epoch
bash box.sh prefer night small 1100               # three variants of the tuning of ProGen2-small, none started after 11:00 UTC
bash box.sh prefer chain base 1.5 fixed           # the same one setting on ProGen2-base for 20 epochs
bash box.sh prefer likelihood small 0.8           # the same one setting with a likelihood term at the weights 1, 0.01, and 0.1
python -m general_models_vs_plms.figures --summary <summary> --out figures/mythos \
    --primary claude-mythos-5-2@harness-low --only preference_long preference_scale
```

The three variants are a learning rate of 0.000001 (`slow`), a low-rank update with every released weight frozen
(`lowrank`), and tuning on every labeled row of a split in place of 256 (`rows`). The `long` run is scored three
ways: after 20 epochs and after 40 epochs with no checkpoint chosen (`long20`, `long40`), and at the checkpoint its
held-out rows choose (`long`). Its records go to `report --trace` with the two files above. The `likelihood` stage
adds the negative log-likelihood of each pair's preferred row to the ranked loss of that setting, and its three runs
are read after 20 epochs with nothing chosen (`nll20`, `nllmid20`, and `nlllight20` for the weights 1, 0.1, and
0.01). Their records go to `report --trace` as well, whose table holds the mean score of the queries after every
epoch beside the Spearman.

The baselines on the rows the tuning holds out need the features the first experiment stored and no GPU.

```bash
bash box.sh prefer inside small                   # each baseline fit on the tuning rows, scored on the held-out rows
bash box.sh prefer states                         # the same for a ridge on the hidden states of each ProGen2 model
python -m general_models_vs_plms.report --tables <new tables> --out <summary> \
    --tuning <state>/prefer/records.json \
    --inside <state>/prefer/inside.json <state>/prefer/inside_small.json <state>/prefer/inside_base.json
```
