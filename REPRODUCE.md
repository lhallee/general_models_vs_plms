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
