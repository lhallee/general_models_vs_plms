#!/usr/bin/env bash
# Score the subagent route from the answer files and write everything the write-up reads: audit, ingest, collect,
# report, post hoc numbers, data checks, figures, and the explainer page.
#
#   STATE=<run state> WORK=<folder with w/ and w_private/> TRANSCRIPTS=<subagent transcripts> bash run_all.sh [figures]
#
# STATE holds flip2/, tasks/ and final/hub/ as pulled from the run's Hub dataset. Everything is written under ROUTE
# (STATE/route unless set), which is emptied first. "figures" redraws the figures and the page from the summaries
# already there. PAGE=1 also builds the explainer page, ROUTE/page/claude_vs_ridge_flip2.html, from the page sources in
# the folder EXPLAINER names.
#
# PROJECT is the folder that holds the general_models_vs_plms package, two above this one, and ROOT the nearest folder
# at or above it that holds foundry/. Written for Git Bash on Windows: the `py` launcher, and `;` between the two
# entries of PYTHONPATH.
set -euo pipefail

: "${STATE:?}" "${WORK:?}" "${TRANSCRIPTS:?}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && (pwd -W 2>/dev/null || pwd))"
PROJECT="$(cd "$HERE/../.." && (pwd -W 2>/dev/null || pwd))"
ROOT="$PROJECT"
while [ ! -d "$ROOT/foundry" ]; do
    [ "$(dirname "$ROOT")" != "$ROOT" ] || { echo "run_all.sh: no folder at or above $PROJECT holds foundry/" >&2; exit 2; }
    ROOT="$(dirname "$ROOT")"
done
ROUTE="${ROUTE:-$STATE/route}"
MYTHOS="claude-mythos-5-2@harness-low"
export PYTHONPATH="$PROJECT;$ROOT" PYTHONDONTWRITEBYTECODE=1
cd "$ROOT"

if [ "${1:-all}" != "figures" ]; then
    rm -rf "$ROUTE"
    py -X utf8 "$HERE/score_route.py" --state "$STATE" --work "$WORK" --transcripts "$TRANSCRIPTS" --route "$ROUTE" --collect
    py -X utf8 -m general_models_vs_plms.report --tables "$ROUTE/tables/main" --out "$ROUTE/summary" \
        --validation "$ROUTE/tables/pilot" "$STATE/final/hub/tables/pilot"
    py -X utf8 -m general_models_vs_plms.recall_probe summarize --data "$STATE/flip2" \
        --probes "$STATE/final/hub/responses/probes" --out "$ROUTE/summary/recall.csv"
    py -X utf8 "$HERE/posthoc.py" "$ROUTE/tables/main" "$ROUTE/summary" "$ROUTE/posthoc/variables.json" > /dev/null
    # The measured facts of each split, for the explainer's split table. This step compares every test row with the
    # training rows and takes several minutes.
    py -X utf8 -m general_models_vs_plms.data_checks --data "$STATE/flip2" --tasks "$STATE/tasks/main" \
        --out "$ROUTE/posthoc/data_checks.csv" > /dev/null
fi

rm -rf "$ROUTE/figures" "$ROUTE/page"
py -X utf8 -m general_models_vs_plms.figures --summary "$ROUTE/summary" --out "$ROUTE/figures/mythos" --primary "$MYTHOS" \
    --only tiers tiers_both headline per_split postcutoff claude_controls exact_matches effort complementarity arms_by_split
# The API models are a second experiment on other queries: their figures are drawn apart and hold no subagent score.
py -X utf8 -m general_models_vs_plms.figures --summary "$ROUTE/summary" --out "$ROUTE/figures/api" \
    --primary "claude-fable-5-2@low" \
    --only headline per_split postcutoff learning_curve claude_models refusals complementarity within_groups api_controls \
    api_effort
# The baselines alone, on every query: these two figures draw no language model.
py -X utf8 -m general_models_vs_plms.figures --summary "$ROUTE/summary" --out "$ROUTE/figures/baselines" \
    --primary "claude-fable-5-2@low" --only ridge_controls plm_models
if [ "${PAGE:-0}" = "1" ]; then
    : "${EXPLAINER:?}"
    mkdir -p "$ROUTE/page"
    cp -r "$ROUTE/figures" "$ROUTE/page/figures"
    py -X utf8 "$EXPLAINER/build_page.py" "$ROUTE/posthoc/data_checks.csv" "$ROUTE/summary" "$MYTHOS" \
        "$ROUTE/posthoc/variables.json" "$ROUTE/page/claude_vs_ridge_flip2.html"
fi
date -u +"finished %Y-%m-%d %H:%M:%S UTC"
