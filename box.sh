#!/usr/bin/env bash
# The Docker entrypoint for general_models_vs_plms on a GPU machine: every workload runs in
# the image, nothing in host Python. Run as
#   bash box.sh <command>
# from this folder. When a launcher ships the tree, the run directory is mounted at its own path.
#
#   bash box.sh build                      the image (Dockerfile)
#   bash box.sh python ...                 any python in the image
#   bash box.sh start                      prepare, then pilot and features side by side, logged to $STATE/*.log
#   bash box.sh prepare                    build, download the data, write the pilot and main tasks
#   bash box.sh pilot                      the pilot: three efforts of Claude Fable 5.2 on validation rows
#   bash box.sh features                   a 64-row smoke run, then both protein language models and the baselines
#   bash box.sh smoke [MODEL ...]          one short request per model through the code every stage uses
#   bash box.sh main SELECTED CHEAP        the main arms at two efforts in order of weight, the recall probes, the tables
#   bash box.sh fullref                    features of every training row, then the ridges fit on all of them
#   bash box.sh more                       ESMC-300M, ESMC-6B and the embedding controls, added to the baselines and tables
#   bash box.sh tables                     the tidy tables again, from whatever has run
#   bash box.sh knowledge SET [MODEL ...]  masked recovery and pseudo-log-likelihood of each sequence of a set
#   bash box.sh extra MODEL EFFORT ARMS    one more model, effort and arm list on the main tasks, then the tables
#   bash box.sh bundle                     tables, manifests and probes as one archive, $STATE/bundle.tar.gz
#   bash box.sh logs [LINES]               the ends of the stage logs
#   bash box.sh emit PATH                  a file under $STATE as base64 on standard output
#   bash box.sh publish REPOSITORY [PATTERN ...]  everything but the Hub cache, or what the patterns match, to a private
#                                          Hub dataset (HF_TOKEN)
#
# State lives under $STATE (~/gmp): data/, tasks/, responses/ (one budget ledger for every Claude call), features/,
# baselines/, tables/, hf/ (the Hub cache). Containers run as root, so what they write is root-owned.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE="${GMP_STATE:-$HOME/gmp}"
IMAGE="${GMP_IMAGE:-general_models_vs_plms:cu128-20261121}"
BUDGET="${GMP_BUDGET_USD:-150}"
mkdir -p "$STATE"

# The run directory (holds launch.py and tree/) when shipped, else the repository root (holds foundry/).
ROOT="$HERE"
while [ "$ROOT" != "/" ] && [ ! -f "$ROOT/launch.py" ] && [ ! -d "$ROOT/foundry" ]; do ROOT="$(dirname "$ROOT")"; done
[ "$ROOT" != "/" ] || { echo "box.sh: neither a shipped run directory nor a repository root holds $HERE" >&2; exit 2; }
if [ -d "$ROOT/tree" ]; then TREE="$ROOT/tree"; else TREE="$ROOT"; fi
# The project, then what the launcher shipped beside the tree (foundry), then the launcher's own path.
IMPORTS="$HERE"
if [ -d "$ROOT/import-roots" ]; then IMPORTS="$IMPORTS:$ROOT/import-roots"; fi
IMPORTS="$IMPORTS:$TREE${PYTHONPATH:+:$PYTHONPATH}"

command=${1:-}
shift || true

docker_run() {
  # sudo resets the environment, so the credential names (never values) pass through it and into the container.
  sudo --preserve-env=ANTHROPIC_API_KEY,HF_TOKEN docker run --rm --gpus all --ipc=host \
    -v "$ROOT:$ROOT" -v "$STATE:/state" -w "$HERE" \
    -e PYTHONPATH="$IMPORTS" -e HF_HOME=/state/hf -e HF_HUB_DISABLE_PROGRESS_BARS=1 \
    -e ANTHROPIC_API_KEY -e HF_TOKEN \
    "$IMAGE" "$@"
}

module() {
  local name=$1
  shift
  docker_run python -m "general_models_vs_plms.$name" "$@"
}

claude() {
  # A run that stops on the budget or on failed calls exits 3; the chain goes on to what is cheaper or independent.
  module run_claude --tasks /state/tasks/main --out /state/responses --budget "$BUDGET" "$@" || echo "run_claude $* exited $?"
}

tables() {
  module collect --data /state/data --tasks /state/tasks/main --responses /state/responses \
    --baselines /state/baselines --out /state/tables/main
}

case "$command" in
  build)
    sudo docker build -t "$IMAGE" "$HERE"
    sudo docker image inspect "$IMAGE" --format 'image {{.Id}} created {{.Created}}'
    ;;
  python)
    docker_run python "$@"
    ;;
  prepare)
    sudo docker image inspect "$IMAGE" > /dev/null 2>&1 || sudo docker build -t "$IMAGE" "$HERE"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    docker_run python -c "import torch; assert torch.cuda.is_available(); print('cuda', torch.cuda.get_device_name(0), 'bf16', torch.cuda.is_bf16_supported())"
    module fetch_data --data /state/data
    module build_tasks --data /state/data --out /state/tasks --stage pilot
    module build_tasks --data /state/data --out /state/tasks --stage main --representation "${1:-substitutions}"
    echo "PREPARE DONE $(date -u +%FT%TZ)"
    ;;
  start)
    bash "$HERE/box.sh" prepare
    bash "$HERE/box.sh" pilot > "$STATE/pilot.log" 2>&1 &
    bash "$HERE/box.sh" features > "$STATE/features.log" 2>&1 &
    wait
    echo "START DONE $(date -u +%FT%TZ)"
    ;;
  smoke)
    # One short request per model through the same code as every stage, so an SDK mismatch costs a cent, not a stage.
    for model in "${@:-claude-haiku-5-5}"; do
      module claude_calls --model "$model" --effort low
    done
    ;;
  lost)
    # Count calls that were billed and left no record against the budget, at a ceiling: box.sh lost NAME USD CALLS.
    printf '{"usd": %s, "calls": %s}\n' "${2:?usd}" "${3:?calls}" | sudo tee "$STATE/responses/spend__lost_${1:?name}.json"
    ;;
  pilot)
    bash "$HERE/box.sh" smoke claude-fable-5-2
    for effort in low medium high; do
      module run_claude --tasks /state/tasks/pilot --out /state/responses --budget "$BUDGET" \
        --model claude-fable-5-2 --effort "$effort" --workers 5 || echo "pilot $effort exited $?"
    done
    module collect --data /state/data --tasks /state/tasks/pilot --responses /state/responses --out /state/tables/pilot
    echo "PILOT DONE $(date -u +%FT%TZ)"
    ;;
  features)
    for model in esm2_650m esmc_600m; do
      module plm_features --data /state/data --tasks /state/tasks/main --out /state/features_smoke --model "$model" --limit 64
    done
    echo "SMOKE DONE $(date -u +%FT%TZ)"
    for model in esm2_650m esmc_600m; do
      module plm_features --data /state/data --tasks /state/tasks/main --out /state/features --model "$model"
    done
    module baselines --data /state/data --tasks /state/tasks/main --features /state/features --out /state/baselines
    echo "FEATURES DONE $(date -u +%FT%TZ)"
    ;;
  main)
    # The effort the pilot selected runs the primary arm and its cipher control; the cheap effort runs every arm.
    # Lines are in order of weight, so what the budget cuts first is what the answer needs least.
    selected=${1:?box.sh main SELECTED_EFFORT CHEAP_EFFORT}
    cheap=${2:?box.sh main SELECTED_EFFORT CHEAP_EFFORT}
    bash "$HERE/box.sh" smoke claude-fable-5-2 claude-opus-5-5 claude-opus-5 claude-haiku-5-5
    claude --model claude-fable-5-2 --effort "$selected" --arms k256 --seeds 0 &
    claude --model claude-fable-5-2 --effort "$cheap" --arms k256,k256_cipher,k256_described,k0_described --seeds 0 &
    claude --model claude-opus-5-5 --effort "$cheap" --arms k256,k256_cipher --seeds 0 &
    wait
    claude --model claude-fable-5-2 --effort "$selected" --arms k256_cipher --seeds 0 &
    claude --model claude-opus-5-5 --effort "$selected" --arms k256 --seeds 0 &
    claude --model claude-opus-5 --effort "$cheap" --arms k256,k256_cipher --seeds 0 &
    claude --model claude-haiku-5-5 --effort "$cheap" --arms k256,k256_cipher --seeds 0 &
    wait
    for model in claude-fable-5-2 claude-opus-5-5 claude-opus-5 claude-haiku-5-5; do
      module recall_probe run --data /state/data --out /state/responses --model "$model" --effort "$cheap" --budget "$BUDGET" \
        || echo "recall_probe $model exited $?"
    done
    tables
    echo "MAIN FIRST HALF DONE $(date -u +%FT%TZ)"
    claude --model claude-fable-5-2 --effort "$cheap" --arms k256_shuffled,k256_flipped,k256,k16,k64 &
    claude --model claude-opus-5-5 --effort "$cheap" --arms k256_described,k0_described --seeds 0 &
    wait
    claude --model claude-fable-5-2 --effort "$cheap" --arms klarge --seeds 0 &
    claude --model claude-opus-5-5 --effort "$cheap" --arms klarge --seeds 0 &
    wait
    tables
    echo "MAIN DONE $(date -u +%FT%TZ)"
    ;;
  fullref)
    for model in esm2_650m esmc_600m; do
      module plm_features --data /state/data --tasks /state/tasks/main --out /state/features_full --model "$model" --all-train
    done
    module baselines --data /state/data --tasks /state/tasks/main --features /state/features_full --out /state/baselines --full-train
    echo "FULLREF DONE $(date -u +%FT%TZ)"
    ;;
  more)
    # Every ESMC model beside ESM-2 650M, and the embedding controls
    # (an untrained ESMC-300M, reordered sequences, amino acid composition, random vectors).
    embed() { module plm_features --data /state/data --tasks /state/tasks/main "$@"; }
    add_baselines() { module baselines --data /state/data --tasks /state/tasks/main --out /state/baselines --add "$@"; }
    for variant in pretrained random_init shuffled permuted; do
      embed --out /state/features_smoke --model esmc_300m --variant "$variant" --limit 64
    done
    echo "MORE SMOKE DONE $(date -u +%FT%TZ)"
    # The 6B model takes most of the time, so it runs beside the small models from the start.
    embed --out /state/features --model esmc_6b > "$STATE/esmc_6b.log" 2>&1 &
    six=$!
    for variant in pretrained random_init shuffled permuted; do
      embed --out /state/features --model esmc_300m --variant "$variant"
    done
    for variant in shuffled permuted; do
      embed --out /state/features --model esmc_600m --variant "$variant"
    done
    add_baselines --features /state/features --models esmc_300m
    tables
    echo "MORE SMALL DONE $(date -u +%FT%TZ)"
    embed --out /state/features_full --model esmc_300m --all-train
    add_baselines --features /state/features_full --models esmc_300m --full-train
    tables
    echo "MORE FULL DONE $(date -u +%FT%TZ)"
    wait "$six"
    add_baselines --features /state/features --models esmc_6b --controls ""
    tables
    echo "MORE DONE $(date -u +%FT%TZ)"
    ;;
  tables)
    tables
    ;;
  knowledge)
    # Masked recovery and pseudo-log-likelihood of every sequence of a set, one model after another, smallest first, so
    # the first model's time sizes the rest. The set is a file the launcher shipped, named from the tree's root.
    set_file=${1:?box.sh knowledge SET_FILE [MODEL ...]}
    shift
    for model in "${@:-esmc_300m esmc_600m esm2_650m esmc_6b}"; do
      for name in $model; do
        echo "KNOWLEDGE $name START $(date -u +%FT%TZ)"
        module sequence_knowledge esmc --set "$TREE/$set_file" --out /state/knowledge --model "$name"
        echo "KNOWLEDGE $name DONE $(date -u +%FT%TZ)"
      done
    done > "$STATE/knowledge.log" 2>&1
    tail -n 12 "$STATE/knowledge.log"
    ;;
  extra)
    model=${1:?box.sh extra MODEL EFFORT ARMS [SEEDS]}
    claude --model "$model" --effort "${2:?effort}" --arms "${3:?arms}" --seeds "${4:-0}"
    tables
    echo "EXTRA DONE $(date -u +%FT%TZ)"
    ;;
  bundle)
    members=()
    for member in tables tasks/pilot/manifest.json tasks/main/manifest.json data/data_manifest.json \
      data/fgfr2/provenance.json responses baselines pilot.log features.log esmc_6b.log; do
      if [ -e "$STATE/$member" ]; then members+=("$member"); fi
    done
    for summary in "$STATE"/features/*/summary*.json "$STATE"/features_full/*/summary*.json; do
      if [ -e "$summary" ]; then members+=("${summary#"$STATE"/}"); fi
    done
    tar -czf "$STATE/bundle.tar.gz" -C "$STATE" "${members[@]}"
    ls -la "$STATE/bundle.tar.gz"
    sha256sum "$STATE/bundle.tar.gz"
    ;;
  logs)
    # The ends of the stage logs, with progress bars split into lines.
    for name in pilot features esmc_6b knowledge; do
      if [ -f "$STATE/$name.log" ]; then
        echo "== $name.log"
        tr '\r' '\n' < "$STATE/$name.log" | tail -n "${1:-12}" | cut -c1-300
      fi
    done
    ls "$STATE"
    df -h / | tail -1
    uptime
    sudo docker images --format '{{.Repository}}:{{.Tag}} {{.Size}} {{.CreatedSince}}'
    sudo docker ps --format '{{.Image}} {{.Status}} {{.Command}}' | cut -c1-160
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    date -u +%FT%TZ
    ;;
  show)
    # One log under $STATE, whole, with progress bars reduced to their last state.
    tr '\r' '\n' < "$STATE/${1:?box.sh show NAME}" | grep -v -E '^\s*$|it/s\]$|s/it\]$|\?it/s\]$' | cut -c1-400
    ;;
  emit)
    # A small file under $STATE as base64 between markers, for a launcher that only returns text. The launcher passes
    # about 15 kB a second whatever the line length, so anything over a megabyte goes through `publish` instead.
    echo "-----BEGIN ${1:?box.sh emit PATH}-----"
    base64 -w 60000 "$STATE/$1"
    echo "-----END $1-----"
    ;;
  publish)
    repository=${1:?box.sh publish REPOSITORY [PATTERN ...]}
    shift
    if [ "$#" -gt 0 ]; then
      module publish --state /state --repository "$repository" --only "$@"
    else
      module publish --state /state --repository "$repository"
    fi
    ;;
  *)
    echo "usage: box.sh build|python|prepare|start|pilot|features|main|fullref|more|tables|extra|bundle|logs|emit|publish ..." >&2
    exit 2
    ;;
esac
