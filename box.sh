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
#   bash box.sh survey                     what else the shared machine runs: load, memory, disk, containers, GPU clients
#   bash box.sh pull REPOSITORY PATTERN .. files of a run's Hub dataset into $STATE (HF_TOKEN)
#   bash box.sh prefer check [MODEL]       load a ProGen2 checkpoint and check it against its authors' reference
#   bash box.sh prefer pilot [MODEL] [HOURS]  preference tuning, every arm, on two splits, into $STATE/prefer, logged
#                                          to $STATE/prefer_MODEL.log; it waits for an idle GPU and yields to another client
#   bash box.sh prefer main [MODEL] [HOURS]   the same on every split not yet done, starting no split after HOURS in all
#   bash box.sh prefer chain [MODEL] [HOURS]  the pilot, its projection, and the rest if the projection fits HOURS
#   bash box.sh prefer trace [MODEL] [HOURS]  the same chain for the untuned score and the ranked-loss arm alone, with the
#                                          first 100 queries scored after every epoch, into $STATE/prefer_trace
#   bash box.sh prefer short [MODEL] [HOURS]  the same chain for a larger model: its untuned score, the ridge on its
#                                          hidden states, and the ranked-loss arm at the lower learning rate alone
#   bash box.sh prefer night [MODEL] [UNTIL]  after any other chain has ended, three variants of the ranked-loss arm, each
#                                          as a chain: a lower learning rate, a low-rank update of frozen weights, and
#                                          the largest example sets; none starts at or after UNTIL (UTC, HHMM)
#   bash box.sh prefer likelihood [MODEL] [HOURS] [share]  after any other chain has ended, the fixed recipe with a
#                                          likelihood term beside the ranked loss, one chain a weight (1, 0.01, 0.1),
#                                          HOURS each; `share` enters beside another line's job inside the ceilings
#   bash box.sh prefer stop                end this stage's waiting chains, never one whose container is running
#   bash box.sh bundle                    tables, manifests and probes as one archive, $STATE/bundle.tar.gz
#   bash box.sh logs [LINES]               the ends of the stage logs
#   bash box.sh emit PATH                  a file under $STATE as base64 on standard output
#   bash box.sh publish REPOSITORY [PATTERN ...]  everything but the Hub cache, or what the patterns match, to a private
#                                          Hub dataset (HF_TOKEN)
#
# State lives under $STATE (~/gmp): data/, tasks/, responses/ (one budget ledger for every Claude call), features/,
# baselines/, tables/, prefer/ (ProGen2 preference tuning), hf/ (the Hub cache). Containers run as root, so what they
# write is root-owned.
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
  sudo --preserve-env=ANTHROPIC_API_KEY,HF_TOKEN docker run --rm --gpus all --ipc=host ${GMP_NAME:+--name "$GMP_NAME"} \
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
  survey)
    uptime
    free -g | sed -n 2p
    df -h / | tail -1
    sudo docker ps --format '{{.Names}} {{.Image}} {{.Status}} {{.Command}}' | cut -c1-160
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader
    date -u +%FT%TZ
    ;;
  pull)
    repository=${1:?box.sh pull REPOSITORY PATTERN ...}
    shift
    module pull --repository "$repository" --out /state --only "${@:?box.sh pull REPOSITORY PATTERN ...}"
    ;;
  prefer)
    # ProGen2 preference tuning on the k256 task of each split. The pilot is the full run's container and settings on
    # one split of equal-length variants and on Rhomax, whose sequences vary in length.
    stage=${1:?box.sh prefer check|inside|states|pilot|main|chain|trace|short|night|likelihood|stop [MODEL] [HOURS or UNTIL] [long|fixed|share]}
    model=${2:-small}
    hours=${3:-5}
    preset=${4:-}
    out=/state/prefer
    log="$STATE/prefer_$model.log"
    extra=()
    if [ "$stage" = chain ] && [ "$preset" = long ]; then
      # One setting of the ranked loss for twice the epochs, with every query scored after every epoch, so that the
      # run can also be read after 20 and after 40 epochs with nothing chosen. It is a `chain`, so a waiting `night`
      # stage waits for it, and it waits itself for a `short` chain to end.
      log="$STATE/prefer_long_$model.log"
      extra=(--variant long --arms zero,dpo --learning-rates 1e-5 --betas 2 --epochs 40 --trace 300 --fixed-epochs 20,40)
      echo "PREFER long $model QUEUED $(date -u +%FT%TZ)" >> "$log"
      while pgrep -f 'box\.sh prefer short ' > /dev/null; do sleep 30; done
    elif [ "$stage" = chain ] && [ "$preset" = fixed ]; then
      # The same setting for 20 epochs on another checkpoint, read after the last epoch with nothing chosen. It
      # waits for a `short` chain and for the `night` stage, whose variants are queued ahead of it.
      log="$STATE/prefer_fixed_$model.log"
      extra=(--variant long --arms zero,dpo --learning-rates 1e-5 --betas 2 --epochs 20 --trace 300 --fixed-epochs 20)
      echo "PREFER fixed $model QUEUED $(date -u +%FT%TZ)" >> "$log"
      while pgrep -f 'box\.sh prefer (short|night) ' > /dev/null; do sleep 30; done
    elif [ "$stage" = trace ]; then
      # The untuned score and the ranked-loss arm again, with the first 100 queries scored after every epoch, as a
      # chain of its own into its own folder: what a reader with the measured values needs to see every epoch's score.
      out=/state/prefer_trace
      log="$STATE/prefer_trace_$model.log"
      extra=(--arms zero,dpo --trace 100)
    elif [ "$stage" = short ]; then
      # For a model too large for every arm within the hours: its untuned score, the ridge on its hidden states, and
      # the ranked-loss arm at the lower learning rate alone, as the same chain.
      extra=(--arms zero,ridge,dpo --learning-rates 1e-5)
    fi
    tune() {
      module preference_tuning --data /state/data --tasks /state/tasks/main --out "$out" --model "$model" --hours "$hours" \
        ${extra[@]+"${extra[@]}"} "$@"
    }
    gpu_clients() {
      # GPU processes other than the MPS daemon, which holds no work, one a line.
      nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader |
        grep -v -e nvidia-cuda-mps-server -e '^$' || true
    }
    guarded() {
      # Enter a GPU that timed runs of another session share without ever running beside one: start only after the
      # GPU has had no client for 90 s, hold it for 90 s without computing, and leave at once if another client
      # starts in that time (the stage then exits 75, and the wait begins again). A session that looks later finds
      # the GPU taken and waits its turn.
      local status idle watcher
      if [ "$preset" = share ]; then
        # Beside another line's job that is not a timed run, as the machine's rules allow inside its ceilings: enter
        # once the GPU has under half its memory in use, so every job together stays under 90 percent, and hold
        # nothing idle. The other job is never touched.
        while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -ge 49000 ]; do sleep 30; done
        echo "GUARD sharing the GPU, $(gpu_clients | wc -l) other client(s), entering $(date -u +%FT%TZ)"
        status=0
        "$@" || status=$?
        return "$status"
      fi
      while true; do
        idle=0
        while [ "$idle" -lt 90 ]; do
          if [ -n "$(gpu_clients)" ]; then idle=0; else idle=$((idle + 10)); fi
          sleep 10
        done
        rm -f "$STATE/prefer_yield"
        echo "GUARD gpu idle for 90 s, entering $(date -u +%FT%TZ)"
        ( for _ in $(seq 20); do
            sleep 5
            if [ "$(gpu_clients | wc -l)" -gt 1 ]; then touch "$STATE/prefer_yield"; break; fi
          done ) &
        watcher=$!
        status=0
        "$@" --hold 90 --yield-file /state/prefer_yield || status=$?
        wait "$watcher" || true
        [ "$status" = 75 ] || return "$status"
        echo "GUARD yielded to another GPU client $(date -u +%FT%TZ)"
      done
    }
    pilot() { GMP_NAME="gmp-prefer-$model" guarded tune --splits amylase/by_mutation,rhomax/by_wild_type; }
    main() { GMP_NAME="gmp-prefer-$model" guarded tune; }
    case "$stage" in
      check)
        module preference_tuning --model "$model" --check
        ;;
      inside)
        # The baselines of the first experiment fit on the rows the tuning fits on and scored on its held-out rows,
        # from the features under $STATE/features. It loads no model and makes no GPU client, so it needs no guard.
        module preference_tuning --data /state/data --tasks /state/tasks/main --out "$out" --model "$model" --inside /state/features
        ;;
      states)
        # A ridge on each untuned model's hidden states, fit on the same tuning rows and scored on the same held-out
        # rows. It loads a model, so it begins once no chain of this stage runs and enters the GPU under the guard, one
        # model after the other.
        log="$STATE/prefer_states.log"
        echo "PREFER states QUEUED $(date -u +%FT%TZ)" >> "$log"
        while pgrep -f 'box\.sh prefer (short|chain|trace|night) ' > /dev/null; do sleep 30; done
        extra=(--inside-states)
        for model in small base; do
          echo "PREFER states $model START $(date -u +%FT%TZ)" >> "$log"
          main >> "$log" 2>&1 || echo "PREFER states $model STOPPED with status $?" >> "$log"
        done
        echo "PREFER states both DONE $(date -u +%FT%TZ)" >> "$log"
        ;;
      pilot|main)
        echo "PREFER $stage $model START $(date -u +%FT%TZ)" >> "$log"
        "$stage" >> "$log" 2>&1
        echo "PREFER $stage $model DONE $(date -u +%FT%TZ)" >> "$log"
        ;;
      chain|trace|short)
        # The pilot, its projection of every split, and the full run in one stay on the GPU: the full run starts only
        # if the pilot ended cleanly and the projection fits the hours.
        echo "PREFER $stage $model START $(date -u +%FT%TZ)" >> "$log"
        { pilot && tune --project && main; } >> "$log" 2>&1 || echo "PREFER $stage $model STOPPED with status $?" >> "$log"
        echo "PREFER $stage $model DONE $(date -u +%FT%TZ)" >> "$log"
        ;;
      night)
        # The exploration of scale: one chain after another on the same model, each a variant that changes one thing
        # about the tuning and writes its own folder under $STATE/prefer. It begins once no other chain of this stage
        # runs, gives each variant its own hours, and starts none at or after the hour UNTIL (UTC, as HHMM).
        until_hhmm=${3:-2359}
        log="$STATE/prefer_night_$model.log"
        echo "PREFER night $model QUEUED $(date -u +%FT%TZ)" >> "$log"
        while pgrep -f 'box\.sh prefer (short|chain|trace) ' > /dev/null; do sleep 30; done
        for entry in "slow 0.8" "lowrank 0.8" "rows 1.4"; do
          variant=${entry% *}
          hours=${entry#* }
          if [ "$(date -u +%H%M)" -ge "$until_hhmm" ]; then
            echo "PREFER $variant $model SKIPPED at $(date -u +%FT%TZ), past $until_hhmm" >> "$log"
            continue
          fi
          case "$variant" in
            slow) extra=(--variant slow --arms zero,dpo --learning-rates 1e-6 --trace 100) ;;
            lowrank) extra=(--variant lowrank --arms zero,dpo --low-rank 8 --learning-rates 1e-4,1e-3 --betas 2 --trace 100) ;;
            rows) extra=(--variant rows --arm klarge --arms zero,dpo --learning-rates 1e-5 --epochs 6) ;;
          esac
          echo "PREFER $variant $model START $(date -u +%FT%TZ)" >> "$log"
          { pilot && tune --project && main; } >> "$log" 2>&1 || echo "PREFER $variant $model STOPPED with status $?" >> "$log"
          echo "PREFER $variant $model ENDED $(date -u +%FT%TZ)" >> "$log"
        done
        echo "PREFER night $model DONE $(date -u +%FT%TZ)" >> "$log"
        ;;
      stop)
        # End this stage's waiting chains, and only waiting ones: with a container of this stage running nothing is
        # touched, since stopping a busy container on this machine faults the shared MPS server.
        if sudo docker ps --format '{{.Names}}' | grep -q '^gmp-prefer-'; then
          echo "a gmp-prefer container is running; nothing was stopped"
        elif pkill -f 'box[.]sh prefer (chain|trace|short|night|likelihood|states) '; then
          echo "stopped the waiting chains at $(date -u +%FT%TZ)"
        else
          echo "no chain was waiting"
        fi
        exit 0
        ;;
      likelihood)
        # A likelihood term beside the ranked loss, on the fixed recipe: the negative log-likelihood of the preferred
        # row of each pair of a batch, at three weights, each as a chain into its own folder under $STATE/prefer. The
        # weight of 1 runs first, then the lightest. It begins once no other chain of this stage runs. With `share` as
        # the fourth argument it enters beside another line's job, where the guard would wait for an idle GPU.
        log="$STATE/prefer_likelihood_$model.log"
        echo "PREFER likelihood $model QUEUED $(date -u +%FT%TZ)" >> "$log"
        while pgrep -f 'box\.sh prefer (short|chain|trace|night) ' > /dev/null; do sleep 30; done
        for entry in "nll 1" "nlllight 0.01" "nllmid 0.1"; do
          variant=${entry% *}
          extra=(--variant "$variant" --arms zero,dpo --learning-rates 1e-5 --betas 2 --epochs 20 --trace 300 --fixed-epochs 20
                 --likelihood-weight "${entry#* }")
          echo "PREFER $variant $model START $(date -u +%FT%TZ)" >> "$log"
          { pilot && tune --project && main; } >> "$log" 2>&1 || echo "PREFER $variant $model STOPPED with status $?" >> "$log"
          echo "PREFER $variant $model ENDED $(date -u +%FT%TZ)" >> "$log"
        done
        echo "PREFER likelihood $model DONE $(date -u +%FT%TZ)" >> "$log"
        ;;
      *)
        echo "usage: box.sh prefer check|inside|states|pilot|main|chain|trace|short|night|likelihood|stop [MODEL] [HOURS or UNTIL] [long|fixed|share]" >&2
        exit 2
        ;;
    esac
    tr '\r' '\n' < "$log" 2>/dev/null | grep -v -E 'it/s\]$|s/it\]$' | tail -n 30 || true
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
    for log in "$STATE"/pilot.log "$STATE"/features.log "$STATE"/esmc_6b.log "$STATE"/knowledge.log "$STATE"/prefer_*.log; do
      if [ -f "$log" ]; then
        echo "== $(basename "$log")"
        tr '\r' '\n' < "$log" | grep -v -E 'it/s\]$|s/it\]$' | tail -n "${1:-12}" | cut -c1-300
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
    echo "usage: box.sh build|python|prepare|start|pilot|features|main|fullref|more|tables|extra|survey|pull|prefer|bundle|logs|emit|publish ..." >&2
    exit 2
    ;;
esac
