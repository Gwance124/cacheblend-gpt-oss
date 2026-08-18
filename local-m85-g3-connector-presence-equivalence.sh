#!/usr/bin/env bash
# Deterministic long-context A/B/A for connector absence versus presence.
# Run on solab-g3. All arms explicitly enable HMA and use identical requests.

set -euo pipefail

readonly CACHEBLEND_REPO=/mnt/nvme3n1/mlee/cacheblend-gpt-oss
readonly CACHEBLEND_BRANCH=cacheblend-scatter-diagnostic-and-checklayer
readonly CACHEBLEND_POINTER=/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m8.5-connector-presence-equivalence-20260818.current
readonly CACHEBLEND_BASE_DIR=/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m8.5-connector-presence-equivalence-20260818

CACHEBLEND_CURRENT_PID=""
CACHEBLEND_LMCACHE_PID=""

stop_current_server() {
  if test -z "$CACHEBLEND_CURRENT_PID"; then
    return 0
  fi
  if kill -0 "$CACHEBLEND_CURRENT_PID" 2>/dev/null; then
    kill -TERM "$CACHEBLEND_CURRENT_PID"
    for _ in $(seq 1 60); do
      if ! kill -0 "$CACHEBLEND_CURRENT_PID" 2>/dev/null; then
        break
      fi
      sleep 1
    done
  fi
  if kill -0 "$CACHEBLEND_CURRENT_PID" 2>/dev/null; then
    echo "VLLM_DID_NOT_STOP_PID=$CACHEBLEND_CURRENT_PID" >&2
    return 1
  fi
  wait "$CACHEBLEND_CURRENT_PID" 2>/dev/null || true
  CACHEBLEND_CURRENT_PID=""
}

stop_lmcache_server() {
  if test -z "$CACHEBLEND_LMCACHE_PID"; then
    return 0
  fi
  if kill -0 "$CACHEBLEND_LMCACHE_PID" 2>/dev/null; then
    kill -TERM "$CACHEBLEND_LMCACHE_PID"
    for _ in $(seq 1 30); do
      if ! kill -0 "$CACHEBLEND_LMCACHE_PID" 2>/dev/null; then
        break
      fi
      sleep 1
    done
  fi
  if kill -0 "$CACHEBLEND_LMCACHE_PID" 2>/dev/null; then
    echo "LMCACHE_DID_NOT_STOP_PID=$CACHEBLEND_LMCACHE_PID" >&2
    return 1
  fi
  wait "$CACHEBLEND_LMCACHE_PID" 2>/dev/null || true
  CACHEBLEND_LMCACHE_PID=""
}

stop_all_servers() {
  local status=0
  stop_current_server || status=1
  stop_lmcache_server || status=1
  return "$status"
}

stop_existing_listener() {
  local port="$1"
  local listener_pid
  listener_pid="$(
    ss -ltnp 2>/dev/null \
      | awk -v pattern=":${port}$" '$4 ~ pattern { x=$NF; sub(/^.*pid=/,"",x); sub(/,.*/,"",x); print x; exit }'
  )"
  case "$listener_pid" in
    '') return 0 ;;
    *[!0-9]*)
      echo "INVALID_PORT_${port}_LISTENER_PID=$listener_pid" >&2
      return 1
      ;;
  esac
  echo "STOPPING_EXISTING_PORT_${port}_PID=$listener_pid"
  kill -TERM "$listener_pid"
  for _ in $(seq 1 60); do
    if ! kill -0 "$listener_pid" 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  echo "EXISTING_PORT_${port}_LISTENER_DID_NOT_STOP=$listener_pid" >&2
  return 1
}

write_launch_command() {
  local path="$1"
  shift
  {
    echo '#!/usr/bin/env bash'
    printf 'exec'
    printf ' %q' "$@"
    echo
  } > "$path"
  chmod 700 "$path"
}

require_clean_tracked_worktree() {
  if test -n "$(git status --porcelain --untracked-files=no)"; then
    echo "STOP_TRACKED_WORKTREE_CHANGES" >&2
    git status --short --untracked-files=no >&2
    return 1
  fi
}

start_lmcache_server() {
  nohup .venv/bin/python -m cacheblend_gpt_oss.storage.lmcache_server_v0_4_3 \
    --host 127.0.0.1 \
    --port 5556 \
    --chunk-size 256 \
    --hash-algorithm blake3 \
    --l1-size-gb 8 \
    --l1-init-size-gb 8 \
    --eviction-policy LRU \
    --max-workers 1 \
    > "$CACHEBLEND_RUN_DIR/lmcache-server.log" 2>&1 < /dev/null &
  CACHEBLEND_LMCACHE_PID=$!
  echo "$CACHEBLEND_LMCACHE_PID" > "$CACHEBLEND_RUN_DIR/lmcache-server.pid"

  local ready=no
  for _ in $(seq 1 90); do
    if ss -ltn 2>/dev/null \
      | awk '$4 ~ /:5556$/ { found=1 } END { exit found ? 0 : 1 }'; then
      ready=yes
      break
    fi
    if ! kill -0 "$CACHEBLEND_LMCACHE_PID" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  echo "LMCACHE_READY=$ready"
  if test "$ready" != yes; then
    tail -n 160 "$CACHEBLEND_RUN_DIR/lmcache-server.log"
    return 1
  fi
}

run_arm() {
  local label="$1"
  local mode="$2"
  local connector_attached="$3"
  local arm_dir="$CACHEBLEND_RUN_DIR/$label"
  mkdir -p "$arm_dir"

  local -a command=(
    .venv/bin/vllm serve
    "$CACHEBLEND_MODEL_PATH"
    --served-model-name openai/gpt-oss-20b
    --tensor-parallel-size 1
    --dtype bfloat16
    --max-model-len 131072
    --gpu-memory-utilization 0.50
    --max-num-seqs 1
    --max-num-batched-tokens 131072
    --long-prefill-token-threshold 0
    --no-async-scheduling
    --enforce-eager
    --enable-prefix-caching
    --kv-cache-dtype auto
    --attention-backend TRITON_ATTN
    --no-disable-hybrid-kv-cache-manager
    --enable-auto-tool-choice
    --tool-call-parser openai
    --generation-config vllm
    --max-logprobs -1
    --port 8000
  )
  if test "$connector_attached" = yes; then
    command+=(--kv-transfer-config "$CACHEBLEND_KV_CONFIG_JSON")
  fi

  write_launch_command "$arm_dir/launch-command.sh" "${command[@]}"
  printf '%s\n' "$mode" > "$arm_dir/mode.txt"
  nvidia-smi \
    --query-gpu=name,memory.total,memory.used,memory.free,driver_version \
    --format=csv,noheader \
    > "$arm_dir/gpu-before.txt"

  nohup "${command[@]}" > "$arm_dir/vllm-server.log" 2>&1 < /dev/null &
  CACHEBLEND_CURRENT_PID=$!
  echo "$CACHEBLEND_CURRENT_PID" > "$arm_dir/vllm-server.pid"

  local ready=no
  for _ in $(seq 1 300); do
    local http_code
    http_code="$(
      curl -sS \
        -o "$arm_dir/models.json" \
        -w '%{http_code}' \
        http://127.0.0.1:8000/v1/models \
        2>/dev/null || true
    )"
    if test "$http_code" = 200; then
      ready=yes
      break
    fi
    if ! kill -0 "$CACHEBLEND_CURRENT_PID" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  echo "${label}_VLLM_READY=$ready"
  if test "$ready" != yes; then
    tail -n 200 "$arm_dir/vllm-server.log"
    return 1
  fi

  curl -fsS http://127.0.0.1:8000/metrics > "$arm_dir/metrics-startup.prom"
  .venv/bin/python scripts/capture_hybrid_flag_responses.py \
    --mode "$mode" \
    --base-url http://127.0.0.1:8000 \
    --timeout-seconds 1800 \
    --warmup \
    --filler-repetitions-per-turn 20000 \
    --output "$arm_dir/responses.json" \
    | tee "$arm_dir/responses-summary.txt"
  curl -fsS http://127.0.0.1:8000/metrics > "$arm_dir/metrics-after.prom"
  nvidia-smi \
    --query-gpu=name,memory.total,memory.used,memory.free \
    --format=csv,noheader \
    > "$arm_dir/gpu-after.txt"
  stop_current_server
}

main() {
  cd "$CACHEBLEND_REPO"
  require_clean_tracked_worktree
  git fetch origin
  git switch "$CACHEBLEND_BRANCH" 2>/dev/null \
    || git switch -c "$CACHEBLEND_BRANCH" --track "origin/$CACHEBLEND_BRANCH"
  git pull --ff-only
  require_clean_tracked_worktree
  if test -n "$(git status --porcelain --untracked-files=normal)"; then
    echo "PRESERVING_UNTRACKED_FILES"
    git status --short --untracked-files=normal
  fi

  export TIKTOKEN_ENCODINGS_BASE=/mnt/nvme3n1/labuser/.cache/tiktoken/encodings
  export TIKTOKEN_RS_CACHE_DIR=/mnt/nvme3n1/labuser/.cache/tiktoken
  export TIKTOKEN_CACHE_DIR=/mnt/nvme3n1/labuser/.cache/tiktoken
  export TRITON_CACHE_DIR="$CACHEBLEND_REPO/.triton-cache"
  export VLLM_USE_V2_MODEL_RUNNER=0
  export CUDA_VISIBLE_DEVICES=0

  export CACHEBLEND_MODEL_PATH=/mnt/nvme3n1/labuser/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee
  export CACHEBLEND_MODEL_REVISION=6cee5e81ee83917806bbde320786a8fb61efebee
  export CACHEBLEND_TOKENIZER_REVISION=6cee5e81ee83917806bbde320786a8fb61efebee
  export CACHEBLEND_MODEL_CONFIG_DIGEST=1c69c7868c1206ea76c372df01e5baa2abcadcd2ca5b9f93b97d94fa6070aae0
  export CACHEBLEND_KV_CONFIG_DIGEST=131eb7ec025bc9a4fa1dabd220bb41b75c7d8f921e537fd8be505e91c6850742
  export CACHEBLEND_PLUGIN_COMMIT="$(git rev-parse HEAD)"

  export CACHEBLEND_RUN_DIR="$CACHEBLEND_BASE_DIR"
  if test -e "$CACHEBLEND_RUN_DIR"; then
    CACHEBLEND_RUN_DIR="${CACHEBLEND_BASE_DIR}-retry$(date +%Y%m%d-%H%M%S)"
    while test -e "$CACHEBLEND_RUN_DIR"; do
      sleep 1
      CACHEBLEND_RUN_DIR="${CACHEBLEND_BASE_DIR}-retry$(date +%Y%m%d-%H%M%S)"
    done
    export CACHEBLEND_RUN_DIR
  fi
  mkdir -p "$CACHEBLEND_RUN_DIR" "$TRITON_CACHE_DIR"
  printf '%s\n' "$CACHEBLEND_RUN_DIR" > "$CACHEBLEND_POINTER"
  git rev-parse HEAD > "$CACHEBLEND_RUN_DIR/git-head.txt"
  echo "RUN_DIR=$CACHEBLEND_RUN_DIR"

  trap stop_all_servers EXIT
  stop_existing_listener 8000
  stop_existing_listener 5556

  .venv/bin/python -c "import importlib.metadata as m,json,torch; observed={'torch':torch.__version__,'cuda':torch.version.cuda,'vllm':m.version('vllm'),'lmcache':m.version('lmcache'),'gpu':torch.cuda.get_device_name(0)}; expected={'torch':'2.10.0+cu128','cuda':'12.8','vllm':'0.19.1','lmcache':'0.4.3','gpu':'NVIDIA A100-SXM4-80GB'}; assert observed == expected, (observed,expected); print(json.dumps(observed,sort_keys=True))" \
    | tee "$CACHEBLEND_RUN_DIR/runtime.json"

  export CACHEBLEND_SIDECAR="$CACHEBLEND_RUN_DIR/sidecar.sqlite3"
  export CACHEBLEND_TRANSFER_CONFIG="$CACHEBLEND_RUN_DIR/transfer-config.json"
  export CACHEBLEND_TRANSFER_EVIDENCE="$CACHEBLEND_RUN_DIR/transfer-evidence.json"
  .venv/bin/python -c "import os; from cacheblend_gpt_oss.storage.sidecar import SidecarMode,open_sidecar_index; i=open_sidecar_index(os.environ['CACHEBLEND_SIDECAR'],SidecarMode.WORKER_READ_WRITE); i.close()"
  start_lmcache_server

  .venv/bin/python scripts/render_transfer_config.py \
    --lmcache-server-url tcp://127.0.0.1:5556 \
    --sidecar-path "$CACHEBLEND_SIDECAR" \
    --model-revision "$CACHEBLEND_MODEL_REVISION" \
    --tokenizer-revision "$CACHEBLEND_TOKENIZER_REVISION" \
    --model-config-digest "$CACHEBLEND_MODEL_CONFIG_DIGEST" \
    --kv-cache-config-digest "$CACHEBLEND_KV_CONFIG_DIGEST" \
    --adapter-revision "$CACHEBLEND_PLUGIN_COMMIT" \
    --transfer-evidence-path "$CACHEBLEND_TRANSFER_EVIDENCE" \
    --staging-token-capacity 131072 \
    --request-timeout-seconds 1800 \
    --allow-prefix-caching \
    --disable-kv-scatter \
    > "$CACHEBLEND_TRANSFER_CONFIG"
  export CACHEBLEND_KV_CONFIG_JSON="$(<"$CACHEBLEND_TRANSFER_CONFIG")"

  # A/B/A brackets the connector arm with two connector-free controls.
  run_arm baseline-a baseline no
  run_arm connector connector yes
  run_arm baseline-b baseline no

  set +e
  .venv/bin/python scripts/evaluate_connector_presence_equivalence.py \
    --baseline-a "$CACHEBLEND_RUN_DIR/baseline-a/responses.json" \
    --baseline-b "$CACHEBLEND_RUN_DIR/baseline-b/responses.json" \
    --connector "$CACHEBLEND_RUN_DIR/connector/responses.json" \
    --latency-ratio-limit 2.0 \
    --minimum-final-input-tokens 50000 \
    --output "$CACHEBLEND_RUN_DIR/connector-presence-verdict.json" \
    | tee "$CACHEBLEND_RUN_DIR/connector-presence-verdict.txt"
  local verdict_status=${PIPESTATUS[0]}
  set -e

  stop_all_servers
  trap - EXIT
  echo "VERDICT_STATUS=$verdict_status"
  echo "RUN_DIR=$CACHEBLEND_RUN_DIR"
  return "$verdict_status"
}

main "$@"
