#!/usr/bin/env bash
# Deterministic connector-free A/B/A for omitted versus explicit-false HMA.
# Run on solab-g3. Each arm gets a fresh vLLM process and identical requests.

set -euo pipefail

readonly CACHEBLEND_REPO=/mnt/nvme3n1/mlee/cacheblend-gpt-oss
readonly CACHEBLEND_BRANCH=cacheblend-scatter-diagnostic-and-checklayer
readonly CACHEBLEND_POINTER=/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m8.5-hybrid-flag-equivalence-20260818.current
readonly CACHEBLEND_BASE_DIR=/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m8.5-hybrid-flag-equivalence-20260818

CACHEBLEND_CURRENT_PID=""

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
    echo "SERVER_DID_NOT_STOP_PID=$CACHEBLEND_CURRENT_PID" >&2
    return 1
  fi
  wait "$CACHEBLEND_CURRENT_PID" 2>/dev/null || true
  CACHEBLEND_CURRENT_PID=""
}

stop_existing_listener() {
  local listener_pid
  listener_pid="$(
    ss -ltnp 2>/dev/null \
      | awk '$4 ~ /:8000$/ { x=$NF; sub(/^.*pid=/,"",x); sub(/,.*/,"",x); print x; exit }'
  )"
  case "$listener_pid" in
    '') return 0 ;;
    *[!0-9]*)
      echo "INVALID_PORT_8000_LISTENER_PID=$listener_pid" >&2
      return 1
      ;;
  esac
  echo "STOPPING_EXISTING_PORT_8000_PID=$listener_pid"
  kill -TERM "$listener_pid"
  for _ in $(seq 1 60); do
    if ! kill -0 "$listener_pid" 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  echo "EXISTING_PORT_8000_LISTENER_DID_NOT_STOP=$listener_pid" >&2
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

run_arm() {
  local label="$1"
  local mode="$2"
  local explicit_false="$3"
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
  )
  if test "$explicit_false" = yes; then
    command+=(--no-disable-hybrid-kv-cache-manager)
  fi
  command+=(
    --enable-auto-tool-choice
    --tool-call-parser openai
    --generation-config vllm
    --max-logprobs -1
    --port 8000
  )

  write_launch_command "$arm_dir/launch-command.sh" "${command[@]}"
  printf '%s\n' "$mode" > "$arm_dir/raw-hybrid-flag-mode.txt"
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

  curl -fsS http://127.0.0.1:8000/metrics > "$arm_dir/metrics-before.prom"

  .venv/bin/python scripts/capture_moved_document.py \
    --mode full_prefill \
    --case moved_document \
    --base-url http://127.0.0.1:8000 \
    --model-revision "$CACHEBLEND_MODEL_REVISION" \
    --tokenizer-revision "$CACHEBLEND_TOKENIZER_REVISION" \
    --plugin-commit "$CACHEBLEND_PLUGIN_COMMIT" \
    --model-config-digest "$CACHEBLEND_MODEL_CONFIG_DIGEST" \
    --kv-cache-config-digest "$CACHEBLEND_KV_CONFIG_DIGEST" \
    --output "$arm_dir/full-vocabulary-logits.json" \
    | tee "$arm_dir/full-vocabulary-logits-summary.txt"

  .venv/bin/python scripts/capture_hybrid_flag_responses.py \
    --mode "$mode" \
    --base-url http://127.0.0.1:8000 \
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
  git fetch origin
  git switch "$CACHEBLEND_BRANCH" 2>/dev/null \
    || git switch -c "$CACHEBLEND_BRANCH" --track "origin/$CACHEBLEND_BRANCH"
  git pull --ff-only
  if test -n "$(git status --porcelain)"; then
    echo "STOP_DIRTY_WORKTREE" >&2
    git status --short >&2
    return 1
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
  echo "RUN_DIR=$CACHEBLEND_RUN_DIR"

  trap stop_current_server EXIT
  stop_existing_listener

  .venv/bin/python -c "import importlib.metadata as m,json,torch; observed={'torch':torch.__version__,'cuda':torch.version.cuda,'vllm':m.version('vllm'),'lmcache':m.version('lmcache'),'gpu':torch.cuda.get_device_name(0)}; expected={'torch':'2.10.0+cu128','cuda':'12.8','vllm':'0.19.1','lmcache':'0.4.3','gpu':'NVIDIA A100-SXM4-80GB'}; assert observed == expected, (observed,expected); print(json.dumps(observed,sort_keys=True))" \
    | tee "$CACHEBLEND_RUN_DIR/runtime.json"

  .venv/bin/python scripts/capture_hybrid_flag_resolution.py \
    --model-path "$CACHEBLEND_MODEL_PATH" \
    --output "$CACHEBLEND_RUN_DIR/hybrid-flag-resolution.json" \
    | tee "$CACHEBLEND_RUN_DIR/hybrid-flag-resolution.txt"

  # A/B/A brackets the explicit-false candidate with two omitted-flag controls.
  run_arm implicit-a implicit no
  run_arm explicit-false explicit_false yes
  run_arm implicit-b implicit no

  set +e
  .venv/bin/python scripts/evaluate_hybrid_flag_equivalence.py \
    --implicit-a-responses "$CACHEBLEND_RUN_DIR/implicit-a/responses.json" \
    --implicit-b-responses "$CACHEBLEND_RUN_DIR/implicit-b/responses.json" \
    --explicit-false-responses "$CACHEBLEND_RUN_DIR/explicit-false/responses.json" \
    --implicit-a-logits "$CACHEBLEND_RUN_DIR/implicit-a/full-vocabulary-logits.json" \
    --implicit-b-logits "$CACHEBLEND_RUN_DIR/implicit-b/full-vocabulary-logits.json" \
    --explicit-false-logits "$CACHEBLEND_RUN_DIR/explicit-false/full-vocabulary-logits.json" \
    --resolution "$CACHEBLEND_RUN_DIR/hybrid-flag-resolution.json" \
    --latency-ratio-limit 2.0 \
    --output "$CACHEBLEND_RUN_DIR/hybrid-flag-equivalence-verdict.json" \
    | tee "$CACHEBLEND_RUN_DIR/hybrid-flag-equivalence-verdict.txt"
  local verdict_status=${PIPESTATUS[0]}
  set -e

  trap - EXIT
  echo "VERDICT_STATUS=$verdict_status"
  echo "RUN_DIR=$CACHEBLEND_RUN_DIR"
  return "$verdict_status"
}

main "$@"
