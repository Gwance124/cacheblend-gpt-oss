#!/usr/bin/env bash

main() {
  cd /mnt/nvme3n1/mlee/cacheblend-gpt-oss || return 0

  export CACHEBLEND_REQUIRED_BRANCH=cacheblend-scatter-diagnostic-and-checklayer
  git fetch origin
  if ! git switch "$CACHEBLEND_REQUIRED_BRANCH" 2>/dev/null &&
    ! git switch -c "$CACHEBLEND_REQUIRED_BRANCH" \
      --track "origin/$CACHEBLEND_REQUIRED_BRANCH"; then
    echo "STOP_BRANCH_SWITCH_FAILED=$(git branch --show-current)"
    return 0
  fi
  if ! git pull --ff-only origin "$CACHEBLEND_REQUIRED_BRANCH"; then
    echo "STOP_GIT_SYNC_FAILED=$(git rev-parse HEAD)"
    return 0
  fi

  export TIKTOKEN_ENCODINGS_BASE=/mnt/nvme3n1/labuser/.cache/tiktoken/encodings
  export TIKTOKEN_RS_CACHE_DIR=/mnt/nvme3n1/labuser/.cache/tiktoken
  export TIKTOKEN_CACHE_DIR=/mnt/nvme3n1/labuser/.cache/tiktoken
  export TRITON_CACHE_DIR=/mnt/nvme3n1/mlee/cacheblend-gpt-oss/.triton-cache
  export VLLM_USE_V2_MODEL_RUNNER=0
  export CUDA_VISIBLE_DEVICES=0
  export CACHEBLEND_ENABLE_CUSTOM_BACKEND=1

  export CACHEBLEND_MODEL_PATH=/mnt/nvme3n1/labuser/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee
  export CACHEBLEND_MODEL_REVISION=6cee5e81ee83917806bbde320786a8fb61efebee
  export CACHEBLEND_TOKENIZER_REVISION=6cee5e81ee83917806bbde320786a8fb61efebee
  export CACHEBLEND_PLUGIN_COMMIT="$(git rev-parse HEAD)"
  export CACHEBLEND_MODEL_CONFIG_DIGEST=1c69c7868c1206ea76c372df01e5baa2abcadcd2ca5b9f93b97d94fa6070aae0
  export CACHEBLEND_KV_CONFIG_DIGEST=131eb7ec025bc9a4fa1dabd220bb41b75c7d8f921e537fd8be505e91c6850742

  export CACHEBLEND_RUN_BASE_DIR=/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m7-selective-correctness-$(date +%Y%m%d-%H%M%S)
  export CACHEBLEND_RUN_DIR="$CACHEBLEND_RUN_BASE_DIR"
  while test -e "$CACHEBLEND_RUN_DIR"; do
    sleep 1
    export CACHEBLEND_RUN_DIR="${CACHEBLEND_RUN_BASE_DIR}-retry$(date +%Y%m%d-%H%M%S)"
  done
  export CACHEBLEND_REFERENCE_DIR="$CACHEBLEND_RUN_DIR/reference"
  mkdir -p "$CACHEBLEND_REFERENCE_DIR" "$TRITON_CACHE_DIR"

  echo "RUN_DIR=$CACHEBLEND_RUN_DIR"
  echo "SERVING_HEAD=$CACHEBLEND_PLUGIN_COMMIT"
  echo "REFERENCE_DIR=$CACHEBLEND_REFERENCE_DIR"

  .venv/bin/python -m pip install --no-deps --no-build-isolation -e . \
    > "$CACHEBLEND_RUN_DIR/plugin-install.log" 2>&1
  echo "PLUGIN_INSTALL_STATUS=$?"

  for PORT in 8000 5556; do
    LISTENER_PID="$(ss -ltnp 2>/dev/null | awk -v p=":$PORT$" '$4 ~ p { x=$NF; sub(/^.*pid=/,"",x); sub(/,.*/,"",x); print x; exit }')"
    case "$LISTENER_PID" in
      ''|*[!0-9]*) ;;
      *) echo "STOPPING_PORT_${PORT}_PID=$LISTENER_PID"; kill -TERM "$LISTENER_PID" 2>/dev/null || true ;;
    esac
  done
  sleep 5

  .venv/bin/python -c "import importlib.metadata as m,torch; print({'torch':torch.__version__,'torch_cuda':torch.version.cuda,'vllm':m.version('vllm'),'lmcache':m.version('lmcache'),'gpu':torch.cuda.get_device_name(0)})" \
    > "$CACHEBLEND_REFERENCE_DIR/runtime.txt" 2>&1

  nohup env \
    CACHEBLEND_ENABLE_CUSTOM_BACKEND=1 \
    TIKTOKEN_ENCODINGS_BASE="$TIKTOKEN_ENCODINGS_BASE" \
    TIKTOKEN_RS_CACHE_DIR="$TIKTOKEN_RS_CACHE_DIR" \
    TIKTOKEN_CACHE_DIR="$TIKTOKEN_CACHE_DIR" \
    TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    VLLM_USE_V2_MODEL_RUNNER=0 \
    CUDA_VISIBLE_DEVICES=0 \
    .venv/bin/vllm serve \
    "$CACHEBLEND_MODEL_PATH" \
    --served-model-name openai/gpt-oss-20b \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.50 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 131072 \
    --long-prefill-token-threshold 0 \
    --no-async-scheduling \
    --enforce-eager \
    --no-enable-prefix-caching \
    --kv-cache-dtype auto \
    --attention-backend CUSTOM \
    --generation-config vllm \
    --max-logprobs -1 \
    --port 8000 \
    > "$CACHEBLEND_REFERENCE_DIR/full-prefill-server.log" 2>&1 < /dev/null &
  export CACHEBLEND_REFERENCE_VLLM_PID=$!
  echo "REFERENCE_VLLM_PID=$CACHEBLEND_REFERENCE_VLLM_PID"

  export CACHEBLEND_REFERENCE_READY=no
  for n in $(seq 1 300); do
    CACHEBLEND_HTTP_CODE="$(curl -sS -o "$CACHEBLEND_REFERENCE_DIR/models.json" -w '%{http_code}' http://127.0.0.1:8000/v1/models 2>/dev/null || true)"
    if test "$CACHEBLEND_HTTP_CODE" = 200; then
      export CACHEBLEND_REFERENCE_READY=yes
      break
    fi
    if ! kill -0 "$CACHEBLEND_REFERENCE_VLLM_PID" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  echo "REFERENCE_VLLM_READY=$CACHEBLEND_REFERENCE_READY"
  if test "$CACHEBLEND_REFERENCE_READY" != yes; then
    tail -n 220 "$CACHEBLEND_REFERENCE_DIR/full-prefill-server.log"
    return 0
  fi

  for LABEL in reference repeat; do
    .venv/bin/python scripts/capture_moved_document.py \
      --mode full_prefill \
      --warm-source-before-target \
      --model-revision "$CACHEBLEND_MODEL_REVISION" \
      --tokenizer-revision "$CACHEBLEND_TOKENIZER_REVISION" \
      --plugin-commit "$CACHEBLEND_PLUGIN_COMMIT" \
      --model-config-digest "$CACHEBLEND_MODEL_CONFIG_DIGEST" \
      --kv-cache-config-digest "$CACHEBLEND_KV_CONFIG_DIGEST" \
      --output "$CACHEBLEND_REFERENCE_DIR/full-prefill-${LABEL}.json" \
      | tee "$CACHEBLEND_REFERENCE_DIR/full-prefill-${LABEL}.txt"
    export CACHEBLEND_CAPTURE_STATUS=${PIPESTATUS[0]}
    echo "FULL_PREFILL_${LABEL^^}_STATUS=$CACHEBLEND_CAPTURE_STATUS"
    if test "$CACHEBLEND_CAPTURE_STATUS" != 0; then
      return 0
    fi
  done

  .venv/bin/python scripts/freeze_correctness_tolerance.py \
    --reference "$CACHEBLEND_REFERENCE_DIR/full-prefill-reference.json" \
    --repeat "$CACHEBLEND_REFERENCE_DIR/full-prefill-repeat.json" \
    --max-abs-floor 0.08 \
    --mean-abs-floor 0.014 \
    --output "$CACHEBLEND_REFERENCE_DIR/selective-tolerance.json" \
    | tee "$CACHEBLEND_REFERENCE_DIR/selective-tolerance.txt"
  export CACHEBLEND_TOLERANCE_STATUS=${PIPESTATUS[0]}
  echo "SELECTIVE_TOLERANCE_STATUS=$CACHEBLEND_TOLERANCE_STATUS"
  if test "$CACHEBLEND_TOLERANCE_STATUS" != 0; then
    return 0
  fi

  kill -TERM "$CACHEBLEND_REFERENCE_VLLM_PID" 2>/dev/null || true
  sleep 5

  # This launcher creates the selective run directory, starts LMCache and the
  # CUSTOM/model wrapper server, and leaves the server running for capture.
  source ./local-m7-selective-g3.sh
  if test "${CACHEBLEND_VLLM_READY:-no}" != yes; then
    echo "STOP_SELECTIVE_SERVER_NOT_READY"
    return 0
  fi

  export CACHEBLEND_SELECTIVE_ARTIFACT="$CACHEBLEND_RUN_DIR/cacheblend-selective.json"
  .venv/bin/python scripts/capture_moved_document.py \
    --mode cacheblend_selective \
    --model-revision "$CACHEBLEND_MODEL_REVISION" \
    --tokenizer-revision "$CACHEBLEND_TOKENIZER_REVISION" \
    --plugin-commit "$CACHEBLEND_PLUGIN_COMMIT" \
    --model-config-digest "$CACHEBLEND_MODEL_CONFIG_DIGEST" \
    --kv-cache-config-digest "$CACHEBLEND_KV_CONFIG_DIGEST" \
    --output "$CACHEBLEND_SELECTIVE_ARTIFACT" \
    | tee "$CACHEBLEND_RUN_DIR/selective-capture.txt"
  export CACHEBLEND_SELECTIVE_CAPTURE_STATUS=${PIPESTATUS[0]}
  echo "SELECTIVE_CAPTURE_STATUS=$CACHEBLEND_SELECTIVE_CAPTURE_STATUS"
  if test "$CACHEBLEND_SELECTIVE_CAPTURE_STATUS" != 0; then
    return 0
  fi

  curl -sS http://127.0.0.1:8000/metrics \
    > "$CACHEBLEND_RUN_DIR/selective-metrics.prom"
  .venv/bin/python -c "import json,os; from pathlib import Path; from cacheblend_gpt_oss.correctness import parse_connector_counter_snapshot,parse_selective_work_counter_snapshot; t=Path(os.path.join(os.environ['CACHEBLEND_RUN_DIR'],'selective-metrics.prom')).read_text(); print(json.dumps({'connector':parse_connector_counter_snapshot(t),'selective_work':parse_selective_work_counter_snapshot(t)},indent=2,sort_keys=True))" \
    | tee "$CACHEBLEND_RUN_DIR/selective-counters.json"

  .venv/bin/python scripts/evaluate_cacheblend_correctness.py \
    --reference "$CACHEBLEND_REFERENCE_DIR/full-prefill-reference.json" \
    --cacheblend "$CACHEBLEND_SELECTIVE_ARTIFACT" \
    --tolerance "$CACHEBLEND_REFERENCE_DIR/selective-tolerance.json" \
    --mode cacheblend_selective \
    --output "$CACHEBLEND_RUN_DIR/selective-verdict.json" \
    | tee "$CACHEBLEND_RUN_DIR/selective-verdict.txt"
  export CACHEBLEND_SELECTIVE_EVAL_STATUS=${PIPESTATUS[0]}
  echo "SELECTIVE_EVALUATION_STATUS=$CACHEBLEND_SELECTIVE_EVAL_STATUS"
  echo "REFERENCE_ARTIFACT=$CACHEBLEND_REFERENCE_DIR/full-prefill-reference.json"
  echo "REPEAT_ARTIFACT=$CACHEBLEND_REFERENCE_DIR/full-prefill-repeat.json"
  echo "SELECTIVE_ARTIFACT=$CACHEBLEND_SELECTIVE_ARTIFACT"
  echo "SELECTIVE_VERDICT=$CACHEBLEND_RUN_DIR/selective-verdict.json"
}

main
