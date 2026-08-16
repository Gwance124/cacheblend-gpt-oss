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
  if ! git pull --ff-only; then
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
  export CACHEBLEND_ENABLE_CUSTOM_MODEL=1

  export CACHEBLEND_MODEL_PATH=/mnt/nvme3n1/labuser/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee
  export CACHEBLEND_MODEL_REVISION=6cee5e81ee83917806bbde320786a8fb61efebee
  export CACHEBLEND_TOKENIZER_REVISION=6cee5e81ee83917806bbde320786a8fb61efebee
  export CACHEBLEND_PLUGIN_COMMIT="$(git rev-parse HEAD)"
  export CACHEBLEND_MODEL_CONFIG_DIGEST=1c69c7868c1206ea76c372df01e5baa2abcadcd2ca5b9f93b97d94fa6070aae0
  export CACHEBLEND_KV_CONFIG_DIGEST=131eb7ec025bc9a4fa1dabd220bb41b75c7d8f921e537fd8be505e91c6850742
  export CACHEBLEND_STAGING_TOKENS=131072
  export CACHEBLEND_MAX_BATCHED_TOKENS=131072
  export CACHEBLEND_L1_SIZE_GB=8
  export CACHEBLEND_RUN_BASE_DIR=/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m7-selective-$(date +%Y%m%d-%H%M%S)
  export CACHEBLEND_RUN_DIR="$CACHEBLEND_RUN_BASE_DIR"
  while test -e "$CACHEBLEND_RUN_DIR"; do
    sleep 1
    export CACHEBLEND_RUN_DIR="${CACHEBLEND_RUN_BASE_DIR}-retry$(date +%Y%m%d-%H%M%S)"
  done
  mkdir -p "$CACHEBLEND_RUN_DIR" "$TRITON_CACHE_DIR"

  export CACHEBLEND_SIDECAR="$CACHEBLEND_RUN_DIR/sidecar.sqlite3"
  export CACHEBLEND_TRANSFER_CONFIG="$CACHEBLEND_RUN_DIR/transfer-config.json"
  echo "RUN_DIR=$CACHEBLEND_RUN_DIR"
  echo "SERVING_HEAD=$CACHEBLEND_PLUGIN_COMMIT"
  echo "SELECTIVE_MODE=transfer_selective"
  echo "SELECTIVE_CHECK_LAYER=1"
  echo "SELECTIVE_RECOMPUTE_RATIO=0.15"
  echo "SELECTIVE_SUFFIX_TOKENS=32"
  echo "SELECTIVE_EVIDENCE=row_work_metrics"

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
    > "$CACHEBLEND_RUN_DIR/runtime.txt" 2>&1
  echo "RUNTIME_IDENTITY=$CACHEBLEND_RUN_DIR/runtime.txt"

  nohup .venv/bin/python -m cacheblend_gpt_oss.storage.lmcache_server_v0_4_3 \
    --host 127.0.0.1 \
    --port 5556 \
    --chunk-size 256 \
    --hash-algorithm blake3 \
    --l1-size-gb "$CACHEBLEND_L1_SIZE_GB" \
    --l1-init-size-gb "$CACHEBLEND_L1_SIZE_GB" \
    --eviction-policy LRU \
    --max-workers 1 \
    > "$CACHEBLEND_RUN_DIR/lmcache-server.log" 2>&1 < /dev/null &
  export CACHEBLEND_LMCACHE_PID=$!
  echo "LMCACHE_PID=$CACHEBLEND_LMCACHE_PID"

  export CACHEBLEND_LMCACHE_READY=no
  for n in $(seq 1 90); do
    if ss -ltn 2>/dev/null | awk '$4 ~ /:5556$/ { found=1 } END { exit found ? 0 : 1 }'; then
      export CACHEBLEND_LMCACHE_READY=yes
      break
    fi
    sleep 1
  done
  echo "LMCACHE_READY=$CACHEBLEND_LMCACHE_READY"

  if test "$CACHEBLEND_LMCACHE_READY" != yes; then
    tail -n 160 "$CACHEBLEND_RUN_DIR/lmcache-server.log"
    return 0
  fi

  .venv/bin/python -c "import os; from cacheblend_gpt_oss.storage.sidecar import SidecarMode,open_sidecar_index; i=open_sidecar_index(os.environ['CACHEBLEND_SIDECAR'],SidecarMode.WORKER_READ_WRITE); i.close()"

  .venv/bin/python scripts/render_transfer_config.py \
    --mode transfer_selective \
    --lmcache-server-url tcp://127.0.0.1:5556 \
    --sidecar-path "$CACHEBLEND_SIDECAR" \
    --model-revision "$CACHEBLEND_MODEL_REVISION" \
    --tokenizer-revision "$CACHEBLEND_TOKENIZER_REVISION" \
    --model-config-digest "$CACHEBLEND_MODEL_CONFIG_DIGEST" \
    --kv-cache-config-digest "$CACHEBLEND_KV_CONFIG_DIGEST" \
    --adapter-revision "$CACHEBLEND_PLUGIN_COMMIT" \
    --staging-token-capacity "$CACHEBLEND_STAGING_TOKENS" \
    --request-timeout-seconds 300 \
    --check-layer 1 \
    --recompute-ratio 0.15 \
    --suffix-tokens 32 \
    > "$CACHEBLEND_TRANSFER_CONFIG"
  echo "TRANSFER_CONFIG=$CACHEBLEND_TRANSFER_CONFIG"
  export CACHEBLEND_KV_CONFIG_JSON="$(<"$CACHEBLEND_TRANSFER_CONFIG")"

  nohup .venv/bin/vllm serve \
    "$CACHEBLEND_MODEL_PATH" \
    --served-model-name openai/gpt-oss-20b \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.50 \
    --max-num-seqs 1 \
    --max-num-batched-tokens "$CACHEBLEND_MAX_BATCHED_TOKENS" \
    --long-prefill-token-threshold 0 \
    --no-async-scheduling \
    --enforce-eager \
    --no-enable-prefix-caching \
    --kv-cache-dtype auto \
    --attention-backend CUSTOM \
    --no-disable-hybrid-kv-cache-manager \
    --enable-auto-tool-choice \
    --tool-call-parser openai \
    --generation-config vllm \
    --max-logprobs -1 \
    --port 8000 \
    --kv-transfer-config "$CACHEBLEND_KV_CONFIG_JSON" \
    > "$CACHEBLEND_RUN_DIR/vllm-server.log" 2>&1 < /dev/null &
  export CACHEBLEND_VLLM_PID=$!
  echo "VLLM_PID=$CACHEBLEND_VLLM_PID"

  export CACHEBLEND_VLLM_READY=no
  for n in $(seq 1 300); do
    CACHEBLEND_HTTP_CODE="$(curl -sS -o "$CACHEBLEND_RUN_DIR/models.json" -w '%{http_code}' http://127.0.0.1:8000/v1/models 2>/dev/null || true)"
    if test "$CACHEBLEND_HTTP_CODE" = 200; then
      export CACHEBLEND_VLLM_READY=yes
      break
    fi
    if ! kill -0 "$CACHEBLEND_VLLM_PID" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  echo "VLLM_READY=$CACHEBLEND_VLLM_READY"
  if test "$CACHEBLEND_VLLM_READY" = yes; then
    .venv/bin/python -c "import json,os; json.dump({'serving_head':os.environ['CACHEBLEND_PLUGIN_COMMIT'],'mode':'transfer_selective','check_layer':1,'recompute_ratio':0.15,'suffix_tokens':32},open(os.path.join(os.environ['CACHEBLEND_RUN_DIR'],'selective-runtime.json'),'w'),indent=2); print('SELECTIVE_SERVER_READY')"
  else
    tail -n 220 "$CACHEBLEND_RUN_DIR/vllm-server.log"
  fi
}

main
