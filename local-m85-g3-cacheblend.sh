#!/usr/bin/env bash

main() {
cd /mnt/nvme3n1/mlee/cacheblend-gpt-oss || return 0

git fetch origin
git switch cacheblend-scatter-diagnostic-and-checklayer 2>/dev/null || git switch -c cacheblend-scatter-diagnostic-and-checklayer --track origin/cacheblend-scatter-diagnostic-and-checklayer
git pull --ff-only

export TIKTOKEN_ENCODINGS_BASE=/mnt/nvme3n1/labuser/.cache/tiktoken/encodings
export TIKTOKEN_RS_CACHE_DIR=/mnt/nvme3n1/labuser/.cache/tiktoken
export TIKTOKEN_CACHE_DIR=/mnt/nvme3n1/labuser/.cache/tiktoken
export TRITON_CACHE_DIR=/mnt/nvme3n1/mlee/cacheblend-gpt-oss/.triton-cache
export VLLM_USE_V2_MODEL_RUNNER=0
export CUDA_VISIBLE_DEVICES=0

export CACHEBLEND_MODEL_PATH=/mnt/nvme3n1/labuser/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee
export CACHEBLEND_MODEL_REVISION=6cee5e81ee83917806bbde320786a8fb61efebee
export CACHEBLEND_TOKENIZER_REVISION=6cee5e81ee83917806bbde320786a8fb61efebee
export CACHEBLEND_PLUGIN_COMMIT="$(git rev-parse HEAD)"
export CACHEBLEND_MODEL_CONFIG_DIGEST=1c69c7868c1206ea76c372df01e5baa2abcadcd2ca5b9f93b97d94fa6070aae0
export CACHEBLEND_KV_CONFIG_DIGEST=131eb7ec025bc9a4fa1dabd220bb41b75c7d8f921e537fd8be505e91c6850742
export CACHEBLEND_STAGING_TOKENS=131072
export CACHEBLEND_RUN_DIR=/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m8.5-3b7bb29-browsecomp-append-only-20260815
export CACHEBLEND_SIDECAR="$CACHEBLEND_RUN_DIR/sidecar.sqlite3"
export CACHEBLEND_TRANSFER_CONFIG="$CACHEBLEND_RUN_DIR/transfer-config.json"
export CACHEBLEND_TRANSFER_EVIDENCE="$CACHEBLEND_RUN_DIR/transfer-evidence.json"

if test "$(git rev-parse --short HEAD)" = "3b7bb29"; then
  if test ! -e "$CACHEBLEND_RUN_DIR"; then
    mkdir -p "$CACHEBLEND_RUN_DIR" "$TRITON_CACHE_DIR"

    nvidia-smi \
      --query-gpu=name,memory.total,memory.free,driver_version \
      --format=csv,noheader \
      > "$CACHEBLEND_RUN_DIR/gpu-before.txt"

    .venv/bin/python -c "import importlib.metadata as m,torch; print({'torch':torch.__version__,'torch_cuda':torch.version.cuda,'vllm':m.version('vllm'),'lmcache':m.version('lmcache'),'gpu':torch.cuda.get_device_name(0)})" \
      > "$CACHEBLEND_RUN_DIR/runtime.txt"

    for PORT in 8000 5556; do
      LISTENER_PID="$(ss -ltnp 2>/dev/null | awk -v p=":$PORT$" '$4 ~ p { x=$NF; sub(/^.*pid=/,"",x); sub(/,.*/,"",x); print x; exit }')"
      case "$LISTENER_PID" in
        ''|*[!0-9]*) ;;
        *) echo "STOPPING_PORT_${PORT}_PID=$LISTENER_PID"; kill -TERM "$LISTENER_PID" 2>/dev/null || true ;;
      esac
    done

    sleep 5

    nohup .venv/bin/python -m cacheblend_gpt_oss.storage.lmcache_server_v0_4_3 \
      --host 127.0.0.1 \
      --port 5556 \
      --chunk-size 256 \
      --hash-algorithm blake3 \
      --l1-size-gb 16 \
      --l1-init-size-gb 16 \
      --eviction-policy LRU \
      --max-workers 1 \
      > "$CACHEBLEND_RUN_DIR/lmcache-server.log" 2>&1 < /dev/null &

    export CACHEBLEND_LMCACHE_PID=$!
    echo "$CACHEBLEND_LMCACHE_PID" > "$CACHEBLEND_RUN_DIR/lmcache-server.pid"

    CACHEBLEND_LMCACHE_READY=no
    for n in $(seq 1 90); do
      if ss -ltn 2>/dev/null | awk '$4 ~ /:5556$/ { found=1 } END { exit found ? 0 : 1 }'; then
        CACHEBLEND_LMCACHE_READY=yes
        break
      fi
      sleep 1
    done

    echo "LMCACHE_READY=$CACHEBLEND_LMCACHE_READY"

    if test "$CACHEBLEND_LMCACHE_READY" = yes; then
      .venv/bin/python -c "import os; from cacheblend_gpt_oss.storage.sidecar import SidecarMode,open_sidecar_index; i=open_sidecar_index(os.environ['CACHEBLEND_SIDECAR'],SidecarMode.WORKER_READ_WRITE); i.close()"

      .venv/bin/python scripts/render_transfer_config.py \
        --lmcache-server-url tcp://127.0.0.1:5556 \
        --sidecar-path "$CACHEBLEND_SIDECAR" \
        --model-revision "$CACHEBLEND_MODEL_REVISION" \
        --tokenizer-revision "$CACHEBLEND_TOKENIZER_REVISION" \
        --model-config-digest "$CACHEBLEND_MODEL_CONFIG_DIGEST" \
        --kv-cache-config-digest "$CACHEBLEND_KV_CONFIG_DIGEST" \
        --adapter-revision "$CACHEBLEND_PLUGIN_COMMIT" \
        --transfer-evidence-path "$CACHEBLEND_TRANSFER_EVIDENCE" \
        --staging-token-capacity "$CACHEBLEND_STAGING_TOKENS" \
        --request-timeout-seconds 300 \
        > "$CACHEBLEND_TRANSFER_CONFIG"

      export CACHEBLEND_KV_CONFIG_JSON="$(<"$CACHEBLEND_TRANSFER_CONFIG")"

      .venv/bin/python -c "import json,os; d={'model_id':'openai/gpt-oss-20b','model_revision':os.environ['CACHEBLEND_MODEL_REVISION'],'tokenizer_revision':os.environ['CACHEBLEND_TOKENIZER_REVISION'],'plugin_commit':os.environ['CACHEBLEND_PLUGIN_COMMIT'],'model_config_digest':os.environ['CACHEBLEND_MODEL_CONFIG_DIGEST'],'kv_cache_config_digest':os.environ['CACHEBLEND_KV_CONFIG_DIGEST'],'vllm_version':'0.19.1','lmcache_version':'0.4.3','torch_version':'2.10.0+cu128','cuda_runtime':'12.8','gpu_name':'NVIDIA A100-SXM4-80GB','dtype':'torch.bfloat16'}; json.dump(d,open(os.path.join(os.environ['CACHEBLEND_RUN_DIR'],'runtime-identity.json'),'w'),indent=2); print('RUNTIME_IDENTITY_OK')"

      nohup .venv/bin/vllm serve \
        "$CACHEBLEND_MODEL_PATH" \
        --served-model-name openai/gpt-oss-20b \
        --tensor-parallel-size 1 \
        --dtype bfloat16 \
        --max-model-len 131072 \
        --gpu-memory-utilization 0.80 \
        --max-num-seqs 1 \
        --max-num-batched-tokens 131072 \
        --long-prefill-token-threshold 0 \
        --no-async-scheduling \
        --enforce-eager \
        --no-enable-prefix-caching \
        --kv-cache-dtype auto \
        --attention-backend TRITON_ATTN \
        --no-disable-hybrid-kv-cache-manager \
        --enable-auto-tool-choice \
        --tool-call-parser openai \
        --generation-config vllm \
        --max-logprobs -1 \
        --port 8000 \
        --kv-transfer-config "$CACHEBLEND_KV_CONFIG_JSON" \
        > "$CACHEBLEND_RUN_DIR/vllm-server.log" 2>&1 < /dev/null &

      export CACHEBLEND_VLLM_PID=$!
      echo "$CACHEBLEND_VLLM_PID" > "$CACHEBLEND_RUN_DIR/vllm-server.pid"

      CACHEBLEND_VLLM_READY=no
      for n in $(seq 1 300); do
        CACHEBLEND_HTTP_CODE="$(curl -sS -o "$CACHEBLEND_RUN_DIR/models.json" -w '%{http_code}' http://127.0.0.1:8000/v1/models 2>/dev/null || true)"
        if test "$CACHEBLEND_HTTP_CODE" = 200; then
          CACHEBLEND_VLLM_READY=yes
          break
        fi
        if ! kill -0 "$CACHEBLEND_VLLM_PID" 2>/dev/null; then
          break
        fi
        sleep 1
      done

      echo "VLLM_READY=$CACHEBLEND_VLLM_READY"
      if test "$CACHEBLEND_VLLM_READY" = yes; then
        nvidia-smi \
          --query-gpu=name,memory.total,memory.used,memory.free \
          --format=csv,noheader \
          > "$CACHEBLEND_RUN_DIR/gpu-after-start.txt"
      else
        tail -n 160 "$CACHEBLEND_RUN_DIR/vllm-server.log"
      fi
    else
      tail -n 120 "$CACHEBLEND_RUN_DIR/lmcache-server.log"
    fi
  else
    echo "STOP_RUN_DIR_ALREADY_EXISTS=$CACHEBLEND_RUN_DIR"
  fi
else
  echo "STOP_WRONG_COMMIT=$(git rev-parse --short HEAD)"
fi
}

main
