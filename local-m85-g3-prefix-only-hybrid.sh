#!/usr/bin/env bash
# Isolation test: prefix cache + hybrid KV cache manager, NO CacheBlend.
# If this matches arm 2 speed (~149s for 30 searches), the hybrid manager
# is innocent and something in the connector init causes the divergence.
# If this diverges from arm 2, the hybrid manager IS the cause.

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
export CACHEBLEND_MAX_BATCHED_TOKENS=131072
export CACHEBLEND_RUN_BASE_DIR=/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m8.5-browsecomp-append-only-prefix-only-hybrid-20260817
export CACHEBLEND_RUN_POINTER=/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m8.5-browsecomp-append-only-prefix-only-hybrid-20260817.current

export CACHEBLEND_RUN_DIR="$CACHEBLEND_RUN_BASE_DIR"
if test -e "$CACHEBLEND_RUN_DIR"; then
  export CACHEBLEND_RUN_DIR="${CACHEBLEND_RUN_BASE_DIR}-retry$(date +%Y%m%d-%H%M%S)"
  while test -e "$CACHEBLEND_RUN_DIR"; do
    sleep 1
    export CACHEBLEND_RUN_DIR="${CACHEBLEND_RUN_BASE_DIR}-retry$(date +%Y%m%d-%H%M%S)"
  done
fi

mkdir -p "$CACHEBLEND_RUN_DIR" "$TRITON_CACHE_DIR"
printf '%s\n' "$CACHEBLEND_RUN_DIR" > "$CACHEBLEND_RUN_POINTER"
echo "RUN_DIR=$CACHEBLEND_RUN_DIR"

nvidia-smi \
  --query-gpu=name,memory.total,memory.free,driver_version \
  --format=csv,noheader \
  > "$CACHEBLEND_RUN_DIR/gpu-before.txt"

.venv/bin/python -c "import importlib.metadata as m,torch; print({'torch':torch.__version__,'torch_cuda':torch.version.cuda,'vllm':m.version('vllm'),'gpu':torch.cuda.get_device_name(0)})" \
  > "$CACHEBLEND_RUN_DIR/runtime.txt"

for PORT in 8000; do
  LISTENER_PID="$(ss -ltnp 2>/dev/null | awk -v p=":$PORT$" '$4 ~ p { x=$NF; sub(/^.*pid=/,"",x); sub(/,.*/,"",x); print x; exit }')"
  case "$LISTENER_PID" in
    ''|*[!0-9]*) ;;
    *) echo "STOPPING_PORT_${PORT}_PID=$LISTENER_PID"; kill -TERM "$LISTENER_PID" 2>/dev/null || true ;;
  esac
done

sleep 5

# Same as arm 2 but with --no-disable-hybrid-kv-cache-manager added.
# No LMCache, no CacheBlend connector, no staging backend.
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
  --enable-prefix-caching \
  --kv-cache-dtype auto \
  --attention-backend TRITON_ATTN \
  --no-disable-hybrid-kv-cache-manager \
  --enable-auto-tool-choice \
  --tool-call-parser openai \
  --generation-config vllm \
  --max-logprobs -1 \
  --port 8000 \
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
}

main
