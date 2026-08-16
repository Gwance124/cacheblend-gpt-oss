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
export CACHEBLEND_ENABLE_CUSTOM_BACKEND=1
export CACHEBLEND_MODEL_PATH=/mnt/nvme3n1/labuser/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee
export CACHEBLEND_RUN_DIR=/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m6-custom-backend-control-$(date +%Y%m%d-%H%M%S)

mkdir -p "$CACHEBLEND_RUN_DIR" "$TRITON_CACHE_DIR"
echo "RUN_DIR=$CACHEBLEND_RUN_DIR"
echo "SERVING_HEAD=$(git rev-parse HEAD)"

.venv/bin/python -m pip install --no-deps --no-build-isolation -e . \
  > "$CACHEBLEND_RUN_DIR/plugin-install.log" 2>&1
echo "PLUGIN_INSTALL_STATUS=$?"

if command -v fuser >/dev/null 2>&1; then
  fuser -k -TERM 8000/tcp >/dev/null 2>&1 || true
fi
sleep 3

.venv/bin/python -c "import importlib.metadata as m,torch; print({'torch':torch.__version__,'torch_cuda':torch.version.cuda,'vllm':m.version('vllm'),'lmcache':m.version('lmcache'),'gpu':torch.cuda.get_device_name(0)})" \
  > "$CACHEBLEND_RUN_DIR/runtime.txt" 2>&1

nohup env CACHEBLEND_ENABLE_CUSTOM_BACKEND=1 \
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
  > "$CACHEBLEND_RUN_DIR/vllm-server.log" 2>&1 < /dev/null &

export CACHEBLEND_VLLM_PID=$!
echo "VLLM_PID=$CACHEBLEND_VLLM_PID"
export CACHEBLEND_VLLM_READY=no

for n in $(seq 1 300); do
  export CACHEBLEND_HTTP_CODE="$(curl -sS -o "$CACHEBLEND_RUN_DIR/models.json" -w '%{http_code}' http://127.0.0.1:8000/v1/models 2>/dev/null || true)"
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
  .venv/bin/python -c "import importlib.metadata as m, json, os, torch; from pathlib import Path; p=Path(os.environ['CACHEBLEND_RUN_DIR']); print({'vllm':m.version('vllm'),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'backend':'CUSTOM'}); json.dump({'serving_head':os.popen('git rev-parse HEAD').read().strip(),'backend':'CUSTOM','model':'openai/gpt-oss-20b'},(p/'custom-backend-control.json').open('w'),indent=2)"
else
  tail -n 180 "$CACHEBLEND_RUN_DIR/vllm-server.log"
fi
}

main
