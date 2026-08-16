#!/usr/bin/env bash

main() {
cd /mnt/nvme3n1/mlee/rag-system || return 0
source .env

export CUDA_VISIBLE_DEVICES=1
export P7_HOST=192.168.3.40
export G3_HOST=192.168.3.4

if curl -fsS http://127.0.0.1:8011/health > /tmp/rag-system-encoder-health.json 2>/dev/null; then
  echo "ENCODER_ALREADY_READY"
else
  mkdir -p /mnt/nvme3n1/mlee/rag-system/results/traces/rag-slim-trace/g3-encoder

  nohup env PYTHONPATH=src python scripts/serve_query_encoder.py \
    --model-path /mnt/nvme3n1/labuser/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-8B/snapshots/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af \
    --host 0.0.0.0 \
    --port 8011 \
    --max-length 512 \
    --trace-jsonl /mnt/nvme3n1/mlee/rag-system/results/traces/rag-slim-trace/g3-encoder/encoder.trace.jsonl \
    --otlp-endpoint http://192.168.3.40:6006/v1/traces \
    --otlp-header x-project-name=rag-slim-trace \
    --service-name rag-system-query-encoder \
    > /mnt/nvme3n1/mlee/rag-system/results/traces/rag-slim-trace/g3-encoder/encoder.log 2>&1 < /dev/null &

  echo "ENCODER_PID=$!"
  echo "ENCODER_LOG=/mnt/nvme3n1/mlee/rag-system/results/traces/rag-slim-trace/g3-encoder/encoder.log"
fi
}

main
