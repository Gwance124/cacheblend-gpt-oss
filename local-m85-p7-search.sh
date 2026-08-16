#!/usr/bin/env bash

main() {
cd /mnt/nvme2/mlee/rag-system || return 0
source .env

export CACHEBLEND_P7_RUN_DIR=/mnt/nvme2/mlee/cacheblend-gpt-oss-artifacts/browsecomp-append-only-cacheblend-3b7bb29-20260815

if curl -fsS http://127.0.0.1:8012/health > /tmp/rag-system-search-health.json 2>/dev/null; then
  echo "SEARCH_ALREADY_READY"
else
  mkdir -p "$CACHEBLEND_P7_RUN_DIR"

  nohup env PYTHONPATH=src python scripts/serve_standard_search.py \
    --corpus-repo "$RAG_CORPUS_REPO" \
    --tokenizer-path "$TOKENIZER_DIR" \
    --encoder-url "$ENCODER_URL" \
    --qdrant-manifest "$RAG_QDRANT_MANIFEST" \
    --qdrant-url "$QDRANT_URL" \
    --trace-jsonl "$CACHEBLEND_P7_RUN_DIR/search.trace.jsonl" \
    --otlp-endpoint "$PHOENIX_OTLP_ENDPOINT" \
    --otlp-header x-project-name=rag-slim-trace \
    --service-name "$SEARCH_SERVICE_NAME" \
    --host 127.0.0.1 \
    --port 8012 \
    > "$CACHEBLEND_P7_RUN_DIR/search-service.log" 2>&1 < /dev/null &

  echo "SEARCH_PID=$!"
  echo "SEARCH_LOG=$CACHEBLEND_P7_RUN_DIR/search-service.log"
fi
}

main
