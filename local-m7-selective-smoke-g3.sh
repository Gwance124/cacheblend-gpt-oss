#!/usr/bin/env bash

main() {
  cd /mnt/nvme3n1/mlee/cacheblend-gpt-oss || return 0

  # The launcher leaves the selective server running and exports the exact
  # run-directory/config variables into this sourced shell.
  source ./local-m7-selective-g3.sh
  if test "${CACHEBLEND_VLLM_READY:-no}" != yes; then
    echo "STOP_SELECTIVE_SERVER_NOT_READY"
    return 0
  fi

  export CACHEBLEND_SELECTIVE_ARTIFACT="$CACHEBLEND_RUN_DIR/cacheblend-selective.json"
  if test -e "$CACHEBLEND_SELECTIVE_ARTIFACT"; then
    echo "STOP_SELECTIVE_ARTIFACT_EXISTS=$CACHEBLEND_SELECTIVE_ARTIFACT"
    return 0
  fi

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
  echo "SELECTIVE_SMOKE_COMPLETE=$CACHEBLEND_SELECTIVE_ARTIFACT"
}

main
