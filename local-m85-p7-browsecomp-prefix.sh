#!/usr/bin/env bash

main() {
  cd /mnt/nvme2/mlee/rag-system || return 0
  source .env
  if test -n "${CACHEBLEND_CONTEXT_BUDGET_TOKENS:-}"; then
    export AGENTIC_CONTEXT_BUDGET_TOKENS="$CACHEBLEND_CONTEXT_BUDGET_TOKENS"
  fi
  export CACHEBLEND_AGENT_MAX_OUTPUT_TOKENS="${CACHEBLEND_AGENT_MAX_OUTPUT_TOKENS:-4096}"
  export CACHEBLEND_AGENT_MAX_ITERATIONS="${CACHEBLEND_AGENT_MAX_ITERATIONS:-$AGENTIC_MAX_ITERATIONS}"
  export CACHEBLEND_AGENT_MAX_SEARCH_CALLS="${CACHEBLEND_AGENT_MAX_SEARCH_CALLS:-$AGENTIC_MAX_SEARCH_CALLS}"
  echo "CONTEXT_BUDGET_TOKENS=$AGENTIC_CONTEXT_BUDGET_TOKENS"

  if test -x /mnt/nvme2/mlee/rag-system/.venv/bin/python; then
    export CACHEBLEND_P7_PYTHON=/mnt/nvme2/mlee/rag-system/.venv/bin/python
  elif test -n "${PYTHON_BIN:-}"; then
    export CACHEBLEND_P7_PYTHON="$PYTHON_BIN"
  else
    export CACHEBLEND_P7_PYTHON=python3
  fi

  export CACHEBLEND_P7_RUN_BASE_DIR="${CACHEBLEND_P7_RUN_BASE_DIR:-/mnt/nvme2/mlee/cacheblend-gpt-oss-artifacts/browsecomp-append-only-prefix-cacheblend-20260817}"
  export CACHEBLEND_P7_RUN_DIR="$CACHEBLEND_P7_RUN_BASE_DIR"
  if test -e "$CACHEBLEND_P7_RUN_DIR"; then
    export CACHEBLEND_P7_RUN_DIR="${CACHEBLEND_P7_RUN_BASE_DIR}-retry$(date +%Y%m%d-%H%M%S)"
    while test -e "$CACHEBLEND_P7_RUN_DIR"; do
      sleep 1
      export CACHEBLEND_P7_RUN_DIR="${CACHEBLEND_P7_RUN_BASE_DIR}-retry$(date +%Y%m%d-%H%M%S)"
    done
  fi

  mkdir -p "$CACHEBLEND_P7_RUN_DIR/run"
  chmod 700 "$CACHEBLEND_P7_RUN_DIR" "$CACHEBLEND_P7_RUN_DIR/run"

  export CACHEBLEND_G3_RUN_POINTER="${CACHEBLEND_G3_RUN_POINTER:-/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m8.5-browsecomp-append-only-prefix-cacheblend-20260817.current}"
  if ! scp labuser@192.168.3.4:"$CACHEBLEND_G3_RUN_POINTER" \
    "$CACHEBLEND_P7_RUN_DIR/g3-run-dir.txt"; then
    echo "G3_RUN_POINTER_TRANSFER_FAILED=$CACHEBLEND_G3_RUN_POINTER"
    return 0
  fi

  export CACHEBLEND_G3_RUN_DIR="$(sed -n '1p' "$CACHEBLEND_P7_RUN_DIR/g3-run-dir.txt")"
  case "$CACHEBLEND_G3_RUN_DIR" in
    /mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/*) ;;
    *) echo "INVALID_G3_RUN_DIR=$CACHEBLEND_G3_RUN_DIR"; return 0 ;;
  esac

  if ! scp labuser@192.168.3.4:"$CACHEBLEND_G3_RUN_DIR/runtime-identity.json" \
    "$CACHEBLEND_P7_RUN_DIR/runtime-identity.json"; then
    echo "RUNTIME_IDENTITY_TRANSFER_FAILED"
    return 0
  fi

  if ! curl -fsS "$ENCODER_URL/health" > "$CACHEBLEND_P7_RUN_DIR/encoder-health.json"; then
    echo "ENCODER_NOT_READY=$ENCODER_URL/health"
    return 0
  fi

  if ! curl -fsS http://127.0.0.1:8012/health > "$CACHEBLEND_P7_RUN_DIR/search-health.json"; then
    echo "SEARCH_NOT_READY=http://127.0.0.1:8012/health"
    return 0
  fi

  if ! curl -fsS "$RAG_GENERATOR_URL/models" > "$CACHEBLEND_P7_RUN_DIR/generator-models.json"; then
    echo "GENERATOR_NOT_READY=$RAG_GENERATOR_URL/models"
    return 0
  fi

  curl -fsS "${RAG_GENERATOR_URL%/v1}/metrics" \
    > "$CACHEBLEND_P7_RUN_DIR/metrics-before.prom"

  PYTHONPATH=src "$CACHEBLEND_P7_PYTHON" scripts/run_oss_standard_agent.py \
    --prepared-dir "$RAG_PREPARED_DIR" \
    --query-id 703 \
    --search-url "$RAG_SEARCH_URL" \
    --generator-url "$RAG_GENERATOR_URL" \
    --model openai/gpt-oss-20b \
    --reasoning-effort "$AGENTIC_REASONING_EFFORT" \
    --max-output-tokens "$CACHEBLEND_AGENT_MAX_OUTPUT_TOKENS" \
    --forced-decision-reasoning-effort "$AGENTIC_FORCED_DECISION_REASONING_EFFORT" \
    --forced-decision-max-output-tokens "$AGENTIC_FORCED_DECISION_MAX_OUTPUT_TOKENS" \
    --max-forced-decision-recoveries "$AGENTIC_MAX_FORCED_DECISION_RECOVERIES" \
    --max-iterations "$CACHEBLEND_AGENT_MAX_ITERATIONS" \
    --max-search-calls "$CACHEBLEND_AGENT_MAX_SEARCH_CALLS" \
    --context-budget-tokens "$AGENTIC_CONTEXT_BUDGET_TOKENS" \
    --context-strategy append_only \
    --cache-mode cacheblend \
    --no-deduplicate-retrieved-documents \
    --generator-timeout-seconds "$GENERATOR_TIMEOUT_SECONDS" \
    --trace-jsonl "$CACHEBLEND_P7_RUN_DIR/trace.jsonl" \
    --output-dir "$CACHEBLEND_P7_RUN_DIR/run" \
    2>&1 | tee "$CACHEBLEND_P7_RUN_DIR/agent.log"

  export CACHEBLEND_AGENT_STATUS=${PIPESTATUS[0]}
  echo "AGENT_STATUS=$CACHEBLEND_AGENT_STATUS"

  export CACHEBLEND_RUN_RECORD="$CACHEBLEND_P7_RUN_DIR/run/run_703.json"
  if test ! -s "$CACHEBLEND_RUN_RECORD"; then
    echo "RUN_RECORD_MISSING=$CACHEBLEND_RUN_RECORD"
    return 0
  fi

  export CACHEBLEND_GENERATIONS="$("$CACHEBLEND_P7_PYTHON" -c "import json; print(json.load(open('$CACHEBLEND_RUN_RECORD'))['diagnostics']['generation_request_count'])" 2>/dev/null || echo 0)"

  for n in $(seq 1 120); do
    curl -fsS "${RAG_GENERATOR_URL%/v1}/metrics" \
      > "$CACHEBLEND_P7_RUN_DIR/metrics-after-live.prom" 2>/dev/null || true

    export CACHEBLEND_CURRENT="$(awk '/vllm:cacheblend_requests_total\{/ {print $NF; exit}' "$CACHEBLEND_P7_RUN_DIR/metrics-after-live.prom")"
    export CACHEBLEND_CURRENT="${CACHEBLEND_CURRENT:-0}"

    if awk -v actual="$CACHEBLEND_CURRENT" -v expected="$CACHEBLEND_GENERATIONS" 'BEGIN { exit !(actual >= expected) }'; then
      break
    fi
    sleep 1
  done

  cp "$CACHEBLEND_P7_RUN_DIR/metrics-after-live.prom" \
    "$CACHEBLEND_P7_RUN_DIR/metrics-after.prom"

  echo "AGENT_STATUS=$CACHEBLEND_AGENT_STATUS"
  echo "RUN_DIR=$CACHEBLEND_P7_RUN_DIR"
}

main
