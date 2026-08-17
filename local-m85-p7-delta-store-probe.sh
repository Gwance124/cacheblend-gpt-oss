#!/usr/bin/env bash

main() {
  cd /mnt/nvme2/mlee/rag-system || return 0
  source .env
  if test -n "${CACHEBLEND_CONTEXT_BUDGET_TOKENS:-}"; then
    export AGENTIC_CONTEXT_BUDGET_TOKENS="$CACHEBLEND_CONTEXT_BUDGET_TOKENS"
  fi

  export CACHEBLEND_P7_PYTHON="${CACHEBLEND_P7_PYTHON:-/mnt/nvme2/mlee/rag-system/.venv/bin/python}"
  if test ! -x "$CACHEBLEND_P7_PYTHON"; then
    export CACHEBLEND_P7_PYTHON="${PYTHON_BIN:-python3}"
  fi

  export CACHEBLEND_P7_RUN_BASE_DIR="${CACHEBLEND_P7_RUN_BASE_DIR:-/mnt/nvme2/mlee/cacheblend-gpt-oss-artifacts/browsecomp-append-only-cacheblend-delta-store-probe-$(date +%Y%m%d-%H%M%S)}"
  export CACHEBLEND_P7_RUN_DIR="$CACHEBLEND_P7_RUN_BASE_DIR"
  while test -e "$CACHEBLEND_P7_RUN_DIR"; do
    export CACHEBLEND_P7_RUN_DIR="${CACHEBLEND_P7_RUN_BASE_DIR}-retry$(date +%Y%m%d-%H%M%S)"
    sleep 1
  done
  mkdir -p "$CACHEBLEND_P7_RUN_DIR/run"
  chmod 700 "$CACHEBLEND_P7_RUN_DIR" "$CACHEBLEND_P7_RUN_DIR/run"

  export CACHEBLEND_G3_RUN_POINTER="${CACHEBLEND_G3_RUN_POINTER:-/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m8.5-selective-browsecomp-append-only-$(date +%Y%m%d).current}"
  export CACHEBLEND_G3_RUN_DIR_FILE="$CACHEBLEND_P7_RUN_DIR/g3-run-dir.txt"
  if ! scp labuser@192.168.3.4:"$CACHEBLEND_G3_RUN_POINTER" "$CACHEBLEND_G3_RUN_DIR_FILE"; then
    echo "G3_RUN_POINTER_TRANSFER_FAILED=$CACHEBLEND_G3_RUN_POINTER"
    return 0
  fi

  export CACHEBLEND_G3_RUN_DIR="$(sed -n '1p' "$CACHEBLEND_G3_RUN_DIR_FILE")"
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

  export CACHEBLEND_METRICS_URL="${RAG_GENERATOR_URL%/v1}/metrics"
  export CACHEBLEND_METRICS_BEFORE="$CACHEBLEND_P7_RUN_DIR/metrics-before.prom"
  curl -fsS "$CACHEBLEND_METRICS_URL" > "$CACHEBLEND_METRICS_BEFORE"

  export CACHEBLEND_METRICS_SAMPLES="$CACHEBLEND_P7_RUN_DIR/metrics-samples.jsonl"
  : > "$CACHEBLEND_METRICS_SAMPLES"
  export CACHEBLEND_PROBE_MARKER="$CACHEBLEND_P7_RUN_DIR/agent-running"
  : > "$CACHEBLEND_PROBE_MARKER"
  export CACHEBLEND_PROBE_MONITOR_PID=""

  record_metrics_sample() {
    CACHEBLEND_SAMPLE_PROM="$1" \
      CACHEBLEND_SAMPLE_TIME="$2" \
      PYTHONPATH=/mnt/nvme2/mlee/cacheblend-gpt-oss/src \
      "$CACHEBLEND_P7_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

from cacheblend_gpt_oss.correctness import (
    parse_connector_counter_snapshot,
    parse_connector_store_counter_snapshot,
    parse_selective_work_counter_snapshot,
)


def metric_total(text: str, name: str) -> float:
    total = 0.0
    for line in text.splitlines():
        fields = line.strip().split()
        if len(fields) < 2:
            continue
        if fields[0].split("{", 1)[0] != name:
            continue
        total += float(fields[1])
    return total


text = Path(os.environ["CACHEBLEND_SAMPLE_PROM"]).read_text()
connector = parse_connector_counter_snapshot(text)
store = parse_connector_store_counter_snapshot(text)
work = parse_selective_work_counter_snapshot(text)
connector_latency = {}
for key in (
    "lookup_latency_seconds",
    "transfer_latency_seconds",
    "position_correction_latency_seconds",
    "selective_recomputation_latency_seconds",
    "store_latency_seconds",
):
    prefix = f"vllm:cacheblend_{key}"
    connector_latency[key] = {
        "count": int(metric_total(text, f"{prefix}_count")),
        "sum_seconds": metric_total(text, f"{prefix}_sum"),
    }
vllm_timing = {}
for key, metric in (
    ("ttft_seconds", "vllm:time_to_first_token_seconds"),
    ("end_to_end_latency_seconds", "vllm:e2e_request_latency_seconds"),
    ("prefill_latency_seconds", "vllm:request_prefill_time_seconds"),
):
    vllm_timing[key] = {
        "count": int(metric_total(text, f"{metric}_count")),
        "sum_seconds": metric_total(text, f"{metric}_sum"),
    }
print(
    json.dumps(
        {
            "time": os.environ["CACHEBLEND_SAMPLE_TIME"],
            "requests": connector["requests"],
            "kv_tokens_found": connector["kv_tokens_found"],
            "kv_tokens_loaded": connector["kv_tokens_loaded"],
            "store_tokens_eligible": store["store_tokens_eligible"],
            "store_tokens_completed": store["store_tokens_completed"],
            "layer_token_rows_recomputed": work["layer_token_rows_recomputed"],
            "layer_token_rows_avoided": work["layer_token_rows_avoided"],
            "connector_latency": connector_latency,
            "vllm_timing": vllm_timing,
        },
        sort_keys=True,
    )
)
PY
  }

  cleanup() {
    rm -f "$CACHEBLEND_PROBE_MARKER"
    if test -n "${CACHEBLEND_PROBE_MONITOR_PID:-}"; then
      kill -TERM "$CACHEBLEND_PROBE_MONITOR_PID" 2>/dev/null || true
    fi
  }
  trap cleanup INT TERM EXIT

  (
    while test -e "$CACHEBLEND_PROBE_MARKER"; do
      export CACHEBLEND_SAMPLE_PROM="$CACHEBLEND_P7_RUN_DIR/metrics-sample.prom"
      if curl -fsS "$CACHEBLEND_METRICS_URL" > "$CACHEBLEND_SAMPLE_PROM" 2>/dev/null; then
        export CACHEBLEND_SAMPLE_TIME="$(date +%s.%N)"
        record_metrics_sample "$CACHEBLEND_SAMPLE_PROM" "$CACHEBLEND_SAMPLE_TIME" \
          >> "$CACHEBLEND_METRICS_SAMPLES" 2>/dev/null || true
      fi
      sleep 1
    done
  ) &
  export CACHEBLEND_PROBE_MONITOR_PID=$!

  export CACHEBLEND_AGENT_MAX_OUTPUT_TOKENS="${CACHEBLEND_PROBE_MAX_OUTPUT_TOKENS:-1024}"
  export CACHEBLEND_AGENT_MAX_ITERATIONS="${CACHEBLEND_PROBE_MAX_ITERATIONS:-3}"
  export CACHEBLEND_AGENT_MAX_SEARCH_CALLS="${CACHEBLEND_PROBE_MAX_SEARCH_CALLS:-3}"

  echo "PROBE_RUN_DIR=$CACHEBLEND_P7_RUN_DIR"
  echo "PROBE_G3_RUN_DIR=$CACHEBLEND_G3_RUN_DIR"
  echo "PROBE_MAX_ITERATIONS=$CACHEBLEND_AGENT_MAX_ITERATIONS"
  echo "PROBE_MAX_SEARCH_CALLS=$CACHEBLEND_AGENT_MAX_SEARCH_CALLS"
  echo "PROBE_CONTEXT_BUDGET_TOKENS=$AGENTIC_CONTEXT_BUDGET_TOKENS"
  echo "PROBE_METRICS_SAMPLES=$CACHEBLEND_METRICS_SAMPLES"
  echo "PROBE_TELEMETRY_ONLY=yes"

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

  rm -f "$CACHEBLEND_PROBE_MARKER"
  wait "$CACHEBLEND_PROBE_MONITOR_PID" 2>/dev/null || true
  export CACHEBLEND_PROBE_MONITOR_PID=""
  curl -fsS "$CACHEBLEND_METRICS_URL" > "$CACHEBLEND_P7_RUN_DIR/metrics-after.prom" 2>/dev/null || true

  if test -s "$CACHEBLEND_METRICS_SAMPLES"; then
    CACHEBLEND_METRICS_SAMPLES="$CACHEBLEND_METRICS_SAMPLES" \
      "$CACHEBLEND_P7_PYTHON" - <<'PY'
import json
import os
from pathlib import Path


rows = [
    json.loads(line)
    for line in Path(os.environ["CACHEBLEND_METRICS_SAMPLES"]).read_text().splitlines()
    if line.strip()
]
print(f"PROBE_SAMPLE_COUNT={len(rows)}")
if rows:
    previous = rows[0]
    print(
        json.dumps(
            {
                "baseline_requests": previous["requests"],
                "baseline_store_tokens_completed": previous["store_tokens_completed"],
                "baseline_store_tokens_eligible": previous["store_tokens_eligible"],
                "baseline_layer_token_rows_recomputed": previous[
                    "layer_token_rows_recomputed"
                ],
                "baseline_layer_token_rows_avoided": previous[
                    "layer_token_rows_avoided"
                ],
                "baseline_connector_latency": previous["connector_latency"],
                "baseline_vllm_timing": previous["vllm_timing"],
            },
            sort_keys=True,
        )
    )
    for row in rows[1:]:
        if row["requests"] == previous["requests"]:
            continue
        print(
            json.dumps(
                {
                    "request": row["requests"],
                    "store_tokens_completed_delta": row["store_tokens_completed"]
                    - previous["store_tokens_completed"],
                    "store_tokens_eligible_delta": row["store_tokens_eligible"]
                    - previous["store_tokens_eligible"],
                    "layer_token_rows_recomputed_delta": row[
                        "layer_token_rows_recomputed"
                    ]
                    - previous["layer_token_rows_recomputed"],
                    "layer_token_rows_avoided_delta": row["layer_token_rows_avoided"]
                    - previous["layer_token_rows_avoided"],
                    "connector_latency_delta": {
                        key: {
                            "count": row["connector_latency"][key]["count"]
                            - previous["connector_latency"][key]["count"],
                            "sum_seconds": row["connector_latency"][key]["sum_seconds"]
                            - previous["connector_latency"][key]["sum_seconds"],
                        }
                        for key in row["connector_latency"]
                    },
                    "vllm_timing_delta": {
                        key: {
                            "count": row["vllm_timing"][key]["count"]
                            - previous["vllm_timing"][key]["count"],
                            "sum_seconds": row["vllm_timing"][key]["sum_seconds"]
                            - previous["vllm_timing"][key]["sum_seconds"],
                        }
                        for key in row["vllm_timing"]
                    },
                },
                sort_keys=True,
            )
        )
        previous = row
PY
      2>/dev/null || true
  else
    echo "PROBE_METRICS_SAMPLES_EMPTY"
  fi

  echo "PROBE_AGENT_STATUS=$CACHEBLEND_AGENT_STATUS"
  echo "PROBE_COMPLETE=$CACHEBLEND_P7_RUN_DIR"
}

main
