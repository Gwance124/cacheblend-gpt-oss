#!/usr/bin/env bash
# Run on g3 AFTER the browsecomp test completes.
# Usage: bash local-m85-check-prefix-diag.sh [RUN_DIR]
#
# If RUN_DIR is omitted, reads from the g3 run pointer.

set -euo pipefail

RUN_DIR="${1:-}"
if test -z "$RUN_DIR"; then
  POINTER=/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m8.5-browsecomp-append-only-prefix-cacheblend-20260817.current
  if test -f "$POINTER"; then
    RUN_DIR="$(cat "$POINTER")"
  else
    echo "No RUN_DIR given and pointer not found: $POINTER" >&2
    exit 1
  fi
fi

LOG="$RUN_DIR/vllm-server.log"
if test ! -f "$LOG"; then
  echo "vLLM server log not found: $LOG" >&2
  exit 1
fi

echo "=== vLLM Prefix Cache Hit Rate (periodic scheduler report) ==="
grep "Prefix cache hit rate" "$LOG" | tail -10
echo ""

echo "=== CACHEBLEND_PREFIX_DIAG (per-request prefix cache hits) ==="
grep "CACHEBLEND_PREFIX_DIAG" "$LOG" | head -40
echo ""

echo "=== CACHEBLEND_SCHED_DIAG (per-request scheduled tokens) ==="
grep "CACHEBLEND_SCHED_DIAG" "$LOG" | head -40
echo ""

echo "=== Summary ==="
TOTAL=$(grep -c "CACHEBLEND_PREFIX_DIAG" "$LOG" 2>/dev/null || echo 0)
HITS=$(grep "CACHEBLEND_PREFIX_DIAG" "$LOG" | grep -v "prefix_cache_hits=0 " | wc -l | tr -d ' ')
echo "Total requests: $TOTAL"
echo "Requests with prefix_cache_hits > 0: $HITS"

if test "$TOTAL" -gt 0; then
  echo ""
  echo "=== Prometheus prefix cache counters ==="
  METRICS="$RUN_DIR/metrics-after.prom"
  if test -f "$METRICS"; then
    grep -E "vllm:prefix_cache_(queries|hits)" "$METRICS" 2>/dev/null || echo "(not found in metrics)"
  else
    echo "(metrics file not found: $METRICS)"
    LIVE_METRICS="$(curl -fsS http://127.0.0.1:8000/metrics 2>/dev/null || true)"
    if test -n "$LIVE_METRICS"; then
      echo "$LIVE_METRICS" | grep -E "vllm:prefix_cache_(queries|hits)" || echo "(not in live metrics either)"
    fi
  fi
fi
