#!/usr/bin/env bash
# Extract CACHEBLEND_TRANSFER_DIAG output from the latest arm 4 run.
# Shows per-request: prefix_cached_tokens, scheduled_tokens, complete_step,
# should_transfer, and whether a transfer was created.
#
# Usage: bash local-m85-check-transfer-diag.sh [VLLM_LOG_DIR]
# If no dir given, reads from the .current pointer.

set -euo pipefail

DIR="${1:-}"
if test -z "$DIR"; then
  # Try noscatter first, then regular cacheblend-prefix
  for PTR in \
    /mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m8.5-browsecomp-append-only-prefix-cacheblend-noscatter-20260817.current \
    /mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m8.5-browsecomp-append-only-prefix-cacheblend-20260817.current; do
    if test -f "$PTR"; then
      DIR="$(cat "$PTR")"
      break
    fi
  done
fi

if test -z "$DIR" || test ! -f "$DIR/vllm-server.log"; then
  echo "No vllm-server.log found. Pass the run dir as argument." >&2
  exit 1
fi

LOG="$DIR/vllm-server.log"
echo "=== Transfer Diagnostic: $DIR ==="
echo ""

echo "--- Lookup results (prefix_cached_tokens, should_transfer) ---"
grep "CACHEBLEND_TRANSFER_DIAG lookup" "$LOG" | head -40
echo ""

echo "--- Alloc results (num_external_tokens) ---"
grep "CACHEBLEND_TRANSFER_DIAG alloc" "$LOG" | head -40
echo ""

echo "--- Build_meta results (scheduled_tokens, complete_step, should_transfer) ---"
grep "CACHEBLEND_TRANSFER_DIAG build_meta" "$LOG" | head -40
echo ""

echo "--- Summary ---"
TOTAL_LOOKUPS=$(grep -c "CACHEBLEND_TRANSFER_DIAG lookup" "$LOG" 2>/dev/null || echo 0)
ELIGIBLE_LOOKUPS=$(grep "CACHEBLEND_TRANSFER_DIAG lookup" "$LOG" | grep -c "should_transfer=True" 2>/dev/null || echo 0)
INELIGIBLE_LOOKUPS=$(grep "CACHEBLEND_TRANSFER_DIAG lookup" "$LOG" | grep -c "should_transfer=False" 2>/dev/null || echo 0)
COMPLETE_STEPS=$(grep "CACHEBLEND_TRANSFER_DIAG build_meta" "$LOG" | grep -c "complete_step=True" 2>/dev/null || echo 0)
INCOMPLETE_STEPS=$(grep "CACHEBLEND_TRANSFER_DIAG build_meta" "$LOG" | grep -c "complete_step=False" 2>/dev/null || echo 0)
ZERO_PREFIX=$(grep "CACHEBLEND_TRANSFER_DIAG lookup" "$LOG" | grep -c "prefix_cached_tokens=0 " 2>/dev/null || echo 0)
NONZERO_EXTERNAL=$(grep "CACHEBLEND_TRANSFER_DIAG alloc" "$LOG" | grep -v "num_external_tokens=0$" | wc -l)

echo "Total lookups:            $TOTAL_LOOKUPS"
echo "  should_transfer=True:   $ELIGIBLE_LOOKUPS"
echo "  should_transfer=False:  $INELIGIBLE_LOOKUPS"
echo "  prefix_cached_tokens=0: $ZERO_PREFIX"
echo "Build_meta:"
echo "  complete_step=True:     $COMPLETE_STEPS"
echo "  complete_step=False:    $INCOMPLETE_STEPS"
echo "Alloc num_external_tokens != 0: $NONZERO_EXTERNAL"
echo ""

echo "--- Decode diag ---"
grep "CACHEBLEND_DECODE_DIAG" "$LOG" | tail -5
echo ""

echo "--- Prometheus output tokens ---"
if curl -fsS http://127.0.0.1:8000/metrics 2>/dev/null | grep "generation_tokens_total"; then
  :
else
  echo "(server not running or metrics unavailable)"
fi
