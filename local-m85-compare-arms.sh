#!/usr/bin/env bash
# Compare arm 2 (prefix only) vs arm 4 (prefix + CacheBlend) results.
# Run on the machine after BOTH arms have completed.
#
# Usage:
#   bash local-m85-compare-arms.sh [PREFIX_RUN_DIR] [CACHEBLEND_RUN_DIR]

set -euo pipefail

PREFIX_DIR="${1:-}"
CACHEBLEND_DIR="${2:-}"

if test -z "$PREFIX_DIR"; then
  PTR=/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m8.5-browsecomp-append-only-prefix-only-20260817.current
  if test -f "$PTR"; then
    PREFIX_DIR="$(cat "$PTR")"
  else
    echo "No PREFIX_DIR given and pointer not found: $PTR" >&2
    exit 1
  fi
fi

if test -z "$CACHEBLEND_DIR"; then
  PTR=/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m8.5-browsecomp-append-only-prefix-cacheblend-20260817.current
  if test -f "$PTR"; then
    CACHEBLEND_DIR="$(cat "$PTR")"
  else
    echo "No CACHEBLEND_DIR given and pointer not found: $PTR" >&2
    exit 1
  fi
fi

echo "=== ARM 2: Prefix Only ==="
echo "DIR: $PREFIX_DIR"
if test -f "$PREFIX_DIR/vllm-server.log"; then
  echo "Prefix cache hit rate:"
  grep "Prefix cache hit rate" "$PREFIX_DIR/vllm-server.log" | tail -3
fi
echo ""

echo "=== ARM 4: Prefix + CacheBlend ==="
echo "DIR: $CACHEBLEND_DIR"
if test -f "$CACHEBLEND_DIR/vllm-server.log"; then
  echo "Prefix cache hit rate:"
  grep "Prefix cache hit rate" "$CACHEBLEND_DIR/vllm-server.log" | tail -3
  echo ""
  echo "Decode step counts:"
  grep "CACHEBLEND_DECODE_DIAG" "$CACHEBLEND_DIR/vllm-server.log" | tail -10
  echo ""
  FINAL=$(grep "CACHEBLEND_DECODE_DIAG finished=" "$CACHEBLEND_DIR/vllm-server.log" | tail -1)
  if test -n "$FINAL"; then
    echo "Final: $FINAL"
  fi
fi
echo ""

echo "=== Agent Timing Comparison ==="
for label_dir in "Prefix-only:$PREFIX_DIR" "Prefix+CB:$CACHEBLEND_DIR"; do
  LABEL="${label_dir%%:*}"
  DIR="${label_dir#*:}"
  AGENT_LOG="$DIR/agent.log"
  if test ! -f "$AGENT_LOG"; then
    # Try the p7 agent log location
    P7_DIR=""
    case "$LABEL" in
      Prefix-only)
        for d in /mnt/nvme2/mlee/cacheblend-gpt-oss-artifacts/browsecomp-append-only-prefix-only-20260817*; do
          if test -f "$d/agent.log"; then P7_DIR="$d"; fi
        done
        ;;
      Prefix+CB)
        for d in /mnt/nvme2/mlee/cacheblend-gpt-oss-artifacts/browsecomp-append-only-cacheblend-prefix-20260817*; do
          if test -f "$d/agent.log"; then P7_DIR="$d"; fi
        done
        ;;
    esac
    if test -n "$P7_DIR"; then
      AGENT_LOG="$P7_DIR/agent.log"
    fi
  fi

  if test -f "$AGENT_LOG"; then
    echo "--- $LABEL ---"
    # Extract total time and search count from last lines
    tail -5 "$AGENT_LOG" | grep -E "total_time|searches|search_count|elapsed" || true
    echo ""
  else
    echo "--- $LABEL: agent.log not found ---"
    echo ""
  fi
done

echo "=== Output Token Comparison ==="
echo "(Run this after the new arm 4 test to compare decode step counts"
echo " with arm 2 timing. If arm 4 has 3x more decode steps, the slowdown"
echo " is from model non-determinism, not connector overhead.)"
