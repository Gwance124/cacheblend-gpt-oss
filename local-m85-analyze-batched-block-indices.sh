#!/usr/bin/env bash
# Recover block-index artifacts from an existing G3 run without rerunning GPUs.

set -euo pipefail

readonly CACHEBLEND_REPO=/mnt/nvme3n1/mlee/cacheblend-gpt-oss
readonly CACHEBLEND_BRANCH=cacheblend-scatter-diagnostic-and-checklayer
readonly CACHEBLEND_POINTER="$CACHEBLEND_REPO/artifacts/solab-g3-m8.5-connector-presence-equivalence-20260818.current"
readonly CACHEBLEND_BASELINE_RUN_DIR="$CACHEBLEND_REPO/artifacts/solab-g3-m8.5-connector-presence-equivalence-20260818-retry20260818-140016"

require_clean_tracked_worktree() {
  if test -n "$(git status --porcelain --untracked-files=no)"; then
    echo "STOP_TRACKED_WORKTREE_CHANGES" >&2
    git status --short --untracked-files=no >&2
    return 1
  fi
}

read_pointer() {
  if ! test -f "$CACHEBLEND_POINTER"; then
    echo "MISSING_RUN_POINTER=$CACHEBLEND_POINTER" >&2
    return 1
  fi
  local run_dir
  run_dir="$(<"$CACHEBLEND_POINTER")"
  if ! test -d "$run_dir"; then
    echo "MISSING_RUN_DIR=$run_dir" >&2
    return 1
  fi
  printf '%s\n' "$run_dir"
}

require_absent() {
  local path="$1"
  if test -e "$path"; then
    echo "STOP_EXISTING_ARTIFACT=$path" >&2
    return 1
  fi
}

require_metric() {
  local metrics_path="$1"
  local metric_name="$2"
  if ! grep -q "^${metric_name}{" "$metrics_path"; then
    echo "MISSING_PROMETHEUS_METRIC=$metric_name" >&2
    return 1
  fi
}

main() {
  if test "$#" -gt 1; then
    echo "usage: $0 [RUN_DIR]" >&2
    return 2
  fi
  cd "$CACHEBLEND_REPO"
  require_clean_tracked_worktree
  git fetch origin
  git switch "$CACHEBLEND_BRANCH" 2>/dev/null \
    || git switch -c "$CACHEBLEND_BRANCH" --track "origin/$CACHEBLEND_BRANCH"
  git pull --ff-only
  require_clean_tracked_worktree
  if test -n "$(git status --porcelain --untracked-files=normal)"; then
    echo "PRESERVING_UNTRACKED_FILES"
    git status --short --untracked-files=normal
  fi

  local run_dir="${1:-}"
  if test -z "$run_dir"; then
    run_dir="$(read_pointer)"
  fi
  case "$run_dir" in
    "$CACHEBLEND_REPO"/artifacts/solab-g3-m8.5-connector-presence-equivalence-*) ;;
    *) echo "INVALID_RUN_DIR=$run_dir" >&2; return 1 ;;
  esac
  if ! test -d "$run_dir"; then
    echo "MISSING_RUN_DIR=$run_dir" >&2
    return 1
  fi
  if ! test -d "$CACHEBLEND_BASELINE_RUN_DIR"; then
    echo "MISSING_BASELINE_RUN_DIR=$CACHEBLEND_BASELINE_RUN_DIR" >&2
    return 1
  fi

  local metrics_after="$run_dir/connector/metrics-after.prom"
  if ! test -f "$metrics_after"; then
    echo "MISSING_METRICS=$metrics_after" >&2
    return 1
  fi
  require_metric "$metrics_after" \
    vllm:cacheblend_store_preflight_block_index_owner_constructions_total
  require_metric "$metrics_after" \
    vllm:cacheblend_store_preflight_block_index_row_views_total
  require_metric "$metrics_after" \
    vllm:cacheblend_store_preflight_staging_view_constructions_total

  local block_output="$run_dir/connector-block-index-view-breakdown.json"
  local gate_output="$run_dir/connector-batched-block-indices.json"
  require_absent "$block_output"
  require_absent "$gate_output"

  .venv/bin/python scripts/analyze_connector_block_index_view.py \
    --run-dir "$run_dir" \
    --output "$block_output" \
    | tee "$run_dir/connector-block-index-view-breakdown.txt"

  set +e
  .venv/bin/python scripts/analyze_connector_batched_block_indices.py \
    --baseline-run-dir "$CACHEBLEND_BASELINE_RUN_DIR" \
    --candidate-run-dir "$run_dir" \
    --output "$gate_output" \
    | tee "$run_dir/connector-batched-block-indices.txt"
  local gate_status=${PIPESTATUS[0]}
  set -e

  echo "BLOCK_INDEX_VIEW_STATUS=0"
  echo "BATCHED_BLOCK_INDICES_STATUS=$gate_status"
  echo "RUN_DIR=$run_dir"
  return "$gate_status"
}

main "$@"
