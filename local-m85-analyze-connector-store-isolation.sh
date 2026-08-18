#!/usr/bin/env bash
# Combine the existing store-on and no-store GPU artifacts without rerunning GPUs.

set -euo pipefail

readonly CACHEBLEND_REPO=/mnt/nvme3n1/mlee/cacheblend-gpt-oss
readonly CACHEBLEND_BRANCH=cacheblend-scatter-diagnostic-and-checklayer
readonly CACHEBLEND_STORE_ON_POINTER="$CACHEBLEND_REPO/artifacts/solab-g3-m8.5-connector-presence-equivalence-20260818.current"
readonly CACHEBLEND_NO_STORE_POINTER="$CACHEBLEND_REPO/artifacts/solab-g3-m8.5-connector-no-store-equivalence-20260818.current"

require_clean_tracked_worktree() {
  if test -n "$(git status --porcelain --untracked-files=no)"; then
    echo "STOP_TRACKED_WORKTREE_CHANGES" >&2
    git status --short --untracked-files=no >&2
    return 1
  fi
}

read_pointer() {
  local pointer="$1"
  if ! test -f "$pointer"; then
    echo "MISSING_RUN_POINTER=$pointer" >&2
    return 1
  fi
  local run_dir
  run_dir="$(<"$pointer")"
  if ! test -d "$run_dir"; then
    echo "MISSING_RUN_DIR=$run_dir" >&2
    return 1
  fi
  printf '%s\n' "$run_dir"
}

main() {
  if test "$#" -gt 2; then
    echo "usage: $0 [STORE_ON_RUN_DIR] [NO_STORE_RUN_DIR]" >&2
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

  local store_on_run_dir="${1:-}"
  local no_store_run_dir="${2:-}"
  if test -z "$store_on_run_dir"; then
    store_on_run_dir="$(read_pointer "$CACHEBLEND_STORE_ON_POINTER")"
  fi
  if test -z "$no_store_run_dir"; then
    no_store_run_dir="$(read_pointer "$CACHEBLEND_NO_STORE_POINTER")"
  fi
  case "$store_on_run_dir" in
    "$CACHEBLEND_REPO"/artifacts/solab-g3-m8.5-connector-presence-equivalence-*) ;;
    *) echo "INVALID_STORE_ON_RUN_DIR=$store_on_run_dir" >&2; return 1 ;;
  esac
  case "$no_store_run_dir" in
    "$CACHEBLEND_REPO"/artifacts/solab-g3-m8.5-connector-no-store-equivalence-*) ;;
    *) echo "INVALID_NO_STORE_RUN_DIR=$no_store_run_dir" >&2; return 1 ;;
  esac
  if ! test -d "$store_on_run_dir"; then
    echo "MISSING_RUN_DIR=$store_on_run_dir" >&2
    return 1
  fi
  if ! test -d "$no_store_run_dir"; then
    echo "MISSING_RUN_DIR=$no_store_run_dir" >&2
    return 1
  fi

  local output="$no_store_run_dir/connector-store-isolation.json"
  .venv/bin/python scripts/analyze_connector_store_isolation.py \
    --store-on-run-dir "$store_on_run_dir" \
    --no-store-run-dir "$no_store_run_dir" \
    --output "$output" \
    | tee "$no_store_run_dir/connector-store-isolation.txt"
  echo "ARTIFACT_WRITTEN=$output"
}

main "$@"
