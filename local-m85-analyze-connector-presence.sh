#!/usr/bin/env bash
# Analyze an existing deterministic connector-presence run without a GPU rerun.

set -euo pipefail

readonly CACHEBLEND_REPO=/mnt/nvme3n1/mlee/cacheblend-gpt-oss
readonly CACHEBLEND_BRANCH=cacheblend-scatter-diagnostic-and-checklayer
readonly CACHEBLEND_POINTER=/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m8.5-connector-presence-equivalence-20260818.current

require_clean_tracked_worktree() {
  if test -n "$(git status --porcelain --untracked-files=no)"; then
    echo "STOP_TRACKED_WORKTREE_CHANGES" >&2
    git status --short --untracked-files=no >&2
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
    if ! test -f "$CACHEBLEND_POINTER"; then
      echo "MISSING_RUN_POINTER=$CACHEBLEND_POINTER" >&2
      return 1
    fi
    run_dir="$(<"$CACHEBLEND_POINTER")"
  fi
  case "$run_dir" in
    "$CACHEBLEND_REPO"/artifacts/solab-g3-m8.5-connector-presence-equivalence-*) ;;
    *)
      echo "INVALID_RUN_DIR=$run_dir" >&2
      return 1
      ;;
  esac
  if ! test -d "$run_dir"; then
    echo "MISSING_RUN_DIR=$run_dir" >&2
    return 1
  fi

  local output="$run_dir/connector-stage-diagnostic.json"
  .venv/bin/python scripts/analyze_connector_presence_run.py \
    --run-dir "$run_dir" \
    --output "$output" \
    | tee "$run_dir/connector-stage-diagnostic.txt"
  echo "ARTIFACT_WRITTEN=$output"
}

main "$@"
