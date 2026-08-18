#!/usr/bin/env bash
# Analyze CacheBlend transfer diagnostics from a completed solab-g3 run.
#
# Usage:
#   bash local-m85-check-transfer-diag.sh [RUN_DIR_OR_POINTER]
#
# With no argument, the unified-mode run pointer is used. The report is written
# next to vllm-server.log as prefix-cache-diagnostic.txt. A diagnostic verdict
# is evidence, not a test failure; only missing or malformed diagnostics return
# a nonzero status.

set -euo pipefail

readonly DEFAULT_POINTER=/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m8.5-browsecomp-append-only-prefix-cacheblend-unified-20260818.current

resolve_run_dir() {
  local target="$1"
  local run_dir

  if test -d "$target"; then
    run_dir="$target"
  elif test -f "$target"; then
    IFS= read -r run_dir < "$target" || true
    run_dir="${run_dir%$'\r'}"
  else
    echo "Run directory or pointer not found: $target" >&2
    return 1
  fi

  if test -z "$run_dir" || test ! -d "$run_dir"; then
    echo "Pointer does not name an existing run directory: $target" >&2
    return 1
  fi
  printf '%s\n' "$run_dir"
}

sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    printf '%s\n' unavailable
  fi
}

print_first_matching() {
  local pattern="$1"
  local limit="$2"
  local path="$3"
  awk -v pattern="$pattern" -v limit="$limit" '
    index($0, pattern) {
      print
      seen++
      if (seen == limit) {
        exit
      }
    }
  ' "$path"
}

print_last_matching() {
  local pattern="$1"
  local limit="$2"
  local path="$3"
  awk -v pattern="$pattern" -v limit="$limit" '
    index($0, pattern) {
      lines[seen % limit] = $0
      seen++
    }
    END {
      start = seen > limit ? seen - limit : 0
      for (i = start; i < seen; i++) {
        print lines[i % limit]
      }
    }
  ' "$path"
}

main() {
  local target="${1:-$DEFAULT_POINTER}"
  local run_dir
  run_dir="$(resolve_run_dir "$target")"

  local log="$run_dir/vllm-server.log"
  if test ! -s "$log"; then
    echo "vLLM server log missing or empty: $log" >&2
    return 1
  fi

  local report="$run_dir/prefix-cache-diagnostic.txt"
  local temporary_report
  temporary_report="$(mktemp "$run_dir/.prefix-cache-diagnostic.XXXXXX")"
  trap 'rm -f -- "${temporary_report:-}"' EXIT

  local summary
  summary="$(awk '
    index($0, "CACHEBLEND_TRANSFER_DIAG") {
      transfer_events++
    }
    index($0, "CACHEBLEND_DECODE_DIAG") {
      decode_events++
    }
    index($0, "CACHEBLEND_TRANSFER_DIAG lookup") {
      lookup_events++
      value_seen = 0
      request_seen = 0
      for (i = 1; i <= NF; i++) {
        if (index($i, "request=") == 1) {
          request = substr($i, length("request=") + 1)
          if (request != "") {
            requests[request] = 1
            request_seen = 1
          }
        }
        if (index($i, "prefix_cached_tokens=") == 1) {
          raw = substr($i, length("prefix_cached_tokens=") + 1)
          if (raw ~ /^[0-9]+$/) {
            value = raw + 0
            value_seen = 1
            valid_values++
            value_sum += value
            if (valid_values == 1 || value < value_min) {
              value_min = value
            }
            if (valid_values == 1 || value > value_max) {
              value_max = value
            }
            if (value == 0) {
              zero_values++
            } else {
              nonzero_values++
            }
          }
        }
      }
      if (!value_seen || !request_seen) {
        malformed_lookups++
      }
    }
    END {
      for (request in requests) {
        unique_requests++
      }
      if (valid_values == 0) {
        value_min = 0
        value_max = 0
      }
      printf "%d %d %d %d %d %d %d %d %d %d\n", \
        transfer_events, decode_events, lookup_events, unique_requests, \
        valid_values, zero_values, nonzero_values, malformed_lookups, \
        value_min, value_max
      printf "%d\n", value_sum
    }
  ' "$log")"

  local first_summary second_summary
  first_summary="$(printf '%s\n' "$summary" | sed -n '1p')"
  second_summary="$(printf '%s\n' "$summary" | sed -n '2p')"

  local transfer_events decode_events lookup_events unique_requests
  local valid_values zero_values nonzero_values malformed_lookups
  local value_min value_max value_sum
  read -r transfer_events decode_events lookup_events unique_requests \
    valid_values zero_values nonzero_values malformed_lookups \
    value_min value_max <<< "$first_summary"
  read -r value_sum <<< "$second_summary"

  local gate key_finding next_action diagnostic_status=0
  if test "$lookup_events" -eq 0; then
    gate=INCONCLUSIVE_NO_LOOKUP_DIAGNOSTICS
    key_finding="No CACHEBLEND_TRANSFER_DIAG lookup events were captured."
    next_action="Rerun with CACHEBLEND_TRANSFER_DIAG=1 and verify stderr is redirected to vllm-server.log."
    diagnostic_status=2
  elif test "$malformed_lookups" -ne 0 || test "$valid_values" -ne "$lookup_events"; then
    gate=INCONCLUSIVE_MALFORMED_LOOKUP_DIAGNOSTICS
    key_finding="At least one lookup event lacks a parseable request or prefix_cached_tokens field."
    next_action="Inspect the lookup excerpts before drawing a prefix-cache conclusion."
    diagnostic_status=2
  elif test "$nonzero_values" -eq 0 && test "$zero_values" -eq "$lookup_events"; then
    gate=FAIL_ALL_LOOKUPS_ZERO
    key_finding="Every connector lookup received prefix_cached_tokens=0; no vLLM prefix reuse reached the connector."
    next_action="Trace vLLM prefix-cache matching before connector lookup and compare arm 2 against connector-attached mode."
  else
    gate=PASS_NONZERO_REUSE_OBSERVED
    key_finding="At least one connector lookup received nonzero prefix-cached tokens."
    next_action="Trace disable_hybrid_kv_cache_manager=None versus True beyond hybrid-spec unification."
  fi

  local log_sha256
  log_sha256="$(sha256_file "$log")"

  {
    echo "RUN_DIR=$run_dir"
    echo "VLLM_LOG=$log"
    echo "VLLM_LOG_SHA256=$log_sha256"
    echo "REPORT=$report"
    echo
    echo "=== CACHEBLEND_TRANSFER_DIAG (first 20) ==="
    print_first_matching CACHEBLEND_TRANSFER_DIAG 20 "$log"
    echo
    echo "=== CACHEBLEND_DECODE_DIAG (last 5) ==="
    print_last_matching CACHEBLEND_DECODE_DIAG 5 "$log"
    echo
    echo "=== Prefix cache scheduler reports (last 10) ==="
    print_last_matching "Prefix cache hit rate" 10 "$log"
    echo
    echo "=== Calculated summary ==="
    echo "TRANSFER_DIAG_EVENTS=$transfer_events"
    echo "DECODE_DIAG_EVENTS=$decode_events"
    echo "LOOKUP_EVENTS=$lookup_events"
    echo "UNIQUE_REQUESTS=$unique_requests"
    echo "PREFIX_CACHED_VALUES=$valid_values"
    echo "PREFIX_CACHED_ZERO_EVENTS=$zero_values"
    echo "PREFIX_CACHED_NONZERO_EVENTS=$nonzero_values"
    echo "MALFORMED_LOOKUP_EVENTS=$malformed_lookups"
    echo "PREFIX_CACHED_MIN=$value_min"
    echo "PREFIX_CACHED_MAX=$value_max"
    echo "PREFIX_CACHED_SUM=$value_sum"
    echo "PREFIX_CACHE_GATE=$gate"
    echo "KEY_FINDING=$key_finding"
    echo "NEXT_ACTION=$next_action"
  } | tee "$temporary_report"

  mv "$temporary_report" "$report"
  trap - EXIT
  echo "ARTIFACT_WRITTEN=$report"
  return "$diagnostic_status"
}

main "$@"
