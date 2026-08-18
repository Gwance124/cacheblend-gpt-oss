#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Attribute a connector-presence run to bounded connector stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CONTRACT = "cacheblend-gpt-oss-connector-stage-diagnostic-v1"
VERDICT_CONTRACT = "cacheblend-gpt-oss-connector-presence-equivalence-v1"
_LATENCY_KEYS = (
    "lookup_latency_seconds",
    "transfer_latency_seconds",
    "position_correction_latency_seconds",
    "selective_recomputation_latency_seconds",
    "store_latency_seconds",
)
_PRIMARY_LATENCY_KEYS = (
    "lookup_latency_seconds",
    "transfer_latency_seconds",
    "store_latency_seconds",
)
_CONNECTOR_COUNTER_KEYS = frozenset(
    {
        "requests",
        "reusable_document_tokens_requested",
        "kv_tokens_found",
        "kv_tokens_loaded",
        "kv_tokens_rejected",
        "tokens_recomputed",
        "prefill_tokens_avoided",
    }
)
_STORE_COUNTER_KEYS = frozenset(
    {"store_tokens_eligible", "store_tokens_completed", "store_fallbacks"}
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"could not read artifact: {path}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_read_text(path))
    except json.JSONDecodeError as exc:
        raise ValueError(f"could not parse JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"could not hash artifact: {path}") from exc


def _positive_number(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"invalid {name}")
    return float(value)


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid {name}")
    return value


def _counter_mapping(
    value: object,
    expected_keys: frozenset[str],
    name: str,
) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError(f"invalid {name} schema")
    return {
        key: _nonnegative_integer(value.get(key), f"{name} {key}")
        for key in expected_keys
    }


def _metric_total(
    text: str,
    metric_name: str,
    *,
    allow_missing: bool = False,
) -> float:
    total = 0.0
    found = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2 or parts[0].split("{", 1)[0] != metric_name:
            continue
        try:
            value = float(parts[1])
        except ValueError as exc:
            raise ValueError(f"invalid Prometheus sample: {metric_name}") from exc
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"invalid Prometheus sample: {metric_name}")
        total += value
        found = True
    if not found and not allow_missing:
        raise ValueError(f"missing Prometheus metric: {metric_name}")
    return total


def _histogram_delta(before: str, after: str, key: str) -> dict[str, int | float]:
    base = f"vllm:cacheblend_{key}"
    # Labeled prometheus_client histograms have no samples until their first
    # observation, so a valid pre-request scrape can omit the whole family.
    before_count = _metric_total(
        before,
        f"{base}_count",
        allow_missing=True,
    )
    after_count = _metric_total(after, f"{base}_count")
    before_sum = _metric_total(before, f"{base}_sum", allow_missing=True)
    after_sum = _metric_total(after, f"{base}_sum")
    count_delta = after_count - before_count
    sum_delta = after_sum - before_sum
    if count_delta < 0 or sum_delta < -1e-12 or not count_delta.is_integer():
        raise ValueError(f"connector histogram moved backwards: {key}")
    count = int(count_delta)
    seconds = max(0.0, sum_delta)
    return {
        "count": count,
        "sum_seconds": seconds,
        "mean_seconds": seconds / count if count else 0.0,
    }


def _require_sequence(value: object, name: str, length: int) -> list[Any]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"invalid {name}")
    return value


def analyze(run_dir: Path) -> dict[str, object]:
    verdict_path = run_dir / "connector-presence-verdict.json"
    startup_path = run_dir / "connector" / "metrics-startup.prom"
    after_path = run_dir / "connector" / "metrics-after.prom"
    verdict = _read_json(verdict_path)
    if (
        verdict.get("schema_version") != 1
        or verdict.get("contract") != VERDICT_CONTRACT
    ):
        raise ValueError("invalid connector-presence verdict identity")

    latency = verdict.get("latency")
    if not isinstance(latency, dict):
        raise ValueError("connector-presence verdict has no latency object")
    baseline_turn_means = _require_sequence(
        latency.get("baseline_turn_mean_seconds"),
        "baseline turn means",
        3,
    )
    connector_turns = _require_sequence(
        latency.get("connector_turn_seconds"),
        "connector turn latencies",
        3,
    )
    baseline_first = _positive_number(
        baseline_turn_means[0], "baseline first-turn latency"
    )
    connector_first = _positive_number(
        connector_turns[0], "connector first-turn latency"
    )
    cold_excess = connector_first - baseline_first

    signatures = verdict.get("response_signatures")
    if not isinstance(signatures, dict):
        raise ValueError("connector-presence verdict has no signatures")
    baseline_a = _require_sequence(signatures.get("baseline_a"), "baseline A", 3)
    baseline_b = _require_sequence(signatures.get("baseline_b"), "baseline B", 3)
    connector = _require_sequence(signatures.get("connector"), "connector", 3)
    if baseline_a != baseline_b:
        raise ValueError("baseline signatures are not stable")
    if not all(isinstance(item, dict) for item in (*baseline_a, *connector)):
        raise ValueError("response signature entry is invalid")

    first_output_divergence: int | None = None
    first_request_divergence: int | None = None
    for index, (baseline_item, connector_item) in enumerate(
        zip(baseline_a, connector, strict=True),
        start=1,
    ):
        if first_output_divergence is None and baseline_item.get(
            "output_digest"
        ) != connector_item.get("output_digest"):
            first_output_divergence = index
        if first_request_divergence is None and baseline_item.get(
            "request_digest"
        ) != connector_item.get("request_digest"):
            first_request_divergence = index

    before_metrics = _read_text(startup_path)
    after_metrics = _read_text(after_path)
    stages = {
        key: _histogram_delta(before_metrics, after_metrics, key)
        for key in _LATENCY_KEYS
    }
    # Correction is nested inside transfer, so summing all five histograms
    # would double-count it.  These three are the non-overlapping connector
    # lifecycle stages recorded around scheduler lookup, worker load, and
    # synchronous writeback.
    primary_stage_sum = sum(
        float(stages[key]["sum_seconds"]) for key in _PRIMARY_LATENCY_KEYS
    )
    store_seconds = float(stages["store_latency_seconds"]["sum_seconds"])
    store_share = store_seconds / cold_excess if cold_excess > 0 else None
    primary_stage_share = primary_stage_sum / cold_excess if cold_excess > 0 else None

    connector_counters = _counter_mapping(
        verdict.get("connector_counters"),
        _CONNECTOR_COUNTER_KEYS,
        "connector counters",
    )
    store_counters = _counter_mapping(
        verdict.get("connector_store_counters"),
        _STORE_COUNTER_KEYS,
        "connector store counters",
    )
    warmup = verdict.get("connector_warmup")
    if not isinstance(warmup, dict):
        raise ValueError("connector warmup evidence is missing")
    warmup_store = _counter_mapping(
        warmup.get("connector_store_counters"),
        _STORE_COUNTER_KEYS,
        "warmup store counters",
    )
    if any(warmup_store.values()):
        raise ValueError("warmup unexpectedly performed connector storage")

    cold_signature_equal = baseline_a[0] == connector[0]
    same_request_at_output_divergence = (
        first_output_divergence is not None
        and baseline_a[first_output_divergence - 1].get("request_digest")
        == connector[first_output_divergence - 1].get("request_digest")
    )
    store_dominates = (
        cold_excess > 0
        and cold_signature_equal
        and store_counters["store_tokens_eligible"] > 0
        and store_counters["store_tokens_completed"]
        == store_counters["store_tokens_eligible"]
        and store_counters["store_fallbacks"] == 0
        and int(stages["store_latency_seconds"]["count"]) > 0
        and store_share is not None
        and store_share >= 0.8
    )
    status = (
        "RECORDED_STORE_DOMINATES_COLD_EXCESS"
        if store_dominates
        else "RECORDED_STORE_DOES_NOT_DOMINATE_COLD_EXCESS"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT,
        "status": status,
        "run_dir": str(run_dir),
        "input_sha256": {
            "verdict": _sha256(verdict_path),
            "connector_metrics_startup": _sha256(startup_path),
            "connector_metrics_after": _sha256(after_path),
        },
        "cold_turn": {
            "signatures_equal": cold_signature_equal,
            "baseline_mean_seconds": baseline_first,
            "connector_seconds": connector_first,
            "excess_seconds": cold_excess,
            "ratio": connector_first / baseline_first,
        },
        "connector_stage_latency": stages,
        "attribution": {
            "policy": "store_time_at_least_80_percent_of_cold_turn_excess",
            "measurement_window": "connector warmup plus three workload turns",
            "primary_stage_keys": list(_PRIMARY_LATENCY_KEYS),
            "primary_stage_sum_seconds": primary_stage_sum,
            "store_sum_seconds": store_seconds,
            "primary_stage_share_of_cold_excess": primary_stage_share,
            "store_share_of_cold_excess": store_share,
        },
        "trajectory": {
            "first_output_divergence_turn": first_output_divergence,
            "same_request_at_first_output_divergence": (
                same_request_at_output_divergence
            ),
            "first_request_divergence_turn": first_request_divergence,
        },
        "connector_counters": connector_counters,
        "connector_store_counters": store_counters,
        "warmup_store_counters": warmup_store,
        "next_action": (
            "Run a transfer-enabled no-store arm against the same fixed transcript."
        ),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError("connector stage diagnostic already exists")
    report = analyze(args.run_dir.resolve())
    rendered = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with args.output.open("x", encoding="utf-8") as output:
        output.write(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
