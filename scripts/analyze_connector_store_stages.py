#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decompose synchronous connector writeback into bounded sub-stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CONTRACT = "cacheblend-gpt-oss-store-stage-breakdown-v1"
VERDICT_CONTRACT = "cacheblend-gpt-oss-connector-presence-equivalence-v1"
BROAD_STORE_KEY = "store_latency_seconds"
STORE_STAGE_KEYS = (
    "store_plan_latency_seconds",
    "store_preflight_latency_seconds",
    "store_gather_latency_seconds",
    "store_lmcache_latency_seconds",
    "store_sidecar_publish_latency_seconds",
)
PRIMARY_WRITE_STAGE_KEYS = (
    "store_gather_latency_seconds",
    "store_lmcache_latency_seconds",
    "store_sidecar_publish_latency_seconds",
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
    before_count = _metric_total(before, f"{base}_count", allow_missing=True)
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


def _store_counters(value: object) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != _STORE_COUNTER_KEYS:
        raise ValueError("invalid connector store counter schema")
    counters: dict[str, int] = {}
    for key in _STORE_COUNTER_KEYS:
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"invalid connector store counter: {key}")
        counters[key] = item
    return counters


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

    store_counters = _store_counters(verdict.get("connector_store_counters"))
    store_completed = (
        store_counters["store_tokens_eligible"] > 0
        and store_counters["store_tokens_completed"]
        == store_counters["store_tokens_eligible"]
        and store_counters["store_fallbacks"] == 0
    )
    if not store_completed:
        raise ValueError("run did not complete a positive connector store")

    before = _read_text(startup_path)
    after = _read_text(after_path)
    broad = _histogram_delta(before, after, BROAD_STORE_KEY)
    stages = {key: _histogram_delta(before, after, key) for key in STORE_STAGE_KEYS}
    broad_count = int(broad["count"])
    if broad_count <= 0 or any(
        int(stage["count"]) != broad_count for stage in stages.values()
    ):
        raise ValueError("store sub-stage observation counts do not reconcile")

    broad_seconds = float(broad["sum_seconds"])
    stage_sum = sum(float(stage["sum_seconds"]) for stage in stages.values())
    tolerance = max(1e-9, broad_seconds * 1e-6)
    if stage_sum > broad_seconds + tolerance:
        raise ValueError("store sub-stage time exceeds enclosing store time")
    residual = max(0.0, broad_seconds - stage_sum)
    dominant_key = max(
        STORE_STAGE_KEYS,
        key=lambda key: float(stages[key]["sum_seconds"]),
    )
    primary_write_sum = sum(
        float(stages[key]["sum_seconds"]) for key in PRIMARY_WRITE_STAGE_KEYS
    )
    shares = {
        key: (
            float(stages[key]["sum_seconds"]) / broad_seconds
            if broad_seconds > 0.0
            else 0.0
        )
        for key in STORE_STAGE_KEYS
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT,
        "status": "CAPTURED_SYNCHRONOUS_STORE_STAGE_BREAKDOWN",
        "run_dir": str(run_dir),
        "input_sha256": {
            "verdict": _sha256(verdict_path),
            "connector_metrics_startup": _sha256(startup_path),
            "connector_metrics_after": _sha256(after_path),
        },
        "measurement_window": "connector warmup plus three workload turns",
        "correctness_context": {
            "presence_verdict_status": verdict.get("status"),
            "baseline_outputs_stable": verdict.get("baseline_outputs_stable"),
            "connector_outputs_match": verdict.get("connector_outputs_match"),
            "stage_breakdown_is_not_output_correctness_evidence": True,
        },
        "connector_store_counters": store_counters,
        "enclosing_store_latency": broad,
        "store_stage_latency": stages,
        "decomposition": {
            "stage_sum_seconds": stage_sum,
            "unattributed_enclosing_seconds": residual,
            "stage_share_of_enclosing_store": shares,
            "primary_write_stage_keys": list(PRIMARY_WRITE_STAGE_KEYS),
            "primary_write_stage_sum_seconds": primary_write_sum,
            "primary_write_share_of_enclosing_store": (
                primary_write_sum / broad_seconds if broad_seconds > 0.0 else 0.0
            ),
            "dominant_stage": dominant_key,
            "dominant_stage_seconds": float(stages[dominant_key]["sum_seconds"]),
            "dominant_stage_share_of_enclosing_store": shares[dominant_key],
        },
        "next_action": (
            "Use the dominant measured sub-stage as the boundary for the next "
            "storage-path optimization experiment."
        ),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError("store-stage breakdown artifact already exists")
    report = analyze(args.run_dir.resolve())
    rendered = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with args.output.open("x", encoding="utf-8") as output:
        output.write(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
