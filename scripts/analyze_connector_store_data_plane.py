#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decompose connector store preflight and gather into worker phases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CONTRACT = "cacheblend-gpt-oss-store-data-plane-phase-breakdown-v1"
VERDICT_CONTRACT = "cacheblend-gpt-oss-connector-presence-equivalence-v1"

PREFLIGHT_ENCLOSING_KEY = "store_preflight_latency_seconds"
GATHER_ENCLOSING_KEY = "store_gather_latency_seconds"
STORAGE_PREFLIGHT_KEY = "store_storage_preflight_latency_seconds"
PREFLIGHT_PHASE_KEYS = (
    "store_preflight_prepare_latency_seconds",
    "store_preflight_enqueue_latency_seconds",
    "store_preflight_synchronize_latency_seconds",
)
GATHER_PHASE_KEYS = (
    "store_gather_prepare_latency_seconds",
    "store_gather_enqueue_latency_seconds",
    "store_gather_synchronize_latency_seconds",
)
PREFLIGHT_OPERATION_KEY = "store_preflight_prepared_copy_operations"
GATHER_OPERATION_KEY = "store_gather_prepared_copy_operations"
PREFLIGHT_SUBMITTED_OPERATION_KEY = "store_preflight_submitted_copy_operations"
GATHER_SUBMITTED_OPERATION_KEY = "store_gather_submitted_copy_operations"

# Exact pinned GPT-OSS-20B/vLLM 0.19.1 geometry. Each prepared operation moves
# either K or V for one 16-token physical block span and one model layer.
PINNED_BLOCK_SIZE = 16
PINNED_LAYER_COUNT = 24
KV_COMPONENT_COUNT = 2
KV_ROW_WIDTH = 8 * 64
KV_ELEMENT_BYTES = 2
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


def _counter_delta(before: str, after: str, key: str) -> int:
    name = f"vllm:cacheblend_{key}_total"
    before_total = _metric_total(before, name, allow_missing=True)
    after_total = _metric_total(after, name)
    delta = after_total - before_total
    if delta < 0 or not delta.is_integer():
        raise ValueError(f"connector counter moved backwards: {key}")
    return int(delta)


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


def _phase_sum(
    phases: dict[str, dict[str, int | float]], keys: tuple[str, ...]
) -> float:
    return sum(float(phases[key]["sum_seconds"]) for key in keys)


def _dominant(phases: dict[str, dict[str, int | float]], keys: tuple[str, ...]) -> str:
    return max(keys, key=lambda key: float(phases[key]["sum_seconds"]))


def _next_action(
    preflight_dominant: str,
    gather_dominant: str,
    submission_reduction_fraction: float,
) -> str:
    if preflight_dominant.endswith("synchronize_latency_seconds"):
        return (
            "Insert a CUDA event at model-forward completion and compare it with "
            "the read-only preflight synchronize boundary."
        )
    if preflight_dominant.endswith("prepare_latency_seconds"):
        if submission_reduction_fraction >= 0.99:
            return (
                "Split residual preflight preparation into span validation and "
                "block-index construction before changing the batched copy path."
            )
        return (
            "Replace duplicate read-only copy preparation with a reusable validated "
            "batch, then rerun the fixed transcript."
        )
    if gather_dominant.endswith("enqueue_latency_seconds"):
        return (
            "Prototype a bounded batched gather that reduces per-span CUDA dispatch, "
            "then rerun the fixed transcript."
        )
    if gather_dominant.endswith("synchronize_latency_seconds"):
        return (
            "Use CUDA events around gather copies to separate queued model work from "
            "copy execution before changing the gather implementation."
        )
    return (
        "Use the dominant measured phase as the boundary for the next data-plane "
        "experiment."
    )


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
    stored_tokens = store_counters["store_tokens_completed"]
    if (
        stored_tokens <= 0
        or stored_tokens != store_counters["store_tokens_eligible"]
        or store_counters["store_fallbacks"] != 0
        or stored_tokens % PINNED_BLOCK_SIZE != 0
    ):
        raise ValueError("run did not complete an aligned positive connector store")

    before = _read_text(startup_path)
    after = _read_text(after_path)
    preflight_enclosing = _histogram_delta(before, after, PREFLIGHT_ENCLOSING_KEY)
    gather_enclosing = _histogram_delta(before, after, GATHER_ENCLOSING_KEY)
    phase_keys = (
        *PREFLIGHT_PHASE_KEYS,
        STORAGE_PREFLIGHT_KEY,
        *GATHER_PHASE_KEYS,
    )
    phases = {key: _histogram_delta(before, after, key) for key in phase_keys}
    observation_count = int(preflight_enclosing["count"])
    if observation_count <= 0 or int(gather_enclosing["count"]) != observation_count:
        raise ValueError("data-plane enclosing observation counts do not reconcile")
    if any(int(phases[key]["count"]) != observation_count for key in phase_keys):
        raise ValueError("data-plane phase observation counts do not reconcile")

    preflight_phase_seconds = _phase_sum(phases, PREFLIGHT_PHASE_KEYS)
    storage_preflight_seconds = float(phases[STORAGE_PREFLIGHT_KEY]["sum_seconds"])
    preflight_attributed_seconds = preflight_phase_seconds + storage_preflight_seconds
    gather_phase_seconds = _phase_sum(phases, GATHER_PHASE_KEYS)
    preflight_enclosing_seconds = float(preflight_enclosing["sum_seconds"])
    gather_enclosing_seconds = float(gather_enclosing["sum_seconds"])
    preflight_tolerance = max(1e-9, preflight_enclosing_seconds * 1e-6)
    gather_tolerance = max(1e-9, gather_enclosing_seconds * 1e-6)
    if preflight_attributed_seconds > preflight_enclosing_seconds + preflight_tolerance:
        raise ValueError("preflight phases exceed enclosing preflight time")
    if gather_phase_seconds > gather_enclosing_seconds + gather_tolerance:
        raise ValueError("gather phases exceed enclosing gather time")

    preflight_operations = _counter_delta(before, after, PREFLIGHT_OPERATION_KEY)
    gather_operations = _counter_delta(before, after, GATHER_OPERATION_KEY)
    preflight_submitted_operations = _counter_delta(
        before,
        after,
        PREFLIGHT_SUBMITTED_OPERATION_KEY,
    )
    gather_submitted_operations = _counter_delta(
        before,
        after,
        GATHER_SUBMITTED_OPERATION_KEY,
    )
    expected_operations = (
        stored_tokens // PINNED_BLOCK_SIZE * PINNED_LAYER_COUNT * KV_COMPONENT_COUNT
    )
    if (
        preflight_operations != expected_operations
        or gather_operations != expected_operations
    ):
        raise ValueError("prepared-copy operation count does not match pinned geometry")
    if (
        preflight_submitted_operations <= 0
        or gather_submitted_operations != preflight_submitted_operations
        or preflight_submitted_operations > expected_operations
    ):
        raise ValueError("submitted-copy operation count does not reconcile")

    payload_bytes = (
        stored_tokens
        * PINNED_LAYER_COUNT
        * KV_COMPONENT_COUNT
        * KV_ROW_WIDTH
        * KV_ELEMENT_BYTES
    )
    payload_gib = payload_bytes / (1024**3)
    submission_reduction_fraction = (
        1.0 - preflight_submitted_operations / expected_operations
    )
    preflight_dominant = _dominant(phases, PREFLIGHT_PHASE_KEYS)
    gather_dominant = _dominant(phases, GATHER_PHASE_KEYS)

    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT,
        "status": "CAPTURED_STORE_DATA_PLANE_PHASE_BREAKDOWN",
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
            "phase_breakdown_is_not_output_correctness_evidence": True,
        },
        "connector_store_counters": store_counters,
        "operation_geometry": {
            "block_size_tokens": PINNED_BLOCK_SIZE,
            "layer_count": PINNED_LAYER_COUNT,
            "kv_components_per_layer": KV_COMPONENT_COUNT,
            "kv_row_width_elements": KV_ROW_WIDTH,
            "kv_element_bytes": KV_ELEMENT_BYTES,
            "stored_tokens": stored_tokens,
            "expected_prepared_copy_operations": expected_operations,
            "preflight_prepared_copy_operations": preflight_operations,
            "gather_prepared_copy_operations": gather_operations,
            "preflight_submitted_copy_operations": (
                preflight_submitted_operations
            ),
            "gather_submitted_copy_operations": gather_submitted_operations,
            "submitted_copy_reduction_fraction": submission_reduction_fraction,
            "bytes_per_full_block_copy": (
                PINNED_BLOCK_SIZE * KV_ROW_WIDTH * KV_ELEMENT_BYTES
            ),
            "logical_payload_bytes": payload_bytes,
            "logical_payload_gib": payload_gib,
        },
        "preflight": {
            "enclosing_latency": preflight_enclosing,
            "phase_latency": {key: phases[key] for key in PREFLIGHT_PHASE_KEYS},
            "storage_preflight_latency": phases[STORAGE_PREFLIGHT_KEY],
            "data_plane_phase_sum_seconds": preflight_phase_seconds,
            "attributed_sum_seconds": preflight_attributed_seconds,
            "unattributed_enclosing_seconds": max(
                0.0, preflight_enclosing_seconds - preflight_attributed_seconds
            ),
            "dominant_data_plane_phase": preflight_dominant,
            "dominant_data_plane_phase_seconds": float(
                phases[preflight_dominant]["sum_seconds"]
            ),
        },
        "gather": {
            "enclosing_latency": gather_enclosing,
            "phase_latency": {key: phases[key] for key in GATHER_PHASE_KEYS},
            "phase_sum_seconds": gather_phase_seconds,
            "unattributed_enclosing_seconds": max(
                0.0, gather_enclosing_seconds - gather_phase_seconds
            ),
            "dominant_phase": gather_dominant,
            "dominant_phase_seconds": float(phases[gather_dominant]["sum_seconds"]),
            "logical_payload_gib_per_enclosing_second": (
                payload_gib / gather_enclosing_seconds
                if gather_enclosing_seconds > 0.0
                else 0.0
            ),
        },
        "next_action": _next_action(
            preflight_dominant,
            gather_dominant,
            submission_reduction_fraction,
        ),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError("store data-plane breakdown artifact already exists")
    report = analyze(args.run_dir.resolve())
    rendered = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with args.output.open("x", encoding="utf-8") as output:
        output.write(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
