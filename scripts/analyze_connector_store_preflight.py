#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decompose retained gather preparation into bounded read-only subphases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CONTRACT = "cacheblend-gpt-oss-store-preflight-preparation-breakdown-v1"
VERDICT_CONTRACT = "cacheblend-gpt-oss-connector-presence-equivalence-v1"
DATA_PLANE_CONTRACT = "cacheblend-gpt-oss-store-data-plane-phase-breakdown-v1"
PREPARE_KEY = "store_preflight_prepare_latency_seconds"
SUBPHASE_KEYS = (
    "store_preflight_input_materialization_latency_seconds",
    "store_preflight_span_validation_latency_seconds",
    "store_preflight_tensor_validation_latency_seconds",
    "store_preflight_range_validation_latency_seconds",
    "store_preflight_block_plan_latency_seconds",
    "store_preflight_block_index_view_latency_seconds",
    "store_preflight_legacy_view_latency_seconds",
)
UNATTRIBUTED_KEY = "unattributed_preparation_latency_seconds"
PINNED_STORED_TOKENS = 19_968
PINNED_PREPARED_COPY_OPERATIONS = 59_904
PINNED_SUBMITTED_COPY_OPERATIONS = 48
ZERO_PHASE_TOLERANCE_SECONDS = 1e-9


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


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"invalid mapping: {label}")
    return value


def _number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"invalid nonnegative number: {label}")
    return float(value)


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid nonnegative integer: {label}")
    return value


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


def _validate_artifact_identity(
    *,
    verdict: dict[str, Any],
    data_plane: dict[str, Any],
    verdict_path: Path,
    startup_path: Path,
    after_path: Path,
) -> None:
    if (
        verdict.get("schema_version") != SCHEMA_VERSION
        or verdict.get("contract") != VERDICT_CONTRACT
    ):
        raise ValueError("invalid connector-presence verdict identity")
    if (
        data_plane.get("schema_version") != SCHEMA_VERSION
        or data_plane.get("contract") != DATA_PLANE_CONTRACT
    ):
        raise ValueError("invalid store data-plane artifact identity")
    inputs = _mapping(data_plane.get("input_sha256"), "data_plane.input_sha256")
    expected = {
        "verdict": _sha256(verdict_path),
        "connector_metrics_startup": _sha256(startup_path),
        "connector_metrics_after": _sha256(after_path),
    }
    if any(inputs.get(key) != digest for key, digest in expected.items()):
        raise ValueError("store data-plane input digest mismatch")


def _validate_pinned_geometry(data_plane: dict[str, Any]) -> dict[str, int]:
    geometry = _mapping(data_plane.get("operation_geometry"), "operation_geometry")
    observed = {
        "stored_tokens": _integer(geometry.get("stored_tokens"), "stored_tokens"),
        "expected_prepared_copy_operations": _integer(
            geometry.get("expected_prepared_copy_operations"),
            "expected_prepared_copy_operations",
        ),
        "preflight_prepared_copy_operations": _integer(
            geometry.get("preflight_prepared_copy_operations"),
            "preflight_prepared_copy_operations",
        ),
        "preflight_submitted_copy_operations": _integer(
            geometry.get("preflight_submitted_copy_operations"),
            "preflight_submitted_copy_operations",
        ),
    }
    expected = {
        "stored_tokens": PINNED_STORED_TOKENS,
        "expected_prepared_copy_operations": PINNED_PREPARED_COPY_OPERATIONS,
        "preflight_prepared_copy_operations": PINNED_PREPARED_COPY_OPERATIONS,
        "preflight_submitted_copy_operations": PINNED_SUBMITTED_COPY_OPERATIONS,
    }
    if observed != expected:
        raise ValueError("store preflight geometry does not match the pinned fast path")
    return observed


def _recorded_prepare(data_plane: dict[str, Any]) -> dict[str, int | float]:
    preflight = _mapping(data_plane.get("preflight"), "preflight")
    phases = _mapping(preflight.get("phase_latency"), "preflight.phase_latency")
    recorded = _mapping(phases.get(PREPARE_KEY), PREPARE_KEY)
    return {
        "count": _integer(recorded.get("count"), f"{PREPARE_KEY}.count"),
        "sum_seconds": _number(
            recorded.get("sum_seconds"), f"{PREPARE_KEY}.sum_seconds"
        ),
        "mean_seconds": _number(
            recorded.get("mean_seconds"), f"{PREPARE_KEY}.mean_seconds"
        ),
    }


def _next_action(dominant: str) -> str:
    actions = {
        "store_preflight_input_materialization_latency_seconds": (
            "Split chunk-sequence materialization from tuple copying before changing "
            "the retained-batch representation."
        ),
        "store_preflight_span_validation_latency_seconds": (
            "Split canonical span validation into ordering, layer coverage, and "
            "range checks before changing validation semantics."
        ),
        "store_preflight_tensor_validation_latency_seconds": (
            "Split shared tensor-owner checks from per-span block-bound checks before "
            "changing the fail-closed validation path."
        ),
        "store_preflight_range_validation_latency_seconds": (
            "Split document-range and overlap checks from receipt construction before "
            "changing range validation."
        ),
        "store_preflight_block_plan_latency_seconds": (
            "Split aligned-span scanning from per-layer block-ID plan construction "
            "before changing the block plan."
        ),
        "store_preflight_block_index_view_latency_seconds": (
            "Split CUDA block-index tensor construction from staging-view validation "
            "before changing the batched copy path."
        ),
        UNATTRIBUTED_KEY: (
            "Instrument logical-operation accounting and prepared-batch publication "
            "before changing the measured data plane."
        ),
    }
    return actions.get(
        dominant,
        "Investigate the unexpected legacy-view preparation before optimizing it.",
    )


def analyze(run_dir: Path) -> dict[str, object]:
    verdict_path = run_dir / "connector-presence-verdict.json"
    data_plane_path = run_dir / "connector-store-data-plane-breakdown.json"
    startup_path = run_dir / "connector" / "metrics-startup.prom"
    after_path = run_dir / "connector" / "metrics-after.prom"
    verdict = _read_json(verdict_path)
    data_plane = _read_json(data_plane_path)
    _validate_artifact_identity(
        verdict=verdict,
        data_plane=data_plane,
        verdict_path=verdict_path,
        startup_path=startup_path,
        after_path=after_path,
    )
    geometry = _validate_pinned_geometry(data_plane)

    before = _read_text(startup_path)
    after = _read_text(after_path)
    enclosing = _histogram_delta(before, after, PREPARE_KEY)
    recorded = _recorded_prepare(data_plane)
    if enclosing != recorded:
        raise ValueError("preflight preparation does not match data-plane artifact")
    observation_count = int(enclosing["count"])
    if observation_count <= 0:
        raise ValueError("preflight preparation has no observations")

    subphases = {key: _histogram_delta(before, after, key) for key in SUBPHASE_KEYS}
    if any(int(value["count"]) != observation_count for value in subphases.values()):
        raise ValueError("preflight preparation observation counts do not reconcile")
    enclosing_seconds = float(enclosing["sum_seconds"])
    subphase_sum = sum(float(value["sum_seconds"]) for value in subphases.values())
    tolerance = max(ZERO_PHASE_TOLERANCE_SECONDS, enclosing_seconds * 1e-6)
    if subphase_sum > enclosing_seconds + tolerance:
        raise ValueError("preflight preparation subphases exceed enclosing time")
    legacy_seconds = float(
        subphases["store_preflight_legacy_view_latency_seconds"]["sum_seconds"]
    )
    if legacy_seconds > ZERO_PHASE_TOLERANCE_SECONDS:
        raise ValueError("block-batched fast path unexpectedly used legacy views")

    residual = max(0.0, enclosing_seconds - subphase_sum)
    component_seconds = {
        key: float(value["sum_seconds"]) for key, value in subphases.items()
    }
    component_seconds[UNATTRIBUTED_KEY] = residual
    dominant = max(component_seconds, key=component_seconds.__getitem__)
    shares = {
        key: seconds / enclosing_seconds if enclosing_seconds > 0.0 else 0.0
        for key, seconds in component_seconds.items()
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT,
        "status": "CAPTURED_STORE_PREFLIGHT_PREPARATION_BREAKDOWN",
        "run_dir": str(run_dir),
        "input_sha256": {
            "verdict": _sha256(verdict_path),
            "store_data_plane_breakdown": _sha256(data_plane_path),
            "connector_metrics_startup": _sha256(startup_path),
            "connector_metrics_after": _sha256(after_path),
        },
        "measurement_window": "connector warmup plus three workload turns",
        "correctness_context": {
            "presence_verdict_status": verdict.get("status"),
            "baseline_outputs_stable": verdict.get("baseline_outputs_stable"),
            "connector_outputs_match": verdict.get("connector_outputs_match"),
            "preparation_breakdown_is_not_output_correctness_evidence": True,
        },
        "operation_geometry": geometry,
        "preparation": {
            "enclosing_latency": enclosing,
            "subphase_latency": subphases,
            "subphase_sum_seconds": subphase_sum,
            "unattributed_enclosing_seconds": residual,
            "component_share_of_enclosing_preparation": shares,
            "dominant_component": dominant,
            "dominant_component_seconds": component_seconds[dominant],
            "dominant_component_share": shares[dominant],
            "legacy_fast_path_observed": legacy_seconds <= ZERO_PHASE_TOLERANCE_SECONDS,
        },
        "next_action": _next_action(dominant),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError("store-preflight breakdown artifact already exists")
    report = analyze(args.run_dir.resolve())
    rendered = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with args.output.open("x", encoding="utf-8") as output:
        output.write(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
