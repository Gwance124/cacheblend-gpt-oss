#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decompose block-index and staging-view preparation into exact subphases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CONTRACT = "cacheblend-gpt-oss-block-index-view-breakdown-v1"
VERDICT_CONTRACT = "cacheblend-gpt-oss-connector-presence-equivalence-v1"
DATA_PLANE_CONTRACT = "cacheblend-gpt-oss-store-data-plane-phase-breakdown-v1"
PREFLIGHT_CONTRACT = "cacheblend-gpt-oss-store-preflight-preparation-breakdown-v1"
ENCLOSING_KEY = "store_preflight_block_index_view_latency_seconds"
SUBPHASE_KEYS = (
    "store_preflight_block_index_construction_latency_seconds",
    "store_preflight_block_index_validation_latency_seconds",
    "store_preflight_staging_view_construction_latency_seconds",
    "store_preflight_staging_view_validation_latency_seconds",
)
UNATTRIBUTED_KEY = "unattributed_block_index_view_latency_seconds"
PINNED_STORED_TOKENS = 19_968
PINNED_PREPARED_COPY_OPERATIONS = 59_904
PINNED_SUBMITTED_COPY_OPERATIONS = 48
PINNED_LAYER_COUNT = 24
PINNED_KV_COMPONENTS = 2


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


def _validate_identity(
    *,
    verdict: dict[str, Any],
    data_plane: dict[str, Any],
    preflight: dict[str, Any],
    verdict_path: Path,
    data_plane_path: Path,
    startup_path: Path,
    after_path: Path,
) -> None:
    identities = (
        (verdict, VERDICT_CONTRACT, "connector-presence verdict"),
        (data_plane, DATA_PLANE_CONTRACT, "store data-plane breakdown"),
        (preflight, PREFLIGHT_CONTRACT, "store preflight breakdown"),
    )
    if any(
        artifact.get("schema_version") != SCHEMA_VERSION
        or artifact.get("contract") != contract
        for artifact, contract, _label in identities
    ):
        raise ValueError("invalid block-index/view input artifact identity")
    inputs = _mapping(preflight.get("input_sha256"), "preflight.input_sha256")
    expected = {
        "verdict": _sha256(verdict_path),
        "store_data_plane_breakdown": _sha256(data_plane_path),
        "connector_metrics_startup": _sha256(startup_path),
        "connector_metrics_after": _sha256(after_path),
    }
    if any(inputs.get(key) != digest for key, digest in expected.items()):
        raise ValueError("store preflight input digest mismatch")


def _geometry(preflight: dict[str, Any]) -> dict[str, int]:
    value = _mapping(preflight.get("operation_geometry"), "operation_geometry")
    observed = {
        "stored_tokens": _integer(value.get("stored_tokens"), "stored_tokens"),
        "expected_prepared_copy_operations": _integer(
            value.get("expected_prepared_copy_operations"),
            "expected_prepared_copy_operations",
        ),
        "preflight_prepared_copy_operations": _integer(
            value.get("preflight_prepared_copy_operations"),
            "preflight_prepared_copy_operations",
        ),
        "preflight_submitted_copy_operations": _integer(
            value.get("preflight_submitted_copy_operations"),
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
        raise ValueError("block-index/view geometry does not match pinned fast path")
    return observed


def _recorded_enclosing(preflight: dict[str, Any]) -> dict[str, int | float]:
    preparation = _mapping(preflight.get("preparation"), "preparation")
    subphases = _mapping(preparation.get("subphase_latency"), "subphase_latency")
    value = _mapping(subphases.get(ENCLOSING_KEY), ENCLOSING_KEY)
    return {
        "count": _integer(value.get("count"), f"{ENCLOSING_KEY}.count"),
        "sum_seconds": _number(
            value.get("sum_seconds"), f"{ENCLOSING_KEY}.sum_seconds"
        ),
        "mean_seconds": _number(
            value.get("mean_seconds"), f"{ENCLOSING_KEY}.mean_seconds"
        ),
    }


def _next_action(dominant: str) -> str:
    actions = {
        "store_preflight_block_index_construction_latency_seconds": (
            "Replace 24 per-layer CUDA index-tensor constructions with one validated "
            "batched index owner, then rerun the fixed transcript."
        ),
        "store_preflight_block_index_validation_latency_seconds": (
            "Separate invariant index metadata from per-layer validation before "
            "changing the fail-closed checks."
        ),
        "store_preflight_staging_view_construction_latency_seconds": (
            "Construct the 48 contiguous staging destinations from one validated "
            "base view, then rerun the fixed transcript."
        ),
        "store_preflight_staging_view_validation_latency_seconds": (
            "Hoist invariant staging owner checks while retaining exact per-view "
            "shape validation, then rerun the fixed transcript."
        ),
        UNATTRIBUTED_KEY: (
            "Instrument prepared-operation assembly before changing block-index or "
            "staging-view construction."
        ),
    }
    return actions[dominant]


def analyze(run_dir: Path) -> dict[str, object]:
    verdict_path = run_dir / "connector-presence-verdict.json"
    data_plane_path = run_dir / "connector-store-data-plane-breakdown.json"
    preflight_path = run_dir / "connector-store-preflight-breakdown.json"
    startup_path = run_dir / "connector" / "metrics-startup.prom"
    after_path = run_dir / "connector" / "metrics-after.prom"
    verdict = _read_json(verdict_path)
    data_plane = _read_json(data_plane_path)
    preflight = _read_json(preflight_path)
    _validate_identity(
        verdict=verdict,
        data_plane=data_plane,
        preflight=preflight,
        verdict_path=verdict_path,
        data_plane_path=data_plane_path,
        startup_path=startup_path,
        after_path=after_path,
    )
    geometry = _geometry(preflight)

    before = _read_text(startup_path)
    after = _read_text(after_path)
    enclosing = _histogram_delta(before, after, ENCLOSING_KEY)
    if enclosing != _recorded_enclosing(preflight):
        raise ValueError("block-index/view envelope does not match preflight artifact")
    observation_count = int(enclosing["count"])
    if observation_count <= 0:
        raise ValueError("block-index/view envelope has no observations")

    subphases = {key: _histogram_delta(before, after, key) for key in SUBPHASE_KEYS}
    if any(int(value["count"]) != observation_count for value in subphases.values()):
        raise ValueError("block-index/view observation counts do not reconcile")
    enclosing_seconds = float(enclosing["sum_seconds"])
    subphase_sum = sum(float(value["sum_seconds"]) for value in subphases.values())
    tolerance = max(1e-9, enclosing_seconds * 1e-6)
    if subphase_sum > enclosing_seconds + tolerance:
        raise ValueError("block-index/view subphases exceed enclosing time")

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
        "status": "CAPTURED_BLOCK_INDEX_VIEW_BREAKDOWN",
        "run_dir": str(run_dir),
        "input_sha256": {
            "verdict": _sha256(verdict_path),
            "store_data_plane_breakdown": _sha256(data_plane_path),
            "store_preflight_breakdown": _sha256(preflight_path),
            "connector_metrics_startup": _sha256(startup_path),
            "connector_metrics_after": _sha256(after_path),
        },
        "measurement_window": "connector warmup plus three workload turns",
        "correctness_context": {
            "presence_verdict_status": verdict.get("status"),
            "baseline_outputs_stable": verdict.get("baseline_outputs_stable"),
            "connector_outputs_match": verdict.get("connector_outputs_match"),
            "block_index_view_breakdown_is_not_output_correctness_evidence": True,
        },
        "operation_geometry": {
            **geometry,
            "expected_index_tensor_constructions_per_store_batch": PINNED_LAYER_COUNT,
            "expected_staging_view_constructions_per_store_batch": (
                PINNED_LAYER_COUNT * PINNED_KV_COMPONENTS
            ),
        },
        "block_index_view": {
            "enclosing_latency": enclosing,
            "subphase_latency": subphases,
            "subphase_sum_seconds": subphase_sum,
            "unattributed_enclosing_seconds": residual,
            "component_share_of_enclosing": shares,
            "dominant_component": dominant,
            "dominant_component_seconds": component_seconds[dominant],
            "dominant_component_share": shares[dominant],
        },
        "next_action": _next_action(dominant),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError("block-index/view breakdown artifact already exists")
    report = analyze(args.run_dir.resolve())
    rendered = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with args.output.open("x", encoding="utf-8") as output:
        output.write(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
