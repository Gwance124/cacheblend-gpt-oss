#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gate one batched CUDA block-index owner against the measured 24-owner run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CONTRACT = "cacheblend-gpt-oss-batched-block-index-owner-v1"
BLOCK_CONTRACT = "cacheblend-gpt-oss-block-index-view-breakdown-v1"
PREFLIGHT_CONTRACT = "cacheblend-gpt-oss-store-preflight-preparation-breakdown-v1"
VERDICT_CONTRACT = "cacheblend-gpt-oss-connector-presence-equivalence-v1"
MINIMUM_CONSTRUCTION_RECOVERY_FRACTION = 0.8
MINIMUM_PREFLIGHT_RECOVERY_FRACTION = 0.8
MAXIMUM_COLD_RATIO = 2.0
PINNED_BASELINE_INDEX_CONSTRUCTIONS = 24
PINNED_OWNER_CONSTRUCTIONS = 1
PINNED_ROW_VIEWS = 24
PINNED_STAGING_VIEWS = 48
CONSTRUCTION_KEY = "store_preflight_block_index_construction_latency_seconds"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON artifact: {path}") from exc
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


def _sequence(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"invalid sequence: {label}")
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


def _validate_identity(
    *,
    run_dir: Path,
    block: dict[str, Any],
    preflight: dict[str, Any],
    verdict: dict[str, Any],
    label: str,
) -> None:
    if (
        block.get("schema_version") != SCHEMA_VERSION
        or block.get("contract") != BLOCK_CONTRACT
        or preflight.get("schema_version") != SCHEMA_VERSION
        or preflight.get("contract") != PREFLIGHT_CONTRACT
        or verdict.get("schema_version") != SCHEMA_VERSION
        or verdict.get("contract") != VERDICT_CONTRACT
    ):
        raise ValueError(f"invalid batched-index input identity: {label}")
    verdict_path = run_dir / "connector-presence-verdict.json"
    preflight_path = run_dir / "connector-store-preflight-breakdown.json"
    block_inputs = _mapping(block.get("input_sha256"), f"{label}.block.input_sha256")
    preflight_inputs = _mapping(
        preflight.get("input_sha256"), f"{label}.preflight.input_sha256"
    )
    if (
        block_inputs.get("verdict") != _sha256(verdict_path)
        or block_inputs.get("store_preflight_breakdown") != _sha256(preflight_path)
        or preflight_inputs.get("verdict") != _sha256(verdict_path)
    ):
        raise ValueError(f"batched-index input digest mismatch: {label}")


def _read_run(run_dir: Path, label: str) -> dict[str, object]:
    block_path = run_dir / "connector-block-index-view-breakdown.json"
    preflight_path = run_dir / "connector-store-preflight-breakdown.json"
    verdict_path = run_dir / "connector-presence-verdict.json"
    block = _read_json(block_path)
    preflight = _read_json(preflight_path)
    verdict = _read_json(verdict_path)
    _validate_identity(
        run_dir=run_dir,
        block=block,
        preflight=preflight,
        verdict=verdict,
        label=label,
    )
    return {
        "block": block,
        "preflight": preflight,
        "verdict": verdict,
        "paths": {
            "block": block_path,
            "preflight": preflight_path,
            "verdict": verdict_path,
        },
    }


def _logical_geometry(block: dict[str, Any]) -> dict[str, int]:
    geometry = _mapping(block.get("operation_geometry"), "operation_geometry")
    keys = (
        "stored_tokens",
        "expected_prepared_copy_operations",
        "preflight_prepared_copy_operations",
        "preflight_submitted_copy_operations",
    )
    return {key: _integer(geometry.get(key), key) for key in keys}


def _construction_seconds(block: dict[str, Any]) -> float:
    envelope = _mapping(block.get("block_index_view"), "block_index_view")
    subphases = _mapping(envelope.get("subphase_latency"), "subphase_latency")
    metric = _mapping(subphases.get(CONSTRUCTION_KEY), CONSTRUCTION_KEY)
    return _number(metric.get("sum_seconds"), f"{CONSTRUCTION_KEY}.sum_seconds")


def _preflight_seconds(preflight: dict[str, Any]) -> float:
    preparation = _mapping(preflight.get("preparation"), "preparation")
    enclosing = _mapping(preparation.get("enclosing_latency"), "enclosing_latency")
    return _number(enclosing.get("sum_seconds"), "preparation.sum_seconds")


def _candidate_mechanics(block: dict[str, Any]) -> dict[str, int | bool]:
    geometry = _mapping(block.get("operation_geometry"), "candidate geometry")
    observed = {
        "block_index_owner_constructions_per_store_batch": _integer(
            geometry.get("observed_block_index_owner_constructions_per_store_batch"),
            "observed owner constructions",
        ),
        "block_index_row_views_per_store_batch": _integer(
            geometry.get("observed_block_index_row_views_per_store_batch"),
            "observed row views",
        ),
        "staging_view_constructions_per_store_batch": _integer(
            geometry.get("observed_staging_view_constructions_per_store_batch"),
            "observed staging views",
        ),
    }
    observed["exact"] = observed == {
        "block_index_owner_constructions_per_store_batch": (PINNED_OWNER_CONSTRUCTIONS),
        "block_index_row_views_per_store_batch": PINNED_ROW_VIEWS,
        "staging_view_constructions_per_store_batch": PINNED_STAGING_VIEWS,
    }
    return observed


def _baseline_mechanics(block: dict[str, Any]) -> dict[str, int | bool]:
    geometry = _mapping(block.get("operation_geometry"), "baseline geometry")
    constructions = _integer(
        geometry.get("expected_index_tensor_constructions_per_store_batch"),
        "baseline index constructions",
    )
    staging_views = _integer(
        geometry.get("expected_staging_view_constructions_per_store_batch"),
        "baseline staging views",
    )
    return {
        "index_tensor_constructions_per_store_batch": constructions,
        "staging_view_constructions_per_store_batch": staging_views,
        "exact": (
            constructions == PINNED_BASELINE_INDEX_CONSTRUCTIONS
            and staging_views == PINNED_STAGING_VIEWS
        ),
    }


def _cold_latency(verdict: dict[str, Any]) -> dict[str, float]:
    latency = _mapping(verdict.get("latency"), "latency")
    controls = _sequence(
        latency.get("baseline_turn_mean_seconds"), "baseline turn means"
    )
    connector = _sequence(latency.get("connector_turn_seconds"), "connector turns")
    if not controls or not connector:
        raise ValueError("missing cold-turn latency")
    control_seconds = _number(controls[0], "cold control")
    connector_seconds = _number(connector[0], "cold connector")
    return {
        "control_mean_seconds": control_seconds,
        "connector_seconds": connector_seconds,
        "excess_seconds": max(0.0, connector_seconds - control_seconds),
        "ratio": connector_seconds / control_seconds if control_seconds else 0.0,
    }


def _cold_signature(verdict: dict[str, Any]) -> dict[str, Any]:
    signatures = _mapping(verdict.get("response_signatures"), "response_signatures")
    connector = _sequence(signatures.get("connector"), "connector signatures")
    if not connector or not isinstance(connector[0], dict):
        raise ValueError("missing connector cold signature")
    return connector[0]


def analyze(baseline_run_dir: Path, candidate_run_dir: Path) -> dict[str, object]:
    baseline = _read_run(baseline_run_dir, "baseline")
    candidate = _read_run(candidate_run_dir, "candidate")
    baseline_block = _mapping(baseline["block"], "baseline.block")
    candidate_block = _mapping(candidate["block"], "candidate.block")
    baseline_preflight = _mapping(baseline["preflight"], "baseline.preflight")
    candidate_preflight = _mapping(candidate["preflight"], "candidate.preflight")
    baseline_verdict = _mapping(baseline["verdict"], "baseline.verdict")
    candidate_verdict = _mapping(candidate["verdict"], "candidate.verdict")

    baseline_geometry = _logical_geometry(baseline_block)
    candidate_geometry = _logical_geometry(candidate_block)
    geometry_equal = baseline_geometry == candidate_geometry
    baseline_mechanics = _baseline_mechanics(baseline_block)
    candidate_mechanics = _candidate_mechanics(candidate_block)

    baseline_construction = _construction_seconds(baseline_block)
    candidate_construction = _construction_seconds(candidate_block)
    if baseline_construction <= 0.0:
        raise ValueError("baseline block-index construction was not measured")
    construction_recovered = max(0.0, baseline_construction - candidate_construction)
    construction_recovery_fraction = construction_recovered / baseline_construction

    baseline_preflight_seconds = _preflight_seconds(baseline_preflight)
    candidate_preflight_seconds = _preflight_seconds(candidate_preflight)
    preflight_recovered = max(
        0.0, baseline_preflight_seconds - candidate_preflight_seconds
    )
    preflight_recovery_fraction = preflight_recovered / baseline_construction

    baseline_cold = _cold_latency(baseline_verdict)
    candidate_cold = _cold_latency(candidate_verdict)
    cold_signature_equal = _cold_signature(baseline_verdict) == _cold_signature(
        candidate_verdict
    )
    store_counters_equal = baseline_verdict.get(
        "connector_store_counters"
    ) == candidate_verdict.get("connector_store_counters")
    workload_equal = (
        baseline_verdict.get("long_context_reached") is True
        and candidate_verdict.get("long_context_reached") is True
        and baseline_verdict.get("prefix_reuse_all") is True
        and candidate_verdict.get("prefix_reuse_all") is True
    )
    passed = (
        geometry_equal
        and baseline_mechanics["exact"] is True
        and candidate_mechanics["exact"] is True
        and construction_recovery_fraction >= MINIMUM_CONSTRUCTION_RECOVERY_FRACTION
        and preflight_recovery_fraction >= MINIMUM_PREFLIGHT_RECOVERY_FRACTION
        and candidate_cold["ratio"] <= MAXIMUM_COLD_RATIO
        and cold_signature_equal
        and store_counters_equal
        and workload_equal
    )

    baseline_paths = _mapping(baseline["paths"], "baseline.paths")
    candidate_paths = _mapping(candidate["paths"], "candidate.paths")
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT,
        "status": (
            "BATCHED_BLOCK_INDEX_OWNER_RECOVERED_CONSTRUCTION"
            if passed
            else "BATCHED_BLOCK_INDEX_OWNER_GATE_FAILED"
        ),
        "passed": passed,
        "run_dirs": {
            "baseline": str(baseline_run_dir),
            "candidate": str(candidate_run_dir),
        },
        "input_sha256": {
            "baseline_block_index_view": _sha256(Path(baseline_paths["block"])),
            "baseline_preflight": _sha256(Path(baseline_paths["preflight"])),
            "baseline_verdict": _sha256(Path(baseline_paths["verdict"])),
            "candidate_block_index_view": _sha256(Path(candidate_paths["block"])),
            "candidate_preflight": _sha256(Path(candidate_paths["preflight"])),
            "candidate_verdict": _sha256(Path(candidate_paths["verdict"])),
        },
        "matched_evidence": {
            "logical_geometry_equal": geometry_equal,
            "cold_connector_signature_equal": cold_signature_equal,
            "connector_store_counters_equal": store_counters_equal,
            "long_context_and_prefix_reuse": workload_equal,
        },
        "mechanics": {
            "baseline": baseline_mechanics,
            "candidate": candidate_mechanics,
        },
        "latency": {
            "baseline_block_index_construction_seconds": baseline_construction,
            "candidate_block_index_construction_seconds": candidate_construction,
            "block_index_construction_recovered_seconds": construction_recovered,
            "block_index_construction_recovery_fraction": (
                construction_recovery_fraction
            ),
            "minimum_construction_recovery_fraction": (
                MINIMUM_CONSTRUCTION_RECOVERY_FRACTION
            ),
            "baseline_preflight_preparation_seconds": baseline_preflight_seconds,
            "candidate_preflight_preparation_seconds": candidate_preflight_seconds,
            "preflight_preparation_recovered_seconds": preflight_recovered,
            "preflight_recovery_fraction_of_baseline_construction": (
                preflight_recovery_fraction
            ),
            "minimum_preflight_recovery_fraction": (
                MINIMUM_PREFLIGHT_RECOVERY_FRACTION
            ),
            "baseline_cold": baseline_cold,
            "candidate_cold": candidate_cold,
            "maximum_cold_ratio": MAXIMUM_COLD_RATIO,
        },
        "correctness_context": {
            "baseline_presence_status": baseline_verdict.get("status"),
            "candidate_presence_status": candidate_verdict.get("status"),
            "batched_index_gate_is_not_output_correctness_evidence": True,
        },
        "next_action": (
            "Use the candidate preflight artifact to select any remaining measured "
            "store stage; retain the independent long-context output verdict."
        ),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError("batched block-index artifact already exists")
    report = analyze(
        args.baseline_run_dir.resolve(),
        args.candidate_run_dir.resolve(),
    )
    rendered = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with args.output.open("x", encoding="utf-8") as output:
        output.write(rendered)
    print(rendered, end="")
    return 0 if report["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
