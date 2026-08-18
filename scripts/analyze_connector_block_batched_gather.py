#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gate block-batched gather against the measured prepared-reuse baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CONTRACT = "cacheblend-gpt-oss-block-batched-gather-v1"
PHASE_CONTRACT = "cacheblend-gpt-oss-store-data-plane-phase-breakdown-v1"
VERDICT_CONTRACT = "cacheblend-gpt-oss-connector-presence-equivalence-v1"
MINIMUM_PREPARE_RECOVERY_FRACTION = 0.8
MINIMUM_COLD_EXCESS_RECOVERY_FRACTION = 0.8
MINIMUM_SUBMISSION_REDUCTION_FRACTION = 0.99
ZERO_PHASE_TOLERANCE_SECONDS = 1e-9
PINNED_SUBMISSIONS_PER_BATCH = 24 * 2

PREFLIGHT_PREPARE_KEY = "store_preflight_prepare_latency_seconds"
PREFLIGHT_ENQUEUE_KEY = "store_preflight_enqueue_latency_seconds"
PREFLIGHT_SYNCHRONIZE_KEY = "store_preflight_synchronize_latency_seconds"
GATHER_PREPARE_KEY = "store_gather_prepare_latency_seconds"


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


def _phase_seconds(phase: dict[str, Any], section: str, key: str) -> float:
    section_value = _mapping(phase.get(section), section)
    phases = _mapping(section_value.get("phase_latency"), f"{section}.phase_latency")
    metric = _mapping(phases.get(key), key)
    return _number(metric.get("sum_seconds"), f"{key}.sum_seconds")


def _logical_geometry(phase: dict[str, Any]) -> dict[str, int]:
    geometry = _mapping(phase.get("operation_geometry"), "operation_geometry")
    expected = _integer(
        geometry.get("expected_prepared_copy_operations"),
        "expected_prepared_copy_operations",
    )
    preflight = _integer(
        geometry.get("preflight_prepared_copy_operations"),
        "preflight_prepared_copy_operations",
    )
    gather = _integer(
        geometry.get("gather_prepared_copy_operations"),
        "gather_prepared_copy_operations",
    )
    stored_tokens = _integer(geometry.get("stored_tokens"), "stored_tokens")
    payload_bytes = _integer(
        geometry.get("logical_payload_bytes"), "logical_payload_bytes"
    )
    if expected <= 0 or preflight != expected or gather != expected:
        raise ValueError("logical prepared-copy geometry does not reconcile")
    return {
        "expected_prepared_copy_operations": expected,
        "preflight_prepared_copy_operations": preflight,
        "gather_prepared_copy_operations": gather,
        "stored_tokens": stored_tokens,
        "logical_payload_bytes": payload_bytes,
    }


def _submitted_geometry(phase: dict[str, Any]) -> dict[str, int | float | bool]:
    geometry = _mapping(phase.get("operation_geometry"), "operation_geometry")
    logical = _integer(
        geometry.get("expected_prepared_copy_operations"),
        "expected_prepared_copy_operations",
    )
    preflight = _integer(
        geometry.get("preflight_submitted_copy_operations"),
        "preflight_submitted_copy_operations",
    )
    gather = _integer(
        geometry.get("gather_submitted_copy_operations"),
        "gather_submitted_copy_operations",
    )
    reduction = 1.0 - preflight / logical if logical else 0.0
    reconciled = (
        preflight > 0
        and preflight == gather
        and preflight % PINNED_SUBMISSIONS_PER_BATCH == 0
        and preflight <= logical
    )
    return {
        "preflight_submitted_copy_operations": preflight,
        "gather_submitted_copy_operations": gather,
        "submissions_per_batch": PINNED_SUBMISSIONS_PER_BATCH,
        "observed_store_batches": (
            preflight // PINNED_SUBMISSIONS_PER_BATCH if reconciled else 0
        ),
        "submission_reduction_fraction": reduction,
        "reconciled": reconciled,
    }


def _cold_latency(verdict: dict[str, Any]) -> dict[str, float]:
    latency = _mapping(verdict.get("latency"), "latency")
    controls = _sequence(
        latency.get("baseline_turn_mean_seconds"), "baseline_turn_mean_seconds"
    )
    connector = _sequence(
        latency.get("connector_turn_seconds"), "connector_turn_seconds"
    )
    if not controls or not connector:
        raise ValueError("missing cold-turn latency")
    control_seconds = _number(controls[0], "cold control mean")
    connector_seconds = _number(connector[0], "cold connector latency")
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


def _validate_identity(
    phase: dict[str, Any],
    verdict: dict[str, Any],
    verdict_path: Path,
    label: str,
) -> None:
    if phase.get("schema_version") != 1 or phase.get("contract") != PHASE_CONTRACT:
        raise ValueError(f"invalid phase artifact identity: {label}")
    if (
        verdict.get("schema_version") != 1
        or verdict.get("contract") != VERDICT_CONTRACT
    ):
        raise ValueError(f"invalid verdict artifact identity: {label}")
    inputs = _mapping(phase.get("input_sha256"), f"{label}.input_sha256")
    if inputs.get("verdict") != _sha256(verdict_path):
        raise ValueError(f"phase/verdict digest mismatch: {label}")


def analyze(baseline_run_dir: Path, candidate_run_dir: Path) -> dict[str, object]:
    baseline_phase_path = baseline_run_dir / "connector-store-data-plane-breakdown.json"
    baseline_verdict_path = baseline_run_dir / "connector-presence-verdict.json"
    candidate_phase_path = (
        candidate_run_dir / "connector-store-data-plane-breakdown.json"
    )
    candidate_verdict_path = candidate_run_dir / "connector-presence-verdict.json"
    baseline_phase = _read_json(baseline_phase_path)
    baseline_verdict = _read_json(baseline_verdict_path)
    candidate_phase = _read_json(candidate_phase_path)
    candidate_verdict = _read_json(candidate_verdict_path)
    _validate_identity(
        baseline_phase, baseline_verdict, baseline_verdict_path, "baseline"
    )
    _validate_identity(
        candidate_phase, candidate_verdict, candidate_verdict_path, "candidate"
    )

    baseline_geometry = _logical_geometry(baseline_phase)
    candidate_geometry = _logical_geometry(candidate_phase)
    geometry_equal = baseline_geometry == candidate_geometry
    submitted = _submitted_geometry(candidate_phase)
    submission_reduced = (
        submitted["reconciled"] is True
        and float(submitted["submission_reduction_fraction"])
        >= MINIMUM_SUBMISSION_REDUCTION_FRACTION
    )

    baseline_prepare = _phase_seconds(
        baseline_phase, "preflight", PREFLIGHT_PREPARE_KEY
    )
    candidate_prepare = _phase_seconds(
        candidate_phase, "preflight", PREFLIGHT_PREPARE_KEY
    )
    if baseline_prepare <= 0.0:
        raise ValueError("baseline did not measure positive preparation")
    prepare_recovered = max(0.0, baseline_prepare - candidate_prepare)
    prepare_recovery_fraction = prepare_recovered / baseline_prepare
    gather_prepare = _phase_seconds(candidate_phase, "gather", GATHER_PREPARE_KEY)
    preflight_enqueue = _phase_seconds(
        candidate_phase, "preflight", PREFLIGHT_ENQUEUE_KEY
    )
    preflight_synchronize = _phase_seconds(
        candidate_phase, "preflight", PREFLIGHT_SYNCHRONIZE_KEY
    )
    one_shot_reuse_preserved = (
        gather_prepare <= ZERO_PHASE_TOLERANCE_SECONDS
        and preflight_enqueue <= ZERO_PHASE_TOLERANCE_SECONDS
        and preflight_synchronize <= ZERO_PHASE_TOLERANCE_SECONDS
    )

    baseline_cold = _cold_latency(baseline_verdict)
    candidate_cold = _cold_latency(candidate_verdict)
    cold_recovered = max(
        0.0,
        baseline_cold["excess_seconds"] - candidate_cold["excess_seconds"],
    )
    cold_recovery_fraction = (
        cold_recovered / baseline_cold["excess_seconds"]
        if baseline_cold["excess_seconds"]
        else 0.0
    )
    cold_signatures_equal = _cold_signature(baseline_verdict) == _cold_signature(
        candidate_verdict
    )
    store_counters_equal = (
        baseline_verdict.get("connector_store_counters")
        == candidate_verdict.get("connector_store_counters")
    )
    workload_equal = (
        baseline_verdict.get("long_context_reached") is True
        and candidate_verdict.get("long_context_reached") is True
        and baseline_verdict.get("prefix_reuse_all") is True
        and candidate_verdict.get("prefix_reuse_all") is True
    )
    passed = (
        geometry_equal
        and submission_reduced
        and one_shot_reuse_preserved
        and prepare_recovery_fraction >= MINIMUM_PREPARE_RECOVERY_FRACTION
        and cold_recovery_fraction >= MINIMUM_COLD_EXCESS_RECOVERY_FRACTION
        and cold_signatures_equal
        and store_counters_equal
        and workload_equal
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT,
        "status": (
            "BLOCK_BATCHED_GATHER_RECOVERED_VIEW_PREPARATION"
            if passed
            else "BLOCK_BATCHED_GATHER_GATE_FAILED"
        ),
        "passed": passed,
        "run_dirs": {
            "baseline": str(baseline_run_dir),
            "candidate": str(candidate_run_dir),
        },
        "input_sha256": {
            "baseline_phase": _sha256(baseline_phase_path),
            "baseline_verdict": _sha256(baseline_verdict_path),
            "candidate_phase": _sha256(candidate_phase_path),
            "candidate_verdict": _sha256(candidate_verdict_path),
        },
        "matched_evidence": {
            "logical_geometry_equal": geometry_equal,
            "cold_connector_signature_equal": cold_signatures_equal,
            "connector_store_counters_equal": store_counters_equal,
            "long_context_and_prefix_reuse": workload_equal,
        },
        "operation_geometry": candidate_geometry | submitted,
        "mechanics": {
            "one_shot_reuse_preserved": one_shot_reuse_preserved,
            "candidate_gather_prepare_seconds": gather_prepare,
            "candidate_preflight_enqueue_seconds": preflight_enqueue,
            "candidate_preflight_synchronize_seconds": preflight_synchronize,
            "minimum_submission_reduction_fraction": (
                MINIMUM_SUBMISSION_REDUCTION_FRACTION
            ),
        },
        "latency": {
            "baseline_preflight_prepare_seconds": baseline_prepare,
            "candidate_preflight_prepare_seconds": candidate_prepare,
            "preflight_prepare_recovered_seconds": prepare_recovered,
            "preflight_prepare_recovery_fraction": prepare_recovery_fraction,
            "minimum_prepare_recovery_fraction": (
                MINIMUM_PREPARE_RECOVERY_FRACTION
            ),
            "baseline_cold": baseline_cold,
            "candidate_cold": candidate_cold,
            "cold_excess_recovered_seconds": cold_recovered,
            "cold_excess_recovery_fraction": cold_recovery_fraction,
            "minimum_cold_excess_recovery_fraction": (
                MINIMUM_COLD_EXCESS_RECOVERY_FRACTION
            ),
        },
        "correctness_context": {
            "baseline_presence_status": baseline_verdict.get("status"),
            "candidate_presence_status": candidate_verdict.get("status"),
            "block_batch_gate_is_not_output_correctness_evidence": True,
        },
        "next_action": (
            "Use the candidate phase artifact to select the next measured stage; "
            "keep the independent long-context output verdict as a correctness gate."
        ),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError("block-batched gather artifact already exists")
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
