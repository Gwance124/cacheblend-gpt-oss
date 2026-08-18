#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gate one-shot prepared-gather reuse against a fixed measured baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CONTRACT = "cacheblend-gpt-oss-prepared-gather-reuse-v1"
PHASE_CONTRACT = "cacheblend-gpt-oss-store-data-plane-phase-breakdown-v1"
VERDICT_CONTRACT = "cacheblend-gpt-oss-connector-presence-equivalence-v1"
MINIMUM_RECOVERY_FRACTION = 0.8
MINIMUM_PRESERVED_PREPARE_RATIO = 0.5
MAXIMUM_PRESERVED_PREPARE_RATIO = 2.0
ZERO_PHASE_TOLERANCE_SECONDS = 1e-9

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


def _operation_geometry(phase: dict[str, Any]) -> dict[str, int | float]:
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
        raise ValueError("prepared-copy geometry does not reconcile")
    return {
        "expected_prepared_copy_operations": expected,
        "preflight_prepared_copy_operations": preflight,
        "gather_prepared_copy_operations": gather,
        "stored_tokens": stored_tokens,
        "logical_payload_bytes": payload_bytes,
    }


def _cold_latency(verdict: dict[str, Any]) -> dict[str, float]:
    latency = _mapping(verdict.get("latency"), "latency")
    control_turn_means = _sequence(
        latency.get("baseline_turn_mean_seconds"), "baseline_turn_mean_seconds"
    )
    connector_turns = _sequence(
        latency.get("connector_turn_seconds"), "connector_turn_seconds"
    )
    if not control_turn_means or not connector_turns:
        raise ValueError("missing cold-turn latency")
    control = _number(control_turn_means[0], "cold control mean")
    connector = _number(connector_turns[0], "cold connector latency")
    return {
        "control_mean_seconds": control,
        "connector_seconds": connector,
        "excess_seconds": max(0.0, connector - control),
        "ratio": connector / control if control > 0.0 else 0.0,
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
    if (
        phase.get("schema_version") != 1
        or phase.get("contract") != PHASE_CONTRACT
    ):
        raise ValueError(f"invalid phase artifact identity: {label}")
    if (
        verdict.get("schema_version") != 1
        or verdict.get("contract") != VERDICT_CONTRACT
    ):
        raise ValueError(f"invalid verdict artifact identity: {label}")
    phase_inputs = _mapping(phase.get("input_sha256"), f"{label}.input_sha256")
    if phase_inputs.get("verdict") != _sha256(verdict_path):
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

    baseline_geometry = _operation_geometry(baseline_phase)
    candidate_geometry = _operation_geometry(candidate_phase)
    geometry_equal = baseline_geometry == candidate_geometry

    baseline_preflight_prepare = _phase_seconds(
        baseline_phase, "preflight", PREFLIGHT_PREPARE_KEY
    )
    candidate_preflight_prepare = _phase_seconds(
        candidate_phase, "preflight", PREFLIGHT_PREPARE_KEY
    )
    baseline_gather_prepare = _phase_seconds(
        baseline_phase, "gather", GATHER_PREPARE_KEY
    )
    candidate_gather_prepare = _phase_seconds(
        candidate_phase, "gather", GATHER_PREPARE_KEY
    )
    candidate_preflight_enqueue = _phase_seconds(
        candidate_phase, "preflight", PREFLIGHT_ENQUEUE_KEY
    )
    candidate_preflight_synchronize = _phase_seconds(
        candidate_phase, "preflight", PREFLIGHT_SYNCHRONIZE_KEY
    )
    if baseline_gather_prepare <= 0.0 or baseline_preflight_prepare <= 0.0:
        raise ValueError("baseline did not measure duplicate positive preparation")

    preserved_prepare_ratio = (
        candidate_preflight_prepare / baseline_preflight_prepare
    )
    preflight_prepare_preserved = (
        MINIMUM_PRESERVED_PREPARE_RATIO
        <= preserved_prepare_ratio
        <= MAXIMUM_PRESERVED_PREPARE_RATIO
    )
    second_prepare_eliminated = (
        candidate_gather_prepare <= ZERO_PHASE_TOLERANCE_SECONDS
        and candidate_preflight_enqueue <= ZERO_PHASE_TOLERANCE_SECONDS
        and candidate_preflight_synchronize <= ZERO_PHASE_TOLERANCE_SECONDS
    )

    baseline_cold = _cold_latency(baseline_verdict)
    candidate_cold = _cold_latency(candidate_verdict)
    recovered_seconds = max(
        0.0,
        baseline_cold["excess_seconds"] - candidate_cold["excess_seconds"],
    )
    recovery_fraction = recovered_seconds / baseline_gather_prepare
    cold_signatures_equal = _cold_signature(baseline_verdict) == _cold_signature(
        candidate_verdict
    )
    store_counters_equal = (
        baseline_verdict.get("connector_store_counters")
        == candidate_verdict.get("connector_store_counters")
    )
    workload_gates_equal = (
        baseline_verdict.get("long_context_reached") is True
        and candidate_verdict.get("long_context_reached") is True
        and baseline_verdict.get("prefix_reuse_all") is True
        and candidate_verdict.get("prefix_reuse_all") is True
    )
    passed = (
        geometry_equal
        and preflight_prepare_preserved
        and second_prepare_eliminated
        and cold_signatures_equal
        and store_counters_equal
        and workload_gates_equal
        and recovery_fraction >= MINIMUM_RECOVERY_FRACTION
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT,
        "status": (
            "PREPARED_GATHER_REUSE_RECOVERED_DUPLICATE_PREPARATION"
            if passed
            else "PREPARED_GATHER_REUSE_GATE_FAILED"
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
            "geometry_equal": geometry_equal,
            "cold_connector_signature_equal": cold_signatures_equal,
            "store_counters_equal": store_counters_equal,
            "long_context_and_prefix_reuse": workload_gates_equal,
        },
        "operation_geometry": candidate_geometry,
        "reuse_mechanics": {
            "second_prepare_eliminated": second_prepare_eliminated,
            "preflight_prepare_preserved": preflight_prepare_preserved,
            "preflight_prepare_preserved_ratio": preserved_prepare_ratio,
            "baseline_preflight_prepare_seconds": baseline_preflight_prepare,
            "candidate_preflight_prepare_seconds": candidate_preflight_prepare,
            "baseline_gather_prepare_seconds": baseline_gather_prepare,
            "candidate_gather_prepare_seconds": candidate_gather_prepare,
            "candidate_preflight_enqueue_seconds": candidate_preflight_enqueue,
            "candidate_preflight_synchronize_seconds": (
                candidate_preflight_synchronize
            ),
            "zero_phase_tolerance_seconds": ZERO_PHASE_TOLERANCE_SECONDS,
        },
        "latency": {
            "baseline_cold": baseline_cold,
            "candidate_cold": candidate_cold,
            "cold_excess_recovered_seconds": recovered_seconds,
            "recovery_fraction_of_prior_gather_prepare": recovery_fraction,
            "minimum_recovery_fraction": MINIMUM_RECOVERY_FRACTION,
        },
        "correctness_context": {
            "baseline_presence_status": baseline_verdict.get("status"),
            "candidate_presence_status": candidate_verdict.get("status"),
            "reuse_gate_is_not_output_correctness_evidence": True,
        },
        "next_action": (
            "Vectorize or batch the remaining single prepared-view construction "
            "while preserving the one-shot batch and exact geometry gates."
        ),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError("prepared-gather reuse artifact already exists")
    report = analyze(
        args.baseline_run_dir.resolve(), args.candidate_run_dir.resolve()
    )
    rendered = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with args.output.open("x", encoding="utf-8") as output:
        output.write(rendered)
    print(rendered, end="")
    return 0 if report["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
