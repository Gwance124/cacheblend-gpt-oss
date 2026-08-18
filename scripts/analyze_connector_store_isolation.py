#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Combine the store-on and no-store runs into one bounded causal artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CONTRACT = "cacheblend-gpt-oss-connector-store-isolation-v1"
PRESENCE_CONTRACT = "cacheblend-gpt-oss-connector-presence-equivalence-v1"
STAGE_CONTRACT = "cacheblend-gpt-oss-connector-stage-diagnostic-v1"
STORE_DOMINANCE_STATUS = "RECORDED_STORE_DOMINATES_COLD_EXCESS"
MINIMUM_RECOVERY_FRACTION = 0.8

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
    parser.add_argument("--store-on-run-dir", type=Path, required=True)
    parser.add_argument("--no-store-run-dir", type=Path, required=True)
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


def _positive_number(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"invalid {name}")
    return float(value)


def _require_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"invalid {name}")
    return value


def _require_sequence(value: object, name: str, length: int) -> list[Any]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"invalid {name}")
    return value


def _counter_mapping(
    value: object,
    expected_keys: frozenset[str],
    name: str,
) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError(f"invalid {name} schema")
    result: dict[str, int] = {}
    for key in expected_keys:
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"invalid {name} {key}")
        result[key] = item
    return result


def _read_presence_verdict(path: Path) -> dict[str, Any]:
    verdict = _read_json(path)
    if (
        verdict.get("schema_version") != 1
        or verdict.get("contract") != PRESENCE_CONTRACT
    ):
        raise ValueError(f"invalid connector-presence verdict identity: {path}")
    return verdict


def _latency(verdict: dict[str, Any], name: str) -> dict[str, Any]:
    latency = _require_mapping(verdict.get("latency"), f"{name} latency")
    _require_sequence(
        latency.get("baseline_turn_mean_seconds"),
        f"{name} baseline turn means",
        3,
    )
    _require_sequence(
        latency.get("connector_turn_seconds"),
        f"{name} connector turn latencies",
        3,
    )
    return latency


def _connector_signatures(verdict: dict[str, Any], name: str) -> list[Any]:
    signatures = _require_mapping(
        verdict.get("response_signatures"), f"{name} signatures"
    )
    return _require_sequence(
        signatures.get("connector"), f"{name} connector signatures", 3
    )


def analyze(store_on_run_dir: Path, no_store_run_dir: Path) -> dict[str, object]:
    store_verdict_path = store_on_run_dir / "connector-presence-verdict.json"
    stage_path = store_on_run_dir / "connector-stage-diagnostic.json"
    no_store_verdict_path = no_store_run_dir / "connector-no-store-verdict.json"

    store_verdict = _read_presence_verdict(store_verdict_path)
    no_store_verdict = _read_presence_verdict(no_store_verdict_path)
    stage = _read_json(stage_path)
    if (
        stage.get("schema_version") != 1
        or stage.get("contract") != STAGE_CONTRACT
        or stage.get("status") != STORE_DOMINANCE_STATUS
    ):
        raise ValueError("store-on stage diagnostic did not establish dominance")
    stage_hashes = _require_mapping(stage.get("input_sha256"), "stage input hashes")
    if stage_hashes.get("verdict") != _sha256(store_verdict_path):
        raise ValueError("store-on stage diagnostic is not bound to its verdict")

    store_counters = _counter_mapping(
        store_verdict.get("connector_counters"),
        _CONNECTOR_COUNTER_KEYS,
        "store-on connector counters",
    )
    no_store_counters = _counter_mapping(
        no_store_verdict.get("connector_counters"),
        _CONNECTOR_COUNTER_KEYS,
        "no-store connector counters",
    )
    store_write_counters = _counter_mapping(
        store_verdict.get("connector_store_counters"),
        _STORE_COUNTER_KEYS,
        "store-on store counters",
    )
    no_store_write_counters = _counter_mapping(
        no_store_verdict.get("connector_store_counters"),
        _STORE_COUNTER_KEYS,
        "no-store store counters",
    )
    no_store_policy = _require_mapping(
        no_store_verdict.get("store_policy"), "no-store policy"
    )

    store_latency = _latency(store_verdict, "store-on")
    no_store_latency = _latency(no_store_verdict, "no-store")
    stage_cold = _require_mapping(stage.get("cold_turn"), "stage cold turn")
    store_cold = _positive_number(
        store_latency["connector_turn_seconds"][0], "store-on cold latency"
    )
    stage_store_cold = _positive_number(
        stage_cold.get("connector_seconds"), "stage connector cold latency"
    )
    if not math.isclose(store_cold, stage_store_cold, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("store-on cold latency disagrees with stage diagnostic")
    no_store_cold = _positive_number(
        no_store_latency["connector_turn_seconds"][0], "no-store cold latency"
    )
    no_store_control_cold = _positive_number(
        no_store_latency["baseline_turn_mean_seconds"][0],
        "no-store control cold latency",
    )
    store_excess = _positive_number(
        stage_cold.get("excess_seconds"), "store-on cold excess"
    )
    recovered = store_cold - no_store_cold
    recovered_fraction = recovered / store_excess

    store_signatures = _connector_signatures(store_verdict, "store-on")
    no_store_signatures = _connector_signatures(no_store_verdict, "no-store")
    cold_signatures_equal = store_signatures[0] == no_store_signatures[0]
    counters_equal = store_counters == no_store_counters
    store_completed = (
        store_write_counters["store_tokens_eligible"] > 0
        and store_write_counters["store_tokens_completed"]
        == store_write_counters["store_tokens_eligible"]
        and store_write_counters["store_fallbacks"] == 0
    )
    zero_store = (
        no_store_policy.get("zero_store_required") is True
        and no_store_policy.get("zero_store_observed") is True
        and all(value == 0 for value in no_store_write_counters.values())
    )
    timing_controls_valid = (
        no_store_latency.get("baseline_stable") is True
        and no_store_latency.get("connector_within_limit") is True
        and no_store_verdict.get("prefix_reuse_all") is True
        and no_store_verdict.get("long_context_reached") is True
    )
    latency_isolated = (
        cold_signatures_equal
        and counters_equal
        and store_completed
        and zero_store
        and timing_controls_valid
        and recovered > 0.0
        and recovered_fraction >= MINIMUM_RECOVERY_FRACTION
    )

    baseline_outputs_stable = no_store_verdict.get("baseline_outputs_stable")
    if not isinstance(baseline_outputs_stable, bool):
        raise ValueError("no-store verdict has invalid baseline output stability")
    output_conclusion = (
        "NOT_TESTABLE_BASELINE_OUTPUT_UNSTABLE"
        if not baseline_outputs_stable
        else (
            "MATCHED"
            if no_store_verdict.get("connector_outputs_match") is True
            else "DIVERGED"
        )
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT,
        "status": (
            "STORE_PATH_REMOVAL_RECOVERED_COLD_LATENCY"
            if latency_isolated
            else "STORE_PATH_ISOLATION_GATE_NOT_MET"
        ),
        "latency_isolated": latency_isolated,
        "output_conclusion": output_conclusion,
        "run_dirs": {
            "store_on": str(store_on_run_dir),
            "no_store": str(no_store_run_dir),
        },
        "input_sha256": {
            "store_on_verdict": _sha256(store_verdict_path),
            "store_on_stage_diagnostic": _sha256(stage_path),
            "no_store_verdict": _sha256(no_store_verdict_path),
        },
        "matched_evidence": {
            "cold_connector_signature_equal": cold_signatures_equal,
            "connector_counters_equal": counters_equal,
            "prefix_reuse_observed": no_store_verdict.get("prefix_reuse_all"),
            "long_context_reached": no_store_verdict.get("long_context_reached"),
        },
        "store_intervention": {
            "store_on": store_write_counters,
            "no_store": no_store_write_counters,
            "store_on_completed_without_fallback": store_completed,
            "no_store_zero_observed": zero_store,
        },
        "latency": {
            "store_on_cold_seconds": store_cold,
            "no_store_cold_seconds": no_store_cold,
            "no_store_control_cold_mean_seconds": no_store_control_cold,
            "no_store_cold_ratio": no_store_cold / no_store_control_cold,
            "store_on_cold_excess_seconds": store_excess,
            "cold_latency_recovered_seconds": recovered,
            "cold_excess_recovery_fraction": recovered_fraction,
            "minimum_recovery_fraction": MINIMUM_RECOVERY_FRACTION,
            "no_store_residual_cold_excess_seconds": (
                no_store_cold - no_store_control_cold
            ),
            "store_on_total_seconds": _positive_number(
                store_latency.get("connector_total_seconds"),
                "store-on total latency",
            ),
            "no_store_total_seconds": _positive_number(
                no_store_latency.get("connector_total_seconds"),
                "no-store total latency",
            ),
        },
        "next_action": (
            "Measure gather, LMCache write, and sidecar publication as separate "
            "sub-stages of synchronous writeback."
        ),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError("connector store-isolation artifact already exists")
    report = analyze(
        args.store_on_run_dir.resolve(),
        args.no_store_run_dir.resolve(),
    )
    rendered = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with args.output.open("x", encoding="utf-8") as output:
        output.write(rendered)
    print(rendered, end="")
    return 0 if report["latency_isolated"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
