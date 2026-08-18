"""CPU-only tests for the prepared-gather reuse cross-run gate."""

from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import pytest

_analysis = runpy.run_path(
    "scripts/analyze_connector_prepared_gather_reuse.py",
    run_name="connector_prepared_gather_reuse_analysis_test",
)


def _phase_artifact(
    *,
    preflight_prepare: float,
    preflight_enqueue: float,
    preflight_synchronize: float,
    gather_prepare: float,
) -> dict[str, object]:
    def metric(seconds: float) -> dict[str, int | float]:
        return {"count": 2, "sum_seconds": seconds, "mean_seconds": seconds / 2}

    return {
        "schema_version": 1,
        "contract": "cacheblend-gpt-oss-store-data-plane-phase-breakdown-v1",
        "operation_geometry": {
            "expected_prepared_copy_operations": 59_904,
            "preflight_prepared_copy_operations": 59_904,
            "gather_prepared_copy_operations": 59_904,
            "stored_tokens": 19_968,
            "logical_payload_bytes": 981_467_136,
        },
        "preflight": {
            "phase_latency": {
                "store_preflight_prepare_latency_seconds": metric(
                    preflight_prepare
                ),
                "store_preflight_enqueue_latency_seconds": metric(
                    preflight_enqueue
                ),
                "store_preflight_synchronize_latency_seconds": metric(
                    preflight_synchronize
                ),
            }
        },
        "gather": {
            "phase_latency": {
                "store_gather_prepare_latency_seconds": metric(gather_prepare)
            }
        },
    }


def _verdict(*, connector_cold: float) -> dict[str, object]:
    cold_signature = {
        "request_digest": "a" * 64,
        "output_digest": "b" * 64,
        "usage": {
            "input_tokens": 20_139,
            "output_tokens": 35,
            "total_tokens": 20_174,
            "cached_tokens": 0,
            "reasoning_tokens": 18,
            "tool_output_tokens": 0,
        },
    }
    return {
        "schema_version": 1,
        "contract": "cacheblend-gpt-oss-connector-presence-equivalence-v1",
        "status": "INCONCLUSIVE_BASELINE_OUTPUT_UNSTABLE",
        "long_context_reached": True,
        "prefix_reuse_all": True,
        "connector_store_counters": {
            "store_tokens_eligible": 19_968,
            "store_tokens_completed": 19_968,
            "store_fallbacks": 0,
        },
        "latency": {
            "baseline_turn_mean_seconds": [4.0, 7.0, 15.0],
            "connector_turn_seconds": [connector_cold, 7.0, 15.0],
        },
        "response_signatures": {"connector": [cold_signature]},
    }


def _write_run(
    run_dir: Path,
    *,
    preflight_prepare: float,
    preflight_enqueue: float,
    preflight_synchronize: float,
    gather_prepare: float,
    connector_cold: float,
) -> None:
    run_dir.mkdir()
    verdict_path = run_dir / "connector-presence-verdict.json"
    verdict_path.write_text(
        json.dumps(_verdict(connector_cold=connector_cold)), encoding="utf-8"
    )
    verdict_sha256 = hashlib.sha256(verdict_path.read_bytes()).hexdigest()
    phase = _phase_artifact(
        preflight_prepare=preflight_prepare,
        preflight_enqueue=preflight_enqueue,
        preflight_synchronize=preflight_synchronize,
        gather_prepare=gather_prepare,
    )
    phase["input_sha256"] = {"verdict": verdict_sha256}
    (run_dir / "connector-store-data-plane-breakdown.json").write_text(
        json.dumps(phase),
        encoding="utf-8",
    )


def test_reuse_gate_requires_zero_second_prepare_and_recovers_latency(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(
        baseline,
        preflight_prepare=6.5,
        preflight_enqueue=0.03,
        preflight_synchronize=0.001,
        gather_prepare=6.2,
        connector_cold=16.6,
    )
    _write_run(
        candidate,
        preflight_prepare=6.4,
        preflight_enqueue=0.0,
        preflight_synchronize=0.0,
        gather_prepare=0.0,
        connector_cold=10.5,
    )

    report = _analysis["analyze"](baseline, candidate)

    assert report["passed"] is True
    assert report["status"] == (
        "PREPARED_GATHER_REUSE_RECOVERED_DUPLICATE_PREPARATION"
    )
    assert report["reuse_mechanics"]["second_prepare_eliminated"] is True
    assert report["latency"][
        "recovery_fraction_of_prior_gather_prepare"
    ] == pytest.approx(6.1 / 6.2)


def test_reuse_gate_fails_when_second_prepare_remains(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(
        baseline,
        preflight_prepare=6.5,
        preflight_enqueue=0.03,
        preflight_synchronize=0.001,
        gather_prepare=6.2,
        connector_cold=16.6,
    )
    _write_run(
        candidate,
        preflight_prepare=6.4,
        preflight_enqueue=0.0,
        preflight_synchronize=0.0,
        gather_prepare=0.1,
        connector_cold=10.5,
    )

    report = _analysis["analyze"](baseline, candidate)

    assert report["passed"] is False
    assert report["reuse_mechanics"]["second_prepare_eliminated"] is False


def test_reuse_gate_fails_without_expected_cold_recovery(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(
        baseline,
        preflight_prepare=6.5,
        preflight_enqueue=0.03,
        preflight_synchronize=0.001,
        gather_prepare=6.2,
        connector_cold=16.6,
    )
    _write_run(
        candidate,
        preflight_prepare=6.4,
        preflight_enqueue=0.0,
        preflight_synchronize=0.0,
        gather_prepare=0.0,
        connector_cold=15.0,
    )

    report = _analysis["analyze"](baseline, candidate)

    assert report["passed"] is False
    assert report["latency"][
        "recovery_fraction_of_prior_gather_prepare"
    ] < 0.8


def test_reuse_gate_rejects_phase_verdict_digest_mismatch(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    for run_dir, gather_prepare, connector_cold in (
        (baseline, 6.2, 16.6),
        (candidate, 0.0, 10.5),
    ):
        _write_run(
            run_dir,
            preflight_prepare=6.5,
            preflight_enqueue=0.0,
            preflight_synchronize=0.0,
            gather_prepare=gather_prepare,
            connector_cold=connector_cold,
        )
    (candidate / "connector-presence-verdict.json").write_text(
        json.dumps(_verdict(connector_cold=10.6)), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="digest mismatch"):
        _analysis["analyze"](baseline, candidate)
