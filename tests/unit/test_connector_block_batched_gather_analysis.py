"""CPU-only tests for the block-batched gather cross-run gate."""

from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import pytest

_analysis = runpy.run_path(
    "scripts/analyze_connector_block_batched_gather.py",
    run_name="connector_block_batched_gather_analysis_test",
)


def _metric(seconds: float) -> dict[str, int | float]:
    return {"count": 2, "sum_seconds": seconds, "mean_seconds": seconds / 2}


def _phase(
    *,
    preflight_prepare: float,
    submitted_operations: int | None,
    gather_prepare: float = 0.0,
) -> dict[str, object]:
    geometry: dict[str, int | float] = {
        "expected_prepared_copy_operations": 59_904,
        "preflight_prepared_copy_operations": 59_904,
        "gather_prepared_copy_operations": 59_904,
        "stored_tokens": 19_968,
        "logical_payload_bytes": 981_467_136,
    }
    if submitted_operations is not None:
        geometry.update(
            {
                "preflight_submitted_copy_operations": submitted_operations,
                "gather_submitted_copy_operations": submitted_operations,
            }
        )
    return {
        "schema_version": 1,
        "contract": "cacheblend-gpt-oss-store-data-plane-phase-breakdown-v1",
        "operation_geometry": geometry,
        "preflight": {
            "phase_latency": {
                "store_preflight_prepare_latency_seconds": _metric(
                    preflight_prepare
                ),
                "store_preflight_enqueue_latency_seconds": _metric(0.0),
                "store_preflight_synchronize_latency_seconds": _metric(0.0),
            }
        },
        "gather": {
            "phase_latency": {
                "store_gather_prepare_latency_seconds": _metric(gather_prepare)
            }
        },
    }


def _verdict(connector_cold: float, *, status: str) -> dict[str, object]:
    signature = {
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
        "status": status,
        "long_context_reached": True,
        "prefix_reuse_all": True,
        "connector_counters": {
            "requests": 3,
            "reusable_document_tokens_requested": 20_139,
            "kv_tokens_found": 0,
            "kv_tokens_loaded": 0,
            "kv_tokens_rejected": 0,
            "tokens_recomputed": 120_426,
            "prefill_tokens_avoided": 0,
        },
        "connector_store_counters": {
            "store_tokens_eligible": 19_968,
            "store_tokens_completed": 19_968,
            "store_fallbacks": 0,
        },
        "latency": {
            "baseline_turn_mean_seconds": [4.0, 7.0, 15.0],
            "connector_turn_seconds": [connector_cold, 7.0, 15.0],
        },
        "response_signatures": {"connector": [signature]},
    }


def _write_run(
    run_dir: Path,
    *,
    preflight_prepare: float,
    connector_cold: float,
    submitted_operations: int | None,
    status: str,
    gather_prepare: float = 0.0,
) -> None:
    run_dir.mkdir()
    verdict_path = run_dir / "connector-presence-verdict.json"
    verdict_path.write_text(
        json.dumps(_verdict(connector_cold, status=status)),
        encoding="utf-8",
    )
    phase = _phase(
        preflight_prepare=preflight_prepare,
        submitted_operations=submitted_operations,
        gather_prepare=gather_prepare,
    )
    phase["input_sha256"] = {
        "verdict": hashlib.sha256(verdict_path.read_bytes()).hexdigest()
    }
    (run_dir / "connector-store-data-plane-breakdown.json").write_text(
        json.dumps(phase),
        encoding="utf-8",
    )


def _write_pair(
    tmp_path: Path,
    *,
    candidate_prepare: float = 0.2,
    candidate_cold: float = 4.5,
    submitted_operations: int = 48,
    gather_prepare: float = 0.0,
) -> tuple[Path, Path]:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(
        baseline,
        preflight_prepare=6.0,
        connector_cold=10.0,
        submitted_operations=None,
        status="FAIL_CONNECTOR_OUTPUT_DIVERGED",
    )
    _write_run(
        candidate,
        preflight_prepare=candidate_prepare,
        connector_cold=candidate_cold,
        submitted_operations=submitted_operations,
        status="FAIL_CONNECTOR_OUTPUT_DIVERGED",
        gather_prepare=gather_prepare,
    )
    return baseline, candidate


def test_block_batch_gate_reconciles_48_submissions_and_recovers_latency(
    tmp_path: Path,
) -> None:
    baseline, candidate = _write_pair(tmp_path)

    report = _analysis["analyze"](baseline, candidate)

    assert report["passed"] is True
    assert report["status"] == (
        "BLOCK_BATCHED_GATHER_RECOVERED_VIEW_PREPARATION"
    )
    assert report["operation_geometry"]["observed_store_batches"] == 1
    assert report["operation_geometry"][
        "submission_reduction_fraction"
    ] == pytest.approx(1.0 - 48 / 59_904)
    assert report["latency"][
        "preflight_prepare_recovery_fraction"
    ] == pytest.approx(5.8 / 6.0)
    assert report["latency"]["cold_excess_recovery_fraction"] == pytest.approx(
        5.5 / 6.0
    )


def test_block_batch_gate_fails_without_preparation_recovery(tmp_path: Path) -> None:
    baseline, candidate = _write_pair(tmp_path, candidate_prepare=3.0)

    report = _analysis["analyze"](baseline, candidate)

    assert report["passed"] is False
    assert report["latency"]["preflight_prepare_recovery_fraction"] < 0.8


def test_block_batch_gate_fails_non_batched_submission_geometry(
    tmp_path: Path,
) -> None:
    baseline, candidate = _write_pair(tmp_path, submitted_operations=59_904)

    report = _analysis["analyze"](baseline, candidate)

    assert report["passed"] is False
    assert report["operation_geometry"]["reconciled"] is True
    assert report["operation_geometry"]["submission_reduction_fraction"] == 0.0


def test_block_batch_gate_preserves_one_shot_reuse(tmp_path: Path) -> None:
    baseline, candidate = _write_pair(tmp_path, gather_prepare=0.01)

    report = _analysis["analyze"](baseline, candidate)

    assert report["passed"] is False
    assert report["mechanics"]["one_shot_reuse_preserved"] is False
