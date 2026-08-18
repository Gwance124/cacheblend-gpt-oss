"""CPU-only tests for the batched block-index owner cross-run gate."""

from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import pytest

_analysis = runpy.run_path(
    "scripts/analyze_connector_batched_block_indices.py",
    run_name="connector_batched_block_indices_analysis_test",
)


def _metric(seconds: float) -> dict[str, int | float]:
    return {"count": 2, "sum_seconds": seconds, "mean_seconds": seconds / 2}


def _verdict(connector_cold: float) -> dict[str, object]:
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
        "response_signatures": {"connector": [signature]},
    }


def _write_run(
    run_dir: Path,
    *,
    baseline: bool,
    construction_seconds: float,
    preflight_seconds: float,
    connector_cold: float,
    owner_constructions: int = 1,
) -> None:
    run_dir.mkdir()
    verdict_path = run_dir / "connector-presence-verdict.json"
    verdict_path.write_text(json.dumps(_verdict(connector_cold)), encoding="utf-8")
    verdict_sha = hashlib.sha256(verdict_path.read_bytes()).hexdigest()

    preflight_path = run_dir / "connector-store-preflight-breakdown.json"
    preflight = {
        "schema_version": 1,
        "contract": "cacheblend-gpt-oss-store-preflight-preparation-breakdown-v1",
        "input_sha256": {"verdict": verdict_sha},
        "preparation": {"enclosing_latency": _metric(preflight_seconds)},
    }
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")

    geometry: dict[str, int] = {
        "stored_tokens": 19_968,
        "expected_prepared_copy_operations": 59_904,
        "preflight_prepared_copy_operations": 59_904,
        "preflight_submitted_copy_operations": 48,
        "expected_staging_view_constructions_per_store_batch": 48,
    }
    if baseline:
        geometry["expected_index_tensor_constructions_per_store_batch"] = 24
    else:
        geometry.update(
            {
                "expected_block_index_owner_constructions_per_store_batch": 1,
                "expected_block_index_row_views_per_store_batch": 24,
                "observed_block_index_owner_constructions_per_store_batch": (
                    owner_constructions
                ),
                "observed_block_index_row_views_per_store_batch": 24,
                "observed_staging_view_constructions_per_store_batch": 48,
            }
        )
    block = {
        "schema_version": 1,
        "contract": "cacheblend-gpt-oss-block-index-view-breakdown-v1",
        "input_sha256": {
            "verdict": verdict_sha,
            "store_preflight_breakdown": hashlib.sha256(
                preflight_path.read_bytes()
            ).hexdigest(),
        },
        "operation_geometry": geometry,
        "block_index_view": {
            "subphase_latency": {
                "store_preflight_block_index_construction_latency_seconds": (
                    _metric(construction_seconds)
                )
            }
        },
    }
    (run_dir / "connector-block-index-view-breakdown.json").write_text(
        json.dumps(block), encoding="utf-8"
    )


def _write_pair(
    tmp_path: Path,
    *,
    candidate_construction: float = 0.1,
    candidate_preflight: float = 1.0,
    owner_constructions: int = 1,
) -> tuple[Path, Path]:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_run(
        baseline,
        baseline=True,
        construction_seconds=1.0,
        preflight_seconds=2.0,
        connector_cold=4.7,
    )
    _write_run(
        candidate,
        baseline=False,
        construction_seconds=candidate_construction,
        preflight_seconds=candidate_preflight,
        connector_cold=4.4,
        owner_constructions=owner_constructions,
    )
    return baseline, candidate


def test_batched_index_gate_requires_exact_mechanics_and_recovers_cost(
    tmp_path: Path,
) -> None:
    baseline, candidate = _write_pair(tmp_path)

    report = _analysis["analyze"](baseline, candidate)

    assert report["passed"] is True
    assert report["status"] == ("BATCHED_BLOCK_INDEX_OWNER_RECOVERED_CONSTRUCTION")
    assert report["mechanics"]["candidate"]["exact"] is True
    assert report["latency"][
        "block_index_construction_recovery_fraction"
    ] == pytest.approx(0.9)
    assert report["latency"][
        "preflight_recovery_fraction_of_baseline_construction"
    ] == pytest.approx(1.0)


def test_batched_index_gate_fails_without_one_owner(tmp_path: Path) -> None:
    baseline, candidate = _write_pair(tmp_path, owner_constructions=2)

    report = _analysis["analyze"](baseline, candidate)

    assert report["passed"] is False
    assert report["mechanics"]["candidate"]["exact"] is False


def test_batched_index_gate_fails_without_measured_recovery(tmp_path: Path) -> None:
    baseline, candidate = _write_pair(
        tmp_path,
        candidate_construction=0.4,
        candidate_preflight=1.5,
    )

    report = _analysis["analyze"](baseline, candidate)

    assert report["passed"] is False
    assert report["latency"]["block_index_construction_recovery_fraction"] < 0.8


def test_batched_index_gate_rejects_digest_mismatch(tmp_path: Path) -> None:
    baseline, candidate = _write_pair(tmp_path)
    (candidate / "connector-presence-verdict.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="identity|digest mismatch"):
        _analysis["analyze"](baseline, candidate)
