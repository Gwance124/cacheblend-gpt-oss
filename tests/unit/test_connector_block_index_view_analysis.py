"""CPU-only tests for block-index and staging-view attribution."""

from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import pytest

_analysis = runpy.run_path(
    "scripts/analyze_connector_block_index_view.py",
    run_name="connector_block_index_view_analysis_test",
)

_SUBPHASE_SECONDS = {
    "store_preflight_block_index_construction_latency_seconds": 0.8,
    "store_preflight_block_index_validation_latency_seconds": 0.1,
    "store_preflight_staging_view_construction_latency_seconds": 0.1,
    "store_preflight_staging_view_validation_latency_seconds": 0.05,
}


def _metrics(histograms: dict[str, float], *, count: int) -> str:
    lines: list[str] = []
    for key, value in histograms.items():
        base = f"vllm:cacheblend_{key}"
        lines.append(f'{base}_count{{engine="0"}} {count}')
        lines.append(f'{base}_sum{{engine="0"}} {value}')
    return "\n".join(lines) + "\n"


def _counters(*, batches: int) -> str:
    values = {
        "store_preflight_block_index_owner_constructions": batches,
        "store_preflight_block_index_row_views": batches * 24,
        "store_preflight_staging_view_constructions": batches * 48,
    }
    return "".join(
        f'vllm:cacheblend_{key}_total{{engine="0"}} {value}\n'
        for key, value in values.items()
    )


def _write_run(
    run_dir: Path,
    *,
    enclosing_seconds: float = 1.1,
    subphases: dict[str, float] | None = None,
    submitted_operations: int = 48,
    mechanic_batches: int = 1,
) -> None:
    connector_dir = run_dir / "connector"
    connector_dir.mkdir(parents=True)
    phase_seconds = dict(_SUBPHASE_SECONDS if subphases is None else subphases)
    histograms = {
        "store_preflight_block_index_view_latency_seconds": enclosing_seconds,
        **phase_seconds,
    }
    startup_path = connector_dir / "metrics-startup.prom"
    after_path = connector_dir / "metrics-after.prom"
    startup_path.write_text(
        _metrics({key: 0.0 for key in histograms}, count=0) + _counters(batches=0),
        encoding="utf-8",
    )
    after_path.write_text(
        _metrics(histograms, count=2) + _counters(batches=mechanic_batches),
        encoding="utf-8",
    )

    verdict_path = run_dir / "connector-presence-verdict.json"
    verdict = {
        "schema_version": 1,
        "contract": "cacheblend-gpt-oss-connector-presence-equivalence-v1",
        "status": "INCONCLUSIVE_BASELINE_OUTPUT_UNSTABLE",
        "baseline_outputs_stable": False,
        "connector_outputs_match": False,
    }
    verdict_path.write_text(json.dumps(verdict), encoding="utf-8")

    data_plane_path = run_dir / "connector-store-data-plane-breakdown.json"
    data_plane_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract": ("cacheblend-gpt-oss-store-data-plane-phase-breakdown-v1"),
            }
        ),
        encoding="utf-8",
    )

    metric = {
        "count": 2,
        "sum_seconds": enclosing_seconds,
        "mean_seconds": enclosing_seconds / 2,
    }
    preflight = {
        "schema_version": 1,
        "contract": "cacheblend-gpt-oss-store-preflight-preparation-breakdown-v1",
        "input_sha256": {
            "verdict": hashlib.sha256(verdict_path.read_bytes()).hexdigest(),
            "store_data_plane_breakdown": hashlib.sha256(
                data_plane_path.read_bytes()
            ).hexdigest(),
            "connector_metrics_startup": hashlib.sha256(
                startup_path.read_bytes()
            ).hexdigest(),
            "connector_metrics_after": hashlib.sha256(
                after_path.read_bytes()
            ).hexdigest(),
        },
        "operation_geometry": {
            "stored_tokens": 19_968,
            "expected_prepared_copy_operations": 59_904,
            "preflight_prepared_copy_operations": 59_904,
            "preflight_submitted_copy_operations": submitted_operations,
        },
        "preparation": {
            "subphase_latency": {
                "store_preflight_block_index_view_latency_seconds": metric,
            }
        },
    }
    (run_dir / "connector-store-preflight-breakdown.json").write_text(
        json.dumps(preflight), encoding="utf-8"
    )


def test_block_index_view_breakdown_reconciles_and_selects_index_construction(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path)

    report = _analysis["analyze"](tmp_path)

    assert report["status"] == "CAPTURED_BLOCK_INDEX_VIEW_BREAKDOWN"
    geometry = report["operation_geometry"]
    assert geometry["expected_block_index_owner_constructions_per_store_batch"] == 1
    assert geometry["expected_block_index_row_views_per_store_batch"] == 24
    assert geometry["expected_staging_view_constructions_per_store_batch"] == 48
    assert geometry["timing_observations"] == 2
    assert geometry["observed_store_batches"] == 1
    assert geometry["observed_block_index_owner_constructions_per_store_batch"] == 1
    assert geometry["observed_block_index_row_views_per_store_batch"] == 24
    assert geometry["observed_staging_view_constructions_per_store_batch"] == 48
    breakdown = report["block_index_view"]
    assert breakdown["subphase_sum_seconds"] == pytest.approx(1.05)
    assert breakdown["unattributed_enclosing_seconds"] == pytest.approx(0.05)
    assert breakdown["dominant_component"] == (
        "store_preflight_block_index_construction_latency_seconds"
    )
    assert breakdown["dominant_component_share"] == pytest.approx(0.8 / 1.1)


def test_block_index_view_breakdown_rejects_subphases_above_envelope(
    tmp_path: Path,
) -> None:
    phases = dict(_SUBPHASE_SECONDS)
    phases["store_preflight_block_index_construction_latency_seconds"] = 0.9
    _write_run(tmp_path, subphases=phases)

    with pytest.raises(ValueError, match="subphases exceed"):
        _analysis["analyze"](tmp_path)


def test_block_index_view_breakdown_rejects_non_batched_geometry(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, submitted_operations=59_904)

    with pytest.raises(ValueError, match="pinned fast path"):
        _analysis["analyze"](tmp_path)


def test_block_index_view_breakdown_rejects_mechanics_per_timing_observation(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, mechanic_batches=2)

    with pytest.raises(ValueError, match="mechanics do not match"):
        _analysis["analyze"](tmp_path)


def test_block_index_view_breakdown_rejects_preflight_digest_mismatch(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path)
    verdict_path = tmp_path / "connector-presence-verdict.json"
    verdict_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="input artifact identity|digest mismatch"):
        _analysis["analyze"](tmp_path)
