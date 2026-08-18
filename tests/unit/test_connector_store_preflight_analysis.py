"""CPU-only tests for retained gather-preparation attribution."""

from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import pytest

_analysis = runpy.run_path(
    "scripts/analyze_connector_store_preflight.py",
    run_name="connector_store_preflight_analysis_test",
)

_SUBPHASE_SECONDS = {
    "store_preflight_input_materialization_latency_seconds": 0.05,
    "store_preflight_span_validation_latency_seconds": 1.2,
    "store_preflight_tensor_validation_latency_seconds": 0.2,
    "store_preflight_range_validation_latency_seconds": 0.1,
    "store_preflight_block_plan_latency_seconds": 0.2,
    "store_preflight_block_index_view_latency_seconds": 0.15,
    "store_preflight_legacy_view_latency_seconds": 0.0,
}


def _metrics(histograms: dict[str, float], *, count: int) -> str:
    lines: list[str] = []
    for key, value in histograms.items():
        base = f"vllm:cacheblend_{key}"
        lines.append(f'{base}_count{{engine="0"}} {count}')
        lines.append(f'{base}_sum{{engine="0"}} {value}')
    return "\n".join(lines) + "\n"


def _write_run(
    run_dir: Path,
    *,
    enclosing_seconds: float = 2.0,
    subphases: dict[str, float] | None = None,
    submitted_operations: int = 48,
) -> None:
    connector_dir = run_dir / "connector"
    connector_dir.mkdir(parents=True)
    phase_seconds = dict(_SUBPHASE_SECONDS if subphases is None else subphases)
    histograms = {
        "store_preflight_prepare_latency_seconds": enclosing_seconds,
        **phase_seconds,
    }
    startup_path = connector_dir / "metrics-startup.prom"
    after_path = connector_dir / "metrics-after.prom"
    startup_path.write_text(
        _metrics({key: 0.0 for key in histograms}, count=0),
        encoding="utf-8",
    )
    after_path.write_text(_metrics(histograms, count=2), encoding="utf-8")

    verdict_path = run_dir / "connector-presence-verdict.json"
    verdict = {
        "schema_version": 1,
        "contract": "cacheblend-gpt-oss-connector-presence-equivalence-v1",
        "status": "INCONCLUSIVE_BASELINE_OUTPUT_UNSTABLE",
        "baseline_outputs_stable": False,
        "connector_outputs_match": False,
    }
    verdict_path.write_text(json.dumps(verdict), encoding="utf-8")

    metric = {
        "count": 2,
        "sum_seconds": enclosing_seconds,
        "mean_seconds": enclosing_seconds / 2,
    }
    data_plane = {
        "schema_version": 1,
        "contract": "cacheblend-gpt-oss-store-data-plane-phase-breakdown-v1",
        "input_sha256": {
            "verdict": hashlib.sha256(verdict_path.read_bytes()).hexdigest(),
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
        "preflight": {
            "phase_latency": {
                "store_preflight_prepare_latency_seconds": metric,
            }
        },
    }
    (run_dir / "connector-store-data-plane-breakdown.json").write_text(
        json.dumps(data_plane), encoding="utf-8"
    )


def test_preflight_breakdown_reconciles_fast_path_and_selects_dominant_phase(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path)

    report = _analysis["analyze"](tmp_path)

    assert report["status"] == "CAPTURED_STORE_PREFLIGHT_PREPARATION_BREAKDOWN"
    assert report["operation_geometry"]["preflight_submitted_copy_operations"] == 48
    preparation = report["preparation"]
    assert preparation["subphase_sum_seconds"] == pytest.approx(1.9)
    assert preparation["unattributed_enclosing_seconds"] == pytest.approx(0.1)
    assert preparation["dominant_component"] == (
        "store_preflight_span_validation_latency_seconds"
    )
    assert preparation["dominant_component_share"] == pytest.approx(0.6)
    assert preparation["legacy_fast_path_observed"] is True


def test_preflight_breakdown_rejects_subphases_above_enclosing_time(
    tmp_path: Path,
) -> None:
    phases = dict(_SUBPHASE_SECONDS)
    phases["store_preflight_span_validation_latency_seconds"] = 1.4
    _write_run(tmp_path, subphases=phases)

    with pytest.raises(ValueError, match="subphases exceed"):
        _analysis["analyze"](tmp_path)


def test_preflight_breakdown_rejects_legacy_view_fallback(tmp_path: Path) -> None:
    phases = dict(_SUBPHASE_SECONDS)
    phases["store_preflight_span_validation_latency_seconds"] = 1.1
    phases["store_preflight_legacy_view_latency_seconds"] = 0.1
    _write_run(tmp_path, subphases=phases)

    with pytest.raises(ValueError, match="legacy views"):
        _analysis["analyze"](tmp_path)


def test_preflight_breakdown_rejects_non_batched_geometry(tmp_path: Path) -> None:
    _write_run(tmp_path, submitted_operations=59_904)

    with pytest.raises(ValueError, match="pinned fast path"):
        _analysis["analyze"](tmp_path)
