"""CPU-only tests for connector store data-plane phase attribution."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

_analysis = runpy.run_path(
    "scripts/analyze_connector_store_data_plane.py",
    run_name="connector_store_data_plane_analysis_test",
)


def _metrics_text(
    histograms: dict[str, float],
    counters: dict[str, int],
    *,
    count: int,
) -> str:
    lines: list[str] = []
    for key, value in histograms.items():
        base = f"vllm:cacheblend_{key}"
        lines.append(f'{base}_count{{engine="0"}} {count}')
        lines.append(f'{base}_sum{{engine="0"}} {value}')
    for key, value in counters.items():
        lines.append(f'vllm:cacheblend_{key}_total{{engine="0"}} {value}')
    return "\n".join(lines) + "\n"


def _write_run(
    run_dir: Path,
    *,
    gather_operations: int = 59_904,
    gather_sync_seconds: float = 1.5,
) -> None:
    connector_dir = run_dir / "connector"
    connector_dir.mkdir(parents=True)
    histograms = {
        "store_preflight_latency_seconds": 8.0,
        "store_gather_latency_seconds": 9.0,
        "store_storage_preflight_latency_seconds": 0.4,
        "store_preflight_prepare_latency_seconds": 3.0,
        "store_preflight_enqueue_latency_seconds": 0.5,
        "store_preflight_synchronize_latency_seconds": 4.0,
        "store_gather_prepare_latency_seconds": 1.0,
        "store_gather_enqueue_latency_seconds": 6.0,
        "store_gather_synchronize_latency_seconds": gather_sync_seconds,
    }
    counters = {
        "store_preflight_prepared_copy_operations": 59_904,
        "store_gather_prepared_copy_operations": gather_operations,
    }
    (connector_dir / "metrics-startup.prom").write_text(
        _metrics_text(
            {key: 0.0 for key in histograms},
            {key: 0 for key in counters},
            count=0,
        ),
        encoding="utf-8",
    )
    (connector_dir / "metrics-after.prom").write_text(
        _metrics_text(histograms, counters, count=2),
        encoding="utf-8",
    )
    verdict = {
        "schema_version": 1,
        "contract": "cacheblend-gpt-oss-connector-presence-equivalence-v1",
        "status": "INCONCLUSIVE_BASELINE_OUTPUT_UNSTABLE",
        "baseline_outputs_stable": False,
        "connector_outputs_match": False,
        "connector_store_counters": {
            "store_tokens_eligible": 19_968,
            "store_tokens_completed": 19_968,
            "store_fallbacks": 0,
        },
    }
    (run_dir / "connector-presence-verdict.json").write_text(
        json.dumps(verdict), encoding="utf-8"
    )


def test_phase_breakdown_reconciles_exact_pinned_copy_geometry(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path)

    report = _analysis["analyze"](tmp_path)

    assert report["status"] == "CAPTURED_STORE_DATA_PLANE_PHASE_BREAKDOWN"
    geometry = report["operation_geometry"]
    assert geometry["expected_prepared_copy_operations"] == 59_904
    assert geometry["preflight_prepared_copy_operations"] == 59_904
    assert geometry["gather_prepared_copy_operations"] == 59_904
    assert geometry["bytes_per_full_block_copy"] == 16_384
    assert geometry["logical_payload_bytes"] == 981_467_136
    assert geometry["logical_payload_gib"] == pytest.approx(0.9140625)
    assert report["preflight"]["dominant_data_plane_phase"] == (
        "store_preflight_synchronize_latency_seconds"
    )
    assert report["preflight"]["attributed_sum_seconds"] == pytest.approx(7.9)
    assert report["gather"]["dominant_phase"] == (
        "store_gather_enqueue_latency_seconds"
    )
    assert report["gather"]["phase_sum_seconds"] == pytest.approx(8.5)


def test_phase_breakdown_rejects_wrong_copy_operation_count(tmp_path: Path) -> None:
    _write_run(tmp_path, gather_operations=59_903)

    with pytest.raises(ValueError, match="pinned geometry"):
        _analysis["analyze"](tmp_path)


def test_phase_breakdown_rejects_phases_above_enclosing_timer(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, gather_sync_seconds=2.1)

    with pytest.raises(ValueError, match="exceed.*gather"):
        _analysis["analyze"](tmp_path)
