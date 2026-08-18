"""CPU-only tests for synchronous store-stage decomposition."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

_analysis = runpy.run_path(
    "scripts/analyze_connector_store_stages.py",
    run_name="connector_store_stage_analysis_test",
)


def _write_metrics(path: Path, values: dict[str, float], *, count: int) -> None:
    lines: list[str] = []
    for key, value in values.items():
        base = f"vllm:cacheblend_{key}"
        lines.append(f'{base}_count{{engine="0"}} {count}')
        lines.append(f'{base}_sum{{engine="0"}} {value}')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_run(run_dir: Path, *, lmcache_seconds: float = 8.0) -> None:
    connector_dir = run_dir / "connector"
    connector_dir.mkdir(parents=True)
    values = {
        "store_latency_seconds": 10.0,
        "store_plan_latency_seconds": 0.1,
        "store_preflight_latency_seconds": 0.2,
        "store_gather_latency_seconds": 1.0,
        "store_lmcache_latency_seconds": lmcache_seconds,
        "store_sidecar_publish_latency_seconds": 0.3,
    }
    _write_metrics(
        connector_dir / "metrics-startup.prom",
        {key: 0.0 for key in values},
        count=0,
    )
    _write_metrics(connector_dir / "metrics-after.prom", values, count=2)
    verdict = {
        "schema_version": 1,
        "contract": "cacheblend-gpt-oss-connector-presence-equivalence-v1",
        "status": "FAIL_CONNECTOR_OUTPUT_DIVERGED",
        "baseline_outputs_stable": True,
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


def test_breakdown_identifies_lmcache_as_dominant_stage(tmp_path: Path) -> None:
    _write_run(tmp_path)

    report = _analysis["analyze"](tmp_path)

    assert report["status"] == "CAPTURED_SYNCHRONOUS_STORE_STAGE_BREAKDOWN"
    assert report["decomposition"]["dominant_stage"] == (
        "store_lmcache_latency_seconds"
    )
    assert report["decomposition"]["stage_sum_seconds"] == pytest.approx(9.6)
    assert report["decomposition"]["unattributed_enclosing_seconds"] == pytest.approx(
        0.4
    )
    assert report["decomposition"][
        "dominant_stage_share_of_enclosing_store"
    ] == pytest.approx(0.8)


def test_breakdown_rejects_substages_larger_than_enclosing_timer(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, lmcache_seconds=10.0)

    with pytest.raises(ValueError, match="exceeds enclosing"):
        _analysis["analyze"](tmp_path)
