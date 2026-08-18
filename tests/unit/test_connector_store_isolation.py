"""CPU-only tests for cross-run connector store isolation."""

from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import pytest

_analysis = runpy.run_path(
    "scripts/analyze_connector_store_isolation.py",
    run_name="connector_store_isolation_test",
)


def _signature(output: str) -> dict[str, object]:
    return {
        "request_digest": "1" * 64,
        "output_digest": output * 64,
        "usage": {
            "input_tokens": 20_139,
            "output_tokens": 35,
            "total_tokens": 20_174,
            "cached_tokens": 0,
            "reasoning_tokens": 18,
            "tool_output_tokens": 0,
        },
    }


def _verdict(*, store: bool) -> dict[str, object]:
    connector_counters = {
        "requests": 3,
        "reusable_document_tokens_requested": 20_139,
        "kv_tokens_found": 0,
        "kv_tokens_loaded": 0,
        "kv_tokens_rejected": 0,
        "tokens_recomputed": 120_438,
        "prefill_tokens_avoided": 0,
    }
    store_counters = {
        "store_tokens_eligible": 19_968 if store else 0,
        "store_tokens_completed": 19_968 if store else 0,
        "store_fallbacks": 0,
    }
    cold = 16.709226039936766 if store else 4.5247651629615575
    total = 40.77574416203424 if store else 28.005189025076106
    connector_signatures = [
        _signature("a"),
        _signature("b"),
        _signature("c"),
    ]
    return {
        "schema_version": 1,
        "contract": "cacheblend-gpt-oss-connector-presence-equivalence-v1",
        "baseline_outputs_stable": store,
        "connector_outputs_match": False,
        "prefix_reuse_all": True,
        "long_context_reached": True,
        "latency": {
            "baseline_turn_mean_seconds": [
                4.1753507029498 if store else 4.080105734989047,
                7.5,
                15.0,
            ],
            "connector_turn_seconds": [cold, 8.0, 16.0],
            "connector_total_seconds": total,
            "baseline_stable": True,
            "connector_within_limit": not store,
        },
        "connector_counters": connector_counters,
        "connector_store_counters": store_counters,
        "store_policy": {
            "zero_store_required": not store,
            "zero_store_observed": not store,
        },
        "response_signatures": {
            "baseline_a": connector_signatures,
            "baseline_b": connector_signatures,
            "connector": connector_signatures,
        },
    }


def _write_runs(root: Path) -> tuple[Path, Path]:
    store_dir = root / "store-on"
    no_store_dir = root / "no-store"
    store_dir.mkdir()
    no_store_dir.mkdir()
    store_verdict_path = store_dir / "connector-presence-verdict.json"
    store_verdict_path.write_text(json.dumps(_verdict(store=True)), encoding="utf-8")
    stage = {
        "schema_version": 1,
        "contract": "cacheblend-gpt-oss-connector-stage-diagnostic-v1",
        "status": "RECORDED_STORE_DOMINATES_COLD_EXCESS",
        "input_sha256": {
            "verdict": hashlib.sha256(store_verdict_path.read_bytes()).hexdigest()
        },
        "cold_turn": {
            "connector_seconds": 16.709226039936766,
            "excess_seconds": 12.533875336986966,
        },
    }
    (store_dir / "connector-stage-diagnostic.json").write_text(
        json.dumps(stage), encoding="utf-8"
    )
    (no_store_dir / "connector-no-store-verdict.json").write_text(
        json.dumps(_verdict(store=False)), encoding="utf-8"
    )
    return store_dir, no_store_dir


def test_store_removal_recovers_measured_cold_excess(tmp_path: Path) -> None:
    store_dir, no_store_dir = _write_runs(tmp_path)

    report = _analysis["analyze"](store_dir, no_store_dir)

    assert report["status"] == "STORE_PATH_REMOVAL_RECOVERED_COLD_LATENCY"
    assert report["latency_isolated"] is True
    assert report["output_conclusion"] == "NOT_TESTABLE_BASELINE_OUTPUT_UNSTABLE"
    assert report["latency"]["cold_latency_recovered_seconds"] == pytest.approx(
        12.184460876975208
    )
    assert report["latency"]["cold_excess_recovery_fraction"] == pytest.approx(
        0.9721223922675655
    )
    assert report["latency"]["no_store_cold_ratio"] == pytest.approx(1.108982329589972)


def test_isolation_rejects_mismatched_connector_work(tmp_path: Path) -> None:
    store_dir, no_store_dir = _write_runs(tmp_path)
    path = no_store_dir / "connector-no-store-verdict.json"
    verdict = json.loads(path.read_text(encoding="utf-8"))
    verdict["connector_counters"]["tokens_recomputed"] += 1
    path.write_text(json.dumps(verdict), encoding="utf-8")

    report = _analysis["analyze"](store_dir, no_store_dir)

    assert report["latency_isolated"] is False
    assert report["matched_evidence"]["connector_counters_equal"] is False


def test_shell_runner_preserves_untracked_gpu_host_files() -> None:
    runner = Path("local-m85-analyze-connector-store-isolation.sh").read_text(
        encoding="utf-8"
    )

    assert "git status --porcelain --untracked-files=no" in runner
    assert "PRESERVING_UNTRACKED_FILES" in runner
    assert "scripts/analyze_connector_store_isolation.py" in runner
