"""CPU-only tests for connector stage attribution."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

_analysis = runpy.run_path(
    "scripts/analyze_connector_presence_run.py",
    run_name="connector_presence_analysis_test",
)


def _write_metrics(path: Path, *, count: int, store: float, lookup: float) -> None:
    sums = {
        "lookup_latency_seconds": lookup,
        "transfer_latency_seconds": 0.0,
        "position_correction_latency_seconds": 0.0,
        "selective_recomputation_latency_seconds": 0.0,
        "store_latency_seconds": store,
    }
    lines: list[str] = []
    for key, value in sums.items():
        base = f"vllm:cacheblend_{key}"
        lines.extend(
            (
                f'{base}_count{{engine="0"}} {count}',
                f'{base}_sum{{engine="0"}} {value}',
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _signature(output: str, request: str) -> dict[str, object]:
    return {
        "output_digest": output * 64,
        "request_digest": request * 64,
        "usage": {
            "input_tokens": 20_000,
            "output_tokens": 10,
            "total_tokens": 20_010,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
            "tool_output_tokens": 0,
        },
    }


def _write_run(run_dir: Path) -> None:
    connector_dir = run_dir / "connector"
    connector_dir.mkdir(parents=True)
    _write_metrics(
        connector_dir / "metrics-startup.prom",
        count=0,
        store=0.0,
        lookup=0.0,
    )
    _write_metrics(
        connector_dir / "metrics-after.prom",
        count=4,
        store=10.5,
        lookup=0.5,
    )
    baseline = [
        _signature("a", "1"),
        _signature("b", "2"),
        _signature("c", "3"),
    ]
    connector = [
        _signature("a", "1"),
        _signature("d", "2"),
        _signature("e", "4"),
    ]
    verdict = {
        "schema_version": 1,
        "contract": "cacheblend-gpt-oss-connector-presence-equivalence-v1",
        "latency": {
            "baseline_turn_mean_seconds": [4.0, 7.0, 15.0],
            "connector_turn_seconds": [16.0, 8.0, 16.0],
        },
        "response_signatures": {
            "baseline_a": baseline,
            "baseline_b": baseline,
            "connector": connector,
        },
        "connector_counters": {
            "requests": 3,
            "reusable_document_tokens_requested": 20_000,
            "kv_tokens_found": 0,
            "kv_tokens_loaded": 0,
            "kv_tokens_rejected": 0,
            "tokens_recomputed": 120_000,
            "prefill_tokens_avoided": 0,
        },
        "connector_store_counters": {
            "store_tokens_eligible": 19_968,
            "store_tokens_completed": 19_968,
            "store_fallbacks": 0,
        },
        "connector_warmup": {
            "connector_store_counters": {
                "store_tokens_eligible": 0,
                "store_tokens_completed": 0,
                "store_fallbacks": 0,
            }
        },
    }
    (run_dir / "connector-presence-verdict.json").write_text(
        json.dumps(verdict),
        encoding="utf-8",
    )


def test_analysis_attributes_cold_excess_and_trajectory(tmp_path: Path) -> None:
    _write_run(tmp_path)

    report = _analysis["analyze"](tmp_path)

    assert report["status"] == "RECORDED_STORE_DOMINATES_COLD_EXCESS"
    assert report["cold_turn"]["excess_seconds"] == pytest.approx(12.0)
    assert report["attribution"]["store_sum_seconds"] == pytest.approx(10.5)
    assert report["attribution"]["store_share_of_cold_excess"] == pytest.approx(0.875)
    assert report["attribution"]["primary_stage_sum_seconds"] == pytest.approx(11.0)
    assert report["trajectory"] == {
        "first_output_divergence_turn": 2,
        "same_request_at_first_output_divergence": True,
        "first_request_divergence_turn": 3,
    }


def test_histogram_delta_rejects_backwards_samples() -> None:
    before = "vllm:cacheblend_store_latency_seconds_count 2\n"
    before += "vllm:cacheblend_store_latency_seconds_sum 3\n"
    after = "vllm:cacheblend_store_latency_seconds_count 1\n"
    after += "vllm:cacheblend_store_latency_seconds_sum 2\n"

    with pytest.raises(ValueError, match="moved backwards"):
        _analysis["_histogram_delta"](
            before,
            after,
            "store_latency_seconds",
        )


def test_histogram_delta_accepts_missing_cold_snapshot() -> None:
    after = "vllm:cacheblend_store_latency_seconds_count 1\n"
    after += "vllm:cacheblend_store_latency_seconds_sum 2.5\n"

    assert _analysis["_histogram_delta"](
        "# no observations yet\n",
        after,
        "store_latency_seconds",
    ) == {"count": 1, "sum_seconds": 2.5, "mean_seconds": 2.5}
