"""CPU-only tests for the deterministic connector-presence diagnostic."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

_capture = runpy.run_path(
    "scripts/capture_hybrid_flag_responses.py",
    run_name="connector_presence_capture_test",
)
_evaluate = runpy.run_path(
    "scripts/evaluate_connector_presence_equivalence.py",
    run_name="connector_presence_evaluate_test",
)


def _artifact(
    mode: str,
    elapsed_seconds: list[float],
    *,
    connector: bool,
) -> dict[str, object]:
    input_tokens = [20_100, 40_200, 60_300]
    cached_tokens = [0, 20_000, 40_000]
    turns: list[dict[str, object]] = []
    for index, (input_count, cached_count, elapsed) in enumerate(
        zip(input_tokens, cached_tokens, elapsed_seconds, strict=True)
    ):
        canonical_output = {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": f"Paris {index}"}],
                }
            ],
        }
        output_tokens = 10 + index
        turns.append(
            {
                "request_digest": str(index) * 64,
                "output_digest": _evaluate["_digest"](canonical_output),
                "canonical_output": canonical_output,
                "usage": {
                    "input_tokens": input_count,
                    "output_tokens": output_tokens,
                    "total_tokens": input_count + output_tokens,
                    "cached_tokens": cached_count,
                    "reasoning_tokens": 5,
                    "tool_output_tokens": 0,
                },
                "elapsed_seconds": elapsed,
            }
        )
    return {
        "schema_version": 1,
        "contract": "cacheblend-gpt-oss-hybrid-flag-responses-v1",
        "mode": mode,
        "model": "openai/gpt-oss-20b",
        "sampling": {"temperature": 0.0, "top_p": 1.0, "seed": 0},
        "stable_replay_ids": True,
        "warmup": {
            "performed": True,
            "request_digest": _evaluate["_digest"](_evaluate["WARMUP_PAYLOAD"]),
            "connector_counters": (
                {
                    "requests": 1,
                    "reusable_document_tokens_requested": 0,
                    "kv_tokens_found": 0,
                    "kv_tokens_loaded": 0,
                    "kv_tokens_rejected": 0,
                    "tokens_recomputed": 8,
                    "prefill_tokens_avoided": 0,
                }
                if connector
                else None
            ),
            "connector_store_counters": (
                {
                    "store_tokens_eligible": 0,
                    "store_tokens_completed": 0,
                    "store_fallbacks": 0,
                }
                if connector
                else None
            ),
        },
        "workload": {
            "filler_units_per_turn": [" alpha", " beta", " gamma"],
            "filler_repetitions_per_turn": 20_000,
            "input_tokens_per_turn": input_tokens,
        },
        "turns": turns,
        "prefix_cache": {
            "cached_tokens_per_turn": cached_tokens,
            "reuse_observed_after_cold_turn": True,
        },
        "total_elapsed_seconds": sum(elapsed_seconds),
        "connector_counters": (
            {
                "requests": 3,
                "reusable_document_tokens_requested": 0,
                "kv_tokens_found": 0,
                "kv_tokens_loaded": 0,
                "kv_tokens_rejected": 0,
                "tokens_recomputed": 60_300,
                "prefill_tokens_avoided": 0,
            }
            if connector
            else None
        ),
        "connector_store_counters": (
            {
                "store_tokens_eligible": 60_160,
                "store_tokens_completed": 60_160,
                "store_fallbacks": 0,
            }
            if connector
            else None
        ),
    }


def test_long_context_filler_is_bounded_and_deterministic() -> None:
    build = _capture["_build_filler"]

    assert build(" alpha", 3) == " alpha alpha alpha"
    assert build(" beta", 0) == ""
    with pytest.raises(ValueError, match="bounded range"):
        build(" gamma", 20_001)
    with pytest.raises(ValueError, match="fixed workload"):
        build(" evidence", 3)


def test_connector_artifact_reader_validates_inert_counters(tmp_path: Path) -> None:
    artifact = _artifact("connector", [1.0, 2.0, 3.0], connector=True)
    path = tmp_path / "connector.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    parsed = _evaluate["_read_responses"](
        path,
        expected_mode="connector",
        connector_expected=True,
    )

    assert parsed["connector_counters"]["kv_tokens_loaded"] == 0


def test_connector_artifact_reader_rejects_loaded_kv(tmp_path: Path) -> None:
    artifact = _artifact("connector", [1.0, 2.0, 3.0], connector=True)
    artifact["connector_counters"]["kv_tokens_found"] = 16
    artifact["connector_counters"]["kv_tokens_loaded"] = 16
    path = tmp_path / "connector.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="counters do not reconcile"):
        _evaluate["_read_responses"](
            path,
            expected_mode="connector",
            connector_expected=True,
        )


def test_connector_presence_gate_passes_matching_long_context_a_b_a() -> None:
    report, status = _evaluate["evaluate"](
        _artifact("baseline", [1.0, 2.0, 3.0], connector=False),
        _artifact("baseline", [1.1, 2.1, 3.1], connector=False),
        _artifact("connector", [1.05, 2.05, 3.05], connector=True),
        latency_ratio_limit=2.0,
        minimum_final_input_tokens=50_000,
    )

    assert status == 0
    assert report["status"] == "PASS_CONNECTOR_PRESENCE_WITHIN_LIMIT"
    assert report["connector_outputs_match"] is True


def test_connector_presence_gate_rejects_latency_divergence() -> None:
    report, status = _evaluate["evaluate"](
        _artifact("baseline", [1.0, 2.0, 3.0], connector=False),
        _artifact("baseline", [1.1, 2.1, 3.1], connector=False),
        _artifact("connector", [3.0, 6.0, 9.0], connector=True),
        latency_ratio_limit=2.0,
        minimum_final_input_tokens=50_000,
    )

    assert status == 1
    assert report["status"] == "FAIL_CONNECTOR_LATENCY_DIVERGED"


def test_connector_presence_gate_rejects_output_divergence() -> None:
    connector = _artifact("connector", [1.0, 2.0, 3.0], connector=True)
    connector["turns"][2]["output_digest"] = "f" * 64

    report, status = _evaluate["evaluate"](
        _artifact("baseline", [1.0, 2.0, 3.0], connector=False),
        _artifact("baseline", [1.1, 2.1, 3.1], connector=False),
        connector,
        latency_ratio_limit=2.0,
        minimum_final_input_tokens=50_000,
    )

    assert status == 1
    assert report["status"] == "FAIL_CONNECTOR_OUTPUT_DIVERGED"
