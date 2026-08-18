"""CPU-only tests for the pinned hybrid-manager flag diagnostic."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from cacheblend_gpt_oss.correctness import (
    ARTIFACT_SCHEMA_VERSION,
    GPT_OSS_VOCAB_SIZE,
    CorrectnessArtifact,
    CorrectnessRunMode,
    CorrectnessRuntimeIdentity,
    FullVocabularyLogprobs,
    build_moved_document_fixture,
)
from cacheblend_gpt_oss.responses_contract import parse_completed_response

_capture = runpy.run_path(
    "scripts/capture_hybrid_flag_responses.py",
    run_name="hybrid_flag_responses_test",
)
_resolution = runpy.run_path(
    "scripts/capture_hybrid_flag_resolution.py",
    run_name="hybrid_flag_resolution_test",
)
_evaluate = runpy.run_path(
    "scripts/evaluate_hybrid_flag_equivalence.py",
    run_name="hybrid_flag_equivalence_test",
)


def _runtime() -> CorrectnessRuntimeIdentity:
    return CorrectnessRuntimeIdentity(
        model_id="openai/gpt-oss-20b",
        model_revision="model-revision",
        tokenizer_revision="tokenizer-revision",
        plugin_commit="a" * 40,
        model_config_digest="b" * 64,
        kv_cache_config_digest="c" * 64,
        vllm_version="0.19.1",
        lmcache_version="0.4.3",
        torch_version="2.10.0+cu128",
        cuda_runtime="12.8",
        gpu_name="NVIDIA A100-SXM4-80GB",
    )


def _artifact(logprob: float = -10.0) -> CorrectnessArtifact:
    values = (logprob,) * GPT_OSS_VOCAB_SIZE
    return CorrectnessArtifact(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        run_mode=CorrectnessRunMode.FULL_PREFILL,
        runtime=_runtime(),
        prompt=build_moved_document_fixture().prompt_identity,
        distribution=FullVocabularyLogprobs(values, 7),
    )


def _responses(mode: str, elapsed: float) -> dict[str, object]:
    usage = {
        "input_tokens": 100,
        "output_tokens": 10,
        "total_tokens": 110,
        "cached_tokens": 64,
        "reasoning_tokens": 8,
        "tool_output_tokens": 0,
    }
    canonical_output = {
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Paris"}],
            }
        ],
    }
    turns = [
        {
            "request_digest": str(index) * 64,
            "output_digest": _evaluate["_digest"](canonical_output),
            "canonical_output": canonical_output,
            "usage": dict(usage),
            "elapsed_seconds": elapsed / 3,
        }
        for index in range(3)
    ]
    return {
        "schema_version": 1,
        "contract": "cacheblend-gpt-oss-hybrid-flag-responses-v1",
        "mode": mode,
        "model": "openai/gpt-oss-20b",
        "sampling": {"temperature": 0.0, "top_p": 1.0, "seed": 0},
        "stable_replay_ids": True,
        "turns": turns,
        "prefix_cache": {
            "cached_tokens_per_turn": [64, 64, 64],
            "reuse_observed_after_cold_turn": True,
        },
        "total_elapsed_seconds": elapsed,
    }


def _resolution_artifact() -> dict[str, object]:
    return {
        "schema_version": 1,
        "contract": "cacheblend-gpt-oss-hybrid-flag-resolution-v1",
        "resolved_snapshots_equal": True,
        "resolved_disable_hybrid_kv_cache_manager": False,
        "gate_passed": True,
    }


def test_responses_probe_pins_sampling_and_stabilizes_replay_ids() -> None:
    payload = _capture["_request_payload"]([{"role": "user", "content": "hello"}])
    assert payload["temperature"] == 0.0
    assert payload["top_p"] == 1.0
    assert payload["seed"] == 0
    assert payload["store"] is False

    raw = {
        "id": "response-random",
        "status": "completed",
        "output": [
            {
                "id": "reasoning-random",
                "type": "reasoning",
                "status": "completed",
                "summary": [],
            },
            {
                "id": "call-random",
                "type": "function_call",
                "status": "completed",
                "encrypted_content": "generated-opaque-value",
                "call_id": "generated-call-id",
                "name": "get_weather",
                "arguments": '{ "city": "Paris" }',
            },
        ],
    }
    parsed = parse_completed_response(raw)
    items, call_ids = _capture["_stable_replay_items"](parsed, turn=1)

    assert items[0]["id"] == "rs_turn_1_0"
    assert items[1]["id"] == "fc_turn_1_1"
    assert items[1]["call_id"] == "call_turn_1_0"
    assert items[1]["arguments"] == '{"city":"Paris"}'
    assert "encrypted_content" not in items[1]
    assert call_ids == {"generated-call-id": "call_turn_1_0"}


def test_response_canonicalization_removes_only_dynamic_identifiers() -> None:
    canonicalize = _capture["_canonical_response_value"]
    left = canonicalize(
        {
            "id": "one",
            "call_id": "random-one",
            "name": "get_weather",
            "arguments": '{"city":"Paris","unit":"C"}',
        }
    )
    right = canonicalize(
        {
            "id": "two",
            "call_id": "random-two",
            "name": "get_weather",
            "arguments": '{"unit":"C","city":"Paris"}',
        }
    )

    assert (
        left
        == right
        == {
            "name": "get_weather",
            "arguments": {"city": "Paris", "unit": "C"},
        }
    )


def test_resolution_snapshot_captures_final_scheduler_value() -> None:
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False,
            max_num_batched_tokens=131_072,
            max_num_seqs=1,
            max_model_len=131_072,
            enable_chunked_prefill=True,
            long_prefill_token_threshold=0,
            async_scheduling=False,
            scheduler_reserve_full_isl=True,
            scheduler_cls="scheduler",
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=True,
            block_size=16,
            cache_dtype="auto",
            sliding_window=None,
        ),
        model_config=SimpleNamespace(
            max_model_len=131_072,
            attention_chunk_size=None,
            dtype="bfloat16",
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        kv_transfer_config=None,
    )

    snapshot = _resolution["resolved_snapshot"](config)

    assert snapshot["scheduler"]["disable_hybrid_kv_cache_manager"] is False
    assert snapshot["kv_transfer_config_present"] is False


def test_response_reader_recalculates_prefix_reuse(tmp_path: Path) -> None:
    artifact = _responses("implicit", 12.0)
    artifact["prefix_cache"]["cached_tokens_per_turn"] = [64, 0, 0]
    path = tmp_path / "responses.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="prefix-cache evidence"):
        _evaluate["_read_responses"](path, "implicit")


def test_equivalence_gate_passes_matching_a_b_a(monkeypatch) -> None:
    artifact = _artifact()
    monkeypatch.setitem(
        _evaluate["evaluate"].__globals__,
        "read_artifact",
        lambda _path: artifact,
    )

    report, status = _evaluate["evaluate"](
        implicit_a_responses=_responses("implicit", 10.0),
        implicit_b_responses=_responses("implicit", 12.0),
        explicit_responses=_responses("explicit_false", 11.0),
        implicit_a_logits=Path("implicit-a.json"),
        implicit_b_logits=Path("implicit-b.json"),
        explicit_logits=Path("explicit.json"),
        resolution=_resolution_artifact(),
        latency_ratio_limit=2.0,
    )

    assert status == 0
    assert report["status"] == "PASS_IMPLICIT_EQUALS_EXPLICIT_FALSE"
    assert report["passed"] is True


def test_equivalence_gate_rejects_explicit_response_divergence(monkeypatch) -> None:
    artifact = _artifact()
    monkeypatch.setitem(
        _evaluate["evaluate"].__globals__,
        "read_artifact",
        lambda _path: artifact,
    )
    explicit = _responses("explicit_false", 11.0)
    explicit["turns"][1]["output_digest"] = "f" * 64

    report, status = _evaluate["evaluate"](
        implicit_a_responses=_responses("implicit", 4.0),
        implicit_b_responses=_responses("implicit", 12.0),
        explicit_responses=explicit,
        implicit_a_logits=Path("implicit-a.json"),
        implicit_b_logits=Path("implicit-b.json"),
        explicit_logits=Path("explicit.json"),
        resolution=_resolution_artifact(),
        latency_ratio_limit=2.0,
    )

    assert status == 1
    assert report["status"] == "FAIL_EXPLICIT_FALSE_OUTPUT_DIVERGED"


def test_equivalence_gate_rejects_logit_divergence(monkeypatch) -> None:
    baseline = _artifact()
    divergent = _artifact(-9.0)
    monkeypatch.setitem(
        _evaluate["evaluate"].__globals__,
        "read_artifact",
        lambda path: divergent if path.name == "explicit.json" else baseline,
    )

    report, status = _evaluate["evaluate"](
        implicit_a_responses=_responses("implicit", 10.0),
        implicit_b_responses=_responses("implicit", 12.0),
        explicit_responses=_responses("explicit_false", 11.0),
        implicit_a_logits=Path("implicit-a.json"),
        implicit_b_logits=Path("implicit-b.json"),
        explicit_logits=Path("explicit.json"),
        resolution=_resolution_artifact(),
        latency_ratio_limit=2.0,
    )

    assert status == 1
    assert report["status"] == "FAIL_EXPLICIT_FALSE_OUTPUT_DIVERGED"
    assert report["numerical_within_baseline_envelope"] is False
