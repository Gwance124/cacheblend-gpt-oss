from __future__ import annotations

from copy import deepcopy

import pytest

from cacheblend_gpt_oss.responses_evidence import (
    RESPONSES_EVIDENCE_CONTRACT,
    RESPONSES_EVIDENCE_SCHEMA_VERSION,
    ResponsesEvidenceError,
    responses_contract_evidence_digest,
    responses_contract_evidence_from_dict,
)


def _runtime() -> dict[str, str]:
    return {
        "model_id": "openai/gpt-oss-20b",
        "model_revision": "model-revision",
        "tokenizer_revision": "tokenizer-revision",
        "plugin_commit": "a" * 40,
        "model_config_digest": "b" * 64,
        "kv_cache_config_digest": "c" * 64,
        "vllm_version": "0.19.1",
        "lmcache_version": "0.4.3",
        "torch_version": "2.10.0+cu128",
        "cuda_runtime": "12.8",
        "gpu_name": "NVIDIA A100-SXM4-80GB",
        "dtype": "torch.bfloat16",
    }


def _report() -> dict[str, object]:
    timing = {
        name: {"count": 3, "sum_seconds": 3.0, "mean_seconds": 1.0}
        for name in (
            "ttft_seconds",
            "end_to_end_latency_seconds",
            "queue_latency_seconds",
            "prefill_latency_seconds",
            "decode_latency_seconds",
        )
    }
    return {
        "schema_version": RESPONSES_EVIDENCE_SCHEMA_VERSION,
        "contract": RESPONSES_EVIDENCE_CONTRACT,
        "runtime": _runtime(),
        "passed": True,
        "turns": [
            {
                "output_types": ["reasoning", "function_call"],
                "reasoning_items": 1,
                "function_calls": 1,
            },
            {
                "output_types": ["reasoning", "message"],
                "reasoning_items": 1,
                "message_text_parts": 1,
            },
            {
                "output_types": ["reasoning", "message"],
                "reasoning_items": 1,
                "message_text_parts": 1,
            },
        ],
        "tool": {
            "name": "get_weather",
            "argument_keys": ["city"],
            "result_city_observed": True,
        },
        "append_only_item_counts": {"initial": 1, "after_tool": 4, "final_input": 7},
        "connector_counter_delta": {
            "requests": 3,
            "reusable_document_tokens_requested": 0,
            "kv_tokens_found": 0,
            "kv_tokens_loaded": 0,
            "kv_tokens_rejected": 0,
            "tokens_recomputed": 811,
            "prefill_tokens_avoided": 0,
        },
        "native_prompt_tokens_processed": 811,
        "native_prompt_source_delta": {
            "local_compute": 811,
            "local_cache_hit": 0,
            "external_kv_transfer": 0,
        },
        "native_prefill_work": {"observations": 3, "kv_computed_tokens": 811},
        "vllm_timing_delta": timing,
    }


def test_valid_report_is_decoded_and_digest_is_stable() -> None:
    evidence = responses_contract_evidence_from_dict(_report())
    assert len(evidence.turns) == 3
    assert evidence.native_prompt_source_delta["local_compute"] == 811
    assert evidence.native_prefill_work.kv_computed_tokens == 811
    assert len(responses_contract_evidence_digest(evidence)) == 64
    assert responses_contract_evidence_digest(evidence) == (
        responses_contract_evidence_digest(evidence)
    )


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("passed", False, "not_passed"),
        ("contract", "other", "invalid_schema"),
        ("schema_version", 1, "invalid_schema"),
        ("native_prompt_tokens_processed", 810, "invalid_prompt_source_metrics"),
    ],
)
def test_report_tampering_fails_closed(
    field: str,
    value: object,
    code: str,
) -> None:
    report = _report()
    report[field] = value
    with pytest.raises(ResponsesEvidenceError, match=code):
        responses_contract_evidence_from_dict(report)


def test_nested_counter_and_timing_tampering_is_rejected() -> None:
    report = _report()
    connector = report["connector_counter_delta"]
    assert isinstance(connector, dict)
    connector["kv_tokens_found"] = 1
    with pytest.raises(ResponsesEvidenceError, match="invalid_connector_metrics"):
        responses_contract_evidence_from_dict(report)

    report = _report()
    timing = report["vllm_timing_delta"]
    assert isinstance(timing, dict)
    timing["ttft_seconds"]["count"] = 2  # type: ignore[index]
    with pytest.raises(ResponsesEvidenceError, match="invalid_timings"):
        responses_contract_evidence_from_dict(report)

    report = _report()
    source = report["native_prompt_source_delta"]
    assert isinstance(source, dict)
    source["external_kv_transfer"] = 1
    with pytest.raises(
        ResponsesEvidenceError,
        match="invalid_prompt_source_metrics",
    ):
        responses_contract_evidence_from_dict(report)

    report = _report()
    prefill = report["native_prefill_work"]
    assert isinstance(prefill, dict)
    prefill["kv_computed_tokens"] = 810
    with pytest.raises(ResponsesEvidenceError, match="invalid_prefill_work"):
        responses_contract_evidence_from_dict(report)

    report = _report()
    connector = report["connector_counter_delta"]
    assert isinstance(connector, dict)
    connector["tokens_recomputed"] = 810
    with pytest.raises(ResponsesEvidenceError, match="invalid_connector_metrics"):
        responses_contract_evidence_from_dict(report)

    report = _report()
    connector = report["connector_counter_delta"]
    assert isinstance(connector, dict)
    connector["reusable_document_tokens_requested"] = 1
    connector["kv_tokens_found"] = 2
    connector["kv_tokens_loaded"] = 2
    with pytest.raises(ResponsesEvidenceError, match="invalid_connector_metrics"):
        responses_contract_evidence_from_dict(report)

    report = _report()
    connector = report["connector_counter_delta"]
    assert isinstance(connector, dict)
    connector["kv_tokens_found"] = 1
    connector["kv_tokens_loaded"] = 2
    connector["kv_tokens_rejected"] = 0
    connector["reusable_document_tokens_requested"] = 2
    with pytest.raises(ResponsesEvidenceError, match="invalid_connector_metrics"):
        responses_contract_evidence_from_dict(report)

    report = _report()
    connector = report["connector_counter_delta"]
    assert isinstance(connector, dict)
    connector["reusable_document_tokens_requested"] = 812
    with pytest.raises(ResponsesEvidenceError, match="invalid_connector_metrics"):
        responses_contract_evidence_from_dict(report)


def test_extra_keys_and_turn_structure_are_rejected() -> None:
    report = _report()
    report["unexpected"] = True
    with pytest.raises(ResponsesEvidenceError, match="invalid_schema"):
        responses_contract_evidence_from_dict(report)

    report = deepcopy(_report())
    turns = report["turns"]
    assert isinstance(turns, list)
    turns[1]["function_calls"] = 0  # type: ignore[index]
    with pytest.raises(ResponsesEvidenceError, match="invalid_turns"):
        responses_contract_evidence_from_dict(report)

    report = deepcopy(_report())
    turns = report["turns"]
    assert isinstance(turns, list)
    turns[1]["output_types"] = [  # type: ignore[index]
        "reasoning",
        "custom_tool_call",
        "message",
    ]
    with pytest.raises(ResponsesEvidenceError, match="invalid_turns"):
        responses_contract_evidence_from_dict(report)

    report = deepcopy(_report())
    turns = report["turns"]
    assert isinstance(turns, list)
    turns[0]["output_types"] = [  # type: ignore[index]
        "message",
        "reasoning",
        "function_call",
    ]
    with pytest.raises(ResponsesEvidenceError, match="invalid_turns"):
        responses_contract_evidence_from_dict(report)

    report = deepcopy(_report())
    turns = report["turns"]
    assert isinstance(turns, list)
    turns[0]["reasoning_items"] = 2  # type: ignore[index]
    with pytest.raises(ResponsesEvidenceError, match="invalid_turns"):
        responses_contract_evidence_from_dict(report)

    report = deepcopy(_report())
    turns = report["turns"]
    assert isinstance(turns, list)
    turns[1]["output_types"] = [  # type: ignore[index]
        "reasoning",
        "function_call",
        "message",
    ]
    with pytest.raises(ResponsesEvidenceError, match="invalid_turns"):
        responses_contract_evidence_from_dict(report)
