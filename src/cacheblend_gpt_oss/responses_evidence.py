# SPDX-License-Identifier: Apache-2.0
"""Offline validation for the pinned Responses contract evidence artifact.

The live harness intentionally writes only bounded structural fields and
aggregate metrics.  This module validates that report without importing vLLM,
LMCache, Torch, or CUDA, so a copied artifact can be audited on the authoring
workstation before it is used as an M8 stop/go input.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import NoReturn, cast

from cacheblend_gpt_oss.correctness.capture import (
    VllmPrefillWorkSnapshot,
    VllmTimingSnapshot,
    VllmTimingSummary,
    require_full_prefill_prompt_source_delta,
)
from cacheblend_gpt_oss.correctness.models import CorrectnessRuntimeIdentity

RESPONSES_EVIDENCE_SCHEMA_VERSION = 2
RESPONSES_EVIDENCE_CONTRACT = "gpt_oss_responses_harmony_tool_append_only_multiturn"
_CONNECTOR_KEYS = frozenset(
    {
        "requests",
        "reusable_document_tokens_requested",
        "kv_tokens_found",
        "kv_tokens_loaded",
        "kv_tokens_rejected",
        "tokens_recomputed",
        "prefill_tokens_avoided",
    }
)
_TIMING_KEYS = frozenset(
    {
        "ttft_seconds",
        "end_to_end_latency_seconds",
        "queue_latency_seconds",
        "prefill_latency_seconds",
        "decode_latency_seconds",
    }
)
_PROMPT_SOURCE_KEYS = frozenset(
    {"local_compute", "local_cache_hit", "external_kv_transfer"}
)
_RUNTIME_KEYS = frozenset(
    {
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "plugin_commit",
        "model_config_digest",
        "kv_cache_config_digest",
        "vllm_version",
        "lmcache_version",
        "torch_version",
        "cuda_runtime",
        "gpu_name",
        "dtype",
    }
)
_HARMONY_OUTPUT_TYPES = frozenset({"reasoning", "function_call", "message"})


class ResponsesEvidenceErrorCode(str, Enum):
    """Bounded report validation failures."""

    INVALID_JSON = "invalid_json"
    INVALID_SCHEMA = "invalid_schema"
    NOT_PASSED = "not_passed"
    INVALID_RUNTIME = "invalid_runtime"
    INVALID_TURNS = "invalid_turns"
    INVALID_TOOL = "invalid_tool"
    INVALID_APPEND_ONLY = "invalid_append_only"
    INVALID_CONNECTOR_METRICS = "invalid_connector_metrics"
    INVALID_PROMPT_METRICS = "invalid_prompt_metrics"
    INVALID_PROMPT_SOURCE_METRICS = "invalid_prompt_source_metrics"
    INVALID_PREFILL_WORK = "invalid_prefill_work"
    INVALID_TIMINGS = "invalid_timings"
    FILE_ERROR = "file_error"


class ResponsesEvidenceError(ValueError):
    """Fail-closed error with a bounded code and no report contents."""

    def __init__(self, code: str) -> None:
        normalized = (
            code.value if isinstance(code, ResponsesEvidenceErrorCode) else code
        )
        self.code = normalized
        super().__init__(f"invalid Responses evidence: {normalized}")


def _fail(code: str) -> NoReturn:
    raise ResponsesEvidenceError(code)


def _mapping(value: object, code: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        _fail(code)
    return value


def _exact_mapping(
    value: object, keys: frozenset[str], code: str
) -> Mapping[str, object]:
    mapping = _mapping(value, code)
    if frozenset(mapping) != keys:
        _fail(code)
    return mapping


def _bounded_count(value: object, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(code)
    return value


def _bounded_text(value: object, code: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 256
        or value.strip() != value
        or any(character.isspace() for character in value)
    ):
        _fail(code)
    return value


@dataclass(frozen=True, slots=True)
class ResponsesTurnEvidence:
    """Bounded structure for one of the three replayed Responses turns."""

    output_types: tuple[str, ...]
    reasoning_items: int
    function_calls: int | None = None
    message_text_parts: int | None = None


@dataclass(frozen=True, slots=True)
class ResponsesToolEvidence:
    """Bounded tool-call facts without call IDs, arguments, or message text."""

    name: str
    argument_keys: tuple[str, ...]
    result_city_observed: bool


@dataclass(frozen=True, slots=True)
class ResponsesAppendOnlyEvidence:
    """Lengths proving that each replay input retained prior items."""

    initial: int
    after_tool: int
    final_input: int


@dataclass(frozen=True, slots=True)
class ResponsesContractEvidence:
    """Validated, identifier-free M8 evidence."""

    runtime: CorrectnessRuntimeIdentity
    turns: tuple[ResponsesTurnEvidence, ...]
    tool: ResponsesToolEvidence
    append_only_item_counts: ResponsesAppendOnlyEvidence
    connector_counter_delta: dict[str, int]
    native_prompt_tokens_processed: int
    native_prompt_source_delta: dict[str, int]
    native_prefill_work: VllmPrefillWorkSnapshot
    vllm_timing_delta: VllmTimingSnapshot


def _parse_runtime(value: object) -> CorrectnessRuntimeIdentity:
    mapping = _exact_mapping(
        value, _RUNTIME_KEYS, ResponsesEvidenceErrorCode.INVALID_RUNTIME
    )
    try:
        return CorrectnessRuntimeIdentity(**cast(dict[str, str], dict(mapping)))
    except (TypeError, ValueError):
        _fail(ResponsesEvidenceErrorCode.INVALID_RUNTIME)


def _parse_turns(value: object) -> tuple[ResponsesTurnEvidence, ...]:
    if not isinstance(value, list) or len(value) != 3:
        _fail(ResponsesEvidenceErrorCode.INVALID_TURNS)
    turns: list[ResponsesTurnEvidence] = []
    for index, raw in enumerate(value):
        if index == 0:
            mapping = _exact_mapping(
                raw,
                frozenset({"output_types", "reasoning_items", "function_calls"}),
                ResponsesEvidenceErrorCode.INVALID_TURNS,
            )
        else:
            mapping = _exact_mapping(
                raw,
                frozenset({"output_types", "reasoning_items", "message_text_parts"}),
                ResponsesEvidenceErrorCode.INVALID_TURNS,
            )
        output_types = mapping["output_types"]
        if (
            not isinstance(output_types, list)
            or not output_types
            or any(
                not isinstance(item, str) or not item or len(item) > 128
                for item in output_types
            )
            or any(item not in _HARMONY_OUTPUT_TYPES for item in output_types)
        ):
            _fail(ResponsesEvidenceErrorCode.INVALID_TURNS)
        reasoning_items = _bounded_count(
            mapping["reasoning_items"], ResponsesEvidenceErrorCode.INVALID_TURNS
        )
        if index == 0:
            if (
                reasoning_items < 1
                or reasoning_items != output_types.count("reasoning")
                or output_types[0] != "reasoning"
            ):
                _fail(ResponsesEvidenceErrorCode.INVALID_TURNS)
            function_calls = _bounded_count(
                mapping["function_calls"], ResponsesEvidenceErrorCode.INVALID_TURNS
            )
            if (
                function_calls != 1
                or function_calls != output_types.count("function_call")
                or output_types[-1] != "function_call"
                or any(item != "reasoning" for item in output_types[:-1])
            ):
                _fail(ResponsesEvidenceErrorCode.INVALID_TURNS)
            turns.append(
                ResponsesTurnEvidence(
                    tuple(output_types), reasoning_items, function_calls, None
                )
            )
        else:
            message_text_parts = _bounded_count(
                mapping["message_text_parts"], ResponsesEvidenceErrorCode.INVALID_TURNS
            )
            reasoning_prefix = (
                reasoning_items >= 1
                and reasoning_items == output_types.count("reasoning")
                and output_types[0] == "reasoning"
                and all(item == "reasoning" for item in output_types[:-1])
            )
            message_only = reasoning_items == 0 and output_types == ["message"]
            if (
                message_text_parts < 1
                or "function_call" in output_types
                or output_types[-1] != "message"
                or not (reasoning_prefix or message_only)
            ):
                _fail(ResponsesEvidenceErrorCode.INVALID_TURNS)
            turns.append(
                ResponsesTurnEvidence(
                    tuple(output_types), reasoning_items, None, message_text_parts
                )
            )
    return tuple(turns)


def _parse_tool(value: object) -> ResponsesToolEvidence:
    mapping = _exact_mapping(
        value,
        frozenset({"name", "argument_keys", "result_city_observed"}),
        ResponsesEvidenceErrorCode.INVALID_TOOL,
    )
    name = _bounded_text(mapping["name"], ResponsesEvidenceErrorCode.INVALID_TOOL)
    argument_keys = mapping["argument_keys"]
    if (
        name != "get_weather"
        or not isinstance(argument_keys, list)
        or tuple(argument_keys) != ("city",)
        or mapping["result_city_observed"] is not True
    ):
        _fail(ResponsesEvidenceErrorCode.INVALID_TOOL)
    return ResponsesToolEvidence(name, ("city",), True)


def _parse_append_only(value: object) -> ResponsesAppendOnlyEvidence:
    mapping = _exact_mapping(
        value,
        frozenset({"initial", "after_tool", "final_input"}),
        ResponsesEvidenceErrorCode.INVALID_APPEND_ONLY,
    )
    initial = _bounded_count(
        mapping["initial"], ResponsesEvidenceErrorCode.INVALID_APPEND_ONLY
    )
    after_tool = _bounded_count(
        mapping["after_tool"], ResponsesEvidenceErrorCode.INVALID_APPEND_ONLY
    )
    final_input = _bounded_count(
        mapping["final_input"], ResponsesEvidenceErrorCode.INVALID_APPEND_ONLY
    )
    if initial != 1 or after_tool <= initial or final_input <= after_tool:
        _fail(ResponsesEvidenceErrorCode.INVALID_APPEND_ONLY)
    return ResponsesAppendOnlyEvidence(initial, after_tool, final_input)


def _parse_connector(value: object) -> dict[str, int]:
    mapping = _exact_mapping(
        value, _CONNECTOR_KEYS, ResponsesEvidenceErrorCode.INVALID_CONNECTOR_METRICS
    )
    result = {
        key: _bounded_count(
            mapping[key], ResponsesEvidenceErrorCode.INVALID_CONNECTOR_METRICS
        )
        for key in _CONNECTOR_KEYS
    }
    if (
        result["requests"] != 3
        or result["tokens_recomputed"] <= 0
        or result["prefill_tokens_avoided"] != 0
        or result["kv_tokens_found"] > result["reusable_document_tokens_requested"]
        or result["kv_tokens_loaded"] > result["kv_tokens_found"]
        or result["kv_tokens_found"]
        != result["kv_tokens_loaded"] + result["kv_tokens_rejected"]
    ):
        _fail(ResponsesEvidenceErrorCode.INVALID_CONNECTOR_METRICS)
    return result


def _parse_timings(value: object) -> VllmTimingSnapshot:
    mapping = _exact_mapping(
        value, _TIMING_KEYS, ResponsesEvidenceErrorCode.INVALID_TIMINGS
    )
    summaries: dict[str, VllmTimingSummary] = {}
    for key in _TIMING_KEYS:
        family = _exact_mapping(
            mapping[key],
            frozenset({"count", "sum_seconds", "mean_seconds"}),
            ResponsesEvidenceErrorCode.INVALID_TIMINGS,
        )
        count = _bounded_count(
            family["count"], ResponsesEvidenceErrorCode.INVALID_TIMINGS
        )
        raw_sum = family["sum_seconds"]
        raw_mean = family["mean_seconds"]
        if (
            isinstance(raw_sum, bool)
            or not isinstance(raw_sum, int | float)
            or not math.isfinite(float(raw_sum))
            or float(raw_sum) < 0.0
            or count != 3
            or raw_mean is None
            or isinstance(raw_mean, bool)
            or not isinstance(raw_mean, int | float)
            or not math.isfinite(float(raw_mean))
            or float(raw_mean) < 0.0
            or not math.isclose(float(raw_mean), float(raw_sum) / count, rel_tol=1e-9)
        ):
            _fail(ResponsesEvidenceErrorCode.INVALID_TIMINGS)
        summaries[key] = VllmTimingSummary(count, float(raw_sum))
    return VllmTimingSnapshot(**summaries)


def _parse_prompt_source(
    value: object,
    *,
    expected_prompt_tokens: int,
) -> dict[str, int]:
    mapping = _exact_mapping(
        value,
        _PROMPT_SOURCE_KEYS,
        ResponsesEvidenceErrorCode.INVALID_PROMPT_SOURCE_METRICS,
    )
    delta = {
        key: _bounded_count(
            mapping[key], ResponsesEvidenceErrorCode.INVALID_PROMPT_SOURCE_METRICS
        )
        for key in _PROMPT_SOURCE_KEYS
    }
    try:
        require_full_prefill_prompt_source_delta(
            delta,
            expected_prompt_tokens=expected_prompt_tokens,
        )
    except ValueError:
        _fail(ResponsesEvidenceErrorCode.INVALID_PROMPT_SOURCE_METRICS)
    return delta


def responses_contract_evidence_from_dict(data: object) -> ResponsesContractEvidence:
    """Validate and decode one generated report."""

    root = _exact_mapping(
        data,
        frozenset(
            {
                "schema_version",
                "contract",
                "runtime",
                "passed",
                "turns",
                "tool",
                "append_only_item_counts",
                "connector_counter_delta",
                "native_prompt_tokens_processed",
                "native_prompt_source_delta",
                "native_prefill_work",
                "vllm_timing_delta",
            }
        ),
        ResponsesEvidenceErrorCode.INVALID_SCHEMA,
    )
    if (
        isinstance(root["schema_version"], bool)
        or not isinstance(root["schema_version"], int)
        or root["schema_version"] != RESPONSES_EVIDENCE_SCHEMA_VERSION
    ):
        _fail(ResponsesEvidenceErrorCode.INVALID_SCHEMA)
    if root["contract"] != RESPONSES_EVIDENCE_CONTRACT:
        _fail(ResponsesEvidenceErrorCode.INVALID_SCHEMA)
    if root["passed"] is not True:
        _fail(ResponsesEvidenceErrorCode.NOT_PASSED)
    runtime = _parse_runtime(root["runtime"])
    turns = _parse_turns(root["turns"])
    tool = _parse_tool(root["tool"])
    append_only = _parse_append_only(root["append_only_item_counts"])
    connector = _parse_connector(root["connector_counter_delta"])
    native_prompt = _bounded_count(
        root["native_prompt_tokens_processed"],
        ResponsesEvidenceErrorCode.INVALID_PROMPT_METRICS,
    )
    if native_prompt <= 0:
        _fail(ResponsesEvidenceErrorCode.INVALID_PROMPT_METRICS)
    native_prompt_source = _parse_prompt_source(
        root["native_prompt_source_delta"],
        expected_prompt_tokens=native_prompt,
    )
    if connector["tokens_recomputed"] != native_prompt:
        _fail(ResponsesEvidenceErrorCode.INVALID_CONNECTOR_METRICS)
    if connector["reusable_document_tokens_requested"] > native_prompt:
        _fail(ResponsesEvidenceErrorCode.INVALID_CONNECTOR_METRICS)
    native_prefill = _exact_mapping(
        root["native_prefill_work"],
        frozenset({"observations", "kv_computed_tokens"}),
        ResponsesEvidenceErrorCode.INVALID_PREFILL_WORK,
    )
    prefill = VllmPrefillWorkSnapshot(
        _bounded_count(
            native_prefill["observations"],
            ResponsesEvidenceErrorCode.INVALID_PREFILL_WORK,
        ),
        _bounded_count(
            native_prefill["kv_computed_tokens"],
            ResponsesEvidenceErrorCode.INVALID_PREFILL_WORK,
        ),
    )
    if prefill.observations != 3 or prefill.kv_computed_tokens != native_prompt:
        _fail(ResponsesEvidenceErrorCode.INVALID_PREFILL_WORK)
    return ResponsesContractEvidence(
        runtime=runtime,
        turns=turns,
        tool=tool,
        append_only_item_counts=append_only,
        connector_counter_delta=connector,
        native_prompt_tokens_processed=native_prompt,
        native_prompt_source_delta=native_prompt_source,
        native_prefill_work=prefill,
        vllm_timing_delta=_parse_timings(root["vllm_timing_delta"]),
    )


def read_responses_contract_evidence(path: Path) -> ResponsesContractEvidence:
    """Read and validate a generated report from a local path."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError):
        _fail(ResponsesEvidenceErrorCode.FILE_ERROR)
    except json.JSONDecodeError:
        _fail(ResponsesEvidenceErrorCode.INVALID_JSON)
    return responses_contract_evidence_from_dict(data)


def canonical_responses_contract_bytes(
    evidence: ResponsesContractEvidence,
) -> bytes:
    """Return canonical bytes for an identifier-free evidence digest."""

    payload = {
        "schema_version": RESPONSES_EVIDENCE_SCHEMA_VERSION,
        "contract": RESPONSES_EVIDENCE_CONTRACT,
        "runtime": {
            key: getattr(evidence.runtime, key) for key in sorted(_RUNTIME_KEYS)
        },
        "turns": [
            {
                "output_types": list(turn.output_types),
                "reasoning_items": turn.reasoning_items,
                **(
                    {"function_calls": turn.function_calls}
                    if turn.function_calls is not None
                    else {"message_text_parts": turn.message_text_parts}
                ),
            }
            for turn in evidence.turns
        ],
        "tool": {
            "name": evidence.tool.name,
            "argument_keys": list(evidence.tool.argument_keys),
            "result_city_observed": evidence.tool.result_city_observed,
        },
        "append_only_item_counts": {
            "initial": evidence.append_only_item_counts.initial,
            "after_tool": evidence.append_only_item_counts.after_tool,
            "final_input": evidence.append_only_item_counts.final_input,
        },
        "connector_counter_delta": evidence.connector_counter_delta,
        "native_prompt_tokens_processed": evidence.native_prompt_tokens_processed,
        "native_prompt_source_delta": evidence.native_prompt_source_delta,
        "native_prefill_work": {
            "observations": evidence.native_prefill_work.observations,
            "kv_computed_tokens": evidence.native_prefill_work.kv_computed_tokens,
        },
        "vllm_timing_delta": evidence.vllm_timing_delta.as_dict(),
    }
    return (
        json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def responses_contract_evidence_digest(evidence: ResponsesContractEvidence) -> str:
    """Return a stable SHA-256 digest of validated evidence."""

    return sha256(canonical_responses_contract_bytes(evidence)).hexdigest()


__all__ = [
    "RESPONSES_EVIDENCE_CONTRACT",
    "RESPONSES_EVIDENCE_SCHEMA_VERSION",
    "ResponsesAppendOnlyEvidence",
    "ResponsesContractEvidence",
    "ResponsesEvidenceError",
    "ResponsesEvidenceErrorCode",
    "ResponsesToolEvidence",
    "ResponsesTurnEvidence",
    "canonical_responses_contract_bytes",
    "read_responses_contract_evidence",
    "responses_contract_evidence_digest",
    "responses_contract_evidence_from_dict",
]
