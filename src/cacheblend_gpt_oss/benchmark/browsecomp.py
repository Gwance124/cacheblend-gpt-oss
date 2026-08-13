# SPDX-License-Identifier: Apache-2.0
"""Offline, identifier-free evidence for one BrowseComp-Plus agent run.

The input run record is deliberately treated as an untrusted private artifact.
Only bounded counts and the already-pinned runtime identity cross the boundary
into the returned report.  In particular, query IDs, prompt/answer/reasoning
payloads, response and call IDs, document IDs, and prompt/document fingerprints
are never copied into evidence.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import NoReturn, cast

from cacheblend_gpt_oss.correctness.capture import (
    VllmPrefillWorkSnapshot,
    VllmTimingSnapshot,
    connector_counter_delta,
    connector_store_counter_delta,
    has_connector_metric_surface,
    has_vllm_prefill_work_metric_surface,
    has_vllm_prompt_metric_surface,
    has_vllm_prompt_source_metric_surface,
    has_vllm_timing_metric_surface,
    parse_connector_counter_snapshot,
    parse_connector_store_counter_snapshot,
    parse_vllm_prefill_work_snapshot,
    parse_vllm_prompt_counter_snapshot,
    parse_vllm_prompt_source_snapshot,
    parse_vllm_timing_snapshot,
    require_full_prefill_prompt_source_delta,
    require_vllm_prefill_work_total,
    require_vllm_timing_delta,
    vllm_prefill_work_snapshot_delta,
    vllm_prompt_counter_delta,
    vllm_prompt_source_delta,
    vllm_timing_snapshot_delta,
)
from cacheblend_gpt_oss.correctness.models import CorrectnessRuntimeIdentity

BROWSECOMP_EVIDENCE_SCHEMA_VERSION = 1
BROWSECOMP_EVIDENCE_CONTRACT = "browsecomp_plus_agentic_append_only_transfer_100pct"

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
_CONNECTOR_METRIC_NAMES = frozenset(
    {
        "vllm:cacheblend_requests_total",
        "vllm:cacheblend_reusable_document_tokens_requested_total",
        "vllm:cacheblend_kv_tokens_found_total",
        "vllm:cacheblend_kv_tokens_loaded_total",
        "vllm:cacheblend_kv_tokens_rejected_total",
        "vllm:cacheblend_tokens_recomputed_total",
        "vllm:cacheblend_prefill_tokens_avoided_total",
    }
)
_STORE_METRIC_NAMES = frozenset(
    {
        "vllm:cacheblend_store_tokens_eligible_total",
        "vllm:cacheblend_store_tokens_completed_total",
        "vllm:cacheblend_store_fallbacks_total",
    }
)
_WORKLOAD_KEYS = (
    "searches",
    "retrieval_requests",
    "generation_requests",
    "completed_generation_steps",
    "context_rebuilds",
    "context_edits",
    "context_truncations",
    "invalid_function_calls",
    "generation_retries",
)
_TOKEN_TOTAL_KEYS = ("input_tokens", "output_tokens", "total_tokens")


class BrowseCompEvidenceErrorCode(str, Enum):
    """Bounded failure reasons for the offline smoke validator."""

    INVALID_SCHEMA = "invalid_schema"
    INVALID_STATUS = "invalid_status"
    INVALID_CONFIGURATION = "invalid_configuration"
    INVALID_WORKLOAD = "invalid_workload"
    INVALID_GENERATION = "invalid_generation"
    INVALID_USAGE = "invalid_usage"
    INVALID_CONTEXT = "invalid_context"
    INVALID_RUNTIME = "invalid_runtime"
    INVALID_PROMETHEUS = "invalid_prometheus"
    INVALID_CONNECTOR_METRICS = "invalid_connector_metrics"
    INVALID_STORE_METRICS = "invalid_store_metrics"
    INVALID_NATIVE_METRICS = "invalid_native_metrics"
    INVALID_TIMING_METRICS = "invalid_timing_metrics"
    INVALID_JSON = "invalid_json"
    FILE_ERROR = "file_error"
    OUTPUT_EXISTS = "output_exists"


_ERROR_MESSAGES = {
    BrowseCompEvidenceErrorCode.INVALID_SCHEMA: (
        "BrowseComp-Plus run record schema is not supported"
    ),
    BrowseCompEvidenceErrorCode.INVALID_STATUS: (
        "BrowseComp-Plus run did not complete without an error"
    ),
    BrowseCompEvidenceErrorCode.INVALID_CONFIGURATION: (
        "BrowseComp-Plus run configuration is outside the append-only "
        "CacheBlend contract"
    ),
    BrowseCompEvidenceErrorCode.INVALID_WORKLOAD: (
        "BrowseComp-Plus workload counts do not reconcile"
    ),
    BrowseCompEvidenceErrorCode.INVALID_GENERATION: (
        "BrowseComp-Plus generation steps are not all completed"
    ),
    BrowseCompEvidenceErrorCode.INVALID_USAGE: (
        "BrowseComp-Plus generation usage does not reconcile exactly"
    ),
    BrowseCompEvidenceErrorCode.INVALID_CONTEXT: (
        "BrowseComp-Plus context was edited, truncated, or retried"
    ),
    BrowseCompEvidenceErrorCode.INVALID_RUNTIME: (
        "runtime identity is invalid or outside the pinned target"
    ),
    BrowseCompEvidenceErrorCode.INVALID_PROMETHEUS: (
        "Prometheus snapshots do not expose the required bounded metric families"
    ),
    BrowseCompEvidenceErrorCode.INVALID_CONNECTOR_METRICS: (
        "CacheBlend connector counters do not reconcile"
    ),
    BrowseCompEvidenceErrorCode.INVALID_STORE_METRICS: (
        "CacheBlend store counters do not reconcile"
    ),
    BrowseCompEvidenceErrorCode.INVALID_NATIVE_METRICS: (
        "native vLLM prompt accounting or prefill work does not reconcile"
    ),
    BrowseCompEvidenceErrorCode.INVALID_TIMING_METRICS: (
        "native vLLM timing families do not reconcile"
    ),
    BrowseCompEvidenceErrorCode.INVALID_JSON: "input is not valid JSON",
    BrowseCompEvidenceErrorCode.FILE_ERROR: "input or output file is unavailable",
    BrowseCompEvidenceErrorCode.OUTPUT_EXISTS: "output evidence file already exists",
}


class BrowseCompEvidenceError(ValueError):
    """Fail-closed error whose code and message contain no run data."""

    def __init__(self, code: BrowseCompEvidenceErrorCode | str) -> None:
        try:
            normalized = BrowseCompEvidenceErrorCode(code)
        except (TypeError, ValueError):
            normalized = BrowseCompEvidenceErrorCode.INVALID_SCHEMA
        self.code = normalized
        self.message = _ERROR_MESSAGES[normalized]
        super().__init__(self.message)


def _fail(code: BrowseCompEvidenceErrorCode) -> NoReturn:
    raise BrowseCompEvidenceError(code)


def _mapping(value: object, code: BrowseCompEvidenceErrorCode) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        _fail(code)
    return cast(Mapping[str, object], value)


def _required_mapping(
    mapping: Mapping[str, object],
    key: str,
    code: BrowseCompEvidenceErrorCode,
) -> Mapping[str, object]:
    if key not in mapping:
        _fail(code)
    return _mapping(mapping[key], code)


def _required_list(
    mapping: Mapping[str, object],
    key: str,
    code: BrowseCompEvidenceErrorCode,
) -> list[object]:
    value = mapping.get(key)
    if not isinstance(value, list):
        _fail(code)
    return value


def _count(value: object, code: BrowseCompEvidenceErrorCode) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(code)
    return value


def _required_count(
    mapping: Mapping[str, object],
    key: str,
    code: BrowseCompEvidenceErrorCode,
) -> int:
    if key not in mapping:
        _fail(code)
    return _count(mapping[key], code)


def _runtime_to_dict(runtime: CorrectnessRuntimeIdentity) -> dict[str, str]:
    return {key: str(getattr(runtime, key)) for key in sorted(_RUNTIME_KEYS)}


def runtime_identity_from_dict(value: object) -> CorrectnessRuntimeIdentity:
    """Decode the existing exact ``CorrectnessRuntimeIdentity`` JSON shape."""

    mapping = _mapping(value, BrowseCompEvidenceErrorCode.INVALID_RUNTIME)
    if frozenset(mapping) != _RUNTIME_KEYS:
        _fail(BrowseCompEvidenceErrorCode.INVALID_RUNTIME)
    try:
        return CorrectnessRuntimeIdentity(**cast(dict[str, str], dict(mapping)))
    except (TypeError, ValueError):
        _fail(BrowseCompEvidenceErrorCode.INVALID_RUNTIME)


@dataclass(frozen=True, slots=True)
class _RunFacts:
    """Only aggregate facts allowed to cross from a private run record."""

    searches: int
    retrieval_requests: int
    generation_requests: int
    context_rebuilds: int
    input_tokens: int
    output_tokens: int
    total_tokens: int


def _usage_counts(value: object) -> tuple[int, int, int]:
    mapping = _mapping(value, BrowseCompEvidenceErrorCode.INVALID_USAGE)
    input_tokens = _required_count(
        mapping, "input_tokens", BrowseCompEvidenceErrorCode.INVALID_USAGE
    )
    output_tokens = _required_count(
        mapping, "output_tokens", BrowseCompEvidenceErrorCode.INVALID_USAGE
    )
    total_tokens = _required_count(
        mapping, "total_tokens", BrowseCompEvidenceErrorCode.INVALID_USAGE
    )
    if input_tokens <= 0 or total_tokens != input_tokens + output_tokens:
        _fail(BrowseCompEvidenceErrorCode.INVALID_USAGE)

    input_details = mapping.get("input_tokens_details")
    if input_details is not None:
        details = _mapping(input_details, BrowseCompEvidenceErrorCode.INVALID_USAGE)
        if "cached_tokens" in details:
            cached_tokens = _count(
                details["cached_tokens"], BrowseCompEvidenceErrorCode.INVALID_USAGE
            )
            if cached_tokens != 0:
                _fail(BrowseCompEvidenceErrorCode.INVALID_USAGE)
    output_details = mapping.get("output_tokens_details")
    if output_details is not None:
        details = _mapping(output_details, BrowseCompEvidenceErrorCode.INVALID_USAGE)
        if "reasoning_tokens" in details:
            reasoning_tokens = _count(
                details["reasoning_tokens"], BrowseCompEvidenceErrorCode.INVALID_USAGE
            )
            if reasoning_tokens > output_tokens:
                _fail(BrowseCompEvidenceErrorCode.INVALID_USAGE)
    return input_tokens, output_tokens, total_tokens


def _require_zero_count(
    mapping: Mapping[str, object],
    key: str,
    code: BrowseCompEvidenceErrorCode,
) -> None:
    if _required_count(mapping, key, code) != 0:
        _fail(code)


def _validate_run_record(
    run_record: object,
    runtime: CorrectnessRuntimeIdentity,
) -> _RunFacts:
    if not isinstance(runtime, CorrectnessRuntimeIdentity):
        _fail(BrowseCompEvidenceErrorCode.INVALID_RUNTIME)
    root = _mapping(run_record, BrowseCompEvidenceErrorCode.INVALID_SCHEMA)
    if root.get("schema_version") != "1.0":
        _fail(BrowseCompEvidenceErrorCode.INVALID_SCHEMA)
    if root.get("status") != "completed":
        _fail(BrowseCompEvidenceErrorCode.INVALID_STATUS)
    if "error" in root and root["error"] is not None:
        _fail(BrowseCompEvidenceErrorCode.INVALID_STATUS)

    metadata = _required_mapping(
        root, "metadata", BrowseCompEvidenceErrorCode.INVALID_CONFIGURATION
    )
    diagnostics = _required_mapping(
        root, "diagnostics", BrowseCompEvidenceErrorCode.INVALID_CONFIGURATION
    )
    if metadata.get("model") != "openai/gpt-oss-20b":
        _fail(BrowseCompEvidenceErrorCode.INVALID_CONFIGURATION)
    if metadata.get("api") not in {"responses", "/v1/responses"}:
        _fail(BrowseCompEvidenceErrorCode.INVALID_CONFIGURATION)
    if "endpoint" in metadata and metadata["endpoint"] != "/v1/responses":
        _fail(BrowseCompEvidenceErrorCode.INVALID_CONFIGURATION)
    if metadata.get("scaffold") != "standard_search_only_top5_first512":
        _fail(BrowseCompEvidenceErrorCode.INVALID_CONFIGURATION)
    if metadata.get("cache_mode") != "cacheblend":
        _fail(BrowseCompEvidenceErrorCode.INVALID_CONFIGURATION)
    if diagnostics.get("cache_mode") != "cacheblend":
        _fail(BrowseCompEvidenceErrorCode.INVALID_CONFIGURATION)
    if metadata.get("context_strategy") != "append_only":
        _fail(BrowseCompEvidenceErrorCode.INVALID_CONFIGURATION)
    if diagnostics.get("context_strategy") != "append_only":
        _fail(BrowseCompEvidenceErrorCode.INVALID_CONFIGURATION)
    if diagnostics.get("deduplicate_retrieved_documents") is not False:
        _fail(BrowseCompEvidenceErrorCode.INVALID_CONFIGURATION)
    if "deduplicate_retrieved_documents" in metadata and metadata[
        "deduplicate_retrieved_documents"
    ] not in {None, False}:
        _fail(BrowseCompEvidenceErrorCode.INVALID_CONFIGURATION)

    tool_counts = _required_mapping(
        root, "tool_call_counts", BrowseCompEvidenceErrorCode.INVALID_WORKLOAD
    )
    if set(tool_counts) != {"search"}:
        _fail(BrowseCompEvidenceErrorCode.INVALID_WORKLOAD)
    searches = _required_count(
        tool_counts, "search", BrowseCompEvidenceErrorCode.INVALID_WORKLOAD
    )
    search_steps = _required_list(
        diagnostics, "search_steps", BrowseCompEvidenceErrorCode.INVALID_WORKLOAD
    )
    retrieval_requests = _required_count(
        diagnostics,
        "retrieval_request_count",
        BrowseCompEvidenceErrorCode.INVALID_WORKLOAD,
    )
    if searches < 1 or searches != len(search_steps) or retrieval_requests != searches:
        _fail(BrowseCompEvidenceErrorCode.INVALID_WORKLOAD)
    for index, raw_step in enumerate(search_steps, start=1):
        step = _mapping(raw_step, BrowseCompEvidenceErrorCode.INVALID_WORKLOAD)
        if (
            _required_count(
                step, "search_call", BrowseCompEvidenceErrorCode.INVALID_WORKLOAD
            )
            != index
        ):
            _fail(BrowseCompEvidenceErrorCode.INVALID_WORKLOAD)
        returned = _required_count(
            step,
            "returned_documents",
            BrowseCompEvidenceErrorCode.INVALID_WORKLOAD,
        )
        if returned > 5:
            _fail(BrowseCompEvidenceErrorCode.INVALID_WORKLOAD)
        if "elided_duplicate_documents" in step:
            _require_zero_count(
                step,
                "elided_duplicate_documents",
                BrowseCompEvidenceErrorCode.INVALID_CONTEXT,
            )

    generation_requests = _required_count(
        diagnostics,
        "generation_request_count",
        BrowseCompEvidenceErrorCode.INVALID_WORKLOAD,
    )
    generation_steps = _required_list(
        diagnostics,
        "generation_steps",
        BrowseCompEvidenceErrorCode.INVALID_GENERATION,
    )
    generation_usage = _required_list(
        diagnostics,
        "generation_usage",
        BrowseCompEvidenceErrorCode.INVALID_USAGE,
    )
    if (
        generation_requests < 2
        or len(generation_steps) != generation_requests
        or len(generation_usage) != generation_requests
    ):
        _fail(BrowseCompEvidenceErrorCode.INVALID_WORKLOAD)
    input_tokens_total = 0
    output_tokens_total = 0
    total_tokens_total = 0
    for index, raw_step in enumerate(generation_steps, start=1):
        step = _mapping(raw_step, BrowseCompEvidenceErrorCode.INVALID_GENERATION)
        if (
            _required_count(
                step, "request_number", BrowseCompEvidenceErrorCode.INVALID_GENERATION
            )
            != index
        ):
            _fail(BrowseCompEvidenceErrorCode.INVALID_GENERATION)
        if step.get("response_status") != "completed":
            _fail(BrowseCompEvidenceErrorCode.INVALID_GENERATION)
        if "status" in step and step["status"] != "completed":
            _fail(BrowseCompEvidenceErrorCode.INVALID_GENERATION)
        if "error" in step and step["error"] is not None:
            _fail(BrowseCompEvidenceErrorCode.INVALID_GENERATION)
        step_usage = _usage_counts(step.get("usage"))
        recorded_usage = _usage_counts(generation_usage[index - 1])
        if step_usage != recorded_usage:
            _fail(BrowseCompEvidenceErrorCode.INVALID_USAGE)
        input_tokens_total += recorded_usage[0]
        output_tokens_total += recorded_usage[1]
        total_tokens_total += recorded_usage[2]

    if (
        "generation_request_count" in root
        and root["generation_request_count"] != generation_requests
    ):
        _fail(BrowseCompEvidenceErrorCode.INVALID_WORKLOAD)

    context_rebuilds = _required_list(
        diagnostics, "context_rebuilds", BrowseCompEvidenceErrorCode.INVALID_CONTEXT
    )
    if len(context_rebuilds) != generation_requests:
        _fail(BrowseCompEvidenceErrorCode.INVALID_CONTEXT)
    for index, raw_rebuild in enumerate(context_rebuilds, start=1):
        rebuild = _mapping(raw_rebuild, BrowseCompEvidenceErrorCode.INVALID_CONTEXT)
        if (
            _required_count(
                rebuild, "request_number", BrowseCompEvidenceErrorCode.INVALID_CONTEXT
            )
            != index
        ):
            _fail(BrowseCompEvidenceErrorCode.INVALID_CONTEXT)
        if rebuild.get("context_strategy") != "append_only":
            _fail(BrowseCompEvidenceErrorCode.INVALID_CONTEXT)
        if rebuild.get("first_edited_item_index", object()) is not None:
            _fail(BrowseCompEvidenceErrorCode.INVALID_CONTEXT)
        for key in ("documents_full", "documents_stubbed", "documents_removed"):
            _require_zero_count(
                rebuild, key, BrowseCompEvidenceErrorCode.INVALID_CONTEXT
            )

    truncations = _required_list(
        diagnostics,
        "context_truncation_events",
        BrowseCompEvidenceErrorCode.INVALID_CONTEXT,
    )
    invalid_calls = _required_list(
        diagnostics,
        "invalid_function_calls",
        BrowseCompEvidenceErrorCode.INVALID_CONTEXT,
    )
    if truncations or invalid_calls:
        _fail(BrowseCompEvidenceErrorCode.INVALID_CONTEXT)
    if diagnostics.get("context_budget_triggered") is not False:
        _fail(BrowseCompEvidenceErrorCode.INVALID_CONTEXT)
    if "elided_duplicate_documents_total" in diagnostics:
        _require_zero_count(
            diagnostics,
            "elided_duplicate_documents_total",
            BrowseCompEvidenceErrorCode.INVALID_CONTEXT,
        )
    for key in ("context_edits", "context_edit_events"):
        if key in diagnostics:
            value = diagnostics[key]
            if not isinstance(value, list) or value:
                _fail(BrowseCompEvidenceErrorCode.INVALID_CONTEXT)
    for key in (
        "generation_retry_count",
        "generation_retries",
        "retry_count",
        "retries",
    ):
        if key in diagnostics:
            value = diagnostics[key]
            if isinstance(value, list):
                if value:
                    _fail(BrowseCompEvidenceErrorCode.INVALID_CONTEXT)
            elif _count(value, BrowseCompEvidenceErrorCode.INVALID_CONTEXT) != 0:
                _fail(BrowseCompEvidenceErrorCode.INVALID_CONTEXT)

    if diagnostics.get("termination_reason") != "final_answer":
        _fail(BrowseCompEvidenceErrorCode.INVALID_STATUS)
    final_validation = _required_mapping(
        diagnostics,
        "final_answer_validation",
        BrowseCompEvidenceErrorCode.INVALID_STATUS,
    )
    if final_validation.get("valid") is not True:
        _fail(BrowseCompEvidenceErrorCode.INVALID_STATUS)

    return _RunFacts(
        searches=searches,
        retrieval_requests=retrieval_requests,
        generation_requests=generation_requests,
        context_rebuilds=len(context_rebuilds),
        input_tokens=input_tokens_total,
        output_tokens=output_tokens_total,
        total_tokens=total_tokens_total,
    )


def _metric_names(text: str) -> set[str]:
    return {
        line.strip().split()[0].split("{", 1)[0]
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def _require_metric_surface(
    text: object,
    names: frozenset[str],
    code: BrowseCompEvidenceErrorCode,
) -> str:
    if not isinstance(text, str):
        _fail(code)
    if not names.issubset(_metric_names(text)):
        _fail(BrowseCompEvidenceErrorCode.INVALID_PROMETHEUS)
    return text


def _require_optional_metric_surface(
    text: object,
    names: frozenset[str],
    code: BrowseCompEvidenceErrorCode,
) -> str:
    """Allow a cold zero baseline, but reject an advertised partial family."""

    if not isinstance(text, str):
        _fail(code)
    present = _metric_names(text) & names
    if present and present != names:
        _fail(code)
    return text


def _parse_metric_deltas(
    metrics_before: str,
    metrics_after: str,
    facts: _RunFacts,
) -> tuple[
    dict[str, int],
    dict[str, int],
    int,
    dict[str, int],
    VllmPrefillWorkSnapshot,
    VllmTimingSnapshot,
]:
    before_text = _require_optional_metric_surface(
        metrics_before,
        _CONNECTOR_METRIC_NAMES,
        BrowseCompEvidenceErrorCode.INVALID_CONNECTOR_METRICS,
    )
    after_text = _require_metric_surface(
        metrics_after,
        _CONNECTOR_METRIC_NAMES,
        BrowseCompEvidenceErrorCode.INVALID_CONNECTOR_METRICS,
    )
    _require_optional_metric_surface(
        before_text,
        _STORE_METRIC_NAMES,
        BrowseCompEvidenceErrorCode.INVALID_STORE_METRICS,
    )
    _require_metric_surface(
        after_text,
        _STORE_METRIC_NAMES,
        BrowseCompEvidenceErrorCode.INVALID_STORE_METRICS,
    )
    if not has_connector_metric_surface(after_text):
        _fail(BrowseCompEvidenceErrorCode.INVALID_CONNECTOR_METRICS)
    if not has_vllm_prompt_metric_surface(after_text):
        _fail(BrowseCompEvidenceErrorCode.INVALID_PROMETHEUS)
    if not has_vllm_prompt_source_metric_surface(after_text):
        _fail(BrowseCompEvidenceErrorCode.INVALID_PROMETHEUS)
    if not has_vllm_prefill_work_metric_surface(after_text):
        _fail(BrowseCompEvidenceErrorCode.INVALID_PROMETHEUS)
    if not has_vllm_timing_metric_surface(after_text):
        _fail(BrowseCompEvidenceErrorCode.INVALID_PROMETHEUS)

    try:
        before_connector = parse_connector_counter_snapshot(before_text)
        after_connector = parse_connector_counter_snapshot(after_text)
    except (TypeError, ValueError):
        _fail(BrowseCompEvidenceErrorCode.INVALID_CONNECTOR_METRICS)
    try:
        before_store = parse_connector_store_counter_snapshot(before_text)
        after_store = parse_connector_store_counter_snapshot(after_text)
    except (TypeError, ValueError):
        _fail(BrowseCompEvidenceErrorCode.INVALID_STORE_METRICS)
    try:
        connector = connector_counter_delta(before_connector, after_connector)
    except (TypeError, ValueError):
        _fail(BrowseCompEvidenceErrorCode.INVALID_CONNECTOR_METRICS)
    try:
        store = connector_store_counter_delta(before_store, after_store)
    except (TypeError, ValueError):
        _fail(BrowseCompEvidenceErrorCode.INVALID_STORE_METRICS)

    try:
        before_prompt = parse_vllm_prompt_counter_snapshot(before_text)
        after_prompt = parse_vllm_prompt_counter_snapshot(after_text)
        prompt_tokens = vllm_prompt_counter_delta(before_prompt, after_prompt)
        before_source = parse_vllm_prompt_source_snapshot(
            before_text, allow_missing=True
        )
        after_source = parse_vllm_prompt_source_snapshot(after_text)
        prompt_source = vllm_prompt_source_delta(before_source, after_source)
        before_prefill = parse_vllm_prefill_work_snapshot(before_text)
        after_prefill = parse_vllm_prefill_work_snapshot(after_text)
        prefill_work = vllm_prefill_work_snapshot_delta(before_prefill, after_prefill)
    except (TypeError, ValueError):
        _fail(BrowseCompEvidenceErrorCode.INVALID_NATIVE_METRICS)
    try:
        before_timing = parse_vllm_timing_snapshot(before_text)
        after_timing = parse_vllm_timing_snapshot(after_text)
        timing = vllm_timing_snapshot_delta(before_timing, after_timing)
    except (TypeError, ValueError):
        _fail(BrowseCompEvidenceErrorCode.INVALID_TIMING_METRICS)

    if (
        connector["requests"] != facts.generation_requests
        or connector["kv_tokens_found"]
        != connector["kv_tokens_loaded"] + connector["kv_tokens_rejected"]
        or connector["kv_tokens_loaded"] <= 0
        or connector["kv_tokens_found"]
        > connector["reusable_document_tokens_requested"]
        or connector["kv_tokens_loaded"] > connector["kv_tokens_found"]
        or connector["tokens_recomputed"] != facts.input_tokens
        or connector["prefill_tokens_avoided"] != 0
    ):
        _fail(BrowseCompEvidenceErrorCode.INVALID_CONNECTOR_METRICS)
    if (
        store["store_fallbacks"] != 0
        or store["store_tokens_eligible"] <= 0
        or store["store_tokens_completed"] <= 0
        or store["store_tokens_completed"] != store["store_tokens_eligible"]
    ):
        _fail(BrowseCompEvidenceErrorCode.INVALID_STORE_METRICS)
    if prompt_tokens != facts.input_tokens:
        _fail(BrowseCompEvidenceErrorCode.INVALID_NATIVE_METRICS)
    try:
        require_full_prefill_prompt_source_delta(
            prompt_source,
            expected_prompt_tokens=facts.input_tokens,
        )
        require_vllm_prefill_work_total(
            prefill_work,
            expected_prompt_tokens=facts.input_tokens,
            expected_requests=facts.generation_requests,
        )
    except (TypeError, ValueError):
        _fail(BrowseCompEvidenceErrorCode.INVALID_NATIVE_METRICS)
    try:
        require_vllm_timing_delta(timing, expected_requests=facts.generation_requests)
    except (TypeError, ValueError):
        _fail(BrowseCompEvidenceErrorCode.INVALID_TIMING_METRICS)
    return connector, store, prompt_tokens, prompt_source, prefill_work, timing


def _empty_workload() -> dict[str, int]:
    return {key: 0 for key in _WORKLOAD_KEYS}


def _empty_tokens() -> dict[str, int]:
    return {key: 0 for key in _TOKEN_TOTAL_KEYS}


def _report_without_digest(
    runtime: CorrectnessRuntimeIdentity | None,
    *,
    passed: bool,
    workload: Mapping[str, int],
    token_totals: Mapping[str, int],
    connector: Mapping[str, int] | None,
    store: Mapping[str, int] | None,
    native: Mapping[str, object] | None,
    timing: Mapping[str, object] | None,
    failure: BrowseCompEvidenceError | None = None,
) -> dict[str, object]:
    report: dict[str, object] = {
        "schema_version": BROWSECOMP_EVIDENCE_SCHEMA_VERSION,
        "contract": BROWSECOMP_EVIDENCE_CONTRACT,
        "runtime": None if runtime is None else _runtime_to_dict(runtime),
        "passed": passed,
        "workload": dict(workload),
        "token_totals": dict(token_totals),
        "connector": None if connector is None else dict(connector),
        "store": None if store is None else dict(store),
        "native": None if native is None else dict(native),
        "timing": None if timing is None else dict(timing),
    }
    if failure is not None:
        report["failure"] = {
            "code": failure.code.value,
            "message": failure.message,
        }
    return report


def canonical_browsecomp_evidence_bytes(report: Mapping[str, object]) -> bytes:
    """Return stable bytes for a sanitized report, excluding its digest."""

    if not isinstance(report, Mapping):
        raise TypeError("BrowseComp evidence report must be a mapping")
    payload = dict(report)
    payload.pop("evidence_digest", None)
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def browsecomp_evidence_digest(report: Mapping[str, object]) -> str:
    """Return the stable SHA-256 digest of a report without its digest field."""

    return sha256(canonical_browsecomp_evidence_bytes(report)).hexdigest()


def failed_browsecomp_report(
    runtime: CorrectnessRuntimeIdentity | None,
    error: BrowseCompEvidenceError,
) -> dict[str, object]:
    """Build a sanitized failed report for CLI inspection."""

    report = _report_without_digest(
        runtime,
        passed=False,
        workload=_empty_workload(),
        token_totals=_empty_tokens(),
        connector=None,
        store=None,
        native=None,
        timing=None,
        failure=error,
    )
    report["evidence_digest"] = browsecomp_evidence_digest(report)
    return report


def validate_browsecomp_append_only(
    run_record: object,
    metrics_before: str,
    metrics_after: str,
    runtime: CorrectnessRuntimeIdentity,
) -> dict[str, object]:
    """Validate one completed append-only CacheBlend transfer smoke run.

    The function raises :class:`BrowseCompEvidenceError` on any failed gate.
    Its successful return value is a sanitized schema-v1 report containing no
    request-, document-, prompt-, or response-level identifiers.
    """

    facts = _validate_run_record(run_record, runtime)
    (
        connector,
        store,
        prompt_tokens,
        prompt_source,
        prefill_work,
        timing,
    ) = _parse_metric_deltas(metrics_before, metrics_after, facts)
    workload = {
        "searches": facts.searches,
        "retrieval_requests": facts.retrieval_requests,
        "generation_requests": facts.generation_requests,
        "completed_generation_steps": facts.generation_requests,
        "context_rebuilds": facts.context_rebuilds,
        "context_edits": 0,
        "context_truncations": 0,
        "invalid_function_calls": 0,
        "generation_retries": 0,
    }
    token_totals = {
        "input_tokens": facts.input_tokens,
        "output_tokens": facts.output_tokens,
        "total_tokens": facts.total_tokens,
    }
    native = {
        "prompt_tokens_processed": prompt_tokens,
        "prompt_source_delta": prompt_source,
        "prefill_work": {
            "observations": prefill_work.observations,
            "kv_computed_tokens": prefill_work.kv_computed_tokens,
        },
    }
    report = _report_without_digest(
        runtime,
        passed=True,
        workload=workload,
        token_totals=token_totals,
        connector=connector,
        store=store,
        native=native,
        timing=timing.as_dict(),
    )
    report["evidence_digest"] = browsecomp_evidence_digest(report)
    return report


__all__ = [
    "BROWSECOMP_EVIDENCE_CONTRACT",
    "BROWSECOMP_EVIDENCE_SCHEMA_VERSION",
    "BrowseCompEvidenceError",
    "BrowseCompEvidenceErrorCode",
    "browsecomp_evidence_digest",
    "canonical_browsecomp_evidence_bytes",
    "failed_browsecomp_report",
    "runtime_identity_from_dict",
    "validate_browsecomp_append_only",
]
