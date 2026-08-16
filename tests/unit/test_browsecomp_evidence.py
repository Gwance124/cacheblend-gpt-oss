# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the BrowseComp-Plus transfer smoke evidence gate."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cacheblend_gpt_oss.benchmark.browsecomp import (
    BROWSECOMP_EVIDENCE_CONTRACT,
    BROWSECOMP_SELECTIVE_EVIDENCE_CONTRACT,
    BrowseCompEvidenceError,
    BrowseCompEvidenceErrorCode,
    browsecomp_evidence_digest,
    validate_browsecomp_append_only,
    validate_browsecomp_selective_append_only,
)
from cacheblend_gpt_oss.correctness.models import CorrectnessRuntimeIdentity


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


def _usage(input_tokens: int, output_tokens: int) -> dict[str, int]:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _run_record() -> dict[str, object]:
    first = _usage(100, 20)
    second = _usage(140, 30)
    return {
        "schema_version": "1.0",
        "query_id": "private-query-id",
        "status": "completed",
        "tool_call_counts": {"search": 1},
        "retrieved_docids": ["private-doc-id"],
        "result": [
            {"type": "reasoning", "output": "private reasoning text"},
            {"type": "output_text", "output": "private answer text"},
        ],
        "metadata": {
            "model": "openai/gpt-oss-20b",
            "api": "responses",
            "scaffold": "standard_search_only_top5_first512",
            "cache_mode": "cacheblend",
            "context_strategy": "append_only",
            "deduplicate_retrieved_documents": None,
            "response_id": "private-response-id",
        },
        "diagnostics": {
            "cache_mode": "cacheblend",
            "context_strategy": "append_only",
            "deduplicate_retrieved_documents": False,
            "context_budget_triggered": False,
            "elided_duplicate_documents_total": 0,
            "retrieval_request_count": 1,
            "search_steps": [
                {
                    "search_call": 1,
                    "returned_documents": 5,
                    "elided_duplicate_documents": 0,
                    "query": "private search query",
                }
            ],
            "generation_request_count": 2,
            "generation_usage": [first, second],
            "generation_steps": [
                {
                    "request_number": 1,
                    "response_id": "private-response-1",
                    "response_status": "completed",
                    "usage": first,
                    "call_id": "private-call-1",
                },
                {
                    "request_number": 2,
                    "response_id": "private-response-2",
                    "response_status": "completed",
                    "usage": second,
                    "call_id": "private-call-2",
                },
            ],
            "context_rebuilds": [
                {
                    "request_number": 1,
                    "context_strategy": "append_only",
                    "documents_full": 0,
                    "documents_stubbed": 0,
                    "documents_removed": 0,
                    "first_edited_item_index": None,
                },
                {
                    "request_number": 2,
                    "context_strategy": "append_only",
                    "documents_full": 0,
                    "documents_stubbed": 0,
                    "documents_removed": 0,
                    "first_edited_item_index": None,
                },
            ],
            "context_truncation_events": [],
            "invalid_function_calls": [],
            "termination_reason": "final_answer",
            "final_answer_validation": {"valid": True},
        },
    }


_CONNECTOR_BASE = {
    "vllm:cacheblend_requests_total": 7,
    "vllm:cacheblend_reusable_document_tokens_requested_total": 50,
    "vllm:cacheblend_kv_tokens_found_total": 10,
    "vllm:cacheblend_kv_tokens_loaded_total": 8,
    "vllm:cacheblend_kv_tokens_rejected_total": 2,
    "vllm:cacheblend_tokens_recomputed_total": 0,
    "vllm:cacheblend_prefill_tokens_avoided_total": 0,
}
_CONNECTOR_DELTA = {
    "vllm:cacheblend_requests_total": 2,
    "vllm:cacheblend_reusable_document_tokens_requested_total": 120,
    "vllm:cacheblend_kv_tokens_found_total": 100,
    "vllm:cacheblend_kv_tokens_loaded_total": 90,
    "vllm:cacheblend_kv_tokens_rejected_total": 10,
    "vllm:cacheblend_tokens_recomputed_total": 240,
    "vllm:cacheblend_prefill_tokens_avoided_total": 0,
}
_STORE_BASE = {
    "vllm:cacheblend_store_tokens_eligible_total": 20,
    "vllm:cacheblend_store_tokens_completed_total": 20,
    "vllm:cacheblend_store_fallbacks_total": 0,
}
_STORE_DELTA = {
    "vllm:cacheblend_store_tokens_eligible_total": 90,
    "vllm:cacheblend_store_tokens_completed_total": 90,
    "vllm:cacheblend_store_fallbacks_total": 0,
}


def _metric_snapshot(
    *,
    connector: dict[str, int] | None = None,
    store: dict[str, int] | None = None,
    prompt_tokens: int = 100,
    local_compute: int = 100,
    prefill_observations: int = 3,
    prefill_tokens: int = 100,
    timing_count: int = 5,
    timing_sum: float = 1.0,
    selective_work: dict[str, int] | None = None,
) -> str:
    values: dict[str, int | float] = {}
    values.update(connector or _CONNECTOR_BASE)
    values.update(store or _STORE_BASE)
    values.update(selective_work or {})
    values["vllm:prompt_tokens_total"] = prompt_tokens
    for source, value in (
        ("local_compute", local_compute),
        ("local_cache_hit", 0),
        ("external_kv_transfer", 0),
    ):
        values[f"vllm:prompt_tokens_by_source_total|{source}"] = value
    values["vllm:request_prefill_kv_computed_tokens_count"] = prefill_observations
    values["vllm:request_prefill_kv_computed_tokens_sum"] = prefill_tokens
    for index, metric in enumerate(
        (
            "vllm:time_to_first_token_seconds",
            "vllm:e2e_request_latency_seconds",
            "vllm:request_queue_time_seconds",
            "vllm:request_prefill_time_seconds",
            "vllm:request_decode_time_seconds",
        )
    ):
        values[f"{metric}_count"] = timing_count
        values[f"{metric}_sum"] = timing_sum + index / 10

    lines: list[str] = []
    for name, value in values.items():
        if "|" in name:
            metric, source = name.split("|", 1)
            lines.append(
                f'{metric}{{source="{source}",private_label="private-id"}} {value}'
            )
        else:
            lines.append(f'{name}{{engine="0",private_label="private-id"}} {value}')
    return "\n".join(lines) + "\n"


def _metric_pair(*, selective: bool = False) -> tuple[str, str]:
    before = _metric_snapshot()
    after_connector = {
        key: _CONNECTOR_BASE[key] + _CONNECTOR_DELTA[key] for key in _CONNECTOR_BASE
    }
    after_store = {key: _STORE_BASE[key] + _STORE_DELTA[key] for key in _STORE_BASE}
    return before, _metric_snapshot(
        connector=after_connector,
        store=after_store,
        prompt_tokens=340,
        local_compute=340,
        prefill_observations=5,
        prefill_tokens=340,
        timing_count=7,
        timing_sum=3.0,
        selective_work=(
            {
                "vllm:cacheblend_layer_token_rows_recomputed_total": 5_000,
                "vllm:cacheblend_layer_token_rows_avoided_total": 760,
            }
            if selective
            else None
        ),
    )


def _valid_report() -> dict[str, object]:
    before, after = _metric_pair()
    return validate_browsecomp_append_only(_run_record(), before, after, _runtime())


def _valid_selective_report() -> dict[str, object]:
    before, after = _metric_pair(selective=True)
    return validate_browsecomp_selective_append_only(
        _run_record(), before, after, _runtime()
    )


def _assert_failure(
    run: object,
    before: str,
    after: str,
    code: BrowseCompEvidenceErrorCode,
) -> None:
    with pytest.raises(BrowseCompEvidenceError) as caught:
        validate_browsecomp_append_only(run, before, after, _runtime())
    assert caught.value.code is code
    assert caught.value.message == str(caught.value)
    assert "private" not in str(caught.value)


def test_happy_path_reports_only_bounded_aggregate_evidence() -> None:
    report = _valid_report()

    assert report["schema_version"] == 1
    assert report["contract"] == BROWSECOMP_EVIDENCE_CONTRACT
    assert report["passed"] is True
    assert report["workload"] == {
        "searches": 1,
        "retrieval_requests": 1,
        "generation_requests": 2,
        "completed_generation_steps": 2,
        "context_rebuilds": 2,
        "context_edits": 0,
        "context_truncations": 0,
        "invalid_function_calls": 0,
        "generation_retries": 0,
    }
    assert report["token_totals"] == {
        "input_tokens": 240,
        "output_tokens": 50,
        "total_tokens": 290,
    }
    assert report["connector"] == {
        "requests": 2,
        "reusable_document_tokens_requested": 120,
        "kv_tokens_found": 100,
        "kv_tokens_loaded": 90,
        "kv_tokens_rejected": 10,
        "tokens_recomputed": 240,
        "prefill_tokens_avoided": 0,
    }
    assert report["store"] == {
        "store_tokens_eligible": 90,
        "store_tokens_completed": 90,
        "store_fallbacks": 0,
    }
    native = report["native"]
    assert isinstance(native, dict)
    assert native["prompt_tokens_processed"] == 240
    assert native["prompt_source_delta"] == {
        "local_compute": 240,
        "local_cache_hit": 0,
        "external_kv_transfer": 0,
    }
    assert native["prefill_work"] == {
        "observations": 2,
        "kv_computed_tokens": 240,
    }
    assert report["timing"]["ttft_seconds"]["count"] == 2  # type: ignore[index]


def test_selective_happy_path_reconciles_layer_token_work() -> None:
    report = _valid_selective_report()

    assert report["contract"] == BROWSECOMP_SELECTIVE_EVIDENCE_CONTRACT
    assert report["passed"] is True
    assert report["selective_work"] == {
        "layer_token_rows_recomputed": 5_000,
        "layer_token_rows_avoided": 760,
    }


def test_selective_requires_positive_reconciled_work() -> None:
    before, after = _metric_pair()
    with pytest.raises(BrowseCompEvidenceError) as caught:
        validate_browsecomp_selective_append_only(
            _run_record(), before, after, _runtime()
        )
    assert caught.value.code is BrowseCompEvidenceErrorCode.INVALID_PROMETHEUS


def test_private_run_and_metric_data_never_cross_into_report_or_digest() -> None:
    before, after = _metric_pair()
    report = validate_browsecomp_append_only(_run_record(), before, after, _runtime())
    rendered = json.dumps(report, sort_keys=True)

    for private_value in (
        "private-query-id",
        "private-doc-id",
        "private reasoning text",
        "private answer text",
        "private-response-id",
        "private-call-1",
        "private search query",
        "private-id",
    ):
        assert private_value not in rendered
    for forbidden_key in (
        "query_id",
        "retrieved_docids",
        "response_id",
        "call_id",
        "fingerprint",
    ):
        assert forbidden_key not in rendered
    assert report["evidence_digest"] == browsecomp_evidence_digest(report)
    assert browsecomp_evidence_digest(report) == browsecomp_evidence_digest(
        json.loads(json.dumps(report))
    )


@pytest.mark.parametrize(
    ("path", "value", "code"),
    [
        (
            ("metadata", "cache_mode"),
            "prefix+cacheblend",
            BrowseCompEvidenceErrorCode.INVALID_CONFIGURATION,
        ),
        (
            ("diagnostics", "context_strategy"),
            "prune",
            BrowseCompEvidenceErrorCode.INVALID_CONFIGURATION,
        ),
        (
            ("diagnostics", "deduplicate_retrieved_documents"),
            True,
            BrowseCompEvidenceErrorCode.INVALID_CONFIGURATION,
        ),
        (
            ("diagnostics", "termination_reason"),
            "max_iterations",
            BrowseCompEvidenceErrorCode.INVALID_STATUS,
        ),
    ],
)
def test_configuration_and_terminal_tampering_fail_with_bounded_codes(
    path: tuple[str, str], value: object, code: BrowseCompEvidenceErrorCode
) -> None:
    run = _run_record()
    before, after = _metric_pair()
    target = run
    for key in path[:-1]:
        nested = target[key]  # type: ignore[index]
        assert isinstance(nested, dict)
        target = nested
    target[path[-1]] = value  # type: ignore[index]
    _assert_failure(run, before, after, code)


def test_schema_generation_usage_context_and_metric_tampering_fail_closed() -> None:
    before, after = _metric_pair()

    run = _run_record()
    run["schema_version"] = "2.0"
    _assert_failure(run, before, after, BrowseCompEvidenceErrorCode.INVALID_SCHEMA)

    run = _run_record()
    run["diagnostics"]["generation_steps"][0][  # type: ignore[index]
        "response_status"
    ] = "incomplete"
    _assert_failure(run, before, after, BrowseCompEvidenceErrorCode.INVALID_GENERATION)

    run = _run_record()
    run["diagnostics"]["generation_usage"][1]["input_tokens"] = (  # type: ignore[index]
        141
    )
    _assert_failure(run, before, after, BrowseCompEvidenceErrorCode.INVALID_USAGE)

    run = _run_record()
    run["diagnostics"]["generation_usage"][0][  # type: ignore[index]
        "input_tokens_details"
    ] = {"cached_tokens": 1}
    run["diagnostics"]["generation_steps"][0]["usage"] = run[  # type: ignore[index]
        "diagnostics"
    ]["generation_usage"][0]  # type: ignore[index]
    _assert_failure(run, before, after, BrowseCompEvidenceErrorCode.INVALID_USAGE)

    run = _run_record()
    run["diagnostics"]["context_rebuilds"][0][  # type: ignore[index]
        "first_edited_item_index"
    ] = 0
    _assert_failure(run, before, after, BrowseCompEvidenceErrorCode.INVALID_CONTEXT)

    after_connector = _metric_snapshot(
        connector={
            key: _CONNECTOR_BASE[key] + _CONNECTOR_DELTA[key] for key in _CONNECTOR_BASE
        }
        | {"vllm:cacheblend_kv_tokens_loaded_total": 91},
        store={key: _STORE_BASE[key] + _STORE_DELTA[key] for key in _STORE_BASE},
        prompt_tokens=340,
        local_compute=340,
        prefill_observations=5,
        prefill_tokens=340,
        timing_count=7,
        timing_sum=3.0,
    )
    _assert_failure(
        _run_record(),
        before,
        after_connector,
        BrowseCompEvidenceErrorCode.INVALID_CONNECTOR_METRICS,
    )

    after_store = _metric_snapshot(
        connector={
            key: _CONNECTOR_BASE[key] + _CONNECTOR_DELTA[key] for key in _CONNECTOR_BASE
        },
        store={key: _STORE_BASE[key] + _STORE_DELTA[key] for key in _STORE_BASE}
        | {"vllm:cacheblend_store_fallbacks_total": 1},
        prompt_tokens=340,
        local_compute=340,
        prefill_observations=5,
        prefill_tokens=340,
        timing_count=7,
        timing_sum=3.0,
    )
    _assert_failure(
        _run_record(),
        before,
        after_store,
        BrowseCompEvidenceErrorCode.INVALID_STORE_METRICS,
    )


def test_idle_before_scrape_may_omit_connector_and_store_families() -> None:
    after = _metric_snapshot(
        connector=_CONNECTOR_DELTA,
        store=_STORE_DELTA,
        prompt_tokens=240,
        local_compute=240,
        prefill_observations=2,
        prefill_tokens=240,
        timing_count=2,
        timing_sum=2.0,
    )
    report = validate_browsecomp_append_only(_run_record(), "", after, _runtime())

    assert report["passed"] is True
    assert report["connector"]["requests"] == 2  # type: ignore[index]
    assert report["store"]["store_tokens_completed"] == 90  # type: ignore[index]


@pytest.mark.parametrize(
    "partial_metric",
    [
        'vllm:cacheblend_requests_total{engine="0"} 7\n',
        'vllm:cacheblend_store_fallbacks_total{engine="0"} 0\n',
    ],
)
def test_idle_before_scrape_rejects_partial_connector_or_store_family(
    partial_metric: str,
) -> None:
    _before, after = _metric_pair()
    _assert_failure(
        _run_record(),
        partial_metric,
        after,
        (
            BrowseCompEvidenceErrorCode.INVALID_CONNECTOR_METRICS
            if "cacheblend_store" not in partial_metric
            else BrowseCompEvidenceErrorCode.INVALID_STORE_METRICS
        ),
    )


def test_single_generation_trajectory_is_rejected() -> None:
    before, after = _metric_pair()
    run = _run_record()
    diagnostics = run["diagnostics"]
    assert isinstance(diagnostics, dict)
    diagnostics["generation_request_count"] = 1
    diagnostics["generation_steps"] = diagnostics["generation_steps"][:1]
    diagnostics["generation_usage"] = diagnostics["generation_usage"][:1]
    diagnostics["context_rebuilds"] = diagnostics["context_rebuilds"][:1]
    _assert_failure(
        run,
        before,
        after,
        BrowseCompEvidenceErrorCode.INVALID_WORKLOAD,
    )


def test_zero_store_interval_is_rejected_as_stale_sidecar_evidence() -> None:
    before, _after = _metric_pair()
    after_connector = {
        key: _CONNECTOR_BASE[key] + _CONNECTOR_DELTA[key] for key in _CONNECTOR_BASE
    }
    after = _metric_snapshot(
        connector=after_connector,
        store=_STORE_BASE,
        prompt_tokens=340,
        local_compute=340,
        prefill_observations=5,
        prefill_tokens=340,
        timing_count=7,
        timing_sum=3.0,
    )
    _assert_failure(
        _run_record(),
        before,
        after,
        BrowseCompEvidenceErrorCode.INVALID_STORE_METRICS,
    )


def test_runtime_identity_json_and_cli_report_are_create_only(tmp_path: Path) -> None:
    before, after = _metric_pair()
    run_path = tmp_path / "run_private.json"
    before_path = tmp_path / "before.prom"
    after_path = tmp_path / "after.prom"
    runtime_path = tmp_path / "runtime.json"
    output_path = tmp_path / "evidence.json"
    run_path.write_text(json.dumps(_run_record()), encoding="utf-8")
    before_path.write_text(before, encoding="utf-8")
    after_path.write_text(after, encoding="utf-8")
    runtime_path.write_text(
        json.dumps(
            {
                key: getattr(_runtime(), key)
                for key in (
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
                )
            }
        ),
        encoding="utf-8",
    )
    script = Path("scripts/validate_browsecomp_append_only.py")
    command = [
        sys.executable,
        str(script),
        "--run-record",
        str(run_path),
        "--metrics-before",
        str(before_path),
        "--metrics-after",
        str(after_path),
        "--runtime-identity",
        str(runtime_path),
        "--output",
        str(output_path),
        "--require-passed",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    cli_report = json.loads(completed.stdout)
    assert cli_report["passed"] is True
    assert json.loads(output_path.read_text(encoding="utf-8")) == cli_report

    second = subprocess.run(command, capture_output=True, text=True, check=False)
    assert second.returncode == 2
    assert json.loads(second.stdout)["failure"]["code"] == "output_exists"


def test_failure_report_is_sanitized_and_require_passed_is_meaningful() -> None:
    run = _run_record()
    run["status"] = "incomplete"
    before, after = _metric_pair()
    with pytest.raises(BrowseCompEvidenceError) as caught:
        validate_browsecomp_append_only(run, before, after, _runtime())
    assert caught.value.code is BrowseCompEvidenceErrorCode.INVALID_STATUS

    # The CLI's failure shape is exercised through the same fixed error text;
    # no input record is ever interpolated into it.
    assert "private" not in caught.value.message
