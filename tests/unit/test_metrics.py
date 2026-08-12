import pytest

from cacheblend_gpt_oss.metrics import (
    InvalidRequestMetricsError,
    MetricField,
    MetricInvariantCode,
    RequestCorrectnessMetrics,
    RequestMetricCounters,
    RequestMetrics,
    RequestMetricTimers,
    require_valid_request_metrics,
    validate_request_metrics,
)


def _timers() -> RequestMetricTimers:
    return RequestMetricTimers(
        lookup_latency_seconds=0.001,
        transfer_latency_seconds=0.002,
        position_correction_latency_seconds=0.0,
        selective_recomputation_latency_seconds=0.0,
        ttft_seconds=None,
        prefill_latency_seconds=0.01,
    )


def _full_recompute_counters() -> RequestMetricCounters:
    return RequestMetricCounters(
        prompt_tokens=100,
        reusable_documents_requested=1,
        reusable_documents_hit=1,
        reusable_document_tokens_requested=60,
        kv_tokens_found=60,
        kv_tokens_loaded=55,
        kv_tokens_rejected=5,
        tokens_recomputed=100,
        prefill_tokens_avoided=0,
    )


def test_full_recompute_snapshot_is_consistent_and_reports_no_savings() -> None:
    counters = _full_recompute_counters()
    metrics = RequestMetrics(counters=counters, timers=_timers())

    assert validate_request_metrics(metrics) == ()
    require_valid_request_metrics(metrics)
    assert counters.is_full_recomputation
    assert counters.document_hit_fraction == 1.0
    assert counters.candidate_token_hit_fraction == 1.0
    assert counters.loaded_token_hit_fraction == pytest.approx(55 / 60)
    assert counters.effective_saved_prefill_fraction == 0.0


def test_selective_recompute_snapshot_derives_effective_saved_fraction() -> None:
    counters = RequestMetricCounters(
        prompt_tokens=100,
        reusable_documents_requested=2,
        reusable_documents_hit=1,
        reusable_document_tokens_requested=80,
        kv_tokens_found=70,
        kv_tokens_loaded=65,
        kv_tokens_rejected=5,
        tokens_recomputed=25,
        prefill_tokens_avoided=75,
    )
    metrics = RequestMetrics(
        counters=counters,
        timers=_timers(),
        correctness=RequestCorrectnessMetrics(
            max_abs_logit_error=0.001,
            mean_abs_logit_error=0.0001,
        ),
    )

    assert validate_request_metrics(metrics) == ()
    assert counters.document_hit_fraction == 0.5
    assert counters.candidate_token_hit_fraction == pytest.approx(70 / 80)
    assert counters.loaded_token_hit_fraction == pytest.approx(65 / 80)
    assert counters.effective_saved_prefill_fraction == 0.75


def test_loaded_tokens_may_not_exceed_found_tokens() -> None:
    counters = RequestMetricCounters(
        prompt_tokens=100,
        reusable_documents_requested=1,
        reusable_documents_hit=1,
        reusable_document_tokens_requested=60,
        kv_tokens_found=50,
        kv_tokens_loaded=51,
        kv_tokens_rejected=0,
        tokens_recomputed=100,
        prefill_tokens_avoided=0,
    )

    issues = validate_request_metrics(RequestMetrics(counters, _timers()))

    assert MetricInvariantCode.LOADED_TOKENS_EXCEED_FOUND in {
        issue.code for issue in issues
    }


def test_found_tokens_reconcile_to_loaded_plus_rejected() -> None:
    counters = RequestMetricCounters(
        prompt_tokens=100,
        reusable_documents_requested=1,
        reusable_documents_hit=1,
        reusable_document_tokens_requested=60,
        kv_tokens_found=60,
        kv_tokens_loaded=55,
        kv_tokens_rejected=4,
        tokens_recomputed=100,
        prefill_tokens_avoided=0,
    )

    issues = validate_request_metrics(RequestMetrics(counters, _timers()))

    assert MetricInvariantCode.REJECTED_TOKEN_RECONCILIATION_FAILED in {
        issue.code for issue in issues
    }


def test_full_recomputation_cannot_report_saved_prefill() -> None:
    counters = RequestMetricCounters(
        prompt_tokens=100,
        reusable_documents_requested=0,
        reusable_documents_hit=0,
        reusable_document_tokens_requested=0,
        kv_tokens_found=0,
        kv_tokens_loaded=0,
        kv_tokens_rejected=0,
        tokens_recomputed=100,
        prefill_tokens_avoided=1,
    )

    issues = validate_request_metrics(RequestMetrics(counters, _timers()))

    assert MetricInvariantCode.SAVED_TOKEN_RECONCILIATION_FAILED in {
        issue.code for issue in issues
    }
    assert MetricInvariantCode.FULL_RECOMPUTE_REPORTED_SAVINGS in {
        issue.code for issue in issues
    }


def test_invalid_values_have_bounded_fields_and_raise_structured_error() -> None:
    timers = RequestMetricTimers(
        lookup_latency_seconds=-0.1,
        transfer_latency_seconds=0.0,
        position_correction_latency_seconds=0.0,
        selective_recomputation_latency_seconds=0.0,
        ttft_seconds=float("nan"),
        prefill_latency_seconds=0.1,
    )
    metrics = RequestMetrics(
        counters=_full_recompute_counters(),
        timers=timers,
        correctness=RequestCorrectnessMetrics(max_abs_logit_error=float("inf")),
    )

    with pytest.raises(InvalidRequestMetricsError) as caught:
        require_valid_request_metrics(metrics)

    assert {issue.field for issue in caught.value.issues} == {
        MetricField.LOOKUP_LATENCY_SECONDS,
        MetricField.TTFT_SECONDS,
        MetricField.MAX_ABS_LOGIT_ERROR,
    }


def test_zero_denominators_produce_bounded_zero_fractions() -> None:
    counters = RequestMetricCounters(
        prompt_tokens=0,
        reusable_documents_requested=0,
        reusable_documents_hit=0,
        reusable_document_tokens_requested=0,
        kv_tokens_found=0,
        kv_tokens_loaded=0,
        kv_tokens_rejected=0,
        tokens_recomputed=0,
        prefill_tokens_avoided=0,
    )

    assert counters.document_hit_fraction == 0.0
    assert counters.candidate_token_hit_fraction == 0.0
    assert counters.loaded_token_hit_fraction == 0.0
    assert counters.effective_saved_prefill_fraction == 0.0
    assert validate_request_metrics(RequestMetrics(counters, _timers())) == ()
