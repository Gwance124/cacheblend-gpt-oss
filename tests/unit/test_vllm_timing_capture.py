from __future__ import annotations

import pytest

from cacheblend_gpt_oss.correctness import (
    VllmNativeRequestEvidence,
    VllmPrefillWorkSnapshot,
    VllmPromptSourceDelta,
    VllmTimingSnapshot,
    VllmTimingSummary,
    has_vllm_prefill_work_metric_surface,
    has_vllm_prompt_metric_surface,
    has_vllm_prompt_source_metric_surface,
    has_vllm_timing_metric_surface,
    parse_vllm_prefill_work_snapshot,
    parse_vllm_prompt_counter_snapshot,
    parse_vllm_prompt_source_snapshot,
    parse_vllm_timing_snapshot,
    require_full_prefill_prompt_source_delta,
    require_vllm_prefill_work_delta,
    require_vllm_prefill_work_total,
    require_vllm_timing_delta,
    vllm_prefill_work_snapshot_delta,
    vllm_prompt_counter_delta,
    vllm_prompt_source_delta,
    vllm_timing_snapshot_delta,
)

_METRICS = (
    "vllm:time_to_first_token_seconds",
    "vllm:e2e_request_latency_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
)


def _snapshot_text(
    *,
    count: int = 2,
    sum_seconds: float = 3.0,
    include_buckets: bool = True,
) -> str:
    lines: list[str] = []
    for index, metric in enumerate(_METRICS):
        lines.append(f'{metric}_count{{engine="0"}} {count}')
        lines.append(
            f'{metric}_sum{{engine="0"}} {sum_seconds + index / 10:.1f}'
        )
        if include_buckets:
            lines.append(f'{metric}_bucket{{engine="0",le="+Inf"}} {count}')
    return "\n".join(lines) + "\n"


def _prompt_source_text(
    local_compute: int,
    local_cache_hit: int,
    external_kv_transfer: int,
) -> str:
    return "\n".join(
        (
            "vllm:prompt_tokens_by_source{model_name=\"m\",engine=\"0\","
            f"source=\"local_compute\"}} {local_compute}",
            "vllm:prompt_tokens_by_source{model_name=\"m\",engine=\"0\","
            f"source=\"local_cache_hit\"}} {local_cache_hit}",
            "vllm:prompt_tokens_by_source{model_name=\"m\",engine=\"0\","
            f"source=\"external_kv_transfer\"}} {external_kv_transfer}",
        )
    ) + "\n"


def test_pinned_timing_histograms_are_aggregated_without_labels() -> None:
    text = _snapshot_text()
    text += (
        'vllm:time_to_first_token_seconds_count{engine="1",model_name="other"} 1\n'
        'vllm:time_to_first_token_seconds_sum{engine="1",model_name="other"} 0.5\n'
    )

    assert has_vllm_timing_metric_surface(text)
    snapshot = parse_vllm_timing_snapshot(text)

    assert snapshot.ttft_seconds == VllmTimingSummary(count=3, sum_seconds=3.5)
    assert snapshot.prefill_latency_seconds.count == 2
    assert snapshot.prefill_latency_seconds.mean_seconds == pytest.approx(1.65)
    assert snapshot.as_dict()["decode_latency_seconds"]["sum_seconds"] == pytest.approx(
        3.4
    )


def test_native_prompt_counter_is_aggregated_and_delta_checked() -> None:
    before = parse_vllm_prompt_counter_snapshot(
        'vllm:prompt_tokens{engine="0"} 11\n'
        'vllm:prompt_tokens{engine="1"} 4\n'
    )
    after = parse_vllm_prompt_counter_snapshot(
        'vllm:prompt_tokens{engine="0"} 291\n'
        'vllm:prompt_tokens{engine="1"} 4\n'
    )

    assert has_vllm_prompt_metric_surface(
        'vllm:prompt_tokens{engine="0"} 11\n'
    )
    assert vllm_prompt_counter_delta(before, after) == 280

    emitted = parse_vllm_prompt_counter_snapshot(
        'vllm:prompt_tokens_total{engine="0"} 291\n'
        'vllm:prompt_tokens_total{engine="1"} 4\n'
    )
    assert emitted == {"prompt_tokens": 295}


def test_native_prompt_counter_rejects_missing_or_backwards_intervals() -> None:
    assert not has_vllm_prompt_metric_surface("vllm:num_requests_running 0\n")
    empty = parse_vllm_prompt_counter_snapshot("")
    assert empty == {"prompt_tokens": 0}
    with pytest.raises(ValueError, match="moved backwards"):
        vllm_prompt_counter_delta(
            {"prompt_tokens": 10},
            {"prompt_tokens": 9},
        )


def test_native_prefill_work_histogram_reconciles_exact_prompt_rows() -> None:
    before = parse_vllm_prefill_work_snapshot(
        'vllm:request_prefill_kv_computed_tokens_count{engine="0"} 2\n'
        'vllm:request_prefill_kv_computed_tokens_sum{engine="0"} 20\n'
    )
    after = parse_vllm_prefill_work_snapshot(
        'vllm:request_prefill_kv_computed_tokens_count{engine="0"} 3\n'
        'vllm:request_prefill_kv_computed_tokens_sum{engine="0"} 300\n'
    )

    assert has_vllm_prefill_work_metric_surface(
        'vllm:request_prefill_kv_computed_tokens_count{engine="0"} 0\n'
        'vllm:request_prefill_kv_computed_tokens_sum{engine="0"} 0\n'
    )
    delta = vllm_prefill_work_snapshot_delta(before, after)
    require_vllm_prefill_work_delta(delta, expected_prompt_tokens=280)
    assert delta == VllmPrefillWorkSnapshot(1, 280)


def test_native_prefill_work_rejects_partial_or_mismatched_histograms() -> None:
    partial = (
        'vllm:request_prefill_kv_computed_tokens_count{engine="0"} 1\n'
    )
    assert not has_vllm_prefill_work_metric_surface(partial)
    with pytest.raises(ValueError, match="family is incomplete"):
        parse_vllm_prefill_work_snapshot(partial)
    with pytest.raises(ValueError, match="does not match"):
        require_vllm_prefill_work_delta(
            VllmPrefillWorkSnapshot(1, 279),
            expected_prompt_tokens=280,
        )
    require_vllm_prefill_work_total(
        VllmPrefillWorkSnapshot(3, 811),
        expected_prompt_tokens=811,
        expected_requests=3,
    )
    with pytest.raises(ValueError, match="total does not match"):
        require_vllm_prefill_work_total(
            VllmPrefillWorkSnapshot(3, 810),
            expected_prompt_tokens=811,
            expected_requests=3,
        )


def test_prompt_source_counters_require_zero_external_credit() -> None:
    before = parse_vllm_prompt_source_snapshot(_prompt_source_text(2, 1, 0))
    after = parse_vllm_prompt_source_snapshot(_prompt_source_text(282, 1, 0))
    assert has_vllm_prompt_source_metric_surface(_prompt_source_text(0, 0, 0))
    delta = vllm_prompt_source_delta(before, after)
    require_full_prefill_prompt_source_delta(delta, expected_prompt_tokens=280)
    assert delta == {
        "local_compute": 280,
        "local_cache_hit": 0,
        "external_kv_transfer": 0,
    }


def test_prompt_source_counters_allow_a_cold_initial_snapshot_only() -> None:
    assert parse_vllm_prompt_source_snapshot(
        "", allow_missing=True
    ) == {
        "local_compute": 0,
        "local_cache_hit": 0,
        "external_kv_transfer": 0,
    }
    with pytest.raises(ValueError, match="family is incomplete"):
        parse_vllm_prompt_source_snapshot(
            'vllm:prompt_tokens_by_source{source="local_compute"} 1\n',
            allow_missing=True,
        )


def test_prompt_source_counters_sum_engines_and_label_order() -> None:
    text = _prompt_source_text(2, 1, 0) + (
        'vllm:prompt_tokens_by_source{source="local_compute",engine="1"} 3\n'
        'vllm:prompt_tokens_by_source{source="local_cache_hit",engine="1"} 4\n'
        'vllm:prompt_tokens_by_source{source="external_kv_transfer",engine="1"} 5\n'
    )
    assert parse_vllm_prompt_source_snapshot(text) == {
        "local_compute": 5,
        "local_cache_hit": 5,
        "external_kv_transfer": 5,
    }

    emitted = _prompt_source_text(3, 4, 5).replace(
        "vllm:prompt_tokens_by_source{",
        "vllm:prompt_tokens_by_source_total{",
    )
    assert parse_vllm_prompt_source_snapshot(emitted) == {
        "local_compute": 3,
        "local_cache_hit": 4,
        "external_kv_transfer": 5,
    }


def test_prompt_source_external_or_unknown_labels_fail_closed() -> None:
    with pytest.raises(ValueError, match="family is incomplete"):
        parse_vllm_prompt_source_snapshot(
            'vllm:prompt_tokens_by_source{source="local_compute"} 1\n'
        )
    with pytest.raises(ValueError, match="prompt-source label"):
        parse_vllm_prompt_source_snapshot(
            'vllm:prompt_tokens_by_source{source="unknown"} 1\n'
        )
    with pytest.raises(ValueError, match="not full-prefill"):
        require_full_prefill_prompt_source_delta(
            {
                "local_compute": 279,
                "local_cache_hit": 1,
                "external_kv_transfer": 0,
            },
            expected_prompt_tokens=280,
        )


def test_native_request_evidence_reconciles_all_pinned_metrics() -> None:
    evidence = VllmNativeRequestEvidence(
        prompt_tokens_processed=280,
        prompt_source_delta=VllmPromptSourceDelta(280, 0, 0),
        prefill_work=VllmPrefillWorkSnapshot(1, 280),
        timing_delta=parse_vllm_timing_snapshot(_snapshot_text(count=1)),
    )
    assert evidence.as_dict()["prompt_tokens_processed"] == 280
    assert evidence.as_dict()["prefill_work"] == {
        "observations": 1,
        "kv_computed_tokens": 280,
    }


def test_native_request_evidence_rejects_nonlocal_or_partial_work() -> None:
    kwargs = {
        "prompt_tokens_processed": 280,
        "prefill_work": VllmPrefillWorkSnapshot(1, 280),
        "timing_delta": parse_vllm_timing_snapshot(_snapshot_text(count=1)),
    }
    with pytest.raises(ValueError, match="does not reconcile"):
        VllmNativeRequestEvidence(
            prompt_source_delta=VllmPromptSourceDelta(279, 1, 0),
            **kwargs,
        )
    with pytest.raises(ValueError, match="does not reconcile"):
        VllmNativeRequestEvidence(
            prompt_source_delta=VllmPromptSourceDelta(280, 0, 0),
            prefill_work=VllmPrefillWorkSnapshot(0, 0),
            timing_delta=kwargs["timing_delta"],
            prompt_tokens_processed=280,
        )


def test_timing_delta_and_complete_request_gate() -> None:
    before = parse_vllm_timing_snapshot(_snapshot_text(count=2, sum_seconds=4.0))
    after = parse_vllm_timing_snapshot(_snapshot_text(count=5, sum_seconds=7.0))

    delta = vllm_timing_snapshot_delta(before, after)
    require_vllm_timing_delta(delta, expected_requests=3)
    assert delta.ttft_seconds == VllmTimingSummary(count=3, sum_seconds=3.0)


def test_timing_surface_requires_all_count_and_sum_pairs() -> None:
    incomplete = _snapshot_text().replace(
        "vllm:request_decode_time_seconds_sum{engine=\"0\"} 3.4\n", ""
    )
    assert not has_vllm_timing_metric_surface(incomplete)
    with pytest.raises(ValueError, match="family is incomplete"):
        parse_vllm_timing_snapshot(incomplete)


def test_timing_delta_rejects_backwards_or_partial_intervals() -> None:
    before = parse_vllm_timing_snapshot(_snapshot_text(count=3, sum_seconds=4.0))
    after = parse_vllm_timing_snapshot(_snapshot_text(count=2, sum_seconds=5.0))
    with pytest.raises(ValueError, match="moved backwards"):
        vllm_timing_snapshot_delta(before, after)

    zero = VllmTimingSnapshot(
        *(VllmTimingSummary(count=0, sum_seconds=0.0) for _ in _METRICS)
    )
    with pytest.raises(ValueError, match="observations"):
        require_vllm_timing_delta(zero, expected_requests=3)


@pytest.mark.parametrize("value", ["nan", "+Inf", "-1"])
def test_invalid_timing_values_fail_closed(value: str) -> None:
    malformed = _snapshot_text().replace(
        'vllm:request_queue_time_seconds_sum{engine="0"} 3.2',
        f'vllm:request_queue_time_seconds_sum{{engine="0"}} {value}',
    )
    with pytest.raises(ValueError, match="timing metric value"):
        parse_vllm_timing_snapshot(malformed)
