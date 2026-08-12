from __future__ import annotations

import pytest

from cacheblend_gpt_oss.correctness import (
    VllmTimingSnapshot,
    VllmTimingSummary,
    has_vllm_prompt_metric_surface,
    has_vllm_timing_metric_surface,
    parse_vllm_prompt_counter_snapshot,
    parse_vllm_timing_snapshot,
    require_vllm_timing_delta,
    vllm_prompt_counter_delta,
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


def test_native_prompt_counter_rejects_missing_or_backwards_intervals() -> None:
    assert not has_vllm_prompt_metric_surface("vllm:num_requests_running 0\n")
    empty = parse_vllm_prompt_counter_snapshot("")
    assert empty == {"prompt_tokens": 0}
    with pytest.raises(ValueError, match="moved backwards"):
        vllm_prompt_counter_delta(
            {"prompt_tokens": 10},
            {"prompt_tokens": 9},
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
