from __future__ import annotations

from dataclasses import replace

import pytest

from cacheblend_gpt_oss.gpt_oss.layout import (
    AttentionKind,
    CacheGroupLayout,
    GptOssHybridCacheLayout,
    GroupBlockTable,
    TokenTransfer,
    plan_token_scatter,
)
from cacheblend_gpt_oss.gpt_oss.selective import ForwardRowPlan
from cacheblend_gpt_oss.gpt_oss.selective_kv import (
    SelectiveWriteError,
    SelectiveWriteErrorCode,
    plan_selective_kv_writes,
)
from cacheblend_gpt_oss.planner import TokenRange


def _layer_name(index: int) -> str:
    return f"model.layers.{index}.attn.attn"


def _scatter_spans():
    layout = GptOssHybridCacheLayout(
        (
            CacheGroupLayout(
                0,
                AttentionKind.FULL,
                tuple(_layer_name(index) for index in range(1, 24, 2)),
                16,
                None,
            ),
            CacheGroupLayout(
                1,
                AttentionKind.SLIDING,
                tuple(_layer_name(index) for index in range(0, 24, 2)),
                16,
                128,
            ),
        )
    )
    return plan_token_scatter(
        layout,
        (
            GroupBlockTable(0, 16, (0, 1)),
            GroupBlockTable(1, 16, (2, 3)),
        ),
        TokenTransfer(TokenRange(100, 120), TokenRange(0, 20)),
    ).layer_spans


def test_full_recompute_keeps_every_original_span_and_slot() -> None:
    full_spans = _scatter_spans()
    result = plan_selective_kv_writes(
        full_spans,
        ForwardRowPlan.full_recompute(20),
    )

    assert result.full_layer_spans == full_spans
    assert result.recompute_layer_spans == full_spans
    assert result.recompute_tokens == 24 * 20
    assert result.cached_tokens == 0
    assert not result.sinks_touched


def test_moved_cached_rows_split_spans_and_preserve_source_positions() -> None:
    ranges = [
        (TokenRange(0, 3), TokenRange(10, 20))
        if index == 0
        else (TokenRange(0, 20),)
        for index in range(24)
    ]
    result = plan_selective_kv_writes(
        _scatter_spans(),
        ForwardRowPlan.from_recompute_ranges(20, ranges),
    )

    layer_zero = result.spans_for_layer(0)
    assert tuple(span.target_range for span in layer_zero) == (
        TokenRange(0, 3),
        TokenRange(10, 16),
        TokenRange(16, 20),
    )
    assert tuple(span.source_range for span in layer_zero) == (
        TokenRange(100, 103),
        TokenRange(110, 116),
        TokenRange(116, 120),
    )
    assert tuple(span.physical_slot_start for span in layer_zero) == (32, 42, 48)
    assert result.cached_ranges_for_layer(0) == (TokenRange(3, 10),)
    assert result.cached_tokens == 7
    assert result.recompute_tokens == 23 * 20 + 13
    assert sum(span.token_count for span in layer_zero) == 13


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (
            lambda spans: spans[1:],
            SelectiveWriteErrorCode.RANGE_COVERAGE_MISMATCH,
        ),
        (
            lambda spans: (
                *spans[:1],
                replace(
                    spans[1],
                    group_span=replace(
                        spans[1].group_span,
                        physical_slot_start=999,
                    ),
                ),
                *spans[2:],
            ),
            SelectiveWriteErrorCode.INVALID_PHYSICAL_SLOT,
        ),
    ],
)
def test_invalid_full_spans_fail_before_selective_plan(mutate, code) -> None:
    with pytest.raises(SelectiveWriteError) as error:
        plan_selective_kv_writes(
            mutate(_scatter_spans()),
            ForwardRowPlan.full_recompute(20),
        )
    assert error.value.code is code


def test_wrong_prompt_extent_and_missing_layer_fail_closed() -> None:
    with pytest.raises(SelectiveWriteError) as extent:
        plan_selective_kv_writes(_scatter_spans(), ForwardRowPlan.full_recompute(19))
    assert extent.value.code is SelectiveWriteErrorCode.RANGE_OUT_OF_BOUNDS

    missing_layer = tuple(span for span in _scatter_spans() if span.layer_index != 23)
    with pytest.raises(SelectiveWriteError) as missing:
        plan_selective_kv_writes(missing_layer, ForwardRowPlan.full_recompute(20))
    assert missing.value.code is SelectiveWriteErrorCode.LAYER_SET_MISMATCH


def test_layer_lookup_rejects_bool_and_out_of_range_indices() -> None:
    result = plan_selective_kv_writes(
        _scatter_spans(),
        ForwardRowPlan.full_recompute(20),
    )
    for index in (True, -1, 24):
        with pytest.raises(SelectiveWriteError) as error:
            result.spans_for_layer(index)
        assert error.value.code is SelectiveWriteErrorCode.INVALID_LAYER
