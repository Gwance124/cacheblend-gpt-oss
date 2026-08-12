from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from cacheblend_gpt_oss.gpt_oss import (
    AttentionKind,
    CacheGroupLayout,
    GptOssHybridCacheLayout,
    GroupBlockTable,
    HybridLayoutError,
    HybridLayoutErrorCode,
    TokenTransfer,
    extract_gpt_oss_layer_index,
    plan_token_scatter,
)
from cacheblend_gpt_oss.planner import TokenRange


def layer_name(index: int) -> str:
    return f"model.layers.{index}.attn.attn"


def group(
    group_id: int,
    kind: AttentionKind,
    indexes: tuple[int, ...],
    *,
    block_size: int,
) -> CacheGroupLayout:
    return CacheGroupLayout(
        group_id=group_id,
        attention_kind=kind,
        layer_names=tuple(layer_name(index) for index in indexes),
        block_size=block_size,
        sliding_window=128 if kind is AttentionKind.SLIDING else None,
    )


@pytest.fixture
def layout() -> GptOssHybridCacheLayout:
    # Deliberately use different block sizes and reverse tuple order. The
    # planner must bind by group ID rather than assume one global table/layout.
    return GptOssHybridCacheLayout(
        groups=(
            group(1, AttentionKind.FULL, tuple(range(1, 24, 2)), block_size=16),
            group(0, AttentionKind.SLIDING, tuple(range(0, 24, 2)), block_size=8),
        )
    )


def assert_error(
    expected: HybridLayoutErrorCode, operation: object
) -> HybridLayoutError:
    assert callable(operation)
    with pytest.raises(HybridLayoutError) as caught:
        operation()
    assert caught.value.code is expected
    assert str(caught.value) == expected.value
    return caught.value


def test_layout_extracts_all_layers_and_binds_expected_groups(
    layout: GptOssHybridCacheLayout,
) -> None:
    assert [layer.layer_index for layer in layout.layers] == list(range(24))
    assert extract_gpt_oss_layer_index(layer_name(23)) == 23
    assert [layer.layer_index for layer in layout.layers_in_group(0)] == list(
        range(0, 24, 2)
    )
    assert [layer.layer_index for layer in layout.layers_in_group(1)] == list(
        range(1, 24, 2)
    )
    assert layout.layer(layer_name(0)).attention_kind is AttentionKind.SLIDING
    assert layout.layer(layer_name(0)).sliding_window == 128
    assert layout.layer(layer_name(1)).attention_kind is AttentionKind.FULL
    assert layout.layer(layer_name(1)).sliding_window is None
    with pytest.raises(FrozenInstanceError):
        layout.groups = ()  # type: ignore[misc]


def test_moved_non_aligned_document_splits_per_group_and_layer(
    layout: GptOssHybridCacheLayout,
) -> None:
    transfer = TokenTransfer(TokenRange(101, 112), TokenRange(5, 16))
    plan = plan_token_scatter(
        layout,
        (
            GroupBlockTable(1, 16, (100,)),
            GroupBlockTable(0, 8, (200, 201)),
        ),
        transfer,
    )

    sliding_spans = plan.spans_for_group(0)
    full_spans = plan.spans_for_group(1)
    assert [span.target_range for span in sliding_spans] == [
        TokenRange(5, 8),
        TokenRange(8, 16),
    ]
    assert [span.source_range for span in sliding_spans] == [
        TokenRange(101, 104),
        TokenRange(104, 112),
    ]
    assert [span.block_offset for span in sliding_spans] == [5, 0]
    assert [span.physical_slot_start for span in sliding_spans] == [1605, 1608]
    assert [span.attention_kind for span in sliding_spans] == [
        AttentionKind.SLIDING,
        AttentionKind.SLIDING,
    ]

    assert len(full_spans) == 1
    assert full_spans[0].source_range == TokenRange(101, 112)
    assert full_spans[0].target_range == TokenRange(5, 16)
    assert full_spans[0].block_offset == 5
    assert full_spans[0].physical_slot_start == 1605
    assert full_spans[0].attention_kind is AttentionKind.FULL
    assert transfer.position_delta == -96

    # Each of 12 sliding layers gets two spans; each of 12 full layers gets one.
    assert len(plan.layer_spans) == 36
    assert len(plan.spans_for_layer(layer_name(0))) == 2
    assert len(plan.spans_for_layer(layer_name(1))) == 1
    assert all(
        span.group_id == 0
        and span.attention_kind is AttentionKind.SLIDING
        for span in plan.spans_for_layer(layer_name(0))
    )
    assert all(
        span.group_id == 1 and span.attention_kind is AttentionKind.FULL
        for span in plan.spans_for_layer(layer_name(1))
    )


def test_exact_block_boundary_is_one_span_per_group(
    layout: GptOssHybridCacheLayout,
) -> None:
    plan = plan_token_scatter(
        layout,
        (
            GroupBlockTable(0, 8, (7, 8)),
            GroupBlockTable(1, 16, (9,)),
        ),
        TokenTransfer(TokenRange(40, 48), TokenRange(8, 16)),
    )

    assert len(plan.spans_for_group(0)) == 1
    assert plan.spans_for_group(0)[0].logical_block_index == 1
    assert plan.spans_for_group(0)[0].block_offset == 0
    assert len(plan.spans_for_group(1)) == 1
    assert plan.spans_for_group(1)[0].block_offset == 8


def test_one_token_crossing_boundary_splits_without_gaps(
    layout: GptOssHybridCacheLayout,
) -> None:
    plan = plan_token_scatter(
        layout,
        (
            GroupBlockTable(0, 8, (11, 12)),
            GroupBlockTable(1, 16, (13,)),
        ),
        TokenTransfer(TokenRange(20, 22), TokenRange(7, 9)),
    )

    spans = plan.spans_for_group(0)
    assert [(span.target_range.start, span.target_range.end) for span in spans] == [
        (7, 8),
        (8, 9),
    ]
    assert [(span.source_range.start, span.source_range.end) for span in spans] == [
        (20, 21),
        (21, 22),
    ]


def test_missing_layer_is_rejected() -> None:
    operation = lambda: GptOssHybridCacheLayout(  # noqa: E731
        groups=(
            group(0, AttentionKind.SLIDING, tuple(range(0, 24, 2)), block_size=8),
            group(1, AttentionKind.FULL, tuple(range(1, 23, 2)), block_size=8),
        )
    )
    assert_error(HybridLayoutErrorCode.INVALID_LAYER_COUNT, operation)


def test_duplicate_layer_name_is_rejected() -> None:
    full_indexes = (*tuple(range(1, 22, 2)), 0)
    operation = lambda: GptOssHybridCacheLayout(  # noqa: E731
        groups=(
            group(0, AttentionKind.SLIDING, tuple(range(0, 24, 2)), block_size=8),
            group(1, AttentionKind.FULL, full_indexes, block_size=8),
        )
    )
    assert_error(HybridLayoutErrorCode.DUPLICATE_LAYER_NAME, operation)


def test_wrong_even_odd_attention_pattern_is_rejected() -> None:
    operation = lambda: GptOssHybridCacheLayout(  # noqa: E731
        groups=(
            group(
                0,
                AttentionKind.SLIDING,
                (1, *tuple(range(2, 24, 2))),
                block_size=8,
            ),
            group(
                1,
                AttentionKind.FULL,
                (0, *tuple(range(3, 24, 2))),
                block_size=8,
            ),
        )
    )
    assert_error(HybridLayoutErrorCode.ATTENTION_PATTERN_MISMATCH, operation)


def test_wrong_sliding_window_is_rejected() -> None:
    operation = lambda: CacheGroupLayout(  # noqa: E731
        group_id=0,
        attention_kind=AttentionKind.SLIDING,
        layer_names=(layer_name(0),),
        block_size=8,
        sliding_window=256,
    )
    assert_error(HybridLayoutErrorCode.SLIDING_WINDOW_MISMATCH, operation)


def test_duplicate_and_noncontiguous_group_ids_are_rejected() -> None:
    sliding = group(
        0, AttentionKind.SLIDING, tuple(range(0, 24, 2)), block_size=8
    )
    assert_error(
        HybridLayoutErrorCode.DUPLICATE_GROUP_ID,
        lambda: GptOssHybridCacheLayout(
            groups=(
                sliding,
                group(0, AttentionKind.FULL, tuple(range(1, 24, 2)), block_size=8),
            )
        ),
    )
    assert_error(
        HybridLayoutErrorCode.INVALID_GROUP_ID,
        lambda: GptOssHybridCacheLayout(
            groups=(
                sliding,
                group(2, AttentionKind.FULL, tuple(range(1, 24, 2)), block_size=8),
            )
        ),
    )


@pytest.mark.parametrize(
    "bad_name",
    [
        "model.layers.0.attn",
        "draft_model.layers.0.attn.attn",
        "model.layers.zero.attn.attn",
        "model.layers.0.attn.attn.extra",
    ],
)
def test_noncanonical_layer_name_is_rejected(bad_name: str) -> None:
    assert_error(
        HybridLayoutErrorCode.INVALID_LAYER_NAME,
        lambda: extract_gpt_oss_layer_index(bad_name),
    )


def test_out_of_range_layer_index_is_rejected() -> None:
    assert_error(
        HybridLayoutErrorCode.LAYER_INDEX_OUT_OF_RANGE,
        lambda: extract_gpt_oss_layer_index(layer_name(24)),
    )


def test_too_short_block_table_fails_before_a_plan_is_returned(
    layout: GptOssHybridCacheLayout,
) -> None:
    operation = lambda: plan_token_scatter(  # noqa: E731
        layout,
        (
            GroupBlockTable(0, 8, (10,)),
            GroupBlockTable(1, 16, (20,)),
        ),
        TokenTransfer(TokenRange(100, 104), TokenRange(8, 12)),
    )
    assert_error(HybridLayoutErrorCode.BLOCK_TABLE_TOO_SHORT, operation)


def test_unavailable_sliding_block_is_rejected(
    layout: GptOssHybridCacheLayout,
) -> None:
    operation = lambda: plan_token_scatter(  # noqa: E731
        layout,
        (
            GroupBlockTable(0, 8, (10, None)),
            GroupBlockTable(1, 16, (20,)),
        ),
        TokenTransfer(TokenRange(100, 104), TokenRange(8, 12)),
    )
    assert_error(HybridLayoutErrorCode.BLOCK_UNAVAILABLE, operation)


def test_missing_or_mismatched_group_table_is_rejected(
    layout: GptOssHybridCacheLayout,
) -> None:
    transfer = TokenTransfer(TokenRange(0, 1), TokenRange(0, 1))
    assert_error(
        HybridLayoutErrorCode.MISSING_BLOCK_TABLE,
        lambda: plan_token_scatter(
            layout, (GroupBlockTable(0, 8, (1,)),), transfer
        ),
    )
    assert_error(
        HybridLayoutErrorCode.BLOCK_SIZE_MISMATCH,
        lambda: plan_token_scatter(
            layout,
            (
                GroupBlockTable(0, 16, (1,)),
                GroupBlockTable(1, 16, (2,)),
            ),
            transfer,
        ),
    )


def test_unknown_layer_and_group_fail_closed(
    layout: GptOssHybridCacheLayout,
) -> None:
    assert_error(HybridLayoutErrorCode.UNKNOWN_GROUP, lambda: layout.group(2))
    assert_error(
        HybridLayoutErrorCode.UNKNOWN_LAYER,
        lambda: layout.layer("model.layers.24.attn.attn"),
    )
    plan = plan_token_scatter(
        layout,
        (
            GroupBlockTable(0, 8, (1,)),
            GroupBlockTable(1, 16, (2,)),
        ),
        TokenTransfer(TokenRange(0, 1), TokenRange(0, 1)),
    )
    assert_error(HybridLayoutErrorCode.UNKNOWN_GROUP, lambda: plan.spans_for_group(2))
    assert_error(
        HybridLayoutErrorCode.UNKNOWN_LAYER,
        lambda: plan.spans_for_layer("model.layers.24.attn.attn"),
    )


def test_invalid_transfer_ranges_and_context_limit_are_rejected() -> None:
    assert_error(
        HybridLayoutErrorCode.EMPTY_TRANSFER,
        lambda: TokenTransfer(TokenRange(1, 1), TokenRange(2, 2)),
    )
    assert_error(
        HybridLayoutErrorCode.RANGE_LENGTH_MISMATCH,
        lambda: TokenTransfer(TokenRange(0, 2), TokenRange(0, 3)),
    )
    assert_error(
        HybridLayoutErrorCode.POSITION_OUT_OF_RANGE,
        lambda: TokenTransfer(TokenRange(0, 1), TokenRange(131_072, 131_073)),
    )
