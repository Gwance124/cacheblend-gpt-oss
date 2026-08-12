# SPDX-License-Identifier: Apache-2.0
"""Plan-aware GPT-OSS KV writes for the future selective attention backend.

The pinned Triton path updates a full prompt's KV rows before attention.  A
selective backend must instead write only rows selected for recomputation and
leave verified, position-corrected cached rows untouched.  This module plans
that split from the dependency-free :class:`ForwardRowPlan` and the existing
hybrid group scatter spans; it performs no tensor mutation and imports neither
vLLM nor Torch.

The source spans are expected to cover one complete prompt for all 24
GPT-OSS layers.  Each selected sub-span keeps the old source positions (for
YaRN correction) while changing only the destination target range and physical
slot offset.  The resulting plan is a CPU-testable contract for a future
``AttentionImpl.do_kv_cache_update`` implementation.  The live connector does
not consume it yet.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import NoReturn

from cacheblend_gpt_oss.gpt_oss.layout import (
    GPT_OSS_MAX_CONTEXT_TOKENS,
    GPT_OSS_NUM_LAYERS,
    AttentionKind,
    GroupTokenScatterSpan,
    LayerTokenScatterSpan,
)
from cacheblend_gpt_oss.gpt_oss.selective import (
    ForwardRowPlan,
    LayerRowSelection,
)
from cacheblend_gpt_oss.planner.models import TokenRange


class SelectiveWriteErrorCode(str, Enum):
    """Bounded failures for selective row-to-slot planning."""

    INVALID_PLAN = "invalid_plan"
    EMPTY_SPANS = "empty_spans"
    LAYER_SET_MISMATCH = "layer_set_mismatch"
    INVALID_LAYER = "invalid_layer"
    INVALID_SPAN = "invalid_span"
    RANGE_OUT_OF_BOUNDS = "range_out_of_bounds"
    RANGE_COVERAGE_MISMATCH = "range_coverage_mismatch"
    OVERLAPPING_SPANS = "overlapping_spans"
    INVALID_PHYSICAL_SLOT = "invalid_physical_slot"
    ATTENTION_PATTERN_MISMATCH = "attention_pattern_mismatch"


class SelectiveWriteError(ValueError):
    """Fail-closed selective write planning error."""

    def __init__(self, code: SelectiveWriteErrorCode) -> None:
        self.code = code
        super().__init__(code.value)


def _fail(code: SelectiveWriteErrorCode) -> NoReturn:
    raise SelectiveWriteError(code)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _expected_kind(layer_index: int) -> AttentionKind:
    return AttentionKind.SLIDING if layer_index % 2 == 0 else AttentionKind.FULL


def _validate_group_span(
    span: LayerTokenScatterSpan,
    *,
    prompt_tokens: int,
) -> None:
    if (
        not isinstance(span, LayerTokenScatterSpan)
        or not _is_int(span.layer_index)
        or not 0 <= span.layer_index < GPT_OSS_NUM_LAYERS
        or span.layer_name != f"model.layers.{span.layer_index}.attn.attn"
    ):
        _fail(SelectiveWriteErrorCode.INVALID_LAYER)
    group_span = span.group_span
    if not isinstance(group_span, GroupTokenScatterSpan):
        _fail(SelectiveWriteErrorCode.INVALID_SPAN)
    if span.attention_kind is not _expected_kind(span.layer_index):
        _fail(SelectiveWriteErrorCode.ATTENTION_PATTERN_MISMATCH)
    if (
        len(span.source_range) == 0
        or len(span.target_range) == 0
        or len(span.source_range) != len(span.target_range)
        or span.source_range.end > GPT_OSS_MAX_CONTEXT_TOKENS
        or span.target_range.end > prompt_tokens
    ):
        _fail(SelectiveWriteErrorCode.RANGE_OUT_OF_BOUNDS)
    if (
        not _is_int(group_span.block_id)
        or group_span.block_id < 0
        or not _is_int(group_span.block_size)
        or group_span.block_size <= 0
        or not 0 <= group_span.block_offset < group_span.block_size
        or group_span.block_offset + span.token_count > group_span.block_size
        or group_span.physical_slot_start
        != group_span.block_id * group_span.block_size + group_span.block_offset
    ):
        _fail(SelectiveWriteErrorCode.INVALID_PHYSICAL_SLOT)


def _split_span(
    span: LayerTokenScatterSpan,
    selection: LayerRowSelection,
) -> tuple[LayerTokenScatterSpan, ...]:
    """Intersect one block-contained span with canonical recompute ranges."""

    output: list[LayerTokenScatterSpan] = []
    for recompute_range in selection.recompute_ranges:
        start = max(span.target_range.start, recompute_range.start)
        end = min(span.target_range.end, recompute_range.end)
        if start >= end:
            continue
        offset = start - span.target_range.start
        count = end - start
        group_span = span.group_span
        source_start = span.source_range.start + offset
        adjusted_group = GroupTokenScatterSpan(
            group_id=group_span.group_id,
            attention_kind=group_span.attention_kind,
            source_range=TokenRange(source_start, source_start + count),
            target_range=TokenRange(start, end),
            logical_block_index=group_span.logical_block_index,
            block_id=group_span.block_id,
            block_size=group_span.block_size,
            block_offset=group_span.block_offset + offset,
            physical_slot_start=group_span.physical_slot_start + offset,
        )
        output.append(
            LayerTokenScatterSpan(
                layer_name=span.layer_name,
                layer_index=span.layer_index,
                group_span=adjusted_group,
            )
        )
    return tuple(output)


def _validate_layer_coverage(
    spans: Sequence[LayerTokenScatterSpan],
    selection: LayerRowSelection,
) -> tuple[LayerTokenScatterSpan, ...]:
    ordered = tuple(sorted(spans, key=lambda span: span.target_range.start))
    cursor = 0
    for span in ordered:
        if span.target_range.start != cursor:
            _fail(SelectiveWriteErrorCode.RANGE_COVERAGE_MISMATCH)
        if span.target_range.end <= span.target_range.start:
            _fail(SelectiveWriteErrorCode.INVALID_SPAN)
        cursor = span.target_range.end
    if cursor != selection.prompt_tokens:
        _fail(SelectiveWriteErrorCode.RANGE_COVERAGE_MISMATCH)
    return ordered


@dataclass(frozen=True, slots=True)
class SelectiveWritePlan:
    """Full prompt spans plus the subset written by recomputed rows."""

    row_plan: ForwardRowPlan
    full_layer_spans: tuple[LayerTokenScatterSpan, ...]
    recompute_layer_spans: tuple[LayerTokenScatterSpan, ...]

    @property
    def cached_tokens(self) -> int:
        return self.row_plan.cached_tokens

    @property
    def recompute_tokens(self) -> int:
        return self.row_plan.recompute_tokens

    @property
    def sinks_touched(self) -> bool:
        return False

    def spans_for_layer(self, layer_index: int) -> tuple[LayerTokenScatterSpan, ...]:
        if not _is_int(layer_index) or not 0 <= layer_index < GPT_OSS_NUM_LAYERS:
            _fail(SelectiveWriteErrorCode.INVALID_LAYER)
        return tuple(
            span
            for span in self.recompute_layer_spans
            if span.layer_index == layer_index
        )

    def cached_ranges_for_layer(self, layer_index: int) -> tuple[TokenRange, ...]:
        if not _is_int(layer_index) or not 0 <= layer_index < GPT_OSS_NUM_LAYERS:
            _fail(SelectiveWriteErrorCode.INVALID_LAYER)
        return self.row_plan.layer(layer_index).cached_ranges


def plan_selective_kv_writes(
    layer_spans: Sequence[LayerTokenScatterSpan],
    row_plan: ForwardRowPlan,
) -> SelectiveWritePlan:
    """Split complete hybrid spans into recompute-only destination writes."""

    if not isinstance(row_plan, ForwardRowPlan):
        _fail(SelectiveWriteErrorCode.INVALID_PLAN)
    spans = tuple(layer_spans)
    if not spans:
        _fail(SelectiveWriteErrorCode.EMPTY_SPANS)

    by_layer: dict[int, list[LayerTokenScatterSpan]] = {}
    for span in spans:
        _validate_group_span(span, prompt_tokens=row_plan.prompt_tokens)
        by_layer.setdefault(span.layer_index, []).append(span)
    if set(by_layer) != set(range(GPT_OSS_NUM_LAYERS)):
        _fail(SelectiveWriteErrorCode.LAYER_SET_MISMATCH)

    ordered_full: list[LayerTokenScatterSpan] = []
    ordered_recompute: list[LayerTokenScatterSpan] = []
    for layer_index in range(GPT_OSS_NUM_LAYERS):
        selection = row_plan.layer(layer_index)
        layer_spans = _validate_layer_coverage(by_layer[layer_index], selection)
        ordered_full.extend(layer_spans)
        for span in layer_spans:
            ordered_recompute.extend(_split_span(span, selection))

    return SelectiveWritePlan(
        row_plan=row_plan,
        full_layer_spans=tuple(ordered_full),
        recompute_layer_spans=tuple(ordered_recompute),
    )


__all__ = [
    "SelectiveWriteError",
    "SelectiveWriteErrorCode",
    "SelectiveWritePlan",
    "plan_selective_kv_writes",
]
