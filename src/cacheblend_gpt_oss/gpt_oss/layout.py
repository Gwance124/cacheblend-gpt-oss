# SPDX-License-Identifier: Apache-2.0
"""Immutable GPT-OSS-20B hybrid KV-cache layout and scatter planning.

This module is a dependency-free description layer.  The version-scoped vLLM
adapter translates the pinned runtime objects into these value objects; this
module itself deliberately imports neither vLLM nor Torch.

The assumptions and slot calculation are tied to vLLM 0.19.1 commit
``b1388b1fbf5aaef47937fabe98931211684666a6``:

* GPT-OSS assigns sliding attention to even layers and full attention to odd
  layers, and passes the learned sinks separately from KV:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/models/gpt_oss.py#L67-L153
* GPT-OSS builds its numbered transformer layers under ``model.layers``:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/models/gpt_oss.py#L261-L285
* each ``KVCacheGroupSpec`` owns one layer-name set and one cache spec:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/kv_cache_interface.py#L461-L490
* grouped block IDs have one outer entry per cache group:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/core/kv_cache_manager.py#L21-L80
* the model runner computes ``block_id * block_size + block_offset`` separately
  for each group and assigns that mapping to every layer in the group:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/worker/gpu/block_table.py#L212-L274
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/worker/gpu/attn_utils.py#L173-L181

These descriptors model only the audited TP=1, PP=1 GPT-OSS-20B target.  A
generic-looking value object is not a support claim for any other model.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import NoReturn

from cacheblend_gpt_oss.planner.models import TokenRange

GPT_OSS_NUM_LAYERS = 24
GPT_OSS_NUM_CACHE_GROUPS = 2
GPT_OSS_SLIDING_WINDOW = 128
GPT_OSS_MAX_CONTEXT_TOKENS = 131_072
MAX_VLLM_BLOCK_ID = (1 << 31) - 1

# ``OAIAttention`` passes ``model.layers.<index>.attn.attn`` to the generic
# Attention layer.  Keep this strict: speculative/draft prefixes and copied
# model trees are outside the audited runtime envelope.
_LAYER_NAME_PATTERN = re.compile(r"^model\.layers\.(\d+)\.attn\.attn$")


class AttentionKind(str, Enum):
    """The two attention/cache kinds in the pinned GPT-OSS-20B layout."""

    SLIDING = "sliding"
    FULL = "full"


class HybridLayoutErrorCode(str, Enum):
    """Stable failure codes safe for bounded logs and metric labels."""

    INVALID_GROUP_COUNT = "invalid_group_count"
    INVALID_GROUP_ID = "invalid_group_id"
    DUPLICATE_GROUP_ID = "duplicate_group_id"
    INVALID_LAYER_COUNT = "invalid_layer_count"
    INVALID_LAYER_NAME = "invalid_layer_name"
    DUPLICATE_LAYER_NAME = "duplicate_layer_name"
    DUPLICATE_LAYER_INDEX = "duplicate_layer_index"
    LAYER_INDEX_OUT_OF_RANGE = "layer_index_out_of_range"
    ATTENTION_PATTERN_MISMATCH = "attention_pattern_mismatch"
    SLIDING_WINDOW_MISMATCH = "sliding_window_mismatch"
    INVALID_BLOCK_SIZE = "invalid_block_size"
    MISSING_BLOCK_TABLE = "missing_block_table"
    UNEXPECTED_BLOCK_TABLE = "unexpected_block_table"
    DUPLICATE_BLOCK_TABLE = "duplicate_block_table"
    BLOCK_SIZE_MISMATCH = "block_size_mismatch"
    INVALID_BLOCK_ID = "invalid_block_id"
    EMPTY_TRANSFER = "empty_transfer"
    RANGE_LENGTH_MISMATCH = "range_length_mismatch"
    POSITION_OUT_OF_RANGE = "position_out_of_range"
    BLOCK_TABLE_TOO_SHORT = "block_table_too_short"
    BLOCK_UNAVAILABLE = "block_unavailable"
    UNKNOWN_GROUP = "unknown_group"
    UNKNOWN_LAYER = "unknown_layer"


class HybridLayoutError(ValueError):
    """A fail-closed layout/planning error with a bounded machine code."""

    def __init__(self, code: HybridLayoutErrorCode) -> None:
        self.code = code
        super().__init__(code.value)


def _raise(code: HybridLayoutErrorCode) -> NoReturn:
    raise HybridLayoutError(code)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def extract_gpt_oss_layer_index(layer_name: str) -> int:
    """Extract and bound the index from one exact GPT-OSS attention name."""

    if not isinstance(layer_name, str):
        _raise(HybridLayoutErrorCode.INVALID_LAYER_NAME)
    match = _LAYER_NAME_PATTERN.fullmatch(layer_name)
    if match is None:
        _raise(HybridLayoutErrorCode.INVALID_LAYER_NAME)
    digits = match.group(1)
    # The pinned module names are canonical decimal indices ``0`` through
    # ``23``. Reject leading-zero aliases and bound conversion before calling
    # ``int`` so malformed external objects cannot raise an unbounded parser
    # exception or alias a different layer identity.
    if len(digits) > 2:
        _raise(HybridLayoutErrorCode.LAYER_INDEX_OUT_OF_RANGE)
    if len(digits) > 1 and digits.startswith("0"):
        _raise(HybridLayoutErrorCode.INVALID_LAYER_NAME)
    layer_index = int(digits)
    if not 0 <= layer_index < GPT_OSS_NUM_LAYERS:
        _raise(HybridLayoutErrorCode.LAYER_INDEX_OUT_OF_RANGE)
    return layer_index


@dataclass(frozen=True, slots=True)
class CacheGroupLayout:
    """One vLLM hybrid cache group and all GPT-OSS layers sharing its table."""

    group_id: int
    attention_kind: AttentionKind
    layer_names: tuple[str, ...]
    block_size: int
    sliding_window: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_names", tuple(self.layer_names))
        if not _is_int(self.group_id) or self.group_id < 0:
            _raise(HybridLayoutErrorCode.INVALID_GROUP_ID)
        if not isinstance(self.attention_kind, AttentionKind):
            _raise(HybridLayoutErrorCode.ATTENTION_PATTERN_MISMATCH)
        if not _is_int(self.block_size) or self.block_size <= 0:
            _raise(HybridLayoutErrorCode.INVALID_BLOCK_SIZE)
        if self.attention_kind is AttentionKind.SLIDING:
            if self.sliding_window != GPT_OSS_SLIDING_WINDOW:
                _raise(HybridLayoutErrorCode.SLIDING_WINDOW_MISMATCH)
        elif self.sliding_window is not None:
            _raise(HybridLayoutErrorCode.SLIDING_WINDOW_MISMATCH)


@dataclass(frozen=True, slots=True)
class AttentionLayerLayout:
    """A validated GPT-OSS attention layer bound to one cache group."""

    layer_name: str
    layer_index: int
    group_id: int
    attention_kind: AttentionKind
    sliding_window: int | None
    block_size: int


@dataclass(frozen=True, slots=True)
class GptOssHybridCacheLayout:
    """The complete, validated 24-layer GPT-OSS-20B hybrid layout."""

    groups: tuple[CacheGroupLayout, ...]
    layers: tuple[AttentionLayerLayout, ...] = field(init=False)

    def __post_init__(self) -> None:
        groups = tuple(self.groups)
        object.__setattr__(self, "groups", groups)
        if len(groups) != GPT_OSS_NUM_CACHE_GROUPS:
            _raise(HybridLayoutErrorCode.INVALID_GROUP_COUNT)

        group_ids = tuple(group.group_id for group in groups)
        if len(set(group_ids)) != len(group_ids):
            _raise(HybridLayoutErrorCode.DUPLICATE_GROUP_ID)
        if set(group_ids) != set(range(GPT_OSS_NUM_CACHE_GROUPS)):
            _raise(HybridLayoutErrorCode.INVALID_GROUP_ID)
        if {group.attention_kind for group in groups} != {
            AttentionKind.SLIDING,
            AttentionKind.FULL,
        }:
            _raise(HybridLayoutErrorCode.ATTENTION_PATTERN_MISMATCH)

        layer_count = sum(len(group.layer_names) for group in groups)
        if layer_count != GPT_OSS_NUM_LAYERS:
            _raise(HybridLayoutErrorCode.INVALID_LAYER_COUNT)

        layers: list[AttentionLayerLayout] = []
        seen_names: set[str] = set()
        seen_indexes: set[int] = set()
        for group in groups:
            for layer_name in group.layer_names:
                if layer_name in seen_names:
                    _raise(HybridLayoutErrorCode.DUPLICATE_LAYER_NAME)
                seen_names.add(layer_name)
                layer_index = extract_gpt_oss_layer_index(layer_name)
                if layer_index in seen_indexes:
                    _raise(HybridLayoutErrorCode.DUPLICATE_LAYER_INDEX)
                seen_indexes.add(layer_index)

                expected_kind = (
                    AttentionKind.SLIDING
                    if layer_index % 2 == 0
                    else AttentionKind.FULL
                )
                if group.attention_kind is not expected_kind:
                    _raise(HybridLayoutErrorCode.ATTENTION_PATTERN_MISMATCH)
                layers.append(
                    AttentionLayerLayout(
                        layer_name=layer_name,
                        layer_index=layer_index,
                        group_id=group.group_id,
                        attention_kind=group.attention_kind,
                        sliding_window=group.sliding_window,
                        block_size=group.block_size,
                    )
                )

        if seen_indexes != set(range(GPT_OSS_NUM_LAYERS)):
            _raise(HybridLayoutErrorCode.INVALID_LAYER_COUNT)
        object.__setattr__(
            self,
            "layers",
            tuple(sorted(layers, key=lambda layer: layer.layer_index)),
        )

    def group(self, group_id: int) -> CacheGroupLayout:
        """Return a group by its exact vLLM group ID or fail closed."""

        for group in self.groups:
            if group.group_id == group_id:
                return group
        _raise(HybridLayoutErrorCode.UNKNOWN_GROUP)

    def layer(self, layer_name: str) -> AttentionLayerLayout:
        """Return a layer descriptor by its exact vLLM name or fail closed."""

        for layer in self.layers:
            if layer.layer_name == layer_name:
                return layer
        _raise(HybridLayoutErrorCode.UNKNOWN_LAYER)

    def layers_in_group(self, group_id: int) -> tuple[AttentionLayerLayout, ...]:
        """Return group members in transformer execution order."""

        self.group(group_id)
        return tuple(layer for layer in self.layers if layer.group_id == group_id)


@dataclass(frozen=True, slots=True)
class GroupBlockTable:
    """One request's destination block table for one hybrid cache group.

    ``None`` represents an explicitly unavailable logical block, for example a
    sliding-window block that the version adapter knows has been reclaimed.
    """

    group_id: int
    block_size: int
    block_ids: tuple[int | None, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_ids", tuple(self.block_ids))
        if not _is_int(self.group_id) or self.group_id < 0:
            _raise(HybridLayoutErrorCode.INVALID_GROUP_ID)
        if not _is_int(self.block_size) or self.block_size <= 0:
            _raise(HybridLayoutErrorCode.INVALID_BLOCK_SIZE)
        for block_id in self.block_ids:
            if block_id is None:
                continue
            if (
                not _is_int(block_id)
                or block_id < 0
                or block_id > MAX_VLLM_BLOCK_ID
            ):
                _raise(HybridLayoutErrorCode.INVALID_BLOCK_ID)

    @property
    def capacity_tokens(self) -> int:
        """Return the logical token capacity represented by this table."""

        return len(self.block_ids) * self.block_size


@dataclass(frozen=True, slots=True)
class TokenTransfer:
    """Equal-length source and destination ranges for one reusable segment."""

    source_range: TokenRange
    target_range: TokenRange

    def __post_init__(self) -> None:
        if not isinstance(self.source_range, TokenRange) or not isinstance(
            self.target_range, TokenRange
        ):
            _raise(HybridLayoutErrorCode.RANGE_LENGTH_MISMATCH)
        if len(self.source_range) == 0 or len(self.target_range) == 0:
            _raise(HybridLayoutErrorCode.EMPTY_TRANSFER)
        if len(self.source_range) != len(self.target_range):
            _raise(HybridLayoutErrorCode.RANGE_LENGTH_MISMATCH)
        if (
            self.source_range.end > GPT_OSS_MAX_CONTEXT_TOKENS
            or self.target_range.end > GPT_OSS_MAX_CONTEXT_TOKENS
        ):
            _raise(HybridLayoutErrorCode.POSITION_OUT_OF_RANGE)

    @property
    def position_delta(self) -> int:
        """Return the RoPE shift from source to target position."""

        return self.target_range.start - self.source_range.start


@dataclass(frozen=True, slots=True)
class GroupTokenScatterSpan:
    """A maximal transfer run contained in one destination cache block."""

    group_id: int
    attention_kind: AttentionKind
    source_range: TokenRange
    target_range: TokenRange
    logical_block_index: int
    block_id: int
    block_size: int
    block_offset: int
    physical_slot_start: int

    @property
    def token_count(self) -> int:
        return len(self.target_range)


@dataclass(frozen=True, slots=True)
class LayerTokenScatterSpan:
    """A group scatter span assigned to one concrete GPT-OSS layer."""

    layer_name: str
    layer_index: int
    group_span: GroupTokenScatterSpan

    @property
    def group_id(self) -> int:
        return self.group_span.group_id

    @property
    def attention_kind(self) -> AttentionKind:
        return self.group_span.attention_kind

    @property
    def source_range(self) -> TokenRange:
        return self.group_span.source_range

    @property
    def target_range(self) -> TokenRange:
        return self.group_span.target_range

    @property
    def token_count(self) -> int:
        return self.group_span.token_count

    @property
    def physical_slot_start(self) -> int:
        return self.group_span.physical_slot_start


@dataclass(frozen=True, slots=True)
class TokenScatterPlan:
    """Complete group- and layer-specific scatter work for one transfer."""

    transfer: TokenTransfer
    group_spans: tuple[GroupTokenScatterSpan, ...]
    layer_spans: tuple[LayerTokenScatterSpan, ...]

    def spans_for_group(self, group_id: int) -> tuple[GroupTokenScatterSpan, ...]:
        spans = tuple(span for span in self.group_spans if span.group_id == group_id)
        if not spans:
            _raise(HybridLayoutErrorCode.UNKNOWN_GROUP)
        return spans

    def spans_for_layer(self, layer_name: str) -> tuple[LayerTokenScatterSpan, ...]:
        spans = tuple(
            span for span in self.layer_spans if span.layer_name == layer_name
        )
        if not spans:
            _raise(HybridLayoutErrorCode.UNKNOWN_LAYER)
        return spans


def _validated_tables(
    layout: GptOssHybridCacheLayout,
    block_tables: Iterable[GroupBlockTable],
    transfer: TokenTransfer,
) -> tuple[GroupBlockTable, ...]:
    tables = tuple(block_tables)
    table_ids = tuple(table.group_id for table in tables)
    if len(set(table_ids)) != len(table_ids):
        _raise(HybridLayoutErrorCode.DUPLICATE_BLOCK_TABLE)

    expected_ids = {group.group_id for group in layout.groups}
    actual_ids = set(table_ids)
    if actual_ids - expected_ids:
        _raise(HybridLayoutErrorCode.UNEXPECTED_BLOCK_TABLE)
    if expected_ids - actual_ids:
        _raise(HybridLayoutErrorCode.MISSING_BLOCK_TABLE)

    by_id = {table.group_id: table for table in tables}
    ordered: list[GroupBlockTable] = []
    for group in sorted(layout.groups, key=lambda item: item.group_id):
        table = by_id[group.group_id]
        if table.block_size != group.block_size:
            _raise(HybridLayoutErrorCode.BLOCK_SIZE_MISMATCH)

        first_block = transfer.target_range.start // group.block_size
        last_block = (transfer.target_range.end - 1) // group.block_size
        if last_block >= len(table.block_ids):
            _raise(HybridLayoutErrorCode.BLOCK_TABLE_TOO_SHORT)
        if any(
            table.block_ids[block_index] is None
            for block_index in range(first_block, last_block + 1)
        ):
            _raise(HybridLayoutErrorCode.BLOCK_UNAVAILABLE)
        ordered.append(table)
    return tuple(ordered)


def _scatter_group(
    group: CacheGroupLayout,
    table: GroupBlockTable,
    transfer: TokenTransfer,
) -> tuple[GroupTokenScatterSpan, ...]:
    spans: list[GroupTokenScatterSpan] = []
    target_cursor = transfer.target_range.start
    while target_cursor < transfer.target_range.end:
        logical_block_index = target_cursor // group.block_size
        block_offset = target_cursor % group.block_size
        token_count = min(
            transfer.target_range.end - target_cursor,
            group.block_size - block_offset,
        )
        source_start = (
            transfer.source_range.start
            + target_cursor
            - transfer.target_range.start
        )
        block_id = table.block_ids[logical_block_index]
        # _validated_tables preflights the entire request before any output is
        # built, so this assertion cannot turn a partial plan into valid work.
        assert block_id is not None
        spans.append(
            GroupTokenScatterSpan(
                group_id=group.group_id,
                attention_kind=group.attention_kind,
                source_range=TokenRange(source_start, source_start + token_count),
                target_range=TokenRange(target_cursor, target_cursor + token_count),
                logical_block_index=logical_block_index,
                block_id=block_id,
                block_size=group.block_size,
                block_offset=block_offset,
                physical_slot_start=block_id * group.block_size + block_offset,
            )
        )
        target_cursor += token_count
    return tuple(spans)


def plan_token_scatter(
    layout: GptOssHybridCacheLayout,
    block_tables: Iterable[GroupBlockTable],
    transfer: TokenTransfer,
) -> TokenScatterPlan:
    """Plan all group/layer writes, failing before returning partial work."""

    tables = _validated_tables(layout, block_tables, transfer)
    group_spans: list[GroupTokenScatterSpan] = []
    spans_by_group: dict[int, tuple[GroupTokenScatterSpan, ...]] = {}
    for table in tables:
        group = layout.group(table.group_id)
        spans = _scatter_group(group, table, transfer)
        spans_by_group[group.group_id] = spans
        group_spans.extend(spans)

    layer_spans = tuple(
        LayerTokenScatterSpan(
            layer_name=layer.layer_name,
            layer_index=layer.layer_index,
            group_span=group_span,
        )
        for layer in layout.layers
        for group_span in spans_by_group[layer.group_id]
    )
    return TokenScatterPlan(
        transfer=transfer,
        group_spans=tuple(group_spans),
        layer_spans=layer_spans,
    )
