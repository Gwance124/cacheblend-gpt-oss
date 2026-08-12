# SPDX-License-Identifier: Apache-2.0
"""Plan-aware GPT-OSS KV writes for the future selective attention backend.

The pinned Triton path updates a full prompt's KV rows before attention.  A
selective backend must instead write only rows selected for recomputation and
leave verified, position-corrected cached rows untouched.  This module plans
that split from the dependency-free :class:`ForwardRowPlan` and the existing
hybrid group scatter spans; it performs no tensor mutation and imports neither
vLLM nor Torch.

The slot vector validated here is the ``slot_mapping`` argument of the pinned
``TritonAttentionImpl.do_kv_cache_update``:
https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/attention/backends/triton_attn.py#L575-L606

The source spans are expected to cover one complete prompt for all 24
GPT-OSS layers.  Each selected sub-span keeps the old source positions (for
YaRN correction) while changing only the destination target range and physical
slot offset.  The resulting plan is a CPU-testable contract for a future
``AttentionImpl.do_kv_cache_update`` implementation.  The live connector does
not consume it yet.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import NoReturn, Protocol, cast

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

GPT_OSS_NUM_KV_HEADS = 8
GPT_OSS_HEAD_DIM = 64


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
    INVALID_SLOT_MAPPING = "invalid_slot_mapping"
    SLOT_MAPPING_LENGTH_MISMATCH = "slot_mapping_length_mismatch"
    SLOT_MAPPING_VALUE_MISMATCH = "slot_mapping_value_mismatch"


class SelectiveUpdateErrorCode(str, Enum):
    """Bounded failures for the tensor-injected selective writer."""

    INVALID_PLAN = "invalid_plan"
    TENSOR_SET_MISMATCH = "tensor_set_mismatch"
    INVALID_SOURCE_SHAPE = "invalid_source_shape"
    INVALID_CACHE_SHAPE = "invalid_cache_shape"
    INVALID_VIEW = "invalid_view"
    DTYPE_MISMATCH = "dtype_mismatch"
    DEVICE_MISMATCH = "device_mismatch"
    INVALID_DEVICE = "invalid_device"
    SLOT_MAPPING_SET_MISMATCH = "slot_mapping_set_mismatch"
    INVALID_SLOT_MAPPING = "invalid_slot_mapping"
    SLOT_MAPPING_LENGTH_MISMATCH = "slot_mapping_length_mismatch"
    SLOT_MAPPING_VALUE_MISMATCH = "slot_mapping_value_mismatch"
    SESSION_INVALID_STATE = "session_invalid_state"
    SESSION_LAYER_ORDER_MISMATCH = "session_layer_order_mismatch"
    SESSION_DUPLICATE_LAYER = "session_duplicate_layer"
    SESSION_INCOMPLETE = "session_incomplete"
    MUTATION_FAILED = "mutation_failed"


class SelectiveWriteError(ValueError):
    """Fail-closed selective write planning error."""

    def __init__(self, code: SelectiveWriteErrorCode) -> None:
        self.code = code
        super().__init__(code.value)


def _fail(code: SelectiveWriteErrorCode) -> NoReturn:
    raise SelectiveWriteError(code)


class SelectiveUpdateError(RuntimeError):
    """Fail-closed tensor update error; callers must discard the request KV."""

    def __init__(self, code: SelectiveUpdateErrorCode) -> None:
        self.code = code
        super().__init__(code.value)


def _fail_update(code: SelectiveUpdateErrorCode) -> NoReturn:
    raise SelectiveUpdateError(code)


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


def validate_slot_mapping(
    plan: SelectiveWritePlan,
    *,
    layer_index: int,
    slot_mapping: Sequence[object],
) -> tuple[int, ...]:
    """Validate vLLM's flattened per-token physical slot mapping.

    The pinned Triton ``do_kv_cache_update`` receives a flattened slot vector
    alongside the post-RoPE K/V rows.  ``SelectiveWritePlan`` already carries
    the destination block and offset for every target token, but a future
    backend must still check that the runtime vector agrees before it writes
    only recomputed rows.  The input is intentionally a CPU-friendly sequence;
    the CUDA adapter must call ``tolist()`` (or an equivalent bounded reader)
    before entering this contract.

    ``-1``/padding slots are rejected.  The first selective experiment uses a
    single eager prompt with prefix caching disabled, so every prompt row must
    have a concrete destination.  Rejecting any other shape or value prevents
    a stale row from being written to an unrelated block.
    """

    if not isinstance(plan, SelectiveWritePlan):
        _fail(SelectiveWriteErrorCode.INVALID_PLAN)
    if not _is_int(layer_index) or not 0 <= layer_index < GPT_OSS_NUM_LAYERS:
        _fail(SelectiveWriteErrorCode.INVALID_LAYER)
    try:
        normalized = tuple(slot_mapping)
    except TypeError as error:
        raise SelectiveWriteError(
            SelectiveWriteErrorCode.INVALID_SLOT_MAPPING
        ) from error
    prompt_tokens = plan.row_plan.prompt_tokens
    if len(normalized) != prompt_tokens:
        _fail(SelectiveWriteErrorCode.SLOT_MAPPING_LENGTH_MISMATCH)
    normalized_slots: list[int] = []
    for slot in normalized:
        if not _is_int(slot):
            _fail(SelectiveWriteErrorCode.SLOT_MAPPING_VALUE_MISMATCH)
        slot_int = cast(int, slot)
        if slot_int < 0:
            _fail(SelectiveWriteErrorCode.SLOT_MAPPING_VALUE_MISMATCH)
        normalized_slots.append(slot_int)
    normalized_ints = tuple(normalized_slots)

    expected_by_position: dict[int, int] = {}
    for span in plan.full_layer_spans:
        if span.layer_index != layer_index:
            continue
        for offset in range(span.token_count):
            position = span.target_range.start + offset
            expected_by_position[position] = span.physical_slot_start + offset
    if len(expected_by_position) != prompt_tokens:
        _fail(SelectiveWriteErrorCode.RANGE_COVERAGE_MISMATCH)
    for position, actual in enumerate(normalized_ints):
        if actual != expected_by_position[position]:
            _fail(SelectiveWriteErrorCode.SLOT_MAPPING_VALUE_MISMATCH)
    return normalized_ints


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


class SelectiveCacheOps(Protocol):
    """Minimal tensor surface for a plan-aware cache update."""

    def shape(self, tensor: object) -> tuple[int, ...]:
        """Return a tensor shape."""

    def dtype_name(self, tensor: object) -> str:
        """Return a stable dtype identifier."""

    def device_name(self, tensor: object) -> str:
        """Return a stable device identifier."""

    def prompt_rows(self, tensor: object, *, start: int, count: int) -> object:
        """Return a ``[count, 8, 64]`` view from model-produced rows."""

    def paged_rows(
        self,
        tensor: object,
        *,
        component: int,
        block_id: int,
        block_offset: int,
        count: int,
    ) -> object:
        """Return a ``[count, 8, 64]`` paged-cache view."""

    def copy(self, destination: object, source: object) -> None:
        """Copy one equal-shaped selected row span."""

    def synchronize(self, tensor: object) -> None:
        """Synchronize the destination device after all writes."""


@dataclass(frozen=True, slots=True)
class SelectiveUpdateReceipt:
    """Counts from one fully preflighted selective KV update."""

    recomputed_token_rows: int
    cached_token_rows: int
    write_span_count: int
    copied_key_rows: int
    copied_value_rows: int
    sinks_touched: bool = False

    def __post_init__(self) -> None:
        if self.sinks_touched:
            _fail_update(SelectiveUpdateErrorCode.INVALID_PLAN)


@dataclass(frozen=True, slots=True)
class _PreparedUpdate:
    destination: object
    source: object


@dataclass(frozen=True, slots=True)
class _PreparedLayer:
    """Read-only work prepared for one vLLM layer callback."""

    operations: tuple[_PreparedUpdate, ...]
    recomputed_token_rows: int
    write_span_count: int
    dtype: str
    device: str


class GptOssSelectiveKvUpdater:
    """Apply recompute-only K/V rows after a complete read-only preflight."""

    def __init__(self, tensor_ops: SelectiveCacheOps) -> None:
        self._ops = tensor_ops

    def _prepare_layer(
        self,
        *,
        plan: SelectiveWritePlan,
        layer_index: int,
        key: object,
        value: object,
        cache: object,
        slot_mapping: Sequence[object],
        expected_dtype: str | None = None,
        expected_device: str | None = None,
    ) -> _PreparedLayer:
        """Validate and prepare one layer without mutating the cache.

        vLLM 0.19.1 invokes ``AttentionImpl.do_kv_cache_update`` once per
        layer, immediately before that layer's attention call.  Keeping this
        preflight separate from the copy operation lets the normal all-layer
        updater retain its atomic preflight while a future backend can bind a
        worker-local session to those per-layer callbacks.
        """

        try:
            validate_slot_mapping(
                plan,
                layer_index=layer_index,
                slot_mapping=slot_mapping,
            )
        except SelectiveWriteError as error:
            update_code = {
                SelectiveWriteErrorCode.INVALID_SLOT_MAPPING: (
                    SelectiveUpdateErrorCode.INVALID_SLOT_MAPPING
                ),
                SelectiveWriteErrorCode.SLOT_MAPPING_LENGTH_MISMATCH: (
                    SelectiveUpdateErrorCode.SLOT_MAPPING_LENGTH_MISMATCH
                ),
                SelectiveWriteErrorCode.SLOT_MAPPING_VALUE_MISMATCH: (
                    SelectiveUpdateErrorCode.SLOT_MAPPING_VALUE_MISMATCH
                ),
            }.get(
                error.code,
                SelectiveUpdateErrorCode.INVALID_SLOT_MAPPING,
            )
            raise SelectiveUpdateError(update_code) from error

        key_shape = self._safe_shape(key)
        value_shape = self._safe_shape(value)
        expected_source_shape = (
            plan.row_plan.prompt_tokens,
            GPT_OSS_NUM_KV_HEADS,
            GPT_OSS_HEAD_DIM,
        )
        if key_shape != expected_source_shape or value_shape != key_shape:
            _fail_update(SelectiveUpdateErrorCode.INVALID_SOURCE_SHAPE)

        cache_shape = self._safe_shape(cache)
        full_layer_spans = tuple(
            span
            for span in plan.full_layer_spans
            if span.layer_index == layer_index
        )
        layer_spans = plan.spans_for_layer(layer_index)
        block_sizes = {
            span.group_span.block_size for span in full_layer_spans
        }
        if (
            len(block_sizes) != 1
            or len(cache_shape) != 5
            or cache_shape[0] <= 0
            or cache_shape[1] != 2
            or cache_shape[2] != next(iter(block_sizes), -1)
            or cache_shape[3:] != (GPT_OSS_NUM_KV_HEADS, GPT_OSS_HEAD_DIM)
        ):
            _fail_update(SelectiveUpdateErrorCode.INVALID_CACHE_SHAPE)

        key_dtype = self._safe_dtype(key)
        value_dtype = self._safe_dtype(value)
        cache_dtype = self._safe_dtype(cache)
        if key_dtype != value_dtype or key_dtype != cache_dtype:
            _fail_update(SelectiveUpdateErrorCode.DTYPE_MISMATCH)
        if key_dtype != "torch.bfloat16":
            _fail_update(SelectiveUpdateErrorCode.DTYPE_MISMATCH)
        key_device = self._safe_device(key)
        value_device = self._safe_device(value)
        cache_device = self._safe_device(cache)
        if key_device != value_device or key_device != cache_device:
            _fail_update(SelectiveUpdateErrorCode.DEVICE_MISMATCH)
        if not key_device.startswith("cuda:"):
            _fail_update(SelectiveUpdateErrorCode.INVALID_DEVICE)
        if expected_dtype is not None and key_dtype != expected_dtype:
            _fail_update(SelectiveUpdateErrorCode.DTYPE_MISMATCH)
        if expected_device is not None and key_device != expected_device:
            _fail_update(SelectiveUpdateErrorCode.DEVICE_MISMATCH)

        prepared: list[_PreparedUpdate] = []
        for span in layer_spans:
            count = span.token_count
            try:
                key_source = self._ops.prompt_rows(
                    key,
                    start=span.target_range.start,
                    count=count,
                )
                value_source = self._ops.prompt_rows(
                    value,
                    start=span.target_range.start,
                    count=count,
                )
                key_destination = self._ops.paged_rows(
                    cache,
                    component=0,
                    block_id=span.group_span.block_id,
                    block_offset=span.group_span.block_offset,
                    count=count,
                )
                value_destination = self._ops.paged_rows(
                    cache,
                    component=1,
                    block_id=span.group_span.block_id,
                    block_offset=span.group_span.block_offset,
                    count=count,
                )
            except Exception as error:
                raise SelectiveUpdateError(
                    SelectiveUpdateErrorCode.INVALID_VIEW
                ) from error
            expected_shape = (count, GPT_OSS_NUM_KV_HEADS, GPT_OSS_HEAD_DIM)
            for view in (
                key_source,
                value_source,
                key_destination,
                value_destination,
            ):
                if self._safe_shape(view) != expected_shape:
                    _fail_update(SelectiveUpdateErrorCode.INVALID_VIEW)
                if self._safe_dtype(view) != key_dtype:
                    _fail_update(SelectiveUpdateErrorCode.DTYPE_MISMATCH)
                if self._safe_device(view) != key_device:
                    _fail_update(SelectiveUpdateErrorCode.DEVICE_MISMATCH)
            prepared.extend(
                (
                    _PreparedUpdate(key_destination, key_source),
                    _PreparedUpdate(value_destination, value_source),
                )
            )

        return _PreparedLayer(
            operations=tuple(prepared),
            recomputed_token_rows=plan.row_plan.layer(layer_index).recompute_tokens,
            write_span_count=len(layer_spans),
            dtype=key_dtype,
            device=key_device,
        )

    def update(
        self,
        *,
        plan: SelectiveWritePlan,
        key_by_layer: Mapping[str, object],
        value_by_layer: Mapping[str, object],
        paged_caches: Mapping[str, object],
        slot_mapping_by_layer: Mapping[str, Sequence[object]],
    ) -> SelectiveUpdateReceipt:
        """Write only recomputed rows; no operation occurs before all checks.

        ``slot_mapping_by_layer`` is the CPU-readable form of the pinned
        Triton ``slot_mapping`` tensor.  A future backend must provide one
        vector per concrete GPT-OSS layer; omitting it would make the physical
        destination contract unverifiable, so the argument is required even
        when every row is recomputed.
        """

        if not isinstance(plan, SelectiveWritePlan):
            _fail_update(SelectiveUpdateErrorCode.INVALID_PLAN)
        expected_names = {
            f"model.layers.{index}.attn.attn" for index in range(GPT_OSS_NUM_LAYERS)
        }
        if (
            set(key_by_layer) != expected_names
            or set(value_by_layer) != expected_names
            or set(paged_caches) != expected_names
        ):
            _fail_update(SelectiveUpdateErrorCode.TENSOR_SET_MISMATCH)
        if set(slot_mapping_by_layer) != expected_names:
            _fail_update(SelectiveUpdateErrorCode.SLOT_MAPPING_SET_MISMATCH)

        dtype: str | None = None
        device: str | None = None
        prepared: list[_PreparedUpdate] = []
        for layer_index in range(GPT_OSS_NUM_LAYERS):
            layer_name = f"model.layers.{layer_index}.attn.attn"
            key = key_by_layer[layer_name]
            value = value_by_layer[layer_name]
            cache = paged_caches[layer_name]
            layer = self._prepare_layer(
                plan=plan,
                layer_index=layer_index,
                key=key,
                value=value,
                cache=cache,
                slot_mapping=slot_mapping_by_layer[layer_name],
                expected_dtype=dtype,
                expected_device=device,
            )
            if dtype is None:
                dtype = layer.dtype
                device = layer.device
            prepared.extend(layer.operations)

        try:
            for operation in prepared:
                self._ops.copy(operation.destination, operation.source)
            if prepared:
                self._ops.synchronize(next(iter(paged_caches.values())))
        except Exception as error:
            raise SelectiveUpdateError(
                SelectiveUpdateErrorCode.MUTATION_FAILED
            ) from error

        return SelectiveUpdateReceipt(
            recomputed_token_rows=plan.recompute_tokens,
            cached_token_rows=plan.cached_tokens,
            write_span_count=len(plan.recompute_layer_spans),
            copied_key_rows=plan.recompute_tokens,
            copied_value_rows=plan.recompute_tokens,
        )

    def _safe_shape(self, tensor: object) -> tuple[int, ...]:
        try:
            shape = tuple(self._ops.shape(tensor))
        except Exception as error:
            raise SelectiveUpdateError(
                SelectiveUpdateErrorCode.INVALID_VIEW
            ) from error
        if any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 0
            for dimension in shape
        ):
            _fail_update(SelectiveUpdateErrorCode.INVALID_VIEW)
        return shape

    def _safe_dtype(self, tensor: object) -> str:
        try:
            dtype = self._ops.dtype_name(tensor)
        except Exception as error:
            raise SelectiveUpdateError(
                SelectiveUpdateErrorCode.INVALID_VIEW
            ) from error
        if not dtype:
            _fail_update(SelectiveUpdateErrorCode.INVALID_VIEW)
        return dtype

    def _safe_device(self, tensor: object) -> str:
        try:
            device = self._ops.device_name(tensor)
        except Exception as error:
            raise SelectiveUpdateError(
                SelectiveUpdateErrorCode.INVALID_VIEW
            ) from error
        if not device:
            _fail_update(SelectiveUpdateErrorCode.INVALID_VIEW)
        return device


class GptOssSelectiveKvSession:
    """Bind the updater to vLLM's ordered, per-layer cache-update callbacks.

    The pinned Triton implementation calls ``do_kv_cache_update`` for one
    layer at a time immediately before attention.  This session validates and
    writes that layer, then requires all 24 canonical layers before it can
    produce a receipt.  A later validation or copy failure makes the session
    terminal: callers must discard the request's KV cache rather than reuse a
    partially updated request.  This is a dormant adapter contract; the live
    connector remains full-prefill/100%-recompute.
    """

    def __init__(
        self,
        updater: GptOssSelectiveKvUpdater,
        *,
        plan: SelectiveWritePlan,
    ) -> None:
        if not isinstance(updater, GptOssSelectiveKvUpdater):
            _fail_update(SelectiveUpdateErrorCode.INVALID_PLAN)
        if not isinstance(plan, SelectiveWritePlan):
            _fail_update(SelectiveUpdateErrorCode.INVALID_PLAN)
        self._updater = updater
        self._plan = plan
        self._next_layer = 0
        self._failed = False
        self._finished = False
        self._recomputed_token_rows = 0
        self._write_span_count = 0
        self._copied_key_rows = 0
        self._copied_value_rows = 0
        self._session_dtype: str | None = None
        self._session_device: str | None = None

    def update_layer(
        self,
        *,
        layer_index: int,
        key: object,
        value: object,
        paged_cache: object,
        slot_mapping: Sequence[object],
    ) -> None:
        """Validate and update one layer in the pinned forward order."""

        self._ensure_active()
        if not _is_int(layer_index) or not 0 <= layer_index < GPT_OSS_NUM_LAYERS:
            self._terminal(SelectiveUpdateErrorCode.INVALID_PLAN)
        if layer_index < self._next_layer:
            self._terminal(SelectiveUpdateErrorCode.SESSION_DUPLICATE_LAYER)
        if layer_index != self._next_layer:
            self._terminal(SelectiveUpdateErrorCode.SESSION_LAYER_ORDER_MISMATCH)

        try:
            prepared = self._updater._prepare_layer(
                plan=self._plan,
                layer_index=layer_index,
                key=key,
                value=value,
                cache=paged_cache,
                slot_mapping=slot_mapping,
                expected_dtype=self._session_dtype,
                expected_device=self._session_device,
            )
            for operation in prepared.operations:
                self._updater._ops.copy(
                    operation.destination,
                    operation.source,
                )
            if prepared.operations:
                self._updater._ops.synchronize(
                    paged_cache
                )
        except SelectiveUpdateError:
            self._failed = True
            raise
        except Exception as error:
            self._failed = True
            raise SelectiveUpdateError(
                SelectiveUpdateErrorCode.MUTATION_FAILED
            ) from error

        if self._session_dtype is None:
            self._session_dtype = prepared.dtype
            self._session_device = prepared.device
        self._recomputed_token_rows += prepared.recomputed_token_rows
        self._write_span_count += prepared.write_span_count
        self._copied_key_rows += prepared.recomputed_token_rows
        self._copied_value_rows += prepared.recomputed_token_rows
        self._next_layer += 1

    def finish(self) -> SelectiveUpdateReceipt:
        """Return aggregate counters only after every layer completed."""

        self._ensure_active()
        if self._next_layer != GPT_OSS_NUM_LAYERS:
            self._failed = True
            _fail_update(SelectiveUpdateErrorCode.SESSION_INCOMPLETE)
        self._finished = True
        return SelectiveUpdateReceipt(
            recomputed_token_rows=self._recomputed_token_rows,
            cached_token_rows=self._plan.cached_tokens,
            write_span_count=self._write_span_count,
            copied_key_rows=self._copied_key_rows,
            copied_value_rows=self._copied_value_rows,
        )

    def _ensure_active(self) -> None:
        if self._failed or self._finished:
            _fail_update(SelectiveUpdateErrorCode.SESSION_INVALID_STATE)

    def _terminal(self, code: SelectiveUpdateErrorCode) -> NoReturn:
        self._failed = True
        _fail_update(code)

__all__ = [
    "GptOssSelectiveKvSession",
    "GptOssSelectiveKvUpdater",
    "SelectiveCacheOps",
    "SelectiveUpdateError",
    "SelectiveUpdateErrorCode",
    "SelectiveUpdateReceipt",
    "SelectiveWriteError",
    "SelectiveWriteErrorCode",
    "SelectiveWritePlan",
    "plan_selective_kv_writes",
]
