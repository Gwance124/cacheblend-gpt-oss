# SPDX-License-Identifier: Apache-2.0
"""Pinned GPT-OSS/LMCache KV staging data plane.

This module translates dependency-free ``LayerTokenScatterSpan`` values into
copies between the two exact layouts used by the target stack:

* vLLM 0.19.1 Triton KV caches are
  ``[blocks, 2, block_size, num_kv_heads, head_size]`` and GPT-OSS uses eight
  KV heads of width 64:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/attention/backends/triton_attn.py#L293-L316
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/models/gpt_oss.py#L67-L108
* LMCache 0.4.3 ``PlainGPUCacheContext`` requires one contiguous
  ``[2, layers, tokens, width]`` tensor:
  https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/v1/multiprocess/gpu_context.py#L340-L406
* Blend V2 retrieval writes a candidate to ``cur_st + offset``.  Therefore
  staging reads use the span's *target* position plus the explicit retrieval
  offset; the old source position is used only for RoPE correction:
  https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/v1/multiprocess/blend_server_v2.py#L597-L687
* GPT-OSS applies YaRN RoPE to K before attention, while V is unchanged, and
  learned sinks are passed separately to the attention layer:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/models/gpt_oss.py#L82-L153
  Triton likewise keeps sinks outside its KV cache tensor:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/attention/backends/triton_attn.py#L401-L407
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/attention/backends/triton_attn.py#L473-L523

The public operations preflight every layer, tensor shape, dtype, device,
source/target range, staging range, and correction result before the first
destination write.  Attention sinks cannot be passed to this API and are never
read, serialized, or mutated.

Torch is imported only when :func:`load_torch_tensor_ops` is called.  CPU unit
tests use an injected ``TensorOps`` implementation and ordinary Python fakes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from math import isfinite
from time import perf_counter
from typing import Any, NoReturn, Protocol

from cacheblend_gpt_oss.gpt_oss.layout import (
    GPT_OSS_MAX_CONTEXT_TOKENS,
    GPT_OSS_NUM_LAYERS,
    AttentionKind,
    LayerTokenScatterSpan,
)
from cacheblend_gpt_oss.planner.models import TokenRange
from cacheblend_gpt_oss.targets import PINNED_TARGET

GPT_OSS_NUM_KV_HEADS = 8
GPT_OSS_HEAD_DIM = 64
GPT_OSS_KV_WIDTH = GPT_OSS_NUM_KV_HEADS * GPT_OSS_HEAD_DIM
KV_COMPONENTS = 2
KEY_COMPONENT = 0
VALUE_COMPONENT = 1


class DataPlaneErrorCode(str, Enum):
    """Bounded failure reasons for fail-closed data-plane handling."""

    EMPTY_SPANS = "empty_spans"
    LAYER_SET_MISMATCH = "layer_set_mismatch"
    INVALID_LAYER = "invalid_layer"
    INVALID_ATTENTION_PATTERN = "invalid_attention_pattern"
    INVALID_GROUP_LAYOUT = "invalid_group_layout"
    INVALID_SPAN = "invalid_span"
    RANGE_COVERAGE_MISMATCH = "range_coverage_mismatch"
    OVERLAPPING_WRITE = "overlapping_write"
    PAGED_CACHE_SET_MISMATCH = "paged_cache_set_mismatch"
    INVALID_PAGED_CACHE_SHAPE = "invalid_paged_cache_shape"
    INVALID_STAGING_SHAPE = "invalid_staging_shape"
    PAGED_RANGE_OUT_OF_BOUNDS = "paged_range_out_of_bounds"
    STAGING_RANGE_OUT_OF_BOUNDS = "staging_range_out_of_bounds"
    DTYPE_MISMATCH = "dtype_mismatch"
    DEVICE_MISMATCH = "device_mismatch"
    INVALID_DEVICE = "invalid_device"
    POSITION_CORRECTION_FAILED = "position_correction_failed"
    INVALID_CORRECTED_KEY = "invalid_corrected_key"
    TENSOR_VIEW_FAILED = "tensor_view_failed"
    MUTATION_FAILED = "mutation_failed"
    TORCH_DEPENDENCY_MISSING = "torch_dependency_missing"
    TORCH_VERSION_MISMATCH = "torch_version_mismatch"
    TORCH_CUDA_MISMATCH = "torch_cuda_mismatch"


class DataPlaneError(RuntimeError):
    """A fail-closed data-plane error with a stable machine code."""

    def __init__(self, code: DataPlaneErrorCode, message: str = "") -> None:
        self.code = code
        super().__init__(message or code.value)


def _fail(code: DataPlaneErrorCode, message: str = "") -> NoReturn:
    raise DataPlaneError(code, message)


class TransferDirection(str, Enum):
    """Direction of one completed staging operation."""

    LOAD_FROM_STAGING = "load_from_staging"
    STORE_TO_STAGING = "store_to_staging"


class TensorOps(Protocol):
    """Minimal injected tensor surface; view methods must not mutate tensors."""

    def shape(self, tensor: object) -> tuple[int, ...]:
        """Return the tensor shape as plain integers."""

    def dtype_name(self, tensor: object) -> str:
        """Return a stable dtype name such as ``torch.bfloat16``."""

    def device_name(self, tensor: object) -> str:
        """Return a device name such as ``cuda:0``."""

    def paged_rows(
        self,
        tensor: object,
        *,
        component: int,
        block_id: int,
        block_offset: int,
        token_count: int,
    ) -> object:
        """Return a ``[tokens, 8, 64]`` view without mutating the cache."""

    def staging_rows(
        self,
        tensor: object,
        *,
        component: int,
        layer_index: int,
        token_start: int,
        token_count: int,
    ) -> object:
        """Return a ``[tokens, 512]`` view without mutating staging."""

    def reshape(self, tensor: object, shape: tuple[int, ...]) -> object:
        """Return a view with ``shape`` without mutating its storage."""

    def copy(self, destination: object, source: object) -> None:
        """Copy equal-shaped rows into ``destination``."""

    def synchronize(self, tensor: object) -> None:
        """Synchronize data-plane work on the destination tensor's device."""


class KeyPositionCorrector(Protocol):
    """Injected GPT-OSS YaRN correction callback.

    Implementations must not mutate ``key_rows``.  They receive cached absolute
    source positions and requested absolute target positions and return a tensor
    with the same shape, dtype, and device as ``key_rows``.
    """

    def __call__(
        self,
        key_rows: object,
        *,
        source_positions: tuple[int, ...],
        target_positions: tuple[int, ...],
        layer_index: int,
    ) -> object:
        """Return position-corrected K rows without changing V or sinks."""


@dataclass(frozen=True, slots=True)
class DataPlaneReceipt:
    """Counts from one synchronous, fully completed data-plane operation."""

    direction: TransferDirection
    logical_tokens: int
    layer_token_rows: int
    span_count: int
    corrected_key_rows: int
    copied_key_rows: int
    copied_value_rows: int
    sinks_touched: bool = False
    position_correction_latency_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.sinks_touched:
            _fail(DataPlaneErrorCode.INVALID_SPAN, "attention sinks cannot be touched")
        if (
            isinstance(self.position_correction_latency_seconds, bool)
            or not isinstance(self.position_correction_latency_seconds, int | float)
            or not isfinite(self.position_correction_latency_seconds)
            or self.position_correction_latency_seconds < 0
        ):
            _fail(
                DataPlaneErrorCode.INVALID_SPAN,
                "position-correction latency must be finite and nonnegative",
            )


@dataclass(frozen=True, slots=True)
class _ValidatedSpans:
    """Canonical layer-ordered spans and their common logical transfer."""

    spans: tuple[LayerTokenScatterSpan, ...]
    source_range: TokenRange
    target_range: TokenRange
    layer_names: tuple[str, ...]

    @property
    def logical_tokens(self) -> int:
        return len(self.target_range)

    @property
    def layer_token_rows(self) -> int:
        return self.logical_tokens * GPT_OSS_NUM_LAYERS


@dataclass(frozen=True, slots=True)
class _PreparedCopy:
    destination: object
    source: object


class GptOssDataPlane:
    """Synchronous gather/scatter implementation for the single pinned target."""

    def __init__(self, tensor_ops: TensorOps) -> None:
        self._ops = tensor_ops

    def scatter_retrieved_kv(
        self,
        *,
        staging: object,
        paged_caches: Mapping[str, object],
        layer_spans: Sequence[LayerTokenScatterSpan],
        retrieval_buffer_offset: int,
        query_token_count: int,
        correct_key_positions: KeyPositionCorrector,
    ) -> DataPlaneReceipt:
        """Load LMCache candidate rows into paged caches.

        LMCache placed each candidate at
        ``retrieval_buffer_offset + span.target_range.start``.  In particular,
        ``span.source_range`` is never used as a staging address; it supplies
        only the old absolute positions passed to ``correct_key_positions``.
        """

        validated = self._preflight_common(staging, paged_caches, layer_spans)
        _require_plain_int("retrieval_buffer_offset", retrieval_buffer_offset, 0)
        _require_plain_int("query_token_count", query_token_count, 1)
        if query_token_count > GPT_OSS_MAX_CONTEXT_TOKENS:
            _fail(
                DataPlaneErrorCode.STAGING_RANGE_OUT_OF_BOUNDS,
                "query_token_count exceeds the GPT-OSS context limit",
            )
        if validated.target_range.end > query_token_count:
            _fail(
                DataPlaneErrorCode.STAGING_RANGE_OUT_OF_BOUNDS,
                "scatter target exceeds the lookup query",
            )
        staging_shape = self._safe_shape(staging)
        if retrieval_buffer_offset + query_token_count > staging_shape[2]:
            _fail(
                DataPlaneErrorCode.STAGING_RANGE_OUT_OF_BOUNDS,
                "LMCache retrieval placement exceeds staging capacity",
            )

        prepared: list[_PreparedCopy] = []
        correction_latency_seconds = 0.0
        for span in validated.spans:
            staging_start = retrieval_buffer_offset + span.target_range.start
            span_prepared, span_correction_latency = self._prepare_scatter_span(
                staging,
                paged_caches[span.layer_name],
                span,
                staging_start=staging_start,
                correct_key_positions=correct_key_positions,
            )
            prepared.extend(span_prepared)
            correction_latency_seconds += span_correction_latency

        self._apply(prepared, next(iter(paged_caches.values())))
        return DataPlaneReceipt(
            direction=TransferDirection.LOAD_FROM_STAGING,
            logical_tokens=validated.logical_tokens,
            layer_token_rows=validated.layer_token_rows,
            span_count=len(validated.spans),
            corrected_key_rows=validated.layer_token_rows,
            copied_key_rows=validated.layer_token_rows,
            copied_value_rows=validated.layer_token_rows,
            position_correction_latency_seconds=correction_latency_seconds,
        )

    def gather_precomputed_kv(
        self,
        *,
        paged_caches: Mapping[str, object],
        staging: object,
        layer_spans: Sequence[LayerTokenScatterSpan],
        document_target_range: TokenRange,
        store_buffer_offset: int,
    ) -> DataPlaneReceipt:
        """Compact one document's post-RoPE K and ordinary V into staging.

        A precomputed-document store sends only the document tokens to LMCache,
        so staging is compact: target position ``document_target_range.start``
        maps exactly to ``store_buffer_offset``.  No position correction occurs
        while storing; correction is deferred until the document is loaded at a
        new position.
        """

        validated = self._preflight_common(staging, paged_caches, layer_spans)
        if not isinstance(document_target_range, TokenRange):
            _fail(DataPlaneErrorCode.INVALID_SPAN, "invalid document target range")
        if document_target_range != validated.target_range:
            _fail(
                DataPlaneErrorCode.RANGE_COVERAGE_MISMATCH,
                "document range must equal the complete span target coverage",
            )
        _require_plain_int("store_buffer_offset", store_buffer_offset, 0)
        staging_shape = self._safe_shape(staging)
        if store_buffer_offset + len(document_target_range) > staging_shape[2]:
            _fail(
                DataPlaneErrorCode.STAGING_RANGE_OUT_OF_BOUNDS,
                "compact document placement exceeds staging capacity",
            )

        prepared: list[_PreparedCopy] = []
        for span in validated.spans:
            relative_start = span.target_range.start - document_target_range.start
            staging_start = store_buffer_offset + relative_start
            prepared.extend(
                self._prepare_gather_span(
                    paged_caches[span.layer_name],
                    staging,
                    span,
                    staging_start=staging_start,
                )
            )

        self._apply(prepared, staging)
        return DataPlaneReceipt(
            direction=TransferDirection.STORE_TO_STAGING,
            logical_tokens=validated.logical_tokens,
            layer_token_rows=validated.layer_token_rows,
            span_count=len(validated.spans),
            corrected_key_rows=0,
            copied_key_rows=validated.layer_token_rows,
            copied_value_rows=validated.layer_token_rows,
        )

    def _preflight_common(
        self,
        staging: object,
        paged_caches: Mapping[str, object],
        layer_spans: Sequence[LayerTokenScatterSpan],
    ) -> _ValidatedSpans:
        validated = _validate_spans(layer_spans)
        actual_names = set(paged_caches)
        expected_names = set(validated.layer_names)
        if actual_names != expected_names:
            _fail(
                DataPlaneErrorCode.PAGED_CACHE_SET_MISMATCH,
                "paged cache names do not match the 24 validated layers",
            )

        staging_shape = self._safe_shape(staging)
        if (
            len(staging_shape) != 4
            or staging_shape[0] != KV_COMPONENTS
            or staging_shape[1] != GPT_OSS_NUM_LAYERS
            or staging_shape[2] <= 0
            or staging_shape[3] != GPT_OSS_KV_WIDTH
        ):
            _fail(
                DataPlaneErrorCode.INVALID_STAGING_SHAPE,
                "staging must have shape [2, 24, tokens, 512]",
            )
        staging_dtype = self._safe_dtype(staging)
        staging_device = self._safe_device(staging)
        if not staging_device.startswith("cuda:"):
            _fail(
                DataPlaneErrorCode.INVALID_DEVICE,
                "the pinned data plane requires one CUDA device",
            )

        spans_by_layer: dict[str, list[LayerTokenScatterSpan]] = {}
        for span in validated.spans:
            spans_by_layer.setdefault(span.layer_name, []).append(span)
        for layer_name in validated.layer_names:
            cache = paged_caches[layer_name]
            shape = self._safe_shape(cache)
            layer_spans_for_cache = spans_by_layer[layer_name]
            block_sizes = {span.group_span.block_size for span in layer_spans_for_cache}
            if len(block_sizes) != 1:
                _fail(
                    DataPlaneErrorCode.INVALID_GROUP_LAYOUT,
                    "one layer referenced multiple block sizes",
                )
            block_size = next(iter(block_sizes))
            if (
                len(shape) != 5
                or shape[0] <= 0
                or shape[1] != KV_COMPONENTS
                or shape[2] != block_size
                or shape[3] != GPT_OSS_NUM_KV_HEADS
                or shape[4] != GPT_OSS_HEAD_DIM
            ):
                _fail(
                    DataPlaneErrorCode.INVALID_PAGED_CACHE_SHAPE,
                    f"{layer_name} must have shape [blocks,2,block,8,64]",
                )
            if block_size % 16 != 0:
                _fail(
                    DataPlaneErrorCode.INVALID_PAGED_CACHE_SHAPE,
                    "Triton block_size must be a multiple of 16",
                )
            if self._safe_dtype(cache) != staging_dtype:
                _fail(
                    DataPlaneErrorCode.DTYPE_MISMATCH,
                    "paged and staging dtypes differ",
                )
            if self._safe_device(cache) != staging_device:
                _fail(
                    DataPlaneErrorCode.DEVICE_MISMATCH,
                    "paged and staging tensors are on different devices",
                )
            for span in layer_spans_for_cache:
                group_span = span.group_span
                if group_span.block_id >= shape[0]:
                    _fail(
                        DataPlaneErrorCode.PAGED_RANGE_OUT_OF_BOUNDS,
                        "span block_id exceeds the paged cache",
                    )
                if group_span.block_offset + span.token_count > shape[2]:
                    _fail(
                        DataPlaneErrorCode.PAGED_RANGE_OUT_OF_BOUNDS,
                        "span exceeds its paged cache block",
                    )
        return validated

    def _prepare_scatter_span(
        self,
        staging: object,
        paged: object,
        span: LayerTokenScatterSpan,
        *,
        staging_start: int,
        correct_key_positions: KeyPositionCorrector,
    ) -> tuple[tuple[_PreparedCopy, _PreparedCopy], float]:
        token_count = span.token_count
        try:
            staged_key = self._ops.staging_rows(
                staging,
                component=KEY_COMPONENT,
                layer_index=span.layer_index,
                token_start=staging_start,
                token_count=token_count,
            )
            staged_value = self._ops.staging_rows(
                staging,
                component=VALUE_COMPONENT,
                layer_index=span.layer_index,
                token_start=staging_start,
                token_count=token_count,
            )
            key_rows = self._ops.reshape(
                staged_key, (token_count, GPT_OSS_NUM_KV_HEADS, GPT_OSS_HEAD_DIM)
            )
            value_rows = self._ops.reshape(
                staged_value,
                (token_count, GPT_OSS_NUM_KV_HEADS, GPT_OSS_HEAD_DIM),
            )
            key_destination = self._ops.paged_rows(
                paged,
                component=KEY_COMPONENT,
                block_id=span.group_span.block_id,
                block_offset=span.group_span.block_offset,
                token_count=token_count,
            )
            value_destination = self._ops.paged_rows(
                paged,
                component=VALUE_COMPONENT,
                block_id=span.group_span.block_id,
                block_offset=span.group_span.block_offset,
                token_count=token_count,
            )
        except Exception as exc:
            raise DataPlaneError(
                DataPlaneErrorCode.TENSOR_VIEW_FAILED,
                "failed to prepare a scatter tensor view",
            ) from exc

        expected_shape = (token_count, GPT_OSS_NUM_KV_HEADS, GPT_OSS_HEAD_DIM)
        self._validate_view(key_rows, expected_shape, staging)
        self._validate_view(value_rows, expected_shape, staging)
        self._validate_view(key_destination, expected_shape, paged)
        self._validate_view(value_destination, expected_shape, paged)
        source_positions = tuple(range(span.source_range.start, span.source_range.end))
        target_positions = tuple(range(span.target_range.start, span.target_range.end))
        correction_started_at = perf_counter()
        try:
            corrected_key = correct_key_positions(
                key_rows,
                source_positions=source_positions,
                target_positions=target_positions,
                layer_index=span.layer_index,
            )
        except Exception as exc:
            raise DataPlaneError(
                DataPlaneErrorCode.POSITION_CORRECTION_FAILED,
                "GPT-OSS YaRN key correction failed before cache mutation",
            ) from exc
        correction_latency_seconds = perf_counter() - correction_started_at
        try:
            self._validate_view(corrected_key, expected_shape, staging)
        except DataPlaneError as exc:
            raise DataPlaneError(
                DataPlaneErrorCode.INVALID_CORRECTED_KEY,
                "corrected K shape, dtype, or device is invalid",
            ) from exc
        return (
            (
                _PreparedCopy(key_destination, corrected_key),
                _PreparedCopy(value_destination, value_rows),
            ),
            correction_latency_seconds,
        )

    def _prepare_gather_span(
        self,
        paged: object,
        staging: object,
        span: LayerTokenScatterSpan,
        *,
        staging_start: int,
    ) -> tuple[_PreparedCopy, _PreparedCopy]:
        token_count = span.token_count
        try:
            paged_key = self._ops.paged_rows(
                paged,
                component=KEY_COMPONENT,
                block_id=span.group_span.block_id,
                block_offset=span.group_span.block_offset,
                token_count=token_count,
            )
            paged_value = self._ops.paged_rows(
                paged,
                component=VALUE_COMPONENT,
                block_id=span.group_span.block_id,
                block_offset=span.group_span.block_offset,
                token_count=token_count,
            )
            staging_key = self._ops.staging_rows(
                staging,
                component=KEY_COMPONENT,
                layer_index=span.layer_index,
                token_start=staging_start,
                token_count=token_count,
            )
            staging_value = self._ops.staging_rows(
                staging,
                component=VALUE_COMPONENT,
                layer_index=span.layer_index,
                token_start=staging_start,
                token_count=token_count,
            )
            flat_key = self._ops.reshape(paged_key, (token_count, GPT_OSS_KV_WIDTH))
            flat_value = self._ops.reshape(
                paged_value, (token_count, GPT_OSS_KV_WIDTH)
            )
        except Exception as exc:
            raise DataPlaneError(
                DataPlaneErrorCode.TENSOR_VIEW_FAILED,
                "failed to prepare a gather tensor view",
            ) from exc

        paged_shape = (token_count, GPT_OSS_NUM_KV_HEADS, GPT_OSS_HEAD_DIM)
        staging_shape = (token_count, GPT_OSS_KV_WIDTH)
        self._validate_view(paged_key, paged_shape, paged)
        self._validate_view(paged_value, paged_shape, paged)
        self._validate_view(flat_key, staging_shape, paged)
        self._validate_view(flat_value, staging_shape, paged)
        self._validate_view(staging_key, staging_shape, staging)
        self._validate_view(staging_value, staging_shape, staging)
        return (
            _PreparedCopy(staging_key, flat_key),
            _PreparedCopy(staging_value, flat_value),
        )

    def _validate_view(
        self, view: object, expected_shape: tuple[int, ...], owner: object
    ) -> None:
        if self._safe_shape(view) != expected_shape:
            _fail(DataPlaneErrorCode.TENSOR_VIEW_FAILED, "tensor view shape mismatch")
        if self._safe_dtype(view) != self._safe_dtype(owner):
            _fail(DataPlaneErrorCode.DTYPE_MISMATCH, "tensor view dtype mismatch")
        if self._safe_device(view) != self._safe_device(owner):
            _fail(DataPlaneErrorCode.DEVICE_MISMATCH, "tensor view device mismatch")

    def _apply(self, prepared: Sequence[_PreparedCopy], sync_tensor: object) -> None:
        # All shape/range/layer checks and every correction callback completed
        # before this first mutation. A backend copy failure makes the request
        # unusable; the caller must discard/fallback rather than consume partial KV.
        try:
            for operation in prepared:
                self._ops.copy(operation.destination, operation.source)
            self._ops.synchronize(sync_tensor)
        except Exception as exc:
            raise DataPlaneError(
                DataPlaneErrorCode.MUTATION_FAILED,
                "tensor copy or synchronization failed; discard the request KV",
            ) from exc

    def _safe_shape(self, tensor: object) -> tuple[int, ...]:
        try:
            shape = self._ops.shape(tensor)
        except Exception as exc:
            raise DataPlaneError(
                DataPlaneErrorCode.TENSOR_VIEW_FAILED,
                "tensor shape inspection failed",
            ) from exc
        if any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim < 0
            for dim in shape
        ):
            _fail(DataPlaneErrorCode.TENSOR_VIEW_FAILED, "invalid tensor shape")
        return tuple(shape)

    def _safe_dtype(self, tensor: object) -> str:
        try:
            dtype = self._ops.dtype_name(tensor)
        except Exception as exc:
            raise DataPlaneError(
                DataPlaneErrorCode.TENSOR_VIEW_FAILED,
                "tensor dtype inspection failed",
            ) from exc
        if not dtype:
            _fail(DataPlaneErrorCode.TENSOR_VIEW_FAILED, "empty tensor dtype")
        return dtype

    def _safe_device(self, tensor: object) -> str:
        try:
            device = self._ops.device_name(tensor)
        except Exception as exc:
            raise DataPlaneError(
                DataPlaneErrorCode.TENSOR_VIEW_FAILED,
                "tensor device inspection failed",
            ) from exc
        if not device:
            _fail(DataPlaneErrorCode.TENSOR_VIEW_FAILED, "empty tensor device")
        return device


class TorchTensorOps:
    """Production tensor operations over a lazily supplied Torch module."""

    def __init__(self, torch_module: object) -> None:
        self._torch = torch_module

    def _require_tensor(self, tensor: object) -> Any:
        tensor_type = getattr(self._torch, "Tensor", None)
        if tensor_type is None or not isinstance(tensor, tensor_type):
            raise TypeError("expected a torch.Tensor")
        return tensor

    def shape(self, tensor: object) -> tuple[int, ...]:
        value = self._require_tensor(tensor)
        return tuple(int(dimension) for dimension in value.shape)

    def dtype_name(self, tensor: object) -> str:
        return str(self._require_tensor(tensor).dtype)

    def device_name(self, tensor: object) -> str:
        return str(self._require_tensor(tensor).device)

    def paged_rows(
        self,
        tensor: object,
        *,
        component: int,
        block_id: int,
        block_offset: int,
        token_count: int,
    ) -> object:
        value = self._require_tensor(tensor)
        return value[
            block_id,
            component,
            block_offset : block_offset + token_count,
            :,
            :,
        ]

    def staging_rows(
        self,
        tensor: object,
        *,
        component: int,
        layer_index: int,
        token_start: int,
        token_count: int,
    ) -> object:
        value = self._require_tensor(tensor)
        return value[
            component,
            layer_index,
            token_start : token_start + token_count,
            :,
        ]

    def reshape(self, tensor: object, shape: tuple[int, ...]) -> object:
        return self._require_tensor(tensor).reshape(shape)

    def copy(self, destination: object, source: object) -> None:
        destination_tensor = self._require_tensor(destination)
        source_tensor = self._require_tensor(source)
        destination_tensor.copy_(source_tensor, non_blocking=False)

    def synchronize(self, tensor: object) -> None:
        value = self._require_tensor(tensor)
        cuda = getattr(self._torch, "cuda", None)
        synchronize = getattr(cuda, "synchronize", None)
        if not callable(synchronize):
            raise RuntimeError("Torch CUDA synchronization is unavailable")
        synchronize(value.device)


def load_torch_tensor_ops() -> TensorOps:
    """Lazily load the exact pinned Torch/CUDA production implementation."""

    try:
        torch = import_module("torch")
    except ImportError as exc:
        raise DataPlaneError(
            DataPlaneErrorCode.TORCH_DEPENDENCY_MISSING,
            "Torch is not installed; install the pinned GPU runtime extras",
        ) from exc
    observed_version = str(getattr(torch, "__version__", ""))
    if observed_version != PINNED_TARGET.torch_version:
        _fail(
            DataPlaneErrorCode.TORCH_VERSION_MISMATCH,
            f"expected Torch {PINNED_TARGET.torch_version}; got {observed_version!r}",
        )
    observed_cuda = str(getattr(getattr(torch, "version", None), "cuda", ""))
    if observed_cuda != PINNED_TARGET.cuda_runtime:
        _fail(
            DataPlaneErrorCode.TORCH_CUDA_MISMATCH,
            f"expected CUDA runtime {PINNED_TARGET.cuda_runtime}; "
            f"got {observed_cuda!r}",
        )
    return TorchTensorOps(torch)


def _validate_spans(
    layer_spans: Sequence[LayerTokenScatterSpan],
) -> _ValidatedSpans:
    spans = tuple(layer_spans)
    if not spans:
        _fail(DataPlaneErrorCode.EMPTY_SPANS)
    if any(not isinstance(span, LayerTokenScatterSpan) for span in spans):
        _fail(DataPlaneErrorCode.INVALID_SPAN, "non-LayerTokenScatterSpan value")

    by_layer: dict[int, list[LayerTokenScatterSpan]] = {}
    layer_name_by_index: dict[int, str] = {}
    group_kind: dict[int, AttentionKind] = {}
    group_block_size: dict[int, int] = {}
    for span in spans:
        expected_name = f"model.layers.{span.layer_index}.attn.attn"
        if (
            span.layer_name != expected_name
            or not 0 <= span.layer_index < GPT_OSS_NUM_LAYERS
        ):
            _fail(DataPlaneErrorCode.INVALID_LAYER)
        expected_kind = (
            AttentionKind.SLIDING
            if span.layer_index % 2 == 0
            else AttentionKind.FULL
        )
        if span.attention_kind is not expected_kind:
            _fail(DataPlaneErrorCode.INVALID_ATTENTION_PATTERN)
        previous_kind = group_kind.setdefault(span.group_id, span.attention_kind)
        if previous_kind is not span.attention_kind:
            _fail(DataPlaneErrorCode.INVALID_GROUP_LAYOUT)
        previous_block_size = group_block_size.setdefault(
            span.group_id, span.group_span.block_size
        )
        if previous_block_size != span.group_span.block_size:
            _fail(DataPlaneErrorCode.INVALID_GROUP_LAYOUT)
        if (
            span.token_count <= 0
            or len(span.source_range) != span.token_count
            or span.group_span.block_size <= 0
            or span.group_span.block_id < 0
            or span.group_span.block_offset < 0
            or span.group_span.block_offset + span.token_count
            > span.group_span.block_size
            or span.physical_slot_start
            != span.group_span.block_id * span.group_span.block_size
            + span.group_span.block_offset
        ):
            _fail(DataPlaneErrorCode.INVALID_SPAN)
        by_layer.setdefault(span.layer_index, []).append(span)
        layer_name_by_index[span.layer_index] = span.layer_name

    if set(by_layer) != set(range(GPT_OSS_NUM_LAYERS)):
        _fail(DataPlaneErrorCode.LAYER_SET_MISMATCH)
    if set(group_kind) != {0, 1} or set(group_kind.values()) != {
        AttentionKind.SLIDING,
        AttentionKind.FULL,
    }:
        _fail(DataPlaneErrorCode.INVALID_GROUP_LAYOUT)

    canonical_source: TokenRange | None = None
    canonical_target: TokenRange | None = None
    ordered: list[LayerTokenScatterSpan] = []
    for layer_index in range(GPT_OSS_NUM_LAYERS):
        layer_spans_for_index = sorted(
            by_layer[layer_index], key=lambda span: span.target_range.start
        )
        source_start = layer_spans_for_index[0].source_range.start
        target_start = layer_spans_for_index[0].target_range.start
        source_cursor = source_start
        target_cursor = target_start
        physical_ranges: list[TokenRange] = []
        for span in layer_spans_for_index:
            if (
                span.source_range.start != source_cursor
                or span.target_range.start != target_cursor
            ):
                _fail(DataPlaneErrorCode.RANGE_COVERAGE_MISMATCH)
            if span.target_range.start - span.source_range.start != (
                target_start - source_start
            ):
                _fail(DataPlaneErrorCode.RANGE_COVERAGE_MISMATCH)
            physical_range = TokenRange(
                span.physical_slot_start,
                span.physical_slot_start + span.token_count,
            )
            if any(physical_range.overlaps(existing) for existing in physical_ranges):
                _fail(DataPlaneErrorCode.OVERLAPPING_WRITE)
            physical_ranges.append(physical_range)
            source_cursor = span.source_range.end
            target_cursor = span.target_range.end
        layer_source = TokenRange(source_start, source_cursor)
        layer_target = TokenRange(target_start, target_cursor)
        if canonical_source is None:
            canonical_source = layer_source
            canonical_target = layer_target
        elif layer_source != canonical_source or layer_target != canonical_target:
            _fail(DataPlaneErrorCode.RANGE_COVERAGE_MISMATCH)
        ordered.extend(layer_spans_for_index)

    assert canonical_source is not None and canonical_target is not None
    return _ValidatedSpans(
        spans=tuple(ordered),
        source_range=canonical_source,
        target_range=canonical_target,
        layer_names=tuple(layer_name_by_index[index] for index in range(24)),
    )


def _require_plain_int(name: str, value: object, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(DataPlaneErrorCode.INVALID_SPAN, f"{name} must be >= {minimum}")


__all__ = [
    "GPT_OSS_HEAD_DIM",
    "GPT_OSS_KV_WIDTH",
    "GPT_OSS_NUM_KV_HEADS",
    "DataPlaneError",
    "DataPlaneErrorCode",
    "DataPlaneReceipt",
    "GptOssDataPlane",
    "KeyPositionCorrector",
    "TensorOps",
    "TorchTensorOps",
    "TransferDirection",
    "load_torch_tensor_ops",
]
