# SPDX-License-Identifier: Apache-2.0
"""Dependency-free translations from pinned vLLM 0.19.1 objects.

This module uses structural inspection rather than importing vLLM or Torch, so
its fail-closed translations can be tested on a CPU-only workstation.  The
accepted object shapes are intentionally tied to vLLM commit
``b1388b1fbf5aaef47937fabe98931211684666a6``:

* ``Request`` stores token prompts, optional prompt embeddings, and the prompt
  length:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/request.py#L40-L125
* ``FullAttentionSpec``, ``SlidingWindowSpec``, and their head/block fields:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/kv_cache_interface.py#L21-L193
* finalized ``KVCacheGroupSpec`` and ``KVCacheConfig`` objects:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/kv_cache_interface.py#L461-L490
* ``KVCacheBlocks.get_block_ids`` returns one ordered block-ID list per cache
  group:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/core/kv_cache_manager.py#L22-L80

Only the audited GPT-OSS-20B, TP=1 hybrid layout is accepted.  These helpers
are translations for a version connector, not generic vLLM adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NoReturn, TypeGuard, cast

from cacheblend_gpt_oss.connector.control_plane import (
    CacheGroupLayout as ControlPlaneCacheGroupLayout,
)
from cacheblend_gpt_oss.connector.control_plane import (
    ControlPlaneError,
    GroupedBlockAllocation,
)
from cacheblend_gpt_oss.gpt_oss.layout import (
    AttentionKind,
    GptOssHybridCacheLayout,
    GroupBlockTable,
    HybridLayoutError,
    extract_gpt_oss_layer_index,
)
from cacheblend_gpt_oss.gpt_oss.layout import (
    CacheGroupLayout as GptOssCacheGroupLayout,
)
from cacheblend_gpt_oss.planner.models import normalize_token_ids

GPT_OSS_NUM_LAYERS = 24
GPT_OSS_NUM_CACHE_GROUPS = 2
GPT_OSS_KV_HEADS = 8
GPT_OSS_HEAD_SIZE = 64
GPT_OSS_BLOCK_SIZE = 16
GPT_OSS_SLIDING_WINDOW = 128

_MISSING = object()


class VllmAdapterErrorCode(str, Enum):
    """Stable failure codes safe for bounded logs and metric labels."""

    INVALID_KV_CACHE_CONFIG = "invalid_kv_cache_config"
    INVALID_NUM_BLOCKS = "invalid_num_blocks"
    CACHE_GROUP_COUNT_MISMATCH = "cache_group_count_mismatch"
    SPEC_TYPE_MISMATCH = "spec_type_mismatch"
    BLOCK_SIZE_MISMATCH = "block_size_mismatch"
    KV_HEAD_COUNT_MISMATCH = "kv_head_count_mismatch"
    HEAD_SIZE_MISMATCH = "head_size_mismatch"
    VALUE_HEAD_SIZE_MISMATCH = "value_head_size_mismatch"
    SLIDING_WINDOW_MISMATCH = "sliding_window_mismatch"
    ATTENTION_CHUNKING_UNSUPPORTED = "attention_chunking_unsupported"
    INVALID_LAYER_NAMES = "invalid_layer_names"
    PROMPT_EMBEDS_UNSUPPORTED = "prompt_embeds_unsupported"
    PROMPT_TOKEN_IDS_MISSING = "prompt_token_ids_missing"
    INVALID_PROMPT_TOKEN_IDS = "invalid_prompt_token_ids"
    PROMPT_TOKEN_COUNT_MISMATCH = "prompt_token_count_mismatch"
    BLOCK_ID_METHOD_MISSING = "block_id_method_missing"
    BLOCK_ID_EXTRACTION_FAILED = "block_id_extraction_failed"
    BLOCK_GROUP_COUNT_MISMATCH = "block_group_count_mismatch"
    INVALID_BLOCK_IDS = "invalid_block_ids"
    BLOCK_ID_OUT_OF_RANGE = "block_id_out_of_range"
    NULL_BLOCK_UNSUPPORTED = "null_block_unsupported"


class VllmAdapterError(ValueError):
    """Fail-closed translation error containing no request-specific values."""

    def __init__(self, code: VllmAdapterErrorCode) -> None:
        self.code = code
        super().__init__(f"vLLM 0.19.1 adapter failure: {code.value}")


def _fail(code: VllmAdapterErrorCode) -> NoReturn:
    raise VllmAdapterError(code)


def _attribute(value: object, name: str) -> object:
    return cast(object, getattr(value, name, _MISSING))


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_exact_int(
    value: object,
    expected: int,
    code: VllmAdapterErrorCode,
) -> None:
    if not _is_int(value) or value != expected:
        _fail(code)


def _canonical_layer_names(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        _fail(VllmAdapterErrorCode.INVALID_LAYER_NAMES)
    normalized_names: list[str] = []
    for name in value:
        if not isinstance(name, str):
            _fail(VllmAdapterErrorCode.INVALID_LAYER_NAMES)
        normalized_names.append(name)
    layer_names = tuple(normalized_names)
    try:
        indexed_names = tuple(
            (extract_gpt_oss_layer_index(name), name) for name in layer_names
        )
    except HybridLayoutError:
        _fail(VllmAdapterErrorCode.INVALID_LAYER_NAMES)
    indexes = tuple(index for index, _ in indexed_names)
    if len(indexes) != len(set(indexes)):
        _fail(VllmAdapterErrorCode.INVALID_LAYER_NAMES)
    return tuple(name for _, name in sorted(indexed_names))


@dataclass(frozen=True, slots=True)
class AdaptedKvCacheConfig:
    """Validated independent layouts derived from one finalized config."""

    num_blocks: int
    gpt_oss_layout: GptOssHybridCacheLayout
    control_plane_layout: ControlPlaneCacheGroupLayout


@dataclass(frozen=True, slots=True)
class AdaptedKvCacheBlocks:
    """Immutable views of one request's grouped vLLM block allocation."""

    grouped_allocation: GroupedBlockAllocation
    group_block_tables: tuple[GroupBlockTable, ...]

    @property
    def block_ids_by_group(self) -> tuple[tuple[int, ...], ...]:
        """Return IDs in exact finalized-config group and logical-block order."""

        return self.grouped_allocation.block_ids_by_group


def adapt_kv_cache_config(kv_cache_config: object) -> AdaptedKvCacheConfig:
    """Validate and translate the exact GPT-OSS finalized KV-cache config."""

    num_blocks = _attribute(kv_cache_config, "num_blocks")
    if not _is_int(num_blocks) or num_blocks < 1:
        _fail(VllmAdapterErrorCode.INVALID_NUM_BLOCKS)

    raw_groups = _attribute(kv_cache_config, "kv_cache_groups")
    if not isinstance(raw_groups, list):
        _fail(VllmAdapterErrorCode.INVALID_KV_CACHE_CONFIG)
    groups = tuple(raw_groups)
    if len(groups) != GPT_OSS_NUM_CACHE_GROUPS:
        _fail(VllmAdapterErrorCode.CACHE_GROUP_COUNT_MISMATCH)

    gpt_oss_groups: list[GptOssCacheGroupLayout] = []
    all_indexes: set[int] = set()
    seen_kinds: set[AttentionKind] = set()
    for group_id, group in enumerate(groups):
        spec = _attribute(group, "kv_cache_spec")
        if spec is _MISSING:
            _fail(VllmAdapterErrorCode.INVALID_KV_CACHE_CONFIG)
        spec_type = type(spec).__name__
        if spec_type == "SlidingWindowSpec":
            attention_kind = AttentionKind.SLIDING
            expected_window: int | None = GPT_OSS_SLIDING_WINDOW
        elif spec_type == "FullAttentionSpec":
            attention_kind = AttentionKind.FULL
            expected_window = None
        else:
            _fail(VllmAdapterErrorCode.SPEC_TYPE_MISMATCH)

        _require_exact_int(
            _attribute(spec, "block_size"),
            GPT_OSS_BLOCK_SIZE,
            VllmAdapterErrorCode.BLOCK_SIZE_MISMATCH,
        )
        _require_exact_int(
            _attribute(spec, "num_kv_heads"),
            GPT_OSS_KV_HEADS,
            VllmAdapterErrorCode.KV_HEAD_COUNT_MISMATCH,
        )
        _require_exact_int(
            _attribute(spec, "head_size"),
            GPT_OSS_HEAD_SIZE,
            VllmAdapterErrorCode.HEAD_SIZE_MISMATCH,
        )
        if attention_kind is AttentionKind.FULL:
            _require_exact_int(
                _attribute(spec, "head_size_v"),
                GPT_OSS_HEAD_SIZE,
                VllmAdapterErrorCode.VALUE_HEAD_SIZE_MISMATCH,
            )
            if _attribute(spec, "attention_chunk_size") is not None:
                _fail(VllmAdapterErrorCode.ATTENTION_CHUNKING_UNSUPPORTED)
        if _attribute(spec, "sliding_window") != expected_window:
            _fail(VllmAdapterErrorCode.SLIDING_WINDOW_MISMATCH)

        layer_names = _canonical_layer_names(_attribute(group, "layer_names"))
        indexes = tuple(extract_gpt_oss_layer_index(name) for name in layer_names)
        if all_indexes.intersection(indexes):
            _fail(VllmAdapterErrorCode.INVALID_LAYER_NAMES)
        all_indexes.update(indexes)
        expected_parity = 0 if attention_kind is AttentionKind.SLIDING else 1
        if any(index % 2 != expected_parity for index in indexes):
            _fail(VllmAdapterErrorCode.INVALID_LAYER_NAMES)
        seen_kinds.add(attention_kind)
        gpt_oss_groups.append(
            GptOssCacheGroupLayout(
                group_id=group_id,
                attention_kind=attention_kind,
                layer_names=layer_names,
                block_size=GPT_OSS_BLOCK_SIZE,
                sliding_window=expected_window,
            )
        )

    if all_indexes != set(range(GPT_OSS_NUM_LAYERS)) or seen_kinds != {
        AttentionKind.SLIDING,
        AttentionKind.FULL,
    }:
        _fail(VllmAdapterErrorCode.INVALID_LAYER_NAMES)

    try:
        gpt_oss_layout = GptOssHybridCacheLayout(tuple(gpt_oss_groups))
        control_plane_layout = ControlPlaneCacheGroupLayout(
            tuple(group.layer_names for group in gpt_oss_groups)
        )
    except (HybridLayoutError, ControlPlaneError):
        _fail(VllmAdapterErrorCode.INVALID_LAYER_NAMES)

    return AdaptedKvCacheConfig(
        num_blocks=num_blocks,
        gpt_oss_layout=gpt_oss_layout,
        control_plane_layout=control_plane_layout,
    )


def copy_request_prompt_token_ids(request: object) -> tuple[int, ...]:
    """Copy exact prompt IDs and reject the unaudited prompt-embeddings path."""

    prompt_embeds = _attribute(request, "prompt_embeds")
    if prompt_embeds is not _MISSING and prompt_embeds is not None:
        _fail(VllmAdapterErrorCode.PROMPT_EMBEDS_UNSUPPORTED)

    raw_token_ids = _attribute(request, "prompt_token_ids")
    if raw_token_ids is _MISSING or raw_token_ids is None:
        _fail(VllmAdapterErrorCode.PROMPT_TOKEN_IDS_MISSING)
    if not isinstance(raw_token_ids, list):
        _fail(VllmAdapterErrorCode.INVALID_PROMPT_TOKEN_IDS)
    try:
        token_ids = normalize_token_ids(raw_token_ids)
    except (TypeError, ValueError):
        _fail(VllmAdapterErrorCode.INVALID_PROMPT_TOKEN_IDS)

    num_prompt_tokens = _attribute(request, "num_prompt_tokens")
    if (
        not _is_int(num_prompt_tokens)
        or num_prompt_tokens != len(token_ids)
    ):
        _fail(VllmAdapterErrorCode.PROMPT_TOKEN_COUNT_MISMATCH)
    return token_ids


def _copy_block_ids(
    kv_cache_blocks: object,
    config: AdaptedKvCacheConfig,
) -> tuple[tuple[int, ...], ...]:
    get_block_ids = _attribute(kv_cache_blocks, "get_block_ids")
    if not callable(get_block_ids):
        _fail(VllmAdapterErrorCode.BLOCK_ID_METHOD_MISSING)
    try:
        raw_block_ids = cast(object, get_block_ids())
    except Exception as exc:
        raise VllmAdapterError(
            VllmAdapterErrorCode.BLOCK_ID_EXTRACTION_FAILED
        ) from exc

    # The pinned method returns tuple[list[int], ...].  Rejecting alternate
    # containers catches API drift before group identities can be misbound.
    if not isinstance(raw_block_ids, tuple):
        _fail(VllmAdapterErrorCode.BLOCK_ID_EXTRACTION_FAILED)
    if len(raw_block_ids) != config.control_plane_layout.group_count:
        _fail(VllmAdapterErrorCode.BLOCK_GROUP_COUNT_MISMATCH)

    copied_groups: list[tuple[int, ...]] = []
    for raw_group in raw_block_ids:
        if not isinstance(raw_group, list):
            _fail(VllmAdapterErrorCode.INVALID_BLOCK_IDS)
        normalized_group: list[int] = []
        for block_id in raw_group:
            if not _is_int(block_id) or block_id < 0:
                _fail(VllmAdapterErrorCode.INVALID_BLOCK_IDS)
            normalized_group.append(block_id)
        group = tuple(normalized_group)
        if any(block_id >= config.num_blocks for block_id in group):
            _fail(VllmAdapterErrorCode.BLOCK_ID_OUT_OF_RANGE)
        copied_groups.append(group)
    return tuple(copied_groups)


def _reject_observable_null_blocks(
    kv_cache_blocks: object,
    block_ids_by_group: tuple[tuple[int, ...], ...],
) -> None:
    """Reject pinned null blocks when raw block objects are observable.

    ``get_block_ids`` does not retain ``KVCacheBlock.is_null``.  Exact runtime
    ``KVCacheBlocks`` also exposes the grouped block objects, so inspect that
    information before treating an ID as a writable destination.  Structural
    CPU fakes may omit ``blocks``; the real pinned object may not.
    """

    raw_groups = _attribute(kv_cache_blocks, "blocks")
    if raw_groups is _MISSING:
        return
    if not isinstance(raw_groups, tuple) or len(raw_groups) != len(
        block_ids_by_group
    ):
        _fail(VllmAdapterErrorCode.BLOCK_GROUP_COUNT_MISMATCH)
    for raw_group, copied_ids in zip(
        raw_groups, block_ids_by_group, strict=True
    ):
        if not isinstance(raw_group, list | tuple):
            _fail(VllmAdapterErrorCode.INVALID_BLOCK_IDS)
        blocks = tuple(raw_group)
        if len(blocks) != len(copied_ids):
            _fail(VllmAdapterErrorCode.INVALID_BLOCK_IDS)
        for block, copied_id in zip(blocks, copied_ids, strict=True):
            raw_block_id = _attribute(block, "block_id")
            if not _is_int(raw_block_id) or raw_block_id != copied_id:
                _fail(VllmAdapterErrorCode.INVALID_BLOCK_IDS)
            is_null = _attribute(block, "is_null")
            if not isinstance(is_null, bool):
                _fail(VllmAdapterErrorCode.INVALID_BLOCK_IDS)
            if is_null:
                _fail(VllmAdapterErrorCode.NULL_BLOCK_UNSUPPORTED)


def adapt_kv_cache_blocks(
    kv_cache_blocks: object,
    config: AdaptedKvCacheConfig,
    *,
    allow_null_blocks: bool = False,
) -> AdaptedKvCacheBlocks:
    """Copy one grouped allocation into control-plane and scatter descriptors."""

    block_ids_by_group = _copy_block_ids(kv_cache_blocks, config)
    if not allow_null_blocks:
        _reject_observable_null_blocks(kv_cache_blocks, block_ids_by_group)
    if any(
        len(group) != len(set(group)) for group in block_ids_by_group
    ):
        _fail(VllmAdapterErrorCode.INVALID_BLOCK_IDS)
    try:
        grouped_allocation = GroupedBlockAllocation.capture(
            config.control_plane_layout,
            block_ids_by_group,
        )
        group_block_tables = tuple(
            GroupBlockTable(
                group_id=group.group_id,
                block_size=group.block_size,
                block_ids=block_ids_by_group[group.group_id],
            )
            for group in config.gpt_oss_layout.groups
        )
    except (ControlPlaneError, HybridLayoutError):
        _fail(VllmAdapterErrorCode.INVALID_BLOCK_IDS)
    return AdaptedKvCacheBlocks(
        grouped_allocation=grouped_allocation,
        group_block_tables=group_block_tables,
    )


__all__ = [
    "GPT_OSS_BLOCK_SIZE",
    "GPT_OSS_HEAD_SIZE",
    "GPT_OSS_KV_HEADS",
    "GPT_OSS_NUM_CACHE_GROUPS",
    "GPT_OSS_NUM_LAYERS",
    "GPT_OSS_SLIDING_WINDOW",
    "AdaptedKvCacheBlocks",
    "AdaptedKvCacheConfig",
    "VllmAdapterError",
    "VllmAdapterErrorCode",
    "adapt_kv_cache_blocks",
    "adapt_kv_cache_config",
    "copy_request_prompt_token_ids",
]
