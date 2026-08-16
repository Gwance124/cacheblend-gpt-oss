# SPDX-License-Identifier: Apache-2.0
"""Pinned vLLM 0.19.1 CUSTOM attention backend adapter.

This module is loaded only by an explicitly enabled vLLM plugin on the pinned
GPU environment.  With no active :class:`ForwardRowPlanContext`, it delegates
to the stock sink-capable Triton implementation byte-for-byte at the Python
boundary.  With a plan, it narrows only the KV-cache write rows selected for
that layer.  It does not yet skip GPT-OSS hidden-state or attention compute;
that remains the next model-override milestone and this adapter must not be
described as a speedup by itself.

Pinned vLLM source contracts:

* ``TritonAttentionBackend`` exposes the sink-capable paged-cache backend and
  ``forward_includes_kv_cache_update=False``:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/attention/backends/triton_attn.py#L257-L355
* ``TritonAttentionImpl.do_kv_cache_update`` receives the layer, token-row K/V,
  paged cache, and slot mapping:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/attention/backends/triton_attn.py#L575-L606
* the generic attention layer invokes this cache update before attention:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/layers/attention/attention.py#L452-L500
"""

from __future__ import annotations

from typing import Any

from cacheblend_gpt_oss.gpt_oss.layout import extract_gpt_oss_layer_index
from cacheblend_gpt_oss.gpt_oss.selective import (
    ForwardRowPlanContext,
    SelectiveForwardState,
)

try:
    import torch  # type: ignore[import-not-found]
    from vllm.v1.attention.backends.triton_attn import (  # type: ignore[import-not-found]
        TritonAttentionBackend,
        TritonAttentionImpl,
    )
except ImportError as error:  # pragma: no cover - exercised on GPU only
    raise RuntimeError(
        "CacheBlend CUSTOM backend requires the pinned vLLM 0.19.1 GPU runtime"
    ) from error


BACKEND_CLASS_PATH = (
    "cacheblend_gpt_oss.vllm_compat.v0_19_1.selective_backend."
    "GptOssCacheBlendAttentionBackend"
)


def _selected_positions(plan: Any, layer_index: int, device: Any) -> Any:
    """Build a device-local row index from the bounded immutable plan."""

    ranges = plan.layer(layer_index).recompute_ranges
    if not ranges:
        return torch.empty(0, dtype=torch.long, device=device)
    chunks = [
        torch.arange(item.start, item.end, dtype=torch.long, device=device)
        for item in ranges
    ]
    return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


def _read_cached_values(
    value_cache: torch.Tensor, flat_slots: torch.Tensor
) -> torch.Tensor:
    """Read the logical NHD view before the stock write overwrites it.

    vLLM passes ``kv_cache.unbind(1)`` to this method.  The resulting view is
    four-dimensional with ``[block, offset, head, dim]`` indexing even when
    the underlying allocation uses the HND stride order; indexing the view
    therefore preserves both pinned layouts.
    """

    if value_cache.dtype == torch.uint8 or value_cache.ndim != 4:
        raise RuntimeError(
            "CacheBlend check-layer scoring requires an unquantized NHD/HND KV cache"
        )
    block_size = value_cache.shape[1]
    if not isinstance(block_size, int) or block_size <= 0:
        raise RuntimeError("CacheBlend received an invalid KV-cache block size")
    if flat_slots.numel() and (
        bool(torch.any(flat_slots < 0).item())
        or bool(torch.any(flat_slots // block_size >= value_cache.shape[0]).item())
    ):
        raise RuntimeError("CacheBlend check-layer slots are outside the KV cache")
    block_ids = torch.div(flat_slots, block_size, rounding_mode="floor")
    offsets = flat_slots.remainder(block_size)
    return value_cache[block_ids, offsets]


def _measure_check_layer(
    state: SelectiveForwardState,
    layer_index: int,
    value: torch.Tensor,
    value_cache: torch.Tensor,
    flat_slots: torch.Tensor,
) -> None:
    """Produce audited CacheBlend value-difference importance scores once."""

    if layer_index != state.check_layer or state.scored:
        return
    old_value = _read_cached_values(value_cache, flat_slots)
    if old_value.shape != value.shape:
        raise RuntimeError("CacheBlend check-layer KV shapes do not match")
    scores = (value.float() - old_value.float()).square().sum(dim=(1, 2))
    state.update_importance_scores(scores.detach().cpu().tolist())


class GptOssCacheBlendAttentionBackend(TritonAttentionBackend):  # type: ignore[misc]
    """Sink-capable Triton backend with an opt-in selective KV write seam."""

    @staticmethod
    def get_name() -> str:
        # vLLM's pinned Attention constructor indexes AttentionBackendEnum by
        # this exact string after resolving the CUSTOM class path.
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[GptOssCacheBlendAttentionImpl]:
        return GptOssCacheBlendAttentionImpl


class GptOssCacheBlendAttentionImpl(TritonAttentionImpl):  # type: ignore[misc]
    """Use stock Triton writes unless a validated worker plan is bound."""

    def do_kv_cache_update(
        self,
        layer: Any,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        state = ForwardRowPlanContext.current_state()
        plan = ForwardRowPlanContext.current_or_none()
        if plan is None:
            super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
            return

        layer_name = getattr(layer, "layer_name", None)
        if not isinstance(layer_name, str):
            raise RuntimeError(
                "CacheBlend selective backend received an invalid GPT-OSS layer"
            )
        try:
            layer_index = extract_gpt_oss_layer_index(layer_name)
        except Exception as error:
            raise RuntimeError(
                "CacheBlend selective backend received an invalid GPT-OSS layer"
            ) from error

        if (
            key.ndim != 3
            or value.ndim != 3
            or key.shape[0] != plan.prompt_tokens
            or value.shape[0] != plan.prompt_tokens
        ):
            raise RuntimeError(
                "CacheBlend selective backend received incompatible KV rows"
            )
        flat_slots = slot_mapping.reshape(-1)
        if flat_slots.numel() != plan.prompt_tokens:
            raise RuntimeError(
                "CacheBlend selective backend received incompatible slot rows"
            )

        key_cache, value_cache = kv_cache.unbind(1)
        if state is not None:
            _measure_check_layer(
                state,
                layer_index,
                value,
                value_cache,
                flat_slots,
            )
            # The check-layer write remains full; later layers see the measured
            # plan through ForwardRowPlanContext.current_or_none().
            plan = ForwardRowPlanContext.current_or_none()
            if plan is None:
                raise RuntimeError("CacheBlend selective plan disappeared mid-forward")

        positions = _selected_positions(plan, layer_index, key.device)
        if positions.numel() == key.shape[0]:
            super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
            return
        if positions.numel() == 0:
            return

        super().do_kv_cache_update(
            layer,
            key.index_select(0, positions),
            value.index_select(0, positions),
            kv_cache,
            flat_slots.index_select(0, positions),
        )


__all__ = [
    "BACKEND_CLASS_PATH",
    "GptOssCacheBlendAttentionBackend",
    "GptOssCacheBlendAttentionImpl",
]
