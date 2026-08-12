# SPDX-License-Identifier: Apache-2.0
"""CPU-only ordering contract for a future GPT-OSS selective backend.

The pinned vLLM 0.19.1 split-update path calls the attention implementation's
``do_kv_cache_update`` before the decorated attention operation.  A selective
backend must update only the rows selected for recomputation, then invoke
attention with the same learned GPT-OSS sink object; a connector wait that runs
inside the decorated operation is too late for overlapping cached rows.

Pinned source evidence:

* ``Attention.forward`` calls ``unified_kv_cache_update`` before attention:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/layers/attention/attention.py#L452-L500
* ``TritonAttentionImpl.do_kv_cache_update`` receives one layer's K/V, paged
  cache, and flattened slot mapping:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/attention/backends/triton_attn.py#L575-L606
* GPT-OSS passes learned sinks into its attention layer:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/models/gpt_oss.py#L67-L153

This bridge is deliberately dependency-injected and does not import vLLM,
Torch, or CUDA.  It is a dormant CPU contract: the live connector still
recomputes 100% of the prompt, and no custom backend is registered.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import NoReturn, Protocol

from cacheblend_gpt_oss.gpt_oss.selective_kv import (
    SelectiveUpdateError,
    SelectiveUpdateReceipt,
)

GPT_OSS_NUM_LAYERS = 24


class SelectiveAttentionErrorCode(str, Enum):
    """Bounded failures for the future per-layer attention adapter."""

    INVALID_SESSION = "invalid_session"
    INVALID_LAYER = "invalid_layer"
    LAYER_ORDER_MISMATCH = "layer_order_mismatch"
    SINK_REQUIRED = "sink_required"
    INVALID_ATTENTION = "invalid_attention"
    UPDATE_FAILED = "update_failed"
    ATTENTION_FAILED = "attention_failed"
    INCOMPLETE = "incomplete"


class SelectiveAttentionError(RuntimeError):
    """Fail-closed ordering error; request KV must be discarded."""

    def __init__(self, code: SelectiveAttentionErrorCode) -> None:
        self.code = code
        super().__init__(f"GPT-OSS selective attention failure: {code.value}")


def _fail(code: SelectiveAttentionErrorCode) -> NoReturn:
    raise SelectiveAttentionError(code)


class SelectiveLayerSession(Protocol):
    """Per-layer KV session consumed by the ordering bridge."""

    def update_layer(
        self,
        *,
        layer_index: int,
        key: object,
        value: object,
        paged_cache: object,
        slot_mapping: Sequence[object],
    ) -> None:
        """Write only recomputed rows for one layer."""

    def finish(self) -> SelectiveUpdateReceipt:
        """Finalize after all canonical layers have run."""


class SelectiveAttentionCall(Protocol):
    """Attention callback with the pinned layer/sink data boundary."""

    def __call__(
        self,
        *,
        layer_index: int,
        query: object,
        key: object,
        value: object,
        kv_cache: object,
        attn_metadata: object,
        sinks: object,
    ) -> object:
        """Run one layer's attention without modifying ``sinks``."""


class SelectiveAttentionBridge:
    """Enforce update-before-attention ordering for all 24 GPT-OSS layers.

    The bridge marks itself terminal after any update or attention failure.
    Earlier cache writes may already be visible when attention fails, so the
    caller must discard the request KV rather than retrying or publishing it.
    """

    def __init__(self, session: SelectiveLayerSession) -> None:
        if not callable(getattr(session, "update_layer", None)) or not callable(
            getattr(session, "finish", None)
        ):
            _fail(SelectiveAttentionErrorCode.INVALID_SESSION)
        self._session = session
        self._next_layer = 0
        self._failed = False
        self._finished = False

    def run_layer(
        self,
        *,
        layer_index: int,
        query: object,
        key: object,
        value: object,
        kv_cache: object,
        slot_mapping: Sequence[object],
        attn_metadata: object,
        sinks: object,
        attention: SelectiveAttentionCall,
    ) -> object:
        """Update selected rows, then invoke attention with the same sinks."""

        self._ensure_active()
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or not 0 <= layer_index < GPT_OSS_NUM_LAYERS
        ):
            self._terminal(SelectiveAttentionErrorCode.INVALID_LAYER)
        if layer_index != self._next_layer:
            self._terminal(SelectiveAttentionErrorCode.LAYER_ORDER_MISMATCH)
        if sinks is None:
            self._terminal(SelectiveAttentionErrorCode.SINK_REQUIRED)
        if not callable(attention):
            self._terminal(SelectiveAttentionErrorCode.INVALID_ATTENTION)

        try:
            self._session.update_layer(
                layer_index=layer_index,
                key=key,
                value=value,
                paged_cache=kv_cache,
                slot_mapping=slot_mapping,
            )
        except SelectiveUpdateError as error:
            self._failed = True
            raise SelectiveAttentionError(
                SelectiveAttentionErrorCode.UPDATE_FAILED
            ) from error
        except Exception as error:
            self._failed = True
            raise SelectiveAttentionError(
                SelectiveAttentionErrorCode.UPDATE_FAILED
            ) from error

        try:
            output = attention(
                layer_index=layer_index,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                sinks=sinks,
            )
        except Exception as error:
            self._failed = True
            raise SelectiveAttentionError(
                SelectiveAttentionErrorCode.ATTENTION_FAILED
            ) from error
        self._next_layer += 1
        return output

    def finish(self) -> SelectiveUpdateReceipt:
        """Finalize only after all layer attention calls completed."""

        self._ensure_active()
        if self._next_layer != GPT_OSS_NUM_LAYERS:
            self._failed = True
            _fail(SelectiveAttentionErrorCode.INCOMPLETE)
        try:
            receipt = self._session.finish()
        except SelectiveUpdateError as error:
            self._failed = True
            raise SelectiveAttentionError(
                SelectiveAttentionErrorCode.UPDATE_FAILED
            ) from error
        except Exception as error:
            self._failed = True
            raise SelectiveAttentionError(
                SelectiveAttentionErrorCode.UPDATE_FAILED
            ) from error
        self._finished = True
        return receipt

    def _ensure_active(self) -> None:
        if self._failed or self._finished:
            _fail(SelectiveAttentionErrorCode.INVALID_SESSION)

    def _terminal(self, code: SelectiveAttentionErrorCode) -> NoReturn:
        self._failed = True
        _fail(code)


__all__ = [
    "GPT_OSS_NUM_LAYERS",
    "SelectiveAttentionBridge",
    "SelectiveAttentionCall",
    "SelectiveAttentionError",
    "SelectiveAttentionErrorCode",
    "SelectiveLayerSession",
]
