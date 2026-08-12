from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from cacheblend_gpt_oss.gpt_oss.selective_attention import (
    SelectiveAttentionBridge,
    SelectiveAttentionError,
    SelectiveAttentionErrorCode,
)
from cacheblend_gpt_oss.gpt_oss.selective_kv import (
    SelectiveUpdateError,
    SelectiveUpdateErrorCode,
    SelectiveUpdateReceipt,
)


@dataclass
class RecordingSession:
    updates: list[int]
    finish_calls: int = 0
    fail_layer: int | None = None

    def update_layer(
        self,
        *,
        layer_index: int,
        key: object,
        value: object,
        paged_cache: object,
        slot_mapping: tuple[object, ...],
    ) -> None:
        del key, value, paged_cache, slot_mapping
        if self.fail_layer == layer_index:
            raise SelectiveUpdateError(SelectiveUpdateErrorCode.INVALID_PLAN)
        self.updates.append(layer_index)

    def finish(self) -> SelectiveUpdateReceipt:
        self.finish_calls += 1
        return SelectiveUpdateReceipt(
            recomputed_token_rows=24,
            cached_token_rows=24,
            write_span_count=24,
            copied_key_rows=24,
            copied_value_rows=24,
        )


def _attention_events() -> tuple[list[tuple[int, object]], Any]:
    events: list[tuple[int, object]] = []

    def attention(
        *,
        layer_index: int,
        query: object,
        key: object,
        value: object,
        kv_cache: object,
        attn_metadata: object,
        sinks: object,
    ) -> object:
        del query, key, value, kv_cache, attn_metadata
        events.append((layer_index, sinks))
        return (layer_index, sinks)

    return events, attention


def _run_all_layers(
    bridge: SelectiveAttentionBridge,
    attention: Any,
    sinks: object,
) -> None:
    for layer_index in range(24):
        bridge.run_layer(
            layer_index=layer_index,
            query=("q", layer_index),
            key=("k", layer_index),
            value=("v", layer_index),
            kv_cache=("cache", layer_index),
            slot_mapping=(layer_index,),
            attn_metadata=("meta", layer_index),
            sinks=sinks,
            attention=attention,
        )


def test_bridge_updates_before_attention_and_preserves_sink_identity() -> None:
    session = RecordingSession([])
    bridge = SelectiveAttentionBridge(session)
    events, attention = _attention_events()
    sinks = object()

    _run_all_layers(bridge, attention, sinks)
    receipt = bridge.finish()

    assert session.updates == list(range(24))
    assert [layer for layer, _ in events] == list(range(24))
    assert all(observed_sinks is sinks for _, observed_sinks in events)
    assert receipt.recomputed_token_rows == 24
    assert session.finish_calls == 1


def test_bridge_rejects_out_of_order_layer_and_becomes_terminal() -> None:
    session = RecordingSession([])
    bridge = SelectiveAttentionBridge(session)
    events, attention = _attention_events()

    with pytest.raises(SelectiveAttentionError) as caught:
        bridge.run_layer(
            layer_index=1,
            query=None,
            key=None,
            value=None,
            kv_cache=None,
            slot_mapping=(),
            attn_metadata=None,
            sinks=object(),
            attention=attention,
        )
    assert caught.value.code is SelectiveAttentionErrorCode.LAYER_ORDER_MISMATCH
    assert session.updates == []
    assert events == []

    with pytest.raises(SelectiveAttentionError) as caught:
        bridge.finish()
    assert caught.value.code is SelectiveAttentionErrorCode.INVALID_SESSION


def test_bridge_requires_sinks_before_mutating_cache() -> None:
    session = RecordingSession([])
    bridge = SelectiveAttentionBridge(session)
    events, attention = _attention_events()

    with pytest.raises(SelectiveAttentionError) as caught:
        bridge.run_layer(
            layer_index=0,
            query=None,
            key=None,
            value=None,
            kv_cache=None,
            slot_mapping=(),
            attn_metadata=None,
            sinks=None,
            attention=attention,
        )
    assert caught.value.code is SelectiveAttentionErrorCode.SINK_REQUIRED
    assert session.updates == []
    assert events == []


def test_update_failure_skips_attention_and_requires_discard() -> None:
    session = RecordingSession([], fail_layer=0)
    bridge = SelectiveAttentionBridge(session)
    events, attention = _attention_events()

    with pytest.raises(SelectiveAttentionError) as caught:
        bridge.run_layer(
            layer_index=0,
            query=None,
            key=None,
            value=None,
            kv_cache=None,
            slot_mapping=(),
            attn_metadata=None,
            sinks=object(),
            attention=attention,
        )
    assert caught.value.code is SelectiveAttentionErrorCode.UPDATE_FAILED
    assert events == []

    with pytest.raises(SelectiveAttentionError) as caught:
        bridge.run_layer(
            layer_index=0,
            query=None,
            key=None,
            value=None,
            kv_cache=None,
            slot_mapping=(),
            attn_metadata=None,
            sinks=object(),
            attention=attention,
        )
    assert caught.value.code is SelectiveAttentionErrorCode.INVALID_SESSION


def test_attention_failure_is_terminal_after_layer_update() -> None:
    session = RecordingSession([])
    bridge = SelectiveAttentionBridge(session)

    def attention(**kwargs: object) -> object:
        del kwargs
        raise RuntimeError("private attention detail")

    with pytest.raises(SelectiveAttentionError) as caught:
        bridge.run_layer(
            layer_index=0,
            query=None,
            key=None,
            value=None,
            kv_cache=None,
            slot_mapping=(),
            attn_metadata=None,
            sinks=object(),
            attention=attention,
        )
    assert caught.value.code is SelectiveAttentionErrorCode.ATTENTION_FAILED
    assert session.updates == [0]
    assert "private attention detail" not in str(caught.value)


def test_finish_requires_all_layers() -> None:
    session = RecordingSession([])
    bridge = SelectiveAttentionBridge(session)

    with pytest.raises(SelectiveAttentionError) as caught:
        bridge.finish()
    assert caught.value.code is SelectiveAttentionErrorCode.INCOMPLETE
    assert session.finish_calls == 0


def test_invalid_session_is_rejected_without_importing_runtime_dependencies() -> None:
    with pytest.raises(SelectiveAttentionError) as caught:
        SelectiveAttentionBridge(object())  # type: ignore[arg-type]
    assert caught.value.code is SelectiveAttentionErrorCode.INVALID_SESSION
