from __future__ import annotations

from dataclasses import dataclass

import pytest

from cacheblend_gpt_oss.gpt_oss import (
    AttentionKind,
    CacheGroupLayout,
    GptOssHybridCacheLayout,
    GroupBlockTable,
    TokenTransfer,
    plan_selective_kv_writes,
    plan_token_scatter,
)
from cacheblend_gpt_oss.gpt_oss.selective import ForwardRowPlan
from cacheblend_gpt_oss.gpt_oss.selective_attention import (
    SelectiveAttentionBridge,
)
from cacheblend_gpt_oss.gpt_oss.selective_kv import (
    GptOssSelectiveKvSession,
    GptOssSelectiveKvUpdater,
    SelectiveUpdateError,
    SelectiveUpdateErrorCode,
)
from cacheblend_gpt_oss.planner import TokenRange

BLOCK_SIZE = 16
PROMPT_TOKENS = 20


def _layer_name(index: int) -> str:
    return f"model.layers.{index}.attn.attn"


def _spans():
    layout = GptOssHybridCacheLayout(
        (
            CacheGroupLayout(
                0,
                AttentionKind.FULL,
                tuple(_layer_name(index) for index in range(1, 24, 2)),
                BLOCK_SIZE,
                None,
            ),
            CacheGroupLayout(
                1,
                AttentionKind.SLIDING,
                tuple(_layer_name(index) for index in range(0, 24, 2)),
                BLOCK_SIZE,
                128,
            ),
        )
    )
    return plan_token_scatter(
        layout,
        (
            GroupBlockTable(0, BLOCK_SIZE, (0, 1)),
            GroupBlockTable(1, BLOCK_SIZE, (2, 3)),
        ),
        TokenTransfer(TokenRange(100, 120), TokenRange(0, PROMPT_TOKENS)),
    ).layer_spans


@dataclass(slots=True)
class FakeTensor:
    shape: tuple[int, ...]
    dtype: str = "torch.bfloat16"
    device: str = "cuda:0"
    rows: dict[tuple[int, ...], object] | None = None

    def __post_init__(self) -> None:
        if self.rows is None:
            self.rows = {}


@dataclass(frozen=True, slots=True)
class FakeView:
    owner: FakeTensor
    refs: tuple[tuple[int, ...], ...]
    shape: tuple[int, ...]


class FakeSelectiveOps:
    def __init__(self) -> None:
        self.copy_count = 0
        self.sync_count = 0

    def shape(self, tensor: object) -> tuple[int, ...]:
        assert isinstance(tensor, FakeTensor | FakeView)
        return tensor.shape

    def dtype_name(self, tensor: object) -> str:
        assert isinstance(tensor, FakeTensor | FakeView)
        return tensor.owner.dtype if isinstance(tensor, FakeView) else tensor.dtype

    def device_name(self, tensor: object) -> str:
        assert isinstance(tensor, FakeTensor | FakeView)
        return tensor.owner.device if isinstance(tensor, FakeView) else tensor.device

    def prompt_rows(self, tensor: object, *, start: int, count: int) -> FakeView:
        assert isinstance(tensor, FakeTensor)
        return FakeView(
            tensor,
            tuple((start + index,) for index in range(count)),
            (count, 8, 64),
        )

    def paged_rows(
        self,
        tensor: object,
        *,
        component: int,
        block_id: int,
        block_offset: int,
        count: int,
    ) -> FakeView:
        assert isinstance(tensor, FakeTensor)
        return FakeView(
            tensor,
            tuple(
                (block_id, component, block_offset + index)
                for index in range(count)
            ),
            (count, 8, 64),
        )

    def copy(self, destination: object, source: object) -> None:
        assert isinstance(destination, FakeView)
        assert isinstance(source, FakeView)
        assert destination.shape == source.shape
        assert destination.owner.rows is not None
        assert source.owner.rows is not None
        for destination_ref, source_ref in zip(
            destination.refs, source.refs, strict=True
        ):
            destination.owner.rows[destination_ref] = source.owner.rows.get(
                source_ref
            )
        self.copy_count += 1

    def synchronize(self, tensor: object) -> None:
        assert isinstance(tensor, FakeTensor)
        self.sync_count += 1


def _tensors() -> tuple[
    dict[str, FakeTensor],
    dict[str, FakeTensor],
    dict[str, FakeTensor],
]:
    keys = {
        _layer_name(index): FakeTensor((PROMPT_TOKENS, 8, 64))
        for index in range(24)
    }
    values = {
        _layer_name(index): FakeTensor((PROMPT_TOKENS, 8, 64))
        for index in range(24)
    }
    caches = {
        _layer_name(index): FakeTensor((4, 2, BLOCK_SIZE, 8, 64))
        for index in range(24)
    }
    for layer_index in range(24):
        key_rows = keys[_layer_name(layer_index)].rows
        value_rows = values[_layer_name(layer_index)].rows
        cache_rows = caches[_layer_name(layer_index)].rows
        assert key_rows is not None
        assert value_rows is not None
        assert cache_rows is not None
        for token in range(PROMPT_TOKENS):
            key_rows[(token,)] = ("key", layer_index, token)
            value_rows[(token,)] = ("value", layer_index, token)
        for block_id in range(4):
            for component in range(2):
                for offset in range(BLOCK_SIZE):
                    cache_rows[(block_id, component, offset)] = (
                        "sentinel",
                        layer_index,
                        component,
                        block_id,
                        offset,
                    )
    return keys, values, caches


def _selective_plan():
    ranges = [
        (TokenRange(0, 3), TokenRange(10, 20))
        if index == 0
        else (TokenRange(0, PROMPT_TOKENS),)
        for index in range(24)
    ]
    return plan_selective_kv_writes(
        _spans(),
        ForwardRowPlan.from_recompute_ranges(PROMPT_TOKENS, ranges),
    )


def _slot_mappings(plan) -> dict[str, tuple[int, ...]]:
    mappings: dict[str, list[int]] = {
        _layer_name(index): [0] * PROMPT_TOKENS for index in range(24)
    }
    for span in plan.full_layer_spans:
        mapping = mappings[span.layer_name]
        for offset in range(span.token_count):
            mapping[span.target_range.start + offset] = (
                span.physical_slot_start + offset
            )
    return {name: tuple(values) for name, values in mappings.items()}


def _cache_value(caches: dict[str, FakeTensor], layer: int, component: int, token: int):
    block_id, offset = ((2, token) if token < 16 else (3, token - 16))
    rows = caches[_layer_name(layer)].rows
    assert rows is not None
    return rows[(block_id, component, offset)]


def test_updater_writes_only_recomputed_rows_and_preserves_cached_rows() -> None:
    ops = FakeSelectiveOps()
    keys, values, caches = _tensors()
    plan = _selective_plan()

    receipt = GptOssSelectiveKvUpdater(ops).update(
        plan=plan,
        key_by_layer=keys,
        value_by_layer=values,
        paged_caches=caches,
        slot_mapping_by_layer=_slot_mappings(plan),
    )

    assert receipt.recomputed_token_rows == 23 * PROMPT_TOKENS + 13
    assert receipt.cached_token_rows == 7
    assert receipt.copied_key_rows == receipt.copied_value_rows
    assert receipt.sinks_touched is False
    assert ops.copy_count == receipt.write_span_count * 2
    assert ops.sync_count == 1
    assert _cache_value(caches, 0, 0, 4)[0] == "sentinel"
    assert _cache_value(caches, 0, 1, 9)[0] == "sentinel"
    assert _cache_value(caches, 0, 0, 0) == ("key", 0, 0)
    assert _cache_value(caches, 0, 1, 19) == ("value", 0, 19)


def test_all_cached_layer_is_valid_and_performs_no_layer_writes() -> None:
    ranges = [
        () if index == 0 else (TokenRange(0, PROMPT_TOKENS),)
        for index in range(24)
    ]
    plan = plan_selective_kv_writes(
        _spans(),
        ForwardRowPlan.from_recompute_ranges(PROMPT_TOKENS, ranges),
    )
    ops = FakeSelectiveOps()
    keys, values, caches = _tensors()
    slot_mappings = _slot_mappings(plan)

    receipt = GptOssSelectiveKvUpdater(ops).update(
        plan=plan,
        key_by_layer=keys,
        value_by_layer=values,
        paged_caches=caches,
        slot_mapping_by_layer=slot_mappings,
    )

    assert receipt.cached_token_rows == PROMPT_TOKENS
    assert all(
        _cache_value(caches, 0, component, token)[0] == "sentinel"
        for component in (0, 1)
        for token in range(PROMPT_TOKENS)
    )


def test_bad_layer_is_preflighted_before_any_copy() -> None:
    ops = FakeSelectiveOps()
    keys, values, caches = _tensors()
    caches[_layer_name(23)] = FakeTensor((4, 2, BLOCK_SIZE, 8, 32))
    plan = _selective_plan()

    with pytest.raises(SelectiveUpdateError) as error:
        GptOssSelectiveKvUpdater(ops).update(
            plan=plan,
            key_by_layer=keys,
            value_by_layer=values,
            paged_caches=caches,
            slot_mapping_by_layer=_slot_mappings(plan),
        )

    assert error.value.code is SelectiveUpdateErrorCode.INVALID_CACHE_SHAPE
    assert ops.copy_count == 0
    assert ops.sync_count == 0


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("dtype", SelectiveUpdateErrorCode.DTYPE_MISMATCH),
        ("device", SelectiveUpdateErrorCode.DEVICE_MISMATCH),
        ("cpu", SelectiveUpdateErrorCode.INVALID_DEVICE),
    ],
)
def test_dtype_and_device_checks_fail_closed(kind: str, expected) -> None:
    ops = FakeSelectiveOps()
    keys, values, caches = _tensors()
    if kind == "dtype":
        values[_layer_name(0)].dtype = "torch.float16"
    elif kind == "device":
        caches[_layer_name(0)].device = "cuda:1"
    else:
        keys[_layer_name(0)].device = "cpu"
        values[_layer_name(0)].device = "cpu"
        caches[_layer_name(0)].device = "cpu"
    plan = _selective_plan()

    with pytest.raises(SelectiveUpdateError) as error:
        GptOssSelectiveKvUpdater(ops).update(
            plan=plan,
            key_by_layer=keys,
            value_by_layer=values,
            paged_caches=caches,
            slot_mapping_by_layer=_slot_mappings(plan),
        )
    assert error.value.code is expected
    assert ops.copy_count == 0


def test_slot_mapping_failure_is_preflighted_before_any_copy() -> None:
    ops = FakeSelectiveOps()
    keys, values, caches = _tensors()
    plan = _selective_plan()
    mappings = _slot_mappings(plan)
    mappings[_layer_name(0)] = (999, *mappings[_layer_name(0)][1:])

    with pytest.raises(SelectiveUpdateError) as error:
        GptOssSelectiveKvUpdater(ops).update(
            plan=plan,
            key_by_layer=keys,
            value_by_layer=values,
            paged_caches=caches,
            slot_mapping_by_layer=mappings,
        )
    assert error.value.code is SelectiveUpdateErrorCode.SLOT_MAPPING_VALUE_MISMATCH
    assert ops.copy_count == 0
    assert ops.sync_count == 0


def test_per_layer_session_matches_atomic_updater_accounting() -> None:
    ops = FakeSelectiveOps()
    keys, values, caches = _tensors()
    plan = _selective_plan()
    mappings = _slot_mappings(plan)
    session = GptOssSelectiveKvSession(
        GptOssSelectiveKvUpdater(ops),
        plan=plan,
    )

    for layer_index in range(24):
        name = _layer_name(layer_index)
        session.update_layer(
            layer_index=layer_index,
            key=keys[name],
            value=values[name],
            paged_cache=caches[name],
            slot_mapping=mappings[name],
        )

    receipt = session.finish()
    assert receipt.recomputed_token_rows == plan.recompute_tokens
    assert receipt.cached_token_rows == plan.cached_tokens
    assert receipt.write_span_count == len(plan.recompute_layer_spans)
    assert receipt.copied_key_rows == plan.recompute_tokens
    assert receipt.copied_value_rows == plan.recompute_tokens
    assert ops.copy_count == receipt.write_span_count * 2
    assert ops.sync_count == 24


def test_per_layer_session_rejects_order_and_becomes_terminal() -> None:
    ops = FakeSelectiveOps()
    keys, values, caches = _tensors()
    plan = _selective_plan()
    mappings = _slot_mappings(plan)
    session = GptOssSelectiveKvSession(
        GptOssSelectiveKvUpdater(ops),
        plan=plan,
    )

    with pytest.raises(SelectiveUpdateError) as error:
        session.update_layer(
            layer_index=1,
            key=keys[_layer_name(1)],
            value=values[_layer_name(1)],
            paged_cache=caches[_layer_name(1)],
            slot_mapping=mappings[_layer_name(1)],
        )
    assert error.value.code is SelectiveUpdateErrorCode.SESSION_LAYER_ORDER_MISMATCH

    with pytest.raises(SelectiveUpdateError) as error:
        session.update_layer(
            layer_index=0,
            key=keys[_layer_name(0)],
            value=values[_layer_name(0)],
            paged_cache=caches[_layer_name(0)],
            slot_mapping=mappings[_layer_name(0)],
        )
    assert error.value.code is SelectiveUpdateErrorCode.SESSION_INVALID_STATE
    assert ops.copy_count == 0


def test_per_layer_session_incomplete_finish_is_terminal() -> None:
    ops = FakeSelectiveOps()
    keys, values, caches = _tensors()
    plan = _selective_plan()
    mappings = _slot_mappings(plan)
    session = GptOssSelectiveKvSession(
        GptOssSelectiveKvUpdater(ops),
        plan=plan,
    )
    name = _layer_name(0)
    session.update_layer(
        layer_index=0,
        key=keys[name],
        value=values[name],
        paged_cache=caches[name],
        slot_mapping=mappings[name],
    )

    with pytest.raises(SelectiveUpdateError) as error:
        session.finish()
    assert error.value.code is SelectiveUpdateErrorCode.SESSION_INCOMPLETE

    with pytest.raises(SelectiveUpdateError) as error:
        session.finish()
    assert error.value.code is SelectiveUpdateErrorCode.SESSION_INVALID_STATE


def test_per_layer_session_late_failure_requires_request_discard() -> None:
    ops = FakeSelectiveOps()
    keys, values, caches = _tensors()
    plan = _selective_plan()
    mappings = _slot_mappings(plan)
    session = GptOssSelectiveKvSession(
        GptOssSelectiveKvUpdater(ops),
        plan=plan,
    )
    first = _layer_name(0)
    session.update_layer(
        layer_index=0,
        key=keys[first],
        value=values[first],
        paged_cache=caches[first],
        slot_mapping=mappings[first],
    )
    bad = _layer_name(1)
    caches[bad] = FakeTensor((4, 2, BLOCK_SIZE, 8, 32))

    with pytest.raises(SelectiveUpdateError) as error:
        session.update_layer(
            layer_index=1,
            key=keys[bad],
            value=values[bad],
            paged_cache=caches[bad],
            slot_mapping=mappings[bad],
        )
    assert error.value.code is SelectiveUpdateErrorCode.INVALID_CACHE_SHAPE
    assert ops.copy_count > 0

    with pytest.raises(SelectiveUpdateError) as error:
        session.update_layer(
            layer_index=1,
            key=keys[bad],
            value=values[bad],
            paged_cache=caches[bad],
            slot_mapping=mappings[bad],
        )
    assert error.value.code is SelectiveUpdateErrorCode.SESSION_INVALID_STATE


def test_attention_bridge_drives_real_per_layer_session() -> None:
    ops = FakeSelectiveOps()
    keys, values, caches = _tensors()
    plan = _selective_plan()
    mappings = _slot_mappings(plan)
    session = GptOssSelectiveKvSession(
        GptOssSelectiveKvUpdater(ops),
        plan=plan,
    )
    bridge = SelectiveAttentionBridge(session)
    sink = object()
    attention_layers: list[tuple[int, object]] = []

    def attention(**kwargs: object) -> object:
        layer_index = kwargs["layer_index"]
        observed_sink = kwargs["sinks"]
        assert isinstance(layer_index, int)
        attention_layers.append((layer_index, observed_sink))
        return layer_index

    for layer_index in range(24):
        name = _layer_name(layer_index)
        assert (
            bridge.run_layer(
                layer_index=layer_index,
                query=None,
                key=keys[name],
                value=values[name],
                kv_cache=caches[name],
                slot_mapping=mappings[name],
                attn_metadata=None,
                sinks=sink,
                attention=attention,
            )
            == layer_index
        )

    receipt = bridge.finish()
    assert [layer for layer, _ in attention_layers] == list(range(24))
    assert all(observed_sink is sink for _, observed_sink in attention_layers)
    assert receipt.recomputed_token_rows == plan.recompute_tokens
