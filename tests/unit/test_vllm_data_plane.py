"""CPU-only tests for the pinned GPT-OSS KV staging data plane."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NoReturn

import pytest

from cacheblend_gpt_oss.gpt_oss import (
    AttentionKind,
    CacheGroupLayout,
    GptOssHybridCacheLayout,
    GroupBlockTable,
    LayerTokenScatterSpan,
    TokenTransfer,
    plan_token_scatter,
)
from cacheblend_gpt_oss.planner import TokenRange
from cacheblend_gpt_oss.vllm_compat.v0_19_1 import data_plane as module
from cacheblend_gpt_oss.vllm_compat.v0_19_1.data_plane import (
    DataPlaneError,
    DataPlaneErrorCode,
    GptOssDataPlane,
    TransferDirection,
)

BLOCK_SIZE = 16
SOURCE = TokenRange(101, 123)
TARGET = TokenRange(5, 27)
RETRIEVAL_OFFSET = 7
STORE_OFFSET = 3


def layer_name(index: int) -> str:
    return f"model.layers.{index}.attn.attn"


def layout() -> GptOssHybridCacheLayout:
    return GptOssHybridCacheLayout(
        groups=(
            CacheGroupLayout(
                group_id=0,
                attention_kind=AttentionKind.SLIDING,
                layer_names=tuple(layer_name(index) for index in range(0, 24, 2)),
                block_size=BLOCK_SIZE,
                sliding_window=128,
            ),
            CacheGroupLayout(
                group_id=1,
                attention_kind=AttentionKind.FULL,
                layer_names=tuple(layer_name(index) for index in range(1, 24, 2)),
                block_size=BLOCK_SIZE,
                sliding_window=None,
            ),
        )
    )


def spans() -> tuple[LayerTokenScatterSpan, ...]:
    plan = plan_token_scatter(
        layout(),
        (
            GroupBlockTable(0, BLOCK_SIZE, (2, 3)),
            GroupBlockTable(1, BLOCK_SIZE, (4, 5)),
        ),
        TokenTransfer(SOURCE, TARGET),
    )
    return plan.layer_spans


@dataclass(slots=True)
class FakeTensor:
    shape: tuple[int, ...]
    dtype: str = "torch.bfloat16"
    device: str = "cuda:0"
    rows: dict[tuple[int, ...], object] | None = None

    def __post_init__(self) -> None:
        if self.rows is None:
            self.rows = {}


@dataclass(slots=True)
class FakeView:
    owner: FakeTensor
    refs: tuple[tuple[int, ...], ...]
    shape: tuple[int, ...]


@dataclass(slots=True)
class FakeValue:
    rows: tuple[object, ...]
    shape: tuple[int, ...]
    dtype: str
    device: str


class FakeTensorOps:
    def __init__(self) -> None:
        self.copy_count = 0
        self.synchronizations: list[FakeTensor] = []
        self.staging_reads: list[tuple[int, int, int, int]] = []

    def shape(self, tensor: object) -> tuple[int, ...]:
        assert isinstance(tensor, FakeTensor | FakeView | FakeValue)
        return tensor.shape

    def dtype_name(self, tensor: object) -> str:
        assert isinstance(tensor, FakeTensor | FakeView | FakeValue)
        if isinstance(tensor, FakeView):
            return tensor.owner.dtype
        return tensor.dtype

    def device_name(self, tensor: object) -> str:
        assert isinstance(tensor, FakeTensor | FakeView | FakeValue)
        if isinstance(tensor, FakeView):
            return tensor.owner.device
        return tensor.device

    def paged_rows(
        self,
        tensor: object,
        *,
        component: int,
        block_id: int,
        block_offset: int,
        token_count: int,
    ) -> FakeView:
        assert isinstance(tensor, FakeTensor)
        refs = tuple(
            (block_id, component, block_offset + index) for index in range(token_count)
        )
        return FakeView(tensor, refs, (token_count, 8, 64))

    def staging_rows(
        self,
        tensor: object,
        *,
        component: int,
        layer_index: int,
        token_start: int,
        token_count: int,
    ) -> FakeView:
        assert isinstance(tensor, FakeTensor)
        self.staging_reads.append((component, layer_index, token_start, token_count))
        refs = tuple(
            (component, layer_index, token_start + index)
            for index in range(token_count)
        )
        return FakeView(tensor, refs, (token_count, 512))

    def reshape(self, tensor: object, shape: tuple[int, ...]) -> FakeView | FakeValue:
        assert isinstance(tensor, FakeView | FakeValue)
        old_elements = _product(tensor.shape)
        new_elements = _product(shape)
        if old_elements != new_elements:
            raise ValueError("reshape changes element count")
        if isinstance(tensor, FakeView):
            return FakeView(tensor.owner, tensor.refs, shape)
        return FakeValue(tensor.rows, shape, tensor.dtype, tensor.device)

    def copy(self, destination: object, source: object) -> None:
        assert isinstance(destination, FakeView)
        assert isinstance(source, FakeView | FakeValue)
        assert destination.shape == source.shape
        source_rows = _rows(source)
        assert len(destination.refs) == len(source_rows)
        assert destination.owner.rows is not None
        for reference, row in zip(destination.refs, source_rows, strict=True):
            destination.owner.rows[reference] = row
        self.copy_count += 1

    def synchronize(self, tensor: object) -> None:
        assert isinstance(tensor, FakeTensor)
        self.synchronizations.append(tensor)


class RecordingCorrector:
    def __init__(self, ops: FakeTensorOps, *, fail_layer: int | None = None) -> None:
        self.ops = ops
        self.fail_layer = fail_layer
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []

    def __call__(
        self,
        key_rows: object,
        *,
        source_positions: tuple[int, ...],
        target_positions: tuple[int, ...],
        layer_index: int,
    ) -> FakeValue:
        if layer_index == self.fail_layer:
            raise ValueError("injected correction failure")
        assert isinstance(key_rows, FakeView | FakeValue)
        self.calls.append((source_positions, target_positions, layer_index))
        corrected = tuple(
            ("corrected-k", value, source, target, layer_index)
            for value, source, target in zip(
                _rows(key_rows), source_positions, target_positions, strict=True
            )
        )
        return FakeValue(
            corrected,
            key_rows.shape,
            self.ops.dtype_name(key_rows),
            self.ops.device_name(key_rows),
        )


def _product(shape: tuple[int, ...]) -> int:
    product = 1
    for dimension in shape:
        product *= dimension
    return product


def _rows(value: FakeView | FakeValue) -> tuple[object, ...]:
    if isinstance(value, FakeValue):
        return value.rows
    assert value.owner.rows is not None
    return tuple(value.owner.rows.get(reference) for reference in value.refs)


def paged_caches() -> dict[str, FakeTensor]:
    return {
        layer_name(index): FakeTensor((6, 2, BLOCK_SIZE, 8, 64)) for index in range(24)
    }


def staging(capacity: int = 64) -> FakeTensor:
    return FakeTensor((2, 24, capacity, 512))


def seed_retrieved_staging(tensor: FakeTensor) -> None:
    assert tensor.rows is not None
    for layer_index in range(24):
        for target_position in range(TARGET.start, TARGET.end):
            staging_position = RETRIEVAL_OFFSET + target_position
            tensor.rows[(0, layer_index, staging_position)] = (
                "retrieved-k",
                layer_index,
                target_position,
            )
            tensor.rows[(1, layer_index, staging_position)] = (
                "retrieved-v",
                layer_index,
                target_position,
            )


def seed_paged(tensors: dict[str, FakeTensor]) -> None:
    for span in spans():
        tensor = tensors[span.layer_name]
        assert tensor.rows is not None
        for index, target_position in enumerate(
            range(span.target_range.start, span.target_range.end)
        ):
            block_position = span.group_span.block_offset + index
            tensor.rows[(span.group_span.block_id, 0, block_position)] = (
                "paged-k",
                span.layer_index,
                target_position,
            )
            tensor.rows[(span.group_span.block_id, 1, block_position)] = (
                "paged-v",
                span.layer_index,
                target_position,
            )


def paged_row(
    tensors: dict[str, FakeTensor],
    span: LayerTokenScatterSpan,
    component: int,
    token_index: int,
) -> object:
    tensor = tensors[span.layer_name]
    assert tensor.rows is not None
    key = (
        span.group_span.block_id,
        component,
        span.group_span.block_offset + token_index,
    )
    return tensor.rows.get(key)


def assert_error(code: DataPlaneErrorCode, operation: object) -> DataPlaneError:
    assert callable(operation)
    with pytest.raises(DataPlaneError) as caught:
        operation()
    assert caught.value.code is code
    return caught.value


def test_scatter_reads_target_plus_offset_corrects_only_k_and_copies_v() -> None:
    ops = FakeTensorOps()
    data_plane = GptOssDataPlane(ops)
    stage = staging()
    caches = paged_caches()
    seed_retrieved_staging(stage)
    corrector = RecordingCorrector(ops)

    receipt = data_plane.scatter_retrieved_kv(
        staging=stage,
        paged_caches=caches,
        layer_spans=spans(),
        retrieval_buffer_offset=RETRIEVAL_OFFSET,
        query_token_count=TARGET.end,
        correct_key_positions=corrector,
    )

    assert receipt.direction is TransferDirection.LOAD_FROM_STAGING
    assert receipt.logical_tokens == len(TARGET)
    assert receipt.layer_token_rows == len(TARGET) * 24
    assert receipt.span_count == 48
    assert receipt.corrected_key_rows == len(TARGET) * 24
    assert receipt.position_correction_latency_seconds >= 0.0
    assert not receipt.sinks_touched
    assert len(corrector.calls) == 48
    assert ops.copy_count == 96
    assert len(ops.synchronizations) == 1

    first = spans()[0]
    first_key = paged_row(caches, first, 0, 0)
    first_value = paged_row(caches, first, 1, 0)
    assert first_key == (
        "corrected-k",
        ("retrieved-k", 0, TARGET.start),
        SOURCE.start,
        TARGET.start,
        0,
    )
    assert first_value == ("retrieved-v", 0, TARGET.start)
    assert min(read[2] for read in ops.staging_reads) == (
        RETRIEVAL_OFFSET + TARGET.start
    )
    assert all(read[2] < SOURCE.start for read in ops.staging_reads)
    first_call = corrector.calls[0]
    assert first_call[0][0] == SOURCE.start
    assert first_call[1][0] == TARGET.start


def test_disabled_scatter_still_corrects_but_never_copies_into_paged_cache() -> None:
    ops = FakeTensorOps()
    data_plane = GptOssDataPlane(ops)
    stage = staging()
    caches = paged_caches()
    seed_retrieved_staging(stage)
    corrector = RecordingCorrector(ops)

    receipt = data_plane.scatter_retrieved_kv(
        staging=stage,
        paged_caches=caches,
        layer_spans=spans(),
        retrieval_buffer_offset=RETRIEVAL_OFFSET,
        query_token_count=TARGET.end,
        correct_key_positions=corrector,
        disable_scatter=True,
    )

    # Correction and every staging/paged view still ran in full.
    assert len(corrector.calls) == 48
    assert receipt.corrected_key_rows == len(TARGET) * 24
    assert receipt.position_correction_latency_seconds >= 0.0

    # But the destination mutation never happened.
    assert ops.copy_count == 0
    assert len(ops.synchronizations) == 0
    assert receipt.scatter_suppressed is True
    assert receipt.copied_key_rows == 0
    assert receipt.copied_value_rows == 0
    for span in spans():
        assert paged_row(caches, span, 0, 0) is None
        assert paged_row(caches, span, 1, 0) is None


def test_default_scatter_path_is_unchanged_and_reports_not_suppressed() -> None:
    ops = FakeTensorOps()
    data_plane = GptOssDataPlane(ops)
    stage = staging()
    caches = paged_caches()
    seed_retrieved_staging(stage)
    corrector = RecordingCorrector(ops)

    receipt = data_plane.scatter_retrieved_kv(
        staging=stage,
        paged_caches=caches,
        layer_spans=spans(),
        retrieval_buffer_offset=RETRIEVAL_OFFSET,
        query_token_count=TARGET.end,
        correct_key_positions=corrector,
    )

    assert receipt.scatter_suppressed is False
    assert receipt.copied_key_rows == len(TARGET) * 24
    assert receipt.copied_value_rows == len(TARGET) * 24
    assert ops.copy_count == 96


def test_batched_scatter_synchronizes_once_for_multiple_candidates() -> None:
    ops = FakeTensorOps()
    data_plane = GptOssDataPlane(ops)
    stage = staging()
    caches = paged_caches()
    seed_retrieved_staging(stage)
    corrector = RecordingCorrector(ops)

    receipts = data_plane.scatter_retrieved_kv_batch(
        staging=stage,
        paged_caches=caches,
        candidate_layer_spans=(spans(), spans()),
        retrieval_buffer_offset=RETRIEVAL_OFFSET,
        query_token_count=TARGET.end,
        correct_key_positions=corrector,
    )

    assert len(receipts) == 2
    assert all(receipt.logical_tokens == len(TARGET) for receipt in receipts)
    assert ops.copy_count == 192
    assert len(ops.synchronizations) == 1


def test_receipt_rejects_suppressed_flag_with_nonzero_copied_rows() -> None:
    with pytest.raises(DataPlaneError) as caught:
        module.DataPlaneReceipt(
            direction=TransferDirection.LOAD_FROM_STAGING,
            logical_tokens=1,
            layer_token_rows=24,
            span_count=1,
            corrected_key_rows=24,
            copied_key_rows=24,
            copied_value_rows=24,
            scatter_suppressed=True,
        )
    assert caught.value.code is DataPlaneErrorCode.INVALID_SPAN


def test_gather_compacts_target_rows_without_position_correction() -> None:
    ops = FakeTensorOps()
    data_plane = GptOssDataPlane(ops)
    stage = staging()
    caches = paged_caches()
    seed_paged(caches)

    receipt = data_plane.gather_precomputed_kv(
        paged_caches=caches,
        staging=stage,
        layer_spans=spans(),
        document_target_range=TARGET,
        store_buffer_offset=STORE_OFFSET,
    )

    assert receipt.direction is TransferDirection.STORE_TO_STAGING
    assert receipt.corrected_key_rows == 0
    assert receipt.copied_key_rows == len(TARGET) * 24
    assert receipt.copied_value_rows == len(TARGET) * 24
    assert not receipt.sinks_touched
    assert ops.copy_count == 96
    assert data_plane.last_gather_timing.prepared_copy_operations == 96
    assert data_plane.last_gather_timing.total_latency_seconds >= 0.0
    assert stage.rows is not None
    assert stage.rows[(0, 0, STORE_OFFSET)] == ("paged-k", 0, TARGET.start)
    assert stage.rows[(1, 0, STORE_OFFSET)] == ("paged-v", 0, TARGET.start)
    assert stage.rows[(0, 0, STORE_OFFSET + len(TARGET) - 1)] == (
        "paged-k",
        0,
        TARGET.end - 1,
    )
    assert stage.rows[(0, 0, STORE_OFFSET + TARGET.start)] == (
        "paged-k",
        0,
        TARGET.start + TARGET.start,
    )
    assert (0, 0, STORE_OFFSET + len(TARGET)) not in stage.rows


def test_prepared_gather_batch_is_read_only_and_executes_exactly_once() -> None:
    ops = FakeTensorOps()
    data_plane = GptOssDataPlane(ops)
    stage = staging()
    caches = paged_caches()
    seed_paged(caches)

    batch = data_plane.prepare_gather_precomputed_kv_batch(
        paged_caches=caches,
        staging=stage,
        chunk_layer_spans=(spans(),),
        document_target_ranges=(TARGET,),
        store_buffer_offsets=(STORE_OFFSET,),
    )

    assert batch.prepared_copy_operations == 96
    assert len(batch.receipts) == 1
    assert data_plane.last_gather_timing.prepared_copy_operations == 96
    assert data_plane.last_gather_timing.enqueue_latency_seconds == 0.0
    assert data_plane.last_gather_timing.synchronize_latency_seconds == 0.0
    assert ops.copy_count == 0
    assert ops.synchronizations == []
    assert stage.rows == {}

    receipts = data_plane.execute_prepared_gather_batch(batch)

    assert receipts == batch.receipts
    assert ops.copy_count == 96
    assert len(ops.synchronizations) == 1
    assert data_plane.last_gather_timing.prepare_latency_seconds == 0.0
    assert data_plane.last_gather_timing.prepared_copy_operations == 96
    with pytest.raises(DataPlaneError) as caught:
        data_plane.execute_prepared_gather_batch(batch)
    assert caught.value.code is DataPlaneErrorCode.INVALID_PREPARED_BATCH


def test_prepared_gather_batch_discard_releases_views_without_mutation() -> None:
    ops = FakeTensorOps()
    data_plane = GptOssDataPlane(ops)
    stage = staging()
    batch = data_plane.prepare_gather_precomputed_kv_batch(
        paged_caches=paged_caches(),
        staging=stage,
        chunk_layer_spans=(spans(),),
        document_target_ranges=(TARGET,),
        store_buffer_offsets=(STORE_OFFSET,),
    )

    data_plane.discard_prepared_gather_batch(batch)

    assert ops.copy_count == 0
    assert ops.synchronizations == []
    assert stage.rows == {}
    with pytest.raises(DataPlaneError) as caught:
        data_plane.execute_prepared_gather_batch(batch)
    assert caught.value.code is DataPlaneErrorCode.INVALID_PREPARED_BATCH


def test_late_shape_failure_preflights_before_any_mutation() -> None:
    ops = FakeTensorOps()
    data_plane = GptOssDataPlane(ops)
    stage = staging()
    caches = paged_caches()
    seed_retrieved_staging(stage)
    caches[layer_name(23)].shape = (6, 2, BLOCK_SIZE, 8, 63)

    assert_error(
        DataPlaneErrorCode.INVALID_PAGED_CACHE_SHAPE,
        lambda: data_plane.scatter_retrieved_kv(
            staging=stage,
            paged_caches=caches,
            layer_spans=spans(),
            retrieval_buffer_offset=RETRIEVAL_OFFSET,
            query_token_count=TARGET.end,
            correct_key_positions=RecordingCorrector(ops),
        ),
    )
    assert ops.copy_count == 0


def test_late_key_correction_failure_preflights_before_any_mutation() -> None:
    ops = FakeTensorOps()
    data_plane = GptOssDataPlane(ops)
    stage = staging()
    caches = paged_caches()
    seed_retrieved_staging(stage)

    assert_error(
        DataPlaneErrorCode.POSITION_CORRECTION_FAILED,
        lambda: data_plane.scatter_retrieved_kv(
            staging=stage,
            paged_caches=caches,
            layer_spans=spans(),
            retrieval_buffer_offset=RETRIEVAL_OFFSET,
            query_token_count=TARGET.end,
            correct_key_positions=RecordingCorrector(ops, fail_layer=23),
        ),
    )
    assert ops.copy_count == 0
    assert all(not tensor.rows for tensor in caches.values())


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (
            lambda tensor: setattr(tensor, "shape", (2, 23, 64, 512)),
            DataPlaneErrorCode.INVALID_STAGING_SHAPE,
        ),
        (
            lambda tensor: setattr(tensor, "dtype", "torch.float32"),
            DataPlaneErrorCode.DTYPE_MISMATCH,
        ),
        (
            lambda tensor: setattr(tensor, "device", "cuda:1"),
            DataPlaneErrorCode.DEVICE_MISMATCH,
        ),
        (
            lambda tensor: setattr(tensor, "device", "cpu"),
            DataPlaneErrorCode.INVALID_DEVICE,
        ),
    ],
)
def test_staging_shape_dtype_and_device_fail_closed(
    mutation: object, code: DataPlaneErrorCode
) -> None:
    assert callable(mutation)
    ops = FakeTensorOps()
    stage = staging()
    caches = paged_caches()
    mutation(stage)

    assert_error(
        code,
        lambda: GptOssDataPlane(ops).gather_precomputed_kv(
            paged_caches=caches,
            staging=stage,
            layer_spans=spans(),
            document_target_range=TARGET,
            store_buffer_offset=STORE_OFFSET,
        ),
    )
    assert ops.copy_count == 0


def test_scatter_rejects_retrieval_placement_outside_staging() -> None:
    ops = FakeTensorOps()
    assert_error(
        DataPlaneErrorCode.STAGING_RANGE_OUT_OF_BOUNDS,
        lambda: GptOssDataPlane(ops).scatter_retrieved_kv(
            staging=staging(capacity=32),
            paged_caches=paged_caches(),
            layer_spans=spans(),
            retrieval_buffer_offset=10,
            query_token_count=TARGET.end,
            correct_key_positions=RecordingCorrector(ops),
        ),
    )
    assert ops.copy_count == 0


def test_missing_layer_cache_and_missing_layer_span_fail_before_mutation() -> None:
    ops = FakeTensorOps()
    caches = paged_caches()
    del caches[layer_name(23)]
    assert_error(
        DataPlaneErrorCode.PAGED_CACHE_SET_MISMATCH,
        lambda: GptOssDataPlane(ops).gather_precomputed_kv(
            paged_caches=caches,
            staging=staging(),
            layer_spans=spans(),
            document_target_range=TARGET,
            store_buffer_offset=0,
        ),
    )
    assert_error(
        DataPlaneErrorCode.LAYER_SET_MISMATCH,
        lambda: GptOssDataPlane(ops).gather_precomputed_kv(
            paged_caches=paged_caches(),
            staging=staging(),
            layer_spans=tuple(span for span in spans() if span.layer_index != 23),
            document_target_range=TARGET,
            store_buffer_offset=0,
        ),
    )
    assert ops.copy_count == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("group_id", True),
        ("block_id", True),
        ("block_size", True),
        ("source_range", None),
    ],
)
def test_malformed_group_span_fails_before_tensor_access(
    field: str, value: object
) -> None:
    ops = FakeTensorOps()
    original = spans()
    malformed_group = replace(original[0].group_span, **{field: value})
    malformed = (replace(original[0], group_span=malformed_group), *original[1:])

    assert_error(
        DataPlaneErrorCode.INVALID_SPAN,
        lambda: GptOssDataPlane(ops).gather_precomputed_kv(
            paged_caches=paged_caches(),
            staging=staging(),
            layer_spans=malformed,
            document_target_range=TARGET,
            store_buffer_offset=STORE_OFFSET,
        ),
    )
    assert ops.copy_count == 0


def test_invalid_corrected_key_shape_fails_before_mutation() -> None:
    class BadCorrector:
        def __call__(
            self,
            key_rows: object,
            *,
            source_positions: tuple[int, ...],
            target_positions: tuple[int, ...],
            layer_index: int,
        ) -> FakeValue:
            del key_rows, source_positions, target_positions, layer_index
            return FakeValue((), (0, 8, 64), "torch.bfloat16", "cuda:0")

    ops = FakeTensorOps()
    assert_error(
        DataPlaneErrorCode.INVALID_CORRECTED_KEY,
        lambda: GptOssDataPlane(ops).scatter_retrieved_kv(
            staging=staging(),
            paged_caches=paged_caches(),
            layer_spans=spans(),
            retrieval_buffer_offset=RETRIEVAL_OFFSET,
            query_token_count=TARGET.end,
            correct_key_positions=BadCorrector(),
        ),
    )
    assert ops.copy_count == 0


def test_copy_failure_marks_result_unusable() -> None:
    class FailingCopyOps(FakeTensorOps):
        def copy(self, destination: object, source: object) -> None:
            del destination, source
            raise ValueError("injected copy failure")

    ops = FailingCopyOps()
    assert_error(
        DataPlaneErrorCode.MUTATION_FAILED,
        lambda: GptOssDataPlane(ops).gather_precomputed_kv(
            paged_caches=paged_caches(),
            staging=staging(),
            layer_spans=spans(),
            document_target_range=TARGET,
            store_buffer_offset=0,
        ),
    )


def test_lazy_torch_loader_has_actionable_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_import(name: str) -> NoReturn:
        assert name == "torch"
        raise ImportError(name)

    monkeypatch.setattr(module, "import_module", missing_import)

    error = assert_error(
        DataPlaneErrorCode.TORCH_DEPENDENCY_MISSING,
        module.load_torch_tensor_ops,
    )
    assert "pinned GPU runtime extras" in str(error)


def test_lazy_torch_loader_rejects_wrong_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCudaVersion:
        cuda = "12.8"

    class FakeTorch:
        __version__ = "2.10.1+cu128"
        version = FakeCudaVersion()

    monkeypatch.setattr(module, "import_module", lambda name: FakeTorch())

    assert_error(
        DataPlaneErrorCode.TORCH_VERSION_MISMATCH,
        module.load_torch_tensor_ops,
    )


def test_span_group_block_size_inconsistency_is_rejected() -> None:
    work = list(spans())
    first = work[0]
    work[0] = replace(
        first,
        group_span=replace(
            first.group_span,
            block_size=32,
            physical_slot_start=(
                first.group_span.block_id * 32 + first.group_span.block_offset
            ),
        ),
    )
    ops = FakeTensorOps()

    assert_error(
        DataPlaneErrorCode.INVALID_GROUP_LAYOUT,
        lambda: GptOssDataPlane(ops).gather_precomputed_kv(
            paged_caches=paged_caches(),
            staging=staging(),
            layer_spans=work,
            document_target_range=TARGET,
            store_buffer_offset=0,
        ),
    )
