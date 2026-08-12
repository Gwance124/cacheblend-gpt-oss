from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast

import pytest

from cacheblend_gpt_oss.connector.control_plane import (
    METADATA_SCHEMA_VERSION,
    GroupedBlockAllocation,
    RequestAllocation,
    RequestHandoffMetadata,
    RequestPlan,
)
from cacheblend_gpt_oss.connector.control_plane import (
    CacheGroupLayout as ControlCacheGroupLayout,
)
from cacheblend_gpt_oss.gpt_oss.layout import (
    AttentionKind,
    CacheGroupLayout,
    GptOssHybridCacheLayout,
    GroupBlockTable,
    LayerTokenScatterSpan,
    TokenTransfer,
    plan_token_scatter,
)
from cacheblend_gpt_oss.planner.fingerprint import SHA256_FINGERPRINTER
from cacheblend_gpt_oss.planner.matching import MatchPlan, VerifiedMatch
from cacheblend_gpt_oss.planner.models import (
    CacheNamespace,
    CacheRecord,
    CandidateMatch,
    TokenRange,
    TokenSegment,
)
from cacheblend_gpt_oss.storage.lmcache_types import (
    LMCACHE_BLEND_PROTOCOL,
    LMCACHE_CACHE_KEY_PREFIX,
    LMCACHE_HASH_ALGORITHM,
    LMCACHE_SOURCE_COMMIT,
    LMCACHE_VERSION,
    LmcacheBlendTransportConfig,
    LmcacheCandidate,
    LmcacheRetrieveReceipt,
    LmcacheServerAttestation,
    LmcacheStoreReceipt,
    VerifiedLmcacheCandidate,
    query_digest,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.adapters import AdaptedKvCacheBlocks
from cacheblend_gpt_oss.vllm_compat.v0_19_1.data_plane import (
    DataPlaneReceipt,
    KeyPositionCorrector,
    TensorOps,
    TransferDirection,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.staging import (
    StagingBackend,
    StagingConfig,
    StagingState,
    StagingTransferDirection,
    StagingTransferLease,
    StagingTransport,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.transfer_runtime import (
    CandidateScatterWork,
    SchedulerTransferMetadata,
    StoreChunkWork,
    WorkerDataPlane,
    WorkerLoadPlan,
    WorkerStorage,
    WorkerStorePlan,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.worker_bridge import (
    DataPlaneOperations,
    GptOssWorkerBridge,
    StagingRuntimeLike,
    WorkerBridgeBufferConfig,
    WorkerBridgeError,
    WorkerBridgeErrorCode,
    WorkerBridgeState,
    WorkerLmcacheTransport,
)


def _namespace() -> CacheNamespace:
    return CacheNamespace(
        schema_version=1,
        model_id="openai/gpt-oss-20b",
        model_revision="model-revision",
        tokenizer_id="openai/gpt-oss-20b",
        tokenizer_revision="tokenizer-revision",
        model_config_digest="1" * 64,
        kv_cache_config_digest="2" * 64,
        adapter_revision="adapter-revision",
        vllm_version="0.19.1",
        lmcache_version="0.4.3",
        torch_version="2.10.0+cu128",
        cuda_runtime="12.8",
    )


def _transport_config() -> LmcacheBlendTransportConfig:
    return LmcacheBlendTransportConfig(
        namespace=_namespace(),
        server_attestation=LmcacheServerAttestation(
            lmcache_version=LMCACHE_VERSION,
            source_commit=LMCACHE_SOURCE_COMMIT,
            protocol=LMCACHE_BLEND_PROTOCOL,
            hash_algorithm=LMCACHE_HASH_ALGORITHM,
        ),
    )


def _layer_name(index: int) -> str:
    return f"model.layers.{index}.attn.attn"


def _layout() -> GptOssHybridCacheLayout:
    return GptOssHybridCacheLayout(
        (
            CacheGroupLayout(
                0,
                AttentionKind.FULL,
                tuple(_layer_name(index) for index in range(1, 24, 2)),
                16,
                None,
            ),
            CacheGroupLayout(
                1,
                AttentionKind.SLIDING,
                tuple(_layer_name(index) for index in range(0, 24, 2)),
                16,
                128,
            ),
        )
    )


def _plans(
    config: LmcacheBlendTransportConfig,
) -> tuple[WorkerLoadPlan, WorkerStorePlan]:
    prompt = tuple(range(600))
    layout = _layout()
    block_count = (len(prompt) + 15) // 16
    block_ids = (
        tuple(range(block_count)),
        tuple(range(block_count, block_count * 2)),
    )
    control_layout = ControlCacheGroupLayout(
        tuple(group.layer_names for group in layout.groups)
    )
    grouped = GroupedBlockAllocation.capture(control_layout, block_ids)
    blocks = AdaptedKvCacheBlocks(
        grouped_allocation=grouped,
        group_block_tables=tuple(
            GroupBlockTable(group.group_id, 16, block_ids[group.group_id])
            for group in layout.groups
        ),
    )

    target = TokenRange(256, 512)
    segment = TokenSegment(target, prompt[target.start : target.end])
    storage_hash = b"\x03" * 32
    record = CacheRecord(
        config.namespace,
        SHA256_FINGERPRINTER.fingerprint(config.namespace, segment.token_ids),
        segment.token_ids,
        TokenRange(1024, 1280),
        LMCACHE_CACHE_KEY_PREFIX + storage_hash.hex(),
    )
    match = VerifiedMatch(CandidateMatch(segment, record.fingerprint, record))
    candidate = VerifiedLmcacheCandidate.bind(
        LmcacheCandidate(
            TokenRange(0, 256),
            target,
            storage_hash,
            config.storage_model_name,
            query_digest(prompt),
        ),
        match,
        expected_namespace=config.namespace,
    )
    request_plan = RequestPlan(
        "opaque-request",
        len(prompt),
        (segment,),
        MatchPlan((match,), (), len(segment)),
    )
    handoff = RequestHandoffMetadata(
        METADATA_SCHEMA_VERSION,
        request_plan,
        RequestAllocation("opaque-request", 0, grouped),
    )
    metadata = SchedulerTransferMetadata(
        config.namespace,
        prompt,
        (candidate,),
        handoff,
        0,
        len(prompt),
        True,
        True,
    )
    scatter = plan_token_scatter(
        layout,
        blocks.group_block_tables,
        TokenTransfer(record.source_range, target),
    )
    load = WorkerLoadPlan(
        metadata,
        blocks,
        (CandidateScatterWork(0, candidate, scatter),),
    )
    chunks = tuple(
        StoreChunkWork(
            chunk_index,
            token_range,
            prompt[token_range.start : token_range.end],
            plan_token_scatter(
                layout,
                blocks.group_block_tables,
                TokenTransfer(token_range, token_range),
            ),
        )
        for chunk_index, token_range in enumerate(
            (TokenRange(0, 256), TokenRange(256, 512))
        )
    )
    return load, WorkerStorePlan(metadata, blocks, chunks)


@dataclass
class _FakeStagingRuntime:
    config: StagingConfig
    trace: list[str]
    fail_open: bool = False
    state: StagingState = StagingState.CREATED

    @property
    def tensor(self) -> object:
        if self.state is not StagingState.REGISTERED:
            raise RuntimeError("staging not locally owned")
        return "staging-tensor"

    def open(self) -> object:
        self.trace.append("staging.open")
        if self.fail_open:
            self.state = StagingState.FAILED
            raise RuntimeError("sensitive registration error")
        self.state = StagingState.REGISTERED
        return self.tensor

    @contextmanager
    def synchronous_transfer(
        self,
        *,
        direction: StagingTransferDirection,
        buffer_offset: int,
        token_extent: int,
    ) -> Iterator[StagingTransferLease]:
        if self.state is not StagingState.REGISTERED:
            raise RuntimeError("invalid staging state")
        self.trace.append(
            f"lease:{direction.value}:{buffer_offset}:{token_extent}"
        )
        self.state = StagingState.TRANSFER_ACTIVE
        try:
            yield StagingTransferLease(
                direction,
                buffer_offset,
                token_extent,
                self.config.token_capacity,
                b"event-handle",
            )
        finally:
            if self.state is StagingState.TRANSFER_ACTIVE:
                self.state = StagingState.REGISTERED

    def close(self) -> None:
        self.trace.append("staging.close")
        self.state = StagingState.CLOSED


class _RuntimeFactory:
    def __init__(self, trace: list[str], *, fail_open: bool = False) -> None:
        self.trace = trace
        self.fail_open = fail_open
        self.transport: StagingTransport | None = None
        self.runtime: _FakeStagingRuntime | None = None

    def __call__(
        self,
        config: StagingConfig,
        backend: StagingBackend,
        transport: StagingTransport,
    ) -> StagingRuntimeLike:
        self.transport = transport
        self.runtime = _FakeStagingRuntime(config, self.trace, self.fail_open)
        return self.runtime


class _FakeTransport:
    def __init__(self, config: LmcacheBlendTransportConfig, trace: list[str]) -> None:
        self.config = config
        self.trace = trace
        self.bad_namespace = False

    def open(self) -> None:
        self.trace.append("transport.open")

    def register_staging_buffer(self, registration: object) -> None:
        self.trace.append("transport.register")

    def unregister_staging_buffer(self) -> None:
        self.trace.append("transport.unregister")

    def retrieve_precomputed(
        self,
        token_ids: Sequence[int],
        verified_candidates: Sequence[VerifiedLmcacheCandidate],
        *,
        buffer_offset: int,
        event_ipc_handle: bytes,
        request_id: str,
    ) -> LmcacheRetrieveReceipt:
        self.trace.append(
            f"retrieve:{buffer_offset}:{len(token_ids)}:{len(verified_candidates)}"
        )
        assert event_ipc_handle == b"event-handle"
        return LmcacheRetrieveReceipt(256, 1)

    def store_precomputed(
        self,
        token_ids: Sequence[int],
        *,
        cache_namespace: CacheNamespace,
        document_source_range: TokenRange,
        buffer_offset: int,
        event_ipc_handle: bytes,
        request_id: str,
    ) -> LmcacheStoreReceipt:
        self.trace.append(f"store:{buffer_offset}:{len(token_ids)}")
        assert event_ipc_handle == b"event-handle"
        namespace = _namespace() if not self.bad_namespace else CacheNamespace(
            **{
                **dict(_namespace().canonical_fields()),
                "schema_version": 1,
                "adapter_revision": "wrong-adapter",
            }
        )
        tokens = tuple(token_ids)
        records = tuple(
            CacheRecord(
                namespace,
                SHA256_FINGERPRINTER.fingerprint(namespace, chunk),
                chunk,
                TokenRange(start, start + 256),
                LMCACHE_CACHE_KEY_PREFIX + (bytes([10 + start // 256]) * 32).hex(),
            )
            for start in range(0, len(tokens), 256)
            for chunk in (tokens[start : start + 256],)
        )
        return LmcacheStoreReceipt(len(tokens), len(records), True, records)

    def close(self) -> None:
        self.trace.append("transport.close")


class _FakeSidecar:
    def __init__(self, trace: list[str]) -> None:
        self.trace = trace
        self.batches: list[tuple[CacheRecord, ...]] = []

    def add_many(self, records: Sequence[CacheRecord]) -> int:
        batch = tuple(records)
        self.trace.append(f"sidecar.add_many:{len(batch)}")
        self.batches.append(batch)
        return len(batch)


class _FakeTensorOps:
    def __init__(self, trace: list[str]) -> None:
        self.trace = trace

    def shape(self, tensor: object) -> tuple[int, ...]:
        return (1,)

    def dtype_name(self, tensor: object) -> str:
        return "torch.bfloat16"

    def device_name(self, tensor: object) -> str:
        return "cuda:0"

    def paged_rows(
        self,
        tensor: object,
        *,
        component: int,
        block_id: int,
        block_offset: int,
        token_count: int,
    ) -> object:
        self.trace.append("view:paged")
        return "paged-view"

    def staging_rows(
        self,
        tensor: object,
        *,
        component: int,
        layer_index: int,
        token_start: int,
        token_count: int,
    ) -> object:
        self.trace.append(f"view:staging:{token_start}:{token_count}")
        return "staging-view"

    def reshape(self, tensor: object, shape: tuple[int, ...]) -> object:
        self.trace.append("view:reshape")
        return tensor

    def copy(self, destination: object, source: object) -> None:
        self.trace.append("tensor.copy")

    def synchronize(self, tensor: object) -> None:
        self.trace.append("tensor.synchronize")


class _FakeCorrector:
    def __init__(self, trace: list[str]) -> None:
        self.trace = trace
        self.positions: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def __call__(
        self,
        key_rows: object,
        *,
        source_positions: tuple[int, ...],
        target_positions: tuple[int, ...],
        layer_index: int,
    ) -> object:
        self.trace.append("yarn.correct")
        self.positions.append((source_positions, target_positions))
        return "corrected-key"


class _FakeDataPlane:
    def __init__(
        self,
        ops: TensorOps,
        trace: list[str],
        instance_index: int,
    ) -> None:
        self.ops = ops
        self.trace = trace
        self.label = "active" if instance_index == 0 else "preflight"
        self.position_correction_latency_seconds = (
            0.25 if self.label == "active" else 0.0
        )

    @staticmethod
    def _ranges(
        spans: Sequence[LayerTokenScatterSpan],
    ) -> tuple[tuple[int, ...], tuple[int, ...], int]:
        first_layer = sorted(
            (span for span in spans if span.layer_index == 0),
            key=lambda span: span.target_range.start,
        )
        source = tuple(
            position
            for span in first_layer
            for position in range(span.source_range.start, span.source_range.end)
        )
        target = tuple(
            position
            for span in first_layer
            for position in range(span.target_range.start, span.target_range.end)
        )
        return source, target, len(target)

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
        source, target, logical = self._ranges(layer_spans)
        self.trace.append(
            f"{self.label}.scatter:{retrieval_buffer_offset}:{query_token_count}"
        )
        self.ops.staging_rows(
            staging,
            component=0,
            layer_index=0,
            token_start=retrieval_buffer_offset + target[0],
            token_count=logical,
        )
        correct_key_positions(
            "key-rows",
            source_positions=source,
            target_positions=target,
            layer_index=0,
        )
        self.ops.copy("paged", "staging")
        self.ops.synchronize("paged")
        rows = logical * 24
        return DataPlaneReceipt(
            TransferDirection.LOAD_FROM_STAGING,
            logical,
            rows,
            len(layer_spans),
            rows,
            rows,
            rows,
            position_correction_latency_seconds=(
                self.position_correction_latency_seconds
            ),
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
        _source, _target, logical = self._ranges(layer_spans)
        self.trace.append(f"{self.label}.gather:{store_buffer_offset}")
        self.ops.staging_rows(
            staging,
            component=0,
            layer_index=0,
            token_start=store_buffer_offset,
            token_count=logical,
        )
        self.ops.copy("staging", "paged")
        self.ops.synchronize("staging")
        rows = logical * 24
        return DataPlaneReceipt(
            TransferDirection.STORE_TO_STAGING,
            logical,
            rows,
            len(layer_spans),
            0,
            rows,
            rows,
        )


class _DataPlaneFactory:
    def __init__(self, trace: list[str]) -> None:
        self.trace = trace
        self.instances: list[_FakeDataPlane] = []

    def __call__(self, tensor_ops: TensorOps) -> DataPlaneOperations:
        instance = _FakeDataPlane(tensor_ops, self.trace, len(self.instances))
        self.instances.append(instance)
        return instance


@dataclass
class _Fixture:
    bridge: GptOssWorkerBridge
    transport: _FakeTransport
    sidecar: _FakeSidecar
    runtime_factory: _RuntimeFactory
    corrector: _FakeCorrector
    trace: list[str]


def _fixture(
    *,
    buffer_config: WorkerBridgeBufferConfig | None = None,
    staging_capacity: int = 1024,
    fail_open: bool = False,
) -> _Fixture:
    trace: list[str] = []
    config = _transport_config()
    transport = _FakeTransport(config, trace)
    sidecar = _FakeSidecar(trace)
    runtime_factory = _RuntimeFactory(trace, fail_open=fail_open)
    data_factory = _DataPlaneFactory(trace)
    corrector = _FakeCorrector(trace)
    bridge = GptOssWorkerBridge(
        staging_config=StagingConfig(17, staging_capacity, "cuda:0"),
        staging_backend=cast(StagingBackend, object()),
        transport=cast(WorkerLmcacheTransport, transport),
        sidecar=sidecar,
        tensor_ops=_FakeTensorOps(trace),
        paged_caches={_layer_name(index): object() for index in range(24)},
        correct_key_positions=corrector,
        buffer_config=buffer_config,
        staging_factory=runtime_factory,
        data_plane_factory=data_factory,
    )
    return _Fixture(
        bridge, transport, sidecar, runtime_factory, corrector, trace
    )


def _assert_error(
    code: WorkerBridgeErrorCode, operation: object
) -> WorkerBridgeError:
    with pytest.raises(WorkerBridgeError) as caught:
        cast("callable", operation)()
    assert caught.value.code is code
    assert "opaque-request" not in str(caught.value)
    return caught.value


def test_bridge_structurally_satisfies_both_runtime_protocols() -> None:
    fixture = _fixture()
    storage: WorkerStorage = fixture.bridge
    data_plane: WorkerDataPlane = fixture.bridge
    assert storage is data_plane


def test_load_preflight_is_read_only_and_preserves_offset_and_yarn_positions() -> (
    None
):
    fixture = _fixture(buffer_config=WorkerBridgeBufferConfig(32, 32))
    load, _store = _plans(fixture.transport.config)
    fixture.bridge.open()
    assert fixture.runtime_factory.transport is fixture.transport
    fixture.trace.clear()

    fixture.bridge.preflight_retrieve(load)
    fixture.bridge.preflight_scatter(load)

    assert "tensor.copy" not in fixture.trace
    assert "yarn.correct" in fixture.trace
    assert "tensor.synchronize" in fixture.trace
    assert "preflight.scatter:32:600" in fixture.trace
    assert "view:staging:288:256" in fixture.trace
    source, target = fixture.corrector.positions[-1]
    assert source == tuple(range(1024, 1280))
    assert target == tuple(range(256, 512))

    receipt = fixture.bridge.retrieve_verified(load)
    fixture.bridge.scatter_retrieved(load)

    assert receipt == LmcacheRetrieveReceipt(256, 1)
    assert "lease:retrieve:32:600" in fixture.trace
    assert "retrieve:32:600:1" in fixture.trace
    assert "active.scatter:32:600" in fixture.trace
    assert fixture.trace.count("tensor.copy") == 1
    assert fixture.bridge.position_correction_latency_seconds == pytest.approx(0.25)


def test_same_staging_region_is_reused_sequentially_for_load_then_store() -> None:
    fixture = _fixture(buffer_config=WorkerBridgeBufferConfig(0, 0))
    load, store = _plans(fixture.transport.config)
    fixture.bridge.open()
    fixture.trace.clear()

    fixture.bridge.preflight_retrieve(load)
    fixture.bridge.preflight_scatter(load)
    fixture.bridge.retrieve_verified(load)
    fixture.bridge.scatter_retrieved(load)
    fixture.bridge.preflight_gather(store)
    fixture.bridge.preflight_store(store)

    assert fixture.trace.count("tensor.copy") == 1
    assert "preflight.gather:0" in fixture.trace
    assert "preflight.gather:256" in fixture.trace
    fixture.bridge.gather_recomputed(store)
    receipt = fixture.bridge.store_precomputed(store)
    inserted = fixture.bridge.publish_sidecar_records_atomically(
        receipt.sidecar_records
    )

    assert fixture.trace.count("tensor.copy") == 3
    assert "lease:retrieve:0:600" in fixture.trace
    assert "lease:store:0:512" in fixture.trace
    assert "store:0:512" in fixture.trace
    assert inserted == 2
    assert fixture.sidecar.batches == [receipt.sidecar_records]


def test_call_order_and_capacity_fail_before_worker_mutation() -> None:
    fixture = _fixture()
    load, store = _plans(fixture.transport.config)
    fixture.bridge.open()
    fixture.trace.clear()

    _assert_error(
        WorkerBridgeErrorCode.PLAN_ORDER_MISMATCH,
        lambda: fixture.bridge.retrieve_verified(load),
    )
    _assert_error(
        WorkerBridgeErrorCode.PLAN_ORDER_MISMATCH,
        lambda: fixture.bridge.preflight_store(store),
    )
    assert "tensor.copy" not in fixture.trace
    assert not any(entry.startswith("retrieve:") for entry in fixture.trace)
    assert not any(entry.startswith("store:") for entry in fixture.trace)

    offset_fixture = _fixture(
        buffer_config=WorkerBridgeBufferConfig(512, 0), staging_capacity=1024
    )
    offset_load, _ = _plans(offset_fixture.transport.config)
    offset_fixture.bridge.open()
    offset_fixture.trace.clear()
    _assert_error(
        WorkerBridgeErrorCode.STAGING_RANGE_OUT_OF_BOUNDS,
        lambda: offset_fixture.bridge.preflight_retrieve(offset_load),
    )
    assert "tensor.copy" not in offset_fixture.trace


def test_store_receipt_records_are_independently_reverified_before_return() -> None:
    fixture = _fixture()
    _load, store = _plans(fixture.transport.config)
    fixture.bridge.open()
    fixture.bridge.preflight_gather(store)
    fixture.bridge.preflight_store(store)
    fixture.bridge.gather_recomputed(store)
    fixture.transport.bad_namespace = True

    _assert_error(
        WorkerBridgeErrorCode.RECEIPT_MISMATCH,
        lambda: fixture.bridge.store_precomputed(store),
    )
    assert fixture.sidecar.batches == []
    assert not any(entry.startswith("sidecar.add_many") for entry in fixture.trace)


def test_direct_atomic_publish_rechecks_fingerprint_source_and_cache_key() -> None:
    fixture = _fixture()
    fixture.bridge.open()
    namespace = fixture.transport.config.namespace
    tokens = tuple(range(256))
    valid = CacheRecord(
        namespace,
        SHA256_FINGERPRINTER.fingerprint(namespace, tokens),
        tokens,
        TokenRange(0, 256),
        LMCACHE_CACHE_KEY_PREFIX + (b"\x09" * 32).hex(),
    )
    invalid = CacheRecord(
        namespace,
        valid.fingerprint,
        tokens,
        TokenRange(0, 256),
        "untrusted-key",
    )

    _assert_error(
        WorkerBridgeErrorCode.INVALID_PLAN,
        lambda: fixture.bridge.publish_sidecar_records_atomically((invalid,)),
    )
    assert fixture.sidecar.batches == []
    assert fixture.bridge.publish_sidecar_records_atomically((valid,)) == 1


def test_direct_atomic_publish_rejects_nonzero_compact_source_start() -> None:
    fixture = _fixture()
    fixture.bridge.open()
    namespace = fixture.transport.config.namespace
    tokens = tuple(range(256))
    record = CacheRecord(
        namespace,
        SHA256_FINGERPRINTER.fingerprint(namespace, tokens),
        tokens,
        TokenRange(256, 512),
        LMCACHE_CACHE_KEY_PREFIX + (b"\x0a" * 32).hex(),
    )

    _assert_error(
        WorkerBridgeErrorCode.INVALID_PLAN,
        lambda: fixture.bridge.publish_sidecar_records_atomically((record,)),
    )
    assert fixture.sidecar.batches == []


def test_failed_open_can_be_closed_once_without_bridge_double_unregister() -> None:
    fixture = _fixture(fail_open=True)

    _assert_error(WorkerBridgeErrorCode.OPEN_FAILED, fixture.bridge.open)
    assert fixture.bridge.state is WorkerBridgeState.FAILED
    fixture.bridge.close()
    fixture.bridge.close()

    assert fixture.trace.count("staging.close") == 1
    assert fixture.trace.count("transport.close") == 1
    assert fixture.trace.count("transport.unregister") == 0
    assert fixture.bridge.state is WorkerBridgeState.CLOSED
