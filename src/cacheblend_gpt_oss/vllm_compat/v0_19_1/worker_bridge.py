# SPDX-License-Identifier: Apache-2.0
"""Concrete worker bridge for the pinned 100%-recompute transfer path.

The bridge makes :class:`~.transfer_runtime.WorkerStorage` and
:class:`~.transfer_runtime.WorkerDataPlane` concrete without importing vLLM,
LMCache, or Torch at module import time.  Production construction defaults to
the audited staging runtime and GPT-OSS data plane; CPU tests inject fakes.

Exact API boundaries:

* LMCache 0.4.3 precomputed retrieval places rows at
  ``retrieval_buffer_offset + current_query_position`` and its synchronous
  future waits for the returned CUDA event:
  https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/v1/multiprocess/blend_server_v2.py#L597-L687
* precomputed stores read a compact contiguous staging range and return one
  hash/sidecar identity per complete 256-token chunk:
  https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/v1/multiprocess/blend_server_v2.py#L444-L513
* this project's data plane uses target positions for staging addresses and
  source/target positions only for out-of-place YaRN correction; gather uses a
  compact explicit store offset.  Its public methods validate all views before
  their first copy.
* ``SqliteSidecarIndex.add_many`` validates the complete record batch and uses
  one immediate SQLite transaction, so publication is all-or-nothing.

The existing data plane has no separate public dry-run method.  To preserve a
whole-request no-mutation preflight, this bridge invokes that same public data
plane through :class:`_ReadOnlyTensorOps`: all real tensor/view/correction and
device checks run, while its copy operation is suppressed.  Only after every
candidate or chunk passes does the bridge allow LMCache or paged-cache mutation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum
from typing import NoReturn, Protocol

from cacheblend_gpt_oss.gpt_oss.layout import (
    GPT_OSS_NUM_LAYERS,
    LayerTokenScatterSpan,
)
from cacheblend_gpt_oss.planner.fingerprint import SHA256_FINGERPRINTER
from cacheblend_gpt_oss.planner.models import CacheNamespace, CacheRecord, TokenRange
from cacheblend_gpt_oss.storage.lmcache_types import (
    LMCACHE_CACHE_KEY_PREFIX,
    LMCACHE_CHUNK_SIZE,
    LmcacheBlendTransportConfig,
    LmcacheRetrieveReceipt,
    LmcacheStoreReceipt,
    VerifiedLmcacheCandidate,
    validate_request_id,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.data_plane import (
    DataPlaneReceipt,
    GptOssDataPlane,
    KeyPositionCorrector,
    TensorOps,
    TransferDirection,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.staging import (
    StagingBackend,
    StagingConfig,
    StagingRuntime,
    StagingState,
    StagingTransferDirection,
    StagingTransferLease,
    StagingTransport,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.transfer_runtime import (
    WorkerLoadPlan,
    WorkerStorePlan,
)

_MAX_LMCACHE_TOKEN_ID = (1 << 32) - 1


class WorkerBridgeErrorCode(str, Enum):
    """Bounded bridge failures safe for logs and aggregate metrics."""

    INVALID_CONFIG = "invalid_config"
    INVALID_STATE = "invalid_state"
    INVALID_PLAN = "invalid_plan"
    PLAN_ORDER_MISMATCH = "plan_order_mismatch"
    STAGING_RANGE_OUT_OF_BOUNDS = "staging_range_out_of_bounds"
    TRANSPORT_CONFIG_MISMATCH = "transport_config_mismatch"
    RECEIPT_MISMATCH = "receipt_mismatch"
    OPEN_FAILED = "open_failed"
    CLOSE_FAILED = "close_failed"


class WorkerBridgeError(RuntimeError):
    """Fail-closed bridge error whose message contains no request data."""

    def __init__(self, code: WorkerBridgeErrorCode) -> None:
        self.code = code
        super().__init__(f"worker transfer bridge failure: {code.value}")


def _fail(code: WorkerBridgeErrorCode) -> NoReturn:
    raise WorkerBridgeError(code)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True, slots=True)
class WorkerBridgeBufferConfig:
    """Fixed staging regions used sequentially by load and store operations."""

    retrieval_buffer_offset: int = 0
    store_buffer_offset: int = 0

    def __post_init__(self) -> None:
        if (
            not _is_int(self.retrieval_buffer_offset)
            or self.retrieval_buffer_offset < 0
            or not _is_int(self.store_buffer_offset)
            or self.store_buffer_offset < 0
        ):
            _fail(WorkerBridgeErrorCode.INVALID_CONFIG)


class WorkerLmcacheTransport(StagingTransport, Protocol):
    """Exact synchronous transport methods used by the production bridge."""

    @property
    def config(self) -> LmcacheBlendTransportConfig:
        """Return the pinned immutable transport configuration."""

    def open(self) -> None:
        """Open and attest the pinned LMCache server connection."""

    def retrieve_precomputed(
        self,
        token_ids: Sequence[int],
        verified_candidates: Sequence[VerifiedLmcacheCandidate],
        *,
        buffer_offset: int,
        event_ipc_handle: bytes,
        request_id: str,
    ) -> LmcacheRetrieveReceipt:
        """Synchronously retrieve exact verified candidates into staging."""

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
        """Synchronously store compact complete chunks from staging."""

    def close(self) -> None:
        """Close the transport and its message-queue resources."""


class AtomicRecordWriter(Protocol):
    """Sidecar subset whose ``add_many`` operation is one atomic transaction."""

    def add_many(self, records: Sequence[CacheRecord]) -> int:
        """Atomically validate and publish all records, or publish none."""


class StagingRuntimeLike(Protocol):
    """Injectable surface of :class:`StagingRuntime` used by this bridge."""

    @property
    def state(self) -> StagingState:
        """Return the current staging lifecycle state."""

    @property
    def config(self) -> StagingConfig:
        """Return the staging allocation configuration."""

    @property
    def tensor(self) -> object:
        """Return staging only while it is locally owned and registered."""

    def open(self) -> object:
        """Allocate and register the staging tensor."""

    def synchronous_transfer(
        self,
        *,
        direction: StagingTransferDirection,
        buffer_offset: int,
        token_extent: int,
    ) -> AbstractContextManager[StagingTransferLease]:
        """Lease the event handle around exactly one synchronous MQ call."""

    def close(self) -> None:
        """Unregister and release staging ownership."""


class StagingRuntimeFactory(Protocol):
    """Factory injection used only to keep CPU tests dependency-free."""

    def __call__(
        self,
        config: StagingConfig,
        backend: StagingBackend,
        transport: StagingTransport,
    ) -> StagingRuntimeLike:
        """Build a runtime bound to the exact supplied transport object."""


class DataPlaneOperations(Protocol):
    """Concrete surface shared by real and CPU-fake GPT-OSS data planes."""

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
        """Scatter one verified candidate into all hybrid groups."""

    def gather_precomputed_kv(
        self,
        *,
        paged_caches: Mapping[str, object],
        staging: object,
        layer_spans: Sequence[LayerTokenScatterSpan],
        document_target_range: TokenRange,
        store_buffer_offset: int,
    ) -> DataPlaneReceipt:
        """Gather one complete prompt chunk into compact staging."""


class DataPlaneFactory(Protocol):
    """Factory for the active and no-copy validation data planes."""

    def __call__(self, tensor_ops: TensorOps) -> DataPlaneOperations:
        """Build a data plane over the supplied tensor operations."""


class WorkerBridgeState(str, Enum):
    """Worker-level ownership state for transport and staging resources."""

    CREATED = "created"
    READY = "ready"
    FAILED = "failed"
    CLOSED = "closed"


def _create_staging_runtime(
    config: StagingConfig,
    backend: StagingBackend,
    transport: StagingTransport,
) -> StagingRuntimeLike:
    return StagingRuntime(config, backend, transport)


def _create_data_plane(tensor_ops: TensorOps) -> DataPlaneOperations:
    return GptOssDataPlane(tensor_ops)


class _ReadOnlyTensorOps:
    """Delegate all inspection/view work while suppressing destination copies."""

    def __init__(self, delegate: TensorOps) -> None:
        self._delegate = delegate

    def shape(self, tensor: object) -> tuple[int, ...]:
        return self._delegate.shape(tensor)

    def dtype_name(self, tensor: object) -> str:
        return self._delegate.dtype_name(tensor)

    def device_name(self, tensor: object) -> str:
        return self._delegate.device_name(tensor)

    def paged_rows(
        self,
        tensor: object,
        *,
        component: int,
        block_id: int,
        block_offset: int,
        token_count: int,
    ) -> object:
        return self._delegate.paged_rows(
            tensor,
            component=component,
            block_id=block_id,
            block_offset=block_offset,
            token_count=token_count,
        )

    def staging_rows(
        self,
        tensor: object,
        *,
        component: int,
        layer_index: int,
        token_start: int,
        token_count: int,
    ) -> object:
        return self._delegate.staging_rows(
            tensor,
            component=component,
            layer_index=layer_index,
            token_start=token_start,
            token_count=token_count,
        )

    def reshape(self, tensor: object, shape: tuple[int, ...]) -> object:
        return self._delegate.reshape(tensor, shape)

    def copy(self, destination: object, source: object) -> None:
        """Intentionally suppress the only tensor mutation in the data plane."""

    def synchronize(self, tensor: object) -> None:
        # Validation may enqueue out-of-place YaRN arithmetic.  Synchronizing
        # makes its failure observable before LMCache or paged-cache mutation.
        self._delegate.synchronize(tensor)


class GptOssWorkerBridge:
    """WorkerStorage/WorkerDataPlane implementation for one pinned GPU worker.

    The bridge constructs :class:`StagingRuntime` itself with the very same
    transport object used for retrieval and storage.  This identity guarantee
    prevents a registered buffer from being confused with another transport's
    CUDA IPC registration.
    """

    def __init__(
        self,
        *,
        staging_config: StagingConfig,
        staging_backend: StagingBackend,
        transport: WorkerLmcacheTransport,
        sidecar: AtomicRecordWriter,
        tensor_ops: TensorOps,
        paged_caches: Mapping[str, object],
        correct_key_positions: KeyPositionCorrector,
        buffer_config: WorkerBridgeBufferConfig | None = None,
        staging_factory: StagingRuntimeFactory = _create_staging_runtime,
        data_plane_factory: DataPlaneFactory = _create_data_plane,
    ) -> None:
        selected_buffer_config = (
            WorkerBridgeBufferConfig() if buffer_config is None else buffer_config
        )
        if not isinstance(staging_config, StagingConfig) or not isinstance(
            selected_buffer_config, WorkerBridgeBufferConfig
        ):
            _fail(WorkerBridgeErrorCode.INVALID_CONFIG)
        try:
            caches = dict(paged_caches)
        except (TypeError, ValueError):
            _fail(WorkerBridgeErrorCode.INVALID_CONFIG)
        expected_names = {
            f"model.layers.{layer_index}.attn.attn"
            for layer_index in range(GPT_OSS_NUM_LAYERS)
        }
        if set(caches) != expected_names or any(
            not isinstance(name, str) for name in caches
        ):
            _fail(WorkerBridgeErrorCode.INVALID_CONFIG)
        if not callable(correct_key_positions):
            _fail(WorkerBridgeErrorCode.INVALID_CONFIG)

        self._transport = transport
        self._sidecar = sidecar
        self._buffer_config = selected_buffer_config
        self._paged_caches: Mapping[str, object] = caches
        self._correct_key_positions = correct_key_positions
        try:
            self._staging = staging_factory(
                staging_config, staging_backend, transport
            )
            self._data_plane = data_plane_factory(tensor_ops)
            self._preflight_data_plane = data_plane_factory(
                _ReadOnlyTensorOps(tensor_ops)
            )
        except Exception as exc:
            raise WorkerBridgeError(WorkerBridgeErrorCode.INVALID_CONFIG) from exc
        self._state = WorkerBridgeState.CREATED
        self._load_storage_preflight: WorkerLoadPlan | None = None
        self._load_data_preflight: WorkerLoadPlan | None = None
        self._retrieved_plan: WorkerLoadPlan | None = None
        self._gather_preflight: WorkerStorePlan | None = None
        self._store_preflight: WorkerStorePlan | None = None
        self._gathered_plan: WorkerStorePlan | None = None

    @property
    def state(self) -> WorkerBridgeState:
        return self._state

    @property
    def staging_runtime(self) -> StagingRuntimeLike:
        """Expose lifecycle state/tensor ownership, never the transport binding."""

        return self._staging

    def open(self) -> object:
        """Open transport first, then allocate/register its exact staging tensor."""

        if self._state is not WorkerBridgeState.CREATED:
            _fail(WorkerBridgeErrorCode.INVALID_STATE)
        try:
            self._transport.open()
            tensor = self._staging.open()
        except Exception as exc:
            self._state = WorkerBridgeState.FAILED
            raise WorkerBridgeError(WorkerBridgeErrorCode.OPEN_FAILED) from exc
        self._state = WorkerBridgeState.READY
        return tensor

    def close(self) -> None:
        """Unregister staging, then close transport; successful close is idempotent."""

        if self._state is WorkerBridgeState.CLOSED:
            return
        failed = False
        try:
            self._staging.close()
        except Exception:
            failed = True
        try:
            self._transport.close()
        except Exception:
            failed = True
        self._clear_plans()
        if failed:
            self._state = WorkerBridgeState.FAILED
            _fail(WorkerBridgeErrorCode.CLOSE_FAILED)
        self._state = WorkerBridgeState.CLOSED

    def preflight_retrieve(self, plan: WorkerLoadPlan) -> None:
        """Validate storage identity/ranges before data-plane dry preflight."""

        self._require_ready()
        if self._retrieved_plan is not None:
            _fail(WorkerBridgeErrorCode.INVALID_STATE)
        self._load_storage_preflight = None
        self._load_data_preflight = None
        self._validate_load_plan(plan)
        self._validate_transport(plan)
        self._validate_staging_range(
            self._buffer_config.retrieval_buffer_offset,
            plan.metadata.prompt_token_count,
        )
        self._load_storage_preflight = plan

    def preflight_scatter(self, plan: WorkerLoadPlan) -> None:
        """Dry-run every real view and YaRN correction before retrieval."""

        self._require_plan(self._load_storage_preflight, plan)
        staging = self._staging.tensor
        for candidate in plan.candidates:
            receipt = self._preflight_data_plane.scatter_retrieved_kv(
                staging=staging,
                paged_caches=self._paged_caches,
                layer_spans=candidate.scatter_plan.layer_spans,
                retrieval_buffer_offset=(
                    self._buffer_config.retrieval_buffer_offset
                ),
                query_token_count=plan.metadata.prompt_token_count,
                correct_key_positions=self._correct_key_positions,
            )
            self._validate_data_receipt(
                receipt,
                TransferDirection.LOAD_FROM_STAGING,
                LMCACHE_CHUNK_SIZE,
                len(candidate.scatter_plan.layer_spans),
            )
        self._load_data_preflight = plan

    def retrieve_verified(self, plan: WorkerLoadPlan) -> LmcacheRetrieveReceipt:
        """Retrieve only after storage and every scatter call were preflighted."""

        self._require_plan(self._load_storage_preflight, plan)
        self._require_plan(self._load_data_preflight, plan)
        try:
            with self._staging.synchronous_transfer(
                direction=StagingTransferDirection.RETRIEVE,
                buffer_offset=self._buffer_config.retrieval_buffer_offset,
                token_extent=plan.metadata.prompt_token_count,
            ) as lease:
                self._validate_lease(
                    lease,
                    StagingTransferDirection.RETRIEVE,
                    self._buffer_config.retrieval_buffer_offset,
                    plan.metadata.prompt_token_count,
                )
                receipt = self._transport.retrieve_precomputed(
                    plan.metadata.prompt_token_ids,
                    tuple(
                        candidate.verified_candidate
                        for candidate in plan.candidates
                    ),
                    buffer_offset=lease.buffer_offset,
                    event_ipc_handle=lease.event_ipc_handle,
                    request_id=plan.metadata.request_id,
                )
            self._validate_retrieve_receipt(receipt, plan)
        except Exception:
            self._retrieved_plan = None
            raise
        finally:
            self._load_storage_preflight = None
            self._load_data_preflight = None
        self._retrieved_plan = plan
        return receipt

    def scatter_retrieved(self, plan: WorkerLoadPlan) -> None:
        """Scatter each already-preflighted candidate using exact target offsets."""

        self._require_plan(self._retrieved_plan, plan)
        staging = self._staging.tensor
        try:
            for candidate in plan.candidates:
                receipt = self._data_plane.scatter_retrieved_kv(
                    staging=staging,
                    paged_caches=self._paged_caches,
                    layer_spans=candidate.scatter_plan.layer_spans,
                    retrieval_buffer_offset=(
                        self._buffer_config.retrieval_buffer_offset
                    ),
                    query_token_count=plan.metadata.prompt_token_count,
                    correct_key_positions=self._correct_key_positions,
                )
                self._validate_data_receipt(
                    receipt,
                    TransferDirection.LOAD_FROM_STAGING,
                    LMCACHE_CHUNK_SIZE,
                    len(candidate.scatter_plan.layer_spans),
                )
        finally:
            self._retrieved_plan = None

    def preflight_gather(self, plan: WorkerStorePlan) -> None:
        """Dry-run every complete-chunk gather before staging mutation."""

        self._require_ready()
        if self._gathered_plan is not None:
            _fail(WorkerBridgeErrorCode.INVALID_STATE)
        self._gather_preflight = None
        self._store_preflight = None
        self._validate_store_plan(plan)
        staging = self._staging.tensor
        for chunk in plan.chunks:
            receipt = self._preflight_data_plane.gather_precomputed_kv(
                paged_caches=self._paged_caches,
                staging=staging,
                layer_spans=chunk.gather_plan.layer_spans,
                document_target_range=chunk.token_range,
                store_buffer_offset=self._chunk_store_offset(chunk.chunk_index),
            )
            self._validate_data_receipt(
                receipt,
                TransferDirection.STORE_TO_STAGING,
                LMCACHE_CHUNK_SIZE,
                len(chunk.gather_plan.layer_spans),
            )
        self._gather_preflight = plan

    def preflight_store(self, plan: WorkerStorePlan) -> None:
        """Validate transport/staging store inputs after all gather dry-runs."""

        self._require_plan(self._gather_preflight, plan)
        self._validate_transport(plan)
        self._validate_staging_range(
            self._buffer_config.store_buffer_offset, plan.expected_tokens
        )
        self._store_preflight = plan

    def gather_recomputed(self, plan: WorkerStorePlan) -> None:
        """Gather exact post-forward KV into a contiguous staging region."""

        self._require_plan(self._gather_preflight, plan)
        self._require_plan(self._store_preflight, plan)
        staging = self._staging.tensor
        try:
            for chunk in plan.chunks:
                receipt = self._data_plane.gather_precomputed_kv(
                    paged_caches=self._paged_caches,
                    staging=staging,
                    layer_spans=chunk.gather_plan.layer_spans,
                    document_target_range=chunk.token_range,
                    store_buffer_offset=self._chunk_store_offset(
                        chunk.chunk_index
                    ),
                )
                self._validate_data_receipt(
                    receipt,
                    TransferDirection.STORE_TO_STAGING,
                    LMCACHE_CHUNK_SIZE,
                    len(chunk.gather_plan.layer_spans),
                )
        except Exception:
            self._gathered_plan = None
            raise
        finally:
            self._gather_preflight = None
        self._gathered_plan = plan

    def store_precomputed(self, plan: WorkerStorePlan) -> LmcacheStoreReceipt:
        """Store the compact gathered prefix through one synchronous event lease."""

        self._require_plan(self._store_preflight, plan)
        self._require_plan(self._gathered_plan, plan)
        try:
            with self._staging.synchronous_transfer(
                direction=StagingTransferDirection.STORE,
                buffer_offset=self._buffer_config.store_buffer_offset,
                token_extent=plan.expected_tokens,
            ) as lease:
                self._validate_lease(
                    lease,
                    StagingTransferDirection.STORE,
                    self._buffer_config.store_buffer_offset,
                    plan.expected_tokens,
                )
                receipt = self._transport.store_precomputed(
                    plan.token_ids,
                    cache_namespace=plan.metadata.cache_namespace,
                    document_source_range=plan.source_range,
                    buffer_offset=lease.buffer_offset,
                    event_ipc_handle=lease.event_ipc_handle,
                    request_id=plan.metadata.request_id,
                )
            self._validate_store_receipt(receipt, plan)
        finally:
            self._store_preflight = None
            self._gathered_plan = None
        return receipt

    def publish_sidecar_records_atomically(
        self, records: tuple[CacheRecord, ...]
    ) -> int:
        """Delegate one already-verified batch to sidecar ``add_many``."""

        self._require_ready()
        if not self._valid_record_batch(records):
            _fail(WorkerBridgeErrorCode.INVALID_PLAN)
        return self._sidecar.add_many(records)

    def _validate_load_plan(self, plan: WorkerLoadPlan) -> None:
        if (
            not isinstance(plan, WorkerLoadPlan)
            or not plan.metadata.transfer_eligible
            or not plan.candidates
            or len(plan.candidates) != len(plan.metadata.verified_candidates)
            or plan.expected_tokens
            != len(plan.candidates) * LMCACHE_CHUNK_SIZE
        ):
            _fail(WorkerBridgeErrorCode.INVALID_PLAN)
        for index, work in enumerate(plan.candidates):
            verified = plan.metadata.verified_candidates[index]
            if (
                work.candidate_index != index
                or work.verified_candidate != verified
                or work.scatter_plan.transfer.source_range
                != verified.match.record.source_range
                or work.scatter_plan.transfer.target_range
                != verified.candidate.target_range
            ):
                _fail(WorkerBridgeErrorCode.INVALID_PLAN)
        self._validate_request_tokens(plan.metadata.prompt_token_ids)
        validate_request_id(plan.metadata.request_id)

    def _validate_store_plan(self, plan: WorkerStorePlan) -> None:
        if (
            not isinstance(plan, WorkerStorePlan)
            or not plan.metadata.store_eligible
            or not plan.chunks
            or plan.expected_tokens != len(plan.chunks) * LMCACHE_CHUNK_SIZE
            or plan.expected_tokens != plan.metadata.complete_store_token_count
            or plan.token_ids
            != plan.metadata.prompt_token_ids[: plan.expected_tokens]
            or plan.source_range != TokenRange(0, plan.expected_tokens)
        ):
            _fail(WorkerBridgeErrorCode.INVALID_PLAN)
        for index, chunk in enumerate(plan.chunks):
            expected_range = TokenRange(
                index * LMCACHE_CHUNK_SIZE,
                (index + 1) * LMCACHE_CHUNK_SIZE,
            )
            if (
                chunk.chunk_index != index
                or chunk.token_range != expected_range
                or chunk.token_ids
                != plan.metadata.prompt_token_ids[
                    expected_range.start : expected_range.end
                ]
                or chunk.gather_plan.transfer.source_range != expected_range
                or chunk.gather_plan.transfer.target_range != expected_range
            ):
                _fail(WorkerBridgeErrorCode.INVALID_PLAN)
        self._validate_request_tokens(plan.token_ids)
        validate_request_id(plan.metadata.request_id)

    def _validate_transport(self, plan: WorkerLoadPlan | WorkerStorePlan) -> None:
        try:
            config = self._transport.config
        except Exception as exc:
            raise WorkerBridgeError(
                WorkerBridgeErrorCode.TRANSPORT_CONFIG_MISMATCH
            ) from exc
        if (
            not isinstance(config, LmcacheBlendTransportConfig)
            or config.namespace != plan.metadata.cache_namespace
            or config.chunk_size != LMCACHE_CHUNK_SIZE
        ):
            _fail(WorkerBridgeErrorCode.TRANSPORT_CONFIG_MISMATCH)
        if isinstance(plan, WorkerLoadPlan) and any(
            work.verified_candidate.candidate.storage_model_name
            != config.storage_model_name
            for work in plan.candidates
        ):
            _fail(WorkerBridgeErrorCode.TRANSPORT_CONFIG_MISMATCH)

    def _validate_staging_range(self, start: int, length: int) -> None:
        self._require_ready()
        capacity = self._staging.config.token_capacity
        if (
            not _is_int(capacity)
            or capacity < 0
            or start + length > capacity
        ):
            _fail(WorkerBridgeErrorCode.STAGING_RANGE_OUT_OF_BOUNDS)
        # Access also proves that registered staging is locally owned now.
        staging_tensor = self._staging.tensor
        if staging_tensor is None:
            _fail(WorkerBridgeErrorCode.INVALID_STATE)

    @staticmethod
    def _validate_request_tokens(token_ids: tuple[int, ...]) -> None:
        if not token_ids or any(
            not _is_int(token_id) or not 0 <= token_id <= _MAX_LMCACHE_TOKEN_ID
            for token_id in token_ids
        ):
            _fail(WorkerBridgeErrorCode.INVALID_PLAN)

    @staticmethod
    def _validate_retrieve_receipt(
        receipt: object, plan: WorkerLoadPlan
    ) -> None:
        if (
            not isinstance(receipt, LmcacheRetrieveReceipt)
            or not _is_int(receipt.retrieved_tokens)
            or not _is_int(receipt.retrieved_chunks)
            or receipt.retrieved_tokens != plan.expected_tokens
            or receipt.retrieved_chunks != plan.expected_chunks
        ):
            _fail(WorkerBridgeErrorCode.RECEIPT_MISMATCH)

    @staticmethod
    def _validate_store_receipt(receipt: object, plan: WorkerStorePlan) -> None:
        if (
            not isinstance(receipt, LmcacheStoreReceipt)
            or receipt.stored_tokens != plan.expected_tokens
            or receipt.stored_chunks != plan.expected_chunks
            or not receipt.candidate_lookup_required
            or len(receipt.sidecar_records) != plan.expected_chunks
        ):
            _fail(WorkerBridgeErrorCode.RECEIPT_MISMATCH)
        for chunk, record in zip(
            plan.chunks, receipt.sidecar_records, strict=True
        ):
            if not GptOssWorkerBridge._valid_record(
                record,
                plan.metadata.cache_namespace,
                chunk.token_ids,
                chunk.token_range,
            ):
                _fail(WorkerBridgeErrorCode.RECEIPT_MISMATCH)

    @staticmethod
    def _valid_record_batch(records: object) -> bool:
        if (
            not isinstance(records, tuple)
            or not records
            or any(not isinstance(record, CacheRecord) for record in records)
        ):
            return False
        previous_end: int | None = None
        namespace: CacheNamespace | None = None
        for record in records:
            if namespace is None:
                namespace = record.namespace
            if (
                record.namespace != namespace
                or len(record.token_ids) != LMCACHE_CHUNK_SIZE
                or (previous_end is not None
                and record.source_range.start != previous_end)
                or not GptOssWorkerBridge._valid_record(
                    record,
                    record.namespace,
                    record.token_ids,
                    record.source_range,
                )
            ):
                return False
            previous_end = record.source_range.end
        return True

    @staticmethod
    def _valid_record(
        record: object,
        namespace: CacheNamespace,
        token_ids: tuple[int, ...],
        source_range: TokenRange,
    ) -> bool:
        if not isinstance(record, CacheRecord) or not isinstance(
            record.cache_key, str
        ):
            return False
        suffix = record.cache_key.removeprefix(LMCACHE_CACHE_KEY_PREFIX)
        return (
            record.namespace == namespace
            and record.token_ids == token_ids
            and record.source_range == source_range
            and record.fingerprint
            == SHA256_FINGERPRINTER.fingerprint(namespace, token_ids)
            and record.cache_key.startswith(LMCACHE_CACHE_KEY_PREFIX)
            and len(suffix) == 64
            and all(character in "0123456789abcdef" for character in suffix)
        )

    @staticmethod
    def _validate_data_receipt(
        receipt: object,
        direction: TransferDirection,
        logical_tokens: int,
        span_count: int,
    ) -> None:
        expected_rows = logical_tokens * GPT_OSS_NUM_LAYERS
        if (
            not isinstance(receipt, DataPlaneReceipt)
            or receipt.direction is not direction
            or receipt.logical_tokens != logical_tokens
            or receipt.layer_token_rows != expected_rows
            or receipt.span_count != span_count
            or receipt.copied_key_rows != expected_rows
            or receipt.copied_value_rows != expected_rows
            or receipt.sinks_touched
            or (
                direction is TransferDirection.LOAD_FROM_STAGING
                and receipt.corrected_key_rows != expected_rows
            )
            or (
                direction is TransferDirection.STORE_TO_STAGING
                and receipt.corrected_key_rows != 0
            )
        ):
            _fail(WorkerBridgeErrorCode.RECEIPT_MISMATCH)

    @staticmethod
    def _validate_lease(
        lease: object,
        direction: StagingTransferDirection,
        buffer_offset: int,
        token_extent: int,
    ) -> None:
        if (
            not isinstance(lease, StagingTransferLease)
            or lease.direction is not direction
            or lease.buffer_offset != buffer_offset
            or lease.token_extent != token_extent
        ):
            _fail(WorkerBridgeErrorCode.RECEIPT_MISMATCH)

    def _chunk_store_offset(self, chunk_index: int) -> int:
        return (
            self._buffer_config.store_buffer_offset
            + chunk_index * LMCACHE_CHUNK_SIZE
        )

    def _require_ready(self) -> None:
        if (
            self._state is not WorkerBridgeState.READY
            or self._staging.state is not StagingState.REGISTERED
        ):
            if self._state is WorkerBridgeState.READY:
                self._state = WorkerBridgeState.FAILED
            _fail(WorkerBridgeErrorCode.INVALID_STATE)

    @staticmethod
    def _require_plan(expected: object, observed: object) -> None:
        if expected is not observed:
            _fail(WorkerBridgeErrorCode.PLAN_ORDER_MISMATCH)

    def _clear_plans(self) -> None:
        self._load_storage_preflight = None
        self._load_data_preflight = None
        self._retrieved_plan = None
        self._gather_preflight = None
        self._store_preflight = None
        self._gathered_plan = None


__all__ = [
    "AtomicRecordWriter",
    "DataPlaneFactory",
    "DataPlaneOperations",
    "GptOssWorkerBridge",
    "StagingRuntimeFactory",
    "StagingRuntimeLike",
    "WorkerBridgeBufferConfig",
    "WorkerBridgeError",
    "WorkerBridgeErrorCode",
    "WorkerBridgeState",
    "WorkerLmcacheTransport",
]
