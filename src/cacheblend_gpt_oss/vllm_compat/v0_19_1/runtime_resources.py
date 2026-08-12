# SPDX-License-Identifier: Apache-2.0
"""Role-scoped production resources for the pinned transfer connector.

This module is the only composition root between the dependency-free
scheduler/transfer runtimes and the concrete LMCache, SQLite, Torch, and CUDA
adapters.  Importing it does not import those third-party packages: the heavy
loaders run only when the worker factory is called.

The scheduler owns a read-only sidecar handle and a separate LMCache message
queue client.  The worker owns a read-write sidecar handle, its own LMCache
client, one registered CUDA staging tensor, and the GPT-OSS transfer bridge.
Those process-local ownership rules follow vLLM 0.19.1's separate connector
construction in scheduler and worker processes:
https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L242-L296
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from typing import NoReturn, Protocol

from cacheblend_gpt_oss.gpt_oss.layout import GptOssHybridCacheLayout
from cacheblend_gpt_oss.gpt_oss.torch_yarn import load_torch_yarn_corrector
from cacheblend_gpt_oss.planner.models import (
    CacheNamespace,
    CacheRecord,
    SegmentFingerprint,
)
from cacheblend_gpt_oss.storage.lmcache_types import (
    LmcacheBlendTransportConfig,
    LmcacheCandidate,
    LmcacheServerAttestation,
)
from cacheblend_gpt_oss.storage.lmcache_v0_4_3 import (
    create_lmcache_blend_transport,
)
from cacheblend_gpt_oss.storage.lookup import LmcacheCandidateLookupCoordinator
from cacheblend_gpt_oss.storage.sidecar import SidecarMode, open_sidecar_index
from cacheblend_gpt_oss.vllm_compat.v0_19_1.data_plane import (
    KeyPositionCorrector,
    TensorOps,
    load_torch_tensor_ops,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.scheduler_runtime import (
    SchedulerLookupRuntime,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.staging import (
    StagingBackend,
    StagingConfig,
    load_staging_backend,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.transfer_config import (
    Transfer100PctConfig,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.transfer_runtime import (
    TransferRuntime,
    WorkerDataPlane,
    WorkerStorage,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.worker_bridge import (
    AtomicRecordWriter,
    GptOssWorkerBridge,
    WorkerLmcacheTransport,
)


class RuntimeResourceErrorCode(str, Enum):
    """Bounded composition/cleanup failures safe for logs and metrics."""

    INVALID_CONFIG = "invalid_config"
    INVALID_DEVICE = "invalid_device"
    SCHEDULER_CREATE_FAILED = "scheduler_create_failed"
    WORKER_CREATE_FAILED = "worker_create_failed"
    CLOSE_FAILED = "close_failed"


class RuntimeResourceError(RuntimeError):
    """Fail-closed resource error without paths, URLs, or request data."""

    def __init__(self, code: RuntimeResourceErrorCode) -> None:
        self.code = code
        super().__init__(f"CacheBlend runtime resource failure: {code.value}")


def _fail(code: RuntimeResourceErrorCode) -> NoReturn:
    raise RuntimeResourceError(code)


class RuntimeTransport(WorkerLmcacheTransport, Protocol):
    """Combined scheduler/worker surface of the pinned LMCache client."""

    def lookup_candidates(
        self, token_ids: Sequence[int], *, request_id: str
    ) -> tuple[LmcacheCandidate, ...]:
        """Return untrusted scheduler candidates."""


class TransportFactory(Protocol):
    def __call__(
        self, server_url: str, config: LmcacheBlendTransportConfig
    ) -> RuntimeTransport: ...


class RuntimeSidecar(AtomicRecordWriter, Protocol):
    def lookup(
        self, namespace: CacheNamespace, fingerprint: SegmentFingerprint
    ) -> Sequence[CacheRecord]: ...

    def close(self) -> None: ...


class SidecarFactory(Protocol):
    def __call__(self, path: str, mode: SidecarMode) -> RuntimeSidecar: ...


class OpenWorkerBridge(WorkerStorage, WorkerDataPlane, Protocol):
    def open(self) -> object: ...

    def close(self) -> None: ...


class WorkerBridgeFactory(Protocol):
    def __call__(
        self,
        *,
        staging_config: StagingConfig,
        staging_backend: StagingBackend,
        transport: WorkerLmcacheTransport,
        sidecar: AtomicRecordWriter,
        tensor_ops: TensorOps,
        paged_caches: Mapping[str, object],
        correct_key_positions: KeyPositionCorrector,
    ) -> OpenWorkerBridge: ...


class SchedulerRuntimeLike(Protocol):
    def discard(self, request_id: str) -> object: ...

    def close(self) -> None: ...


def _transport_factory(
    server_url: str, config: LmcacheBlendTransportConfig
) -> RuntimeTransport:
    return create_lmcache_blend_transport(server_url, config)


def _sidecar_factory(path: str, mode: SidecarMode) -> RuntimeSidecar:
    return open_sidecar_index(path, mode)


def _worker_bridge_factory(
    *,
    staging_config: StagingConfig,
    staging_backend: StagingBackend,
    transport: WorkerLmcacheTransport,
    sidecar: AtomicRecordWriter,
    tensor_ops: TensorOps,
    paged_caches: Mapping[str, object],
    correct_key_positions: KeyPositionCorrector,
) -> OpenWorkerBridge:
    return GptOssWorkerBridge(
        staging_config=staging_config,
        staging_backend=staging_backend,
        transport=transport,
        sidecar=sidecar,
        tensor_ops=tensor_ops,
        paged_caches=paged_caches,
        correct_key_positions=correct_key_positions,
    )


def build_lmcache_transport_config(
    config: Transfer100PctConfig,
) -> LmcacheBlendTransportConfig:
    """Translate the strict connector config without weakening attestation."""

    if not isinstance(config, Transfer100PctConfig):
        _fail(RuntimeResourceErrorCode.INVALID_CONFIG)
    attestation = config.lmcache_server_attestation
    return LmcacheBlendTransportConfig(
        namespace=config.namespace,
        server_attestation=LmcacheServerAttestation(
            lmcache_version=attestation.lmcache_version,
            source_commit=attestation.source_commit,
            protocol=attestation.protocol,
            hash_algorithm=attestation.hash_algorithm,
        ),
        request_timeout_seconds=config.request_timeout_seconds,
    )


@dataclass(slots=True)
class SchedulerRuntimeResources:
    """All resources owned by one scheduler connector instance."""

    runtime: SchedulerLookupRuntime
    sidecar: RuntimeSidecar
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        failed = False
        try:
            self.runtime.close()
        except Exception:
            failed = True
        try:
            self.sidecar.close()
        except Exception:
            failed = True
        self._closed = True
        if failed:
            _fail(RuntimeResourceErrorCode.CLOSE_FAILED)


@dataclass(slots=True)
class WorkerRuntimeResources:
    """All resources owned by one worker connector instance."""

    bridge: OpenWorkerBridge
    transfer_runtime: TransferRuntime
    sidecar: RuntimeSidecar
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        failed = False
        try:
            self.bridge.close()
        except Exception:
            failed = True
        try:
            self.sidecar.close()
        except Exception:
            failed = True
        self._closed = True
        if failed:
            _fail(RuntimeResourceErrorCode.CLOSE_FAILED)


def create_scheduler_runtime_resources(
    config: Transfer100PctConfig,
    *,
    transport_factory: TransportFactory = _transport_factory,
    sidecar_factory: SidecarFactory = _sidecar_factory,
) -> SchedulerRuntimeResources:
    """Create scheduler lookup resources; network opening remains lazy."""

    transport_config = build_lmcache_transport_config(config)
    sidecar: RuntimeSidecar | None = None
    transport: RuntimeTransport | None = None
    try:
        sidecar = sidecar_factory(
            config.sidecar_path, SidecarMode.SCHEDULER_READ_ONLY
        )
        transport = transport_factory(config.lmcache_server_url, transport_config)

        def replacement_transport_factory() -> RuntimeTransport:
            return transport_factory(config.lmcache_server_url, transport_config)

        runtime = SchedulerLookupRuntime(
            config,
            transport,
            LmcacheCandidateLookupCoordinator(sidecar),
            replacement_transport_factory=replacement_transport_factory,
        )
        return SchedulerRuntimeResources(runtime, sidecar)
    except Exception as exc:
        if transport is not None:
            with suppress(Exception):
                transport.close()
        if sidecar is not None:
            with suppress(Exception):
                sidecar.close()
        raise RuntimeResourceError(
            RuntimeResourceErrorCode.SCHEDULER_CREATE_FAILED
        ) from exc


def create_worker_runtime_resources(
    config: Transfer100PctConfig,
    layout: GptOssHybridCacheLayout,
    paged_caches: Mapping[str, object],
    *,
    device: str,
    instance_id: int | None = None,
    transport_factory: TransportFactory = _transport_factory,
    sidecar_factory: SidecarFactory = _sidecar_factory,
    staging_backend_factory: Callable[[], StagingBackend] = load_staging_backend,
    tensor_ops_factory: Callable[[], TensorOps] = load_torch_tensor_ops,
    key_corrector_factory: Callable[[], KeyPositionCorrector] = (
        load_torch_yarn_corrector
    ),
    bridge_factory: WorkerBridgeFactory = _worker_bridge_factory,
) -> WorkerRuntimeResources:
    """Create and open one worker's exact CUDA/LMCache ownership graph."""

    if not isinstance(config, Transfer100PctConfig) or not isinstance(
        layout, GptOssHybridCacheLayout
    ):
        _fail(RuntimeResourceErrorCode.INVALID_CONFIG)
    selected_instance_id = os.getpid() if instance_id is None else instance_id
    try:
        staging_config = StagingConfig(
            selected_instance_id,
            config.staging_token_capacity,
            device,
        )
    except Exception as exc:
        raise RuntimeResourceError(RuntimeResourceErrorCode.INVALID_DEVICE) from exc

    sidecar: RuntimeSidecar | None = None
    bridge: OpenWorkerBridge | None = None
    transport: RuntimeTransport | None = None
    try:
        transport_config = build_lmcache_transport_config(config)
        sidecar = sidecar_factory(config.sidecar_path, SidecarMode.WORKER_READ_WRITE)
        transport = transport_factory(config.lmcache_server_url, transport_config)
        bridge = bridge_factory(
            staging_config=staging_config,
            staging_backend=staging_backend_factory(),
            transport=transport,
            sidecar=sidecar,
            tensor_ops=tensor_ops_factory(),
            paged_caches=dict(paged_caches),
            correct_key_positions=key_corrector_factory(),
        )
        bridge.open()
        runtime = TransferRuntime(layout, bridge, bridge)
        return WorkerRuntimeResources(bridge, runtime, sidecar)
    except Exception as exc:
        if bridge is not None:
            with suppress(Exception):
                bridge.close()
        elif transport is not None:
            with suppress(Exception):
                transport.close()
        if sidecar is not None:
            with suppress(Exception):
                sidecar.close()
        raise RuntimeResourceError(
            RuntimeResourceErrorCode.WORKER_CREATE_FAILED
        ) from exc


__all__ = [
    "OpenWorkerBridge",
    "RuntimeResourceError",
    "RuntimeResourceErrorCode",
    "RuntimeSidecar",
    "RuntimeTransport",
    "SchedulerRuntimeResources",
    "SidecarFactory",
    "TransportFactory",
    "WorkerBridgeFactory",
    "WorkerRuntimeResources",
    "build_lmcache_transport_config",
    "create_scheduler_runtime_resources",
    "create_worker_runtime_resources",
]
