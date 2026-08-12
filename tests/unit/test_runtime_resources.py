"""CPU-only tests for role-scoped production resource composition."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pytest

from cacheblend_gpt_oss.gpt_oss.layout import (
    AttentionKind,
    CacheGroupLayout,
    GptOssHybridCacheLayout,
)
from cacheblend_gpt_oss.planner.models import (
    CacheNamespace,
    CacheRecord,
    SegmentFingerprint,
)
from cacheblend_gpt_oss.storage.lmcache_types import (
    LMCACHE_BLEND_PROTOCOL,
    LMCACHE_HASH_ALGORITHM,
    LMCACHE_SOURCE_COMMIT,
    LMCACHE_VERSION,
    LmcacheBlendTransportConfig,
    LmcacheCandidate,
)
from cacheblend_gpt_oss.storage.sidecar import SidecarMode
from cacheblend_gpt_oss.vllm_compat.v0_19_1.runtime_resources import (
    CudaRuntimeIdentity,
    OpenWorkerBridge,
    RuntimeResourceError,
    RuntimeResourceErrorCode,
    RuntimeSidecar,
    RuntimeTransport,
    SchedulerRuntimeResources,
    WorkerRuntimeResources,
    build_lmcache_transport_config,
    create_scheduler_runtime_resources,
    create_worker_runtime_resources,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.scheduler_runtime import (
    SchedulerRuntimeState,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.staging import StagingConfig
from cacheblend_gpt_oss.vllm_compat.v0_19_1.transfer_config import (
    PinnedLmcacheServerAttestation,
    Transfer100PctConfig,
    TransferFailurePolicy,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.transfer_runtime import (
    TransferRuntime,
)


def _config(tmp_path: object) -> Transfer100PctConfig:
    return Transfer100PctConfig(
        lmcache_server_url="tcp://127.0.0.1:5555",
        sidecar_path=f"{tmp_path}/sidecar.sqlite3",
        lmcache_server_attestation=PinnedLmcacheServerAttestation(
            lmcache_version=LMCACHE_VERSION,
            source_commit=LMCACHE_SOURCE_COMMIT,
            protocol=LMCACHE_BLEND_PROTOCOL,
            hash_algorithm=LMCACHE_HASH_ALGORITHM,
        ),
        model_revision="model-revision",
        tokenizer_revision="tokenizer-revision",
        model_config_digest="a" * 64,
        kv_cache_config_digest="b" * 64,
        adapter_revision="adapter-revision",
        staging_token_capacity=512,
        request_timeout_seconds=7.5,
        transfer_failure_policy=TransferFailurePolicy.FULL_PREFILL,
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


def _cuda_identity(device: str) -> CudaRuntimeIdentity:
    return CudaRuntimeIdentity(
        device_index=int(device.rsplit(":", 1)[1]),
        torch_version="2.10.0+cu128",
        cuda_runtime="12.8",
        gpu_name="NVIDIA A100-SXM4-80GB",
        compute_capability="8.0",
    )


class FakeTransport:
    def __init__(self, config: LmcacheBlendTransportConfig) -> None:
        self.config = config
        self.open_calls = 0
        self.close_calls = 0

    def open(self) -> None:
        self.open_calls += 1

    def lookup_candidates(
        self, token_ids: Sequence[int], *, request_id: str
    ) -> tuple[LmcacheCandidate, ...]:
        del token_ids, request_id
        return ()

    def close(self) -> None:
        self.close_calls += 1


class FakeSidecar:
    def __init__(self, *, close_error: bool = False) -> None:
        self.close_error = close_error
        self.close_calls = 0

    def lookup(
        self, namespace: CacheNamespace, fingerprint: SegmentFingerprint
    ) -> Sequence[CacheRecord]:
        del namespace, fingerprint
        return ()

    def add_many(self, records: object) -> int:
        del records
        return 0

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error:
            raise RuntimeError("bounded fake cleanup failure")


class FakeBridge:
    def __init__(self, *, open_error: bool = False, close_error: bool = False) -> None:
        self.open_error = open_error
        self.close_error = close_error
        self.open_calls = 0
        self.close_calls = 0

    def open(self) -> object:
        self.open_calls += 1
        if self.open_error:
            raise RuntimeError("bounded fake open failure")
        return object()

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error:
            raise RuntimeError("bounded fake cleanup failure")


def test_transport_config_preserves_strict_identity(tmp_path: object) -> None:
    config = _config(tmp_path)
    result = build_lmcache_transport_config(config)

    assert result.namespace == config.namespace
    assert result.request_timeout_seconds == 7.5
    assert result.server_attestation.lmcache_version == LMCACHE_VERSION
    assert result.server_attestation.source_commit == LMCACHE_SOURCE_COMMIT
    assert result.server_attestation.protocol == LMCACHE_BLEND_PROTOCOL
    assert result.server_attestation.hash_algorithm == LMCACHE_HASH_ALGORITHM


def test_scheduler_resources_are_role_scoped_and_network_lazy(
    tmp_path: object,
) -> None:
    config = _config(tmp_path)
    calls: list[tuple[str, object, object]] = []
    sidecar = FakeSidecar()
    transport = FakeTransport(build_lmcache_transport_config(config))

    def sidecar_factory(path: str, mode: SidecarMode) -> RuntimeSidecar:
        calls.append(("sidecar", path, mode))
        return cast(RuntimeSidecar, sidecar)

    def transport_factory(
        url: str, transport_config: LmcacheBlendTransportConfig
    ) -> RuntimeTransport:
        calls.append(("transport", url, transport_config))
        return cast(RuntimeTransport, transport)

    resources = create_scheduler_runtime_resources(
        config,
        sidecar_factory=sidecar_factory,
        transport_factory=transport_factory,
    )

    assert resources.runtime.state is SchedulerRuntimeState.CREATED
    assert transport.open_calls == 0
    assert calls == [
        ("sidecar", config.sidecar_path, SidecarMode.SCHEDULER_READ_ONLY),
        ("transport", config.lmcache_server_url, transport.config),
    ]
    resources.close()
    resources.close()
    assert transport.close_calls == 1
    assert sidecar.close_calls == 1


def test_scheduler_creation_failure_cleans_opened_sidecar(tmp_path: object) -> None:
    config = _config(tmp_path)
    sidecar = FakeSidecar()

    def sidecar_factory(path: str, mode: SidecarMode) -> RuntimeSidecar:
        del path, mode
        return cast(RuntimeSidecar, sidecar)

    def transport_factory(
        url: str, transport_config: LmcacheBlendTransportConfig
    ) -> RuntimeTransport:
        del url, transport_config
        raise RuntimeError("bounded fake factory failure")

    with pytest.raises(RuntimeResourceError) as error:
        create_scheduler_runtime_resources(
            config,
            sidecar_factory=sidecar_factory,
            transport_factory=transport_factory,
        )

    assert error.value.code is RuntimeResourceErrorCode.SCHEDULER_CREATE_FAILED
    assert sidecar.close_calls == 1


def test_worker_resources_bind_one_device_and_open_bridge(tmp_path: object) -> None:
    config = _config(tmp_path)
    layout = _layout()
    caches = {_layer_name(index): object() for index in range(24)}
    sidecar = FakeSidecar()
    transport = FakeTransport(build_lmcache_transport_config(config))
    bridge = FakeBridge()
    captured: dict[str, object] = {}
    backend = cast(object, object())
    tensor_ops = cast(object, object())
    corrector = cast(object, lambda *args: object())

    def sidecar_factory(path: str, mode: SidecarMode) -> RuntimeSidecar:
        captured["sidecar_factory"] = (path, mode)
        return cast(RuntimeSidecar, sidecar)

    def transport_factory(
        url: str, transport_config: LmcacheBlendTransportConfig
    ) -> RuntimeTransport:
        captured["transport_factory"] = (url, transport_config)
        return cast(RuntimeTransport, transport)

    def bridge_factory(**kwargs: object) -> OpenWorkerBridge:
        captured.update(kwargs)
        return cast(OpenWorkerBridge, bridge)

    resources = create_worker_runtime_resources(
        config,
        layout,
        caches,
        device="cuda:3",
        instance_id=77,
        sidecar_factory=sidecar_factory,
        transport_factory=transport_factory,
        staging_backend_factory=lambda: cast(object, backend),
        tensor_ops_factory=lambda: cast(object, tensor_ops),
        key_corrector_factory=lambda: cast(object, corrector),
        cuda_runtime_factory=_cuda_identity,
        bridge_factory=bridge_factory,
    )

    assert isinstance(resources.transfer_runtime, TransferRuntime)
    assert bridge.open_calls == 1
    assert captured["staging_config"] == StagingConfig(77, 512, "cuda:3")
    assert captured["paged_caches"] == caches
    assert captured["transport_factory"] == (
        config.lmcache_server_url,
        transport.config,
    )
    assert captured["transport"] is transport
    assert captured["sidecar_factory"] == (
        config.sidecar_path,
        SidecarMode.WORKER_READ_WRITE,
    )
    assert captured["sidecar"] is sidecar
    assert captured["staging_backend"] is backend
    assert captured["tensor_ops"] is tensor_ops
    assert captured["correct_key_positions"] is corrector
    resources.close()
    resources.close()
    assert bridge.close_calls == 1
    assert sidecar.close_calls == 1


@pytest.mark.parametrize("device", ["cuda", "cpu", "cuda:-1", "cuda:1:2"])
def test_worker_rejects_nonindexed_cuda_device(
    tmp_path: object, device: str
) -> None:
    with pytest.raises(RuntimeResourceError) as error:
        create_worker_runtime_resources(
            _config(tmp_path), _layout(), {}, device=device
        )
    assert error.value.code is RuntimeResourceErrorCode.INVALID_DEVICE


def test_worker_open_failure_closes_bridge_and_sidecar(tmp_path: object) -> None:
    config = _config(tmp_path)
    bridge = FakeBridge(open_error=True)
    sidecar = FakeSidecar()
    transport = FakeTransport(build_lmcache_transport_config(config))

    with pytest.raises(RuntimeResourceError) as error:
        create_worker_runtime_resources(
            config,
            _layout(),
            {_layer_name(index): object() for index in range(24)},
            device="cuda:0",
            instance_id=12,
            transport_factory=lambda _url, _config: cast(
                RuntimeTransport, transport
            ),
            sidecar_factory=lambda _path, _mode: cast(RuntimeSidecar, sidecar),
            staging_backend_factory=lambda: cast(object, object()),
            tensor_ops_factory=lambda: cast(object, object()),
            key_corrector_factory=lambda: cast(object, lambda *args: object()),
            cuda_runtime_factory=_cuda_identity,
            bridge_factory=lambda **_kwargs: cast(OpenWorkerBridge, bridge),
        )

    assert error.value.code is RuntimeResourceErrorCode.WORKER_CREATE_FAILED
    assert bridge.open_calls == 1
    assert bridge.close_calls == 1
    assert sidecar.close_calls == 1


@pytest.mark.parametrize(
    "field",
    ["torch_version", "cuda_runtime", "gpu_name", "compute_capability"],
)
def test_worker_rejects_unpinned_cuda_runtime_identity(
    tmp_path: object, field: str
) -> None:
    identity = _cuda_identity("cuda:0")
    values = {
        "device_index": identity.device_index,
        "torch_version": identity.torch_version,
        "cuda_runtime": identity.cuda_runtime,
        "gpu_name": identity.gpu_name,
        "compute_capability": identity.compute_capability,
    }
    values[field] = "unsupported"
    invalid = CudaRuntimeIdentity(**values)

    with pytest.raises(RuntimeResourceError) as error:
        create_worker_runtime_resources(
            _config(tmp_path),
            _layout(),
            {},
            device="cuda:0",
            cuda_runtime_factory=lambda _device: invalid,
        )
    assert error.value.code is RuntimeResourceErrorCode.INVALID_DEVICE


def test_cuda_identity_rejects_boolean_device_index() -> None:
    with pytest.raises(RuntimeResourceError) as error:
        CudaRuntimeIdentity(
            device_index=True,  # type: ignore[arg-type]
            torch_version="2.10.0+cu128",
            cuda_runtime="12.8",
            gpu_name="NVIDIA A100-SXM4-80GB",
            compute_capability="8.0",
        )

    assert error.value.code is RuntimeResourceErrorCode.INVALID_DEVICE


class FakeClosableRuntime:
    def __init__(self, *, close_error: bool = False) -> None:
        self.close_error = close_error
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error:
            raise RuntimeError("bounded fake cleanup failure")


def test_resource_close_attempts_every_owner_after_failure() -> None:
    runtime = FakeClosableRuntime(close_error=True)
    sidecar = FakeSidecar(close_error=True)
    scheduler = SchedulerRuntimeResources(
        cast(object, runtime), cast(RuntimeSidecar, sidecar)
    )

    with pytest.raises(RuntimeResourceError) as error:
        scheduler.close()
    assert error.value.code is RuntimeResourceErrorCode.CLOSE_FAILED
    assert runtime.close_calls == 1
    assert sidecar.close_calls == 1

    bridge = FakeBridge(close_error=True)
    worker_sidecar = FakeSidecar(close_error=True)
    worker = WorkerRuntimeResources(
        cast(OpenWorkerBridge, bridge),
        cast(TransferRuntime, object()),
        cast(RuntimeSidecar, worker_sidecar),
    )
    with pytest.raises(RuntimeResourceError) as error:
        worker.close()
    assert error.value.code is RuntimeResourceErrorCode.CLOSE_FAILED
    assert bridge.close_calls == 1
    assert worker_sidecar.close_calls == 1
