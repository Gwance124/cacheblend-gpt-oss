"""CPU-fake tests for the pinned CUDA/LMCache staging runtime."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from cacheblend_gpt_oss.storage.lmcache_types import LmcacheStagingRegistration
from cacheblend_gpt_oss.vllm_compat.v0_19_1 import staging as module
from cacheblend_gpt_oss.vllm_compat.v0_19_1.staging import (
    STAGING_DTYPE,
    StagingConfig,
    StagingError,
    StagingErrorCode,
    StagingRuntime,
    StagingState,
    StagingTransferDirection,
    TorchLmcacheStagingBackend,
)

CAPACITY = 512
SHAPE = (2, 24, CAPACITY, 512)
STRIDE = (24 * CAPACITY * 512, CAPACITY * 512, 512, 1)


@dataclass(slots=True)
class FakeTensor:
    shape: tuple[int, ...] = SHAPE
    dtype: str = STAGING_DTYPE
    device: str = "cuda:0"
    contiguous: bool = True
    storage_offset: int = 0


@dataclass(slots=True)
class FakeWrapper:
    shape: tuple[int, ...]
    dtype: str
    storage_offset: int
    stride: tuple[int, ...]
    device_uuid: str
    handle: object


@dataclass(slots=True)
class FakeEvent:
    handle: bytes = b"stable-interprocess-event"


class FakeBackend:
    def __init__(self, trace: list[str] | None = None) -> None:
        self.trace = trace if trace is not None else []
        self.torch_version_value = "2.10.0+cu128"
        self.cuda_runtime_value = "12.8"
        self.lmcache_version_value = "0.4.3"
        self.tensor_value = FakeTensor()
        self.wrapper_value: FakeWrapper | None = None
        self.event_value = FakeEvent()
        self.device_uuid_value = "GPU-a100"
        self.device_error: Exception | None = None
        self.allocate_error: Exception | None = None
        self.wrapper_error: Exception | None = None
        self.event_error: Exception | None = None
        self.record_error: Exception | None = None
        self.event_sync_error: Exception | None = None
        self.handle_error: Exception | None = None
        self.sync_error_at: set[int] = set()
        self.sync_count = 0
        self.event_create_count = 0
        self.wrapper_create_count = 0

    @property
    def torch_version(self) -> str:
        return self.torch_version_value

    @property
    def cuda_runtime(self) -> str:
        return self.cuda_runtime_value

    @property
    def lmcache_version(self) -> str:
        return self.lmcache_version_value

    def validate_device(self, device_index: int, expected_name: str) -> None:
        self.trace.append(f"validate-device:{device_index}:{expected_name}")
        if self.device_error is not None:
            raise self.device_error

    def device_uuid(self, device_index: int) -> str:
        assert device_index == 0
        return self.device_uuid_value

    def allocate(
        self, shape: tuple[int, ...], dtype_name: str, device: str
    ) -> object:
        self.trace.append(f"allocate:{shape}:{dtype_name}:{device}")
        if self.allocate_error is not None:
            raise self.allocate_error
        return self.tensor_value

    def tensor_shape(self, tensor: object) -> tuple[int, ...]:
        assert tensor is self.tensor_value
        return self.tensor_value.shape

    def tensor_dtype_name(self, tensor: object) -> str:
        assert tensor is self.tensor_value
        return self.tensor_value.dtype

    def tensor_device_name(self, tensor: object) -> str:
        assert tensor is self.tensor_value
        return self.tensor_value.device

    def tensor_is_contiguous(self, tensor: object) -> bool:
        assert tensor is self.tensor_value
        return self.tensor_value.contiguous

    def tensor_storage_offset(self, tensor: object) -> int:
        assert tensor is self.tensor_value
        return self.tensor_value.storage_offset

    def make_cuda_ipc_wrapper(self, tensor: object) -> object:
        assert tensor is self.tensor_value
        self.trace.append("make-wrapper")
        self.wrapper_create_count += 1
        if self.wrapper_error is not None:
            raise self.wrapper_error
        if self.wrapper_value is None:
            self.wrapper_value = FakeWrapper(
                shape=self.tensor_value.shape,
                dtype=self.tensor_value.dtype,
                storage_offset=self.tensor_value.storage_offset,
                stride=STRIDE,
                device_uuid=self.device_uuid_value,
                handle=("cuda-storage-handle",),
            )
        return self.wrapper_value

    def wrapper_shape(self, wrapper: object) -> tuple[int, ...]:
        assert isinstance(wrapper, FakeWrapper)
        return wrapper.shape

    def wrapper_dtype_name(self, wrapper: object) -> str:
        assert isinstance(wrapper, FakeWrapper)
        return wrapper.dtype

    def wrapper_storage_offset(self, wrapper: object) -> int:
        assert isinstance(wrapper, FakeWrapper)
        return wrapper.storage_offset

    def wrapper_stride(self, wrapper: object) -> tuple[int, ...]:
        assert isinstance(wrapper, FakeWrapper)
        return wrapper.stride

    def wrapper_device_uuid(self, wrapper: object) -> str:
        assert isinstance(wrapper, FakeWrapper)
        return wrapper.device_uuid

    def wrapper_has_handle(self, wrapper: object) -> bool:
        assert isinstance(wrapper, FakeWrapper)
        return bool(wrapper.handle)

    def create_interprocess_event(self, device_index: int) -> object:
        assert device_index == 0
        self.trace.append("create-interprocess-event")
        self.event_create_count += 1
        if self.event_error is not None:
            raise self.event_error
        return self.event_value

    def synchronize_device(self, device_index: int) -> None:
        assert device_index == 0
        self.sync_count += 1
        self.trace.append(f"device-sync:{self.sync_count}")
        if self.sync_count in self.sync_error_at:
            raise RuntimeError("injected device synchronize failure")

    def record_event(self, event: object, device_index: int) -> None:
        assert event is self.event_value
        assert device_index == 0
        self.trace.append("record-event")
        if self.record_error is not None:
            raise self.record_error

    def synchronize_event(self, event: object) -> None:
        assert event is self.event_value
        self.trace.append("event-sync")
        if self.event_sync_error is not None:
            raise self.event_sync_error

    def event_ipc_handle(self, event: object) -> bytes:
        assert event is self.event_value
        self.trace.append("event-handle")
        if self.handle_error is not None:
            raise self.handle_error
        return self.event_value.handle


class FakeTransport:
    def __init__(self, trace: list[str] | None = None) -> None:
        self.trace = trace if trace is not None else []
        self.registrations: list[LmcacheStagingRegistration] = []
        self.unregister_count = 0
        self.register_error: Exception | None = None
        self.unregister_failures = 0

    def register_staging_buffer(
        self, registration: LmcacheStagingRegistration
    ) -> None:
        self.trace.append("register")
        self.registrations.append(registration)
        if self.register_error is not None:
            raise self.register_error

    def unregister_staging_buffer(self) -> None:
        self.trace.append("unregister")
        self.unregister_count += 1
        if self.unregister_failures > 0:
            self.unregister_failures -= 1
            raise RuntimeError("injected unregister failure")


def config() -> StagingConfig:
    return StagingConfig(instance_id=17, token_capacity=CAPACITY, device="cuda:0")


def runtime(
    backend: FakeBackend | None = None, transport: FakeTransport | None = None
) -> tuple[StagingRuntime, FakeBackend, FakeTransport]:
    selected_backend = backend or FakeBackend()
    selected_transport = transport or FakeTransport(selected_backend.trace)
    return (
        StagingRuntime(config(), selected_backend, selected_transport),
        selected_backend,
        selected_transport,
    )


def assert_error(
    code: StagingErrorCode, operation: Callable[[], object]
) -> StagingError:
    with pytest.raises(StagingError) as caught:
        operation()
    assert caught.value.code is code
    return caught.value


def test_open_allocates_exact_tensor_wrapper_event_and_registration() -> None:
    value, backend, transport = runtime()

    tensor = value.open()

    assert tensor is backend.tensor_value
    assert value.tensor is tensor
    assert value.state is StagingState.REGISTERED
    assert backend.wrapper_create_count == 1
    assert backend.event_create_count == 1
    assert backend.trace == [
        "validate-device:0:NVIDIA A100-SXM4-80GB",
        f"allocate:{SHAPE}:{STAGING_DTYPE}:cuda:0",
        "create-interprocess-event",
        "device-sync:1",
        "record-event",
        "event-sync",
        "event-handle",
        "make-wrapper",
        "register",
    ]
    assert len(transport.registrations) == 1
    registration = transport.registrations[0]
    assert registration.instance_id == 17
    assert registration.kv_cache_payload == (backend.wrapper_value,)
    assert registration.layout.shape == SHAPE
    assert registration.layout.dtype_name == STAGING_DTYPE


def test_synchronous_transfer_reuses_handle_with_explicit_ordering() -> None:
    trace: list[str] = []
    backend = FakeBackend(trace)
    transport = FakeTransport(trace)
    value, _, _ = runtime(backend, transport)
    value.open()
    trace.clear()

    with value.synchronous_transfer(
        direction=StagingTransferDirection.RETRIEVE,
        buffer_offset=64,
        token_extent=300,
    ) as first_lease:
        trace.append("synchronous-transport-call")
        assert value.state is StagingState.TRANSFER_ACTIVE
        assert first_lease.event_ipc_handle == b"stable-interprocess-event"
        assert first_lease.buffer_offset == 64
        assert first_lease.token_extent == 300
        assert first_lease.staging_position(17) == 81

    with value.synchronous_transfer(
        direction=StagingTransferDirection.STORE,
        buffer_offset=7,
        token_extent=256,
    ) as second_lease:
        trace.append("second-synchronous-transport-call")

    assert first_lease.event_ipc_handle == second_lease.event_ipc_handle
    assert backend.event_create_count == 1
    assert value.state is StagingState.REGISTERED
    assert trace == [
        "device-sync:2",
        "record-event",
        "event-sync",
        "event-handle",
        "synchronous-transport-call",
        "device-sync:3",
        "device-sync:4",
        "record-event",
        "event-sync",
        "event-handle",
        "second-synchronous-transport-call",
        "device-sync:5",
    ]


def test_staging_tensor_is_not_exposed_while_lmcache_owns_transfer() -> None:
    value, _, _ = runtime()
    value.open()

    with value.synchronous_transfer(
        direction=StagingTransferDirection.RETRIEVE,
        buffer_offset=0,
        token_extent=256,
    ):
        assert_error(StagingErrorCode.INVALID_STATE, lambda: value.tensor)

    assert value.state is StagingState.REGISTERED


@pytest.mark.parametrize(
    ("direction", "buffer_offset", "token_extent"),
    [
        ("retrieve", 0, 256),
        (StagingTransferDirection.RETRIEVE, -1, 256),
        (StagingTransferDirection.RETRIEVE, True, 256),
        (StagingTransferDirection.RETRIEVE, 500, 13),
        (StagingTransferDirection.RETRIEVE, 0, 0),
        (StagingTransferDirection.STORE, 0, 255),
    ],
)
def test_invalid_transfer_placement_fails_before_event_recording(
    direction: object, buffer_offset: object, token_extent: object
) -> None:
    value, backend, _ = runtime()
    value.open()
    prior_sync_count = backend.sync_count

    assert_error(
        StagingErrorCode.INVALID_CONFIG,
        lambda: _transfer_with_untrusted_placement(
            value, direction, buffer_offset, token_extent
        ),
    )
    assert value.state is StagingState.REGISTERED
    assert backend.sync_count == prior_sync_count


def _transfer_with_untrusted_placement(
    value: StagingRuntime,
    direction: object,
    buffer_offset: object,
    token_extent: object,
) -> None:
    with value.synchronous_transfer(
        direction=direction,  # type: ignore[arg-type]
        buffer_offset=buffer_offset,  # type: ignore[arg-type]
        token_extent=token_extent,  # type: ignore[arg-type]
    ):
        pass


def test_lease_rejects_position_outside_bound_transfer() -> None:
    value, _, _ = runtime()
    value.open()

    with value.synchronous_transfer(
        direction=StagingTransferDirection.RETRIEVE,
        buffer_offset=11,
        token_extent=17,
    ) as lease:
        assert lease.staging_position(0) == 11
        assert lease.staging_position(16) == 27
        assert lease.end_offset == 28
        assert_error(
            StagingErrorCode.INVALID_CONFIG,
            lambda: lease.staging_position(17),
        )
        assert_error(
            StagingErrorCode.INVALID_CONFIG,
            lambda: lease.staging_position(True),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"instance_id": -1, "token_capacity": CAPACITY, "device": "cuda:0"},
        {"instance_id": True, "token_capacity": CAPACITY, "device": "cuda:0"},
        {"instance_id": 1, "token_capacity": 255, "device": "cuda:0"},
        {"instance_id": 1, "token_capacity": 257, "device": "cuda:0"},
        {"instance_id": 1, "token_capacity": 131_328, "device": "cuda:0"},
        {"instance_id": 1, "token_capacity": True, "device": "cuda:0"},
        {"instance_id": 1, "token_capacity": CAPACITY, "device": "cuda"},
        {"instance_id": 1, "token_capacity": CAPACITY, "device": "cuda:-1"},
        {"instance_id": 1, "token_capacity": CAPACITY, "device": "cuda:01"},
        {
            "instance_id": 1,
            "token_capacity": CAPACITY,
            "device": "cuda:" + "9" * 10_000,
        },
        {"instance_id": 1, "token_capacity": CAPACITY, "device": "cpu"},
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, object]) -> None:
    assert_error(StagingErrorCode.INVALID_CONFIG, lambda: StagingConfig(**kwargs))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("attribute", "observed", "code"),
    [
        (
            "torch_version_value",
            "2.10.1+cu128",
            StagingErrorCode.TORCH_VERSION_MISMATCH,
        ),
        (
            "cuda_runtime_value",
            "12.7",
            StagingErrorCode.CUDA_VERSION_MISMATCH,
        ),
        (
            "lmcache_version_value",
            "0.4.4",
            StagingErrorCode.LMCACHE_VERSION_MISMATCH,
        ),
    ],
)
def test_unpinned_version_rejects_construction(
    attribute: str, observed: str, code: StagingErrorCode
) -> None:
    backend = FakeBackend()
    setattr(backend, attribute, observed)
    assert_error(code, lambda: StagingRuntime(config(), backend, FakeTransport()))


def test_wrong_gpu_is_preserved_as_a_specific_fail_closed_error() -> None:
    backend = FakeBackend()
    backend.device_error = StagingError(
        StagingErrorCode.GPU_MISMATCH, "wrong GPU"
    )
    value, _, transport = runtime(backend)

    assert_error(StagingErrorCode.GPU_MISMATCH, value.open)
    assert value.state is StagingState.FAILED
    assert transport.registrations == []


def test_raw_device_failure_is_classified_and_fails_before_allocation() -> None:
    backend = FakeBackend()
    backend.device_error = RuntimeError("CUDA unavailable")
    value, _, transport = runtime(backend)

    assert_error(StagingErrorCode.CUDA_DEVICE_UNAVAILABLE, value.open)
    assert value.state is StagingState.FAILED
    assert transport.registrations == []
    assert not any(entry.startswith("allocate:") for entry in backend.trace)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda tensor: setattr(tensor, "shape", (2, 23, CAPACITY, 512)),
        lambda tensor: setattr(tensor, "dtype", "torch.float16"),
        lambda tensor: setattr(tensor, "device", "cuda:1"),
        lambda tensor: setattr(tensor, "contiguous", False),
        lambda tensor: setattr(tensor, "storage_offset", 1),
        lambda tensor: setattr(tensor, "storage_offset", True),
    ],
)
def test_invalid_allocated_tensor_fails_before_event_and_registration(
    mutation: Callable[[FakeTensor], None],
) -> None:
    backend = FakeBackend()
    mutation(backend.tensor_value)
    value, _, transport = runtime(backend)

    assert_error(StagingErrorCode.INVALID_TENSOR, value.open)
    assert value.state is StagingState.FAILED
    assert backend.event_create_count == 0
    assert backend.wrapper_create_count == 0
    assert transport.registrations == []


@pytest.mark.parametrize(
    "mutation",
    [
        lambda wrapper: setattr(wrapper, "shape", (2, 23, CAPACITY, 512)),
        lambda wrapper: setattr(wrapper, "dtype", "torch.float16"),
        lambda wrapper: setattr(wrapper, "storage_offset", 1),
        lambda wrapper: setattr(wrapper, "stride", (1, 2, 3, 4)),
        lambda wrapper: setattr(wrapper, "device_uuid", "GPU-other"),
        lambda wrapper: setattr(wrapper, "handle", ()),
    ],
)
def test_invalid_cuda_ipc_declaration_fails_before_registration(
    mutation: Callable[[FakeWrapper], None],
) -> None:
    backend = FakeBackend()
    wrapper = FakeWrapper(
        shape=SHAPE,
        dtype=STAGING_DTYPE,
        storage_offset=0,
        stride=STRIDE,
        device_uuid=backend.device_uuid_value,
        handle=("cuda-storage-handle",),
    )
    mutation(wrapper)
    backend.wrapper_value = wrapper
    value, _, transport = runtime(backend)

    assert_error(StagingErrorCode.IPC_WRAPPER_FAILED, value.open)
    assert value.state is StagingState.FAILED
    assert transport.registrations == []


@pytest.mark.parametrize(
    ("attribute", "error"),
    [
        ("allocate_error", RuntimeError("allocation")),
        ("event_error", RuntimeError("event")),
        ("record_error", RuntimeError("record")),
        ("event_sync_error", RuntimeError("event synchronize")),
        ("handle_error", RuntimeError("event handle")),
        ("wrapper_error", RuntimeError("wrapper")),
    ],
)
def test_creation_failures_are_bounded_and_never_register(
    attribute: str, error: Exception
) -> None:
    backend = FakeBackend()
    setattr(backend, attribute, error)
    value, _, transport = runtime(backend)
    expected = {
        "allocate_error": StagingErrorCode.ALLOCATION_FAILED,
        "event_error": StagingErrorCode.EVENT_FAILED,
        "record_error": StagingErrorCode.EVENT_FAILED,
        "event_sync_error": StagingErrorCode.EVENT_FAILED,
        "handle_error": StagingErrorCode.EVENT_FAILED,
        "wrapper_error": StagingErrorCode.IPC_WRAPPER_FAILED,
    }[attribute]

    assert_error(expected, value.open)
    assert value.state is StagingState.FAILED
    assert transport.registrations == []


def test_empty_event_ipc_handle_is_rejected() -> None:
    backend = FakeBackend()
    backend.event_value.handle = b""
    value, _, _ = runtime(backend)

    assert_error(StagingErrorCode.EVENT_FAILED, value.open)
    assert value.state is StagingState.FAILED


def test_initial_device_synchronization_failure_fails_before_registration() -> None:
    backend = FakeBackend()
    backend.sync_error_at.add(1)
    value, _, transport = runtime(backend)

    assert_error(StagingErrorCode.SYNCHRONIZATION_FAILED, value.open)
    assert value.state is StagingState.FAILED
    assert transport.registrations == []


def test_changed_reexported_handle_fails_before_transfer_body() -> None:
    backend = FakeBackend()
    value, _, _ = runtime(backend)
    value.open()
    backend.event_value.handle = b"changed-event-handle"
    body_called = False

    def operation() -> None:
        nonlocal body_called
        with value.synchronous_transfer(
            direction=StagingTransferDirection.RETRIEVE,
            buffer_offset=0,
            token_extent=256,
        ):
            body_called = True

    assert_error(StagingErrorCode.EVENT_FAILED, operation)
    assert not body_called
    assert value.state is StagingState.FAILED


def test_pre_transfer_device_sync_failure_prevents_body_and_fails_closed() -> None:
    backend = FakeBackend()
    value, _, _ = runtime(backend)
    value.open()
    backend.sync_error_at.add(2)
    body_called = False

    def operation() -> None:
        nonlocal body_called
        with value.synchronous_transfer(
            direction=StagingTransferDirection.RETRIEVE,
            buffer_offset=0,
            token_extent=256,
        ):
            body_called = True

    assert_error(StagingErrorCode.SYNCHRONIZATION_FAILED, operation)
    assert not body_called
    assert value.state is StagingState.FAILED


def test_post_transfer_device_sync_failure_fails_closed() -> None:
    backend = FakeBackend()
    value, _, _ = runtime(backend)
    value.open()
    backend.sync_error_at.add(3)

    assert_error(
        StagingErrorCode.SYNCHRONIZATION_FAILED,
        lambda: _successful_transfer(value),
    )
    assert value.state is StagingState.FAILED


def _successful_transfer(value: StagingRuntime) -> None:
    with value.synchronous_transfer(
        direction=StagingTransferDirection.RETRIEVE,
        buffer_offset=0,
        token_extent=256,
    ):
        pass


def test_transport_body_failure_is_wrapped_and_staging_fails_closed() -> None:
    backend = FakeBackend()
    value, _, _ = runtime(backend)
    value.open()

    def operation() -> None:
        with value.synchronous_transfer(
            direction=StagingTransferDirection.RETRIEVE,
            buffer_offset=0,
            token_extent=256,
        ):
            raise TimeoutError("LMCache timeout")

    error = assert_error(StagingErrorCode.TRANSFER_FAILED, operation)
    assert isinstance(error.__cause__, TimeoutError)
    assert value.state is StagingState.FAILED
    assert backend.sync_count == 3


def test_registration_failure_is_conservative_and_close_retries_unregister() -> None:
    trace: list[str] = []
    backend = FakeBackend(trace)
    transport = FakeTransport(trace)
    transport.register_error = TimeoutError("ambiguous registration timeout")
    value, _, _ = runtime(backend, transport)

    assert_error(StagingErrorCode.REGISTRATION_FAILED, value.open)
    assert value.state is StagingState.FAILED
    assert len(transport.registrations) == 1
    value.close()
    assert transport.unregister_count == 1
    assert value.state is StagingState.CLOSED


def test_close_unregisters_once_releases_owners_and_is_idempotent() -> None:
    value, _, transport = runtime()
    value.open()

    value.close()
    value.close()

    assert transport.unregister_count == 1
    assert value.state is StagingState.CLOSED
    assert_error(StagingErrorCode.INVALID_STATE, lambda: value.tensor)


def test_unregister_failure_retains_lifecycle_for_safe_retry() -> None:
    value, _, transport = runtime()
    value.open()
    transport.unregister_failures = 1

    assert_error(StagingErrorCode.UNREGISTER_FAILED, value.close)
    assert value.state is StagingState.FAILED
    value.close()
    assert transport.unregister_count == 2
    assert value.state is StagingState.CLOSED


def test_open_twice_and_close_during_transfer_are_rejected() -> None:
    value, _, _ = runtime()
    value.open()
    assert_error(StagingErrorCode.INVALID_STATE, value.open)

    with value.synchronous_transfer(
        direction=StagingTransferDirection.RETRIEVE,
        buffer_offset=0,
        token_extent=256,
    ):
        assert_error(StagingErrorCode.INVALID_STATE, value.close)

    assert value.state is StagingState.REGISTERED


def test_close_before_open_is_idempotent_and_does_not_unregister() -> None:
    value, _, transport = runtime()

    value.close()
    value.close()

    assert value.state is StagingState.CLOSED
    assert transport.unregister_count == 0


def test_lazy_loader_rejects_missing_distribution_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(distribution: str) -> str:
        raise StagingError(
            StagingErrorCode.DEPENDENCY_MISSING,
            f"{distribution} missing",
        )

    import_called = False

    def unexpected_import(name: str) -> object:
        nonlocal import_called
        import_called = True
        raise AssertionError(name)

    monkeypatch.setattr(module, "_distribution_version", missing)
    monkeypatch.setattr(module, "import_module", unexpected_import)

    assert_error(StagingErrorCode.DEPENDENCY_MISSING, module.load_staging_backend)
    assert not import_called


def test_lazy_loader_wraps_missing_module_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        module,
        "_distribution_version",
        {"torch": "2.10.0+cu128", "lmcache": "0.4.3"}.__getitem__,
    )

    def missing_import(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr(module, "import_module", missing_import)
    assert_error(StagingErrorCode.DEPENDENCY_MISSING, module.load_staging_backend)


@pytest.mark.parametrize(
    ("versions", "code"),
    [
        (
            {"torch": "2.10.1+cu128", "lmcache": "0.4.3"},
            StagingErrorCode.TORCH_VERSION_MISMATCH,
        ),
        (
            {"torch": "2.10.0+cu128", "lmcache": "0.4.4"},
            StagingErrorCode.LMCACHE_VERSION_MISMATCH,
        ),
    ],
)
def test_lazy_loader_rejects_unpinned_distributions(
    monkeypatch: pytest.MonkeyPatch,
    versions: dict[str, str],
    code: StagingErrorCode,
) -> None:
    monkeypatch.setattr(module, "_distribution_version", versions.__getitem__)
    assert_error(code, module.load_staging_backend)


def test_lazy_loader_rejects_unpinned_cuda_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        module,
        "_distribution_version",
        {"torch": "2.10.0+cu128", "lmcache": "0.4.3"}.__getitem__,
    )
    fake_torch = SimpleNamespace(
        __version__="2.10.0+cu128",
        version=SimpleNamespace(cuda="12.7"),
    )
    fake_custom_types = SimpleNamespace(CudaIPCWrapper=type("Wrapper", (), {}))
    monkeypatch.setattr(
        module,
        "import_module",
        lambda name: fake_torch if name == "torch" else fake_custom_types,
    )

    assert_error(StagingErrorCode.CUDA_VERSION_MISMATCH, module.load_staging_backend)


def test_lazy_loader_rejects_module_distribution_version_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        module,
        "_distribution_version",
        {"torch": "2.10.0+cu128", "lmcache": "0.4.3"}.__getitem__,
    )
    fake_torch = SimpleNamespace(
        __version__="2.10.1+cu128",
        version=SimpleNamespace(cuda="12.8"),
    )
    fake_custom_types = SimpleNamespace(CudaIPCWrapper=type("Wrapper", (), {}))
    monkeypatch.setattr(
        module,
        "import_module",
        lambda name: fake_torch if name == "torch" else fake_custom_types,
    )

    assert_error(StagingErrorCode.TORCH_VERSION_MISMATCH, module.load_staging_backend)


def test_lazy_loader_requires_exact_cuda_wrapper_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        module,
        "_distribution_version",
        {"torch": "2.10.0+cu128", "lmcache": "0.4.3"}.__getitem__,
    )
    fake_torch = SimpleNamespace(
        __version__="2.10.0+cu128",
        version=SimpleNamespace(cuda="12.8"),
    )
    fake_custom_types = SimpleNamespace(CudaIPCWrapper=None)
    monkeypatch.setattr(
        module,
        "import_module",
        lambda name: fake_torch if name == "torch" else fake_custom_types,
    )

    assert_error(StagingErrorCode.DEPENDENCY_MISSING, module.load_staging_backend)


def test_lazy_loader_accepts_exact_modules_without_allocating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        module,
        "_distribution_version",
        {"torch": "2.10.0+cu128", "lmcache": "0.4.3"}.__getitem__,
    )
    fake_torch = SimpleNamespace(
        __version__="2.10.0+cu128",
        version=SimpleNamespace(cuda="12.8"),
    )
    fake_custom_types = SimpleNamespace(CudaIPCWrapper=type("Wrapper", (), {}))
    imports: list[str] = []

    def fake_import(name: str) -> object:
        imports.append(name)
        return fake_torch if name == "torch" else fake_custom_types

    monkeypatch.setattr(module, "import_module", fake_import)

    backend = module.load_staging_backend()

    assert isinstance(backend, TorchLmcacheStagingBackend)
    assert imports == ["torch", "lmcache.v1.multiprocess.custom_types"]
    assert backend.torch_version == "2.10.0+cu128"
    assert backend.cuda_runtime == "12.8"
    assert backend.lmcache_version == "0.4.3"


def test_production_backend_uses_exact_torch_event_and_allocation_apis() -> None:
    class FakeDtype:
        def __str__(self) -> str:
            return STAGING_DTYPE

    class ProductionTensor:
        def __init__(
            self, shape: tuple[int, ...], dtype: object, device: str
        ) -> None:
            self.shape = shape
            self.dtype = dtype
            self.device = device

        def is_contiguous(self) -> bool:
            return True

        def storage_offset(self) -> int:
            return 0

    class ProductionEvent:
        def __init__(self) -> None:
            self.recorded_streams: list[object] = []
            self.synchronize_count = 0

        def record(self, stream: object) -> None:
            self.recorded_streams.append(stream)

        def synchronize(self) -> None:
            self.synchronize_count += 1

        def ipc_handle(self) -> bytes:
            return b"production-fake-event"

    class DeviceContext:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *args: object) -> None:
            return None

    class ProductionCuda:
        def __init__(self) -> None:
            self.initialized = False
            self.event_kwargs: list[dict[str, object]] = []
            self.events: list[ProductionEvent] = []
            self.synchronized_devices: list[int] = []
            self.stream = object()

        def is_available(self) -> bool:
            return True

        def init(self) -> None:
            self.initialized = True

        def device_count(self) -> int:
            return 1

        def get_device_properties(self, device_index: int) -> SimpleNamespace:
            assert device_index == 0
            return SimpleNamespace(
                name="NVIDIA A100-SXM4-80GB",
                uuid="GPU-a100",
            )

        def device(self, device_index: int) -> DeviceContext:
            assert device_index == 0
            return DeviceContext()

        def Event(self, **kwargs: object) -> ProductionEvent:
            self.event_kwargs.append(kwargs)
            event = ProductionEvent()
            self.events.append(event)
            return event

        def synchronize(self, device_index: int) -> None:
            self.synchronized_devices.append(device_index)

        def current_stream(self, device_index: int) -> object:
            assert device_index == 0
            return self.stream

    class ProductionTorch:
        Tensor = ProductionTensor

        def __init__(self) -> None:
            self.bfloat16 = FakeDtype()
            self.cuda = ProductionCuda()
            self.empty_calls: list[tuple[tuple[int, ...], object, str]] = []

        def empty(
            self, shape: tuple[int, ...], *, dtype: object, device: str
        ) -> ProductionTensor:
            self.empty_calls.append((shape, dtype, device))
            return ProductionTensor(shape, dtype, device)

    class ProductionWrapper:
        creation_count = 0

        def __init__(self, tensor: ProductionTensor) -> None:
            type(self).creation_count += 1
            self.shape = tensor.shape
            self.dtype = tensor.dtype
            self.stride = STRIDE
            self.storage_offset = 0
            self.device_uuid = "GPU-a100"
            self.handle = ("cuda-storage-handle",)

    fake_torch = ProductionTorch()
    backend = TorchLmcacheStagingBackend(
        fake_torch,
        ProductionWrapper,
        torch_version="2.10.0+cu128",
        cuda_runtime="12.8",
        lmcache_version="0.4.3",
    )
    transport = FakeTransport()
    value = StagingRuntime(config(), backend, transport)

    tensor = value.open()
    with value.synchronous_transfer(
        direction=StagingTransferDirection.RETRIEVE,
        buffer_offset=0,
        token_extent=256,
    ) as lease:
        assert lease.event_ipc_handle == b"production-fake-event"

    assert isinstance(tensor, ProductionTensor)
    assert fake_torch.cuda.initialized
    assert fake_torch.empty_calls == [(SHAPE, fake_torch.bfloat16, "cuda:0")]
    assert fake_torch.cuda.event_kwargs == [
        {
            "enable_timing": False,
            "blocking": False,
            "interprocess": True,
        }
    ]
    assert len(fake_torch.cuda.events) == 1
    event = fake_torch.cuda.events[0]
    assert event.recorded_streams == [fake_torch.cuda.stream] * 2
    assert event.synchronize_count == 2
    assert fake_torch.cuda.synchronized_devices == [0, 0, 0]
    assert ProductionWrapper.creation_count == 1
    assert len(transport.registrations) == 1
