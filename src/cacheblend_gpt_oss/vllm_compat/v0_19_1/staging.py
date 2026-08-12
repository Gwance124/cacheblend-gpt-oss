# SPDX-License-Identifier: Apache-2.0
"""Pinned CUDA/LMCache staging-buffer lifecycle for synchronous hooks.

This module owns the one contiguous buffer shared with LMCache, but does not
wire it into the connector.  Its source boundary is exact:

* LMCache 0.4.3 ``CudaIPCWrapper`` requires a contiguous tensor and serializes
  its CUDA storage handle, dtype, shape, stride, storage offset, and device UUID:
  https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/v1/multiprocess/custom_types.py#L23-L99
* Blend registration accepts exactly ``[instance_id, KVCache, model_name,
  world_size]`` and ``PlainGPUCacheContext`` accepts exactly one contiguous
  ``[2,L,T,D]`` tensor:
  https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/v1/multiprocess/protocols/blend.py#L95-L108
  https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/v1/multiprocess/gpu_context.py#L340-L406
* A store imports the client event, waits for it on LMCache's stream, copies
  staging to storage, and records a returned event. Retrieval records a returned
  event after writing staging (the inbound handle is present in the schema but
  is not waited on by this pinned implementation):
  https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/v1/multiprocess/blend_server_v2.py#L465-L595
  https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/v1/multiprocess/blend_server_v2.py#L597-L687
* LMCache's CUDA future imports and synchronizes that returned event before
  returning the transfer result:
  https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/v1/multiprocess/futures.py#L82-L139
* PyTorch 2.10 CUDA events are lazily initialized, can be exported only when
  created with ``interprocess=True``, record on a same-device stream, and expose
  ``record``, ``synchronize``, and ``ipc_handle``:
  https://github.com/pytorch/pytorch/blob/v2.10.0/torch/cuda/streams.py#L139-L225

For the first correctness milestone, connector hooks are synchronous. Before
each MQ call this runtime synchronizes the staging device, records and
synchronizes one reusable interprocess event, then yields its stable IPC handle.
The hook must perform the synchronous LMCache transport call inside that context.
On normal exit a second device synchronization establishes that the transport's
returned event has completed before staging is reused. This deliberately gives
up overlap in favor of an explicit, testable ownership order.

Registration is worker-process local: the worker that owns the CUDA allocation
creates one runtime and registers one wrapper for its ``instance_id``. The
scheduler process must not allocate or register staging. The tensor, wrapper,
and reusable event remain strongly referenced until synchronous unregister.

The exact event order is asymmetric in pinned LMCache. For stores, LMCache
imports and waits on the client event before reading staging, then records its
server event after enqueueing the store copies. For retrieval, LMCache 0.4.3
does not wait on the inbound client event; it records its server event after
enqueueing writes to staging. In both cases the existing transport's CUDA future
imports and synchronizes the returned server event before the context body can
finish. This runtime then performs its post-body device synchronization.

Torch and LMCache imports are lazy. CPU tests inject :class:`StagingBackend` and
:class:`StagingTransport` fakes and do not require either package.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any, NoReturn, Protocol

from cacheblend_gpt_oss.gpt_oss.layout import GPT_OSS_MAX_CONTEXT_TOKENS
from cacheblend_gpt_oss.storage.lmcache_types import (
    LMCACHE_CHUNK_SIZE,
    LMCACHE_VERSION,
    LmcacheStagingLayout,
    LmcacheStagingRegistration,
    validate_event_handle,
)
from cacheblend_gpt_oss.targets import PINNED_TARGET

STAGING_COMPONENTS = 2
STAGING_LAYERS = 24
STAGING_KV_WIDTH = 512
STAGING_DTYPE = "torch.bfloat16"


class StagingState(str, Enum):
    """Fail-closed lifecycle of one staging allocation."""

    CREATED = "created"
    REGISTERED = "registered"
    TRANSFER_ACTIVE = "transfer_active"
    FAILED = "failed"
    CLOSED = "closed"


class StagingTransferDirection(str, Enum):
    """Which synchronous LMCache operation owns staging during one lease."""

    STORE = "store"
    RETRIEVE = "retrieve"


class StagingErrorCode(str, Enum):
    """Bounded staging failures suitable for startup/request fallback policy."""

    INVALID_CONFIG = "invalid_config"
    INVALID_STATE = "invalid_state"
    DEPENDENCY_MISSING = "dependency_missing"
    TORCH_VERSION_MISMATCH = "torch_version_mismatch"
    LMCACHE_VERSION_MISMATCH = "lmcache_version_mismatch"
    CUDA_VERSION_MISMATCH = "cuda_version_mismatch"
    CUDA_DEVICE_UNAVAILABLE = "cuda_device_unavailable"
    GPU_MISMATCH = "gpu_mismatch"
    ALLOCATION_FAILED = "allocation_failed"
    INVALID_TENSOR = "invalid_tensor"
    EVENT_FAILED = "event_failed"
    IPC_WRAPPER_FAILED = "ipc_wrapper_failed"
    REGISTRATION_FAILED = "registration_failed"
    SYNCHRONIZATION_FAILED = "synchronization_failed"
    TRANSFER_FAILED = "transfer_failed"
    UNREGISTER_FAILED = "unregister_failed"


class StagingError(RuntimeError):
    """Fail-closed staging error with a stable machine-readable code."""

    def __init__(self, code: StagingErrorCode, message: str = "") -> None:
        self.code = code
        super().__init__(message or code.value)


def _fail(code: StagingErrorCode, message: str = "") -> NoReturn:
    raise StagingError(code, message)


@dataclass(frozen=True, slots=True)
class StagingConfig:
    """Immutable allocation identity for the sole supported staging layout."""

    instance_id: int
    token_capacity: int
    device: str

    def __post_init__(self) -> None:
        _require_plain_int("instance_id", self.instance_id, minimum=0)
        _require_plain_int(
            "token_capacity", self.token_capacity, minimum=LMCACHE_CHUNK_SIZE
        )
        if self.token_capacity % LMCACHE_CHUNK_SIZE != 0:
            _fail(
                StagingErrorCode.INVALID_CONFIG,
                f"token_capacity must be aligned to {LMCACHE_CHUNK_SIZE} tokens",
            )
        if self.token_capacity > GPT_OSS_MAX_CONTEXT_TOKENS:
            _fail(
                StagingErrorCode.INVALID_CONFIG,
                "token_capacity exceeds GPT-OSS-20B's 131072-token context",
            )
        _parse_indexed_cuda_device(self.device)

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Return the exact LMCache staging tensor shape."""

        return (
            STAGING_COMPONENTS,
            STAGING_LAYERS,
            self.token_capacity,
            STAGING_KV_WIDTH,
        )

    @property
    def device_index(self) -> int:
        """Return the explicitly configured CUDA ordinal."""

        return _parse_indexed_cuda_device(self.device)


@dataclass(frozen=True, slots=True)
class StagingTransferLease:
    """Validated bridge inputs bound to one client-event recording.

    ``buffer_offset`` is the exact offset sent on the LMCache wire and later
    passed to the data plane. For retrieval, a candidate at query-relative
    target position ``p`` is written/read at ``buffer_offset + p``.
    """

    direction: StagingTransferDirection
    buffer_offset: int
    token_extent: int
    capacity: int
    event_ipc_handle: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.direction, StagingTransferDirection):
            _fail(StagingErrorCode.INVALID_CONFIG, "invalid staging direction")
        _require_plain_int("buffer_offset", self.buffer_offset, minimum=0)
        _require_plain_int("token_extent", self.token_extent, minimum=1)
        _require_plain_int("capacity", self.capacity, minimum=1)
        if self.buffer_offset + self.token_extent > self.capacity:
            _fail(
                StagingErrorCode.INVALID_CONFIG,
                "staging transfer placement exceeds tensor capacity",
            )
        validate_event_handle(self.event_ipc_handle)

    @property
    def end_offset(self) -> int:
        """Return the exclusive end of the staging placement."""

        return self.buffer_offset + self.token_extent

    def staging_position(self, relative_position: int) -> int:
        """Map a query/document-relative position into registered staging."""

        _require_plain_int("relative_position", relative_position, minimum=0)
        if relative_position >= self.token_extent:
            _fail(
                StagingErrorCode.INVALID_CONFIG,
                "relative position exceeds this staging transfer",
            )
        return self.buffer_offset + relative_position


class StagingTransport(Protocol):
    """Injected subset of the pinned synchronous LMCache transport."""

    def register_staging_buffer(
        self, registration: LmcacheStagingRegistration
    ) -> None:
        """Synchronously register exactly one CUDA IPC wrapper."""

    def unregister_staging_buffer(self) -> None:
        """Synchronously unregister the active staging buffer."""


class StagingBackend(Protocol):
    """Injected CUDA/Torch/LMCache primitive boundary."""

    @property
    def torch_version(self) -> str:
        """Return the exact Torch module version."""

    @property
    def cuda_runtime(self) -> str:
        """Return the CUDA runtime against which Torch was built."""

    @property
    def lmcache_version(self) -> str:
        """Return the installed LMCache distribution version."""

    def validate_device(self, device_index: int, expected_name: str) -> None:
        """Fail unless the exact indexed target GPU is available."""

    def device_uuid(self, device_index: int) -> str:
        """Return the UUID used by LMCache to resolve an IPC device."""

    def allocate(self, shape: tuple[int, ...], dtype_name: str, device: str) -> object:
        """Allocate one new contiguous tensor."""

    def tensor_shape(self, tensor: object) -> tuple[int, ...]:
        """Return tensor shape."""

    def tensor_dtype_name(self, tensor: object) -> str:
        """Return tensor dtype name."""

    def tensor_device_name(self, tensor: object) -> str:
        """Return the indexed tensor device name."""

    def tensor_is_contiguous(self, tensor: object) -> bool:
        """Return whether tensor uses a contiguous layout."""

    def tensor_storage_offset(self, tensor: object) -> int:
        """Return the tensor's storage offset."""

    def make_cuda_ipc_wrapper(self, tensor: object) -> object:
        """Construct exactly one LMCache 0.4.3 ``CudaIPCWrapper``."""

    def wrapper_shape(self, wrapper: object) -> tuple[int, ...]:
        """Return the shape declared in a CUDA IPC wrapper."""

    def wrapper_dtype_name(self, wrapper: object) -> str:
        """Return the dtype declared in a CUDA IPC wrapper."""

    def wrapper_storage_offset(self, wrapper: object) -> int:
        """Return the storage offset declared in a CUDA IPC wrapper."""

    def wrapper_stride(self, wrapper: object) -> tuple[int, ...]:
        """Return the stride declared in a CUDA IPC wrapper."""

    def wrapper_device_uuid(self, wrapper: object) -> str:
        """Return the device UUID declared in a CUDA IPC wrapper."""

    def wrapper_has_handle(self, wrapper: object) -> bool:
        """Return whether the wrapper contains an exported CUDA storage handle."""

    def create_interprocess_event(self, device_index: int) -> object:
        """Create a reusable, non-timing interprocess CUDA event."""

    def synchronize_device(self, device_index: int) -> None:
        """Synchronize all work on the staging CUDA device."""

    def record_event(self, event: object, device_index: int) -> None:
        """Record the event on the current stream of its device."""

    def synchronize_event(self, event: object) -> None:
        """Wait for all work captured by the recorded event."""

    def event_ipc_handle(self, event: object) -> bytes:
        """Export the event's reusable IPC handle."""


class StagingRuntime:
    """Own one registered staging tensor and one reusable IPC event."""

    def __init__(
        self,
        config: StagingConfig,
        backend: StagingBackend,
        transport: StagingTransport,
    ) -> None:
        self._config = config
        self._backend = backend
        self._transport = transport
        self._state = StagingState.CREATED
        self._tensor: object | None = None
        self._wrapper: object | None = None
        self._event: object | None = None
        self._event_handle: bytes | None = None
        self._registration_may_exist = False
        self._validate_versions()

    @property
    def state(self) -> StagingState:
        """Return the current fail-closed lifecycle state."""

        return self._state

    @property
    def config(self) -> StagingConfig:
        """Return the immutable staging configuration."""

        return self._config

    @property
    def tensor(self) -> object:
        """Return staging only while registered and not owned by LMCache."""

        self._require_state(StagingState.REGISTERED)
        if self._tensor is None:
            self._mark_failed()
            _fail(StagingErrorCode.INVALID_STATE, "registered staging has no tensor")
        return self._tensor

    def open(self) -> object:
        """Allocate, validate, wrap, initialize the event, and register staging."""

        self._require_state(StagingState.CREATED)
        try:
            self._backend.validate_device(
                self._config.device_index, PINNED_TARGET.gpu_name
            )
        except StagingError:
            self._mark_failed()
            raise
        except Exception as exc:
            self._mark_failed()
            raise StagingError(
                StagingErrorCode.CUDA_DEVICE_UNAVAILABLE,
                "the exact indexed A100 staging device is unavailable",
            ) from exc

        try:
            tensor = self._backend.allocate(
                self._config.shape, STAGING_DTYPE, self._config.device
            )
        except Exception as exc:
            self._mark_failed()
            raise StagingError(
                StagingErrorCode.ALLOCATION_FAILED,
                "failed to allocate contiguous BF16 CUDA staging",
            ) from exc
        self._tensor = tensor
        self._validate_tensor(tensor)

        try:
            event = self._backend.create_interprocess_event(
                self._config.device_index
            )
        except Exception as exc:
            self._mark_failed()
            raise StagingError(
                StagingErrorCode.EVENT_FAILED,
                "failed to create the interprocess CUDA event",
            ) from exc
        self._event = event
        self._event_handle = self._record_and_export_event(expected_handle=None)

        try:
            wrapper = self._backend.make_cuda_ipc_wrapper(tensor)
        except Exception as exc:
            self._mark_failed()
            raise StagingError(
                StagingErrorCode.IPC_WRAPPER_FAILED,
                "LMCache CudaIPCWrapper construction failed",
            ) from exc
        self._wrapper = wrapper
        self._validate_wrapper(wrapper)
        layout = LmcacheStagingLayout(
            layer_count=STAGING_LAYERS,
            token_capacity=self._config.token_capacity,
            kv_width=STAGING_KV_WIDTH,
            dtype_name=STAGING_DTYPE,
        )
        registration = LmcacheStagingRegistration(
            instance_id=self._config.instance_id,
            kv_cache_payload=(wrapper,),
            layout=layout,
        )
        self._registration_may_exist = True
        try:
            self._transport.register_staging_buffer(registration)
        except Exception as exc:
            self._mark_failed()
            raise StagingError(
                StagingErrorCode.REGISTRATION_FAILED,
                "synchronous LMCache staging registration failed",
            ) from exc
        self._state = StagingState.REGISTERED
        return tensor

    @contextmanager
    def synchronous_transfer(
        self,
        *,
        direction: StagingTransferDirection,
        buffer_offset: int,
        token_extent: int,
    ) -> Iterator[StagingTransferLease]:
        """Lease validated bridge inputs around one synchronous MQ call.

        Data-plane writes must finish before entering this context. The body must
        make exactly one synchronous LMCache store or retrieve call. A body
        exception marks staging failed so the handle cannot be silently reused.
        """

        self._require_state(StagingState.REGISTERED)
        if not isinstance(direction, StagingTransferDirection):
            _fail(StagingErrorCode.INVALID_CONFIG, "invalid staging direction")
        _require_plain_int("buffer_offset", buffer_offset, minimum=0)
        _require_plain_int("token_extent", token_extent, minimum=1)
        if buffer_offset + token_extent > self._config.token_capacity:
            _fail(
                StagingErrorCode.INVALID_CONFIG,
                "staging transfer placement exceeds tensor capacity",
            )
        if (
            direction is StagingTransferDirection.STORE
            and token_extent % LMCACHE_CHUNK_SIZE != 0
        ):
            _fail(
                StagingErrorCode.INVALID_CONFIG,
                "LMCache stores require a complete-chunk token extent",
            )
        handle = self._record_and_export_event(expected_handle=self._event_handle)
        lease = StagingTransferLease(
            direction=direction,
            buffer_offset=buffer_offset,
            token_extent=token_extent,
            capacity=self._config.token_capacity,
            event_ipc_handle=handle,
        )
        self._state = StagingState.TRANSFER_ACTIVE
        try:
            yield lease
        except Exception as exc:
            self._best_effort_synchronize()
            self._mark_failed()
            raise StagingError(
                StagingErrorCode.TRANSFER_FAILED,
                "synchronous LMCache staging transfer failed",
            ) from exc
        except BaseException:
            self._best_effort_synchronize()
            self._mark_failed()
            raise
        try:
            self._synchronize_device()
        except StagingError:
            self._mark_failed()
            raise
        self._state = StagingState.REGISTERED

    def close(self) -> None:
        """Synchronously unregister, then release local owners; success is idempotent.

        If unregister fails, CUDA objects remain strongly referenced and state is
        ``FAILED``. A later call may retry; staging storage is never deliberately
        released while the server might still hold its IPC declaration.
        """

        if self._state is StagingState.CLOSED:
            return
        if self._state is StagingState.TRANSFER_ACTIVE:
            _fail(
                StagingErrorCode.INVALID_STATE,
                "cannot close staging during an active transfer",
            )
        if self._registration_may_exist:
            try:
                self._transport.unregister_staging_buffer()
            except Exception as exc:
                self._mark_failed()
                raise StagingError(
                    StagingErrorCode.UNREGISTER_FAILED,
                    "LMCache staging unregister failed; CUDA owners were retained",
                ) from exc
            self._registration_may_exist = False
        self._tensor = None
        self._wrapper = None
        self._event = None
        self._event_handle = None
        self._state = StagingState.CLOSED

    def _validate_versions(self) -> None:
        try:
            torch_version = self._backend.torch_version
            cuda_runtime = self._backend.cuda_runtime
            lmcache_version = self._backend.lmcache_version
        except Exception as exc:
            raise StagingError(
                StagingErrorCode.DEPENDENCY_MISSING,
                "could not inspect staging runtime versions",
            ) from exc
        if torch_version != PINNED_TARGET.torch_version:
            _fail(
                StagingErrorCode.TORCH_VERSION_MISMATCH,
                f"expected Torch {PINNED_TARGET.torch_version}; got {torch_version!r}",
            )
        if cuda_runtime != PINNED_TARGET.cuda_runtime:
            _fail(
                StagingErrorCode.CUDA_VERSION_MISMATCH,
                f"expected CUDA {PINNED_TARGET.cuda_runtime}; got {cuda_runtime!r}",
            )
        if lmcache_version != LMCACHE_VERSION:
            _fail(
                StagingErrorCode.LMCACHE_VERSION_MISMATCH,
                f"expected LMCache {LMCACHE_VERSION}; got {lmcache_version!r}",
            )

    def _validate_tensor(self, tensor: object) -> None:
        try:
            shape = self._backend.tensor_shape(tensor)
            dtype_name = self._backend.tensor_dtype_name(tensor)
            device_name = self._backend.tensor_device_name(tensor)
            contiguous = self._backend.tensor_is_contiguous(tensor)
            storage_offset = self._backend.tensor_storage_offset(tensor)
        except Exception as exc:
            self._mark_failed()
            raise StagingError(
                StagingErrorCode.INVALID_TENSOR,
                "could not inspect the allocated staging tensor",
            ) from exc
        if (
            shape != self._config.shape
            or dtype_name != STAGING_DTYPE
            or device_name != self._config.device
            or contiguous is not True
            or isinstance(storage_offset, bool)
            or not isinstance(storage_offset, int)
            or storage_offset != 0
        ):
            self._mark_failed()
            _fail(
                StagingErrorCode.INVALID_TENSOR,
                "staging must be a direct contiguous BF16 CUDA [2,24,T,512] tensor",
            )

    def _validate_wrapper(self, wrapper: object) -> None:
        try:
            shape = self._backend.wrapper_shape(wrapper)
            dtype_name = self._backend.wrapper_dtype_name(wrapper)
            storage_offset = self._backend.wrapper_storage_offset(wrapper)
            stride = self._backend.wrapper_stride(wrapper)
            wrapper_device_uuid = self._backend.wrapper_device_uuid(wrapper)
            expected_device_uuid = self._backend.device_uuid(
                self._config.device_index
            )
            has_handle = self._backend.wrapper_has_handle(wrapper)
        except Exception as exc:
            self._mark_failed()
            raise StagingError(
                StagingErrorCode.IPC_WRAPPER_FAILED,
                "could not inspect the LMCache CUDA IPC declaration",
            ) from exc
        if (
            shape != self._config.shape
            or dtype_name != STAGING_DTYPE
            or isinstance(storage_offset, bool)
            or not isinstance(storage_offset, int)
            or storage_offset != 0
            or stride != _contiguous_stride(self._config.shape)
            or not wrapper_device_uuid
            or wrapper_device_uuid != expected_device_uuid
            or has_handle is not True
        ):
            self._mark_failed()
            _fail(
                StagingErrorCode.IPC_WRAPPER_FAILED,
                "LMCache CUDA IPC declaration does not match staging",
            )

    def _record_and_export_event(self, expected_handle: bytes | None) -> bytes:
        event = self._event
        if event is None:
            self._mark_failed()
            _fail(StagingErrorCode.EVENT_FAILED, "staging event is unavailable")
        try:
            self._synchronize_device()
        except StagingError:
            self._mark_failed()
            raise
        try:
            self._backend.record_event(event, self._config.device_index)
            self._backend.synchronize_event(event)
            handle = self._backend.event_ipc_handle(event)
            validate_event_handle(handle)
        except Exception as exc:
            self._mark_failed()
            raise StagingError(
                StagingErrorCode.EVENT_FAILED,
                "CUDA event record, synchronize, or IPC export failed",
            ) from exc
        if expected_handle is not None and handle != expected_handle:
            self._mark_failed()
            _fail(
                StagingErrorCode.EVENT_FAILED,
                "re-exported CUDA event IPC handle changed unexpectedly",
            )
        return handle

    def _synchronize_device(self) -> None:
        try:
            self._backend.synchronize_device(self._config.device_index)
        except Exception as exc:
            raise StagingError(
                StagingErrorCode.SYNCHRONIZATION_FAILED,
                "staging CUDA device synchronization failed",
            ) from exc

    def _best_effort_synchronize(self) -> None:
        with suppress(Exception):
            self._backend.synchronize_device(self._config.device_index)

    def _mark_failed(self) -> None:
        if self._state is not StagingState.CLOSED:
            self._state = StagingState.FAILED

    def _require_state(self, expected: StagingState) -> None:
        if self._state is not expected:
            _fail(
                StagingErrorCode.INVALID_STATE,
                f"staging must be {expected.value}; "
                f"current state is {self._state.value}",
            )


class TorchLmcacheStagingBackend:
    """Production backend over lazily supplied pinned Torch and LMCache types."""

    def __init__(
        self,
        torch_module: object,
        cuda_wrapper_type: type[Any],
        *,
        torch_version: str,
        cuda_runtime: str,
        lmcache_version: str,
    ) -> None:
        self._torch: Any = torch_module
        self._cuda_wrapper_type = cuda_wrapper_type
        self._torch_version = torch_version
        self._cuda_runtime = cuda_runtime
        self._lmcache_version = lmcache_version

    @property
    def torch_version(self) -> str:
        return self._torch_version

    @property
    def cuda_runtime(self) -> str:
        return self._cuda_runtime

    @property
    def lmcache_version(self) -> str:
        return self._lmcache_version

    def validate_device(self, device_index: int, expected_name: str) -> None:
        cuda = self._torch.cuda
        if not cuda.is_available():
            _fail(StagingErrorCode.CUDA_DEVICE_UNAVAILABLE, "CUDA is unavailable")
        cuda.init()
        if device_index >= int(cuda.device_count()):
            _fail(
                StagingErrorCode.CUDA_DEVICE_UNAVAILABLE,
                "CUDA device index is unavailable",
            )
        observed_name = str(cuda.get_device_properties(device_index).name)
        if observed_name != expected_name:
            _fail(
                StagingErrorCode.GPU_MISMATCH,
                f"expected GPU {expected_name!r}; got {observed_name!r}"
            )

    def device_uuid(self, device_index: int) -> str:
        return str(self._torch.cuda.get_device_properties(device_index).uuid)

    def allocate(self, shape: tuple[int, ...], dtype_name: str, device: str) -> object:
        if dtype_name != STAGING_DTYPE:
            raise ValueError("unsupported staging dtype")
        return self._torch.empty(
            shape,
            dtype=self._torch.bfloat16,
            device=device,
        )

    def _require_tensor(self, tensor: object) -> Any:
        tensor_type = getattr(self._torch, "Tensor", None)
        if tensor_type is None or not isinstance(tensor, tensor_type):
            raise TypeError("expected a torch.Tensor")
        return tensor

    def tensor_shape(self, tensor: object) -> tuple[int, ...]:
        value = self._require_tensor(tensor)
        return tuple(int(dimension) for dimension in value.shape)

    def tensor_dtype_name(self, tensor: object) -> str:
        return str(self._require_tensor(tensor).dtype)

    def tensor_device_name(self, tensor: object) -> str:
        return str(self._require_tensor(tensor).device)

    def tensor_is_contiguous(self, tensor: object) -> bool:
        return bool(self._require_tensor(tensor).is_contiguous())

    def tensor_storage_offset(self, tensor: object) -> int:
        return int(self._require_tensor(tensor).storage_offset())

    def make_cuda_ipc_wrapper(self, tensor: object) -> object:
        return self._cuda_wrapper_type(self._require_tensor(tensor))

    def _require_wrapper(self, wrapper: object) -> Any:
        if not isinstance(wrapper, self._cuda_wrapper_type):
            raise TypeError("expected an LMCache CudaIPCWrapper")
        return wrapper

    def wrapper_shape(self, wrapper: object) -> tuple[int, ...]:
        value = self._require_wrapper(wrapper)
        return tuple(int(dimension) for dimension in value.shape)

    def wrapper_dtype_name(self, wrapper: object) -> str:
        return str(self._require_wrapper(wrapper).dtype)

    def wrapper_storage_offset(self, wrapper: object) -> int:
        return int(self._require_wrapper(wrapper).storage_offset)

    def wrapper_stride(self, wrapper: object) -> tuple[int, ...]:
        value = self._require_wrapper(wrapper)
        return tuple(int(dimension) for dimension in value.stride)

    def wrapper_device_uuid(self, wrapper: object) -> str:
        return str(self._require_wrapper(wrapper).device_uuid)

    def wrapper_has_handle(self, wrapper: object) -> bool:
        return bool(self._require_wrapper(wrapper).handle)

    def create_interprocess_event(self, device_index: int) -> object:
        with self._torch.cuda.device(device_index):
            return self._torch.cuda.Event(
                enable_timing=False,
                blocking=False,
                interprocess=True,
            )

    def synchronize_device(self, device_index: int) -> None:
        self._torch.cuda.synchronize(device_index)

    def record_event(self, event: object, device_index: int) -> None:
        with self._torch.cuda.device(device_index):
            stream = self._torch.cuda.current_stream(device_index)
            event_value: Any = event
            event_value.record(stream)

    def synchronize_event(self, event: object) -> None:
        event_value: Any = event
        event_value.synchronize()

    def event_ipc_handle(self, event: object) -> bytes:
        event_value: Any = event
        handle = event_value.ipc_handle()
        if not isinstance(handle, bytes):
            raise TypeError("Torch CUDA event returned a non-bytes IPC handle")
        return handle


def load_staging_backend() -> StagingBackend:
    """Lazily load the exact pinned Torch and LMCache production primitives."""

    torch_distribution = _distribution_version("torch")
    lmcache_distribution = _distribution_version("lmcache")
    if torch_distribution != PINNED_TARGET.torch_version:
        _fail(
            StagingErrorCode.TORCH_VERSION_MISMATCH,
            f"expected Torch {PINNED_TARGET.torch_version}; "
            f"got {torch_distribution!r}",
        )
    if lmcache_distribution != LMCACHE_VERSION:
        _fail(
            StagingErrorCode.LMCACHE_VERSION_MISMATCH,
            f"expected LMCache {LMCACHE_VERSION}; got {lmcache_distribution!r}",
        )
    try:
        torch = import_module("torch")
        custom_types = import_module("lmcache.v1.multiprocess.custom_types")
    except ImportError as exc:
        raise StagingError(
            StagingErrorCode.DEPENDENCY_MISSING,
            "could not import pinned Torch/LMCache staging dependencies",
        ) from exc
    module_torch_version = str(getattr(torch, "__version__", ""))
    if module_torch_version != PINNED_TARGET.torch_version:
        _fail(
            StagingErrorCode.TORCH_VERSION_MISMATCH,
            "Torch module version differs from the pinned distribution",
        )
    cuda_runtime = str(getattr(getattr(torch, "version", None), "cuda", ""))
    if cuda_runtime != PINNED_TARGET.cuda_runtime:
        _fail(
            StagingErrorCode.CUDA_VERSION_MISMATCH,
            f"expected CUDA {PINNED_TARGET.cuda_runtime}; got {cuda_runtime!r}",
        )
    wrapper_type = getattr(custom_types, "CudaIPCWrapper", None)
    if not isinstance(wrapper_type, type):
        _fail(
            StagingErrorCode.DEPENDENCY_MISSING,
            "LMCache 0.4.3 CudaIPCWrapper is unavailable",
        )
    return TorchLmcacheStagingBackend(
        torch,
        wrapper_type,
        torch_version=module_torch_version,
        cuda_runtime=cuda_runtime,
        lmcache_version=lmcache_distribution,
    )


def create_staging_runtime(
    config: StagingConfig,
    transport: StagingTransport,
) -> StagingRuntime:
    """Lazily allocate and synchronously register production staging."""

    runtime = StagingRuntime(config, load_staging_backend(), transport)
    runtime.open()
    return runtime


def _distribution_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError as exc:
        raise StagingError(
            StagingErrorCode.DEPENDENCY_MISSING,
            f"required distribution {distribution!r} is not installed",
        ) from exc


def _parse_indexed_cuda_device(device: object) -> int:
    if not isinstance(device, str) or not device.startswith("cuda:"):
        _fail(
            StagingErrorCode.INVALID_CONFIG,
            "device must be an explicitly indexed CUDA device",
        )
    suffix = device.removeprefix("cuda:")
    if (
        not suffix.isdigit()
        or len(suffix) > 6
        or str(int(suffix)) != suffix
    ):
        _fail(
            StagingErrorCode.INVALID_CONFIG,
            "device must use canonical form cuda:<non-negative integer>",
        )
    return int(suffix)


def _require_plain_int(name: str, value: object, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(
            StagingErrorCode.INVALID_CONFIG,
            f"{name} must be a plain integer >= {minimum}",
        )


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    reversed_strides: list[int] = []
    for dimension in reversed(shape):
        reversed_strides.append(stride)
        stride *= dimension
    return tuple(reversed(reversed_strides))


__all__ = [
    "STAGING_COMPONENTS",
    "STAGING_DTYPE",
    "STAGING_KV_WIDTH",
    "STAGING_LAYERS",
    "StagingBackend",
    "StagingConfig",
    "StagingError",
    "StagingErrorCode",
    "StagingRuntime",
    "StagingState",
    "StagingTransferDirection",
    "StagingTransferLease",
    "StagingTransport",
    "TorchLmcacheStagingBackend",
    "create_staging_runtime",
    "load_staging_backend",
]
