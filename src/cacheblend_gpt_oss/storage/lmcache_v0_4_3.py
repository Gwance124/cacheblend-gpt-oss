"""LMCache 0.4.3 BlendEngineV2 message-queue boundary.

This module follows the exact public LMCache commit
``7f326118a2f1afc7801988dd02e3055bdf21ef6b``:

* ``IPCCacheEngineKey`` and ``CBMatchResult`` wire fields:
  https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/v1/multiprocess/custom_types.py#L121-L180
  and
  https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/v1/multiprocess/custom_types.py#L234-L250
* exact Blend V2 lookup/retrieve payload schemas:
  https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/v1/multiprocess/protocols/blend_v2.py#L29-L56
* register, precomputed-store, final-store, and unregister schemas:
  https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/v1/multiprocess/protocols/blend.py#L31-L108
* ``MessageQueueClient.submit_request`` and future behavior:
  https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/v1/multiprocess/mq.py#L110-L247
* server candidate generation and storage-prefetch filtering:
  https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/v1/multiprocess/blend_server_v2.py#L340-L460
* one contiguous ``[2,L,T,D]`` staging buffer:
  https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/v1/multiprocess/gpu_context.py#L340-L406

LMCache's matcher is a rolling-hash candidate index, not exact-token proof.
Accordingly, :meth:`LmcacheBlendTransport.lookup_candidates` returns only
``LmcacheCandidate`` values and retrieval accepts only independently bound
``VerifiedLmcacheCandidate`` values.

There are no top-level LMCache, Torch, CUDA, or ZeroMQ imports.  Production
bindings are loaded lazily by :func:`create_lmcache_blend_transport`; unit tests
inject protocol and message-queue fakes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from types import GenericAlias
from typing import Any, NoReturn, Protocol, cast

from cacheblend_gpt_oss.planner.models import TokenRange, normalize_token_ids
from cacheblend_gpt_oss.storage.lmcache_types import (
    LMCACHE_VERSION,
    LmcacheBlendTransportConfig,
    LmcacheCandidate,
    LmcacheCloseError,
    LmcacheConfigurationError,
    LmcacheDependencyError,
    LmcacheLifecycleError,
    LmcacheOperationError,
    LmcacheProtocolError,
    LmcacheRetrieveReceipt,
    LmcacheStagingRegistration,
    LmcacheStoreReceipt,
    LmcacheTransportState,
    VerifiedLmcacheCandidate,
    query_digest,
    validate_buffer_range,
    validate_event_handle,
    validate_request_id,
)


class LmcacheRequest(str, Enum):
    """The exact LMCache request members used by this adapter."""

    PING = "PING"
    GET_CHUNK_SIZE = "GET_CHUNK_SIZE"
    REGISTER = "CB_REGISTER_KV_CACHE"
    UNREGISTER = "CB_UNREGISTER_KV_CACHE"
    LOOKUP = "CB_LOOKUP_PRE_COMPUTED_V2"
    STORE_PRECOMPUTED = "CB_STORE_PRE_COMPUTED"
    RETRIEVE = "CB_RETRIEVE_PRE_COMPUTED_V2"
    STORE_FINAL = "CB_STORE_FINAL"


class MessageFuture(Protocol):
    """Small subset of LMCache ``MessagingFuture`` consumed by the adapter."""

    def result(self, timeout: float | None = None) -> object:
        """Wait for and return the decoded response."""


class MessageQueue(Protocol):
    """Injected subset of LMCache ``MessageQueueClient``."""

    def submit_request(
        self,
        request_type: object,
        request_payloads: list[object],
        response_cls: object | None = None,
    ) -> MessageFuture:
        """Submit one exact-schema request."""

    def close(self) -> None:
        """Close client sockets and the worker thread."""


class LmcacheBindings(Protocol):
    """Dependency-injected protocol/type adapter for LMCache 0.4.3."""

    @property
    def lmcache_version(self) -> str:
        """Return the installed LMCache distribution version."""

    def validate_protocol_schema(self) -> None:
        """Fail unless all used request payload and response types are exact."""

    def request_type(self, request: LmcacheRequest) -> object:
        """Resolve a pinned request enum member."""

    def response_class(self, request: LmcacheRequest) -> object | None:
        """Resolve the response class registered by LMCache."""

    def make_key(
        self,
        *,
        model_name: str,
        world_size: int,
        worker_id: int | None,
        token_ids: tuple[int, ...],
        start: int,
        end: int,
        request_id: str,
    ) -> object:
        """Construct an exact ``IPCCacheEngineKey``."""

    def parse_match(self, raw_match: object) -> tuple[int, int, int, int, bytes]:
        """Validate and unpack one exact ``CBMatchResult``."""

    def make_match(
        self, *, old_st: int, old_ed: int, cur_st: int, cur_ed: int, hash: bytes
    ) -> object:
        """Construct an exact ``CBMatchResult`` for retrieval."""

    def validate_kv_cache_payload(
        self,
        payload: list[object],
        *,
        expected_shape: tuple[int, int, int, int],
        expected_dtype_name: str,
    ) -> None:
        """Validate the one wrapped staging tensor without opening CUDA IPC."""

    def wait(self, future: MessageFuture, timeout: float) -> object:
        """Wait for an ordinary message response."""

    def wait_cuda(self, future: MessageFuture, timeout: float) -> object:
        """Wait for both the message response and returned CUDA event."""


@dataclass(frozen=True, slots=True)
class _RuntimeLmcacheBindings:
    """Runtime objects imported from the exact installed LMCache package."""

    _version: str
    _request_type_cls: Any
    _key_type: type[Any]
    _match_type: type[Any]
    _cuda_wrapper_type: type[Any]
    _get_payload_classes: Any
    _get_response_class: Any

    @property
    def lmcache_version(self) -> str:
        return self._version

    def validate_protocol_schema(self) -> None:
        key = self._key_type
        match = self._match_type
        wrapper = self._cuda_wrapper_type
        wrapper_list = GenericAlias(list, wrapper)
        match_list = GenericAlias(list, match)
        transfer_response = GenericAlias(tuple, (bytes, bool))
        expected: dict[LmcacheRequest, tuple[list[object], object | None]] = {
            LmcacheRequest.PING: ([], bool),
            LmcacheRequest.GET_CHUNK_SIZE: ([], int),
            LmcacheRequest.REGISTER: ([int, wrapper_list, str, int], None),
            LmcacheRequest.UNREGISTER: ([int], None),
            LmcacheRequest.LOOKUP: ([key], match_list),
            LmcacheRequest.STORE_PRECOMPUTED: (
                [key, int, int, bytes],
                transfer_response,
            ),
            LmcacheRequest.RETRIEVE: (
                [key, match_list, int, int, bytes],
                transfer_response,
            ),
            LmcacheRequest.STORE_FINAL: (
                [key, int, int, bytes],
                transfer_response,
            ),
        }
        for request, (payload_classes, response_class) in expected.items():
            request_type = self.request_type(request)
            actual_payloads = self._get_payload_classes(request_type)
            actual_response = self._get_response_class(request_type)
            if actual_payloads != payload_classes or actual_response != response_class:
                raise LmcacheProtocolError(
                    f"LMCache 0.4.3 protocol schema mismatch for {request.value}"
                )

    def request_type(self, request: LmcacheRequest) -> object:
        try:
            return getattr(self._request_type_cls, request.value)
        except AttributeError as exc:
            raise LmcacheProtocolError(
                f"LMCache 0.4.3 lacks request type {request.value}"
            ) from exc

    def response_class(self, request: LmcacheRequest) -> object | None:
        return cast(object | None, self._get_response_class(self.request_type(request)))

    def make_key(
        self,
        *,
        model_name: str,
        world_size: int,
        worker_id: int | None,
        token_ids: tuple[int, ...],
        start: int,
        end: int,
        request_id: str,
    ) -> object:
        return self._key_type(
            model_name=model_name,
            world_size=world_size,
            worker_id=worker_id,
            token_ids=token_ids,
            start=start,
            end=end,
            request_id=request_id,
        )

    def parse_match(self, raw_match: object) -> tuple[int, int, int, int, bytes]:
        if not isinstance(raw_match, self._match_type):
            raise LmcacheProtocolError("lookup returned a non-CBMatchResult value")
        return (
            raw_match.old_st,
            raw_match.old_ed,
            raw_match.cur_st,
            raw_match.cur_ed,
            raw_match.hash,
        )

    def make_match(
        self, *, old_st: int, old_ed: int, cur_st: int, cur_ed: int, hash: bytes
    ) -> object:
        return self._match_type(
            old_st=old_st,
            old_ed=old_ed,
            cur_st=cur_st,
            cur_ed=cur_ed,
            hash=hash,
        )

    def validate_kv_cache_payload(
        self,
        payload: list[object],
        *,
        expected_shape: tuple[int, int, int, int],
        expected_dtype_name: str,
    ) -> None:
        if len(payload) != 1 or not isinstance(payload[0], self._cuda_wrapper_type):
            raise LmcacheConfigurationError(
                "registration requires one LMCache CudaIPCWrapper"
            )
        wrapper = payload[0]
        if tuple(wrapper.shape) != expected_shape:
            raise LmcacheConfigurationError(
                "CUDA-IPC wrapper shape does not match the staging layout"
            )
        if str(wrapper.dtype) != expected_dtype_name:
            raise LmcacheConfigurationError(
                "CUDA-IPC wrapper dtype does not match the staging layout"
            )

    def wait(self, future: MessageFuture, timeout: float) -> object:
        return future.result(timeout=timeout)

    def wait_cuda(self, future: MessageFuture, timeout: float) -> object:
        to_cuda_future = getattr(future, "to_cuda_future", None)
        if not callable(to_cuda_future):
            raise LmcacheProtocolError(
                "LMCache transfer future lacks to_cuda_future()"
            )
        cuda_future = to_cuda_future()
        return cast(object, cuda_future.result(timeout=timeout))


class LmcacheBlendTransport:
    """Fail-closed synchronous client for the audited BlendEngineV2 protocol.

    Any timeout, malformed response, rejected transfer, or schema mismatch moves
    the transport to ``FAILED``.  Callers must then use ordinary full prefill or
    reject the request; failures are never converted into cache misses.
    """

    def __init__(
        self,
        config: LmcacheBlendTransportConfig,
        bindings: LmcacheBindings,
        message_queue: MessageQueue,
    ) -> None:
        self._config = config
        self._bindings = bindings
        self._message_queue = message_queue
        self._state = LmcacheTransportState.CREATED
        self._registration: LmcacheStagingRegistration | None = None
        if bindings.lmcache_version != LMCACHE_VERSION:
            raise LmcacheDependencyError(
                f"LMCache {LMCACHE_VERSION} is required; "
                f"got {bindings.lmcache_version!r}"
            )
        bindings.validate_protocol_schema()

    @property
    def state(self) -> LmcacheTransportState:
        """Return the current fail-closed lifecycle state."""

        return self._state

    @property
    def config(self) -> LmcacheBlendTransportConfig:
        """Return the immutable transport configuration."""

        return self._config

    def open(self) -> None:
        """Probe server reachability and the only remotely visible schema value."""

        self._require_state(LmcacheTransportState.CREATED)
        ping = self._request(LmcacheRequest.PING, [], cuda=False)
        if ping is not True:
            self._fail("PING returned an invalid response")
        chunk_size = self._request(LmcacheRequest.GET_CHUNK_SIZE, [], cuda=False)
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            self._fail("GET_CHUNK_SIZE returned a non-integer response")
        if chunk_size != self._config.chunk_size:
            self._fail("server chunk size does not match the pinned client")
        self._state = LmcacheTransportState.READY

    def register_staging_buffer(
        self, registration: LmcacheStagingRegistration
    ) -> None:
        """Register exactly one declared ``[2,L,T,D]`` CUDA-IPC tensor."""

        self._require_state(LmcacheTransportState.READY)
        payload = list(registration.kv_cache_payload)
        self._bindings.validate_kv_cache_payload(
            payload,
            expected_shape=registration.layout.shape,
            expected_dtype_name=registration.layout.dtype_name,
        )
        response = self._request(
            LmcacheRequest.REGISTER,
            [
                registration.instance_id,
                payload,
                self._config.storage_model_name,
                self._config.world_size,
            ],
            cuda=False,
        )
        if response is not None:
            self._fail("CB_REGISTER_KV_CACHE returned an invalid response")
        self._registration = registration
        self._state = LmcacheTransportState.REGISTERED

    def unregister_staging_buffer(self) -> None:
        """Synchronously unregister the active staging tensor."""

        self._require_state(LmcacheTransportState.REGISTERED)
        registration = self._registration
        if registration is None:
            self._fail("registered state has no staging registration")
        response = self._request(
            LmcacheRequest.UNREGISTER,
            [registration.instance_id],
            cuda=False,
        )
        if response is not None:
            self._fail("CB_UNREGISTER_KV_CACHE returned an invalid response")
        self._registration = None
        self._state = LmcacheTransportState.READY

    def lookup_candidates(
        self, token_ids: Sequence[int], *, request_id: str
    ) -> tuple[LmcacheCandidate, ...]:
        """Return storage-present rolling-hash candidates, never verified hits."""

        self._require_state(
            LmcacheTransportState.READY, LmcacheTransportState.REGISTERED
        )
        validate_request_id(request_id)
        tokens = normalize_token_ids(token_ids)
        if len(tokens) < self._config.chunk_size:
            return ()
        key = self._make_key(tokens, request_id=request_id, worker_id=None)
        raw_response = self._request(LmcacheRequest.LOOKUP, [key], cuda=False)
        if not isinstance(raw_response, list):
            self._fail("CB_LOOKUP_PRE_COMPUTED_V2 returned a non-list response")
        digest = query_digest(tokens)
        candidates: list[LmcacheCandidate] = []
        try:
            for raw_match in raw_response:
                old_st, old_ed, cur_st, cur_ed, storage_hash = (
                    self._bindings.parse_match(raw_match)
                )
                source_range = _candidate_range(old_st, old_ed, "source")
                target_range = _candidate_range(cur_st, cur_ed, "target")
                if len(source_range) != self._config.chunk_size:
                    raise LmcacheProtocolError(
                        "candidate source range is not one complete chunk"
                    )
                if target_range.end > len(tokens):
                    raise LmcacheProtocolError(
                        "candidate target range exceeds the query tokens"
                    )
                candidate = LmcacheCandidate(
                    source_relative_range=source_range,
                    target_range=target_range,
                    storage_hash=storage_hash,
                    storage_model_name=self._config.storage_model_name,
                    query_digest=digest,
                )
                candidates.append(candidate)
        except (AttributeError, TypeError, ValueError, LmcacheProtocolError) as exc:
            self._state = LmcacheTransportState.FAILED
            raise LmcacheProtocolError(
                "CB_LOOKUP_PRE_COMPUTED_V2 returned an invalid candidate"
            ) from exc
        return tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    candidate.target_range.start,
                    candidate.source_relative_range.start,
                    candidate.storage_hash,
                ),
            )
        )

    def store_precomputed(
        self,
        token_ids: Sequence[int],
        *,
        buffer_offset: int,
        event_ipc_handle: bytes,
        request_id: str,
    ) -> LmcacheStoreReceipt:
        """Store only complete chunks and register them for non-prefix lookup."""

        tokens = self._validate_store_inputs(
            token_ids,
            buffer_offset=buffer_offset,
            event_ipc_handle=event_ipc_handle,
            request_id=request_id,
        )
        key = self._make_key(
            tokens, request_id=request_id, worker_id=self._config.worker_id
        )
        registration = self._active_registration()
        response = self._request(
            LmcacheRequest.STORE_PRECOMPUTED,
            [key, buffer_offset, registration.instance_id, event_ipc_handle],
            cuda=True,
        )
        self._require_transfer_success(response, "CB_STORE_PRE_COMPUTED")
        return LmcacheStoreReceipt(
            stored_tokens=len(tokens),
            stored_chunks=len(tokens) // self._config.chunk_size,
            candidate_lookup_required=True,
        )

    def store_final(
        self,
        token_ids: Sequence[int],
        *,
        buffer_offset: int,
        event_ipc_handle: bytes,
        request_id: str,
    ) -> LmcacheStoreReceipt:
        """Store complete final chunks for ordinary prefix lookup.

        The pinned server does not add ``CB_STORE_FINAL`` data to its non-prefix
        matcher, so ``candidate_lookup_required`` is false and this method must
        not be presented as creating a CacheBlend document entry.
        """

        tokens = self._validate_store_inputs(
            token_ids,
            buffer_offset=buffer_offset,
            event_ipc_handle=event_ipc_handle,
            request_id=request_id,
        )
        key = self._make_key(
            tokens, request_id=request_id, worker_id=self._config.worker_id
        )
        registration = self._active_registration()
        response = self._request(
            LmcacheRequest.STORE_FINAL,
            [key, buffer_offset, registration.instance_id, event_ipc_handle],
            cuda=True,
        )
        self._require_transfer_success(response, "CB_STORE_FINAL")
        return LmcacheStoreReceipt(
            stored_tokens=len(tokens),
            stored_chunks=len(tokens) // self._config.chunk_size,
            candidate_lookup_required=False,
        )

    def retrieve_precomputed(
        self,
        token_ids: Sequence[int],
        verified_candidates: Sequence[VerifiedLmcacheCandidate],
        *,
        buffer_offset: int,
        event_ipc_handle: bytes,
        request_id: str,
    ) -> LmcacheRetrieveReceipt:
        """Synchronously copy independently verified chunks into staging."""

        registration = self._active_registration()
        validate_request_id(request_id)
        validate_event_handle(event_ipc_handle)
        tokens = normalize_token_ids(token_ids)
        if not tokens:
            raise LmcacheConfigurationError("retrieve query tokens must not be empty")
        if not verified_candidates:
            return LmcacheRetrieveReceipt(retrieved_tokens=0, retrieved_chunks=0)
        digest = query_digest(tokens)
        raw_matches: list[object] = []
        ranges: list[TokenRange] = []
        for verified in verified_candidates:
            if not isinstance(verified, VerifiedLmcacheCandidate):
                raise LmcacheConfigurationError(
                    "retrieve requires independently verified LMCache candidates"
                )
            candidate = verified.candidate
            if candidate.storage_model_name != self._config.storage_model_name:
                raise LmcacheProtocolError("candidate storage namespace does not match")
            if candidate.query_digest != digest:
                raise LmcacheProtocolError(
                    "candidate was produced for a different lookup query"
                )
            target = candidate.target_range
            if target.end > len(tokens):
                raise LmcacheProtocolError("verified candidate exceeds query tokens")
            query_slice = tuple(tokens[target.start : target.end])
            if query_slice != verified.match.record.token_ids:
                raise LmcacheProtocolError(
                    "query tokens changed after exact candidate verification"
                )
            if len(target) != self._config.chunk_size:
                raise LmcacheProtocolError(
                    "verified candidate is not one complete LMCache chunk"
                )
            if any(target.overlaps(existing) for existing in ranges):
                raise LmcacheProtocolError("verified candidate target ranges overlap")
            ranges.append(target)
            source = candidate.source_relative_range
            raw_matches.append(
                self._bindings.make_match(
                    old_st=source.start,
                    old_ed=source.end,
                    cur_st=target.start,
                    cur_ed=target.end,
                    hash=candidate.storage_hash,
                )
            )
        max_target_end = max(token_range.end for token_range in ranges)
        validate_buffer_range(
            start=buffer_offset,
            length=max_target_end,
            capacity=registration.layout.token_capacity,
            field_name="retrieve buffer_offset",
        )
        key = self._make_key(
            tokens, request_id=request_id, worker_id=self._config.worker_id
        )
        response = self._request(
            LmcacheRequest.RETRIEVE,
            [
                key,
                raw_matches,
                buffer_offset,
                registration.instance_id,
                event_ipc_handle,
            ],
            cuda=True,
        )
        self._require_transfer_success(response, "CB_RETRIEVE_PRE_COMPUTED_V2")
        return LmcacheRetrieveReceipt(
            retrieved_tokens=sum(len(token_range) for token_range in ranges),
            retrieved_chunks=len(ranges),
        )

    def close(self) -> None:
        """Always close the MQ client, even when unregistering fails.

        The method is idempotent.  If a staging buffer is still registered, one
        synchronous unregister attempt is made first.  Any unregister or MQ
        close failure is reported after MQ cleanup; it is never suppressed.
        """

        if self._state is LmcacheTransportState.CLOSED:
            return
        failures: list[str] = []
        if self._registration is not None:
            try:
                request_type = self._bindings.request_type(LmcacheRequest.UNREGISTER)
                response_cls = self._bindings.response_class(LmcacheRequest.UNREGISTER)
                future = self._message_queue.submit_request(
                    request_type,
                    [self._registration.instance_id],
                    response_cls,
                )
                response = self._bindings.wait(
                    future, float(self._config.request_timeout_seconds)
                )
                if response is not None:
                    failures.append("unregister returned an invalid response")
            except Exception:  # cleanup must continue through transport failures
                failures.append("unregister failed")
        try:
            self._message_queue.close()
        except Exception:
            failures.append("message queue close failed")
        finally:
            self._registration = None
            self._state = LmcacheTransportState.CLOSED
        if failures:
            raise LmcacheCloseError("; ".join(failures))

    def _validate_store_inputs(
        self,
        token_ids: Sequence[int],
        *,
        buffer_offset: int,
        event_ipc_handle: bytes,
        request_id: str,
    ) -> tuple[int, ...]:
        registration = self._active_registration()
        validate_request_id(request_id)
        validate_event_handle(event_ipc_handle)
        tokens = normalize_token_ids(token_ids)
        if not tokens:
            raise LmcacheConfigurationError("store token_ids must not be empty")
        if len(tokens) % self._config.chunk_size != 0:
            raise LmcacheConfigurationError(
                "Blend V2 stores only complete chunks; partial stores are rejected"
            )
        validate_buffer_range(
            start=buffer_offset,
            length=len(tokens),
            capacity=registration.layout.token_capacity,
            field_name="store buffer_offset",
        )
        return tokens

    def _make_key(
        self,
        tokens: tuple[int, ...],
        *,
        request_id: str,
        worker_id: int | None,
    ) -> object:
        return self._bindings.make_key(
            model_name=self._config.storage_model_name,
            world_size=self._config.world_size,
            worker_id=worker_id,
            token_ids=tokens,
            start=0,
            end=len(tokens),
            request_id=request_id,
        )

    def _active_registration(self) -> LmcacheStagingRegistration:
        self._require_state(LmcacheTransportState.REGISTERED)
        if self._registration is None:
            self._fail("registered state has no staging registration")
        return self._registration

    def _request(
        self, request: LmcacheRequest, payloads: list[object], *, cuda: bool
    ) -> object:
        try:
            request_type = self._bindings.request_type(request)
            response_cls = self._bindings.response_class(request)
            future = self._message_queue.submit_request(
                request_type, payloads, response_cls
            )
            timeout = float(self._config.request_timeout_seconds)
            if cuda:
                return self._bindings.wait_cuda(future, timeout)
            return self._bindings.wait(future, timeout)
        except LmcacheOperationError:
            raise
        except Exception as exc:
            self._state = LmcacheTransportState.FAILED
            raise LmcacheOperationError(
                f"LMCache {request.value} request failed"
            ) from exc

    def _require_transfer_success(self, response: object, operation: str) -> None:
        # ``wait_cuda`` unwraps the raw ``tuple[event_handle, bool]`` response and
        # returns only its boolean payload, matching LMCache CUDAMessagingFuture.
        if response is not True:
            self._state = LmcacheTransportState.FAILED
            raise LmcacheOperationError(f"LMCache {operation} reported failure")

    def _require_state(self, *allowed: LmcacheTransportState) -> None:
        if self._state not in allowed:
            names = ", ".join(state.value for state in allowed)
            raise LmcacheLifecycleError(
                f"operation requires state {names}; "
                f"current state is {self._state.value}"
            )

    def _fail(self, message: str) -> NoReturn:
        self._state = LmcacheTransportState.FAILED
        raise LmcacheOperationError(message)


def load_lmcache_v0_4_3_bindings() -> LmcacheBindings:
    """Lazily import and validate the exact public LMCache 0.4.3 protocol."""

    try:
        installed_version = version("lmcache")
    except PackageNotFoundError as exc:
        raise LmcacheDependencyError(
            "LMCache 0.4.3 is not installed; install the pinned GPU runtime extras"
        ) from exc
    if installed_version != LMCACHE_VERSION:
        raise LmcacheDependencyError(
            f"LMCache {LMCACHE_VERSION} is required; got {installed_version!r}"
        )
    try:
        custom_types = import_module("lmcache.v1.multiprocess.custom_types")
        protocol = import_module("lmcache.v1.multiprocess.protocol")
    except ImportError as exc:
        raise LmcacheDependencyError(
            "LMCache 0.4.3 Blend V2 modules could not be imported; "
            "verify the pinned wheel and CUDA runtime"
        ) from exc
    bindings = _RuntimeLmcacheBindings(
        _version=installed_version,
        _request_type_cls=protocol.RequestType,
        _key_type=custom_types.IPCCacheEngineKey,
        _match_type=custom_types.CBMatchResult,
        _cuda_wrapper_type=custom_types.CudaIPCWrapper,
        _get_payload_classes=protocol.get_payload_classes,
        _get_response_class=protocol.get_response_class,
    )
    bindings.validate_protocol_schema()
    return bindings


def create_lmcache_blend_transport(
    server_url: str,
    config: LmcacheBlendTransportConfig,
    *,
    zmq_context: object | None = None,
) -> LmcacheBlendTransport:
    """Lazily construct the production LMCache MQ client.

    Construction validates the local package/schema.  The caller must still
    invoke :meth:`LmcacheBlendTransport.open` before any cache operation.
    """

    _validate_server_url(server_url)
    bindings = load_lmcache_v0_4_3_bindings()
    try:
        mq_module = import_module("lmcache.v1.multiprocess.mq")
        zmq = import_module("zmq")
        context = zmq_context if zmq_context is not None else zmq.Context.instance()
        message_queue = mq_module.MessageQueueClient(
            server_url=server_url, context=context
        )
    except ImportError as exc:
        raise LmcacheDependencyError(
            "LMCache 0.4.3 message-queue dependencies could not be imported"
        ) from exc
    except Exception as exc:
        raise LmcacheOperationError(
            "LMCache message-queue client creation failed"
        ) from exc
    return LmcacheBlendTransport(config, bindings, cast(MessageQueue, message_queue))


def _candidate_range(start: object, end: object, name: str) -> TokenRange:
    if isinstance(start, bool) or not isinstance(start, int):
        raise LmcacheProtocolError(f"candidate {name} start is not an integer")
    if isinstance(end, bool) or not isinstance(end, int):
        raise LmcacheProtocolError(f"candidate {name} end is not an integer")
    try:
        token_range = TokenRange(start, end)
    except (TypeError, ValueError) as exc:
        raise LmcacheProtocolError(f"candidate {name} range is invalid") from exc
    if len(token_range) == 0:
        raise LmcacheProtocolError(f"candidate {name} range is empty")
    return token_range


def _validate_server_url(server_url: str) -> None:
    if not isinstance(server_url, str) or not server_url.startswith("tcp://"):
        raise LmcacheConfigurationError("LMCache server_url must use tcp://")
    if len(server_url.encode("utf-8")) > 512 or any(
        character.isspace() for character in server_url
    ):
        raise LmcacheConfigurationError("LMCache server_url is invalid")
