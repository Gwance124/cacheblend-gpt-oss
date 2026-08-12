"""Pure value objects for the pinned LMCache Blend V2 transport.

These types intentionally do not import LMCache, Torch, CUDA, or ZeroMQ.  The
wire adapter lives in :mod:`cacheblend_gpt_oss.storage.lmcache_v0_4_3` and only
loads those runtime dependencies when explicitly requested.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from enum import Enum

from cacheblend_gpt_oss.planner.fingerprint import SHA256_FINGERPRINTER
from cacheblend_gpt_oss.planner.matching import VerifiedMatch
from cacheblend_gpt_oss.planner.models import (
    CacheNamespace,
    TokenRange,
    normalize_token_ids,
)
from cacheblend_gpt_oss.targets import PINNED_TARGET

LMCACHE_VERSION = "0.4.3"
LMCACHE_SOURCE_COMMIT = "7f326118a2f1afc7801988dd02e3055bdf21ef6b"
LMCACHE_BLEND_PROTOCOL = "multiprocess-blend-v2"
LMCACHE_HASH_ALGORITHM = "blake3"
LMCACHE_HASH_BYTES = 32
LMCACHE_CHUNK_SIZE = 256
LMCACHE_CACHE_KEY_PREFIX = "lmcache:0.4.3:blake3:"

_STORAGE_NAMESPACE_DOMAIN = b"cacheblend-gpt-oss\x00lmcache-storage-namespace\x00v1\x00"
_QUERY_DOMAIN = b"cacheblend-gpt-oss\x00lmcache-query\x00v1\x00"
_MAX_NAMESPACE_FIELD_BYTES = 1024
_MAX_REQUEST_ID_BYTES = 256
_MAX_EVENT_HANDLE_BYTES = 16 * 1024


class LmcacheTransportError(RuntimeError):
    """Base class for fail-closed LMCache transport errors."""


class LmcacheConfigurationError(LmcacheTransportError):
    """The requested transport configuration is outside the audited envelope."""


class LmcacheDependencyError(LmcacheTransportError):
    """Pinned runtime dependencies could not be loaded or validated."""


class LmcacheProtocolError(LmcacheTransportError):
    """The local or remote protocol did not match the pinned schema."""


class LmcacheLifecycleError(LmcacheTransportError):
    """An operation was attempted in an invalid transport state."""


class LmcacheOperationError(LmcacheTransportError):
    """A message-queue operation failed or returned an invalid response."""


class LmcacheCloseError(LmcacheTransportError):
    """Closing the client failed after all cleanup attempts were made."""


class LmcacheTransportState(str, Enum):
    """Observable lifecycle states for the transport."""

    CREATED = "created"
    READY = "ready"
    REGISTERED = "registered"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class LmcacheServerAttestation:
    """Operator-supplied identity for a separately launched cache server.

    LMCache 0.4.3 has ``PING`` and ``GET_CHUNK_SIZE`` RPCs, but it has no RPC
    that reports the server package version, source commit, protocol revision,
    or hash algorithm.  Requiring this explicit attestation prevents the client
    from silently assuming that a reachable server is the audited server.
    """

    lmcache_version: str
    source_commit: str
    protocol: str
    hash_algorithm: str

    def validate(self) -> None:
        expected = (
            ("lmcache_version", LMCACHE_VERSION, self.lmcache_version),
            ("source_commit", LMCACHE_SOURCE_COMMIT, self.source_commit),
            ("protocol", LMCACHE_BLEND_PROTOCOL, self.protocol),
            ("hash_algorithm", LMCACHE_HASH_ALGORITHM, self.hash_algorithm),
        )
        for name, expected_value, observed_value in expected:
            if observed_value != expected_value:
                raise LmcacheConfigurationError(
                    f"LMCache server {name} must be {expected_value!r}; "
                    f"got {observed_value!r}"
                )


@dataclass(frozen=True, slots=True)
class LmcacheBlendTransportConfig:
    """Pinned client identity and timeout policy.

    ``storage_model_name`` hashes the complete project cache namespace into the
    only model-identity field exposed by ``IPCCacheEngineKey``.  Consequently,
    caches from different model/tokenizer revisions or hybrid-layout digests do
    not share an LMCache ``ObjectKey`` namespace.
    """

    namespace: CacheNamespace
    server_attestation: LmcacheServerAttestation
    world_size: int = 1
    worker_id: int = 0
    chunk_size: int = LMCACHE_CHUNK_SIZE
    request_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        self.server_attestation.validate()
        _validate_namespace(self.namespace)
        _require_plain_int("world_size", self.world_size, minimum=1)
        _require_plain_int("worker_id", self.worker_id, minimum=0)
        _require_plain_int("chunk_size", self.chunk_size, minimum=1)
        if self.world_size != 1 or self.worker_id != 0:
            raise LmcacheConfigurationError(
                "the audited GPT-OSS transport supports world_size=1 and worker_id=0"
            )
        if self.chunk_size != LMCACHE_CHUNK_SIZE:
            raise LmcacheConfigurationError(
                f"the audited Blend V2 chunk size is {LMCACHE_CHUNK_SIZE}"
            )
        if isinstance(self.request_timeout_seconds, bool) or not isinstance(
            self.request_timeout_seconds, int | float
        ):
            raise LmcacheConfigurationError(
                "request_timeout_seconds must be a finite positive number"
            )
        timeout = float(self.request_timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0:
            raise LmcacheConfigurationError(
                "request_timeout_seconds must be a finite positive number"
            )

    @property
    def storage_model_name(self) -> str:
        """Return the strongly namespaced value sent as LMCache ``model_name``."""

        digest = hashlib.sha256()
        digest.update(_STORAGE_NAMESPACE_DOMAIN)
        for field_name, field_value in self.namespace.canonical_fields():
            _update_length_prefixed(digest, field_name.encode("utf-8"))
            _update_length_prefixed(digest, field_value.encode("utf-8"))
        _update_length_prefixed(digest, str(self.world_size).encode("ascii"))
        _update_length_prefixed(digest, str(self.chunk_size).encode("ascii"))
        _update_length_prefixed(
            digest, self.server_attestation.protocol.encode("ascii")
        )
        _update_length_prefixed(
            digest, self.server_attestation.hash_algorithm.encode("ascii")
        )
        return f"{PINNED_TARGET.model_id}#cacheblend-v1-{digest.hexdigest()}"


@dataclass(frozen=True, slots=True)
class LmcacheStagingLayout:
    """Declared shape of the one contiguous CUDA-IPC staging tensor.

    The server's ``PlainGPUCacheContext`` requires one rank-four ``[2,L,T,D]``
    tensor.  Model-specific validation of ``L`` and ``D`` stays in the GPT-OSS
    adapter; this storage boundary validates the declared shape and payload agree.
    """

    layer_count: int
    token_capacity: int
    kv_width: int
    dtype_name: str

    def __post_init__(self) -> None:
        _require_plain_int("layer_count", self.layer_count, minimum=1)
        _require_plain_int("token_capacity", self.token_capacity, minimum=1)
        _require_plain_int("kv_width", self.kv_width, minimum=1)
        if not self.dtype_name or len(self.dtype_name.encode("utf-8")) > 128:
            raise LmcacheConfigurationError("dtype_name must be a bounded string")
        if self.token_capacity < LMCACHE_CHUNK_SIZE:
            raise LmcacheConfigurationError(
                "the staging tensor must hold at least one complete LMCache chunk"
            )

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Return the required ``[2, layers, tokens, width]`` shape."""

        return (2, self.layer_count, self.token_capacity, self.kv_width)


@dataclass(frozen=True, slots=True)
class LmcacheStagingRegistration:
    """Opaque CUDA-IPC payload plus its independently validated layout."""

    instance_id: int
    kv_cache_payload: tuple[object, ...]
    layout: LmcacheStagingLayout

    def __post_init__(self) -> None:
        _require_plain_int("instance_id", self.instance_id, minimum=0)
        if len(self.kv_cache_payload) != 1:
            raise LmcacheConfigurationError(
                "Blend V2 requires exactly one CUDA-IPC staging tensor"
            )


@dataclass(frozen=True, slots=True)
class LmcacheCandidate:
    """Untrusted candidate returned by LMCache's rolling-hash matcher.

    A candidate is *not* a cache hit.  It does not contain the stored token
    sequence or this project's full compatibility record, so it cannot be sent
    to retrieval until :meth:`VerifiedLmcacheCandidate.bind` succeeds.
    """

    source_relative_range: TokenRange
    target_range: TokenRange
    storage_hash: bytes
    storage_model_name: str
    query_digest: bytes

    def __post_init__(self) -> None:
        if len(self.source_relative_range) == 0 or len(self.target_range) == 0:
            raise LmcacheProtocolError("LMCache candidate ranges must not be empty")
        if len(self.source_relative_range) != len(self.target_range):
            raise LmcacheProtocolError("LMCache candidate range lengths differ")
        if not isinstance(self.storage_hash, bytes) or len(self.storage_hash) != 32:
            raise LmcacheProtocolError(
                "LMCache candidate hash must be a 32-byte BLAKE3 digest"
            )
        if not self.storage_model_name:
            raise LmcacheProtocolError("LMCache candidate model namespace is empty")
        if not isinstance(self.query_digest, bytes) or len(self.query_digest) != 32:
            raise LmcacheProtocolError("LMCache candidate query digest is invalid")

    @property
    def cache_key(self) -> str:
        """Return the sidecar-record key bound to this LMCache storage hash."""

        return LMCACHE_CACHE_KEY_PREFIX + self.storage_hash.hex()


@dataclass(frozen=True, slots=True, init=False)
class VerifiedLmcacheCandidate:
    """A raw LMCache candidate bound to an exact-token verified sidecar record."""

    candidate: LmcacheCandidate
    match: VerifiedMatch

    @classmethod
    def bind(
        cls,
        candidate: LmcacheCandidate,
        match: VerifiedMatch,
        *,
        expected_namespace: CacheNamespace,
    ) -> VerifiedLmcacheCandidate:
        """Recheck every planner invariant and bind the storage hash to the record."""

        record = match.record
        target = match.target_segment
        if record.namespace != expected_namespace:
            raise LmcacheProtocolError("verified record namespace does not match")
        if candidate.target_range != target.token_range:
            raise LmcacheProtocolError("candidate and verified target ranges differ")
        if candidate.cache_key != record.cache_key:
            raise LmcacheProtocolError(
                "candidate storage hash is not bound to the verified sidecar record"
            )
        if record.token_ids != target.token_ids:
            raise LmcacheProtocolError("verified record token sequence does not match")
        expected_fingerprint = SHA256_FINGERPRINTER.fingerprint(
            expected_namespace, target.token_ids
        )
        if (
            match.candidate.query_fingerprint != expected_fingerprint
            or record.fingerprint != expected_fingerprint
        ):
            raise LmcacheProtocolError("verified record fingerprint does not match")
        if len(candidate.target_range) != len(target):
            raise LmcacheProtocolError("candidate and verified token counts differ")
        instance = object.__new__(cls)
        object.__setattr__(instance, "candidate", candidate)
        object.__setattr__(instance, "match", match)
        return instance


@dataclass(frozen=True, slots=True)
class LmcacheStoreReceipt:
    """Synchronous acknowledgement of complete precomputed chunks."""

    stored_tokens: int
    stored_chunks: int
    candidate_lookup_required: bool


@dataclass(frozen=True, slots=True)
class LmcacheRetrieveReceipt:
    """Synchronous acknowledgement of candidate KV copied into staging."""

    retrieved_tokens: int
    retrieved_chunks: int


def query_digest(token_ids: tuple[int, ...]) -> bytes:
    """Bind a candidate to the exact full query used for LMCache lookup."""

    normalized = normalize_token_ids(token_ids)
    digest = hashlib.sha256()
    digest.update(_QUERY_DOMAIN)
    digest.update(struct.pack(">Q", len(normalized)))
    for token_id in normalized:
        digest.update(struct.pack(">Q", token_id))
    return digest.digest()


def validate_request_id(request_id: str) -> None:
    """Validate an opaque protocol correlation ID without logging its value."""

    if not isinstance(request_id, str) or not request_id:
        raise LmcacheConfigurationError("request_id must be a non-empty string")
    if len(request_id.encode("utf-8")) > _MAX_REQUEST_ID_BYTES:
        raise LmcacheConfigurationError("request_id exceeds the bounded wire limit")


def validate_event_handle(event_ipc_handle: bytes) -> None:
    """Validate an opaque CUDA event IPC handle without decoding it locally."""

    if not isinstance(event_ipc_handle, bytes) or not event_ipc_handle:
        raise LmcacheConfigurationError("event_ipc_handle must be non-empty bytes")
    if len(event_ipc_handle) > _MAX_EVENT_HANDLE_BYTES:
        raise LmcacheConfigurationError("event_ipc_handle exceeds the wire limit")


def validate_buffer_range(
    *, start: int, length: int, capacity: int, field_name: str
) -> None:
    """Validate a half-open staging-buffer token range."""

    _require_plain_int(field_name, start, minimum=0)
    _require_plain_int("token length", length, minimum=0)
    if start + length > capacity:
        raise LmcacheConfigurationError(
            f"{field_name} range exceeds the registered staging capacity"
        )


def _validate_namespace(namespace: CacheNamespace) -> None:
    if namespace.schema_version != 1:
        raise LmcacheConfigurationError("cache namespace schema_version must be 1")
    expected = (
        ("model_id", PINNED_TARGET.model_id, namespace.model_id),
        ("tokenizer_id", PINNED_TARGET.model_id, namespace.tokenizer_id),
        ("vllm_version", PINNED_TARGET.vllm_version, namespace.vllm_version),
        ("lmcache_version", PINNED_TARGET.lmcache_version, namespace.lmcache_version),
        ("torch_version", PINNED_TARGET.torch_version, namespace.torch_version),
        ("cuda_runtime", PINNED_TARGET.cuda_runtime, namespace.cuda_runtime),
    )
    for name, expected_value, observed_value in expected:
        if observed_value != expected_value:
            raise LmcacheConfigurationError(
                f"cache namespace {name} must be {expected_value!r}; "
                f"got {observed_value!r}"
            )
    for field_name, field_value in namespace.canonical_fields():
        if len(field_value.encode("utf-8")) > _MAX_NAMESPACE_FIELD_BYTES:
            raise LmcacheConfigurationError(
                f"cache namespace {field_name} exceeds the bounded wire limit"
            )


def _require_plain_int(name: str, value: object, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LmcacheConfigurationError(f"{name} must be an integer")
    if value < minimum:
        raise LmcacheConfigurationError(f"{name} must be at least {minimum}")


def _update_length_prefixed(digest: object, value: bytes) -> None:
    # hashlib's common protocol is intentionally kept private to avoid importing
    # private typing aliases that differ across supported Python minor versions.
    digest.update(struct.pack(">Q", len(value)))  # type: ignore[attr-defined]
    digest.update(value)  # type: ignore[attr-defined]
