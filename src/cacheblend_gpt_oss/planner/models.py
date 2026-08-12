"""Immutable value objects used by the cache-match planner."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

MAX_TOKEN_ID = (1 << 64) - 1


def normalize_token_ids(token_ids: Iterable[int]) -> tuple[int, ...]:
    """Return validated token IDs in their canonical in-memory representation."""

    normalized = tuple(token_ids)
    for token_id in normalized:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("token IDs must be integers")
        if not 0 <= token_id <= MAX_TOKEN_ID:
            raise ValueError(f"token ID must be in [0, {MAX_TOKEN_ID}]")
    return normalized


@dataclass(frozen=True, slots=True, order=True)
class TokenRange:
    """A half-open range in the token sequence of one request."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if isinstance(self.start, bool) or isinstance(self.end, bool):
            raise TypeError("token range bounds must be integers")
        if not isinstance(self.start, int) or not isinstance(self.end, int):
            raise TypeError("token range bounds must be integers")
        if self.start < 0:
            raise ValueError("token range start must be non-negative")
        if self.end < self.start:
            raise ValueError("token range end must not precede its start")

    def __len__(self) -> int:
        return self.end - self.start

    def overlaps(self, other: TokenRange) -> bool:
        """Return whether two non-empty half-open ranges overlap."""

        return self.start < other.end and other.start < self.end


@dataclass(frozen=True, slots=True)
class TokenSegment:
    """An exact token tuple and its position in a source or target prompt."""

    token_range: TokenRange
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        normalized = normalize_token_ids(self.token_ids)
        object.__setattr__(self, "token_ids", normalized)
        if not normalized:
            raise ValueError("a token segment must not be empty")
        if len(self.token_range) != len(normalized):
            raise ValueError("token range length must equal the token count")

    @classmethod
    def at(cls, start: int, token_ids: Iterable[int]) -> TokenSegment:
        """Build a segment beginning at ``start``."""

        normalized = normalize_token_ids(token_ids)
        return cls(TokenRange(start, start + len(normalized)), normalized)

    def __len__(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True, slots=True)
class CacheNamespace:
    """All compatibility inputs that separate mutually reusable cache entries.

    The two configuration digests are produced by the version-scoped model and
    KV-cache adapters. They cover the complete normalized model configuration
    and hybrid cache-group layout, respectively.
    """

    schema_version: int
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    model_config_digest: str
    kv_cache_config_digest: str
    adapter_revision: str
    vllm_version: str
    lmcache_version: str
    torch_version: str
    cuda_runtime: str

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(
            self.schema_version, int
        ):
            raise TypeError("schema version must be an integer")
        if self.schema_version < 1:
            raise ValueError("schema version must be positive")
        for name, value in self.canonical_fields():
            if name == "schema_version":
                continue
            if not value:
                raise ValueError(f"{name} must not be empty")

    def canonical_fields(self) -> tuple[tuple[str, str], ...]:
        """Return a stable, explicitly named representation for hashing."""

        return (
            ("schema_version", str(self.schema_version)),
            ("model_id", self.model_id),
            ("model_revision", self.model_revision),
            ("tokenizer_id", self.tokenizer_id),
            ("tokenizer_revision", self.tokenizer_revision),
            ("model_config_digest", self.model_config_digest),
            ("kv_cache_config_digest", self.kv_cache_config_digest),
            ("adapter_revision", self.adapter_revision),
            ("vllm_version", self.vllm_version),
            ("lmcache_version", self.lmcache_version),
            ("torch_version", self.torch_version),
            ("cuda_runtime", self.cuda_runtime),
        )


@dataclass(frozen=True, slots=True)
class SegmentFingerprint:
    """A complete SHA-256 segment identity."""

    digest: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.digest, bytes):
            raise TypeError("fingerprint digest must be bytes")
        if len(self.digest) != 32:
            raise ValueError("a SHA-256 fingerprint must contain 32 bytes")

    @property
    def hex_digest(self) -> str:
        return self.digest.hex()


@dataclass(frozen=True, slots=True)
class CacheRecord:
    """Persisted reusable-token metadata; KV payloads remain behind ``cache_key``."""

    namespace: CacheNamespace
    fingerprint: SegmentFingerprint
    token_ids: tuple[int, ...]
    source_range: TokenRange
    cache_key: str

    def __post_init__(self) -> None:
        normalized = normalize_token_ids(self.token_ids)
        object.__setattr__(self, "token_ids", normalized)
        if not normalized:
            raise ValueError("a cache record must not be empty")
        if len(self.source_range) != len(normalized):
            raise ValueError("source range length must equal the token count")
        if not self.cache_key:
            raise ValueError("cache key must not be empty")


@dataclass(frozen=True, slots=True)
class CandidateMatch:
    """A lookup candidate that has not yet passed all verification gates."""

    target_segment: TokenSegment
    query_fingerprint: SegmentFingerprint
    record: CacheRecord

    @property
    def position_delta(self) -> int:
        """The RoPE correction shift from cached to requested position."""

        return self.target_segment.token_range.start - self.record.source_range.start
