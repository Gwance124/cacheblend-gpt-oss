"""Position-independent, strongly namespaced segment fingerprints."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterable
from typing import Protocol

from cacheblend_gpt_oss.planner.models import (
    CacheNamespace,
    SegmentFingerprint,
    normalize_token_ids,
)

_FINGERPRINT_DOMAIN = b"cacheblend-gpt-oss\x00segment-fingerprint\x00v1\x00"
_TOKEN_ENCODING_DOMAIN = b"cacheblend-gpt-oss\x00token-sequence\x00u64be-v1\x00"


def _length_prefix(value: bytes) -> bytes:
    return struct.pack(">Q", len(value)) + value


def canonical_token_bytes(token_ids: Iterable[int]) -> bytes:
    """Encode a token tuple without decimal/string concatenation ambiguity."""

    normalized = normalize_token_ids(token_ids)
    output = bytearray(_TOKEN_ENCODING_DOMAIN)
    output.extend(struct.pack(">Q", len(normalized)))
    for token_id in normalized:
        output.extend(struct.pack(">Q", token_id))
    return bytes(output)


class SegmentFingerprinter(Protocol):
    """Dependency-injection boundary for strong segment identity."""

    def fingerprint(
        self, namespace: CacheNamespace, token_ids: Iterable[int]
    ) -> SegmentFingerprint:
        """Return the namespace-bound fingerprint for ``token_ids``."""


class Sha256SegmentFingerprinter:
    """The production segment fingerprinter."""

    def fingerprint(
        self, namespace: CacheNamespace, token_ids: Iterable[int]
    ) -> SegmentFingerprint:
        digest = hashlib.sha256()
        digest.update(_FINGERPRINT_DOMAIN)
        for field_name, field_value in namespace.canonical_fields():
            digest.update(_length_prefix(field_name.encode("utf-8")))
            digest.update(_length_prefix(field_value.encode("utf-8")))
        digest.update(_length_prefix(canonical_token_bytes(token_ids)))
        return SegmentFingerprint(digest.digest())


SHA256_FINGERPRINTER = Sha256SegmentFingerprinter()


def fingerprint_segment(
    namespace: CacheNamespace, token_ids: Iterable[int]
) -> SegmentFingerprint:
    """Fingerprint tokens using the required SHA-256 implementation."""

    return SHA256_FINGERPRINTER.fingerprint(namespace, token_ids)
