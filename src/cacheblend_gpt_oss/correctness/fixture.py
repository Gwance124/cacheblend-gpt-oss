# SPDX-License-Identifier: Apache-2.0
"""Deterministic raw-token fixture for the first moved-document GPU gate."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from cacheblend_gpt_oss.correctness.models import (
    CorrectnessCase,
    PromptCaseIdentity,
    ReusableSegmentIdentity,
)

_DOCUMENT_A_TOKENS = tuple(range(1_024, 1_280))
_DOCUMENT_B_TOKENS = tuple(range(1_280, 1_536))
_MISS_DOCUMENT_TOKENS = tuple(range(4_096, 4_352))
_TARGET_PREFIX_TOKENS = tuple(range(2_048, 2_065))
_TARGET_SUFFIX_TOKENS = tuple(range(3_072, 3_079))


def digest_token_ids(token_ids: tuple[int, ...]) -> str:
    """Hash exact u32 token IDs without serializing them into artifacts."""

    digest = sha256(b"cacheblend-gpt-oss-correctness-tokens-v1\0")
    digest.update(len(token_ids).to_bytes(8, "big"))
    for token_id in token_ids:
        digest.update(token_id.to_bytes(4, "big"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CorrectnessFixture:
    """One deterministic source/target pair and its identifier-free identity."""

    source_prompt_token_ids: tuple[int, ...]
    target_prompt_token_ids: tuple[int, ...]
    prompt_identity: PromptCaseIdentity


MovedDocumentFixture = CorrectnessFixture


def _identity(
    case: CorrectnessCase,
    source: tuple[int, ...],
    target: tuple[int, ...],
    mappings: tuple[tuple[tuple[int, ...], int, int], ...],
) -> PromptCaseIdentity:
    return PromptCaseIdentity(
        case=case,
        source_prompt_digest=digest_token_ids(source),
        source_prompt_tokens=len(source),
        target_prompt_digest=digest_token_ids(target),
        target_prompt_tokens=len(target),
        reusable_segments=tuple(
            ReusableSegmentIdentity(
                token_digest=digest_token_ids(tokens),
                tokens=len(tokens),
                source_start=source_start,
                target_start=target_start,
            )
            for tokens, source_start, target_start in mappings
        ),
    )


def build_exact_prefix_fixture() -> CorrectnessFixture:
    source = _DOCUMENT_A_TOKENS
    target = (*_DOCUMENT_A_TOKENS, *_TARGET_SUFFIX_TOKENS)
    return CorrectnessFixture(
        source,
        target,
        _identity(
            CorrectnessCase.EXACT_PREFIX,
            source,
            target,
            ((_DOCUMENT_A_TOKENS, 0, 0),),
        ),
    )


def build_moved_document_fixture() -> CorrectnessFixture:
    """Return the immutable raw-token fixture used by all M3 captures."""

    source = _DOCUMENT_A_TOKENS
    target = (*_TARGET_PREFIX_TOKENS, *_DOCUMENT_A_TOKENS, *_TARGET_SUFFIX_TOKENS)
    return CorrectnessFixture(
        source,
        target,
        _identity(
            CorrectnessCase.MOVED_DOCUMENT,
            source,
            target,
            ((_DOCUMENT_A_TOKENS, 0, len(_TARGET_PREFIX_TOKENS)),),
        ),
    )


def build_reordered_documents_fixture() -> CorrectnessFixture:
    source = (*_DOCUMENT_A_TOKENS, *_DOCUMENT_B_TOKENS)
    target = (
        *_TARGET_PREFIX_TOKENS,
        *_DOCUMENT_B_TOKENS,
        *_DOCUMENT_A_TOKENS,
        *_TARGET_SUFFIX_TOKENS,
    )
    prefix = len(_TARGET_PREFIX_TOKENS)
    return CorrectnessFixture(
        source,
        target,
        _identity(
            CorrectnessCase.REORDERED_DOCUMENTS,
            source,
            target,
            (
                (_DOCUMENT_A_TOKENS, 0, prefix + len(_DOCUMENT_B_TOKENS)),
                (_DOCUMENT_B_TOKENS, len(_DOCUMENT_A_TOKENS), prefix),
            ),
        ),
    )


def build_cache_miss_fixture() -> CorrectnessFixture:
    source = _DOCUMENT_A_TOKENS
    target = (
        *_TARGET_PREFIX_TOKENS,
        *_MISS_DOCUMENT_TOKENS,
        *_TARGET_SUFFIX_TOKENS,
    )
    return CorrectnessFixture(
        source,
        target,
        _identity(CorrectnessCase.CACHE_MISS, source, target, ()),
    )


def build_correctness_fixture(case: CorrectnessCase) -> CorrectnessFixture:
    builders = {
        CorrectnessCase.EXACT_PREFIX: build_exact_prefix_fixture,
        CorrectnessCase.MOVED_DOCUMENT: build_moved_document_fixture,
        CorrectnessCase.REORDERED_DOCUMENTS: build_reordered_documents_fixture,
        CorrectnessCase.CACHE_MISS: build_cache_miss_fixture,
    }
    if not isinstance(case, CorrectnessCase):
        raise ValueError("unsupported correctness fixture case")
    return builders[case]()


__all__ = [
    "CorrectnessFixture",
    "MovedDocumentFixture",
    "build_cache_miss_fixture",
    "build_correctness_fixture",
    "build_exact_prefix_fixture",
    "build_moved_document_fixture",
    "build_reordered_documents_fixture",
    "digest_token_ids",
]
