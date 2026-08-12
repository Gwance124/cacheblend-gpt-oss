# SPDX-License-Identifier: Apache-2.0
"""Deterministic raw-token fixture for the first moved-document GPU gate."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from cacheblend_gpt_oss.correctness.models import (
    CorrectnessCase,
    PromptCaseIdentity,
)

_DOCUMENT_TOKENS = tuple(range(1_024, 1_280))
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
class MovedDocumentFixture:
    """A 256-token source chunk moved by 17 non-block-aligned positions."""

    source_prompt_token_ids: tuple[int, ...]
    target_prompt_token_ids: tuple[int, ...]
    prompt_identity: PromptCaseIdentity


def build_moved_document_fixture() -> MovedDocumentFixture:
    """Return the immutable raw-token fixture used by all M3 captures."""

    source = _DOCUMENT_TOKENS
    target = (*_TARGET_PREFIX_TOKENS, *_DOCUMENT_TOKENS, *_TARGET_SUFFIX_TOKENS)
    return MovedDocumentFixture(
        source_prompt_token_ids=source,
        target_prompt_token_ids=target,
        prompt_identity=PromptCaseIdentity(
            case=CorrectnessCase.MOVED_DOCUMENT,
            target_prompt_digest=digest_token_ids(target),
            target_prompt_tokens=len(target),
            reusable_token_digest=digest_token_ids(_DOCUMENT_TOKENS),
            reusable_tokens=len(_DOCUMENT_TOKENS),
            source_start=0,
            target_start=len(_TARGET_PREFIX_TOKENS),
        ),
    )


__all__ = ["MovedDocumentFixture", "build_moved_document_fixture", "digest_token_ids"]
