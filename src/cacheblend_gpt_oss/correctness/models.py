# SPDX-License-Identifier: Apache-2.0
"""Immutable correctness evidence for the pinned GPT-OSS milestone."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum

from cacheblend_gpt_oss.targets import PINNED_TARGET

ARTIFACT_SCHEMA_VERSION = 1
GPT_OSS_VOCAB_SIZE = 201_088
_HEX_40 = re.compile(r"[0-9a-f]{40}")
_HEX_64 = re.compile(r"[0-9a-f]{64}")


class CorrectnessCase(str, Enum):
    """Required comparison cases; the first live fixture is moved-document."""

    EXACT_PREFIX = "exact_prefix"
    MOVED_DOCUMENT = "moved_document"
    REORDERED_DOCUMENTS = "reordered_documents"
    CACHE_MISS = "cache_miss"


class CorrectnessRunMode(str, Enum):
    """Execution mode represented by one final-distribution artifact."""

    FULL_PREFILL = "full_prefill"
    CACHEBLEND_100PCT = "cacheblend_100pct"


def _require_text(name: str, value: str, *, maximum: int = 256) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or value.strip() != value
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"invalid correctness artifact {name}")


def _require_count(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid correctness artifact {name}")


@dataclass(frozen=True, slots=True)
class CorrectnessRuntimeIdentity:
    """Exact runtime identity required before two artifacts are comparable."""

    model_id: str
    model_revision: str
    tokenizer_revision: str
    plugin_commit: str
    model_config_digest: str
    kv_cache_config_digest: str
    vllm_version: str
    lmcache_version: str
    torch_version: str
    cuda_runtime: str
    gpu_name: str
    dtype: str = "torch.bfloat16"

    def __post_init__(self) -> None:
        expected = (
            ("model_id", self.model_id, PINNED_TARGET.model_id),
            ("vllm_version", self.vllm_version, PINNED_TARGET.vllm_version),
            ("lmcache_version", self.lmcache_version, PINNED_TARGET.lmcache_version),
            ("torch_version", self.torch_version, PINNED_TARGET.torch_version),
            ("cuda_runtime", self.cuda_runtime, PINNED_TARGET.cuda_runtime),
            ("gpu_name", self.gpu_name, PINNED_TARGET.gpu_name),
            ("dtype", self.dtype, "torch.bfloat16"),
        )
        if any(observed != required for _, observed, required in expected):
            raise ValueError("correctness artifact is outside the pinned runtime")
        _require_text("model_revision", self.model_revision)
        _require_text("tokenizer_revision", self.tokenizer_revision)
        if _HEX_40.fullmatch(self.plugin_commit) is None:
            raise ValueError("invalid correctness artifact plugin_commit")
        for name, digest in (
            ("model_config_digest", self.model_config_digest),
            ("kv_cache_config_digest", self.kv_cache_config_digest),
        ):
            if _HEX_64.fullmatch(digest) is None:
                raise ValueError(f"invalid correctness artifact {name}")


@dataclass(frozen=True, slots=True)
class PromptCaseIdentity:
    """Position-only metadata and digests; raw token sequences stay private."""

    case: CorrectnessCase
    target_prompt_digest: str
    target_prompt_tokens: int
    reusable_token_digest: str
    reusable_tokens: int
    source_start: int
    target_start: int

    def __post_init__(self) -> None:
        if not isinstance(self.case, CorrectnessCase):
            raise ValueError("invalid correctness case")
        for name, digest in (
            ("target_prompt_digest", self.target_prompt_digest),
            ("reusable_token_digest", self.reusable_token_digest),
        ):
            if _HEX_64.fullmatch(digest) is None:
                raise ValueError(f"invalid correctness artifact {name}")
        for name, value in (
            ("target_prompt_tokens", self.target_prompt_tokens),
            ("reusable_tokens", self.reusable_tokens),
            ("source_start", self.source_start),
            ("target_start", self.target_start),
        ):
            _require_count(name, value)
        if (
            self.reusable_tokens == 0
            or self.target_start + self.reusable_tokens > self.target_prompt_tokens
        ):
            raise ValueError("reusable range is outside the target prompt")
        if (
            self.case is CorrectnessCase.MOVED_DOCUMENT
            and self.source_start == self.target_start
        ):
            raise ValueError("moved-document positions must differ")
        if (
            self.case is CorrectnessCase.EXACT_PREFIX
            and self.source_start != self.target_start
        ):
            raise ValueError("exact-prefix positions must agree")


@dataclass(frozen=True, slots=True)
class ConnectorCorrectnessEvidence:
    """Per-target-request metric deltas required for the 100% reuse proof."""

    reusable_document_tokens_requested: int
    kv_tokens_found: int
    kv_tokens_loaded: int
    kv_tokens_rejected: int
    tokens_recomputed: int
    prefill_tokens_avoided: int

    def __post_init__(self) -> None:
        for name, value in (
            (
                "reusable_document_tokens_requested",
                self.reusable_document_tokens_requested,
            ),
            ("kv_tokens_found", self.kv_tokens_found),
            ("kv_tokens_loaded", self.kv_tokens_loaded),
            ("kv_tokens_rejected", self.kv_tokens_rejected),
            ("tokens_recomputed", self.tokens_recomputed),
            ("prefill_tokens_avoided", self.prefill_tokens_avoided),
        ):
            _require_count(name, value)
        if self.kv_tokens_found != self.kv_tokens_loaded + self.kv_tokens_rejected:
            raise ValueError("found KV tokens must be loaded or rejected")
        if self.kv_tokens_loaded > self.reusable_document_tokens_requested:
            raise ValueError("loaded KV tokens exceed reusable requested tokens")


@dataclass(frozen=True, slots=True)
class FullVocabularyLogprobs:
    """Complete one-token output log-softmax vector in token-ID order."""

    values: tuple[float, ...]
    sampled_token_id: int

    def __post_init__(self) -> None:
        try:
            values = tuple(self.values)
        except TypeError as exc:
            raise ValueError("invalid full-vocabulary distribution") from exc
        object.__setattr__(self, "values", values)
        if len(values) != GPT_OSS_VOCAB_SIZE:
            raise ValueError("full-vocabulary distribution has the wrong size")
        if (
            isinstance(self.sampled_token_id, bool)
            or not isinstance(self.sampled_token_id, int)
            or not 0 <= self.sampled_token_id < len(values)
        ):
            raise ValueError("invalid sampled token ID")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or math.isnan(value)
            or value == math.inf
            for value in values
        ):
            raise ValueError("invalid full-vocabulary log probability")

    @property
    def top_token_id(self) -> int:
        return max(range(len(self.values)), key=self.values.__getitem__)


@dataclass(frozen=True, slots=True)
class CorrectnessArtifact:
    """One baseline or CacheBlend final-distribution observation."""

    schema_version: int
    run_mode: CorrectnessRunMode
    runtime: CorrectnessRuntimeIdentity
    prompt: PromptCaseIdentity
    distribution: FullVocabularyLogprobs
    connector: ConnectorCorrectnessEvidence | None = None

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported correctness artifact schema")
        if (
            not isinstance(self.run_mode, CorrectnessRunMode)
            or not isinstance(self.runtime, CorrectnessRuntimeIdentity)
            or not isinstance(self.prompt, PromptCaseIdentity)
            or not isinstance(self.distribution, FullVocabularyLogprobs)
        ):
            raise ValueError("invalid correctness artifact")
        if self.run_mode is CorrectnessRunMode.FULL_PREFILL:
            if self.connector is not None:
                raise ValueError("full-prefill artifact cannot claim connector work")
            return
        if not isinstance(self.connector, ConnectorCorrectnessEvidence):
            raise ValueError("CacheBlend artifact requires connector evidence")
        if (
            self.connector.kv_tokens_loaded == 0
            or self.connector.kv_tokens_loaded != self.prompt.reusable_tokens
            or self.connector.reusable_document_tokens_requested
            != self.prompt.target_prompt_tokens
            or self.connector.tokens_recomputed != self.prompt.target_prompt_tokens
            or self.connector.prefill_tokens_avoided != 0
        ):
            raise ValueError("CacheBlend 100% artifact has invalid work evidence")


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "GPT_OSS_VOCAB_SIZE",
    "ConnectorCorrectnessEvidence",
    "CorrectnessArtifact",
    "CorrectnessCase",
    "CorrectnessRunMode",
    "CorrectnessRuntimeIdentity",
    "FullVocabularyLogprobs",
    "PromptCaseIdentity",
]
