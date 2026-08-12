# SPDX-License-Identifier: Apache-2.0
"""Deterministic cache-compatibility digests for the pinned vLLM target.

Persistent KV must never be trusted merely because an operator supplied two
well-formed hexadecimal strings.  This module derives those strings from the
finalized vLLM configuration and the already validated GPT-OSS hybrid layout,
without importing vLLM, Torch, CUDA, or LMCache.

The model digest covers the complete normalized Hugging Face configuration
returned by ``PretrainedConfig.to_dict`` plus the small set of vLLM model
settings that can change executed activations.  The KV digest covers the exact
two-group physical/logical layout, dtypes, and parallel topology.  Runtime
capacity (``num_blocks``) is deliberately excluded: it changes how many
requests fit, not the representation of a token's K/V rows.

The consumed fields are pinned to vLLM 0.19.1 commit
``b1388b1fbf5aaef47937fabe98931211684666a6``:

* finalized model configuration:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/config/model.py#L120-L220
* cache dtype and block configuration:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/config/cache.py#L20-L90
* finalized hybrid ``KVCacheConfig`` groups:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/kv_cache_interface.py#L461-L490
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import NoReturn, cast

from cacheblend_gpt_oss.vllm_compat.v0_19_1.adapters import AdaptedKvCacheConfig

_MODEL_DIGEST_DOMAIN = b"cacheblend-gpt-oss\x00model-config\x00v1\x00"
_KV_DIGEST_DOMAIN = b"cacheblend-gpt-oss\x00kv-config\x00v1\x00"
_MAX_CANONICAL_DEPTH = 32
_MAX_CANONICAL_ITEMS = 100_000
_MAX_CANONICAL_BYTES = 4 * 1024 * 1024
_MISSING = object()


class CompatibilityDigestErrorCode(str, Enum):
    """Bounded digest failures safe for startup diagnostics."""

    INVALID_VLLM_CONFIG = "invalid_vllm_config"
    INVALID_HF_CONFIG = "invalid_hf_config"
    UNSUPPORTED_CONFIG_VALUE = "unsupported_config_value"
    CONFIG_TOO_LARGE = "config_too_large"
    MODEL_DIGEST_MISMATCH = "model_digest_mismatch"
    KV_DIGEST_MISMATCH = "kv_digest_mismatch"


class CompatibilityDigestError(RuntimeError):
    """Fail-closed configuration identity error with a stable code."""

    def __init__(self, code: CompatibilityDigestErrorCode) -> None:
        self.code = code
        super().__init__(f"runtime compatibility digest failure: {code.value}")


def _fail(code: CompatibilityDigestErrorCode) -> NoReturn:
    raise CompatibilityDigestError(code)


@dataclass(frozen=True, slots=True)
class RuntimeCompatibilityDigests:
    """SHA-256 identities derived from one finalized runtime."""

    model_config_digest: str
    kv_cache_config_digest: str


@dataclass(slots=True)
class _CanonicalBudget:
    items: int = 0

    def consume(self) -> None:
        self.items += 1
        if self.items > _MAX_CANONICAL_ITEMS:
            _fail(CompatibilityDigestErrorCode.CONFIG_TOO_LARGE)


def _canonicalize(value: object, budget: _CanonicalBudget, depth: int = 0) -> object:
    if depth > _MAX_CANONICAL_DEPTH:
        _fail(CompatibilityDigestErrorCode.CONFIG_TOO_LARGE)
    budget.consume()
    if value is None or isinstance(value, bool | str | int):
        if isinstance(value, str) and "\x00" in value:
            _fail(CompatibilityDigestErrorCode.UNSUPPORTED_CONFIG_VALUE)
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(CompatibilityDigestErrorCode.UNSUPPORTED_CONFIG_VALUE)
        return value
    if isinstance(value, list | tuple):
        return [_canonicalize(item, budget, depth + 1) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) or "\x00" in key for key in value):
            _fail(CompatibilityDigestErrorCode.UNSUPPORTED_CONFIG_VALUE)
        return {
            key: _canonicalize(value[key], budget, depth + 1)
            for key in sorted(value)
        }
    _fail(CompatibilityDigestErrorCode.UNSUPPORTED_CONFIG_VALUE)


def _encode(value: object) -> bytes:
    canonical = _canonicalize(value, _CanonicalBudget())
    try:
        encoded = json.dumps(
            canonical,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CompatibilityDigestError(
            CompatibilityDigestErrorCode.UNSUPPORTED_CONFIG_VALUE
        ) from exc
    if len(encoded) > _MAX_CANONICAL_BYTES:
        _fail(CompatibilityDigestErrorCode.CONFIG_TOO_LARGE)
    return encoded


def _attribute(value: object, name: str) -> object:
    observed = getattr(value, name, _MISSING)
    if observed is _MISSING:
        _fail(CompatibilityDigestErrorCode.INVALID_VLLM_CONFIG)
    return observed


def _stringified(value: object) -> str:
    rendered = str(value)
    if not rendered or "\x00" in rendered:
        _fail(CompatibilityDigestErrorCode.INVALID_VLLM_CONFIG)
    return rendered


def _named_or_stringified(value: object) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str) and name and "\x00" not in name:
        return name
    return _stringified(value)


def _served_names(model_config: object) -> list[str]:
    raw = _attribute(model_config, "served_model_name")
    if isinstance(raw, str):
        names = [raw]
    elif isinstance(raw, list | tuple) and all(isinstance(item, str) for item in raw):
        names = list(raw)
    else:
        _fail(CompatibilityDigestErrorCode.INVALID_VLLM_CONFIG)
    if not names or any(not name for name in names):
        _fail(CompatibilityDigestErrorCode.INVALID_VLLM_CONFIG)
    return sorted(names)


def _hf_config_dictionary(model_config: object) -> dict[str, object]:
    hf_config = _attribute(model_config, "hf_config")
    to_dict = getattr(hf_config, "to_dict", None)
    if not callable(to_dict):
        _fail(CompatibilityDigestErrorCode.INVALID_HF_CONFIG)
    try:
        raw = to_dict()
    except Exception as exc:
        raise CompatibilityDigestError(
            CompatibilityDigestErrorCode.INVALID_HF_CONFIG
        ) from exc
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        _fail(CompatibilityDigestErrorCode.INVALID_HF_CONFIG)
    return cast(dict[str, object], raw)


def _sha256(domain: bytes, value: object) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(_encode(value))
    return digest.hexdigest()


def derive_runtime_compatibility_digests(
    vllm_config: object,
    adapted_kv_cache_config: AdaptedKvCacheConfig,
) -> RuntimeCompatibilityDigests:
    """Derive both persistent-cache identities from finalized configuration."""

    if not isinstance(adapted_kv_cache_config, AdaptedKvCacheConfig):
        _fail(CompatibilityDigestErrorCode.INVALID_VLLM_CONFIG)
    model = _attribute(vllm_config, "model_config")
    cache = _attribute(vllm_config, "cache_config")
    parallel = _attribute(vllm_config, "parallel_config")
    attention = _attribute(vllm_config, "attention_config")
    attention_backend = _attribute(attention, "backend")

    model_view = {
        "schema": "gpt-oss-model-config-v1",
        "hf_config": _hf_config_dictionary(model),
        "served_model_names": _served_names(model),
        "dtype": _stringified(_attribute(model, "dtype")),
        "max_model_len": _attribute(model, "max_model_len"),
        "disable_sliding_window": _attribute(model, "disable_sliding_window"),
        "runner_type": _attribute(model, "runner_type"),
    }

    layout = adapted_kv_cache_config.gpt_oss_layout
    groups = [
        {
            "group_id": group.group_id,
            "attention_kind": group.attention_kind.value,
            "layer_names": list(group.layer_names),
            "block_size": group.block_size,
            "sliding_window": group.sliding_window,
        }
        for group in sorted(layout.groups, key=lambda item: item.group_id)
    ]
    kv_view = {
        "schema": "gpt-oss-kv-config-v1",
        "cache_dtype": _stringified(_attribute(cache, "cache_dtype")),
        "cache_block_size": _attribute(cache, "block_size"),
        "model_dtype": _stringified(_attribute(model, "dtype")),
        "attention_backend": _named_or_stringified(attention_backend),
        "tensor_parallel_size": _attribute(parallel, "tensor_parallel_size"),
        "pipeline_parallel_size": _attribute(parallel, "pipeline_parallel_size"),
        "data_parallel_size": _attribute(parallel, "data_parallel_size"),
        "prefill_context_parallel_size": _attribute(
            parallel, "prefill_context_parallel_size"
        ),
        "decode_context_parallel_size": _attribute(
            parallel, "decode_context_parallel_size"
        ),
        "kv_components": 2,
        "kv_heads": 8,
        "head_dimension": 64,
        "groups": groups,
    }
    return RuntimeCompatibilityDigests(
        model_config_digest=_sha256(_MODEL_DIGEST_DOMAIN, model_view),
        kv_cache_config_digest=_sha256(_KV_DIGEST_DOMAIN, kv_view),
    )


def require_runtime_compatibility_digests(
    vllm_config: object,
    adapted_kv_cache_config: AdaptedKvCacheConfig,
    *,
    expected_model_config_digest: str,
    expected_kv_cache_config_digest: str,
) -> RuntimeCompatibilityDigests:
    """Derive identities and reject any operator/runtime mismatch."""

    observed = derive_runtime_compatibility_digests(
        vllm_config, adapted_kv_cache_config
    )
    if observed.model_config_digest != expected_model_config_digest:
        _fail(CompatibilityDigestErrorCode.MODEL_DIGEST_MISMATCH)
    if observed.kv_cache_config_digest != expected_kv_cache_config_digest:
        _fail(CompatibilityDigestErrorCode.KV_DIGEST_MISMATCH)
    return observed


__all__ = [
    "CompatibilityDigestError",
    "CompatibilityDigestErrorCode",
    "RuntimeCompatibilityDigests",
    "derive_runtime_compatibility_digests",
    "require_runtime_compatibility_digests",
]
