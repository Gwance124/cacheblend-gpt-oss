from dataclasses import replace

import pytest

from cacheblend_gpt_oss.planner import (
    CacheNamespace,
    TokenSegment,
    canonical_token_bytes,
    fingerprint_segment,
)


def namespace() -> CacheNamespace:
    return CacheNamespace(
        schema_version=1,
        model_id="openai/gpt-oss-20b",
        model_revision="model-revision",
        tokenizer_id="openai/gpt-oss-20b",
        tokenizer_revision="tokenizer-revision",
        model_config_digest="model-config-sha256",
        kv_cache_config_digest="hybrid-cache-config-sha256",
        adapter_revision="adapter-revision",
        vllm_version="0.19.1",
        lmcache_version="0.4.3",
        torch_version="2.10.0+cu128",
        cuda_runtime="12.8",
    )


def test_fingerprint_is_position_independent_but_namespace_bound() -> None:
    source = TokenSegment.at(10, [17, 23, 41])
    moved = TokenSegment.at(800, [17, 23, 41])

    source_fingerprint = fingerprint_segment(namespace(), source.token_ids)

    assert source_fingerprint.hex_digest == (
        "d5d5a594dd690ce8efb92dc3dbfb5a08"
        "79026e5a95a365fbcf2001ccccfad7fe"
    )
    assert source_fingerprint == fingerprint_segment(namespace(), moved.token_ids)
    assert source_fingerprint != fingerprint_segment(
        replace(namespace(), model_revision="other-model-revision"),
        moved.token_ids,
    )
    assert source_fingerprint != fingerprint_segment(namespace(), [17, 23, 42])


def test_canonical_token_encoding_has_sequence_boundaries() -> None:
    assert canonical_token_bytes([1, 256]) != canonical_token_bytes([257])
    assert canonical_token_bytes([1, 2]) != canonical_token_bytes([1, 2, 0])


@pytest.mark.parametrize("invalid_token", [-1, 1 << 64, True, "5"])
def test_canonical_token_encoding_rejects_invalid_ids(invalid_token: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        canonical_token_bytes([invalid_token])  # type: ignore[list-item]
