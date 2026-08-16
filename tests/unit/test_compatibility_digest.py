"""CPU-only tests for pinned runtime compatibility identities."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from cacheblend_gpt_oss.vllm_compat.v0_19_1.adapters import adapt_kv_cache_config
from cacheblend_gpt_oss.vllm_compat.v0_19_1.compatibility_digest import (
    CompatibilityDigestError,
    CompatibilityDigestErrorCode,
    derive_runtime_compatibility_digests,
    require_runtime_compatibility_digests,
)


class FullAttentionSpec(SimpleNamespace):
    pass


class SlidingWindowSpec(SimpleNamespace):
    pass


class FakeHfConfig:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values

    def to_dict(self) -> dict[str, object]:
        return deepcopy(self.values)


def _runtime() -> tuple[SimpleNamespace, SimpleNamespace]:
    hf_values: dict[str, object] = {
        "architectures": ["GptOssForCausalLM"],
        "model_type": "gpt_oss",
        "hidden_size": 2880,
        "num_hidden_layers": 24,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 32.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        },
        "quantization_config": {"quant_method": "mxfp4"},
    }
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=FakeHfConfig(hf_values),
            served_model_name=["openai/gpt-oss-20b"],
            dtype="torch.bfloat16",
            max_model_len=131_072,
            disable_sliding_window=False,
            runner_type="generate",
        ),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=16),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        attention_config=SimpleNamespace(
            backend=SimpleNamespace(name="TRITON_ATTN")
        ),
    )
    kv_config = SimpleNamespace(
        num_blocks=128,
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=[
                    f"model.layers.{index}.attn.attn" for index in range(0, 24, 2)
                ],
                kv_cache_spec=SlidingWindowSpec(
                    block_size=16,
                    num_kv_heads=8,
                    head_size=64,
                    sliding_window=128,
                ),
            ),
            SimpleNamespace(
                layer_names=[
                    f"model.layers.{index}.attn.attn" for index in range(1, 24, 2)
                ],
                kv_cache_spec=FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=8,
                    head_size=64,
                    head_size_v=64,
                    sliding_window=None,
                    attention_chunk_size=None,
                ),
            ),
        ],
    )
    return config, kv_config


def _derive(
    config: SimpleNamespace, kv_config: SimpleNamespace
) -> tuple[str, str]:
    result = derive_runtime_compatibility_digests(
        config, adapt_kv_cache_config(kv_config)
    )
    assert len(result.model_config_digest) == 64
    assert len(result.kv_cache_config_digest) == 64
    assert result.model_config_digest.islower()
    assert result.kv_cache_config_digest.islower()
    return result.model_config_digest, result.kv_cache_config_digest


def test_digests_are_deterministic_and_ignore_mapping_insertion_order() -> None:
    config, kv_config = _runtime()
    first = _derive(config, kv_config)
    values = config.model_config.hf_config.values
    config.model_config.hf_config.values = dict(reversed(tuple(values.items())))

    assert _derive(config, kv_config) == first


def test_hf_config_integer_mapping_keys_are_canonicalized_without_collision() -> None:
    config, kv_config = _runtime()
    config.model_config.hf_config.values["id2label"] = {
        0: "LABEL_0",
        1: "LABEL_1",
    }

    integer_key_digest = _derive(config, kv_config)

    config.model_config.hf_config.values["id2label"] = {
        "0": "LABEL_0",
        "1": "LABEL_1",
    }
    string_key_digest = _derive(config, kv_config)

    assert integer_key_digest != string_key_digest


def test_model_and_kv_changes_are_separated() -> None:
    config, kv_config = _runtime()
    baseline_model, baseline_kv = _derive(config, kv_config)

    changed_model = deepcopy(config)
    changed_model.model_config.hf_config.values["hidden_size"] = 2881
    model_digest, kv_digest = _derive(changed_model, kv_config)
    assert model_digest != baseline_model
    assert kv_digest == baseline_kv

    changed_kv = deepcopy(config)
    changed_kv.cache_config.cache_dtype = "bfloat16"
    model_digest, kv_digest = _derive(changed_kv, kv_config)
    assert model_digest == baseline_model
    assert kv_digest != baseline_kv


def test_selective_custom_dispatch_alias_preserves_triton_kv_identity() -> None:
    config, kv_config = _runtime()
    triton_model, triton_kv = _derive(config, kv_config)

    config.attention_config.backend.name = "CUSTOM"
    custom_model, custom_kv = _derive(config, kv_config)

    assert custom_model == triton_model
    assert custom_kv == triton_kv


def test_runtime_capacity_is_not_part_of_kv_representation_identity() -> None:
    config, kv_config = _runtime()
    baseline = _derive(config, kv_config)
    kv_config.num_blocks = 4096

    assert _derive(config, kv_config) == baseline


def test_canonical_adapter_order_removes_input_layer_order_noise() -> None:
    config, kv_config = _runtime()
    baseline = _derive(config, kv_config)
    kv_config.kv_cache_groups[0].layer_names.reverse()
    kv_config.kv_cache_groups[1].layer_names.reverse()

    assert _derive(config, kv_config) == baseline


def test_expected_digest_attestation_is_checked_independently() -> None:
    config, kv_config = _runtime()
    adapted = adapt_kv_cache_config(kv_config)
    observed = derive_runtime_compatibility_digests(config, adapted)

    assert (
        require_runtime_compatibility_digests(
            config,
            adapted,
            expected_model_config_digest=observed.model_config_digest,
            expected_kv_cache_config_digest=observed.kv_cache_config_digest,
        )
        == observed
    )
    with pytest.raises(CompatibilityDigestError) as model_error:
        require_runtime_compatibility_digests(
            config,
            adapted,
            expected_model_config_digest="0" * 64,
            expected_kv_cache_config_digest=observed.kv_cache_config_digest,
        )
    assert model_error.value.code is CompatibilityDigestErrorCode.MODEL_DIGEST_MISMATCH

    with pytest.raises(CompatibilityDigestError) as kv_error:
        require_runtime_compatibility_digests(
            config,
            adapted,
            expected_model_config_digest=observed.model_config_digest,
            expected_kv_cache_config_digest="0" * 64,
        )
    assert kv_error.value.code is CompatibilityDigestErrorCode.KV_DIGEST_MISMATCH


def test_hf_config_must_be_bounded_plain_json_data() -> None:
    config, kv_config = _runtime()
    config.model_config.hf_config.values["unsupported"] = {1, 2, 3}

    with pytest.raises(CompatibilityDigestError) as error:
        _derive(config, kv_config)
    assert error.value.code is CompatibilityDigestErrorCode.UNSUPPORTED_CONFIG_VALUE

    config, kv_config = _runtime()
    nested: object = "leaf"
    for _ in range(40):
        nested = [nested]
    config.model_config.hf_config.values["too_deep"] = nested
    with pytest.raises(CompatibilityDigestError) as error:
        _derive(config, kv_config)
    assert error.value.code is CompatibilityDigestErrorCode.CONFIG_TOO_LARGE


def test_missing_to_dict_fails_closed() -> None:
    config, kv_config = _runtime()
    config.model_config.hf_config = SimpleNamespace()

    with pytest.raises(CompatibilityDigestError) as error:
        _derive(config, kv_config)
    assert error.value.code is CompatibilityDigestErrorCode.INVALID_HF_CONFIG


def test_module_has_no_heavy_runtime_imports() -> None:
    source = Path(
        "src/cacheblend_gpt_oss/vllm_compat/v0_19_1/compatibility_digest.py"
    ).read_text(encoding="utf-8")
    for dependency in ("vllm", "lmcache", "torch", "cuda"):
        assert f"\nimport {dependency}" not in source
        assert f"\nfrom {dependency}" not in source
