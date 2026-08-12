"""CPU-only tests for the exact vLLM 0.19.1 configuration boundary."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from cacheblend_gpt_oss.vllm_compat.v0_19_1.config_validation import (
    UnsupportedPinnedConfigError,
    collect_pinned_config_issues,
    require_pinned_config,
)


class FullAttentionSpec(SimpleNamespace):
    pass


class SlidingWindowSpec(SimpleNamespace):
    pass


def _valid_config() -> tuple[SimpleNamespace, SimpleNamespace]:
    rope = {
        "rope_type": "yarn",
        "factor": 32.0,
        "original_max_position_embeddings": 4096,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "truncate": False,
    }
    hf_config = SimpleNamespace(
        architectures=["GptOssForCausalLM"],
        model_type="gpt_oss",
        num_hidden_layers=24,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=64,
        sliding_window=128,
        max_position_embeddings=131_072,
        num_local_experts=32,
        num_experts_per_tok=4,
        quantization_config={"quant_method": "mxfp4"},
        attention_bias=True,
        rope_parameters=rope,
        rope_theta=150_000,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="/models/gpt-oss-20b",
            served_model_name=["openai/gpt-oss-20b"],
            hf_config=hf_config,
            disable_sliding_window=False,
            enable_prompt_embeds=False,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_dbo=False,
            enable_expert_parallel=False,
        ),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False,
        ),
        cache_config=SimpleNamespace(
            block_size=16,
            kv_offloading_size=None,
            kv_sharing_fast_prefill=False,
        ),
        attention_config=SimpleNamespace(
            backend=SimpleNamespace(name="TRITON_ATTN")
        ),
        speculative_config=None,
        lora_config=None,
    )
    sliding = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=64,
        sliding_window=128,
    )
    full = FullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=64,
        sliding_window=None,
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=sliding,
                layer_names=[
                    f"model.layers.{index}.attn.attn" for index in range(0, 24, 2)
                ],
            ),
            SimpleNamespace(
                kv_cache_spec=full,
                layer_names=[
                    f"model.layers.{index}.attn.attn" for index in range(1, 24, 2)
                ],
            ),
        ]
    )
    return vllm_config, kv_cache_config


def test_exact_pinned_configuration_is_accepted() -> None:
    vllm_config, kv_cache_config = _valid_config()

    assert (
        collect_pinned_config_issues(
            vllm_config,
            kv_cache_config,
            v2_model_runner_enabled=False,
        )
        == ()
    )
    require_pinned_config(
        vllm_config,
        kv_cache_config,
        v2_model_runner_enabled=False,
    )


@pytest.mark.parametrize(
    ("mutation", "expected_field"),
    [
        (
            lambda config, _: setattr(
                config.model_config.hf_config, "head_dim", 128
            ),
            "model.head_dim",
        ),
        (
            lambda config, _: setattr(
                config.attention_config.backend, "name", "FLASH_ATTN"
            ),
            "attention.backend",
        ),
        (
            lambda config, _: setattr(
                config.parallel_config, "tensor_parallel_size", 2
            ),
            "parallel.tensor_parallel_size",
        ),
        (
            lambda config, _: setattr(
                config.cache_config, "kv_offloading_size", 1.0
            ),
            "cache.kv_offloading_size",
        ),
        (
            lambda _, cache: cache.kv_cache_groups[0].layer_names.append(
                "model.layers.1.attn.attn"
            ),
            "kv.layer.1.spec_type",
        ),
    ],
)
def test_any_unaudited_configuration_fails_closed(
    mutation: Callable[[Any, Any], None],
    expected_field: str,
) -> None:
    vllm_config, kv_cache_config = deepcopy(_valid_config())
    mutation(vllm_config, kv_cache_config)

    with pytest.raises(UnsupportedPinnedConfigError) as error:
        require_pinned_config(
            vllm_config,
            kv_cache_config,
            v2_model_runner_enabled=False,
        )

    assert expected_field in {issue.field for issue in error.value.issues}


def test_v2_runner_and_incomplete_hybrid_layout_are_both_reported() -> None:
    vllm_config, kv_cache_config = _valid_config()
    kv_cache_config.kv_cache_groups[0].layer_names.pop()

    issues = collect_pinned_config_issues(
        vllm_config,
        kv_cache_config,
        v2_model_runner_enabled=True,
    )

    assert {issue.field for issue in issues} >= {"runner.v2_enabled", "kv.layers"}
