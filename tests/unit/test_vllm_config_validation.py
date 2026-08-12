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
    collect_transfer_100pct_config_issues,
    require_pinned_config,
    require_transfer_100pct_config,
)


class FullAttentionSpec(SimpleNamespace):
    pass


class SlidingWindowSpec(SimpleNamespace):
    pass


class _StringLike:
    def __str__(self) -> str:
        return "openai/gpt-oss-20b"


def _valid_config() -> tuple[SimpleNamespace, SimpleNamespace]:
    rope = {
        "rope_type": "yarn",
        "rope_theta": 150_000,
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
        layer_types=["sliding_attention", "full_attention"] * 12,
        num_attention_heads=64,
        num_key_value_heads=8,
        vocab_size=201_088,
        head_dim=64,
        sliding_window=128,
        max_position_embeddings=131_072,
        num_local_experts=32,
        num_experts_per_tok=4,
        quantization_config={"quant_method": "mxfp4"},
        attention_bias=True,
        rope_parameters=rope,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="/models/gpt-oss-20b",
            served_model_name=["openai/gpt-oss-20b"],
            hf_config=hf_config,
            disable_sliding_window=False,
            enable_prompt_embeds=False,
            dtype="torch.bfloat16",
            enforce_eager=True,
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
            max_num_seqs=1,
            max_num_batched_tokens=4096,
            max_num_scheduled_tokens=4096,
            long_prefill_token_threshold=0,
            async_scheduling=False,
            scheduler_cls=None,
        ),
        cache_config=SimpleNamespace(
            block_size=16,
            kv_offloading_size=None,
            kv_sharing_fast_prefill=False,
            enable_prefix_caching=False,
            cache_dtype="auto",
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


def test_finalized_rope_parameters_must_contain_theta() -> None:
    """Do not treat raw top-level rope_theta as the model-consumed value.

    The pinned vLLM loader patches legacy ``rope_theta`` and ``rope_scaling``
    fields into ``hf_config.rope_parameters`` before GPT-OSS construction.
    This fixture models a pre-patch/raw config and must fail closed rather than
    silently accepting a value that the model adapter will not read.
    """

    vllm_config, kv_cache_config = _valid_config()
    raw_rope_scaling = dict(vllm_config.model_config.hf_config.rope_parameters)
    raw_rope_scaling.pop("rope_theta")
    vllm_config.model_config.hf_config.rope_parameters = None
    vllm_config.model_config.hf_config.rope_scaling = raw_rope_scaling
    vllm_config.model_config.hf_config.rope_theta = 150_000

    issues = collect_pinned_config_issues(
        vllm_config,
        kv_cache_config,
        v2_model_runner_enabled=False,
    )

    assert "rope.theta" in {issue.field for issue in issues}


def test_complete_raw_rope_scaling_is_not_treated_as_finalized() -> None:
    """A legacy mapping must not bypass vLLM's normalization boundary."""

    vllm_config, kv_cache_config = _valid_config()
    raw_rope_scaling = dict(vllm_config.model_config.hf_config.rope_parameters)
    vllm_config.model_config.hf_config.rope_parameters = None
    vllm_config.model_config.hf_config.rope_scaling = raw_rope_scaling

    issues = collect_pinned_config_issues(
        vllm_config,
        kv_cache_config,
        v2_model_runner_enabled=False,
    )

    assert "rope.type" in {issue.field for issue in issues}
    assert "rope.theta" in {issue.field for issue in issues}


def test_vllm_rope_normalization_shape_is_accepted() -> None:
    """A vLLM-finalized flat rope_parameters mapping is the accepted shape."""

    vllm_config, kv_cache_config = _valid_config()
    finalized = dict(vllm_config.model_config.hf_config.rope_parameters)
    vllm_config.model_config.hf_config.rope_parameters = finalized
    vllm_config.model_config.hf_config.rope_scaling = {
        key: value for key, value in finalized.items() if key != "rope_theta"
    }
    vllm_config.model_config.hf_config.rope_theta = 150_000

    assert (
        collect_pinned_config_issues(
            vllm_config,
            kv_cache_config,
            v2_model_runner_enabled=False,
        )
        == ()
    )


@pytest.mark.parametrize(
    ("mutation", "expected_field"),
    [
        (
            lambda config, _: setattr(
                config.model_config,
                "served_model_name",
                [_StringLike()],
            ),
            "model.served_name",
        ),
        (
            lambda _, cache: cache.kv_cache_groups[0].layer_names.__setitem__(
                0, _StringLike()
            ),
            "kv.groups.0.layer_name",
        ),
    ],
)
def test_pinned_config_does_not_coerce_object_names_to_strings(
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


@pytest.mark.parametrize(
    "layer_name_value",
    [
        "model.layers.00.attn.attn",
        "model.layers." + "9" * 10_000 + ".attn.attn",
    ],
)
def test_malformed_numeric_layer_names_fail_as_bounded_issues(
    layer_name_value: str,
) -> None:
    vllm_config, kv_cache_config = _valid_config()
    kv_cache_config.kv_cache_groups[0].layer_names[0] = layer_name_value

    issues = collect_pinned_config_issues(
        vllm_config,
        kv_cache_config,
        v2_model_runner_enabled=False,
    )

    assert "kv.groups.0.layer_name" in {issue.field for issue in issues}


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
                config.model_config.hf_config,
                "layer_types",
                ["full_attention", "sliding_attention"] * 12,
            ),
            "model.layer_types",
        ),
        (
            lambda config, _: setattr(
                config.model_config.hf_config, "vocab_size", 201_087
            ),
            "model.vocab_size",
        ),
        (
            lambda config, _: config.model_config.hf_config.rope_parameters.update(
                rope_theta=149_999
            ),
            "rope.theta",
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


@pytest.mark.parametrize(
    ("mutation", "expected_field"),
    [
        (
            lambda config, _: setattr(
                config.model_config.hf_config, "num_hidden_layers", True
            ),
            "model.num_hidden_layers",
        ),
        (
            lambda config, _: setattr(
                config.parallel_config, "tensor_parallel_size", True
            ),
            "parallel.tensor_parallel_size",
        ),
        (
            lambda config, _: setattr(
                config.model_config.hf_config, "attention_bias", 1
            ),
            "model.attention_bias",
        ),
        (
            lambda _, cache: setattr(
                cache.kv_cache_groups[0].kv_cache_spec, "block_size", True
            ),
            "kv.groups.0.block_size",
        ),
    ],
)
def test_pinned_configuration_rejects_boolean_numeric_coercion(
    mutation: Callable[[Any, Any], None],
    expected_field: str,
) -> None:
    vllm_config, kv_cache_config = _valid_config()
    mutation(vllm_config, kv_cache_config)

    with pytest.raises(UnsupportedPinnedConfigError) as error:
        require_pinned_config(
            vllm_config,
            kv_cache_config,
            v2_model_runner_enabled=False,
        )

    assert expected_field in {issue.field for issue in error.value.issues}


def test_exact_transfer_100pct_scheduler_envelope_is_accepted() -> None:
    vllm_config, _ = _valid_config()

    assert (
        collect_transfer_100pct_config_issues(
            vllm_config,
            staging_token_capacity=4096,
        )
        == ()
    )
    require_transfer_100pct_config(
        vllm_config,
        staging_token_capacity=4096,
    )


@pytest.mark.parametrize(
    ("mutation", "expected_field"),
    [
        (
            lambda config: setattr(config.model_config, "dtype", "torch.float16"),
            "transfer.model.dtype",
        ),
        (
            lambda config: setattr(config.model_config, "enforce_eager", False),
            "transfer.model.enforce_eager",
        ),
        (
            lambda config: setattr(config.cache_config, "enable_prefix_caching", True),
            "transfer.cache.enable_prefix_caching",
        ),
        (
            lambda config: setattr(config.cache_config, "cache_dtype", "fp8"),
            "transfer.cache.cache_dtype",
        ),
        (
            lambda config: setattr(config.scheduler_config, "max_num_seqs", 2),
            "transfer.scheduler.max_num_seqs",
        ),
        (
            lambda config: setattr(
                config.scheduler_config, "long_prefill_token_threshold", 2048
            ),
            "transfer.scheduler.long_prefill_token_threshold",
        ),
        (
            lambda config: setattr(config.scheduler_config, "async_scheduling", True),
            "transfer.scheduler.async_scheduling",
        ),
        (
            lambda config: setattr(
                config.scheduler_config, "scheduler_cls", "custom.Scheduler"
            ),
            "transfer.scheduler.scheduler_cls",
        ),
        (
            lambda config: setattr(
                config.scheduler_config, "max_num_batched_tokens", 4095
            ),
            "transfer.scheduler.max_num_batched_tokens",
        ),
        (
            lambda config: setattr(
                config.scheduler_config, "max_num_scheduled_tokens", 4095
            ),
            "transfer.scheduler.max_num_scheduled_tokens",
        ),
    ],
)
def test_transfer_100pct_rejects_unsafe_scheduler_or_dtype_configuration(
    mutation: Callable[[Any], None],
    expected_field: str,
) -> None:
    vllm_config, _ = _valid_config()
    mutation(vllm_config)

    with pytest.raises(UnsupportedPinnedConfigError) as error:
        require_transfer_100pct_config(
            vllm_config,
            staging_token_capacity=4096,
        )

    assert expected_field in {issue.field for issue in error.value.issues}


def test_transfer_100pct_accepts_pinned_scheduler_default_budget_fallback() -> None:
    vllm_config, _ = _valid_config()
    vllm_config.scheduler_config.max_num_scheduled_tokens = None

    require_transfer_100pct_config(
        vllm_config,
        staging_token_capacity=4096,
    )


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    [
        ("max_num_seqs", True),
        ("long_prefill_token_threshold", False),
        ("async_scheduling", 0),
    ],
)
def test_transfer_100pct_rejects_boolean_scheduler_coercion(
    field_name: str,
    invalid: object,
) -> None:
    vllm_config, _ = _valid_config()
    setattr(vllm_config.scheduler_config, field_name, invalid)

    with pytest.raises(UnsupportedPinnedConfigError) as error:
        require_transfer_100pct_config(
            vllm_config,
            staging_token_capacity=4096,
        )

    assert f"transfer.scheduler.{field_name}" in {
        issue.field for issue in error.value.issues
    }


@pytest.mark.parametrize("invalid", [0, False, "4096"])
def test_transfer_100pct_rejects_invalid_explicit_scheduled_budget(
    invalid: object,
) -> None:
    vllm_config, _ = _valid_config()
    vllm_config.scheduler_config.max_num_scheduled_tokens = invalid

    with pytest.raises(UnsupportedPinnedConfigError) as error:
        require_transfer_100pct_config(
            vllm_config,
            staging_token_capacity=4096,
        )

    assert "transfer.scheduler.max_num_scheduled_tokens" in {
        issue.field for issue in error.value.issues
    }


@pytest.mark.parametrize("capacity", [0, -1, True, 1.0])
def test_transfer_100pct_rejects_invalid_staging_capacity(capacity: object) -> None:
    vllm_config, _ = _valid_config()

    with pytest.raises(ValueError, match="positive integer"):
        collect_transfer_100pct_config_issues(
            vllm_config,
            staging_token_capacity=capacity,  # type: ignore[arg-type]
        )
