from dataclasses import FrozenInstanceError, replace

import pytest

from cacheblend_gpt_oss.connector import (
    GPT_OSS_ATTENTION_PATTERN,
    AttentionLayerKind,
    MismatchAction,
    RuntimeMode,
    RuntimeObservation,
    RuntimeValidationPolicy,
    RuntimeValidator,
    UnsupportedFeature,
    ValidationFailureCode,
)


def _valid_observation() -> RuntimeObservation:
    return RuntimeObservation(
        model_id="openai/gpt-oss-20b",
        vllm_version="0.19.1",
        lmcache_version="0.4.3",
        torch_version="2.10.0+cu128",
        cuda_runtime="12.8",
        gpu_name="NVIDIA A100-SXM4-80GB",
        gpu_compute_capability="8.0",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        num_hidden_layers=24,
        attention_pattern=GPT_OSS_ATTENTION_PATTERN,
        sliding_window=128,
        hybrid_kv_manager_enabled=True,
        v2_model_runner_enabled=False,
        attention_backend="TRITON_ATTN",
        attention_backend_supports_sinks=True,
        unsupported_features=frozenset(),
        full_prefill_fallback_available=True,
    )


def test_exact_pinned_runtime_enables_cacheblend() -> None:
    result = RuntimeValidator().validate(_valid_observation())

    assert result.mode is RuntimeMode.CACHEBLEND
    assert result.cacheblend_enabled
    assert not result.requires_full_prefill
    assert result.issues == ()
    assert len(GPT_OSS_ATTENTION_PATTERN) == 24
    assert GPT_OSS_ATTENTION_PATTERN[::2] == (AttentionLayerKind.SLIDING,) * 12
    assert GPT_OSS_ATTENTION_PATTERN[1::2] == (AttentionLayerKind.FULL,) * 12


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    [
        ("model_id", "other/model", ValidationFailureCode.MODEL_ID_MISMATCH),
        ("vllm_version", "0.20.0", ValidationFailureCode.VLLM_VERSION_MISMATCH),
        (
            "lmcache_version",
            "0.4.4",
            ValidationFailureCode.LMCACHE_VERSION_MISMATCH,
        ),
        (
            "torch_version",
            "2.10.0+cu126",
            ValidationFailureCode.TORCH_VERSION_MISMATCH,
        ),
        ("cuda_runtime", "12.6", ValidationFailureCode.CUDA_RUNTIME_MISMATCH),
        ("gpu_name", "NVIDIA H100", ValidationFailureCode.GPU_NAME_MISMATCH),
        (
            "gpu_compute_capability",
            "9.0",
            ValidationFailureCode.GPU_COMPUTE_CAPABILITY_MISMATCH,
        ),
        (
            "tensor_parallel_size",
            2,
            ValidationFailureCode.TENSOR_PARALLEL_SIZE_MISMATCH,
        ),
        (
            "pipeline_parallel_size",
            2,
            ValidationFailureCode.PIPELINE_PARALLEL_SIZE_MISMATCH,
        ),
        ("num_hidden_layers", 25, ValidationFailureCode.LAYER_COUNT_MISMATCH),
        (
            "attention_pattern",
            tuple(reversed(GPT_OSS_ATTENTION_PATTERN)),
            ValidationFailureCode.ATTENTION_PATTERN_MISMATCH,
        ),
        (
            "sliding_window",
            256,
            ValidationFailureCode.SLIDING_WINDOW_MISMATCH,
        ),
        (
            "hybrid_kv_manager_enabled",
            False,
            ValidationFailureCode.HYBRID_KV_MANAGER_DISABLED,
        ),
        (
            "v2_model_runner_enabled",
            True,
            ValidationFailureCode.V2_MODEL_RUNNER_ENABLED,
        ),
        (
            "attention_backend",
            "FLASH_ATTN",
            ValidationFailureCode.ATTENTION_BACKEND_MISMATCH,
        ),
        (
            "attention_backend_supports_sinks",
            False,
            ValidationFailureCode.ATTENTION_SINKS_UNSUPPORTED,
        ),
        (
            "unsupported_features",
            frozenset({UnsupportedFeature.SPECULATIVE_DECODING}),
            ValidationFailureCode.UNSUPPORTED_FEATURE_ENABLED,
        ),
    ],
)
def test_each_unsupported_runtime_condition_rejects_startup(
    field: str,
    value: object,
    expected_code: ValidationFailureCode,
) -> None:
    result = RuntimeValidator().validate(
        replace(_valid_observation(), **{field: value})
    )

    assert result.mode is RuntimeMode.REJECTED
    assert not result.cacheblend_enabled
    assert not result.requires_full_prefill
    assert expected_code in {issue.code for issue in result.issues}


def test_policy_can_force_a_reuse_disabled_full_prefill_fallback() -> None:
    validator = RuntimeValidator(
        policy=RuntimeValidationPolicy(
            mismatch_action=MismatchAction.FALL_BACK_TO_FULL_PREFILL
        )
    )
    observation = replace(_valid_observation(), attention_backend="FLASH_ATTN")

    result = validator.validate(observation)

    assert result.mode is RuntimeMode.FULL_PREFILL
    assert result.requires_full_prefill
    assert not result.cacheblend_enabled
    assert [issue.code for issue in result.issues] == [
        ValidationFailureCode.ATTENTION_BACKEND_MISMATCH
    ]


def test_missing_full_prefill_path_escalates_fallback_policy_to_rejection() -> None:
    validator = RuntimeValidator(
        policy=RuntimeValidationPolicy(
            mismatch_action=MismatchAction.FALL_BACK_TO_FULL_PREFILL
        )
    )
    observation = replace(
        _valid_observation(),
        v2_model_runner_enabled=True,
        full_prefill_fallback_available=False,
    )

    result = validator.validate(observation)

    assert result.mode is RuntimeMode.REJECTED
    assert [issue.code for issue in result.issues] == [
        ValidationFailureCode.V2_MODEL_RUNNER_ENABLED,
        ValidationFailureCode.FULL_PREFILL_FALLBACK_UNAVAILABLE,
    ]


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    [
        (
            "tensor_parallel_size",
            True,
            ValidationFailureCode.TENSOR_PARALLEL_SIZE_MISMATCH,
        ),
        (
            "num_hidden_layers",
            True,
            ValidationFailureCode.LAYER_COUNT_MISMATCH,
        ),
        (
            "hybrid_kv_manager_enabled",
            1,
            ValidationFailureCode.HYBRID_KV_MANAGER_DISABLED,
        ),
    ],
)
def test_bool_integer_coercion_cannot_pass_runtime_validation(
    field: str,
    value: object,
    expected_code: ValidationFailureCode,
) -> None:
    result = RuntimeValidator().validate(
        replace(_valid_observation(), **{field: value})
    )

    assert result.mode is RuntimeMode.REJECTED
    assert expected_code in {issue.code for issue in result.issues}


def test_truthy_non_boolean_fallback_flag_cannot_enable_fallback() -> None:
    validator = RuntimeValidator(
        policy=RuntimeValidationPolicy(
            mismatch_action=MismatchAction.FALL_BACK_TO_FULL_PREFILL
        )
    )
    observation = replace(
        _valid_observation(),
        attention_backend="FLASH_ATTN",
        full_prefill_fallback_available=1,  # type: ignore[arg-type]
    )

    result = validator.validate(observation)

    assert result.mode is RuntimeMode.REJECTED
    assert ValidationFailureCode.FULL_PREFILL_FALLBACK_UNAVAILABLE in {
        issue.code for issue in result.issues
    }


def test_observations_are_immutable() -> None:
    observation = _valid_observation()

    with pytest.raises(FrozenInstanceError):
        observation.vllm_version = "0.20.0"  # type: ignore[misc]
