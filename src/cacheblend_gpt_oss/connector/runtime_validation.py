"""Fail-closed validation of the sole supported runtime envelope.

This module deliberately consumes plain immutable observations.  Collection of
those observations belongs in the version-scoped vLLM compatibility package,
so importing or testing this module never imports vLLM, LMCache, Torch, or CUDA.

The GPT-OSS attention layout and sink behavior are taken from the pinned vLLM
source:
https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/models/gpt_oss.py#L67-L153
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from cacheblend_gpt_oss.targets import PINNED_TARGET


class AttentionLayerKind(str, Enum):
    """The two attention kinds present in the pinned GPT-OSS model."""

    SLIDING = "sliding"
    FULL = "full"


GPT_OSS_ATTENTION_PATTERN = tuple(
    AttentionLayerKind.SLIDING if layer_index % 2 == 0 else AttentionLayerKind.FULL
    for layer_index in range(24)
)


class UnsupportedFeature(str, Enum):
    """Bounded names for execution features not audited by this prototype."""

    SPECULATIVE_DECODING = "speculative_decoding"
    PIPELINE_PARALLELISM = "pipeline_parallelism"
    LORA = "lora"
    DUAL_BATCH_OVERLAP = "dual_batch_overlap"
    KV_CACHE_OFFLOAD = "kv_cache_offload"
    KV_CACHE_COMPRESSION = "kv_cache_compression"
    CUSTOM_SCHEDULER = "custom_scheduler"
    OTHER_UNAUDITED = "other_unaudited"


class ValidationFailureCode(str, Enum):
    """Stable, bounded codes suitable for logs and metric labels."""

    MODEL_ID_MISMATCH = "model_id_mismatch"
    VLLM_VERSION_MISMATCH = "vllm_version_mismatch"
    LMCACHE_VERSION_MISMATCH = "lmcache_version_mismatch"
    TORCH_VERSION_MISMATCH = "torch_version_mismatch"
    CUDA_RUNTIME_MISMATCH = "cuda_runtime_mismatch"
    GPU_NAME_MISMATCH = "gpu_name_mismatch"
    GPU_COMPUTE_CAPABILITY_MISMATCH = "gpu_compute_capability_mismatch"
    TENSOR_PARALLEL_SIZE_MISMATCH = "tensor_parallel_size_mismatch"
    PIPELINE_PARALLEL_SIZE_MISMATCH = "pipeline_parallel_size_mismatch"
    LAYER_COUNT_MISMATCH = "layer_count_mismatch"
    ATTENTION_PATTERN_MISMATCH = "attention_pattern_mismatch"
    SLIDING_WINDOW_MISMATCH = "sliding_window_mismatch"
    HYBRID_KV_MANAGER_DISABLED = "hybrid_kv_manager_disabled"
    V2_MODEL_RUNNER_ENABLED = "v2_model_runner_enabled"
    ATTENTION_BACKEND_MISMATCH = "attention_backend_mismatch"
    ATTENTION_SINKS_UNSUPPORTED = "attention_sinks_unsupported"
    UNSUPPORTED_FEATURE_ENABLED = "unsupported_feature_enabled"
    FULL_PREFILL_FALLBACK_UNAVAILABLE = "full_prefill_fallback_unavailable"


class MismatchAction(str, Enum):
    """Configured response to an incompatible runtime."""

    FALL_BACK_TO_FULL_PREFILL = "fall_back_to_full_prefill"
    REJECT_STARTUP = "reject_startup"


class RuntimeMode(str, Enum):
    """The only modes the caller may enter after validation."""

    CACHEBLEND = "cacheblend"
    FULL_PREFILL = "full_prefill"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class RuntimeExpectations:
    """Expected values for the single audited deployment target."""

    model_id: str = PINNED_TARGET.model_id
    vllm_version: str = PINNED_TARGET.vllm_version
    lmcache_version: str = PINNED_TARGET.lmcache_version
    torch_version: str = PINNED_TARGET.torch_version
    cuda_runtime: str = PINNED_TARGET.cuda_runtime
    gpu_name: str = PINNED_TARGET.gpu_name
    gpu_compute_capability: str = "8.0"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    num_hidden_layers: int = 24
    attention_pattern: tuple[AttentionLayerKind, ...] = GPT_OSS_ATTENTION_PATTERN
    sliding_window: int = 128
    hybrid_kv_manager_enabled: bool = True
    v2_model_runner_enabled: bool = False
    attention_backend: str = "TRITON_ATTN"
    attention_backend_supports_sinks: bool = True
    unsupported_features: frozenset[UnsupportedFeature] = frozenset()


@dataclass(frozen=True, slots=True)
class RuntimeObservation:
    """Values collected by a dependency-injected, version-specific probe."""

    model_id: str
    vllm_version: str
    lmcache_version: str
    torch_version: str
    cuda_runtime: str
    gpu_name: str
    gpu_compute_capability: str
    tensor_parallel_size: int
    pipeline_parallel_size: int
    num_hidden_layers: int
    attention_pattern: tuple[AttentionLayerKind, ...]
    sliding_window: int
    hybrid_kv_manager_enabled: bool
    v2_model_runner_enabled: bool
    attention_backend: str
    attention_backend_supports_sinks: bool
    unsupported_features: frozenset[UnsupportedFeature]
    full_prefill_fallback_available: bool


@dataclass(frozen=True, slots=True)
class RuntimeValidationPolicy:
    """Explicitly selects rejection or a reuse-disabled fallback."""

    mismatch_action: MismatchAction = MismatchAction.REJECT_STARTUP


@dataclass(frozen=True, slots=True)
class RuntimeValidationIssue:
    """One compatibility failure with a bounded code and diagnostic values."""

    code: ValidationFailureCode
    expected: str
    observed: str


@dataclass(frozen=True, slots=True)
class RuntimeValidationResult:
    """A complete validation result that the connector must honor."""

    mode: RuntimeMode
    issues: tuple[RuntimeValidationIssue, ...]

    @property
    def cacheblend_enabled(self) -> bool:
        """Return whether KV reuse is safe to enable."""

        return self.mode is RuntimeMode.CACHEBLEND

    @property
    def requires_full_prefill(self) -> bool:
        """Return whether all prompt tokens must follow ordinary prefill."""

        return self.mode is RuntimeMode.FULL_PREFILL


def _display(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, tuple | frozenset):
        values = sorted(
            item.value if isinstance(item, Enum) else str(item) for item in value
        )
        return ",".join(values)
    return str(value)


def _strict_equal(expected: object, observed: object) -> bool:
    """Compare runtime fields without Python's bool/int equality trap."""

    if isinstance(expected, bool):
        return isinstance(observed, bool) and observed is expected
    if isinstance(expected, int):
        return (
            isinstance(observed, int)
            and not isinstance(observed, bool)
            and observed == expected
        )
    if isinstance(expected, tuple):
        return isinstance(observed, tuple) and observed == expected
    if isinstance(expected, frozenset):
        return isinstance(observed, frozenset) and observed == expected
    return type(observed) is type(expected) and observed == expected


@dataclass(frozen=True, slots=True)
class RuntimeValidator:
    """Pure validator with injected expectations and mismatch policy."""

    expectations: RuntimeExpectations = RuntimeExpectations()
    policy: RuntimeValidationPolicy = RuntimeValidationPolicy()

    def validate(self, observation: RuntimeObservation) -> RuntimeValidationResult:
        """Validate every audited condition and choose a fail-closed mode."""

        expected = self.expectations
        issues: list[RuntimeValidationIssue] = []

        def compare(
            code: ValidationFailureCode,
            expected_value: object,
            observed_value: object,
        ) -> None:
            if not _strict_equal(expected_value, observed_value):
                issues.append(
                    RuntimeValidationIssue(
                        code=code,
                        expected=_display(expected_value),
                        observed=_display(observed_value),
                    )
                )

        compare(
            ValidationFailureCode.MODEL_ID_MISMATCH,
            expected.model_id,
            observation.model_id,
        )
        compare(
            ValidationFailureCode.VLLM_VERSION_MISMATCH,
            expected.vllm_version,
            observation.vllm_version,
        )
        compare(
            ValidationFailureCode.LMCACHE_VERSION_MISMATCH,
            expected.lmcache_version,
            observation.lmcache_version,
        )
        compare(
            ValidationFailureCode.TORCH_VERSION_MISMATCH,
            expected.torch_version,
            observation.torch_version,
        )
        compare(
            ValidationFailureCode.CUDA_RUNTIME_MISMATCH,
            expected.cuda_runtime,
            observation.cuda_runtime,
        )
        compare(
            ValidationFailureCode.GPU_NAME_MISMATCH,
            expected.gpu_name,
            observation.gpu_name,
        )
        compare(
            ValidationFailureCode.GPU_COMPUTE_CAPABILITY_MISMATCH,
            expected.gpu_compute_capability,
            observation.gpu_compute_capability,
        )
        compare(
            ValidationFailureCode.TENSOR_PARALLEL_SIZE_MISMATCH,
            expected.tensor_parallel_size,
            observation.tensor_parallel_size,
        )
        compare(
            ValidationFailureCode.PIPELINE_PARALLEL_SIZE_MISMATCH,
            expected.pipeline_parallel_size,
            observation.pipeline_parallel_size,
        )
        compare(
            ValidationFailureCode.LAYER_COUNT_MISMATCH,
            expected.num_hidden_layers,
            observation.num_hidden_layers,
        )
        compare(
            ValidationFailureCode.ATTENTION_PATTERN_MISMATCH,
            expected.attention_pattern,
            observation.attention_pattern,
        )
        compare(
            ValidationFailureCode.SLIDING_WINDOW_MISMATCH,
            expected.sliding_window,
            observation.sliding_window,
        )
        compare(
            ValidationFailureCode.HYBRID_KV_MANAGER_DISABLED,
            expected.hybrid_kv_manager_enabled,
            observation.hybrid_kv_manager_enabled,
        )
        compare(
            ValidationFailureCode.V2_MODEL_RUNNER_ENABLED,
            expected.v2_model_runner_enabled,
            observation.v2_model_runner_enabled,
        )
        compare(
            ValidationFailureCode.ATTENTION_BACKEND_MISMATCH,
            expected.attention_backend,
            observation.attention_backend,
        )
        compare(
            ValidationFailureCode.ATTENTION_SINKS_UNSUPPORTED,
            expected.attention_backend_supports_sinks,
            observation.attention_backend_supports_sinks,
        )
        compare(
            ValidationFailureCode.UNSUPPORTED_FEATURE_ENABLED,
            expected.unsupported_features,
            observation.unsupported_features,
        )

        if not issues:
            return RuntimeValidationResult(mode=RuntimeMode.CACHEBLEND, issues=())

        if (
            self.policy.mismatch_action is MismatchAction.FALL_BACK_TO_FULL_PREFILL
            and observation.full_prefill_fallback_available is True
        ):
            return RuntimeValidationResult(
                mode=RuntimeMode.FULL_PREFILL,
                issues=tuple(issues),
            )

        if (
            self.policy.mismatch_action is MismatchAction.FALL_BACK_TO_FULL_PREFILL
            and observation.full_prefill_fallback_available is not True
        ):
            issues.append(
                RuntimeValidationIssue(
                    code=ValidationFailureCode.FULL_PREFILL_FALLBACK_UNAVAILABLE,
                    expected="true",
                    observed="false",
                )
            )

        return RuntimeValidationResult(mode=RuntimeMode.REJECTED, issues=tuple(issues))
