# SPDX-License-Identifier: Apache-2.0
"""Pinned benchmark evidence models for GPT-OSS CacheBlend experiments.

This module describes *evidence*, not execution.  A worker on ``solab-g3`` can
write one artifact containing raw trial counters and timings for a controlled
arm.  The authoring workstation can then validate identity, reconcile work,
and calculate confidence intervals without importing vLLM, LMCache, Torch, or
CUDA.  A report is not benchmark-ready until every recorded trial has an
independent passing correctness artifact.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from math import sqrt
from statistics import median, stdev
from typing import NoReturn

from cacheblend_gpt_oss.correctness.models import (
    CorrectnessCase,
    CorrectnessRuntimeIdentity,
)
from cacheblend_gpt_oss.metrics.request import (
    RequestMetrics,
    validate_request_metrics,
)

BENCHMARK_SCHEMA_VERSION = 1
BENCHMARK_ATTENTION_BACKEND = "TRITON_ATTN"
BENCHMARK_BLOCK_SIZE = 16
BENCHMARK_MAX_MODEL_LEN = 131_072
BENCHMARK_TENSOR_PARALLEL_SIZE = 1
BENCHMARK_PIPELINE_PARALLEL_SIZE = 1
_HEX_64 = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[A-Za-z0-9._-]{1,256}")


class BenchmarkArm(str, Enum):
    """Isolated serving arm from the M9 comparison matrix."""

    FULL_PREFILL = "full_prefill"
    VLLM_PREFIX_EXACT = "vllm_prefix_exact"
    VLLM_PREFIX_MOVED = "vllm_prefix_moved"
    CACHEBLEND_100PCT = "cacheblend_100pct"
    CACHEBLEND_SELECTIVE = "cacheblend_selective"
    PREFIX_PLUS_CACHEBLEND = "prefix_plus_cacheblend"


class BenchmarkCacheState(str, Enum):
    """Whether reusable cache content was cold or warmed before the trial."""

    COLD = "cold"
    WARM = "warm"


class BenchmarkFailureCode(str, Enum):
    """Bounded failure reasons; free-form server text is never persisted."""

    CONNECTOR_FALLBACK = "connector_fallback"
    CORRECTNESS_FAILED = "correctness_failed"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    UNSUPPORTED_CONFIGURATION = "unsupported_configuration"


class BenchmarkErrorCode(str, Enum):
    """Bounded validation failures for benchmark artifacts."""

    INVALID_SCHEMA = "invalid_schema"
    INVALID_RUNTIME = "invalid_runtime"
    INVALID_ARM = "invalid_arm"
    INVALID_CASE = "invalid_case"
    INVALID_CACHE_STATE = "invalid_cache_state"
    INVALID_TRIAL_INDEX = "invalid_trial_index"
    INVALID_PROMPT_TOKENS = "invalid_prompt_tokens"
    INVALID_RATIO = "invalid_ratio"
    INVALID_IDENTITY = "invalid_identity"
    INVALID_METRICS = "invalid_metrics"
    INVALID_DIGEST = "invalid_digest"
    INVALID_FAILURE = "invalid_failure"
    ARM_CASE_MISMATCH = "arm_case_mismatch"
    ARM_METRIC_MISMATCH = "arm_metric_mismatch"
    CORRECTNESS_MISSING = "correctness_missing"
    TRANSFER_EVIDENCE_MISSING = "transfer_evidence_missing"
    DUPLICATE_TRIAL = "duplicate_trial"
    INCONSISTENT_TRIAL = "inconsistent_trial"
    EMPTY_TRIALS = "empty_trials"
    FILE_EXISTS = "file_exists"
    FILE_ERROR = "file_error"
    INVALID_JSON = "invalid_json"


class BenchmarkError(ValueError):
    """Fail-closed benchmark error without prompt or request identifiers."""

    def __init__(self, code: BenchmarkErrorCode) -> None:
        self.code = code
        super().__init__(f"CacheBlend benchmark evidence failure: {code.value}")


def _fail(code: BenchmarkErrorCode) -> NoReturn:
    raise BenchmarkError(code)


def _finite_nonnegative(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _require_id(value: object, code: BenchmarkErrorCode) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        _fail(code)
    return value


def _validate_arm_case(arm: BenchmarkArm, case: CorrectnessCase) -> None:
    expected: dict[BenchmarkArm, tuple[CorrectnessCase, ...]] = {
        BenchmarkArm.VLLM_PREFIX_EXACT: (CorrectnessCase.EXACT_PREFIX,),
        BenchmarkArm.VLLM_PREFIX_MOVED: (CorrectnessCase.MOVED_DOCUMENT,),
        BenchmarkArm.CACHEBLEND_SELECTIVE: (
            CorrectnessCase.MOVED_DOCUMENT,
            CorrectnessCase.REORDERED_DOCUMENTS,
        ),
        BenchmarkArm.PREFIX_PLUS_CACHEBLEND: (
            CorrectnessCase.EXACT_PREFIX,
            CorrectnessCase.MOVED_DOCUMENT,
            CorrectnessCase.REORDERED_DOCUMENTS,
        ),
    }
    allowed = expected.get(arm)
    if allowed is not None and case not in allowed:
        _fail(BenchmarkErrorCode.ARM_CASE_MISMATCH)


@dataclass(frozen=True, slots=True)
class BenchmarkTrial:
    """One raw, identifier-free trial from one benchmark arm."""

    arm: BenchmarkArm
    case: CorrectnessCase
    cache_state: BenchmarkCacheState
    trial_index: int
    metrics: RequestMetrics
    recompute_ratio: float | None
    peak_memory_bytes: int
    correctness_passed: bool
    correctness_artifact_digest: str | None = None
    failure: BenchmarkFailureCode | None = None
    staging_overhead_bytes: int = 0
    transfer_evidence_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.arm, BenchmarkArm):
            _fail(BenchmarkErrorCode.INVALID_ARM)
        if not isinstance(self.case, CorrectnessCase):
            _fail(BenchmarkErrorCode.INVALID_CASE)
        if not isinstance(self.cache_state, BenchmarkCacheState):
            _fail(BenchmarkErrorCode.INVALID_CACHE_STATE)
        _validate_arm_case(self.arm, self.case)
        if (
            isinstance(self.trial_index, bool)
            or not isinstance(self.trial_index, int)
            or self.trial_index < 1
        ):
            _fail(BenchmarkErrorCode.INVALID_TRIAL_INDEX)
        if not isinstance(self.metrics, RequestMetrics):
            _fail(BenchmarkErrorCode.INVALID_METRICS)
        if validate_request_metrics(self.metrics):
            _fail(BenchmarkErrorCode.INVALID_METRICS)
        if self.metrics.counters.prompt_tokens <= 0:
            _fail(BenchmarkErrorCode.INVALID_PROMPT_TOKENS)
        if self.recompute_ratio is not None and (
            isinstance(self.recompute_ratio, bool)
            or not isinstance(self.recompute_ratio, int | float)
            or not math.isfinite(float(self.recompute_ratio))
            or not 0.0 <= float(self.recompute_ratio) <= 1.0
        ):
            _fail(BenchmarkErrorCode.INVALID_RATIO)
        if self.arm is BenchmarkArm.CACHEBLEND_100PCT:
            if self.recompute_ratio != 1.0 or not (
                self.metrics.counters.is_full_recomputation
            ):
                _fail(BenchmarkErrorCode.ARM_METRIC_MISMATCH)
        elif self.arm is BenchmarkArm.CACHEBLEND_SELECTIVE:
            if self.recompute_ratio is None or self.recompute_ratio >= 1.0:
                _fail(BenchmarkErrorCode.ARM_METRIC_MISMATCH)
        elif self.recompute_ratio is not None:
            _fail(BenchmarkErrorCode.ARM_METRIC_MISMATCH)
        if (
            isinstance(self.peak_memory_bytes, bool)
            or not isinstance(self.peak_memory_bytes, int)
            or self.peak_memory_bytes < 0
        ):
            _fail(BenchmarkErrorCode.INVALID_METRICS)
        if not isinstance(self.correctness_passed, bool):
            _fail(BenchmarkErrorCode.CORRECTNESS_MISSING)
        if self.correctness_artifact_digest is not None and (
            not isinstance(self.correctness_artifact_digest, str)
            or _HEX_64.fullmatch(self.correctness_artifact_digest) is None
        ):
            _fail(BenchmarkErrorCode.INVALID_DIGEST)
        if self.correctness_passed and self.correctness_artifact_digest is None:
            _fail(BenchmarkErrorCode.CORRECTNESS_MISSING)
        if self.transfer_evidence_digest is not None and (
            not isinstance(self.transfer_evidence_digest, str)
            or _HEX_64.fullmatch(self.transfer_evidence_digest) is None
        ):
            _fail(BenchmarkErrorCode.INVALID_DIGEST)
        if (
            self.arm
            in {BenchmarkArm.CACHEBLEND_100PCT, BenchmarkArm.CACHEBLEND_SELECTIVE}
            and self.correctness_passed
            and self.transfer_evidence_digest is None
        ):
            _fail(BenchmarkErrorCode.TRANSFER_EVIDENCE_MISSING)
        if self.failure is not None and not isinstance(
            self.failure, BenchmarkFailureCode
        ):
            _fail(BenchmarkErrorCode.INVALID_FAILURE)
        if self.correctness_passed and self.failure is not None:
            _fail(BenchmarkErrorCode.CORRECTNESS_MISSING)
        if (
            isinstance(self.staging_overhead_bytes, bool)
            or not isinstance(self.staging_overhead_bytes, int)
            or self.staging_overhead_bytes < 0
        ):
            _fail(BenchmarkErrorCode.INVALID_METRICS)


@dataclass(frozen=True, slots=True)
class BenchmarkArtifact:
    """One controlled comparison case and all its repeated arm trials."""

    schema_version: int
    runtime: CorrectnessRuntimeIdentity
    case: CorrectnessCase
    prompt_tokens: int
    prompt_fixture_digest: str
    host_id: str
    attention_backend: str
    hybrid_kv_cache_enabled: bool
    block_size: int
    max_model_len: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    sampling_seed: int
    temperature: float
    top_p: float
    trials: tuple[BenchmarkTrial, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != BENCHMARK_SCHEMA_VERSION
        ):
            _fail(BenchmarkErrorCode.INVALID_SCHEMA)
        if not isinstance(self.runtime, CorrectnessRuntimeIdentity):
            _fail(BenchmarkErrorCode.INVALID_RUNTIME)
        if not isinstance(self.case, CorrectnessCase):
            _fail(BenchmarkErrorCode.INVALID_CASE)
        if (
            isinstance(self.prompt_tokens, bool)
            or not isinstance(self.prompt_tokens, int)
            or self.prompt_tokens <= 0
            or self.prompt_tokens > BENCHMARK_MAX_MODEL_LEN
        ):
            _fail(BenchmarkErrorCode.INVALID_PROMPT_TOKENS)
        if (
            not isinstance(self.prompt_fixture_digest, str)
            or _HEX_64.fullmatch(self.prompt_fixture_digest) is None
        ):
            _fail(BenchmarkErrorCode.INVALID_DIGEST)
        _require_id(self.host_id, BenchmarkErrorCode.INVALID_IDENTITY)
        if self.attention_backend != BENCHMARK_ATTENTION_BACKEND:
            _fail(BenchmarkErrorCode.INVALID_IDENTITY)
        if self.hybrid_kv_cache_enabled is not True:
            _fail(BenchmarkErrorCode.INVALID_IDENTITY)
        if self.block_size != BENCHMARK_BLOCK_SIZE:
            _fail(BenchmarkErrorCode.INVALID_IDENTITY)
        if self.max_model_len != BENCHMARK_MAX_MODEL_LEN:
            _fail(BenchmarkErrorCode.INVALID_IDENTITY)
        if self.tensor_parallel_size != BENCHMARK_TENSOR_PARALLEL_SIZE:
            _fail(BenchmarkErrorCode.INVALID_IDENTITY)
        if self.pipeline_parallel_size != BENCHMARK_PIPELINE_PARALLEL_SIZE:
            _fail(BenchmarkErrorCode.INVALID_IDENTITY)
        if (
            isinstance(self.sampling_seed, bool)
            or not isinstance(self.sampling_seed, int)
            or self.sampling_seed != 0
        ):
            _fail(BenchmarkErrorCode.INVALID_IDENTITY)
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, int | float)
            or not math.isfinite(float(self.temperature))
            or isinstance(self.top_p, bool)
            or not isinstance(self.top_p, int | float)
            or not math.isfinite(float(self.top_p))
            or self.temperature != 0.0
            or self.top_p != 1.0
        ):
            _fail(BenchmarkErrorCode.INVALID_IDENTITY)
        try:
            trials = tuple(self.trials)
        except TypeError:
            _fail(BenchmarkErrorCode.EMPTY_TRIALS)
        if not trials or any(not isinstance(trial, BenchmarkTrial) for trial in trials):
            _fail(BenchmarkErrorCode.EMPTY_TRIALS)
        seen: set[tuple[BenchmarkArm, BenchmarkCacheState, int]] = set()
        for trial in trials:
            if trial.case is not self.case:
                _fail(BenchmarkErrorCode.INCONSISTENT_TRIAL)
            if trial.metrics.counters.prompt_tokens != self.prompt_tokens:
                _fail(BenchmarkErrorCode.INCONSISTENT_TRIAL)
            key = (trial.arm, trial.cache_state, trial.trial_index)
            if key in seen:
                _fail(BenchmarkErrorCode.DUPLICATE_TRIAL)
            seen.add(key)
        object.__setattr__(self, "trials", trials)

    @property
    def arms(self) -> tuple[BenchmarkArm, ...]:
        """Stable arm order, independent of insertion order."""

        return tuple(
            arm
            for arm in BenchmarkArm
            if any(trial.arm is arm for trial in self.trials)
        )

    @property
    def missing_required_arms(self) -> tuple[BenchmarkArm, ...]:
        required = (BenchmarkArm.FULL_PREFILL, BenchmarkArm.CACHEBLEND_100PCT)
        return tuple(arm for arm in required if arm not in self.arms)

    @property
    def benchmark_ready(self) -> bool:
        """Whether all recorded trials have the evidence needed for comparison."""

        def complete_latency(trial: BenchmarkTrial) -> bool:
            timers = trial.metrics.timers
            return all(
                value is not None
                for value in (
                    timers.ttft_seconds,
                    timers.queue_latency_seconds,
                    timers.prefill_latency_seconds,
                    timers.decode_latency_seconds,
                    timers.end_to_end_latency_seconds,
                )
            )

        return not self.missing_required_arms and all(
            trial.correctness_passed and complete_latency(trial)
            for trial in self.trials
        )


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """Mean, median, and normal-approximation 95% interval for one metric."""

    count: int
    mean: float
    median: float
    ci95_low: float
    ci95_high: float


@dataclass(frozen=True, slots=True)
class BenchmarkArmSummary:
    """Derived repeated-trial summary; raw trials remain authoritative."""

    arm: BenchmarkArm
    trial_count: int
    correctness_passed: bool
    ttft_seconds: ConfidenceInterval | None
    queue_latency_seconds: ConfidenceInterval | None
    decode_latency_seconds: ConfidenceInterval | None
    prefill_latency_seconds: ConfidenceInterval
    end_to_end_latency_seconds: ConfidenceInterval | None
    recomputed_tokens: ConfidenceInterval
    saved_prefill_fraction: ConfidenceInterval
    peak_memory_bytes: ConfidenceInterval
    staging_overhead_bytes: ConfidenceInterval


def _confidence(values: list[float]) -> ConfidenceInterval:
    if not values or any(not _finite_nonnegative(value) for value in values):
        _fail(BenchmarkErrorCode.INVALID_METRICS)
    count = len(values)
    mean = sum(values) / count
    spread = stdev(values) if count > 1 else 0.0
    margin = 1.96 * spread / sqrt(count)
    return ConfidenceInterval(
        count=count,
        mean=mean,
        median=float(median(values)),
        ci95_low=max(0.0, mean - margin),
        ci95_high=mean + margin,
    )


def _optional_confidence(values: list[float | None]) -> ConfidenceInterval | None:
    """Summarize an optional metric only when every trial supplied it."""

    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        _fail(BenchmarkErrorCode.INVALID_METRICS)
    return _confidence([float(value) for value in values if value is not None])


def summarize_benchmark(artifact: BenchmarkArtifact) -> tuple[BenchmarkArmSummary, ...]:
    """Calculate deterministic per-arm summaries from raw trial evidence."""

    if not isinstance(artifact, BenchmarkArtifact):
        _fail(BenchmarkErrorCode.INVALID_SCHEMA)
    summaries: list[BenchmarkArmSummary] = []
    for arm in artifact.arms:
        trials = [trial for trial in artifact.trials if trial.arm is arm]
        summaries.append(
            BenchmarkArmSummary(
                arm=arm,
                trial_count=len(trials),
                correctness_passed=all(
                    trial.correctness_passed for trial in trials
                ),
                ttft_seconds=_optional_confidence(
                    [
                        None
                        if trial.metrics.timers.ttft_seconds is None
                        else float(trial.metrics.timers.ttft_seconds)
                        for trial in trials
                    ]
                ),
                queue_latency_seconds=_optional_confidence(
                    [
                        None
                        if trial.metrics.timers.queue_latency_seconds is None
                        else float(trial.metrics.timers.queue_latency_seconds)
                        for trial in trials
                    ]
                ),
                decode_latency_seconds=_optional_confidence(
                    [
                        None
                        if trial.metrics.timers.decode_latency_seconds is None
                        else float(trial.metrics.timers.decode_latency_seconds)
                        for trial in trials
                    ]
                ),
                prefill_latency_seconds=_confidence(
                    [
                        float(trial.metrics.timers.prefill_latency_seconds)
                        for trial in trials
                    ]
                ),
                end_to_end_latency_seconds=_optional_confidence(
                    [
                        None
                        if trial.metrics.timers.end_to_end_latency_seconds is None
                        else float(trial.metrics.timers.end_to_end_latency_seconds)
                        for trial in trials
                    ]
                ),
                recomputed_tokens=_confidence(
                    [
                        float(trial.metrics.counters.tokens_recomputed)
                        for trial in trials
                    ]
                ),
                saved_prefill_fraction=_confidence(
                    [
                        float(
                            trial.metrics.counters.effective_saved_prefill_fraction
                        )
                        for trial in trials
                    ]
                ),
                peak_memory_bytes=_confidence(
                    [float(trial.peak_memory_bytes) for trial in trials]
                ),
                staging_overhead_bytes=_confidence(
                    [float(trial.staging_overhead_bytes) for trial in trials]
                ),
            )
        )
    return tuple(summaries)


__all__ = [
    "BENCHMARK_ATTENTION_BACKEND",
    "BENCHMARK_BLOCK_SIZE",
    "BENCHMARK_MAX_MODEL_LEN",
    "BENCHMARK_PIPELINE_PARALLEL_SIZE",
    "BENCHMARK_SCHEMA_VERSION",
    "BENCHMARK_TENSOR_PARALLEL_SIZE",
    "BenchmarkArm",
    "BenchmarkArmSummary",
    "BenchmarkArtifact",
    "BenchmarkCacheState",
    "BenchmarkError",
    "BenchmarkErrorCode",
    "BenchmarkFailureCode",
    "BenchmarkTrial",
    "ConfidenceInterval",
    "summarize_benchmark",
]
