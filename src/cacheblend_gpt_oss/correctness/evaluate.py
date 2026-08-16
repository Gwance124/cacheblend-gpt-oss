# SPDX-License-Identifier: Apache-2.0
"""Freeze baseline error envelopes before evaluating CacheBlend output."""

from __future__ import annotations

import math
from dataclasses import dataclass

from cacheblend_gpt_oss.correctness.io import artifact_digest
from cacheblend_gpt_oss.correctness.models import (
    CorrectnessArtifact,
    CorrectnessRunMode,
    FullVocabularyLogprobs,
)


@dataclass(frozen=True, slots=True)
class DistributionComparison:
    max_abs_error: float
    mean_abs_error: float
    max_relative_error: float
    mean_relative_error: float
    compared_values: int
    negative_infinity_values: int
    sampled_token_agreement: bool
    top_token_agreement: bool


@dataclass(frozen=True, slots=True)
class FrozenFullPrefillTolerance:
    """Tolerance derived and persisted before a CacheBlend artifact is judged."""

    reference_artifact_digest: str
    repeat_artifact_digest: str
    baseline_max_abs_error: float
    baseline_mean_abs_error: float
    allowed_max_abs_error: float
    allowed_mean_abs_error: float

    def __post_init__(self) -> None:
        for digest in (
            self.reference_artifact_digest,
            self.repeat_artifact_digest,
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("invalid frozen tolerance artifact digest")
        values = (
            self.baseline_max_abs_error,
            self.baseline_mean_abs_error,
            self.allowed_max_abs_error,
            self.allowed_mean_abs_error,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("invalid frozen numerical tolerance")
        if (
            self.allowed_max_abs_error < self.baseline_max_abs_error
            or self.allowed_mean_abs_error < self.baseline_mean_abs_error
        ):
            raise ValueError("allowed tolerance is tighter than its baseline")


@dataclass(frozen=True, slots=True)
class CacheBlendCorrectnessVerdict:
    comparison: DistributionComparison
    passed: bool
    failure_reasons: tuple[str, ...]


def compare_distributions(
    left: FullVocabularyLogprobs,
    right: FullVocabularyLogprobs,
) -> DistributionComparison:
    """Compare every vocabulary entry; top-k approximations are not accepted."""

    if len(left.values) != len(right.values):
        raise ValueError("full-vocabulary distributions have different sizes")
    absolute_errors: list[float] = []
    relative_errors: list[float] = []
    negative_infinity_values = 0
    for left_value, right_value in zip(left.values, right.values, strict=True):
        if left_value == -math.inf or right_value == -math.inf:
            if left_value != right_value:
                return DistributionComparison(
                    max_abs_error=math.inf,
                    mean_abs_error=math.inf,
                    max_relative_error=math.inf,
                    mean_relative_error=math.inf,
                    compared_values=len(absolute_errors),
                    negative_infinity_values=negative_infinity_values,
                    sampled_token_agreement=(
                        left.sampled_token_id == right.sampled_token_id
                    ),
                    top_token_agreement=(left.top_token_id == right.top_token_id),
                )
            negative_infinity_values += 1
            continue
        difference = abs(float(left_value) - float(right_value))
        absolute_errors.append(difference)
        relative_errors.append(
            difference / max(abs(float(left_value)), abs(float(right_value)), 1e-12)
        )
    if not absolute_errors:
        raise ValueError("distributions contain no finite comparable values")
    return DistributionComparison(
        max_abs_error=max(absolute_errors),
        mean_abs_error=sum(absolute_errors) / len(absolute_errors),
        max_relative_error=max(relative_errors),
        mean_relative_error=sum(relative_errors) / len(relative_errors),
        compared_values=len(absolute_errors),
        negative_infinity_values=negative_infinity_values,
        sampled_token_agreement=(left.sampled_token_id == right.sampled_token_id),
        top_token_agreement=(left.top_token_id == right.top_token_id),
    )


def _require_same_case(left: CorrectnessArtifact, right: CorrectnessArtifact) -> None:
    if left.runtime != right.runtime or left.prompt != right.prompt:
        raise ValueError("correctness artifacts have incompatible identities")


def freeze_full_prefill_tolerance(
    reference: CorrectnessArtifact,
    repeat: CorrectnessArtifact,
    *,
    max_abs_floor: float,
    mean_abs_floor: float,
    multiplier: float = 1.0,
) -> FrozenFullPrefillTolerance:
    """Freeze an envelope from two baselines before CacheBlend is evaluated."""

    if (
        reference.run_mode is not CorrectnessRunMode.FULL_PREFILL
        or repeat.run_mode is not CorrectnessRunMode.FULL_PREFILL
    ):
        raise ValueError("tolerance inputs must both be full-prefill artifacts")
    _require_same_case(reference, repeat)
    parameters = (max_abs_floor, mean_abs_floor, multiplier)
    if (
        any(not math.isfinite(value) or value < 0 for value in parameters)
        or multiplier < 1
    ):
        raise ValueError("invalid tolerance freeze parameters")
    comparison = compare_distributions(reference.distribution, repeat.distribution)
    if not comparison.sampled_token_agreement or not comparison.top_token_agreement:
        raise ValueError("repeated full prefill changed the selected token")
    return FrozenFullPrefillTolerance(
        reference_artifact_digest=artifact_digest(reference),
        repeat_artifact_digest=artifact_digest(repeat),
        baseline_max_abs_error=comparison.max_abs_error,
        baseline_mean_abs_error=comparison.mean_abs_error,
        allowed_max_abs_error=max(
            max_abs_floor, comparison.max_abs_error * multiplier
        ),
        allowed_mean_abs_error=max(
            mean_abs_floor, comparison.mean_abs_error * multiplier
        ),
    )


def evaluate_cacheblend_100pct(
    reference: CorrectnessArtifact,
    cacheblend: CorrectnessArtifact,
    tolerance: FrozenFullPrefillTolerance,
) -> CacheBlendCorrectnessVerdict:
    """Evaluate numerical equivalence only after transfer evidence validated."""

    if reference.run_mode is not CorrectnessRunMode.FULL_PREFILL:
        raise ValueError("reference must be a full-prefill artifact")
    if cacheblend.run_mode is not CorrectnessRunMode.CACHEBLEND_100PCT:
        raise ValueError("candidate must be a CacheBlend 100% artifact")
    _require_same_case(reference, cacheblend)
    if artifact_digest(reference) != tolerance.reference_artifact_digest:
        raise ValueError("frozen tolerance belongs to another reference artifact")
    comparison = compare_distributions(reference.distribution, cacheblend.distribution)
    reasons: list[str] = []
    if not comparison.sampled_token_agreement:
        reasons.append("sampled_token_mismatch")
    if not comparison.top_token_agreement:
        reasons.append("top_token_mismatch")
    if comparison.max_abs_error > tolerance.allowed_max_abs_error:
        reasons.append("max_abs_error_exceeded")
    if comparison.mean_abs_error > tolerance.allowed_mean_abs_error:
        reasons.append("mean_abs_error_exceeded")
    return CacheBlendCorrectnessVerdict(
        comparison=comparison,
        passed=not reasons,
        failure_reasons=tuple(reasons),
    )


def evaluate_cacheblend_selective(
    reference: CorrectnessArtifact,
    cacheblend: CorrectnessArtifact,
    tolerance: FrozenFullPrefillTolerance,
) -> CacheBlendCorrectnessVerdict:
    """Compare a selective smoke artifact without claiming 100% overwrite."""

    if reference.run_mode is not CorrectnessRunMode.FULL_PREFILL:
        raise ValueError("reference must be a full-prefill artifact")
    if cacheblend.run_mode is not CorrectnessRunMode.CACHEBLEND_SELECTIVE:
        raise ValueError("candidate must be a CacheBlend selective artifact")
    _require_same_case(reference, cacheblend)
    if artifact_digest(reference) != tolerance.reference_artifact_digest:
        raise ValueError("frozen tolerance belongs to another reference artifact")
    comparison = compare_distributions(reference.distribution, cacheblend.distribution)
    reasons: list[str] = []
    if not comparison.sampled_token_agreement:
        reasons.append("sampled_token_mismatch")
    if not comparison.top_token_agreement:
        reasons.append("top_token_mismatch")
    if comparison.max_abs_error > tolerance.allowed_max_abs_error:
        reasons.append("max_abs_error_exceeded")
    if comparison.mean_abs_error > tolerance.allowed_mean_abs_error:
        reasons.append("mean_abs_error_exceeded")
    return CacheBlendCorrectnessVerdict(
        comparison=comparison,
        passed=not reasons,
        failure_reasons=tuple(reasons),
    )


__all__ = [
    "CacheBlendCorrectnessVerdict",
    "DistributionComparison",
    "FrozenFullPrefillTolerance",
    "compare_distributions",
    "evaluate_cacheblend_100pct",
    "evaluate_cacheblend_selective",
    "freeze_full_prefill_tolerance",
]
