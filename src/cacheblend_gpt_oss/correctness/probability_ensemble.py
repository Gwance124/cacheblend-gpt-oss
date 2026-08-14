# SPDX-License-Identifier: Apache-2.0
"""Prospective probability-mass-aware numerical gate.

The strict full-vocabulary maximum remains a diagnostic in this policy.  It
is deliberately not an acceptance metric because a single negligible-mass
BF16 tail coordinate caused the immutable strict-v1 failure.  This policy is
prospective: its constants are code-owned, its five controls are frozen before
the next candidate, and a candidate must pass every probability-aware metric.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Any, cast

from cacheblend_gpt_oss.correctness.io import artifact_digest
from cacheblend_gpt_oss.correctness.models import (
    CorrectnessArtifact,
    CorrectnessRunMode,
    FullVocabularyLogprobs,
)

PROBABILITY_ENSEMBLE_SCHEMA_VERSION = 1
PROBABILITY_ENSEMBLE_POLICY_VERSION = (
    "cacheblend-gpt-oss-probability-mass-v1"
)
PROBABILITY_BASELINE_COUNT = 5
PROBABILITY_PAIR_COUNT = 10
PROBABILITY_TAIL_EPSILON = 1e-4
HARD_FULL_MEAN_ABS_LOGPROB_CEILING = 0.014
HARD_TOTAL_VARIATION_CEILING = 0.02
HARD_JENSEN_SHANNON_CEILING = 0.001
HARD_HIGH_MASS_MAX_ABS_LOGPROB_CEILING = 0.08


class ProbabilityEnsembleStatus(str, Enum):
    """Outcome categories for the prospective probability-aware gate."""

    PASS = "PASS"
    FAIL = "FAIL"
    INDETERMINATE_BASELINE_UNSTABLE = "INDETERMINATE_BASELINE_UNSTABLE"


def _finite_nonnegative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"invalid probability ensemble {name}")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"invalid probability ensemble {name}")
    return converted


def _digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"invalid probability ensemble {name}")
    return value


def _index(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid probability ensemble {name}")
    return value


def _reasons(value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple) or any(
        not isinstance(reason, str) or not reason for reason in value
    ):
        raise ValueError("invalid probability ensemble failure reasons")
    return value


@dataclass(frozen=True, slots=True)
class ProbabilityMetrics:
    """Probability-aware metrics plus the non-gating strict diagnostic."""

    full_mean_abs_logprob: float
    total_variation: float
    jensen_shannon_divergence: float
    high_mass_max_abs_logprob: float
    strict_full_max_abs_logprob: float
    high_mass_support_tokens: int
    sampled_token_agreement: bool
    top_token_agreement: bool

    def __post_init__(self) -> None:
        for name in (
            "full_mean_abs_logprob",
            "total_variation",
            "jensen_shannon_divergence",
            "high_mass_max_abs_logprob",
            "strict_full_max_abs_logprob",
        ):
            _finite_nonnegative(name, getattr(self, name))
        if self.total_variation > 1.0 + 1e-12:
            raise ValueError("probability ensemble total variation exceeds one")
        if self.jensen_shannon_divergence > math.log(2.0) + 1e-12:
            raise ValueError("probability ensemble Jensen-Shannon exceeds log(2)")
        support = _index("high-mass support tokens", self.high_mass_support_tokens)
        if support == 0:
            raise ValueError("probability ensemble high-mass support is empty")
        if not isinstance(self.sampled_token_agreement, bool) or not isinstance(
            self.top_token_agreement, bool
        ):
            raise ValueError("invalid probability ensemble token agreement")


@dataclass(frozen=True, slots=True)
class ProbabilityBaselineComparison:
    """One of the ten probability-aware baseline comparisons."""

    left_index: int
    right_index: int
    left_artifact_digest: str
    right_artifact_digest: str
    metrics: ProbabilityMetrics

    def __post_init__(self) -> None:
        left = _index("left index", self.left_index)
        right = _index("right index", self.right_index)
        if not left < right or right >= PROBABILITY_BASELINE_COUNT:
            raise ValueError("invalid probability baseline pair indices")
        _digest("left artifact digest", self.left_artifact_digest)
        _digest("right artifact digest", self.right_artifact_digest)
        if not isinstance(self.metrics, ProbabilityMetrics):
            raise ValueError("invalid probability baseline metrics")


@dataclass(frozen=True, slots=True)
class ProbabilityCandidateComparison:
    """One probability-aware candidate-to-baseline comparison."""

    baseline_index: int
    baseline_artifact_digest: str
    metrics: ProbabilityMetrics

    def __post_init__(self) -> None:
        index = _index("baseline index", self.baseline_index)
        if index >= PROBABILITY_BASELINE_COUNT:
            raise ValueError("invalid probability candidate baseline index")
        _digest("baseline artifact digest", self.baseline_artifact_digest)
        if not isinstance(self.metrics, ProbabilityMetrics):
            raise ValueError("invalid probability candidate metrics")


@dataclass(frozen=True, slots=True)
class ProbabilityBaselineManifest:
    """Immutable policy and empirical envelopes for five controls."""

    policy_version: str
    artifact_digests: tuple[str, ...]
    excluded_candidate_artifact_digests: tuple[str, ...]
    tail_epsilon: float
    hard_full_mean_abs_logprob_ceiling: float
    hard_total_variation_ceiling: float
    hard_jensen_shannon_ceiling: float
    hard_high_mass_max_abs_logprob_ceiling: float
    u_full_mean_abs_logprob: float
    u_total_variation: float
    u_jensen_shannon_divergence: float
    u_high_mass_max_abs_logprob: float
    diagnostic_u_strict_full_max_abs_logprob: float
    baseline_status: ProbabilityEnsembleStatus
    baseline_failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.policy_version != PROBABILITY_ENSEMBLE_POLICY_VERSION:
            raise ValueError("unsupported probability ensemble policy")
        if len(self.artifact_digests) != PROBABILITY_BASELINE_COUNT:
            raise ValueError("probability manifest must contain five digests")
        digests = tuple(
            _digest("artifact digest", item) for item in self.artifact_digests
        )
        if len(set(digests)) != PROBABILITY_BASELINE_COUNT:
            raise ValueError("probability manifest requires unique digests")
        object.__setattr__(self, "artifact_digests", digests)
        if len(self.excluded_candidate_artifact_digests) != 1:
            raise ValueError("probability manifest must exclude one pilot candidate")
        excluded = tuple(
            _digest("excluded candidate digest", item)
            for item in self.excluded_candidate_artifact_digests
        )
        if any(item in digests for item in excluded):
            raise ValueError("excluded candidate digest is a baseline digest")
        object.__setattr__(
            self,
            "excluded_candidate_artifact_digests",
            excluded,
        )
        fixed = (
            ("tail epsilon", self.tail_epsilon, PROBABILITY_TAIL_EPSILON),
            (
                "hard full-mean ceiling",
                self.hard_full_mean_abs_logprob_ceiling,
                HARD_FULL_MEAN_ABS_LOGPROB_CEILING,
            ),
            (
                "hard total-variation ceiling",
                self.hard_total_variation_ceiling,
                HARD_TOTAL_VARIATION_CEILING,
            ),
            (
                "hard Jensen-Shannon ceiling",
                self.hard_jensen_shannon_ceiling,
                HARD_JENSEN_SHANNON_CEILING,
            ),
            (
                "hard high-mass ceiling",
                self.hard_high_mass_max_abs_logprob_ceiling,
                HARD_HIGH_MASS_MAX_ABS_LOGPROB_CEILING,
            ),
        )
        for name, observed, expected in fixed:
            if _finite_nonnegative(name, observed) != expected:
                raise ValueError(f"probability manifest changed fixed {name}")
        for name in (
            "u_full_mean_abs_logprob",
            "u_total_variation",
            "u_jensen_shannon_divergence",
            "u_high_mass_max_abs_logprob",
            "diagnostic_u_strict_full_max_abs_logprob",
        ):
            _finite_nonnegative(name, getattr(self, name))
        if not isinstance(self.baseline_status, ProbabilityEnsembleStatus):
            raise ValueError("invalid probability baseline status")
        object.__setattr__(
            self,
            "baseline_failure_reasons",
            _reasons(self.baseline_failure_reasons),
        )
        if self.baseline_status is ProbabilityEnsembleStatus.FAIL:
            raise ValueError("FAIL is not a probability baseline status")
        if self.stable != (not self.baseline_failure_reasons):
            raise ValueError("probability baseline status and reasons disagree")
        if self.stable and (
            self.u_full_mean_abs_logprob
            > HARD_FULL_MEAN_ABS_LOGPROB_CEILING
            or self.u_total_variation > HARD_TOTAL_VARIATION_CEILING
            or self.u_jensen_shannon_divergence
            > HARD_JENSEN_SHANNON_CEILING
            or self.u_high_mass_max_abs_logprob
            > HARD_HIGH_MASS_MAX_ABS_LOGPROB_CEILING
        ):
            raise ValueError("stable probability baseline exceeds a hard ceiling")

    @property
    def stable(self) -> bool:
        return self.baseline_status is ProbabilityEnsembleStatus.PASS


@dataclass(frozen=True, slots=True)
class ProbabilityBaselineEnsemble:
    """Five controls, all ten comparisons, and their frozen manifest."""

    baselines: tuple[CorrectnessArtifact, ...]
    comparisons: tuple[ProbabilityBaselineComparison, ...]
    manifest: ProbabilityBaselineManifest

    def __post_init__(self) -> None:
        if len(self.baselines) != PROBABILITY_BASELINE_COUNT:
            raise ValueError("probability ensemble must contain five baselines")
        if len(self.comparisons) != PROBABILITY_PAIR_COUNT:
            raise ValueError("probability ensemble must contain ten comparisons")
        if tuple(artifact_digest(item) for item in self.baselines) != (
            self.manifest.artifact_digests
        ):
            raise ValueError("probability manifest does not bind its baselines")
        expected_pairs = set(
            itertools.combinations(range(PROBABILITY_BASELINE_COUNT), 2)
        )
        observed_pairs = {
            (item.left_index, item.right_index) for item in self.comparisons
        }
        if observed_pairs != expected_pairs:
            raise ValueError("probability ensemble pair coverage is incomplete")

    @property
    def status(self) -> ProbabilityEnsembleStatus:
        return self.manifest.baseline_status


@dataclass(frozen=True, slots=True)
class ProbabilityCandidateVerdict:
    """Prospective candidate result against all five controls."""

    status: ProbabilityEnsembleStatus
    candidate_artifact_digest: str
    manifest_digest: str
    comparisons: tuple[ProbabilityCandidateComparison, ...]
    q_full_mean_abs_logprob: float
    q_total_variation: float
    q_jensen_shannon_divergence: float
    q_high_mass_max_abs_logprob: float
    diagnostic_q_strict_full_max_abs_logprob: float
    u_full_mean_abs_logprob: float
    u_total_variation: float
    u_jensen_shannon_divergence: float
    u_high_mass_max_abs_logprob: float
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.status, ProbabilityEnsembleStatus):
            raise ValueError("invalid probability candidate status")
        _digest("candidate artifact digest", self.candidate_artifact_digest)
        _digest("manifest digest", self.manifest_digest)
        if len(self.comparisons) != PROBABILITY_BASELINE_COUNT:
            raise ValueError("candidate must be compared with five baselines")
        if {item.baseline_index for item in self.comparisons} != set(
            range(PROBABILITY_BASELINE_COUNT)
        ):
            raise ValueError("probability candidate baseline coverage is incomplete")
        for name in (
            "q_full_mean_abs_logprob",
            "q_total_variation",
            "q_jensen_shannon_divergence",
            "q_high_mass_max_abs_logprob",
            "diagnostic_q_strict_full_max_abs_logprob",
            "u_full_mean_abs_logprob",
            "u_total_variation",
            "u_jensen_shannon_divergence",
            "u_high_mass_max_abs_logprob",
        ):
            _finite_nonnegative(name, getattr(self, name))
        object.__setattr__(self, "failure_reasons", _reasons(self.failure_reasons))
        if self.passed != (not self.failure_reasons):
            raise ValueError("probability candidate status and reasons disagree")

    @property
    def passed(self) -> bool:
        return self.status is ProbabilityEnsembleStatus.PASS


@dataclass(frozen=True, slots=True)
class _DistributionView:
    logprobs: tuple[float, ...]
    probabilities: tuple[float, ...]
    high_mass_support: frozenset[int]


def _probabilities(values: tuple[float, ...]) -> tuple[float, ...]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("probability gate requires finite full logprobs")
    maximum = max(values)
    weights = tuple(math.exp(value - maximum) for value in values)
    total = math.fsum(weights)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("probability gate cannot normalize logprobs")
    return tuple(weight / total for weight in weights)


def _high_mass_support(probabilities: tuple[float, ...]) -> frozenset[int]:
    order = sorted(
        range(len(probabilities)),
        key=lambda index: (-probabilities[index], index),
    )
    target = 1.0 - PROBABILITY_TAIL_EPSILON
    cumulative = 0.0
    selected: list[int] = []
    for index in order:
        selected.append(index)
        cumulative += probabilities[index]
        if cumulative >= target:
            break
    if not selected:
        raise ValueError("probability gate selected an empty support")
    return frozenset(selected)


def _view(distribution: FullVocabularyLogprobs) -> _DistributionView:
    probabilities = _probabilities(distribution.values)
    return _DistributionView(
        logprobs=distribution.values,
        probabilities=probabilities,
        high_mass_support=_high_mass_support(probabilities),
    )


def _compare_views(
    left: FullVocabularyLogprobs,
    right: FullVocabularyLogprobs,
    left_view: _DistributionView,
    right_view: _DistributionView,
) -> ProbabilityMetrics:
    if len(left_view.logprobs) != len(right_view.logprobs):
        raise ValueError("probability distributions have different sizes")
    count = len(left_view.logprobs)
    absolute_sum = 0.0
    strict_max = 0.0
    total_variation_sum = 0.0
    js_sum = 0.0
    for left_logprob, right_logprob, left_probability, right_probability in zip(
        left_view.logprobs,
        right_view.logprobs,
        left_view.probabilities,
        right_view.probabilities,
        strict=True,
    ):
        difference = abs(left_logprob - right_logprob)
        absolute_sum += difference
        strict_max = max(strict_max, difference)
        total_variation_sum += abs(left_probability - right_probability)
        midpoint = 0.5 * (left_probability + right_probability)
        if left_probability > 0:
            js_sum += 0.5 * left_probability * math.log(
                left_probability / midpoint
            )
        if right_probability > 0:
            js_sum += 0.5 * right_probability * math.log(
                right_probability / midpoint
            )
    support = left_view.high_mass_support | right_view.high_mass_support
    high_mass_max = max(
        abs(left_view.logprobs[index] - right_view.logprobs[index])
        for index in support
    )
    return ProbabilityMetrics(
        full_mean_abs_logprob=absolute_sum / count,
        total_variation=0.5 * total_variation_sum,
        jensen_shannon_divergence=max(js_sum, 0.0),
        high_mass_max_abs_logprob=high_mass_max,
        strict_full_max_abs_logprob=strict_max,
        high_mass_support_tokens=len(support),
        sampled_token_agreement=(
            left.sampled_token_id == right.sampled_token_id
        ),
        top_token_agreement=(left.top_token_id == right.top_token_id),
    )


def _require_compatible_baselines(
    baselines: tuple[CorrectnessArtifact, ...],
) -> None:
    if len(baselines) != PROBABILITY_BASELINE_COUNT:
        raise ValueError("exactly five probability baselines are required")
    if any(not isinstance(item, CorrectnessArtifact) for item in baselines):
        raise ValueError("probability baselines must be correctness artifacts")
    first = baselines[0]
    if any(
        item.run_mode is not CorrectnessRunMode.FULL_PREFILL
        for item in baselines
    ):
        raise ValueError("probability baselines must be FULL_PREFILL artifacts")
    if any(
        item.runtime != first.runtime or item.prompt != first.prompt
        for item in baselines[1:]
    ):
        raise ValueError("probability baselines have incompatible identities")


def _baseline_failure_reasons(
    comparisons: tuple[ProbabilityBaselineComparison, ...],
    *,
    u_full_mean: float,
    u_tv: float,
    u_js: float,
    u_high_mass: float,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if u_full_mean > HARD_FULL_MEAN_ABS_LOGPROB_CEILING:
        reasons.append("baseline_full_mean_exceeds_hard_ceiling")
    if u_tv > HARD_TOTAL_VARIATION_CEILING:
        reasons.append("baseline_total_variation_exceeds_hard_ceiling")
    if u_js > HARD_JENSEN_SHANNON_CEILING:
        reasons.append("baseline_jensen_shannon_exceeds_hard_ceiling")
    if u_high_mass > HARD_HIGH_MASS_MAX_ABS_LOGPROB_CEILING:
        reasons.append("baseline_high_mass_max_exceeds_hard_ceiling")
    if any(not item.metrics.sampled_token_agreement for item in comparisons):
        reasons.append("baseline_sampled_token_mismatch")
    if any(not item.metrics.top_token_agreement for item in comparisons):
        reasons.append("baseline_top_token_mismatch")
    return tuple(reasons)


def build_probability_baseline_ensemble(
    baselines: Sequence[CorrectnessArtifact],
    *,
    excluded_candidates: Sequence[CorrectnessArtifact],
) -> ProbabilityBaselineEnsemble:
    """Build the fixed prospective policy from exactly five controls."""

    artifacts = tuple(baselines)
    _require_compatible_baselines(artifacts)
    digests = tuple(artifact_digest(item) for item in artifacts)
    if len(set(digests)) != PROBABILITY_BASELINE_COUNT:
        raise ValueError("probability baselines must have unique digests")
    excluded = tuple(excluded_candidates)
    if len(excluded) != 1:
        raise ValueError("exactly one strict-v1 pilot candidate must be excluded")
    pilot = excluded[0]
    if (
        not isinstance(pilot, CorrectnessArtifact)
        or pilot.run_mode is not CorrectnessRunMode.CACHEBLEND_100PCT
        or pilot.runtime != artifacts[0].runtime
        or pilot.prompt != artifacts[0].prompt
    ):
        raise ValueError("excluded pilot candidate has an incompatible identity")
    excluded_digests = tuple(artifact_digest(item) for item in excluded)
    if any(item in digests for item in excluded_digests):
        raise ValueError("excluded pilot candidate is a baseline")
    views = tuple(_view(item.distribution) for item in artifacts)
    records: list[ProbabilityBaselineComparison] = []
    for left_index, right_index in itertools.combinations(
        range(PROBABILITY_BASELINE_COUNT), 2
    ):
        records.append(
            ProbabilityBaselineComparison(
                left_index=left_index,
                right_index=right_index,
                left_artifact_digest=digests[left_index],
                right_artifact_digest=digests[right_index],
                metrics=_compare_views(
                    artifacts[left_index].distribution,
                    artifacts[right_index].distribution,
                    views[left_index],
                    views[right_index],
                ),
            )
        )
    comparisons = tuple(records)
    u_full_mean = max(
        item.metrics.full_mean_abs_logprob for item in comparisons
    )
    u_tv = max(item.metrics.total_variation for item in comparisons)
    u_js = max(
        item.metrics.jensen_shannon_divergence for item in comparisons
    )
    u_high_mass = max(
        item.metrics.high_mass_max_abs_logprob for item in comparisons
    )
    diagnostic_u_strict = max(
        item.metrics.strict_full_max_abs_logprob for item in comparisons
    )
    reasons = _baseline_failure_reasons(
        comparisons,
        u_full_mean=u_full_mean,
        u_tv=u_tv,
        u_js=u_js,
        u_high_mass=u_high_mass,
    )
    status = (
        ProbabilityEnsembleStatus.INDETERMINATE_BASELINE_UNSTABLE
        if reasons
        else ProbabilityEnsembleStatus.PASS
    )
    manifest = ProbabilityBaselineManifest(
        policy_version=PROBABILITY_ENSEMBLE_POLICY_VERSION,
        artifact_digests=digests,
        excluded_candidate_artifact_digests=excluded_digests,
        tail_epsilon=PROBABILITY_TAIL_EPSILON,
        hard_full_mean_abs_logprob_ceiling=(
            HARD_FULL_MEAN_ABS_LOGPROB_CEILING
        ),
        hard_total_variation_ceiling=HARD_TOTAL_VARIATION_CEILING,
        hard_jensen_shannon_ceiling=HARD_JENSEN_SHANNON_CEILING,
        hard_high_mass_max_abs_logprob_ceiling=(
            HARD_HIGH_MASS_MAX_ABS_LOGPROB_CEILING
        ),
        u_full_mean_abs_logprob=u_full_mean,
        u_total_variation=u_tv,
        u_jensen_shannon_divergence=u_js,
        u_high_mass_max_abs_logprob=u_high_mass,
        diagnostic_u_strict_full_max_abs_logprob=diagnostic_u_strict,
        baseline_status=status,
        baseline_failure_reasons=reasons,
    )
    return ProbabilityBaselineEnsemble(
        baselines=artifacts,
        comparisons=comparisons,
        manifest=manifest,
    )


def evaluate_probability_candidate(
    ensemble: ProbabilityBaselineEnsemble,
    candidate: CorrectnessArtifact,
) -> ProbabilityCandidateVerdict:
    """Evaluate one fresh candidate; strict-v1 max remains diagnostic only."""

    if not isinstance(ensemble, ProbabilityBaselineEnsemble):
        raise ValueError("invalid probability baseline ensemble")
    if (
        not isinstance(candidate, CorrectnessArtifact)
        or candidate.run_mode is not CorrectnessRunMode.CACHEBLEND_100PCT
    ):
        raise ValueError("candidate must be a CACHEBLEND_100PCT artifact")
    reference = ensemble.baselines[0]
    if candidate.runtime != reference.runtime or candidate.prompt != reference.prompt:
        raise ValueError("candidate has an incompatible probability identity")
    if artifact_digest(candidate) in (
        ensemble.manifest.excluded_candidate_artifact_digests
    ):
        raise ValueError("strict-v1 pilot candidate is ineligible for prospective v2")
    candidate_view = _view(candidate.distribution)
    baseline_views = tuple(
        _view(baseline.distribution) for baseline in ensemble.baselines
    )
    records: list[ProbabilityCandidateComparison] = []
    for index, (baseline, baseline_view) in enumerate(
        zip(ensemble.baselines, baseline_views, strict=True)
    ):
        records.append(
            ProbabilityCandidateComparison(
                baseline_index=index,
                baseline_artifact_digest=artifact_digest(baseline),
                metrics=_compare_views(
                    candidate.distribution,
                    baseline.distribution,
                    candidate_view,
                    baseline_view,
                ),
            )
        )
    comparisons = tuple(records)
    q_full_mean = max(
        item.metrics.full_mean_abs_logprob for item in comparisons
    )
    q_tv = max(item.metrics.total_variation for item in comparisons)
    q_js = max(
        item.metrics.jensen_shannon_divergence for item in comparisons
    )
    q_high_mass = max(
        item.metrics.high_mass_max_abs_logprob for item in comparisons
    )
    diagnostic_q_strict = max(
        item.metrics.strict_full_max_abs_logprob for item in comparisons
    )
    manifest = ensemble.manifest
    reasons: list[str] = []
    if ensemble.status is not ProbabilityEnsembleStatus.PASS:
        reasons.append("baseline_unstable")
    if any(not item.metrics.sampled_token_agreement for item in comparisons):
        reasons.append("candidate_sampled_token_mismatch")
    if any(not item.metrics.top_token_agreement for item in comparisons):
        reasons.append("candidate_top_token_mismatch")
    checks = (
        (
            q_full_mean,
            manifest.u_full_mean_abs_logprob,
            "candidate_full_mean_exceeds_empirical_envelope",
        ),
        (
            q_tv,
            manifest.u_total_variation,
            "candidate_total_variation_exceeds_empirical_envelope",
        ),
        (
            q_js,
            manifest.u_jensen_shannon_divergence,
            "candidate_jensen_shannon_exceeds_empirical_envelope",
        ),
        (
            q_high_mass,
            manifest.u_high_mass_max_abs_logprob,
            "candidate_high_mass_max_exceeds_empirical_envelope",
        ),
        (
            q_full_mean,
            HARD_FULL_MEAN_ABS_LOGPROB_CEILING,
            "candidate_full_mean_exceeds_hard_ceiling",
        ),
        (
            q_tv,
            HARD_TOTAL_VARIATION_CEILING,
            "candidate_total_variation_exceeds_hard_ceiling",
        ),
        (
            q_js,
            HARD_JENSEN_SHANNON_CEILING,
            "candidate_jensen_shannon_exceeds_hard_ceiling",
        ),
        (
            q_high_mass,
            HARD_HIGH_MASS_MAX_ABS_LOGPROB_CEILING,
            "candidate_high_mass_max_exceeds_hard_ceiling",
        ),
    )
    reasons.extend(reason for observed, limit, reason in checks if observed > limit)
    if ensemble.status is not ProbabilityEnsembleStatus.PASS:
        status = ProbabilityEnsembleStatus.INDETERMINATE_BASELINE_UNSTABLE
    else:
        status = ProbabilityEnsembleStatus.FAIL if reasons else (
            ProbabilityEnsembleStatus.PASS
        )
    return ProbabilityCandidateVerdict(
        status=status,
        candidate_artifact_digest=artifact_digest(candidate),
        manifest_digest=manifest_digest(manifest),
        comparisons=comparisons,
        q_full_mean_abs_logprob=q_full_mean,
        q_total_variation=q_tv,
        q_jensen_shannon_divergence=q_js,
        q_high_mass_max_abs_logprob=q_high_mass,
        diagnostic_q_strict_full_max_abs_logprob=diagnostic_q_strict,
        u_full_mean_abs_logprob=manifest.u_full_mean_abs_logprob,
        u_total_variation=manifest.u_total_variation,
        u_jensen_shannon_divergence=manifest.u_jensen_shannon_divergence,
        u_high_mass_max_abs_logprob=manifest.u_high_mass_max_abs_logprob,
        failure_reasons=tuple(reasons),
    )


def manifest_to_dict(
    manifest: ProbabilityBaselineManifest,
) -> dict[str, object]:
    return {
        "schema_version": PROBABILITY_ENSEMBLE_SCHEMA_VERSION,
        "policy_version": manifest.policy_version,
        "artifact_digests": list(manifest.artifact_digests),
        "excluded_candidate_artifact_digests": list(
            manifest.excluded_candidate_artifact_digests
        ),
        "tail_epsilon": manifest.tail_epsilon,
        "hard_full_mean_abs_logprob_ceiling": (
            manifest.hard_full_mean_abs_logprob_ceiling
        ),
        "hard_total_variation_ceiling": manifest.hard_total_variation_ceiling,
        "hard_jensen_shannon_ceiling": manifest.hard_jensen_shannon_ceiling,
        "hard_high_mass_max_abs_logprob_ceiling": (
            manifest.hard_high_mass_max_abs_logprob_ceiling
        ),
        "u_full_mean_abs_logprob": manifest.u_full_mean_abs_logprob,
        "u_total_variation": manifest.u_total_variation,
        "u_jensen_shannon_divergence": (
            manifest.u_jensen_shannon_divergence
        ),
        "u_high_mass_max_abs_logprob": manifest.u_high_mass_max_abs_logprob,
        "diagnostic_u_strict_full_max_abs_logprob": (
            manifest.diagnostic_u_strict_full_max_abs_logprob
        ),
        "baseline_status": manifest.baseline_status.value,
        "baseline_failure_reasons": list(manifest.baseline_failure_reasons),
    }


def _mapping(value: object, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("invalid probability ensemble manifest schema")
    return cast(dict[str, Any], value)


def manifest_from_dict(data: object) -> ProbabilityBaselineManifest:
    keys = {
        "schema_version",
        "policy_version",
        "artifact_digests",
        "excluded_candidate_artifact_digests",
        "tail_epsilon",
        "hard_full_mean_abs_logprob_ceiling",
        "hard_total_variation_ceiling",
        "hard_jensen_shannon_ceiling",
        "hard_high_mass_max_abs_logprob_ceiling",
        "u_full_mean_abs_logprob",
        "u_total_variation",
        "u_jensen_shannon_divergence",
        "u_high_mass_max_abs_logprob",
        "diagnostic_u_strict_full_max_abs_logprob",
        "baseline_status",
        "baseline_failure_reasons",
    }
    root = _mapping(data, keys)
    if (
        isinstance(root["schema_version"], bool)
        or not isinstance(root["schema_version"], int)
        or root["schema_version"] != PROBABILITY_ENSEMBLE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported probability ensemble manifest schema")
    if (
        not isinstance(root["artifact_digests"], list)
        or not isinstance(root["excluded_candidate_artifact_digests"], list)
        or not isinstance(root["baseline_failure_reasons"], list)
    ):
        raise ValueError("invalid probability ensemble manifest lists")
    try:
        return ProbabilityBaselineManifest(
            policy_version=root["policy_version"],
            artifact_digests=tuple(root["artifact_digests"]),
            excluded_candidate_artifact_digests=tuple(
                root["excluded_candidate_artifact_digests"]
            ),
            tail_epsilon=root["tail_epsilon"],
            hard_full_mean_abs_logprob_ceiling=(
                root["hard_full_mean_abs_logprob_ceiling"]
            ),
            hard_total_variation_ceiling=root["hard_total_variation_ceiling"],
            hard_jensen_shannon_ceiling=root["hard_jensen_shannon_ceiling"],
            hard_high_mass_max_abs_logprob_ceiling=(
                root["hard_high_mass_max_abs_logprob_ceiling"]
            ),
            u_full_mean_abs_logprob=root["u_full_mean_abs_logprob"],
            u_total_variation=root["u_total_variation"],
            u_jensen_shannon_divergence=root[
                "u_jensen_shannon_divergence"
            ],
            u_high_mass_max_abs_logprob=root[
                "u_high_mass_max_abs_logprob"
            ],
            diagnostic_u_strict_full_max_abs_logprob=root[
                "diagnostic_u_strict_full_max_abs_logprob"
            ],
            baseline_status=ProbabilityEnsembleStatus(root["baseline_status"]),
            baseline_failure_reasons=tuple(root["baseline_failure_reasons"]),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid probability ensemble manifest values") from exc


def canonical_manifest_bytes(manifest: ProbabilityBaselineManifest) -> bytes:
    return json.dumps(
        manifest_to_dict(manifest),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def manifest_digest(manifest: ProbabilityBaselineManifest) -> str:
    return sha256(canonical_manifest_bytes(manifest)).hexdigest()


__all__ = [
    "HARD_FULL_MEAN_ABS_LOGPROB_CEILING",
    "HARD_HIGH_MASS_MAX_ABS_LOGPROB_CEILING",
    "HARD_JENSEN_SHANNON_CEILING",
    "HARD_TOTAL_VARIATION_CEILING",
    "PROBABILITY_BASELINE_COUNT",
    "PROBABILITY_ENSEMBLE_POLICY_VERSION",
    "PROBABILITY_ENSEMBLE_SCHEMA_VERSION",
    "PROBABILITY_PAIR_COUNT",
    "PROBABILITY_TAIL_EPSILON",
    "ProbabilityBaselineComparison",
    "ProbabilityBaselineEnsemble",
    "ProbabilityBaselineManifest",
    "ProbabilityCandidateComparison",
    "ProbabilityCandidateVerdict",
    "ProbabilityEnsembleStatus",
    "ProbabilityMetrics",
    "build_probability_baseline_ensemble",
    "canonical_manifest_bytes",
    "evaluate_probability_candidate",
    "manifest_digest",
    "manifest_from_dict",
    "manifest_to_dict",
]
