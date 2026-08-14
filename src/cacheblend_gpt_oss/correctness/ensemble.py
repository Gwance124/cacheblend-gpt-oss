# SPDX-License-Identifier: Apache-2.0
"""An empirical five-baseline numerical envelope for CacheBlend.

This module deliberately does not assign a confidence level or make a
statistical claim.  Its thresholds are the observed maximum pairwise errors
among five identity-compatible full-prefill artifacts, bounded by explicit
hard ceilings chosen before evaluating a candidate.
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

from cacheblend_gpt_oss.correctness.evaluate import (
    DistributionComparison,
    compare_distributions,
)
from cacheblend_gpt_oss.correctness.io import artifact_digest
from cacheblend_gpt_oss.correctness.models import (
    CorrectnessArtifact,
    CorrectnessRunMode,
)

ENSEMBLE_SCHEMA_VERSION = 1
ENSEMBLE_POLICY_VERSION = "cacheblend-gpt-oss-five-baseline-v1"
FIVE_BASELINE_COUNT = 5
PAIRWISE_COMPARISON_COUNT = 10


class EnsembleStatus(str, Enum):
    """Outcome categories for the baseline and candidate gates."""

    PASS = "PASS"
    FAIL = "FAIL"
    INDETERMINATE_BASELINE_UNSTABLE = "INDETERMINATE_BASELINE_UNSTABLE"


def _require_finite_nonnegative(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"invalid ensemble {name}")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"invalid ensemble {name}")
    return converted


def _require_digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"invalid ensemble {name}")
    return value


def _require_index(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid ensemble {name}")
    return value


def _require_reason_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple) or any(
        not isinstance(reason, str) or not reason for reason in value
    ):
        raise ValueError("invalid ensemble failure reasons")
    return value


@dataclass(frozen=True, slots=True)
class BaselinePairwiseComparison:
    """One of the ten full-vocabulary comparisons between baselines."""

    left_index: int
    right_index: int
    left_artifact_digest: str
    right_artifact_digest: str
    comparison: DistributionComparison

    def __post_init__(self) -> None:
        left_index = _require_index("left index", self.left_index)
        right_index = _require_index("right index", self.right_index)
        if not left_index < right_index or right_index >= FIVE_BASELINE_COUNT:
            raise ValueError("invalid baseline pair indices")
        _require_digest("left artifact digest", self.left_artifact_digest)
        _require_digest("right artifact digest", self.right_artifact_digest)
        if not isinstance(self.comparison, DistributionComparison):
            raise ValueError("invalid baseline pair comparison")

    @property
    def max_abs(self) -> float:
        """The pair's maximum absolute full-vocabulary error."""

        return self.comparison.max_abs_error

    @property
    def mean_abs(self) -> float:
        """The pair's mean absolute full-vocabulary error."""

        return self.comparison.mean_abs_error


@dataclass(frozen=True, slots=True)
class CandidateBaselineComparison:
    """One full-vocabulary comparison between a candidate and a baseline."""

    baseline_index: int
    baseline_artifact_digest: str
    comparison: DistributionComparison

    def __post_init__(self) -> None:
        baseline_index = _require_index("baseline index", self.baseline_index)
        if baseline_index >= FIVE_BASELINE_COUNT:
            raise ValueError("invalid candidate baseline index")
        _require_digest("baseline artifact digest", self.baseline_artifact_digest)
        if not isinstance(self.comparison, DistributionComparison):
            raise ValueError("invalid candidate baseline comparison")

    @property
    def max_abs(self) -> float:
        """The comparison's maximum absolute full-vocabulary error."""

        return self.comparison.max_abs_error

    @property
    def mean_abs(self) -> float:
        """The comparison's mean absolute full-vocabulary error."""

        return self.comparison.mean_abs_error


@dataclass(frozen=True, slots=True)
class BaselineEnsembleManifest:
    """Immutable, serializable policy inputs and observed baseline envelope."""

    policy_version: str
    artifact_digests: tuple[str, ...]
    medoid_artifact_digest: str
    u_max_abs: float
    u_mean_abs: float
    hard_max_abs_ceiling: float
    hard_mean_abs_ceiling: float
    baseline_status: EnsembleStatus = EnsembleStatus.PASS
    baseline_failure_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.policy_version != ENSEMBLE_POLICY_VERSION:
            raise ValueError("unsupported ensemble policy version")
        if len(self.artifact_digests) != FIVE_BASELINE_COUNT:
            raise ValueError("ensemble manifest must contain exactly five digests")
        digests = tuple(
            _require_digest("artifact digest", digest)
            for digest in self.artifact_digests
        )
        if len(set(digests)) != FIVE_BASELINE_COUNT:
            raise ValueError("ensemble manifest requires five unique digests")
        object.__setattr__(self, "artifact_digests", digests)
        medoid_digest = _require_digest(
            "medoid artifact digest", self.medoid_artifact_digest
        )
        if medoid_digest not in digests:
            raise ValueError("medoid digest is not a baseline artifact digest")
        object.__setattr__(self, "medoid_artifact_digest", medoid_digest)
        for name in (
            "u_max_abs",
            "u_mean_abs",
            "hard_max_abs_ceiling",
            "hard_mean_abs_ceiling",
        ):
            _require_finite_nonnegative(name, getattr(self, name))
        if not isinstance(self.baseline_status, EnsembleStatus):
            raise ValueError("invalid ensemble baseline status")
        object.__setattr__(
            self,
            "baseline_failure_reasons",
            _require_reason_tuple(self.baseline_failure_reasons),
        )

    @property
    def medoid_digest(self) -> str:
        """Short alias for the digest of the actual medoid artifact."""

        return self.medoid_artifact_digest

    @property
    def stable(self) -> bool:
        """Whether the observed baseline envelope is eligible for evaluation."""

        return self.baseline_status is EnsembleStatus.PASS


@dataclass(frozen=True, slots=True)
class FiveBaselineCorrectnessEnsemble:
    """Five baselines, all ten pairwise comparisons, and their manifest."""

    baselines: tuple[CorrectnessArtifact, ...]
    pairwise_comparisons: tuple[BaselinePairwiseComparison, ...]
    medoid: CorrectnessArtifact
    manifest: BaselineEnsembleManifest

    def __post_init__(self) -> None:
        if len(self.baselines) != FIVE_BASELINE_COUNT:
            raise ValueError("ensemble must contain exactly five baselines")
        if len(self.pairwise_comparisons) != PAIRWISE_COMPARISON_COUNT:
            raise ValueError("ensemble must contain all ten pairwise comparisons")
        if any(
            not isinstance(artifact, CorrectnessArtifact)
            for artifact in self.baselines
        ):
            raise ValueError("ensemble contains an invalid baseline artifact")
        if not isinstance(self.medoid, CorrectnessArtifact):
            raise ValueError("ensemble contains an invalid medoid artifact")
        if not isinstance(self.manifest, BaselineEnsembleManifest):
            raise ValueError("ensemble contains an invalid manifest")
        digests = tuple(artifact_digest(artifact) for artifact in self.baselines)
        if digests != self.manifest.artifact_digests:
            raise ValueError("ensemble manifest does not bind its baselines")
        if artifact_digest(self.medoid) != self.manifest.medoid_artifact_digest:
            raise ValueError("ensemble manifest does not bind its medoid")

    @property
    def status(self) -> EnsembleStatus:
        """The baseline eligibility status recorded in the manifest."""

        return self.manifest.baseline_status

    @property
    def u_max_abs(self) -> float:
        return self.manifest.u_max_abs

    @property
    def u_mean_abs(self) -> float:
        return self.manifest.u_mean_abs


@dataclass(frozen=True, slots=True)
class CacheBlendEnsembleVerdict:
    """Candidate result with explicit pass, fail, or indeterminate status."""

    status: EnsembleStatus
    candidate_artifact_digest: str
    manifest_digest: str
    candidate_comparisons: tuple[CandidateBaselineComparison, ...]
    q_max_abs: float
    q_mean_abs: float
    u_max_abs: float
    u_mean_abs: float
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.status, EnsembleStatus):
            raise ValueError("invalid candidate ensemble status")
        _require_digest("candidate artifact digest", self.candidate_artifact_digest)
        _require_digest("manifest digest", self.manifest_digest)
        if len(self.candidate_comparisons) != FIVE_BASELINE_COUNT:
            raise ValueError("candidate must be compared with all five baselines")
        for name in ("q_max_abs", "q_mean_abs", "u_max_abs", "u_mean_abs"):
            _require_finite_nonnegative(name, getattr(self, name))
        object.__setattr__(
            self,
            "failure_reasons",
            _require_reason_tuple(self.failure_reasons),
        )

    @property
    def passed(self) -> bool:
        """Whether the candidate satisfied the stable empirical envelope."""

        return self.status is EnsembleStatus.PASS


def _require_identity_compatible(
    artifacts: tuple[CorrectnessArtifact, ...],
) -> None:
    first = artifacts[0]
    if any(
        artifact.run_mode is not CorrectnessRunMode.FULL_PREFILL
        for artifact in artifacts
    ):
        raise ValueError("all ensemble baselines must be FULL_PREFILL artifacts")
    if any(
        artifact.runtime != first.runtime or artifact.prompt != first.prompt
        for artifact in artifacts[1:]
    ):
        raise ValueError("ensemble baselines have incompatible identities")


def _select_medoid(
    artifacts: tuple[CorrectnessArtifact, ...],
    pairwise: tuple[BaselinePairwiseComparison, ...],
) -> CorrectnessArtifact:
    """Select an actual artifact by mean-error sum, with stable tie-breakers."""

    scores: list[tuple[float, float, str, int]] = []
    for index, artifact in enumerate(artifacts):
        related = [
            pair
            for pair in pairwise
            if pair.left_index == index or pair.right_index == index
        ]
        mean_sum = sum(pair.mean_abs for pair in related)
        max_sum = sum(pair.max_abs for pair in related)
        scores.append((mean_sum, max_sum, artifact_digest(artifact), index))
    return artifacts[min(scores)[3]]


def _baseline_failure_reasons(
    pairwise: tuple[BaselinePairwiseComparison, ...],
    *,
    u_max_abs: float,
    u_mean_abs: float,
    hard_max_abs_ceiling: float,
    hard_mean_abs_ceiling: float,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if u_max_abs > hard_max_abs_ceiling:
        reasons.append("baseline_u_max_abs_exceeds_hard_ceiling")
    if u_mean_abs > hard_mean_abs_ceiling:
        reasons.append("baseline_u_mean_abs_exceeds_hard_ceiling")
    if any(not pair.comparison.sampled_token_agreement for pair in pairwise):
        reasons.append("baseline_sampled_token_mismatch")
    if any(not pair.comparison.top_token_agreement for pair in pairwise):
        reasons.append("baseline_top_token_mismatch")
    return tuple(reasons)


def build_five_baseline_ensemble(
    baselines: Sequence[CorrectnessArtifact],
    *,
    hard_max_abs_ceiling: float,
    hard_mean_abs_ceiling: float,
    policy_version: str = ENSEMBLE_POLICY_VERSION,
) -> FiveBaselineCorrectnessEnsemble:
    """Build an immutable empirical envelope from exactly five baselines.

    The hard ceilings are policy inputs and must be fixed before a candidate is
    evaluated.  If observed baseline variation exceeds either ceiling, the
    returned ensemble is explicitly indeterminate and cannot produce PASS.
    """

    artifacts = tuple(baselines)
    if len(artifacts) != FIVE_BASELINE_COUNT:
        raise ValueError("exactly five baseline artifacts are required")
    if any(not isinstance(artifact, CorrectnessArtifact) for artifact in artifacts):
        raise ValueError("ensemble baselines must be correctness artifacts")
    _require_identity_compatible(artifacts)
    digests = tuple(artifact_digest(artifact) for artifact in artifacts)
    if len(set(digests)) != FIVE_BASELINE_COUNT:
        raise ValueError("ensemble baselines must have five unique artifact digests")
    hard_max = _require_finite_nonnegative(
        "hard max absolute ceiling", hard_max_abs_ceiling
    )
    hard_mean = _require_finite_nonnegative(
        "hard mean absolute ceiling", hard_mean_abs_ceiling
    )
    if policy_version != ENSEMBLE_POLICY_VERSION:
        raise ValueError("unsupported ensemble policy version")

    pairs: list[BaselinePairwiseComparison] = []
    for left_index, right_index in itertools.combinations(
        range(FIVE_BASELINE_COUNT), 2
    ):
        comparison = compare_distributions(
            artifacts[left_index].distribution,
            artifacts[right_index].distribution,
        )
        pairs.append(
            BaselinePairwiseComparison(
                left_index=left_index,
                right_index=right_index,
                left_artifact_digest=digests[left_index],
                right_artifact_digest=digests[right_index],
                comparison=comparison,
            )
        )
    pairwise = tuple(pairs)
    u_max_abs = max(pair.max_abs for pair in pairwise)
    u_mean_abs = max(pair.mean_abs for pair in pairwise)
    reasons = _baseline_failure_reasons(
        pairwise,
        u_max_abs=u_max_abs,
        u_mean_abs=u_mean_abs,
        hard_max_abs_ceiling=hard_max,
        hard_mean_abs_ceiling=hard_mean,
    )
    status = (
        EnsembleStatus.INDETERMINATE_BASELINE_UNSTABLE
        if reasons
        else EnsembleStatus.PASS
    )
    medoid = _select_medoid(artifacts, pairwise)
    manifest = BaselineEnsembleManifest(
        policy_version=policy_version,
        artifact_digests=digests,
        medoid_artifact_digest=artifact_digest(medoid),
        u_max_abs=u_max_abs,
        u_mean_abs=u_mean_abs,
        hard_max_abs_ceiling=hard_max,
        hard_mean_abs_ceiling=hard_mean,
        baseline_status=status,
        baseline_failure_reasons=reasons,
    )
    return FiveBaselineCorrectnessEnsemble(
        baselines=artifacts,
        pairwise_comparisons=pairwise,
        medoid=medoid,
        manifest=manifest,
    )


def evaluate_cacheblend_100pct_ensemble(
    ensemble: FiveBaselineCorrectnessEnsemble,
    candidate: CorrectnessArtifact,
) -> CacheBlendEnsembleVerdict:
    """Evaluate one candidate against every baseline in the frozen manifest."""

    if not isinstance(ensemble, FiveBaselineCorrectnessEnsemble):
        raise ValueError("invalid five-baseline ensemble")
    if (
        not isinstance(candidate, CorrectnessArtifact)
        or candidate.run_mode is not CorrectnessRunMode.CACHEBLEND_100PCT
    ):
        raise ValueError("candidate must be a CACHEBLEND_100PCT artifact")
    reference = ensemble.baselines[0]
    if candidate.runtime != reference.runtime or candidate.prompt != reference.prompt:
        raise ValueError("candidate has an incompatible correctness identity")

    candidate_digest = artifact_digest(candidate)
    comparisons = tuple(
        CandidateBaselineComparison(
            baseline_index=index,
            baseline_artifact_digest=artifact_digest(baseline),
            comparison=compare_distributions(
                candidate.distribution,
                baseline.distribution,
            ),
        )
        for index, baseline in enumerate(ensemble.baselines)
    )
    q_max_abs = max(item.max_abs for item in comparisons)
    q_mean_abs = max(item.mean_abs for item in comparisons)
    reasons: list[str] = []
    if ensemble.status is not EnsembleStatus.PASS:
        reasons.append("baseline_unstable")
    if any(
        not item.comparison.sampled_token_agreement for item in comparisons
    ):
        reasons.append("candidate_sampled_token_mismatch")
    if any(not item.comparison.top_token_agreement for item in comparisons):
        reasons.append("candidate_top_token_mismatch")
    if q_max_abs > ensemble.u_max_abs:
        reasons.append("candidate_q_max_abs_exceeds_u_max_abs")
    if q_mean_abs > ensemble.u_mean_abs:
        reasons.append("candidate_q_mean_abs_exceeds_u_mean_abs")

    if ensemble.status is not EnsembleStatus.PASS:
        status = EnsembleStatus.INDETERMINATE_BASELINE_UNSTABLE
    else:
        status = EnsembleStatus.FAIL if reasons else EnsembleStatus.PASS
    return CacheBlendEnsembleVerdict(
        status=status,
        candidate_artifact_digest=candidate_digest,
        manifest_digest=manifest_digest(ensemble.manifest),
        candidate_comparisons=comparisons,
        q_max_abs=q_max_abs,
        q_mean_abs=q_mean_abs,
        u_max_abs=ensemble.u_max_abs,
        u_mean_abs=ensemble.u_mean_abs,
        failure_reasons=tuple(reasons),
    )


def manifest_to_dict(
    manifest: BaselineEnsembleManifest,
) -> dict[str, object]:
    """Return the canonical JSON-compatible representation of a manifest."""

    return {
        "schema_version": ENSEMBLE_SCHEMA_VERSION,
        "policy_version": manifest.policy_version,
        "artifact_digests": list(manifest.artifact_digests),
        "medoid_artifact_digest": manifest.medoid_artifact_digest,
        "u_max_abs": manifest.u_max_abs,
        "u_mean_abs": manifest.u_mean_abs,
        "hard_max_abs_ceiling": manifest.hard_max_abs_ceiling,
        "hard_mean_abs_ceiling": manifest.hard_mean_abs_ceiling,
        "baseline_status": manifest.baseline_status.value,
        "baseline_failure_reasons": list(manifest.baseline_failure_reasons),
    }


def _exact_mapping(value: object, expected_keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("invalid ensemble manifest schema")
    return cast(dict[str, Any], value)


def manifest_from_dict(data: object) -> BaselineEnsembleManifest:
    """Parse a fail-closed manifest representation."""

    root = _exact_mapping(
        data,
        {
            "schema_version",
            "policy_version",
            "artifact_digests",
            "medoid_artifact_digest",
            "u_max_abs",
            "u_mean_abs",
            "hard_max_abs_ceiling",
            "hard_mean_abs_ceiling",
            "baseline_status",
            "baseline_failure_reasons",
        },
    )
    if (
        isinstance(root["schema_version"], bool)
        or not isinstance(root["schema_version"], int)
        or root["schema_version"] != ENSEMBLE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported ensemble manifest schema")
    if not isinstance(root["artifact_digests"], list):
        raise ValueError("invalid ensemble manifest artifact digests")
    if not isinstance(root["baseline_failure_reasons"], list):
        raise ValueError("invalid ensemble manifest failure reasons")
    try:
        return BaselineEnsembleManifest(
            policy_version=root["policy_version"],
            artifact_digests=tuple(root["artifact_digests"]),
            medoid_artifact_digest=root["medoid_artifact_digest"],
            u_max_abs=root["u_max_abs"],
            u_mean_abs=root["u_mean_abs"],
            hard_max_abs_ceiling=root["hard_max_abs_ceiling"],
            hard_mean_abs_ceiling=root["hard_mean_abs_ceiling"],
            baseline_status=EnsembleStatus(root["baseline_status"]),
            baseline_failure_reasons=tuple(root["baseline_failure_reasons"]),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid ensemble manifest values") from exc


def canonical_manifest_bytes(manifest: BaselineEnsembleManifest) -> bytes:
    return json.dumps(
        manifest_to_dict(manifest),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def manifest_digest(manifest: BaselineEnsembleManifest) -> str:
    """Hash the canonical immutable manifest for evidence binding."""

    return sha256(canonical_manifest_bytes(manifest)).hexdigest()


# Descriptive aliases keep the public surface easy to discover without
# changing the established pairwise functions in correctness.evaluate.
FiveBaselineEnsemble = FiveBaselineCorrectnessEnsemble
FiveBaselineManifest = BaselineEnsembleManifest
build_baseline_ensemble = build_five_baseline_ensemble
evaluate_cacheblend_ensemble = evaluate_cacheblend_100pct_ensemble


__all__ = [
    "ENSEMBLE_POLICY_VERSION",
    "ENSEMBLE_SCHEMA_VERSION",
    "FIVE_BASELINE_COUNT",
    "PAIRWISE_COMPARISON_COUNT",
    "BaselineEnsembleManifest",
    "BaselinePairwiseComparison",
    "CacheBlendEnsembleVerdict",
    "CandidateBaselineComparison",
    "EnsembleStatus",
    "FiveBaselineCorrectnessEnsemble",
    "FiveBaselineEnsemble",
    "FiveBaselineManifest",
    "build_baseline_ensemble",
    "build_five_baseline_ensemble",
    "canonical_manifest_bytes",
    "evaluate_cacheblend_100pct_ensemble",
    "evaluate_cacheblend_ensemble",
    "manifest_digest",
    "manifest_from_dict",
    "manifest_to_dict",
]
