# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the prospective probability-mass-aware gate."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace

import pytest

from cacheblend_gpt_oss.correctness import (
    ARTIFACT_SCHEMA_VERSION,
    GPT_OSS_VOCAB_SIZE,
    ConnectorCorrectnessEvidence,
    CorrectnessArtifact,
    CorrectnessRunMode,
    CorrectnessRuntimeIdentity,
    FullVocabularyLogprobs,
    build_moved_document_fixture,
)
from cacheblend_gpt_oss.correctness.probability_ensemble import (
    HARD_FULL_MEAN_ABS_LOGPROB_CEILING,
    PROBABILITY_TAIL_EPSILON,
    ProbabilityBaselineEnsemble,
    ProbabilityEnsembleStatus,
    _high_mass_support,
    _probabilities,
    build_probability_baseline_ensemble,
    evaluate_probability_candidate,
    manifest_digest,
    manifest_from_dict,
    manifest_to_dict,
)


def _runtime(*, model_revision: str = "model-revision") -> CorrectnessRuntimeIdentity:
    return CorrectnessRuntimeIdentity(
        model_id="openai/gpt-oss-20b",
        model_revision=model_revision,
        tokenizer_revision="tokenizer-revision",
        plugin_commit="a" * 40,
        model_config_digest="b" * 64,
        kv_cache_config_digest="c" * 64,
        vllm_version="0.19.1",
        lmcache_version="0.4.3",
        torch_version="2.10.0+cu128",
        cuda_runtime="12.8",
        gpu_name="NVIDIA A100-SXM4-80GB",
    )


def _distribution(
    shift: float,
    *,
    tail_outlier: float = 0.0,
    sampled_token_id: int = 7,
) -> FullVocabularyLogprobs:
    values = [-50.0 + shift] * GPT_OSS_VOCAB_SIZE
    values[sampled_token_id] = -0.25 + shift
    values[-1] += tail_outlier
    return FullVocabularyLogprobs(tuple(values), sampled_token_id)


def _artifact(
    shift: float,
    *,
    candidate: bool = False,
    tail_outlier: float = 0.0,
    sampled_token_id: int = 7,
    model_revision: str = "model-revision",
) -> CorrectnessArtifact:
    prompt = build_moved_document_fixture().prompt_identity
    return CorrectnessArtifact(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        run_mode=(
            CorrectnessRunMode.CACHEBLEND_100PCT
            if candidate
            else CorrectnessRunMode.FULL_PREFILL
        ),
        runtime=_runtime(model_revision=model_revision),
        prompt=prompt,
        distribution=_distribution(
            shift,
            tail_outlier=tail_outlier,
            sampled_token_id=sampled_token_id,
        ),
        connector=(
            ConnectorCorrectnessEvidence(
                reusable_document_tokens_requested=prompt.target_prompt_tokens,
                kv_tokens_found=prompt.reusable_tokens,
                kv_tokens_loaded=prompt.reusable_tokens,
                kv_tokens_rejected=0,
                tokens_recomputed=prompt.target_prompt_tokens,
                prefill_tokens_avoided=0,
            )
            if candidate
            else None
        ),
    )


@pytest.fixture(scope="module")
def ensemble_and_candidate() -> tuple[
    ProbabilityBaselineEnsemble, CorrectnessArtifact
]:
    baselines = tuple(
        _artifact(index / 1_000, tail_outlier=index * 0.03)
        for index in range(5)
    )
    pilot = _artifact(0.0015, candidate=True, tail_outlier=0.08)
    ensemble = build_probability_baseline_ensemble(
        baselines,
        excluded_candidates=(pilot,),
    )
    candidate = _artifact(0.002, candidate=True, tail_outlier=0.09)
    return ensemble, candidate


def test_probability_normalization_and_support_are_deterministic() -> None:
    probabilities = _probabilities((0.0, math.log(0.5), math.log(0.25)))

    assert math.fsum(probabilities) == pytest.approx(1.0)
    assert probabilities[0] > probabilities[1] > probabilities[2]
    assert PROBABILITY_TAIL_EPSILON == 1e-4
    assert _high_mass_support((0.99995, 0.00004, 0.00001)) == frozenset({0})


def test_builds_fixed_five_control_manifest_and_round_trips(
    ensemble_and_candidate: tuple[ProbabilityBaselineEnsemble, CorrectnessArtifact],
) -> None:
    ensemble, _ = ensemble_and_candidate
    manifest = ensemble.manifest

    assert ensemble.status is ProbabilityEnsembleStatus.PASS
    assert len(ensemble.comparisons) == 10
    assert manifest.tail_epsilon == PROBABILITY_TAIL_EPSILON
    assert (
        manifest.hard_full_mean_abs_logprob_ceiling
        == HARD_FULL_MEAN_ABS_LOGPROB_CEILING
    )
    assert manifest_from_dict(manifest_to_dict(manifest)) == manifest
    assert len(manifest_digest(manifest)) == 64
    invalid_schema = manifest_to_dict(manifest)
    invalid_schema["schema_version"] = True
    with pytest.raises(ValueError, match="schema"):
        manifest_from_dict(invalid_schema)
    changed_policy = manifest_to_dict(manifest)
    changed_policy["tail_epsilon"] = 0.1
    with pytest.raises(ValueError, match="manifest values"):
        manifest_from_dict(changed_policy)
    with pytest.raises(FrozenInstanceError):
        manifest.tail_epsilon = 0.1


def test_negligible_tail_strict_max_is_diagnostic_not_acceptance(
    ensemble_and_candidate: tuple[ProbabilityBaselineEnsemble, CorrectnessArtifact],
) -> None:
    ensemble, candidate = ensemble_and_candidate
    verdict = evaluate_probability_candidate(ensemble, candidate)

    assert "candidate_high_mass_max_exceeds_empirical_envelope" not in (
        verdict.failure_reasons
    )
    assert "candidate_high_mass_max_exceeds_hard_ceiling" not in (
        verdict.failure_reasons
    )
    assert verdict.status is ProbabilityEnsembleStatus.PASS
    assert verdict.passed
    assert verdict.diagnostic_q_strict_full_max_abs_logprob > 0.08
    assert verdict.q_high_mass_max_abs_logprob <= (
        verdict.u_high_mass_max_abs_logprob
    )


def test_high_mass_max_is_diagnostic_when_controls_exceed_limit() -> None:
    base = _artifact(0.0)
    baselines: list[CorrectnessArtifact] = []
    for index in range(5):
        values = list(base.distribution.values)
        values[base.distribution.sampled_token_id] += index * 0.025
        baselines.append(
            replace(
                base,
                distribution=FullVocabularyLogprobs(
                    tuple(values), base.distribution.sampled_token_id
                ),
            )
        )
    ensemble = build_probability_baseline_ensemble(
        tuple(baselines),
        excluded_candidates=(_artifact(0.0015, candidate=True),),
    )

    candidate_values = list(base.distribution.values)
    candidate_values[base.distribution.sampled_token_id] += 0.2
    candidate = replace(
        _artifact(0.0015, candidate=True),
        distribution=FullVocabularyLogprobs(
            tuple(candidate_values), base.distribution.sampled_token_id
        ),
    )
    verdict = evaluate_probability_candidate(ensemble, candidate)

    assert ensemble.status is ProbabilityEnsembleStatus.PASS
    assert ensemble.manifest.u_high_mass_max_abs_logprob > 0.08
    assert verdict.q_high_mass_max_abs_logprob > (
        verdict.u_high_mass_max_abs_logprob
    )
    assert "candidate_high_mass_max_exceeds_empirical_envelope" not in (
        verdict.failure_reasons
    )
    assert "candidate_high_mass_max_exceeds_hard_ceiling" not in (
        verdict.failure_reasons
    )


def test_requires_five_unique_compatible_full_prefill_controls() -> None:
    baselines = tuple(_artifact(index / 1_000) for index in range(5))
    with pytest.raises(ValueError, match="exactly five"):
        build_probability_baseline_ensemble(
            baselines[:4],
            excluded_candidates=(_artifact(0.002, candidate=True),),
        )
    with pytest.raises(ValueError, match="unique"):
        build_probability_baseline_ensemble(
            (baselines[0], baselines[0], *baselines[2:]),
            excluded_candidates=(_artifact(0.002, candidate=True),),
        )
    with pytest.raises(ValueError, match="incompatible"):
        build_probability_baseline_ensemble(
            (*baselines[:4], _artifact(0.004, model_revision="other")),
            excluded_candidates=(_artifact(0.002, candidate=True),),
        )


def test_strict_v1_pilot_is_digest_excluded() -> None:
    baselines = tuple(_artifact(index / 1_000) for index in range(5))
    pilot = _artifact(0.002, candidate=True, tail_outlier=0.09)
    ensemble = build_probability_baseline_ensemble(
        baselines,
        excluded_candidates=(pilot,),
    )

    assert ensemble.manifest.excluded_candidate_artifact_digests
    with pytest.raises(ValueError, match="ineligible"):
        evaluate_probability_candidate(ensemble, pilot)


def test_candidate_identity_and_token_agreement_fail_closed(
    ensemble_and_candidate: tuple[ProbabilityBaselineEnsemble, CorrectnessArtifact],
) -> None:
    ensemble, _ = ensemble_and_candidate
    mismatch = _artifact(0.002, candidate=True, sampled_token_id=8)
    verdict = evaluate_probability_candidate(ensemble, mismatch)

    assert verdict.status is ProbabilityEnsembleStatus.FAIL
    assert "candidate_sampled_token_mismatch" in verdict.failure_reasons
    assert "candidate_top_token_mismatch" in verdict.failure_reasons

    incompatible = _artifact(
        0.002,
        candidate=True,
        model_revision="other",
    )
    with pytest.raises(ValueError, match="incompatible probability identity"):
        evaluate_probability_candidate(ensemble, incompatible)
