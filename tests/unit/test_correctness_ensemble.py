# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the five-baseline correctness ensemble."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

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
from cacheblend_gpt_oss.correctness.ensemble import (
    ENSEMBLE_POLICY_VERSION,
    EnsembleStatus,
    build_five_baseline_ensemble,
    evaluate_cacheblend_100pct_ensemble,
    manifest_digest,
    manifest_from_dict,
    manifest_to_dict,
)


def _runtime() -> CorrectnessRuntimeIdentity:
    return CorrectnessRuntimeIdentity(
        model_id="openai/gpt-oss-20b",
        model_revision="model-revision",
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
    sampled_token_id: int = 7,
) -> FullVocabularyLogprobs:
    values = [-10.0 + shift] * GPT_OSS_VOCAB_SIZE
    values[sampled_token_id] = -0.25 + shift
    values[-1] = -math.inf
    return FullVocabularyLogprobs(tuple(values), sampled_token_id)


def _baseline(
    shift: float,
    *,
    sampled_token_id: int = 7,
    model_revision: str = "model-revision",
) -> CorrectnessArtifact:
    runtime = _runtime()
    if model_revision != runtime.model_revision:
        runtime = CorrectnessRuntimeIdentity(
            model_id=runtime.model_id,
            model_revision=model_revision,
            tokenizer_revision=runtime.tokenizer_revision,
            plugin_commit=runtime.plugin_commit,
            model_config_digest=runtime.model_config_digest,
            kv_cache_config_digest=runtime.kv_cache_config_digest,
            vllm_version=runtime.vllm_version,
            lmcache_version=runtime.lmcache_version,
            torch_version=runtime.torch_version,
            cuda_runtime=runtime.cuda_runtime,
            gpu_name=runtime.gpu_name,
            dtype=runtime.dtype,
        )
    return CorrectnessArtifact(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        run_mode=CorrectnessRunMode.FULL_PREFILL,
        runtime=runtime,
        prompt=build_moved_document_fixture().prompt_identity,
        distribution=_distribution(shift, sampled_token_id=sampled_token_id),
    )


def _candidate(
    shift: float,
    *,
    sampled_token_id: int = 7,
) -> CorrectnessArtifact:
    prompt = build_moved_document_fixture().prompt_identity
    return CorrectnessArtifact(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        run_mode=CorrectnessRunMode.CACHEBLEND_100PCT,
        runtime=_runtime(),
        prompt=prompt,
        distribution=_distribution(shift, sampled_token_id=sampled_token_id),
        connector=ConnectorCorrectnessEvidence(
            reusable_document_tokens_requested=prompt.target_prompt_tokens,
            kv_tokens_found=256,
            kv_tokens_loaded=256,
            kv_tokens_rejected=0,
            tokens_recomputed=prompt.target_prompt_tokens,
            prefill_tokens_avoided=0,
        ),
    )


def _baselines() -> tuple[CorrectnessArtifact, ...]:
    return tuple(_baseline(index / 1_000) for index in range(5))


def test_builds_all_ten_pairs_and_selects_an_actual_medoid() -> None:
    ensemble = build_five_baseline_ensemble(
        _baselines(),
        hard_max_abs_ceiling=0.005,
        hard_mean_abs_ceiling=0.005,
    )

    assert ensemble.status is EnsembleStatus.PASS
    assert len(ensemble.pairwise_comparisons) == 10
    assert [
        (pair.left_index, pair.right_index)
        for pair in ensemble.pairwise_comparisons
    ] == [
        (left, right)
        for left in range(5)
        for right in range(left + 1, 5)
    ]
    assert ensemble.u_max_abs == pytest.approx(0.004)
    assert ensemble.u_mean_abs == pytest.approx(0.004)
    assert ensemble.medoid is ensemble.baselines[2]
    assert ensemble.manifest.medoid_digest == (
        ensemble.manifest.artifact_digests[2]
    )


def test_requires_five_unique_identity_compatible_full_prefills() -> None:
    baselines = _baselines()
    with pytest.raises(ValueError, match="exactly five"):
        build_five_baseline_ensemble(
            baselines[:4],
            hard_max_abs_ceiling=1,
            hard_mean_abs_ceiling=1,
        )
    with pytest.raises(ValueError, match="unique artifact digests"):
        build_five_baseline_ensemble(
            (baselines[0], baselines[0], *baselines[2:]),
            hard_max_abs_ceiling=1,
            hard_mean_abs_ceiling=1,
        )
    with pytest.raises(ValueError, match="incompatible identities"):
        build_five_baseline_ensemble(
            (*baselines[:4], _baseline(0.004, model_revision="other")),
            hard_max_abs_ceiling=1,
            hard_mean_abs_ceiling=1,
        )


def test_manifest_is_immutable_and_canonical_round_trips() -> None:
    ensemble = build_five_baseline_ensemble(
        _baselines(),
        hard_max_abs_ceiling=0.005,
        hard_mean_abs_ceiling=0.005,
    )
    manifest = ensemble.manifest

    with pytest.raises(FrozenInstanceError):
        manifest.u_max_abs = 0.0  # type: ignore[misc]

    encoded = manifest_to_dict(manifest)
    assert encoded["policy_version"] == ENSEMBLE_POLICY_VERSION
    assert len(encoded["artifact_digests"]) == 5  # type: ignore[arg-type]
    assert manifest_from_dict(encoded) == manifest
    assert manifest_digest(manifest) == manifest_digest(manifest_from_dict(encoded))


def test_ceiling_excess_is_indeterminate_and_cannot_pass_candidate() -> None:
    ensemble = build_five_baseline_ensemble(
        _baselines(),
        hard_max_abs_ceiling=0.001,
        hard_mean_abs_ceiling=0.001,
    )

    assert ensemble.status is EnsembleStatus.INDETERMINATE_BASELINE_UNSTABLE
    assert ensemble.manifest.baseline_failure_reasons == (
        "baseline_u_max_abs_exceeds_hard_ceiling",
        "baseline_u_mean_abs_exceeds_hard_ceiling",
    )
    verdict = evaluate_cacheblend_100pct_ensemble(ensemble, _candidate(0.002))
    assert verdict.status is EnsembleStatus.INDETERMINATE_BASELINE_UNSTABLE
    assert not verdict.passed
    assert "baseline_unstable" in verdict.failure_reasons


def test_candidate_passes_only_with_token_agreement_and_q_within_u() -> None:
    ensemble = build_five_baseline_ensemble(
        _baselines(),
        hard_max_abs_ceiling=0.005,
        hard_mean_abs_ceiling=0.005,
    )

    passing = evaluate_cacheblend_100pct_ensemble(ensemble, _candidate(0.002))
    assert passing.status is EnsembleStatus.PASS
    assert passing.passed
    assert passing.q_max_abs == pytest.approx(0.002)
    assert passing.q_mean_abs == pytest.approx(0.002)

    outside = evaluate_cacheblend_100pct_ensemble(ensemble, _candidate(0.01))
    assert outside.status is EnsembleStatus.FAIL
    assert not outside.passed
    assert "candidate_q_max_abs_exceeds_u_max_abs" in outside.failure_reasons
    assert "candidate_q_mean_abs_exceeds_u_mean_abs" in outside.failure_reasons

    mismatch = evaluate_cacheblend_100pct_ensemble(
        ensemble,
        _candidate(0.002, sampled_token_id=8),
    )
    assert mismatch.status is EnsembleStatus.FAIL
    assert "candidate_sampled_token_mismatch" in mismatch.failure_reasons
    assert "candidate_top_token_mismatch" in mismatch.failure_reasons


def test_candidate_requires_cacheblend_mode_and_matching_identity() -> None:
    ensemble = build_five_baseline_ensemble(
        _baselines(),
        hard_max_abs_ceiling=0.005,
        hard_mean_abs_ceiling=0.005,
    )
    with pytest.raises(ValueError, match="CACHEBLEND_100PCT"):
        evaluate_cacheblend_100pct_ensemble(ensemble, ensemble.baselines[0])
    incompatible = _candidate(0.002)
    incompatible = CorrectnessArtifact(
        schema_version=incompatible.schema_version,
        run_mode=incompatible.run_mode,
        runtime=CorrectnessRuntimeIdentity(
            model_id="openai/gpt-oss-20b",
            model_revision="other",
            tokenizer_revision="tokenizer-revision",
            plugin_commit="a" * 40,
            model_config_digest="b" * 64,
            kv_cache_config_digest="c" * 64,
            vllm_version="0.19.1",
            lmcache_version="0.4.3",
            torch_version="2.10.0+cu128",
            cuda_runtime="12.8",
            gpu_name="NVIDIA A100-SXM4-80GB",
        ),
        prompt=incompatible.prompt,
        distribution=incompatible.distribution,
        connector=incompatible.connector,
    )
    with pytest.raises(ValueError, match="incompatible correctness identity"):
        evaluate_cacheblend_100pct_ensemble(ensemble, incompatible)


def test_baseline_token_disagreement_is_indeterminate() -> None:
    baselines = _baselines()
    unstable = (*baselines[:4], _baseline(0.004, sampled_token_id=8))
    ensemble = build_five_baseline_ensemble(
        unstable,
        hard_max_abs_ceiling=0.005,
        hard_mean_abs_ceiling=0.005,
    )

    assert ensemble.status is EnsembleStatus.INDETERMINATE_BASELINE_UNSTABLE
    assert "baseline_sampled_token_mismatch" in (
        ensemble.manifest.baseline_failure_reasons
    )
