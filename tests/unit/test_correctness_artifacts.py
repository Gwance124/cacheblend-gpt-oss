from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from cacheblend_gpt_oss.correctness import (
    ARTIFACT_SCHEMA_VERSION,
    GPT_OSS_VOCAB_SIZE,
    ConnectorCorrectnessEvidence,
    CorrectnessArtifact,
    CorrectnessCase,
    CorrectnessFixture,
    CorrectnessRunMode,
    CorrectnessRuntimeIdentity,
    FullVocabularyLogprobs,
    artifact_digest,
    artifact_from_dict,
    artifact_to_dict,
    build_cache_miss_fixture,
    build_correctness_fixture,
    build_exact_prefix_fixture,
    build_moved_document_fixture,
    build_reordered_documents_fixture,
    connector_counter_delta,
    connector_evidence_from_snapshots,
    connector_store_counter_delta,
    digest_token_ids,
    evaluate_cacheblend_100pct,
    freeze_full_prefill_tolerance,
    has_connector_metric_surface,
    parse_completion_distribution,
    parse_connector_counter_snapshot,
    parse_connector_store_counter_snapshot,
    read_artifact,
    read_frozen_tolerance,
    tolerance_from_dict,
    tolerance_to_dict,
    write_artifact,
    write_frozen_tolerance,
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
    *,
    top_value: float = -0.25,
    other_value: float = -10.0,
    sampled_token_id: int = 7,
) -> FullVocabularyLogprobs:
    values = [other_value] * GPT_OSS_VOCAB_SIZE
    values[sampled_token_id] = top_value
    values[-1] = -math.inf
    return FullVocabularyLogprobs(tuple(values), sampled_token_id)


def _baseline(
    *, distribution: FullVocabularyLogprobs | None = None
) -> CorrectnessArtifact:
    return CorrectnessArtifact(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        run_mode=CorrectnessRunMode.FULL_PREFILL,
        runtime=_runtime(),
        prompt=build_moved_document_fixture().prompt_identity,
        distribution=distribution or _distribution(),
    )


def _cacheblend(
    *, distribution: FullVocabularyLogprobs | None = None
) -> CorrectnessArtifact:
    prompt = build_moved_document_fixture().prompt_identity
    return CorrectnessArtifact(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        run_mode=CorrectnessRunMode.CACHEBLEND_100PCT,
        runtime=_runtime(),
        prompt=prompt,
        distribution=distribution or _distribution(),
        connector=ConnectorCorrectnessEvidence(
            reusable_document_tokens_requested=prompt.target_prompt_tokens,
            kv_tokens_found=256,
            kv_tokens_loaded=256,
            kv_tokens_rejected=0,
            tokens_recomputed=prompt.target_prompt_tokens,
            prefill_tokens_avoided=0,
        ),
    )


def test_moved_document_fixture_is_exact_non_block_aligned_reuse() -> None:
    fixture = build_moved_document_fixture()
    segment = fixture.prompt_identity.reusable_segments[0]

    assert len(fixture.source_prompt_token_ids) == 256
    assert len(fixture.target_prompt_token_ids) == 280
    assert segment.source_start == 0
    assert segment.target_start == 17
    assert fixture.target_prompt_token_ids[17:273] == fixture.source_prompt_token_ids
    assert fixture.prompt_identity.target_prompt_digest != (
        segment.token_digest
    )


def test_all_required_correctness_fixtures_bind_exact_ranges() -> None:
    exact = build_exact_prefix_fixture()
    moved = build_moved_document_fixture()
    reordered = build_reordered_documents_fixture()
    miss = build_cache_miss_fixture()

    assert exact.prompt_identity.reusable_segments[0].source_start == 0
    assert exact.prompt_identity.reusable_segments[0].target_start == 0
    assert len(exact.target_prompt_token_ids) == 263
    assert moved.prompt_identity.case is CorrectnessCase.MOVED_DOCUMENT
    assert len(reordered.source_prompt_token_ids) == 512
    assert len(reordered.target_prompt_token_ids) == 536
    assert [
        segment.source_start for segment in reordered.prompt_identity.reusable_segments
    ] == [0, 256]
    assert [
        segment.target_start for segment in reordered.prompt_identity.reusable_segments
    ] == [273, 17]
    for segment in reordered.prompt_identity.reusable_segments:
        source_tokens = reordered.source_prompt_token_ids[
            segment.source_start : segment.source_start + segment.tokens
        ]
        target_tokens = reordered.target_prompt_token_ids[
            segment.target_start : segment.target_start + segment.tokens
        ]
        assert source_tokens == target_tokens
        assert digest_token_ids(source_tokens) == segment.token_digest
    assert miss.prompt_identity.case is CorrectnessCase.CACHE_MISS
    assert miss.prompt_identity.reusable_segments == ()
    assert miss.prompt_identity.reusable_tokens == 0
    assert build_correctness_fixture(CorrectnessCase.EXACT_PREFIX) == exact
    assert build_correctness_fixture(CorrectnessCase.REORDERED_DOCUMENTS) == reordered


@pytest.mark.parametrize(
    ("fixture", "expected_loaded"),
    [
        (build_exact_prefix_fixture(), 256),
        (build_moved_document_fixture(), 256),
        (build_reordered_documents_fixture(), 512),
        (build_cache_miss_fixture(), 0),
    ],
)
def test_cacheblend_artifact_accepts_exact_case_specific_transfer_coverage(
    fixture: CorrectnessFixture,
    expected_loaded: int,
) -> None:
    prompt = fixture.prompt_identity
    artifact = CorrectnessArtifact(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        run_mode=CorrectnessRunMode.CACHEBLEND_100PCT,
        runtime=_runtime(),
        prompt=prompt,
        distribution=_distribution(),
        connector=ConnectorCorrectnessEvidence(
            reusable_document_tokens_requested=prompt.target_prompt_tokens,
            kv_tokens_found=expected_loaded,
            kv_tokens_loaded=expected_loaded,
            kv_tokens_rejected=0,
            tokens_recomputed=prompt.target_prompt_tokens,
            prefill_tokens_avoided=0,
        ),
    )

    assert artifact.connector is not None
    assert artifact.connector.kv_tokens_loaded == expected_loaded


def test_artifact_canonical_round_trip_preserves_negative_infinity(
    tmp_path: Path,
) -> None:
    artifact = _baseline()
    path = tmp_path / "baseline.json"

    write_artifact(path, artifact)
    loaded = read_artifact(path)

    assert loaded == artifact
    assert artifact_to_dict(artifact)["distribution"]["values"][-1] == "-inf"
    assert artifact_from_dict(artifact_to_dict(artifact)) == artifact
    assert artifact_digest(loaded) == artifact_digest(artifact)
    assert not path.read_text(encoding="utf-8").endswith("\n\n")
    with pytest.raises(FileExistsError):
        write_artifact(path, artifact)


def test_freeze_then_evaluate_full_vocabulary_cacheblend_equivalence() -> None:
    reference = _baseline()
    repeat_values = list(reference.distribution.values)
    repeat_values[100] += 0.001
    repeat = _baseline(
        distribution=FullVocabularyLogprobs(
            tuple(repeat_values), reference.distribution.sampled_token_id
        )
    )
    tolerance = freeze_full_prefill_tolerance(
        reference,
        repeat,
        max_abs_floor=0.002,
        mean_abs_floor=1e-6,
        multiplier=2.0,
    )

    verdict = evaluate_cacheblend_100pct(reference, _cacheblend(), tolerance)

    assert verdict.passed
    assert verdict.failure_reasons == ()
    assert verdict.comparison.compared_values == GPT_OSS_VOCAB_SIZE - 1
    assert verdict.comparison.negative_infinity_values == 1
    assert tolerance.allowed_max_abs_error == pytest.approx(0.002)
    assert tolerance.baseline_max_abs_error == pytest.approx(0.001)


def test_numerical_or_selected_token_mismatch_fails_with_bounded_reasons() -> None:
    reference = _baseline()
    tolerance = freeze_full_prefill_tolerance(
        reference,
        reference,
        max_abs_floor=0.01,
        mean_abs_floor=1e-6,
    )
    changed_values = list(reference.distribution.values)
    changed_values[9] = 1.0
    changed = _cacheblend(
        distribution=FullVocabularyLogprobs(tuple(changed_values), 9)
    )

    verdict = evaluate_cacheblend_100pct(reference, changed, tolerance)

    assert not verdict.passed
    assert verdict.failure_reasons == (
        "sampled_token_mismatch",
        "top_token_mismatch",
        "max_abs_error_exceeded",
        "mean_abs_error_exceeded",
    )


def test_cacheblend_artifact_requires_nonzero_load_full_recompute_zero_savings() -> (
    None
):
    prompt = build_moved_document_fixture().prompt_identity
    with pytest.raises(ValueError, match="work evidence"):
        CorrectnessArtifact(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            run_mode=CorrectnessRunMode.CACHEBLEND_100PCT,
            runtime=_runtime(),
            prompt=prompt,
            distribution=_distribution(),
            connector=ConnectorCorrectnessEvidence(
                reusable_document_tokens_requested=prompt.target_prompt_tokens,
                kv_tokens_found=0,
                kv_tokens_loaded=0,
                kv_tokens_rejected=0,
                tokens_recomputed=prompt.target_prompt_tokens,
                prefill_tokens_avoided=0,
            ),
        )


@pytest.mark.parametrize(
    ("requested_tokens", "loaded_tokens"),
    [(279, 256), (280, 255), (280, 257)],
)
def test_cacheblend_artifact_requires_exact_fixture_transfer_coverage(
    requested_tokens: int,
    loaded_tokens: int,
) -> None:
    prompt = build_moved_document_fixture().prompt_identity
    with pytest.raises(ValueError, match="work evidence"):
        CorrectnessArtifact(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            run_mode=CorrectnessRunMode.CACHEBLEND_100PCT,
            runtime=_runtime(),
            prompt=prompt,
            distribution=_distribution(),
            connector=ConnectorCorrectnessEvidence(
                reusable_document_tokens_requested=requested_tokens,
                kv_tokens_found=loaded_tokens,
                kv_tokens_loaded=loaded_tokens,
                kv_tokens_rejected=0,
                tokens_recomputed=prompt.target_prompt_tokens,
                prefill_tokens_avoided=0,
            ),
        )


def test_artifact_identity_and_schema_mismatches_fail_closed(tmp_path: Path) -> None:
    artifact = _baseline()
    with pytest.raises(ValueError, match="unsupported correctness artifact schema"):
        replace(artifact, schema_version=True)

    malformed = artifact_to_dict(artifact)
    malformed["unexpected"] = True
    with pytest.raises(ValueError, match="root schema"):
        artifact_from_dict(malformed)

    different_runtime = CorrectnessArtifact(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        run_mode=CorrectnessRunMode.FULL_PREFILL,
        runtime=_runtime(model_revision="different-revision"),
        prompt=artifact.prompt,
        distribution=artifact.distribution,
    )
    with pytest.raises(ValueError, match="incompatible identities"):
        freeze_full_prefill_tolerance(
            artifact,
            different_runtime,
            max_abs_floor=0,
            mean_abs_floor=0,
        )

    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="valid correctness artifact"):
        read_artifact(bad_path)


def test_frozen_tolerance_round_trip_is_canonical(tmp_path: Path) -> None:
    reference = _baseline()
    tolerance = freeze_full_prefill_tolerance(
        reference,
        reference,
        max_abs_floor=0.01,
        mean_abs_floor=1e-6,
    )
    path = tmp_path / "tolerance.json"

    write_frozen_tolerance(path, tolerance)

    assert read_frozen_tolerance(path) == tolerance
    assert '"schema_version": 1' in path.read_text(encoding="utf-8")
    malformed = tolerance_to_dict(tolerance)
    malformed["schema_version"] = True
    with pytest.raises(ValueError, match="unsupported frozen tolerance schema"):
        tolerance_from_dict(malformed)
    with pytest.raises(FileExistsError):
        write_frozen_tolerance(path, tolerance)


def test_invalid_runtime_distribution_and_tolerance_binding_are_rejected() -> None:
    with pytest.raises(ValueError, match="pinned runtime"):
        CorrectnessRuntimeIdentity(
            **{
                **artifact_to_dict(_baseline())["runtime"],
                "vllm_version": "0.20.0",
            }
        )
    with pytest.raises(ValueError, match="wrong size"):
        FullVocabularyLogprobs((0.0,), 0)

    reference = _baseline()
    tolerance = freeze_full_prefill_tolerance(
        reference,
        reference,
        max_abs_floor=0,
        mean_abs_floor=0,
    )
    other_values = list(reference.distribution.values)
    other_values[100] += 0.01
    other_reference = _baseline(
        distribution=FullVocabularyLogprobs(
            tuple(other_values), reference.distribution.sampled_token_id
        )
    )
    with pytest.raises(ValueError, match="another reference"):
        evaluate_cacheblend_100pct(other_reference, _cacheblend(), tolerance)


def test_full_vocabulary_completion_and_prometheus_deltas_are_parsed() -> None:
    raw_logprobs = {
        f"token_id:{token_id}": (-0.25 if token_id == 7 else -10.0)
        for token_id in range(GPT_OSS_VOCAB_SIZE)
    }
    distribution = parse_completion_distribution(
        {
            "choices": [
                {
                    "token_ids": [7],
                    "logprobs": {"top_logprobs": [raw_logprobs]},
                }
            ]
        }
    )
    before = parse_connector_counter_snapshot(
        """
# HELP ignored ignored
vllm:cacheblend_requests_total{engine="0"} 1
vllm:cacheblend_reusable_document_tokens_requested_total{engine="0"} 256
vllm:cacheblend_kv_tokens_found_total{engine="0"} 0
vllm:cacheblend_kv_tokens_loaded_total{engine="0"} 0
vllm:cacheblend_kv_tokens_rejected_total{engine="0"} 0
vllm:cacheblend_tokens_recomputed_total{engine="0"} 256
vllm:cacheblend_prefill_tokens_avoided_total{engine="0"} 0
"""
    )
    assert has_connector_metric_surface(
        "vllm:cacheblend_requests_total{engine=\"0\"} 1\n"
    )
    assert not has_connector_metric_surface("vllm:num_requests_running 0\n")
    after = parse_connector_counter_snapshot(
        """
vllm:cacheblend_requests_total{engine="0"} 2
vllm:cacheblend_reusable_document_tokens_requested_total{engine="0"} 536
vllm:cacheblend_kv_tokens_found_total{engine="0"} 256
vllm:cacheblend_kv_tokens_loaded_total{engine="0"} 256
vllm:cacheblend_kv_tokens_rejected_total{engine="0"} 0
vllm:cacheblend_tokens_recomputed_total{engine="0"} 536
vllm:cacheblend_prefill_tokens_avoided_total{engine="0"} 0
"""
    )

    evidence = connector_evidence_from_snapshots(before, after)

    assert len(distribution.values) == GPT_OSS_VOCAB_SIZE
    assert distribution.sampled_token_id == 7
    assert distribution.top_token_id == 7
    assert evidence == ConnectorCorrectnessEvidence(
        reusable_document_tokens_requested=280,
        kv_tokens_found=256,
        kv_tokens_loaded=256,
        kv_tokens_rejected=0,
        tokens_recomputed=280,
        prefill_tokens_avoided=0,
    )


def test_partial_completion_or_ambiguous_metric_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="not full-vocabulary"):
        parse_completion_distribution(
            {
                "choices": [
                    {
                        "token_ids": [7],
                        "logprobs": {"top_logprobs": [{"token_id:7": -0.25}]},
                    }
                ]
            }
        )
    empty = parse_connector_counter_snapshot("")
    two_requests = dict(empty)
    two_requests["requests"] = 2
    with pytest.raises(ValueError, match="exactly one target request"):
        connector_evidence_from_snapshots(empty, two_requests)
    assert connector_counter_delta(empty, two_requests)["requests"] == 2


def test_store_counter_surface_is_parsed_and_reconciled_separately() -> None:
    before = parse_connector_store_counter_snapshot(
        """
vllm:cacheblend_store_tokens_eligible_total{engine="0"} 256
vllm:cacheblend_store_tokens_completed_total{engine="0"} 256
vllm:cacheblend_store_fallbacks_total{engine="0"} 0
"""
    )
    after = parse_connector_store_counter_snapshot(
        """
vllm:cacheblend_store_tokens_eligible_total{engine="0"} 768
vllm:cacheblend_store_tokens_completed_total{engine="0"} 512
vllm:cacheblend_store_fallbacks_total{engine="0"} 1
"""
    )

    assert connector_store_counter_delta(before, after) == {
        "store_tokens_eligible": 512,
        "store_tokens_completed": 256,
        "store_fallbacks": 1,
    }

    with pytest.raises(ValueError, match="store counter"):
        connector_store_counter_delta(after, before)
