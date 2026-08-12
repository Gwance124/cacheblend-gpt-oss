"""CPU-only tests for the pinned benchmark evidence contract."""

from __future__ import annotations

import json

import pytest

from cacheblend_gpt_oss.benchmark import (
    BENCHMARK_MAX_MODEL_LEN,
    BenchmarkArm,
    BenchmarkArtifact,
    BenchmarkCacheState,
    BenchmarkError,
    BenchmarkErrorCode,
    BenchmarkFailureCode,
    BenchmarkTrial,
    benchmark_artifact_digest,
    benchmark_artifact_from_dict,
    benchmark_artifact_to_dict,
    read_benchmark_artifact,
    summarize_benchmark,
    write_benchmark_artifact,
)
from cacheblend_gpt_oss.correctness.models import (
    CorrectnessCase,
    CorrectnessRuntimeIdentity,
)
from cacheblend_gpt_oss.metrics import (
    RequestCorrectnessMetrics,
    RequestMetricCounters,
    RequestMetrics,
    RequestMetricTimers,
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


def _metrics(*, recomputed: int = 100, avoided: int = 0) -> RequestMetrics:
    return RequestMetrics(
        counters=RequestMetricCounters(
            prompt_tokens=100,
            reusable_documents_requested=1,
            reusable_documents_hit=1,
            reusable_document_tokens_requested=60,
            kv_tokens_found=60,
            kv_tokens_loaded=55,
            kv_tokens_rejected=5,
            tokens_recomputed=recomputed,
            prefill_tokens_avoided=avoided,
        ),
        timers=RequestMetricTimers(
            lookup_latency_seconds=0.001,
            transfer_latency_seconds=0.002,
            position_correction_latency_seconds=0.0,
            selective_recomputation_latency_seconds=0.0,
            ttft_seconds=0.1,
            prefill_latency_seconds=0.4,
            store_latency_seconds=0.003,
            queue_latency_seconds=0.004,
            decode_latency_seconds=0.005,
            end_to_end_latency_seconds=0.406,
        ),
        correctness=RequestCorrectnessMetrics(
            max_abs_logit_error=0.0,
            mean_abs_logit_error=0.0,
        ),
    )


def _trial(
    arm: BenchmarkArm,
    *,
    index: int = 1,
    state: BenchmarkCacheState = BenchmarkCacheState.WARM,
    passed: bool = True,
) -> BenchmarkTrial:
    selective = arm is BenchmarkArm.CACHEBLEND_SELECTIVE
    full_recompute = arm in {
        BenchmarkArm.FULL_PREFILL,
        BenchmarkArm.CACHEBLEND_100PCT,
    }
    return BenchmarkTrial(
        arm=arm,
        case=CorrectnessCase.MOVED_DOCUMENT,
        cache_state=state,
        trial_index=index,
        metrics=_metrics(
            recomputed=100 if full_recompute else 40,
            avoided=0 if full_recompute else 60,
        ),
        recompute_ratio=(
            0.5
            if selective
            else (1.0 if arm is BenchmarkArm.CACHEBLEND_100PCT else None)
        ),
        peak_memory_bytes=80_000,
        correctness_passed=passed,
        correctness_artifact_digest=("d" * 64 if passed else None),
        failure=None if passed else BenchmarkFailureCode.CORRECTNESS_FAILED,
        staging_overhead_bytes=4_096,
    )


def _artifact(*trials: BenchmarkTrial) -> BenchmarkArtifact:
    return BenchmarkArtifact(
        schema_version=1,
        runtime=_runtime(),
        case=CorrectnessCase.MOVED_DOCUMENT,
        prompt_tokens=100,
        host_id="solab-g3",
        attention_backend="TRITON_ATTN",
        hybrid_kv_cache_enabled=True,
        block_size=16,
        max_model_len=BENCHMARK_MAX_MODEL_LEN,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        sampling_seed=0,
        temperature=0.0,
        top_p=1.0,
        trials=trials,
    )


def test_ready_artifact_requires_full_and_cacheblend_controls() -> None:
    artifact = _artifact(
        _trial(BenchmarkArm.FULL_PREFILL),
        _trial(BenchmarkArm.CACHEBLEND_100PCT),
        _trial(BenchmarkArm.CACHEBLEND_SELECTIVE),
    )
    assert artifact.benchmark_ready
    assert artifact.missing_required_arms == ()
    assert artifact.arms == (
        BenchmarkArm.FULL_PREFILL,
        BenchmarkArm.CACHEBLEND_100PCT,
        BenchmarkArm.CACHEBLEND_SELECTIVE,
    )


def test_summaries_have_repeated_trial_confidence_intervals() -> None:
    artifact = _artifact(
        _trial(BenchmarkArm.FULL_PREFILL, index=1),
        _trial(BenchmarkArm.FULL_PREFILL, index=2),
        _trial(BenchmarkArm.CACHEBLEND_100PCT),
    )
    summaries = summarize_benchmark(artifact)
    assert summaries[0].arm is BenchmarkArm.FULL_PREFILL
    assert summaries[0].trial_count == 2
    assert summaries[0].ttft_seconds is not None
    assert summaries[0].ttft_seconds.count == 2
    assert summaries[0].end_to_end_latency_seconds.mean == pytest.approx(0.406)
    assert summaries[1].recomputed_tokens.mean == pytest.approx(100.0)


def test_missing_correctness_makes_report_not_ready_but_remains_recordable() -> None:
    artifact = _artifact(
        _trial(BenchmarkArm.FULL_PREFILL),
        _trial(BenchmarkArm.CACHEBLEND_100PCT, passed=False),
    )
    assert not artifact.benchmark_ready
    assert not summarize_benchmark(artifact)[1].correctness_passed


def test_missing_latency_is_not_reported_as_zero_or_ready() -> None:
    metrics = _metrics()
    incomplete = RequestMetrics(
        counters=metrics.counters,
        timers=RequestMetricTimers(
            lookup_latency_seconds=0.001,
            transfer_latency_seconds=0.002,
            position_correction_latency_seconds=0.0,
            selective_recomputation_latency_seconds=0.0,
            ttft_seconds=None,
            prefill_latency_seconds=0.4,
        ),
        correctness=metrics.correctness,
    )
    full = BenchmarkTrial(
        arm=BenchmarkArm.FULL_PREFILL,
        case=CorrectnessCase.MOVED_DOCUMENT,
        cache_state=BenchmarkCacheState.WARM,
        trial_index=1,
        metrics=incomplete,
        recompute_ratio=None,
        peak_memory_bytes=1,
        correctness_passed=True,
        correctness_artifact_digest="d" * 64,
    )
    control = _trial(BenchmarkArm.CACHEBLEND_100PCT)
    artifact = _artifact(full, control)
    assert not artifact.benchmark_ready
    assert summarize_benchmark(artifact)[0].queue_latency_seconds is None


def test_round_trip_digest_and_create_only_writer(tmp_path) -> None:
    artifact = _artifact(
        _trial(BenchmarkArm.FULL_PREFILL),
        _trial(BenchmarkArm.CACHEBLEND_100PCT),
    )
    payload = benchmark_artifact_to_dict(artifact)
    assert benchmark_artifact_from_dict(payload) == artifact
    assert benchmark_artifact_digest(artifact) == benchmark_artifact_digest(
        benchmark_artifact_from_dict(payload)
    )
    path = tmp_path / "benchmark.json"
    write_benchmark_artifact(path, artifact)
    assert read_benchmark_artifact(path) == artifact
    with pytest.raises(BenchmarkError) as caught:
        write_benchmark_artifact(path, artifact)
    assert caught.value.code is BenchmarkErrorCode.FILE_EXISTS


def test_artifact_is_identifier_free() -> None:
    rendered = json.dumps(
        benchmark_artifact_to_dict(
            _artifact(
                _trial(BenchmarkArm.FULL_PREFILL),
                _trial(BenchmarkArm.CACHEBLEND_100PCT),
            )
        ),
        sort_keys=True,
    )
    assert "request_id" not in rendered
    assert "token_ids" not in rendered
    assert "prompt_text" not in rendered


def test_invalid_arm_ratio_case_and_duplicate_trials_fail_closed() -> None:
    with pytest.raises(BenchmarkError) as caught:
        BenchmarkTrial(
            arm=BenchmarkArm.CACHEBLEND_SELECTIVE,
            case=CorrectnessCase.EXACT_PREFIX,
            cache_state=BenchmarkCacheState.WARM,
            trial_index=1,
            metrics=_metrics(recomputed=40, avoided=60),
            recompute_ratio=0.5,
            peak_memory_bytes=1,
            correctness_passed=True,
            correctness_artifact_digest="d" * 64,
        )
    assert caught.value.code is BenchmarkErrorCode.ARM_CASE_MISMATCH

    with pytest.raises(BenchmarkError) as caught:
        BenchmarkTrial(
            arm=BenchmarkArm.CACHEBLEND_100PCT,
            case=CorrectnessCase.MOVED_DOCUMENT,
            cache_state=BenchmarkCacheState.WARM,
            trial_index=1,
            metrics=_metrics(),
            recompute_ratio=0.5,
            peak_memory_bytes=1,
            correctness_passed=True,
            correctness_artifact_digest="d" * 64,
        )
    assert caught.value.code is BenchmarkErrorCode.ARM_METRIC_MISMATCH

    with pytest.raises(BenchmarkError) as caught:
        _artifact(
            _trial(BenchmarkArm.FULL_PREFILL),
            _trial(BenchmarkArm.FULL_PREFILL),
        )
    assert caught.value.code is BenchmarkErrorCode.DUPLICATE_TRIAL


def test_root_and_nested_schema_tampering_is_bounded() -> None:
    payload = benchmark_artifact_to_dict(
        _artifact(
            _trial(BenchmarkArm.FULL_PREFILL),
            _trial(BenchmarkArm.CACHEBLEND_100PCT),
        )
    )
    payload["unknown"] = 1
    with pytest.raises(BenchmarkError) as caught:
        benchmark_artifact_from_dict(payload)
    assert caught.value.code is BenchmarkErrorCode.INVALID_SCHEMA

    payload = benchmark_artifact_to_dict(
        _artifact(
            _trial(BenchmarkArm.FULL_PREFILL),
            _trial(BenchmarkArm.CACHEBLEND_100PCT),
        )
    )
    payload["trials"][0]["metrics"]["counters"]["prompt_tokens"] = True  # type: ignore[index]
    with pytest.raises(BenchmarkError) as caught:
        benchmark_artifact_from_dict(payload)
    assert caught.value.code is BenchmarkErrorCode.INVALID_METRICS


def test_invalid_json_is_bounded(tmp_path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(BenchmarkError) as caught:
        read_benchmark_artifact(path)
    assert caught.value.code is BenchmarkErrorCode.INVALID_JSON
