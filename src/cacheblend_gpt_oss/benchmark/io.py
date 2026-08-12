# SPDX-License-Identifier: Apache-2.0
"""Canonical JSON I/O for pinned benchmark evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import NoReturn, cast

from cacheblend_gpt_oss.benchmark.models import (
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkArm,
    BenchmarkArtifact,
    BenchmarkCacheState,
    BenchmarkError,
    BenchmarkErrorCode,
    BenchmarkFailureCode,
    BenchmarkTrial,
)
from cacheblend_gpt_oss.correctness.models import (
    CorrectnessCase,
    CorrectnessRuntimeIdentity,
)
from cacheblend_gpt_oss.metrics.request import (
    RequestCorrectnessMetrics,
    RequestMetricCounters,
    RequestMetrics,
    RequestMetricTimers,
)


def _fail(code: BenchmarkErrorCode) -> NoReturn:
    raise BenchmarkError(code)


def _mapping(value: object, code: BenchmarkErrorCode) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _exact_mapping(
    value: object,
    keys: set[str],
    code: BenchmarkErrorCode,
) -> Mapping[str, object]:
    mapping = _mapping(value, code)
    if set(mapping) != keys:
        _fail(code)
    return mapping


def _runtime_to_dict(runtime: CorrectnessRuntimeIdentity) -> dict[str, object]:
    return {
        "cuda_runtime": runtime.cuda_runtime,
        "dtype": runtime.dtype,
        "gpu_name": runtime.gpu_name,
        "kv_cache_config_digest": runtime.kv_cache_config_digest,
        "lmcache_version": runtime.lmcache_version,
        "model_config_digest": runtime.model_config_digest,
        "model_id": runtime.model_id,
        "model_revision": runtime.model_revision,
        "plugin_commit": runtime.plugin_commit,
        "tokenizer_revision": runtime.tokenizer_revision,
        "torch_version": runtime.torch_version,
        "vllm_version": runtime.vllm_version,
    }


def _runtime_from_dict(value: object) -> CorrectnessRuntimeIdentity:
    mapping = _exact_mapping(
        value,
        {
            "cuda_runtime",
            "dtype",
            "gpu_name",
            "kv_cache_config_digest",
            "lmcache_version",
            "model_config_digest",
            "model_id",
            "model_revision",
            "plugin_commit",
            "tokenizer_revision",
            "torch_version",
            "vllm_version",
        },
        BenchmarkErrorCode.INVALID_RUNTIME,
    )
    try:
        return CorrectnessRuntimeIdentity(
            **cast(dict[str, str], dict(mapping))
        )
    except (TypeError, ValueError):
        _fail(BenchmarkErrorCode.INVALID_RUNTIME)


def _metrics_to_dict(metrics: RequestMetrics) -> dict[str, object]:
    counters = metrics.counters
    timers = metrics.timers
    correctness = metrics.correctness
    return {
        "counters": {
            "kv_tokens_found": counters.kv_tokens_found,
            "kv_tokens_loaded": counters.kv_tokens_loaded,
            "kv_tokens_rejected": counters.kv_tokens_rejected,
            "prefill_tokens_avoided": counters.prefill_tokens_avoided,
            "prompt_tokens": counters.prompt_tokens,
            "reusable_document_tokens_requested": (
                counters.reusable_document_tokens_requested
            ),
            "reusable_documents_hit": counters.reusable_documents_hit,
            "reusable_documents_requested": counters.reusable_documents_requested,
            "tokens_recomputed": counters.tokens_recomputed,
        },
        "timers": {
            "decode_latency_seconds": timers.decode_latency_seconds,
            "end_to_end_latency_seconds": timers.end_to_end_latency_seconds,
            "lookup_latency_seconds": timers.lookup_latency_seconds,
            "position_correction_latency_seconds": (
                timers.position_correction_latency_seconds
            ),
            "prefill_latency_seconds": timers.prefill_latency_seconds,
            "queue_latency_seconds": timers.queue_latency_seconds,
            "selective_recomputation_latency_seconds": (
                timers.selective_recomputation_latency_seconds
            ),
            "store_latency_seconds": timers.store_latency_seconds,
            "transfer_latency_seconds": timers.transfer_latency_seconds,
            "ttft_seconds": timers.ttft_seconds,
        },
        "correctness": {
            "max_abs_logit_error": correctness.max_abs_logit_error,
            "mean_abs_logit_error": correctness.mean_abs_logit_error,
        },
    }


def _metrics_from_dict(value: object) -> RequestMetrics:
    root = _exact_mapping(
        value,
        {"counters", "timers", "correctness"},
        BenchmarkErrorCode.INVALID_METRICS,
    )
    counters = _exact_mapping(
        root["counters"],
        {
            "kv_tokens_found",
            "kv_tokens_loaded",
            "kv_tokens_rejected",
            "prefill_tokens_avoided",
            "prompt_tokens",
            "reusable_document_tokens_requested",
            "reusable_documents_hit",
            "reusable_documents_requested",
            "tokens_recomputed",
        },
        BenchmarkErrorCode.INVALID_METRICS,
    )
    timers = _exact_mapping(
        root["timers"],
        {
            "decode_latency_seconds",
            "end_to_end_latency_seconds",
            "lookup_latency_seconds",
            "position_correction_latency_seconds",
            "prefill_latency_seconds",
            "queue_latency_seconds",
            "selective_recomputation_latency_seconds",
            "store_latency_seconds",
            "transfer_latency_seconds",
            "ttft_seconds",
        },
        BenchmarkErrorCode.INVALID_METRICS,
    )
    correctness = _exact_mapping(
        root["correctness"],
        {"max_abs_logit_error", "mean_abs_logit_error"},
        BenchmarkErrorCode.INVALID_METRICS,
    )
    try:
        return RequestMetrics(
            counters=RequestMetricCounters(
                **cast(dict[str, int], dict(counters))
            ),
            timers=RequestMetricTimers(
                lookup_latency_seconds=cast(
                    float, timers["lookup_latency_seconds"]
                ),
                decode_latency_seconds=cast(
                    float | None, timers["decode_latency_seconds"]
                ),
                end_to_end_latency_seconds=cast(
                    float | None, timers["end_to_end_latency_seconds"]
                ),
                position_correction_latency_seconds=cast(
                    float, timers["position_correction_latency_seconds"]
                ),
                prefill_latency_seconds=cast(
                    float, timers["prefill_latency_seconds"]
                ),
                queue_latency_seconds=cast(
                    float | None, timers["queue_latency_seconds"]
                ),
                selective_recomputation_latency_seconds=cast(
                    float, timers["selective_recomputation_latency_seconds"]
                ),
                store_latency_seconds=cast(
                    float, timers["store_latency_seconds"]
                ),
                transfer_latency_seconds=cast(
                    float, timers["transfer_latency_seconds"]
                ),
                ttft_seconds=cast(float | None, timers["ttft_seconds"]),
            ),
            correctness=RequestCorrectnessMetrics(
                **cast(dict[str, float | None], dict(correctness))
            ),
        )
    except (TypeError, ValueError):
        _fail(BenchmarkErrorCode.INVALID_METRICS)


def _trial_to_dict(trial: BenchmarkTrial) -> dict[str, object]:
    return {
        "arm": trial.arm.value,
        "cache_state": trial.cache_state.value,
        "case": trial.case.value,
        "correctness_artifact_digest": trial.correctness_artifact_digest,
        "correctness_passed": trial.correctness_passed,
        "failure": None if trial.failure is None else trial.failure.value,
        "metrics": _metrics_to_dict(trial.metrics),
        "peak_memory_bytes": trial.peak_memory_bytes,
        "recompute_ratio": trial.recompute_ratio,
        "staging_overhead_bytes": trial.staging_overhead_bytes,
        "transfer_evidence_digest": trial.transfer_evidence_digest,
        "trial_index": trial.trial_index,
    }


def _trial_from_dict(value: object) -> BenchmarkTrial:
    mapping = _exact_mapping(
        value,
        {
            "arm",
            "cache_state",
            "case",
            "correctness_artifact_digest",
            "correctness_passed",
            "failure",
            "metrics",
            "peak_memory_bytes",
            "recompute_ratio",
            "staging_overhead_bytes",
            "transfer_evidence_digest",
            "trial_index",
        },
        BenchmarkErrorCode.INVALID_SCHEMA,
    )
    try:
        arm = BenchmarkArm(mapping["arm"])
        cache_state = BenchmarkCacheState(mapping["cache_state"])
        case = CorrectnessCase(mapping["case"])
        failure_value = mapping["failure"]
        failure = (
            None
            if failure_value is None
            else BenchmarkFailureCode(failure_value)
        )
        return BenchmarkTrial(
            arm=arm,
            cache_state=cache_state,
            case=case,
            correctness_artifact_digest=cast(
                str | None, mapping["correctness_artifact_digest"]
            ),
            correctness_passed=cast(bool, mapping["correctness_passed"]),
            failure=failure,
            metrics=_metrics_from_dict(mapping["metrics"]),
            peak_memory_bytes=cast(int, mapping["peak_memory_bytes"]),
            recompute_ratio=cast(float | None, mapping["recompute_ratio"]),
            staging_overhead_bytes=cast(
                int, mapping["staging_overhead_bytes"]
            ),
            transfer_evidence_digest=cast(
                str | None, mapping["transfer_evidence_digest"]
            ),
            trial_index=cast(int, mapping["trial_index"]),
        )
    except (TypeError, ValueError, BenchmarkError) as exc:
        if isinstance(exc, BenchmarkError):
            raise
        _fail(BenchmarkErrorCode.INVALID_SCHEMA)


def benchmark_artifact_to_dict(artifact: BenchmarkArtifact) -> dict[str, object]:
    """Return the strict, non-sensitive benchmark mapping."""

    if not isinstance(artifact, BenchmarkArtifact):
        _fail(BenchmarkErrorCode.INVALID_SCHEMA)
    payload = {
        "attention_backend": artifact.attention_backend,
        "block_size": artifact.block_size,
        "case": artifact.case.value,
        "host_id": artifact.host_id,
        "hybrid_kv_cache_enabled": artifact.hybrid_kv_cache_enabled,
        "max_model_len": artifact.max_model_len,
        "pipeline_parallel_size": artifact.pipeline_parallel_size,
        "prompt_tokens": artifact.prompt_tokens,
        "prompt_fixture_digest": artifact.prompt_fixture_digest,
        "runtime": _runtime_to_dict(artifact.runtime),
        "sampling_seed": artifact.sampling_seed,
        "schema_version": artifact.schema_version,
        "temperature": artifact.temperature,
        "tensor_parallel_size": artifact.tensor_parallel_size,
        "top_p": artifact.top_p,
        "trials": [_trial_to_dict(trial) for trial in artifact.trials],
    }
    benchmark_artifact_from_dict(payload)
    return payload


def benchmark_artifact_from_dict(data: object) -> BenchmarkArtifact:
    """Parse and independently validate one benchmark artifact."""

    root = _exact_mapping(
        data,
        {
            "attention_backend",
            "block_size",
            "case",
            "host_id",
            "hybrid_kv_cache_enabled",
            "max_model_len",
            "pipeline_parallel_size",
            "prompt_tokens",
            "prompt_fixture_digest",
            "runtime",
            "sampling_seed",
            "schema_version",
            "temperature",
            "tensor_parallel_size",
            "top_p",
            "trials",
        },
        BenchmarkErrorCode.INVALID_SCHEMA,
    )
    if root["schema_version"] != BENCHMARK_SCHEMA_VERSION:
        _fail(BenchmarkErrorCode.INVALID_SCHEMA)
    raw_trials = root["trials"]
    if not isinstance(raw_trials, list):
        _fail(BenchmarkErrorCode.EMPTY_TRIALS)
    try:
        artifact = BenchmarkArtifact(
            attention_backend=cast(str, root["attention_backend"]),
            block_size=cast(int, root["block_size"]),
            case=CorrectnessCase(root["case"]),
            host_id=cast(str, root["host_id"]),
            hybrid_kv_cache_enabled=cast(
                bool, root["hybrid_kv_cache_enabled"]
            ),
            max_model_len=cast(int, root["max_model_len"]),
            pipeline_parallel_size=cast(int, root["pipeline_parallel_size"]),
            prompt_tokens=cast(int, root["prompt_tokens"]),
            prompt_fixture_digest=cast(
                str, root["prompt_fixture_digest"]
            ),
            runtime=_runtime_from_dict(root["runtime"]),
            sampling_seed=cast(int, root["sampling_seed"]),
            schema_version=cast(int, root["schema_version"]),
            temperature=cast(float, root["temperature"]),
            tensor_parallel_size=cast(int, root["tensor_parallel_size"]),
            top_p=cast(float, root["top_p"]),
            trials=tuple(_trial_from_dict(raw) for raw in raw_trials),
        )
    except (TypeError, ValueError, BenchmarkError) as exc:
        if isinstance(exc, BenchmarkError):
            raise
        _fail(BenchmarkErrorCode.INVALID_SCHEMA)
    return artifact


def canonical_benchmark_bytes(artifact: BenchmarkArtifact) -> bytes:
    """Return deterministic JSON bytes for hashing and evidence storage."""

    return json.dumps(
        benchmark_artifact_to_dict(artifact),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def benchmark_artifact_digest(artifact: BenchmarkArtifact) -> str:
    """Return the SHA-256 digest of canonical benchmark bytes."""

    return sha256(canonical_benchmark_bytes(artifact)).hexdigest()


def read_benchmark_artifact(path: Path) -> BenchmarkArtifact:
    """Read one benchmark artifact and map file/JSON failures to bounded codes."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _fail(BenchmarkErrorCode.INVALID_JSON)
    except (OSError, UnicodeError, TypeError):
        _fail(BenchmarkErrorCode.FILE_ERROR)
    return benchmark_artifact_from_dict(data)


def write_benchmark_artifact(path: Path, artifact: BenchmarkArtifact) -> None:
    """Create one canonical artifact without overwriting prior evidence."""

    try:
        with path.open("xb") as output:
            output.write(canonical_benchmark_bytes(artifact) + b"\n")
    except FileExistsError:
        _fail(BenchmarkErrorCode.FILE_EXISTS)
    except OSError:
        _fail(BenchmarkErrorCode.FILE_ERROR)


__all__ = [
    "benchmark_artifact_digest",
    "benchmark_artifact_from_dict",
    "benchmark_artifact_to_dict",
    "canonical_benchmark_bytes",
    "read_benchmark_artifact",
    "write_benchmark_artifact",
]
