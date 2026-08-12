# SPDX-License-Identifier: Apache-2.0
"""Identifier-free derived reports for pinned benchmark artifacts."""

from __future__ import annotations

from typing import NoReturn

from cacheblend_gpt_oss.benchmark.io import benchmark_artifact_digest
from cacheblend_gpt_oss.benchmark.models import (
    BenchmarkArmSummary,
    BenchmarkArtifact,
    BenchmarkError,
    BenchmarkErrorCode,
    ConfidenceInterval,
    summarize_benchmark,
)

BENCHMARK_REPORT_SCHEMA_VERSION = 1


def _fail(code: BenchmarkErrorCode) -> NoReturn:
    raise BenchmarkError(code)


def _confidence(
    value: ConfidenceInterval | None,
) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "count": value.count,
        "mean": value.mean,
        "median": value.median,
        "ci95_low": value.ci95_low,
        "ci95_high": value.ci95_high,
    }


def _runtime(artifact: BenchmarkArtifact) -> dict[str, object]:
    runtime = artifact.runtime
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


def _summary(summary: BenchmarkArmSummary) -> dict[str, object]:
    return {
        "arm": summary.arm.value,
        "correctness_passed": summary.correctness_passed,
        "candidate_token_hit_fraction": _confidence(
            summary.candidate_token_hit_fraction
        ),
        "document_hit_fraction": _confidence(summary.document_hit_fraction),
        "end_to_end_latency_seconds": _confidence(
            summary.end_to_end_latency_seconds
        ),
        "failure_codes": [failure.value for failure in summary.failure_codes],
        "failure_count": summary.failure_count,
        "decode_latency_seconds": _confidence(summary.decode_latency_seconds),
        "kv_tokens_found": _confidence(summary.kv_tokens_found),
        "kv_tokens_loaded": _confidence(summary.kv_tokens_loaded),
        "kv_tokens_rejected": _confidence(summary.kv_tokens_rejected),
        "loaded_token_hit_fraction": _confidence(
            summary.loaded_token_hit_fraction
        ),
        "lookup_latency_seconds": _confidence(summary.lookup_latency_seconds),
        "max_abs_logit_error": _confidence(summary.max_abs_logit_error),
        "mean_abs_logit_error": _confidence(summary.mean_abs_logit_error),
        "position_correction_latency_seconds": _confidence(
            summary.position_correction_latency_seconds
        ),
        "prefill_latency_seconds": _confidence(summary.prefill_latency_seconds),
        "prefill_tokens_avoided": _confidence(summary.prefill_tokens_avoided),
        "peak_memory_bytes": _confidence(summary.peak_memory_bytes),
        "queue_latency_seconds": _confidence(summary.queue_latency_seconds),
        "recomputed_tokens": _confidence(summary.recomputed_tokens),
        "reusable_document_tokens_requested": _confidence(
            summary.reusable_document_tokens_requested
        ),
        "reusable_documents_hit": _confidence(summary.reusable_documents_hit),
        "reusable_documents_requested": _confidence(
            summary.reusable_documents_requested
        ),
        "selective_recomputation_latency_seconds": _confidence(
            summary.selective_recomputation_latency_seconds
        ),
        "saved_prefill_fraction": _confidence(summary.saved_prefill_fraction),
        "staging_overhead_bytes": _confidence(summary.staging_overhead_bytes),
        "store_latency_seconds": _confidence(summary.store_latency_seconds),
        "transfer_latency_seconds": _confidence(summary.transfer_latency_seconds),
        "trial_count": summary.trial_count,
        "ttft_seconds": _confidence(summary.ttft_seconds),
    }


def build_benchmark_report(artifact: BenchmarkArtifact) -> dict[str, object]:
    """Build a copy-safe report retaining all non-sensitive run identity."""

    if not isinstance(artifact, BenchmarkArtifact):
        _fail(BenchmarkErrorCode.INVALID_SCHEMA)
    summaries = summarize_benchmark(artifact)
    return {
        "schema_version": BENCHMARK_REPORT_SCHEMA_VERSION,
        "artifact_schema_version": artifact.schema_version,
        "artifact_digest": benchmark_artifact_digest(artifact),
        "attention_backend": artifact.attention_backend,
        "benchmark_ready": artifact.benchmark_ready,
        "block_size": artifact.block_size,
        "cache_state": artifact.cache_state.value,
        "case": artifact.case.value,
        "host_id": artifact.host_id,
        "hybrid_kv_cache_enabled": artifact.hybrid_kv_cache_enabled,
        "max_model_len": artifact.max_model_len,
        "missing_required_arms": [
            arm.value for arm in artifact.missing_required_arms
        ],
        "passed": artifact.benchmark_ready,
        "pipeline_parallel_size": artifact.pipeline_parallel_size,
        "prompt_fixture_digest": artifact.prompt_fixture_digest,
        "prompt_tokens": artifact.prompt_tokens,
        "runtime": _runtime(artifact),
        "sampling_seed": artifact.sampling_seed,
        "summaries": [_summary(summary) for summary in summaries],
        "tensor_parallel_size": artifact.tensor_parallel_size,
        "temperature": artifact.temperature,
        "top_p": artifact.top_p,
        "trial_count": len(artifact.trials),
    }


__all__ = [
    "BENCHMARK_REPORT_SCHEMA_VERSION",
    "build_benchmark_report",
]
