#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate and summarize one pinned GPT-OSS benchmark evidence artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cacheblend_gpt_oss.benchmark import (
    BenchmarkArtifact,
    BenchmarkError,
    benchmark_artifact_digest,
    read_benchmark_artifact,
    summarize_benchmark,
)


def _confidence(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    interval = value
    return {
        "count": interval.count,
        "mean": interval.mean,
        "median": interval.median,
        "ci95_low": interval.ci95_low,
        "ci95_high": interval.ci95_high,
    }


def _report(artifact: BenchmarkArtifact) -> dict[str, object]:
    summaries = summarize_benchmark(artifact)
    return {
        "artifact_digest": benchmark_artifact_digest(artifact),
        "attention_backend": artifact.attention_backend,
        "benchmark_ready": artifact.benchmark_ready,
        "case": artifact.case.value,
        "host_id": artifact.host_id,
        "missing_required_arms": [
            arm.value for arm in artifact.missing_required_arms
        ],
        "passed": artifact.benchmark_ready,
        "prompt_tokens": artifact.prompt_tokens,
        "summaries": [
            {
                "arm": summary.arm.value,
                "correctness_passed": summary.correctness_passed,
                "end_to_end_latency_seconds": _confidence(
                    summary.end_to_end_latency_seconds
                ),
                "decode_latency_seconds": _confidence(
                    summary.decode_latency_seconds
                ),
                "prefill_latency_seconds": _confidence(
                    summary.prefill_latency_seconds
                ),
                "peak_memory_bytes": _confidence(summary.peak_memory_bytes),
                "queue_latency_seconds": _confidence(
                    summary.queue_latency_seconds
                ),
                "recomputed_tokens": _confidence(summary.recomputed_tokens),
                "saved_prefill_fraction": _confidence(
                    summary.saved_prefill_fraction
                ),
                "staging_overhead_bytes": _confidence(
                    summary.staging_overhead_bytes
                ),
                "trial_count": summary.trial_count,
                "ttft_seconds": _confidence(summary.ttft_seconds),
            }
            for summary in summaries
        ],
        "trial_count": len(artifact.trials),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="fail unless required arms and correctness evidence are present",
    )
    args = parser.parse_args()
    try:
        artifact = read_benchmark_artifact(args.input)
        report = _report(artifact)
    except BenchmarkError as exc:
        parser.error(exc.code.value)
    if args.require_ready and not report["benchmark_ready"]:
        parser.error("benchmark_not_ready")
    rendered = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        try:
            with args.output.open("x", encoding="utf-8") as output:
                output.write(rendered)
        except FileExistsError:
            parser.error("file_exists")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
