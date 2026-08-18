#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Evaluate omitted-vs-explicit-false HMA configuration equivalence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import ensure_source_path

ensure_source_path()

from cacheblend_gpt_oss.correctness import (  # noqa: E402
    CorrectnessRunMode,
    DistributionComparison,
    compare_distributions,
    read_artifact,
)

SCHEMA_VERSION = 1
CONTRACT = "cacheblend-gpt-oss-hybrid-flag-equivalence-v1"
RESPONSES_CONTRACT = "cacheblend-gpt-oss-hybrid-flag-responses-v1"
RESOLUTION_CONTRACT = "cacheblend-gpt-oss-hybrid-flag-resolution-v1"
RESPONSES_MODEL = "openai/gpt-oss-20b"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implicit-a-responses", type=Path, required=True)
    parser.add_argument("--implicit-b-responses", type=Path, required=True)
    parser.add_argument("--explicit-false-responses", type=Path, required=True)
    parser.add_argument("--implicit-a-logits", type=Path, required=True)
    parser.add_argument("--implicit-b-logits", type=Path, required=True)
    parser.add_argument("--explicit-false-logits", type=Path, required=True)
    parser.add_argument("--resolution", type=Path, required=True)
    parser.add_argument("--latency-ratio-limit", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def _nonnegative_number(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"invalid Responses {name}")
    return float(value)


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid Responses {name}")
    return value


def _read_responses(path: Path, expected_mode: str) -> dict[str, Any]:
    artifact = _read_json(path)
    if (
        artifact.get("schema_version") != 1
        or artifact.get("contract") != RESPONSES_CONTRACT
        or artifact.get("mode") != expected_mode
        or artifact.get("model") != RESPONSES_MODEL
        or artifact.get("sampling") != {"temperature": 0.0, "top_p": 1.0, "seed": 0}
        or artifact.get("stable_replay_ids") is not True
    ):
        raise ValueError(f"invalid Responses equivalence artifact: {path}")
    turns = artifact.get("turns")
    if not isinstance(turns, list) or len(turns) != 3:
        raise ValueError(f"Responses artifact must contain three turns: {path}")
    cached_tokens: list[int] = []
    elapsed_seconds: list[float] = []
    for turn in turns:
        if not isinstance(turn, dict):
            raise ValueError(f"invalid Responses turn: {path}")
        canonical_output = turn.get("canonical_output")
        if turn.get("output_digest") != _digest(canonical_output):
            raise ValueError(f"Responses output digest mismatch: {path}")
        request_digest = turn.get("request_digest")
        if (
            not isinstance(request_digest, str)
            or len(request_digest) != 64
            or any(character not in "0123456789abcdef" for character in request_digest)
        ):
            raise ValueError(f"invalid Responses request digest: {path}")
        usage = turn.get("usage")
        if not isinstance(usage, dict):
            raise ValueError(f"invalid Responses usage: {path}")
        for key in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_tokens",
            "reasoning_tokens",
            "tool_output_tokens",
        ):
            value = _nonnegative_integer(usage.get(key), f"usage {key}")
            if key == "cached_tokens":
                cached_tokens.append(value)
        elapsed_seconds.append(
            _nonnegative_number(turn.get("elapsed_seconds"), "turn latency")
        )
    prefix_cache = artifact.get("prefix_cache")
    reuse_observed = any(value > 0 for value in cached_tokens[1:])
    if (
        not isinstance(prefix_cache, dict)
        or prefix_cache.get("cached_tokens_per_turn") != cached_tokens
        or prefix_cache.get("reuse_observed_after_cold_turn") is not reuse_observed
    ):
        raise ValueError(f"invalid Responses prefix-cache evidence: {path}")
    total_elapsed = _nonnegative_number(
        artifact.get("total_elapsed_seconds"), "total latency"
    )
    if not math.isclose(
        total_elapsed,
        sum(elapsed_seconds),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(f"Responses latency total does not reconcile: {path}")
    return artifact


def _response_signature(artifact: dict[str, Any]) -> list[dict[str, object]]:
    signature = []
    for turn in artifact["turns"]:
        signature.append(
            {
                "request_digest": turn["request_digest"],
                "output_digest": turn["output_digest"],
                "usage": turn["usage"],
            }
        )
    return signature


def _comparison_dict(comparison: DistributionComparison) -> dict[str, object]:
    def finite(value: float) -> float | str:
        return value if math.isfinite(value) else "inf"

    return {
        "max_abs_logprob_error": finite(comparison.max_abs_error),
        "mean_abs_logprob_error": finite(comparison.mean_abs_error),
        "max_relative_logprob_error": finite(comparison.max_relative_error),
        "mean_relative_logprob_error": finite(comparison.mean_relative_error),
        "compared_values": comparison.compared_values,
        "negative_infinity_values": comparison.negative_infinity_values,
        "sampled_token_agreement": comparison.sampled_token_agreement,
        "top_token_agreement": comparison.top_token_agreement,
    }


def evaluate(
    *,
    implicit_a_responses: dict[str, Any],
    implicit_b_responses: dict[str, Any],
    explicit_responses: dict[str, Any],
    implicit_a_logits: Path,
    implicit_b_logits: Path,
    explicit_logits: Path,
    resolution: dict[str, Any],
    latency_ratio_limit: float,
) -> tuple[dict[str, object], int]:
    if not math.isfinite(latency_ratio_limit) or latency_ratio_limit <= 1.0:
        raise ValueError("latency ratio limit must be finite and greater than one")
    resolution_passed = (
        resolution.get("schema_version") == 1
        and resolution.get("contract") == RESOLUTION_CONTRACT
        and resolution.get("resolved_snapshots_equal") is True
        and resolution.get("resolved_disable_hybrid_kv_cache_manager") is False
        and resolution.get("gate_passed") is True
    )

    implicit_a = read_artifact(implicit_a_logits)
    implicit_b = read_artifact(implicit_b_logits)
    explicit = read_artifact(explicit_logits)
    if any(
        artifact.run_mode is not CorrectnessRunMode.FULL_PREFILL
        for artifact in (implicit_a, implicit_b, explicit)
    ):
        raise ValueError("hybrid-flag logits must all be full-prefill artifacts")
    if not (
        implicit_a.runtime == implicit_b.runtime == explicit.runtime
        and implicit_a.prompt == implicit_b.prompt == explicit.prompt
    ):
        raise ValueError("hybrid-flag logits have incompatible identities")

    baseline_comparison = compare_distributions(
        implicit_a.distribution,
        implicit_b.distribution,
    )
    candidate_a_comparison = compare_distributions(
        implicit_a.distribution,
        explicit.distribution,
    )
    candidate_b_comparison = compare_distributions(
        implicit_b.distribution,
        explicit.distribution,
    )
    baseline_numerically_stable = (
        baseline_comparison.sampled_token_agreement
        and baseline_comparison.top_token_agreement
        and math.isfinite(baseline_comparison.max_abs_error)
        and math.isfinite(baseline_comparison.mean_abs_error)
    )
    numerical_within_baseline_envelope = baseline_numerically_stable and all(
        comparison.sampled_token_agreement
        and comparison.top_token_agreement
        and comparison.max_abs_error <= baseline_comparison.max_abs_error
        and comparison.mean_abs_error <= baseline_comparison.mean_abs_error
        for comparison in (candidate_a_comparison, candidate_b_comparison)
    )

    implicit_a_signature = _response_signature(implicit_a_responses)
    implicit_b_signature = _response_signature(implicit_b_responses)
    explicit_signature = _response_signature(explicit_responses)
    baseline_responses_stable = implicit_a_signature == implicit_b_signature
    explicit_responses_match = (
        explicit_signature == implicit_a_signature == implicit_b_signature
    )
    prefix_reuse_all = all(
        artifact["prefix_cache"].get("reuse_observed_after_cold_turn") is True
        for artifact in (
            implicit_a_responses,
            implicit_b_responses,
            explicit_responses,
        )
    )

    implicit_latencies = [
        float(implicit_a_responses["total_elapsed_seconds"]),
        float(implicit_b_responses["total_elapsed_seconds"]),
    ]
    explicit_latency = float(explicit_responses["total_elapsed_seconds"])
    implicit_mean_latency = sum(implicit_latencies) / len(implicit_latencies)
    implicit_spread_ratio = max(implicit_latencies) / max(
        min(implicit_latencies),
        1e-12,
    )
    explicit_latency_ratio = explicit_latency / max(implicit_mean_latency, 1e-12)
    baseline_latency_stable = implicit_spread_ratio <= latency_ratio_limit
    explicit_latency_within_limit = explicit_latency_ratio <= latency_ratio_limit

    if not resolution_passed:
        status = "FAIL_RUNTIME_RESOLUTION_MISMATCH"
        exit_status = 1
    elif not baseline_numerically_stable or not baseline_responses_stable:
        status = "INCONCLUSIVE_IMPLICIT_BASELINE_UNSTABLE"
        exit_status = 2
    elif not prefix_reuse_all:
        status = "FAIL_PREFIX_REUSE_MISSING"
        exit_status = 1
    elif not numerical_within_baseline_envelope or not explicit_responses_match:
        status = "FAIL_EXPLICIT_FALSE_OUTPUT_DIVERGED"
        exit_status = 1
    elif not baseline_latency_stable:
        status = "INCONCLUSIVE_IMPLICIT_LATENCY_UNSTABLE"
        exit_status = 2
    elif not explicit_latency_within_limit:
        status = "FAIL_EXPLICIT_FALSE_LATENCY_DIVERGED"
        exit_status = 1
    else:
        status = "PASS_IMPLICIT_EQUALS_EXPLICIT_FALSE"
        exit_status = 0

    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT,
        "status": status,
        "passed": exit_status == 0,
        "runtime_resolution_passed": resolution_passed,
        "baseline_responses_stable": baseline_responses_stable,
        "explicit_responses_match": explicit_responses_match,
        "prefix_reuse_all": prefix_reuse_all,
        "baseline_numerically_stable": baseline_numerically_stable,
        "numerical_within_baseline_envelope": numerical_within_baseline_envelope,
        "numerical_envelope": {
            "policy": "candidate_error_not_greater_than_implicit_repeat_error",
            "baseline": _comparison_dict(baseline_comparison),
            "explicit_vs_implicit_a": _comparison_dict(candidate_a_comparison),
            "explicit_vs_implicit_b": _comparison_dict(candidate_b_comparison),
        },
        "latency": {
            "ratio_limit": latency_ratio_limit,
            "implicit_seconds": implicit_latencies,
            "implicit_mean_seconds": implicit_mean_latency,
            "implicit_spread_ratio": implicit_spread_ratio,
            "explicit_false_seconds": explicit_latency,
            "explicit_to_implicit_mean_ratio": explicit_latency_ratio,
            "baseline_stable": baseline_latency_stable,
            "explicit_within_limit": explicit_latency_within_limit,
        },
        "sampling": {"temperature": 0.0, "top_p": 1.0, "seed": 0},
        "response_signatures": {
            "implicit_a": implicit_a_signature,
            "implicit_b": implicit_b_signature,
            "explicit_false": explicit_signature,
        },
    }
    return report, exit_status


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError("hybrid-flag verdict output already exists")
    report, exit_status = evaluate(
        implicit_a_responses=_read_responses(
            args.implicit_a_responses,
            "implicit",
        ),
        implicit_b_responses=_read_responses(
            args.implicit_b_responses,
            "implicit",
        ),
        explicit_responses=_read_responses(
            args.explicit_false_responses,
            "explicit_false",
        ),
        implicit_a_logits=args.implicit_a_logits,
        implicit_b_logits=args.implicit_b_logits,
        explicit_logits=args.explicit_false_logits,
        resolution=_read_json(args.resolution),
        latency_ratio_limit=args.latency_ratio_limit,
    )
    rendered = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with args.output.open("x", encoding="utf-8") as output:
        output.write(rendered)
    print(rendered, end="")
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
