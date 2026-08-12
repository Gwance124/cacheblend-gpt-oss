#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Evaluate one CacheBlend 100% artifact against a frozen baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from cacheblend_gpt_oss.correctness import (
    artifact_digest,
    evaluate_cacheblend_100pct,
    read_artifact,
    read_frozen_tolerance,
)


def _number(value: float) -> float | str:
    return value if math.isfinite(value) else "inf"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cacheblend", type=Path, required=True)
    parser.add_argument("--tolerance", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError("correctness verdict output already exists")

    reference = read_artifact(args.reference)
    cacheblend = read_artifact(args.cacheblend)
    verdict = evaluate_cacheblend_100pct(
        reference,
        cacheblend,
        read_frozen_tolerance(args.tolerance),
    )
    comparison = verdict.comparison
    report = {
        "schema_version": 1,
        "reference_artifact_digest": artifact_digest(reference),
        "cacheblend_artifact_digest": artifact_digest(cacheblend),
        "passed": verdict.passed,
        "failure_reasons": list(verdict.failure_reasons),
        "max_abs_logprob_error": _number(comparison.max_abs_error),
        "mean_abs_logprob_error": _number(comparison.mean_abs_error),
        "max_relative_logprob_error": _number(comparison.max_relative_error),
        "mean_relative_logprob_error": _number(comparison.mean_relative_error),
        "compared_values": comparison.compared_values,
        "negative_infinity_values": comparison.negative_infinity_values,
        "sampled_token_agreement": comparison.sampled_token_agreement,
        "top_token_agreement": comparison.top_token_agreement,
    }
    rendered = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as output:
            output.write(rendered)
    print(rendered, end="")
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
