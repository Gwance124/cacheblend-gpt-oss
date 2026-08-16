#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Evaluate one CacheBlend 100% artifact against a frozen baseline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import ensure_source_path

ensure_source_path()

from cacheblend_gpt_oss.correctness import (  # noqa: E402
    CorrectnessArtifact,
    CorrectnessCase,
    CorrectnessRunMode,
    artifact_digest,
    evaluate_cacheblend_100pct,
    evaluate_cacheblend_selective,
    read_artifact,
    read_frozen_tolerance,
    read_transfer_evidence,
    transfer_evidence_digest,
    validate_transfer_evidence_binding,
)


def _number(value: float) -> float | str:
    return value if math.isfinite(value) else "inf"


def _validate_no_transfer_cache_miss(cacheblend: CorrectnessArtifact) -> None:
    """Accept no sidecar only for an explicit zero-transfer cache miss."""

    connector = cacheblend.connector
    if (
        cacheblend.prompt.case is not CorrectnessCase.CACHE_MISS
        or connector is None
        or connector.kv_tokens_found != 0
        or connector.kv_tokens_loaded != 0
        or connector.kv_tokens_rejected != 0
    ):
        raise ValueError(
            "no-transfer evaluation requires a CACHE_MISS artifact with zero "
            "found, loaded, and rejected KV tokens"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cacheblend", type=Path, required=True)
    parser.add_argument("--tolerance", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=(
            CorrectnessRunMode.CACHEBLEND_100PCT.value,
            CorrectnessRunMode.CACHEBLEND_SELECTIVE.value,
        ),
        default=CorrectnessRunMode.CACHEBLEND_100PCT.value,
        help=(
            "candidate artifact mode; selective mode does not require "
            "overwrite evidence"
        ),
    )
    parser.add_argument(
        "--transfer-evidence",
        type=Path,
        help="all-layer transfer sidecar bound into the verdict",
    )
    parser.add_argument(
        "--allow-cache-miss-no-transfer",
        action="store_true",
        help=(
            "allow the explicit cache-miss case to be judged without a "
            "transfer sidecar when all transfer counters are zero"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    candidate_mode = CorrectnessRunMode(args.mode)
    if (
        args.transfer_evidence is None
        and candidate_mode is not CorrectnessRunMode.CACHEBLEND_SELECTIVE
        and not args.allow_cache_miss_no_transfer
    ):
        parser.error(
            "--transfer-evidence is required unless "
            "--allow-cache-miss-no-transfer is set (selective mode uses "
            "row-work evidence)"
        )
    if args.output is not None and args.output.exists():
        raise FileExistsError("correctness verdict output already exists")

    reference = read_artifact(args.reference)
    cacheblend = read_artifact(args.cacheblend)
    if cacheblend.run_mode is not candidate_mode:
        raise ValueError("candidate artifact mode does not match --mode")
    transfer = None
    if args.transfer_evidence is not None:
        if candidate_mode is CorrectnessRunMode.CACHEBLEND_SELECTIVE:
            raise ValueError(
                "selective candidate uses row-work evidence, not 100% "
                "overwrite evidence"
            )
        transfer = read_transfer_evidence(args.transfer_evidence)
        validate_transfer_evidence_binding(cacheblend, transfer)
    elif candidate_mode is CorrectnessRunMode.CACHEBLEND_100PCT:
        try:
            _validate_no_transfer_cache_miss(cacheblend)
        except ValueError as exc:
            parser.error(str(exc))
    tolerance = read_frozen_tolerance(args.tolerance)
    verdict = (
        evaluate_cacheblend_selective(reference, cacheblend, tolerance)
        if candidate_mode is CorrectnessRunMode.CACHEBLEND_SELECTIVE
        else evaluate_cacheblend_100pct(reference, cacheblend, tolerance)
    )
    comparison = verdict.comparison
    report = {
        "schema_version": 1,
        "reference_artifact_digest": artifact_digest(reference),
        "cacheblend_artifact_digest": artifact_digest(cacheblend),
        "candidate_mode": candidate_mode.value,
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
        "transfer_evidence_digest": (
            None if transfer is None else transfer_evidence_digest(transfer)
        ),
        "transfer_evidence_bound": transfer is not None,
        "transfer_all_layers_loaded_and_overwritten": (
            False if transfer is None else transfer.all_layers_loaded_and_overwritten
        ),
        "explicit_cache_miss_without_transfer": transfer is None,
    }
    rendered = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as output:
            output.write(rendered)
    print(rendered, end="")
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
