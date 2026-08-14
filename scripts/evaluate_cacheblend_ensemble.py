#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Evaluate one fresh CacheBlend candidate against a frozen five-run envelope."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import ensure_source_path

ensure_source_path()

from cacheblend_gpt_oss.correctness import (  # noqa: E402
    FIVE_BASELINE_COUNT,
    CandidateBaselineComparison,
    CorrectnessArtifact,
    CorrectnessCase,
    build_five_baseline_ensemble,
    evaluate_cacheblend_100pct_ensemble,
    manifest_digest,
    manifest_from_dict,
    read_artifact,
    read_transfer_evidence,
    transfer_evidence_digest,
    validate_transfer_evidence_binding,
)


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


def _candidate_comparison_dict(
    item: CandidateBaselineComparison,
) -> dict[str, object]:
    comparison = item.comparison
    return {
        "baseline_index": item.baseline_index,
        "baseline_artifact_digest": item.baseline_artifact_digest,
        "max_abs_error": comparison.max_abs_error,
        "mean_abs_error": comparison.mean_abs_error,
        "sampled_token_agreement": comparison.sampled_token_agreement,
        "top_token_agreement": comparison.top_token_agreement,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        action="append",
        type=Path,
        required=True,
        help="repeat exactly five times in the manifest's canonical order",
    )
    parser.add_argument("--cacheblend", type=Path, required=True)
    parser.add_argument(
        "--transfer-evidence",
        type=Path,
        help="all-layer transfer sidecar bound into the verdict",
    )
    parser.add_argument(
        "--allow-cache-miss-no-transfer",
        action="store_true",
        help=(
            "allow the explicit cache-miss case without a transfer sidecar "
            "when all transfer counters are zero"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.baseline) != FIVE_BASELINE_COUNT:
        parser.error(f"--baseline must be supplied exactly {FIVE_BASELINE_COUNT} times")
    if args.transfer_evidence is None and not args.allow_cache_miss_no_transfer:
        parser.error(
            "--transfer-evidence is required unless "
            "--allow-cache-miss-no-transfer is set"
        )
    if args.output.exists():
        raise FileExistsError("ensemble verdict output already exists")

    manifest = manifest_from_dict(
        json.loads(args.manifest.read_text(encoding="utf-8"))
    )
    ensemble = build_five_baseline_ensemble(
        tuple(read_artifact(path) for path in args.baseline),
        hard_max_abs_ceiling=manifest.hard_max_abs_ceiling,
        hard_mean_abs_ceiling=manifest.hard_mean_abs_ceiling,
        policy_version=manifest.policy_version,
    )
    if ensemble.manifest != manifest:
        raise ValueError("frozen ensemble manifest does not bind these baselines")

    candidate = read_artifact(args.cacheblend)
    transfer = None
    if args.transfer_evidence is not None:
        transfer = read_transfer_evidence(args.transfer_evidence)
        validate_transfer_evidence_binding(candidate, transfer)
    else:
        try:
            _validate_no_transfer_cache_miss(candidate)
        except ValueError as exc:
            parser.error(str(exc))
    verdict = evaluate_cacheblend_100pct_ensemble(ensemble, candidate)
    comparisons = [
        _candidate_comparison_dict(item)
        for item in verdict.candidate_comparisons
    ]
    medoid_index = manifest.artifact_digests.index(
        manifest.medoid_artifact_digest
    )
    report = {
        "schema_version": 1,
        "status": verdict.status.value,
        "passed": verdict.passed,
        "failure_reasons": list(verdict.failure_reasons),
        "candidate_artifact_digest": verdict.candidate_artifact_digest,
        "manifest_digest": manifest_digest(manifest),
        "transfer_evidence_digest": (
            None if transfer is None else transfer_evidence_digest(transfer)
        ),
        "transfer_evidence_bound": transfer is not None,
        "explicit_cache_miss_without_transfer": transfer is None,
        "u_max_abs": verdict.u_max_abs,
        "u_mean_abs": verdict.u_mean_abs,
        "q_max_abs": verdict.q_max_abs,
        "q_mean_abs": verdict.q_mean_abs,
        "medoid_artifact_digest": manifest.medoid_artifact_digest,
        "candidate_to_medoid": comparisons[medoid_index],
        "candidate_comparisons": comparisons,
    }
    rendered = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with args.output.open("x", encoding="utf-8") as output:
        output.write(rendered)
    print(rendered, end="")
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
