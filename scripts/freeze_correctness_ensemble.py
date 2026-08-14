#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Freeze five ordinary full-prefill artifacts before judging CacheBlend."""

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
    BaselinePairwiseComparison,
    EnsembleStatus,
    build_five_baseline_ensemble,
    canonical_manifest_bytes,
    manifest_digest,
    manifest_to_dict,
    read_artifact,
)


def _comparison_dict(
    pair: BaselinePairwiseComparison,
) -> dict[str, object]:
    comparison = pair.comparison
    return {
        "left_index": pair.left_index,
        "right_index": pair.right_index,
        "left_artifact_digest": pair.left_artifact_digest,
        "right_artifact_digest": pair.right_artifact_digest,
        "max_abs_error": comparison.max_abs_error,
        "mean_abs_error": comparison.mean_abs_error,
        "sampled_token_agreement": comparison.sampled_token_agreement,
        "top_token_agreement": comparison.top_token_agreement,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        action="append",
        type=Path,
        required=True,
        help="repeat exactly five times in canonical capture order",
    )
    parser.add_argument("--hard-max-abs-ceiling", type=float, required=True)
    parser.add_argument("--hard-mean-abs-ceiling", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.baseline) != FIVE_BASELINE_COUNT:
        parser.error(f"--baseline must be supplied exactly {FIVE_BASELINE_COUNT} times")
    if args.output.exists():
        raise FileExistsError("ensemble manifest output already exists")

    ensemble = build_five_baseline_ensemble(
        tuple(read_artifact(path) for path in args.baseline),
        hard_max_abs_ceiling=args.hard_max_abs_ceiling,
        hard_mean_abs_ceiling=args.hard_mean_abs_ceiling,
    )
    with args.output.open("xb") as output:
        output.write(canonical_manifest_bytes(ensemble.manifest) + b"\n")
    report = {
        "schema_version": 1,
        "manifest": manifest_to_dict(ensemble.manifest),
        "manifest_digest": manifest_digest(ensemble.manifest),
        "stable": ensemble.status is EnsembleStatus.PASS,
        "pairwise_comparisons": [
            _comparison_dict(pair) for pair in ensemble.pairwise_comparisons
        ],
    }
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0 if ensemble.status is EnsembleStatus.PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
