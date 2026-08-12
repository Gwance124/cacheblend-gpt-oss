#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate and summarize a GPT-OSS selective-ratio experiment artifact.

This command is intended for reports copied from ``solab-g3``.  It validates
the strict CPU-side schema and reports whether real error/latency measurements
are present; it never treats absent measurements as passing evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cacheblend_gpt_oss.gpt_oss.selective_policy_io import (
    SelectionSweepIoError,
    SelectionSweepIoErrorCode,
    read_selection_sweep,
    selection_sweep_digest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-measurements",
        action="store_true",
        help="fail unless every ratio has explicit error and latency data",
    )
    args = parser.parse_args()
    try:
        sweep = read_selection_sweep(args.input)
        measured = all(point.measurement is not None for point in sweep.points)
        if args.require_measurements and not measured:
            raise SelectionSweepIoError(
                SelectionSweepIoErrorCode.INCOMPLETE_MEASUREMENTS
            )
    except SelectionSweepIoError as exc:
        parser.error(exc.code.value)
    report = {
        "artifact_digest": selection_sweep_digest(sweep),
        "measured": measured,
        "point_count": len(sweep.points),
        "ratios": list(sweep.ratios),
        "work_curve": [list(point) for point in sweep.work_curve],
        "error_curve": (
            [list(point) for point in sweep.error_curve] if measured else None
        ),
        "latency_curve": (
            [list(point) for point in sweep.latency_curve] if measured else None
        ),
        "passed": measured,
    }
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
