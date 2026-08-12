#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Freeze a full-prefill numerical envelope before CacheBlend is judged."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import ensure_source_path

ensure_source_path()

from cacheblend_gpt_oss.correctness import (  # noqa: E402
    freeze_full_prefill_tolerance,
    read_artifact,
    tolerance_to_dict,
    write_frozen_tolerance,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, required=True)
    parser.add_argument("--max-abs-floor", type=float, required=True)
    parser.add_argument("--mean-abs-floor", type=float, required=True)
    parser.add_argument("--multiplier", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("frozen tolerance output already exists")

    tolerance = freeze_full_prefill_tolerance(
        read_artifact(args.reference),
        read_artifact(args.repeat),
        max_abs_floor=args.max_abs_floor,
        mean_abs_floor=args.mean_abs_floor,
        multiplier=args.multiplier,
    )
    write_frozen_tolerance(args.output, tolerance)
    print(json.dumps(tolerance_to_dict(tolerance), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
