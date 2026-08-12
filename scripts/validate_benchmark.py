#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate and summarize one pinned GPT-OSS benchmark evidence artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import ensure_source_path

ensure_source_path()

from cacheblend_gpt_oss.benchmark import (  # noqa: E402
    BenchmarkError,
    read_benchmark_artifact,
)
from cacheblend_gpt_oss.benchmark.report import build_benchmark_report  # noqa: E402


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
        report = build_benchmark_report(artifact)
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
