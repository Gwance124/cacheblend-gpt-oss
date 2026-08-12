#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hash the five reviewed solab-g3 artifacts used by the dormant M6 gate.

This command performs no semantic approval and never enables a vLLM plugin. It
only creates the immutable digest bundle that a reviewer can compare with the
artifacts named in the selective-gate hand-off.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import ensure_source_path

ensure_source_path()

from cacheblend_gpt_oss.vllm_compat.v0_19_1.selective_registry import (  # noqa: E402
    SelectiveGateEvidence,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--full-prefill", type=Path, required=True)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--yarn", type=Path, required=True)
    parser.add_argument("--hybrid-sink", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evidence = SelectiveGateEvidence.from_artifact_paths(
        runtime=args.runtime,
        full_prefill=args.full_prefill,
        transfer=args.transfer,
        yarn=args.yarn,
        hybrid_sink=args.hybrid_sink,
    )
    rendered = json.dumps(evidence.to_dict(), allow_nan=False, indent=2) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as output:
            output.write(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
