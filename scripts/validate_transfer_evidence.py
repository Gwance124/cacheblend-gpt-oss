#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate and summarize one real GPT-OSS transfer-evidence sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cacheblend_gpt_oss.correctness import (
    read_transfer_evidence,
    transfer_evidence_digest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evidence = read_transfer_evidence(args.input)
    report = {
        "schema_version": evidence.schema_version,
        "evidence_digest": transfer_evidence_digest(evidence),
        "loaded_tokens": evidence.loaded_tokens,
        "target_prompt_tokens": evidence.target_prompt_tokens,
        "recomputed_tokens": evidence.recomputed_tokens,
        "prefill_tokens_avoided": evidence.prefill_tokens_avoided,
        "sliding_layers": len(evidence.sliding_layers),
        "full_layers": len(evidence.full_layers),
        "all_layers_loaded_and_overwritten": (
            evidence.all_layers_loaded_and_overwritten
        ),
        "passed": evidence.all_layers_loaded_and_overwritten,
    }
    rendered = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as output:
            output.write(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
