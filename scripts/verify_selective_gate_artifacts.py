#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify that reviewed selective-gate files still match a digest bundle.

This command checks identity and freshness only.  It does not approve the
M3--M5 contents, register a vLLM plugin, or enable selective execution.
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
    SELECTIVE_GATE_EVIDENCE_SCHEMA_VERSION,
    SelectiveGateEvidence,
    SelectiveRegistrationError,
)

_MAX_EVIDENCE_BYTES = 1024 * 1024


def _read_evidence(path: Path) -> SelectiveGateEvidence:
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
    ):
        raise ValueError("invalid_evidence")
    try:
        stat = path.stat()
        if stat.st_size <= 0 or stat.st_size > _MAX_EVIDENCE_BYTES:
            raise ValueError("invalid_evidence")
        with path.open("rb") as handle:
            raw = handle.read(_MAX_EVIDENCE_BYTES + 1)
        if len(raw) != stat.st_size or len(raw) > _MAX_EVIDENCE_BYTES:
            raise ValueError("invalid_evidence")
        after = path.stat()
        if (
            after.st_size != stat.st_size
            or after.st_mtime_ns != stat.st_mtime_ns
            or after.st_ino != stat.st_ino
            or after.st_dev != stat.st_dev
        ):
            raise ValueError("invalid_evidence")
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid_evidence") from exc
    return SelectiveGateEvidence.from_dict(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--full-prefill", type=Path, required=True)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--yarn", type=Path, required=True)
    parser.add_argument("--hybrid-sink", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        evidence = _read_evidence(args.evidence)
        evidence.verify_artifact_paths(
            runtime=args.runtime,
            full_prefill=args.full_prefill,
            transfer=args.transfer,
            yarn=args.yarn,
            hybrid_sink=args.hybrid_sink,
        )
    except (SelectiveRegistrationError, ValueError, TypeError, OSError) as exc:
        parser.error("invalid_evidence")
        raise AssertionError("argparse.error must exit") from exc
    report = {
        "schema_version": SELECTIVE_GATE_EVIDENCE_SCHEMA_VERSION,
        "evidence": evidence.to_dict(),
        "verified": True,
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
