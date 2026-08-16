#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate one offline BrowseComp-Plus append-only transfer smoke run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import ensure_source_path

ensure_source_path()

from cacheblend_gpt_oss.benchmark.browsecomp import (  # noqa: E402
    BrowseCompEvidenceError,
    BrowseCompEvidenceErrorCode,
    failed_browsecomp_report,
    runtime_identity_from_dict,
    validate_browsecomp_append_only,
    validate_browsecomp_selective_append_only,
)


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant: {value}")


def _objects_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _read_json(path: Path) -> object:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise BrowseCompEvidenceError(BrowseCompEvidenceErrorCode.FILE_ERROR) from None
    try:
        return json.loads(
            text,
            object_pairs_hook=_objects_without_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        raise BrowseCompEvidenceError(
            BrowseCompEvidenceErrorCode.INVALID_JSON
        ) from None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise BrowseCompEvidenceError(BrowseCompEvidenceErrorCode.FILE_ERROR) from None


def _write_create_only(path: Path, rendered: str) -> None:
    try:
        with path.open("x", encoding="utf-8") as output:
            output.write(rendered)
    except FileExistsError:
        raise BrowseCompEvidenceError(
            BrowseCompEvidenceErrorCode.OUTPUT_EXISTS
        ) from None
    except (OSError, UnicodeError):
        raise BrowseCompEvidenceError(BrowseCompEvidenceErrorCode.FILE_ERROR) from None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-record", type=Path, required=True)
    parser.add_argument("--metrics-before", type=Path, required=True)
    parser.add_argument("--metrics-after", type=Path, required=True)
    parser.add_argument("--runtime-identity", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional create-only sanitized JSON report",
    )
    parser.add_argument(
        "--require-passed",
        action="store_true",
        help="return nonzero when the evidence gates do not pass",
    )
    parser.add_argument(
        "--selective",
        action="store_true",
        help=(
            "validate the transfer_selective contract and require positive, "
            "fully reconciled layer-token work"
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    runtime = None
    output_error = False
    try:
        runtime = runtime_identity_from_dict(_read_json(args.runtime_identity))
        validator = (
            validate_browsecomp_selective_append_only
            if args.selective
            else validate_browsecomp_append_only
        )
        report = validator(
            _read_json(args.run_record),
            _read_text(args.metrics_before),
            _read_text(args.metrics_after),
            runtime,
        )
    except BrowseCompEvidenceError as error:
        report = failed_browsecomp_report(runtime, error, selective=args.selective)

    rendered = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        try:
            _write_create_only(args.output, rendered)
        except BrowseCompEvidenceError as error:
            output_error = True
            report = failed_browsecomp_report(
                runtime,
                error,
                selective=args.selective,
            )
            rendered = (
                json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
            )
    print(rendered, end="")
    if output_error:
        return 2
    if args.require_passed and report.get("passed") is not True:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
