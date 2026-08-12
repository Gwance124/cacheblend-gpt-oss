# SPDX-License-Identifier: Apache-2.0
"""Canonical persistence for a pre-CacheBlend baseline tolerance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cacheblend_gpt_oss.correctness.evaluate import FrozenFullPrefillTolerance

TOLERANCE_SCHEMA_VERSION = 1
_FIELDS = {
    "schema_version",
    "reference_artifact_digest",
    "repeat_artifact_digest",
    "baseline_max_abs_error",
    "baseline_mean_abs_error",
    "allowed_max_abs_error",
    "allowed_mean_abs_error",
}


def tolerance_to_dict(
    tolerance: FrozenFullPrefillTolerance,
) -> dict[str, int | float | str]:
    return {
        "schema_version": TOLERANCE_SCHEMA_VERSION,
        "reference_artifact_digest": tolerance.reference_artifact_digest,
        "repeat_artifact_digest": tolerance.repeat_artifact_digest,
        "baseline_max_abs_error": tolerance.baseline_max_abs_error,
        "baseline_mean_abs_error": tolerance.baseline_mean_abs_error,
        "allowed_max_abs_error": tolerance.allowed_max_abs_error,
        "allowed_mean_abs_error": tolerance.allowed_mean_abs_error,
    }


def tolerance_from_dict(data: object) -> FrozenFullPrefillTolerance:
    if not isinstance(data, dict) or set(data) != _FIELDS:
        raise ValueError("invalid frozen tolerance schema")
    if (
        isinstance(data["schema_version"], bool)
        or not isinstance(data["schema_version"], int)
        or data["schema_version"] != TOLERANCE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported frozen tolerance schema")
    values: dict[str, Any] = dict(data)
    del values["schema_version"]
    try:
        return FrozenFullPrefillTolerance(**values)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid frozen tolerance values") from exc


def write_frozen_tolerance(
    path: Path, tolerance: FrozenFullPrefillTolerance
) -> None:
    encoded = json.dumps(
        tolerance_to_dict(tolerance),
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )
    with path.open("x", encoding="utf-8") as output:
        output.write(encoded + "\n")


def read_frozen_tolerance(path: Path) -> FrozenFullPrefillTolerance:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return tolerance_from_dict(data)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("could not read a valid frozen tolerance") from exc


__all__ = [
    "TOLERANCE_SCHEMA_VERSION",
    "read_frozen_tolerance",
    "tolerance_from_dict",
    "tolerance_to_dict",
    "write_frozen_tolerance",
]
