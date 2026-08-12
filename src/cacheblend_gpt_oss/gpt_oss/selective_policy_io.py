# SPDX-License-Identifier: Apache-2.0
"""Strict JSON artifacts for future GPT-OSS selective-ratio experiments.

The selection policy is intentionally CPU-only and does not retain prompt
tokens, fingerprints, or request identifiers.  This module gives a future
``solab-g3`` worker a small, reproducible hand-off format for its ratio/work
curve and measured correctness data.  A curve is only considered measured when
every point contains an explicit error and latency observation; missing data is
never represented as zero.

The format is versioned independently from vLLM and LMCache.  It is an
experiment artifact, not a serving protocol, and writing one does not enable
selective execution in the live connector.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from enum import Enum
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from typing import NoReturn

from cacheblend_gpt_oss.gpt_oss.layout import (
    GPT_OSS_MAX_CONTEXT_TOKENS,
    GPT_OSS_NUM_LAYERS,
)
from cacheblend_gpt_oss.gpt_oss.selective import ForwardRowPlan
from cacheblend_gpt_oss.gpt_oss.selective_policy import (
    SelectionMeasurement,
    SelectionPolicyError,
    SelectionPolicyResult,
    SelectionSweep,
    SelectionSweepPoint,
)
from cacheblend_gpt_oss.planner.models import TokenRange

SELECTION_SWEEP_SCHEMA_VERSION = 1
SELECTION_SWEEP_KIND = "cacheblend_gpt_oss_selection_sweep"


class SelectionSweepIoErrorCode(str, Enum):
    """Bounded failures for selection-sweep artifact I/O."""

    INVALID_SCHEMA = "invalid_schema"
    INVALID_JSON = "invalid_json"
    INVALID_POINT = "invalid_point"
    INVALID_RANGE = "invalid_range"
    POINT_MISMATCH = "point_mismatch"
    INCONSISTENT_SWEEP = "inconsistent_sweep"
    INCOMPLETE_MEASUREMENTS = "incomplete_measurements"
    FILE_EXISTS = "file_exists"
    FILE_ERROR = "file_error"


class SelectionSweepIoError(ValueError):
    """Fail-closed artifact error without prompt or request identifiers."""

    def __init__(self, code: SelectionSweepIoErrorCode) -> None:
        self.code = code
        super().__init__(f"CacheBlend selection-sweep I/O failure: {code.value}")


def _fail(code: SelectionSweepIoErrorCode) -> NoReturn:
    raise SelectionSweepIoError(code)


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail(SelectionSweepIoErrorCode.INVALID_SCHEMA)
    return value


def _exact_mapping(
    value: object,
    keys: set[str],
    code: SelectionSweepIoErrorCode = SelectionSweepIoErrorCode.INVALID_SCHEMA,
) -> Mapping[str, object]:
    mapping = _mapping(value)
    if set(mapping) != keys:
        _fail(code)
    return mapping


def _bounded_int(value: object, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(SelectionSweepIoErrorCode.INVALID_POINT)
    if maximum is not None and value > maximum:
        _fail(SelectionSweepIoErrorCode.INVALID_POINT)
    return value


def _bounded_float(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        _fail(SelectionSweepIoErrorCode.INVALID_POINT)
    return float(value)


def _ranges(value: object, prompt_tokens: int) -> tuple[TokenRange, ...]:
    if not isinstance(value, list):
        _fail(SelectionSweepIoErrorCode.INVALID_RANGE)
    parsed: list[TokenRange] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            _fail(SelectionSweepIoErrorCode.INVALID_RANGE)
        start = _bounded_int(item[0], maximum=prompt_tokens)
        end = _bounded_int(item[1], maximum=prompt_tokens)
        if end <= start:
            _fail(SelectionSweepIoErrorCode.INVALID_RANGE)
        try:
            parsed.append(TokenRange(start, end))
        except (TypeError, ValueError):
            _fail(SelectionSweepIoErrorCode.INVALID_RANGE)
    normalized = tuple(parsed)
    if tuple(sorted(normalized, key=lambda item: (item.start, item.end))) != normalized:
        _fail(SelectionSweepIoErrorCode.INVALID_RANGE)
    if any(left.overlaps(right) for left, right in pairwise(normalized)):
        _fail(SelectionSweepIoErrorCode.INVALID_RANGE)
    return normalized


def _range_list(ranges: Sequence[TokenRange]) -> list[list[int]]:
    return [[item.start, item.end] for item in ranges]


def _complement(
    prompt_tokens: int, ranges: Sequence[TokenRange]
) -> tuple[TokenRange, ...]:
    result: list[TokenRange] = []
    cursor = 0
    for item in ranges:
        if cursor < item.start:
            result.append(TokenRange(cursor, item.start))
        cursor = item.end
    if cursor < prompt_tokens:
        result.append(TokenRange(cursor, prompt_tokens))
    return tuple(result)


def _merge_ranges(ranges: Sequence[TokenRange]) -> tuple[TokenRange, ...]:
    ordered = sorted(ranges, key=lambda item: (item.start, item.end))
    merged: list[TokenRange] = []
    for item in ordered:
        if not merged or item.start > merged[-1].end:
            merged.append(item)
        elif item.end > merged[-1].end:
            merged[-1] = TokenRange(merged[-1].start, item.end)
    return tuple(merged)


def _selected_rows(
    value: object,
    prompt_tokens: int,
    candidate_ranges: Sequence[TokenRange],
    suffix_tokens: int,
) -> tuple[int, ...]:
    if not isinstance(value, list):
        _fail(SelectionSweepIoErrorCode.INVALID_POINT)
    rows = tuple(
        _bounded_int(item, maximum=prompt_tokens - 1) for item in value
    )
    if rows != tuple(sorted(set(rows))):
        _fail(SelectionSweepIoErrorCode.INVALID_POINT)
    suffix_start = prompt_tokens - suffix_tokens
    if any(
        not any(
            token_range.start <= row < token_range.end
            for token_range in candidate_ranges
        )
        or row >= suffix_start
        for row in rows
    ):
        _fail(SelectionSweepIoErrorCode.POINT_MISMATCH)
    return rows


def _measurement(value: object) -> SelectionMeasurement | None:
    if value is None:
        return None
    mapping = _exact_mapping(
        value,
        {"max_abs_error", "mean_abs_error", "selective_latency_seconds"},
        SelectionSweepIoErrorCode.INVALID_POINT,
    )
    try:
        return SelectionMeasurement(
            _bounded_float(mapping["max_abs_error"]),
            _bounded_float(mapping["mean_abs_error"]),
            _bounded_float(mapping["selective_latency_seconds"]),
        )
    except (TypeError, ValueError, SelectionSweepIoError):
        _fail(SelectionSweepIoErrorCode.INVALID_POINT)


def _point_to_dict(point: SelectionSweepPoint) -> dict[str, object]:
    result = point.result
    measurement = point.measurement
    return {
        "candidate_cached_ranges": _range_list(result.candidate_cached_ranges),
        "check_layer": result.check_layer,
        "measurement": (
            None
            if measurement is None
            else {
                "max_abs_error": measurement.max_abs_error,
                "mean_abs_error": measurement.mean_abs_error,
                "selective_latency_seconds": measurement.selective_latency_seconds,
            }
        ),
        "prompt_tokens": result.prompt_tokens,
        "recompute_ranges": _range_list(result.recompute_ranges),
        "recompute_ratio": result.recompute_ratio,
        "selected_cached_rows": list(result.selected_cached_rows),
        "suffix_tokens": result.suffix_tokens,
    }


def _validate_sweep_context(sweep: SelectionSweep) -> None:
    first = sweep.points[0].result
    context = (
        first.prompt_tokens,
        first.check_layer,
        first.suffix_tokens,
        first.candidate_cached_ranges,
    )
    if any(
        (
            point.result.prompt_tokens,
            point.result.check_layer,
            point.result.suffix_tokens,
            point.result.candidate_cached_ranges,
        )
        != context
        for point in sweep.points[1:]
    ):
        _fail(SelectionSweepIoErrorCode.INCONSISTENT_SWEEP)
    measurements = tuple(point.measurement is not None for point in sweep.points)
    if any(measurements) and not all(measurements):
        _fail(SelectionSweepIoErrorCode.INCOMPLETE_MEASUREMENTS)


def selection_sweep_to_dict(sweep: SelectionSweep) -> dict[str, object]:
    """Return the strict, non-sensitive mapping used by the artifact format."""

    if not isinstance(sweep, SelectionSweep):
        _fail(SelectionSweepIoErrorCode.INVALID_SCHEMA)
    _validate_sweep_context(sweep)
    payload = {
        "kind": SELECTION_SWEEP_KIND,
        "points": [_point_to_dict(point) for point in sweep.points],
        "schema_version": SELECTION_SWEEP_SCHEMA_VERSION,
    }
    # Dataclasses are intentionally lightweight and do not duplicate every
    # cross-field invariant.  Parse the emitted mapping once before exposing it
    # so a hand-constructed malformed result cannot become evidence.
    selection_sweep_from_dict(payload)
    return payload


def selection_sweep_from_dict(data: object) -> SelectionSweep:
    """Parse and independently validate one selection-sweep artifact."""

    root = _exact_mapping(data, {"kind", "points", "schema_version"})
    if (
        root["kind"] != SELECTION_SWEEP_KIND
        or isinstance(root["schema_version"], bool)
        or not isinstance(root["schema_version"], int)
        or root["schema_version"] != SELECTION_SWEEP_SCHEMA_VERSION
        or not isinstance(root["points"], list)
        or not root["points"]
    ):
        _fail(SelectionSweepIoErrorCode.INVALID_SCHEMA)

    points: list[SelectionSweepPoint] = []
    for raw in root["points"]:
        point = _exact_mapping(
            raw,
            {
                "candidate_cached_ranges",
                "check_layer",
                "measurement",
                "prompt_tokens",
                "recompute_ranges",
                "recompute_ratio",
                "selected_cached_rows",
                "suffix_tokens",
            },
            SelectionSweepIoErrorCode.INVALID_POINT,
        )
        prompt = _bounded_int(
            point["prompt_tokens"], maximum=GPT_OSS_MAX_CONTEXT_TOKENS
        )
        check_layer = _bounded_int(point["check_layer"], maximum=23)
        ratio = _bounded_float(point["recompute_ratio"])
        if not 0.0 <= ratio <= 1.0:
            _fail(SelectionSweepIoErrorCode.INVALID_POINT)
        suffix = _bounded_int(point["suffix_tokens"], maximum=prompt)
        candidates = _ranges(point["candidate_cached_ranges"], prompt)
        recompute = _ranges(point["recompute_ranges"], prompt)
        selected = _selected_rows(
            point["selected_cached_rows"], prompt, candidates, suffix
        )
        suffix_start = prompt - suffix
        eligible_count = sum(
            sum(
                1
                for row in range(token_range.start, token_range.end)
                if row < suffix_start
            )
            for token_range in candidates
        )
        if len(selected) != int(eligible_count * ratio):
            _fail(SelectionSweepIoErrorCode.POINT_MISMATCH)
        forced_suffix = (TokenRange(prompt - suffix, prompt),) if suffix else ()
        expected_recompute = _merge_ranges(
            (
                *_complement(prompt, candidates),
                *forced_suffix,
                *(TokenRange(row, row + 1) for row in selected),
            )
        )
        if recompute != expected_recompute:
            _fail(SelectionSweepIoErrorCode.POINT_MISMATCH)
        try:
            row_plan = ForwardRowPlan.from_recompute_ranges(
                prompt, tuple(recompute for _ in range(GPT_OSS_NUM_LAYERS))
            )
            result = SelectionPolicyResult(
                check_layer=check_layer,
                recompute_ratio=ratio,
                suffix_tokens=suffix,
                candidate_cached_ranges=candidates,
                recompute_ranges=recompute,
                selected_cached_rows=selected,
                row_plan=row_plan,
            )
        except (TypeError, ValueError, SelectionPolicyError):
            _fail(SelectionSweepIoErrorCode.POINT_MISMATCH)
        points.append(SelectionSweepPoint(result, _measurement(point["measurement"])))
    try:
        sweep = SelectionSweep(tuple(points))
    except (TypeError, ValueError, SelectionPolicyError):
        _fail(SelectionSweepIoErrorCode.INVALID_POINT)
    _validate_sweep_context(sweep)
    return sweep


def canonical_selection_sweep_bytes(sweep: SelectionSweep) -> bytes:
    """Return deterministic UTF-8 JSON bytes for hashing and storage."""

    return json.dumps(
        selection_sweep_to_dict(sweep),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def selection_sweep_digest(sweep: SelectionSweep) -> str:
    """Return the SHA-256 digest of canonical artifact bytes."""

    return sha256(canonical_selection_sweep_bytes(sweep)).hexdigest()


def read_selection_sweep(path: Path) -> SelectionSweep:
    """Read one artifact and map filesystem/JSON failures to bounded codes."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _fail(SelectionSweepIoErrorCode.INVALID_JSON)
    except (OSError, UnicodeError, TypeError):
        _fail(SelectionSweepIoErrorCode.FILE_ERROR)
    return selection_sweep_from_dict(data)


def write_selection_sweep(path: Path, sweep: SelectionSweep) -> None:
    """Create one canonical artifact without overwriting prior evidence."""

    try:
        with path.open("xb") as output:
            output.write(canonical_selection_sweep_bytes(sweep) + b"\n")
    except FileExistsError:
        _fail(SelectionSweepIoErrorCode.FILE_EXISTS)
    except OSError:
        _fail(SelectionSweepIoErrorCode.FILE_ERROR)


__all__ = [
    "SELECTION_SWEEP_KIND",
    "SELECTION_SWEEP_SCHEMA_VERSION",
    "SelectionSweepIoError",
    "SelectionSweepIoErrorCode",
    "canonical_selection_sweep_bytes",
    "read_selection_sweep",
    "selection_sweep_digest",
    "selection_sweep_from_dict",
    "selection_sweep_to_dict",
    "write_selection_sweep",
]
