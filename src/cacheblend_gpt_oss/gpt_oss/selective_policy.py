# SPDX-License-Identifier: Apache-2.0
"""Deterministic CPU selection policy for the future CacheBlend data plane.

The public CacheBlend snapshot's Llama path performs a check-layer comparison:
it ranks cached-row K/V differences, recomputes a configured fraction of the
largest differences, and always recomputes a suffix before propagating the
selected rows through later layers.  This module preserves that *planning*
idea without copying the old vLLM implementation or importing Torch/vLLM.

Reference snapshot (conceptual behavior only), pinned by this repository:
``55ad02675939f783a38d579393527d218a7fd581``

* https://github.com/YaoJiayi/CacheBlend/blob/55ad02675939f783a38d579393527d218a7fd581/vllm_blend/vllm/model_executor/models/llama.py#L300-L365
* https://github.com/YaoJiayi/CacheBlend/blob/55ad02675939f783a38d579393527d218a7fd581/vllm_blend/vllm/attention/backends/xformers.py#L220-L305

The current connector never consumes this policy.  It is deliberately
fail-closed and records an immutable :class:`ForwardRowPlan` so a future
GPT-OSS model/backend can compare each ratio against full prefill before any
selective ratio is enabled in production.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
from typing import NoReturn

from cacheblend_gpt_oss.gpt_oss.layout import GPT_OSS_NUM_LAYERS
from cacheblend_gpt_oss.gpt_oss.selective import ForwardRowPlan
from cacheblend_gpt_oss.planner.models import TokenRange


class SelectionPolicyErrorCode(str, Enum):
    """Bounded failures for check-layer selection inputs."""

    INVALID_PROMPT_LENGTH = "invalid_prompt_length"
    INVALID_CHECK_LAYER = "invalid_check_layer"
    INVALID_RECOMPUTE_RATIO = "invalid_recompute_ratio"
    INVALID_SUFFIX_LENGTH = "invalid_suffix_length"
    INVALID_IMPORTANCE_SCORES = "invalid_importance_scores"
    INVALID_CACHE_RANGES = "invalid_cache_ranges"
    NON_CANONICAL_CACHE_RANGES = "non_canonical_cache_ranges"
    OVERLAPPING_CACHE_RANGES = "overlapping_cache_ranges"
    CACHE_RANGE_OUT_OF_BOUNDS = "cache_range_out_of_bounds"
    INVALID_RATIO_SWEEP = "invalid_ratio_sweep"
    INCONSISTENT_SWEEP = "inconsistent_sweep"
    INVALID_MEASUREMENT = "invalid_measurement"
    MEASUREMENT_MISMATCH = "measurement_mismatch"


class SelectionPolicyError(ValueError):
    """Fail-closed error without request, document, or token identifiers."""

    def __init__(self, code: SelectionPolicyErrorCode) -> None:
        self.code = code
        super().__init__(f"CacheBlend selection policy failure: {code.value}")


def _fail(code: SelectionPolicyErrorCode) -> NoReturn:
    raise SelectionPolicyError(code)


def _validate_prompt_tokens(prompt_tokens: object) -> int:
    if (
        isinstance(prompt_tokens, bool)
        or not isinstance(prompt_tokens, int)
        or prompt_tokens < 0
        or prompt_tokens > 131_072
    ):
        _fail(SelectionPolicyErrorCode.INVALID_PROMPT_LENGTH)
    return prompt_tokens


def _validate_cache_ranges(
    prompt_tokens: int,
    cache_ranges: Sequence[TokenRange],
) -> tuple[TokenRange, ...]:
    try:
        normalized = tuple(cache_ranges)
    except TypeError:
        _fail(SelectionPolicyErrorCode.INVALID_CACHE_RANGES)
    if any(not isinstance(item, TokenRange) or len(item) == 0 for item in normalized):
        _fail(SelectionPolicyErrorCode.INVALID_CACHE_RANGES)
    if any(item.end > prompt_tokens for item in normalized):
        _fail(SelectionPolicyErrorCode.CACHE_RANGE_OUT_OF_BOUNDS)
    if tuple(sorted(normalized, key=lambda item: (item.start, item.end))) != normalized:
        _fail(SelectionPolicyErrorCode.NON_CANONICAL_CACHE_RANGES)
    if any(left.overlaps(right) for left, right in pairwise(normalized)):
        _fail(SelectionPolicyErrorCode.OVERLAPPING_CACHE_RANGES)
    return normalized


def _merge_ranges(ranges: Sequence[TokenRange]) -> tuple[TokenRange, ...]:
    """Merge sorted/unsorted ranges, including adjacent singleton rows."""

    ordered = sorted(ranges, key=lambda item: (item.start, item.end))
    merged: list[TokenRange] = []
    for item in ordered:
        if not merged or item.start > merged[-1].end:
            merged.append(item)
        elif item.end > merged[-1].end:
            merged[-1] = TokenRange(merged[-1].start, item.end)
    return tuple(merged)


def _complement(
    prompt_tokens: int,
    cache_ranges: Sequence[TokenRange],
) -> tuple[TokenRange, ...]:
    ranges: list[TokenRange] = []
    cursor = 0
    for item in cache_ranges:
        if cursor < item.start:
            ranges.append(TokenRange(cursor, item.start))
        cursor = item.end
    if cursor < prompt_tokens:
        ranges.append(TokenRange(cursor, prompt_tokens))
    return tuple(ranges)


def _contains(ranges: Sequence[TokenRange], position: int) -> bool:
    return any(item.start <= position < item.end for item in ranges)


@dataclass(frozen=True, slots=True)
class SelectionPolicyResult:
    """Immutable ratio result consumed by a future worker-local row context.

    ``recompute_ranges`` is the selective row set applied to every layer
    *after* ``check_layer``.  Layers ``0..check_layer`` inclusive always
    recompute the full prompt in ``row_plan``, because CacheBlend's importance
    scores are measured at the check layer and require a correct prefix
    through it.  ``layer_token_rows_recomputed``/``layer_token_rows_avoided``
    report work summed across all 24 layers (layer-token accounting), not
    per-layer prompt-token accounting.
    """

    check_layer: int
    recompute_ratio: float
    suffix_tokens: int
    candidate_cached_ranges: tuple[TokenRange, ...]
    recompute_ranges: tuple[TokenRange, ...]
    selected_cached_rows: tuple[int, ...]
    row_plan: ForwardRowPlan

    @property
    def prompt_tokens(self) -> int:
        return self.row_plan.prompt_tokens

    @property
    def cached_ranges(self) -> tuple[TokenRange, ...]:
        """Verified candidate rows left untouched by the selective layers."""

        return _complement(self.prompt_tokens, self.recompute_ranges)

    @property
    def recompute_tokens_per_layer(self) -> int:
        """Rows recomputed by each selective layer (after ``check_layer``)."""

        return sum(len(item) for item in self.recompute_ranges)

    @property
    def cached_tokens_per_layer(self) -> int:
        """Rows left cached by each selective layer (after ``check_layer``)."""

        return self.prompt_tokens - self.recompute_tokens_per_layer

    @property
    def recompute_fraction(self) -> float:
        if self.prompt_tokens == 0:
            return 0.0
        return self.recompute_tokens_per_layer / self.prompt_tokens

    @property
    def layer_token_rows_recomputed(self) -> int:
        """Total recomputed rows summed across all 24 layers."""

        return self.row_plan.recompute_tokens

    @property
    def layer_token_rows_avoided(self) -> int:
        """Rows avoided vs. 100% recomputation at every layer, summed."""

        full = GPT_OSS_NUM_LAYERS * self.prompt_tokens
        return full - self.layer_token_rows_recomputed


@dataclass(frozen=True, slots=True)
class SelectionMeasurement:
    """Measured error/work data for one ratio, supplied by an external runner."""

    max_abs_error: float
    mean_abs_error: float
    selective_latency_seconds: float

    def __post_init__(self) -> None:
        values = (
            self.max_abs_error,
            self.mean_abs_error,
            self.selective_latency_seconds,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in values
        ):
            _fail(SelectionPolicyErrorCode.INVALID_MEASUREMENT)


@dataclass(frozen=True, slots=True)
class SelectionSweepPoint:
    """One deterministic policy result and optional measured GPU evidence."""

    result: SelectionPolicyResult
    measurement: SelectionMeasurement | None = None

    @property
    def ratio(self) -> float:
        return self.result.recompute_ratio

    @property
    def recompute_fraction(self) -> float:
        return self.result.recompute_fraction


@dataclass(frozen=True, slots=True)
class SelectionSweep:
    """Ordered ratio/work curve; error fields require explicit measurements."""

    points: tuple[SelectionSweepPoint, ...]

    def __post_init__(self) -> None:
        try:
            points = tuple(self.points)
        except TypeError:
            _fail(SelectionPolicyErrorCode.INVALID_RATIO_SWEEP)
        if not points or any(
            not isinstance(point, SelectionSweepPoint) for point in points
        ):
            _fail(SelectionPolicyErrorCode.INVALID_RATIO_SWEEP)
        ratios = tuple(point.ratio for point in points)
        if any(left <= right for left, right in pairwise(ratios)):
            _fail(SelectionPolicyErrorCode.INVALID_RATIO_SWEEP)
        first = points[0].result
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
            for point in points[1:]
        ):
            _fail(SelectionPolicyErrorCode.INCONSISTENT_SWEEP)
        object.__setattr__(self, "points", points)

    @property
    def ratios(self) -> tuple[float, ...]:
        return tuple(point.ratio for point in self.points)

    @property
    def work_curve(self) -> tuple[tuple[float, float], ...]:
        """Return ``(ratio, recomputed fraction)`` without inventing errors."""

        return tuple(
            (point.ratio, point.recompute_fraction) for point in self.points
        )

    @property
    def error_curve(self) -> tuple[tuple[float, float, float], ...]:
        """Return ``(ratio, max error, mean error)`` after all measurements."""

        if any(point.measurement is None for point in self.points):
            _fail(SelectionPolicyErrorCode.MEASUREMENT_MISMATCH)
        return tuple(
            (
                point.ratio,
                point.measurement.max_abs_error,  # type: ignore[union-attr]
                point.measurement.mean_abs_error,  # type: ignore[union-attr]
            )
            for point in self.points
        )

    @property
    def latency_curve(self) -> tuple[tuple[float, float], ...]:
        """Return ``(ratio, selective latency)`` after all measurements."""

        if any(point.measurement is None for point in self.points):
            _fail(SelectionPolicyErrorCode.MEASUREMENT_MISMATCH)
        return tuple(
            (
                point.ratio,
                point.measurement.selective_latency_seconds,  # type: ignore[union-attr]
            )
            for point in self.points
        )

    def with_measurements(
        self,
        measurements: Sequence[SelectionMeasurement],
    ) -> SelectionSweep:
        """Attach externally measured results in exactly the sweep's order."""

        try:
            normalized = tuple(measurements)
        except TypeError:
            _fail(SelectionPolicyErrorCode.MEASUREMENT_MISMATCH)
        if len(normalized) != len(self.points) or any(
            not isinstance(measurement, SelectionMeasurement)
            for measurement in normalized
        ):
            _fail(SelectionPolicyErrorCode.MEASUREMENT_MISMATCH)
        return SelectionSweep(
            tuple(
                SelectionSweepPoint(point.result, measurement)
                for point, measurement in zip(self.points, normalized, strict=True)
            )
        )


class CacheBlendSelectionPolicy:
    """Choose check-layer rows with deterministic top-score tie breaking.

    ``recompute_ratio`` applies only to eligible cached rows outside the
    forced suffix.  Every non-cached target row is always recomputed.  This
    makes the accounting explicit: lowering the ratio cannot accidentally
    skip prompt tokens for which no verified KV exists.
    """

    def select(
        self,
        *,
        prompt_tokens: object,
        cache_ranges: Sequence[TokenRange],
        importance_scores: Sequence[object],
        check_layer: object,
        recompute_ratio: object,
        suffix_tokens: object,
    ) -> SelectionPolicyResult:
        prompt = _validate_prompt_tokens(prompt_tokens)
        if (
            isinstance(check_layer, bool)
            or not isinstance(check_layer, int)
            or not 0 <= check_layer < GPT_OSS_NUM_LAYERS
        ):
            _fail(SelectionPolicyErrorCode.INVALID_CHECK_LAYER)
        if (
            isinstance(recompute_ratio, bool)
            or not isinstance(recompute_ratio, int | float)
            or not math.isfinite(float(recompute_ratio))
            or not 0.0 <= float(recompute_ratio) <= 1.0
        ):
            _fail(SelectionPolicyErrorCode.INVALID_RECOMPUTE_RATIO)
        ratio = float(recompute_ratio)
        if (
            isinstance(suffix_tokens, bool)
            or not isinstance(suffix_tokens, int)
            or suffix_tokens < 0
            or suffix_tokens > prompt
        ):
            _fail(SelectionPolicyErrorCode.INVALID_SUFFIX_LENGTH)
        suffix = suffix_tokens
        try:
            scores = tuple(importance_scores)
        except TypeError:
            _fail(SelectionPolicyErrorCode.INVALID_IMPORTANCE_SCORES)
        if len(scores) != prompt:
            _fail(SelectionPolicyErrorCode.INVALID_IMPORTANCE_SCORES)
        normalized_scores: list[float] = []
        for score in scores:
            if (
                isinstance(score, bool)
                or not isinstance(score, int | float)
                or not math.isfinite(float(score))
                or float(score) < 0.0
            ):
                _fail(SelectionPolicyErrorCode.INVALID_IMPORTANCE_SCORES)
            normalized_scores.append(float(score))

        cached = _validate_cache_ranges(prompt, cache_ranges)
        forced_suffix = (
            (TokenRange(prompt - suffix, prompt),) if suffix else ()
        )
        non_cached = _complement(prompt, cached)
        eligible = [
            position
            for item in cached
            for position in range(item.start, item.end)
            if not _contains(forced_suffix, position)
        ]
        budget = int(len(eligible) * ratio)
        ranked = sorted(
            eligible,
            key=lambda position: (-normalized_scores[position], position),
        )
        selected = tuple(sorted(ranked[:budget]))
        selected_ranges = tuple(
            TokenRange(position, position + 1) for position in selected
        )
        recompute = _merge_ranges((*non_cached, *forced_suffix, *selected_ranges))
        # Layers 0..check_layer inclusive recompute the full prompt: the
        # check-layer importance scores are measured at that layer and need a
        # correct prefix through it.  Only layers after check_layer apply the
        # selective (non-cached + suffix + top-score) recompute set.
        full_prompt_range = (TokenRange(0, prompt),) if prompt else ()
        plan = ForwardRowPlan.from_recompute_ranges(
            prompt,
            tuple(
                full_prompt_range if layer_index <= check_layer else recompute
                for layer_index in range(GPT_OSS_NUM_LAYERS)
            ),
        )
        return SelectionPolicyResult(
            check_layer=check_layer,
            recompute_ratio=ratio,
            suffix_tokens=suffix,
            candidate_cached_ranges=cached,
            recompute_ranges=recompute,
            selected_cached_rows=selected,
            row_plan=plan,
        )

    def sweep(
        self,
        *,
        prompt_tokens: object,
        cache_ranges: Sequence[TokenRange],
        importance_scores: Sequence[object],
        check_layer: object,
        recompute_ratios: Sequence[object],
        suffix_tokens: object,
    ) -> SelectionSweep:
        """Build a descending ratio sweep without claiming measured accuracy."""

        try:
            ratios = tuple(recompute_ratios)
        except TypeError:
            _fail(SelectionPolicyErrorCode.INVALID_RATIO_SWEEP)
        if not ratios:
            _fail(SelectionPolicyErrorCode.INVALID_RATIO_SWEEP)
        points: list[SelectionSweepPoint] = []
        for ratio in ratios:
            result = self.select(
                prompt_tokens=prompt_tokens,
                cache_ranges=cache_ranges,
                importance_scores=importance_scores,
                check_layer=check_layer,
                recompute_ratio=ratio,
                suffix_tokens=suffix_tokens,
            )
            points.append(SelectionSweepPoint(result))
        return SelectionSweep(tuple(points))


__all__ = [
    "CacheBlendSelectionPolicy",
    "SelectionMeasurement",
    "SelectionPolicyError",
    "SelectionPolicyErrorCode",
    "SelectionPolicyResult",
    "SelectionSweep",
    "SelectionSweepPoint",
]
