# SPDX-License-Identifier: Apache-2.0
"""CPU-only row-plan contract for the future GPT-OSS selective data plane.

This module deliberately does not import vLLM, Torch, CUDA, or model code.  It
describes the full-shaped row invariant that a future M6 model override and
attention backend must preserve.  The current connector does not consume this
context and continues to recompute every prompt token.

The contract is specific to the audited GPT-OSS-20B target: 24 transformer
layers and a 131,072-token context.  A row plan is not evidence that selective
execution is numerically correct; that remains gated on the M3--M5 GPU/model
results and the M6 stop/go experiment.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
from typing import NoReturn

from cacheblend_gpt_oss.gpt_oss.layout import (
    GPT_OSS_MAX_CONTEXT_TOKENS,
    GPT_OSS_NUM_LAYERS,
)
from cacheblend_gpt_oss.planner.models import TokenRange


class SelectivePlanErrorCode(str, Enum):
    """Bounded validation errors for row plans and forward context."""

    INVALID_PROMPT_LENGTH = "invalid_prompt_length"
    INVALID_LAYER_COUNT = "invalid_layer_count"
    INVALID_LAYER_INDEX = "invalid_layer_index"
    INVALID_RANGE = "invalid_range"
    RANGE_OUT_OF_BOUNDS = "range_out_of_bounds"
    OVERLAPPING_RANGES = "overlapping_ranges"
    NON_CANONICAL_RANGES = "non_canonical_ranges"
    ACTIVE_CONTEXT = "active_context"
    MISSING_CONTEXT = "missing_context"


class SelectivePlanError(ValueError):
    """Fail-closed row-plan error with a bounded machine-readable code."""

    def __init__(self, code: SelectivePlanErrorCode) -> None:
        self.code = code
        super().__init__(code.value)


def _fail(code: SelectivePlanErrorCode) -> NoReturn:
    raise SelectivePlanError(code)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_prompt_tokens(prompt_tokens: int) -> None:
    if (
        not _is_int(prompt_tokens)
        or prompt_tokens < 0
        or prompt_tokens > GPT_OSS_MAX_CONTEXT_TOKENS
    ):
        _fail(SelectivePlanErrorCode.INVALID_PROMPT_LENGTH)


def _normalize_ranges(
    prompt_tokens: int,
    ranges: Sequence[TokenRange],
) -> tuple[TokenRange, ...]:
    """Validate and canonicalize non-empty, sorted, non-overlapping ranges."""

    _validate_prompt_tokens(prompt_tokens)
    try:
        normalized = tuple(ranges)
    except TypeError:
        _fail(SelectivePlanErrorCode.INVALID_RANGE)

    for token_range in normalized:
        if not isinstance(token_range, TokenRange) or len(token_range) == 0:
            _fail(SelectivePlanErrorCode.INVALID_RANGE)
        if token_range.end > prompt_tokens:
            _fail(SelectivePlanErrorCode.RANGE_OUT_OF_BOUNDS)

    ordered = tuple(sorted(normalized, key=lambda value: (value.start, value.end)))
    if ordered != normalized:
        _fail(SelectivePlanErrorCode.NON_CANONICAL_RANGES)
    if any(left.overlaps(right) for left, right in pairwise(ordered)):
        _fail(SelectivePlanErrorCode.OVERLAPPING_RANGES)
    return ordered


def _complement(
    prompt_tokens: int,
    excluded: Sequence[TokenRange],
) -> tuple[TokenRange, ...]:
    """Return the canonical complement of ``excluded`` in prompt rows."""

    ranges: list[TokenRange] = []
    cursor = 0
    for token_range in excluded:
        if cursor < token_range.start:
            ranges.append(TokenRange(cursor, token_range.start))
        cursor = token_range.end
    if cursor < prompt_tokens:
        ranges.append(TokenRange(cursor, prompt_tokens))
    return tuple(ranges)


def _range_token_count(ranges: Sequence[TokenRange]) -> int:
    return sum(len(token_range) for token_range in ranges)


@dataclass(frozen=True, slots=True)
class LayerRowSelection:
    """Rows recomputed for one GPT-OSS layer.

    ``recompute_ranges`` is canonical and contains every row that the model
    must materialize for this layer.  Its complement is the only set a future
    backend may treat as accepted cached rows.  Keeping both sets derived from
    one bounded representation prevents gaps, overlaps, and out-of-range slot
    writes.
    """

    layer_index: int
    prompt_tokens: int
    recompute_ranges: tuple[TokenRange, ...]

    def __post_init__(self) -> None:
        if (
            not _is_int(self.layer_index)
            or not 0 <= self.layer_index < GPT_OSS_NUM_LAYERS
        ):
            _fail(SelectivePlanErrorCode.INVALID_LAYER_INDEX)
        _validate_prompt_tokens(self.prompt_tokens)
        object.__setattr__(
            self,
            "recompute_ranges",
            _normalize_ranges(self.prompt_tokens, self.recompute_ranges),
        )

    @property
    def recompute_tokens(self) -> int:
        return _range_token_count(self.recompute_ranges)

    @property
    def cached_ranges(self) -> tuple[TokenRange, ...]:
        """Rows that may be read from verified corrected KV, not written."""

        return _complement(self.prompt_tokens, self.recompute_ranges)

    @property
    def cached_tokens(self) -> int:
        return _range_token_count(self.cached_ranges)

    @property
    def covers_prompt(self) -> bool:
        return self.recompute_tokens + self.cached_tokens == self.prompt_tokens

    @property
    def is_full_recompute(self) -> bool:
        return self.cached_tokens == 0


@dataclass(frozen=True, slots=True)
class ForwardRowPlan:
    """Full-shaped per-layer row selections for one forward pass."""

    prompt_tokens: int
    layers: tuple[LayerRowSelection, ...]

    def __post_init__(self) -> None:
        _validate_prompt_tokens(self.prompt_tokens)
        try:
            layers = tuple(self.layers)
        except TypeError:
            _fail(SelectivePlanErrorCode.INVALID_LAYER_COUNT)
        if len(layers) != GPT_OSS_NUM_LAYERS:
            _fail(SelectivePlanErrorCode.INVALID_LAYER_COUNT)
        for expected_index, layer in enumerate(layers):
            if not isinstance(layer, LayerRowSelection):
                _fail(SelectivePlanErrorCode.INVALID_LAYER_INDEX)
            if layer.layer_index != expected_index:
                _fail(SelectivePlanErrorCode.INVALID_LAYER_INDEX)
            if layer.prompt_tokens != self.prompt_tokens or not layer.covers_prompt:
                _fail(SelectivePlanErrorCode.INVALID_PROMPT_LENGTH)
        object.__setattr__(self, "layers", layers)

    @classmethod
    def full_recompute(cls, prompt_tokens: int) -> ForwardRowPlan:
        """Build the only plan consumed by the current connector milestone."""

        _validate_prompt_tokens(prompt_tokens)
        full_range = (TokenRange(0, prompt_tokens),) if prompt_tokens else ()
        return cls(
            prompt_tokens,
            tuple(
                LayerRowSelection(index, prompt_tokens, full_range)
                for index in range(GPT_OSS_NUM_LAYERS)
            ),
        )

    @classmethod
    def from_recompute_ranges(
        cls,
        prompt_tokens: int,
        ranges_by_layer: Sequence[Sequence[TokenRange]],
    ) -> ForwardRowPlan:
        """Build a plan while keeping all layer rows full-shaped.

        This constructor is intentionally not connected to vLLM.  It exists so
        the future model/backend spike can test row accounting independently of
        CUDA and can compare every selective plan with ``full_recompute``.
        """

        _validate_prompt_tokens(prompt_tokens)
        try:
            ranges = tuple(tuple(layer_ranges) for layer_ranges in ranges_by_layer)
        except TypeError:
            _fail(SelectivePlanErrorCode.INVALID_LAYER_COUNT)
        if len(ranges) != GPT_OSS_NUM_LAYERS:
            _fail(SelectivePlanErrorCode.INVALID_LAYER_COUNT)
        return cls(
            prompt_tokens,
            tuple(
                LayerRowSelection(index, prompt_tokens, layer_ranges)
                for index, layer_ranges in enumerate(ranges)
            ),
        )

    @property
    def recompute_tokens(self) -> int:
        return sum(layer.recompute_tokens for layer in self.layers)

    @property
    def cached_tokens(self) -> int:
        return sum(layer.cached_tokens for layer in self.layers)

    @property
    def is_full_recompute(self) -> bool:
        return self.cached_tokens == 0

    def layer(self, layer_index: int) -> LayerRowSelection:
        if (
            not _is_int(layer_index)
            or not 0 <= layer_index < GPT_OSS_NUM_LAYERS
        ):
            _fail(SelectivePlanErrorCode.INVALID_LAYER_INDEX)
        return self.layers[layer_index]


_ACTIVE_PLAN: ContextVar[ForwardRowPlan | None] = ContextVar(
    "cacheblend_gpt_oss_forward_row_plan", default=None
)


class ForwardRowPlanContext:
    """Worker-local, lifetime-bounded plan binding for one forward pass."""

    @staticmethod
    @contextmanager
    def bind(plan: ForwardRowPlan) -> Iterator[ForwardRowPlan]:
        if not isinstance(plan, ForwardRowPlan):
            _fail(SelectivePlanErrorCode.INVALID_LAYER_COUNT)
        if _ACTIVE_PLAN.get() is not None:
            _fail(SelectivePlanErrorCode.ACTIVE_CONTEXT)
        token = _ACTIVE_PLAN.set(plan)
        try:
            yield plan
        finally:
            _ACTIVE_PLAN.reset(token)

    @staticmethod
    def current() -> ForwardRowPlan:
        plan = _ACTIVE_PLAN.get()
        if plan is None:
            _fail(SelectivePlanErrorCode.MISSING_CONTEXT)
        return plan


__all__ = [
    "ForwardRowPlan",
    "ForwardRowPlanContext",
    "LayerRowSelection",
    "SelectivePlanError",
    "SelectivePlanErrorCode",
]
