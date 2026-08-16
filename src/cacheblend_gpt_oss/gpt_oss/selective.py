# SPDX-License-Identifier: Apache-2.0
"""CPU-only row-plan contract for the future GPT-OSS selective data plane.

This module deliberately does not import vLLM, Torch, CUDA, or model code.  It
describes the full-shaped row invariant that the GPT-OSS model override and
attention backend preserve.  The connector may install one plan around a
single worker forward; CPU tests can still use the scoped ``bind`` helper.

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
    IMPORTANCE_SCORES_ALREADY_SET = "importance_scores_already_set"


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
    def recompute_positions(self) -> tuple[int, ...]:
        """Return the canonical prompt-row positions selected for this layer."""

        return tuple(
            position
            for token_range in self.recompute_ranges
            for position in range(token_range.start, token_range.end)
        )

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


@dataclass(slots=True)
class SelectiveForwardState:
    """Mutable worker-local selective state for one model forward.

    The initial plan recomputes every row through ``check_layer`` and leaves
    only the configured suffix/non-cached rows selected afterwards.  The
    attention backend replaces that provisional plan once, before writing the
    check-layer KV, using measured loaded-versus-fresh value differences.
    Keeping this state separate from the immutable plan lets the model query
    the final plan on every later layer without widening the vLLM boundary.
    """

    plan: ForwardRowPlan
    candidate_cached_ranges: tuple[TokenRange, ...]
    check_layer: int
    recompute_ratio: float
    suffix_tokens: int
    importance_scores: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ForwardRowPlan):
            _fail(SelectivePlanErrorCode.INVALID_LAYER_COUNT)
        try:
            ranges = tuple(self.candidate_cached_ranges)
        except TypeError:
            _fail(SelectivePlanErrorCode.INVALID_RANGE)
        if self.plan.prompt_tokens < 0 or any(
            not isinstance(item, TokenRange)
            or len(item) == 0
            or item.end > self.plan.prompt_tokens
            for item in ranges
        ):
            _fail(SelectivePlanErrorCode.INVALID_RANGE)
        if tuple(sorted(ranges, key=lambda item: (item.start, item.end))) != ranges:
            _fail(SelectivePlanErrorCode.NON_CANONICAL_RANGES)
        if any(left.overlaps(right) for left, right in pairwise(ranges)):
            _fail(SelectivePlanErrorCode.OVERLAPPING_RANGES)
        if (
            not _is_int(self.check_layer)
            or not 0 <= self.check_layer < GPT_OSS_NUM_LAYERS
            or isinstance(self.recompute_ratio, bool)
            or not isinstance(self.recompute_ratio, int | float)
            or not 0.0 <= float(self.recompute_ratio) <= 1.0
            or not _is_int(self.suffix_tokens)
            or not 0 <= self.suffix_tokens <= self.plan.prompt_tokens
        ):
            _fail(SelectivePlanErrorCode.INVALID_RANGE)
        object.__setattr__(self, "candidate_cached_ranges", ranges)

    @property
    def scored(self) -> bool:
        """Whether the check-layer measurement has replaced the provisional plan."""

        return self.importance_scores is not None

    def update_importance_scores(self, scores: Sequence[object]) -> None:
        """Install one validated score vector and rebuild the later-layer plan."""

        if self.scored:
            _fail(SelectivePlanErrorCode.IMPORTANCE_SCORES_ALREADY_SET)
        normalized = tuple(scores)
        # Keep the policy import local: selective_policy imports ForwardRowPlan.
        from cacheblend_gpt_oss.gpt_oss.selective_policy import (
            CacheBlendSelectionPolicy,
        )

        result = CacheBlendSelectionPolicy().select(
            prompt_tokens=self.plan.prompt_tokens,
            cache_ranges=self.candidate_cached_ranges,
            importance_scores=normalized,
            check_layer=self.check_layer,
            recompute_ratio=self.recompute_ratio,
            suffix_tokens=self.suffix_tokens,
        )
        self.plan = result.row_plan
        self.importance_scores = tuple(float(score) for score in normalized)


_ACTIVE_PLAN: ContextVar[ForwardRowPlan | SelectiveForwardState | None] = ContextVar(
    "cacheblend_gpt_oss_forward_row_plan", default=None
)


class ForwardRowPlanContext:
    """Worker-local, lifetime-bounded plan binding for one forward pass."""

    @staticmethod
    def install(plan: ForwardRowPlan | SelectiveForwardState) -> object:
        """Install a plan before vLLM enters the model forward.

        The V1 connector's ``start_load_kv`` and ``wait_for_save`` hooks are
        separate callbacks, so a normal ``with`` block cannot span the model
        runner call.  The opaque token is intentionally returned to the
        connector and must be passed to :meth:`reset` exactly once.
        """

        if not isinstance(plan, ForwardRowPlan | SelectiveForwardState):
            _fail(SelectivePlanErrorCode.INVALID_LAYER_COUNT)
        if _ACTIVE_PLAN.get() is not None:
            _fail(SelectivePlanErrorCode.ACTIVE_CONTEXT)
        return _ACTIVE_PLAN.set(plan)

    @staticmethod
    def reset(token: object) -> None:
        """Clear a plan installed by :meth:`install`."""

        try:
            _ACTIVE_PLAN.reset(token)  # type: ignore[arg-type]
        except (RuntimeError, TypeError, ValueError) as error:
            raise SelectivePlanError(SelectivePlanErrorCode.MISSING_CONTEXT) from error

    @staticmethod
    @contextmanager
    def bind(plan: ForwardRowPlan) -> Iterator[ForwardRowPlan]:
        token = ForwardRowPlanContext.install(plan)
        try:
            yield plan
        finally:
            ForwardRowPlanContext.reset(token)

    @staticmethod
    def current() -> ForwardRowPlan:
        plan = _ACTIVE_PLAN.get()
        if plan is None:
            _fail(SelectivePlanErrorCode.MISSING_CONTEXT)
        return plan.plan if isinstance(plan, SelectiveForwardState) else plan

    @staticmethod
    def current_or_none() -> ForwardRowPlan | None:
        """Return the active plan without turning ordinary vLLM into an error."""

        active = _ACTIVE_PLAN.get()
        if isinstance(active, SelectiveForwardState):
            return active.plan
        return active

    @staticmethod
    def current_state() -> SelectiveForwardState | None:
        """Return mutable selective state, if this forward is selective."""

        active = _ACTIVE_PLAN.get()
        return active if isinstance(active, SelectiveForwardState) else None


__all__ = [
    "ForwardRowPlan",
    "ForwardRowPlanContext",
    "LayerRowSelection",
    "SelectiveForwardState",
    "SelectivePlanError",
    "SelectivePlanErrorCode",
]
