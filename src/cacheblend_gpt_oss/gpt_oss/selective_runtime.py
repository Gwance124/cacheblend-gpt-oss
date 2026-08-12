# SPDX-License-Identifier: Apache-2.0
"""CPU-testable forward bridge for the future GPT-OSS selective adapter.

The pinned V1 runner needs a selective model/backend to see one immutable row
plan during model execution, then receive the same full-shaped hidden output it
would receive from ordinary GPT-OSS.  This module owns that narrow lifecycle:
it binds :class:`ForwardRowPlanContext` only around an injected forward call and
validates the runner's output shape and logits indices before returning.

It deliberately does not import vLLM, Torch, or CUDA and is not a serving
registration point.  The concrete model override and sink-aware attention
backend remain gated on the M3--M5 ``solab-g3`` evidence described in
``docs/plans/gpt-oss-cacheblend-feasibility.md``.

The model-call adapter follows the exact pinned GPT-OSS forward signature:
https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/models/gpt_oss.py#L218-L240
It accepts token IDs and positions only; prompt embeddings are outside the
first validated CacheBlend envelope and fail closed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import NoReturn, Protocol

from cacheblend_gpt_oss.gpt_oss.forward_output import (
    ForwardOutputContract,
    ForwardOutputError,
)
from cacheblend_gpt_oss.gpt_oss.selective import (
    ForwardRowPlan,
    ForwardRowPlanContext,
    SelectivePlanError,
    SelectivePlanErrorCode,
)


class SelectiveForwardErrorCode(str, Enum):
    """Bounded failures for the worker-local forward bridge."""

    INVALID_PLAN = "invalid_plan"
    INVALID_FORWARD = "invalid_forward"
    FORWARD_FAILED = "forward_failed"
    INVALID_OUTPUT = "invalid_output"
    ACTIVE_CONTEXT = "active_context"
    INPUT_IDS_REQUIRED = "input_ids_required"
    POSITIONS_REQUIRED = "positions_required"
    PROMPT_EMBEDS_UNSUPPORTED = "prompt_embeds_unsupported"


class SelectiveForwardError(RuntimeError):
    """Fail-closed bridge error without request or tensor details."""

    def __init__(self, code: SelectiveForwardErrorCode) -> None:
        self.code = code
        super().__init__(f"GPT-OSS selective forward failure: {code.value}")


def _fail(code: SelectiveForwardErrorCode) -> NoReturn:
    raise SelectiveForwardError(code)


class HiddenShapeReader(Protocol):
    """Extract the two-dimensional hidden shape from a model result."""

    def __call__(self, output: object) -> Sequence[object]:
        """Return ``(rows, hidden_size)`` without retaining tensor data."""


class GptOssModelForward(Protocol):
    """The pinned ``GptOssForCausalLM.forward`` call surface."""

    def __call__(
        self,
        *,
        input_ids: object,
        positions: object,
        intermediate_tensors: object | None = None,
        inputs_embeds: object | None = None,
    ) -> object:
        """Run one ordinary GPT-OSS model forward."""


@dataclass(frozen=True, slots=True)
class SelectiveForwardResult:
    """Validated output and runner metadata from one bounded forward call."""

    output: object = field(repr=False, compare=False)
    contract: ForwardOutputContract


class SelectiveForwardBridge:
    """Bind one plan around an injected forward and validate its output.

    ``forward`` is intentionally a zero-argument callable: a future model
    override closes over vLLM's prepared inputs while the attention backend
    reads ``ForwardRowPlanContext.current()`` during the call.  The context is
    always cleared before this method returns, including on model or shape
    failure, so a plan cannot leak into the next request.
    """

    def run(
        self,
        plan: ForwardRowPlan,
        *,
        expected_rows: object,
        hidden_size: object,
        logits_indices: Sequence[object],
        forward: Callable[[], object],
        hidden_shape: HiddenShapeReader,
    ) -> SelectiveForwardResult:
        if not isinstance(plan, ForwardRowPlan):
            _fail(SelectiveForwardErrorCode.INVALID_PLAN)
        if not callable(forward) or not callable(hidden_shape):
            _fail(SelectiveForwardErrorCode.INVALID_FORWARD)

        try:
            with ForwardRowPlanContext.bind(plan):
                output = forward()
        except SelectivePlanError as error:
            if error.code is SelectivePlanErrorCode.ACTIVE_CONTEXT:
                _fail(SelectiveForwardErrorCode.ACTIVE_CONTEXT)
            raise SelectiveForwardError(
                SelectiveForwardErrorCode.FORWARD_FAILED
            ) from error
        except Exception as error:
            raise SelectiveForwardError(
                SelectiveForwardErrorCode.FORWARD_FAILED
            ) from error

        try:
            shape = hidden_shape(output)
            contract = ForwardOutputContract.validate(
                expected_rows=expected_rows,
                actual_hidden_shape=shape,
                hidden_size=hidden_size,
                logits_indices=logits_indices,
            )
        except (ForwardOutputError, TypeError, ValueError) as error:
            raise SelectiveForwardError(
                SelectiveForwardErrorCode.INVALID_OUTPUT
            ) from error
        return SelectiveForwardResult(output=output, contract=contract)


class GptOssSelectiveModelAdapter:
    """Bind the exact GPT-OSS model-forward arguments to the row bridge.

    This is the model-side seam a future vLLM registry override can delegate
    to. It keeps the runner's full-shaped output contract while making the
    first target envelope explicit: token IDs and positions are required, and
    ``inputs_embeds`` is rejected rather than silently mixing a prompt-
    embedding path with token-identity-based cache records.
    """

    def __init__(self, forward_bridge: SelectiveForwardBridge | None = None) -> None:
        self._bridge = forward_bridge or SelectiveForwardBridge()

    def run(
        self,
        plan: ForwardRowPlan,
        *,
        input_ids: object | None,
        positions: object | None,
        intermediate_tensors: object | None = None,
        inputs_embeds: object | None = None,
        expected_rows: object,
        hidden_size: object,
        logits_indices: Sequence[object],
        model_forward: GptOssModelForward,
        hidden_shape: HiddenShapeReader,
    ) -> SelectiveForwardResult:
        """Run a token-ID GPT-OSS forward under one bounded row plan."""

        if input_ids is None:
            _fail(SelectiveForwardErrorCode.INPUT_IDS_REQUIRED)
        if positions is None:
            _fail(SelectiveForwardErrorCode.POSITIONS_REQUIRED)
        if inputs_embeds is not None:
            _fail(SelectiveForwardErrorCode.PROMPT_EMBEDS_UNSUPPORTED)
        if not callable(model_forward):
            _fail(SelectiveForwardErrorCode.INVALID_FORWARD)

        return self._bridge.run(
            plan,
            expected_rows=expected_rows,
            hidden_size=hidden_size,
            logits_indices=logits_indices,
            forward=lambda: model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=None,
            ),
            hidden_shape=hidden_shape,
        )


__all__ = [
    "GptOssModelForward",
    "GptOssSelectiveModelAdapter",
    "HiddenShapeReader",
    "SelectiveForwardBridge",
    "SelectiveForwardError",
    "SelectiveForwardErrorCode",
    "SelectiveForwardResult",
]
