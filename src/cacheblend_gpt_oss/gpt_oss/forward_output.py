# SPDX-License-Identifier: Apache-2.0
"""Full-shaped model-output contract for the pinned GPT-OSS runner.

The vLLM 0.19.1 V1 GPU runner computes ``logits_indices`` while preparing a
request and later gathers ``hidden_states[logits_indices]`` before computing
logits.  A future selective model override must therefore preserve the exact
row shape produced by ordinary model forward; returning only recomputed rows
would silently select the wrong sampling positions.

Pinned source evidence:

* ``GPUModelRunner._prepare_inputs`` derives ``logits_indices`` from
  ``query_start_loc``:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/worker/gpu_model_runner.py#L2040-L2051
* ``GPUModelRunner.execute_model`` gathers hidden rows with those indices:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/worker/gpu_model_runner.py#L4060-L4074

This module is deliberately tensor-free.  It validates metadata before a
future model/backend touches CUDA tensors and does not claim that selective
execution itself is implemented.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum


class ForwardOutputErrorCode(str, Enum):
    """Bounded failures for the model-output contract."""

    INVALID_EXPECTED_ROWS = "invalid_expected_rows"
    INVALID_ACTUAL_ROWS = "invalid_actual_rows"
    ROW_COUNT_MISMATCH = "row_count_mismatch"
    INVALID_HIDDEN_SIZE = "invalid_hidden_size"
    INVALID_HIDDEN_RANK = "invalid_hidden_rank"
    INVALID_LOGITS_INDEX = "invalid_logits_index"
    LOGITS_INDEX_OUT_OF_RANGE = "logits_index_out_of_range"
    LOGITS_INDICES_NOT_INCREASING = "logits_indices_not_increasing"


class ForwardOutputError(ValueError):
    """Fail-closed error without prompt/request identifiers."""

    def __init__(self, code: ForwardOutputErrorCode) -> None:
        self.code = code
        super().__init__(f"GPT-OSS forward-output contract failure: {code.value}")


def _bounded_nonnegative(value: object, code: ForwardOutputErrorCode) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ForwardOutputError(code)
    return value


@dataclass(frozen=True, slots=True)
class ForwardOutputContract:
    """Validated shape and sampling-row metadata for one model forward.

    ``expected_rows`` is the number of rows ordinary vLLM would return for the
    prepared input, including any runner padding.  ``actual_rows`` must match
    it exactly.  ``logits_indices`` are the runner-provided row offsets and are
    retained in their original order for the subsequent logits gather.
    """

    expected_rows: int
    actual_rows: int
    hidden_size: int
    logits_indices: tuple[int, ...]

    @classmethod
    def validate(
        cls,
        *,
        expected_rows: object,
        actual_hidden_shape: Sequence[object],
        hidden_size: object,
        logits_indices: Sequence[object],
    ) -> ForwardOutputContract:
        expected = _bounded_nonnegative(
            expected_rows, ForwardOutputErrorCode.INVALID_EXPECTED_ROWS
        )
        shape = tuple(actual_hidden_shape)
        if len(shape) != 2:
            raise ForwardOutputError(ForwardOutputErrorCode.INVALID_HIDDEN_RANK)
        actual = _bounded_nonnegative(
            shape[0], ForwardOutputErrorCode.INVALID_ACTUAL_ROWS
        )
        if actual != expected:
            raise ForwardOutputError(ForwardOutputErrorCode.ROW_COUNT_MISMATCH)
        width = _bounded_nonnegative(
            hidden_size, ForwardOutputErrorCode.INVALID_HIDDEN_SIZE
        )
        shape_width = _bounded_nonnegative(
            shape[1], ForwardOutputErrorCode.INVALID_HIDDEN_SIZE
        )
        if width == 0 or shape_width != width:
            raise ForwardOutputError(ForwardOutputErrorCode.INVALID_HIDDEN_SIZE)

        normalized: list[int] = []
        previous = -1
        for index in logits_indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise ForwardOutputError(ForwardOutputErrorCode.INVALID_LOGITS_INDEX)
            if index < 0 or index >= actual:
                raise ForwardOutputError(
                    ForwardOutputErrorCode.LOGITS_INDEX_OUT_OF_RANGE
                )
            if index <= previous:
                raise ForwardOutputError(
                    ForwardOutputErrorCode.LOGITS_INDICES_NOT_INCREASING
                )
            normalized.append(index)
            previous = index
        return cls(expected, actual, width, tuple(normalized))


__all__ = [
    "ForwardOutputContract",
    "ForwardOutputError",
    "ForwardOutputErrorCode",
]
