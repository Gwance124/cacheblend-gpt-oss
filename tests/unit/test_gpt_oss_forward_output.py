"""CPU-only tests for the pinned model-output row contract."""

from __future__ import annotations

import pytest

from cacheblend_gpt_oss.gpt_oss.forward_output import (
    ForwardOutputContract,
    ForwardOutputError,
    ForwardOutputErrorCode,
)


def test_valid_contract_preserves_full_rows_and_sampling_indices() -> None:
    result = ForwardOutputContract.validate(
        expected_rows=8,
        actual_hidden_shape=(8, 4096),
        hidden_size=4096,
        logits_indices=(3, 7),
    )
    assert result.expected_rows == result.actual_rows == 8
    assert result.hidden_size == 4096
    assert result.logits_indices == (3, 7)


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        (
            {
                "expected_rows": 8,
                "actual_hidden_shape": (7, 4096),
                "hidden_size": 4096,
                "logits_indices": (3,),
            },
            ForwardOutputErrorCode.ROW_COUNT_MISMATCH,
        ),
        (
            {
                "expected_rows": 8,
                "actual_hidden_shape": (8, 4096, 1),
                "hidden_size": 4096,
                "logits_indices": (3,),
            },
            ForwardOutputErrorCode.INVALID_HIDDEN_RANK,
        ),
        (
            {
                "expected_rows": 8,
                "actual_hidden_shape": (8, 4096),
                "hidden_size": 2048,
                "logits_indices": (3,),
            },
            ForwardOutputErrorCode.INVALID_HIDDEN_SIZE,
        ),
        (
            {
                "expected_rows": 8,
                "actual_hidden_shape": (8, 4096),
                "hidden_size": 4096,
                "logits_indices": (8,),
            },
            ForwardOutputErrorCode.LOGITS_INDEX_OUT_OF_RANGE,
        ),
        (
            {
                "expected_rows": 8,
                "actual_hidden_shape": (8, 4096),
                "hidden_size": 4096,
                "logits_indices": (4, 4),
            },
            ForwardOutputErrorCode.LOGITS_INDICES_NOT_INCREASING,
        ),
    ],
)
def test_invalid_shape_or_indices_fail_closed(
    kwargs: dict[str, object], code: ForwardOutputErrorCode
) -> None:
    with pytest.raises(ForwardOutputError) as caught:
        ForwardOutputContract.validate(**kwargs)  # type: ignore[arg-type]
    assert caught.value.code is code


@pytest.mark.parametrize(
    ("expected_rows", "shape", "hidden_size", "indices", "code"),
    [
        (True, (1, 4), 4, (), ForwardOutputErrorCode.INVALID_EXPECTED_ROWS),
        (1, (True, 4), 4, (), ForwardOutputErrorCode.INVALID_ACTUAL_ROWS),
        (1, (1, 4), True, (), ForwardOutputErrorCode.INVALID_HIDDEN_SIZE),
        (1, (1, 4), 4, (True,), ForwardOutputErrorCode.INVALID_LOGITS_INDEX),
    ],
)
def test_bool_metadata_is_not_accepted(
    expected_rows: object,
    shape: tuple[object, ...],
    hidden_size: object,
    indices: tuple[object, ...],
    code: ForwardOutputErrorCode,
) -> None:
    with pytest.raises(ForwardOutputError) as caught:
        ForwardOutputContract.validate(
            expected_rows=expected_rows,
            actual_hidden_shape=shape,
            hidden_size=hidden_size,
            logits_indices=indices,
        )
    assert caught.value.code is code


def test_empty_sampling_indices_are_allowed_for_nonempty_output() -> None:
    result = ForwardOutputContract.validate(
        expected_rows=1,
        actual_hidden_shape=(1, 16),
        hidden_size=16,
        logits_indices=(),
    )
    assert result.logits_indices == ()
