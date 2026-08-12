"""CPU-only tests for the dormant CacheBlend check-layer policy."""

from __future__ import annotations

import math

import pytest

from cacheblend_gpt_oss.gpt_oss.selective_policy import (
    CacheBlendSelectionPolicy,
    SelectionMeasurement,
    SelectionPolicyError,
    SelectionPolicyErrorCode,
)
from cacheblend_gpt_oss.planner.models import TokenRange


def _scores(length: int) -> list[float]:
    return [0.0] * length


def test_policy_recomputes_non_cached_rows_suffix_and_top_scored_cached_rows() -> None:
    scores = _scores(12)
    scores[3] = 10.0
    scores[9] = 9.0
    scores[2] = 8.0
    scores[8] = 7.0
    result = CacheBlendSelectionPolicy().select(
        prompt_tokens=12,
        cache_ranges=(TokenRange(2, 6), TokenRange(8, 12)),
        importance_scores=scores,
        check_layer=1,
        recompute_ratio=0.5,
        suffix_tokens=2,
    )

    # Six cached rows are eligible outside the forced suffix; floor(6*.5)=3.
    assert result.selected_cached_rows == (2, 3, 9)
    assert result.candidate_cached_ranges == (
        TokenRange(2, 6),
        TokenRange(8, 12),
    )
    assert result.cached_ranges == (
        TokenRange(4, 6),
        TokenRange(8, 9),
    )
    assert result.recompute_ranges == (
        TokenRange(0, 4),
        TokenRange(6, 8),
        TokenRange(9, 12),
    )
    assert result.row_plan.recompute_tokens == 9 * 24
    assert result.row_plan.cached_tokens == 3 * 24
    assert result.recompute_tokens_per_layer == 9
    assert result.cached_tokens_per_layer == 3
    assert result.recompute_fraction == pytest.approx(9 / 12)
    assert result.check_layer == 1


def test_policy_ratio_zero_keeps_only_verified_cached_rows_outside_suffix() -> None:
    result = CacheBlendSelectionPolicy().select(
        prompt_tokens=10,
        cache_ranges=(TokenRange(2, 8),),
        importance_scores=_scores(10),
        check_layer=0,
        recompute_ratio=0.0,
        suffix_tokens=2,
    )
    assert result.selected_cached_rows == ()
    assert result.recompute_ranges == (
        TokenRange(0, 2),
        TokenRange(8, 10),
    )
    assert result.cached_tokens_per_layer == 6


def test_policy_ratio_one_recomputes_every_prompt_row() -> None:
    result = CacheBlendSelectionPolicy().select(
        prompt_tokens=10,
        cache_ranges=(TokenRange(0, 10),),
        importance_scores=_scores(10),
        check_layer=23,
        recompute_ratio=1.0,
        suffix_tokens=2,
    )
    assert result.selected_cached_rows == tuple(range(8))
    assert result.recompute_ranges == (TokenRange(0, 10),)
    assert result.row_plan.is_full_recompute


def test_ties_choose_lower_token_position_deterministically() -> None:
    result = CacheBlendSelectionPolicy().select(
        prompt_tokens=8,
        cache_ranges=(TokenRange(0, 8),),
        importance_scores=[1.0] * 8,
        check_layer=2,
        recompute_ratio=0.5,
        suffix_tokens=0,
    )
    assert result.selected_cached_rows == (0, 1, 2, 3)


def test_no_cache_candidates_cannot_reduce_recomputation() -> None:
    result = CacheBlendSelectionPolicy().select(
        prompt_tokens=5,
        cache_ranges=(),
        importance_scores=[100.0] * 5,
        check_layer=3,
        recompute_ratio=0.0,
        suffix_tokens=0,
    )
    assert result.recompute_ranges == (TokenRange(0, 5),)
    assert result.selected_cached_rows == ()
    assert result.row_plan.is_full_recompute


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        (
            {
                "prompt_tokens": -1,
                "cache_ranges": (),
                "importance_scores": (),
                "check_layer": 0,
                "recompute_ratio": 0.0,
                "suffix_tokens": 0,
            },
            SelectionPolicyErrorCode.INVALID_PROMPT_LENGTH,
        ),
        (
            {
                "prompt_tokens": 2,
                "cache_ranges": (),
                "importance_scores": [0.0, 0.0],
                "check_layer": 24,
                "recompute_ratio": 0.0,
                "suffix_tokens": 0,
            },
            SelectionPolicyErrorCode.INVALID_CHECK_LAYER,
        ),
        (
            {
                "prompt_tokens": 2,
                "cache_ranges": (),
                "importance_scores": [0.0, 0.0],
                "check_layer": 0,
                "recompute_ratio": 1.1,
                "suffix_tokens": 0,
            },
            SelectionPolicyErrorCode.INVALID_RECOMPUTE_RATIO,
        ),
        (
            {
                "prompt_tokens": 2,
                "cache_ranges": (),
                "importance_scores": [0.0, 0.0],
                "check_layer": 0,
                "recompute_ratio": 0.0,
                "suffix_tokens": 3,
            },
            SelectionPolicyErrorCode.INVALID_SUFFIX_LENGTH,
        ),
        (
            {
                "prompt_tokens": 2,
                "cache_ranges": (),
                "importance_scores": [0.0],
                "check_layer": 0,
                "recompute_ratio": 0.0,
                "suffix_tokens": 0,
            },
            SelectionPolicyErrorCode.INVALID_IMPORTANCE_SCORES,
        ),
        (
            {
                "prompt_tokens": 4,
                "cache_ranges": (object(),),
                "importance_scores": [0.0] * 4,
                "check_layer": 0,
                "recompute_ratio": 0.0,
                "suffix_tokens": 0,
            },
            SelectionPolicyErrorCode.INVALID_CACHE_RANGES,
        ),
        (
            {
                "prompt_tokens": 4,
                "cache_ranges": (TokenRange(3, 5),),
                "importance_scores": [0.0] * 4,
                "check_layer": 0,
                "recompute_ratio": 0.0,
                "suffix_tokens": 0,
            },
            SelectionPolicyErrorCode.CACHE_RANGE_OUT_OF_BOUNDS,
        ),
        (
            {
                "prompt_tokens": 4,
                "cache_ranges": (TokenRange(2, 4), TokenRange(1, 2)),
                "importance_scores": [0.0] * 4,
                "check_layer": 0,
                "recompute_ratio": 0.0,
                "suffix_tokens": 0,
            },
            SelectionPolicyErrorCode.NON_CANONICAL_CACHE_RANGES,
        ),
        (
            {
                "prompt_tokens": 4,
                "cache_ranges": (TokenRange(1, 3), TokenRange(2, 4)),
                "importance_scores": [0.0] * 4,
                "check_layer": 0,
                "recompute_ratio": 0.0,
                "suffix_tokens": 0,
            },
            SelectionPolicyErrorCode.OVERLAPPING_CACHE_RANGES,
        ),
    ],
)
def test_invalid_policy_inputs_fail_closed(
    kwargs: dict[str, object], code: SelectionPolicyErrorCode
) -> None:
    with pytest.raises(SelectionPolicyError) as caught:
        CacheBlendSelectionPolicy().select(**kwargs)  # type: ignore[arg-type]
    assert caught.value.code is code


@pytest.mark.parametrize(
    "scores",
    ([math.inf, 0.0], [math.nan, 0.0], [True, 0.0], [-1.0, 0.0]),
)
def test_invalid_importance_values_fail_closed(scores: list[object]) -> None:
    with pytest.raises(SelectionPolicyError) as caught:
        CacheBlendSelectionPolicy().select(
            prompt_tokens=2,
            cache_ranges=(TokenRange(0, 2),),
            importance_scores=scores,
            check_layer=0,
            recompute_ratio=0.5,
            suffix_tokens=0,
        )
    assert caught.value.code is SelectionPolicyErrorCode.INVALID_IMPORTANCE_SCORES


def test_boolean_control_values_are_not_accepted() -> None:
    with pytest.raises(SelectionPolicyError) as caught:
        CacheBlendSelectionPolicy().select(
            prompt_tokens=2,
            cache_ranges=(),
            importance_scores=[0.0, 0.0],
            check_layer=0,
            recompute_ratio=True,
            suffix_tokens=0,
        )
    assert caught.value.code is SelectionPolicyErrorCode.INVALID_RECOMPUTE_RATIO


def test_ratio_sweep_is_descending_and_reports_work_without_fake_error() -> None:
    sweep = CacheBlendSelectionPolicy().sweep(
        prompt_tokens=12,
        cache_ranges=(TokenRange(0, 12),),
        importance_scores=_scores(12),
        check_layer=1,
        recompute_ratios=(1.0, 0.5, 0.0),
        suffix_tokens=2,
    )
    assert sweep.ratios == (1.0, 0.5, 0.0)
    assert sweep.work_curve == (
        (1.0, 1.0),
        (0.5, 7 / 12),
        (0.0, 2 / 12),
    )
    with pytest.raises(SelectionPolicyError) as caught:
        _ = sweep.error_curve
    assert caught.value.code is SelectionPolicyErrorCode.MEASUREMENT_MISMATCH


def test_ratio_sweep_accepts_explicit_measurements_and_exposes_curves() -> None:
    policy = CacheBlendSelectionPolicy()
    sweep = policy.sweep(
        prompt_tokens=8,
        cache_ranges=(TokenRange(0, 8),),
        importance_scores=_scores(8),
        check_layer=1,
        recompute_ratios=(1.0, 0.25),
        suffix_tokens=2,
    ).with_measurements(
        (
            SelectionMeasurement(0.01, 0.001, 0.4),
            SelectionMeasurement(0.5, 0.1, 0.2),
        )
    )
    assert sweep.error_curve == ((1.0, 0.01, 0.001), (0.25, 0.5, 0.1))
    assert sweep.latency_curve == ((1.0, 0.4), (0.25, 0.2))


def test_ratio_sweep_rejects_duplicate_or_ascending_ratios() -> None:
    with pytest.raises(SelectionPolicyError) as caught:
        CacheBlendSelectionPolicy().sweep(
            prompt_tokens=4,
            cache_ranges=(TokenRange(0, 4),),
            importance_scores=_scores(4),
            check_layer=0,
            recompute_ratios=(0.0, 0.5),
            suffix_tokens=0,
        )
    assert caught.value.code is SelectionPolicyErrorCode.INVALID_RATIO_SWEEP


def test_measurement_mismatch_is_bounded() -> None:
    sweep = CacheBlendSelectionPolicy().sweep(
        prompt_tokens=4,
        cache_ranges=(TokenRange(0, 4),),
        importance_scores=_scores(4),
        check_layer=0,
        recompute_ratios=(1.0,),
        suffix_tokens=0,
    )
    with pytest.raises(SelectionPolicyError) as caught:
        sweep.with_measurements(())
    assert caught.value.code is SelectionPolicyErrorCode.MEASUREMENT_MISMATCH


@pytest.mark.parametrize(
    "values",
    ((-1.0, 0.0, 0.1), (0.0, float("nan"), 0.1), (0.0, 0.0, float("inf"))),
)
def test_invalid_measurement_values_fail_closed(
    values: tuple[float, float, float],
) -> None:
    with pytest.raises(SelectionPolicyError) as caught:
        SelectionMeasurement(*values)
    assert caught.value.code is SelectionPolicyErrorCode.INVALID_MEASUREMENT
