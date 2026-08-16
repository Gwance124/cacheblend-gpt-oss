from __future__ import annotations

import pytest

from cacheblend_gpt_oss.gpt_oss.selective import (
    ForwardRowPlan,
    ForwardRowPlanContext,
    LayerRowSelection,
    SelectivePlanError,
    SelectivePlanErrorCode,
)
from cacheblend_gpt_oss.planner import TokenRange


def _full_ranges(prompt_tokens: int) -> tuple[tuple[TokenRange, ...], ...]:
    ranges = (TokenRange(0, prompt_tokens),) if prompt_tokens else ()
    return tuple(ranges for _ in range(24))


def test_full_recompute_plan_covers_all_gpt_oss_layers() -> None:
    plan = ForwardRowPlan.full_recompute(17)

    assert len(plan.layers) == 24
    assert plan.recompute_tokens == 24 * 17
    assert plan.cached_tokens == 0
    assert plan.is_full_recompute
    assert all(layer.covers_prompt for layer in plan.layers)
    assert all(layer.recompute_ranges == (TokenRange(0, 17),) for layer in plan.layers)


def test_zero_length_prompt_has_no_phantom_range() -> None:
    plan = ForwardRowPlan.full_recompute(0)

    assert plan.recompute_tokens == 0
    assert plan.cached_tokens == 0
    assert all(not layer.recompute_ranges for layer in plan.layers)
    assert all(not layer.cached_ranges for layer in plan.layers)


def test_complement_is_exact_for_a_moved_document() -> None:
    ranges = list(_full_ranges(20))
    ranges[0] = (TokenRange(0, 3), TokenRange(10, 20))
    plan = ForwardRowPlan.from_recompute_ranges(20, tuple(ranges))

    first = plan.layer(0)
    assert first.recompute_ranges == (TokenRange(0, 3), TokenRange(10, 20))
    assert first.cached_ranges == (TokenRange(3, 10),)
    assert first.recompute_tokens == 13
    assert first.cached_tokens == 7
    assert plan.cached_tokens == 7
    assert not plan.is_full_recompute
    assert plan.layer(1).is_full_recompute


def test_context_is_bounded_and_clears_after_forward() -> None:
    plan = ForwardRowPlan.full_recompute(3)

    assert ForwardRowPlanContext.current_or_none() is None

    with pytest.raises(SelectivePlanError) as missing:
        ForwardRowPlanContext.current()
    assert missing.value.code is SelectivePlanErrorCode.MISSING_CONTEXT

    with ForwardRowPlanContext.bind(plan) as active:
        assert active is plan
        assert ForwardRowPlanContext.current() is plan
        assert ForwardRowPlanContext.current_or_none() is plan
        with (
            pytest.raises(SelectivePlanError) as nested,
            ForwardRowPlanContext.bind(plan),
        ):
            pass
        assert nested.value.code is SelectivePlanErrorCode.ACTIVE_CONTEXT

    with pytest.raises(SelectivePlanError) as cleared:
        ForwardRowPlanContext.current()
    assert cleared.value.code is SelectivePlanErrorCode.MISSING_CONTEXT
    assert ForwardRowPlanContext.current_or_none() is None


def test_context_supports_connector_lifetime_install_and_reset() -> None:
    plan = ForwardRowPlan.full_recompute(3)
    token = ForwardRowPlanContext.install(plan)
    assert ForwardRowPlanContext.current() is plan
    ForwardRowPlanContext.reset(token)
    assert ForwardRowPlanContext.current_or_none() is None

    with pytest.raises(SelectivePlanError) as cleared:
        ForwardRowPlanContext.reset(token)
    assert cleared.value.code is SelectivePlanErrorCode.MISSING_CONTEXT


def test_recompute_positions_follow_canonical_ranges() -> None:
    plan = ForwardRowPlan.from_recompute_ranges(
        8,
        tuple(
            (
                (TokenRange(0, 2), TokenRange(5, 8))
                if layer == 0
                else (TokenRange(0, 8),)
            )
            for layer in range(24)
        ),
    )

    assert plan.layer(0).recompute_positions == (0, 1, 5, 6, 7)
    assert plan.layer(1).recompute_positions == tuple(range(8))


@pytest.mark.parametrize(
    ("factory", "code"),
    [
        (
            lambda: ForwardRowPlan.full_recompute(-1),
            SelectivePlanErrorCode.INVALID_PROMPT_LENGTH,
        ),
        (
            lambda: ForwardRowPlan.full_recompute(131_073),
            SelectivePlanErrorCode.INVALID_PROMPT_LENGTH,
        ),
        (
            lambda: ForwardRowPlan.from_recompute_ranges(4, _full_ranges(4)[:-1]),
            SelectivePlanErrorCode.INVALID_LAYER_COUNT,
        ),
        (
            lambda: LayerRowSelection(24, 4, (TokenRange(0, 4),)),
            SelectivePlanErrorCode.INVALID_LAYER_INDEX,
        ),
        (
            lambda: LayerRowSelection(0, 4, (TokenRange(0, 3), TokenRange(2, 4))),
            SelectivePlanErrorCode.OVERLAPPING_RANGES,
        ),
        (
            lambda: LayerRowSelection(0, 4, (TokenRange(2, 4), TokenRange(0, 1))),
            SelectivePlanErrorCode.NON_CANONICAL_RANGES,
        ),
        (
            lambda: LayerRowSelection(0, 3, (TokenRange(0, 4),)),
            SelectivePlanErrorCode.RANGE_OUT_OF_BOUNDS,
        ),
    ],
)
def test_row_plan_validation_is_fail_closed(factory, code) -> None:
    with pytest.raises(SelectivePlanError) as error:
        factory()
    assert error.value.code is code


def test_layer_lookup_rejects_bool_and_out_of_range_indices() -> None:
    plan = ForwardRowPlan.full_recompute(1)

    for index in (True, -1, 24):
        with pytest.raises(SelectivePlanError) as error:
            plan.layer(index)
        assert error.value.code is SelectivePlanErrorCode.INVALID_LAYER_INDEX
