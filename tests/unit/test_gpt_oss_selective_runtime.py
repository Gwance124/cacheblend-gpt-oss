"""CPU-only tests for the dormant selective forward bridge."""

from __future__ import annotations

import pytest

from cacheblend_gpt_oss.gpt_oss.selective import (
    ForwardRowPlan,
    ForwardRowPlanContext,
    SelectivePlanError,
)
from cacheblend_gpt_oss.gpt_oss.selective_runtime import (
    SelectiveForwardBridge,
    SelectiveForwardError,
    SelectiveForwardErrorCode,
)


def _plan() -> ForwardRowPlan:
    return ForwardRowPlan.full_recompute(4)


def test_forward_bridge_binds_plan_only_during_forward_and_validates_output() -> None:
    seen: list[ForwardRowPlan] = []

    def forward() -> object:
        seen.append(ForwardRowPlanContext.current())
        return {"hidden": "full-shaped"}

    result = SelectiveForwardBridge().run(
        _plan(),
        expected_rows=4,
        hidden_size=16,
        logits_indices=(1, 3),
        forward=forward,
        hidden_shape=lambda output: (4, 16),
    )

    assert seen == [_plan()]
    assert result.output == {"hidden": "full-shaped"}
    assert result.contract.logits_indices == (1, 3)
    with pytest.raises(SelectivePlanError):
        ForwardRowPlanContext.current()


def test_forward_failure_clears_context_and_hides_model_detail() -> None:
    def forward() -> object:
        assert ForwardRowPlanContext.current().prompt_tokens == 4
        raise RuntimeError("sensitive model detail")

    with pytest.raises(SelectiveForwardError) as caught:
        SelectiveForwardBridge().run(
            _plan(),
            expected_rows=4,
            hidden_size=16,
            logits_indices=(),
            forward=forward,
            hidden_shape=lambda output: (4, 16),
        )
    assert caught.value.code is SelectiveForwardErrorCode.FORWARD_FAILED
    assert "sensitive model detail" not in str(caught.value)
    with pytest.raises(SelectivePlanError):
        ForwardRowPlanContext.current()


def test_invalid_output_is_rejected_after_context_is_cleared() -> None:
    with pytest.raises(SelectiveForwardError) as caught:
        SelectiveForwardBridge().run(
            _plan(),
            expected_rows=4,
            hidden_size=16,
            logits_indices=(4,),
            forward=lambda: object(),
            hidden_shape=lambda output: (3, 16),
        )
    assert caught.value.code is SelectiveForwardErrorCode.INVALID_OUTPUT
    with pytest.raises(SelectivePlanError):
        ForwardRowPlanContext.current()


def test_invalid_inputs_and_nested_context_fail_closed() -> None:
    bridge = SelectiveForwardBridge()
    with pytest.raises(SelectiveForwardError) as caught:
        bridge.run(
            object(),  # type: ignore[arg-type]
            expected_rows=1,
            hidden_size=8,
            logits_indices=(),
            forward=lambda: object(),
            hidden_shape=lambda output: (1, 8),
        )
    assert caught.value.code is SelectiveForwardErrorCode.INVALID_PLAN

    with ForwardRowPlanContext.bind(_plan()):
        with pytest.raises(SelectiveForwardError) as nested:
            bridge.run(
                _plan(),
                expected_rows=4,
                hidden_size=8,
                logits_indices=(),
                forward=lambda: object(),
                hidden_shape=lambda output: (4, 8),
            )
        assert nested.value.code is SelectiveForwardErrorCode.ACTIVE_CONTEXT


def test_non_callable_boundaries_are_rejected() -> None:
    with pytest.raises(SelectiveForwardError) as caught:
        SelectiveForwardBridge().run(
            _plan(),
            expected_rows=4,
            hidden_size=8,
            logits_indices=(),
            forward=object(),  # type: ignore[arg-type]
            hidden_shape=lambda output: (4, 8),
        )
    assert caught.value.code is SelectiveForwardErrorCode.INVALID_FORWARD
