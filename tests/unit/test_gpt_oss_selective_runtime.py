"""CPU-only tests for the dormant selective forward bridge."""

from __future__ import annotations

import pytest

from cacheblend_gpt_oss.gpt_oss.selective import (
    ForwardRowPlan,
    ForwardRowPlanContext,
    SelectivePlanError,
    SelectivePlanErrorCode,
)
from cacheblend_gpt_oss.gpt_oss.selective_runtime import (
    GptOssSelectiveModelAdapter,
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


def test_plan_error_raised_by_forward_is_a_forward_failure() -> None:
    def forward() -> object:
        raise SelectivePlanError(SelectivePlanErrorCode.MISSING_CONTEXT)

    with pytest.raises(SelectiveForwardError) as caught:
        SelectiveForwardBridge().run(
            _plan(),
            expected_rows=4,
            hidden_size=8,
            logits_indices=(),
            forward=forward,
            hidden_shape=lambda output: (4, 8),
        )
    assert caught.value.code is SelectiveForwardErrorCode.FORWARD_FAILED


def test_model_adapter_preserves_pinned_forward_arguments_and_output_contract() -> None:
    observed: dict[str, object] = {}
    plan = _plan()

    def model_forward(**kwargs: object) -> object:
        observed.update(kwargs)
        assert ForwardRowPlanContext.current() == plan
        return {"hidden": "full-shaped"}

    result = GptOssSelectiveModelAdapter().run(
        plan,
        input_ids=(1, 2, 3, 4),
        positions=(0, 1, 2, 3),
        intermediate_tensors=None,
        expected_rows=4,
        hidden_size=16,
        logits_indices=(3,),
        model_forward=model_forward,
        hidden_shape=lambda output: (4, 16),
    )

    assert result.output == {"hidden": "full-shaped"}
    assert observed == {
        "input_ids": (1, 2, 3, 4),
        "positions": (0, 1, 2, 3),
        "intermediate_tensors": None,
        "inputs_embeds": None,
    }
    with pytest.raises(SelectivePlanError):
        ForwardRowPlanContext.current()


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        (
            "input_ids",
            None,
            SelectiveForwardErrorCode.INPUT_IDS_REQUIRED,
        ),
        (
            "positions",
            None,
            SelectiveForwardErrorCode.POSITIONS_REQUIRED,
        ),
        (
            "inputs_embeds",
            ("embedded",),
            SelectiveForwardErrorCode.PROMPT_EMBEDS_UNSUPPORTED,
        ),
    ],
)
def test_model_adapter_rejects_unsupported_input_envelopes(
    field: str,
    value: object,
    code: SelectiveForwardErrorCode,
) -> None:
    kwargs: dict[str, object] = {
        "input_ids": (1,),
        "positions": (0,),
        "inputs_embeds": None,
    }
    kwargs[field] = value

    with pytest.raises(SelectiveForwardError) as caught:
        GptOssSelectiveModelAdapter().run(
            _plan(),
            input_ids=kwargs["input_ids"],
            positions=kwargs["positions"],
            inputs_embeds=kwargs["inputs_embeds"],
            expected_rows=1,
            hidden_size=8,
            logits_indices=(),
            model_forward=lambda **ignored: object(),
            hidden_shape=lambda output: (1, 8),
        )
    assert caught.value.code is code
