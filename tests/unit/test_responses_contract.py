from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import pytest

from cacheblend_gpt_oss.responses_contract import (
    FunctionCallObservation,
    append_response_and_user,
    append_tool_result,
    parse_completed_response,
    require_forced_tool_call,
    require_reasoned_message,
)


def _tool_response() -> dict[str, object]:
    return {
        "id": "resp_first",
        "status": "completed",
        "output": [
            {
                "id": "reasoning_first",
                "type": "reasoning",
                "status": "completed",
                "summary": [],
                "content": [],
            },
            {
                "id": "function_first",
                "type": "function_call",
                "status": "completed",
                "call_id": "call_first",
                "name": "get_weather",
                "arguments": '{"city":"Paris"}',
            },
        ],
    }


def _message_response(response_id: str = "resp_second") -> dict[str, object]:
    return {
        "id": response_id,
        "status": "completed",
        "output": [
            {
                "id": f"reasoning_{response_id}",
                "type": "reasoning",
                "status": "completed",
                "summary": [],
                "content": [],
            },
            {
                "id": f"message_{response_id}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "The weather in Paris is 21 C.",
                        "annotations": [],
                    }
                ],
            },
        ],
    }


def test_parse_and_append_exact_harmony_tool_turn() -> None:
    initial = [{"role": "user", "content": "Weather in Paris?"}]
    raw = _tool_response()
    response = parse_completed_response(raw)
    call = require_forced_tool_call(response, expected_name="get_weather")

    history = append_tool_result(
        initial,
        response,
        call,
        output='{"city":"Paris","temperature_celsius":21}',
    )

    assert response.output_types == ("reasoning", "function_call")
    assert response.reasoning_items == 1
    assert call.arguments == {"city": "Paris"}
    assert history[:1] == initial
    assert history[1:3] == raw["output"]
    assert history[3] == {
        "type": "function_call_output",
        "call_id": "call_first",
        "output": '{"city":"Paris","temperature_celsius":21}',
    }


def test_append_only_multi_turn_preserves_prior_items() -> None:
    first = parse_completed_response(_tool_response())
    call = require_forced_tool_call(first, expected_name="get_weather")
    tool_history = append_tool_result(
        [{"role": "user", "content": "Weather in Paris?"}],
        first,
        call,
        output="21 C",
    )
    second = parse_completed_response(_message_response())

    texts = require_reasoned_message(second)
    final_history = append_response_and_user(
        tool_history,
        second,
        user_text="Name the city only.",
    )

    assert texts == ("The weather in Paris is 21 C.",)
    assert final_history[: len(tool_history)] == tool_history
    assert final_history[len(tool_history) : -1] == list(second.output_items)
    assert final_history[-1] == {
        "role": "user",
        "content": "Name the city only.",
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda response: response.update(status="incomplete"), "did not complete"),
        (lambda response: response.update(output=[]), "output is empty"),
        (lambda response: response.pop("id"), "response ID"),
    ],
)
def test_response_root_failures_are_rejected(
    mutation: Callable[[dict[str, object]], object],
    message: str,
) -> None:
    raw = _tool_response()
    mutation(raw)
    with pytest.raises(ValueError, match=message):
        parse_completed_response(raw)


def test_invalid_function_arguments_and_wrong_forced_name_are_rejected() -> None:
    malformed = _tool_response()
    output = malformed["output"]
    assert isinstance(output, list)
    call_item = output[1]
    assert isinstance(call_item, dict)
    call_item["arguments"] = "not-json"
    with pytest.raises(ValueError, match="arguments are invalid JSON"):
        parse_completed_response(malformed)

    parsed = parse_completed_response(_tool_response())
    with pytest.raises(ValueError, match="name does not match"):
        require_forced_tool_call(parsed, expected_name="another_tool")


def test_unknown_items_and_non_harmony_order_are_rejected() -> None:
    unknown = _message_response()
    output = unknown["output"]
    assert isinstance(output, list)
    output.append({"type": "custom_tool_call", "status": "completed"})
    with pytest.raises(ValueError, match="item type is unsupported"):
        parse_completed_response(unknown)

    misplaced = _tool_response()
    misplaced_output = misplaced["output"]
    assert isinstance(misplaced_output, list)
    misplaced_output.insert(0, {"type": "message", "content": []})
    parsed = parse_completed_response(misplaced)
    with pytest.raises(ValueError, match="structurally incomplete"):
        require_forced_tool_call(parsed, expected_name="get_weather")

    misplaced_message = _message_response()
    message_output = misplaced_message["output"]
    assert isinstance(message_output, list)
    message_output.insert(0, {"type": "message", "content": []})
    parsed_message = parse_completed_response(misplaced_message)
    with pytest.raises(ValueError, match="structurally incomplete"):
        require_reasoned_message(parsed_message)


def test_tool_result_must_match_a_call_in_the_response() -> None:
    parsed = parse_completed_response(_tool_response())
    mismatched = FunctionCallObservation(
        call_id="other_call",
        name="get_weather",
        arguments={"city": "Paris"},
    )
    with pytest.raises(ValueError, match="not present"):
        append_tool_result(
            [{"role": "user", "content": "Weather?"}],
            parsed,
            mismatched,
            output="21 C",
        )


def test_missing_reasoning_or_message_text_fails_closed() -> None:
    no_reasoning = _tool_response()
    output = no_reasoning["output"]
    assert isinstance(output, list)
    del output[0]
    parsed_tool = parse_completed_response(no_reasoning)
    with pytest.raises(ValueError, match="structurally incomplete"):
        require_forced_tool_call(parsed_tool, expected_name="get_weather")

    no_text = _message_response()
    message_output = no_text["output"]
    assert isinstance(message_output, list)
    message = message_output[1]
    assert isinstance(message, dict)
    message["content"] = []
    parsed_message = parse_completed_response(no_text)
    with pytest.raises(ValueError, match="structurally incomplete"):
        require_reasoned_message(parsed_message)

    unexpected_call = _message_response()
    unexpected_output = unexpected_call["output"]
    assert isinstance(unexpected_output, list)
    unexpected_output.append(_tool_response()["output"][1])  # type: ignore[index]
    parsed_unexpected_call = parse_completed_response(unexpected_call)
    with pytest.raises(ValueError, match="structurally incomplete"):
        require_reasoned_message(parsed_unexpected_call)


def test_parsing_and_history_builders_do_not_alias_caller_data() -> None:
    raw = _tool_response()
    original = deepcopy(raw)
    parsed = parse_completed_response(raw)
    call = require_forced_tool_call(parsed, expected_name="get_weather")
    initial = [{"role": "user", "content": "Weather?"}]
    history = append_tool_result(initial, parsed, call, output="21 C")

    raw.clear()
    initial[0]["content"] = "mutated"

    assert parsed.output_types == ("reasoning", "function_call")
    assert history[0]["content"] == "Weather?"
    assert list(parsed.output_items) == original["output"]
