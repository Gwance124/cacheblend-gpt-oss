# SPDX-License-Identifier: Apache-2.0
"""Dependency-free validation of the pinned GPT-OSS Responses item contract.

Pinned vLLM accepts both input and prior output items in ``ResponsesRequest``:
https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/entrypoints/openai/responses/protocol.py#L119-L154

Its exact GPT-OSS integration tests require completed Harmony responses,
reasoning items, function calls, append-only function-call output, and a later
text response:
https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/tests/entrypoints/openai/responses/test_harmony.py#L82-L105
https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/tests/entrypoints/openai/responses/test_function_call.py#L111-L169

This module validates only protocol structure and exact append-only item
propagation. It does not treat natural-language quality as numerical evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TypeAlias, cast

JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)
JsonObject: TypeAlias = dict[str, JsonValue]

MAX_RESPONSE_ID_BYTES = 256
MAX_ITEM_TYPE_BYTES = 128
MAX_CALL_FIELD_BYTES = 1_024
_HARMONY_OUTPUT_TYPES = frozenset({"reasoning", "function_call", "message"})


def _json_object(value: object, name: str) -> JsonObject:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"Responses {name} is not a JSON object")
    try:
        encoded = json.dumps(value, allow_nan=False, separators=(",", ":"))
        decoded = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Responses {name} is not finite JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"Responses {name} is not a JSON object")
    return cast(JsonObject, decoded)


def _bounded_text(value: object, name: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not value.strip()
        or len(value.encode("utf-8")) > maximum
    ):
        raise ValueError(f"Responses {name} is invalid")
    return value


@dataclass(frozen=True, slots=True)
class FunctionCallObservation:
    """Validated function-call item needed for the next append-only turn."""

    call_id: str = field(repr=False)
    name: str
    arguments: JsonObject = field(repr=False)

    def __post_init__(self) -> None:
        _bounded_text(self.call_id, "function call ID", MAX_CALL_FIELD_BYTES)
        _bounded_text(self.name, "function name", MAX_CALL_FIELD_BYTES)
        object.__setattr__(self, "arguments", _json_object(self.arguments, "arguments"))


def _nonnegative_count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Responses {name} is invalid")
    return value


@dataclass(frozen=True, slots=True)
class ResponseUsageObservation:
    """Bounded usage counters emitted by pinned non-streaming Responses."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    tool_output_tokens: int

    def __post_init__(self) -> None:
        values = (
            ("input tokens", self.input_tokens),
            ("output tokens", self.output_tokens),
            ("total tokens", self.total_tokens),
            ("cached tokens", self.cached_tokens),
            ("reasoning tokens", self.reasoning_tokens),
            ("tool output tokens", self.tool_output_tokens),
        )
        for name, value in values:
            _nonnegative_count(value, name)
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("Responses usage total tokens do not reconcile")
        if self.cached_tokens > self.input_tokens:
            raise ValueError("Responses cached tokens exceed input tokens")


@dataclass(frozen=True, slots=True)
class ResponseObservation:
    """Validated non-streaming response with opaque items retained for replay."""

    output_items: tuple[JsonObject, ...] = field(repr=False)
    output_types: tuple[str, ...]
    reasoning_items: int
    function_calls: tuple[FunctionCallObservation, ...]
    message_texts: tuple[str, ...] = field(repr=False)
    usage: ResponseUsageObservation | None = None

    def __post_init__(self) -> None:
        if (
            not self.output_items
            or len(self.output_items) != len(self.output_types)
            or isinstance(self.reasoning_items, bool)
            or not isinstance(self.reasoning_items, int)
            or any(
                not isinstance(call, FunctionCallObservation)
                for call in self.function_calls
            )
        ):
            raise ValueError("Responses output item accounting is invalid")
        if self.reasoning_items < 0:
            raise ValueError("Responses reasoning item accounting is invalid")


def _parse_usage(data: object, *, required: bool) -> ResponseUsageObservation | None:
    if data is None:
        if required:
            raise ValueError("Responses usage is missing")
        return None
    usage = _json_object(data, "usage")
    input_details = _json_object(
        usage.get("input_tokens_details"), "input token details"
    )
    output_details = _json_object(
        usage.get("output_tokens_details"), "output token details"
    )
    return ResponseUsageObservation(
        input_tokens=_nonnegative_count(
            usage.get("input_tokens"), "input tokens"
        ),
        output_tokens=_nonnegative_count(
            usage.get("output_tokens"), "output tokens"
        ),
        total_tokens=_nonnegative_count(
            usage.get("total_tokens"), "total tokens"
        ),
        cached_tokens=_nonnegative_count(
            input_details.get("cached_tokens"), "cached tokens"
        ),
        reasoning_tokens=_nonnegative_count(
            output_details.get("reasoning_tokens"), "reasoning tokens"
        ),
        tool_output_tokens=_nonnegative_count(
            output_details.get("tool_output_tokens"), "tool output tokens"
        ),
    )


def parse_completed_response(
    data: object, *, require_usage: bool = False, require_completed_items: bool = False
) -> ResponseObservation:
    """Require a completed, nonempty pinned Responses JSON object.

    ``usage`` is optional in the pinned protocol model, so CPU fixtures may
    omit it. The live non-streaming contract harness passes ``require_usage``
    to prove that vLLM emitted the structured counters before accepting a
    Harmony/tool/multi-turn response. It also passes
    ``require_completed_items`` because a completed root response must not
    carry an in-progress output item.
    """

    root = _json_object(data, "response")
    _bounded_text(root.get("id"), "response ID", MAX_RESPONSE_ID_BYTES)
    if root.get("status") != "completed":
        raise ValueError("Responses request did not complete")
    usage = _parse_usage(root.get("usage"), required=require_usage)
    raw_output = root.get("output")
    if not isinstance(raw_output, list) or not raw_output:
        raise ValueError("Responses output is empty")

    output_items: list[JsonObject] = []
    output_types: list[str] = []
    function_calls: list[FunctionCallObservation] = []
    message_texts: list[str] = []
    reasoning_items = 0
    for raw_item in raw_output:
        item = _json_object(raw_item, "output item")
        item_type = _bounded_text(
            item.get("type"), "output item type", MAX_ITEM_TYPE_BYTES
        )
        if item_type not in _HARMONY_OUTPUT_TYPES:
            raise ValueError("Responses output item type is unsupported")
        # vLLM 0.19.1 can emit a completed root response whose Harmony
        # reasoning item has a null status.  Reasoning is informational and
        # does not represent an unfinished tool/message item.  Keep the live
        # contract strict for every actionable output item and for explicit
        # non-completed reasoning statuses.
        item_status = item.get("status")
        reasoning_status_is_acceptable = (
            item_type == "reasoning" and item_status is None
        )
        if require_completed_items and not (
            item_status == "completed" or reasoning_status_is_acceptable
        ):
            raise ValueError("Responses output item did not complete")
        output_items.append(item)
        output_types.append(item_type)
        if item_type == "reasoning":
            reasoning_items += 1
        elif item_type == "function_call":
            arguments_text = _bounded_text(
                item.get("arguments"), "function arguments", 1_000_000
            )
            try:
                arguments = json.loads(arguments_text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Responses function arguments are invalid JSON"
                ) from exc
            function_calls.append(
                FunctionCallObservation(
                    call_id=_bounded_text(
                        item.get("call_id"), "function call ID", MAX_CALL_FIELD_BYTES
                    ),
                    name=_bounded_text(
                        item.get("name"), "function name", MAX_CALL_FIELD_BYTES
                    ),
                    arguments=_json_object(arguments, "function arguments"),
                )
            )
        elif item_type == "message":
            content = item.get("content")
            if not isinstance(content, list):
                raise ValueError("Responses message content is invalid")
            for raw_part in content:
                part = _json_object(raw_part, "message content")
                if part.get("type") == "output_text":
                    message_texts.append(
                        _bounded_text(part.get("text"), "output text", 10_000_000)
                    )
    return ResponseObservation(
        output_items=tuple(output_items),
        output_types=tuple(output_types),
        reasoning_items=reasoning_items,
        function_calls=tuple(function_calls),
        message_texts=tuple(message_texts),
        usage=usage,
    )


def require_forced_tool_call(
    response: ResponseObservation,
    *,
    expected_name: str,
) -> FunctionCallObservation:
    """Require one named call plus an explicit Harmony reasoning item."""

    if (
        response.reasoning_items < 1
        or len(response.function_calls) != 1
        or response.output_types[-1] != "function_call"
        or any(item_type != "reasoning" for item_type in response.output_types[:-1])
    ):
        raise ValueError("Responses forced tool turn is structurally incomplete")
    call = response.function_calls[0]
    if call.name != expected_name:
        raise ValueError("Responses forced tool name does not match")
    return call


def require_reasoned_message(response: ResponseObservation) -> tuple[str, ...]:
    """Require a Harmony reasoning item and at least one nonempty text part."""

    if (
        response.reasoning_items < 1
        or response.function_calls
        or not response.message_texts
        or response.output_types[-1] != "message"
        or any(item_type != "reasoning" for item_type in response.output_types[:-1])
    ):
        raise ValueError("Responses reasoned message turn is structurally incomplete")
    return response.message_texts


def append_tool_result(
    initial_input: list[JsonObject],
    response: ResponseObservation,
    call: FunctionCallObservation,
    *,
    output: str,
) -> list[JsonObject]:
    """Append every server output item and its exact matching tool result."""

    if call not in response.function_calls:
        raise ValueError("function call is not present in the response")
    tool_output = _bounded_text(output, "function output", 1_000_000)
    history = [_json_object(item, "input item") for item in initial_input]
    history.extend(_json_object(item, "output item") for item in response.output_items)
    history.append(
        {
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": tool_output,
        }
    )
    return history


def append_response_and_user(
    history: list[JsonObject],
    response: ResponseObservation,
    *,
    user_text: str,
) -> list[JsonObject]:
    """Append the complete response followed by one new user message."""

    text = _bounded_text(user_text, "user text", 1_000_000)
    result = [_json_object(item, "history item") for item in history]
    result.extend(_json_object(item, "output item") for item in response.output_items)
    result.append({"role": "user", "content": text})
    return result


__all__ = [
    "FunctionCallObservation",
    "JsonObject",
    "JsonValue",
    "ResponseObservation",
    "ResponseUsageObservation",
    "append_response_and_user",
    "append_tool_result",
    "parse_completed_response",
    "require_forced_tool_call",
    "require_reasoned_message",
]
