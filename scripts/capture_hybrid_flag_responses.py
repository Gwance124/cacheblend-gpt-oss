#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture deterministic append-only Responses probes for causal A/B gates.

Pinned vLLM 0.19.1 defaults an omitted Responses sampling configuration to
temperature 1.0, top-p 1.0, and no seed. This diagnostic overrides all three:
https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/entrypoints/openai/responses/protocol.py#L298-L325
https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/entrypoints/openai/responses/protocol.py#L245-L246
https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/entrypoints/openai/responses/protocol.py#L363-L374

The bounded warmup uses the pinned completions request contract:
https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/entrypoints/openai/completion/protocol.py#L42-L60
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import ensure_source_path

ensure_source_path()

from cacheblend_gpt_oss.correctness import (  # noqa: E402
    connector_counter_delta,
    connector_store_counter_delta,
    has_connector_metric_surface,
    parse_connector_counter_snapshot,
    parse_connector_store_counter_snapshot,
)
from cacheblend_gpt_oss.responses_contract import (  # noqa: E402
    JsonObject,
    ResponseObservation,
    parse_completed_response,
    require_forced_tool_call,
    require_reasoned_message,
)
from cacheblend_gpt_oss.targets import PINNED_TARGET  # noqa: E402

SCHEMA_VERSION = 1
CONTRACT = "cacheblend-gpt-oss-hybrid-flag-responses-v1"
_TOOL_NAME = "get_weather"
_TOOL = {
    "type": "function",
    "name": _TOOL_NAME,
    "description": "Return the fixed local weather observation for one city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    },
    "strict": True,
}
_DYNAMIC_RESPONSE_KEYS = frozenset({"call_id", "created_at", "encrypted_content", "id"})
_ITEM_ID_PREFIXES = {
    "reasoning": "rs",
    "function_call": "fc",
    "message": "msg",
}
_MAX_FILLER_REPETITIONS_PER_TURN = 20_000
_FILLER_UNITS = (" alpha", " beta", " gamma")
_METRIC_WAIT_SECONDS = 30.0
_WARMUP_PAYLOAD = {
    "model": PINNED_TARGET.model_id,
    "prompt": "Isolated connector-presence warmup.",
    "max_tokens": 1,
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 0,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("implicit", "explicit_false", "baseline", "connector"),
        required=True,
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="run one bounded completion and settle its connector metrics first",
    )
    parser.add_argument(
        "--filler-repetitions-per-turn",
        type=int,
        default=0,
        help=(
            "append this many bounded, turn-distinct filler units at each of the three "
            "turns; the maximum is 20000"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _build_filler(unit: str, repetitions: int) -> str:
    if unit not in _FILLER_UNITS:
        raise ValueError("filler unit is not part of the fixed workload")
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or not 0 <= repetitions <= _MAX_FILLER_REPETITIONS_PER_TURN
    ):
        raise ValueError("filler repetitions are outside the bounded range")
    return unit * repetitions


def _with_filler(filler: str, suffix: str) -> str:
    return f"{filler}\n{suffix}" if filler else suffix


def _run_warmup(client: LocalVllmClient) -> None:
    raw = client.post_json("/v1/completions", _WARMUP_PAYLOAD)
    if (
        not isinstance(raw, dict)
        or not isinstance(raw.get("choices"), list)
        or len(raw["choices"]) != 1
    ):
        raise ValueError("warmup completion did not return exactly one choice")


def _wait_for_one_connector_request(
    client: LocalVllmClient,
    before: dict[str, int],
) -> str:
    deadline = time.monotonic() + _METRIC_WAIT_SECONDS
    while True:
        metrics = client.get_text("/metrics")
        delta = connector_counter_delta(
            before,
            parse_connector_counter_snapshot(metrics),
        )
        if delta["requests"] == 1 and delta["tokens_recomputed"] > 0:
            return metrics
        if delta["requests"] > 1:
            raise ValueError("warmup produced more than one connector request")
        if time.monotonic() >= deadline:
            raise TimeoutError("warmup connector request metric did not settle")
        time.sleep(0.25)


def _canonical_response_value(value: object, *, key: str | None = None) -> object:
    """Remove generated identifiers while retaining every generated token field."""

    if isinstance(value, dict):
        normalized: dict[str, object] = {}
        for item_key, item_value in value.items():
            if item_key in _DYNAMIC_RESPONSE_KEYS:
                continue
            normalized[item_key] = _canonical_response_value(
                item_value,
                key=item_key,
            )
        return normalized
    if isinstance(value, list):
        return [_canonical_response_value(item) for item in value]
    if key == "arguments" and isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _stable_replay_items(
    response: ResponseObservation,
    *,
    turn: int,
) -> tuple[list[JsonObject], dict[str, str]]:
    """Replace server-generated IDs before replay so both arms get exact bytes."""

    items = cast(list[JsonObject], deepcopy(list(response.output_items)))
    call_ids: dict[str, str] = {}
    call_index = 0
    for item_index, item in enumerate(items):
        item.pop("encrypted_content", None)
        if "id" in item:
            item_type = item.get("type")
            if not isinstance(item_type, str) or item_type not in _ITEM_ID_PREFIXES:
                raise ValueError("cannot stabilize an unsupported output item ID")
            item["id"] = f"{_ITEM_ID_PREFIXES[item_type]}_turn_{turn}_{item_index}"
        arguments = item.get("arguments")
        if isinstance(arguments, str):
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ValueError("cannot stabilize invalid function arguments") from exc
            item["arguments"] = _canonical_bytes(parsed_arguments).decode("utf-8")
        old_call_id = item.get("call_id")
        if isinstance(old_call_id, str):
            stable_call_id = f"call_turn_{turn}_{call_index}"
            item["call_id"] = stable_call_id
            call_ids[old_call_id] = stable_call_id
            call_index += 1
    return items, call_ids


def _request_payload(input_items: list[JsonObject]) -> dict[str, Any]:
    return {
        "model": PINNED_TARGET.model_id,
        "input": input_items,
        "max_output_tokens": 512,
        "reasoning": {"effort": "low", "summary": "auto"},
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "truncation": "disabled",
        "store": False,
        "stream": False,
    }


class LocalVllmClient:
    def __init__(self, base_url: str, api_key: str, timeout_seconds: float) -> None:
        parsed = urlparse(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("base URL must be a local HTTP vLLM endpoint")
        if timeout_seconds <= 0:
            raise ValueError("timeout must be positive")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds

    def get_text(self, path: str) -> str:
        request = Request(
            f"{self._base_url}{path}",
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        with urlopen(request, timeout=self._timeout_seconds) as response:
            return response.read().decode("utf-8")

    def get_json(self, path: str) -> object:
        return json.loads(self.get_text(path))

    def post_json(self, path: str, payload: dict[str, Any]) -> object:
        request = Request(
            f"{self._base_url}{path}",
            data=_canonical_bytes(payload),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=self._timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))


def _require_served_model(data: object) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise ValueError("invalid /v1/models response")
    model_ids = {item.get("id") for item in data["data"] if isinstance(item, dict)}
    if PINNED_TARGET.model_id not in model_ids:
        raise ValueError("pinned GPT-OSS model is not served")


def _capture_turn(
    client: LocalVllmClient,
    payload: dict[str, Any],
) -> tuple[ResponseObservation, dict[str, object]]:
    started_at = time.monotonic()
    raw = client.post_json("/v1/responses", payload)
    elapsed_seconds = time.monotonic() - started_at
    response = parse_completed_response(
        raw,
        require_usage=True,
        require_completed_items=True,
    )
    if not isinstance(raw, dict):
        raise ValueError("Responses result is not a JSON object")
    canonical_output = {
        "status": raw.get("status"),
        "output": _canonical_response_value(raw.get("output")),
    }
    if response.usage is None:
        raise ValueError("Responses usage was not captured")
    return response, {
        "request_digest": _digest(payload),
        "output_digest": _digest(canonical_output),
        "canonical_output": canonical_output,
        "usage": asdict(response.usage),
        "elapsed_seconds": elapsed_seconds,
    }


def capture(
    mode: str,
    client: LocalVllmClient,
    *,
    filler_repetitions_per_turn: int = 0,
    warmup: bool = False,
) -> dict[str, object]:
    _require_served_model(client.get_json("/v1/models"))
    initial_metrics = client.get_text("/metrics")
    connector_surface_present = has_connector_metric_surface(initial_metrics)
    connector_expected = mode == "connector"
    if connector_surface_present is not connector_expected:
        expected = "present" if connector_expected else "absent"
        raise ValueError(f"connector metric surface must be {expected}")
    warmup_connector_counters: dict[str, int] | None = None
    warmup_store_counters: dict[str, int] | None = None
    if warmup:
        before_warmup = (
            parse_connector_counter_snapshot(initial_metrics)
            if connector_surface_present
            else None
        )
        before_warmup_store = (
            parse_connector_store_counter_snapshot(initial_metrics)
            if connector_surface_present
            else None
        )
        _run_warmup(client)
        initial_metrics = (
            _wait_for_one_connector_request(client, before_warmup)
            if before_warmup is not None
            else client.get_text("/metrics")
        )
        if before_warmup is not None and before_warmup_store is not None:
            warmup_connector_counters = connector_counter_delta(
                before_warmup,
                parse_connector_counter_snapshot(initial_metrics),
            )
            warmup_store_counters = connector_store_counter_delta(
                before_warmup_store,
                parse_connector_store_counter_snapshot(initial_metrics),
            )
            if (
                warmup_connector_counters["requests"] != 1
                or warmup_connector_counters["reusable_document_tokens_requested"] != 0
                or warmup_connector_counters["kv_tokens_found"] != 0
                or warmup_connector_counters["kv_tokens_loaded"] != 0
                or warmup_connector_counters["kv_tokens_rejected"] != 0
                or warmup_connector_counters["tokens_recomputed"] <= 0
                or warmup_connector_counters["prefill_tokens_avoided"] != 0
                or any(warmup_store_counters.values())
            ):
                raise ValueError("warmup connector counters do not reconcile")
    initial_connector = (
        parse_connector_counter_snapshot(initial_metrics)
        if connector_surface_present
        else None
    )
    initial_store = (
        parse_connector_store_counter_snapshot(initial_metrics)
        if connector_surface_present
        else None
    )
    fillers = tuple(
        _build_filler(unit, filler_repetitions_per_turn) for unit in _FILLER_UNITS
    )
    initial_input: list[JsonObject] = [
        {
            "role": "user",
            "content": _with_filler(
                fillers[0],
                "Use get_weather to look up Paris. Do not answer before the tool.",
            ),
        }
    ]
    first_payload = _request_payload(initial_input)
    first_payload["tools"] = [_TOOL]
    first_payload["tool_choice"] = "auto"
    first, first_turn = _capture_turn(client, first_payload)
    call = require_forced_tool_call(first, expected_name=_TOOL_NAME)
    if call.arguments.get("city") != "Paris":
        raise ValueError("tool call did not preserve the requested city")

    stable_first_items, first_call_ids = _stable_replay_items(first, turn=1)
    stable_call_id = first_call_ids.get(call.call_id)
    if stable_call_id is None:
        raise ValueError("stable function-call ID was not created")
    tool_history = [*initial_input, *stable_first_items]
    tool_history.append(
        {
            "type": "function_call_output",
            "call_id": stable_call_id,
            "output": _with_filler(
                fillers[1],
                '{"city":"Paris","temperature_celsius":21}',
            ),
        }
    )
    second_payload = _request_payload(tool_history)
    second, second_turn = _capture_turn(client, second_payload)
    second_texts = require_reasoned_message(second)
    if "paris" not in " ".join(second_texts).casefold():
        raise ValueError("tool continuation did not preserve Paris")

    stable_second_items, _ = _stable_replay_items(second, turn=2)
    final_history = [*tool_history, *stable_second_items]
    final_history.append(
        {
            "role": "user",
            "content": _with_filler(
                fillers[2],
                "Reply with only the city name from this conversation.",
            ),
        }
    )
    third_payload = _request_payload(final_history)
    third, third_turn = _capture_turn(client, third_payload)
    third_texts = require_reasoned_message(third)
    if "paris" not in " ".join(third_texts).casefold():
        raise ValueError("append-only continuation did not preserve Paris")

    turns = [first_turn, second_turn, third_turn]
    cached_tokens = [
        cast(dict[str, int], turn["usage"])["cached_tokens"] for turn in turns
    ]
    input_tokens = [
        cast(dict[str, int], turn["usage"])["input_tokens"] for turn in turns
    ]
    connector_counters: dict[str, int] | None = None
    connector_store_counters: dict[str, int] | None = None
    if connector_surface_present:
        if initial_connector is None or initial_store is None:
            raise ValueError("connector metric baseline was not captured")
        deadline = time.monotonic() + 30.0
        while True:
            final_metrics = client.get_text("/metrics")
            final_connector = parse_connector_counter_snapshot(final_metrics)
            connector_counters = connector_counter_delta(
                initial_connector,
                final_connector,
            )
            if connector_counters["requests"] >= len(turns):
                connector_store_counters = connector_store_counter_delta(
                    initial_store,
                    parse_connector_store_counter_snapshot(final_metrics),
                )
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("connector counters did not reach three requests")
            time.sleep(0.25)
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT,
        "mode": mode,
        "model": PINNED_TARGET.model_id,
        "sampling": {"temperature": 0.0, "top_p": 1.0, "seed": 0},
        "stable_replay_ids": True,
        "warmup": {
            "performed": warmup,
            "request_digest": _digest(_WARMUP_PAYLOAD) if warmup else None,
            "connector_counters": warmup_connector_counters,
            "connector_store_counters": warmup_store_counters,
        },
        "workload": {
            "filler_units_per_turn": list(_FILLER_UNITS),
            "filler_repetitions_per_turn": filler_repetitions_per_turn,
            "input_tokens_per_turn": input_tokens,
        },
        "turns": turns,
        "prefix_cache": {
            "cached_tokens_per_turn": cached_tokens,
            "reuse_observed_after_cold_turn": any(
                value > 0 for value in cached_tokens[1:]
            ),
        },
        "total_elapsed_seconds": sum(
            cast(float, turn["elapsed_seconds"]) for turn in turns
        ),
        "connector_counters": connector_counters,
        "connector_store_counters": connector_store_counters,
    }


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError("Responses diagnostic output already exists")
    artifact = capture(
        args.mode,
        LocalVllmClient(args.base_url, args.api_key, args.timeout_seconds),
        filler_repetitions_per_turn=args.filler_repetitions_per_turn,
        warmup=args.warmup,
    )
    rendered = json.dumps(artifact, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with args.output.open("x", encoding="utf-8") as output:
        output.write(rendered)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "mode": artifact["mode"],
                "warmup": artifact["warmup"],
                "prefix_cache": artifact["prefix_cache"],
                "workload": artifact["workload"],
                "connector_counters": artifact["connector_counters"],
                "connector_store_counters": artifact["connector_store_counters"],
                "total_elapsed_seconds": artifact["total_elapsed_seconds"],
                "output_digests": [
                    turn["output_digest"]
                    for turn in cast(list[dict[str, object]], artifact["turns"])
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
