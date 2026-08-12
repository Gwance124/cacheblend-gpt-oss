#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise GPT-OSS Harmony/tool/append-only Responses on ``solab-g3``.

The request and replay shape follows the exact pinned vLLM tests:
https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/tests/entrypoints/openai/responses/test_harmony.py#L82-L105
https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/tests/entrypoints/openai/responses/test_function_call.py#L111-L169

Native prompt accounting and timing names come from the pinned vLLM
``PrometheusStatLogger``:
https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/metrics/loggers.py#L580-L903
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from cacheblend_gpt_oss.correctness import (
    CorrectnessRuntimeIdentity,
    connector_counter_delta,
    has_connector_metric_surface,
    has_vllm_prefill_work_metric_surface,
    has_vllm_prompt_metric_surface,
    has_vllm_timing_metric_surface,
    parse_connector_counter_snapshot,
    parse_vllm_prefill_work_snapshot,
    parse_vllm_prompt_counter_snapshot,
    parse_vllm_timing_snapshot,
    require_vllm_prefill_work_total,
    require_vllm_timing_delta,
    vllm_prefill_work_snapshot_delta,
    vllm_prompt_counter_delta,
    vllm_timing_snapshot_delta,
)
from cacheblend_gpt_oss.responses_contract import (
    JsonObject,
    append_response_and_user,
    append_tool_result,
    parse_completed_response,
    require_forced_tool_call,
    require_reasoned_message,
)
from cacheblend_gpt_oss.targets import PINNED_TARGET

_TOOL_NAME = "get_weather"
_EXPECTED_CITY = "Paris"
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--plugin-commit", required=True)
    parser.add_argument("--model-config-digest", required=True)
    parser.add_argument("--kv-cache-config-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--metric-wait-seconds", type=float, default=30.0)
    return parser


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
            data=json.dumps(payload, allow_nan=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=self._timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))


def _runtime_identity(args: argparse.Namespace) -> CorrectnessRuntimeIdentity:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("this contract check must run on solab-g3 with CUDA")
    return CorrectnessRuntimeIdentity(
        model_id=PINNED_TARGET.model_id,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        plugin_commit=args.plugin_commit,
        model_config_digest=args.model_config_digest,
        kv_cache_config_digest=args.kv_cache_config_digest,
        vllm_version=version("vllm"),
        lmcache_version=version("lmcache"),
        torch_version=torch.__version__,
        cuda_runtime=str(torch.version.cuda),
        gpu_name=torch.cuda.get_device_name(0),
    )


def _require_served_model(data: object) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise ValueError("invalid /v1/models response")
    model_ids = {
        item.get("id") for item in data["data"] if isinstance(item, dict)
    }
    if PINNED_TARGET.model_id not in model_ids:
        raise ValueError("pinned GPT-OSS model is not served")


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


def _wait_for_requests(
    client: LocalVllmClient,
    minimum: int,
    wait_seconds: float,
) -> tuple[dict[str, int], str]:
    if wait_seconds <= 0:
        raise ValueError("metric wait must be positive")
    deadline = time.monotonic() + wait_seconds
    while True:
        snapshot = parse_connector_counter_snapshot(client.get_text("/metrics"))
        if snapshot["requests"] >= minimum:
            metrics = client.get_text("/metrics")
            return parse_connector_counter_snapshot(metrics), metrics
        if time.monotonic() >= deadline:
            raise TimeoutError("connector metrics did not reach three requests")
        time.sleep(0.25)


def _require_metric_delta(delta: dict[str, int]) -> None:
    if (
        delta["requests"] != 3
        or delta["tokens_recomputed"] <= 0
        or delta["prefill_tokens_avoided"] != 0
        or delta["kv_tokens_found"]
        != delta["kv_tokens_loaded"] + delta["kv_tokens_rejected"]
    ):
        raise ValueError("Responses connector metric delta is inconsistent")


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError("Responses contract output already exists")
    runtime = _runtime_identity(args)
    client = LocalVllmClient(args.base_url, args.api_key, args.timeout_seconds)
    _require_served_model(client.get_json("/v1/models"))
    initial_metrics = client.get_text("/metrics")
    if not has_connector_metric_surface(initial_metrics):
        raise ValueError("CacheBlend connector metrics are not present")
    if not has_vllm_timing_metric_surface(initial_metrics):
        raise ValueError("pinned vLLM timing metrics are not present")
    if not has_vllm_prompt_metric_surface(initial_metrics):
        raise ValueError("pinned vLLM prompt metrics are not present")
    if not has_vllm_prefill_work_metric_surface(initial_metrics):
        raise ValueError("pinned vLLM prefill-work metrics are not present")
    before = parse_connector_counter_snapshot(initial_metrics)
    before_prompt = parse_vllm_prompt_counter_snapshot(initial_metrics)
    before_prefill_work = parse_vllm_prefill_work_snapshot(initial_metrics)
    before_timing = parse_vllm_timing_snapshot(initial_metrics)

    initial_input: list[JsonObject] = [
        {
            "role": "user",
            "content": (
                "Use get_weather to look up Paris. Do not answer before the tool."
            ),
        }
    ]
    first_payload = _request_payload(initial_input)
    first_payload["tools"] = [_TOOL]
    first_payload["tool_choice"] = {"type": "function", "name": _TOOL_NAME}
    first = parse_completed_response(client.post_json("/v1/responses", first_payload))
    call = require_forced_tool_call(first, expected_name=_TOOL_NAME)
    city = call.arguments.get("city")
    if not isinstance(city, str) or city.casefold() != _EXPECTED_CITY.casefold():
        raise ValueError("forced tool arguments did not preserve the requested city")

    tool_output = json.dumps(
        {"city": _EXPECTED_CITY, "temperature_celsius": 21},
        separators=(",", ":"),
        sort_keys=True,
    )
    tool_history = append_tool_result(
        initial_input,
        first,
        call,
        output=tool_output,
    )
    second = parse_completed_response(
        client.post_json("/v1/responses", _request_payload(tool_history))
    )
    second_texts = require_reasoned_message(second)
    if _EXPECTED_CITY.casefold() not in " ".join(second_texts).casefold():
        raise ValueError("tool continuation did not reference the tool result city")

    final_history = append_response_and_user(
        tool_history,
        second,
        user_text="Reply with only the city name from this conversation.",
    )
    third = parse_completed_response(
        client.post_json("/v1/responses", _request_payload(final_history))
    )
    third_texts = require_reasoned_message(third)
    if _EXPECTED_CITY.casefold() not in " ".join(third_texts).casefold():
        raise ValueError("append-only multi-turn response lost the city")

    after, after_metrics = _wait_for_requests(
        client,
        before["requests"] + 3,
        args.metric_wait_seconds,
    )
    delta = connector_counter_delta(before, after)
    _require_metric_delta(delta)
    timing_delta = vllm_timing_snapshot_delta(
        before_timing,
        parse_vllm_timing_snapshot(after_metrics),
    )
    require_vllm_timing_delta(timing_delta, expected_requests=3)
    after_prompt = parse_vllm_prompt_counter_snapshot(after_metrics)
    after_prefill_work = parse_vllm_prefill_work_snapshot(after_metrics)
    native_prompt_tokens = vllm_prompt_counter_delta(before_prompt, after_prompt)
    native_prefill_work = vllm_prefill_work_snapshot_delta(
        before_prefill_work,
        after_prefill_work,
    )
    require_vllm_prefill_work_total(
        native_prefill_work,
        expected_prompt_tokens=native_prompt_tokens,
        expected_requests=3,
    )
    report = {
        "schema_version": 1,
        "contract": "gpt_oss_responses_harmony_tool_append_only_multiturn",
        "runtime": asdict(runtime),
        "passed": True,
        "turns": [
            {
                "output_types": list(first.output_types),
                "reasoning_items": first.reasoning_items,
                "function_calls": len(first.function_calls),
            },
            {
                "output_types": list(second.output_types),
                "reasoning_items": second.reasoning_items,
                "message_text_parts": len(second.message_texts),
            },
            {
                "output_types": list(third.output_types),
                "reasoning_items": third.reasoning_items,
                "message_text_parts": len(third.message_texts),
            },
        ],
        "tool": {
            "name": call.name,
            "argument_keys": sorted(call.arguments),
            "result_city_observed": True,
        },
        "append_only_item_counts": {
            "initial": len(initial_input),
            "after_tool": len(tool_history),
            "final_input": len(final_history),
        },
        "connector_counter_delta": delta,
        "native_prompt_tokens_processed": native_prompt_tokens,
        "native_prefill_work": {
            "observations": native_prefill_work.observations,
            "kv_computed_tokens": native_prefill_work.kv_computed_tokens,
        },
        "vllm_timing_delta": timing_delta.as_dict(),
    }
    rendered = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with args.output.open("x", encoding="utf-8") as output:
        output.write(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
