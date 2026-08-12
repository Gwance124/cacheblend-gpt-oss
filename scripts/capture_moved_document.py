#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Capture one pinned full-vocabulary moved-document artifact on solab-g3.

The script deliberately uses ``/v1/completions`` with raw token IDs for the
numerical gate because pinned vLLM rejects logprobs for Harmony/GPT-OSS
``/v1/responses`` requests:
https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/entrypoints/openai/responses/serving.py#L296-L301

Pinned completions accept integer token prompts and token-ID logprob output:
https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/entrypoints/openai/completion/protocol.py#L42-L60
https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/entrypoints/openai/completion/protocol.py#L126-L143
"""

from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from cacheblend_gpt_oss.correctness import (
    ARTIFACT_SCHEMA_VERSION,
    GPT_OSS_VOCAB_SIZE,
    CorrectnessArtifact,
    CorrectnessCase,
    CorrectnessRunMode,
    CorrectnessRuntimeIdentity,
    build_correctness_fixture,
    connector_evidence_from_snapshots,
    has_connector_metric_surface,
    parse_completion_distribution,
    parse_connector_counter_snapshot,
    parse_connector_store_counter_snapshot,
    write_artifact,
)
from cacheblend_gpt_oss.storage.lmcache_types import LMCACHE_CHUNK_SIZE
from cacheblend_gpt_oss.targets import PINNED_TARGET


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in CorrectnessRunMode],
        required=True,
    )
    parser.add_argument(
        "--case",
        choices=[case.value for case in CorrectnessCase],
        default=CorrectnessCase.MOVED_DOCUMENT.value,
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--plugin-commit", required=True)
    parser.add_argument("--model-config-digest", required=True)
    parser.add_argument("--kv-cache-config-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
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


def _completion_payload(token_ids: tuple[int, ...], *, full: bool) -> dict[str, Any]:
    return {
        "model": PINNED_TARGET.model_id,
        "prompt": list(token_ids),
        "add_special_tokens": False,
        "max_tokens": 1,
        "min_tokens": 1,
        "ignore_eos": True,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "logprobs": GPT_OSS_VOCAB_SIZE if full else 1,
        "return_tokens_as_token_ids": True,
        "return_token_ids": True,
        "stream": False,
    }


def _require_served_model(data: object) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise ValueError("invalid /v1/models response")
    model_ids = {
        item.get("id") for item in data["data"] if isinstance(item, dict)
    }
    if PINNED_TARGET.model_id not in model_ids:
        raise ValueError("pinned GPT-OSS model is not served")


def _require_connector_metric_surface(text: str, *, expected: bool) -> None:
    present = has_connector_metric_surface(text)
    if present is not expected:
        mode = "present" if expected else "absent"
        raise ValueError(f"connector metric surface must be {mode} for this run")


def _wait_for_request_counter(
    client: LocalVllmClient,
    minimum: int,
    wait_seconds: float,
    *,
    minimum_store_tokens: int | None = None,
) -> dict[str, int]:
    if wait_seconds <= 0:
        raise ValueError("metric wait must be positive")
    deadline = time.monotonic() + wait_seconds
    while True:
        metrics = client.get_text("/metrics")
        snapshot = parse_connector_counter_snapshot(metrics)
        stores = parse_connector_store_counter_snapshot(metrics)
        if (
            snapshot["requests"] >= minimum
            and (
                minimum_store_tokens is None
                or stores["store_tokens_completed"] >= minimum_store_tokens
            )
        ):
            return snapshot
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "connector request/store metrics did not reach the expected "
                "milestone"
            )
        time.sleep(0.25)


def _runtime_identity(args: argparse.Namespace) -> CorrectnessRuntimeIdentity:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("this capture must run on solab-g3 with CUDA")
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


def main() -> int:
    args = _parser().parse_args()
    mode = CorrectnessRunMode(args.mode)
    if args.output.exists():
        raise FileExistsError("correctness output already exists")
    runtime = _runtime_identity(args)
    client = LocalVllmClient(args.base_url, args.api_key, args.timeout_seconds)
    _require_served_model(client.get_json("/v1/models"))
    fixture = build_correctness_fixture(CorrectnessCase(args.case))
    connector = None
    if mode is CorrectnessRunMode.CACHEBLEND_100PCT:
        initial_metrics = client.get_text("/metrics")
        _require_connector_metric_surface(initial_metrics, expected=True)
        initial = parse_connector_counter_snapshot(initial_metrics)
        initial_store = parse_connector_store_counter_snapshot(initial_metrics)
        source_store_tokens = (
            len(fixture.source_prompt_token_ids) // LMCACHE_CHUNK_SIZE
        ) * LMCACHE_CHUNK_SIZE
        target_store_tokens = (
            len(fixture.target_prompt_token_ids) // LMCACHE_CHUNK_SIZE
        ) * LMCACHE_CHUNK_SIZE
        client.post_json(
            "/v1/completions",
            _completion_payload(fixture.source_prompt_token_ids, full=False),
        )
        after_source = _wait_for_request_counter(
            client,
            initial["requests"] + 1,
            args.metric_wait_seconds,
            minimum_store_tokens=(
                initial_store["store_tokens_completed"] + source_store_tokens
            ),
        )
        target_response = client.post_json(
            "/v1/completions",
            _completion_payload(fixture.target_prompt_token_ids, full=True),
        )
        after_target = _wait_for_request_counter(
            client,
            after_source["requests"] + 1,
            args.metric_wait_seconds,
            minimum_store_tokens=(
                initial_store["store_tokens_completed"]
                + source_store_tokens
                + target_store_tokens
            ),
        )
        connector = connector_evidence_from_snapshots(after_source, after_target)
    else:
        _require_connector_metric_surface(client.get_text("/metrics"), expected=False)
        target_response = client.post_json(
            "/v1/completions",
            _completion_payload(fixture.target_prompt_token_ids, full=True),
        )
    artifact = CorrectnessArtifact(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        run_mode=mode,
        runtime=runtime,
        prompt=fixture.prompt_identity,
        distribution=parse_completion_distribution(target_response),
        connector=connector,
    )
    write_artifact(args.output, artifact)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "run_mode": mode.value,
                "case": artifact.prompt.case.value,
                "top_token_id": artifact.distribution.top_token_id,
                "sampled_token_id": artifact.distribution.sampled_token_id,
                "connector": (
                    None
                    if artifact.connector is None
                    else {
                        "kv_tokens_found": artifact.connector.kv_tokens_found,
                        "kv_tokens_loaded": artifact.connector.kv_tokens_loaded,
                        "kv_tokens_rejected": artifact.connector.kv_tokens_rejected,
                        "tokens_recomputed": artifact.connector.tokens_recomputed,
                        "prefill_tokens_avoided": (
                            artifact.connector.prefill_tokens_avoided
                        ),
                    }
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
