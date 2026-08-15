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

The native prompt-token and timing checks use the pinned vLLM Prometheus
logger's ``prompt_tokens`` counter and request timing histograms:
https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/metrics/loggers.py#L580-L903
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import ensure_source_path

ensure_source_path()

from cacheblend_gpt_oss.correctness import (  # noqa: E402
    ARTIFACT_SCHEMA_VERSION,
    GPT_OSS_VOCAB_SIZE,
    CorrectnessArtifact,
    CorrectnessCase,
    CorrectnessRunMode,
    CorrectnessRuntimeIdentity,
    VllmNativeRequestEvidence,
    VllmPrefillWorkSnapshot,
    VllmPromptSourceDelta,
    VllmTimingSnapshot,
    build_correctness_fixture,
    connector_counter_delta,
    connector_evidence_from_snapshots,
    connector_store_counter_delta,
    has_connector_metric_surface,
    has_vllm_prefill_work_metric_surface,
    has_vllm_prompt_source_metric_surface,
    has_vllm_timing_metric_surface,
    parse_completion_distribution,
    parse_connector_counter_snapshot,
    parse_connector_store_counter_snapshot,
    parse_vllm_prefill_work_snapshot,
    parse_vllm_prompt_counter_snapshot,
    parse_vllm_prompt_source_snapshot,
    parse_vllm_timing_snapshot,
    require_full_prefill_prompt_source_delta,
    require_vllm_prefill_work_delta,
    require_vllm_timing_delta,
    vllm_prefill_work_snapshot_delta,
    vllm_prompt_counter_delta,
    vllm_prompt_source_delta,
    vllm_timing_snapshot_delta,
    write_artifact,
)
from cacheblend_gpt_oss.storage.lmcache_types import LMCACHE_CHUNK_SIZE  # noqa: E402
from cacheblend_gpt_oss.targets import PINNED_TARGET  # noqa: E402


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
    parser.add_argument(
        "--warm-source-before-target",
        action="store_true",
        help=(
            "in full_prefill mode, issue the fixture source request and wait "
            "for native metrics before capturing the target"
        ),
    )
    parser.add_argument(
        "--connector-attached-control",
        action="store_true",
        help=(
            "in full_prefill mode, require a scatter-disabled connector and "
            "save a full-prefill control artifact"
        ),
    )
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
    minimum_prompt_tokens: int | None = None,
    minimum_prompt_source_local_compute: int | None = None,
    minimum_timing_count: int | None = None,
    minimum_prefill_observations: int | None = None,
) -> tuple[dict[str, int], dict[str, int], str]:
    if wait_seconds <= 0:
        raise ValueError("metric wait must be positive")
    deadline = time.monotonic() + wait_seconds
    while True:
        metrics = client.get_text("/metrics")
        snapshot = parse_connector_counter_snapshot(metrics)
        stores = parse_connector_store_counter_snapshot(metrics)
        native_prompt_ready = True
        if minimum_prompt_tokens is not None:
            native_prompt = parse_vllm_prompt_counter_snapshot(metrics)
            native_prompt_ready = (
                native_prompt["prompt_tokens"] >= minimum_prompt_tokens
            )
        native_prompt_source_ready = True
        if minimum_prompt_source_local_compute is not None:
            native_prompt_source_ready = (
                has_vllm_prompt_source_metric_surface(metrics)
                and parse_vllm_prompt_source_snapshot(metrics)["local_compute"]
                >= minimum_prompt_source_local_compute
            )
        native_timing_ready = True
        if minimum_timing_count is not None:
            timing = parse_vllm_timing_snapshot(metrics)
            native_timing_ready = all(
                summary.count >= minimum_timing_count
                for summary in (
                    timing.ttft_seconds,
                    timing.end_to_end_latency_seconds,
                    timing.queue_latency_seconds,
                    timing.prefill_latency_seconds,
                    timing.decode_latency_seconds,
                )
            )
        native_prefill_ready = True
        if minimum_prefill_observations is not None:
            prefill_work = parse_vllm_prefill_work_snapshot(metrics)
            native_prefill_ready = (
                prefill_work.observations >= minimum_prefill_observations
            )
        if (
            snapshot["requests"] >= minimum
            and (
                minimum_store_tokens is None
                or stores["store_tokens_completed"] >= minimum_store_tokens
            )
            and native_prompt_ready
            and native_prompt_source_ready
            and native_timing_ready
            and native_prefill_ready
        ):
            return snapshot, stores, metrics
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "connector request/store metrics did not reach the expected "
                "milestone"
            )
        time.sleep(0.25)


def _wait_for_prompt_counter(
    client: LocalVllmClient,
    minimum: int,
    wait_seconds: float,
    *,
    minimum_prompt_source_local_compute: int | None = None,
    minimum_timing_count: int | None = None,
    minimum_prefill_observations: int | None = None,
) -> tuple[dict[str, int], str]:
    """Wait for one native vLLM prompt-token counter milestone."""

    if wait_seconds <= 0:
        raise ValueError("metric wait must be positive")
    deadline = time.monotonic() + wait_seconds
    while True:
        metrics = client.get_text("/metrics")
        snapshot = parse_vllm_prompt_counter_snapshot(metrics)
        prompt_source_snapshot: dict[str, int] | None = None
        native_prompt_source_ready = True
        if minimum_prompt_source_local_compute is not None:
            if has_vllm_prompt_source_metric_surface(metrics):
                prompt_source_snapshot = parse_vllm_prompt_source_snapshot(metrics)
                native_prompt_source_ready = (
                    prompt_source_snapshot["local_compute"]
                    >= minimum_prompt_source_local_compute
                )
            else:
                native_prompt_source_ready = False
        timing: VllmTimingSnapshot | None = None
        native_timing_ready = True
        if minimum_timing_count is not None:
            timing = parse_vllm_timing_snapshot(metrics)
            native_timing_ready = all(
                summary.count >= minimum_timing_count
                for summary in (
                    timing.ttft_seconds,
                    timing.end_to_end_latency_seconds,
                    timing.queue_latency_seconds,
                    timing.prefill_latency_seconds,
                    timing.decode_latency_seconds,
                )
            )
        prefill_work: VllmPrefillWorkSnapshot | None = None
        native_prefill_ready = True
        if minimum_prefill_observations is not None:
            prefill_work = parse_vllm_prefill_work_snapshot(metrics)
            native_prefill_ready = (
                prefill_work.observations >= minimum_prefill_observations
            )
        if (
            snapshot["prompt_tokens"] >= minimum
            and native_prompt_source_ready
            and native_timing_ready
            and native_prefill_ready
        ):
            return snapshot, metrics
        if time.monotonic() >= deadline:
            source_value = (
                "missing"
                if prompt_source_snapshot is None
                else str(prompt_source_snapshot["local_compute"])
            )
            timing_count = "not-required" if timing is None else str(
                timing.ttft_seconds.count
            )
            prefill_count = "not-required" if prefill_work is None else str(
                prefill_work.observations
            )
            raise TimeoutError(
                "vLLM prompt-token metrics did not reach the expected milestone: "
                f"prompt_tokens={snapshot['prompt_tokens']}/{minimum}, "
                f"prompt_source_local_compute={source_value}/"
                f"{minimum_prompt_source_local_compute}, "
                f"prompt_source_ready={native_prompt_source_ready}, "
                f"ttft_observations={timing_count}/"
                f"{minimum_timing_count}, "
                f"prefill_observations={prefill_count}/"
                f"{minimum_prefill_observations}"
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
    if (
        args.warm_source_before_target
        and mode is not CorrectnessRunMode.FULL_PREFILL
    ):
        raise ValueError(
            "--warm-source-before-target is valid only in full_prefill mode"
        )
    if args.connector_attached_control and mode is not CorrectnessRunMode.FULL_PREFILL:
        raise ValueError(
            "--connector-attached-control is valid only in full_prefill mode"
        )
    if args.connector_attached_control and not args.warm_source_before_target:
        raise ValueError(
            "--connector-attached-control requires --warm-source-before-target"
        )
    if args.output.exists():
        raise FileExistsError("correctness output already exists")
    runtime = _runtime_identity(args)
    client = LocalVllmClient(args.base_url, args.api_key, args.timeout_seconds)
    _require_served_model(client.get_json("/v1/models"))
    fixture = build_correctness_fixture(CorrectnessCase(args.case))
    connector = None
    target_prompt_tokens_processed: int | None = None
    target_prompt_source_delta: dict[str, int] | None = None
    target_prefill_work_delta: VllmPrefillWorkSnapshot | None = None
    target_timing_delta: VllmTimingSnapshot | None = None
    connector_control_evidence: dict[str, dict[str, int]] | None = None
    if mode is CorrectnessRunMode.CACHEBLEND_100PCT:
        initial_metrics = client.get_text("/metrics")
        _require_connector_metric_surface(initial_metrics, expected=True)
        if not has_vllm_timing_metric_surface(initial_metrics):
            raise ValueError("pinned vLLM timing metrics are not present")
        if not has_vllm_prefill_work_metric_surface(initial_metrics):
            raise ValueError("pinned vLLM prefill-work metrics are not present")
        initial = parse_connector_counter_snapshot(initial_metrics)
        initial_store = parse_connector_store_counter_snapshot(initial_metrics)
        initial_prompt = parse_vllm_prompt_counter_snapshot(initial_metrics)
        initial_prompt_source = parse_vllm_prompt_source_snapshot(
            initial_metrics,
            allow_missing=True,
        )
        initial_prefill_work = parse_vllm_prefill_work_snapshot(initial_metrics)
        initial_timing = parse_vllm_timing_snapshot(initial_metrics)
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
        after_source, after_source_store, after_source_metrics = (
            _wait_for_request_counter(
                client,
                initial["requests"] + 1,
                args.metric_wait_seconds,
                minimum_store_tokens=(
                    initial_store["store_tokens_completed"] + source_store_tokens
                ),
                minimum_prompt_tokens=(
                    initial_prompt["prompt_tokens"]
                    + len(fixture.source_prompt_token_ids)
                ),
                minimum_prompt_source_local_compute=(
                    initial_prompt_source["local_compute"]
                    + len(fixture.source_prompt_token_ids)
                ),
                minimum_timing_count=initial_timing.ttft_seconds.count + 1,
                minimum_prefill_observations=initial_prefill_work.observations + 1,
            )
        )
        target_response = client.post_json(
            "/v1/completions",
            _completion_payload(fixture.target_prompt_token_ids, full=True),
        )
        after_target, after_target_store, after_target_metrics = (
            _wait_for_request_counter(
                client,
                after_source["requests"] + 1,
                args.metric_wait_seconds,
                minimum_store_tokens=(
                    initial_store["store_tokens_completed"]
                    + source_store_tokens
                    + target_store_tokens
                ),
                minimum_prompt_tokens=(
                    initial_prompt["prompt_tokens"]
                    + len(fixture.source_prompt_token_ids)
                    + len(fixture.target_prompt_token_ids)
                ),
                minimum_prompt_source_local_compute=(
                    initial_prompt_source["local_compute"]
                    + len(fixture.source_prompt_token_ids)
                    + len(fixture.target_prompt_token_ids)
                ),
                minimum_timing_count=(
                    parse_vllm_timing_snapshot(after_source_metrics)
                    .ttft_seconds.count
                    + 1
                ),
                minimum_prefill_observations=(
                    parse_vllm_prefill_work_snapshot(after_source_metrics)
                    .observations
                    + 1
                ),
            )
        )
        source_prompt = parse_vllm_prompt_counter_snapshot(after_source_metrics)
        target_prompt = parse_vllm_prompt_counter_snapshot(after_target_metrics)
        source_prompt_source = parse_vllm_prompt_source_snapshot(after_source_metrics)
        target_prompt_source = parse_vllm_prompt_source_snapshot(after_target_metrics)
        source_prefill_work = parse_vllm_prefill_work_snapshot(after_source_metrics)
        target_prefill_work = parse_vllm_prefill_work_snapshot(after_target_metrics)
        source_prompt_delta = vllm_prompt_counter_delta(initial_prompt, source_prompt)
        target_prompt_tokens_processed = vllm_prompt_counter_delta(
            source_prompt, target_prompt
        )
        if source_prompt_delta != len(fixture.source_prompt_token_ids):
            raise ValueError("source native prompt-token delta does not match fixture")
        if target_prompt_tokens_processed != len(fixture.target_prompt_token_ids):
            raise ValueError("target native prompt-token delta does not match fixture")
        require_full_prefill_prompt_source_delta(
            vllm_prompt_source_delta(initial_prompt_source, source_prompt_source),
            expected_prompt_tokens=len(fixture.source_prompt_token_ids),
        )
        target_prompt_source_delta = vllm_prompt_source_delta(
            source_prompt_source, target_prompt_source
        )
        require_full_prefill_prompt_source_delta(
            target_prompt_source_delta,
            expected_prompt_tokens=len(fixture.target_prompt_token_ids),
        )
        source_prefill_delta = vllm_prefill_work_snapshot_delta(
            initial_prefill_work,
            source_prefill_work,
        )
        require_vllm_prefill_work_delta(
            source_prefill_delta,
            expected_prompt_tokens=len(fixture.source_prompt_token_ids),
        )
        target_prefill_delta = vllm_prefill_work_snapshot_delta(
            source_prefill_work,
            target_prefill_work,
        )
        require_vllm_prefill_work_delta(
            target_prefill_delta,
            expected_prompt_tokens=len(fixture.target_prompt_token_ids),
        )
        target_prefill_work_delta = target_prefill_delta
        source_timing = vllm_timing_snapshot_delta(
            initial_timing,
            parse_vllm_timing_snapshot(after_source_metrics),
        )
        require_vllm_timing_delta(source_timing, expected_requests=1)
        target_timing_delta = vllm_timing_snapshot_delta(
            parse_vllm_timing_snapshot(after_source_metrics),
            parse_vllm_timing_snapshot(after_target_metrics),
        )
        require_vllm_timing_delta(target_timing_delta, expected_requests=1)
        source_store_delta = connector_store_counter_delta(
            initial_store, after_source_store
        )
        target_store_delta = connector_store_counter_delta(
            after_source_store, after_target_store
        )
        for label, delta, expected_tokens in (
            ("source", source_store_delta, source_store_tokens),
            ("target", target_store_delta, target_store_tokens),
        ):
            if (
                delta["store_tokens_eligible"] != expected_tokens
                or delta["store_tokens_completed"] != expected_tokens
                or delta["store_fallbacks"] != 0
            ):
                raise ValueError(
                    f"{label} connector store counters do not reconcile"
                )
        connector = connector_evidence_from_snapshots(after_source, after_target)
    else:
        initial_metrics = client.get_text("/metrics")
        _require_connector_metric_surface(
            initial_metrics,
            expected=args.connector_attached_control,
        )
        if not has_vllm_timing_metric_surface(initial_metrics):
            raise ValueError("pinned vLLM timing metrics are not present")
        if not has_vllm_prefill_work_metric_surface(initial_metrics):
            raise ValueError("pinned vLLM prefill-work metrics are not present")
        initial_prompt = parse_vllm_prompt_counter_snapshot(initial_metrics)
        initial_prompt_source = parse_vllm_prompt_source_snapshot(
            initial_metrics,
            allow_missing=True,
        )
        initial_prefill_work = parse_vllm_prefill_work_snapshot(initial_metrics)
        initial_timing = parse_vllm_timing_snapshot(initial_metrics)
        initial_connector = (
            parse_connector_counter_snapshot(initial_metrics)
            if args.connector_attached_control
            else None
        )
        if args.warm_source_before_target:
            client.post_json(
                "/v1/completions",
                _completion_payload(fixture.source_prompt_token_ids, full=False),
            )
            after_source_prompt, after_source_metrics = _wait_for_prompt_counter(
                client,
                initial_prompt["prompt_tokens"]
                + len(fixture.source_prompt_token_ids),
                args.metric_wait_seconds,
                minimum_timing_count=initial_timing.ttft_seconds.count + 1,
                minimum_prompt_source_local_compute=(
                    initial_prompt_source["local_compute"]
                    + len(fixture.source_prompt_token_ids)
                ),
                minimum_prefill_observations=(
                    initial_prefill_work.observations + 1
                ),
            )
            source_prompt_delta = vllm_prompt_counter_delta(
                initial_prompt, after_source_prompt
            )
            if source_prompt_delta != len(fixture.source_prompt_token_ids):
                raise ValueError(
                    "source warm-up prompt-token delta does not match fixture"
                )
            after_source_prompt_source = parse_vllm_prompt_source_snapshot(
                after_source_metrics
            )
            require_full_prefill_prompt_source_delta(
                vllm_prompt_source_delta(
                    initial_prompt_source,
                    after_source_prompt_source,
                ),
                expected_prompt_tokens=len(fixture.source_prompt_token_ids),
            )
            after_source_prefill_work = parse_vllm_prefill_work_snapshot(
                after_source_metrics
            )
            require_vllm_prefill_work_delta(
                vllm_prefill_work_snapshot_delta(
                    initial_prefill_work,
                    after_source_prefill_work,
                ),
                expected_prompt_tokens=len(fixture.source_prompt_token_ids),
            )
            require_vllm_timing_delta(
                vllm_timing_snapshot_delta(
                    initial_timing,
                    parse_vllm_timing_snapshot(after_source_metrics),
                ),
                expected_requests=1,
            )
            initial_prompt = after_source_prompt
            initial_prompt_source = after_source_prompt_source
            initial_prefill_work = after_source_prefill_work
            initial_timing = parse_vllm_timing_snapshot(after_source_metrics)
        target_response = client.post_json(
            "/v1/completions",
            _completion_payload(fixture.target_prompt_token_ids, full=True),
        )
        after_prompt, after_metrics = _wait_for_prompt_counter(
            client,
            initial_prompt["prompt_tokens"] + len(fixture.target_prompt_token_ids),
            args.metric_wait_seconds,
            minimum_timing_count=initial_timing.ttft_seconds.count + 1,
            minimum_prompt_source_local_compute=(
                initial_prompt_source["local_compute"]
                + len(fixture.target_prompt_token_ids)
            ),
            minimum_prefill_observations=initial_prefill_work.observations + 1,
        )
        target_prompt_tokens_processed = vllm_prompt_counter_delta(
            initial_prompt, after_prompt
        )
        if target_prompt_tokens_processed != len(fixture.target_prompt_token_ids):
            raise ValueError("target native prompt-token delta does not match fixture")
        after_prompt_source = parse_vllm_prompt_source_snapshot(after_metrics)
        target_prompt_source_delta = vllm_prompt_source_delta(
            initial_prompt_source, after_prompt_source
        )
        require_full_prefill_prompt_source_delta(
            target_prompt_source_delta,
            expected_prompt_tokens=len(fixture.target_prompt_token_ids),
        )
        target_prefill_work = parse_vllm_prefill_work_snapshot(after_metrics)
        target_prefill_delta = vllm_prefill_work_snapshot_delta(
            initial_prefill_work,
            target_prefill_work,
        )
        require_vllm_prefill_work_delta(
            target_prefill_delta,
            expected_prompt_tokens=len(fixture.target_prompt_token_ids),
        )
        target_prefill_work_delta = target_prefill_delta
        target_timing_delta = vllm_timing_snapshot_delta(
            initial_timing,
            parse_vllm_timing_snapshot(after_metrics),
        )
        require_vllm_timing_delta(target_timing_delta, expected_requests=1)
        if args.connector_attached_control:
            if initial_connector is None:
                raise ValueError("connector control did not capture initial counters")
            source_connector = connector_counter_delta(
                initial_connector,
                parse_connector_counter_snapshot(after_source_metrics),
            )
            target_connector = connector_counter_delta(
                parse_connector_counter_snapshot(after_source_metrics),
                parse_connector_counter_snapshot(after_metrics),
            )
            if (
                source_connector["requests"] != 1
                or source_connector["tokens_recomputed"]
                != len(fixture.source_prompt_token_ids)
                or source_connector["prefill_tokens_avoided"] != 0
                or source_connector["kv_tokens_loaded"] != 0
                or target_connector["requests"] != 1
                or target_connector["reusable_document_tokens_requested"]
                != len(fixture.target_prompt_token_ids)
                or target_connector["kv_tokens_found"]
                != len(fixture.source_prompt_token_ids)
                or target_connector["kv_tokens_loaded"] != 0
                or target_connector["kv_tokens_rejected"]
                != len(fixture.source_prompt_token_ids)
                or target_connector["tokens_recomputed"]
                != len(fixture.target_prompt_token_ids)
                or target_connector["prefill_tokens_avoided"] != 0
            ):
                raise ValueError(
                    "connector-attached scatter-disabled control counters "
                    "do not reconcile"
                )
            connector_control_evidence = {
                "source": source_connector,
                "target": target_connector,
            }
    if (
        target_prompt_tokens_processed is None
        or target_prompt_source_delta is None
        or target_prefill_work_delta is None
        or target_timing_delta is None
    ):
        raise ValueError("native target evidence was not captured")
    native_request_evidence = VllmNativeRequestEvidence(
        prompt_tokens_processed=target_prompt_tokens_processed,
        prompt_source_delta=VllmPromptSourceDelta(**target_prompt_source_delta),
        prefill_work=target_prefill_work_delta,
        timing_delta=target_timing_delta,
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
                "connector_control_evidence": connector_control_evidence,
                "native_prompt_tokens_processed": target_prompt_tokens_processed,
                "native_prompt_source_delta": target_prompt_source_delta,
                "native_request_evidence": native_request_evidence.as_dict(),
                "native_prefill_work": (
                    None
                    if target_prefill_work_delta is None
                    else {
                        "observations": target_prefill_work_delta.observations,
                        "kv_computed_tokens": (
                            target_prefill_work_delta.kv_computed_tokens
                        ),
                    }
                ),
                "vllm_timing_delta": (
                    None
                    if target_timing_delta is None
                    else target_timing_delta.as_dict()
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
