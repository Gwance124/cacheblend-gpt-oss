#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Resolve omitted and explicit-false HMA flags through pinned EngineArgs.

The tri-state scheduler input is finalized during ``VllmConfig`` validation:
https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/config/vllm.py#L1190-L1258

The diagnostic constructs the exact pinned ``EngineArgs`` and resolves them
through its pinned engine-config builder:
https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/engine/arg_utils.py#L373-L604
https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/engine/arg_utils.py#L1516-L1527
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CONTRACT = "cacheblend-gpt-oss-hybrid-flag-resolution-v1"
PINNED_VLLM_VERSION = "0.19.1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _display(value: object) -> object:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return str(value)


def resolved_snapshot(config: Any) -> dict[str, object]:
    scheduler = config.scheduler_config
    cache = config.cache_config
    model = config.model_config
    parallel = config.parallel_config
    return {
        "scheduler": {
            "disable_hybrid_kv_cache_manager": (
                scheduler.disable_hybrid_kv_cache_manager
            ),
            "max_num_batched_tokens": scheduler.max_num_batched_tokens,
            "max_num_seqs": scheduler.max_num_seqs,
            "max_model_len": scheduler.max_model_len,
            "enable_chunked_prefill": scheduler.enable_chunked_prefill,
            "long_prefill_token_threshold": (scheduler.long_prefill_token_threshold),
            "async_scheduling": scheduler.async_scheduling,
            "scheduler_reserve_full_isl": scheduler.scheduler_reserve_full_isl,
            "scheduler_cls": _display(scheduler.scheduler_cls),
        },
        "cache": {
            "enable_prefix_caching": cache.enable_prefix_caching,
            "block_size": cache.block_size,
            "cache_dtype": _display(cache.cache_dtype),
            "sliding_window": cache.sliding_window,
        },
        "model": {
            "max_model_len": model.max_model_len,
            "attention_chunk_size": model.attention_chunk_size,
            "dtype": _display(model.dtype),
        },
        "parallel": {
            "tensor_parallel_size": parallel.tensor_parallel_size,
            "pipeline_parallel_size": parallel.pipeline_parallel_size,
        },
        "kv_transfer_config_present": config.kv_transfer_config is not None,
    }


def _create_config(model_path: str, raw_value: bool | None) -> object:
    from vllm.engine.arg_utils import EngineArgs

    engine_args = EngineArgs(
        model=model_path,
        served_model_name=["openai/gpt-oss-20b"],
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=131_072,
        gpu_memory_utilization=0.50,
        max_num_seqs=1,
        max_num_batched_tokens=131_072,
        long_prefill_token_threshold=0,
        async_scheduling=False,
        enforce_eager=True,
        enable_prefix_caching=True,
        kv_cache_dtype="auto",
        attention_backend="TRITON_ATTN",
        generation_config="vllm",
        max_logprobs=-1,
        disable_hybrid_kv_cache_manager=raw_value,
    )
    return engine_args.create_engine_config()


def capture(model_path: str) -> dict[str, Any]:
    observed_version = version("vllm")
    if observed_version != PINNED_VLLM_VERSION:
        raise RuntimeError(
            f"expected vLLM {PINNED_VLLM_VERSION}, found {observed_version}"
        )
    implicit = resolved_snapshot(_create_config(model_path, None))
    explicit_false = resolved_snapshot(_create_config(model_path, False))
    implicit_scheduler = implicit["scheduler"]
    if not isinstance(implicit_scheduler, dict):
        raise RuntimeError("resolved scheduler snapshot is invalid")
    final_value = implicit_scheduler["disable_hybrid_kv_cache_manager"]
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT,
        "vllm_version": observed_version,
        "model_path": model_path,
        "cases": {
            "implicit": {"raw_value": None, "resolved": implicit},
            "explicit_false": {
                "raw_value": False,
                "resolved": explicit_false,
            },
        },
        "resolved_snapshots_equal": implicit == explicit_false,
        "resolved_disable_hybrid_kv_cache_manager": final_value,
        "gate_passed": implicit == explicit_false and final_value is False,
    }


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError("hybrid-flag resolution output already exists")
    artifact = capture(args.model_path)
    rendered = json.dumps(artifact, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with args.output.open("x", encoding="utf-8") as output:
        output.write(rendered)
    print(rendered, end="")
    return 0 if artifact["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
