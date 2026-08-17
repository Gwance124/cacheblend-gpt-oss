#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render and validate the exact vLLM ``--kv-transfer-config`` JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import ensure_source_path

ensure_source_path()

from cacheblend_gpt_oss.vllm_compat.v0_19_1.transfer_config import (  # noqa: E402
    LMCACHE_BLEND_PROTOCOL,
    LMCACHE_HASH_ALGORITHM,
    LMCACHE_SOURCE_COMMIT,
    parse_connector_extra_config,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("transfer_100pct", "transfer_selective"),
        default="transfer_100pct",
    )
    parser.add_argument("--lmcache-server-url", default="tcp://127.0.0.1:5555")
    parser.add_argument("--sidecar-path", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--model-config-digest", required=True)
    parser.add_argument("--kv-cache-config-digest", required=True)
    parser.add_argument("--adapter-revision", required=True)
    parser.add_argument("--staging-token-capacity", type=int, default=512)
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--transfer-evidence-path", type=Path)
    parser.add_argument("--check-layer", type=int, default=1)
    parser.add_argument("--recompute-ratio", type=float, default=0.15)
    parser.add_argument("--suffix-tokens", type=int, default=32)
    parser.add_argument(
        "--disable-kv-scatter",
        action="store_true",
        help=(
            "Diagnostic only: run lookup/retrieve/YaRN-correction but skip "
            "the scatter copy into vLLM's paged KV cache. Never claims a "
            "real transfer; kv_tokens_loaded stays zero for this run."
        ),
    )
    parser.add_argument(
        "--allow-prefix-caching",
        action="store_true",
        help="Allow vLLM prefix caching alongside CacheBlend transfer.",
    )
    args = parser.parse_args()

    extra = {
        "mode": args.mode,
        "lmcache_server_url": args.lmcache_server_url,
        "sidecar_path": str(args.sidecar_path),
        "lmcache_server_attestation": {
            "lmcache_version": "0.4.3",
            "source_commit": LMCACHE_SOURCE_COMMIT,
            "protocol": LMCACHE_BLEND_PROTOCOL,
            "hash_algorithm": LMCACHE_HASH_ALGORITHM,
        },
        "model_revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "model_config_digest": args.model_config_digest,
        "kv_cache_config_digest": args.kv_cache_config_digest,
        "adapter_revision": args.adapter_revision,
        "staging_token_capacity": args.staging_token_capacity,
        "request_timeout_seconds": args.request_timeout_seconds,
        "transfer_failure_policy": "full_prefill",
    }
    if args.mode == "transfer_selective":
        extra.update(
            {
                "check_layer": args.check_layer,
                "recompute_ratio": args.recompute_ratio,
                "suffix_tokens": args.suffix_tokens,
            }
        )
    if args.transfer_evidence_path is not None:
        extra["transfer_evidence_path"] = str(args.transfer_evidence_path)
    if args.disable_kv_scatter:
        extra["disable_kv_scatter"] = True
    if args.allow_prefix_caching:
        extra["allow_prefix_caching"] = True
    parse_connector_extra_config(extra)
    rendered = {
        "kv_connector": "GptOssCacheBlendConnector",
        "kv_connector_module_path": (
            "cacheblend_gpt_oss.vllm_compat.v0_19_1.connector"
        ),
        "kv_role": "kv_both",
        "kv_load_failure_policy": "fail",
        "kv_connector_extra_config": extra,
    }
    print(json.dumps(rendered, allow_nan=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
