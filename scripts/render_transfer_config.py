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
    parser.add_argument("--lmcache-server-url", default="tcp://127.0.0.1:5555")
    parser.add_argument("--sidecar-path", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--model-config-digest", required=True)
    parser.add_argument("--kv-cache-config-digest", required=True)
    parser.add_argument("--adapter-revision", required=True)
    parser.add_argument("--staging-token-capacity", type=int, default=512)
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()

    extra = {
        "mode": "transfer_100pct",
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
