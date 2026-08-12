#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate one bounded GPT-OSS Responses contract report offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cacheblend_gpt_oss.responses_evidence import (
    RESPONSES_EVIDENCE_CONTRACT,
    RESPONSES_EVIDENCE_SCHEMA_VERSION,
    ResponsesEvidenceError,
    read_responses_contract_evidence,
    responses_contract_evidence_digest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        evidence = read_responses_contract_evidence(args.input)
    except ResponsesEvidenceError as exc:
        parser.error(exc.code)
    report = {
        "schema_version": RESPONSES_EVIDENCE_SCHEMA_VERSION,
        "contract": RESPONSES_EVIDENCE_CONTRACT,
        "passed": True,
        "evidence_digest": responses_contract_evidence_digest(evidence),
        "runtime": {
            "model_id": evidence.runtime.model_id,
            "model_revision": evidence.runtime.model_revision,
            "tokenizer_revision": evidence.runtime.tokenizer_revision,
            "plugin_commit": evidence.runtime.plugin_commit,
            "vllm_version": evidence.runtime.vllm_version,
            "lmcache_version": evidence.runtime.lmcache_version,
            "torch_version": evidence.runtime.torch_version,
            "cuda_runtime": evidence.runtime.cuda_runtime,
            "gpu_name": evidence.runtime.gpu_name,
        },
        "turn_count": len(evidence.turns),
        "native_prompt_tokens_processed": evidence.native_prompt_tokens_processed,
        "native_prefill_work": {
            "observations": evidence.native_prefill_work.observations,
            "kv_computed_tokens": evidence.native_prefill_work.kv_computed_tokens,
        },
        "connector_counter_delta": evidence.connector_counter_delta,
        "vllm_timing_delta": evidence.vllm_timing_delta.as_dict(),
    }
    rendered = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        try:
            with args.output.open("x", encoding="utf-8") as output:
                output.write(rendered)
        except FileExistsError:
            parser.error("file_exists")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
