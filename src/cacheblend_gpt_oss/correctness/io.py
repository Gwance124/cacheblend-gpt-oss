# SPDX-License-Identifier: Apache-2.0
"""Canonical JSON I/O for correctness artifacts and frozen tolerances."""

from __future__ import annotations

import json
import math
from hashlib import sha256
from pathlib import Path
from typing import Any

from cacheblend_gpt_oss.correctness.models import (
    ConnectorCorrectnessEvidence,
    CorrectnessArtifact,
    CorrectnessCase,
    CorrectnessRunMode,
    CorrectnessRuntimeIdentity,
    FullVocabularyLogprobs,
    PromptCaseIdentity,
    ReusableSegmentIdentity,
)


def _encode_logprob(value: float) -> float | str:
    if value == -math.inf:
        return "-inf"
    return float(value)


def _decode_logprob(value: object) -> float:
    if value == "-inf":
        return -math.inf
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("invalid correctness artifact log probability")
    return float(value)


def artifact_to_dict(artifact: CorrectnessArtifact) -> dict[str, Any]:
    """Return the only accepted canonical serializable representation."""

    runtime = artifact.runtime
    prompt = artifact.prompt
    connector = artifact.connector
    return {
        "schema_version": artifact.schema_version,
        "run_mode": artifact.run_mode.value,
        "runtime": {
            "model_id": runtime.model_id,
            "model_revision": runtime.model_revision,
            "tokenizer_revision": runtime.tokenizer_revision,
            "plugin_commit": runtime.plugin_commit,
            "model_config_digest": runtime.model_config_digest,
            "kv_cache_config_digest": runtime.kv_cache_config_digest,
            "vllm_version": runtime.vllm_version,
            "lmcache_version": runtime.lmcache_version,
            "torch_version": runtime.torch_version,
            "cuda_runtime": runtime.cuda_runtime,
            "gpu_name": runtime.gpu_name,
            "dtype": runtime.dtype,
        },
        "prompt": {
            "case": prompt.case.value,
            "source_prompt_digest": prompt.source_prompt_digest,
            "source_prompt_tokens": prompt.source_prompt_tokens,
            "target_prompt_digest": prompt.target_prompt_digest,
            "target_prompt_tokens": prompt.target_prompt_tokens,
            "reusable_segments": [
                {
                    "token_digest": segment.token_digest,
                    "tokens": segment.tokens,
                    "source_start": segment.source_start,
                    "target_start": segment.target_start,
                }
                for segment in prompt.reusable_segments
            ],
        },
        "distribution": {
            "kind": "full_vocabulary_logprobs",
            "sampled_token_id": artifact.distribution.sampled_token_id,
            "values": [
                _encode_logprob(value) for value in artifact.distribution.values
            ],
        },
        "connector": (
            None
            if connector is None
            else {
                "reusable_document_tokens_requested": (
                    connector.reusable_document_tokens_requested
                ),
                "kv_tokens_found": connector.kv_tokens_found,
                "kv_tokens_loaded": connector.kv_tokens_loaded,
                "kv_tokens_rejected": connector.kv_tokens_rejected,
                "tokens_recomputed": connector.tokens_recomputed,
                "prefill_tokens_avoided": connector.prefill_tokens_avoided,
            }
        ),
    }


def _exact_mapping(value: object, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"invalid correctness artifact {name} schema")
    return value


def artifact_from_dict(data: object) -> CorrectnessArtifact:
    """Parse fail-closed JSON data; unknown and missing fields are rejected."""

    root = _exact_mapping(
        data,
        {
            "schema_version",
            "run_mode",
            "runtime",
            "prompt",
            "distribution",
            "connector",
        },
        "root",
    )
    runtime = _exact_mapping(
        root["runtime"],
        {
            "model_id",
            "model_revision",
            "tokenizer_revision",
            "plugin_commit",
            "model_config_digest",
            "kv_cache_config_digest",
            "vllm_version",
            "lmcache_version",
            "torch_version",
            "cuda_runtime",
            "gpu_name",
            "dtype",
        },
        "runtime",
    )
    prompt = _exact_mapping(
        root["prompt"],
        {
            "case",
            "source_prompt_digest",
            "source_prompt_tokens",
            "target_prompt_digest",
            "target_prompt_tokens",
            "reusable_segments",
        },
        "prompt",
    )
    distribution = _exact_mapping(
        root["distribution"],
        {"kind", "sampled_token_id", "values"},
        "distribution",
    )
    if distribution["kind"] != "full_vocabulary_logprobs" or not isinstance(
        distribution["values"], list
    ):
        raise ValueError("invalid correctness artifact distribution")
    reusable_segments = prompt["reusable_segments"]
    if not isinstance(reusable_segments, list):
        raise ValueError("invalid correctness artifact reusable segments")
    parsed_segments: list[ReusableSegmentIdentity] = []
    for raw_segment in reusable_segments:
        segment = _exact_mapping(
            raw_segment,
            {"token_digest", "tokens", "source_start", "target_start"},
            "reusable segment",
        )
        parsed_segments.append(ReusableSegmentIdentity(**segment))
    connector_data = root["connector"]
    connector: ConnectorCorrectnessEvidence | None
    if connector_data is None:
        connector = None
    else:
        connector_mapping = _exact_mapping(
            connector_data,
            {
                "reusable_document_tokens_requested",
                "kv_tokens_found",
                "kv_tokens_loaded",
                "kv_tokens_rejected",
                "tokens_recomputed",
                "prefill_tokens_avoided",
            },
            "connector",
        )
        connector = ConnectorCorrectnessEvidence(**connector_mapping)
    return CorrectnessArtifact(
        schema_version=root["schema_version"],
        run_mode=CorrectnessRunMode(root["run_mode"]),
        runtime=CorrectnessRuntimeIdentity(**runtime),
        prompt=PromptCaseIdentity(
            case=CorrectnessCase(prompt["case"]),
            source_prompt_digest=prompt["source_prompt_digest"],
            source_prompt_tokens=prompt["source_prompt_tokens"],
            target_prompt_digest=prompt["target_prompt_digest"],
            target_prompt_tokens=prompt["target_prompt_tokens"],
            reusable_segments=tuple(parsed_segments),
        ),
        distribution=FullVocabularyLogprobs(
            values=tuple(_decode_logprob(value) for value in distribution["values"]),
            sampled_token_id=distribution["sampled_token_id"],
        ),
        connector=connector,
    )


def canonical_artifact_bytes(artifact: CorrectnessArtifact) -> bytes:
    return json.dumps(
        artifact_to_dict(artifact),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def artifact_digest(artifact: CorrectnessArtifact) -> str:
    return sha256(canonical_artifact_bytes(artifact)).hexdigest()


def read_artifact(path: Path) -> CorrectnessArtifact:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return artifact_from_dict(data)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("could not read a valid correctness artifact") from exc


def write_artifact(path: Path, artifact: CorrectnessArtifact) -> None:
    """Create one canonical artifact without overwriting prior evidence."""

    with path.open("xb") as output:
        output.write(canonical_artifact_bytes(artifact) + b"\n")


__all__ = [
    "artifact_digest",
    "artifact_from_dict",
    "artifact_to_dict",
    "canonical_artifact_bytes",
    "read_artifact",
    "write_artifact",
]
