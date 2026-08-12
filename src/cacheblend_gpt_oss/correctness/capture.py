# SPDX-License-Identifier: Apache-2.0
"""Parse pinned vLLM completion output and connector metric snapshots."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping

from cacheblend_gpt_oss.correctness.models import (
    GPT_OSS_VOCAB_SIZE,
    ConnectorCorrectnessEvidence,
    FullVocabularyLogprobs,
)

_TOKEN_KEY = re.compile(r"token_id:(\d+)")
_COUNTER_METRICS = {
    "requests": "vllm:cacheblend_requests_total",
    "reusable_document_tokens_requested": (
        "vllm:cacheblend_reusable_document_tokens_requested_total"
    ),
    "kv_tokens_found": "vllm:cacheblend_kv_tokens_found_total",
    "kv_tokens_loaded": "vllm:cacheblend_kv_tokens_loaded_total",
    "kv_tokens_rejected": "vllm:cacheblend_kv_tokens_rejected_total",
    "tokens_recomputed": "vllm:cacheblend_tokens_recomputed_total",
    "prefill_tokens_avoided": "vllm:cacheblend_prefill_tokens_avoided_total",
}


def has_connector_metric_surface(text: str) -> bool:
    """Return whether a scrape advertises this connector's request counter."""

    if not isinstance(text, str):
        raise TypeError("Prometheus snapshot must be text")
    return _COUNTER_METRICS["requests"] in text


def parse_completion_distribution(data: object) -> FullVocabularyLogprobs:
    """Require one generated token and every GPT-OSS vocabulary logprob."""

    if not isinstance(data, dict):
        raise ValueError("completion response is not an object")
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("completion response must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("completion choice is invalid")
    token_ids = choice.get("token_ids")
    if (
        not isinstance(token_ids, list)
        or len(token_ids) != 1
        or isinstance(token_ids[0], bool)
        or not isinstance(token_ids[0], int)
    ):
        raise ValueError("completion must expose exactly one sampled token ID")
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        raise ValueError("completion response has no log probabilities")
    top = logprobs.get("top_logprobs")
    if not isinstance(top, list) or len(top) != 1 or not isinstance(top[0], dict):
        raise ValueError("completion response has no full output distribution")
    raw_values = top[0]
    values: list[float | None] = [None] * GPT_OSS_VOCAB_SIZE
    for raw_key, raw_value in raw_values.items():
        if not isinstance(raw_key, str):
            raise ValueError("completion logprob token key is invalid")
        match = _TOKEN_KEY.fullmatch(raw_key)
        if match is None:
            raise ValueError("completion did not return token-ID logprob keys")
        token_id = int(match.group(1))
        if not 0 <= token_id < GPT_OSS_VOCAB_SIZE or values[token_id] is not None:
            raise ValueError("completion logprob token ID is invalid")
        if (
            isinstance(raw_value, bool)
            or not isinstance(raw_value, int | float)
            or not math.isfinite(raw_value)
        ):
            raise ValueError("completion log probability is invalid")
        values[token_id] = float(raw_value)
    if any(value is None for value in values):
        raise ValueError("completion output distribution is not full-vocabulary")
    return FullVocabularyLogprobs(
        values=tuple(value for value in values if value is not None),
        sampled_token_id=token_ids[0],
    )


def parse_connector_counter_snapshot(text: str) -> dict[str, int]:
    """Sum bounded engine samples for each connector counter."""

    if not isinstance(text, str):
        raise TypeError("Prometheus snapshot must be text")
    samples: dict[str, float] = {key: 0.0 for key in _COUNTER_METRICS}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            raise ValueError("invalid Prometheus sample")
        sample_name = parts[0].split("{", 1)[0]
        matching_keys = [
            key
            for key, metric_name in _COUNTER_METRICS.items()
            if metric_name == sample_name
        ]
        if not matching_keys:
            continue
        try:
            value = float(parts[1])
        except ValueError as exc:
            raise ValueError("invalid connector metric value") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid connector metric value")
        samples[matching_keys[0]] += value
    result: dict[str, int] = {}
    for key, value in samples.items():
        if not value.is_integer():
            raise ValueError("connector counter is not integer-valued")
        result[key] = int(value)
    return result


def connector_evidence_from_snapshots(
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> ConnectorCorrectnessEvidence:
    """Build exact per-request deltas and require one target lookup."""

    delta = connector_counter_delta(before, after)
    if delta.pop("requests") != 1:
        raise ValueError("metric interval must contain exactly one target request")
    return ConnectorCorrectnessEvidence(**delta)


def connector_counter_delta(
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> dict[str, int]:
    """Return a monotonic delta for the exact bounded connector counters."""

    if set(before) != set(_COUNTER_METRICS) or set(after) != set(_COUNTER_METRICS):
        raise ValueError("connector counter snapshot schema mismatch")
    delta: dict[str, int] = {}
    for key in _COUNTER_METRICS:
        old_value = before[key]
        new_value = after[key]
        if (
            isinstance(old_value, bool)
            or not isinstance(old_value, int)
            or isinstance(new_value, bool)
            or not isinstance(new_value, int)
            or old_value < 0
            or new_value < old_value
        ):
            raise ValueError("connector counters are invalid or moved backwards")
        delta[key] = new_value - old_value
    return delta


__all__ = [
    "connector_counter_delta",
    "connector_evidence_from_snapshots",
    "has_connector_metric_surface",
    "parse_completion_distribution",
    "parse_connector_counter_snapshot",
]
