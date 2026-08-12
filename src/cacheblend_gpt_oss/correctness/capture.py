# SPDX-License-Identifier: Apache-2.0
"""Parse pinned vLLM completion and bounded Prometheus snapshots.

The timing metric names are taken from the pinned vLLM 0.19.1
``PrometheusStatLogger`` implementation:

https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/metrics/loggers.py#L727-L889

Only histogram ``_count`` and ``_sum`` samples are retained.  Labels and bucket
boundaries are deliberately discarded so a scrape cannot create unbounded
request- or document-level state in the client-side artifact.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

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
_STORE_COUNTER_METRICS = {
    "store_tokens_eligible": "vllm:cacheblend_store_tokens_eligible_total",
    "store_tokens_completed": "vllm:cacheblend_store_tokens_completed_total",
    "store_fallbacks": "vllm:cacheblend_store_fallbacks_total",
}
_VLLM_TIMING_METRICS = {
    "ttft_seconds": "vllm:time_to_first_token_seconds",
    "end_to_end_latency_seconds": "vllm:e2e_request_latency_seconds",
    "queue_latency_seconds": "vllm:request_queue_time_seconds",
    "prefill_latency_seconds": "vllm:request_prefill_time_seconds",
    "decode_latency_seconds": "vllm:request_decode_time_seconds",
}


@dataclass(frozen=True, slots=True)
class VllmTimingSummary:
    """Cumulative count and sum for one pinned vLLM histogram family."""

    count: int
    sum_seconds: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or self.count < 0
        ):
            raise ValueError("timing histogram count must be a non-negative integer")
        if (
            isinstance(self.sum_seconds, bool)
            or not isinstance(self.sum_seconds, int | float)
            or not math.isfinite(float(self.sum_seconds))
            or self.sum_seconds < 0.0
        ):
            raise ValueError("timing histogram sum must be finite and non-negative")

    @property
    def mean_seconds(self) -> float | None:
        """Return the observed mean, or ``None`` when no requests were counted."""

        if self.count == 0:
            return None
        return self.sum_seconds / self.count


@dataclass(frozen=True, slots=True)
class VllmTimingSnapshot:
    """All timing families needed by the request metric contract."""

    ttft_seconds: VllmTimingSummary
    end_to_end_latency_seconds: VllmTimingSummary
    queue_latency_seconds: VllmTimingSummary
    prefill_latency_seconds: VllmTimingSummary
    decode_latency_seconds: VllmTimingSummary

    def as_dict(self) -> dict[str, dict[str, int | float | None]]:
        """Return bounded JSON-ready values without metric labels."""

        return {
            name: {
                "count": summary.count,
                "sum_seconds": summary.sum_seconds,
                "mean_seconds": summary.mean_seconds,
            }
            for name, summary in (
                ("ttft_seconds", self.ttft_seconds),
                ("end_to_end_latency_seconds", self.end_to_end_latency_seconds),
                ("queue_latency_seconds", self.queue_latency_seconds),
                ("prefill_latency_seconds", self.prefill_latency_seconds),
                ("decode_latency_seconds", self.decode_latency_seconds),
            )
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

    return _parse_counter_snapshot(text, _COUNTER_METRICS)


def parse_connector_store_counter_snapshot(text: str) -> dict[str, int]:
    """Parse store-only counters used to gate source-cache persistence."""

    return _parse_counter_snapshot(text, _STORE_COUNTER_METRICS)


def has_vllm_timing_metric_surface(text: str) -> bool:
    """Return whether all pinned vLLM timing histogram pairs are advertised."""

    if not isinstance(text, str):
        raise TypeError("Prometheus snapshot must be text")
    names: set[str] = set()
    for line in text.splitlines():
        parts = line.strip().split()
        if not parts or parts[0].startswith("#"):
            continue
        names.add(parts[0].split("{", 1)[0])
    return all(
        f"{metric}_count" in names and f"{metric}_sum" in names
        for metric in _VLLM_TIMING_METRICS.values()
    )


def parse_vllm_timing_snapshot(text: str) -> VllmTimingSnapshot:
    """Parse the pinned vLLM TTFT/prefill/queue/decode histogram families.

    Every engine/model series is summed into one bounded aggregate.  A metric
    family that is partially present (only ``_count`` or only ``_sum``) is
    rejected rather than silently treated as zero.
    """

    if not isinstance(text, str):
        raise TypeError("Prometheus snapshot must be text")
    values: dict[str, dict[str, float]] = {
        key: {"count": 0.0, "sum": 0.0, "count_seen": 0.0, "sum_seen": 0.0}
        for key in _VLLM_TIMING_METRICS
    }
    metric_to_key = {
        f"{metric}_{suffix}": (key, suffix)
        for key, metric in _VLLM_TIMING_METRICS.items()
        for suffix in ("count", "sum")
    }
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            raise ValueError("invalid Prometheus sample")
        sample_name = parts[0].split("{", 1)[0]
        match = metric_to_key.get(sample_name)
        if match is None:
            continue
        key, suffix = match
        try:
            value = float(parts[1])
        except ValueError as exc:
            raise ValueError("invalid vLLM timing metric value") from exc
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("invalid vLLM timing metric value")
        values[key][suffix] += value
        values[key][f"{suffix}_seen"] += 1.0

    summaries: dict[str, VllmTimingSummary] = {}
    for key, raw in values.items():
        count_seen = raw["count_seen"] > 0.0
        sum_seen = raw["sum_seen"] > 0.0
        if count_seen != sum_seen:
            raise ValueError(f"vLLM timing metric family is incomplete: {key}")
        count = raw["count"]
        if not count.is_integer():
            raise ValueError("vLLM timing histogram count is not integer-valued")
        summaries[key] = VllmTimingSummary(
            count=int(count),
            sum_seconds=raw["sum"],
        )
    return VllmTimingSnapshot(**summaries)


def vllm_timing_snapshot_delta(
    before: VllmTimingSnapshot,
    after: VllmTimingSnapshot,
) -> VllmTimingSnapshot:
    """Return a monotonic interval delta for all pinned timing families."""

    if not isinstance(before, VllmTimingSnapshot) or not isinstance(
        after, VllmTimingSnapshot
    ):
        raise TypeError("vLLM timing snapshots have an invalid type")
    values: dict[str, VllmTimingSummary] = {}
    for key in _VLLM_TIMING_METRICS:
        old = getattr(before, key)
        new = getattr(after, key)
        if new.count < old.count or new.sum_seconds < old.sum_seconds:
            raise ValueError("vLLM timing histogram moved backwards")
        values[key] = VllmTimingSummary(
            count=new.count - old.count,
            sum_seconds=new.sum_seconds - old.sum_seconds,
        )
    return VllmTimingSnapshot(**values)


def require_vllm_timing_delta(
    delta: VllmTimingSnapshot,
    *,
    expected_requests: int,
) -> None:
    """Require one complete timing observation for each expected request."""

    if (
        isinstance(expected_requests, bool)
        or not isinstance(expected_requests, int)
        or expected_requests <= 0
    ):
        raise ValueError("expected timing request count must be positive")
    if not isinstance(delta, VllmTimingSnapshot):
        raise TypeError("vLLM timing delta has an invalid type")
    for key in _VLLM_TIMING_METRICS:
        summary = getattr(delta, key)
        if summary.count != expected_requests:
            raise ValueError(
                f"vLLM timing family {key} has {summary.count} observations, "
                f"expected {expected_requests}"
            )


def _parse_counter_snapshot(
    text: str,
    metric_names: Mapping[str, str],
) -> dict[str, int]:
    """Parse one bounded metric family without retaining labels."""

    if not isinstance(text, str):
        raise TypeError("Prometheus snapshot must be text")
    samples: dict[str, float] = {key: 0.0 for key in metric_names}
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
            for key, metric_name in metric_names.items()
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


def connector_store_counter_delta(
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> dict[str, int]:
    """Return monotonic deltas for the bounded store-counter family."""

    if set(before) != set(_STORE_COUNTER_METRICS) or set(after) != set(
        _STORE_COUNTER_METRICS
    ):
        raise ValueError("connector store counter snapshot schema mismatch")
    delta: dict[str, int] = {}
    for key in _STORE_COUNTER_METRICS:
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
            raise ValueError(
                "connector store counters are invalid or moved backwards"
            )
        delta[key] = new_value - old_value
    return delta


__all__ = [
    "VllmTimingSnapshot",
    "VllmTimingSummary",
    "connector_counter_delta",
    "connector_evidence_from_snapshots",
    "connector_store_counter_delta",
    "has_connector_metric_surface",
    "has_vllm_timing_metric_surface",
    "parse_completion_distribution",
    "parse_connector_counter_snapshot",
    "parse_connector_store_counter_snapshot",
    "parse_vllm_timing_snapshot",
    "require_vllm_timing_delta",
    "vllm_timing_snapshot_delta",
]
