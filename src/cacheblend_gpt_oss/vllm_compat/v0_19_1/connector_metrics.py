# SPDX-License-Identifier: Apache-2.0
"""Serializable aggregate metrics for the pinned vLLM connector hook.

This module intentionally lives under the vLLM 0.19.1 compatibility boundary.
It implements the exact ``KVConnectorStats`` and ``KVConnectorPromMetrics``
contracts called by the worker/model runner and scheduler/logger processes:
https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/distributed/kv_transfer/kv_connector/v1/metrics.py#L19-L145

All Prometheus labels are inherited bounded engine labels. Request IDs, token
IDs, document identities, fingerprints, and failure text never enter metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (  # type: ignore[import-not-found]
    KVConnectorPromMetrics,
    KVConnectorStats,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig  # type: ignore[import-not-found]
    from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
        PromMetric,
        PromMetricT,
    )

_COUNTER_KEYS = (
    "requests",
    "reusable_segments_requested",
    "reusable_segments_hit",
    "reusable_document_tokens_requested",
    "kv_tokens_found",
    "kv_tokens_verified",
    "kv_tokens_rejected",
    "kv_tokens_loaded",
    "kv_tokens_scatter_suppressed",
    "tokens_recomputed",
    "layer_token_rows_recomputed",
    "layer_token_rows_avoided",
    "prefill_tokens_avoided",
    "store_tokens_eligible",
    "store_tokens_completed",
    "store_preflight_prepared_copy_operations",
    "store_preflight_submitted_copy_operations",
    "store_gather_prepared_copy_operations",
    "store_gather_submitted_copy_operations",
    "load_fallbacks",
    "store_fallbacks",
)
_LATENCY_KEYS = (
    "lookup_latency_seconds",
    "transfer_latency_seconds",
    "position_correction_latency_seconds",
    "selective_recomputation_latency_seconds",
    "store_latency_seconds",
    "store_plan_latency_seconds",
    "store_preflight_latency_seconds",
    "store_gather_latency_seconds",
    "store_lmcache_latency_seconds",
    "store_sidecar_publish_latency_seconds",
    "store_storage_preflight_latency_seconds",
    "store_preflight_prepare_latency_seconds",
    "store_preflight_enqueue_latency_seconds",
    "store_preflight_synchronize_latency_seconds",
    "store_gather_prepare_latency_seconds",
    "store_gather_enqueue_latency_seconds",
    "store_gather_synchronize_latency_seconds",
)
_ALL_KEYS = (*_COUNTER_KEYS, *_LATENCY_KEYS)


@dataclass(frozen=True, slots=True)
class CacheBlendLookupObservation:
    """Identifier-free scheduler telemetry transferred to one worker."""

    prompt_tokens: int
    reusable_segments_requested: int
    reusable_segments_hit: int
    reusable_document_tokens_requested: int
    kv_tokens_found: int
    kv_tokens_verified: int
    kv_tokens_rejected: int
    latency_seconds: float

    def __post_init__(self) -> None:
        counters = (
            self.prompt_tokens,
            self.reusable_segments_requested,
            self.reusable_segments_hit,
            self.reusable_document_tokens_requested,
            self.kv_tokens_found,
            self.kv_tokens_verified,
            self.kv_tokens_rejected,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counters
        ):
            raise ValueError("invalid CacheBlend lookup observation")
        if (
            self.reusable_segments_hit > self.reusable_segments_requested
            or self.kv_tokens_verified > self.kv_tokens_found
            or self.kv_tokens_verified > self.reusable_document_tokens_requested
            or self.kv_tokens_found > self.reusable_document_tokens_requested
            or self.kv_tokens_found != self.kv_tokens_verified + self.kv_tokens_rejected
            or self.reusable_document_tokens_requested > self.prompt_tokens
            or isinstance(self.latency_seconds, bool)
            or not isinstance(self.latency_seconds, int | float)
            or not isfinite(self.latency_seconds)
            or self.latency_seconds < 0
        ):
            raise ValueError("invalid CacheBlend lookup observation")


def _empty_data() -> dict[str, list[int | float]]:
    return {key: [] for key in _ALL_KEYS}


def _validate_data(data: object) -> dict[str, list[int | float]]:
    if not isinstance(data, dict) or set(data) != set(_ALL_KEYS):
        raise ValueError("invalid CacheBlend connector stats schema")
    normalized: dict[str, list[int | float]] = {}
    for key in _ALL_KEYS:
        values = data[key]
        if not isinstance(values, list):
            raise ValueError("invalid CacheBlend connector stats values")
        copied: list[int | float] = []
        for value in values:
            expected_type = int if key in _COUNTER_KEYS else (int, float)
            if isinstance(value, bool) or not isinstance(value, expected_type):
                raise ValueError("invalid CacheBlend connector stats value")
            if not isfinite(value) or value < 0:
                raise ValueError("negative CacheBlend connector stats value")
            copied.append(value)
        normalized[key] = copied
    return normalized


@dataclass
class GptOssCacheBlendStats(KVConnectorStats):  # type: ignore[misc]
    """One serializable interval of identifier-free connector observations."""

    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = _empty_data() if not self.data else _validate_data(self.data)

    def reset(self) -> None:
        self.data = _empty_data()

    def is_empty(self) -> bool:
        return not any(self.data[key] for key in _ALL_KEYS)

    def clone_and_reset(self) -> GptOssCacheBlendStats:
        cloned = GptOssCacheBlendStats(
            data={key: list(values) for key, values in self.data.items()}
        )
        self.reset()
        return cloned

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if not isinstance(other, GptOssCacheBlendStats):
            raise TypeError("cannot aggregate incompatible connector stats")
        for key in _ALL_KEYS:
            self.data[key].extend(other.data[key])
        return self

    def reduce(self) -> dict[str, int | float]:
        reduced: dict[str, int | float] = {
            key: int(sum(self.data[key])) for key in _COUNTER_KEYS
        }
        for key in _LATENCY_KEYS:
            values = self.data[key]
            reduced[key] = round(float(sum(values)) / len(values), 6) if values else 0.0
        reduced["document_hit_fraction"] = _fraction(
            int(reduced["reusable_segments_hit"]),
            int(reduced["reusable_segments_requested"]),
        )
        reduced["token_hit_fraction"] = _fraction(
            int(reduced["kv_tokens_verified"]),
            int(reduced["reusable_document_tokens_requested"]),
        )
        reduced["effective_saved_prefill_fraction"] = _fraction(
            int(reduced["prefill_tokens_avoided"]),
            int(reduced["tokens_recomputed"]) + int(reduced["prefill_tokens_avoided"]),
        )
        return reduced

    def record_lookup(self, observation: CacheBlendLookupObservation) -> None:
        if not isinstance(observation, CacheBlendLookupObservation):
            raise TypeError("expected a CacheBlend lookup observation")
        self._append("requests", 1)
        self._append(
            "reusable_segments_requested",
            observation.reusable_segments_requested,
        )
        self._append("reusable_segments_hit", observation.reusable_segments_hit)
        self._append(
            "reusable_document_tokens_requested",
            observation.reusable_document_tokens_requested,
        )
        self._append("kv_tokens_found", observation.kv_tokens_found)
        self._append("kv_tokens_verified", observation.kv_tokens_verified)
        self._append("kv_tokens_rejected", observation.kv_tokens_rejected)
        self._append("lookup_latency_seconds", observation.latency_seconds)

    def record_load(
        self,
        *,
        verified_tokens: int,
        loaded_tokens: int,
        rejected_tokens: int,
        recomputed_tokens: int,
        fallback: bool,
        latency_seconds: float,
        position_correction_latency_seconds: float = 0.0,
        selective_recomputation_latency_seconds: float = 0.0,
        scatter_suppressed_tokens: int = 0,
        layer_token_rows_recomputed: int = 0,
        layer_token_rows_avoided: int = 0,
    ) -> None:
        counters = (
            verified_tokens,
            loaded_tokens,
            rejected_tokens,
            recomputed_tokens,
            scatter_suppressed_tokens,
            layer_token_rows_recomputed,
            layer_token_rows_avoided,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counters
        ):
            raise ValueError("CacheBlend load counters require non-negative integers")
        if not isinstance(fallback, bool):
            raise ValueError("CacheBlend load fallback requires a boolean")
        if verified_tokens != loaded_tokens + rejected_tokens:
            raise ValueError("verified KV tokens must be loaded or rejected")
        if scatter_suppressed_tokens and loaded_tokens:
            raise ValueError(
                "a suppressed-scatter observation must not also report loaded KV"
            )
        self._append("kv_tokens_loaded", loaded_tokens)
        self._append("kv_tokens_scatter_suppressed", scatter_suppressed_tokens)
        self._append("kv_tokens_rejected", rejected_tokens)
        self._append("tokens_recomputed", recomputed_tokens)
        self._append("layer_token_rows_recomputed", layer_token_rows_recomputed)
        self._append("layer_token_rows_avoided", layer_token_rows_avoided)
        self._append("prefill_tokens_avoided", 0)
        self._append("load_fallbacks", int(fallback))
        self._append("transfer_latency_seconds", latency_seconds)
        self._append(
            "position_correction_latency_seconds",
            position_correction_latency_seconds,
        )
        self._append(
            "selective_recomputation_latency_seconds",
            selective_recomputation_latency_seconds,
        )

    def record_store(
        self,
        *,
        eligible_tokens: int,
        stored_tokens: int,
        fallback: bool,
        latency_seconds: float,
        store_plan_latency_seconds: float = 0.0,
        store_preflight_latency_seconds: float = 0.0,
        store_gather_latency_seconds: float = 0.0,
        store_lmcache_latency_seconds: float = 0.0,
        store_sidecar_publish_latency_seconds: float = 0.0,
        store_storage_preflight_latency_seconds: float = 0.0,
        store_preflight_prepare_latency_seconds: float = 0.0,
        store_preflight_enqueue_latency_seconds: float = 0.0,
        store_preflight_synchronize_latency_seconds: float = 0.0,
        store_preflight_prepared_copy_operations: int = 0,
        store_preflight_submitted_copy_operations: int = 0,
        store_gather_prepare_latency_seconds: float = 0.0,
        store_gather_enqueue_latency_seconds: float = 0.0,
        store_gather_synchronize_latency_seconds: float = 0.0,
        store_gather_prepared_copy_operations: int = 0,
        store_gather_submitted_copy_operations: int = 0,
    ) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                eligible_tokens,
                stored_tokens,
                store_preflight_prepared_copy_operations,
                store_preflight_submitted_copy_operations,
                store_gather_prepared_copy_operations,
                store_gather_submitted_copy_operations,
            )
        ):
            raise ValueError("CacheBlend store counters require non-negative integers")
        if not isinstance(fallback, bool):
            raise ValueError("CacheBlend store fallback requires a boolean")
        if stored_tokens > eligible_tokens:
            raise ValueError("stored KV tokens cannot exceed eligible tokens")
        if (
            store_preflight_submitted_copy_operations
            > store_preflight_prepared_copy_operations
            or store_gather_submitted_copy_operations
            > store_gather_prepared_copy_operations
        ):
            raise ValueError(
                "submitted copy operations cannot exceed prepared operations"
            )
        self._append("store_tokens_eligible", eligible_tokens)
        self._append("store_tokens_completed", stored_tokens)
        self._append(
            "store_preflight_prepared_copy_operations",
            store_preflight_prepared_copy_operations,
        )
        self._append(
            "store_preflight_submitted_copy_operations",
            store_preflight_submitted_copy_operations,
        )
        self._append(
            "store_gather_prepared_copy_operations",
            store_gather_prepared_copy_operations,
        )
        self._append(
            "store_gather_submitted_copy_operations",
            store_gather_submitted_copy_operations,
        )
        self._append("store_fallbacks", int(fallback))
        self._append("store_latency_seconds", latency_seconds)
        self._append("store_plan_latency_seconds", store_plan_latency_seconds)
        self._append("store_preflight_latency_seconds", store_preflight_latency_seconds)
        self._append("store_gather_latency_seconds", store_gather_latency_seconds)
        self._append("store_lmcache_latency_seconds", store_lmcache_latency_seconds)
        self._append(
            "store_sidecar_publish_latency_seconds",
            store_sidecar_publish_latency_seconds,
        )
        self._append(
            "store_storage_preflight_latency_seconds",
            store_storage_preflight_latency_seconds,
        )
        self._append(
            "store_preflight_prepare_latency_seconds",
            store_preflight_prepare_latency_seconds,
        )
        self._append(
            "store_preflight_enqueue_latency_seconds",
            store_preflight_enqueue_latency_seconds,
        )
        self._append(
            "store_preflight_synchronize_latency_seconds",
            store_preflight_synchronize_latency_seconds,
        )
        self._append(
            "store_gather_prepare_latency_seconds",
            store_gather_prepare_latency_seconds,
        )
        self._append(
            "store_gather_enqueue_latency_seconds",
            store_gather_enqueue_latency_seconds,
        )
        self._append(
            "store_gather_synchronize_latency_seconds",
            store_gather_synchronize_latency_seconds,
        )

    def _append(self, key: str, value: int | float) -> None:
        if (
            key not in self.data
            or isinstance(value, bool)
            or not isinstance(value, int | float)
            or not isfinite(value)
            or value < 0
        ):
            raise ValueError("invalid CacheBlend connector observation")
        if key in _COUNTER_KEYS and not isinstance(value, int):
            raise ValueError("CacheBlend counters require integer observations")
        self.data[key].append(value)


def _fraction(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


class GptOssCacheBlendPromMetrics(KVConnectorPromMetrics):  # type: ignore[misc]
    """Prometheus counters/histograms with bounded engine labels only."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> None:
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        self._counters = {
            key: self._counter_cls(
                name=f"vllm:cacheblend_{key}_total",
                documentation=f"CacheBlend aggregate {key.replace('_', ' ')}.",
                labelnames=labelnames,
            )
            for key in _COUNTER_KEYS
        }
        latency_buckets = (
            0.0001,
            0.0005,
            0.001,
            0.005,
            0.01,
            0.05,
            0.1,
            0.5,
            1.0,
            5.0,
        )
        self._histograms = {
            key: self._histogram_cls(
                name=f"vllm:cacheblend_{key}",
                documentation=f"CacheBlend {key.replace('_', ' ')}.",
                labelnames=labelnames,
                buckets=latency_buckets,
            )
            for key in _LATENCY_KEYS
        }
        self._gauges = {
            key: self._gauge_cls(
                name=f"vllm:cacheblend_{key}",
                documentation=f"CacheBlend {key.replace('_', ' ')}.",
                labelnames=labelnames,
            )
            for key in (
                "document_hit_fraction",
                "token_hit_fraction",
                "effective_saved_prefill_fraction",
            )
        }

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0) -> None:
        data = _validate_data(transfer_stats_data)
        labels = self.per_engine_labelvalues[engine_idx]
        for key in _COUNTER_KEYS:
            metric = self._counters[key].labels(*labels)
            metric.inc(sum(data[key]))
        for key in _LATENCY_KEYS:
            metric = self._histograms[key].labels(*labels)
            for value in data[key]:
                metric.observe(value)
        reduced = GptOssCacheBlendStats(data=data).reduce()
        for key, metric_family in self._gauges.items():
            metric_family.labels(*labels).set(reduced[key])


__all__ = [
    "CacheBlendLookupObservation",
    "GptOssCacheBlendPromMetrics",
    "GptOssCacheBlendStats",
]
