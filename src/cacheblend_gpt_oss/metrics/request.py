"""Dependency-free request metric state and consistency checks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class MetricField(str, Enum):
    """Bounded field identifiers; safe for aggregate metric labels."""

    PROMPT_TOKENS = "prompt_tokens"
    REUSABLE_DOCUMENTS_REQUESTED = "reusable_documents_requested"
    REUSABLE_DOCUMENTS_HIT = "reusable_documents_hit"
    REUSABLE_DOCUMENT_TOKENS_REQUESTED = "reusable_document_tokens_requested"
    KV_TOKENS_FOUND = "kv_tokens_found"
    KV_TOKENS_LOADED = "kv_tokens_loaded"
    KV_TOKENS_REJECTED = "kv_tokens_rejected"
    TOKENS_RECOMPUTED = "tokens_recomputed"
    PREFILL_TOKENS_AVOIDED = "prefill_tokens_avoided"
    LOOKUP_LATENCY_SECONDS = "lookup_latency_seconds"
    TRANSFER_LATENCY_SECONDS = "transfer_latency_seconds"
    POSITION_CORRECTION_LATENCY_SECONDS = "position_correction_latency_seconds"
    SELECTIVE_RECOMPUTATION_LATENCY_SECONDS = (
        "selective_recomputation_latency_seconds"
    )
    TTFT_SECONDS = "ttft_seconds"
    PREFILL_LATENCY_SECONDS = "prefill_latency_seconds"
    STORE_LATENCY_SECONDS = "store_latency_seconds"
    MAX_ABS_LOGIT_ERROR = "max_abs_logit_error"
    MEAN_ABS_LOGIT_ERROR = "mean_abs_logit_error"


class MetricInvariantCode(str, Enum):
    """Stable, bounded request metric invariant failures."""

    NEGATIVE_COUNTER = "negative_counter"
    INVALID_COUNTER_TYPE = "invalid_counter_type"
    INVALID_TIMER_TYPE = "invalid_timer_type"
    INVALID_ERROR_TYPE = "invalid_error_type"
    NONFINITE_OR_NEGATIVE_TIMER = "nonfinite_or_negative_timer"
    NONFINITE_OR_NEGATIVE_ERROR = "nonfinite_or_negative_error"
    DOCUMENT_HITS_EXCEED_REQUESTS = "document_hits_exceed_requests"
    REUSABLE_TOKENS_EXCEED_PROMPT = "reusable_tokens_exceed_prompt"
    FOUND_TOKENS_EXCEED_REQUESTED = "found_tokens_exceed_requested"
    LOADED_TOKENS_EXCEED_FOUND = "loaded_tokens_exceed_found"
    REJECTED_TOKEN_RECONCILIATION_FAILED = (
        "rejected_token_reconciliation_failed"
    )
    RECOMPUTED_TOKENS_EXCEED_PROMPT = "recomputed_tokens_exceed_prompt"
    SAVED_TOKENS_EXCEED_PROMPT = "saved_tokens_exceed_prompt"
    SAVED_TOKEN_RECONCILIATION_FAILED = "saved_token_reconciliation_failed"
    FULL_RECOMPUTE_REPORTED_SAVINGS = "full_recompute_reported_savings"


@dataclass(frozen=True, slots=True)
class RequestMetricCounters:
    """Per-request token/document counters with no identifying labels."""

    prompt_tokens: int
    reusable_documents_requested: int
    reusable_documents_hit: int
    reusable_document_tokens_requested: int
    kv_tokens_found: int
    kv_tokens_loaded: int
    kv_tokens_rejected: int
    tokens_recomputed: int
    prefill_tokens_avoided: int

    @property
    def document_hit_fraction(self) -> float:
        """Fraction of requested reusable documents with a verified KV hit."""

        return _fraction(self.reusable_documents_hit, self.reusable_documents_requested)

    @property
    def candidate_token_hit_fraction(self) -> float:
        """Fraction of requested document tokens found before verification."""

        return _fraction(self.kv_tokens_found, self.reusable_document_tokens_requested)

    @property
    def loaded_token_hit_fraction(self) -> float:
        """Fraction of requested document tokens actually loaded."""

        return _fraction(self.kv_tokens_loaded, self.reusable_document_tokens_requested)

    @property
    def effective_saved_prefill_fraction(self) -> float:
        """Fraction of prompt-token forward work reported as avoided."""

        return _fraction(self.prefill_tokens_avoided, self.prompt_tokens)

    @property
    def is_full_recomputation(self) -> bool:
        """Whether every prompt token was recomputed."""

        return self.tokens_recomputed == self.prompt_tokens


@dataclass(frozen=True, slots=True)
class RequestMetricTimers:
    """Durations in seconds; TTFT may be unavailable for non-streaming clients."""

    lookup_latency_seconds: float
    transfer_latency_seconds: float
    position_correction_latency_seconds: float
    selective_recomputation_latency_seconds: float
    ttft_seconds: float | None
    prefill_latency_seconds: float
    store_latency_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class RequestCorrectnessMetrics:
    """Optional deterministic logit-comparison errors for a request."""

    max_abs_logit_error: float | None = None
    mean_abs_logit_error: float | None = None


@dataclass(frozen=True, slots=True)
class RequestMetrics:
    """Immutable request metric snapshot passed to an aggregate exporter."""

    counters: RequestMetricCounters
    timers: RequestMetricTimers
    correctness: RequestCorrectnessMetrics = RequestCorrectnessMetrics()


@dataclass(frozen=True, slots=True)
class MetricInvariantIssue:
    """One invalid metric condition, identified without request data."""

    code: MetricInvariantCode
    field: MetricField | None = None


class InvalidRequestMetricsError(ValueError):
    """Raised when an invalid snapshot is submitted for aggregation."""

    def __init__(self, issues: tuple[MetricInvariantIssue, ...]) -> None:
        self.issues = issues
        codes = ", ".join(issue.code.value for issue in issues)
        super().__init__(f"invalid CacheBlend request metrics: {codes}")


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _valid_counter(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_request_metrics(
    metrics: RequestMetrics,
) -> tuple[MetricInvariantIssue, ...]:
    """Return every invariant violation in deterministic order."""

    counters = metrics.counters
    issues: list[MetricInvariantIssue] = []

    counter_fields = (
        (MetricField.PROMPT_TOKENS, counters.prompt_tokens),
        (
            MetricField.REUSABLE_DOCUMENTS_REQUESTED,
            counters.reusable_documents_requested,
        ),
        (MetricField.REUSABLE_DOCUMENTS_HIT, counters.reusable_documents_hit),
        (
            MetricField.REUSABLE_DOCUMENT_TOKENS_REQUESTED,
            counters.reusable_document_tokens_requested,
        ),
        (MetricField.KV_TOKENS_FOUND, counters.kv_tokens_found),
        (MetricField.KV_TOKENS_LOADED, counters.kv_tokens_loaded),
        (MetricField.KV_TOKENS_REJECTED, counters.kv_tokens_rejected),
        (MetricField.TOKENS_RECOMPUTED, counters.tokens_recomputed),
        (MetricField.PREFILL_TOKENS_AVOIDED, counters.prefill_tokens_avoided),
    )
    for counter_field, counter_value in counter_fields:
        if isinstance(counter_value, bool) or not isinstance(counter_value, int):
            issues.append(
                MetricInvariantIssue(
                    code=MetricInvariantCode.INVALID_COUNTER_TYPE,
                    field=counter_field,
                )
            )
        elif counter_value < 0:
            issues.append(
                MetricInvariantIssue(
                    code=MetricInvariantCode.NEGATIVE_COUNTER,
                    field=counter_field,
                )
            )

    timer_fields = (
        (MetricField.LOOKUP_LATENCY_SECONDS, metrics.timers.lookup_latency_seconds),
        (
            MetricField.TRANSFER_LATENCY_SECONDS,
            metrics.timers.transfer_latency_seconds,
        ),
        (
            MetricField.POSITION_CORRECTION_LATENCY_SECONDS,
            metrics.timers.position_correction_latency_seconds,
        ),
        (
            MetricField.SELECTIVE_RECOMPUTATION_LATENCY_SECONDS,
            metrics.timers.selective_recomputation_latency_seconds,
        ),
        (MetricField.TTFT_SECONDS, metrics.timers.ttft_seconds),
        (
            MetricField.PREFILL_LATENCY_SECONDS,
            metrics.timers.prefill_latency_seconds,
        ),
        (MetricField.STORE_LATENCY_SECONDS, metrics.timers.store_latency_seconds),
    )
    for timer_field, timer_value in timer_fields:
        if timer_value is not None and (
            isinstance(timer_value, bool)
            or not isinstance(timer_value, int | float)
        ):
            issues.append(
                MetricInvariantIssue(
                    code=MetricInvariantCode.INVALID_TIMER_TYPE,
                    field=timer_field,
                )
            )
        elif timer_value is not None and (
            not math.isfinite(float(timer_value)) or float(timer_value) < 0.0
        ):
            issues.append(
                MetricInvariantIssue(
                    code=MetricInvariantCode.NONFINITE_OR_NEGATIVE_TIMER,
                    field=timer_field,
                )
            )

    error_fields = (
        (MetricField.MAX_ABS_LOGIT_ERROR, metrics.correctness.max_abs_logit_error),
        (
            MetricField.MEAN_ABS_LOGIT_ERROR,
            metrics.correctness.mean_abs_logit_error,
        ),
    )
    for error_field, error_value in error_fields:
        if error_value is not None and (
            isinstance(error_value, bool)
            or not isinstance(error_value, int | float)
        ):
            issues.append(
                MetricInvariantIssue(
                    code=MetricInvariantCode.INVALID_ERROR_TYPE,
                    field=error_field,
                )
            )
        elif error_value is not None and (
            not math.isfinite(float(error_value)) or float(error_value) < 0.0
        ):
            issues.append(
                MetricInvariantIssue(
                    code=MetricInvariantCode.NONFINITE_OR_NEGATIVE_ERROR,
                    field=error_field,
                )
            )

    if (
        _valid_counter(counters.reusable_documents_hit)
        and _valid_counter(counters.reusable_documents_requested)
        and counters.reusable_documents_hit > counters.reusable_documents_requested
    ):
        issues.append(
            MetricInvariantIssue(
                code=MetricInvariantCode.DOCUMENT_HITS_EXCEED_REQUESTS,
                field=MetricField.REUSABLE_DOCUMENTS_HIT,
            )
        )
    if (
        _valid_counter(counters.reusable_document_tokens_requested)
        and _valid_counter(counters.prompt_tokens)
        and counters.reusable_document_tokens_requested > counters.prompt_tokens
    ):
        issues.append(
            MetricInvariantIssue(
                code=MetricInvariantCode.REUSABLE_TOKENS_EXCEED_PROMPT,
                field=MetricField.REUSABLE_DOCUMENT_TOKENS_REQUESTED,
            )
        )
    if (
        _valid_counter(counters.kv_tokens_found)
        and _valid_counter(counters.reusable_document_tokens_requested)
        and counters.kv_tokens_found > counters.reusable_document_tokens_requested
    ):
        issues.append(
            MetricInvariantIssue(
                code=MetricInvariantCode.FOUND_TOKENS_EXCEED_REQUESTED,
                field=MetricField.KV_TOKENS_FOUND,
            )
        )
    if (
        _valid_counter(counters.kv_tokens_loaded)
        and _valid_counter(counters.kv_tokens_found)
        and counters.kv_tokens_loaded > counters.kv_tokens_found
    ):
        issues.append(
            MetricInvariantIssue(
                code=MetricInvariantCode.LOADED_TOKENS_EXCEED_FOUND,
                field=MetricField.KV_TOKENS_LOADED,
            )
        )
    if (
        _valid_counter(counters.kv_tokens_found)
        and _valid_counter(counters.kv_tokens_loaded)
        and _valid_counter(counters.kv_tokens_rejected)
        and counters.kv_tokens_found
        != counters.kv_tokens_loaded + counters.kv_tokens_rejected
    ):
        issues.append(
            MetricInvariantIssue(
                code=MetricInvariantCode.REJECTED_TOKEN_RECONCILIATION_FAILED,
            )
        )
    if (
        _valid_counter(counters.tokens_recomputed)
        and _valid_counter(counters.prompt_tokens)
        and counters.tokens_recomputed > counters.prompt_tokens
    ):
        issues.append(
            MetricInvariantIssue(
                code=MetricInvariantCode.RECOMPUTED_TOKENS_EXCEED_PROMPT,
                field=MetricField.TOKENS_RECOMPUTED,
            )
        )
    if (
        _valid_counter(counters.prefill_tokens_avoided)
        and _valid_counter(counters.prompt_tokens)
        and counters.prefill_tokens_avoided > counters.prompt_tokens
    ):
        issues.append(
            MetricInvariantIssue(
                code=MetricInvariantCode.SAVED_TOKENS_EXCEED_PROMPT,
                field=MetricField.PREFILL_TOKENS_AVOIDED,
            )
        )

    expected_saved_tokens = (
        counters.prompt_tokens - counters.tokens_recomputed
        if _valid_counter(counters.prompt_tokens)
        and _valid_counter(counters.tokens_recomputed)
        else -1
    )
    if expected_saved_tokens >= 0 and (
        _valid_counter(counters.prefill_tokens_avoided)
        and counters.prefill_tokens_avoided != expected_saved_tokens
    ):
        issues.append(
            MetricInvariantIssue(
                code=MetricInvariantCode.SAVED_TOKEN_RECONCILIATION_FAILED,
                field=MetricField.PREFILL_TOKENS_AVOIDED,
            )
        )
    if counters.is_full_recomputation and counters.prefill_tokens_avoided != 0:
        issues.append(
            MetricInvariantIssue(
                code=MetricInvariantCode.FULL_RECOMPUTE_REPORTED_SAVINGS,
                field=MetricField.PREFILL_TOKENS_AVOIDED,
            )
        )

    return tuple(issues)


def require_valid_request_metrics(metrics: RequestMetrics) -> None:
    """Raise a structured error unless all request invariants hold."""

    issues = validate_request_metrics(metrics)
    if issues:
        raise InvalidRequestMetricsError(issues)
