"""CPU-only tests for the pinned vLLM scheduler lookup runtime."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import FrozenInstanceError, replace
from typing import cast

import pytest

from cacheblend_gpt_oss.planner import (
    InMemoryRecordIndex,
    SegmentFingerprint,
    TokenRange,
    TokenSegment,
    build_cache_record,
)
from cacheblend_gpt_oss.planner.models import CacheNamespace, CacheRecord
from cacheblend_gpt_oss.storage.lmcache_types import (
    LMCACHE_BLEND_PROTOCOL,
    LMCACHE_CACHE_KEY_PREFIX,
    LMCACHE_HASH_ALGORITHM,
    LMCACHE_SOURCE_COMMIT,
    LMCACHE_VERSION,
    LmcacheBlendTransportConfig,
    LmcacheCandidate,
    LmcacheOperationError,
    LmcacheProtocolError,
    LmcacheServerAttestation,
    query_digest,
)
from cacheblend_gpt_oss.storage.lookup import LmcacheCandidateLookupCoordinator
from cacheblend_gpt_oss.storage.sidecar import (
    SidecarCorruptionError,
    SidecarErrorCode,
    SidecarOperationError,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.scheduler_runtime import (
    SchedulerLookupRequest,
    SchedulerLookupRuntime,
    SchedulerLookupStatus,
    SchedulerRuntimeError,
    SchedulerRuntimeErrorCode,
    SchedulerRuntimeState,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.transfer_config import (
    PinnedLmcacheServerAttestation,
    Transfer100PctConfig,
    TransferFailurePolicy,
)


def _transfer_config(*, capacity: int = 512) -> Transfer100PctConfig:
    return Transfer100PctConfig(
        lmcache_server_url="tcp://127.0.0.1:5555",
        sidecar_path="/tmp/cacheblend-sidecar-test.sqlite3",
        lmcache_server_attestation=PinnedLmcacheServerAttestation(
            lmcache_version=LMCACHE_VERSION,
            source_commit=LMCACHE_SOURCE_COMMIT,
            protocol=LMCACHE_BLEND_PROTOCOL,
            hash_algorithm=LMCACHE_HASH_ALGORITHM,
        ),
        model_revision="model-revision",
        tokenizer_revision="tokenizer-revision",
        model_config_digest="a" * 64,
        kv_cache_config_digest="b" * 64,
        adapter_revision="adapter-revision",
        staging_token_capacity=capacity,
        request_timeout_seconds=10.0,
        transfer_failure_policy=TransferFailurePolicy.FULL_PREFILL,
    )


def _transport_config(config: Transfer100PctConfig) -> LmcacheBlendTransportConfig:
    return LmcacheBlendTransportConfig(
        namespace=config.namespace,
        server_attestation=LmcacheServerAttestation(
            lmcache_version=LMCACHE_VERSION,
            source_commit=LMCACHE_SOURCE_COMMIT,
            protocol=LMCACHE_BLEND_PROTOCOL,
            hash_algorithm=LMCACHE_HASH_ALGORITHM,
        ),
        request_timeout_seconds=config.request_timeout_seconds,
    )


CandidateFactory = Callable[[tuple[int, ...]], object]


class FakeCandidateTransport:
    def __init__(
        self,
        config: LmcacheBlendTransportConfig,
        candidate_factory: CandidateFactory | None = None,
        *,
        open_error: BaseException | None = None,
        lookup_error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.config = config
        self.candidate_factory = candidate_factory or (lambda _prompt: ())
        self.open_error = open_error
        self.lookup_error = lookup_error
        self.close_error = close_error
        self.open_calls = 0
        self.lookup_calls: list[tuple[tuple[int, ...], str]] = []
        self.close_calls = 0

    def open(self) -> None:
        self.open_calls += 1
        if self.open_error is not None:
            raise self.open_error

    def lookup_candidates(
        self, token_ids: Sequence[int], *, request_id: str
    ) -> tuple[LmcacheCandidate, ...]:
        prompt = tuple(token_ids)
        self.lookup_calls.append((prompt, request_id))
        if self.lookup_error is not None:
            raise self.lookup_error
        return cast(tuple[LmcacheCandidate, ...], self.candidate_factory(prompt))

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class ErrorRecordLookup:
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.calls = 0

    def lookup(
        self,
        namespace: CacheNamespace,
        fingerprint: SegmentFingerprint,
    ) -> Sequence[CacheRecord]:
        del namespace, fingerprint
        self.calls += 1
        raise self.error


def _runtime(
    config: Transfer100PctConfig,
    transport: FakeCandidateTransport,
    coordinator: LmcacheCandidateLookupCoordinator,
    *,
    replacement_factory: Callable[[], FakeCandidateTransport] | None = None,
) -> SchedulerLookupRuntime:
    return SchedulerLookupRuntime(
        config,
        transport,
        coordinator,
        replacement_transport_factory=(
            replacement_factory
            if replacement_factory is not None
            else lambda: FakeCandidateTransport(transport.config)
        ),
    )


def _request(
    prompt: tuple[int, ...],
    *,
    request_id: str = "request-a",
    sequence_count: int = 1,
    scheduler_step_index: int = 0,
    num_computed_tokens: int = 0,
    num_external_tokens: int = 0,
    preemption_count: int = 0,
) -> SchedulerLookupRequest:
    return SchedulerLookupRequest(
        request_id=request_id,
        prompt_token_ids=prompt,
        sequence_count=sequence_count,
        scheduler_step_index=scheduler_step_index,
        num_computed_tokens=num_computed_tokens,
        num_external_tokens=num_external_tokens,
        preemption_count=preemption_count,
    )


def _candidate_for(
    prompt: tuple[int, ...],
    *,
    target_start: int,
    storage_hash: bytes = b"h" * 32,
) -> LmcacheCandidate:
    return LmcacheCandidate(
        source_relative_range=TokenRange(0, 256),
        target_range=TokenRange(target_start, target_start + 256),
        storage_hash=storage_hash,
        storage_model_name="filled-by-test",
        query_digest=query_digest(prompt),
    )


def _hit_runtime(
    *,
    source_start: int = 1_000,
    target_start: int = 3,
) -> tuple[
    SchedulerLookupRuntime,
    FakeCandidateTransport,
    tuple[int, ...],
]:
    config = _transfer_config()
    document = tuple(range(10_000, 10_256))
    prompt = (7, 8, 9, *document, 20, 21, 22, 23, 24)
    storage_hash = b"h" * 32
    record = build_cache_record(
        config.namespace,
        TokenSegment.at(source_start, document),
        LMCACHE_CACHE_KEY_PREFIX + storage_hash.hex(),
    )

    def candidates(tokens: tuple[int, ...]) -> tuple[LmcacheCandidate, ...]:
        candidate = _candidate_for(
            tokens,
            target_start=target_start,
            storage_hash=storage_hash,
        )
        return (
            replace(
                candidate,
                storage_model_name=transport.config.storage_model_name,
            ),
        )

    transport = FakeCandidateTransport(_transport_config(config))
    transport.candidate_factory = candidates
    runtime = _runtime(
        config,
        transport,
        LmcacheCandidateLookupCoordinator(InMemoryRecordIndex((record,))),
    )
    return runtime, transport, prompt


def test_moved_document_hit_retains_compact_exact_handoff_metadata() -> None:
    runtime, transport, prompt = _hit_runtime(source_start=1_000, target_start=3)

    metadata = runtime.lookup(_request(prompt))

    assert runtime.state is SchedulerRuntimeState.READY
    assert transport.open_calls == 1
    assert transport.lookup_calls == [(prompt, "request-a")]
    assert metadata.status is SchedulerLookupStatus.TRANSFER_READY
    assert metadata.should_transfer
    assert metadata.external_scheduler_tokens == 0
    assert metadata.load_kv_async is False
    assert metadata.prompt_token_ids == prompt
    assert len(metadata.query_windows) == len(prompt) - 256 + 1
    assert metadata.query_windows[0] == TokenRange(0, 256)
    assert metadata.query_windows[-1] == TokenRange(len(prompt) - 256, len(prompt))
    assert len(metadata.verified_candidates) == 1
    verified = metadata.verified_candidates[0]
    assert verified.match.record.source_range == TokenRange(1_000, 1_256)
    assert verified.match.target_segment.token_range == TokenRange(3, 259)
    assert verified.match.position_delta == -997
    assert metadata.request_plan.match_plan.matches == (verified.match,)
    assert metadata.request_plan.external_scheduler_tokens == 0
    assert "prompt_token_ids" not in repr(metadata)
    assert "prompt_token_ids" not in repr(_request(prompt))
    with pytest.raises(FrozenInstanceError):
        metadata.status = SchedulerLookupStatus.FULL_PREFILL_MISS  # type: ignore[misc]


def test_empty_transport_result_is_explicit_full_prefill_miss() -> None:
    config = _transfer_config()
    transport = FakeCandidateTransport(_transport_config(config))
    runtime = _runtime(
        config,
        transport,
        LmcacheCandidateLookupCoordinator(InMemoryRecordIndex()),
    )

    metadata = runtime.lookup(_request(tuple(range(256))))

    assert metadata.status is SchedulerLookupStatus.FULL_PREFILL_MISS
    assert metadata.status.requires_full_prefill
    assert not metadata.should_transfer
    assert metadata.verified_candidates == ()
    assert metadata.lookup_plan.counters.raw_candidates == 0


def test_unverified_candidate_is_counted_miss_not_transfer() -> None:
    config = _transfer_config()

    def candidates(prompt: tuple[int, ...]) -> tuple[LmcacheCandidate, ...]:
        return (
            replace(
                _candidate_for(prompt, target_start=0),
                storage_model_name=transport.config.storage_model_name,
            ),
        )

    transport = FakeCandidateTransport(_transport_config(config))
    transport.candidate_factory = candidates
    runtime = _runtime(
        config,
        transport,
        LmcacheCandidateLookupCoordinator(InMemoryRecordIndex()),
    )

    metadata = runtime.lookup(_request(tuple(range(256))))

    assert metadata.status is SchedulerLookupStatus.FULL_PREFILL_MISS
    assert metadata.lookup_plan.counters.raw_candidates == 1
    assert metadata.lookup_plan.counters.rejected_candidates == 1
    assert metadata.request_plan.match_plan.matches == ()


@pytest.mark.parametrize("at_open", [True, False])
def test_transport_operation_error_degrades_to_full_prefill(at_open: bool) -> None:
    config = _transfer_config()
    error = LmcacheOperationError("private transport detail")
    transport = FakeCandidateTransport(
        _transport_config(config),
        open_error=error if at_open else None,
        lookup_error=None if at_open else error,
    )
    replacement = FakeCandidateTransport(_transport_config(config))
    runtime = _runtime(
        config,
        transport,
        LmcacheCandidateLookupCoordinator(InMemoryRecordIndex()),
        replacement_factory=lambda: replacement,
    )

    first = runtime.lookup(_request(tuple(range(256))))
    second = runtime.lookup(
        _request(tuple(range(256)), request_id="request-b")
    )

    assert first.status is SchedulerLookupStatus.FULL_PREFILL_TRANSPORT_ERROR
    assert second.status is SchedulerLookupStatus.FULL_PREFILL_MISS
    assert runtime.state is SchedulerRuntimeState.READY
    assert transport.open_calls == 1
    assert len(transport.lookup_calls) == (0 if at_open else 1)
    assert transport.close_calls == 1
    assert replacement.open_calls == 1
    assert replacement.lookup_calls == [(tuple(range(256)), "request-b")]
    assert "private transport detail" not in repr(first)


def test_protocol_drift_is_fatal_and_clears_retained_prompts() -> None:
    config = _transfer_config()
    transport = FakeCandidateTransport(
        _transport_config(config),
        lookup_error=LmcacheProtocolError("unbounded protocol response"),
    )
    runtime = _runtime(
        config,
        transport,
        LmcacheCandidateLookupCoordinator(InMemoryRecordIndex()),
    )

    fatal = runtime.lookup(_request(tuple(range(256))))
    after_fatal = runtime.lookup(
        _request(tuple(range(256)), request_id="request-b")
    )

    assert fatal.status is SchedulerLookupStatus.FATAL_PROTOCOL_DRIFT
    assert after_fatal.status is SchedulerLookupStatus.FATAL_PROTOCOL_DRIFT
    assert runtime.state is SchedulerRuntimeState.FATAL
    assert runtime.active_request_count == 0
    assert len(transport.lookup_calls) == 1
    assert "unbounded protocol response" not in repr(fatal)


@pytest.mark.parametrize(
    ("sidecar_error", "expected_status", "expected_state"),
    [
        (
            SidecarOperationError(SidecarErrorCode.SQLITE_OPERATION_FAILED),
            SchedulerLookupStatus.FULL_PREFILL_SIDECAR_ERROR,
            SchedulerRuntimeState.READY,
        ),
        (
            SidecarCorruptionError(SidecarErrorCode.RECORD_CORRUPT),
            SchedulerLookupStatus.FATAL_SIDECAR_CORRUPTION,
            SchedulerRuntimeState.FATAL,
        ),
    ],
)
def test_sidecar_operation_and_corruption_have_distinct_fail_closed_statuses(
    sidecar_error: BaseException,
    expected_status: SchedulerLookupStatus,
    expected_state: SchedulerRuntimeState,
) -> None:
    config = _transfer_config()

    def candidates(prompt: tuple[int, ...]) -> tuple[LmcacheCandidate, ...]:
        return (
            replace(
                _candidate_for(prompt, target_start=0),
                storage_model_name=transport.config.storage_model_name,
            ),
        )

    transport = FakeCandidateTransport(_transport_config(config))
    transport.candidate_factory = candidates
    lookup = ErrorRecordLookup(sidecar_error)
    runtime = _runtime(
        config,
        transport,
        LmcacheCandidateLookupCoordinator(lookup),
    )

    metadata = runtime.lookup(_request(tuple(range(256))))

    assert metadata.status is expected_status
    assert runtime.state is expected_state
    assert lookup.calls == 1
    assert metadata.verified_candidates == ()


@pytest.mark.parametrize(
    ("request_factory", "expected_status"),
    [
        (
            lambda prompt: _request(prompt, sequence_count=2),
            SchedulerLookupStatus.FULL_PREFILL_SEQUENCE_INELIGIBLE,
        ),
        (
            lambda prompt: _request(prompt, scheduler_step_index=1),
            SchedulerLookupStatus.FULL_PREFILL_STEP_INELIGIBLE,
        ),
        (
            lambda prompt: _request(prompt, num_computed_tokens=1),
            SchedulerLookupStatus.FULL_PREFILL_STEP_INELIGIBLE,
        ),
    ],
)
def test_ineligible_sequence_or_step_never_opens_transport(
    request_factory: Callable[[tuple[int, ...]], SchedulerLookupRequest],
    expected_status: SchedulerLookupStatus,
) -> None:
    config = _transfer_config()
    transport = FakeCandidateTransport(_transport_config(config))
    runtime = _runtime(
        config,
        transport,
        LmcacheCandidateLookupCoordinator(InMemoryRecordIndex()),
    )

    metadata = runtime.lookup(request_factory(tuple(range(256))))

    assert metadata.status is expected_status
    assert transport.open_calls == 0
    assert transport.lookup_calls == []


def test_prompt_above_staging_capacity_falls_back_before_lookup() -> None:
    config = _transfer_config(capacity=256)
    transport = FakeCandidateTransport(_transport_config(config))
    runtime = _runtime(
        config,
        transport,
        LmcacheCandidateLookupCoordinator(InMemoryRecordIndex()),
    )

    metadata = runtime.lookup(_request(tuple(range(257))))

    assert metadata.status is SchedulerLookupStatus.FULL_PREFILL_PROMPT_TOO_LARGE
    assert metadata.query_windows == ()
    assert transport.open_calls == 0


def test_nonzero_external_tokens_is_fatal_before_transport_access() -> None:
    config = _transfer_config()
    transport = FakeCandidateTransport(_transport_config(config))
    runtime = _runtime(
        config,
        transport,
        LmcacheCandidateLookupCoordinator(InMemoryRecordIndex()),
    )

    metadata = runtime.lookup(
        _request(tuple(range(256)), num_external_tokens=1)
    )

    assert metadata.status is SchedulerLookupStatus.FATAL_NONZERO_EXTERNAL_TOKENS
    assert runtime.state is SchedulerRuntimeState.FATAL
    assert runtime.active_request_count == 0
    assert transport.open_calls == 0


@pytest.mark.parametrize(
    "candidate_result",
    [
        [],
        (object(),),
        tuple(
            LmcacheCandidate(
                source_relative_range=TokenRange(0, 256),
                target_range=TokenRange(0, 256),
                storage_hash=bytes((index + 1,)) * 32,
                storage_model_name="wrong",
                query_digest=b"q" * 32,
            )
            for index in range(2)
        ),
    ],
)
def test_malformed_or_impossible_candidate_response_is_fatal(
    candidate_result: object,
) -> None:
    config = _transfer_config()
    transport = FakeCandidateTransport(
        _transport_config(config),
        lambda _prompt: candidate_result,
    )
    runtime = _runtime(
        config,
        transport,
        LmcacheCandidateLookupCoordinator(InMemoryRecordIndex()),
    )

    metadata = runtime.lookup(_request(tuple(range(256))))

    assert metadata.status is SchedulerLookupStatus.FATAL_PROTOCOL_DRIFT
    assert runtime.state is SchedulerRuntimeState.FATAL


def test_duplicate_lookup_and_one_step_preemption_are_idempotent() -> None:
    runtime, transport, prompt = _hit_runtime()
    request = _request(prompt)

    first = runtime.lookup(request)
    duplicate = runtime.lookup(request)
    preempted_request = replace(request, preemption_count=1)
    preempted = runtime.lookup(preempted_request)
    preempted_duplicate = runtime.lookup(preempted_request)

    assert duplicate is first
    assert preempted_duplicate is preempted
    assert preempted is not first
    assert preempted.lookup_plan is first.lookup_plan
    assert preempted.allocation_generation == 1
    assert preempted.preemption_count == 1
    assert len(transport.lookup_calls) == 1


@pytest.mark.parametrize(
    "conflicting_request",
    [
        lambda request: replace(request, preemption_count=2),
        lambda request: replace(
            request,
            prompt_token_ids=(0,) * len(request.prompt_token_ids),
        ),
        lambda request: replace(request, scheduler_step_index=1),
    ],
)
def test_duplicate_conflict_or_preemption_jump_is_fatal(
    conflicting_request: Callable[[SchedulerLookupRequest], SchedulerLookupRequest],
) -> None:
    runtime, transport, prompt = _hit_runtime()
    request = _request(prompt)
    runtime.lookup(request)

    metadata = runtime.lookup(conflicting_request(request))

    assert metadata.status is SchedulerLookupStatus.FATAL_DUPLICATE_CONFLICT
    assert runtime.state is SchedulerRuntimeState.FATAL
    assert runtime.active_request_count == 0
    assert len(transport.lookup_calls) == 1


def test_discard_allows_request_id_reuse_and_is_idempotent() -> None:
    config = _transfer_config()
    transport = FakeCandidateTransport(_transport_config(config))
    runtime = _runtime(
        config,
        transport,
        LmcacheCandidateLookupCoordinator(InMemoryRecordIndex()),
    )
    request = _request(tuple(range(256)))

    first = runtime.lookup(request)
    assert runtime.discard(request.request_id) is first
    assert runtime.discard(request.request_id) is None
    second = runtime.lookup(request)

    assert second is not first
    assert len(transport.lookup_calls) == 2
    assert runtime.active_request_count == 1


def test_open_and_close_are_idempotent_and_closed_lookup_is_fatal() -> None:
    config = _transfer_config()
    transport = FakeCandidateTransport(_transport_config(config))
    runtime = _runtime(
        config,
        transport,
        LmcacheCandidateLookupCoordinator(InMemoryRecordIndex()),
    )

    assert runtime.open() is SchedulerRuntimeState.READY
    assert runtime.open() is SchedulerRuntimeState.READY
    runtime.lookup(_request(tuple(range(256))))
    runtime.close()
    runtime.close()
    closed = runtime.lookup(
        _request(tuple(range(256)), request_id="request-after-close")
    )

    assert transport.open_calls == 1
    assert transport.close_calls == 1
    assert runtime.state is SchedulerRuntimeState.CLOSED
    assert runtime.active_request_count == 0
    assert closed.status is SchedulerLookupStatus.FATAL_RUNTIME_CLOSED


def test_close_failure_is_bounded_and_still_terminal() -> None:
    config = _transfer_config()
    transport = FakeCandidateTransport(
        _transport_config(config), close_error=RuntimeError("private close detail")
    )
    runtime = _runtime(
        config,
        transport,
        LmcacheCandidateLookupCoordinator(InMemoryRecordIndex()),
    )

    with pytest.raises(SchedulerRuntimeError) as caught:
        runtime.close()

    assert caught.value.code is SchedulerRuntimeErrorCode.CLOSE_FAILED
    assert "private close detail" not in str(caught.value)
    assert runtime.state is SchedulerRuntimeState.CLOSED


def test_constructor_rejects_transport_namespace_mismatch() -> None:
    config = _transfer_config()
    other = _transfer_config()
    object.__setattr__(
        other,
        "namespace",
        replace(other.namespace, adapter_revision="different-adapter"),
    )
    transport = FakeCandidateTransport(_transport_config(other))

    with pytest.raises(SchedulerRuntimeError) as caught:
        _runtime(
            config,
            transport,
            LmcacheCandidateLookupCoordinator(InMemoryRecordIndex()),
        )

    assert caught.value.code is SchedulerRuntimeErrorCode.INVALID_CONFIG


@pytest.mark.parametrize(
    "operation",
    [
        lambda: _request(()),
        lambda: _request((1, True)),
        lambda: _request((1,), request_id=""),
        lambda: _request((1,), sequence_count=True),
        lambda: _request((1,), num_computed_tokens=2),
    ],
)
def test_request_inputs_fail_with_bounded_errors(
    operation: Callable[[], object],
) -> None:
    with pytest.raises(SchedulerRuntimeError) as caught:
        operation()

    assert caught.value.code is SchedulerRuntimeErrorCode.INVALID_INPUT
