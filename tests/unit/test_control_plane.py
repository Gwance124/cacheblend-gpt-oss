from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from cacheblend_gpt_oss.connector import (
    FULL_RECOMPUTE_EXTERNAL_TOKENS,
    CacheGroupLayout,
    ControlPlaneError,
    ControlPlaneErrorCode,
    RequestControlPlane,
    RequestPhase,
)
from cacheblend_gpt_oss.metrics import require_valid_request_metrics
from cacheblend_gpt_oss.metrics.request import RequestMetrics, RequestMetricTimers
from cacheblend_gpt_oss.planner import (
    CacheNamespace,
    InMemoryRecordIndex,
    MatchPlan,
    MatchPlanner,
    TokenSegment,
    build_cache_record,
)


def _namespace() -> CacheNamespace:
    return CacheNamespace(
        schema_version=1,
        model_id="openai/gpt-oss-20b",
        model_revision="model-revision",
        tokenizer_id="openai/gpt-oss-20b",
        tokenizer_revision="tokenizer-revision",
        model_config_digest="model-config-sha256",
        kv_cache_config_digest="hybrid-cache-config-sha256",
        adapter_revision="adapter-revision",
        vllm_version="0.19.1",
        lmcache_version="0.4.3",
        torch_version="2.10.0+cu128",
        cuda_runtime="12.8",
    )


def _layout() -> CacheGroupLayout:
    return CacheGroupLayout(
        (
            ("model.layers.0.attn", "model.layers.2.attn"),
            ("model.layers.1.attn", "model.layers.3.attn"),
        )
    )


def _plan(
    query_segments: tuple[TokenSegment, ...],
    cached_segments: tuple[tuple[TokenSegment, str], ...],
) -> MatchPlan:
    records = tuple(
        build_cache_record(_namespace(), segment, cache_key)
        for segment, cache_key in cached_segments
    )
    return MatchPlanner(InMemoryRecordIndex(records)).plan(
        _namespace(), query_segments
    )


def _timers() -> RequestMetricTimers:
    return RequestMetricTimers(
        lookup_latency_seconds=0.0,
        transfer_latency_seconds=0.0,
        position_correction_latency_seconds=0.0,
        selective_recomputation_latency_seconds=0.0,
        ttft_seconds=None,
        prefill_latency_seconds=0.0,
    )


def _assert_error(
    expected: ControlPlaneErrorCode,
    operation: Callable[[], object],
) -> None:
    with pytest.raises(ControlPlaneError) as caught:
        operation()
    assert caught.value.code is expected


def test_moved_document_flows_through_scheduler_and_worker_with_full_recompute() -> (
    None
):
    cached = TokenSegment.at(100, [11, 12, 13, 14])
    moved = TokenSegment.at(2, [11, 12, 13, 14])
    match_plan = _plan((moved,), ((cached, "document-a"),))
    scheduler = RequestControlPlane(_layout())

    looked_up = scheduler.lookup(
        request_id="request-moved",
        prompt_tokens=8,
        query_segments=(moved,),
        match_plan=match_plan,
    )

    assert looked_up.phase is RequestPhase.LOOKED_UP
    assert looked_up.plan.match_plan.matches[0].position_delta == -98
    assert looked_up.plan.external_scheduler_tokens == 0
    assert scheduler.external_scheduler_tokens("request-moved") == 0

    mutable_blocks = [[3, 4], [11, 12]]
    allocated = scheduler.allocate(
        "request-moved",
        mutable_blocks,
        external_scheduler_tokens=FULL_RECOMPUTE_EXTERNAL_TOKENS,
    )
    mutable_blocks[0][0] = 999
    assert allocated.allocation is not None
    assert allocated.allocation.grouped_blocks.block_ids_by_group == (
        (3, 4),
        (11, 12),
    )

    metadata = scheduler.handoff("request-moved")
    worker = RequestControlPlane(_layout())
    worker_state = worker.accept_handoff(metadata)
    assert worker_state.phase is RequestPhase.HANDED_OFF
    receipt = worker.validate_worker(
        "request-moved",
        loaded_match_indexes=(0,),
        rejected_match_indexes=(),
    )
    worker.finish("request-moved")

    validated = scheduler.apply_worker_validation(receipt)
    counters = validated.derive_metric_counters()
    require_valid_request_metrics(RequestMetrics(counters, _timers()))
    assert counters.reusable_documents_requested == 1
    assert counters.reusable_documents_hit == 1
    assert counters.kv_tokens_found == 4
    assert counters.kv_tokens_loaded == 4
    assert counters.kv_tokens_rejected == 0
    assert counters.tokens_recomputed == 8
    assert counters.prefill_tokens_avoided == 0
    assert scheduler.finish("request-moved").phase is RequestPhase.FINISHED


def test_reordered_documents_have_deterministic_match_and_group_order() -> None:
    cached_a = TokenSegment.at(20, [1, 2, 3])
    cached_b = TokenSegment.at(40, [4, 5, 6])
    requested_b = TokenSegment.at(0, [4, 5, 6])
    requested_a = TokenSegment.at(4, [1, 2, 3])
    match_plan = _plan(
        (requested_b, requested_a),
        ((cached_a, "document-a"), (cached_b, "document-b")),
    )
    scheduler = RequestControlPlane(_layout())
    state = scheduler.lookup(
        request_id="request-reordered",
        prompt_tokens=7,
        # Deliberately reverse input ordering; RequestPlan canonicalizes it.
        query_segments=(requested_a, requested_b),
        match_plan=match_plan,
    )
    assert state.plan.query_segments == (requested_b, requested_a)
    assert [
        match.record.cache_key for match in state.plan.match_plan.matches
    ] == ["document-b", "document-a"]

    state = scheduler.allocate(
        "request-reordered",
        ((8, 9), (1, 7)),
        external_scheduler_tokens=0,
    )
    assert state.allocation is not None
    assert [
        group.group_index for group in state.allocation.grouped_blocks.groups
    ] == [0, 1]
    assert state.allocation.grouped_blocks.block_ids_by_group == (
        (8, 9),
        (1, 7),
    )

    scheduler.handoff("request-reordered")
    receipt = scheduler.validate_worker(
        "request-reordered",
        loaded_match_indexes=(1, 0),
        rejected_match_indexes=(),
    )
    # Index input ordering is canonicalized, making duplicate receipts equal.
    assert receipt.loaded_match_indexes == (0, 1)
    assert scheduler.validate_worker(
        "request-reordered",
        loaded_match_indexes=(0, 1),
        rejected_match_indexes=(),
    ) == receipt
    counters = scheduler.state("request-reordered").derive_metric_counters()
    assert counters.reusable_documents_hit == 2
    assert counters.kv_tokens_loaded == 6
    assert counters.tokens_recomputed == 7


def test_cache_miss_still_requires_worker_validation_and_full_prefill() -> None:
    query = TokenSegment.at(3, [8, 9, 10])
    miss = _plan((query,), ())
    control = RequestControlPlane(_layout())
    control.lookup(
        request_id="request-miss",
        prompt_tokens=8,
        query_segments=(query,),
        match_plan=miss,
    )
    control.allocate(
        "request-miss", ((2,), (6,)), external_scheduler_tokens=0
    )
    control.handoff("request-miss")
    control.validate_worker(
        "request-miss",
        loaded_match_indexes=(),
        rejected_match_indexes=(),
    )

    counters = control.state("request-miss").derive_metric_counters()
    assert counters.reusable_documents_requested == 1
    assert counters.reusable_documents_hit == 0
    assert counters.kv_tokens_found == 0
    assert counters.kv_tokens_loaded == 0
    assert counters.tokens_recomputed == 8
    assert counters.prefill_tokens_avoided == 0


def test_discard_is_idempotent_for_completed_and_cancelled_requests() -> None:
    query = TokenSegment.at(0, [1, 2, 3])
    control = RequestControlPlane(_layout())
    looked_up = control.lookup(
        request_id="request-discard",
        prompt_tokens=3,
        query_segments=(query,),
        match_plan=_plan((query,), ()),
    )

    assert control.discard("request-discard") is looked_up
    assert control.discard("request-discard") is None
    _assert_error(
        ControlPlaneErrorCode.UNKNOWN_REQUEST,
        lambda: control.state("request-discard"),
    )


def test_nonzero_external_scheduler_tokens_fail_closed() -> None:
    query = TokenSegment.at(0, [1, 2, 3])
    control = RequestControlPlane(_layout())
    control.lookup(
        request_id="request-zero",
        prompt_tokens=3,
        query_segments=(query,),
        match_plan=_plan((query,), ((query, "document"),)),
    )

    _assert_error(
        ControlPlaneErrorCode.NONZERO_EXTERNAL_TOKENS,
        lambda: control.allocate(
            "request-zero", ((1,), (2,)), external_scheduler_tokens=1
        ),
    )
    _assert_error(
        ControlPlaneErrorCode.NONZERO_EXTERNAL_TOKENS,
        lambda: control.allocate(
            "request-zero", ((1,), (2,)), external_scheduler_tokens=False
        ),
    )
    assert control.state("request-zero").phase is RequestPhase.LOOKED_UP


def test_group_count_and_worker_layout_mismatches_fail_closed() -> None:
    query = TokenSegment.at(0, [1, 2])
    scheduler = RequestControlPlane(_layout())
    scheduler.lookup(
        request_id="request-groups",
        prompt_tokens=2,
        query_segments=(query,),
        match_plan=_plan((query,), ((query, "document"),)),
    )
    _assert_error(
        ControlPlaneErrorCode.GROUP_COUNT_MISMATCH,
        lambda: scheduler.allocate(
            "request-groups", ((1,),), external_scheduler_tokens=0
        ),
    )

    scheduler.allocate(
        "request-groups", ((1,), (2,)), external_scheduler_tokens=0
    )
    metadata = scheduler.handoff("request-groups")
    wrong_worker_layout = CacheGroupLayout(
        (
            ("model.layers.0.attn",),
            ("model.layers.1.attn",),
        )
    )
    worker = RequestControlPlane(wrong_worker_layout)
    _assert_error(
        ControlPlaneErrorCode.GROUP_LAYOUT_MISMATCH,
        lambda: worker.accept_handoff(metadata),
    )


def test_lifecycle_misuse_duplicate_conflict_and_preemption_are_bounded() -> None:
    query = TokenSegment.at(0, [1, 2, 3])
    match_plan = _plan((query,), ((query, "document"),))
    scheduler = RequestControlPlane(_layout())
    first_lookup = scheduler.lookup(
        request_id="request-lifecycle",
        prompt_tokens=3,
        query_segments=(query,),
        match_plan=match_plan,
    )
    assert scheduler.lookup(
        request_id="request-lifecycle",
        prompt_tokens=3,
        query_segments=(query,),
        match_plan=match_plan,
    ) is first_lookup

    _assert_error(
        ControlPlaneErrorCode.LIFECYCLE_MISUSE,
        lambda: scheduler.handoff("request-lifecycle"),
    )
    scheduler.allocate(
        "request-lifecycle", ((1,), (2,)), external_scheduler_tokens=0
    )
    # Exact duplicate allocation is idempotent; a conflicting snapshot is not.
    duplicate = scheduler.allocate(
        "request-lifecycle", ((1,), (2,)), external_scheduler_tokens=0
    )
    assert duplicate.phase is RequestPhase.ALLOCATED
    _assert_error(
        ControlPlaneErrorCode.DUPLICATE_REQUEST_CONFLICT,
        lambda: scheduler.allocate(
            "request-lifecycle", ((10,), (20,)), external_scheduler_tokens=0
        ),
    )

    old_metadata = scheduler.handoff("request-lifecycle")
    old_worker = RequestControlPlane(_layout())
    old_worker.accept_handoff(old_metadata)
    stale_receipt = old_worker.validate_worker(
        "request-lifecycle",
        loaded_match_indexes=(0,),
        rejected_match_indexes=(),
    )

    preempted = scheduler.preempt("request-lifecycle")
    assert preempted.phase is RequestPhase.LOOKED_UP
    assert preempted.allocation_generation == 1
    assert preempted.preemption_count == 1
    # Duplicate preemption before reallocation is harmless.
    assert scheduler.preempt("request-lifecycle") is preempted
    _assert_error(
        ControlPlaneErrorCode.STALE_ALLOCATION,
        lambda: scheduler.apply_worker_validation(stale_receipt),
    )

    scheduler.allocate(
        "request-lifecycle", ((10,), (20,)), external_scheduler_tokens=0
    )
    scheduler.handoff("request-lifecycle")
    _assert_error(
        ControlPlaneErrorCode.INVALID_WORKER_RESULT,
        lambda: scheduler.validate_worker(
            "request-lifecycle",
            loaded_match_indexes=(),
            rejected_match_indexes=(),
        ),
    )
    scheduler.validate_worker(
        "request-lifecycle",
        loaded_match_indexes=(),
        rejected_match_indexes=(0,),
    )
    rejected_counters = scheduler.state(
        "request-lifecycle"
    ).derive_metric_counters()
    assert rejected_counters.reusable_documents_hit == 1
    assert rejected_counters.kv_tokens_found == 3
    assert rejected_counters.kv_tokens_loaded == 0
    assert rejected_counters.kv_tokens_rejected == 3
    assert rejected_counters.tokens_recomputed == 3
    assert rejected_counters.prefill_tokens_avoided == 0
    finished = scheduler.finish("request-lifecycle")
    assert scheduler.finish("request-lifecycle") is finished
    _assert_error(
        ControlPlaneErrorCode.LIFECYCLE_MISUSE,
        lambda: scheduler.preempt("request-lifecycle"),
    )


def test_conflicting_duplicate_lookup_and_unknown_request_fail_closed() -> None:
    query = TokenSegment.at(0, [1, 2])
    control = RequestControlPlane(_layout())
    control.lookup(
        request_id="duplicate",
        prompt_tokens=2,
        query_segments=(query,),
        match_plan=_plan((query,), ()),
    )
    conflicting_plan = _plan((query,), ((query, "now-present"),))

    _assert_error(
        ControlPlaneErrorCode.DUPLICATE_REQUEST_CONFLICT,
        lambda: control.lookup(
            request_id="duplicate",
            prompt_tokens=2,
            query_segments=(query,),
            match_plan=conflicting_plan,
        ),
    )
    invalid_plan = replace(_plan((query,), ()), requested_tokens=0)
    _assert_error(
        ControlPlaneErrorCode.INVALID_REQUEST_PLAN,
        lambda: RequestControlPlane(_layout()).lookup(
            request_id="invalid",
            prompt_tokens=2,
            query_segments=(query,),
            match_plan=invalid_plan,
        ),
    )
    _assert_error(
        ControlPlaneErrorCode.UNKNOWN_REQUEST,
        lambda: control.external_scheduler_tokens("missing"),
    )
