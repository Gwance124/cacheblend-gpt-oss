from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass, replace

import pytest

from cacheblend_gpt_oss.connector.control_plane import (
    METADATA_SCHEMA_VERSION,
    GroupedBlockAllocation,
    RequestAllocation,
    RequestHandoffMetadata,
    RequestPlan,
)
from cacheblend_gpt_oss.connector.control_plane import (
    CacheGroupLayout as ControlCacheGroupLayout,
)
from cacheblend_gpt_oss.gpt_oss.layout import (
    AttentionKind,
    CacheGroupLayout,
    GptOssHybridCacheLayout,
    GroupBlockTable,
)
from cacheblend_gpt_oss.planner.fingerprint import SHA256_FINGERPRINTER
from cacheblend_gpt_oss.planner.matching import MatchPlan, VerifiedMatch
from cacheblend_gpt_oss.planner.models import (
    CacheNamespace,
    CacheRecord,
    CandidateMatch,
    TokenRange,
    TokenSegment,
)
from cacheblend_gpt_oss.storage.lmcache_types import (
    LMCACHE_CACHE_KEY_PREFIX,
    LMCACHE_CHUNK_SIZE,
    LmcacheCandidate,
    LmcacheRetrieveReceipt,
    LmcacheStoreReceipt,
    VerifiedLmcacheCandidate,
    query_digest,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.adapters import AdaptedKvCacheBlocks
from cacheblend_gpt_oss.vllm_compat.v0_19_1.transfer_config import (
    LMCACHE_BLEND_PROTOCOL,
    LMCACHE_HASH_ALGORITHM,
    LMCACHE_SOURCE_COMMIT,
    PinnedLmcacheServerAttestation,
    TransferFailurePolicy,
    TransferSelectiveConfig,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.transfer_runtime import (
    FullPrefillCompletion,
    PreForwardOutcome,
    SchedulerTransferMetadata,
    TransferAttemptState,
    TransferFallbackCode,
    TransferRuntime,
    TransferRuntimeError,
    TransferRuntimeErrorCode,
    WorkerLoadPlan,
    WorkerStorePlan,
)


def _namespace(*, adapter_revision: str = "adapter-revision") -> CacheNamespace:
    return CacheNamespace(
        schema_version=1,
        model_id="openai/gpt-oss-20b",
        model_revision="model-revision",
        tokenizer_id="openai/gpt-oss-20b",
        tokenizer_revision="tokenizer-revision",
        model_config_digest="1" * 64,
        kv_cache_config_digest="2" * 64,
        adapter_revision=adapter_revision,
        vllm_version="0.19.1",
        lmcache_version="0.4.3",
        torch_version="2.10.0+cu128",
        cuda_runtime="12.8",
    )


def _layer_name(index: int) -> str:
    return f"model.layers.{index}.attn.attn"


def _layout() -> GptOssHybridCacheLayout:
    return GptOssHybridCacheLayout(
        (
            CacheGroupLayout(
                group_id=0,
                attention_kind=AttentionKind.FULL,
                layer_names=tuple(_layer_name(index) for index in range(1, 24, 2)),
                block_size=16,
                sliding_window=None,
            ),
            CacheGroupLayout(
                group_id=1,
                attention_kind=AttentionKind.SLIDING,
                layer_names=tuple(_layer_name(index) for index in range(0, 24, 2)),
                block_size=16,
                sliding_window=128,
            ),
        )
    )


def _adapted_blocks(prompt_tokens: int = 600) -> AdaptedKvCacheBlocks:
    layout = _layout()
    block_count = (prompt_tokens + 15) // 16
    ids = (
        tuple(range(block_count)),
        tuple(range(block_count, block_count * 2)),
    )
    control_layout = ControlCacheGroupLayout(
        tuple(group.layer_names for group in layout.groups)
    )
    grouped = GroupedBlockAllocation.capture(control_layout, ids)
    return AdaptedKvCacheBlocks(
        grouped_allocation=grouped,
        group_block_tables=tuple(
            GroupBlockTable(group.group_id, group.block_size, ids[group.group_id])
            for group in layout.groups
        ),
    )


def _verified_candidate(
    prompt: tuple[int, ...],
    namespace: CacheNamespace,
    *,
    target_start: int,
    source_start: int,
    hash_byte: int,
    length: int = LMCACHE_CHUNK_SIZE,
) -> VerifiedLmcacheCandidate:
    target = TokenRange(target_start, target_start + length)
    segment = TokenSegment(target, prompt[target.start : target.end])
    storage_hash = bytes([hash_byte]) * 32
    record = CacheRecord(
        namespace=namespace,
        fingerprint=SHA256_FINGERPRINTER.fingerprint(
            namespace, segment.token_ids
        ),
        token_ids=segment.token_ids,
        source_range=TokenRange(source_start, source_start + length),
        cache_key=LMCACHE_CACHE_KEY_PREFIX + storage_hash.hex(),
    )
    match = VerifiedMatch(
        CandidateMatch(segment, record.fingerprint, record)
    )
    raw = LmcacheCandidate(
        source_relative_range=TokenRange(0, length),
        target_range=target,
        storage_hash=storage_hash,
        storage_model_name="pinned-storage-namespace",
        query_digest=query_digest(prompt),
    )
    return VerifiedLmcacheCandidate.bind(
        raw, match, expected_namespace=namespace
    )


def _metadata(
    *,
    candidate_specs: tuple[tuple[int, int, int], ...] = ((256, 1024, 3),),
    transfer_eligible: bool | None = None,
    store_eligible: bool = True,
    prompt_length: int = 600,
) -> tuple[SchedulerTransferMetadata, AdaptedKvCacheBlocks]:
    prompt = tuple(range(prompt_length))
    namespace = _namespace()
    candidates = tuple(
        _verified_candidate(
            prompt,
            namespace,
            target_start=target,
            source_start=source,
            hash_byte=hash_byte,
        )
        for target, source, hash_byte in candidate_specs
    )
    matches = tuple(candidate.match for candidate in candidates)
    query_segments = tuple(match.target_segment for match in matches)
    match_plan = MatchPlan(
        matches=matches,
        rejected_candidates=(),
        requested_tokens=sum(len(segment) for segment in query_segments),
    )
    blocks = _adapted_blocks(prompt_length)
    plan = RequestPlan(
        request_id="opaque-request",
        prompt_tokens=len(prompt),
        query_segments=query_segments,
        match_plan=match_plan,
    )
    allocation = RequestAllocation(
        request_id=plan.request_id,
        allocation_generation=0,
        grouped_blocks=blocks.grouped_allocation,
    )
    handoff = RequestHandoffMetadata(
        schema_version=METADATA_SCHEMA_VERSION,
        plan=plan,
        allocation=allocation,
    )
    metadata = SchedulerTransferMetadata(
        cache_namespace=namespace,
        prompt_token_ids=prompt,
        verified_candidates=candidates,
        handoff=handoff,
        num_computed_tokens_before_step=0,
        scheduled_token_count=len(prompt),
        transfer_eligible=(
            bool(candidates) if transfer_eligible is None else transfer_eligible
        ),
        store_eligible=store_eligible,
    )
    return metadata, blocks


@dataclass
class _FakeStorage:
    calls: list[str]
    mutations: list[str]
    fail_at: str | None = None
    invalid_retrieve_receipt: bool = False
    wrong_record_namespace: bool = False

    def preflight_retrieve(self, plan: WorkerLoadPlan) -> None:
        self.calls.append("storage.preflight_retrieve")
        if self.fail_at == "storage.preflight_retrieve":
            raise RuntimeError("sensitive transport failure")

    def retrieve_verified(self, plan: WorkerLoadPlan) -> LmcacheRetrieveReceipt:
        self.calls.append("storage.retrieve_verified")
        self.mutations.append("retrieve")
        if self.fail_at == "retrieve":
            raise RuntimeError("sensitive transport failure")
        token_count = plan.expected_tokens
        if self.invalid_retrieve_receipt:
            token_count -= 1
        return LmcacheRetrieveReceipt(token_count, plan.expected_chunks)

    def preflight_store(self, plan: WorkerStorePlan) -> None:
        self.calls.append("storage.preflight_store")
        if self.fail_at == "storage.preflight_store":
            raise RuntimeError("sensitive storage failure")

    def store_precomputed(self, plan: WorkerStorePlan) -> LmcacheStoreReceipt:
        self.calls.append("storage.store_precomputed")
        self.mutations.append("store")
        if self.fail_at == "store":
            raise RuntimeError("sensitive storage failure")
        namespace = (
            _namespace(adapter_revision="wrong-adapter")
            if self.wrong_record_namespace
            else plan.metadata.cache_namespace
        )
        records = tuple(
            CacheRecord(
                namespace=namespace,
                fingerprint=SHA256_FINGERPRINTER.fingerprint(
                    namespace, chunk.token_ids
                ),
                token_ids=chunk.token_ids,
                source_range=chunk.token_range,
                cache_key=LMCACHE_CACHE_KEY_PREFIX
                + (bytes([chunk.chunk_index + 10]) * 32).hex(),
            )
            for chunk in plan.chunks
        )
        return LmcacheStoreReceipt(
            stored_tokens=plan.expected_tokens,
            stored_chunks=plan.expected_chunks,
            candidate_lookup_required=True,
            sidecar_records=records,
        )

    def publish_sidecar_records_atomically(
        self, records: tuple[CacheRecord, ...]
    ) -> int:
        self.calls.append("storage.publish_sidecar_records_atomically")
        if self.fail_at == "publish":
            raise RuntimeError("sensitive sidecar failure")
        self.mutations.append(f"publish:{len(records)}")
        return len(records)


@dataclass
class _FakeDataPlane:
    calls: list[str]
    mutations: list[str]
    fail_at: str | None = None

    def preflight_scatter(self, plan: WorkerLoadPlan) -> None:
        self.calls.append("data.preflight_scatter")
        if self.fail_at == "preflight_scatter":
            raise RuntimeError("sensitive tensor failure")

    def scatter_retrieved(self, plan: WorkerLoadPlan) -> None:
        self.calls.append("data.scatter_retrieved")
        self.mutations.append("scatter")
        if self.fail_at == "scatter":
            raise RuntimeError("sensitive tensor failure")

    def preflight_gather(self, plan: WorkerStorePlan) -> None:
        self.calls.append("data.preflight_gather")
        if self.fail_at == "preflight_gather":
            raise RuntimeError("sensitive tensor failure")

    def gather_recomputed(self, plan: WorkerStorePlan) -> None:
        self.calls.append("data.gather_recomputed")
        self.mutations.append("gather")
        if self.fail_at == "gather":
            raise RuntimeError("sensitive tensor failure")


def _runtime(
    *,
    storage_fail: str | None = None,
    data_fail: str | None = None,
    disable_kv_scatter: bool = False,
) -> tuple[TransferRuntime, _FakeStorage, _FakeDataPlane, list[str], list[str]]:
    calls: list[str] = []
    mutations: list[str] = []
    storage = _FakeStorage(calls, mutations, fail_at=storage_fail)
    data_plane = _FakeDataPlane(calls, mutations, fail_at=data_fail)
    return (
        TransferRuntime(
            _layout(), storage, data_plane, disable_kv_scatter=disable_kv_scatter
        ),
        storage,
        data_plane,
        calls,
        mutations,
    )


def _selective_config() -> TransferSelectiveConfig:
    return TransferSelectiveConfig(
        lmcache_server_url="tcp://127.0.0.1:5555",
        sidecar_path="/var/lib/cacheblend/sidecar.sqlite3",
        lmcache_server_attestation=PinnedLmcacheServerAttestation(
            lmcache_version="0.4.3",
            source_commit=LMCACHE_SOURCE_COMMIT,
            protocol=LMCACHE_BLEND_PROTOCOL,
            hash_algorithm=LMCACHE_HASH_ALGORITHM,
        ),
        model_revision="model-revision",
        tokenizer_revision="tokenizer-revision",
        model_config_digest="1" * 64,
        kv_cache_config_digest="2" * 64,
        adapter_revision="adapter-revision",
        staging_token_capacity=512,
        request_timeout_seconds=7.5,
        transfer_failure_policy=TransferFailurePolicy.FULL_PREFILL,
        check_layer=1,
        recompute_ratio=0.0,
        suffix_tokens=32,
    )


def _assert_runtime_error(
    code: TransferRuntimeErrorCode, operation: Callable[[], object]
) -> None:
    with pytest.raises(TransferRuntimeError) as caught:
        operation()
    assert caught.value.code is code
    assert "opaque-request" not in str(caught.value)


def test_moved_candidate_loads_then_full_prompt_recomputes_and_stores_chunks() -> (
    None
):
    metadata, blocks = _metadata()
    runtime, _storage, data_plane, calls, mutations = _runtime()
    data_plane.position_correction_latency_seconds = 0.25

    loaded = runtime.before_forward(metadata, blocks)

    assert calls == [
        "storage.preflight_retrieve",
        "data.preflight_scatter",
        "storage.retrieve_verified",
        "data.scatter_retrieved",
    ]
    assert mutations == ["retrieve", "scatter"]
    assert loaded.state is TransferAttemptState.SUCCEEDED
    assert loaded.loaded_candidate_indexes == (0,)
    assert loaded.rejected_candidate_indexes == ()
    assert loaded.loaded_kv_tokens == 256
    assert loaded.tokens_to_recompute == 600
    assert loaded.external_scheduler_tokens == 0
    assert loaded.prefill_tokens_avoided == 0
    assert loaded.position_correction_latency_seconds == pytest.approx(0.25)
    receipt = loaded.to_worker_validation_receipt()
    assert receipt.loaded_match_indexes == (0,)

    completion = runtime.mark_full_prefill_complete(
        loaded, recomputed_token_count=600
    )
    stored = runtime.after_forward(completion, blocks)

    assert stored.state is TransferAttemptState.SUCCEEDED
    assert stored.eligible_store_tokens == 512
    assert stored.stored_tokens == 512
    assert stored.stored_chunks == 2
    assert stored.sidecar_records_available == 2
    assert stored.sidecar_records_inserted == 2
    assert stored.prefill_tokens_avoided == 0
    assert calls[-5:] == [
        "data.preflight_gather",
        "storage.preflight_store",
        "data.gather_recomputed",
        "storage.store_precomputed",
        "storage.publish_sidecar_records_atomically",
    ]
    assert mutations[-3:] == ["gather", "store", "publish:2"]


def test_selective_transfer_emits_partial_full_shaped_row_plan() -> None:
    metadata, blocks = _metadata()
    runtime, _storage, _data_plane, _calls, _mutations = _runtime()
    runtime = TransferRuntime(
        _layout(),
        _storage,
        _data_plane,
        selective_config=_selective_config(),
    )

    outcome = runtime.before_forward(metadata, blocks)

    assert outcome.state is TransferAttemptState.SUCCEEDED
    assert outcome.row_plan is not None
    assert outcome.selective_state is not None
    assert outcome.selective_state.plan is outcome.row_plan
    assert not outcome.selective_state.scored
    assert outcome.row_plan.layer(0).is_full_recompute
    assert outcome.row_plan.layer(1).is_full_recompute
    assert outcome.row_plan.layer(2).recompute_tokens == 344
    assert outcome.layer_token_rows_recomputed == 8_768
    assert outcome.layer_token_rows_avoided == 5_632
    assert (
        outcome.layer_token_rows_recomputed + outcome.layer_token_rows_avoided
        == 24 * metadata.prompt_token_count
    )


def test_disabled_scatter_runs_worker_scatter_but_reports_fallback_zero_loaded() -> (
    None
):
    metadata, blocks = _metadata()
    runtime, _storage, data_plane, calls, mutations = _runtime(
        disable_kv_scatter=True
    )
    data_plane.position_correction_latency_seconds = 0.25

    outcome = runtime.before_forward(metadata, blocks)

    # Retrieval and the worker's scatter call still ran; only the
    # transfer-evidence accounting reports it as suppressed.
    assert calls == [
        "storage.preflight_retrieve",
        "data.preflight_scatter",
        "storage.retrieve_verified",
        "data.scatter_retrieved",
    ]
    assert mutations == ["retrieve", "scatter"]
    assert outcome.state is TransferAttemptState.FULL_PREFILL_FALLBACK
    assert (
        outcome.failure_code
        is TransferFallbackCode.SCATTER_SUPPRESSED_DIAGNOSTIC
    )
    assert outcome.loaded_candidate_indexes == ()
    assert outcome.rejected_candidate_indexes == (0,)
    assert outcome.loaded_kv_tokens == 0
    assert outcome.scatter_suppressed_tokens == 256
    assert outcome.tokens_to_recompute == 600
    assert outcome.position_correction_latency_seconds == pytest.approx(0.25)
    receipt = outcome.to_worker_validation_receipt()
    assert receipt.loaded_match_indexes == ()
    assert receipt.rejected_match_indexes == (0,)


def test_scatter_suppressed_tokens_must_be_zero_outside_the_diagnostic_fallback() -> (
    None
):
    metadata, _blocks = _metadata()
    succeeded = PreForwardOutcome(
        metadata=metadata,
        state=TransferAttemptState.SUCCEEDED,
        failure_code=None,
        loaded_candidate_indexes=(0,),
        rejected_candidate_indexes=(),
        loaded_kv_tokens=256,
        tokens_to_recompute=600,
    )
    assert succeeded.scatter_suppressed_tokens == 0
    with pytest.raises(TransferRuntimeError) as caught:
        PreForwardOutcome(
            metadata=metadata,
            state=TransferAttemptState.SUCCEEDED,
            failure_code=None,
            loaded_candidate_indexes=(0,),
            rejected_candidate_indexes=(),
            loaded_kv_tokens=256,
            tokens_to_recompute=600,
            scatter_suppressed_tokens=256,
        )
    assert caught.value.code is TransferRuntimeErrorCode.INVALID_OUTCOME

    with pytest.raises(TransferRuntimeError) as caught:
        PreForwardOutcome(
            metadata=metadata,
            state=TransferAttemptState.FULL_PREFILL_FALLBACK,
            failure_code=TransferFallbackCode.SCATTER_FAILED,
            loaded_candidate_indexes=(),
            rejected_candidate_indexes=(0,),
            loaded_kv_tokens=0,
            tokens_to_recompute=600,
            scatter_suppressed_tokens=256,
        )
    assert caught.value.code is TransferRuntimeErrorCode.INVALID_OUTCOME

    with pytest.raises(TransferRuntimeError) as caught:
        PreForwardOutcome(
            metadata=metadata,
            state=TransferAttemptState.FULL_PREFILL_FALLBACK,
            failure_code=TransferFallbackCode.SCATTER_SUPPRESSED_DIAGNOSTIC,
            loaded_candidate_indexes=(),
            rejected_candidate_indexes=(0,),
            loaded_kv_tokens=0,
            tokens_to_recompute=600,
            scatter_suppressed_tokens=0,
        )
    assert caught.value.code is TransferRuntimeErrorCode.INVALID_OUTCOME


def test_reordered_candidates_preserve_verified_identity_and_scatter_positions() -> (
    None
):
    metadata, blocks = _metadata(
        candidate_specs=((0, 2048, 4), (256, 1024, 5))
    )
    runtime, storage, data_plane, _calls, _mutations = _runtime()
    captured: list[WorkerLoadPlan] = []

    def capture(plan: WorkerLoadPlan) -> None:
        captured.append(plan)

    storage.preflight_retrieve = capture  # type: ignore[method-assign]
    data_plane.preflight_scatter = capture  # type: ignore[method-assign]
    outcome = runtime.before_forward(metadata, blocks)

    assert outcome.state is TransferAttemptState.SUCCEEDED
    assert outcome.loaded_candidate_indexes == (0, 1)
    plan = captured[0]
    assert tuple(work.verified_candidate for work in plan.candidates) == (
        metadata.verified_candidates
    )
    assert [
        work.scatter_plan.transfer.source_range for work in plan.candidates
    ] == [TokenRange(2048, 2304), TokenRange(1024, 1280)]
    assert [
        work.scatter_plan.transfer.target_range for work in plan.candidates
    ] == [TokenRange(0, 256), TokenRange(256, 512)]


def test_cache_miss_uses_no_worker_transfer_and_still_marks_full_recompute() -> None:
    metadata, blocks = _metadata(
        candidate_specs=(), transfer_eligible=False, store_eligible=False
    )
    runtime, _storage, _data_plane, calls, mutations = _runtime()

    outcome = runtime.before_forward(metadata, blocks)

    assert outcome.state is TransferAttemptState.NOT_ELIGIBLE
    assert outcome.loaded_kv_tokens == 0
    assert outcome.tokens_to_recompute == 600
    assert outcome.prefill_tokens_avoided == 0
    assert calls == []
    assert mutations == []
    completion = runtime.mark_full_prefill_complete(
        outcome, recomputed_token_count=600
    )
    stored = runtime.after_forward(completion, blocks)
    assert stored.state is TransferAttemptState.NOT_ELIGIBLE
    assert stored.eligible_store_tokens == 0
    assert stored.stored_tokens == 0
    assert calls == []


def test_metadata_is_frozen_and_requires_zero_credit_one_complete_initial_step() -> (
    None
):
    metadata, _blocks = _metadata()
    rendered = repr(metadata)
    assert "prompt_token_ids" not in rendered
    assert "verified_candidates" not in rendered
    assert "handoff" not in rendered
    assert "opaque-request" not in rendered
    assert metadata.verified_candidates[0].candidate.cache_key not in rendered
    with pytest.raises(FrozenInstanceError):
        metadata.scheduled_token_count = 1  # type: ignore[misc]

    _assert_runtime_error(
        TransferRuntimeErrorCode.NOT_INITIAL_PREFILL,
        lambda: replace(metadata, num_computed_tokens_before_step=1),
    )
    _assert_runtime_error(
        TransferRuntimeErrorCode.NOT_INITIAL_PREFILL,
        lambda: replace(metadata, num_computed_tokens_before_step=False),
    )
    _assert_runtime_error(
        TransferRuntimeErrorCode.INCOMPLETE_FULL_PROMPT_STEP,
        lambda: replace(metadata, scheduled_token_count=599),
    )
    _assert_runtime_error(
        TransferRuntimeErrorCode.INCOMPLETE_FULL_PROMPT_STEP,
        lambda: replace(metadata, scheduled_token_count=True),
    )


def test_outcomes_cannot_be_forged_to_credit_external_or_saved_tokens() -> None:
    metadata, blocks = _metadata()
    runtime, _storage, _data_plane, _calls, _mutations = _runtime()
    before = runtime.before_forward(metadata, blocks)

    _assert_runtime_error(
        TransferRuntimeErrorCode.INVALID_OUTCOME,
        lambda: replace(before, external_scheduler_tokens=1),
    )
    _assert_runtime_error(
        TransferRuntimeErrorCode.INVALID_OUTCOME,
        lambda: replace(before, prefill_tokens_avoided=1),
    )
    _assert_runtime_error(
        TransferRuntimeErrorCode.INVALID_OUTCOME,
        lambda: PreForwardOutcome(
            metadata,
            TransferAttemptState.SUCCEEDED,
            None,
            (),
            (0,),
            0,
            600,
        ),
    )

    completion = runtime.mark_full_prefill_complete(
        before, recomputed_token_count=600
    )
    stored = runtime.after_forward(completion, blocks)
    _assert_runtime_error(
        TransferRuntimeErrorCode.INVALID_OUTCOME,
        lambda: replace(stored, prefill_tokens_avoided=1),
    )


def test_metadata_rechecks_chunk_query_namespace_and_handoff_identity() -> None:
    metadata, _blocks = _metadata()
    stale_prompt = (*metadata.prompt_token_ids[:-1], 9999)
    _assert_runtime_error(
        TransferRuntimeErrorCode.CANDIDATE_MISMATCH,
        lambda: replace(metadata, prompt_token_ids=stale_prompt),
    )

    short_prompt = tuple(range(300))
    namespace = _namespace()
    short_candidate = _verified_candidate(
        short_prompt,
        namespace,
        target_start=0,
        source_start=1000,
        hash_byte=8,
        length=128,
    )
    query = short_candidate.match.target_segment
    plan = RequestPlan(
        request_id="short-request",
        prompt_tokens=len(short_prompt),
        query_segments=(query,),
        match_plan=MatchPlan((short_candidate.match,), (), len(query)),
    )
    blocks = _adapted_blocks(len(short_prompt))
    handoff = RequestHandoffMetadata(
        METADATA_SCHEMA_VERSION,
        plan,
        RequestAllocation("short-request", 0, blocks.grouped_allocation),
    )
    _assert_runtime_error(
        TransferRuntimeErrorCode.CANDIDATE_CHUNK_MISMATCH,
        lambda: SchedulerTransferMetadata(
            namespace,
            short_prompt,
            (short_candidate,),
            handoff,
            0,
            len(short_prompt),
            True,
            True,
        ),
    )

    _assert_runtime_error(
        TransferRuntimeErrorCode.NAMESPACE_MISMATCH,
        lambda: replace(
            metadata,
            cache_namespace=_namespace(adapter_revision="other-adapter"),
        ),
    )


def test_allocation_or_group_table_mismatch_falls_back_before_worker_calls() -> None:
    metadata, blocks = _metadata()
    runtime, _storage, _data_plane, calls, mutations = _runtime()
    wrong = _adapted_blocks(256)

    outcome = runtime.before_forward(metadata, wrong)

    assert outcome.state is TransferAttemptState.FULL_PREFILL_FALLBACK
    assert outcome.failure_code is TransferFallbackCode.LOAD_PLAN_REJECTED
    assert outcome.loaded_kv_tokens == 0
    assert outcome.tokens_to_recompute == 600
    assert calls == []
    assert mutations == []

    short_tables = replace(
        blocks,
        group_block_tables=tuple(
            GroupBlockTable(table.group_id, table.block_size, table.block_ids[:1])
            for table in blocks.group_block_tables
        ),
    )
    second = runtime.before_forward(metadata, short_tables)
    assert second.failure_code is TransferFallbackCode.LOAD_PLAN_REJECTED
    assert calls == []


@pytest.mark.parametrize(
    ("storage_fail", "data_fail", "invalid_receipt", "expected", "mutations"),
    [
        (
            "storage.preflight_retrieve",
            None,
            False,
            TransferFallbackCode.LOAD_PREFLIGHT_FAILED,
            [],
        ),
        (
            None,
            "preflight_scatter",
            False,
            TransferFallbackCode.LOAD_PREFLIGHT_FAILED,
            [],
        ),
        (
            "retrieve",
            None,
            False,
            TransferFallbackCode.RETRIEVE_FAILED,
            ["retrieve"],
        ),
        (
            None,
            None,
            True,
            TransferFallbackCode.RETRIEVE_RECEIPT_INVALID,
            ["retrieve"],
        ),
        (
            None,
            "scatter",
            False,
            TransferFallbackCode.SCATTER_FAILED,
            ["retrieve", "scatter"],
        ),
    ],
)
def test_every_load_failure_falls_back_with_no_reuse_or_savings_credit(
    storage_fail: str | None,
    data_fail: str | None,
    invalid_receipt: bool,
    expected: TransferFallbackCode,
    mutations: list[str],
) -> None:
    metadata, blocks = _metadata()
    runtime, storage, _data_plane, _calls, observed_mutations = _runtime(
        storage_fail=storage_fail, data_fail=data_fail
    )
    storage.invalid_retrieve_receipt = invalid_receipt

    outcome = runtime.before_forward(metadata, blocks)

    assert outcome.state is TransferAttemptState.FULL_PREFILL_FALLBACK
    assert outcome.failure_code is expected
    assert outcome.loaded_candidate_indexes == ()
    assert outcome.rejected_candidate_indexes == (0,)
    assert outcome.loaded_kv_tokens == 0
    assert outcome.tokens_to_recompute == 600
    assert outcome.prefill_tokens_avoided == 0
    assert observed_mutations == mutations


def test_forward_completion_requires_every_scheduled_prompt_token() -> None:
    metadata, blocks = _metadata()
    runtime, _storage, _data_plane, calls, mutations = _runtime()
    before = runtime.before_forward(metadata, blocks)
    calls.clear()
    mutations.clear()

    _assert_runtime_error(
        TransferRuntimeErrorCode.INVALID_FORWARD_COMPLETION,
        lambda: runtime.mark_full_prefill_complete(
            before, recomputed_token_count=599
        ),
    )
    _assert_runtime_error(
        TransferRuntimeErrorCode.INVALID_FORWARD_COMPLETION,
        lambda: runtime.mark_full_prefill_complete(
            before, recomputed_token_count=True
        ),
    )
    assert calls == []
    assert mutations == []


@pytest.mark.parametrize(
    ("storage_fail", "data_fail", "wrong_namespace", "expected", "mutations"),
    [
        (
            None,
            "preflight_gather",
            False,
            TransferFallbackCode.STORE_PREFLIGHT_FAILED,
            [],
        ),
        (
            "storage.preflight_store",
            None,
            False,
            TransferFallbackCode.STORE_PREFLIGHT_FAILED,
            [],
        ),
        (
            None,
            "gather",
            False,
            TransferFallbackCode.GATHER_FAILED,
            ["gather"],
        ),
        (
            "store",
            None,
            False,
            TransferFallbackCode.STORE_FAILED,
            ["gather", "store"],
        ),
        (
            None,
            None,
            True,
            TransferFallbackCode.STORE_RECEIPT_INVALID,
            ["gather", "store"],
        ),
        (
            "publish",
            None,
            False,
            TransferFallbackCode.SIDECAR_PUBLISH_FAILED,
            ["gather", "store"],
        ),
    ],
)
def test_every_store_failure_falls_back_without_store_or_savings_credit(
    storage_fail: str | None,
    data_fail: str | None,
    wrong_namespace: bool,
    expected: TransferFallbackCode,
    mutations: list[str],
) -> None:
    metadata, blocks = _metadata(
        candidate_specs=(), transfer_eligible=False, store_eligible=True
    )
    runtime, storage, _data_plane, calls, observed_mutations = _runtime(
        storage_fail=storage_fail, data_fail=data_fail
    )
    storage.wrong_record_namespace = wrong_namespace
    before = runtime.before_forward(metadata, blocks)
    completion = runtime.mark_full_prefill_complete(
        before, recomputed_token_count=600
    )
    calls.clear()
    observed_mutations.clear()

    outcome = runtime.after_forward(completion, blocks)

    assert outcome.state is TransferAttemptState.FULL_PREFILL_FALLBACK
    assert outcome.failure_code is expected
    assert outcome.eligible_store_tokens == 512
    assert outcome.stored_tokens == 0
    assert outcome.stored_chunks == 0
    assert outcome.sidecar_records_available == 0
    assert outcome.sidecar_records_inserted == 0
    assert outcome.prefill_tokens_avoided == 0
    assert observed_mutations == mutations
    assert "storage.publish_sidecar_records_atomically" not in calls or (
        expected is TransferFallbackCode.SIDECAR_PUBLISH_FAILED
    )


def test_load_failure_does_not_prevent_storing_the_recomputed_prompt() -> None:
    metadata, blocks = _metadata()
    runtime, storage, _data_plane, _calls, _mutations = _runtime(
        storage_fail="retrieve"
    )
    before = runtime.before_forward(metadata, blocks)
    assert before.state is TransferAttemptState.FULL_PREFILL_FALLBACK
    storage.fail_at = None
    completion = runtime.mark_full_prefill_complete(
        before, recomputed_token_count=600
    )

    stored = runtime.after_forward(completion, blocks)

    assert stored.state is TransferAttemptState.SUCCEEDED
    assert stored.stored_tokens == 512
    assert stored.prefill_tokens_avoided == 0


def test_first_request_with_no_verified_candidates_stores_the_whole_prefix() -> (
    None
):
    """No prior turn means nothing is already stored: store every chunk."""

    metadata, blocks = _metadata(
        candidate_specs=(), transfer_eligible=False, store_eligible=True
    )
    runtime, storage, data_plane, calls, mutations = _runtime()
    captured: list[WorkerStorePlan] = []

    def capture(plan: WorkerStorePlan) -> None:
        captured.append(plan)

    storage.preflight_store = capture  # type: ignore[method-assign]
    before = runtime.before_forward(metadata, blocks)
    completion = runtime.mark_full_prefill_complete(
        before, recomputed_token_count=600
    )

    stored = runtime.after_forward(completion, blocks)

    assert stored.state is TransferAttemptState.SUCCEEDED
    assert stored.eligible_store_tokens == 512
    assert stored.stored_tokens == 512
    assert stored.stored_chunks == 2
    assert len(captured[0].chunks) == 2
    assert captured[0].chunks[0].token_range == TokenRange(0, 256)
    assert captured[0].chunks[1].token_range == TokenRange(256, 512)
    assert captured[0].token_ids == metadata.prompt_token_ids[:512]
    assert captured[0].source_range == TokenRange(0, 512)
    assert "data.preflight_gather" in calls
    assert "gather" in mutations


def test_follow_up_request_stores_only_the_newly_appended_tail_chunk() -> None:
    """A verified candidate proving chunk 0 is present skips re-storing it."""

    metadata, blocks = _metadata(
        candidate_specs=((0, 4096, 7),),
        transfer_eligible=True,
        store_eligible=True,
    )
    runtime, storage, data_plane, calls, mutations = _runtime()
    captured: list[WorkerStorePlan] = []

    def capture(plan: WorkerStorePlan) -> None:
        captured.append(plan)

    storage.preflight_store = capture  # type: ignore[method-assign]
    before = runtime.before_forward(metadata, blocks)
    completion = runtime.mark_full_prefill_complete(
        before, recomputed_token_count=600
    )

    stored = runtime.after_forward(completion, blocks)

    assert stored.state is TransferAttemptState.SUCCEEDED
    # Only the second (256, 512) chunk is new; the first was already stored.
    assert stored.eligible_store_tokens == 256
    assert stored.stored_tokens == 256
    assert stored.stored_chunks == 1
    assert len(captured[0].chunks) == 1
    assert captured[0].chunks[0].chunk_index == 1
    assert captured[0].chunks[0].token_range == TokenRange(256, 512)
    assert captured[0].token_ids == metadata.prompt_token_ids[256:512]
    assert captured[0].source_range == TokenRange(256, 512)


def test_all_complete_chunks_already_stored_skips_worker_calls_entirely() -> None:
    """Every complete chunk already verified present: nothing to gather/store."""

    metadata, blocks = _metadata(
        candidate_specs=((0, 4096, 7), (256, 8192, 9)),
        transfer_eligible=True,
        store_eligible=True,
    )
    runtime, _storage, _data_plane, calls, mutations = _runtime()

    before = runtime.before_forward(metadata, blocks)
    completion = runtime.mark_full_prefill_complete(
        before, recomputed_token_count=600
    )
    calls.clear()
    mutations.clear()

    stored = runtime.after_forward(completion, blocks)

    assert stored.state is TransferAttemptState.SUCCEEDED
    assert stored.eligible_store_tokens == 0
    assert stored.stored_tokens == 0
    assert stored.stored_chunks == 0
    assert stored.sidecar_records_available == 0
    assert stored.sidecar_records_inserted == 0
    # No gather, store, or publish call was ever made.
    assert calls == []
    assert mutations == []


def test_skip_stops_at_first_gap_even_when_a_later_chunk_is_also_verified() -> (
    None
):
    """A verified candidate does not prove earlier, unverified chunks."""

    metadata, blocks = _metadata(
        candidate_specs=((0, 4096, 7), (512, 8192, 9)),
        transfer_eligible=True,
        store_eligible=True,
        prompt_length=856,
    )
    runtime, storage, _data_plane, _calls, _mutations = _runtime()
    captured: list[WorkerStorePlan] = []

    def capture(plan: WorkerStorePlan) -> None:
        captured.append(plan)

    storage.preflight_store = capture  # type: ignore[method-assign]
    before = runtime.before_forward(metadata, blocks)
    completion = runtime.mark_full_prefill_complete(
        before, recomputed_token_count=856
    )

    stored = runtime.after_forward(completion, blocks)

    # complete_store_token_count = 768 (3 chunks). Chunk 0 is verified, chunk
    # 2 (512-768) is verified, but chunk 1 (256-512) is not: fail-closed
    # storing stops skipping at chunk 1 and stores everything from there on.
    assert stored.state is TransferAttemptState.SUCCEEDED
    assert stored.eligible_store_tokens == 512
    assert stored.stored_chunks == 2
    assert [chunk.token_range for chunk in captured[0].chunks] == [
        TokenRange(256, 512),
        TokenRange(512, 768),
    ]


def test_misaligned_verified_candidate_is_never_used_to_skip_a_store_chunk() -> (
    None
):
    """A candidate not aligned to a store-chunk boundary proves nothing here."""

    metadata, blocks = _metadata(
        candidate_specs=((128, 4096, 7),),
        transfer_eligible=True,
        store_eligible=True,
    )
    runtime, storage, _data_plane, _calls, _mutations = _runtime()
    captured: list[WorkerStorePlan] = []

    def capture(plan: WorkerStorePlan) -> None:
        captured.append(plan)

    storage.preflight_store = capture  # type: ignore[method-assign]
    before = runtime.before_forward(metadata, blocks)
    completion = runtime.mark_full_prefill_complete(
        before, recomputed_token_count=600
    )

    stored = runtime.after_forward(completion, blocks)

    assert stored.state is TransferAttemptState.SUCCEEDED
    assert stored.eligible_store_tokens == 512
    assert stored.stored_chunks == 2
    assert [chunk.token_range for chunk in captured[0].chunks] == [
        TokenRange(0, 256),
        TokenRange(256, 512),
    ]


def test_partial_prompt_tail_is_never_gathered_stored_or_published() -> None:
    metadata, blocks = _metadata(
        candidate_specs=(),
        transfer_eligible=False,
        store_eligible=True,
        prompt_length=511,
    )
    runtime, storage, data_plane, _calls, _mutations = _runtime()
    captured: list[WorkerStorePlan] = []

    def capture(plan: WorkerStorePlan) -> None:
        captured.append(plan)

    storage.preflight_store = capture  # type: ignore[method-assign]
    data_plane.preflight_gather = capture  # type: ignore[method-assign]
    before = runtime.before_forward(metadata, blocks)
    completion = FullPrefillCompletion(before, 511)
    result = runtime.after_forward(completion, blocks)

    assert result.state is TransferAttemptState.SUCCEEDED
    assert result.stored_tokens == 256
    assert len(captured[0].chunks) == 1
    assert captured[0].chunks[0].token_range == TokenRange(0, 256)
    assert captured[0].token_ids == metadata.prompt_token_ids[:256]
