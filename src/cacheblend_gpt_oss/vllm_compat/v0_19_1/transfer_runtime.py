# SPDX-License-Identifier: Apache-2.0
"""Dependency-free orchestration for the 100%-recompute transfer milestone.

This module is the narrow scheduler/worker seam for the pinned vLLM 0.19.1
integration.  It imports no vLLM, LMCache, Torch, or CUDA modules.  Concrete
worker adapters implement the protocols below using the already-audited
storage and data-plane components.

The lifecycle assumptions are tied to exact public sources:

* vLLM schedules prompt tokens and carries grouped ``KVCacheBlocks`` from the
  scheduler to model execution:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/core/sched/scheduler.py#L705-L805
* ``KVCacheBlocks.get_block_ids`` preserves one logical table per hybrid
  cache group:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/core/kv_cache_manager.py#L22-L80
* LMCache 0.4.3 Blend V2 retrieves verified matches at their current-query
  positions and stores only complete chunks:
  https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/v1/multiprocess/blend_server_v2.py#L597-L687
  https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/v1/multiprocess/blend_server_v2.py#L444-L513

Loaded KV is instrumentation data only.  Every prompt token remains scheduled
for ordinary prefill, and every outcome reports zero saved-prefill tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import NoReturn, Protocol

from cacheblend_gpt_oss.connector.control_plane import (
    FULL_RECOMPUTE_EXTERNAL_TOKENS,
    RequestHandoffMetadata,
    WorkerValidationReceipt,
)
from cacheblend_gpt_oss.gpt_oss.layout import (
    GPT_OSS_MAX_CONTEXT_TOKENS,
    GptOssHybridCacheLayout,
    HybridLayoutError,
    TokenScatterPlan,
    TokenTransfer,
    plan_token_scatter,
)
from cacheblend_gpt_oss.planner.fingerprint import SHA256_FINGERPRINTER
from cacheblend_gpt_oss.planner.models import (
    CacheNamespace,
    CacheRecord,
    TokenRange,
    normalize_token_ids,
)
from cacheblend_gpt_oss.storage.lmcache_types import (
    LMCACHE_CACHE_KEY_PREFIX,
    LMCACHE_CHUNK_SIZE,
    LmcacheRetrieveReceipt,
    LmcacheStoreReceipt,
    VerifiedLmcacheCandidate,
    query_digest,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.adapters import AdaptedKvCacheBlocks


class TransferRuntimeErrorCode(str, Enum):
    """Bounded fatal input/lifecycle errors safe for aggregate reporting."""

    INVALID_METADATA = "invalid_metadata"
    NONZERO_EXTERNAL_TOKENS = "nonzero_external_tokens"
    NOT_INITIAL_PREFILL = "not_initial_prefill"
    INCOMPLETE_FULL_PROMPT_STEP = "incomplete_full_prompt_step"
    NAMESPACE_MISMATCH = "namespace_mismatch"
    CANDIDATE_MISMATCH = "candidate_mismatch"
    CANDIDATE_CHUNK_MISMATCH = "candidate_chunk_mismatch"
    CANDIDATE_OVERLAP = "candidate_overlap"
    INVALID_OUTCOME = "invalid_outcome"
    INVALID_FORWARD_COMPLETION = "invalid_forward_completion"


class TransferRuntimeError(RuntimeError):
    """Fail-closed error whose message never contains request data."""

    def __init__(self, code: TransferRuntimeErrorCode) -> None:
        self.code = code
        super().__init__(f"transfer runtime failure: {code.value}")


def _fail(code: TransferRuntimeErrorCode) -> NoReturn:
    raise TransferRuntimeError(code)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True, slots=True)
class SchedulerTransferMetadata:
    """Exact immutable scheduler-to-worker metadata for one prefill step."""

    cache_namespace: CacheNamespace
    prompt_token_ids: tuple[int, ...] = field(repr=False)
    verified_candidates: tuple[VerifiedLmcacheCandidate, ...] = field(repr=False)
    handoff: RequestHandoffMetadata = field(repr=False)
    num_computed_tokens_before_step: int
    scheduled_token_count: int
    transfer_eligible: bool
    store_eligible: bool

    def __post_init__(self) -> None:
        if not isinstance(self.cache_namespace, CacheNamespace):
            _fail(TransferRuntimeErrorCode.INVALID_METADATA)
        try:
            prompt = normalize_token_ids(self.prompt_token_ids)
            candidates = tuple(self.verified_candidates)
        except (TypeError, ValueError):
            _fail(TransferRuntimeErrorCode.INVALID_METADATA)
        if not prompt or len(prompt) > GPT_OSS_MAX_CONTEXT_TOKENS:
            _fail(TransferRuntimeErrorCode.INVALID_METADATA)
        if any(
            not isinstance(candidate, VerifiedLmcacheCandidate)
            for candidate in candidates
        ):
            _fail(TransferRuntimeErrorCode.INVALID_METADATA)
        if not isinstance(self.handoff, RequestHandoffMetadata):
            _fail(TransferRuntimeErrorCode.INVALID_METADATA)
        if not isinstance(self.transfer_eligible, bool) or not isinstance(
            self.store_eligible, bool
        ):
            _fail(TransferRuntimeErrorCode.INVALID_METADATA)

        if (
            self.handoff.plan.external_scheduler_tokens
            != FULL_RECOMPUTE_EXTERNAL_TOKENS
            or self.handoff.allocation.external_scheduler_tokens
            != FULL_RECOMPUTE_EXTERNAL_TOKENS
        ):
            _fail(TransferRuntimeErrorCode.NONZERO_EXTERNAL_TOKENS)
        if (
            not _is_int(self.num_computed_tokens_before_step)
            or self.num_computed_tokens_before_step != 0
        ):
            _fail(TransferRuntimeErrorCode.NOT_INITIAL_PREFILL)
        if (
            not _is_int(self.scheduled_token_count)
            or self.scheduled_token_count != len(prompt)
            or self.handoff.plan.prompt_tokens != len(prompt)
        ):
            _fail(TransferRuntimeErrorCode.INCOMPLETE_FULL_PROMPT_STEP)
        if self.transfer_eligible and not candidates:
            _fail(TransferRuntimeErrorCode.INVALID_METADATA)
        if self.store_eligible and len(prompt) < LMCACHE_CHUNK_SIZE:
            _fail(TransferRuntimeErrorCode.INVALID_METADATA)

        expected_digest = query_digest(prompt)
        expected_matches = self.handoff.plan.match_plan.matches
        if tuple(candidate.match for candidate in candidates) != expected_matches:
            _fail(TransferRuntimeErrorCode.CANDIDATE_MISMATCH)

        previous_end = 0
        for index, verified in enumerate(candidates):
            candidate = verified.candidate
            match = verified.match
            target = candidate.target_range
            record = match.record
            if len(target) != LMCACHE_CHUNK_SIZE:
                _fail(TransferRuntimeErrorCode.CANDIDATE_CHUNK_MISMATCH)
            if target.end > len(prompt):
                _fail(TransferRuntimeErrorCode.CANDIDATE_MISMATCH)
            if index > 0 and target.start < previous_end:
                _fail(TransferRuntimeErrorCode.CANDIDATE_OVERLAP)
            if candidate.query_digest != expected_digest:
                _fail(TransferRuntimeErrorCode.CANDIDATE_MISMATCH)
            if (
                candidate.target_range != match.target_segment.token_range
                or candidate.cache_key != record.cache_key
                or record.namespace != self.cache_namespace
                or match.target_segment.token_ids
                != prompt[target.start : target.end]
                or record.token_ids != match.target_segment.token_ids
                or record.fingerprint
                != SHA256_FINGERPRINTER.fingerprint(
                    self.cache_namespace, record.token_ids
                )
            ):
                code = (
                    TransferRuntimeErrorCode.NAMESPACE_MISMATCH
                    if record.namespace != self.cache_namespace
                    else TransferRuntimeErrorCode.CANDIDATE_MISMATCH
                )
                _fail(code)
            previous_end = target.end

        object.__setattr__(self, "prompt_token_ids", prompt)
        object.__setattr__(self, "verified_candidates", candidates)

    @property
    def request_id(self) -> str:
        """Return the opaque request correlation ID without exposing it in errors."""

        return self.handoff.plan.request_id

    @property
    def prompt_token_count(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def complete_store_token_count(self) -> int:
        """Return only the complete 256-token prefix of the prompt."""

        return (
            self.prompt_token_count // LMCACHE_CHUNK_SIZE
        ) * LMCACHE_CHUNK_SIZE


@dataclass(frozen=True, slots=True)
class CandidateScatterWork:
    """One exact verified candidate and its all-group scatter plan."""

    candidate_index: int
    verified_candidate: VerifiedLmcacheCandidate = field(repr=False)
    scatter_plan: TokenScatterPlan


@dataclass(frozen=True, slots=True)
class WorkerLoadPlan:
    """Fully preplanned retrieval/scatter work; constructing it does no I/O."""

    metadata: SchedulerTransferMetadata
    adapted_blocks: AdaptedKvCacheBlocks
    candidates: tuple[CandidateScatterWork, ...]

    @property
    def expected_tokens(self) -> int:
        return sum(
            len(candidate.verified_candidate.candidate.target_range)
            for candidate in self.candidates
        )

    @property
    def expected_chunks(self) -> int:
        return len(self.candidates)


@dataclass(frozen=True, slots=True)
class StoreChunkWork:
    """One complete prompt chunk and its all-group gather plan."""

    chunk_index: int
    token_range: TokenRange
    token_ids: tuple[int, ...] = field(repr=False)
    gather_plan: TokenScatterPlan


@dataclass(frozen=True, slots=True)
class WorkerStorePlan:
    """Fully preplanned gather/store work for complete prompt chunks."""

    metadata: SchedulerTransferMetadata
    adapted_blocks: AdaptedKvCacheBlocks
    chunks: tuple[StoreChunkWork, ...]

    @property
    def token_ids(self) -> tuple[int, ...]:
        return self.metadata.prompt_token_ids[: self.expected_tokens]

    @property
    def source_range(self) -> TokenRange:
        return TokenRange(0, self.expected_tokens)

    @property
    def expected_tokens(self) -> int:
        return len(self.chunks) * LMCACHE_CHUNK_SIZE

    @property
    def expected_chunks(self) -> int:
        return len(self.chunks)


class WorkerStorage(Protocol):
    """Injected LMCache/sidecar worker boundary.

    Both ``preflight_*`` methods must be read-only.  Publication must make the
    complete tuple visible atomically or raise without partial visibility.
    """

    def preflight_retrieve(self, plan: WorkerLoadPlan) -> None:
        """Validate all retrieval inputs without mutating staging or storage."""

    def retrieve_verified(self, plan: WorkerLoadPlan) -> LmcacheRetrieveReceipt:
        """Synchronously retrieve every verified candidate into staging."""

    def preflight_store(self, plan: WorkerStorePlan) -> None:
        """Validate all store inputs without gathering or writing storage."""

    def store_precomputed(self, plan: WorkerStorePlan) -> LmcacheStoreReceipt:
        """Synchronously store gathered chunks and return sidecar records."""

    def publish_sidecar_records_atomically(
        self, records: tuple[CacheRecord, ...]
    ) -> int:
        """Publish one validated batch and return the newly inserted count."""


class WorkerDataPlane(Protocol):
    """Injected GPT-OSS paged-cache/staging boundary.

    Preflight methods must inspect every span and tensor without mutation.
    """

    def preflight_scatter(self, plan: WorkerLoadPlan) -> None:
        """Validate all candidate scatter work without copying KV."""

    def scatter_retrieved(self, plan: WorkerLoadPlan) -> None:
        """Scatter retrieved KV into every hybrid group synchronously."""

    def preflight_gather(self, plan: WorkerStorePlan) -> None:
        """Validate all chunk gather work without copying KV."""

    def gather_recomputed(self, plan: WorkerStorePlan) -> None:
        """Gather post-forward KV for every complete prompt chunk."""


class TransferAttemptState(str, Enum):
    """Bounded terminal state for load or store orchestration."""

    NOT_ELIGIBLE = "not_eligible"
    SUCCEEDED = "succeeded"
    FULL_PREFILL_FALLBACK = "full_prefill_fallback"


class TransferFallbackCode(str, Enum):
    """Bounded worker failure stages; exception values are never retained."""

    LOAD_PLAN_REJECTED = "load_plan_rejected"
    LOAD_PREFLIGHT_FAILED = "load_preflight_failed"
    RETRIEVE_FAILED = "retrieve_failed"
    RETRIEVE_RECEIPT_INVALID = "retrieve_receipt_invalid"
    SCATTER_FAILED = "scatter_failed"
    STORE_PLAN_REJECTED = "store_plan_rejected"
    STORE_PREFLIGHT_FAILED = "store_preflight_failed"
    GATHER_FAILED = "gather_failed"
    STORE_FAILED = "store_failed"
    STORE_RECEIPT_INVALID = "store_receipt_invalid"
    SIDECAR_PUBLISH_FAILED = "sidecar_publish_failed"


@dataclass(frozen=True, slots=True)
class PreForwardOutcome:
    """Load result plus the mandatory full-prefill scheduling decision."""

    metadata: SchedulerTransferMetadata
    state: TransferAttemptState
    failure_code: TransferFallbackCode | None
    loaded_candidate_indexes: tuple[int, ...]
    rejected_candidate_indexes: tuple[int, ...]
    loaded_kv_tokens: int
    tokens_to_recompute: int
    external_scheduler_tokens: int = FULL_RECOMPUTE_EXTERNAL_TOKENS
    prefill_tokens_avoided: int = 0
    position_correction_latency_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, SchedulerTransferMetadata) or not isinstance(
            self.state, TransferAttemptState
        ):
            _fail(TransferRuntimeErrorCode.INVALID_OUTCOME)
        try:
            loaded = tuple(self.loaded_candidate_indexes)
            rejected = tuple(self.rejected_candidate_indexes)
        except TypeError:
            _fail(TransferRuntimeErrorCode.INVALID_OUTCOME)
        candidate_count = len(self.metadata.verified_candidates)
        expected_indexes = tuple(range(candidate_count))
        if (
            any(not _is_int(index) for index in (*loaded, *rejected))
            or any(
                not _is_int(value)
                for value in (
                    self.loaded_kv_tokens,
                    self.tokens_to_recompute,
                    self.external_scheduler_tokens,
                    self.prefill_tokens_avoided,
                )
            )
            or len(loaded) != len(set(loaded))
            or len(rejected) != len(set(rejected))
            or set(loaded).intersection(rejected)
            or tuple(sorted((*loaded, *rejected))) != expected_indexes
            or self.tokens_to_recompute != self.metadata.prompt_token_count
            or self.external_scheduler_tokens != FULL_RECOMPUTE_EXTERNAL_TOKENS
            or self.prefill_tokens_avoided != 0
        ):
            _fail(TransferRuntimeErrorCode.INVALID_OUTCOME)
        if (
            isinstance(self.position_correction_latency_seconds, bool)
            or not isinstance(self.position_correction_latency_seconds, int | float)
            or not isfinite(self.position_correction_latency_seconds)
            or self.position_correction_latency_seconds < 0
        ):
            _fail(TransferRuntimeErrorCode.INVALID_OUTCOME)
        expected_loaded_tokens = sum(
            len(self.metadata.verified_candidates[index].candidate.target_range)
            for index in loaded
        )
        if self.state is TransferAttemptState.SUCCEEDED:
            if (
                not self.metadata.transfer_eligible
                or self.failure_code is not None
                or loaded != expected_indexes
                or rejected
                or self.loaded_kv_tokens != expected_loaded_tokens
            ):
                _fail(TransferRuntimeErrorCode.INVALID_OUTCOME)
        elif (
            loaded
            or rejected != expected_indexes
            or self.loaded_kv_tokens != 0
            or (
                self.state is TransferAttemptState.NOT_ELIGIBLE
                and (
                    self.metadata.transfer_eligible
                    or self.failure_code is not None
                )
            )
            or (
                self.state is TransferAttemptState.FULL_PREFILL_FALLBACK
                and not isinstance(self.failure_code, TransferFallbackCode)
            )
        ):
            _fail(TransferRuntimeErrorCode.INVALID_OUTCOME)
        object.__setattr__(self, "loaded_candidate_indexes", loaded)
        object.__setattr__(self, "rejected_candidate_indexes", rejected)

    def to_worker_validation_receipt(self) -> WorkerValidationReceipt:
        """Translate exact load accounting back to the control-plane receipt."""

        return WorkerValidationReceipt(
            request_id=self.metadata.request_id,
            allocation=self.metadata.handoff.allocation,
            loaded_match_indexes=self.loaded_candidate_indexes,
            rejected_match_indexes=self.rejected_candidate_indexes,
        )


@dataclass(frozen=True, slots=True)
class FullPrefillCompletion:
    """Proof supplied after ordinary forward recomputed the entire prompt."""

    pre_forward: PreForwardOutcome
    recomputed_token_count: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.pre_forward, PreForwardOutcome)
            or not _is_int(self.recomputed_token_count)
            or self.recomputed_token_count
            != self.pre_forward.metadata.prompt_token_count
            or self.recomputed_token_count != self.pre_forward.tokens_to_recompute
        ):
            _fail(TransferRuntimeErrorCode.INVALID_FORWARD_COMPLETION)


@dataclass(frozen=True, slots=True)
class PostForwardOutcome:
    """Store/publication result with zero saved-prefill credit by construction."""

    completion: FullPrefillCompletion
    state: TransferAttemptState
    failure_code: TransferFallbackCode | None
    eligible_store_tokens: int
    stored_tokens: int
    stored_chunks: int
    sidecar_records_available: int
    sidecar_records_inserted: int
    prefill_tokens_avoided: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.completion, FullPrefillCompletion) or not isinstance(
            self.state, TransferAttemptState
        ):
            _fail(TransferRuntimeErrorCode.INVALID_OUTCOME)
        values = (
            self.eligible_store_tokens,
            self.stored_tokens,
            self.stored_chunks,
            self.sidecar_records_available,
            self.sidecar_records_inserted,
            self.prefill_tokens_avoided,
        )
        if any(not _is_int(value) or value < 0 for value in values):
            _fail(TransferRuntimeErrorCode.INVALID_OUTCOME)
        metadata = self.completion.pre_forward.metadata
        expected_eligible = (
            metadata.complete_store_token_count if metadata.store_eligible else 0
        )
        expected_chunks = expected_eligible // LMCACHE_CHUNK_SIZE
        if self.eligible_store_tokens != expected_eligible:
            _fail(TransferRuntimeErrorCode.INVALID_OUTCOME)
        if self.state is TransferAttemptState.SUCCEEDED:
            if (
                not self.completion.pre_forward.metadata.store_eligible
                or self.failure_code is not None
                or self.stored_tokens != expected_eligible
                or self.stored_chunks != expected_chunks
                or self.sidecar_records_available != expected_chunks
                or self.sidecar_records_inserted > expected_chunks
                or self.prefill_tokens_avoided != 0
            ):
                _fail(TransferRuntimeErrorCode.INVALID_OUTCOME)
        elif (
            any(
                value != 0
                for value in (
                    self.stored_tokens,
                    self.stored_chunks,
                    self.sidecar_records_available,
                    self.sidecar_records_inserted,
                    self.prefill_tokens_avoided,
                )
            )
            or (
                self.state is TransferAttemptState.NOT_ELIGIBLE
                and (
                    self.completion.pre_forward.metadata.store_eligible
                    or self.failure_code is not None
                )
            )
            or (
                self.state is TransferAttemptState.FULL_PREFILL_FALLBACK
                and not isinstance(self.failure_code, TransferFallbackCode)
            )
        ):
            _fail(TransferRuntimeErrorCode.INVALID_OUTCOME)


class TransferRuntime:
    """Orchestrate one initial full-prefill transfer without runtime imports."""

    def __init__(
        self,
        layout: GptOssHybridCacheLayout,
        storage: WorkerStorage,
        data_plane: WorkerDataPlane,
    ) -> None:
        if not isinstance(layout, GptOssHybridCacheLayout):
            _fail(TransferRuntimeErrorCode.INVALID_METADATA)
        self._layout = layout
        self._storage = storage
        self._data_plane = data_plane

    def before_forward(
        self,
        metadata: SchedulerTransferMetadata,
        adapted_blocks: AdaptedKvCacheBlocks,
    ) -> PreForwardOutcome:
        """Retrieve/scatter instrumentation while scheduling full recomputation."""

        if not isinstance(metadata, SchedulerTransferMetadata):
            _fail(TransferRuntimeErrorCode.INVALID_METADATA)
        all_indexes = tuple(range(len(metadata.verified_candidates)))
        try:
            plan = self._build_load_plan(metadata, adapted_blocks)
        except (TransferRuntimeError, HybridLayoutError):
            return PreForwardOutcome(
                metadata=metadata,
                state=TransferAttemptState.FULL_PREFILL_FALLBACK,
                failure_code=TransferFallbackCode.LOAD_PLAN_REJECTED,
                loaded_candidate_indexes=(),
                rejected_candidate_indexes=all_indexes,
                loaded_kv_tokens=0,
                tokens_to_recompute=metadata.prompt_token_count,
            )

        if not metadata.transfer_eligible:
            return PreForwardOutcome(
                metadata=metadata,
                state=TransferAttemptState.NOT_ELIGIBLE,
                failure_code=None,
                loaded_candidate_indexes=(),
                rejected_candidate_indexes=all_indexes,
                loaded_kv_tokens=0,
                tokens_to_recompute=metadata.prompt_token_count,
            )

        try:
            self._storage.preflight_retrieve(plan)
            self._data_plane.preflight_scatter(plan)
        except Exception:
            return self._load_fallback(
                metadata,
                all_indexes,
                TransferFallbackCode.LOAD_PREFLIGHT_FAILED,
            )
        try:
            receipt = self._storage.retrieve_verified(plan)
        except Exception:
            return self._load_fallback(
                metadata, all_indexes, TransferFallbackCode.RETRIEVE_FAILED
            )
        if not self._valid_retrieve_receipt(receipt, plan):
            return self._load_fallback(
                metadata,
                all_indexes,
                TransferFallbackCode.RETRIEVE_RECEIPT_INVALID,
            )
        try:
            self._data_plane.scatter_retrieved(plan)
        except Exception:
            return self._load_fallback(
                metadata, all_indexes, TransferFallbackCode.SCATTER_FAILED
            )
        return PreForwardOutcome(
            metadata=metadata,
            state=TransferAttemptState.SUCCEEDED,
            failure_code=None,
            loaded_candidate_indexes=all_indexes,
            rejected_candidate_indexes=(),
            loaded_kv_tokens=plan.expected_tokens,
            tokens_to_recompute=metadata.prompt_token_count,
            position_correction_latency_seconds=self._read_position_correction_latency(),
        )

    def mark_full_prefill_complete(
        self,
        pre_forward: PreForwardOutcome,
        *,
        recomputed_token_count: int,
    ) -> FullPrefillCompletion:
        """Create the only value accepted by post-forward storage."""

        return FullPrefillCompletion(pre_forward, recomputed_token_count)

    def after_forward(
        self,
        completion: FullPrefillCompletion,
        adapted_blocks: AdaptedKvCacheBlocks,
    ) -> PostForwardOutcome:
        """Gather/store complete prompt chunks, then publish one atomic batch."""

        if not isinstance(completion, FullPrefillCompletion):
            _fail(TransferRuntimeErrorCode.INVALID_FORWARD_COMPLETION)
        metadata = completion.pre_forward.metadata
        eligible_tokens = (
            metadata.complete_store_token_count if metadata.store_eligible else 0
        )
        if not metadata.store_eligible:
            return self._store_outcome(
                completion,
                TransferAttemptState.NOT_ELIGIBLE,
                None,
                eligible_tokens,
            )
        try:
            plan = self._build_store_plan(metadata, adapted_blocks)
        except (TransferRuntimeError, HybridLayoutError):
            return self._store_outcome(
                completion,
                TransferAttemptState.FULL_PREFILL_FALLBACK,
                TransferFallbackCode.STORE_PLAN_REJECTED,
                eligible_tokens,
            )
        try:
            self._data_plane.preflight_gather(plan)
            self._storage.preflight_store(plan)
        except Exception:
            return self._store_outcome(
                completion,
                TransferAttemptState.FULL_PREFILL_FALLBACK,
                TransferFallbackCode.STORE_PREFLIGHT_FAILED,
                eligible_tokens,
            )
        try:
            self._data_plane.gather_recomputed(plan)
        except Exception:
            return self._store_outcome(
                completion,
                TransferAttemptState.FULL_PREFILL_FALLBACK,
                TransferFallbackCode.GATHER_FAILED,
                eligible_tokens,
            )
        try:
            receipt = self._storage.store_precomputed(plan)
        except Exception:
            return self._store_outcome(
                completion,
                TransferAttemptState.FULL_PREFILL_FALLBACK,
                TransferFallbackCode.STORE_FAILED,
                eligible_tokens,
            )
        if not self._valid_store_receipt(receipt, plan):
            return self._store_outcome(
                completion,
                TransferAttemptState.FULL_PREFILL_FALLBACK,
                TransferFallbackCode.STORE_RECEIPT_INVALID,
                eligible_tokens,
            )
        try:
            inserted = self._storage.publish_sidecar_records_atomically(
                receipt.sidecar_records
            )
        except Exception:
            return self._store_outcome(
                completion,
                TransferAttemptState.FULL_PREFILL_FALLBACK,
                TransferFallbackCode.SIDECAR_PUBLISH_FAILED,
                eligible_tokens,
            )
        if (
            not _is_int(inserted)
            or inserted < 0
            or inserted > len(receipt.sidecar_records)
        ):
            return self._store_outcome(
                completion,
                TransferAttemptState.FULL_PREFILL_FALLBACK,
                TransferFallbackCode.SIDECAR_PUBLISH_FAILED,
                eligible_tokens,
            )
        return PostForwardOutcome(
            completion=completion,
            state=TransferAttemptState.SUCCEEDED,
            failure_code=None,
            eligible_store_tokens=eligible_tokens,
            stored_tokens=receipt.stored_tokens,
            stored_chunks=receipt.stored_chunks,
            sidecar_records_available=len(receipt.sidecar_records),
            sidecar_records_inserted=inserted,
        )

    def _build_load_plan(
        self,
        metadata: SchedulerTransferMetadata,
        adapted_blocks: AdaptedKvCacheBlocks,
    ) -> WorkerLoadPlan:
        self._validate_allocation(metadata, adapted_blocks)
        candidates = tuple(
            CandidateScatterWork(
                candidate_index=index,
                verified_candidate=verified,
                scatter_plan=plan_token_scatter(
                    self._layout,
                    adapted_blocks.group_block_tables,
                    TokenTransfer(
                        source_range=verified.match.record.source_range,
                        target_range=verified.candidate.target_range,
                    ),
                ),
            )
            for index, verified in enumerate(metadata.verified_candidates)
        )
        return WorkerLoadPlan(metadata, adapted_blocks, candidates)

    def _build_store_plan(
        self,
        metadata: SchedulerTransferMetadata,
        adapted_blocks: AdaptedKvCacheBlocks,
    ) -> WorkerStorePlan:
        self._validate_allocation(metadata, adapted_blocks)
        chunks: list[StoreChunkWork] = []
        for start in range(0, metadata.complete_store_token_count, LMCACHE_CHUNK_SIZE):
            token_range = TokenRange(start, start + LMCACHE_CHUNK_SIZE)
            chunks.append(
                StoreChunkWork(
                    chunk_index=start // LMCACHE_CHUNK_SIZE,
                    token_range=token_range,
                    token_ids=metadata.prompt_token_ids[
                        token_range.start : token_range.end
                    ],
                    gather_plan=plan_token_scatter(
                        self._layout,
                        adapted_blocks.group_block_tables,
                        TokenTransfer(token_range, token_range),
                    ),
                )
            )
        if not chunks:
            _fail(TransferRuntimeErrorCode.INVALID_METADATA)
        return WorkerStorePlan(metadata, adapted_blocks, tuple(chunks))

    @staticmethod
    def _validate_allocation(
        metadata: SchedulerTransferMetadata,
        adapted_blocks: AdaptedKvCacheBlocks,
    ) -> None:
        if (
            not isinstance(adapted_blocks, AdaptedKvCacheBlocks)
            or adapted_blocks.grouped_allocation
            != metadata.handoff.allocation.grouped_blocks
        ):
            _fail(TransferRuntimeErrorCode.INVALID_METADATA)

    @staticmethod
    def _valid_retrieve_receipt(
        receipt: object, plan: WorkerLoadPlan
    ) -> bool:
        return (
            isinstance(receipt, LmcacheRetrieveReceipt)
            and _is_int(receipt.retrieved_tokens)
            and _is_int(receipt.retrieved_chunks)
            and receipt.retrieved_tokens == plan.expected_tokens
            and receipt.retrieved_chunks == plan.expected_chunks
        )

    @staticmethod
    def _valid_store_receipt(receipt: object, plan: WorkerStorePlan) -> bool:
        if not isinstance(receipt, LmcacheStoreReceipt):
            return False
        if (
            receipt.stored_tokens != plan.expected_tokens
            or receipt.stored_chunks != plan.expected_chunks
            or not receipt.candidate_lookup_required
            or len(receipt.sidecar_records) != plan.expected_chunks
        ):
            return False
        for chunk, record in zip(
            plan.chunks, receipt.sidecar_records, strict=True
        ):
            if not isinstance(record, CacheRecord) or not isinstance(
                record.cache_key, str
            ):
                return False
            suffix = record.cache_key.removeprefix(LMCACHE_CACHE_KEY_PREFIX)
            if (
                record.namespace != plan.metadata.cache_namespace
                or record.token_ids != chunk.token_ids
                or record.source_range != chunk.token_range
                or record.fingerprint
                != SHA256_FINGERPRINTER.fingerprint(
                    plan.metadata.cache_namespace, chunk.token_ids
                )
                or not record.cache_key.startswith(LMCACHE_CACHE_KEY_PREFIX)
                or len(suffix) != 64
                or any(character not in "0123456789abcdef" for character in suffix)
            ):
                return False
        return True

    @staticmethod
    def _load_fallback(
        metadata: SchedulerTransferMetadata,
        all_indexes: tuple[int, ...],
        failure_code: TransferFallbackCode,
    ) -> PreForwardOutcome:
        return PreForwardOutcome(
            metadata=metadata,
            state=TransferAttemptState.FULL_PREFILL_FALLBACK,
            failure_code=failure_code,
            loaded_candidate_indexes=(),
            rejected_candidate_indexes=all_indexes,
            loaded_kv_tokens=0,
            tokens_to_recompute=metadata.prompt_token_count,
        )

    def _read_position_correction_latency(self) -> float:
        """Read optional worker timing without making telemetry a validity gate."""

        value = getattr(self._data_plane, "position_correction_latency_seconds", 0.0)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not isfinite(value)
            or value < 0
        ):
            return 0.0
        return float(value)

    @staticmethod
    def _store_outcome(
        completion: FullPrefillCompletion,
        state: TransferAttemptState,
        failure_code: TransferFallbackCode | None,
        eligible_tokens: int,
    ) -> PostForwardOutcome:
        return PostForwardOutcome(
            completion=completion,
            state=state,
            failure_code=failure_code,
            eligible_store_tokens=eligible_tokens,
            stored_tokens=0,
            stored_chunks=0,
            sidecar_records_available=0,
            sidecar_records_inserted=0,
        )


__all__ = [
    "CandidateScatterWork",
    "FullPrefillCompletion",
    "PostForwardOutcome",
    "PreForwardOutcome",
    "SchedulerTransferMetadata",
    "StoreChunkWork",
    "TransferAttemptState",
    "TransferFallbackCode",
    "TransferRuntime",
    "TransferRuntimeError",
    "TransferRuntimeErrorCode",
    "WorkerDataPlane",
    "WorkerLoadPlan",
    "WorkerStorage",
    "WorkerStorePlan",
]
