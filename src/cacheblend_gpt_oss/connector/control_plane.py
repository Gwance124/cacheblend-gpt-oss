"""Dependency-free request control plane for the full-recompute milestone.

The objects in this module deliberately know nothing about vLLM, LMCache,
Torch, or CUDA.  A version-scoped connector can translate vLLM requests and
``KVCacheBlocks`` into these immutable values while keeping scheduler and
worker lifecycle rules testable on a CPU-only workstation.

At this milestone a verified non-prefix hit is transfer instrumentation only.
The scheduler must always receive zero externally computed tokens, and every
prompt token must still pass through ordinary prefill.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, replace
from enum import Enum
from typing import NoReturn

_duplicate_block_ids_allowed: ContextVar[bool] = ContextVar(
    "_duplicate_block_ids_allowed", default=False
)

from cacheblend_gpt_oss.metrics import RequestMetricCounters
from cacheblend_gpt_oss.planner import MatchPlan, TokenRange, TokenSegment

METADATA_SCHEMA_VERSION = 1
FULL_RECOMPUTE_EXTERNAL_TOKENS = 0


class RequestPhase(str, Enum):
    """Bounded phases shared by scheduler- and worker-side state machines."""

    LOOKED_UP = "looked_up"
    ALLOCATED = "allocated"
    HANDED_OFF = "handed_off"
    WORKER_VALIDATED = "worker_validated"
    FINISHED = "finished"


class ControlPlaneErrorCode(str, Enum):
    """Stable failure codes suitable for bounded logging and metrics."""

    UNKNOWN_REQUEST = "unknown_request"
    DUPLICATE_REQUEST_CONFLICT = "duplicate_request_conflict"
    LIFECYCLE_MISUSE = "lifecycle_misuse"
    INVALID_REQUEST_PLAN = "invalid_request_plan"
    INVALID_GROUP_LAYOUT = "invalid_group_layout"
    GROUP_COUNT_MISMATCH = "group_count_mismatch"
    GROUP_LAYOUT_MISMATCH = "group_layout_mismatch"
    INVALID_BLOCK_ID = "invalid_block_id"
    NONZERO_EXTERNAL_TOKENS = "nonzero_external_tokens"
    METADATA_SCHEMA_MISMATCH = "metadata_schema_mismatch"
    STALE_ALLOCATION = "stale_allocation"
    INVALID_WORKER_RESULT = "invalid_worker_result"


class ControlPlaneError(RuntimeError):
    """Fail-closed control-plane error with a bounded machine-readable code."""

    def __init__(self, code: ControlPlaneErrorCode) -> None:
        self.code = code
        super().__init__(f"CacheBlend control-plane failure: {code.value}")


def _fail(code: ControlPlaneErrorCode) -> NoReturn:
    raise ControlPlaneError(code)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _covered_token_count(segments: Sequence[TokenSegment]) -> int:
    ranges = sorted(segment.token_range for segment in segments)
    if not ranges:
        return 0

    covered = 0
    current_start = ranges[0].start
    current_end = ranges[0].end
    for token_range in ranges[1:]:
        if token_range.start <= current_end:
            current_end = max(current_end, token_range.end)
            continue
        covered += current_end - current_start
        current_start = token_range.start
        current_end = token_range.end
    return covered + current_end - current_start


def _segment_sort_key(segment: TokenSegment) -> tuple[int, int, tuple[int, ...]]:
    return (segment.token_range.start, segment.token_range.end, segment.token_ids)


@dataclass(frozen=True, slots=True)
class CacheGroupLayout:
    """Ordered layer membership for every hybrid KV-cache group."""

    layer_names_by_group: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        try:
            groups = tuple(tuple(names) for names in self.layer_names_by_group)
        except TypeError:
            _fail(ControlPlaneErrorCode.INVALID_GROUP_LAYOUT)
        if not groups or any(not group for group in groups):
            _fail(ControlPlaneErrorCode.INVALID_GROUP_LAYOUT)

        flattened: list[str] = []
        for group in groups:
            if any(not isinstance(name, str) or not name for name in group):
                _fail(ControlPlaneErrorCode.INVALID_GROUP_LAYOUT)
            flattened.extend(group)
        if len(flattened) != len(set(flattened)):
            _fail(ControlPlaneErrorCode.INVALID_GROUP_LAYOUT)
        object.__setattr__(self, "layer_names_by_group", groups)

    @property
    def group_count(self) -> int:
        return len(self.layer_names_by_group)


@dataclass(frozen=True, slots=True)
class GroupBlockSnapshot:
    """One group's ordered logical block table at allocation time."""

    group_index: int
    layer_names: tuple[str, ...]
    block_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not _is_int(self.group_index) or self.group_index < 0:
            _fail(ControlPlaneErrorCode.INVALID_GROUP_LAYOUT)
        layer_names = tuple(self.layer_names)
        if not layer_names or any(
            not isinstance(name, str) or not name for name in layer_names
        ):
            _fail(ControlPlaneErrorCode.INVALID_GROUP_LAYOUT)

        block_ids = tuple(self.block_ids)
        if any(not _is_int(block_id) or block_id < 0 for block_id in block_ids):
            _fail(ControlPlaneErrorCode.INVALID_BLOCK_ID)
        if len(block_ids) != len(set(block_ids)):
            if not _duplicate_block_ids_allowed.get(False):
                _fail(ControlPlaneErrorCode.INVALID_BLOCK_ID)
        object.__setattr__(self, "layer_names", layer_names)
        object.__setattr__(self, "block_ids", block_ids)


@dataclass(frozen=True, slots=True)
class GroupedBlockAllocation:
    """Immutable, deterministic snapshot of all hybrid-group block tables."""

    group_layout: CacheGroupLayout
    groups: tuple[GroupBlockSnapshot, ...]

    def __post_init__(self) -> None:
        groups = tuple(self.groups)
        if len(groups) != self.group_layout.group_count:
            _fail(ControlPlaneErrorCode.GROUP_COUNT_MISMATCH)
        for group_index, group in enumerate(groups):
            if group.group_index != group_index:
                _fail(ControlPlaneErrorCode.INVALID_GROUP_LAYOUT)
            if (
                group.layer_names
                != self.group_layout.layer_names_by_group[group_index]
            ):
                _fail(ControlPlaneErrorCode.GROUP_LAYOUT_MISMATCH)
        object.__setattr__(self, "groups", groups)

    @classmethod
    def capture(
        cls,
        group_layout: CacheGroupLayout,
        block_ids_by_group: Sequence[Sequence[int]],
        *,
        allow_duplicate_block_ids: bool = False,
    ) -> GroupedBlockAllocation:
        """Copy grouped block tables without sorting away logical block order."""

        try:
            block_groups = tuple(tuple(block_ids) for block_ids in block_ids_by_group)
        except TypeError:
            _fail(ControlPlaneErrorCode.GROUP_COUNT_MISMATCH)
        if len(block_groups) != group_layout.group_count:
            _fail(ControlPlaneErrorCode.GROUP_COUNT_MISMATCH)
        token = _duplicate_block_ids_allowed.set(allow_duplicate_block_ids)
        try:
            return cls(
                group_layout=group_layout,
                groups=tuple(
                    GroupBlockSnapshot(
                        group_index=group_index,
                        layer_names=group_layout.layer_names_by_group[group_index],
                        block_ids=block_ids,
                    )
                    for group_index, block_ids in enumerate(block_groups)
                ),
            )
        finally:
            _duplicate_block_ids_allowed.reset(token)

    @property
    def block_ids_by_group(self) -> tuple[tuple[int, ...], ...]:
        return tuple(group.block_ids for group in self.groups)


@dataclass(frozen=True, slots=True)
class RequestPlan:
    """Verified lookup result for reusable segments in one complete prompt."""

    request_id: str
    prompt_tokens: int
    query_segments: tuple[TokenSegment, ...]
    match_plan: MatchPlan

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            _fail(ControlPlaneErrorCode.INVALID_REQUEST_PLAN)
        if not _is_int(self.prompt_tokens) or self.prompt_tokens < 0:
            _fail(ControlPlaneErrorCode.INVALID_REQUEST_PLAN)

        queries = tuple(sorted(tuple(self.query_segments), key=_segment_sort_key))
        if len(queries) != len(set(queries)):
            _fail(ControlPlaneErrorCode.INVALID_REQUEST_PLAN)
        if any(segment.token_range.end > self.prompt_tokens for segment in queries):
            _fail(ControlPlaneErrorCode.INVALID_REQUEST_PLAN)
        if self.match_plan.requested_tokens != _covered_token_count(queries):
            _fail(ControlPlaneErrorCode.INVALID_REQUEST_PLAN)
        if not 0 <= self.match_plan.matched_tokens <= self.match_plan.requested_tokens:
            _fail(ControlPlaneErrorCode.INVALID_REQUEST_PLAN)

        matched_ranges: list[TokenRange] = []
        previous_sort_key: tuple[int, int, tuple[int, ...]] | None = None
        for verified_match in self.match_plan.matches:
            target = verified_match.target_segment
            if target not in queries:
                _fail(ControlPlaneErrorCode.INVALID_REQUEST_PLAN)
            if target.token_ids != verified_match.record.token_ids:
                _fail(ControlPlaneErrorCode.INVALID_REQUEST_PLAN)
            if (
                verified_match.candidate.query_fingerprint
                != verified_match.record.fingerprint
            ):
                _fail(ControlPlaneErrorCode.INVALID_REQUEST_PLAN)
            if any(
                target.token_range.overlaps(existing) for existing in matched_ranges
            ):
                _fail(ControlPlaneErrorCode.INVALID_REQUEST_PLAN)
            sort_key = _segment_sort_key(target)
            if previous_sort_key is not None and sort_key < previous_sort_key:
                _fail(ControlPlaneErrorCode.INVALID_REQUEST_PLAN)
            previous_sort_key = sort_key
            matched_ranges.append(target.token_range)

        if any(
            rejected.candidate.target_segment not in queries
            for rejected in self.match_plan.rejected_candidates
        ):
            _fail(ControlPlaneErrorCode.INVALID_REQUEST_PLAN)
        object.__setattr__(self, "query_segments", queries)

    @property
    def external_scheduler_tokens(self) -> int:
        """Return the hard-zero scheduler credit for 100% recomputation."""

        return FULL_RECOMPUTE_EXTERNAL_TOKENS


@dataclass(frozen=True, slots=True)
class RequestAllocation:
    """A request plan bound to one generation of physical block tables."""

    request_id: str
    allocation_generation: int
    grouped_blocks: GroupedBlockAllocation
    external_scheduler_tokens: int = FULL_RECOMPUTE_EXTERNAL_TOKENS

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            _fail(ControlPlaneErrorCode.INVALID_REQUEST_PLAN)
        if (
            not _is_int(self.allocation_generation)
            or self.allocation_generation < 0
        ):
            _fail(ControlPlaneErrorCode.STALE_ALLOCATION)
        if (
            not _is_int(self.external_scheduler_tokens)
            or self.external_scheduler_tokens != FULL_RECOMPUTE_EXTERNAL_TOKENS
        ):
            _fail(ControlPlaneErrorCode.NONZERO_EXTERNAL_TOKENS)


@dataclass(frozen=True, slots=True)
class RequestHandoffMetadata:
    """Opaque scheduler-to-worker payload independent of vLLM metadata types."""

    schema_version: int
    plan: RequestPlan
    allocation: RequestAllocation

    def __post_init__(self) -> None:
        if (
            not _is_int(self.schema_version)
            or self.schema_version != METADATA_SCHEMA_VERSION
        ):
            _fail(ControlPlaneErrorCode.METADATA_SCHEMA_MISMATCH)
        if self.plan.request_id != self.allocation.request_id:
            _fail(ControlPlaneErrorCode.INVALID_REQUEST_PLAN)
        if (
            self.plan.external_scheduler_tokens
            != self.allocation.external_scheduler_tokens
        ):
            _fail(ControlPlaneErrorCode.NONZERO_EXTERNAL_TOKENS)


def _normalize_match_indexes(indexes: Iterable[int]) -> tuple[int, ...]:
    normalized = tuple(indexes)
    if any(not _is_int(index) or index < 0 for index in normalized):
        _fail(ControlPlaneErrorCode.INVALID_WORKER_RESULT)
    if len(normalized) != len(set(normalized)):
        _fail(ControlPlaneErrorCode.INVALID_WORKER_RESULT)
    return tuple(sorted(normalized))


@dataclass(frozen=True, slots=True)
class WorkerValidationReceipt:
    """Worker result tied to the exact allocation it inspected or loaded."""

    request_id: str
    allocation: RequestAllocation
    loaded_match_indexes: tuple[int, ...]
    rejected_match_indexes: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.request_id != self.allocation.request_id:
            _fail(ControlPlaneErrorCode.INVALID_WORKER_RESULT)
        loaded = _normalize_match_indexes(self.loaded_match_indexes)
        rejected = _normalize_match_indexes(self.rejected_match_indexes)
        if set(loaded).intersection(rejected):
            _fail(ControlPlaneErrorCode.INVALID_WORKER_RESULT)
        object.__setattr__(self, "loaded_match_indexes", loaded)
        object.__setattr__(self, "rejected_match_indexes", rejected)


@dataclass(frozen=True, slots=True)
class RequestLifecycleState:
    """Immutable state for one scheduler or worker request lifecycle."""

    plan: RequestPlan
    phase: RequestPhase = RequestPhase.LOOKED_UP
    allocation_generation: int = 0
    allocation: RequestAllocation | None = None
    handoff_metadata: RequestHandoffMetadata | None = None
    worker_validation: WorkerValidationReceipt | None = None
    preemption_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.phase, RequestPhase):
            _fail(ControlPlaneErrorCode.LIFECYCLE_MISUSE)
        if not _is_int(self.allocation_generation) or self.allocation_generation < 0:
            _fail(ControlPlaneErrorCode.STALE_ALLOCATION)
        if not _is_int(self.preemption_count) or self.preemption_count < 0:
            _fail(ControlPlaneErrorCode.STALE_ALLOCATION)

        if self.phase is RequestPhase.LOOKED_UP:
            if any(
                value is not None
                for value in (
                    self.allocation,
                    self.handoff_metadata,
                    self.worker_validation,
                )
            ):
                _fail(ControlPlaneErrorCode.LIFECYCLE_MISUSE)
            return
        if self.allocation is None:
            _fail(ControlPlaneErrorCode.LIFECYCLE_MISUSE)
        if self.allocation.request_id != self.plan.request_id:
            _fail(ControlPlaneErrorCode.INVALID_REQUEST_PLAN)
        if self.allocation.allocation_generation != self.allocation_generation:
            _fail(ControlPlaneErrorCode.STALE_ALLOCATION)

        if self.phase is RequestPhase.ALLOCATED:
            if self.handoff_metadata is not None or self.worker_validation is not None:
                _fail(ControlPlaneErrorCode.LIFECYCLE_MISUSE)
            return
        if self.handoff_metadata is None:
            _fail(ControlPlaneErrorCode.LIFECYCLE_MISUSE)
        if (
            self.handoff_metadata.plan != self.plan
            or self.handoff_metadata.allocation != self.allocation
        ):
            _fail(ControlPlaneErrorCode.STALE_ALLOCATION)

        if self.phase is RequestPhase.HANDED_OFF:
            if self.worker_validation is not None:
                _fail(ControlPlaneErrorCode.LIFECYCLE_MISUSE)
            return
        if self.worker_validation is None:
            _fail(ControlPlaneErrorCode.LIFECYCLE_MISUSE)
        if self.worker_validation.allocation != self.allocation:
            _fail(ControlPlaneErrorCode.STALE_ALLOCATION)

    @classmethod
    def from_handoff(
        cls,
        metadata: RequestHandoffMetadata,
        expected_group_layout: CacheGroupLayout,
    ) -> RequestLifecycleState:
        """Create worker-side state after validating the configured group layout."""

        if metadata.schema_version != METADATA_SCHEMA_VERSION:
            _fail(ControlPlaneErrorCode.METADATA_SCHEMA_MISMATCH)
        if metadata.allocation.grouped_blocks.group_layout != expected_group_layout:
            _fail(ControlPlaneErrorCode.GROUP_LAYOUT_MISMATCH)
        return cls(
            plan=metadata.plan,
            phase=RequestPhase.HANDED_OFF,
            allocation_generation=metadata.allocation.allocation_generation,
            allocation=metadata.allocation,
            handoff_metadata=metadata,
        )

    def allocate(
        self,
        group_layout: CacheGroupLayout,
        block_ids_by_group: Sequence[Sequence[int]],
        *,
        external_scheduler_tokens: int,
        allow_duplicate_block_ids: bool = False,
    ) -> RequestLifecycleState:
        """Bind the lookup plan to blocks, accepting only exact duplicates."""

        if (
            not _is_int(external_scheduler_tokens)
            or external_scheduler_tokens != FULL_RECOMPUTE_EXTERNAL_TOKENS
        ):
            _fail(ControlPlaneErrorCode.NONZERO_EXTERNAL_TOKENS)
        candidate = RequestAllocation(
            request_id=self.plan.request_id,
            allocation_generation=self.allocation_generation,
            grouped_blocks=GroupedBlockAllocation.capture(
                group_layout,
                block_ids_by_group,
                allow_duplicate_block_ids=allow_duplicate_block_ids,
            ),
            external_scheduler_tokens=external_scheduler_tokens,
        )
        if self.phase is RequestPhase.LOOKED_UP:
            return replace(
                self,
                phase=RequestPhase.ALLOCATED,
                allocation=candidate,
            )
        if self.allocation == candidate:
            return self
        if self.phase is RequestPhase.FINISHED:
            _fail(ControlPlaneErrorCode.LIFECYCLE_MISUSE)
        _fail(ControlPlaneErrorCode.DUPLICATE_REQUEST_CONFLICT)

    def handoff(self) -> RequestLifecycleState:
        """Freeze scheduler allocation metadata for the worker process."""

        if self.phase in {
            RequestPhase.HANDED_OFF,
            RequestPhase.WORKER_VALIDATED,
            RequestPhase.FINISHED,
        }:
            if self.handoff_metadata is None:
                _fail(ControlPlaneErrorCode.LIFECYCLE_MISUSE)
            return self
        if self.phase is not RequestPhase.ALLOCATED or self.allocation is None:
            _fail(ControlPlaneErrorCode.LIFECYCLE_MISUSE)
        metadata = RequestHandoffMetadata(
            schema_version=METADATA_SCHEMA_VERSION,
            plan=self.plan,
            allocation=self.allocation,
        )
        return replace(
            self,
            phase=RequestPhase.HANDED_OFF,
            handoff_metadata=metadata,
        )

    def apply_worker_validation(
        self, receipt: WorkerValidationReceipt
    ) -> RequestLifecycleState:
        """Accept a complete result for the current allocation generation."""

        if receipt.request_id != self.plan.request_id:
            _fail(ControlPlaneErrorCode.DUPLICATE_REQUEST_CONFLICT)
        if receipt.allocation.allocation_generation != self.allocation_generation:
            _fail(ControlPlaneErrorCode.STALE_ALLOCATION)
        if receipt.allocation != self.allocation:
            _fail(ControlPlaneErrorCode.STALE_ALLOCATION)

        expected_indexes = set(range(len(self.plan.match_plan.matches)))
        observed_indexes = set(receipt.loaded_match_indexes).union(
            receipt.rejected_match_indexes
        )
        if observed_indexes != expected_indexes:
            _fail(ControlPlaneErrorCode.INVALID_WORKER_RESULT)

        if self.phase is RequestPhase.HANDED_OFF:
            return replace(
                self,
                phase=RequestPhase.WORKER_VALIDATED,
                worker_validation=receipt,
            )
        if self.phase in {RequestPhase.WORKER_VALIDATED, RequestPhase.FINISHED}:
            if self.worker_validation == receipt:
                return self
            _fail(ControlPlaneErrorCode.DUPLICATE_REQUEST_CONFLICT)
        _fail(ControlPlaneErrorCode.LIFECYCLE_MISUSE)

    def preempt(self) -> RequestLifecycleState:
        """Invalidate all allocation-bound data and advance its generation."""

        if self.phase is RequestPhase.FINISHED:
            _fail(ControlPlaneErrorCode.LIFECYCLE_MISUSE)
        if self.phase is RequestPhase.LOOKED_UP:
            return self
        return replace(
            self,
            phase=RequestPhase.LOOKED_UP,
            allocation_generation=self.allocation_generation + 1,
            allocation=None,
            handoff_metadata=None,
            worker_validation=None,
            preemption_count=self.preemption_count + 1,
        )

    def finish(self) -> RequestLifecycleState:
        """Finish only after every planned match has a worker disposition."""

        if self.phase is RequestPhase.FINISHED:
            return self
        if self.phase is not RequestPhase.WORKER_VALIDATED:
            _fail(ControlPlaneErrorCode.LIFECYCLE_MISUSE)
        return replace(self, phase=RequestPhase.FINISHED)

    def derive_metric_counters(self) -> RequestMetricCounters:
        """Derive reconciled counters after worker validation.

        ``kv_tokens_found`` counts the unique requested token positions covered
        by any lookup candidate, including candidates rejected by the planner.
        ``reusable_documents_hit`` counts fully verified selected query
        segments, while ``kv_tokens_loaded`` counts only those accepted by the
        worker.  Counting unique positions keeps candidate collisions and
        overlapping rolling windows bounded by requested tokens.  Every
        verified match must still receive a worker disposition.
        """

        if self.phase not in {
            RequestPhase.WORKER_VALIDATED,
            RequestPhase.FINISHED,
        } or self.worker_validation is None:
            _fail(ControlPlaneErrorCode.LIFECYCLE_MISUSE)

        matches = self.plan.match_plan.matches
        loaded_indexes = self.worker_validation.loaded_match_indexes
        rejected_indexes = self.worker_validation.rejected_match_indexes
        loaded_tokens = sum(
            len(matches[index].target_segment) for index in loaded_indexes
        )
        candidate_targets = tuple(
            match.target_segment for match in matches
        ) + tuple(
            rejected.candidate.target_segment
            for rejected in self.plan.match_plan.rejected_candidates
        )
        found_tokens = _covered_token_count(candidate_targets)
        rejected_tokens = found_tokens - loaded_tokens
        if rejected_tokens < 0:  # defensive: selected matches are candidates
            _fail(ControlPlaneErrorCode.INVALID_WORKER_RESULT)
        verified_targets = {match.target_segment for match in matches}
        document_hits = sum(
            query in verified_targets for query in self.plan.query_segments
        )
        # Reading both partitions here makes the required complete worker
        # disposition explicit even though rejected token count includes
        # planner-stage rejection as well.
        if len(loaded_indexes) + len(rejected_indexes) != len(matches):
            _fail(ControlPlaneErrorCode.INVALID_WORKER_RESULT)

        return RequestMetricCounters(
            prompt_tokens=self.plan.prompt_tokens,
            reusable_documents_requested=len(self.plan.query_segments),
            reusable_documents_hit=document_hits,
            reusable_document_tokens_requested=self.plan.match_plan.requested_tokens,
            kv_tokens_found=found_tokens,
            kv_tokens_loaded=loaded_tokens,
            kv_tokens_rejected=rejected_tokens,
            tokens_recomputed=self.plan.prompt_tokens,
            prefill_tokens_avoided=0,
        )


class RequestControlPlane:
    """Small owner for immutable per-request scheduler or worker states."""

    def __init__(self, group_layout: CacheGroupLayout) -> None:
        self._group_layout = group_layout
        self._states: dict[str, RequestLifecycleState] = {}

    @property
    def group_layout(self) -> CacheGroupLayout:
        return self._group_layout

    def _require_state(self, request_id: str) -> RequestLifecycleState:
        try:
            return self._states[request_id]
        except KeyError:
            _fail(ControlPlaneErrorCode.UNKNOWN_REQUEST)

    def state(self, request_id: str) -> RequestLifecycleState:
        """Return the current immutable state for diagnostics or adaptation."""

        return self._require_state(request_id)

    def lookup(
        self,
        *,
        request_id: str,
        prompt_tokens: int,
        query_segments: Iterable[TokenSegment],
        match_plan: MatchPlan,
    ) -> RequestLifecycleState:
        """Record a verified planner result; exact duplicate calls are harmless."""

        plan = RequestPlan(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            query_segments=tuple(query_segments),
            match_plan=match_plan,
        )
        existing = self._states.get(request_id)
        if existing is not None:
            if existing.plan == plan:
                return existing
            _fail(ControlPlaneErrorCode.DUPLICATE_REQUEST_CONFLICT)
        state = RequestLifecycleState(plan=plan)
        self._states[request_id] = state
        return state

    def external_scheduler_tokens(self, request_id: str) -> int:
        """Return scheduler credit, hard-coded to zero for this milestone."""

        return self._require_state(request_id).plan.external_scheduler_tokens

    def allocate(
        self,
        request_id: str,
        block_ids_by_group: Sequence[Sequence[int]],
        *,
        external_scheduler_tokens: int,
        allow_duplicate_block_ids: bool = False,
    ) -> RequestLifecycleState:
        state = self._require_state(request_id).allocate(
            self._group_layout,
            block_ids_by_group,
            external_scheduler_tokens=external_scheduler_tokens,
            allow_duplicate_block_ids=allow_duplicate_block_ids,
        )
        self._states[request_id] = state
        return state

    def handoff(self, request_id: str) -> RequestHandoffMetadata:
        state = self._require_state(request_id).handoff()
        self._states[request_id] = state
        if state.handoff_metadata is None:  # defensive: protected by state invariant
            _fail(ControlPlaneErrorCode.LIFECYCLE_MISUSE)
        return state.handoff_metadata

    def accept_handoff(
        self, metadata: RequestHandoffMetadata
    ) -> RequestLifecycleState:
        """Validate scheduler metadata and install worker-side request state."""

        existing = self._states.get(metadata.plan.request_id)
        if existing is not None:
            if existing.handoff_metadata == metadata:
                return existing
            _fail(ControlPlaneErrorCode.DUPLICATE_REQUEST_CONFLICT)
        state = RequestLifecycleState.from_handoff(metadata, self._group_layout)
        self._states[metadata.plan.request_id] = state
        return state

    def validate_worker(
        self,
        request_id: str,
        *,
        loaded_match_indexes: Iterable[int],
        rejected_match_indexes: Iterable[int],
    ) -> WorkerValidationReceipt:
        """Record the worker's complete loaded/rejected match partition."""

        state = self._require_state(request_id)
        if state.allocation is None:
            _fail(ControlPlaneErrorCode.LIFECYCLE_MISUSE)
        receipt = WorkerValidationReceipt(
            request_id=request_id,
            allocation=state.allocation,
            loaded_match_indexes=tuple(loaded_match_indexes),
            rejected_match_indexes=tuple(rejected_match_indexes),
        )
        state = state.apply_worker_validation(receipt)
        self._states[request_id] = state
        return receipt

    def apply_worker_validation(
        self, receipt: WorkerValidationReceipt
    ) -> RequestLifecycleState:
        """Apply a worker receipt to matching scheduler-side state."""

        state = self._require_state(receipt.request_id).apply_worker_validation(
            receipt
        )
        self._states[receipt.request_id] = state
        return state

    def preempt(self, request_id: str) -> RequestLifecycleState:
        state = self._require_state(request_id).preempt()
        self._states[request_id] = state
        return state

    def finish(self, request_id: str) -> RequestLifecycleState:
        state = self._require_state(request_id).finish()
        self._states[request_id] = state
        return state

    def discard(self, request_id: str) -> RequestLifecycleState | None:
        """Forget one request after vLLM has released its lifecycle.

        vLLM may finish or cancel a request before it reaches every normal
        connector phase.  Removing dependency-free bookkeeping is always safe
        at the 100%-recompute milestone because the scheduler has credited zero
        external tokens and this control plane never owns vLLM cache blocks.
        """

        return self._states.pop(request_id, None)


__all__ = [
    "FULL_RECOMPUTE_EXTERNAL_TOKENS",
    "METADATA_SCHEMA_VERSION",
    "CacheGroupLayout",
    "ControlPlaneError",
    "ControlPlaneErrorCode",
    "GroupBlockSnapshot",
    "GroupedBlockAllocation",
    "RequestAllocation",
    "RequestControlPlane",
    "RequestHandoffMetadata",
    "RequestLifecycleState",
    "RequestPhase",
    "RequestPlan",
    "WorkerValidationReceipt",
]
