# SPDX-License-Identifier: Apache-2.0
"""Dependency-free scheduler lookup runtime for 100% recomputation.

This module is the pre-allocation half of the pinned vLLM 0.19.1 connector.
It retains a non-prefix lookup plan while always reporting ``(0, False)`` to
the scheduler.  A later connector adapter can bind :class:`RequestPlan` to
hybrid-group blocks and combine this module's immutable metadata with
``RequestHandoffMetadata``.

The lifecycle is based on vLLM commit
``b1388b1fbf5aaef47937fabe98931211684666a6``:

* ``get_num_new_matched_tokens`` may be called repeatedly and must be
  side-effect free; zero matched tokens requires a false async flag:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L449-L482
* the V1 scheduler queries the connector only before a request has computed
  tokens, then allocates ordinary prompt slots and calls
  ``update_state_after_alloc`` even when the external count is zero:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/core/sched/scheduler.py#L605-L644
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/core/sched/scheduler.py#L656-L681
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/core/sched/scheduler.py#L746-L775
* scheduler metadata is built after allocation and must not mutate
  ``SchedulerOutput``:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L485-L518
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/core/sched/scheduler.py#L932-L954
* preemption frees blocks, resets computed tokens to zero, and increments the
  request's preemption counter before it is queued again:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/core/sched/scheduler.py#L956-L976
* worker metadata is bound and ``start_load_kv`` runs before model forward;
  metadata is cleared after connector finalization:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/worker/kv_connector_model_runner_mixin.py#L91-L129
* the attention wrapper waits for each layer, executes attention, and then
  offers that layer to ``save_kv_layer``:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/layers/attention/kv_transfer_utils.py#L14-L58

LMCache 0.4.3 already performs the same 256-token sliding-window search over
the full query and returns compact-document source positions plus current
query positions:
https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/v1/multiprocess/blend_server_v2.py#L182-L248

No vLLM, LMCache, Torch, CUDA, or networking package is imported here.  The
candidate transport and exact-token sidecar coordinator are injected.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import NoReturn, Protocol

from cacheblend_gpt_oss.connector.control_plane import (
    FULL_RECOMPUTE_EXTERNAL_TOKENS,
    METADATA_SCHEMA_VERSION,
    RequestPlan,
)
from cacheblend_gpt_oss.gpt_oss.layout import GPT_OSS_MAX_CONTEXT_TOKENS
from cacheblend_gpt_oss.planner.matching import MatchPlan
from cacheblend_gpt_oss.planner.models import (
    CacheNamespace,
    TokenRange,
    normalize_token_ids,
)
from cacheblend_gpt_oss.storage.lmcache_types import (
    LMCACHE_CHUNK_SIZE,
    LmcacheBlendTransportConfig,
    LmcacheCandidate,
    LmcacheOperationError,
    LmcacheProtocolError,
    LmcacheTransportError,
    VerifiedLmcacheCandidate,
)
from cacheblend_gpt_oss.storage.lookup import (
    LmcacheLookupCounters,
    LmcacheLookupError,
    LmcacheLookupErrorCode,
    LmcacheLookupPlan,
)
from cacheblend_gpt_oss.storage.sidecar import (
    SidecarCorruptionError,
    SidecarSchemaError,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.transfer_config import (
    Transfer100PctConfig,
)

MAX_REQUEST_ID_BYTES = 256


class SchedulerRuntimeErrorCode(str, Enum):
    """Bounded construction/input/cleanup failures."""

    INVALID_CONFIG = "invalid_config"
    INVALID_INPUT = "invalid_input"
    INVALID_METADATA = "invalid_metadata"
    CLOSE_FAILED = "close_failed"


class SchedulerRuntimeError(RuntimeError):
    """Fail-closed error that never contains request or prompt data."""

    def __init__(self, code: SchedulerRuntimeErrorCode) -> None:
        self.code = code
        super().__init__(f"scheduler runtime failure: {code.value}")


def _fail(code: SchedulerRuntimeErrorCode) -> NoReturn:
    raise SchedulerRuntimeError(code)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class SchedulerRuntimeState(str, Enum):
    """Process-local resource state for the injected scheduler transport."""

    CREATED = "created"
    READY = "ready"
    DEGRADED = "degraded"
    FATAL = "fatal"
    CLOSED = "closed"


class SchedulerLookupStatus(str, Enum):
    """Bounded per-request outcomes safe for logs and metric labels."""

    TRANSFER_READY = "transfer_ready"
    FULL_PREFILL_MISS = "full_prefill_miss"
    FULL_PREFILL_TRANSPORT_ERROR = "full_prefill_transport_error"
    FULL_PREFILL_SIDECAR_ERROR = "full_prefill_sidecar_error"
    FULL_PREFILL_SEQUENCE_INELIGIBLE = "full_prefill_sequence_ineligible"
    FULL_PREFILL_STEP_INELIGIBLE = "full_prefill_step_ineligible"
    FULL_PREFILL_PROMPT_TOO_LARGE = "full_prefill_prompt_too_large"
    FATAL_NONZERO_EXTERNAL_TOKENS = "fatal_nonzero_external_tokens"
    FATAL_PROTOCOL_DRIFT = "fatal_protocol_drift"
    FATAL_TRANSPORT_CLEANUP = "fatal_transport_cleanup"
    FATAL_SIDECAR_CORRUPTION = "fatal_sidecar_corruption"
    FATAL_DUPLICATE_CONFLICT = "fatal_duplicate_conflict"
    FATAL_RUNTIME_CLOSED = "fatal_runtime_closed"

    @property
    def is_fatal(self) -> bool:
        return self.value.startswith("fatal_")

    @property
    def requires_full_prefill(self) -> bool:
        return self is not SchedulerLookupStatus.TRANSFER_READY


class CandidateTransport(Protocol):
    """Scheduler-side subset of the pinned LMCache transport."""

    @property
    def config(self) -> LmcacheBlendTransportConfig:
        """Return the exact namespace and chunk configuration."""

    def open(self) -> None:
        """Synchronously probe and open the candidate transport."""

    def lookup_candidates(
        self, token_ids: Sequence[int], *, request_id: str
    ) -> tuple[LmcacheCandidate, ...]:
        """Return untrusted rolling-window candidates for the full prompt."""

    def close(self) -> None:
        """Release transport resources."""


class CandidateLookupCoordinator(Protocol):
    """Injected read-only exact-token sidecar verification boundary."""

    def plan(
        self,
        prompt_token_ids: Iterable[int],
        namespace: CacheNamespace,
        candidates: Iterable[LmcacheCandidate],
    ) -> LmcacheLookupPlan:
        """Verify untrusted candidates without mutating the sidecar."""


class CandidateTransportFactory(Protocol):
    """Create one fresh transport for a later-request availability retry."""

    def __call__(self) -> CandidateTransport:
        """Return a newly constructed transport with the exact same config."""


@dataclass(frozen=True, slots=True)
class SchedulerLookupRequest:
    """Exact inputs available at the pinned scheduler lookup hook.

    The full prompt is copied and hidden from ``repr`` so retaining this value
    for scheduler-to-worker handoff cannot accidentally log token IDs.
    ``scheduler_step_index`` is zero for the only eligible full-prefill step.
    """

    request_id: str
    prompt_token_ids: tuple[int, ...] = field(repr=False)
    sequence_count: int
    scheduler_step_index: int
    num_computed_tokens: int
    num_external_tokens: int
    preemption_count: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.request_id, str)
            or not self.request_id
            or len(self.request_id.encode("utf-8")) > MAX_REQUEST_ID_BYTES
        ):
            _fail(SchedulerRuntimeErrorCode.INVALID_INPUT)
        try:
            prompt = normalize_token_ids(self.prompt_token_ids)
        except (TypeError, ValueError):
            _fail(SchedulerRuntimeErrorCode.INVALID_INPUT)
        if not prompt or len(prompt) > GPT_OSS_MAX_CONTEXT_TOKENS:
            _fail(SchedulerRuntimeErrorCode.INVALID_INPUT)
        integer_fields = (
            self.sequence_count,
            self.scheduler_step_index,
            self.num_computed_tokens,
            self.num_external_tokens,
            self.preemption_count,
        )
        if any(not _is_int(value) or value < 0 for value in integer_fields):
            _fail(SchedulerRuntimeErrorCode.INVALID_INPUT)
        if self.sequence_count < 1 or self.num_computed_tokens > len(prompt):
            _fail(SchedulerRuntimeErrorCode.INVALID_INPUT)
        object.__setattr__(self, "prompt_token_ids", prompt)


@dataclass(frozen=True, slots=True)
class SchedulerLookupMetadata:
    """Immutable pre-allocation metadata retained for worker handoff.

    ``request_plan`` is directly consumable by the generic control plane after
    its fields are passed to ``RequestControlPlane.lookup``.  It contains only
    selected, non-overlapping exact matches.  ``query_windows`` compactly
    records the complete rolling-window search space without duplicating each
    window's 256 token IDs; those remain in ``prompt_token_ids`` exactly once.
    """

    schema_version: int
    request_plan: RequestPlan = field(repr=False)
    prompt_token_ids: tuple[int, ...] = field(repr=False)
    query_windows: tuple[TokenRange, ...]
    lookup_plan: LmcacheLookupPlan = field(repr=False)
    status: SchedulerLookupStatus
    preemption_count: int
    allocation_generation: int
    external_scheduler_tokens: int = FULL_RECOMPUTE_EXTERNAL_TOKENS
    load_kv_async: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != METADATA_SCHEMA_VERSION:
            _fail(SchedulerRuntimeErrorCode.INVALID_METADATA)
        if not isinstance(self.request_plan, RequestPlan):
            _fail(SchedulerRuntimeErrorCode.INVALID_METADATA)
        try:
            prompt = normalize_token_ids(self.prompt_token_ids)
            windows = tuple(self.query_windows)
        except (TypeError, ValueError):
            _fail(SchedulerRuntimeErrorCode.INVALID_METADATA)
        if not prompt or self.request_plan.prompt_tokens != len(prompt):
            _fail(SchedulerRuntimeErrorCode.INVALID_METADATA)
        if any(
            not isinstance(window, TokenRange)
            or len(window) != LMCACHE_CHUNK_SIZE
            or window.end > len(prompt)
            for window in windows
        ):
            _fail(SchedulerRuntimeErrorCode.INVALID_METADATA)
        expected_window_count = max(0, len(prompt) - LMCACHE_CHUNK_SIZE + 1)
        if windows and (
            len(windows) != expected_window_count
            or any(
                window.start != start
                or window.end != start + LMCACHE_CHUNK_SIZE
                for start, window in enumerate(windows)
            )
        ):
            _fail(SchedulerRuntimeErrorCode.INVALID_METADATA)
        if not isinstance(self.lookup_plan, LmcacheLookupPlan) or not isinstance(
            self.status, SchedulerLookupStatus
        ):
            _fail(SchedulerRuntimeErrorCode.INVALID_METADATA)
        if (
            not _is_int(self.preemption_count)
            or self.preemption_count < 0
            or not _is_int(self.allocation_generation)
            or self.allocation_generation < 0
        ):
            _fail(SchedulerRuntimeErrorCode.INVALID_METADATA)
        if (
            not _is_int(self.external_scheduler_tokens)
            or self.external_scheduler_tokens != FULL_RECOMPUTE_EXTERNAL_TOKENS
            or self.load_kv_async is not False
        ):
            _fail(SchedulerRuntimeErrorCode.INVALID_METADATA)

        verified = self.lookup_plan.verified_candidates
        expected_matches = tuple(candidate.match for candidate in verified)
        if self.request_plan.match_plan.matches != expected_matches:
            _fail(SchedulerRuntimeErrorCode.INVALID_METADATA)
        if self.status is SchedulerLookupStatus.TRANSFER_READY:
            if not verified or not windows:
                _fail(SchedulerRuntimeErrorCode.INVALID_METADATA)
        elif expected_matches:
            _fail(SchedulerRuntimeErrorCode.INVALID_METADATA)
        object.__setattr__(self, "prompt_token_ids", prompt)
        object.__setattr__(self, "query_windows", windows)

    @property
    def request_id(self) -> str:
        return self.request_plan.request_id

    @property
    def verified_candidates(self) -> tuple[VerifiedLmcacheCandidate, ...]:
        """Return the exact-token-verified candidates selected for transfer."""

        return self.lookup_plan.verified_candidates

    @property
    def should_transfer(self) -> bool:
        return self.status is SchedulerLookupStatus.TRANSFER_READY


@dataclass(frozen=True, slots=True)
class _StoredRequest:
    request: SchedulerLookupRequest = field(repr=False)
    metadata: SchedulerLookupMetadata = field(repr=False)


def _empty_lookup_plan() -> LmcacheLookupPlan:
    return LmcacheLookupPlan(
        verified_candidates=(),
        rejected_candidates=(),
        counters=LmcacheLookupCounters(
            raw_candidates=0,
            raw_candidate_tokens=0,
            found_candidates=0,
            found_candidate_tokens=0,
            verified_candidates=0,
            verified_candidate_tokens=0,
            rejected_candidates=0,
            rejected_candidate_tokens=0,
        ),
    )


def _request_plan(
    request: SchedulerLookupRequest,
    lookup_plan: LmcacheLookupPlan,
) -> RequestPlan:
    matches = tuple(candidate.match for candidate in lookup_plan.verified_candidates)
    segments = tuple(match.target_segment for match in matches)
    return RequestPlan(
        request_id=request.request_id,
        prompt_tokens=len(request.prompt_token_ids),
        query_segments=segments,
        match_plan=MatchPlan(
            matches=matches,
            rejected_candidates=(),
            requested_tokens=sum(len(segment) for segment in segments),
        ),
    )


def _query_windows(prompt_tokens: int) -> tuple[TokenRange, ...]:
    if prompt_tokens < LMCACHE_CHUNK_SIZE:
        return ()
    return tuple(
        TokenRange(start, start + LMCACHE_CHUNK_SIZE)
        for start in range(prompt_tokens - LMCACHE_CHUNK_SIZE + 1)
    )


def _sidecar_corruption(error: BaseException) -> bool:
    cause: BaseException | None = error
    while cause is not None:
        if isinstance(cause, SidecarCorruptionError | SidecarSchemaError):
            return True
        cause = cause.__cause__
    return False


class SchedulerLookupRuntime:
    """Own candidate-transport lifecycle and exact per-request lookup state."""

    def __init__(
        self,
        config: Transfer100PctConfig,
        candidate_transport: CandidateTransport,
        lookup_coordinator: CandidateLookupCoordinator,
        *,
        replacement_transport_factory: CandidateTransportFactory,
    ) -> None:
        if not isinstance(config, Transfer100PctConfig):
            _fail(SchedulerRuntimeErrorCode.INVALID_CONFIG)
        try:
            transport_config = candidate_transport.config
        except Exception as exc:
            raise SchedulerRuntimeError(
                SchedulerRuntimeErrorCode.INVALID_CONFIG
            ) from exc
        if (
            not isinstance(transport_config, LmcacheBlendTransportConfig)
            or transport_config.namespace != config.namespace
            or transport_config.chunk_size != LMCACHE_CHUNK_SIZE
        ):
            _fail(SchedulerRuntimeErrorCode.INVALID_CONFIG)
        self._config = config
        self._transport = candidate_transport
        self._transport_config = transport_config
        if not callable(replacement_transport_factory):
            _fail(SchedulerRuntimeErrorCode.INVALID_CONFIG)
        self._replacement_transport_factory = replacement_transport_factory
        self._transport_needs_close = True
        self._coordinator = lookup_coordinator
        self._state = SchedulerRuntimeState.CREATED
        self._fatal_status: SchedulerLookupStatus | None = None
        self._requests: dict[str, _StoredRequest] = {}

    @property
    def state(self) -> SchedulerRuntimeState:
        return self._state

    @property
    def active_request_count(self) -> int:
        return len(self._requests)

    def open(self) -> SchedulerRuntimeState:
        """Open once; connectivity failure degrades to explicit full prefill."""

        if self._state in {
            SchedulerRuntimeState.READY,
            SchedulerRuntimeState.DEGRADED,
            SchedulerRuntimeState.FATAL,
        }:
            return self._state
        if self._state is SchedulerRuntimeState.CLOSED:
            _fail(SchedulerRuntimeErrorCode.INVALID_INPUT)
        try:
            self._transport.open()
        except LmcacheOperationError:
            self._state = SchedulerRuntimeState.DEGRADED
        except LmcacheTransportError:
            self._mark_fatal(SchedulerLookupStatus.FATAL_PROTOCOL_DRIFT)
        except Exception:
            self._mark_fatal(SchedulerLookupStatus.FATAL_PROTOCOL_DRIFT)
        else:
            self._state = SchedulerRuntimeState.READY
        return self._state

    def lookup(self, request: SchedulerLookupRequest) -> SchedulerLookupMetadata:
        """Lookup, exactly verify, and retain one immutable request result."""

        if not isinstance(request, SchedulerLookupRequest):
            _fail(SchedulerRuntimeErrorCode.INVALID_INPUT)

        if self._state is SchedulerRuntimeState.CLOSED:
            return self._build_metadata(
                request,
                SchedulerLookupStatus.FATAL_RUNTIME_CLOSED,
                _empty_lookup_plan(),
                (),
                allocation_generation=request.preemption_count,
            )
        if self._state is SchedulerRuntimeState.FATAL:
            return self._build_metadata(
                request,
                self._fatal_status or SchedulerLookupStatus.FATAL_PROTOCOL_DRIFT,
                _empty_lookup_plan(),
                (),
                allocation_generation=request.preemption_count,
            )

        existing = self._requests.get(request.request_id)
        if existing is not None:
            return self._repeat_or_preempt(existing, request)

        generation = 0
        status = self._eligibility_status(request)
        if status is not None:
            metadata = self._build_metadata(
                request,
                status,
                _empty_lookup_plan(),
                (),
                allocation_generation=generation,
            )
            if not status.is_fatal:
                self._requests[request.request_id] = _StoredRequest(request, metadata)
            return metadata

        runtime_state: SchedulerRuntimeState
        if self._state is SchedulerRuntimeState.CREATED:
            runtime_state = self.open()
        elif self._state is SchedulerRuntimeState.DEGRADED:
            runtime_state = self._retry_degraded_transport_once()
        else:
            runtime_state = self.state
        windows = _query_windows(len(request.prompt_token_ids))
        if runtime_state is SchedulerRuntimeState.DEGRADED:
            metadata = self._build_metadata(
                request,
                SchedulerLookupStatus.FULL_PREFILL_TRANSPORT_ERROR,
                _empty_lookup_plan(),
                windows,
                allocation_generation=generation,
            )
            self._requests[request.request_id] = _StoredRequest(request, metadata)
            return metadata
        if runtime_state is SchedulerRuntimeState.FATAL:
            return self._build_metadata(
                request,
                self._fatal_status or SchedulerLookupStatus.FATAL_PROTOCOL_DRIFT,
                _empty_lookup_plan(),
                windows,
                allocation_generation=generation,
            )

        try:
            candidates = self._transport.lookup_candidates(
                request.prompt_token_ids,
                request_id=request.request_id,
            )
        except LmcacheOperationError:
            self._state = SchedulerRuntimeState.DEGRADED
            status = SchedulerLookupStatus.FULL_PREFILL_TRANSPORT_ERROR
            lookup_plan = _empty_lookup_plan()
        except LmcacheProtocolError:
            self._mark_fatal(SchedulerLookupStatus.FATAL_PROTOCOL_DRIFT)
            status = SchedulerLookupStatus.FATAL_PROTOCOL_DRIFT
            lookup_plan = _empty_lookup_plan()
        except LmcacheTransportError:
            self._mark_fatal(SchedulerLookupStatus.FATAL_PROTOCOL_DRIFT)
            status = SchedulerLookupStatus.FATAL_PROTOCOL_DRIFT
            lookup_plan = _empty_lookup_plan()
        except Exception:
            self._mark_fatal(SchedulerLookupStatus.FATAL_PROTOCOL_DRIFT)
            status = SchedulerLookupStatus.FATAL_PROTOCOL_DRIFT
            lookup_plan = _empty_lookup_plan()
        else:
            status, lookup_plan = self._verify_candidates(
                request, candidates, len(windows)
            )

        metadata = self._build_metadata(
            request,
            status,
            lookup_plan,
            windows,
            allocation_generation=generation,
        )
        if not status.is_fatal:
            self._requests[request.request_id] = _StoredRequest(request, metadata)
        return metadata

    def discard(self, request_id: str) -> SchedulerLookupMetadata | None:
        """Forget prompt-bearing request state; duplicate cleanup is harmless."""

        if not isinstance(request_id, str):
            _fail(SchedulerRuntimeErrorCode.INVALID_INPUT)
        stored = self._requests.pop(request_id, None)
        return None if stored is None else stored.metadata

    def close(self) -> None:
        """Release the owned transport and all retained prompt tuples once."""

        if self._state is SchedulerRuntimeState.CLOSED:
            return
        self._requests.clear()
        if not self._transport_needs_close:
            self._state = SchedulerRuntimeState.CLOSED
            return
        try:
            self._transport.close()
        except Exception as exc:
            self._state = SchedulerRuntimeState.CLOSED
            raise SchedulerRuntimeError(SchedulerRuntimeErrorCode.CLOSE_FAILED) from exc
        self._transport_needs_close = False
        self._state = SchedulerRuntimeState.CLOSED

    def _retry_degraded_transport_once(self) -> SchedulerRuntimeState:
        """Replace once for this distinct request; never loop in scheduler code."""

        if self._transport_needs_close:
            try:
                self._transport.close()
            except Exception:
                self._mark_fatal(
                    SchedulerLookupStatus.FATAL_TRANSPORT_CLEANUP
                )
                return self._state
            self._transport_needs_close = False

        try:
            replacement_transport = self._replacement_transport_factory()
        except LmcacheOperationError:
            self._state = SchedulerRuntimeState.DEGRADED
            return self._state
        except Exception:
            self._mark_fatal(SchedulerLookupStatus.FATAL_PROTOCOL_DRIFT)
            return self._state

        if replacement_transport is self._transport:
            self._mark_fatal(SchedulerLookupStatus.FATAL_PROTOCOL_DRIFT)
            return self._state
        try:
            replacement_config = replacement_transport.config
            valid_config = (
                isinstance(replacement_config, LmcacheBlendTransportConfig)
                and replacement_config == self._transport_config
                and replacement_config.namespace == self._config.namespace
                and replacement_config.chunk_size == LMCACHE_CHUNK_SIZE
            )
        except Exception:
            valid_config = False
        if not valid_config:
            try:
                replacement_transport.close()
            except Exception:
                self._mark_fatal(
                    SchedulerLookupStatus.FATAL_TRANSPORT_CLEANUP
                )
                return self._state
            self._mark_fatal(SchedulerLookupStatus.FATAL_PROTOCOL_DRIFT)
            return self._state

        self._transport = replacement_transport
        self._transport_needs_close = True
        self._state = SchedulerRuntimeState.CREATED
        return self.open()

    def _eligibility_status(
        self, request: SchedulerLookupRequest
    ) -> SchedulerLookupStatus | None:
        if request.num_external_tokens != FULL_RECOMPUTE_EXTERNAL_TOKENS:
            self._mark_fatal(
                SchedulerLookupStatus.FATAL_NONZERO_EXTERNAL_TOKENS
            )
            return SchedulerLookupStatus.FATAL_NONZERO_EXTERNAL_TOKENS
        if request.sequence_count != 1:
            return SchedulerLookupStatus.FULL_PREFILL_SEQUENCE_INELIGIBLE
        if request.scheduler_step_index != 0 or request.num_computed_tokens != 0:
            return SchedulerLookupStatus.FULL_PREFILL_STEP_INELIGIBLE
        if len(request.prompt_token_ids) > self._config.staging_token_capacity:
            return SchedulerLookupStatus.FULL_PREFILL_PROMPT_TOO_LARGE
        return None

    def _verify_candidates(
        self,
        request: SchedulerLookupRequest,
        candidates: object,
        query_window_count: int,
    ) -> tuple[SchedulerLookupStatus, LmcacheLookupPlan]:
        if not isinstance(candidates, tuple) or any(
            not isinstance(candidate, LmcacheCandidate) for candidate in candidates
        ):
            self._mark_fatal(SchedulerLookupStatus.FATAL_PROTOCOL_DRIFT)
            return (
                SchedulerLookupStatus.FATAL_PROTOCOL_DRIFT,
                _empty_lookup_plan(),
            )
        if len(candidates) > query_window_count:
            self._mark_fatal(SchedulerLookupStatus.FATAL_PROTOCOL_DRIFT)
            return (
                SchedulerLookupStatus.FATAL_PROTOCOL_DRIFT,
                _empty_lookup_plan(),
            )
        try:
            lookup_plan = self._coordinator.plan(
                request.prompt_token_ids,
                self._config.namespace,
                candidates,
            )
        except LmcacheLookupError as exc:
            if _sidecar_corruption(exc):
                status = SchedulerLookupStatus.FATAL_SIDECAR_CORRUPTION
                self._mark_fatal(status)
                return status, _empty_lookup_plan()
            if exc.code is LmcacheLookupErrorCode.SIDECAR_LOOKUP_FAILED:
                return (
                    SchedulerLookupStatus.FULL_PREFILL_SIDECAR_ERROR,
                    _empty_lookup_plan(),
                )
            status = SchedulerLookupStatus.FATAL_PROTOCOL_DRIFT
            self._mark_fatal(status)
            return status, _empty_lookup_plan()
        except (SidecarCorruptionError, SidecarSchemaError):
            status = SchedulerLookupStatus.FATAL_SIDECAR_CORRUPTION
            self._mark_fatal(status)
            return status, _empty_lookup_plan()
        except Exception:
            status = SchedulerLookupStatus.FATAL_PROTOCOL_DRIFT
            self._mark_fatal(status)
            return status, _empty_lookup_plan()
        if not isinstance(lookup_plan, LmcacheLookupPlan):
            status = SchedulerLookupStatus.FATAL_PROTOCOL_DRIFT
            self._mark_fatal(status)
            return status, _empty_lookup_plan()
        if lookup_plan.verified_candidates:
            return SchedulerLookupStatus.TRANSFER_READY, lookup_plan
        return SchedulerLookupStatus.FULL_PREFILL_MISS, lookup_plan

    def _repeat_or_preempt(
        self,
        existing: _StoredRequest,
        request: SchedulerLookupRequest,
    ) -> SchedulerLookupMetadata:
        previous = existing.request
        if request == previous:
            return existing.metadata
        if (
            request.preemption_count != previous.preemption_count + 1
            or replace(request, preemption_count=previous.preemption_count) != previous
        ):
            self._mark_fatal(SchedulerLookupStatus.FATAL_DUPLICATE_CONFLICT)
            return self._build_metadata(
                request,
                SchedulerLookupStatus.FATAL_DUPLICATE_CONFLICT,
                _empty_lookup_plan(),
                (),
                allocation_generation=existing.metadata.allocation_generation,
            )

        # vLLM invalidates allocation on preemption but the exact token lookup
        # remains valid.  Reusing it is both side-effect free and safe: a later
        # LMCache eviction is caught by synchronous worker retrieval and falls
        # back to full prefill.
        metadata = replace(
            existing.metadata,
            preemption_count=request.preemption_count,
            allocation_generation=existing.metadata.allocation_generation + 1,
        )
        self._requests[request.request_id] = _StoredRequest(request, metadata)
        return metadata

    def _build_metadata(
        self,
        request: SchedulerLookupRequest,
        status: SchedulerLookupStatus,
        lookup_plan: LmcacheLookupPlan,
        windows: tuple[TokenRange, ...],
        *,
        allocation_generation: int,
    ) -> SchedulerLookupMetadata:
        return SchedulerLookupMetadata(
            schema_version=METADATA_SCHEMA_VERSION,
            request_plan=_request_plan(request, lookup_plan),
            prompt_token_ids=request.prompt_token_ids,
            query_windows=windows,
            lookup_plan=lookup_plan,
            status=status,
            preemption_count=request.preemption_count,
            allocation_generation=allocation_generation,
        )

    def _mark_fatal(self, status: SchedulerLookupStatus) -> None:
        if not status.is_fatal:
            _fail(SchedulerRuntimeErrorCode.INVALID_METADATA)
        self._fatal_status = status
        self._requests.clear()
        self._state = SchedulerRuntimeState.FATAL


__all__ = [
    "CandidateLookupCoordinator",
    "CandidateTransport",
    "CandidateTransportFactory",
    "SchedulerLookupMetadata",
    "SchedulerLookupRequest",
    "SchedulerLookupRuntime",
    "SchedulerLookupStatus",
    "SchedulerRuntimeError",
    "SchedulerRuntimeErrorCode",
    "SchedulerRuntimeState",
]
