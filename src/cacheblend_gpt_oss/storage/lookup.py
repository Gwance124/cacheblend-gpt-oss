# SPDX-License-Identifier: Apache-2.0
"""Exact sidecar verification for untrusted LMCache lookup candidates.

LMCache's Blend matcher establishes only that a rolling storage hash matched.
This dependency-free coordinator slices the exact current prompt, derives this
project's SHA-256 fingerprint, queries a :class:`RecordLookup` collision
bucket, requires the LMCache cache key, and independently verifies namespace,
fingerprint, and every token before binding a ``VerifiedLmcacheCandidate``.

No candidate selected here is scheduler prefix credit.  The current milestone
still recomputes the complete prompt after any instrumented KV transfer.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import NoReturn

from cacheblend_gpt_oss.planner.fingerprint import SHA256_FINGERPRINTER
from cacheblend_gpt_oss.planner.matching import RecordLookup, VerifiedMatch
from cacheblend_gpt_oss.planner.models import (
    CacheNamespace,
    CacheRecord,
    CandidateMatch,
    SegmentFingerprint,
    TokenSegment,
    normalize_token_ids,
)
from cacheblend_gpt_oss.storage.lmcache_types import (
    LmcacheCandidate,
    LmcacheProtocolError,
    VerifiedLmcacheCandidate,
    query_digest,
)


class LmcacheLookupErrorCode(str, Enum):
    """Bounded fatal errors safe for logs and aggregate metric labels."""

    INVALID_PROMPT_TOKENS = "invalid_prompt_tokens"
    INVALID_CANDIDATE = "invalid_candidate"
    TARGET_RANGE_OUT_OF_BOUNDS = "target_range_out_of_bounds"
    SIDECAR_LOOKUP_FAILED = "sidecar_lookup_failed"
    INVALID_SIDECAR_RESPONSE = "invalid_sidecar_response"
    BINDING_INVARIANT_FAILED = "binding_invariant_failed"
    ACCOUNTING_INVARIANT_FAILED = "accounting_invariant_failed"


class LmcacheLookupError(RuntimeError):
    """Fail-closed lookup error whose message contains no request data."""

    def __init__(self, code: LmcacheLookupErrorCode) -> None:
        self.code = code
        super().__init__(f"LMCache sidecar lookup failure: {code.value}")


def _fail(code: LmcacheLookupErrorCode) -> NoReturn:
    raise LmcacheLookupError(code)


class LmcacheCandidateRejectionReason(str, Enum):
    """Bounded reasons for rejecting one otherwise well-formed candidate."""

    STALE_QUERY = "stale_query"
    NO_SIDECAR_RECORD = "no_sidecar_record"
    CACHE_KEY_MISMATCH = "cache_key_mismatch"
    NAMESPACE_MISMATCH = "namespace_mismatch"
    FINGERPRINT_MISMATCH = "fingerprint_mismatch"
    TOKEN_MISMATCH = "token_mismatch"
    DUPLICATE_CANDIDATE = "duplicate_candidate"
    OVERLAPS_SELECTED_CANDIDATE = "overlaps_selected_candidate"


_VERIFICATION_REASON_PRIORITY = {
    LmcacheCandidateRejectionReason.NAMESPACE_MISMATCH: 0,
    LmcacheCandidateRejectionReason.FINGERPRINT_MISMATCH: 1,
    LmcacheCandidateRejectionReason.TOKEN_MISMATCH: 2,
}


@dataclass(frozen=True, slots=True)
class RejectedLmcacheCandidate:
    """A raw candidate and its bounded terminal rejection reason."""

    candidate: LmcacheCandidate
    reason: LmcacheCandidateRejectionReason


@dataclass(frozen=True, slots=True)
class LmcacheLookupCounters:
    """Identifier-free candidate/token accounting for one prompt lookup.

    ``found`` means that the SHA-256 bucket contained at least one record whose
    cache key matched the raw LMCache candidate.  ``verified`` counts only the
    exact candidates selected in the final non-overlapping plan.  Raw values
    include duplicates and overlaps; each raw candidate is ultimately either
    verified or rejected.
    """

    raw_candidates: int
    raw_candidate_tokens: int
    found_candidates: int
    found_candidate_tokens: int
    verified_candidates: int
    verified_candidate_tokens: int
    rejected_candidates: int
    rejected_candidate_tokens: int

    def __post_init__(self) -> None:
        values = (
            self.raw_candidates,
            self.raw_candidate_tokens,
            self.found_candidates,
            self.found_candidate_tokens,
            self.verified_candidates,
            self.verified_candidate_tokens,
            self.rejected_candidates,
            self.rejected_candidate_tokens,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            _fail(LmcacheLookupErrorCode.ACCOUNTING_INVARIANT_FAILED)
        if self.raw_candidates != (
            self.verified_candidates + self.rejected_candidates
        ):
            _fail(LmcacheLookupErrorCode.ACCOUNTING_INVARIANT_FAILED)
        if self.raw_candidate_tokens != (
            self.verified_candidate_tokens + self.rejected_candidate_tokens
        ):
            _fail(LmcacheLookupErrorCode.ACCOUNTING_INVARIANT_FAILED)
        if self.found_candidates > self.raw_candidates:
            _fail(LmcacheLookupErrorCode.ACCOUNTING_INVARIANT_FAILED)
        if self.found_candidate_tokens > self.raw_candidate_tokens:
            _fail(LmcacheLookupErrorCode.ACCOUNTING_INVARIANT_FAILED)
        if self.verified_candidates > self.found_candidates:
            _fail(LmcacheLookupErrorCode.ACCOUNTING_INVARIANT_FAILED)
        if self.verified_candidate_tokens > self.found_candidate_tokens:
            _fail(LmcacheLookupErrorCode.ACCOUNTING_INVARIANT_FAILED)


@dataclass(frozen=True, slots=True)
class LmcacheLookupPlan:
    """Final independently verified values and complete rejection accounting."""

    verified_candidates: tuple[VerifiedLmcacheCandidate, ...]
    rejected_candidates: tuple[RejectedLmcacheCandidate, ...]
    counters: LmcacheLookupCounters


@dataclass(frozen=True, slots=True)
class _PotentialCandidate:
    raw_index: int
    verified: VerifiedLmcacheCandidate

    @property
    def candidate(self) -> LmcacheCandidate:
        return self.verified.candidate

    @property
    def target_segment(self) -> TokenSegment:
        return self.verified.match.target_segment


@dataclass(frozen=True, slots=True)
class _Solution:
    token_count: int
    potential_indexes: tuple[int, ...]
    tie_break_key: tuple[tuple[object, ...], ...]


def _candidate_sort_key(candidate: LmcacheCandidate) -> tuple[object, ...]:
    return (
        candidate.target_range.start,
        candidate.target_range.end,
        candidate.cache_key,
        candidate.source_relative_range.start,
        candidate.source_relative_range.end,
        candidate.storage_model_name,
        candidate.query_digest,
    )


def _potential_sort_key(potential: _PotentialCandidate) -> tuple[object, ...]:
    record = potential.verified.match.record
    return (
        *_candidate_sort_key(potential.candidate),
        record.source_range.start,
        record.source_range.end,
        potential.raw_index,
    )


def _record_sort_key(record: CacheRecord) -> tuple[object, ...]:
    return (
        record.cache_key,
        record.source_range.start,
        record.source_range.end,
        record.fingerprint.digest,
        record.token_ids,
        record.namespace.canonical_fields(),
    )


def _better_solution(left: _Solution, right: _Solution) -> _Solution:
    if left.token_count != right.token_count:
        return left if left.token_count > right.token_count else right
    return left if left.tie_break_key <= right.tie_break_key else right


def _select_non_overlapping(
    potentials: Sequence[_PotentialCandidate],
) -> tuple[int, ...]:
    """Use weighted interval scheduling to maximize exact token coverage."""

    ordered_indexes = sorted(
        range(len(potentials)),
        key=lambda index: (
            potentials[index].candidate.target_range.end,
            *_potential_sort_key(potentials[index]),
        ),
    )
    ordered = tuple(potentials[index] for index in ordered_indexes)
    solutions: list[_Solution] = [_Solution(0, (), ())]
    for ordered_index, potential in enumerate(ordered):
        target = potential.candidate.target_range
        predecessor = ordered_index - 1
        while (
            predecessor >= 0
            and ordered[predecessor].candidate.target_range.end > target.start
        ):
            predecessor -= 1
        base = solutions[predecessor + 1]
        original_index = ordered_indexes[ordered_index]
        include = _Solution(
            token_count=base.token_count + len(target),
            potential_indexes=(*base.potential_indexes, original_index),
            tie_break_key=(
                *base.tie_break_key,
                _potential_sort_key(potential),
            ),
        )
        solutions.append(_better_solution(include, solutions[-1]))
    return solutions[-1].potential_indexes


def _verify_record(
    namespace: CacheNamespace,
    target_segment: TokenSegment,
    query_fingerprint: SegmentFingerprint,
    record: CacheRecord,
) -> LmcacheCandidateRejectionReason | None:
    if record.namespace != namespace:
        return LmcacheCandidateRejectionReason.NAMESPACE_MISMATCH
    if record.fingerprint != query_fingerprint:
        return LmcacheCandidateRejectionReason.FINGERPRINT_MISMATCH
    if record.token_ids != target_segment.token_ids:
        return LmcacheCandidateRejectionReason.TOKEN_MISMATCH
    if (
        SHA256_FINGERPRINTER.fingerprint(record.namespace, record.token_ids)
        != record.fingerprint
    ):
        return LmcacheCandidateRejectionReason.FINGERPRINT_MISMATCH
    return None


class LmcacheCandidateLookupCoordinator:
    """Bind raw LMCache matches to independently verified sidecar records."""

    def __init__(
        self,
        record_lookup: RecordLookup,
    ) -> None:
        self._record_lookup = record_lookup

    def plan(
        self,
        prompt_token_ids: Iterable[int],
        namespace: CacheNamespace,
        candidates: Iterable[LmcacheCandidate],
    ) -> LmcacheLookupPlan:
        """Verify candidates and select deterministic maximum token coverage."""

        try:
            prompt = normalize_token_ids(prompt_token_ids)
        except (TypeError, ValueError) as exc:
            raise LmcacheLookupError(
                LmcacheLookupErrorCode.INVALID_PROMPT_TOKENS
            ) from exc
        try:
            raw_candidates = tuple(candidates)
        except TypeError as exc:
            raise LmcacheLookupError(
                LmcacheLookupErrorCode.INVALID_CANDIDATE
            ) from exc
        if any(
            not isinstance(candidate, LmcacheCandidate)
            for candidate in raw_candidates
        ):
            _fail(LmcacheLookupErrorCode.INVALID_CANDIDATE)

        expected_query_digest = query_digest(prompt)
        bucket_cache: dict[SegmentFingerprint, tuple[CacheRecord, ...]] = {}
        potentials: list[_PotentialCandidate] = []
        rejected: list[tuple[int, RejectedLmcacheCandidate]] = []
        found_raw_indexes: set[int] = set()

        for raw_index, candidate in enumerate(raw_candidates):
            if candidate.target_range.end > len(prompt):
                _fail(LmcacheLookupErrorCode.TARGET_RANGE_OUT_OF_BOUNDS)
            if candidate.query_digest != expected_query_digest:
                rejected.append(
                    (
                        raw_index,
                        RejectedLmcacheCandidate(
                            candidate,
                            LmcacheCandidateRejectionReason.STALE_QUERY,
                        ),
                    )
                )
                continue

            target = candidate.target_range
            target_segment = TokenSegment(
                target,
                prompt[target.start : target.end],
            )
            fingerprint = SHA256_FINGERPRINTER.fingerprint(
                namespace, target_segment.token_ids
            )
            bucket = bucket_cache.get(fingerprint)
            if bucket is None:
                bucket = self._lookup_bucket(namespace, fingerprint)
                bucket_cache[fingerprint] = bucket
            if not bucket:
                rejected.append(
                    (
                        raw_index,
                        RejectedLmcacheCandidate(
                            candidate,
                            LmcacheCandidateRejectionReason.NO_SIDECAR_RECORD,
                        ),
                    )
                )
                continue

            bound_records = tuple(
                record for record in bucket if record.cache_key == candidate.cache_key
            )
            if not bound_records:
                rejected.append(
                    (
                        raw_index,
                        RejectedLmcacheCandidate(
                            candidate,
                            LmcacheCandidateRejectionReason.CACHE_KEY_MISMATCH,
                        ),
                    )
                )
                continue
            found_raw_indexes.add(raw_index)

            valid_records: list[CacheRecord] = []
            invalid_reasons: list[LmcacheCandidateRejectionReason] = []
            for record in bound_records:
                reason = _verify_record(
                    namespace,
                    target_segment,
                    fingerprint,
                    record,
                )
                if reason is None:
                    valid_records.append(record)
                else:
                    invalid_reasons.append(reason)
            if not valid_records:
                reason = min(
                    invalid_reasons,
                    key=_VERIFICATION_REASON_PRIORITY.__getitem__,
                )
                rejected.append(
                    (raw_index, RejectedLmcacheCandidate(candidate, reason))
                )
                continue

            record = min(valid_records, key=_record_sort_key)
            match = VerifiedMatch(
                CandidateMatch(
                    target_segment=target_segment,
                    query_fingerprint=fingerprint,
                    record=record,
                )
            )
            try:
                verified = VerifiedLmcacheCandidate.bind(
                    candidate,
                    match,
                    expected_namespace=namespace,
                )
            except LmcacheProtocolError as exc:
                raise LmcacheLookupError(
                    LmcacheLookupErrorCode.BINDING_INVARIANT_FAILED
                ) from exc
            potentials.append(_PotentialCandidate(raw_index, verified))

        selected_potential_indexes = set(_select_non_overlapping(potentials))
        selected = tuple(
            sorted(
                (
                    potential.verified
                    for index, potential in enumerate(potentials)
                    if index in selected_potential_indexes
                ),
                key=lambda verified: (
                    *_candidate_sort_key(verified.candidate),
                    verified.match.record.source_range.start,
                    verified.match.record.source_range.end,
                ),
            )
        )
        selected_candidates = tuple(item.candidate for item in selected)
        for index, potential in enumerate(potentials):
            if index in selected_potential_indexes:
                continue
            reason = (
                LmcacheCandidateRejectionReason.DUPLICATE_CANDIDATE
                if potential.candidate in selected_candidates
                else LmcacheCandidateRejectionReason.OVERLAPS_SELECTED_CANDIDATE
            )
            rejected.append(
                (
                    potential.raw_index,
                    RejectedLmcacheCandidate(potential.candidate, reason),
                )
            )

        ordered_rejections = tuple(
            item
            for _, item in sorted(
                rejected,
                key=lambda indexed: (
                    *_candidate_sort_key(indexed[1].candidate),
                    indexed[1].reason.value,
                    indexed[0],
                ),
            )
        )
        verified_tokens = sum(
            len(verified.candidate.target_range) for verified in selected
        )
        rejected_tokens = sum(
            len(rejection.candidate.target_range)
            for rejection in ordered_rejections
        )
        counters = LmcacheLookupCounters(
            raw_candidates=len(raw_candidates),
            raw_candidate_tokens=sum(
                len(candidate.target_range) for candidate in raw_candidates
            ),
            found_candidates=len(found_raw_indexes),
            found_candidate_tokens=sum(
                len(raw_candidates[index].target_range)
                for index in found_raw_indexes
            ),
            verified_candidates=len(selected),
            verified_candidate_tokens=verified_tokens,
            rejected_candidates=len(ordered_rejections),
            rejected_candidate_tokens=rejected_tokens,
        )
        return LmcacheLookupPlan(selected, ordered_rejections, counters)

    def _lookup_bucket(
        self,
        namespace: CacheNamespace,
        fingerprint: SegmentFingerprint,
    ) -> tuple[CacheRecord, ...]:
        try:
            response = self._record_lookup.lookup(namespace, fingerprint)
            bucket = tuple(response)
        except Exception as exc:
            raise LmcacheLookupError(
                LmcacheLookupErrorCode.SIDECAR_LOOKUP_FAILED
            ) from exc
        if any(not isinstance(record, CacheRecord) for record in bucket):
            _fail(LmcacheLookupErrorCode.INVALID_SIDECAR_RESPONSE)
        return tuple(sorted(bucket, key=_record_sort_key))


__all__ = [
    "LmcacheCandidateLookupCoordinator",
    "LmcacheCandidateRejectionReason",
    "LmcacheLookupCounters",
    "LmcacheLookupError",
    "LmcacheLookupErrorCode",
    "LmcacheLookupPlan",
    "RejectedLmcacheCandidate",
]
