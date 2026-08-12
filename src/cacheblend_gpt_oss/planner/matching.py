"""Exact-token verified cache-match planning."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from cacheblend_gpt_oss.planner.fingerprint import (
    SHA256_FINGERPRINTER,
    SegmentFingerprinter,
)
from cacheblend_gpt_oss.planner.models import (
    CacheNamespace,
    CacheRecord,
    CandidateMatch,
    SegmentFingerprint,
    TokenSegment,
)


class RecordLookup(Protocol):
    """Read-only lookup interface supplied by the storage/transport layer."""

    def lookup(
        self, namespace: CacheNamespace, fingerprint: SegmentFingerprint
    ) -> Sequence[CacheRecord]:
        """Return untrusted records in the requested digest bucket."""


class CandidateRejectionReason(str, Enum):
    """Fail-closed reasons exposed to bounded metrics."""

    NAMESPACE_MISMATCH = "namespace_mismatch"
    FINGERPRINT_MISMATCH = "fingerprint_mismatch"
    TOKEN_MISMATCH = "token_mismatch"
    OVERLAPS_SELECTED_MATCH = "overlaps_selected_match"


@dataclass(frozen=True, slots=True)
class RejectedCandidate:
    candidate: CandidateMatch
    reason: CandidateRejectionReason


@dataclass(frozen=True, slots=True)
class VerifiedMatch:
    """A candidate that passed namespace, digest, and exact-token checks."""

    candidate: CandidateMatch

    @property
    def target_segment(self) -> TokenSegment:
        return self.candidate.target_segment

    @property
    def record(self) -> CacheRecord:
        return self.candidate.record

    @property
    def position_delta(self) -> int:
        return self.candidate.position_delta


@dataclass(frozen=True, slots=True)
class MatchPlan:
    """A deterministic, non-overlapping set of verified cache matches."""

    matches: tuple[VerifiedMatch, ...]
    rejected_candidates: tuple[RejectedCandidate, ...]
    requested_tokens: int

    @property
    def matched_tokens(self) -> int:
        return sum(len(match.target_segment) for match in self.matches)

    @property
    def unmatched_tokens(self) -> int:
        return self.requested_tokens - self.matched_tokens

    @property
    def token_hit_fraction(self) -> float:
        if self.requested_tokens == 0:
            return 0.0
        return self.matched_tokens / self.requested_tokens


def build_cache_record(
    namespace: CacheNamespace,
    segment: TokenSegment,
    cache_key: str,
    *,
    fingerprinter: SegmentFingerprinter = SHA256_FINGERPRINTER,
) -> CacheRecord:
    """Build a record whose digest is derived from its exact token tuple."""

    return CacheRecord(
        namespace=namespace,
        fingerprint=fingerprinter.fingerprint(namespace, segment.token_ids),
        token_ids=segment.token_ids,
        source_range=segment.token_range,
        cache_key=cache_key,
    )


class InMemoryRecordIndex:
    """Small deterministic index suitable for CPU tests and planner prototypes."""

    def __init__(self, records: Iterable[CacheRecord] = ()) -> None:
        self._records: dict[
            tuple[CacheNamespace, SegmentFingerprint], list[CacheRecord]
        ] = {}
        for record in records:
            self.add(record)

    def add(self, record: CacheRecord) -> None:
        key = (record.namespace, record.fingerprint)
        self._records.setdefault(key, []).append(record)

    def lookup(
        self, namespace: CacheNamespace, fingerprint: SegmentFingerprint
    ) -> Sequence[CacheRecord]:
        records = self._records.get((namespace, fingerprint), ())
        return tuple(sorted(records, key=_record_sort_key))


def _record_sort_key(record: CacheRecord) -> tuple[str, int, int]:
    return (record.cache_key, record.source_range.start, record.source_range.end)


def _candidate_sort_key(
    candidate: CandidateMatch,
) -> tuple[int, int, str, int, int]:
    target = candidate.target_segment.token_range
    source = candidate.record.source_range
    return (
        target.start,
        target.end,
        candidate.record.cache_key,
        source.start,
        source.end,
    )


def _covered_token_count(segments: Iterable[TokenSegment]) -> int:
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


@dataclass(frozen=True, slots=True)
class _Solution:
    token_count: int
    candidate_indexes: tuple[int, ...]
    tie_break_key: tuple[tuple[int, int, str, int, int], ...]


def _better_solution(left: _Solution, right: _Solution) -> _Solution:
    if left.token_count != right.token_count:
        return left if left.token_count > right.token_count else right
    return left if left.tie_break_key <= right.tie_break_key else right


def _select_non_overlapping(
    candidates: Sequence[CandidateMatch],
) -> tuple[int, ...]:
    """Use weighted interval scheduling to maximize verified token coverage."""

    ordered_indexes = sorted(
        range(len(candidates)),
        key=lambda index: (
            candidates[index].target_segment.token_range.end,
            *_candidate_sort_key(candidates[index]),
        ),
    )
    ordered = [candidates[index] for index in ordered_indexes]
    solutions: list[_Solution] = [_Solution(0, (), ())]

    for ordered_index, candidate in enumerate(ordered):
        target = candidate.target_segment.token_range
        predecessor = ordered_index - 1
        while (
            predecessor >= 0
            and ordered[predecessor].target_segment.token_range.end > target.start
        ):
            predecessor -= 1
        base = solutions[predecessor + 1]
        original_index = ordered_indexes[ordered_index]
        include = _Solution(
            token_count=base.token_count + len(candidate.target_segment),
            candidate_indexes=(*base.candidate_indexes, original_index),
            tie_break_key=(*base.tie_break_key, _candidate_sort_key(candidate)),
        )
        exclude = solutions[-1]
        solutions.append(_better_solution(include, exclude))
    return solutions[-1].candidate_indexes


class MatchPlanner:
    """Resolve rolling query candidates into verified, non-overlapping hits."""

    def __init__(
        self,
        lookup: RecordLookup,
        *,
        fingerprinter: SegmentFingerprinter = SHA256_FINGERPRINTER,
    ) -> None:
        self._lookup = lookup
        self._fingerprinter = fingerprinter

    def plan(
        self,
        namespace: CacheNamespace,
        query_segments: Iterable[TokenSegment],
    ) -> MatchPlan:
        queries = tuple(query_segments)
        verified: list[CandidateMatch] = []
        rejected: list[RejectedCandidate] = []

        for query in queries:
            query_fingerprint = self._fingerprinter.fingerprint(
                namespace, query.token_ids
            )
            for record in self._lookup.lookup(namespace, query_fingerprint):
                candidate = CandidateMatch(query, query_fingerprint, record)
                rejection_reason = self._verify_candidate(namespace, candidate)
                if rejection_reason is None:
                    verified.append(candidate)
                else:
                    rejected.append(RejectedCandidate(candidate, rejection_reason))

        selected_indexes = set(_select_non_overlapping(verified))
        selected = tuple(
            VerifiedMatch(candidate)
            for index, candidate in enumerate(verified)
            if index in selected_indexes
        )
        for index, candidate in enumerate(verified):
            if index not in selected_indexes:
                rejected.append(
                    RejectedCandidate(
                        candidate,
                        CandidateRejectionReason.OVERLAPS_SELECTED_MATCH,
                    )
                )

        return MatchPlan(
            matches=tuple(
                sorted(
                    selected,
                    key=lambda item: _candidate_sort_key(item.candidate),
                )
            ),
            rejected_candidates=tuple(rejected),
            requested_tokens=_covered_token_count(queries),
        )

    def _verify_candidate(
        self, namespace: CacheNamespace, candidate: CandidateMatch
    ) -> CandidateRejectionReason | None:
        record = candidate.record
        if record.namespace != namespace:
            return CandidateRejectionReason.NAMESPACE_MISMATCH
        if record.fingerprint != candidate.query_fingerprint:
            return CandidateRejectionReason.FINGERPRINT_MISMATCH
        # This exact comparison remains mandatory even though SHA-256 collisions
        # are impractical; lookup digests are candidate indexes, not proof.
        if record.token_ids != candidate.target_segment.token_ids:
            return CandidateRejectionReason.TOKEN_MISMATCH
        if (
            self._fingerprinter.fingerprint(record.namespace, record.token_ids)
            != record.fingerprint
        ):
            return CandidateRejectionReason.FINGERPRINT_MISMATCH
        return None
