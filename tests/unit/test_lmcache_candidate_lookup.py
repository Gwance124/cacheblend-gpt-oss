from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

import pytest

from cacheblend_gpt_oss.planner import (
    SHA256_FINGERPRINTER,
    CacheNamespace,
    CacheRecord,
    InMemoryRecordIndex,
    SegmentFingerprint,
    TokenRange,
    TokenSegment,
    build_cache_record,
)
from cacheblend_gpt_oss.storage.lmcache_types import (
    LmcacheCandidate,
    query_digest,
)
from cacheblend_gpt_oss.storage.lookup import (
    LmcacheCandidateLookupCoordinator,
    LmcacheCandidateRejectionReason,
    LmcacheLookupCounters,
    LmcacheLookupError,
    LmcacheLookupErrorCode,
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


def _candidate(
    prompt: tuple[int, ...],
    start: int,
    end: int,
    storage_byte: bytes,
    *,
    digest: bytes | None = None,
) -> LmcacheCandidate:
    return LmcacheCandidate(
        source_relative_range=TokenRange(0, end - start),
        target_range=TokenRange(start, end),
        storage_hash=storage_byte * 32,
        storage_model_name="pinned-storage-namespace",
        query_digest=query_digest(prompt) if digest is None else digest,
    )


def _record_for(
    prompt: tuple[int, ...],
    candidate: LmcacheCandidate,
    *,
    source_start: int,
    namespace: CacheNamespace | None = None,
) -> CacheRecord:
    target = candidate.target_range
    return build_cache_record(
        _namespace() if namespace is None else namespace,
        TokenSegment.at(source_start, prompt[target.start : target.end]),
        candidate.cache_key,
    )


def _assert_error(
    expected: LmcacheLookupErrorCode,
    operation: Callable[[], object],
) -> LmcacheLookupError:
    with pytest.raises(LmcacheLookupError) as caught:
        operation()
    assert caught.value.code is expected
    assert str(caught.value).endswith(expected.value)
    return caught.value


class UntrustedLookup:
    def __init__(self, records: Sequence[object] = ()) -> None:
        self.records = records
        self.calls: list[tuple[CacheNamespace, SegmentFingerprint]] = []
        self.error: Exception | None = None

    def lookup(
        self,
        namespace: CacheNamespace,
        fingerprint: SegmentFingerprint,
    ) -> Sequence[CacheRecord]:
        self.calls.append((namespace, fingerprint))
        if self.error is not None:
            raise self.error
        return self.records  # type: ignore[return-value]


def test_moved_candidate_binds_exact_record_and_preserves_identity() -> None:
    prompt = (90, 91, 11, 12, 13, 92)
    candidate = _candidate(prompt, 2, 5, b"a")
    record = _record_for(prompt, candidate, source_start=100)

    plan = LmcacheCandidateLookupCoordinator(
        InMemoryRecordIndex((record,))
    ).plan(prompt, _namespace(), (candidate,))

    assert len(plan.verified_candidates) == 1
    verified = plan.verified_candidates[0]
    assert verified.candidate is candidate
    assert verified.match.record is record
    assert verified.match.target_segment == TokenSegment.at(2, (11, 12, 13))
    assert verified.match.position_delta == -98
    assert plan.rejected_candidates == ()
    assert plan.counters == LmcacheLookupCounters(
        raw_candidates=1,
        raw_candidate_tokens=3,
        found_candidates=1,
        found_candidate_tokens=3,
        verified_candidates=1,
        verified_candidate_tokens=3,
        rejected_candidates=0,
        rejected_candidate_tokens=0,
    )


def test_no_sidecar_record_is_a_counted_miss() -> None:
    prompt = (1, 2, 3)
    candidate = _candidate(prompt, 0, 3, b"n")

    plan = LmcacheCandidateLookupCoordinator(InMemoryRecordIndex()).plan(
        prompt, _namespace(), (candidate,)
    )

    assert plan.verified_candidates == ()
    assert plan.rejected_candidates[0].candidate is candidate
    assert plan.rejected_candidates[0].reason is (
        LmcacheCandidateRejectionReason.NO_SIDECAR_RECORD
    )
    assert plan.counters.raw_candidates == 1
    assert plan.counters.found_candidates == 0
    assert plan.counters.verified_candidates == 0
    assert plan.counters.rejected_candidates == 1
    assert plan.counters.raw_candidate_tokens == 3
    assert plan.counters.rejected_candidate_tokens == 3


def test_sha_bucket_collision_cannot_substitute_a_different_cache_key() -> None:
    prompt = (1, 2, 3)
    candidate = _candidate(prompt, 0, 3, b"a")
    wrong_key = replace(
        _record_for(prompt, candidate, source_start=10),
        cache_key="different-lmcache-object",
    )
    lookup = UntrustedLookup((wrong_key,))

    plan = LmcacheCandidateLookupCoordinator(lookup).plan(
        prompt, _namespace(), (candidate,)
    )

    assert plan.verified_candidates == ()
    assert plan.rejected_candidates[0].reason is (
        LmcacheCandidateRejectionReason.CACHE_KEY_MISMATCH
    )
    assert plan.counters.found_candidates == 0


def test_collision_bucket_selects_deterministic_exact_record_only() -> None:
    prompt = (4, 5, 6)
    candidate = _candidate(prompt, 0, 3, b"c")
    fingerprint = SHA256_FINGERPRINTER.fingerprint(_namespace(), prompt)
    stale_tokens = CacheRecord(
        namespace=_namespace(),
        fingerprint=fingerprint,
        token_ids=(9, 9, 9),
        source_range=TokenRange(50, 53),
        cache_key=candidate.cache_key,
    )
    later_exact = _record_for(prompt, candidate, source_start=100)
    earlier_exact = _record_for(prompt, candidate, source_start=20)
    lookup = UntrustedLookup((later_exact, stale_tokens, earlier_exact))

    plan = LmcacheCandidateLookupCoordinator(lookup).plan(
        prompt, _namespace(), (candidate,)
    )

    assert plan.verified_candidates[0].match.record is earlier_exact
    assert plan.counters.found_candidates == 1
    assert plan.counters.verified_candidates == 1
    assert plan.counters.rejected_candidates == 0


@pytest.mark.parametrize(
    ("record_factory", "reason"),
    [
        (
            lambda prompt, candidate: CacheRecord(
                namespace=replace(_namespace(), model_revision="stale"),
                fingerprint=SHA256_FINGERPRINTER.fingerprint(
                    _namespace(), prompt
                ),
                token_ids=prompt,
                source_range=TokenRange(10, 13),
                cache_key=candidate.cache_key,
            ),
            LmcacheCandidateRejectionReason.NAMESPACE_MISMATCH,
        ),
        (
            lambda prompt, candidate: replace(
                _record_for(prompt, candidate, source_start=10),
                fingerprint=SegmentFingerprint(b"f" * 32),
            ),
            LmcacheCandidateRejectionReason.FINGERPRINT_MISMATCH,
        ),
        (
            lambda prompt, candidate: CacheRecord(
                namespace=_namespace(),
                fingerprint=SHA256_FINGERPRINTER.fingerprint(
                    _namespace(), prompt
                ),
                token_ids=(7, 8, 9),
                source_range=TokenRange(10, 13),
                cache_key=candidate.cache_key,
            ),
            LmcacheCandidateRejectionReason.TOKEN_MISMATCH,
        ),
    ],
)
def test_untrusted_sidecar_records_are_independently_rechecked(
    record_factory: Callable[
        [tuple[int, ...], LmcacheCandidate], CacheRecord
    ],
    reason: LmcacheCandidateRejectionReason,
) -> None:
    prompt = (1, 2, 3)
    candidate = _candidate(prompt, 0, 3, b"r")
    lookup = UntrustedLookup((record_factory(prompt, candidate),))

    plan = LmcacheCandidateLookupCoordinator(lookup).plan(
        prompt, _namespace(), (candidate,)
    )

    assert plan.verified_candidates == ()
    assert plan.rejected_candidates[0].reason is reason
    assert plan.counters.found_candidates == 1
    assert plan.counters.found_candidate_tokens == 3
    assert plan.counters.rejected_candidate_tokens == 3


def test_candidate_from_stale_full_query_is_rejected_without_sidecar_access() -> (
    None
):
    prompt = (1, 2, 3)
    candidate = _candidate(
        prompt,
        0,
        3,
        b"s",
        digest=query_digest((1, 2, 99)),
    )
    lookup = UntrustedLookup()

    plan = LmcacheCandidateLookupCoordinator(lookup).plan(
        prompt, _namespace(), (candidate,)
    )

    assert lookup.calls == []
    assert plan.rejected_candidates[0].reason is (
        LmcacheCandidateRejectionReason.STALE_QUERY
    )
    assert plan.counters.found_candidates == 0


def test_duplicate_candidates_are_memoized_counted_and_deduplicated() -> None:
    prompt = (1, 2, 3)
    candidate = _candidate(prompt, 0, 3, b"d")
    record = _record_for(prompt, candidate, source_start=10)
    lookup = UntrustedLookup((record,))

    plan = LmcacheCandidateLookupCoordinator(lookup).plan(
        prompt, _namespace(), (candidate, candidate)
    )

    assert len(lookup.calls) == 1
    assert [item.candidate for item in plan.verified_candidates] == [candidate]
    assert len(plan.rejected_candidates) == 1
    assert plan.rejected_candidates[0].candidate is candidate
    assert plan.rejected_candidates[0].reason is (
        LmcacheCandidateRejectionReason.DUPLICATE_CANDIDATE
    )
    assert plan.counters == LmcacheLookupCounters(
        raw_candidates=2,
        raw_candidate_tokens=6,
        found_candidates=2,
        found_candidate_tokens=6,
        verified_candidates=1,
        verified_candidate_tokens=3,
        rejected_candidates=1,
        rejected_candidate_tokens=3,
    )


def test_maximum_non_overlap_prefers_greater_total_token_coverage() -> None:
    prompt = (10, 11, 12, 13, 14)
    long = _candidate(prompt, 0, 4, b"l")
    left = _candidate(prompt, 0, 2, b"a")
    right = _candidate(prompt, 2, 5, b"b")
    records = tuple(
        _record_for(prompt, candidate, source_start=100 + index * 10)
        for index, candidate in enumerate((long, left, right))
    )

    plan = LmcacheCandidateLookupCoordinator(
        InMemoryRecordIndex(records)
    ).plan(prompt, _namespace(), (long, right, left))

    assert [
        item.candidate.target_range for item in plan.verified_candidates
    ] == [TokenRange(0, 2), TokenRange(2, 5)]
    assert len(plan.rejected_candidates) == 1
    assert plan.rejected_candidates[0].candidate is long
    assert plan.rejected_candidates[0].reason is (
        LmcacheCandidateRejectionReason.OVERLAPS_SELECTED_CANDIDATE
    )
    assert plan.counters.raw_candidates == 3
    assert plan.counters.raw_candidate_tokens == 9
    assert plan.counters.found_candidates == 3
    assert plan.counters.found_candidate_tokens == 9
    assert plan.counters.verified_candidates == 2
    assert plan.counters.verified_candidate_tokens == 5
    assert plan.counters.rejected_candidates == 1
    assert plan.counters.rejected_candidate_tokens == 4


def test_equal_coverage_tie_break_is_independent_of_input_order() -> None:
    prompt = (1, 2, 3)
    candidate_b = _candidate(prompt, 0, 3, b"b")
    candidate_a = _candidate(prompt, 0, 3, b"a")
    records = (
        _record_for(prompt, candidate_b, source_start=20),
        _record_for(prompt, candidate_a, source_start=10),
    )
    coordinator = LmcacheCandidateLookupCoordinator(InMemoryRecordIndex(records))

    forward = coordinator.plan(
        prompt, _namespace(), (candidate_b, candidate_a)
    )
    reverse = coordinator.plan(
        prompt, _namespace(), (candidate_a, candidate_b)
    )

    assert forward.verified_candidates[0].candidate is candidate_a
    assert reverse.verified_candidates[0].candidate is candidate_a


def test_invalid_prompt_candidate_range_and_sidecar_response_fail_bounded() -> None:
    prompt = (1, 2, 3)
    candidate = _candidate(prompt, 0, 3, b"e")
    coordinator = LmcacheCandidateLookupCoordinator(UntrustedLookup())
    _assert_error(
        LmcacheLookupErrorCode.INVALID_PROMPT_TOKENS,
        lambda: coordinator.plan((1, True), _namespace(), (candidate,)),
    )
    _assert_error(
        LmcacheLookupErrorCode.INVALID_CANDIDATE,
        lambda: coordinator.plan(
            prompt,
            _namespace(),
            (object(),),  # type: ignore[arg-type]
        ),
    )

    out_of_bounds = LmcacheCandidate(
        source_relative_range=TokenRange(0, 4),
        target_range=TokenRange(0, 4),
        storage_hash=b"o" * 32,
        storage_model_name="pinned-storage-namespace",
        query_digest=query_digest(prompt),
    )
    _assert_error(
        LmcacheLookupErrorCode.TARGET_RANGE_OUT_OF_BOUNDS,
        lambda: coordinator.plan(prompt, _namespace(), (out_of_bounds,)),
    )

    invalid_response = UntrustedLookup((object(),))
    _assert_error(
        LmcacheLookupErrorCode.INVALID_SIDECAR_RESPONSE,
        lambda: LmcacheCandidateLookupCoordinator(invalid_response).plan(
            prompt, _namespace(), (candidate,)
        ),
    )


def test_sidecar_failure_is_not_silently_converted_to_a_miss() -> None:
    prompt = (1, 2, 3)
    candidate = _candidate(prompt, 0, 3, b"x")
    lookup = UntrustedLookup()
    lookup.error = RuntimeError("sensitive sidecar detail")

    error = _assert_error(
        LmcacheLookupErrorCode.SIDECAR_LOOKUP_FAILED,
        lambda: LmcacheCandidateLookupCoordinator(lookup).plan(
            prompt, _namespace(), (candidate,)
        ),
    )

    assert "sensitive sidecar detail" not in str(error)


def test_empty_candidate_input_has_zero_reconciled_counters() -> None:
    plan = LmcacheCandidateLookupCoordinator(InMemoryRecordIndex()).plan(
        (), _namespace(), ()
    )
    assert plan.verified_candidates == ()
    assert plan.rejected_candidates == ()
    assert plan.counters == LmcacheLookupCounters(0, 0, 0, 0, 0, 0, 0, 0)


def test_counter_constructor_rejects_nonreconciling_values() -> None:
    _assert_error(
        LmcacheLookupErrorCode.ACCOUNTING_INVARIANT_FAILED,
        lambda: LmcacheLookupCounters(
            raw_candidates=1,
            raw_candidate_tokens=3,
            found_candidates=1,
            found_candidate_tokens=3,
            verified_candidates=1,
            verified_candidate_tokens=3,
            rejected_candidates=1,
            rejected_candidate_tokens=3,
        ),
    )
