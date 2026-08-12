from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace

from cacheblend_gpt_oss.planner import (
    CacheNamespace,
    CacheRecord,
    CandidateRejectionReason,
    DelimiterSegmenter,
    InMemoryRecordIndex,
    MatchPlanner,
    RollingQuerySegmenter,
    SegmentFingerprint,
    TokenSegment,
    build_cache_record,
)


def namespace() -> CacheNamespace:
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


def test_moved_document_is_found_with_position_delta() -> None:
    document = TokenSegment.at(100, [11, 12, 13, 14])
    record = build_cache_record(namespace(), document, "document-a")
    query_region = TokenSegment.at(0, [7, 8, 11, 12, 13, 14, 9, 10])
    queries = RollingQuerySegmenter([len(document)]).segment([query_region])

    plan = MatchPlanner(InMemoryRecordIndex([record])).plan(namespace(), queries)

    assert len(plan.matches) == 1
    assert plan.matches[0].target_segment.token_range.start == 2
    assert plan.matches[0].position_delta == -98
    assert plan.matched_tokens == 4
    assert plan.requested_tokens == 8


def test_reordered_documents_and_duplicate_document_are_each_matched() -> None:
    document_a = TokenSegment.at(10, [1, 2, 3])
    document_b = TokenSegment.at(30, [4, 5, 6])
    records = [
        build_cache_record(namespace(), document_a, "document-a"),
        build_cache_record(namespace(), document_b, "document-b"),
    ]
    # B, A, A: the same immutable cache record may serve both A occurrences.
    query_documents = DelimiterSegmenter([[0]]).segment(
        [4, 5, 6, 0, 1, 2, 3, 0, 1, 2, 3]
    )

    plan = MatchPlanner(InMemoryRecordIndex(records)).plan(
        namespace(), query_documents
    )

    assert [match.record.cache_key for match in plan.matches] == [
        "document-b",
        "document-a",
        "document-a",
    ]
    assert [match.target_segment.token_range.start for match in plan.matches] == [
        0,
        4,
        8,
    ]
    assert plan.matched_tokens == 9
    assert plan.unmatched_tokens == 0


def test_cache_miss_returns_no_verified_match() -> None:
    cached = build_cache_record(
        namespace(), TokenSegment.at(0, [1, 2, 3]), "document-a"
    )

    plan = MatchPlanner(InMemoryRecordIndex([cached])).plan(
        namespace(), [TokenSegment.at(50, [8, 9, 10])]
    )

    assert plan.matches == ()
    assert plan.matched_tokens == 0
    assert plan.unmatched_tokens == 3


class ConstantFingerprinter:
    """Adversarial digest implementation used to exercise collision handling."""

    _fingerprint = SegmentFingerprint(b"\x00" * 32)

    def fingerprint(
        self, namespace: CacheNamespace, token_ids: Iterable[int]
    ) -> SegmentFingerprint:
        del namespace, token_ids
        return self._fingerprint


def test_digest_collision_remains_an_untrusted_candidate() -> None:
    fingerprinter = ConstantFingerprinter()
    correct = build_cache_record(
        namespace(),
        TokenSegment.at(0, [1, 2, 3]),
        "correct",
        fingerprinter=fingerprinter,
    )
    colliding = build_cache_record(
        namespace(),
        TokenSegment.at(10, [7, 8, 9]),
        "colliding",
        fingerprinter=fingerprinter,
    )

    plan = MatchPlanner(
        InMemoryRecordIndex([colliding, correct]),
        fingerprinter=fingerprinter,
    ).plan(namespace(), [TokenSegment.at(100, [1, 2, 3])])

    assert [match.record.cache_key for match in plan.matches] == ["correct"]
    assert [item.reason for item in plan.rejected_candidates] == [
        CandidateRejectionReason.TOKEN_MISMATCH
    ]


class UntrustedLookup:
    def __init__(self, records: Sequence[CacheRecord]) -> None:
        self.records = records

    def lookup(
        self, namespace: CacheNamespace, fingerprint: SegmentFingerprint
    ) -> Sequence[CacheRecord]:
        del namespace, fingerprint
        return self.records


def test_cross_namespace_record_is_rejected_even_if_lookup_returns_it() -> None:
    other_namespace = replace(namespace(), model_revision="incompatible")
    other_record = build_cache_record(
        other_namespace,
        TokenSegment.at(0, [1, 2, 3]),
        "other-namespace",
    )

    plan = MatchPlanner(UntrustedLookup([other_record])).plan(
        namespace(), [TokenSegment.at(100, [1, 2, 3])]
    )

    assert plan.matches == ()
    assert plan.rejected_candidates[0].reason is (
        CandidateRejectionReason.NAMESPACE_MISMATCH
    )

