"""Position-independent matching and recomputation planning boundary."""

from cacheblend_gpt_oss.planner.fingerprint import (
    SHA256_FINGERPRINTER,
    SegmentFingerprinter,
    Sha256SegmentFingerprinter,
    canonical_token_bytes,
    fingerprint_segment,
)
from cacheblend_gpt_oss.planner.matching import (
    CandidateRejectionReason,
    InMemoryRecordIndex,
    MatchPlan,
    MatchPlanner,
    RecordLookup,
    RejectedCandidate,
    VerifiedMatch,
    build_cache_record,
)
from cacheblend_gpt_oss.planner.models import (
    CacheNamespace,
    CacheRecord,
    CandidateMatch,
    SegmentFingerprint,
    TokenRange,
    TokenSegment,
)
from cacheblend_gpt_oss.planner.segmentation import (
    DelimiterSegmenter,
    FixedChunkStorageSegmenter,
    RollingQuerySegmenter,
)

__all__ = [
    "SHA256_FINGERPRINTER",
    "CacheNamespace",
    "CacheRecord",
    "CandidateMatch",
    "CandidateRejectionReason",
    "DelimiterSegmenter",
    "FixedChunkStorageSegmenter",
    "InMemoryRecordIndex",
    "MatchPlan",
    "MatchPlanner",
    "RecordLookup",
    "RejectedCandidate",
    "RollingQuerySegmenter",
    "SegmentFingerprint",
    "SegmentFingerprinter",
    "Sha256SegmentFingerprinter",
    "TokenRange",
    "TokenSegment",
    "VerifiedMatch",
    "build_cache_record",
    "canonical_token_bytes",
    "fingerprint_segment",
]
