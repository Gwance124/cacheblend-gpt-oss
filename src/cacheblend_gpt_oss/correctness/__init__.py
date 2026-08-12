"""Pinned GPT-OSS correctness artifacts and numerical gates."""

from cacheblend_gpt_oss.correctness.capture import (
    connector_counter_delta,
    connector_evidence_from_snapshots,
    has_connector_metric_surface,
    parse_completion_distribution,
    parse_connector_counter_snapshot,
)
from cacheblend_gpt_oss.correctness.evaluate import (
    CacheBlendCorrectnessVerdict,
    DistributionComparison,
    FrozenFullPrefillTolerance,
    compare_distributions,
    evaluate_cacheblend_100pct,
    freeze_full_prefill_tolerance,
)
from cacheblend_gpt_oss.correctness.fixture import (
    MovedDocumentFixture,
    build_moved_document_fixture,
    digest_token_ids,
)
from cacheblend_gpt_oss.correctness.io import (
    artifact_digest,
    artifact_from_dict,
    artifact_to_dict,
    read_artifact,
    write_artifact,
)
from cacheblend_gpt_oss.correctness.models import (
    ARTIFACT_SCHEMA_VERSION,
    GPT_OSS_VOCAB_SIZE,
    ConnectorCorrectnessEvidence,
    CorrectnessArtifact,
    CorrectnessCase,
    CorrectnessRunMode,
    CorrectnessRuntimeIdentity,
    FullVocabularyLogprobs,
    PromptCaseIdentity,
)
from cacheblend_gpt_oss.correctness.tolerance_io import (
    TOLERANCE_SCHEMA_VERSION,
    read_frozen_tolerance,
    tolerance_from_dict,
    tolerance_to_dict,
    write_frozen_tolerance,
)

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "GPT_OSS_VOCAB_SIZE",
    "TOLERANCE_SCHEMA_VERSION",
    "CacheBlendCorrectnessVerdict",
    "ConnectorCorrectnessEvidence",
    "CorrectnessArtifact",
    "CorrectnessCase",
    "CorrectnessRunMode",
    "CorrectnessRuntimeIdentity",
    "DistributionComparison",
    "FrozenFullPrefillTolerance",
    "FullVocabularyLogprobs",
    "MovedDocumentFixture",
    "PromptCaseIdentity",
    "artifact_digest",
    "artifact_from_dict",
    "artifact_to_dict",
    "build_moved_document_fixture",
    "compare_distributions",
    "connector_counter_delta",
    "connector_evidence_from_snapshots",
    "digest_token_ids",
    "evaluate_cacheblend_100pct",
    "freeze_full_prefill_tolerance",
    "has_connector_metric_surface",
    "parse_completion_distribution",
    "parse_connector_counter_snapshot",
    "read_artifact",
    "read_frozen_tolerance",
    "tolerance_from_dict",
    "tolerance_to_dict",
    "write_artifact",
    "write_frozen_tolerance",
]
