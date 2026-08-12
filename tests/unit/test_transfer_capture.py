"""CPU-only tests for incremental transfer-evidence assembly."""

from __future__ import annotations

from dataclasses import replace

import pytest

from cacheblend_gpt_oss.correctness import (
    TRANSFER_EVIDENCE_SCHEMA_VERSION,
    LayerTransferEvidence,
    TransferCaptureError,
    TransferCaptureErrorCode,
    TransferEvidenceBuilder,
    TransferEvidenceCaptureMetadata,
)
from cacheblend_gpt_oss.gpt_oss.layout import AttentionKind


def _digest(index: int) -> str:
    return f"{index:064x}"


def _layer(index: int) -> LayerTransferEvidence:
    base = index * 20
    return LayerTransferEvidence(
        layer_index=index,
        attention_kind=(
            AttentionKind.SLIDING if index % 2 == 0 else AttentionKind.FULL
        ),
        token_count=256,
        key_before_digest=_digest(base + 1),
        key_source_digest=_digest(base + 2),
        key_after_load_digest=_digest(base + 2),
        key_target_prefill_digest=_digest(base + 3),
        key_after_prefill_digest=_digest(base + 3),
        value_before_digest=_digest(base + 4),
        value_source_digest=_digest(base + 5),
        value_after_load_digest=_digest(base + 5),
        value_target_prefill_digest=_digest(base + 6),
        value_after_prefill_digest=_digest(base + 6),
    )


def _metadata() -> TransferEvidenceCaptureMetadata:
    return TransferEvidenceCaptureMetadata(
        source_prompt_digest=_digest(10_000),
        target_prompt_digest=_digest(10_001),
        loaded_tokens=256,
        target_prompt_tokens=280,
        recomputed_tokens=280,
    )


def test_builder_requires_canonical_layers_and_is_idempotent() -> None:
    builder = TransferEvidenceBuilder(_metadata())
    with pytest.raises(TransferCaptureError) as caught:
        builder.finish()
    assert caught.value.code is TransferCaptureErrorCode.INCOMPLETE_LAYERS

    for index in range(24):
        builder.add_layer(_layer(index))
    evidence = builder.finish()

    assert builder.layer_count == 24
    assert builder.finalized
    assert evidence.schema_version == TRANSFER_EVIDENCE_SCHEMA_VERSION
    assert builder.finish() is evidence


@pytest.mark.parametrize(
    ("operation", "code"),
    [
        (
            lambda builder: builder.add_layer(_layer(1)),
            TransferCaptureErrorCode.LAYER_ORDER_MISMATCH,
        ),
        (
            lambda builder: builder.add_layer(object()),
            TransferCaptureErrorCode.INVALID_LAYER,
        ),
    ],
)
def test_builder_rejects_invalid_or_out_of_order_samples(operation, code) -> None:
    builder = TransferEvidenceBuilder(_metadata())
    with pytest.raises(TransferCaptureError) as caught:
        operation(builder)
    assert caught.value.code is code
    assert builder.layer_count == 0


def test_builder_rejects_mutation_after_finalization() -> None:
    builder = TransferEvidenceBuilder(_metadata())
    for index in range(24):
        builder.add_layer(_layer(index))
    builder.finish()

    with pytest.raises(TransferCaptureError) as caught:
        builder.add_layer(_layer(0))
    assert caught.value.code is TransferCaptureErrorCode.FINALIZED


@pytest.mark.parametrize(
    "mutation",
    [
        {"loaded_tokens": 281},
        {"target_prompt_tokens": 0},
        {"recomputed_tokens": 279},
        {"prefill_tokens_avoided": 1},
        {"source_prompt_digest": "not-a-digest"},
    ],
)
def test_capture_metadata_is_strict(mutation) -> None:
    with pytest.raises(TransferCaptureError) as caught:
        replace(_metadata(), **mutation)
    assert caught.value.code is TransferCaptureErrorCode.INVALID_METADATA
