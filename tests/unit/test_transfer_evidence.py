"""CPU-only tests for the 100%-recompute transfer evidence sidecar."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from cacheblend_gpt_oss.correctness import (
    TRANSFER_EVIDENCE_SCHEMA_VERSION,
    LayerTransferEvidence,
    TransferEvidence,
    TransferEvidenceError,
    TransferEvidenceErrorCode,
    canonical_transfer_evidence_bytes,
    read_transfer_evidence,
    transfer_evidence_digest,
    transfer_evidence_from_dict,
    transfer_evidence_to_dict,
    validate_transfer_evidence_binding,
    write_transfer_evidence,
)
from cacheblend_gpt_oss.correctness.fixture import build_moved_document_fixture
from cacheblend_gpt_oss.correctness.models import (
    ARTIFACT_SCHEMA_VERSION,
    GPT_OSS_VOCAB_SIZE,
    ConnectorCorrectnessEvidence,
    CorrectnessArtifact,
    CorrectnessRunMode,
    CorrectnessRuntimeIdentity,
    FullVocabularyLogprobs,
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


def _evidence() -> TransferEvidence:
    return TransferEvidence(
        schema_version=TRANSFER_EVIDENCE_SCHEMA_VERSION,
        source_prompt_digest=_digest(10_000),
        target_prompt_digest=_digest(10_001),
        loaded_tokens=256,
        target_prompt_tokens=280,
        recomputed_tokens=280,
        prefill_tokens_avoided=0,
        layers=tuple(_layer(index) for index in range(24)),
    )


def _bound_artifact() -> CorrectnessArtifact:
    fixture = build_moved_document_fixture()
    values = [-1.0] * GPT_OSS_VOCAB_SIZE
    values[0] = 0.0
    return CorrectnessArtifact(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        run_mode=CorrectnessRunMode.CACHEBLEND_100PCT,
        runtime=CorrectnessRuntimeIdentity(
            model_id="openai/gpt-oss-20b",
            model_revision="model-revision",
            tokenizer_revision="tokenizer-revision",
            plugin_commit="a" * 40,
            model_config_digest="b" * 64,
            kv_cache_config_digest="c" * 64,
            vllm_version="0.19.1",
            lmcache_version="0.4.3",
            torch_version="2.10.0+cu128",
            cuda_runtime="12.8",
            gpu_name="NVIDIA A100-SXM4-80GB",
        ),
        prompt=fixture.prompt_identity,
        distribution=FullVocabularyLogprobs(tuple(values), 0),
        connector=ConnectorCorrectnessEvidence(
            reusable_document_tokens_requested=280,
            kv_tokens_found=256,
            kv_tokens_loaded=256,
            kv_tokens_rejected=0,
            tokens_recomputed=280,
            prefill_tokens_avoided=0,
        ),
    )


def test_all_layers_are_explicitly_split_between_sliding_and_full() -> None:
    evidence = _evidence()
    assert evidence.all_layers_loaded_and_overwritten
    assert tuple(layer.layer_index for layer in evidence.sliding_layers) == tuple(
        range(0, 24, 2)
    )
    assert tuple(layer.layer_index for layer in evidence.full_layers) == tuple(
        range(1, 24, 2)
    )


def test_transfer_evidence_binds_to_the_exact_correctness_artifact() -> None:
    artifact = _bound_artifact()
    evidence = _evidence()
    bound = replace(
        evidence,
        source_prompt_digest=artifact.prompt.source_prompt_digest,
        target_prompt_digest=artifact.prompt.target_prompt_digest,
    )

    validate_transfer_evidence_binding(artifact, bound)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda evidence, artifact: replace(
            evidence, source_prompt_digest=_digest(77_001)
        ),
        lambda evidence, artifact: replace(
            evidence,
            loaded_tokens=255,
            layers=tuple(
                replace(layer, token_count=255) for layer in evidence.layers
            ),
        ),
        lambda evidence, artifact: replace(
            evidence, target_prompt_tokens=279, recomputed_tokens=279
        ),
        lambda evidence, artifact: replace(
            evidence, target_prompt_tokens=281, recomputed_tokens=281
        ),
    ],
)
def test_transfer_evidence_binding_rejects_substitution(mutation) -> None:
    artifact = _bound_artifact()
    evidence = replace(
        _evidence(),
        source_prompt_digest=artifact.prompt.source_prompt_digest,
        target_prompt_digest=artifact.prompt.target_prompt_digest,
    )
    with pytest.raises(TransferEvidenceError) as caught:
        validate_transfer_evidence_binding(artifact, mutation(evidence, artifact))
    assert caught.value.code is TransferEvidenceErrorCode.ARTIFACT_BINDING_MISMATCH


def test_canonical_round_trip_and_digest_are_stable(tmp_path: Path) -> None:
    evidence = _evidence()
    encoded = canonical_transfer_evidence_bytes(evidence)
    assert encoded == canonical_transfer_evidence_bytes(
        transfer_evidence_from_dict(transfer_evidence_to_dict(evidence))
    )
    assert transfer_evidence_digest(evidence) == transfer_evidence_digest(
        transfer_evidence_from_dict(transfer_evidence_to_dict(evidence))
    )

    path = tmp_path / "transfer-evidence.json"
    write_transfer_evidence(path, evidence)
    assert read_transfer_evidence(path) == evidence
    with pytest.raises(TransferEvidenceError) as caught:
        write_transfer_evidence(path, evidence)
    assert caught.value.code is TransferEvidenceErrorCode.FILE_EXISTS


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (
            lambda evidence: replace(evidence, schema_version=True),
            TransferEvidenceErrorCode.INVALID_SCHEMA,
        ),
        (
            lambda evidence: replace(
                evidence, layers=evidence.layers[:-1]
            ),
            TransferEvidenceErrorCode.INCOMPLETE_LAYERS,
        ),
        (
            lambda evidence: replace(
                evidence,
                layers=(
                    replace(evidence.layers[0], attention_kind=AttentionKind.FULL),
                    *evidence.layers[1:],
                ),
            ),
            TransferEvidenceErrorCode.LAYER_KIND_MISMATCH,
        ),
        (
            lambda evidence: replace(evidence, loaded_tokens=257),
            TransferEvidenceErrorCode.INVALID_TOKEN_COUNT,
        ),
    ],
)
def test_top_level_evidence_invariants_fail_closed(mutation, code) -> None:
    with pytest.raises(TransferEvidenceError) as caught:
        mutation(_evidence())
    assert caught.value.code is code


@pytest.mark.parametrize(
    ("field", "code"),
    [
        ("key_before_digest", TransferEvidenceErrorCode.LOAD_NOT_OBSERVED),
        ("key_source_digest", TransferEvidenceErrorCode.SOURCE_MISMATCH),
        (
            "key_target_prefill_digest",
            TransferEvidenceErrorCode.PREFILL_MISMATCH,
        ),
        ("key_after_prefill_digest", TransferEvidenceErrorCode.PREFILL_MISMATCH),
    ],
)
def test_layer_digest_chain_failures_are_bounded(field: str, code) -> None:
    layer = _layer(0)
    value = getattr(layer, field)
    replacement = (
        layer.key_source_digest
        if field == "key_before_digest"
        else _digest(999_999)
    )
    if field == "key_after_prefill_digest":
        replacement = layer.key_after_load_digest
    if field == "key_target_prefill_digest":
        replacement = _digest(999_998)
    with pytest.raises(TransferEvidenceError) as caught:
        replace(layer, **{field: replacement})
    assert caught.value.code is code
    assert value != replacement


def test_layer_overwrite_failure_requires_matching_fresh_prefill_digest() -> None:
    layer = _layer(0)
    with pytest.raises(TransferEvidenceError) as caught:
        replace(
            layer,
            key_target_prefill_digest=layer.key_after_load_digest,
            key_after_prefill_digest=layer.key_after_load_digest,
        )
    assert caught.value.code is TransferEvidenceErrorCode.OVERWRITE_NOT_OBSERVED


def test_unknown_json_fields_and_invalid_attention_values_fail_closed() -> None:
    data = transfer_evidence_to_dict(_evidence())
    data["unexpected"] = True
    with pytest.raises(TransferEvidenceError) as caught:
        transfer_evidence_from_dict(data)
    assert caught.value.code is TransferEvidenceErrorCode.INVALID_JSON

    layers = data["layers"]
    assert isinstance(layers, list)
    layers[0]["attention_kind"] = "unknown"
    with pytest.raises(TransferEvidenceError) as caught:
        transfer_evidence_from_dict(data)
    assert caught.value.code is TransferEvidenceErrorCode.INVALID_JSON


def test_file_tampering_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "tampered.json"
    write_transfer_evidence(path, _evidence())
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["layers"][0]["key_after_load_digest"] = _digest(77_777)
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(TransferEvidenceError) as caught:
        read_transfer_evidence(path)
    assert caught.value.code is TransferEvidenceErrorCode.SOURCE_MISMATCH
