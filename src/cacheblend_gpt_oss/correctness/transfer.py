# SPDX-License-Identifier: Apache-2.0
"""CPU-validated evidence for the pinned 100%-recompute transfer gate.

The live numerical artifact compares final output distributions, but that alone
cannot prove that a non-prefix KV transfer happened and was subsequently
overwritten.  This sidecar schema records bounded digests sampled from every
GPT-OSS layer and requires the independent sequence:

``destination-before -> loaded/source -> fresh-prefill``.

It is intentionally separate from the output artifact so a capture tool can
write it from a worker-side tensor probe without exposing raw KV or prompt
tokens.  No digest is accepted as evidence unless all 24 canonical layers are
present with the expected alternating sliding/full attention kind.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, NoReturn

from cacheblend_gpt_oss.correctness.models import (
    ConnectorCorrectnessEvidence,
    CorrectnessArtifact,
    CorrectnessRunMode,
    CorrectnessRuntimeIdentity,
    ReusableSegmentIdentity,
)
from cacheblend_gpt_oss.gpt_oss.layout import (
    GPT_OSS_NUM_LAYERS,
    AttentionKind,
)
from cacheblend_gpt_oss.planner.models import CacheNamespace

TRANSFER_EVIDENCE_SCHEMA_VERSION = 2
_HEX_64 = re.compile(r"[0-9a-f]{64}")
_NAMESPACE_DIGEST_DOMAIN = b"cacheblend-gpt-oss-transfer-namespace-v1\0"


class TransferEvidenceErrorCode(str, Enum):
    """Bounded transfer-evidence validation failures."""

    INVALID_SCHEMA = "invalid_schema"
    INVALID_DIGEST = "invalid_digest"
    INVALID_LAYER = "invalid_layer"
    INVALID_ATTENTION_KIND = "invalid_attention_kind"
    INVALID_TOKEN_COUNT = "invalid_token_count"
    INCOMPLETE_LAYERS = "incomplete_layers"
    LAYER_KIND_MISMATCH = "layer_kind_mismatch"
    LOAD_NOT_OBSERVED = "load_not_observed"
    OVERWRITE_NOT_OBSERVED = "overwrite_not_observed"
    SOURCE_MISMATCH = "source_mismatch"
    PREFILL_MISMATCH = "prefill_mismatch"
    ARTIFACT_BINDING_MISMATCH = "artifact_binding_mismatch"
    INVALID_JSON = "invalid_json"
    FILE_EXISTS = "file_exists"


class TransferEvidenceError(ValueError):
    """Fail-closed error without request IDs, token IDs, or prompt text."""

    def __init__(self, code: TransferEvidenceErrorCode) -> None:
        self.code = code
        super().__init__(f"CacheBlend transfer evidence failure: {code.value}")


def _fail(code: TransferEvidenceErrorCode) -> NoReturn:
    raise TransferEvidenceError(code)


def _require_digest(value: object) -> str:
    if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
        _fail(TransferEvidenceErrorCode.INVALID_DIGEST)
    return value


def _require_count(value: object, *, allow_zero: bool = False) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or (value < 0 if allow_zero else value <= 0)
    ):
        _fail(TransferEvidenceErrorCode.INVALID_TOKEN_COUNT)
    return value


def _expected_kind(layer_index: int) -> AttentionKind:
    return AttentionKind.SLIDING if layer_index % 2 == 0 else AttentionKind.FULL


def cache_namespace_digest(namespace: CacheNamespace) -> str:
    """Hash every persistent namespace field with unambiguous framing."""

    if not isinstance(namespace, CacheNamespace):
        _fail(TransferEvidenceErrorCode.INVALID_DIGEST)
    digest = sha256(_NAMESPACE_DIGEST_DOMAIN)
    for name, value in namespace.canonical_fields():
        encoded_name = name.encode("utf-8")
        encoded_value = value.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(2, "big"))
        digest.update(encoded_name)
        digest.update(len(encoded_value).to_bytes(4, "big"))
        digest.update(encoded_value)
    return digest.hexdigest()


def correctness_runtime_namespace_digest(
    runtime: CorrectnessRuntimeIdentity,
) -> str:
    """Derive the cache namespace represented by one correctness artifact."""

    if not isinstance(runtime, CorrectnessRuntimeIdentity):
        _fail(TransferEvidenceErrorCode.ARTIFACT_BINDING_MISMATCH)
    return cache_namespace_digest(
        CacheNamespace(
            schema_version=1,
            model_id=runtime.model_id,
            model_revision=runtime.model_revision,
            tokenizer_id=runtime.model_id,
            tokenizer_revision=runtime.tokenizer_revision,
            model_config_digest=runtime.model_config_digest,
            kv_cache_config_digest=runtime.kv_cache_config_digest,
            adapter_revision=runtime.plugin_commit,
            vllm_version=runtime.vllm_version,
            lmcache_version=runtime.lmcache_version,
            torch_version=runtime.torch_version,
            cuda_runtime=runtime.cuda_runtime,
        )
    )


@dataclass(frozen=True, slots=True)
class LayerTransferEvidence:
    """One layer's digest chain for a loaded and overwritten KV span."""

    layer_index: int
    attention_kind: AttentionKind
    token_count: int
    load_write_observed: bool
    prefill_write_observed: bool
    key_before_digest: str
    key_raw_source_digest: str
    key_corrected_source_digest: str
    key_after_load_digest: str
    key_target_prefill_digest: str
    key_after_prefill_digest: str
    value_before_digest: str
    value_source_digest: str
    value_after_load_digest: str
    value_target_prefill_digest: str
    value_after_prefill_digest: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or not 0 <= self.layer_index < GPT_OSS_NUM_LAYERS
        ):
            _fail(TransferEvidenceErrorCode.INVALID_LAYER)
        if not isinstance(self.attention_kind, AttentionKind):
            _fail(TransferEvidenceErrorCode.INVALID_ATTENTION_KIND)
        if self.attention_kind is not _expected_kind(self.layer_index):
            _fail(TransferEvidenceErrorCode.LAYER_KIND_MISMATCH)
        _require_count(self.token_count)
        if self.load_write_observed is not True:
            _fail(TransferEvidenceErrorCode.LOAD_NOT_OBSERVED)
        if self.prefill_write_observed is not True:
            _fail(TransferEvidenceErrorCode.OVERWRITE_NOT_OBSERVED)
        digests = (
            self.key_before_digest,
            self.key_raw_source_digest,
            self.key_corrected_source_digest,
            self.key_after_load_digest,
            self.key_target_prefill_digest,
            self.key_after_prefill_digest,
            self.value_before_digest,
            self.value_source_digest,
            self.value_after_load_digest,
            self.value_target_prefill_digest,
            self.value_after_prefill_digest,
        )
        for digest in digests:
            _require_digest(digest)
        if self.key_after_load_digest != self.key_corrected_source_digest or (
            self.value_after_load_digest != self.value_source_digest
        ):
            _fail(TransferEvidenceErrorCode.SOURCE_MISMATCH)
        if self.key_after_prefill_digest != self.key_target_prefill_digest or (
            self.value_after_prefill_digest != self.value_target_prefill_digest
        ):
            _fail(TransferEvidenceErrorCode.PREFILL_MISMATCH)
        # A write can legitimately reproduce the bytes already present in a
        # cache row.  In particular, GPT-OSS layer 0 K/V for a moved token can
        # equal the position-corrected source K/V bit-for-bit.  Operation
        # observation therefore comes from the audited load/save hooks above;
        # digest equality independently proves source and full-prefill content.


@dataclass(frozen=True, slots=True)
class TransferEvidence:
    """All-layer transfer proof associated with one moved-document request."""

    schema_version: int
    namespace_digest: str
    source_prompt_digest: str
    source_prompt_tokens: int
    target_prompt_digest: str
    loaded_tokens: int
    target_prompt_tokens: int
    recomputed_tokens: int
    prefill_tokens_avoided: int
    reusable_segments: tuple[ReusableSegmentIdentity, ...]
    layers: tuple[LayerTransferEvidence, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != TRANSFER_EVIDENCE_SCHEMA_VERSION
        ):
            _fail(TransferEvidenceErrorCode.INVALID_SCHEMA)
        _require_digest(self.namespace_digest)
        _require_digest(self.source_prompt_digest)
        _require_digest(self.target_prompt_digest)
        source = _require_count(self.source_prompt_tokens)
        loaded = _require_count(self.loaded_tokens)
        target = _require_count(self.target_prompt_tokens)
        recomputed = _require_count(self.recomputed_tokens)
        avoided = _require_count(self.prefill_tokens_avoided, allow_zero=True)
        if loaded > target or recomputed != target or avoided != 0:
            _fail(TransferEvidenceErrorCode.INVALID_TOKEN_COUNT)
        try:
            reusable_segments = tuple(self.reusable_segments)
        except TypeError:
            _fail(TransferEvidenceErrorCode.INVALID_TOKEN_COUNT)
        if (
            not reusable_segments
            or any(
                not isinstance(segment, ReusableSegmentIdentity)
                for segment in reusable_segments
            )
            or sum(segment.tokens for segment in reusable_segments) != loaded
            or max(
                segment.source_start + segment.tokens
                for segment in reusable_segments
            ) > source
        ):
            _fail(TransferEvidenceErrorCode.INVALID_TOKEN_COUNT)
        try:
            layers = tuple(self.layers)
        except TypeError:
            _fail(TransferEvidenceErrorCode.INCOMPLETE_LAYERS)
        if len(layers) != GPT_OSS_NUM_LAYERS or any(
            not isinstance(layer, LayerTransferEvidence) for layer in layers
        ):
            _fail(TransferEvidenceErrorCode.INCOMPLETE_LAYERS)
        if tuple(layer.layer_index for layer in layers) != tuple(
            range(GPT_OSS_NUM_LAYERS)
        ):
            _fail(TransferEvidenceErrorCode.INCOMPLETE_LAYERS)
        if any(layer.token_count != loaded for layer in layers):
            _fail(TransferEvidenceErrorCode.INVALID_TOKEN_COUNT)
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "reusable_segments", reusable_segments)

    @property
    def sliding_layers(self) -> tuple[LayerTransferEvidence, ...]:
        return tuple(
            layer
            for layer in self.layers
            if layer.attention_kind is AttentionKind.SLIDING
        )

    @property
    def full_layers(self) -> tuple[LayerTransferEvidence, ...]:
        return tuple(
            layer
            for layer in self.layers
            if layer.attention_kind is AttentionKind.FULL
        )

    @property
    def all_layers_loaded_and_overwritten(self) -> bool:
        """All layer constructors already enforce this; expose it for reports."""

        return len(self.layers) == GPT_OSS_NUM_LAYERS


def validate_transfer_evidence_binding(
    artifact: CorrectnessArtifact,
    evidence: TransferEvidence,
) -> None:
    """Bind a layer sidecar to one exact CacheBlend correctness artifact.

    The sidecar's own schema proves that every layer observed a load followed
    by a full-prefill overwrite.  It does not, by itself, prove that the
    sampled prompt, connector counter interval, and output distribution came
    from the same request.  This explicit binding closes that evidence
    substitution gap without storing prompt text or token IDs.
    """

    if not isinstance(artifact, CorrectnessArtifact) or not isinstance(
        evidence, TransferEvidence
    ):
        _fail(TransferEvidenceErrorCode.ARTIFACT_BINDING_MISMATCH)
    connector = artifact.connector
    if (
        artifact.run_mode is not CorrectnessRunMode.CACHEBLEND_100PCT
        or not isinstance(connector, ConnectorCorrectnessEvidence)
        or evidence.namespace_digest
        != correctness_runtime_namespace_digest(artifact.runtime)
        or evidence.source_prompt_digest != artifact.prompt.source_prompt_digest
        or evidence.source_prompt_tokens != artifact.prompt.source_prompt_tokens
        or evidence.target_prompt_digest != artifact.prompt.target_prompt_digest
        or evidence.reusable_segments != artifact.prompt.reusable_segments
        or evidence.target_prompt_tokens != artifact.prompt.target_prompt_tokens
        or evidence.loaded_tokens != connector.kv_tokens_loaded
        or evidence.recomputed_tokens != connector.tokens_recomputed
        or evidence.prefill_tokens_avoided
        != connector.prefill_tokens_avoided
    ):
        _fail(TransferEvidenceErrorCode.ARTIFACT_BINDING_MISMATCH)


def transfer_evidence_to_dict(evidence: TransferEvidence) -> dict[str, Any]:
    """Return the canonical JSON representation."""

    return {
        "schema_version": evidence.schema_version,
        "namespace_digest": evidence.namespace_digest,
        "source_prompt_digest": evidence.source_prompt_digest,
        "source_prompt_tokens": evidence.source_prompt_tokens,
        "target_prompt_digest": evidence.target_prompt_digest,
        "loaded_tokens": evidence.loaded_tokens,
        "target_prompt_tokens": evidence.target_prompt_tokens,
        "recomputed_tokens": evidence.recomputed_tokens,
        "prefill_tokens_avoided": evidence.prefill_tokens_avoided,
        "reusable_segments": [
            {
                "token_digest": segment.token_digest,
                "tokens": segment.tokens,
                "source_start": segment.source_start,
                "target_start": segment.target_start,
            }
            for segment in evidence.reusable_segments
        ],
        "layers": [
            {
                "layer_index": layer.layer_index,
                "attention_kind": layer.attention_kind.value,
                "token_count": layer.token_count,
                "load_write_observed": layer.load_write_observed,
                "prefill_write_observed": layer.prefill_write_observed,
                "key_before_digest": layer.key_before_digest,
                "key_raw_source_digest": layer.key_raw_source_digest,
                "key_corrected_source_digest": layer.key_corrected_source_digest,
                "key_after_load_digest": layer.key_after_load_digest,
                "key_target_prefill_digest": layer.key_target_prefill_digest,
                "key_after_prefill_digest": layer.key_after_prefill_digest,
                "value_before_digest": layer.value_before_digest,
                "value_source_digest": layer.value_source_digest,
                "value_after_load_digest": layer.value_after_load_digest,
                "value_target_prefill_digest": layer.value_target_prefill_digest,
                "value_after_prefill_digest": layer.value_after_prefill_digest,
            }
            for layer in evidence.layers
        ],
    }


def _exact_mapping(value: object, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        _fail(TransferEvidenceErrorCode.INVALID_JSON)
    return value


def transfer_evidence_from_dict(data: object) -> TransferEvidence:
    """Parse the fail-closed transfer sidecar schema."""

    root = _exact_mapping(
        data,
        {
            "schema_version",
            "namespace_digest",
            "source_prompt_digest",
            "source_prompt_tokens",
            "target_prompt_digest",
            "loaded_tokens",
            "target_prompt_tokens",
            "recomputed_tokens",
            "prefill_tokens_avoided",
            "reusable_segments",
            "layers",
        },
        "root",
    )
    raw_layers = root["layers"]
    if not isinstance(raw_layers, list):
        _fail(TransferEvidenceErrorCode.INVALID_JSON)
    layers: list[LayerTransferEvidence] = []
    layer_keys = {
        "layer_index",
        "attention_kind",
        "token_count",
        "load_write_observed",
        "prefill_write_observed",
        "key_before_digest",
        "key_raw_source_digest",
        "key_corrected_source_digest",
        "key_after_load_digest",
        "key_target_prefill_digest",
        "key_after_prefill_digest",
        "value_before_digest",
        "value_source_digest",
        "value_after_load_digest",
        "value_target_prefill_digest",
        "value_after_prefill_digest",
    }
    for raw_layer in raw_layers:
        layer = _exact_mapping(raw_layer, layer_keys, "layer")
        try:
            layers.append(
                LayerTransferEvidence(
                    layer_index=layer["layer_index"],
                    attention_kind=AttentionKind(layer["attention_kind"]),
                    token_count=layer["token_count"],
                    load_write_observed=layer["load_write_observed"],
                    prefill_write_observed=layer["prefill_write_observed"],
                    key_before_digest=layer["key_before_digest"],
                    key_raw_source_digest=layer["key_raw_source_digest"],
                    key_corrected_source_digest=(
                        layer["key_corrected_source_digest"]
                    ),
                    key_after_load_digest=layer["key_after_load_digest"],
                    key_target_prefill_digest=layer["key_target_prefill_digest"],
                    key_after_prefill_digest=layer["key_after_prefill_digest"],
                    value_before_digest=layer["value_before_digest"],
                    value_source_digest=layer["value_source_digest"],
                    value_after_load_digest=layer["value_after_load_digest"],
                    value_target_prefill_digest=layer["value_target_prefill_digest"],
                    value_after_prefill_digest=layer["value_after_prefill_digest"],
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, TransferEvidenceError):
                raise
            _fail(TransferEvidenceErrorCode.INVALID_JSON)
    raw_segments = root["reusable_segments"]
    if not isinstance(raw_segments, list):
        _fail(TransferEvidenceErrorCode.INVALID_JSON)
    segment_keys = {"token_digest", "tokens", "source_start", "target_start"}
    segments: list[ReusableSegmentIdentity] = []
    for raw_segment in raw_segments:
        segment = _exact_mapping(raw_segment, segment_keys, "reusable_segment")
        try:
            segments.append(
                ReusableSegmentIdentity(
                    token_digest=segment["token_digest"],
                    tokens=segment["tokens"],
                    source_start=segment["source_start"],
                    target_start=segment["target_start"],
                )
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, TransferEvidenceError):
                raise
            _fail(TransferEvidenceErrorCode.INVALID_JSON)
    return TransferEvidence(
        schema_version=root["schema_version"],
        namespace_digest=root["namespace_digest"],
        source_prompt_digest=root["source_prompt_digest"],
        source_prompt_tokens=root["source_prompt_tokens"],
        target_prompt_digest=root["target_prompt_digest"],
        loaded_tokens=root["loaded_tokens"],
        target_prompt_tokens=root["target_prompt_tokens"],
        recomputed_tokens=root["recomputed_tokens"],
        prefill_tokens_avoided=root["prefill_tokens_avoided"],
        reusable_segments=tuple(segments),
        layers=tuple(layers),
    )


def canonical_transfer_evidence_bytes(evidence: TransferEvidence) -> bytes:
    return json.dumps(
        transfer_evidence_to_dict(evidence),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def transfer_evidence_digest(evidence: TransferEvidence) -> str:
    return sha256(canonical_transfer_evidence_bytes(evidence)).hexdigest()


def read_transfer_evidence(path: Path) -> TransferEvidence:
    try:
        return transfer_evidence_from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        if isinstance(exc, TransferEvidenceError):
            raise
        _fail(TransferEvidenceErrorCode.INVALID_JSON)


def write_transfer_evidence(path: Path, evidence: TransferEvidence) -> None:
    try:
        with path.open("xb") as output:
            output.write(canonical_transfer_evidence_bytes(evidence) + b"\n")
    except FileExistsError as exc:
        raise TransferEvidenceError(TransferEvidenceErrorCode.FILE_EXISTS) from exc


__all__ = [
    "TRANSFER_EVIDENCE_SCHEMA_VERSION",
    "LayerTransferEvidence",
    "TransferEvidence",
    "TransferEvidenceError",
    "TransferEvidenceErrorCode",
    "cache_namespace_digest",
    "canonical_transfer_evidence_bytes",
    "correctness_runtime_namespace_digest",
    "read_transfer_evidence",
    "transfer_evidence_digest",
    "transfer_evidence_from_dict",
    "transfer_evidence_to_dict",
    "validate_transfer_evidence_binding",
    "write_transfer_evidence",
]
