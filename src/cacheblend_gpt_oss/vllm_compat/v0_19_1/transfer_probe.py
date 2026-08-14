# SPDX-License-Identifier: Apache-2.0
"""Worker-side all-layer tensor evidence for the 100%-recompute gate.

The probe is enabled only by an explicit create-only output path.  It samples
the exact paged-cache rows addressed by the already validated worker load plan,
derives the independently position-corrected source K rows from LMCache staging,
and records the ordinary-attention save hook for every GPT-OSS layer.  Raw KV
and token IDs never leave the worker process.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import NoReturn, Protocol

from cacheblend_gpt_oss.correctness import (
    LayerTransferEvidence,
    ReusableSegmentIdentity,
    TransferEvidenceBuilder,
    TransferEvidenceCaptureMetadata,
    cache_namespace_digest,
    write_transfer_evidence,
)
from cacheblend_gpt_oss.correctness.fixture import digest_token_ids
from cacheblend_gpt_oss.gpt_oss.layout import (
    GPT_OSS_NUM_LAYERS,
    AttentionKind,
    LayerTokenScatterSpan,
    extract_gpt_oss_layer_index,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.data_plane import (
    GPT_OSS_HEAD_DIM,
    GPT_OSS_NUM_KV_HEADS,
    KeyPositionCorrector,
    TensorOps,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.transfer_runtime import (
    WorkerLoadPlan,
)

_KEY_COMPONENT = 0
_VALUE_COMPONENT = 1
_TENSOR_DIGEST_DOMAIN = b"cacheblend-gpt-oss-transfer-tensor-v1\0"


class TransferProbeErrorCode(str, Enum):
    """Bounded failures that do not expose request, token, or tensor data."""

    INVALID_CONFIG = "invalid_config"
    INVALID_STATE = "invalid_state"
    INVALID_PLAN = "invalid_plan"
    INVALID_TENSOR = "invalid_tensor"
    INCOMPLETE_LAYERS = "incomplete_layers"


class TransferProbeError(RuntimeError):
    """Fail-closed worker probe error with a stable bounded code."""

    def __init__(self, code: TransferProbeErrorCode) -> None:
        self.code = code
        super().__init__(f"CacheBlend transfer probe failure: {code.value}")


def _fail(code: TransferProbeErrorCode) -> NoReturn:
    raise TransferProbeError(code)


class TensorBytesReader(Protocol):
    """Copy one tensor view to canonical row-major bytes."""

    def __call__(self, tensor: object) -> bytes: ...


def load_torch_tensor_bytes_reader() -> TensorBytesReader:
    """Load the production Torch byte reader without a top-level dependency."""

    torch = import_module("torch")
    tensor_type = getattr(torch, "Tensor", None)
    uint8 = getattr(torch, "uint8", None)
    cuda = getattr(torch, "cuda", None)
    synchronize = getattr(cuda, "synchronize", None)
    if tensor_type is None or uint8 is None or not callable(synchronize):
        _fail(TransferProbeErrorCode.INVALID_CONFIG)

    def read(tensor: object) -> bytes:
        if not isinstance(tensor, tensor_type):
            _fail(TransferProbeErrorCode.INVALID_TENSOR)
        try:
            synchronize(tensor.device)
            byte_tensor = tensor.detach().contiguous().view(uint8).cpu()
            array = byte_tensor.numpy()
            return bytes(array.tobytes(order="C"))
        except Exception as exc:
            raise TransferProbeError(
                TransferProbeErrorCode.INVALID_TENSOR
            ) from exc

    return read


@dataclass(slots=True)
class _LayerSamples:
    before_key: str
    before_value: str
    raw_source_key: str
    corrected_source_key: str
    source_value: str
    after_load_key: str | None = None
    after_load_value: str | None = None
    target_prefill_key: str | None = None
    target_prefill_value: str | None = None


class GptOssTransferEvidenceProbe:
    """Capture one successful positive-load request across all 24 layers."""

    def __init__(
        self,
        *,
        output_path: Path,
        tensor_ops: TensorOps,
        tensor_bytes_reader: TensorBytesReader,
        paged_caches: Mapping[str, object],
        correct_key_positions: KeyPositionCorrector,
    ) -> None:
        if (
            not isinstance(output_path, Path)
            or not output_path.is_absolute()
            or not output_path.parent.is_dir()
            or output_path.exists()
            or output_path.is_symlink()
        ):
            _fail(TransferProbeErrorCode.INVALID_CONFIG)
        if not callable(tensor_bytes_reader) or not callable(correct_key_positions):
            _fail(TransferProbeErrorCode.INVALID_CONFIG)
        caches = dict(paged_caches)
        expected_names = {
            f"model.layers.{index}.attn.attn"
            for index in range(GPT_OSS_NUM_LAYERS)
        }
        if set(caches) != expected_names:
            _fail(TransferProbeErrorCode.INVALID_CONFIG)
        self._output_path = output_path
        self._ops = tensor_ops
        self._read_bytes = tensor_bytes_reader
        self._paged_caches = caches
        self._correct_key_positions = correct_key_positions
        self._plan: WorkerLoadPlan | None = None
        self._spans: dict[int, tuple[LayerTokenScatterSpan, ...]] = {}
        self._samples: dict[int, _LayerSamples] = {}
        self._source_prompt_digest: str | None = None
        self._target_prompt_digest: str | None = None
        self._load_write_observed = False
    @property
    def active(self) -> bool:
        return self._plan is not None

    def begin_load(
        self,
        plan: WorkerLoadPlan,
        *,
        staging: object,
        retrieval_buffer_offset: int,
    ) -> None:
        """Sample destination-before and independent staging/source content."""

        if self._plan is not None:
            _fail(TransferProbeErrorCode.INVALID_STATE)
        spans = self._canonical_spans(plan)
        source_tokens = self._source_prompt_tokens(plan)
        samples: dict[int, _LayerSamples] = {}
        for layer_index in range(GPT_OSS_NUM_LAYERS):
            layer_spans = spans[layer_index]
            before_key, before_value = self._paged_digests(
                layer_index, layer_spans
            )
            raw_source_key, corrected_source_key, source_value = (
                self._staging_source_digests(
                    staging,
                    layer_index,
                    layer_spans,
                    retrieval_buffer_offset=retrieval_buffer_offset,
                )
            )
            samples[layer_index] = _LayerSamples(
                before_key=before_key,
                before_value=before_value,
                raw_source_key=raw_source_key,
                corrected_source_key=corrected_source_key,
                source_value=source_value,
            )
        self._plan = plan
        self._spans = spans
        self._samples = samples
        self._source_prompt_digest = digest_token_ids(source_tokens)
        self._target_prompt_digest = digest_token_ids(
            tuple(plan.metadata.prompt_token_ids)
        )
        self._load_write_observed = False

    def mark_load_complete(self) -> None:
        """Sample destination rows after every validated scatter returned."""

        if self._plan is None:
            _fail(TransferProbeErrorCode.INVALID_STATE)
        for layer_index in range(GPT_OSS_NUM_LAYERS):
            key_digest, value_digest = self._paged_digests(
                layer_index, self._spans[layer_index]
            )
            sample = self._samples[layer_index]
            sample.after_load_key = key_digest
            sample.after_load_value = value_digest
        self._load_write_observed = True

    def record_prefill_layer(self, layer_name: str) -> None:
        """Record the pinned attention save hook after this layer's prefill."""

        if self._plan is None or not self._load_write_observed:
            _fail(TransferProbeErrorCode.INVALID_STATE)
        try:
            layer_index = extract_gpt_oss_layer_index(layer_name)
        except Exception as exc:
            raise TransferProbeError(TransferProbeErrorCode.INVALID_PLAN) from exc
        sample = self._samples[layer_index]
        if sample.target_prefill_key is not None:
            _fail(TransferProbeErrorCode.INVALID_STATE)
        key_digest, value_digest = self._paged_digests(
            layer_index, self._spans[layer_index]
        )
        sample.target_prefill_key = key_digest
        sample.target_prefill_value = value_digest

    def finish(
        self,
        *,
        recomputed_tokens: int,
        prefill_tokens_avoided: int,
    ) -> None:
        """Take final samples and atomically create the bound sidecar."""

        plan = self._plan
        source_digest = self._source_prompt_digest
        target_digest = self._target_prompt_digest
        if (
            plan is None
            or source_digest is None
            or target_digest is None
            or not self._load_write_observed
            or any(
                sample.target_prefill_key is None
                or sample.target_prefill_value is None
                for sample in self._samples.values()
            )
        ):
            _fail(TransferProbeErrorCode.INCOMPLETE_LAYERS)
        builder = TransferEvidenceBuilder(
            TransferEvidenceCaptureMetadata(
                namespace_digest=cache_namespace_digest(
                    plan.metadata.cache_namespace
                ),
                source_prompt_digest=source_digest,
                source_prompt_tokens=len(self._source_prompt_tokens(plan)),
                target_prompt_digest=target_digest,
                loaded_tokens=plan.expected_tokens,
                target_prompt_tokens=plan.metadata.prompt_token_count,
                recomputed_tokens=recomputed_tokens,
                reusable_segments=self._reusable_segments(plan),
                prefill_tokens_avoided=prefill_tokens_avoided,
            )
        )
        for layer_index in range(GPT_OSS_NUM_LAYERS):
            sample = self._samples[layer_index]
            after_key, after_value = self._paged_digests(
                layer_index, self._spans[layer_index]
            )
            assert sample.after_load_key is not None
            assert sample.after_load_value is not None
            assert sample.target_prefill_key is not None
            assert sample.target_prefill_value is not None
            builder.add_layer(
                LayerTransferEvidence(
                    layer_index=layer_index,
                    attention_kind=(
                        AttentionKind.SLIDING
                        if layer_index % 2 == 0
                        else AttentionKind.FULL
                    ),
                    token_count=plan.expected_tokens,
                    load_write_observed=True,
                    prefill_write_observed=True,
                    key_before_digest=sample.before_key,
                    key_raw_source_digest=sample.raw_source_key,
                    key_corrected_source_digest=sample.corrected_source_key,
                    key_after_load_digest=sample.after_load_key,
                    key_target_prefill_digest=sample.target_prefill_key,
                    key_after_prefill_digest=after_key,
                    value_before_digest=sample.before_value,
                    value_source_digest=sample.source_value,
                    value_after_load_digest=sample.after_load_value,
                    value_target_prefill_digest=sample.target_prefill_value,
                    value_after_prefill_digest=after_value,
                )
            )
        evidence = builder.finish()
        write_transfer_evidence(self._output_path, evidence)
        self._clear()

    def abort(self) -> None:
        """Discard only in-memory samples after a failed scatter."""

        self._clear()

    def _clear(self) -> None:
        self._plan = None
        self._spans = {}
        self._samples = {}
        self._source_prompt_digest = None
        self._target_prompt_digest = None
        self._load_write_observed = False

    def _canonical_spans(
        self, plan: WorkerLoadPlan
    ) -> dict[int, tuple[LayerTokenScatterSpan, ...]]:
        if not isinstance(plan, WorkerLoadPlan) or not plan.candidates:
            _fail(TransferProbeErrorCode.INVALID_PLAN)
        by_layer: dict[int, list[LayerTokenScatterSpan]] = {
            index: [] for index in range(GPT_OSS_NUM_LAYERS)
        }
        for candidate in plan.candidates:
            for span in candidate.scatter_plan.layer_spans:
                if span.layer_index not in by_layer:
                    _fail(TransferProbeErrorCode.INVALID_PLAN)
                by_layer[span.layer_index].append(span)
        result: dict[int, tuple[LayerTokenScatterSpan, ...]] = {}
        for layer_index, layer_spans in by_layer.items():
            ordered = tuple(
                sorted(layer_spans, key=lambda span: span.target_range.start)
            )
            if (
                not ordered
                or sum(span.token_count for span in ordered) != plan.expected_tokens
                or any(span.layer_index != layer_index for span in ordered)
            ):
                _fail(TransferProbeErrorCode.INVALID_PLAN)
            result[layer_index] = ordered
        return result

    @staticmethod
    def _source_prompt_tokens(plan: WorkerLoadPlan) -> tuple[int, ...]:
        records = sorted(
            (
                candidate.verified_candidate.match.record
                for candidate in plan.candidates
            ),
            key=lambda record: record.source_range.start,
        )
        cursor = 0
        tokens: list[int] = []
        for record in records:
            if record.source_range.start != cursor:
                _fail(TransferProbeErrorCode.INVALID_PLAN)
            tokens.extend(record.token_ids)
            cursor = record.source_range.end
        if cursor != plan.expected_tokens:
            _fail(TransferProbeErrorCode.INVALID_PLAN)
        return tuple(tokens)

    @staticmethod
    def _reusable_segments(
        plan: WorkerLoadPlan,
    ) -> tuple[ReusableSegmentIdentity, ...]:
        segments = tuple(
            sorted(
                (
                    ReusableSegmentIdentity(
                        token_digest=digest_token_ids(
                            candidate.verified_candidate.match.record.token_ids
                        ),
                        tokens=len(
                            candidate.verified_candidate.match.record.token_ids
                        ),
                        source_start=(
                            candidate.verified_candidate.match.record.source_range.start
                        ),
                        target_start=(
                            candidate.verified_candidate.candidate.target_range.start
                        ),
                    )
                    for candidate in plan.candidates
                ),
                key=lambda segment: segment.source_start,
            )
        )
        if sum(segment.tokens for segment in segments) != plan.expected_tokens:
            _fail(TransferProbeErrorCode.INVALID_PLAN)
        return segments

    def _paged_digests(
        self,
        layer_index: int,
        spans: Sequence[LayerTokenScatterSpan],
    ) -> tuple[str, str]:
        paged = self._paged_caches[
            f"model.layers.{layer_index}.attn.attn"
        ]
        keys: list[object] = []
        values: list[object] = []
        for span in spans:
            keys.append(
                self._ops.paged_rows(
                    paged,
                    component=_KEY_COMPONENT,
                    block_id=span.group_span.block_id,
                    block_offset=span.group_span.block_offset,
                    token_count=span.token_count,
                )
            )
            values.append(
                self._ops.paged_rows(
                    paged,
                    component=_VALUE_COMPONENT,
                    block_id=span.group_span.block_id,
                    block_offset=span.group_span.block_offset,
                    token_count=span.token_count,
                )
            )
        return self._digest_views(keys), self._digest_views(values)

    def _staging_source_digests(
        self,
        staging: object,
        layer_index: int,
        spans: Sequence[LayerTokenScatterSpan],
        *,
        retrieval_buffer_offset: int,
    ) -> tuple[str, str, str]:
        raw_keys: list[object] = []
        corrected_keys: list[object] = []
        values: list[object] = []
        expected_shape_tail = (GPT_OSS_NUM_KV_HEADS, GPT_OSS_HEAD_DIM)
        for span in spans:
            staging_start = retrieval_buffer_offset + span.target_range.start
            shape = (span.token_count, *expected_shape_tail)
            key_rows = self._ops.reshape(
                self._ops.staging_rows(
                    staging,
                    component=_KEY_COMPONENT,
                    layer_index=layer_index,
                    token_start=staging_start,
                    token_count=span.token_count,
                ),
                shape,
            )
            value_rows = self._ops.reshape(
                self._ops.staging_rows(
                    staging,
                    component=_VALUE_COMPONENT,
                    layer_index=layer_index,
                    token_start=staging_start,
                    token_count=span.token_count,
                ),
                shape,
            )
            corrected_keys.append(
                self._correct_key_positions(
                    key_rows,
                    source_positions=tuple(
                        range(span.source_range.start, span.source_range.end)
                    ),
                    target_positions=tuple(
                        range(span.target_range.start, span.target_range.end)
                    ),
                    layer_index=layer_index,
                )
            )
            raw_keys.append(key_rows)
            values.append(value_rows)
        return (
            self._digest_views(raw_keys),
            self._digest_views(corrected_keys),
            self._digest_views(values),
        )

    def _digest_views(self, tensors: Sequence[object]) -> str:
        digest = sha256(_TENSOR_DIGEST_DOMAIN)
        digest.update(len(tensors).to_bytes(4, "big"))
        for tensor in tensors:
            shape = self._ops.shape(tensor)
            dtype_name = self._ops.dtype_name(tensor)
            payload = self._read_bytes(tensor)
            if not shape or not dtype_name or not payload:
                _fail(TransferProbeErrorCode.INVALID_TENSOR)
            digest.update(len(shape).to_bytes(2, "big"))
            for dimension in shape:
                if isinstance(dimension, bool) or not isinstance(dimension, int):
                    _fail(TransferProbeErrorCode.INVALID_TENSOR)
                digest.update(dimension.to_bytes(8, "big"))
            encoded_dtype = dtype_name.encode("utf-8")
            digest.update(len(encoded_dtype).to_bytes(2, "big"))
            digest.update(encoded_dtype)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        return digest.hexdigest()


TransferProbeFactory = Callable[..., GptOssTransferEvidenceProbe]

__all__ = [
    "GptOssTransferEvidenceProbe",
    "TensorBytesReader",
    "TransferProbeError",
    "TransferProbeErrorCode",
    "TransferProbeFactory",
    "load_torch_tensor_bytes_reader",
]
