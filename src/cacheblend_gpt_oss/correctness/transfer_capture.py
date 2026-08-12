# SPDX-License-Identifier: Apache-2.0
"""Incremental, fail-closed assembly of one transfer-evidence sidecar.

The worker-side CUDA probe owns tensor sampling and digest calculation.  This
module owns only the bounded state machine around those samples: exactly one
canonical layer in order, no mutation after finalization, and the same
top-level counts that are later bound to the correctness artifact.  Keeping
this seam dependency-free lets the probe be unit-tested without vLLM, Torch,
LMCache, or model weights.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import NoReturn

from cacheblend_gpt_oss.correctness.transfer import (
    TRANSFER_EVIDENCE_SCHEMA_VERSION,
    LayerTransferEvidence,
    TransferEvidence,
)
from cacheblend_gpt_oss.gpt_oss.layout import GPT_OSS_NUM_LAYERS

_HEX_64 = re.compile(r"[0-9a-f]{64}")


class TransferCaptureErrorCode(str, Enum):
    """Bounded capture-state failures safe for probe logs."""

    INVALID_METADATA = "invalid_metadata"
    INVALID_LAYER = "invalid_layer"
    LAYER_ORDER_MISMATCH = "layer_order_mismatch"
    INCOMPLETE_LAYERS = "incomplete_layers"
    FINALIZED = "finalized"


class TransferCaptureError(RuntimeError):
    """Capture failure without prompt, request, or tensor details."""

    def __init__(self, code: TransferCaptureErrorCode) -> None:
        self.code = code
        super().__init__(f"CacheBlend transfer capture failure: {code.value}")


def _fail(code: TransferCaptureErrorCode) -> NoReturn:
    raise TransferCaptureError(code)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True, slots=True)
class TransferEvidenceCaptureMetadata:
    """Prompt/counter identity supplied before any layer sample is captured."""

    source_prompt_digest: str
    target_prompt_digest: str
    loaded_tokens: int
    target_prompt_tokens: int
    recomputed_tokens: int
    prefill_tokens_avoided: int = 0

    def __post_init__(self) -> None:
        for digest in (self.source_prompt_digest, self.target_prompt_digest):
            if not isinstance(digest, str) or _HEX_64.fullmatch(digest) is None:
                _fail(TransferCaptureErrorCode.INVALID_METADATA)
        counts = (
            self.loaded_tokens,
            self.target_prompt_tokens,
            self.recomputed_tokens,
            self.prefill_tokens_avoided,
        )
        if any(not _is_int(value) or value < 0 for value in counts):
            _fail(TransferCaptureErrorCode.INVALID_METADATA)
        if (
            self.target_prompt_tokens == 0
            or self.loaded_tokens > self.target_prompt_tokens
            or self.recomputed_tokens != self.target_prompt_tokens
            or self.prefill_tokens_avoided != 0
        ):
            _fail(TransferCaptureErrorCode.INVALID_METADATA)


class TransferEvidenceBuilder:
    """Collect all canonical layer records before producing immutable evidence."""

    __slots__ = ("_layers", "_metadata", "_result")

    def __init__(self, metadata: TransferEvidenceCaptureMetadata) -> None:
        if not isinstance(metadata, TransferEvidenceCaptureMetadata):
            _fail(TransferCaptureErrorCode.INVALID_METADATA)
        self._metadata = metadata
        self._layers: list[LayerTransferEvidence] = []
        self._result: TransferEvidence | None = None

    @property
    def layer_count(self) -> int:
        """Return the bounded number of accepted layer samples."""

        return len(self._layers)

    @property
    def finalized(self) -> bool:
        """Return whether :meth:`finish` has produced immutable evidence."""

        return self._result is not None

    def add_layer(self, layer: LayerTransferEvidence) -> None:
        """Append exactly the next canonical layer, rejecting all mutations."""

        if self._result is not None:
            _fail(TransferCaptureErrorCode.FINALIZED)
        if not isinstance(layer, LayerTransferEvidence):
            _fail(TransferCaptureErrorCode.INVALID_LAYER)
        if layer.layer_index != len(self._layers):
            _fail(TransferCaptureErrorCode.LAYER_ORDER_MISMATCH)
        self._layers.append(layer)

    def finish(self) -> TransferEvidence:
        """Return immutable evidence once all 24 layers have been captured."""

        if self._result is not None:
            return self._result
        if len(self._layers) != GPT_OSS_NUM_LAYERS:
            _fail(TransferCaptureErrorCode.INCOMPLETE_LAYERS)
        metadata = self._metadata
        self._result = TransferEvidence(
            schema_version=TRANSFER_EVIDENCE_SCHEMA_VERSION,
            source_prompt_digest=metadata.source_prompt_digest,
            target_prompt_digest=metadata.target_prompt_digest,
            loaded_tokens=metadata.loaded_tokens,
            target_prompt_tokens=metadata.target_prompt_tokens,
            recomputed_tokens=metadata.recomputed_tokens,
            prefill_tokens_avoided=metadata.prefill_tokens_avoided,
            layers=tuple(self._layers),
        )
        return self._result


__all__ = [
    "TransferCaptureError",
    "TransferCaptureErrorCode",
    "TransferEvidenceBuilder",
    "TransferEvidenceCaptureMetadata",
]
