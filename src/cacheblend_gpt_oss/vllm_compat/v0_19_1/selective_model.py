# SPDX-License-Identifier: Apache-2.0
"""Lazy pinned GPT-OSS model wrapper for the full-plan control.

The wrapper is deliberately a matched-control seam, not selective execution:
it binds a full 24-layer :class:`ForwardRowPlan` around the pinned
``GptOssForCausalLM.forward`` call.  The CUSTOM attention backend therefore
sees an explicit plan while every prompt row is still written exactly as in
ordinary full prefill.  Later work can replace the full plan with a validated
selective plan after the connector and model-row computation contracts exist.

The module is imported only through vLLM's lazy model registry on the pinned
GPU runtime.  It must stay out of CPU-only package imports.

Pinned source contract:

* ``GptOssForCausalLM.forward`` accepts ``(input_ids, positions,
  intermediate_tensors=None, inputs_embeds=None)``:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/models/gpt_oss.py#L1190-L1203
* ``ModelRegistry.register_model`` accepts a lazy ``<module>:<class>`` path:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/models/registry.py#L894-L938
"""

from __future__ import annotations

from typing import Any

from cacheblend_gpt_oss.gpt_oss.selective import (
    ForwardRowPlan,
    ForwardRowPlanContext,
)

try:
    from vllm.model_executor.models.gpt_oss import (  # type: ignore[import-not-found]
        GptOssForCausalLM as _PinnedGptOssForCausalLM,
    )
except ImportError as error:  # pragma: no cover - exercised on GPU only
    raise RuntimeError(
        "CacheBlend model wrapper requires the pinned vLLM 0.19.1 runtime"
    ) from error


MODEL_CLASS_PATH = (
    "cacheblend_gpt_oss.vllm_compat.v0_19_1.selective_model:"
    "GptOssCacheBlendForCausalLM"
)


class GptOssCacheBlendForCausalLM(
    _PinnedGptOssForCausalLM,  # type: ignore[misc]
):
    """Pinned GPT-OSS with an explicit full-plan forward context."""

    def forward(
        self,
        input_ids: Any,
        positions: Any,
        intermediate_tensors: Any = None,
        inputs_embeds: Any = None,
    ) -> Any:
        if input_ids is None:
            raise RuntimeError(
                "CacheBlend model wrapper requires token IDs for the full-plan control"
            )
        if inputs_embeds is not None:
            raise RuntimeError(
                "CacheBlend model wrapper does not support prompt embeddings"
            )
        if getattr(input_ids, "ndim", None) != 1:
            raise RuntimeError(
                "CacheBlend model wrapper requires flattened token IDs"
            )

        plan = ForwardRowPlan.full_recompute(int(input_ids.shape[0]))
        with ForwardRowPlanContext.bind(plan):
            return super().forward(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
            )


__all__ = ["MODEL_CLASS_PATH", "GptOssCacheBlendForCausalLM"]
