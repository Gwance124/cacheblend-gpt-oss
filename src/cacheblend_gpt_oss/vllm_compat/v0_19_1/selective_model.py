# SPDX-License-Identifier: Apache-2.0
"""Lazy pinned GPT-OSS model wrapper for full and selective plans.

The wrapper keeps the pinned top-level ``GptOssForCausalLM`` contract.  With no
connector-installed plan it binds a full plan and delegates byte-for-byte to
the parent.  With ``transfer_selective`` it temporarily routes the pinned
inner ``GptOssModel`` loop through a full-shaped attention path that skips the
MoE/MLP call for cached rows after the check layer.  Attention still receives
all rows, and the CUSTOM backend writes only selected K/V rows, so the first
selective arm preserves the hybrid cache and learned-sink shape invariants.

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
    import torch  # type: ignore[import-not-found]
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


def _selective_inner_model_forward(
    model: Any,
    input_ids: Any,
    positions: Any,
    intermediate_tensors: Any = None,
    inputs_embeds: Any = None,
) -> Any:
    """Run the pinned GPT-OSS model loop with selected-row MLP work.

    The loop mirrors the pinned ``GptOssModel.forward`` implementation and
    deliberately keeps a full ``[prompt, hidden]`` tensor at every layer.
    Cached rows are not needed to produce later selected-row queries because
    their verified K/V is already present in the paged cache; their hidden
    rows therefore become zero placeholders until the final norm.  The
    configured suffix guarantees that the final sampling row is always real
    recompute work.

    Pinned source:
    https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/models/gpt_oss.py#L200-L245
    """

    if intermediate_tensors is not None or inputs_embeds is not None:
        raise RuntimeError(
            "CacheBlend selective model requires the pinned single-stage token path"
        )
    if input_ids is None or getattr(input_ids, "ndim", None) != 1:
        raise RuntimeError("CacheBlend selective model requires flattened token IDs")
    plan = ForwardRowPlanContext.current()
    if plan.prompt_tokens != int(input_ids.shape[0]):
        raise RuntimeError("CacheBlend selective plan does not match prompt rows")

    hidden_states = model.embed_input_ids(input_ids)
    residual = None
    for layer_index in range(model.start_layer, model.end_layer):
        layer = model.layers[layer_index]
        selection = plan.layer(layer_index)
        if selection.is_full_recompute:
            hidden_states, residual = layer(hidden_states, positions, residual)
            continue

        if residual is None:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
        else:
            hidden_states, residual = layer.input_layernorm(
                hidden_states, residual
            )
        hidden_states = layer.attn(hidden_states, positions)
        hidden_states, residual = layer.post_attention_layernorm(
            hidden_states, residual
        )

        selected_positions = torch.tensor(
            selection.recompute_positions,
            dtype=torch.long,
            device=hidden_states.device,
        )
        if selected_positions.numel() == hidden_states.shape[0]:
            hidden_states = layer.mlp(hidden_states)
        elif selected_positions.numel() == 0:
            hidden_states = torch.zeros_like(hidden_states)
        else:
            selected_hidden = hidden_states.index_select(0, selected_positions)
            selected_output = layer.mlp(selected_hidden)
            output = torch.zeros_like(hidden_states)
            hidden_states = output.index_copy_(
                0, selected_positions, selected_output
            )

    hidden_states, _ = model.norm(hidden_states, residual)
    return hidden_states


class GptOssCacheBlendForCausalLM(
    _PinnedGptOssForCausalLM,  # type: ignore[misc]
):
    """Pinned GPT-OSS with an explicit full or selective row plan."""

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

        active_plan = ForwardRowPlanContext.current_or_none()
        if active_plan is None:
            plan = ForwardRowPlan.full_recompute(int(input_ids.shape[0]))
            with ForwardRowPlanContext.bind(plan):
                return super().forward(
                    input_ids,
                    positions,
                    intermediate_tensors,
                    inputs_embeds,
                )

        if active_plan.prompt_tokens != int(input_ids.shape[0]):
            raise RuntimeError(
                "CacheBlend model wrapper received a mismatched forward plan"
            )

        original_model_forward = self.model.forward

        def selective_forward(
            inner_input_ids: Any,
            inner_positions: Any,
            inner_intermediate_tensors: Any = None,
            inner_inputs_embeds: Any = None,
        ) -> Any:
            return _selective_inner_model_forward(
                self.model,
                inner_input_ids,
                inner_positions,
                inner_intermediate_tensors,
                inner_inputs_embeds,
            )

        self.model.forward = selective_forward
        try:
            return super().forward(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
            )
        finally:
            self.model.forward = original_model_forward


__all__ = ["MODEL_CLASS_PATH", "GptOssCacheBlendForCausalLM"]
