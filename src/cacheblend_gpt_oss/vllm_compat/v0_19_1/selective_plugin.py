# SPDX-License-Identifier: Apache-2.0
"""Explicit opt-in vLLM general-plugin hook for the matched CUSTOM backend.

The plugin always registers the matched CUSTOM backend when explicitly enabled.
It optionally registers the GPT-OSS model wrapper when
``CACHEBLEND_ENABLE_CUSTOM_MODEL=1``.  The wrapper preserves full recompute
for ``transfer_100pct`` and activates row-selective MLP work only when the
connector has installed an explicit ``transfer_selective`` plan.
"""

from __future__ import annotations

import os
from importlib.metadata import version

BACKEND_CLASS_PATH = (
    "cacheblend_gpt_oss.vllm_compat.v0_19_1.selective_backend."
    "GptOssCacheBlendAttentionBackend"
)
MODEL_CLASS_PATH = (
    "cacheblend_gpt_oss.vllm_compat.v0_19_1.selective_model:"
    "GptOssCacheBlendForCausalLM"
)


def register_cacheblend_backend() -> None:
    """Register CUSTOM only when the operator explicitly opts in."""

    if os.environ.get("CACHEBLEND_ENABLE_CUSTOM_BACKEND") != "1":
        return
    if version("vllm") != "0.19.1":
        raise RuntimeError("CacheBlend CUSTOM backend requires vLLM 0.19.1")

    from vllm.v1.attention.backends.registry import (  # type: ignore[import-not-found]
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(AttentionBackendEnum.CUSTOM, BACKEND_CLASS_PATH)

    if os.environ.get("CACHEBLEND_ENABLE_CUSTOM_MODEL") == "1":
        from vllm.model_executor.models import (  # type: ignore[import-not-found]
            ModelRegistry,
        )

        ModelRegistry.register_model("GptOssForCausalLM", MODEL_CLASS_PATH)


__all__ = ["BACKEND_CLASS_PATH", "MODEL_CLASS_PATH", "register_cacheblend_backend"]
