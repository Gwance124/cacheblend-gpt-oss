# SPDX-License-Identifier: Apache-2.0
"""Explicit opt-in vLLM general-plugin hook for the matched CUSTOM backend.

The plugin deliberately registers only the backend.  It does not register a
model override or activate selective recomputation.  That keeps the current
g3 experiment a matched CUSTOM-at-100%-recompute control until the model
override and connector ``transfer_selective`` mode are implemented and gated.
"""

from __future__ import annotations

import os
from importlib.metadata import version

BACKEND_CLASS_PATH = (
    "cacheblend_gpt_oss.vllm_compat.v0_19_1.selective_backend."
    "GptOssCacheBlendAttentionBackend"
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


__all__ = ["register_cacheblend_backend"]
