# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the explicit CUSTOM-backend plugin opt-in boundary."""

from __future__ import annotations

from cacheblend_gpt_oss.vllm_compat.v0_19_1.selective_plugin import (
    BACKEND_CLASS_PATH,
    MODEL_CLASS_PATH,
    register_cacheblend_backend,
)


def test_custom_backend_plugin_is_dormant_by_default(monkeypatch) -> None:
    monkeypatch.delenv("CACHEBLEND_ENABLE_CUSTOM_BACKEND", raising=False)
    register_cacheblend_backend()


def test_custom_backend_path_is_version_scoped() -> None:
    assert BACKEND_CLASS_PATH == (
        "cacheblend_gpt_oss.vllm_compat.v0_19_1.selective_backend."
        "GptOssCacheBlendAttentionBackend"
    )
    assert MODEL_CLASS_PATH == (
        "cacheblend_gpt_oss.vllm_compat.v0_19_1.selective_model:"
        "GptOssCacheBlendForCausalLM"
    )
