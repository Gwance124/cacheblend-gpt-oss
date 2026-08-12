# SPDX-License-Identifier: Apache-2.0
"""Opt-in GPT-OSS-20B checkpoint configuration gate.

This test intentionally loads only the local checkpoint configuration.  It is
an early model-boundary check for the pinned adapter; loading weights and
running logits remain separate GPU/model gates in the solab-g3 runbooks.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.integration, pytest.mark.model]


def _load_text_config() -> Any:
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    model_path = os.environ.get("CACHEBLEND_MODEL_PATH")
    if not model_path:
        pytest.skip("CACHEBLEND_MODEL_PATH is required for the model gate")
    try:
        config = transformers.AutoConfig.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        pytest.fail(f"could not load the local GPT-OSS checkpoint config: {exc}")
    return getattr(config, "text_config", config)


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        pytest.fail("GPT-OSS rope_scaling must be a mapping")
    return value


def test_pinned_gpt_oss_hybrid_model_config() -> None:
    config = _load_text_config()

    assert getattr(config, "model_type", None) == "gpt_oss"
    assert getattr(config, "num_hidden_layers", None) == 24
    assert getattr(config, "num_attention_heads", None) == 64
    assert getattr(config, "num_key_value_heads", None) == 8
    assert getattr(config, "head_dim", None) == 64
    assert getattr(config, "max_position_embeddings", None) == 131_072
    assert getattr(config, "sliding_window", None) == 128
    assert getattr(config, "vocab_size", None) == 201_088
    assert getattr(config, "num_local_experts", None) == 32
    assert getattr(config, "num_experts_per_tok", None) == 4

    layer_types = tuple(getattr(config, "layer_types", ()))
    assert layer_types == (
        ("sliding_attention", "full_attention") * 12
    )

    rope_scaling = _mapping(getattr(config, "rope_scaling", None))
    assert rope_scaling.get("rope_type") == "yarn"
    assert rope_scaling.get("factor") == 32
    assert rope_scaling.get("original_max_position_embeddings") == 4096
    assert getattr(config, "rope_theta", None) == 150_000
