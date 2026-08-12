# SPDX-License-Identifier: Apache-2.0
"""Manual solab-g3 inspection of the pinned Triton selective hook boundary."""

from __future__ import annotations

import inspect
from importlib.metadata import version

import pytest

from cacheblend_gpt_oss import PINNED_TARGET


@pytest.mark.gpu
@pytest.mark.integration
def test_pinned_triton_backend_selective_hook_contract() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")

    if not torch.cuda.is_available():
        pytest.skip("manual solab-g3 test: CUDA is not available")

    from vllm.v1.attention.backends.triton_attn import (
        TritonAttentionBackend,
        TritonAttentionImpl,
    )

    assert version("vllm") == PINNED_TARGET.vllm_version
    assert torch.__version__ == PINNED_TARGET.torch_version
    assert torch.version.cuda == PINNED_TARGET.cuda_runtime
    assert torch.cuda.get_device_name(0) == PINNED_TARGET.gpu_name
    assert TritonAttentionBackend.forward_includes_kv_cache_update is False
    assert TritonAttentionBackend.supports_sink() is True
    assert TritonAttentionBackend.get_kv_cache_shape(2, 16, 8, 64) == (
        2,
        2,
        16,
        8,
        64,
    )
    parameters = list(
        inspect.signature(TritonAttentionImpl.do_kv_cache_update).parameters
    )
    assert parameters == [
        "self",
        "layer",
        "key",
        "value",
        "kv_cache",
        "slot_mapping",
    ]
