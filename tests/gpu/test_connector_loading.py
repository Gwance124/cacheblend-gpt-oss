# SPDX-License-Identifier: Apache-2.0
"""Runtime import contract for the pinned external vLLM connector."""

from __future__ import annotations

import importlib
import inspect

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def test_connector_module_loads_with_pinned_vllm() -> None:
    vllm = pytest.importorskip("vllm")
    if getattr(vllm, "__version__", None) != "0.19.1":
        pytest.fail("connector loading requires vLLM==0.19.1")

    base = importlib.import_module(
        "vllm.distributed.kv_transfer.kv_connector.v1.base"
    )
    module = importlib.import_module(
        "cacheblend_gpt_oss.vllm_compat.v0_19_1.connector"
    )
    connector = getattr(module, "GptOssCacheBlendConnector", None)
    assert connector is not None
    assert issubclass(connector, base.KVConnectorBase_V1)
    assert issubclass(connector, base.SupportsHMA)

    signature = inspect.signature(connector.__init__)
    assert tuple(signature.parameters) == (
        "self",
        "vllm_config",
        "role",
        "kv_cache_config",
    )
