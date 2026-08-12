"""CPU-only contract tests for the pinned out-of-tree vLLM connector."""

from __future__ import annotations

import builtins
import importlib
import inspect
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

MODULE_NAME = "cacheblend_gpt_oss.vllm_compat.v0_19_1.connector"


def test_import_without_vllm_fails_with_actionable_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, MODULE_NAME, raising=False)
    real_import = builtins.__import__

    def import_without_vllm(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "vllm" or name.startswith("vllm."):
            raise ModuleNotFoundError("vllm deliberately hidden by contract test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_vllm)
    with pytest.raises(RuntimeError, match=r"requires the pinned vLLM==0\.19\.1"):
        importlib.import_module(MODULE_NAME)
    monkeypatch.delitem(sys.modules, MODULE_NAME, raising=False)


def _install_fake_vllm(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    class FakeKVConnectorRole(Enum):
        SCHEDULER = 0
        WORKER = 1

    class FakeKVConnectorMetadata:
        pass

    class FakeSupportsHMA:
        pass

    class FakeKVConnectorBase:
        def __init__(self, vllm_config: Any, role: Any, kv_cache_config: Any) -> None:
            self._vllm_config = vllm_config
            self._kv_transfer_config = vllm_config.kv_transfer_config
            self._kv_cache_config = kv_cache_config
            self._role = role
            self._connector_metadata: Any = None

        @property
        def role(self) -> Any:
            return self._role

        def bind_connector_metadata(self, metadata: Any) -> None:
            self._connector_metadata = metadata

        def _get_connector_metadata(self) -> Any:
            assert self._connector_metadata is not None
            return self._connector_metadata

    modules = {
        "vllm": ModuleType("vllm"),
        "vllm.distributed": ModuleType("vllm.distributed"),
        "vllm.distributed.kv_transfer": ModuleType("vllm.distributed.kv_transfer"),
        "vllm.distributed.kv_transfer.kv_connector": ModuleType(
            "vllm.distributed.kv_transfer.kv_connector"
        ),
        "vllm.distributed.kv_transfer.kv_connector.v1": ModuleType(
            "vllm.distributed.kv_transfer.kv_connector.v1"
        ),
        "vllm.distributed.kv_transfer.kv_connector.v1.base": ModuleType(
            "vllm.distributed.kv_transfer.kv_connector.v1.base"
        ),
    }
    modules["vllm"].__version__ = "0.19.1"  # type: ignore[attr-defined]
    for name, module in modules.items():
        if name != "vllm.distributed.kv_transfer.kv_connector.v1.base":
            module.__path__ = []  # type: ignore[attr-defined]

    base = modules["vllm.distributed.kv_transfer.kv_connector.v1.base"]
    base.KVConnectorBase_V1 = FakeKVConnectorBase  # type: ignore[attr-defined]
    base.KVConnectorMetadata = FakeKVConnectorMetadata  # type: ignore[attr-defined]
    base.KVConnectorRole = FakeKVConnectorRole  # type: ignore[attr-defined]
    base.SupportsHMA = FakeSupportsHMA  # type: ignore[attr-defined]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    return SimpleNamespace(
        base=FakeKVConnectorBase,
        metadata=FakeKVConnectorMetadata,
        role=FakeKVConnectorRole,
        supports_hma=FakeSupportsHMA,
    )


@pytest.fixture
def loaded_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ModuleType, SimpleNamespace]:
    monkeypatch.delitem(sys.modules, MODULE_NAME, raising=False)
    fake = _install_fake_vllm(monkeypatch)
    module = importlib.import_module(MODULE_NAME)
    try:
        yield module, fake
    finally:
        sys.modules.pop(MODULE_NAME, None)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
    )


def _kv_cache_config() -> SimpleNamespace:
    return SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=["model.layers.0.attn"]),
            SimpleNamespace(layer_names=["model.layers.1.attn"]),
        ]
    )


def test_exact_runtime_shape_and_hma_contract(
    loaded_connector: tuple[ModuleType, SimpleNamespace],
) -> None:
    module, fake = loaded_connector
    connector_type = module.GptOssCacheBlendConnector

    assert issubclass(connector_type, fake.base)
    assert issubclass(connector_type, fake.supports_hma)
    assert list(inspect.signature(connector_type.__init__).parameters) == [
        "self",
        "vllm_config",
        "role",
        "kv_cache_config",
    ]


def test_scheduler_records_all_groups_while_recomputing_every_token(
    loaded_connector: tuple[ModuleType, SimpleNamespace],
) -> None:
    module, fake = loaded_connector
    connector = module.GptOssCacheBlendConnector(
        _config(), fake.role.SCHEDULER, _kv_cache_config()
    )
    request = SimpleNamespace(request_id="request-1")
    blocks = SimpleNamespace(get_block_ids=lambda: ([3, 4], [11, 12]))

    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    connector.update_state_after_alloc(request, blocks, num_external_tokens=0)
    metadata = connector.build_connector_meta(SimpleNamespace())

    assert isinstance(metadata, fake.metadata)
    assert metadata.transfer_enabled is False
    assert metadata.group_layer_names == (
        ("model.layers.0.attn",),
        ("model.layers.1.attn",),
    )
    assert len(metadata.allocations) == 1
    assert metadata.allocations[0].block_ids_by_group == ((3, 4), (11, 12))
    assert metadata.allocations[0].num_external_tokens == 0
    assert connector.build_connector_meta(SimpleNamespace()).allocations == ()

    with pytest.raises(RuntimeError, match="must report zero external tokens"):
        connector.update_state_after_alloc(request, blocks, num_external_tokens=1)
    assert connector.request_finished_all_groups(request, ([3], [11])) == (
        False,
        None,
    )


def test_worker_registers_every_layer_and_rejects_transfer_claims(
    loaded_connector: tuple[ModuleType, SimpleNamespace],
) -> None:
    module, fake = loaded_connector
    connector = module.GptOssCacheBlendConnector(
        _config(), fake.role.WORKER, _kv_cache_config()
    )
    caches = {
        "model.layers.0.attn": object(),
        "model.layers.1.attn": object(),
    }
    connector.register_kv_caches(caches)
    metadata = module.GptOssCacheBlendMetadata(
        schema_version=1,
        group_layer_names=(
            ("model.layers.0.attn",),
            ("model.layers.1.attn",),
        ),
        allocations=(
            module.CacheBlendAllocation("request-1", ((3,), (11,)), 0),
        ),
    )
    connector.bind_connector_metadata(metadata)
    connector.start_load_kv(SimpleNamespace())
    connector.wait_for_layer_load("model.layers.0.attn")
    connector.save_kv_layer("model.layers.1.attn", object(), object())
    connector.wait_for_save()
    assert connector.get_finished(set()) == (None, None)
    assert connector.get_block_ids_with_load_errors() == set()

    bad_metadata = module.GptOssCacheBlendMetadata(
        schema_version=1,
        group_layer_names=metadata.group_layer_names,
        allocations=metadata.allocations,
        transfer_enabled=True,
    )
    connector.bind_connector_metadata(bad_metadata)
    with pytest.raises(RuntimeError, match="does not implement KV transfer"):
        connector.start_load_kv(SimpleNamespace())


def test_source_contains_the_pinned_loader_class_name() -> None:
    source = Path(
        "src/cacheblend_gpt_oss/vllm_compat/v0_19_1/connector.py"
    ).read_text(encoding="utf-8")
    assert "class GptOssCacheBlendConnector" in source
    assert "KVConnectorBase_V1, SupportsHMA" in source
    assert "return 0, False" in source
