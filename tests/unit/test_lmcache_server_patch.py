"""CPU-only tests for the pinned LMCache store-order backport."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

import cacheblend_gpt_oss.storage.lmcache_server_v0_4_3 as patch_module
from cacheblend_gpt_oss.storage.lmcache_server_v0_4_3 import (
    patch_lmcache_blend_module,
    patched_store_gpu_copy,
)


def test_patch_replaces_only_the_expected_engine_method() -> None:
    class BlendEngineV2:
        __module__ = "lmcache.v1.multiprocess.blend_server_v2"

        def _cb_store_gpu_copy(self) -> None:
            pass

    module = SimpleNamespace(
        __name__="lmcache.v1.multiprocess.blend_server_v2",
        BlendEngineV2=BlendEngineV2,
    )

    patched = patch_lmcache_blend_module(module)

    assert patched is BlendEngineV2
    assert cast(object, BlendEngineV2._cb_store_gpu_copy) is patched_store_gpu_copy


def test_patch_fails_closed_for_another_engine_module() -> None:
    class BlendEngineV2:
        __module__ = "lmcache.other_release.blend_server_v2"

    module = SimpleNamespace(
        __name__="lmcache.v1.multiprocess.blend_server_v2",
        BlendEngineV2=BlendEngineV2,
    )

    with pytest.raises(RuntimeError, match="unexpected BlendEngineV2"):
        patch_lmcache_blend_module(module)


def test_patch_fails_closed_when_engine_is_missing() -> None:
    module = SimpleNamespace(__name__="lmcache.v1.multiprocess.blend_server_v2")

    with pytest.raises(RuntimeError, match="unavailable"):
        patch_lmcache_blend_module(module)


def test_store_completion_is_queued_before_client_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    class Context:
        def __enter__(self) -> Context:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class Event:
        def __init__(self, *, interprocess: bool) -> None:
            assert interprocess

        @classmethod
        def from_ipc_handle(cls, device: object, handle: bytes) -> Event:
            assert device == "cuda:0" and handle == b"vllm-event"
            return cls(interprocess=True)

        def wait(self, *, stream: object) -> None:
            assert stream == "stream"

        def record(self) -> None:
            order.append("event")

    class Cuda:
        @staticmethod
        def device(device: object) -> Context:
            assert device == "cuda:0"
            return Context()

        @staticmethod
        def stream(stream: object) -> Context:
            assert stream == "stream"
            return Context()

    Cuda.Event = Event  # type: ignore[attr-defined]

    class Torch:
        cuda = Cuda()

    class ExternalStream:
        def launch_host_func(self, callback: object, keys: list[object]) -> None:
            assert getattr(callback, "__self__", None) is storage
            assert getattr(callback, "__name__", None) == "finish_write"
            assert keys == ["key"]
            order.append("finish_write")

    class Storage:
        def reserve_write(
            self, keys: list[object], layout: object, policy: str
        ) -> dict[object, object]:
            assert keys == ["key"] and policy == "new"
            return {"key": "memory"}

        def finish_write(self, keys: list[object]) -> None:
            assert keys == ["key"]

    class Lock:
        def __enter__(self) -> Lock:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class Buffer:
        def copy_(self, value: object, *, non_blocking: bool) -> None:
            assert value == "slice" and non_blocking

    class GpuContext:
        device = "cuda:0"
        stream = "stream"
        cupy_stream = ExternalStream()
        dtype = "bf16"

        @staticmethod
        def get_kv_buffer_shape(tokens: int) -> tuple[int]:
            assert tokens == 256
            return (tokens,)

        @staticmethod
        def get_tmp_gpu_buffer(tokens: int) -> Buffer:
            assert tokens == 256
            return Buffer()

        @staticmethod
        def slice_kv_cache_on_tokens(start: int, end: int) -> str:
            assert (start, end) == (0, 256)
            return "slice"

    class MemoryLayoutDesc:
        def __init__(self, *, shapes: list[tuple[int]], dtypes: list[str]) -> None:
            assert shapes == [(256,)] and dtypes == ["bf16"]

    def fake_import(name: str) -> object:
        if name == "torch":
            return Torch()
        if name == "lmcache.v1.distributed.api":
            return SimpleNamespace(MemoryLayoutDesc=MemoryLayoutDesc)
        if name == "lmcache.v1.gpu_connector.gpu_ops":
            return SimpleNamespace(
                lmcache_memcpy_async_d2h=lambda buffer, memory: order.append(
                    "copy"
                )
            )
        raise AssertionError(name)

    monkeypatch.setattr(patch_module, "import_module", fake_import)
    storage = Storage()
    engine = SimpleNamespace(
        chunk_size=256,
        storage_manager=storage,
        lock=Lock(),
    )

    patched_store_gpu_copy(
        engine,
        ["key"],
        GpuContext(),
        0,
        b"vllm-event",
    )

    assert order == ["copy", "finish_write", "event"]
