"""Pinned LMCache 0.4.3 Blend V2 server entry point with one safe backport.

LMCache 0.4.3 enqueues ``finish_write`` after the interprocess CUDA event in
``BlendEngineV2._cb_store_gpu_copy``.  A client can therefore observe the event,
submit the immediately following CacheBlend lookup, and have the server probe
the storage index before the just-written object is committed.  The pinned
server then removes the fresh fingerprint as stale and reports a miss.

The upstream issue analysis identified this ordering correction after the
pinned release.  The minimal ordering change is backported here without
copying or replacing LMCache: the
``finish_write`` host callback is enqueued before the event is recorded on the
same CUDA stream.  The wrapper imports LMCache lazily, patches only the exact
0.4.3 class, and delegates all argument parsing, handlers, storage, and server
lifecycle to the pinned public module.

Source attribution:

* LMCache 0.4.3 ``BlendEngineV2._cb_store_gpu_copy``:
  https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/v1/multiprocess/blend_server_v2.py#L450-L532
* Upstream race-fix change (the ordering backport only):
  https://github.com/LMCache/LMCache/pull/3179

This module is a server-process entry point.  It intentionally imports Torch
and LMCache only inside the patch/``main`` paths so CPU unit tests can import it
without GPU packages.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any, cast


def patched_store_gpu_copy(
    self: Any,
    obj_keys: list[Any],
    gpu_context: Any,
    offset: int,
    event_ipc_handle: bytes,
) -> tuple[Any, dict[Any, Any]]:
    """Copy the pinned method with the upstream store-completion ordering fix.

    ``cupy_stream`` is the external-stream view of ``gpu_context.stream`` in the
    pinned LMCache implementation.  Queueing ``finish_write`` before
    ``event.record`` makes the client's event wait cover storage-index commit,
    eliminating the fresh-store lookup race.
    """

    torch = import_module("torch")
    MemoryLayoutDesc = import_module(
        "lmcache.v1.distributed.api"
    ).MemoryLayoutDesc
    lmcache_memcpy_async_d2h = import_module(
        "lmcache.v1.gpu_connector.gpu_ops"
    ).lmcache_memcpy_async_d2h

    with (
        torch.cuda.device(gpu_context.device),
        torch.cuda.stream(gpu_context.stream),
    ):
        event = torch.cuda.Event(interprocess=True)

        vllm_event = torch.cuda.Event.from_ipc_handle(
            gpu_context.device, event_ipc_handle
        )
        vllm_event.wait(stream=gpu_context.stream)

        num_tokens = self.chunk_size
        cpu_shape = gpu_context.get_kv_buffer_shape(num_tokens)
        layout_desc = MemoryLayoutDesc(
            shapes=[cpu_shape], dtypes=[gpu_context.dtype]
        )
        reserved_dict = self.storage_manager.reserve_write(
            obj_keys, layout_desc, "new"
        )

        for index, obj_key in enumerate(obj_keys):
            if obj_key not in reserved_dict:
                continue
            memory_obj = reserved_dict[obj_key]
            offset_start = index * self.chunk_size + offset
            offset_end = offset_start + self.chunk_size
            tmp_buffer = gpu_context.get_tmp_gpu_buffer(offset_end - offset_start)
            gpu_kv_slice = gpu_context.slice_kv_cache_on_tokens(
                offset_start, offset_end
            )
            with self.lock:
                tmp_buffer.copy_(gpu_kv_slice, non_blocking=True)
                lmcache_memcpy_async_d2h(tmp_buffer, memory_obj)

        # This ordering is the complete fix.  Both operations target the same
        # underlying CUDA stream in pinned LMCache 0.4.3.
        gpu_context.cupy_stream.launch_host_func(
            self.storage_manager.finish_write,
            list(reserved_dict.keys()),
        )
        event.record()

    return event, reserved_dict


def patch_lmcache_blend_module(module: Any) -> Callable[..., Any]:
    """Patch and return the original ``BlendEngineV2`` class.

    The identity and method check make accidental use with another LMCache
    release fail closed instead of silently applying an incompatible patch.
    """

    engine_class = getattr(module, "BlendEngineV2", None)
    if engine_class is None:
        raise RuntimeError("LMCache 0.4.3 BlendEngineV2 is unavailable")
    if getattr(engine_class, "__module__", None) != module.__name__:
        raise RuntimeError("unexpected BlendEngineV2 implementation module")
    engine_class._cb_store_gpu_copy = patched_store_gpu_copy
    return cast(Callable[..., Any], engine_class)


def main() -> None:
    """Run the pinned LMCache server with the minimal race backport."""

    module = import_module("lmcache.v1.multiprocess.blend_server_v2")
    patch_lmcache_blend_module(module)
    print(
        "CacheBlend: LMCache 0.4.3 store-completion ordering backport active",
        flush=True,
    )
    args = module.parse_args()
    mp_config = module.parse_args_to_mp_server_config(args)
    storage_manager_config = module.parse_args_to_config(args)
    obs_config = module.parse_args_to_observability_config(args)
    module.run_cache_server(
        mp_config=mp_config,
        storage_manager_config=storage_manager_config,
        obs_config=obs_config,
    )


if __name__ == "__main__":
    main()
