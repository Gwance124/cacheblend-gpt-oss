"""Pinned LMCache 0.4.3 Blend V2 server entry point with safe backports.

LMCache 0.4.3 records the interprocess CUDA event before it enqueues the
``finish_write`` host callback in ``BlendEngineV2._cb_store_gpu_copy``.  A
client can therefore observe the event, submit the immediately following
CacheBlend lookup, and have the server probe the storage index before the
just-written object is committed.  The pinned server then removes the fresh
fingerprint as stale and reports a miss.

An initial backport placed the callback before the event on the external view
of the same CUDA stream.  A live sequential source/target gate still observed
zero storage-backed candidates.  This wrapper now takes the conservative
correctness path: synchronize the server stream after the asynchronous D2H
copies, call ``finish_write`` directly, and only then record the client-visible
event.  The first transfer milestone is intentionally correctness-first, so
the extra server-side barrier is accepted and measured rather than hidden.
The wrapper imports LMCache lazily, patches only the exact 0.4.3 class, and
delegates all argument parsing, handlers, storage, and server lifecycle to the
pinned public module.

The public matcher also uses only a truncated direct-address polynomial hash
for candidate generation.  A live GPT-OSS gate stored an exact 256-token
document successfully but returned no candidate when that document moved to
offset 17.  The wrapper therefore replaces only the matcher's in-memory
candidate index with a bounded exact-token index.  LMCache still owns object
storage, prefetch, and retrieval; the connector still independently verifies
the namespace, cache key, SHA-256 fingerprint, and complete token sequence
before loading any KV.

Pinned LMCache also initializes each ``TokenHasher`` from vLLM's
process-global ``NONE_HASH``.  With no ``PYTHONHASHSEED``, pinned vLLM assigns
that value from ``os.urandom(32)``, so the standalone server and vLLM worker
derive different object keys for identical chunks.  The wrapper and client
bindings install LMCache's deterministic local fallback seed on each hasher
instance without changing vLLM's global prefix-cache state.

Source attribution:

* LMCache 0.4.3 ``BlendEngineV2._cb_store_gpu_copy``:
  https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/v1/multiprocess/blend_server_v2.py#L450-L532
* LMCache 0.4.3 ``DistributedStorageManager.finish_write``:
  https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/v1/distributed/storage_manager.py#L164-L187
* vLLM 0.19.1 ``init_none_hash`` random-seed behavior:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/core/kv_cache_utils.py#L71-L97

This module is a server-process entry point.  It intentionally imports Torch
and LMCache only inside the patch/``main`` paths so CPU unit tests can import it
without GPU packages.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from threading import RLock
from typing import Any, cast

from cacheblend_gpt_oss.storage.lmcache_v0_4_3 import (
    pin_lmcache_blake3_none_hash,
)

_EXACT_INDEX_LIMIT = 1 << 20


def patched_token_hasher_init(self: Any, *args: Any, **kwargs: Any) -> None:
    """Initialize one server hasher with the cross-process stable seed."""

    original = getattr(type(self), "_cacheblend_original_init", None)
    if not callable(original):
        raise RuntimeError("LMCache TokenHasher original initializer is unavailable")
    original(self, *args, **kwargs)
    pin_lmcache_blake3_none_hash(self)


def patched_matcher_init(
    self: Any,
    chunk_size: int = 256,
) -> None:
    """Initialize the pinned matcher plus a bounded exact-token index."""

    original = getattr(type(self), "_cacheblend_original_init", None)
    if not callable(original):
        raise RuntimeError("LMCache matcher original initializer is unavailable")
    original(self, chunk_size)
    self._cacheblend_exact_lock = RLock()
    self._cacheblend_exact_by_first_token = {}
    self._cacheblend_exact_by_hash = {}


def patched_on_new_token_hashes(
    self: Any,
    token_ids: list[int],
    token_hashes: list[bytes],
) -> None:
    """Register complete chunks by exact tokens instead of a table slot."""

    chunk_size = getattr(self, "chunk_size", None)
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
        or any(
            isinstance(token, bool) or not isinstance(token, int) for token in token_ids
        )
        or any(
            not isinstance(token_hash, bytes) or not token_hash
            for token_hash in token_hashes
        )
    ):
        raise RuntimeError("LMCache exact matcher received invalid chunk data")
    complete_chunks = len(token_ids) // chunk_size
    if len(token_hashes) != complete_chunks:
        raise RuntimeError("LMCache exact matcher chunk/hash count mismatch")

    additions: list[tuple[bytes, tuple[int, ...], int]] = []
    for chunk_index, token_hash in enumerate(token_hashes):
        start = chunk_index * chunk_size
        chunk = tuple(token_ids[start : start + chunk_size])
        if len(chunk) != chunk_size:
            raise RuntimeError("LMCache exact matcher received a partial chunk")
        additions.append((token_hash, chunk, start))

    lock = getattr(self, "_cacheblend_exact_lock", None)
    by_first = getattr(self, "_cacheblend_exact_by_first_token", None)
    by_hash = getattr(self, "_cacheblend_exact_by_hash", None)
    if lock is None or not isinstance(by_first, dict) or not isinstance(by_hash, dict):
        raise RuntimeError("LMCache exact matcher is not initialized")
    with lock:
        new_hashes = {token_hash for token_hash, _, _ in additions} - set(by_hash)
        if len(by_hash) + len(new_hashes) > _EXACT_INDEX_LIMIT:
            raise RuntimeError("LMCache exact matcher capacity exceeded")
        for token_hash, chunk, start in additions:
            existing = by_hash.get(token_hash)
            if existing is not None and existing != (chunk, start):
                raise RuntimeError("LMCache exact matcher hash identity conflict")
        for token_hash, chunk, start in additions:
            by_hash[token_hash] = (chunk, start)
            bucket = by_first.setdefault(chunk[0], {})
            bucket[token_hash] = (chunk, start)
        indexed_chunks = len(by_hash)
    logger = getattr(type(self), "_cacheblend_logger", None)
    if logger is not None:
        logger.info(
            "CacheBlend exact matcher registered %d chunks; indexed chunks: %d",
            len(additions),
            indexed_chunks,
        )


def patched_match_sub_sequence(
    self: Any,
    token_ids: list[int],
) -> list[Any]:
    """Return first exact query occurrence for every stored chunk hash."""

    chunk_size = getattr(self, "chunk_size", None)
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
        or any(
            isinstance(token, bool) or not isinstance(token, int) for token in token_ids
        )
    ):
        raise RuntimeError("LMCache exact matcher received invalid query tokens")
    query_windows = max(0, len(token_ids) - chunk_size + 1)
    if not query_windows:
        logger = getattr(type(self), "_cacheblend_logger", None)
        if logger is not None:
            logger.info(
                "CacheBlend exact matcher searched 0 windows; exact matches: 0"
            )
        return []
    lock = getattr(self, "_cacheblend_exact_lock", None)
    by_first = getattr(self, "_cacheblend_exact_by_first_token", None)
    match_type = getattr(type(self), "_cacheblend_match_type", None)
    if lock is None or not isinstance(by_first, dict) or not callable(match_type):
        raise RuntimeError("LMCache exact matcher is not initialized")

    with lock:
        snapshot = {
            first_token: tuple(bucket.items())
            for first_token, bucket in by_first.items()
        }
    results: list[Any] = []
    seen_hashes: set[bytes] = set()
    for query_start in range(len(token_ids) - chunk_size + 1):
        bucket = snapshot.get(token_ids[query_start], ())
        if not bucket:
            continue
        query_chunk: tuple[int, ...] | None = None
        for token_hash, (stored_chunk, source_start) in bucket:
            if token_hash in seen_hashes:
                continue
            if query_chunk is None:
                query_chunk = tuple(token_ids[query_start : query_start + chunk_size])
            if query_chunk != stored_chunk:
                continue
            results.append(
                match_type(
                    old_st=source_start,
                    old_ed=source_start + chunk_size,
                    cur_st=query_start,
                    cur_ed=query_start + chunk_size,
                    hash=token_hash,
                )
            )
            seen_hashes.add(token_hash)
    logger = getattr(type(self), "_cacheblend_logger", None)
    if logger is not None:
        logger.info(
            "CacheBlend exact matcher searched %d windows; exact matches: %d",
            query_windows,
            len(results),
        )
    return results


def patched_remove_chunks(self: Any, token_hashes: list[bytes]) -> None:
    """Remove evicted object hashes from the exact-token index."""

    if any(
        not isinstance(token_hash, bytes) or not token_hash
        for token_hash in token_hashes
    ):
        raise RuntimeError("LMCache exact matcher received invalid eviction hashes")
    lock = getattr(self, "_cacheblend_exact_lock", None)
    by_first = getattr(self, "_cacheblend_exact_by_first_token", None)
    by_hash = getattr(self, "_cacheblend_exact_by_hash", None)
    if lock is None or not isinstance(by_first, dict) or not isinstance(by_hash, dict):
        raise RuntimeError("LMCache exact matcher is not initialized")
    with lock:
        for token_hash in token_hashes:
            removed = by_hash.pop(token_hash, None)
            if removed is None:
                continue
            chunk, _ = removed
            bucket = by_first.get(chunk[0])
            if bucket is None:
                continue
            bucket.pop(token_hash, None)
            if not bucket:
                by_first.pop(chunk[0], None)


def patched_store_gpu_copy(
    self: Any,
    obj_keys: list[Any],
    gpu_context: Any,
    offset: int,
    event_ipc_handle: bytes,
) -> tuple[Any, dict[Any, Any]]:
    """Publish copied chunks before recording the client-visible event.

    The pinned D2H helper is explicitly asynchronous.  Synchronizing the
    server stream makes those copies complete before the direct
    ``finish_write`` call publishes the reserved objects to L1.  Recording the
    interprocess event last makes a successful client wait strong evidence
    that both the bytes and storage index are visible.
    """

    torch = import_module("torch")
    MemoryLayoutDesc = import_module("lmcache.v1.distributed.api").MemoryLayoutDesc
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
        layout_desc = MemoryLayoutDesc(shapes=[cpu_shape], dtypes=[gpu_context.dtype])
        reserved_dict = self.storage_manager.reserve_write(obj_keys, layout_desc, "new")

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

        # Correctness-first publication barrier for the 100%-recompute gate.
        # The pinned copy helper does not synchronize the stream.
        gpu_context.stream.synchronize()
        self.storage_manager.finish_write(list(reserved_dict.keys()))
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
    matcher_class = getattr(module, "BlendTokenRangeMatcher", None)
    match_type = getattr(module, "CBMatchResult", None)
    if matcher_class is None or not callable(match_type):
        raise RuntimeError("LMCache 0.4.3 BlendTokenRangeMatcher is unavailable")
    if getattr(matcher_class, "__module__", None) != module.__name__:
        raise RuntimeError("unexpected BlendTokenRangeMatcher implementation module")
    original_init = getattr(matcher_class, "__init__", None)
    if not callable(original_init):
        raise RuntimeError("LMCache matcher initializer is unavailable")
    token_hasher_class = getattr(module, "TokenHasher", None)
    if token_hasher_class is None:
        raise RuntimeError("LMCache 0.4.3 TokenHasher is unavailable")
    token_hasher_init = getattr(token_hasher_class, "__init__", None)
    if not callable(token_hasher_init):
        raise RuntimeError("LMCache TokenHasher initializer is unavailable")
    token_hasher_class._cacheblend_original_init = token_hasher_init
    token_hasher_class.__init__ = patched_token_hasher_init
    matcher_class._cacheblend_original_init = original_init
    matcher_class._cacheblend_match_type = match_type
    matcher_class._cacheblend_logger = getattr(module, "logger", None)
    matcher_class.__init__ = patched_matcher_init
    matcher_class.on_new_token_hashes = patched_on_new_token_hashes
    matcher_class.match_sub_sequence = patched_match_sub_sequence
    matcher_class.remove_chunks = patched_remove_chunks
    engine_class._cb_store_gpu_copy = patched_store_gpu_copy
    return cast(Callable[..., Any], engine_class)


def main() -> None:
    """Run the pinned server with store ordering and exact matching patched."""

    module = import_module("lmcache.v1.multiprocess.blend_server_v2")
    patch_lmcache_blend_module(module)
    print(
        "CacheBlend: LMCache 0.4.3 deterministic-hash, store-order, and "
        "exact-matcher backports active",
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
