"""Pinned vLLM V1 connector boundary for the 100%-recompute milestone.

This module intentionally performs no KV transfer.  It proves that vLLM can
load the project out of tree, propagate allocation metadata for every hybrid
KV-cache group, and execute the ordinary full prefill.  Loading and saving are
synchronous no-ops until the planner and storage data plane are connected.

The API references below are pinned to vLLM 0.19.1 commit
``b1388b1fbf5aaef47937fabe98931211684666a6``:

* ``KVConnectorBase_V1``, ``KVConnectorMetadata``, ``KVConnectorRole``, and
  ``SupportsHMA``:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L82-L207
* Scheduler lookup/allocation/metadata hooks:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L449-L524
* ``KVCacheBlocks.get_block_ids`` and its grouped return value:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/core/kv_cache_manager.py#L22-L76
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cacheblend_gpt_oss.vllm_compat.v0_19_1.adapters import (
    AdaptedKvCacheConfig,
    adapt_kv_cache_blocks,
    adapt_kv_cache_config,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.config_validation import (
    require_pinned_config,
)

try:
    from vllm import __version__ as _VLLM_VERSION  # type: ignore[import-not-found]
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # type: ignore[import-not-found]
        KVConnectorBase_V1,
        KVConnectorMetadata,
        KVConnectorRole,
        SupportsHMA,
    )
except ImportError as exc:  # pragma: no cover - exact message tested in isolation
    raise RuntimeError(
        "GptOssCacheBlendConnector requires the pinned vLLM==0.19.1 runtime; "
        "the lightweight CPU-only package does not install vLLM by default."
    ) from exc

if TYPE_CHECKING:
    import torch  # type: ignore[import-not-found]
    from vllm.config import VllmConfig  # type: ignore[import-not-found]
    from vllm.forward_context import ForwardContext  # type: ignore[import-not-found]
    from vllm.v1.attention.backend import (  # type: ignore[import-not-found]
        AttentionMetadata,
    )
    from vllm.v1.core.kv_cache_manager import (  # type: ignore[import-not-found]
        KVCacheBlocks,
    )
    from vllm.v1.core.sched.output import (  # type: ignore[import-not-found]
        SchedulerOutput,
    )
    from vllm.v1.kv_cache_interface import (  # type: ignore[import-not-found]
        KVCacheConfig,
    )
    from vllm.v1.outputs import KVConnectorOutput  # type: ignore[import-not-found]
    from vllm.v1.request import Request  # type: ignore[import-not-found]

_SUPPORTED_VLLM_VERSION = "0.19.1"
_METADATA_SCHEMA_VERSION = 1


def _v2_model_runner_enabled() -> bool:
    """Mirror the pinned vLLM environment flag without importing GPU workers."""

    raw_value = os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "0")
    try:
        return bool(int(raw_value))
    except ValueError as exc:
        raise RuntimeError(
            "VLLM_USE_V2_MODEL_RUNNER must be an integer and must remain 0 "
            "for the pinned CacheBlend connector."
        ) from exc


@dataclass(frozen=True, slots=True)
class CacheBlendAllocation:
    """Scheduler allocation captured without crediting externally cached tokens."""

    request_id: str
    block_ids_by_group: tuple[tuple[int, ...], ...]
    num_external_tokens: int


@dataclass(frozen=True, slots=True)
class GptOssCacheBlendMetadata(KVConnectorMetadata):  # type: ignore[misc]
    """Opaque scheduler-to-worker metadata for the control-flow-only milestone."""

    schema_version: int
    group_layer_names: tuple[tuple[str, ...], ...]
    allocations: tuple[CacheBlendAllocation, ...]
    transfer_enabled: bool = False


class GptOssCacheBlendConnector(
    KVConnectorBase_V1, SupportsHMA  # type: ignore[misc]
):
    """vLLM 0.19.1 connector skeleton that always performs full prefill.

    The connector returns zero external matches, never copies or persists KV,
    and never assumes ownership of vLLM blocks.  Any attempt to credit external
    tokens or use the non-HMA completion hook fails closed.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None,
    ) -> None:
        if _VLLM_VERSION != _SUPPORTED_VLLM_VERSION:
            raise RuntimeError(
                "GptOssCacheBlendConnector supports only vLLM==0.19.1; "
                f"found {_VLLM_VERSION!r}."
            )
        if kv_cache_config is None:
            raise RuntimeError(
                "GptOssCacheBlendConnector requires the vLLM 0.19.1 "
                "three-argument constructor with a finalized KVCacheConfig."
            )

        require_pinned_config(
            vllm_config,
            kv_cache_config,
            v2_model_runner_enabled=_v2_model_runner_enabled(),
        )

        # vLLM otherwise defaults to disabling HMA whenever a connector is set.
        # Pinned source:
        # https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/config/vllm.py#L1227-L1247
        if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager is not False:
            raise RuntimeError(
                "GPT-OSS CacheBlend requires vLLM's hybrid KV-cache manager; "
                "start vLLM with --no-disable-hybrid-kv-cache-manager."
            )

        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )

        # Translate the same finalized object again through the strict runtime
        # adapter.  The configuration validator above reports operator-facing
        # startup issues; this adapter also constructs immutable group/layout
        # descriptors used for every later block-table translation.
        self._adapted_kv_cache_config: AdaptedKvCacheConfig = (
            adapt_kv_cache_config(kv_cache_config)
        )
        self._group_layer_names = (
            self._adapted_kv_cache_config.control_plane_layout.layer_names_by_group
        )
        self._pending_allocations: dict[str, CacheBlendAllocation] = {}
        self._registered_kv_caches: dict[str, torch.Tensor] = {}

    def _require_role(self, expected: KVConnectorRole, operation: str) -> None:
        if self.role is not expected:
            raise RuntimeError(
                f"{operation} is valid only for the {expected.name.lower()} "
                f"connector; this instance has role {self.role.name.lower()}."
            )

    def _validate_group_count(self, block_ids: tuple[Any, ...]) -> None:
        expected = len(self._group_layer_names)
        if len(block_ids) != expected:
            raise RuntimeError(
                "KV-cache group mismatch: connector was initialized for "
                f"{expected} groups but received {len(block_ids)}."
            )

    # Worker-side hooks. vLLM registers per-layer tensors here:
    # https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/worker/gpu_model_runner.py#L6809-L6819
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._require_role(KVConnectorRole.WORKER, "register_kv_caches")
        expected_names = {
            name for group_names in self._group_layer_names for name in group_names
        }
        actual_names = set(kv_caches)
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            unexpected = sorted(actual_names - expected_names)
            raise RuntimeError(
                "Registered KV-cache layers do not match KVCacheConfig; "
                f"missing={missing}, unexpected={unexpected}."
            )
        # Retain every layer tensor reference. No tensor is read or written yet.
        self._registered_kv_caches = dict(kv_caches)

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        self._require_role(KVConnectorRole.WORKER, "start_load_kv")
        del forward_context, kwargs
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, GptOssCacheBlendMetadata):
            raise RuntimeError("Received metadata from an incompatible connector.")
        if metadata.schema_version != _METADATA_SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported connector metadata schema {metadata.schema_version}."
            )
        if metadata.group_layer_names != self._group_layer_names:
            raise RuntimeError("Scheduler and worker KV-cache groups do not match.")
        if metadata.transfer_enabled:
            raise RuntimeError(
                "This connector milestone does not implement KV transfer."
            )
        for allocation in metadata.allocations:
            self._validate_group_count(allocation.block_ids_by_group)
            if allocation.num_external_tokens != 0:
                raise RuntimeError(
                    "The 100%-recompute milestone cannot load external tokens."
                )

    def wait_for_layer_load(self, layer_name: str) -> None:
        self._require_role(KVConnectorRole.WORKER, "wait_for_layer_load")
        if layer_name not in self._registered_kv_caches:
            raise RuntimeError(f"KV cache for layer {layer_name!r} was not registered.")

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        self._require_role(KVConnectorRole.WORKER, "save_kv_layer")
        del kv_layer, attn_metadata, kwargs
        if layer_name not in self._registered_kv_caches:
            raise RuntimeError(f"KV cache for layer {layer_name!r} was not registered.")

    def wait_for_save(self) -> None:
        self._require_role(KVConnectorRole.WORKER, "wait_for_save")

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        self._require_role(KVConnectorRole.WORKER, "get_finished")
        del finished_req_ids
        return None, None

    def get_block_ids_with_load_errors(self) -> set[int]:
        self._require_role(KVConnectorRole.WORKER, "get_block_ids_with_load_errors")
        return set()

    def get_kv_connector_stats(self) -> None:
        """Return no vLLM connector stats until project metrics are connected."""

        self._require_role(KVConnectorRole.WORKER, "get_kv_connector_stats")
        return None

    # Scheduler-side hooks. update_state_after_alloc is called even when the
    # external count is zero:
    # https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/core/sched/scheduler.py#L746-L775
    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        self._require_role(KVConnectorRole.SCHEDULER, "get_num_new_matched_tokens")
        del request, num_computed_tokens
        return 0, False

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        self._require_role(KVConnectorRole.SCHEDULER, "update_state_after_alloc")
        if num_external_tokens != 0:
            raise RuntimeError(
                "The 100%-recompute milestone must report zero external tokens."
            )
        adapted_blocks = adapt_kv_cache_blocks(
            blocks,
            self._adapted_kv_cache_config,
        )
        block_ids_by_group = adapted_blocks.block_ids_by_group
        self._pending_allocations[request.request_id] = CacheBlendAllocation(
            request_id=request.request_id,
            block_ids_by_group=block_ids_by_group,
            num_external_tokens=0,
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> GptOssCacheBlendMetadata:
        self._require_role(KVConnectorRole.SCHEDULER, "build_connector_meta")
        del scheduler_output
        metadata = GptOssCacheBlendMetadata(
            schema_version=_METADATA_SCHEMA_VERSION,
            group_layer_names=self._group_layer_names,
            allocations=tuple(self._pending_allocations.values()),
            transfer_enabled=False,
        )
        self._pending_allocations.clear()
        return metadata

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        self._require_role(KVConnectorRole.SCHEDULER, "update_connector_output")
        del connector_output

    # vLLM selects this hook for SupportsHMA connectors and supplies one block
    # list per group:
    # https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L84-L120
    def request_finished_all_groups(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        self._require_role(KVConnectorRole.SCHEDULER, "request_finished_all_groups")
        del request
        self._validate_group_count(block_ids)
        return False, None

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        del request, block_ids
        raise RuntimeError(
            "Single-group completion is unsupported; GPT-OSS must run with the "
            "hybrid KV-cache manager enabled."
        )

    def shutdown(self) -> None:
        self._pending_allocations.clear()
        self._registered_kv_caches.clear()


__all__ = [
    "CacheBlendAllocation",
    "GptOssCacheBlendConnector",
    "GptOssCacheBlendMetadata",
]
