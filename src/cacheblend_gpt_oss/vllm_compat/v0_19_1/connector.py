"""Pinned vLLM V1 connector for instrumented transfer plus full recomputation.

In ``transfer_100pct`` mode this module performs exact non-prefix lookup,
synchronous KV transfer and GPT-OSS position correction, then deliberately
recomputes every prompt token through ordinary vLLM prefill.  The transferred
KV is instrumentation data and receives zero scheduler credit.  The default
``control_flow`` mode retains the dependency-free no-transfer smoke path.

The API references below are pinned to vLLM 0.19.1 commit
``b1388b1fbf5aaef47937fabe98931211684666a6``:

* ``KVConnectorBase_V1``, ``KVConnectorMetadata``, ``KVConnectorRole``, and
  ``SupportsHMA``:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L82-L207
* Scheduler lookup/allocation/metadata hooks:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L449-L524
* ``KVCacheBlocks.get_block_ids`` and its grouped return value:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/core/kv_cache_manager.py#L22-L76
* ``Request`` computed/external/preemption counters:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/request.py#L135-L164
* ``SchedulerOutput.num_scheduled_tokens``:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/core/sched/output.py#L179-L211
* Per-layer save hook receives the registered attention KV tensor:
  https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/layers/attention/kv_transfer_utils.py#L47-L57
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any

from cacheblend_gpt_oss.connector.control_plane import (
    METADATA_SCHEMA_VERSION,
    RequestControlPlane,
    RequestHandoffMetadata,
    WorkerValidationReceipt,
)
from cacheblend_gpt_oss.gpt_oss.layout import AttentionKind, GroupBlockTable
from cacheblend_gpt_oss.gpt_oss.selective import (
    ForwardRowPlan,
    ForwardRowPlanContext,
    SelectiveForwardState,
)
from cacheblend_gpt_oss.planner import MatchPlan
from cacheblend_gpt_oss.vllm_compat.v0_19_1.adapters import (
    AdaptedKvCacheBlocks,
    AdaptedKvCacheConfig,
    adapt_kv_cache_blocks,
    adapt_kv_cache_config,
    copy_request_prompt_token_ids,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.compatibility_digest import (
    derive_runtime_compatibility_digests,
    require_runtime_compatibility_digests,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.config_validation import (
    require_pinned_config,
    require_transfer_100pct_config,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.runtime_resources import (
    SchedulerRuntimeResources,
    WorkerRuntimeResources,
    create_scheduler_runtime_resources,
    create_worker_runtime_resources,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.scheduler_runtime import (
    SchedulerLookupMetadata,
    SchedulerLookupRequest,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.transfer_config import (
    LMCACHE_CHUNK_SIZE,
    CompatibilityProbeConfig,
    Transfer100PctConfig,
    TransferSelectiveConfig,
    parse_connector_extra_config,
)
from cacheblend_gpt_oss.vllm_compat.v0_19_1.transfer_runtime import (
    PreForwardOutcome,
    SchedulerTransferMetadata,
    TransferAttemptState,
)

try:
    from vllm import __version__ as _VLLM_VERSION  # type: ignore[import-not-found]
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # type: ignore[import-not-found]
        KVConnectorBase_V1,
        KVConnectorMetadata,
        KVConnectorRole,
        KVConnectorWorkerMetadata,
        SupportsHMA,
    )

    from cacheblend_gpt_oss.vllm_compat.v0_19_1.connector_metrics import (
        CacheBlendLookupObservation,
        GptOssCacheBlendPromMetrics,
        GptOssCacheBlendStats,
    )
except ImportError as exc:  # pragma: no cover - exact message tested in isolation
    raise RuntimeError(
        "GptOssCacheBlendConnector requires the pinned vLLM==0.19.1 runtime; "
        "the lightweight CPU-only package does not install vLLM by default."
    ) from exc

if TYPE_CHECKING:
    import torch  # type: ignore[import-not-found]
    from vllm.config import VllmConfig  # type: ignore[import-not-found]
    from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (  # type: ignore[import-not-found]
        KVConnectorPromMetrics,
        KVConnectorStats,
        PromMetric,
        PromMetricT,
    )
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
_METADATA_SCHEMA_VERSION = METADATA_SCHEMA_VERSION
# vLLM's pinned BlockPool reserves block 0 as the permanent null block.  The
# ``request_finished_all_groups`` hook receives only integer IDs, so it cannot
# preserve the ``KVCacheBlock.is_null`` bit after sliding-window blocks have
# been replaced.  Keep this value version-scoped and strip it only from the
# completion observation; block 0 is rejected from allocation snapshots by
# ``adapt_kv_cache_blocks``.
# https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/core/block_pool.py#L95-L108
_VLLM_NULL_BLOCK_ID = 0


def _is_ordered_subsequence(
    observed: tuple[tuple[object, ...], ...],
    allocated: tuple[tuple[int, ...], ...],
    group_kinds: tuple[AttentionKind, ...],
    num_blocks: int,
) -> bool:
    """Validate a completion table against the allocation-time table.

    The pinned scheduler calls ``remove_skipped_blocks`` immediately before
    collecting block IDs and invoking ``request_finished_all_groups``.  Full
    attention therefore retains the allocation table and may append decode
    blocks.  Sliding attention can replace a leading allocation prefix with
    the permanent null block and may also append decode blocks.

    Pinned vLLM completion behavior:

    * ``Scheduler._connector_finished`` performs the cleanup, reads the
      current grouped table, and invokes the HMA hook:
      https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/core/sched/scheduler.py#L2021-L2038
    * ``SingleTypeKVCacheManager.remove_skipped_blocks`` replaces skipped
      entries with ``null_block`` in place:
      https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/core/single_type_kv_cache_manager.py#L358-L399
    * ``KVCacheBlocks.get_block_ids`` returns one integer-ID list per group:
      https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/core/kv_cache_manager.py#L53-L80
    """

    if (
        len(observed) != len(allocated)
        or len(observed) != len(group_kinds)
        or isinstance(num_blocks, bool)
        or not isinstance(num_blocks, int)
        or num_blocks < 1
    ):
        return False

    for current, original, group_kind in zip(
        observed, allocated, group_kinds, strict=True
    ):
        if any(
            isinstance(block_id, bool)
            or not isinstance(block_id, int)
            or block_id < 0
            or block_id >= num_blocks
            for block_id in (*current, *original)
        ):
            return False

        # Allocation snapshots are captured from real writable blocks.  The
        # pinned adapter rejects observable null blocks before they enter the
        # control plane, and a request cannot own the same physical block twice
        # within one cache group.  With prefix caching the allocation table
        # may contain null-block entries, so filter them before checking.
        original_real_blocks = tuple(
            block_id for block_id in original if block_id != _VLLM_NULL_BLOCK_ID
        )
        if len(original_real_blocks) != len(set(original_real_blocks)):
            return False

        # A block can be reused after a sliding-window block is freed, but a
        # current table cannot contain the same real block twice.  Repeated
        # null IDs are the intentional representation of dropped positions.
        current_real_blocks = tuple(
            block_id for block_id in current if block_id != _VLLM_NULL_BLOCK_ID
        )
        if len(current_real_blocks) != len(set(current_real_blocks)):
            return False

        if len(current) < len(original):
            return False

        if group_kind is AttentionKind.FULL:
            # Full attention never drops a logical position, so allocation
            # blocks must remain an exact prefix; only decode-growth blocks may
            # follow them.  Null block 0 is never valid in this group.
            if (
                any(block_id == _VLLM_NULL_BLOCK_ID for block_id in current)
                or current[: len(original)] != original
            ):
                return False
            continue

        if group_kind is not AttentionKind.SLIDING:
            return False

        # Sliding-window cleanup replaces a leading allocation prefix with
        # null blocks.  The prefix can extend beyond the allocation-time table
        # after a long decode: vLLM caps the skipped-block count to the
        # request's current table length and replaces every skipped entry in
        # place.  See the pinned ``remove_skipped_blocks`` implementation:
        # https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/core/single_type_kv_cache_manager.py#L327-L367
        #
        # When that happens, there is no allocation suffix left to bind.  The
        # structural checks above still require a sliding group, a leading
        # null-only prefix, in-range IDs, unique real IDs, and no later nulls;
        # any surviving allocation suffix must remain in order.
        null_prefix_length = 0
        while (
            null_prefix_length < len(current)
            and current[null_prefix_length] == _VLLM_NULL_BLOCK_ID
        ):
            null_prefix_length += 1
        allocation_suffix_start = min(null_prefix_length, len(original))
        if current[allocation_suffix_start : len(original)] != original[
            allocation_suffix_start:
        ] or any(
            block_id == _VLLM_NULL_BLOCK_ID for block_id in current[null_prefix_length:]
        ):
            return False
    return True


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
class GptOssCacheBlendMetadata(KVConnectorMetadata):  # type: ignore[misc]
    """Opaque immutable request handoffs for one pinned scheduler step."""

    schema_version: int
    group_layer_names: tuple[tuple[str, ...], ...]
    handoffs: tuple[RequestHandoffMetadata, ...]
    transfers: tuple[SchedulerTransferMetadata, ...] = ()
    lookup_observations: tuple[CacheBlendLookupObservation, ...] = ()
    transfer_enabled: bool = False


class _DecodeStepMetadata(KVConnectorMetadata):  # type: ignore[misc]
    """Minimal no-op metadata for decode steps with no pending handoffs.

    Kept intentionally tiny so the pickle+ZMQ IPC between the scheduler
    process and the GPU worker serialises a handful of bytes instead of
    the full ``GptOssCacheBlendMetadata`` with 24 layer-name strings.
    """

    __slots__ = ()
    _singleton: _DecodeStepMetadata | None = None

    def __new__(cls) -> _DecodeStepMetadata:
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
        return cls._singleton


@dataclass(frozen=True, slots=True)
class GptOssCacheBlendWorkerMetadata(KVConnectorWorkerMetadata):  # type: ignore[misc]
    """Worker dispositions returned to the scheduler after one model step."""

    receipts: tuple[WorkerValidationReceipt, ...]

    def aggregate(
        self,
        other: KVConnectorWorkerMetadata,
    ) -> GptOssCacheBlendWorkerMetadata:
        if not isinstance(other, GptOssCacheBlendWorkerMetadata):
            raise RuntimeError("Cannot aggregate incompatible connector metadata.")
        by_request = {receipt.request_id: receipt for receipt in self.receipts}
        for receipt in other.receipts:
            existing = by_request.get(receipt.request_id)
            if existing is not None and existing != receipt:
                raise RuntimeError("Workers returned conflicting CacheBlend receipts.")
            by_request[receipt.request_id] = receipt
        return GptOssCacheBlendWorkerMetadata(
            receipts=tuple(by_request[key] for key in sorted(by_request))
        )


@dataclass(slots=True)
class _ActiveWorkerTransfer:
    """One max-seq-one transfer retained across the per-layer forward hooks."""

    metadata: SchedulerTransferMetadata
    adapted_blocks: AdaptedKvCacheBlocks
    pre_forward: PreForwardOutcome
    saved_layer_names: set[str]
    load_latency_seconds: float


class GptOssCacheBlendConnector(
    KVConnectorBase_V1,
    SupportsHMA,  # type: ignore[misc]
):
    """vLLM 0.19.1 connector for full-prefill and explicit selective modes.

    ``transfer_100pct`` remains synchronous instrumentation with ordinary full
    prefill.  ``transfer_selective`` installs one bounded row plan across the
    worker's model forward and KV writeback hooks; it still credits zero
    external scheduler tokens and remains an experimental GPU gate.
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

        transfer_config = parse_connector_extra_config(
            vllm_config.kv_transfer_config.kv_connector_extra_config
        )
        _prefix_caching_enabled = (
            isinstance(transfer_config, Transfer100PctConfig)
            and transfer_config.allow_prefix_caching
        )
        require_pinned_config(
            vllm_config,
            kv_cache_config,
            v2_model_runner_enabled=_v2_model_runner_enabled(),
            allow_custom_attention_backend=isinstance(
                transfer_config, TransferSelectiveConfig
            ),
            allow_unified_kv_mode=_prefix_caching_enabled,
        )

        # vLLM otherwise defaults to disabling HMA whenever a connector is set.
        # Pinned source:
        # https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/config/vllm.py#L1227-L1247
        if (
            vllm_config.scheduler_config.disable_hybrid_kv_cache_manager is not False
            and not _prefix_caching_enabled
        ):
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
        self._adapted_kv_cache_config: AdaptedKvCacheConfig = adapt_kv_cache_config(
            kv_cache_config
        )
        self._transfer_config = transfer_config
        self._allow_prefix_caching = (
            isinstance(transfer_config, Transfer100PctConfig)
            and transfer_config.allow_prefix_caching
        )
        if isinstance(self._transfer_config, CompatibilityProbeConfig):
            digests = derive_runtime_compatibility_digests(
                vllm_config, self._adapted_kv_cache_config
            )
            raise RuntimeError(
                "CacheBlend compatibility probe (transfer remains disabled): "
                f"model_config_digest={digests.model_config_digest}; "
                f"kv_cache_config_digest={digests.kv_cache_config_digest}."
            )
        if isinstance(self._transfer_config, Transfer100PctConfig):
            require_transfer_100pct_config(
                vllm_config,
                staging_token_capacity=(self._transfer_config.staging_token_capacity),
                allow_prefix_caching=(self._transfer_config.allow_prefix_caching),
            )
            # In unified mode (1 KV cache group, hybrid manager disabled),
            # the pre-computed kv_cache_config_digest was derived from the
            # 2-group hybrid layout and will not match the 1-group layout.
            # CacheBlend transfers are inert under prefix caching, so the
            # digest mismatch is harmless — skip the check entirely.
            _unified_kv_mode = (
                len(self._adapted_kv_cache_config.gpt_oss_layout.groups) == 1
            )
            if not _unified_kv_mode:
                require_runtime_compatibility_digests(
                    vllm_config,
                    self._adapted_kv_cache_config,
                    expected_model_config_digest=(
                        self._transfer_config.model_config_digest
                    ),
                    expected_kv_cache_config_digest=(
                        self._transfer_config.kv_cache_config_digest
                    ),
                )
        self._group_layer_names = (
            self._adapted_kv_cache_config.control_plane_layout.layer_names_by_group
        )
        self._control_plane = RequestControlPlane(
            self._adapted_kv_cache_config.control_plane_layout
        )
        self._known_request_ids: set[str] = set()
        self._request_preemptions: dict[str, int] = {}
        self._pending_handoff_ids: list[str] = []
        self._pending_worker_receipts: list[WorkerValidationReceipt] = []
        # vLLM 0.19.1 may free a completed request before the same
        # ``KVConnectorOutput`` is applied to the scheduler.  Retain the
        # scheduler control-plane state until that worker receipt arrives.
        self._finished_request_ids: set[str] = set()
        self._registered_kv_caches: dict[str, torch.Tensor] = {}
        self._scheduler_lookup_metadata: dict[str, SchedulerLookupMetadata] = {}
        self._scheduler_lookup_observations: dict[str, CacheBlendLookupObservation] = {}
        self._stats = GptOssCacheBlendStats()
        self._scheduler_resources: SchedulerRuntimeResources | None = None
        self._worker_resources: WorkerRuntimeResources | None = None
        self._active_worker_transfer: _ActiveWorkerTransfer | None = None
        self._active_forward_plan_token: object | None = None
        self._prefix_cached_tokens: dict[str, int] = {}
        self._decode_step_count: int = 0
        self._prefill_step_count: int = 0
        self._decode_diag: bool = os.environ.get("CACHEBLEND_DECODE_DIAG") == "1"
        self._transfer_diag: bool = os.environ.get("CACHEBLEND_TRANSFER_DIAG") == "1"
        if (
            isinstance(self._transfer_config, Transfer100PctConfig)
            and role is KVConnectorRole.SCHEDULER
        ):
            self._scheduler_resources = create_scheduler_runtime_resources(
                self._transfer_config
            )

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

    @property
    def _transfer_enabled(self) -> bool:
        return isinstance(self._transfer_config, Transfer100PctConfig)

    @property
    def _selective_enabled(self) -> bool:
        return isinstance(self._transfer_config, TransferSelectiveConfig)

    def _install_forward_plan(
        self, plan: ForwardRowPlan | SelectiveForwardState
    ) -> None:
        if not self._selective_enabled:
            return
        if self._active_forward_plan_token is not None:
            raise RuntimeError("A selective forward plan is already active.")
        self._active_forward_plan_token = ForwardRowPlanContext.install(plan)

    def _clear_forward_plan(self) -> None:
        token = self._active_forward_plan_token
        if token is None:
            return
        self._active_forward_plan_token = None
        ForwardRowPlanContext.reset(token)

    def _adapt_handoff_blocks(
        self, handoff: RequestHandoffMetadata
    ) -> AdaptedKvCacheBlocks:
        grouped = handoff.allocation.grouped_blocks
        block_ids_by_group = grouped.block_ids_by_group
        self._validate_group_count(block_ids_by_group)
        if any(
            block_id >= self._adapted_kv_cache_config.num_blocks
            for group in block_ids_by_group
            for block_id in group
        ):
            raise RuntimeError("Worker handoff contains an out-of-range block ID.")
        tables = tuple(
            GroupBlockTable(
                group_id=group.group_id,
                block_size=group.block_size,
                block_ids=block_ids_by_group[group.group_id],
            )
            for group in self._adapted_kv_cache_config.gpt_oss_layout.groups
        )
        return AdaptedKvCacheBlocks(grouped, tables)

    def _registered_cuda_device(self) -> str:
        devices: set[str] = set()
        for tensor in self._registered_kv_caches.values():
            device = getattr(tensor, "device", None)
            if device is None:
                raise RuntimeError(
                    "Registered KV-cache tensors must expose one CUDA device."
                )
            devices.add(str(device))
        if len(devices) != 1:
            raise RuntimeError(
                "All registered KV-cache tensors must share one CUDA device."
            )
        return next(iter(devices))

    # Worker-side hooks. vLLM registers per-layer tensors here:
    # https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/worker/gpu_model_runner.py#L6809-L6819
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._require_role(KVConnectorRole.WORKER, "register_kv_caches")
        if self._registered_kv_caches:
            raise RuntimeError("KV caches were already registered on this worker.")
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
        self._registered_kv_caches = dict(kv_caches)
        if self._transfer_enabled:
            if not isinstance(self._transfer_config, Transfer100PctConfig):
                raise RuntimeError("Transfer configuration state is inconsistent.")
            self._worker_resources = create_worker_runtime_resources(
                self._transfer_config,
                self._adapted_kv_cache_config.gpt_oss_layout,
                self._registered_kv_caches,
                device=self._registered_cuda_device(),
            )

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        self._require_role(KVConnectorRole.WORKER, "start_load_kv")
        del forward_context, kwargs
        metadata = self._get_connector_metadata()
        if isinstance(metadata, _DecodeStepMetadata):
            self._decode_step_count += 1
            if self._decode_diag and self._decode_step_count % 500 == 0:
                import sys

                print(
                    f"CACHEBLEND_DECODE_DIAG decode_steps={self._decode_step_count} "
                    f"prefill_steps={self._prefill_step_count}",
                    file=sys.stderr,
                    flush=True,
                )
            return
        self._prefill_step_count += 1
        if not isinstance(metadata, GptOssCacheBlendMetadata):
            raise RuntimeError("Received metadata from an incompatible connector.")
        if (
            isinstance(metadata.schema_version, bool)
            or not isinstance(metadata.schema_version, int)
            or metadata.schema_version != _METADATA_SCHEMA_VERSION
        ):
            raise RuntimeError(
                f"Unsupported connector metadata schema {metadata.schema_version}."
            )
        if metadata.group_layer_names != self._group_layer_names:
            raise RuntimeError("Scheduler and worker KV-cache groups do not match.")
        if metadata.transfer_enabled is not self._transfer_enabled:
            raise RuntimeError("Scheduler and worker transfer modes do not match.")
        if self._active_worker_transfer is not None:
            raise RuntimeError("A previous CacheBlend transfer is still active.")
        if self._selective_enabled and len(metadata.handoffs) > 1:
            raise RuntimeError(
                "The selective transfer envelope allows one request per step."
            )
        if len(metadata.transfers) > 1:
            raise RuntimeError(
                "The 100%-recompute transfer envelope allows one request per step."
            )
        if metadata.transfer_enabled:
            if len(metadata.lookup_observations) != len(metadata.handoffs):
                raise RuntimeError(
                    "Transfer metadata must carry one lookup observation per "
                    "request handoff."
                )
        elif metadata.lookup_observations:
            raise RuntimeError(
                "Control-flow metadata cannot claim transfer lookup observations."
            )
        for observation in metadata.lookup_observations:
            self._stats.record_lookup(observation)
        transfers_by_request = {
            transfer.request_id: transfer for transfer in metadata.transfers
        }
        if len(transfers_by_request) != len(metadata.transfers):
            raise RuntimeError("Connector metadata contains duplicate transfers.")
        for handoff in metadata.handoffs:
            self._validate_group_count(
                handoff.allocation.grouped_blocks.block_ids_by_group
            )
            if handoff.allocation.external_scheduler_tokens != 0:
                raise RuntimeError(
                    "The 100%-recompute milestone cannot credit external tokens."
                )
            self._control_plane.accept_handoff(handoff)
            transfer = transfers_by_request.pop(handoff.plan.request_id, None)
            if transfer is None:
                verified_tokens = sum(
                    len(match.target_segment.token_range)
                    for match in handoff.plan.match_plan.matches
                )
                self._stats.record_load(
                    verified_tokens=verified_tokens,
                    loaded_tokens=0,
                    rejected_tokens=verified_tokens,
                    recomputed_tokens=handoff.plan.prompt_tokens,
                    # In transfer mode, an absent transfer metadata entry
                    # means the request was not eligible for the one-step
                    # staging envelope (for example, a partial scheduler
                    # step).  Ordinary control-flow mode has no transfer
                    # attempt and therefore is not a fallback.
                    fallback=self._transfer_enabled,
                    latency_seconds=0.0,
                )
                receipt = self._control_plane.validate_worker(
                    handoff.plan.request_id,
                    loaded_match_indexes=(),
                    rejected_match_indexes=range(len(handoff.plan.match_plan.matches)),
                )
                self._pending_worker_receipts.append(receipt)
                if self._selective_enabled:
                    self._install_forward_plan(
                        ForwardRowPlan.full_recompute(handoff.plan.prompt_tokens)
                    )
                continue
            if transfer.handoff != handoff:
                raise RuntimeError(
                    "Transfer metadata does not match its request handoff."
                )
            if self._worker_resources is None:
                raise RuntimeError(
                    "Worker transfer resources were not initialized by "
                    "register_kv_caches."
                )
            adapted_blocks = self._adapt_handoff_blocks(handoff)
            started_at = perf_counter()
            outcome = self._worker_resources.transfer_runtime.before_forward(
                transfer, adapted_blocks
            )
            load_latency_seconds = perf_counter() - started_at
            # Selective row work is not known until the check-layer backend
            # measures fresh-vs-loaded values.  Defer this one aggregate load
            # observation until wait_for_save, where the final plan exists.
            if not (
                self._selective_enabled
                and outcome.state is TransferAttemptState.SUCCEEDED
                and outcome.selective_state is not None
            ):
                self._stats.record_load(
                    verified_tokens=sum(
                        len(candidate.candidate.target_range)
                        for candidate in transfer.verified_candidates
                    ),
                    loaded_tokens=outcome.loaded_kv_tokens,
                    rejected_tokens=sum(
                        len(transfer.verified_candidates[index].candidate.target_range)
                        for index in outcome.rejected_candidate_indexes
                    ),
                    recomputed_tokens=outcome.tokens_to_recompute,
                    fallback=(
                        outcome.state is TransferAttemptState.FULL_PREFILL_FALLBACK
                    ),
                    latency_seconds=load_latency_seconds,
                    position_correction_latency_seconds=(
                        outcome.position_correction_latency_seconds
                    ),
                    scatter_suppressed_tokens=outcome.scatter_suppressed_tokens,
                    layer_token_rows_recomputed=outcome.layer_token_rows_recomputed,
                    layer_token_rows_avoided=outcome.layer_token_rows_avoided,
                )
            receipt = self._control_plane.validate_worker(
                handoff.plan.request_id,
                loaded_match_indexes=outcome.loaded_candidate_indexes,
                rejected_match_indexes=outcome.rejected_candidate_indexes,
            )
            if receipt != outcome.to_worker_validation_receipt():
                raise RuntimeError("Worker transfer receipt reconciliation failed.")
            self._pending_worker_receipts.append(receipt)
            if self._selective_enabled:
                self._install_forward_plan(
                    outcome.selective_state
                    or outcome.row_plan
                    or ForwardRowPlan.full_recompute(transfer.prompt_token_count)
                )
            self._active_worker_transfer = _ActiveWorkerTransfer(
                transfer,
                adapted_blocks,
                outcome,
                set(),
                load_latency_seconds,
            )
        if transfers_by_request:
            raise RuntimeError("Transfer metadata has no matching request handoff.")

    def wait_for_layer_load(self, layer_name: str) -> None:
        if self._active_worker_transfer is None:
            return
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
        active = self._active_worker_transfer
        if active is None:
            return
        self._require_role(KVConnectorRole.WORKER, "save_kv_layer")
        del attn_metadata, kwargs
        if layer_name not in self._registered_kv_caches:
            raise RuntimeError(f"KV cache for layer {layer_name!r} was not registered.")
        if kv_layer is not self._registered_kv_caches[layer_name]:
            raise RuntimeError("save_kv_layer received an unregistered tensor.")
        if active is not None:
            if layer_name in active.saved_layer_names:
                raise RuntimeError("A KV-cache layer was saved more than once.")
            active.saved_layer_names.add(layer_name)
            if (
                isinstance(self._transfer_config, Transfer100PctConfig)
                and self._transfer_config.transfer_evidence_path is not None
            ):
                if self._worker_resources is None:
                    raise RuntimeError("Worker transfer resources are unavailable.")
                self._worker_resources.bridge.capture_prefill_layer(layer_name)

    def wait_for_save(self) -> None:
        active = self._active_worker_transfer
        if active is None:
            return
        self._require_role(KVConnectorRole.WORKER, "wait_for_save")
        try:
            expected_layers = set(self._registered_kv_caches)
            if active.saved_layer_names != expected_layers:
                raise RuntimeError(
                    "Full-prefill KV writeback did not visit every GPT-OSS layer."
                )
            if self._worker_resources is None:
                raise RuntimeError("Worker transfer resources are unavailable.")
            if (
                self._selective_enabled
                and active.pre_forward.selective_state is not None
            ):
                selective_state = active.pre_forward.selective_state
                if not selective_state.scored:
                    raise RuntimeError(
                        "Selective forward completed without a check-layer score."
                    )
                final_plan = selective_state.plan
                verified_tokens = sum(
                    len(candidate.candidate.target_range)
                    for candidate in active.metadata.verified_candidates
                )
                rejected_tokens = sum(
                    len(
                        active.metadata.verified_candidates[
                            index
                        ].candidate.target_range
                    )
                    for index in active.pre_forward.rejected_candidate_indexes
                )
                self._stats.record_load(
                    verified_tokens=verified_tokens,
                    loaded_tokens=active.pre_forward.loaded_kv_tokens,
                    rejected_tokens=rejected_tokens,
                    recomputed_tokens=active.pre_forward.tokens_to_recompute,
                    fallback=(
                        active.pre_forward.state
                        is TransferAttemptState.FULL_PREFILL_FALLBACK
                    ),
                    latency_seconds=active.load_latency_seconds,
                    position_correction_latency_seconds=(
                        active.pre_forward.position_correction_latency_seconds
                    ),
                    layer_token_rows_recomputed=final_plan.recompute_tokens,
                    layer_token_rows_avoided=final_plan.cached_tokens,
                )
            completion = (
                self._worker_resources.transfer_runtime.mark_full_prefill_complete(
                    active.pre_forward,
                    recomputed_token_count=active.metadata.prompt_token_count,
                )
            )
            started_at = perf_counter()
            post_forward = self._worker_resources.transfer_runtime.after_forward(
                completion, active.adapted_blocks
            )
            if (
                isinstance(self._transfer_config, Transfer100PctConfig)
                and self._transfer_config.transfer_evidence_path is not None
            ):
                if (
                    active.pre_forward.state is TransferAttemptState.SUCCEEDED
                    and post_forward.state is TransferAttemptState.SUCCEEDED
                ):
                    self._worker_resources.bridge.finish_transfer_evidence(
                        recomputed_tokens=active.metadata.prompt_token_count,
                        prefill_tokens_avoided=post_forward.prefill_tokens_avoided,
                    )
                else:
                    self._worker_resources.bridge.abort_transfer_evidence()
            self._stats.record_store(
                eligible_tokens=post_forward.eligible_store_tokens,
                stored_tokens=post_forward.stored_tokens,
                fallback=(
                    post_forward.state is TransferAttemptState.FULL_PREFILL_FALLBACK
                ),
                latency_seconds=perf_counter() - started_at,
                store_plan_latency_seconds=(post_forward.store_plan_latency_seconds),
                store_preflight_latency_seconds=(
                    post_forward.store_preflight_latency_seconds
                ),
                store_gather_latency_seconds=(
                    post_forward.store_gather_latency_seconds
                ),
                store_lmcache_latency_seconds=(
                    post_forward.store_lmcache_latency_seconds
                ),
                store_sidecar_publish_latency_seconds=(
                    post_forward.store_sidecar_publish_latency_seconds
                ),
                store_storage_preflight_latency_seconds=(
                    post_forward.store_storage_preflight_latency_seconds
                ),
                store_preflight_prepare_latency_seconds=(
                    post_forward.store_preflight_data_plane_timing.prepare_latency_seconds
                ),
                store_preflight_input_materialization_latency_seconds=(
                    post_forward.store_preflight_data_plane_timing.input_materialization_latency_seconds
                ),
                store_preflight_span_validation_latency_seconds=(
                    post_forward.store_preflight_data_plane_timing.span_validation_latency_seconds
                ),
                store_preflight_tensor_validation_latency_seconds=(
                    post_forward.store_preflight_data_plane_timing.tensor_validation_latency_seconds
                ),
                store_preflight_range_validation_latency_seconds=(
                    post_forward.store_preflight_data_plane_timing.range_validation_latency_seconds
                ),
                store_preflight_block_plan_latency_seconds=(
                    post_forward.store_preflight_data_plane_timing.block_plan_latency_seconds
                ),
                store_preflight_block_index_view_latency_seconds=(
                    post_forward.store_preflight_data_plane_timing.block_index_view_latency_seconds
                ),
                store_preflight_block_index_construction_latency_seconds=(
                    post_forward.store_preflight_data_plane_timing.block_index_construction_latency_seconds
                ),
                store_preflight_block_index_validation_latency_seconds=(
                    post_forward.store_preflight_data_plane_timing.block_index_validation_latency_seconds
                ),
                store_preflight_staging_view_construction_latency_seconds=(
                    post_forward.store_preflight_data_plane_timing.staging_view_construction_latency_seconds
                ),
                store_preflight_staging_view_validation_latency_seconds=(
                    post_forward.store_preflight_data_plane_timing.staging_view_validation_latency_seconds
                ),
                store_preflight_legacy_view_latency_seconds=(
                    post_forward.store_preflight_data_plane_timing.legacy_view_latency_seconds
                ),
                store_preflight_enqueue_latency_seconds=(
                    post_forward.store_preflight_data_plane_timing.enqueue_latency_seconds
                ),
                store_preflight_synchronize_latency_seconds=(
                    post_forward.store_preflight_data_plane_timing.synchronize_latency_seconds
                ),
                store_preflight_prepared_copy_operations=(
                    post_forward.store_preflight_data_plane_timing.prepared_copy_operations
                ),
                store_preflight_submitted_copy_operations=(
                    post_forward.store_preflight_data_plane_timing.submitted_copy_operations
                ),
                store_preflight_block_index_owner_constructions=(
                    post_forward.store_preflight_data_plane_timing.block_index_owner_constructions
                ),
                store_preflight_block_index_row_views=(
                    post_forward.store_preflight_data_plane_timing.block_index_row_views
                ),
                store_preflight_staging_view_constructions=(
                    post_forward.store_preflight_data_plane_timing.staging_view_constructions
                ),
                store_gather_prepare_latency_seconds=(
                    post_forward.store_gather_data_plane_timing.prepare_latency_seconds
                ),
                store_gather_enqueue_latency_seconds=(
                    post_forward.store_gather_data_plane_timing.enqueue_latency_seconds
                ),
                store_gather_synchronize_latency_seconds=(
                    post_forward.store_gather_data_plane_timing.synchronize_latency_seconds
                ),
                store_gather_prepared_copy_operations=(
                    post_forward.store_gather_data_plane_timing.prepared_copy_operations
                ),
                store_gather_submitted_copy_operations=(
                    post_forward.store_gather_data_plane_timing.submitted_copy_operations
                ),
            )
            self._active_worker_transfer = None
        finally:
            self._clear_forward_plan()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        self._require_role(KVConnectorRole.WORKER, "get_finished")
        active = self._active_worker_transfer
        if active is not None and active.metadata.request_id in finished_req_ids:
            raise RuntimeError(
                "A request finished before CacheBlend completed KV writeback."
            )
        if self._decode_diag and finished_req_ids:
            import sys

            print(
                f"CACHEBLEND_DECODE_DIAG finished={len(finished_req_ids)} "
                f"total_decode_steps={self._decode_step_count} "
                f"total_prefill_steps={self._prefill_step_count}",
                file=sys.stderr,
                flush=True,
            )
        for request_id in finished_req_ids:
            self._control_plane.discard(request_id)
            self._known_request_ids.discard(request_id)
        return None, None

    def build_connector_worker_meta(
        self,
    ) -> GptOssCacheBlendWorkerMetadata | None:
        self._require_role(KVConnectorRole.WORKER, "build_connector_worker_meta")
        if not self._pending_worker_receipts:
            return None
        metadata = GptOssCacheBlendWorkerMetadata(
            receipts=tuple(self._pending_worker_receipts)
        )
        self._pending_worker_receipts.clear()
        return metadata

    def get_block_ids_with_load_errors(self) -> set[int]:
        self._require_role(KVConnectorRole.WORKER, "get_block_ids_with_load_errors")
        return set()

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """Return and atomically reset identifier-free worker observations.

        The pinned vLLM scheduler also invokes this hook on its scheduler-role
        connector from ``Scheduler.update_from_output``.  Statistics are
        produced by the worker/model-runner connector and carried in the
        ``KVConnectorOutput``; the scheduler instance has no local worker
        observations to drain.  Returning ``None`` for that role is therefore
        the intended base-class behavior and avoids turning a normal scheduler
        bookkeeping call into a fatal request error.

        Pinned call sites:
        https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/core/sched/scheduler.py#L1320-L1335
        https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/worker/gpu_model_runner.py#L7050-L7070
        """
        if self.role is KVConnectorRole.SCHEDULER:
            return None
        if self._stats.is_empty():
            return None
        return self._stats.clone_and_reset()

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats:
        """Rebuild the serializable stats object in vLLM's logger process."""

        return GptOssCacheBlendStats(data={} if data is None else data)

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics:
        """Register the pinned connector's bounded Prometheus metrics."""

        return GptOssCacheBlendPromMetrics(
            vllm_config,
            metric_types,
            labelnames,
            per_engine_labelvalues,
        )

    # Scheduler-side hooks. update_state_after_alloc is called even when the
    # external count is zero:
    # https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/core/sched/scheduler.py#L746-L775
    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        self._require_role(KVConnectorRole.SCHEDULER, "get_num_new_matched_tokens")
        prompt_token_ids = copy_request_prompt_token_ids(request)
        request_id = request.request_id
        # These fields are present on the pinned vLLM 0.19.1 ``Request``.
        # Do not default them to zero: doing so would silently treat an API
        # drift (or a malformed request object) as an eligible zero-credit
        # transfer.  The scheduler's external-token counter is especially
        # safety-critical because every CacheBlend transfer in this milestone
        # must still be fully recomputed.
        raw_preemptions = getattr(request, "num_preemptions", None)
        if (
            isinstance(raw_preemptions, bool)
            or not isinstance(raw_preemptions, int)
            or raw_preemptions < 0
        ):
            raise RuntimeError(
                "Pinned vLLM Request.num_preemptions is missing or invalid."
            )
        previous_preemptions = self._request_preemptions.get(request_id)
        if previous_preemptions is not None:
            if raw_preemptions < previous_preemptions:
                raise RuntimeError("Request preemption count moved backwards.")
            if raw_preemptions > previous_preemptions:
                self._control_plane.preempt(request_id)
        self._request_preemptions[request_id] = raw_preemptions
        if self._transfer_enabled:
            if self._scheduler_resources is None:
                raise RuntimeError("Scheduler transfer resources are unavailable.")
            started_at = perf_counter()
            lookup = self._scheduler_resources.runtime.lookup(
                SchedulerLookupRequest(
                    request_id=request_id,
                    prompt_token_ids=prompt_token_ids,
                    sequence_count=1,
                    scheduler_step_index=0,
                    num_computed_tokens=num_computed_tokens,
                    num_external_tokens=self._request_external_tokens(request),
                    preemption_count=raw_preemptions,
                )
            )
            lookup_latency = perf_counter() - started_at
            if lookup.status.is_fatal:
                raise RuntimeError(
                    f"CacheBlend scheduler lookup failed closed: {lookup.status.value}."
                )
            plan = lookup.request_plan
            self._control_plane.lookup(
                request_id=plan.request_id,
                prompt_tokens=plan.prompt_tokens,
                query_segments=plan.query_segments,
                match_plan=plan.match_plan,
            )
            self._scheduler_lookup_metadata[request_id] = lookup
            if self._transfer_diag:
                import sys

                print(
                    f"CACHEBLEND_TRANSFER_DIAG lookup"
                    f" request={request_id}"
                    f" prompt_tokens={len(prompt_token_ids)}"
                    f" prefix_cached_tokens={num_computed_tokens}"
                    f" lookup_status={lookup.status.value}"
                    f" should_transfer={lookup.should_transfer}"
                    f" verified_candidates={len(lookup.verified_candidates)}",
                    file=sys.stderr,
                    flush=True,
                )
            counters = lookup.lookup_plan.counters
            self._scheduler_lookup_observations[request_id] = (
                CacheBlendLookupObservation(
                    prompt_tokens=len(prompt_token_ids),
                    reusable_segments_requested=len(lookup.query_windows),
                    reusable_segments_hit=counters.verified_candidates,
                    reusable_document_tokens_requested=(
                        len(prompt_token_ids) if lookup.query_windows else 0
                    ),
                    # Rolling query windows may overlap.  Export unique
                    # prompt coverage so token-hit fractions stay bounded;
                    # raw candidate totals remain available in the lookup
                    # plan for diagnostics.
                    kv_tokens_found=lookup.lookup_plan.found_target_token_count,
                    kv_tokens_verified=counters.verified_candidate_tokens,
                    kv_tokens_rejected=(
                        lookup.lookup_plan.found_target_token_count
                        - counters.verified_candidate_tokens
                    ),
                    latency_seconds=lookup_latency,
                )
            )
        else:
            self._control_plane.lookup(
                request_id=request_id,
                prompt_tokens=len(prompt_token_ids),
                query_segments=(),
                match_plan=MatchPlan((), (), 0),
            )
        self._known_request_ids.add(request_id)
        return 0, False

    @staticmethod
    def _request_external_tokens(request: Request) -> int:
        """Read the pinned scheduler counter without a compatibility default."""

        value = getattr(request, "num_external_computed_tokens", None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(
                "Pinned vLLM Request.num_external_computed_tokens is missing "
                "or invalid."
            )
        return value

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        self._require_role(KVConnectorRole.SCHEDULER, "update_state_after_alloc")
        if self._transfer_diag:
            import sys

            print(
                f"CACHEBLEND_TRANSFER_DIAG alloc"
                f" request={request.request_id}"
                f" num_external_tokens={num_external_tokens}",
                file=sys.stderr,
                flush=True,
            )
        if num_external_tokens != 0 and not self._allow_prefix_caching:
            raise RuntimeError(
                "The 100%-recompute milestone must report zero external tokens."
            )
        if self._allow_prefix_caching and num_external_tokens > 0:
            self._prefix_cached_tokens[request.request_id] = num_external_tokens
        adapted_blocks = adapt_kv_cache_blocks(
            blocks,
            self._adapted_kv_cache_config,
            allow_null_blocks=self._allow_prefix_caching,
        )
        block_ids_by_group = adapted_blocks.block_ids_by_group
        request_id = request.request_id
        if request_id not in self._known_request_ids:
            raise RuntimeError("Allocation arrived before CacheBlend lookup.")
        self._control_plane.allocate(
            request_id,
            block_ids_by_group,
            external_scheduler_tokens=0,
            allow_duplicate_block_ids=self._allow_prefix_caching,
        )
        if request_id not in self._pending_handoff_ids:
            self._pending_handoff_ids.append(request_id)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> GptOssCacheBlendMetadata | _DecodeStepMetadata:
        self._require_role(KVConnectorRole.SCHEDULER, "build_connector_meta")
        if not self._pending_handoff_ids:
            return _DecodeStepMetadata()
        handoffs = tuple(
            self._control_plane.handoff(request_id)
            for request_id in self._pending_handoff_ids
        )
        transfers: list[SchedulerTransferMetadata] = []
        if self._transfer_enabled:
            scheduled_by_request = getattr(
                scheduler_output, "num_scheduled_tokens", None
            )
            if not isinstance(scheduled_by_request, dict):
                raise RuntimeError(
                    "Pinned SchedulerOutput.num_scheduled_tokens is unavailable."
                )
            if not isinstance(self._transfer_config, Transfer100PctConfig):
                raise RuntimeError("Transfer configuration state is inconsistent.")
            for handoff in handoffs:
                request_id = handoff.plan.request_id
                lookup = self._scheduler_lookup_metadata.get(request_id)
                scheduled_tokens = scheduled_by_request.get(request_id)
                if lookup is None:
                    raise RuntimeError("Transfer handoff has no scheduler lookup.")
                if (
                    isinstance(scheduled_tokens, bool)
                    or not isinstance(scheduled_tokens, int)
                    or scheduled_tokens < 0
                    or scheduled_tokens > len(lookup.prompt_token_ids)
                ):
                    raise RuntimeError(
                        "SchedulerOutput contains an invalid scheduled token count."
                    )
                if (
                    handoff.allocation.allocation_generation
                    != lookup.allocation_generation
                ):
                    raise RuntimeError(
                        "Scheduler lookup and block allocation generations differ."
                    )
                complete_step = scheduled_tokens == len(lookup.prompt_token_ids)
                within_staging = (
                    len(lookup.prompt_token_ids)
                    <= self._transfer_config.staging_token_capacity
                )
                if self._transfer_diag:
                    import sys

                    print(
                        f"CACHEBLEND_TRANSFER_DIAG build_meta"
                        f" request={request_id}"
                        f" scheduled_tokens={scheduled_tokens}"
                        f" prompt_tokens={len(lookup.prompt_token_ids)}"
                        f" complete_step={complete_step}"
                        f" within_staging={within_staging}"
                        f" should_transfer={lookup.should_transfer}"
                        f" verified_candidates={len(lookup.verified_candidates)}",
                        file=sys.stderr,
                        flush=True,
                    )
                if not complete_step or not within_staging:
                    continue
                prefix_cached = self._prefix_cached_tokens.get(request_id, 0)
                if (
                    prefix_cached > 0
                    and lookup.verified_candidates
                    and all(
                        candidate.candidate.target_range.end <= prefix_cached
                        for candidate in lookup.verified_candidates
                    )
                ):
                    continue
                transfers.append(
                    SchedulerTransferMetadata(
                        cache_namespace=self._transfer_config.namespace,
                        prompt_token_ids=lookup.prompt_token_ids,
                        verified_candidates=lookup.verified_candidates,
                        handoff=handoff,
                        num_computed_tokens_before_step=0,
                        scheduled_token_count=scheduled_tokens,
                        transfer_eligible=lookup.should_transfer,
                        store_eligible=(
                            not self._transfer_config.disable_kv_store
                            and len(lookup.prompt_token_ids) >= LMCACHE_CHUNK_SIZE
                        ),
                    )
                )
        metadata = GptOssCacheBlendMetadata(
            schema_version=_METADATA_SCHEMA_VERSION,
            group_layer_names=self._group_layer_names,
            handoffs=handoffs,
            transfers=tuple(transfers),
            lookup_observations=(
                tuple(
                    self._scheduler_lookup_observations[handoff.plan.request_id]
                    for handoff in handoffs
                )
                if self._transfer_enabled
                else ()
            ),
            transfer_enabled=self._transfer_enabled,
        )
        self._pending_handoff_ids.clear()
        return metadata

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        self._require_role(KVConnectorRole.SCHEDULER, "update_connector_output")
        worker_metadata = connector_output.kv_connector_worker_meta
        if worker_metadata is None:
            return
        if not isinstance(worker_metadata, GptOssCacheBlendWorkerMetadata):
            raise RuntimeError("Received incompatible CacheBlend worker metadata.")
        for receipt in worker_metadata.receipts:
            self._control_plane.apply_worker_validation(receipt)
            if receipt.request_id in self._finished_request_ids:
                self._control_plane.discard(receipt.request_id)
                self._finished_request_ids.discard(receipt.request_id)

    # vLLM selects this hook for SupportsHMA connectors and supplies one block
    # list per group:
    # https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L84-L120
    def request_finished_all_groups(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        self._require_role(KVConnectorRole.SCHEDULER, "request_finished_all_groups")
        self._validate_group_count(block_ids)
        request_id = request.request_id
        # vLLM calls this HMA hook before releasing the group's blocks.  The
        # count check above prevents a single-group fallback, but a table from
        # another request would still be an invalid completion receipt. vLLM
        # may drop old sliding-window blocks before this hook, so validate the
        # current table against the allocation-time table while allowing legal
        # decode-growth suffixes. This remains permissive for early
        # cancellation, where no allocation exists.
        state = self._control_plane.state(request_id)
        if state.allocation is not None:
            try:
                observed_block_ids = tuple(tuple(group) for group in block_ids)
            except TypeError as exc:
                raise RuntimeError(
                    "Completion block tables are not valid grouped sequences."
                ) from exc
            if not _is_ordered_subsequence(
                observed_block_ids,
                state.allocation.grouped_blocks.block_ids_by_group,
                tuple(
                    group.attention_kind
                    for group in self._adapted_kv_cache_config.gpt_oss_layout.groups
                ),
                self._adapted_kv_cache_config.num_blocks,
            ):
                raise RuntimeError(
                    "Completion block tables are not compatible with the "
                    "request allocation."
                )
        if self._scheduler_resources is not None:
            self._scheduler_resources.runtime.discard(request_id)
        self._scheduler_lookup_metadata.pop(request_id, None)
        self._scheduler_lookup_observations.pop(request_id, None)
        self._prefix_cached_tokens.pop(request_id, None)
        # The pinned scheduler can invoke this completion hook before it
        # applies the worker's validation receipt (see
        # ``Scheduler._update_from_kv_xfer_finished``).  Discarding here would
        # make that valid late receipt look like an unknown request and kill
        # EngineCore.  Keep transfer state until ``update_connector_output``
        # applies the receipt; the TP=1 target has one synchronous receipt.
        # In control-flow mode there is no worker receipt, so discard normally.
        if self._transfer_enabled and state.worker_validation is None:
            self._finished_request_ids.add(request_id)
        else:
            self._control_plane.discard(request_id)
        self._known_request_ids.discard(request_id)
        self._request_preemptions.pop(request_id, None)
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
        close_error: BaseException | None = None
        if self._scheduler_resources is not None:
            scheduler_resources = self._scheduler_resources
            try:
                scheduler_resources.close()
            except Exception as exc:
                close_error = exc
            else:
                self._scheduler_resources = None
        if self._worker_resources is not None:
            worker_resources = self._worker_resources
            try:
                worker_resources.close()
            except Exception as exc:
                close_error = close_error or exc
            else:
                self._worker_resources = None
        for request_id in tuple(self._known_request_ids):
            self._control_plane.discard(request_id)
        self._known_request_ids.clear()
        self._request_preemptions.clear()
        self._pending_handoff_ids.clear()
        self._pending_worker_receipts.clear()
        self._finished_request_ids.clear()
        self._scheduler_lookup_metadata.clear()
        self._scheduler_lookup_observations.clear()
        self._prefix_cached_tokens.clear()
        self._active_worker_transfer = None
        self._registered_kv_caches.clear()
        if close_error is not None:
            raise RuntimeError("CacheBlend resource shutdown failed.") from close_error


__all__ = [
    "GptOssCacheBlendConnector",
    "GptOssCacheBlendMetadata",
    "GptOssCacheBlendWorkerMetadata",
]
