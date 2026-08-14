# Pinned source audit

Audit date: 2026-08-12 (initial audit 2026-08-11). This audit was read-only with respect to vLLM,
LMCache, CacheBlend, and `rag-system`. Temporary source archives were inspected
outside this repository; no upstream checkout or workload file was changed.

## Decision

**GO: the first connector-loading milestone is completely out of tree and does
not require a vLLM patch.** vLLM 0.19.1 explicitly accepts a Python module path,
imports the named connector class, and constructs it independently in scheduler
and worker roles. The upstream test suite exercises that external-module path.

**GO: the 100%-recomputation transfer milestone can also remain out of tree.**
The connector can return `(0, False)` from scheduler lookup, receive the fully
allocated grouped block tables, synchronously transfer candidate KV for
instrumentation, and allow ordinary GPT-OSS prefill to overwrite every loaded
slot. This establishes matching, metadata, transfer, mapping, writeback, and
output equivalence, but deliberately provides no speedup.

**HOLD: the connector API alone cannot express selective non-prefix
recomputation.** Its scheduler result is one scalar constrained to the largest
cached prompt prefix. The next experiment must exhaust the public out-of-tree
model registry and custom attention-backend registry. A small pinned patch is a
conditional fallback, not part of the first milestone.

## Audited artifacts

| Project | Exact source | Verification |
|---|---|---|
| vLLM | tag `v0.19.1`, commit [`b1388b1fbf5aaef47937fabe98931211684666a6`](https://github.com/vllm-project/vllm/commit/b1388b1fbf5aaef47937fabe98931211684666a6), tree `33b782e425e42d42851a33f7876e97a8deeabb29` | The lightweight tag resolves directly to this commit. A local GitHub tag archive used for the audit had SHA-256 `49ee6f462817e2e9a0dab47c7924c6b7716be712b7dcf54aed9fc144bee2e2cc`. |
| LMCache | tag `v0.4.3`, commit [`7f326118a2f1afc7801988dd02e3055bdf21ef6b`](https://github.com/LMCache/LMCache/commit/7f326118a2f1afc7801988dd02e3055bdf21ef6b) | PyPI attestation points to the same tag/commit. The 0.4.3 sdist SHA-256 is `dfa71984af7a79842eb635e64882b17604aebac0aa5e520e8f0e9d21df53ad20`; its `lmcache/` tree differs from the tag only by generated `_version.py`. |
| CacheBlend reference | public repository snapshot [`55ad02675939f783a38d579393527d218a7fd581`](https://github.com/YaoJiayi/CacheBlend/commit/55ad02675939f783a38d579393527d218a7fd581) | This is an attribution/reference source, not a dependency or compatibility target. The inspected archive SHA-256 was `69a21fe7af9307af5fce7137f4096c98e73c477fa13680cbc41ef822a46c50e6`. |
| RAG workload | read-only branch `slim`, commit `7f952009d458ea280ff26095e0cfebae4a4a194b` | Its 152-line `AGENTS.md` was read before project files. No files were created or modified there. |

The version pins agree at the package level: vLLM 0.19.1 and LMCache 0.4.3
require Python `>=3.10,<3.14` and target PyTorch 2.10.0. This project further
locks the CUDA wheel to `torch==2.10.0+cu128`.

## vLLM 0.19.1 integration audit

All links in this section point to the audited commit, not `main`.

### Connector discovery and construction

- [`KVTransferConfig`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/config/kv_transfer.py#L23-L73)
  defines `kv_connector`, `kv_role`, `kv_connector_extra_config`,
  `kv_connector_module_path`, and `kv_load_failure_policy`. The module path is
  explicitly documented as V1 dynamic loading.
- [`KVConnectorFactory._get_connector_class_with_compat`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/distributed/kv_transfer/kv_connector/factory.py#L103-L139)
  imports `kv_connector_module_path`, looks up the class named by
  `kv_connector`, and detects the old versus current constructor signature.
- [`KVConnectorFactory.create_connector`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/distributed/kv_transfer/kv_connector/factory.py#L43-L82)
  constructs isolated scheduler- and worker-role instances and passes the final
  `KVCacheConfig` to the current three-argument constructor.
- [`test_backwards_compatibility.py`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/tests/v1/kv_connector/unit/test_backwards_compatibility.py#L131-L176)
  tests an external module-path connector in both roles. This is direct source
  evidence for the out-of-tree loading decision.
- [`EngineArgs.add_cli_args`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/engine/arg_utils.py#L1293-L1318)
  exposes `--kv-transfer-config` to `vllm serve`.
- [`Scheduler.__init__`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/core/sched/scheduler.py#L117-L137)
  creates the scheduler-role connector with the finalized cache configuration.
- [`GPUWorker.initialize_from_config`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/worker/gpu_worker.py#L514-L536)
  initializes KV transfer before allocating worker cache tensors;
  [`ensure_kv_transfer_initialized`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/distributed/kv_transfer/kv_transfer_state.py#L51-L71)
  asks the same factory for the worker-role connector.
- [`GPUModelRunner.initialize_kv_cache`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/worker/gpu_model_runner.py#L6769-L6819)
  supplies the allocated per-layer tensors through `register_kv_caches` and the
  platform block-copy operation through `set_host_xfer_buffer_ops`.

The eventual configuration shape is therefore:

```json
{
  "kv_connector": "GptOssCacheBlendConnector",
  "kv_connector_module_path": "cacheblend_gpt_oss.vllm_compat.v0_19_1.connector",
  "kv_role": "kv_both",
  "kv_load_failure_policy": "fail"
}
```

The implemented class now exists at that exact module path. Connector loading,
the transfer-disabled control-flow mode, and the `transfer_100pct` lifecycle
are CPU-fake tested; real construction/model execution is still a manual
`solab-g3` gate and is not claimed as passed.

### V1 connector contract

[`KVConnectorBase_V1`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L170-L207)
is explicitly experimental. The plugin must implement or deliberately handle:

- Worker hooks: `register_kv_caches`, `start_load_kv`,
  `wait_for_layer_load`, `save_kv_layer`, `wait_for_save`, `get_finished`,
  load-error reporting, connector statistics, and shutdown.
- Scheduler hooks: `get_num_new_matched_tokens`, `update_state_after_alloc`,
  `build_connector_meta`, `update_connector_output`, and request completion.
- Hybrid completion: [`SupportsHMA.request_finished_all_groups`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L84-L120).

The decisive limitation is the documented contract of
[`get_num_new_matched_tokens`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/distributed/kv_transfer/kv_connector/v1/base.py#L449-L482):
it returns one count and may consider only the largest prompt prefix actually
available. `build_connector_meta` is also forbidden from modifying
`SchedulerOutput`. Arbitrary hit ranges or recompute positions cannot be
reported as scheduler-computed tokens through this API.

### Request lifecycle

1. [`Request`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/request.py#L58-L126)
   exposes the tokenized prompt and optional connector transfer parameters.
   [`EngineCore.step`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/engine/core.py#L380-L409)
   executes `schedule -> model_executor.execute_model -> update_from_output`.
2. [`Scheduler.schedule`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/core/sched/scheduler.py#L605-L644)
   performs local prefix lookup, then calls connector lookup. It adds the local
   and external scalar counts as a contiguous computed prefix.
3. The scheduler calls grouped
   [`KVCacheManager.allocate_slots`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/core/sched/scheduler.py#L746-L775),
   then `update_state_after_alloc` even when the connector returned zero
   external tokens. `KVCacheBlocks.blocks[i]` represents each cache group.
4. [`Scheduler._build_kv_connector_meta`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/core/sched/scheduler.py#L914-L954)
   carries an opaque planner result to the worker.
5. [`GPUModelRunner.execute_model`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/worker/gpu_model_runner.py#L3769-L4040)
   prepares contiguous inputs, positions, grouped slot mappings, and attention
   metadata, then enters a forward context around model execution.
6. [`KVConnectorModelRunnerMixin._get_kv_connector_output`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/worker/kv_connector_model_runner_mixin.py#L91-L129)
   binds scheduler metadata, calls `start_load_kv`, waits for save completion,
   collects errors/stats/completion, and clears metadata.
7. [`maybe_transfer_kv_layer`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/layers/attention/kv_transfer_utils.py#L14-L58)
   waits for a layer load immediately around the decorated attention operation
   and calls `save_kv_layer` afterward.
8. [`unified_kv_cache_update`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/layers/attention/attention.py#L662-L700)
   writes K/V into paged slots through the backend; `unified_attention` and
   `unified_attention_with_output` perform the actual attention call.
9. [`Scheduler.update_from_output`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/core/sched/scheduler.py#L1302-L1504)
   aggregates worker statistics, reacts to invalid blocks, advances requests,
   and processes transfer completion.
10. [`Scheduler._connector_finished`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/core/sched/scheduler.py#L2009-L2038)
    passes all hybrid group block tables to `request_finished_all_groups` before
    freeing them. `_update_from_kv_xfer_finished` handles delayed send/receive
    completion.

The stock block-copy helper transfers whole blocks, so moved token ranges that
are not block aligned require version-scoped token gather/scatter rather than
only [`copy_kv_blocks`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/distributed/kv_transfer/kv_connector/utils.py#L176-L222).

### Hybrid cache, GPT-OSS, and attention facts

- [`OAIAttention`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/models/gpt_oss.py#L67-L153)
  applies YaRN RoPE to Q/K before calling generic `Attention`, passes learned
  sinks as parameters, and selects a sliding window on even layers while odd
  layers use full attention.
- [`Attention.get_kv_cache_spec`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/layers/attention/attention.py#L537-L560)
  emits `SlidingWindowSpec` or `FullAttentionSpec`; `KVCacheConfig.kv_cache_groups`
  records their layer-name sets.
- vLLM normally auto-disables the hybrid allocator when any connector is
  configured unless the launch explicitly requests
  `--no-disable-hybrid-kv-cache-manager` ([configuration logic](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/config/vllm.py#L1227-L1244)).
  With it enabled, the factory rejects connectors that do not implement HMA.
- Sliding-window blocks may be dropped as the sequence advances, so their
  reusable KV must be captured during layer/prefill writeback, not only when the
  request finishes ([sliding manager](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/core/single_type_kv_cache_manager.py#L358-L399)).
- GPT-OSS caches post-RoPE K and ordinary V. A moved key therefore needs exact
  old-to-new YaRN correction; V does not. Sinks remain model/backend inputs and
  are never cache tokens.
- The released checkpoint's raw `config.json` uses top-level `rope_theta` plus
  a `rope_scaling` mapping. Before vLLM constructs `OAIAttention`, its pinned
  `get_config()` path calls
  [`patch_rope_parameters`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/transformers_utils/config.py#L377-L426),
  which merges those legacy fields into the finalized flat
  `hf_config.rope_parameters` mapping. GPT-OSS then reads that mapping directly
  ([`gpt_oss.py`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/models/gpt_oss.py#L80-L99)).
  The startup validator therefore checks the finalized mapping and deliberately
  rejects a raw/unpatched object even if it still exposes top-level
  `rope_theta`; accepting the latter could validate a value the model will not
  consume.
- On A100, the pinned FlashAttention backend rejects learned sinks below compute
  capability 9.0, while the Triton backend supports them. The plugin must check
  the selected backend at startup rather than assume it.

There is an important ordering hazard. Triton declares
`forward_includes_kv_cache_update=False`; generic `Attention.forward` performs
the cache update before entering the decorated attention call where
`wait_for_layer_load` runs. Ordinary prefix transfer writes disjoint slots, but
selective CacheBlend intentionally overlaps loaded and recomputed slots. The
100% milestone must finish its load synchronously before model forward. The
selective path needs a custom backend with load-before-selected-write ordering
or a pinned ordering patch.

### Public out-of-tree extension points beyond the connector

- [`vllm.general_plugins`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/plugins/__init__.py#L12-L82)
  executes installed plugin functions in API, engine, and worker processes.
- [`ModelRegistry.register_model`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/models/registry.py#L886-L938)
  explicitly registers external models and allows an existing architecture to
  be overridden lazily.
- [`AttentionBackendEnum.CUSTOM`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/attention/backends/registry.py#L34-L118)
  and [`register_backend`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/attention/backends/registry.py#L205-L262)
  explicitly support a third-party backend.

These justify an out-of-tree selective-recomputation spike, but do not yet prove
that sparse hidden-state/output invariants work. No patch should be written
until that spike reaches its stop/go gate.

### Follow-up attention ordering audit (2026-08-12)

The registry seam is real, but the pinned attention call order determines what
the first selective spike must implement. The relevant source evidence is:

- [`AttentionBackend`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/attention/backend.py#L46-L91)
  requires a backend name, implementation class, metadata-builder class, and
  KV-cache shape. Its default
  [`supports_sink`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/attention/backend.py#L199-L205)
  is false, and
  [`validate_configuration`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/attention/backend.py#L250-L308)
  rejects a sink-enabled configuration unless the selected backend explicitly
  supports sinks.
- [`TritonAttentionBackend`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/attention/backends/triton_attn.py#L257-L355)
  is the pinned sink-capable A100 path. It sets
  `forward_includes_kv_cache_update=False`, uses the paged shape
  `[blocks, 2, block, kv_heads, head_dim]`, and advertises sink support.
  [`TritonAttentionImpl`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/attention/backends/triton_attn.py#L357-L495)
  receives the learned sink tensor and passes it to the unified attention
  operation; sinks are not cache rows.
- In [`Attention.forward`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/layers/attention/attention.py#L452-L500),
  the split-update path calls `unified_kv_cache_update` before
  `unified_attention_with_output`. That update invokes
  `impl.do_kv_cache_update` through
  [`unified_kv_cache_update`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/layers/attention/attention.py#L662-L684).
  A selective implementation must therefore make its custom implementation's
  update operation skip accepted cached rows; merely waiting in the connector
  decorator is too late for overlapping load/recompute slots.
- The stock [`TritonAttentionImpl.do_kv_cache_update`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/attention/backends/triton_attn.py#L575-L606)
  accepts the layer, flattened post-RoPE K/V, paged cache, and slot mapping,
  then calls `triton_reshape_and_cache_flash` for every slot. The new
  tensor-free selective planner/updater mirrors this exact input shape and
  physical-slot contract, but only for its recompute spans; it is not yet an
  implementation of that vLLM class.
- [`maybe_transfer_kv_layer`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/layers/attention/kv_transfer_utils.py#L14-L58)
  waits for a layer load on entry and saves it on exit. The pinned
  [`GPUModelRunner.execute_model`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/worker/gpu_model_runner.py#L4029-L4040)
  enters the connector context before `_model_forward`, so the current
  100%-recompute connector can complete its synchronous load before any stock
  Triton update. This does not provide a sparse-row mask.

This evidence changes the M6 boundary as follows:

1. Registration of a model override and a `CUSTOM` backend remains completely
   out of tree; no vLLM patch is needed to load either class.
2. The first custom backend must preserve the exact Triton constructor,
   metadata, sink, and paged-layout contracts while replacing the cache-update
   operation with a plan-aware implementation. A model override must preserve
   full-shaped hidden-state/logit outputs and runner sampling indices.
3. No general-plugin entry point is enabled yet. Registering incomplete classes
   in every vLLM process would turn an unvalidated experiment into an implicit
   serving default. The entry point is added only after the M3--M5 GPU gates
   supply the baseline and transfer evidence.
4. There is no evidence for a patch at this point. If the pinned single-request
   model/backend spike fails an output-shape, slot-mapping, or ordering
   invariant after these public registries are exhausted, record that failure
   and design a version-scoped patch before writing one.

The CPU-side guard for this boundary is
`cacheblend_gpt_oss.vllm_compat.v0_19_1.selective_registry`. It is deliberately
not an entry point: it accepts only a proof object and project-owned lazy class
paths, and it treats a partial registry mutation as terminal for that process.
The companion `cacheblend_gpt_oss.gpt_oss.selective_kv` planner supplies the
row-to-slot split that a custom `do_kv_cache_update` must consume; it does not
mutate cache tensors or alter the stock runner. Its injected updater is
CPU-tested for all-preflight-before-mutation, selected-row-only writes, and
fail-closed shape/dtype/device checks. `validate_slot_mapping` also checks the
flattened per-token slot vector that the pinned method receives against every
target block/offset, rejecting negative/padding entries and cross-group
mismatches before mutation; the updater requires one vector per layer rather
than making this check optional. These contracts remain dormant until the
M3--M5 GPU gates and a real Triton implementation exist.

## LMCache 0.4.3 integration audit

### Ordinary V1 connector and engine

LMCache ships
[`LMCacheConnectorV1Dynamic`](https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/integration/vllm/lmcache_connector_v1.py#L30)
and documents this module-path configuration:

```text
kv_connector = "LMCacheConnectorV1Dynamic"
kv_connector_module_path = "lmcache.integration.vllm.lmcache_connector_v1"
```

The wrapper delegates to
[`LMCacheConnectorV1Impl`](https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/integration/vllm/vllm_v1_adapter.py#L442),
whose exact metadata/state types include `LoadSpec`, `SaveSpec`, `DisaggSpec`,
`RequestTracker`, `ReqMeta`, and `LMCacheConnectorMetadata`. It implements the
same scheduler lookup/allocation/metadata and worker load/layer-save lifecycle
described above.

Reusable storage operations are on
[`LMCacheEngine`](https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/v1/cache_engine.py#L78):
`lookup`, `lookup_unpin`, `retrieve`, `retrieve_layer`, `store`, and
`store_layer`. `LMCacheManager` and `VllmServiceFactory` construct the service;
`LMCacheMetadata` namespaces model, parallel-rank, dtype, layout, layer-group,
engine, and connector information.

This connector is not a GPT-OSS solution. In
[`RequestTracker.from_new_request`](https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/integration/vllm/vllm_v1_adapter.py#L144-L181),
grouped block IDs are reduced to `new_request.block_ids[0]` with a TODO for
multiple groups. LMCache `KVLayerGroupsManager` groups tensors by physical shape
and dtype; that is not vLLM's full/sliding hybrid allocation contract.

The exact vLLM release also ships
[`LMCacheMPConnector`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py#L421-L479),
which imports LMCache's exact
[`LMCacheMPSchedulerAdapter` and `LMCacheMPWorkerAdapter`](https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/integration/vllm/vllm_multi_process_adapter.py#L179-L220).
It is not a CacheBlend connector and cannot be used as the GPT-OSS base:
`reformat_block_ids` explicitly raises when more than one cache group is present
and instructs the operator to disable the hybrid manager
([source](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py#L62-L76)).

### Blending code that is present

LMCache 0.4.3 contains two relevant but incomplete paths:

1. Legacy in-process
   [`LMCBlender`](https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/v1/compute/blend/blender.py#L18)
   implements `process_qkv`, `blend_layer`, and `blend`. Its
   `SegmentTokenDatabase` splits at a configured separator and hashes segments
   without a prompt prefix.
2. Multiprocess
   [`BlendEngineV2`](https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/v1/multiprocess/blend_server_v2.py#L278)
   implements `cb_register_kv_cache`, `cb_lookup_pre_computed`,
   `cb_store_pre_computed`, `cb_retrieve_pre_computed`, `cb_store_final`, and
   `run_cache_server`. `BlendTokenRangeMatcher`, `CBMatchResult`,
   `MessageQueueClient`, `CudaIPCWrapper`, and `IPCCacheEngineKey` are useful
   version-pinned server/protocol building blocks.

The legacy path is explicitly unsuitable:

- [`infer_model_from_vllm`](https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/v1/compute/models/utils.py#L14-L30)
  recognizes only Llama, Qwen2, and Qwen3 adapter shapes; it assumes
  `self_attn`, whereas pinned GPT-OSS uses `attn`/`OAIAttention`.
- [`validate_rope_params`](https://github.com/LMCache/LMCache/blob/7f326118a2f1afc7801988dd02e3055bdf21ef6b/lmcache/v1/compute/positional_encoding.py#L85-L109)
  rejects any non-null RoPE scaling. GPT-OSS YaRN is therefore unsupported.
- It has no learned-sink behavior and no vLLM hybrid-group handling.

BlendEngineV2 is useful only as a server-side substrate. Its CUDA IPC staging
layout is one contiguous rank-four `[2, layers, tokens, dimension]` tensor. It
does not scatter into vLLM paged/hybrid groups, correct positions, select rows,
preserve sinks, account for scheduler work, or emit the required metrics.

`BlendTokenRangeMatcher` uses rolling hashes and a fixed direct-address table.
Table collisions can overwrite entries, and a returned match is not verified
against the complete token sequence. The plugin must treat it as candidate
generation and perform a strong digest plus exact-token check before transfer.
The first live moved-document gate additionally observed that the public
matcher returned no candidate for an exact 256-token document moved from
offset 0 to offset 17, after storage and sidecar publication both completed.
The version-scoped server wrapper therefore replaces only this in-memory
candidate table with a bounded, lock-protected exact-token index. LMCache still
owns object storage, prefetch, and retrieval, and the connector still performs
its independent namespace, cache-key, SHA-256, and complete-token checks.

The pinned 0.4.3 server also has a live store-completion race in
`BlendEngineV2._cb_store_gpu_copy`: it records the client-visible CUDA event and
only then queues `storage_manager.finish_write`. A subsequent lookup can observe
the fingerprint, fail the storage prefetch, and evict that freshly stored
fingerprint before the storage index is committed. An initial backport queued
the callback before the event, but a sequential live source/target gate still
returned no storage-backed candidates. The version-scoped server entry point
therefore synchronizes the server copy stream, calls `finish_write` directly,
and records the client-visible event last. It also reports bounded aggregate
matcher registration/window/hit counts so a remaining miss can be attributed
without logging request IDs, token IDs, hashes, or prompt content. The exact
pinned sources are linked in `storage/lmcache_server_v0_4_3.py`.

The pinned `TokenHasher` also delegates its initial rolling-hash seed to
vLLM's process-global `NONE_HASH`. With no `PYTHONHASHSEED`, pinned vLLM assigns
that global from `os.urandom(32)`. The standalone LMCache server and the
scheduler/worker processes consequently derive different LMCache object keys
for identical chunks. A live gate exposed this after the exact matcher and L1
prefetch both returned one hit while sidecar binding reported zero found
tokens. The version-scoped bindings now install LMCache's own deterministic
fallback seed, `hash_func((0, (0,), None))`, on each LMCache hasher instance.
They do not mutate vLLM's global prefix-cache hash state.

### Code that is absent

The exact LMCache tag, PyPI sdist/wheel, and tests contain no occurrence of:

- `LMCacheMPCBConnector`
- `CBKVConnector`
- `MPCBConnector`
- `lmcache_cacheblend`

The exact vLLM tag contains `LMCacheMPConnector`, not an MPCB connector. A newer
LMCache development operator manifest names `CBKVConnector` with module path
`lmcache_cacheblend.connector`, but that development tree still contains no
Python implementation. It expects a separately supplied image. No published
private connector is assumed or required by this plan.

## Original public CacheBlend audit

The public repository is a complete vLLM 0.4.1 research fork, not a modern V1
plugin. The relevant changes are tightly coupled across:

- [`llama.py`](https://github.com/YaoJiayi/CacheBlend/blob/55ad02675939f783a38d579393527d218a7fd581/vllm_blend/vllm/model_executor/models/llama.py),
  which collects K/V, applies position correction, selects important rows, and
  shrinks residuals/positions after a check layer.
- [`xformers.py`](https://github.com/YaoJiayi/CacheBlend/blob/55ad02675939f783a38d579393527d218a7fd581/vllm_blend/vllm/attention/backends/xformers.py),
  which compares cached and new tensors, chooses top-k recomputation rows,
  merges K/V, and constructs partial attention masks.
- [`paged_attn.py`](https://github.com/YaoJiayi/CacheBlend/blob/55ad02675939f783a38d579393527d218a7fd581/vllm_blend/vllm/attention/ops/paged_attn.py),
  which adds blend-specific cache writes.
- [`model_runner.py`](https://github.com/YaoJiayi/CacheBlend/blob/55ad02675939f783a38d579393527d218a7fd581/vllm_blend/vllm/worker/model_runner.py),
  which adjusts sampling indices after the hidden-state row count changes.

Those concepts inform the planner and tests, but the code cannot be copied as a
drop-in implementation: it targets Llama/XFormers and predates V1 connectors,
GPT-OSS, YaRN, learned sinks, and hybrid KV groups. Any adapted code must retain
the original Apache-2.0 attribution and identify the source commit.

## Read-only RAG workload findings

The external workload already provides the desired transparent boundary:

- `scripts/serve_oss_generator.sh` launches the local GPT-OSS model on
  `solab-g3`, enables the OpenAI tool parser, and passes
  `--kv-transfer-config` for CacheBlend modes. It rejects a missing connector
  config.
- `VllmResponsesClient.complete` posts non-streaming requests to
  `/v1/responses`. The payload contains `model`, `input`, `tools`,
  `max_output_tokens`, `truncation`, and `reasoning`; it has no segment or cache
  field. `cache_mode` is experiment metadata, not request data.
- The agent resends the entire append-only item history. Retrieved hits appear
  inside `function_call_output` as pretty JSON objects containing `docid`, a
  floating-point `score`, and `snippet`.
- The stable reusable content is therefore the tokenized snippet, not
  necessarily the score, rank, JSON punctuation, or surrounding tool item.
- Harmony reasoning, messages, function calls, normalized MCP-style calls, and
  forced-decision recovery are all above the connector boundary.
- Existing metrics already normalize prefix hits, CacheBlend-hit/recomputed
  tokens when supplied, TTFT, and prefill/decode/queue latency. This project now
  exposes its aggregate transfer metrics through vLLM's `/metrics` endpoint;
  no RAG request-schema change is needed. A future per-request artifact schema
  remains a separate gate before changing RAG analysis code.

No RAG runtime dependency or prompt change is needed for a transparent
CacheBlend-enabled endpoint. Future work should only configure that endpoint
and record the model revision, vLLM/LMCache versions, connector/config digest,
deployment ID, and CacheBlend commit SHA. Any proposed file edits remain
document-only until separately authorized.

## Pinned numerical-observability boundary

vLLM 0.19.1's GPT-OSS/Harmony Responses handler explicitly rejects requests
that include output logprobs in
[`OpenAIServingResponses._validate_create_responses_input`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/entrypoints/openai/responses/serving.py#L293-L302).
The first deterministic numerical gate therefore cannot obtain logits through
the production `/v1/responses` route.

The pinned completions route provides a source-supported test surface without
changing model execution: [`CompletionRequest.prompt`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/entrypoints/openai/completion/protocol.py#L42-L60)
accepts exact integer token IDs, and its token-ID output fields are defined at
[`CompletionRequest.return_tokens_as_token_ids`/`return_token_ids`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/entrypoints/openai/completion/protocol.py#L126-L142).
With server `--max-logprobs -1`, pinned
[`SamplingParams._validate_logprobs`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/sampling_params.py#L638-L653)
expands the limit to the model vocabulary. The released GPT-OSS configuration
is additionally rejected unless its vocabulary is exactly 201,088, matching
the pinned [vLLM GPT-OSS kernel fixture](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/tests/kernels/moe/test_gpt_oss_triton_kernels.py#L195-L205).

The resulting artifact is a complete normalized output-logprob vector, not an
unnormalized raw-logit tensor. It preserves every relative final-logit
difference except that the pinned completion serializer clamps non-finite
values to `-9999.0` in
[`_create_completion_logprobs`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/entrypoints/openai/completion/serving.py#L610-L638).
Artifacts label this representation precisely and freeze the BF16
full-prefill-versus-full-prefill envelope before CacheBlend is run. Harmony and
tool correctness must still be checked separately on `/v1/responses`.

The pinned [`ResponsesRequest.input`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/entrypoints/openai/responses/protocol.py#L119-L154)
accepts both input and prior output items. The upstream
[`test_named_tool_use`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/tests/entrypoints/openai/responses/test_function_call.py#L111-L169)
provides direct evidence for replaying a function call followed by matching
`function_call_output`; the exact GPT-OSS
[`test_harmony.py`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/tests/entrypoints/openai/responses/test_harmony.py#L82-L105)
provides the completed-response and low-effort reasoning contract. The local
manual harness follows those pinned surfaces while retaining every prior output
item to match the external RAG client's append-only history model.

## Patch conclusion

No vLLM patch is supported by current evidence for connector loading,
position-independent lookup, the 100%-recomputation transfer proof, YaRN key
correction, or group-aware scatter/gather.

A pinned patch becomes justified only if the out-of-tree GPT-OSS model override
and custom backend cannot preserve vLLM's runner/output invariants, or if native
scheduler accounting for arbitrary recompute positions is required. The likely
seam would be deliberately small but cross-cutting: `SchedulerOutput`,
`Scheduler.schedule`, `GPUModelRunner._prepare_inputs`/slot mapping, and a formal
model/attention hook. That decision is deferred to the selective-data-plane
spike.
