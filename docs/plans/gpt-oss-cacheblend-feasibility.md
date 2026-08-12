# GPT-OSS CacheBlend feasibility plan

## Current decision and next action

The exact pinned source audit is complete. The result remains a **go** for an
out-of-tree connector and a **no-patch** decision for the 100%-recomputation
transfer proof. The connector, scheduler lookup, LMCache transport, persistent
sidecar, worker staging bridge, YaRN corrector, and full/sliding scatter-gather
path are now implemented and CPU-tested. Identifier-free aggregate connector
metrics are wired through vLLM 0.19.1's public stats/Prometheus hooks. The next
gate is manual CUDA and connector/model execution on `solab-g3`; no such pass
is claimed yet.

The connector stats schema now includes separate position-correction and
selective-recomputation latency histograms. Both are explicitly zero in the
100%-recompute path and can become nonzero only when a future worker supplies
measured values; this keeps the required observability surface stable without
fabricating GPU measurements.

Selective non-prefix computation remains conditional. The V1 connector API can
carry an opaque plan but can only credit a contiguous cached prefix to the
scheduler. An external GPT-OSS model override and custom attention backend must
be tested before deciding whether a pinned vLLM patch is necessary.

## Milestone map

| Milestone | State | Primary result | Patch policy |
|---|---|---|---|
| M0. Pinned audit and scaffold | Complete | Exact integration boundary and CPU/GPU test scaffold | No patch |
| M1. Connector-load smoke test | Implemented; GPU pending | Scheduler and worker load an external connector | No patch |
| M2. Segmentation and verified lookup | CPU complete | One document is found after moving positions with exact verification | No patch |
| M3. 100% recomputation transfer proof | Implemented; GPU/logit gate pending | Candidate KV is transferred, verified, then fully overwritten | No patch |
| M4. GPT-OSS YaRN correction | Implemented; CUDA gate pending | Shifted cached K matches direct target-position K | No patch |
| M5. Hybrid groups and sinks | CPU layout/data-plane complete; model gate pending | Full/sliding layers map correctly and sink behavior is unchanged | No patch |
| M6. Out-of-tree selective-data-plane spike | Pinned boundary audited; implementation waits for M3--M5 GPU evidence | Registered model/backend must skip selected rows while preserving runner output shape | Decide at gate |
| M7. Reduced recomputation correctness | CPU policy contract; GPU pending | Deterministic check-layer row plans; error/work curves still require model runs | No optimization yet |
| M8. Responses/Harmony/multi-turn validation | CPU harness and offline evidence validator complete; GPU/API run pending | Transparent validated endpoint with native source-credit proof | Patch only for proven API blocker |
| M9. Controlled benchmark | CPU evidence contract; GPU pending | Full-prefill and prefix-cache comparisons with complete metrics and confidence intervals | Optimize only after correctness |

## M0: pinned audit and repository scaffold

Completed work:

- Audited vLLM tag `v0.19.1` at
  `b1388b1fbf5aaef47937fabe98931211684666a6`.
- Audited LMCache tag/PyPI release `v0.4.3` at
  `7f326118a2f1afc7801988dd02e3055bdf21ef6b`.
- Audited public CacheBlend snapshot
  `55ad02675939f783a38d579393527d218a7fd581` as attribution/reference code.
- Inspected `rag-system` read-only after reading its `AGENTS.md`.
- Recorded exact scheduler, connector, model runner, attention, writeback,
  hybrid-group, GPT-OSS, LMCache, and completion symbols in
  [the source audit](../source-audit.md).
- Created a dependency-light package/test scaffold with exact optional runtime
  pins. Subsequent commits added the implementation without vendoring either
  upstream repository.

Stop/go result: **GO**. Dynamic loading is an explicit and tested vLLM 0.19.1
path. No patch is justified.

## M1: connector-load smoke test

Implementation status: the external three-argument `SupportsHMA` connector and
all pinned startup guards are complete. CPU contract tests simulate both vLLM
roles. The actual import/construction and `/v1/responses` checks remain manual
on `solab-g3`.

Deliverables:

- Add
  `cacheblend_gpt_oss.vllm_compat.v0_19_1.connector.GptOssCacheBlendConnector`.
- Use the current `KVConnectorBase_V1` three-argument constructor and implement
  `SupportsHMA` from the first revision.
- Separate scheduler and worker state. All hooks are safe no-ops except bounded
  lifecycle logging/metrics.
- Validate exact runtime versions, model architecture, HMA enabled, TP=1,
  default V1 GPU model runner, and supported sink-aware attention backend.
- Add a pure unit test with injected/stubbed vLLM types where possible and an
  exact pinned-runtime import/construction test on `solab-g3`.
- Start vLLM with the external module path and complete one ordinary
  `/v1/responses` request with reuse disabled.

Go criteria:

- The external class is imported in both scheduler and worker processes.
- Both receive the non-null `KVCacheConfig`; the worker registers all 24 layer
  tensors and both attention-spec groups.
- `request_finished_all_groups` receives all group block tables.
- With reuse disabled, deterministic output/logits equal the no-connector
  baseline within the frozen full-vs-full tolerance.
- Startup rejects a wrong vLLM version, disabled HMA, V2 model runner, or wrong
  model architecture with a clear message.

Stop criteria:

- If loading fails, first reproduce the upstream external-connector test and
  fix packaging/module visibility. A loading failure does not authorize a vLLM
  patch because the pinned source explicitly supports this path.
- Do not proceed if the connector only works after disabling the hybrid
  allocator.

## M2: position-independent segmentation and verified lookup

Implementation status: complete for CPU. The live transparent connector uses
LMCache's 256-token store chunks plus rolling query windows; delimiter and
generic segmentation remain planner-level APIs. Candidates require namespace,
cache-key, SHA-256, and exact-token verification. A token-hash object associated
with multiple absolute K source positions is rejected as ambiguous.

Deliverables:

- Define dependency-injected `Segmenter`, `CacheIndex`, `CacheTransport`, and
  immutable plan/record types without importing vLLM.
- Implement a deterministic delimiter segmenter for the first integration
  fixture and a fixed-chunk/rolling-query segmenter for transparent prompts.
- Use SHA-256 domain-separated fingerprints over canonical token IDs, excluding
  absolute position.
- Define a cache namespace containing model artifact digest, tokenizer,
  topology, versions, dtypes, block/group schema, and YaRN parameters.
- Treat LMCache `BlendTokenRangeMatcher` output as candidates. Verify a strong
  digest and the complete token sequence before accepting any hit.
- Add bounded rejection reason codes and required counters/timers.

CPU test cases:

- Same tokens at the same position -> exact match.
- Same document moved by one token and by a non-chunk-aligned offset -> match.
- Two documents swapped -> each maps to the new range.
- One-token mutation, truncation, separator mutation, and cache miss -> reject.
- Duplicate document occurrences -> deterministic one-to-one range mapping.
- Synthetic rolling-hash collision -> reject after exact-token verification.
- Different tokenizer/model/config namespace -> no lookup or hard rejection.
- Rank/score/JSON wrapper changes around an unchanged snippet -> inner snippet
  chunks still match.

Go criteria:

- The moved-document plan contains correct old/new half-open token ranges with
  no position in its identity digest.
- Every accepted candidate has exact-token evidence; counters reconcile
  `found = loaded_candidates + rejected` at planning time.
- The entire planner test suite runs with no GPU, model weights, vLLM, LMCache,
  or network service.

Stop criteria:

- Do not use a matcher result that cannot return or validate the source token
  sequence.
- If transparent rolling matches have unacceptable ambiguity, retain the
  delimiter policy for the first milestone and redesign segmentation before
  any approximate reuse.

## M3: 100% recomputation transfer proof

Implementation status: the complete control/data path is wired into the
connector. GPU evidence that transfer occurred, was overwritten, and preserved
deterministic logits is still required.

The first live artifact harness and exact manual commands are now implemented
in `docs/runbooks/solab-g3-moved-document-correctness.md`. Because pinned
vLLM's Harmony Responses service rejects GPT-OSS logprobs, this gate uses raw
token IDs through `/v1/completions` and compares all 201,088 normalized output
logprobs. `/v1/responses` Harmony/tool/multi-turn behavior remains a separate
M8 contract gate. No live result has yet been supplied from `solab-g3`.

Deliverables:

- Return `(0, False)` from `get_num_new_matched_tokens` while retaining the
  non-prefix plan internally.
- Bind requested ranges to all destination group block tables in
  `update_state_after_alloc` and carry them in connector metadata.
- Wrap LMCache BlendEngineV2 protocol/IPC access behind the injected transport.
- Retrieve into a contiguous staging tensor; validate layer, token, head,
  dtype, stride, and digest metadata before scatter.
- Complete the transfer synchronously in `start_load_kv` before stock Triton can
  update overlapping slots.
- Instrument pre-load, post-load, and post-prefill checksums/sampled tensors so
  tests prove that KV was loaded and then overwritten.
- Capture recomputed full- and sliding-layer K/V during `save_kv_layer`, not
  solely at request completion.

The CPU-only `correctness.transfer` sidecar schema now defines this evidence
boundary: every layer must carry K/V digests for destination-before, loaded
source, and fresh-prefill values, with exact source/loaded and target/prefill
agreement. It remains an empty contract until a worker-side `solab-g3` probe
supplies real tensor samples.

The dependency-free `TransferEvidenceBuilder` now supplies the worker probe's
assembly seam. It accepts only the next canonical layer, rejects duplicates or
out-of-order samples, requires all 24 layers before finalization, and becomes
immutable after producing the sidecar. It performs no tensor sampling itself;
the CUDA probe still must provide real digest values.

The moved-document capture script also waits for the pinned connector's
`store_tokens_completed` counter after each source/target request. A request
counter alone is not sufficient evidence that the source document is visible
to the next lookup; each request's eligible/completed chunk deltas must also
match and its store-fallback delta must be zero.

GPU correctness sequence:

1. Run prompt A with document D at source position and store its reusable KV.
2. Run prompt B with identical D at a different position and reuse enabled at
   100% recomputation.
3. Assert nonzero candidate and loaded token counts.
4. Before forward, compare loaded staging/destination samples with source KV.
5. After forward, compare destination samples with fresh prompt-B KV and prove
   candidate slots were overwritten.
6. Compare prompt-B per-layer hidden states or final logits with an ordinary
   full-prefill prompt-B run under identical deterministic settings.

Go criteria:

- Transfer coverage and checksums pass for every tested layer/group.
- No loaded KV affects attention output in this phase.
- CacheBlend-versus-full error is no worse than the frozen full-versus-full
  numerical envelope, and the selected next token agrees.
- Metrics show requested/found/loaded tokens, all prompt rows recomputed, and
  exactly zero effective saved-prefill fraction.
- A miss and every injected validation/transfer failure either visibly fail or
  execute ordinary full prefill according to the configured policy.

Stop criteria:

- Any output agreement without independent evidence that transfer occurred is
  insufficient.
- Any asynchronous overlap between load and stock cache update blocks progress
  until ordering is made deterministic.
- Do not lower the recomputation ratio at this milestone.

## M4: GPT-OSS YaRN/RoPE correction

Implementation status: the production BF16 CUDA corrector and CPU arithmetic
tests are complete. The opt-in CUDA test has been authored but not run here.

Deliverables:

- Implement the GPT-OSS-only delta rotation using the exact loaded YaRN inverse
  frequencies, NeoX pairing, truncation behavior, and float32 trig path.
- Record source absolute position per token and compute target positions from
  the pinned runner metadata.
- Rotate K only; copy V unchanged; never reapply YaRN magnitude scaling.
- Measure correction latency separately from transport/scatter.

CPU tests:

- Directly rotate raw K at source and target positions, then compare corrected
  source K to directly generated target K.
- Cover zero delta, positive/negative deltas, the configured original-context
  boundary, long-context positions, every head pair, and noncontiguous ranges.
- Reject mismatched YaRN configuration, head dimension, dtype, or model
  namespace.

GPU tests:

- Compare corrected cached keys with keys captured from fresh prompt-B prefill
  before attention for representative even and odd layers.
- Compare downstream attention output with the corresponding fresh-KV path.

Go criteria:

- CPU float32 correction is within a predeclared analytic tolerance.
- BF16/GPU error stays inside a separately frozen direct-RoPE baseline envelope.
- V is byte-identical when its dtype/layout permits, otherwise tensor-equal.
- Moving to the same position is an identity operation within dtype tolerance.

Stop criteria:

- A generic RoPE helper that rejects or approximates YaRN is not acceptable.
- Any unexplained position-dependent drift blocks selective recomputation.

## M5: hybrid full/sliding groups and attention sinks

Implementation status: exact 24-layer layouts, grouped physical spans,
scatter/gather, null-block rejection, and the rule that sinks never enter the
data plane are CPU-tested. Real Triton/sink parity and reclamation behavior are
still GPU/model gates.

Deliverables:

- Validate all 24 layer names and their `KVCacheGroupSpec` membership.
- Map even sliding-window-128 layers and odd full-attention layers without
  assuming one physical cache tensor.
- Store sliding KV while its blocks are live and exercise multi-step/chunked
  prefill reclamation.
- Verify that sinks are absent from cache records and passed unchanged to the
  sink-capable backend.
- Add group- and layer-specific metrics without high-cardinality labels.

GPU matrix:

- Early, middle, and final even sliding layers.
- Early, middle, and final odd full layers.
- Document lengths below, equal to, and above 128 tokens.
- Source/destination positions inside and outside a current sliding window.
- Single-step and chunked prefill.
- Sink-enabled stock baseline versus connector-disabled/no-reuse path.

Go criteria:

- No layer/group is missing, duplicated, or written through the wrong block
  table.
- Sliding captures remain retrievable after vLLM reclaims old request blocks.
- Sink tensor values and sink-aware attention results match baseline.
- Full and sliding layer correctness reports are separately visible.

Stop criteria:

- Do not disable HMA or sinks to make a test pass.
- Do not treat request-final block tables as complete evidence for sliding KV.

## M6: out-of-tree selective-data-plane spike

Follow-up audit of the exact pinned source is complete. The public registries
are sufficient to load a lazy GPT-OSS model override and a `CUSTOM` attention
backend, but the stock split-update path calls `unified_kv_cache_update` before
the decorated attention wait. The custom implementation must therefore make
`do_kv_cache_update` plan-aware; a connector-only wait cannot protect
overlapping loaded/recomputed rows. Triton remains the sink-capable target and
its paged KV shape and sink argument must be preserved. See the
[follow-up source audit](../source-audit.md#follow-up-attention-ordering-audit-2026-08-12)
for the pinned line evidence.

No general-plugin entry point is enabled yet. It would execute in every API,
engine, and worker process and could silently activate incomplete classes. The
entry point is a deliverable only after M3--M5 provide real baseline, transfer,
YaRN, hybrid-group, and sink results on `solab-g3`.

The repository now contains a dormant CPU-only `ForwardRowPlan` and
`ForwardRowPlanContext` contract. It validates 24 layer selections, exact
recompute/cached complements, and nested-context/lifetime failures without
importing vLLM or Torch. The current connector does not consume it and remains
100% recompute.

`vllm_compat.v0_19_1.selective_registry` now guards the public registration
calls with the same boundary: all M3--M5 proof flags must be true, class paths
must remain inside this project, `CUSTOM` is bound before the model override,
and a partial registration fails closed for the process lifetime. It is a
library seam only; no plugin entry point is enabled.

The tensor-free `gpt_oss.selective_kv` planner now splits complete hybrid
layer spans by those row selections and preserves source-position metadata and
physical destination slots. It is a structural contract for
`do_kv_cache_update`; its injected updater preflights all model/cache views
before any copy and records only recompute-row writes. It is not wired into the
live attention path. The pinned stock method still calls
`triton_reshape_and_cache_flash` for every slot, so a real custom backend must
adapt this contract and pass the M6 GPU ordering/shape gate before it can claim
selective execution or correctness.

The companion `gpt_oss.forward_output` contract guards the model-runner side:
a future model override must preserve the full hidden-state row shape and the
pinned runner's logits-index ordering. It is CPU-tested and dormant until
M3--M5 GPU evidence permits registering a concrete model/backend.

This milestone decides the patch boundary.

Deliverables:

- Register an idempotent `vllm.general_plugins` entry point in all processes.
- Lazily override only `GptOssForCausalLM` through `ModelRegistry.register_model`.
- Register a `CUSTOM` sink-aware backend based on pinned Triton semantics.
- Publish the immutable plan through a worker-local per-forward context.
- At a configurable check layer, recompute 100% initially, then demonstrate one
  controlled set of cached rows while preserving a full-shaped output tensor
  and the model runner's logits indices.
- Enforce load-before-selected-write and merge recomputed K/V into exact
  destination slots.
- Use eager, single-request, TP=1, no speculative decoding, no CUDA graphs, and
  no compile optimization.

Out-of-tree go criteria:

- The model override and backend load without editing vLLM.
- At a logically 100% selection, logits/hidden states match stock GPT-OSS.
- A controlled sparse selection preserves model-runner output shape, sampling
  position, residual flow, MoE routing for selected rows, and connector
  completion.
- Loaded and recomputed rows merge without a race or stale-slot read.
- The implementation does not monkey-patch vLLM functions or copy the whole
  GPT-OSS/vLLM model tree.

Patch trigger:

A pinned patch is allowed only if an evidence note demonstrates at least one of
these failures after the public registries are exhausted:

- vLLM requires material computation of every intermediate row to satisfy
  runner/output invariants and a full-shaped scatter cannot preserve them;
- arbitrary target slot mappings cannot be supplied to the backend/model
  override;
- stock runner logits/sampling selection cannot address the preserved final
  row; or
- native scheduler accounting of arbitrary recompute positions becomes a hard
  experiment requirement rather than a desired optimization.

If triggered, stop feature work and write a patch design first. The candidate
seam is `SchedulerOutput`, the prefix-scalar portion of `Scheduler.schedule`,
`GPUModelRunner._prepare_inputs`/slot mapping, and a formal GPT-OSS
model/attention hook. The patch must apply only to the audited commit and reject
every other version.

## M7: reduce recomputation only under measured correctness

The CPU-only `gpt_oss.selective_policy` contract is now implemented. It follows
the audited CacheBlend check-layer idea without copying its old vLLM code:
verified cached rows are ranked by injected importance scores, ties choose the
lower token position, the configured fraction is selected, and uncached rows
plus a forced suffix always remain recomputed. It emits the exact retained
cached/recomputed ranges for all 24 layers and remains disconnected from the
live connector. Its descending ratio sweep exposes a work curve locally while
requiring explicit externally supplied measurements before exposing error or
latency curves; it does not fabricate GPU evidence.

`gpt_oss.selective_policy_io` now provides a strict, non-sensitive JSON
artifact for this sweep. It records the ratio, verified cached/recomputed row
ranges, and the work curve. Error and latency fields are either present for
every point or absent for every point; a partial measurement set is rejected.
The artifact digest and the `scripts/validate_selection_sweep.py` report make
future `solab-g3` results reproducible without storing prompts, token IDs,
fingerprints, or request identifiers. This remains an experiment artifact and
does not connect the dormant policy to the serving connector.

Deliverables:

- Implement an injected selection policy modeled on CacheBlend's check-layer
  importance comparison, with deterministic tie-breaking.
- Evaluate successively lower ratios; do not preselect a production ratio.
- Record exactly which token rows are recomputed at each layer/check point.
- Emit an error curve and latency/work curve for every test case.
- Write one selection-sweep artifact per test case and validate it on the
  authoring workstation before comparing ratios across runs.

Go criteria for each lower ratio:

- All cache identity, correction, group, sink, and merge checks still pass.
- Logit/hidden-state error, top-token agreement, and task-output differences are
  reported against the same full-prefill request.
- The measured token-layer work reduction agrees with connector/model metrics.
- A ratio becomes a benchmark candidate only after a written accuracy budget is
  chosen from observed data.

Stop criteria:

- Stop lowering the ratio at the first accuracy-budget violation.
- Fluent or semantically plausible text cannot override a logit/hidden-state
  failure.
- Do not begin fused CUDA optimization to rescue a correctness failure.

## M8: `/v1/responses`, Harmony, tools, and multi-turn

Implementation status: the dependency-free response-item validator, exact
three-turn append-only harness, bounded structural report, independent offline
evidence validator/digest, and manual `solab-g3` commands are implemented. No
model/API run has been supplied, so M8 remains pending. See
`docs/runbooks/solab-g3-responses-contract.md`.

Required API scenarios:

- Plain response with Harmony reasoning summary and final message.
- One local tool call followed by `function_call_output` and a final response.
- Normalized Harmony/MCP-style tool recipient used by the workload.
- Multi-turn append-only history with a repeated retrieved snippet moving as
  new items are appended.
- Reordered retrieval hits, changing floating-point scores, cache miss, context
  truncation behavior, and forced-decision recovery.
- A connector failure under both full-prefill fallback and explicit-fail mode.

Go criteria:

- Response item types, call IDs/arguments, reasoning summaries, finish status,
  usage, and final visible text match the full-prefill endpoint contract.
- The client sends no CacheBlend-specific field and imports no project package.
- Server-side metrics correlate with the request artifact without embedding
  prompt/document data in labels.
- Multi-turn correctness is evaluated with logits/hidden states as well as the
  serialized response.

Stop criteria:

- Any Harmony/tool-call regression blocks benchmarking.
- Do not alter `rag-system` to compensate for a serving-plugin bug.

## M9: benchmark design

The dependency-free `cacheblend_gpt_oss.benchmark` package now defines the
controlled-trial evidence boundary. A trial records one arm, cache state,
reconciled request counters/timers, peak memory, recomputation ratio, and a
digest of its independent correctness artifact. CacheBlend arms also require a
transfer-evidence digest, while one prompt-fixture digest binds all arms in a
case. The ordinary `full_prefill` arm is additionally required to recompute
every prompt row with zero reusable-document/KV counters; it cannot silently
become a cached baseline. Artifacts pin the model and software identity,
Triton attention backend, hybrid-cache requirement, block size, context limit,
deterministic sampling settings, and TP/PP=1. The host validator computes
per-arm means, medians, and 95% confidence intervals from repeated raw trials.
Summaries retain reusable-document and found/loaded/rejected-KV counts,
document/candidate/loaded hit fractions, lookup/transfer/correction/selective/
store timings, native serving timings, recomputed/avoided rows, and absolute/
mean logit error. It reports `benchmark_ready=false` until both the ordinary
full-prefill and CacheBlend-100%-recompute control arms exist and every
recorded trial has passing correctness evidence.

Run these isolated arms from identical model/runtime/config snapshots:

1. Ordinary full prefill with vLLM prefix caching off.
2. Ordinary vLLM prefix caching with an exact-prefix repeat.
3. Ordinary vLLM prefix caching with the document moved/reordered.
4. CacheBlend at 100% recomputation as a correctness/control-flow control; no
   speedup is expected or claimed.
5. CacheBlend at each correctness-approved selective ratio with the document
   moved/reordered.
6. Prefix plus CacheBlend only after the individual mechanisms and interaction
   semantics pass independently.

Control model revision, tokenizer, prompt tokens, batch/concurrency, block
size, max model length, HMA, attention backend, random seed, sampling settings,
cache warm/cold state, and GPU clock/load conditions. Warm kernels separately
from warming cache content. Preserve raw per-request artifacts and before/after
Prometheus snapshots.

Report:

- requested/found/loaded/rejected/recomputed token counts;
- document and token hit fractions;
- nominal/effective saved-prefill fraction;
- lookup, transfer, correction, selective-compute, queue, prefill, TTFT, decode,
  and end-to-end latency;
- peak memory/staging overhead and failures;
- full correctness metrics and recomputation ratio;
- exact model, vLLM, LMCache, PyTorch/CUDA, plugin commit, config digest, host,
  and attention backend identity.

Go criteria:

- Every latency point has a passing correctness artifact and reconciled work
  counters.
- Baseline prefix hits behave as expected: high on exact prefix, absent for a
  moved non-prefix document.
- Results separate lookup/transfer overhead from actual model work saved.
- Repeated trials and confidence intervals are reported; isolated fluent output
  examples are not benchmark evidence.

## Correctness test matrix

| Case | Cache expectation | 100% expectation | Selective expectation |
|---|---|---|---|
| Exact prefix | May hit vLLM prefix cache in its dedicated arm | Same logits as full prefill | Error reported separately; no double counting |
| One moved document | Verified non-prefix candidate and load | Full recompute, zero saved fraction, baseline-equivalent logits | Main approximation/error curve |
| Two reordered documents | Two verified destination mappings | Full recompute, baseline-equivalent logits | Deterministic merge with no overlap |
| Cache miss | No accepted load | Ordinary full prefill | Ordinary full prefill |
| Mutated/collision candidate | Found candidate, then rejected | Ordinary full prefill or visible fail | Never consume stale KV |
| Full-attention layer | Full group mapping | Fresh K/V overwrites loaded values | Corrected cached rows plus selected writes |
| Sliding-128 layer | Sliding group and live-window capture | Fresh K/V overwrites loaded values | Window-correct attention and no reclaimed-source read |
| Sink behavior | Sink is not in cached record | Stock sink result unchanged | Custom backend matches sink semantics |
| Harmony tool/multi-turn | Transparent request | Serialized response contract unchanged | Same contract plus measured approximation |

Numerical thresholds are frozen before judging CacheBlend. First measure
full-prefill versus repeated full-prefill under identical deterministic BF16
settings. At 100%, CacheBlend error must remain inside that recorded envelope
for each compared tensor, with identical top-token selection. Lower ratios use a
written error budget derived from data; no universal tolerance is invented
after seeing a failing result.

## Risk register

| Risk | Evidence/signal | Mitigation and stop rule |
|---|---|---|
| Experimental V1 connector API | vLLM marks `KVConnectorBase_V1` experimental | Isolate every import in `v0_19_1`; exact-version startup rejection |
| Model artifact not revision-pinned | A model ID alone does not identify cached tensor semantics | Require immutable revision/file digest before persistent store/reuse |
| Prefix-scalar scheduler contract | Base API allows only largest prefix | Report zero external tokens through M5; use internal work metrics; patch only at M6 gate |
| HMA auto-disabled for connectors | vLLM config changes default when connector exists | Require `--no-disable-hybrid-kv-cache-manager`; implement `SupportsHMA`; reject otherwise |
| LMCache ordinary connector assumes one group | `RequestTracker` uses `block_ids[0]` | New connector owns all group mappings; do not subclass that assumption |
| LMCache matcher collision/false candidate | Direct-address rolling table lacks exact-token verification | Strong digest plus full-token comparison; reject on any ambiguity |
| LMCache blend APIs are internal | Version-pinned multiprocess protocol, not stable public surface | Hide behind injected adapter; pin exact version; fake in unit tests |
| YaRN correction subtly rescales K | GPT-OSS caches post-RoPE K with magnitude scaling | Exact delta unit rotation; direct source/target tensor tests; no generic LMCache RoPE helper |
| Cached deeper-layer KV is context dependent | Moving a document changes preceding attention/hidden states | 100% exact proof first; treat lower ratios as approximation with error curves |
| Sliding KV reclaimed before finish | vLLM removes out-of-window blocks | Capture per layer/prefill step while live; test >128-token documents/chunking |
| Learned sinks lost or serialized | Sinks are model parameters, not KV | Sink-capable backend validation; explicit absence from records; parity tests |
| Triton load/write race | Cache update occurs before decorated layer wait | Synchronous load at 100%; custom load-before-write backend for overlap |
| Dynamic sparse rows break runner/logits | Runner prepares contiguous positions and output indices | Preserve full output shape in external model spike; patch only with written reproducer |
| MXFP4/MoE numerical sensitivity | Selected rows route through quantized experts | Compare hidden/logits by layer and ratio; deterministic routing fixtures |
| Memory pressure on one A100 | Model, paged KV, and contiguous staging coexist | Start one request/one staged document; measure peak; bound buffers; no CUDA optimization yet |
| Harmony/API regression | Model override sits under Responses/tool parser | Full response-item/tool/multi-turn regression suite before benchmark |
| Local host has no target GPU | GPU work is on `solab-g3` only | CPU unit tests locally; user-run commands; never infer or claim remote pass |
| Premature RAG coupling | Client currently needs only endpoint configuration | Keep `rag-system` read-only and dependency-free; document future metadata only |

## Commands and execution topology

Local workstation source/unit check:

```bash
python -m pytest -m "not gpu"
```

The user runs GPU checks manually on `solab-g3` after synchronizing the
repository. The non-model GPU suite contains the environment contract, BF16
YaRN shift comparison, and hybrid gather/scatter round trip:

```bash
CACHEBLEND_REPO=/path/to/cacheblend-gpt-oss
cd "$CACHEBLEND_REPO"
uv sync --extra gpu --extra test
uv run pytest -m "gpu and integration and not model" -vv
```

Once model tests are added, the manual command will be:

```bash
CACHEBLEND_MODEL_PATH=/path/to/pinned/gpt-oss-20b
export CACHEBLEND_MODEL_PATH
uv run pytest -m "gpu and integration and model" -vv
```

Before interpreting either run, preserve this identity output:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
uv run python -c "import importlib.metadata as m, torch; print({'torch': torch.__version__, 'torch_cuda': torch.version.cuda, 'vllm': m.version('vllm'), 'lmcache': m.version('lmcache'), 'gpu': torch.cuda.get_device_name(0)})"
```

No GPU or model test has been run as part of this local feasibility audit. A
skip on the workstation is not a pass. Only output returned from `solab-g3` may
be recorded as a GPU result. `solab-p7` remains the orchestration host and is not
assumed reachable during implementation.

## Future RAG integration proposal (documentation only)

The validated server should remain transparent: configure
`RAG_GENERATOR_URL`/served model, preserve `/v1/models`, `/v1/responses`, and
`/metrics`, and record deployment identity. No runtime import or segment field
is proposed.

Only after explicit authorization, likely metadata/analysis touch points in the
external repository are:

- `scripts/run_oss_standard_agent.py` and `scripts/run_single_pass.py` for
  CacheBlend/plugin SHA, model revision, serving versions, connector/config
  digest, and deployment ID;
- `scripts/run_dev100.sh` to carry the same immutable experiment identity;
- `src/rag_system/workflows/oss_standard_agent.py` and
  `src/rag_system/analysis/dev100.py` only after a stable per-request metric
  schema exists;
- `src/rag_system/analysis/vllm_metrics.py` for new aggregate Prometheus
  metrics; and
- corresponding documentation/tests.

These are proposed interfaces only. This project must not modify or import
`rag-system` during feasibility work.
