# solab-g3 moved-document correctness gate

This is the first live CacheBlend gate: store one exact 256-token document,
move it from position 0 to position 17, load its KV through every GPT-OSS cache
group/layer, recompute all 280 target-prompt tokens, and compare the complete
201,088-token output distribution with ordinary full prefill.

No command in this runbook has been executed on the authoring workstation. A
run passes only after the user returns the `solab-g3` outputs and artifacts.

## Why this gate uses completions

The production contract remains `/v1/responses`. Pinned vLLM rejects response
logprobs when its GPT-OSS Harmony path is active, however, so the numerical
gate uses `/v1/completions` with raw token IDs. Pinned completions accept an
integer-token prompt, can expose generated token IDs, and can return token-ID
logprobs for the complete vocabulary when the server starts with
`--max-logprobs -1`.

The captured values are vLLM's normalized output logprobs, with non-finite
tails clamped to `-9999.0` by its OpenAI serializer. This preserves relative
final-logit differences across every unclamped vocabulary entry, but it is not
an unnormalized raw-logit or per-layer-hidden-state dump. Harmony, reasoning,
tool calls, and append-only multi-turn behavior are exercised separately on
`/v1/responses`; output fluency is never used as numerical evidence.

Pinned evidence:

- [Responses rejects GPT-OSS logprobs](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/entrypoints/openai/responses/serving.py#L293-L302).
- [Completions accepts integer token prompts](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/entrypoints/openai/completion/protocol.py#L42-L60).
- [Completions exposes token-ID logprob keys and sampled token IDs](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/entrypoints/openai/completion/protocol.py#L126-L142).
- [`max_logprobs=-1` expands to model vocabulary size](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/sampling_params.py#L638-L653).
- [Completion serialization clamps the non-finite tail](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/entrypoints/openai/completion/serving.py#L610-L638).

## 1. Pin the run identity

Use a clean `main` checkout. Install exactly as described in
`solab-g3-connector-smoke.md`, then record these immutable values:

```bash
cd /mnt/nvme3n1/mlee/cacheblend-gpt-oss

export CACHEBLEND_MODEL_PATH=/mnt/nvme3n1/labuser/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee
export CACHEBLEND_MODEL_REVISION=6cee5e81ee83917806bbde320786a8fb61efebee
export CACHEBLEND_TOKENIZER_REVISION=6cee5e81ee83917806bbde320786a8fb61efebee
export CACHEBLEND_PLUGIN_COMMIT="$(git rev-parse HEAD)"
export CACHEBLEND_RUN_DIR="/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m3-${CACHEBLEND_PLUGIN_COMMIT:0:7}-formal1-20260813"

test "$(git branch --show-current)" = main
test -z "$(git status --porcelain)"
mkdir -p "$CACHEBLEND_RUN_DIR"

nvidia-smi \
  --query-gpu=name,memory.total,driver_version \
  --format=csv,noheader | tee "$CACHEBLEND_RUN_DIR/gpu.txt"
.venv/bin/python -c "import importlib.metadata as m, torch; print({'torch': torch.__version__, 'torch_cuda': torch.version.cuda, 'vllm': m.version('vllm'), 'lmcache': m.version('lmcache'), 'gpu': torch.cuda.get_device_name(0)})" \
  | tee "$CACHEBLEND_RUN_DIR/runtime.txt"
```

The expected identity is vLLM `0.19.1`, LMCache `0.4.3`, PyTorch
`2.10.0+cu128`, CUDA runtime `12.8`, and `NVIDIA A100-SXM4-80GB`. Stop on any
difference.

## 2. Derive the finalized compatibility digests

Run the transfer-disabled compatibility probe with the exact flags used by
both later servers:

```bash
CACHEBLEND_PROBE_CONFIG='{"kv_connector":"GptOssCacheBlendConnector","kv_connector_module_path":"cacheblend_gpt_oss.vllm_compat.v0_19_1.connector","kv_role":"kv_both","kv_load_failure_policy":"fail","kv_connector_extra_config":{"mode":"compatibility_probe"}}'
export VLLM_USE_V2_MODEL_RUNNER=0

.venv/bin/vllm serve "$CACHEBLEND_MODEL_PATH" \
  --served-model-name openai/gpt-oss-20b \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 1024 \
  --long-prefill-token-threshold 0 \
  --no-async-scheduling \
  --enforce-eager \
  --no-enable-prefix-caching \
  --kv-cache-dtype auto \
  --attention-backend TRITON_ATTN \
  --no-disable-hybrid-kv-cache-manager \
  --generation-config vllm \
  --max-logprobs -1 \
  --kv-transfer-config "$CACHEBLEND_PROBE_CONFIG" \
  2>&1 | tee "$CACHEBLEND_RUN_DIR/compatibility-probe.log"
```

A successful probe intentionally terminates startup after printing the two
lowercase SHA-256 values below. Export the pinned values and stop if the probe
prints anything different:

```bash
export CACHEBLEND_MODEL_CONFIG_DIGEST=1c69c7868c1206ea76c372df01e5baa2abcadcd2ca5b9f93b97d94fa6070aae0
export CACHEBLEND_KV_CONFIG_DIGEST=131eb7ec025bc9a4fa1dabd220bb41b75c7d8f921e537fd8be505e91c6850742
```

Any other startup failure is not a successful probe.

## 3. Capture and freeze ordinary full-prefill behavior

Start a fresh server with no connector and otherwise identical flags. The
baseline uses `--max-num-batched-tokens 1024`, matching the CacheBlend server:

```bash
export VLLM_USE_V2_MODEL_RUNNER=0

.venv/bin/vllm serve "$CACHEBLEND_MODEL_PATH" \
  --served-model-name openai/gpt-oss-20b \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 1024 \
  --long-prefill-token-threshold 0 \
  --no-async-scheduling \
  --enforce-eager \
  --no-enable-prefix-caching \
  --kv-cache-dtype auto \
  --attention-backend TRITON_ATTN \
  --no-disable-hybrid-kv-cache-manager \
  --generation-config vllm \
  --max-logprobs -1 \
  2>&1 | tee "$CACHEBLEND_RUN_DIR/full-prefill-server.log"
```

From a second shell, capture five ordinary full-prefill controls. Every control
uses the same source-warm-up-then-target protocol: the harness first issues the
256-token source prompt, waits for the native prompt/prefill/timing metrics, and
then captures the 280-token target prompt. This warm-up is ordinary full
prefill; it does not load external KV. The `--warm-source-before-target` flag
enforces this ordering:

```bash
cd /mnt/nvme3n1/mlee/cacheblend-gpt-oss

export CACHEBLEND_MODEL_PATH=/mnt/nvme3n1/labuser/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee
export CACHEBLEND_MODEL_REVISION=6cee5e81ee83917806bbde320786a8fb61efebee
export CACHEBLEND_TOKENIZER_REVISION=6cee5e81ee83917806bbde320786a8fb61efebee
export CACHEBLEND_PLUGIN_COMMIT="$(git rev-parse HEAD)"
export CACHEBLEND_MODEL_CONFIG_DIGEST=1c69c7868c1206ea76c372df01e5baa2abcadcd2ca5b9f93b97d94fa6070aae0
export CACHEBLEND_KV_CONFIG_DIGEST=131eb7ec025bc9a4fa1dabd220bb41b75c7d8f921e537fd8be505e91c6850742
export CACHEBLEND_RUN_DIR="/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m3-${CACHEBLEND_PLUGIN_COMMIT:0:7}-formal1-20260813"

for CONTROL in 1 2 3 4 5; do
  .venv/bin/python scripts/capture_moved_document.py \
    --mode full_prefill \
    --warm-source-before-target \
    --model-revision "$CACHEBLEND_MODEL_REVISION" \
    --tokenizer-revision "$CACHEBLEND_TOKENIZER_REVISION" \
    --plugin-commit "$CACHEBLEND_PLUGIN_COMMIT" \
    --model-config-digest "$CACHEBLEND_MODEL_CONFIG_DIGEST" \
    --kv-cache-config-digest "$CACHEBLEND_KV_CONFIG_DIGEST" \
    --output "$CACHEBLEND_RUN_DIR/full-prefill-1024-control-${CONTROL}.json" \
    | tee "$CACHEBLEND_RUN_DIR/full-prefill-1024-control-${CONTROL}.txt"
done
```

Before starting CacheBlend, freeze an immutable manifest containing all five
control artifact digests, their identical runtime/prompt identity, and the
empirical full-vocabulary envelope:

```text
Umax  = max(max_abs_error(B_i, B_j))  for every i < j
Umean = max(mean_abs_error(B_i, B_j)) for every i < j
```

Freeze the manifest with the pre-registered BF16 baseline-stability ceilings
`max_abs <= 0.08` and `mean_abs <= 0.014`. These ceilings decide whether the
ordinary baseline itself is stable enough to judge a candidate; they do not
replace the tighter empirical `Umax` and `Umean` candidate limits:

```bash
.venv/bin/python scripts/freeze_correctness_ensemble.py \
  --baseline "$CACHEBLEND_RUN_DIR/full-prefill-1024-control-1.json" \
  --baseline "$CACHEBLEND_RUN_DIR/full-prefill-1024-control-2.json" \
  --baseline "$CACHEBLEND_RUN_DIR/full-prefill-1024-control-3.json" \
  --baseline "$CACHEBLEND_RUN_DIR/full-prefill-1024-control-4.json" \
  --baseline "$CACHEBLEND_RUN_DIR/full-prefill-1024-control-5.json" \
  --hard-max-abs-ceiling 0.08 \
  --hard-mean-abs-ceiling 0.014 \
  --output "$CACHEBLEND_RUN_DIR/frozen-five-baseline-manifest.json" \
  | tee "$CACHEBLEND_RUN_DIR/frozen-five-baseline-report.json"
```

Stop if the report does not show `stable: true`. Preserve its manifest digest
before stopping the ordinary server.

The five controls and this policy must be complete before the candidate is
captured. Do not use a CacheBlend output to choose, widen, or otherwise loosen
`Umax` or `Umean`. The older two-artifact `frozen-bf16-tolerance.json` remains
useful as historical evidence, but it is not the formal five-control M3 policy.

Stop the ordinary server before continuing.

## 4. Start LMCache and the 100% connector

Start the pinned LMCache server in one shell, using the wrapper command in
`solab-g3-connector-smoke.md`. It applies the 0.4.3 store-completion ordering
backport and the live-gated exact-token candidate index; do not launch
`lmcache.v1.multiprocess.blend_server_v2` directly.
In another shell, create a new sidecar path. Do not reuse a database from
another model/config/plugin identity:

```bash
CACHEBLEND_SIDECAR="$CACHEBLEND_RUN_DIR/sidecar.sqlite3"
CACHEBLEND_TRANSFER_EVIDENCE="$CACHEBLEND_RUN_DIR/transfer-evidence.json"
export CACHEBLEND_SIDECAR CACHEBLEND_TRANSFER_EVIDENCE
test ! -e "$CACHEBLEND_SIDECAR"
test ! -e "$CACHEBLEND_TRANSFER_EVIDENCE"
.venv/bin/python -c "import os; from cacheblend_gpt_oss.storage.sidecar import SidecarMode, open_sidecar_index; index = open_sidecar_index(os.environ['CACHEBLEND_SIDECAR'], SidecarMode.WORKER_READ_WRITE); index.close()"

CACHEBLEND_KV_CONFIG=$(
  .venv/bin/python scripts/render_transfer_config.py \
    --lmcache-server-url tcp://127.0.0.1:5556 \
    --sidecar-path "$CACHEBLEND_SIDECAR" \
    --model-revision "$CACHEBLEND_MODEL_REVISION" \
    --tokenizer-revision "$CACHEBLEND_TOKENIZER_REVISION" \
    --model-config-digest "$CACHEBLEND_MODEL_CONFIG_DIGEST" \
    --kv-cache-config-digest "$CACHEBLEND_KV_CONFIG_DIGEST" \
    --adapter-revision "$CACHEBLEND_PLUGIN_COMMIT" \
    --transfer-evidence-path "$CACHEBLEND_TRANSFER_EVIDENCE" \
    --staging-token-capacity 1024 \
    --request-timeout-seconds 120
)
export CACHEBLEND_KV_CONFIG
```

The transfer run below uses a 1024-token one-step/staging envelope because the
required reordered-documents fixture is 536 tokens; a 512-token envelope would
correctly fall back before lookup for that case.

Start vLLM using precisely the baseline flags plus the rendered connector:

```bash
export VLLM_USE_V2_MODEL_RUNNER=0

.venv/bin/vllm serve "$CACHEBLEND_MODEL_PATH" \
  --served-model-name openai/gpt-oss-20b \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 1024 \
  --long-prefill-token-threshold 0 \
  --no-async-scheduling \
  --enforce-eager \
  --no-enable-prefix-caching \
  --kv-cache-dtype auto \
  --attention-backend TRITON_ATTN \
  --no-disable-hybrid-kv-cache-manager \
  --generation-config vllm \
  --max-logprobs -1 \
  --kv-transfer-config "$CACHEBLEND_KV_CONFIG" \
  2>&1 | tee "$CACHEBLEND_RUN_DIR/cacheblend-server.log"
```

## 4a. Optional M3 diagnostic: scatter-disabled control

Run this control **before** step 5's real transfer comparison, as the first
M3 numerical-equivalence diagnostic. It isolates connector-presence drift
(staging allocation, an attached LMCache client, YaRN correction arithmetic,
extra CUDA allocator traffic) from genuine loaded-KV contamination of the
paged cache.

`disable_kv_scatter` is an explicit, opt-in diagnostic switch on
`transfer_100pct`
(`src/cacheblend_gpt_oss/vllm_compat/v0_19_1/transfer_config.py`). With it
set, the connector still runs lookup, retrieval into the staging tensor, and
GPT-OSS YaRN key correction exactly as in a real transfer
(`GptOssDataPlane.scatter_retrieved_kv` in
`src/cacheblend_gpt_oss/vllm_compat/v0_19_1/data_plane.py`), but the final
copy of corrected K/V into vLLM's real paged KV cache is skipped
(`GptOssWorkerBridge.scatter_retrieved` in
`src/cacheblend_gpt_oss/vllm_compat/v0_19_1/worker_bridge.py`). Every prompt
token is still recomputed by ordinary prefill, exactly as in the normal
100%-recompute milestone.

Render a second connector config identical to step 4's, with a separate
sidecar/evidence path and `--disable-kv-scatter` added:

```bash
CACHEBLEND_DIAG_SIDECAR="$CACHEBLEND_RUN_DIR/sidecar-scatter-disabled.sqlite3"
CACHEBLEND_DIAG_EVIDENCE="$CACHEBLEND_RUN_DIR/transfer-evidence-scatter-disabled.json"
export CACHEBLEND_DIAG_SIDECAR CACHEBLEND_DIAG_EVIDENCE
test ! -e "$CACHEBLEND_DIAG_SIDECAR"
test ! -e "$CACHEBLEND_DIAG_EVIDENCE"
.venv/bin/python -c "import os; from cacheblend_gpt_oss.storage.sidecar import SidecarMode, open_sidecar_index; index = open_sidecar_index(os.environ['CACHEBLEND_DIAG_SIDECAR'], SidecarMode.WORKER_READ_WRITE); index.close()"

CACHEBLEND_DIAG_KV_CONFIG=$(
  .venv/bin/python scripts/render_transfer_config.py \
    --lmcache-server-url tcp://127.0.0.1:5556 \
    --sidecar-path "$CACHEBLEND_DIAG_SIDECAR" \
    --model-revision "$CACHEBLEND_MODEL_REVISION" \
    --tokenizer-revision "$CACHEBLEND_TOKENIZER_REVISION" \
    --model-config-digest "$CACHEBLEND_MODEL_CONFIG_DIGEST" \
    --kv-cache-config-digest "$CACHEBLEND_KV_CONFIG_DIGEST" \
    --adapter-revision "$CACHEBLEND_PLUGIN_COMMIT" \
    --transfer-evidence-path "$CACHEBLEND_DIAG_EVIDENCE" \
    --staging-token-capacity 1024 \
    --request-timeout-seconds 120 \
    --disable-kv-scatter
)
export CACHEBLEND_DIAG_KV_CONFIG
```

Start a second vLLM instance with the exact same baseline flags as step 4,
substituting `--kv-transfer-config "$CACHEBLEND_DIAG_KV_CONFIG"`, and repeat
the store/move/load/recompute/compare sequence from step 5 against this
server instead.

For the prospective connector-attached probability envelope, capture each
scatter-disabled control as a full-prefill artifact. The dedicated flag keeps
the connector metrics in the capture contract while correctly leaving
`connector` empty in the artifact; this prevents a diagnostic run from being
mistaken for a real 100%-transfer candidate:

```bash
for CONTROL in 1 2 3 4 5; do
  .venv/bin/python scripts/capture_moved_document.py \
    --mode full_prefill \
    --warm-source-before-target \
    --connector-attached-control \
    --model-revision "$CACHEBLEND_MODEL_REVISION" \
    --tokenizer-revision "$CACHEBLEND_TOKENIZER_REVISION" \
    --plugin-commit "$CACHEBLEND_PLUGIN_COMMIT" \
    --model-config-digest "$CACHEBLEND_MODEL_CONFIG_DIGEST" \
    --kv-cache-config-digest "$CACHEBLEND_KV_CONFIG_DIGEST" \
    --output "$CACHEBLEND_RUN_DIR/scatter-disabled-control-${CONTROL}.json" \
    | tee "$CACHEBLEND_RUN_DIR/scatter-disabled-control-${CONTROL}.txt"
done
```

How to read the result:

- The connector artifact for this run must report `kv_tokens_loaded == 0`
  and a fallback with
  `failure_code == "scatter_suppressed_diagnostic"`
  (`TransferFallbackCode.SCATTER_SUPPRESSED_DIAGNOSTIC` in
  `src/cacheblend_gpt_oss/vllm_compat/v0_19_1/transfer_runtime.py`). This is
  by construction, not something to troubleshoot: a scatter-disabled run can
  never satisfy the same-transfer-evidence gate as a real transfer, and the
  existing `kv_tokens_loaded <= 0` fail-closed check in
  `scripts/validate_browsecomp_append_only.py` /
  `benchmark/browsecomp.py` already rejects it as "not a real 100%-transfer"
  for that reason.
- The suppressed-but-would-have-loaded token count is reported separately
  and honestly, never as a fabricated zero: connector Prometheus stats carry
  it in the bounded `vllm:cacheblend_kv_tokens_scatter_suppressed_total`
  counter, and the worker bridge exposes the same count locally through
  `GptOssWorkerBridge.kv_tokens_scatter_suppressed`.
- **If the moved-document numerical discrepancy from step 5 persists in this
  scatter-disabled run**, the drift is caused by connector presence itself
  (staging tensor allocation, the attached LMCache client, YaRN correction
  arithmetic, or CUDA allocator/caching-context changes from those
  allocations) and not by loaded KV reaching attention. Investigate the
  connector/allocator path before re-examining `scatter_retrieved_kv`.
- **If the discrepancy vanishes in this scatter-disabled run**, the
  connector's presence alone is numerically inert, and the discrepancy in
  step 5 is caused specifically by loaded KV contaminating the paged cache
  (i.e. the scatter/correction data path itself, not incidental connector
  overhead). Focus the next diagnostic there.

No GPU results for this control have been captured or claimed by this
document; only the mechanism and how to interpret an eventual run are
described here.

## 5. Store, move, load, fully recompute, and compare

The capture command first sends the 256-token source document and waits for its
complete store counter before issuing the target. It then sends a 280-token
target where the exact document starts at position 17. It refuses to write a
CacheBlend artifact unless the target interval contains exactly one connector
request, the native vLLM prompt-token counter advances by exactly the expected
source and target lengths, the native prefill-KV histogram reports exactly the
expected source/target rows, each target timing histogram records one
observation, all 256 reusable tokens were loaded, all 280 prompt tokens were
recomputed, and saved prefill remained zero. This store-counter wait prevents request
completion from racing the source sidecar/LMCache publication; the capture also
reconciles each request's eligible/completed chunk count and requires zero store
fallbacks. Each source/target interval must additionally report all prompt
tokens as `local_compute`, with zero `local_cache_hit` and
`external_kv_transfer`; this proves the connector's loaded KV receives no
scheduler credit during the 100% recomputation milestone. Native prompt/source/
timing names are taken from the pinned
[`PrometheusStatLogger`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/metrics/loggers.py#L580-L903),
and TTFT is never estimated from client wall time. The wait also requires the
native prompt, prompt-source, and histogram observation milestones before taking
each interval snapshot, so asynchronous exporter lag cannot produce a false
mismatch.

The capture command's JSON summary retains the target native prompt-token,
prompt-source, prefill-work, and timing deltas, plus a reconciled
`native_request_evidence` object, alongside the artifact path;
preserve that summary from the `tee` output with the server metrics and logs.

The candidate below must be fresh and captured only after the five-control
manifest was frozen. The ensemble evaluator re-reads all five controls,
recomputes their envelope, requires an exact manifest match, binds the transfer
sidecar, and compares the candidate with every control.

```bash
.venv/bin/python scripts/capture_moved_document.py \
  --mode cacheblend_100pct \
  --model-revision "$CACHEBLEND_MODEL_REVISION" \
  --tokenizer-revision "$CACHEBLEND_TOKENIZER_REVISION" \
  --plugin-commit "$CACHEBLEND_PLUGIN_COMMIT" \
  --model-config-digest "$CACHEBLEND_MODEL_CONFIG_DIGEST" \
  --kv-cache-config-digest "$CACHEBLEND_KV_CONFIG_DIGEST" \
  --output "$CACHEBLEND_RUN_DIR/cacheblend-100pct.json" \
  | tee "$CACHEBLEND_RUN_DIR/cacheblend-capture.txt"

.venv/bin/python scripts/evaluate_cacheblend_ensemble.py \
  --manifest "$CACHEBLEND_RUN_DIR/frozen-five-baseline-manifest.json" \
  --baseline "$CACHEBLEND_RUN_DIR/full-prefill-1024-control-1.json" \
  --baseline "$CACHEBLEND_RUN_DIR/full-prefill-1024-control-2.json" \
  --baseline "$CACHEBLEND_RUN_DIR/full-prefill-1024-control-3.json" \
  --baseline "$CACHEBLEND_RUN_DIR/full-prefill-1024-control-4.json" \
  --baseline "$CACHEBLEND_RUN_DIR/full-prefill-1024-control-5.json" \
  --cacheblend "$CACHEBLEND_RUN_DIR/cacheblend-100pct.json" \
  --transfer-evidence "$CACHEBLEND_RUN_DIR/transfer-evidence.json" \
  --output "$CACHEBLEND_RUN_DIR/cacheblend-ensemble-verdict.json" \
  | tee "$CACHEBLEND_RUN_DIR/cacheblend-ensemble-verdict.txt"

curl --fail-with-body http://127.0.0.1:8000/metrics \
  | grep 'vllm:cacheblend_' \
  > "$CACHEBLEND_RUN_DIR/cacheblend-metrics.txt"
```

The explicit `--transfer-evidence-path` enables the worker-side probe for this
single create-only capture. Validate its required per-layer digest sidecar
independently before accepting the final-distribution result:

```bash
.venv/bin/python scripts/validate_transfer_evidence.py \
  --input "$CACHEBLEND_RUN_DIR/transfer-evidence.json" \
  --correctness-artifact "$CACHEBLEND_RUN_DIR/cacheblend-100pct.json" \
  --output "$CACHEBLEND_RUN_DIR/transfer-evidence-report.json"
```

When `--correctness-artifact` is supplied, validation additionally requires
the sidecar source/target prompt digests, target length, loaded-token count,
recomputed-token count, and zero-savings counter to match that exact
CacheBlend artifact. A sidecar that is valid in isolation but belongs to a
different request fails this binding check.

The report must show 12 sliding and 12 full layers, all layers loaded and
overwritten, and zero prefill tokens avoided. This command is read-only with
respect to KV; it cannot create evidence when the worker probe is absent.
Schema v2 records successful load-copy and ordinary-attention save-hook
observations explicitly. Digest equality proves source/load and
fresh-prefill/final content; byte inequality is not used as proof of a write,
because a real layer-0 write can legitimately reproduce identical K/V bytes.
This first probe intentionally supports the M3 fixture in which the cached
256-token document is the complete source prompt. It is not yet a generic
BrowseComp+ per-request evidence producer.

The strict-v1 evaluator exits nonzero unless the baseline is stable, every
digest and identity matches, sampled/top tokens agree, `Qmax <= Umax`,
`Qmean <= Umean`, and the transfer sidecar binds to the candidate. Preserve the
manifest, report, verdict, all six artifacts, metrics, and service logs. A
strict-v1 result is immutable: a later probability-aware policy must not edit,
replace, or reinterpret that verdict.

## Observed solab-g3 evidence and formal status

The user-supplied run produced two distinct results. The historical
two-baseline evaluator returned `passed: false`: CacheBlend versus the selected
reference had `max_abs_error=0.07559013366699219` and
`mean_abs_error=0.009438995041906416`, exceeding the then-frozen limits of
`0.07054328918457031` and `0.006880644322352224`. Sampled and top-token
agreement were both true, and transfer evidence was bound and complete; the
failure was numerical under that two-baseline policy.

A later same-configuration comparison used `--max-num-batched-tokens 1024`
and the source-warm-up-then-target protocol for five ordinary controls. Its
empirical envelope was:

```text
Umax  = 0.07423019409179688
Umean = 0.01318507041618522
```

The already-observed CacheBlend candidate had the following diagnostic
distances to those controls:

```text
Qmax  = 0.05914115905761719
Qmean = 0.01111537456170986
```

Sampled-token and top-token agreement were true for every comparison. The
independent schema-v2 transfer report also passed: 256 tokens loaded, 280
tokens recomputed, zero prefill tokens avoided, 12 sliding-window layers and
12 full-attention layers covered, all layers loaded and overwritten, and the
artifact binding passed.

The five-control policy was defined after this candidate had already been
observed. Therefore the candidate is **diagnostic-only**: its `Q` values must
not be promoted to a formal M3 pass. A formal M3 candidate must be captured
after the five-control manifest and envelope are frozen, with no post-hoc
loosening. Formal M3 remains unpassed.

### Formal strict-v1 result and prospective probability-aware v2

The fresh candidate captured after the five-control manifest was frozen received
an immutable **strict-v1 `FAIL`**. Exactly one vocabulary coordinate violated
the strict full-vocabulary maximum envelope: token ID `71784` had
`Qmax=0.0984172821` versus `Umax=0.0742301941`. Its candidate rank was
`199583`, its probability was `5.0698e-8`, and it was outside the baseline
range by `0.0326042`. The full-vocabulary mean remained within its envelope
(`Qmean=0.012911056652712563` versus `Umean=0.01318507041618522`); sampled
token and top-token agreement remained true for every comparison; and the
independent transfer evidence passed, including all-layer load/overwrite
binding. These facts are the permanent v1 record. They do not constitute a
v1 pass or an M3 pass.

The probability-aware v2 response is **prospective only**. It reuses the same
five ordinary control artifacts but freezes a separate probability-v2 manifest
from them before capturing exactly one new CacheBlend candidate. The strict-v1
manifest remains immutable historical evidence and is not a v2 policy input.
The v2 manifest binds the strict-v1 failed candidate digest as an explicit
excluded pilot, and the evaluator rejects reuse of that artifact as the v2
candidate.

The controls are not rerun or replaced. Their serving identity is commit
`75bfe75db794a77d305c495be5f8114e520d119f`, so the new candidate must use
that exact serving checkout; the later gate-tooling commit runs from a detached
worktree. Any identity mismatch stops the run. V2 artifacts belong in a new
directory, not over the v1 files:

```bash
cd /mnt/nvme3n1/mlee/cacheblend-gpt-oss

export CACHEBLEND_SERVING_COMMIT=75bfe75db794a77d305c495be5f8114e520d119f
export CACHEBLEND_V2_RUN_DIR=/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m3-75bfe75-formal-v2-20260813

test "$(git rev-parse HEAD)" = "$CACHEBLEND_SERVING_COMMIT"
test ! -e "$CACHEBLEND_V2_RUN_DIR/probability-v2-manifest.json"
test ! -e "$CACHEBLEND_V2_RUN_DIR/cacheblend-100pct-v2.json"
test ! -e "$CACHEBLEND_V2_RUN_DIR/probability-aware-v2-verdict.json"
```

The v2 evaluator uses a minimal descending token support covering at least
`1-epsilon` probability mass with fixed `epsilon=1e-4`, so at most 0.01% of
probability mass is outside the max-logprob check. Epsilon is code-owned and
cannot be tuned from the CLI or after seeing the candidate. It must compute the
full-vocabulary mean error, total variation (TV), Jensen--Shannon (JS)
divergence, and the high-mass maximum error. The candidate must be within the
empirical five-control baseline envelope for **every** one of those metrics,
and must also satisfy these pre-registered hard ceilings:

```text
full-vocabulary mean error <= 0.014
TV                         <= 0.02
JS                         <= 0.001
high-mass maximum error    <= 0.08
```

The v2 response is admissible only if all four metric-envelope checks pass,
sampled/top-token agreement remains true, and the independent transfer report
passes. The v1 `FAIL` remains unchanged even if a future v2 candidate passes;
v2 has not been run or passed, and formal M3 remains blocked until that
prospective evidence exists.

## Required case matrix

The capture harness also has deterministic fixtures for the other required
correctness cases. Each case needs its own five source-warmed baseline controls,
frozen manifest/envelope, and fresh CacheBlend candidate because the target
prompt digest changes. Do not reuse one case's policy for another case.
Keep the same server identity and run directory, but use distinct filenames:

```bash
for CASE in exact_prefix moved_document reordered_documents cache_miss; do
  for CONTROL in 1 2 3 4 5; do
    .venv/bin/python scripts/capture_moved_document.py \
      --mode full_prefill \
      --case "$CASE" \
      --warm-source-before-target \
      --model-revision "$CACHEBLEND_MODEL_REVISION" \
      --tokenizer-revision "$CACHEBLEND_TOKENIZER_REVISION" \
      --plugin-commit "$CACHEBLEND_PLUGIN_COMMIT" \
      --model-config-digest "$CACHEBLEND_MODEL_CONFIG_DIGEST" \
      --kv-cache-config-digest "$CACHEBLEND_KV_CONFIG_DIGEST" \
      --output "$CACHEBLEND_RUN_DIR/${CASE}-full-${CONTROL}.json"
  done
done
```

With the CacheBlend server and a clean sidecar state appropriate to the case,
repeat the same loop with `--mode cacheblend_100pct` and output names
`${CASE}-cacheblend.json`; then evaluate the fresh candidate against that
case's already-frozen five-control envelope. Supply `--transfer-evidence` for
every case with positive loaded KV.
For the explicit cache miss, omit that file only with
`--allow-cache-miss-no-transfer`; the evaluator then requires zero found,
loaded, and rejected KV counters before comparing the ordinary full-prefill
fallback. The expected transfer evidence is:

| Case | Reusable segments | Expected loaded tokens | Position relation |
|---|---:|---:|---|
| `exact_prefix` | 1 | 256 | source and target start at 0 |
| `moved_document` | 1 | 256 | source 0, target 17 |
| `reordered_documents` | 2 | 512 | two 256-token documents swap order |
| `cache_miss` | 0 | 0 | target contains a new 256-token document |

Every case must still report full target-prompt recomputation and zero saved
prefill. A cache miss is a successful ordinary-prefill fallback only when its
artifact explicitly reports zero found/loaded/rejected tokens and the final
distribution passes the corresponding full-prefill comparison. Do not use one
case's tolerance or sidecar evidence to judge another case.

## Formal M3 status and stop/go decision

The current user-supplied artifact is not a formal M3 pass. Its all-layer
transfer evidence passes, but the historical two-baseline numerical verdict
failed and the later five-control comparison was defined after the candidate
was observed. Do not advance to the `/v1/responses` or BrowseComp gates on the
diagnostic candidate alone.

Go to the `/v1/responses` Harmony/tool/multi-turn gate only if:

- five same-configuration 1024-token controls were captured with
  source-warm-up-then-target ordering and their immutable envelope was frozen
  before the candidate;
- the fresh candidate was captured after that freeze, with no post-hoc
  tolerance loosening;
- the candidate satisfies `Qmax <= Umax` and `Qmean <= Umean`, with sampled and
  top-token agreement;
- the artifact reports `kv_tokens_loaded == 256`;
- `tokens_recomputed == 280` and `prefill_tokens_avoided == 0`;
- independent schema-v2 transfer evidence reports all 24 layers loaded and
  overwritten and passes artifact binding;
- all identities/digests match; and
- no server log contains a fallback, rejected configuration, transfer error,
  correction error, or partial group/layer operation.

Until those conditions are met, formal M3 remains blocked. Stop on any failed
invariant, timeout, fallback, distribution mismatch, or identity drift. Do not
lower recomputation or interpret generated text as evidence. This gate
demonstrates the final output distribution and the connector's all-layer/group
success accounting. A supplied transfer sidecar must validate independently;
raw per-layer evidence cannot be inferred from the output artifact or
connector counters.
