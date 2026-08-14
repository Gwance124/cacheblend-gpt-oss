# solab-g3 connector-loading smoke test

This is the manual GPU-host runbook for connector loading and real CUDA
primitive readiness. The connector also implements `transfer_100pct`, but the
first section deliberately uses `control_flow`: it validates external loading,
hybrid groups, and the unchanged Responses path without starting LMCache.
Passing only this section does **not** demonstrate CacheBlend reuse or speedup.

No command in this runbook was executed during local development.

## Preconditions

- Run on `solab-g3`, not the authoring workstation or `solab-p7`.
- Synchronize a clean `main` checkout of this repository to the host.
- Use the pinned local GPT-OSS-20B artifact. Record an immutable model revision
  or file-manifest digest before any later persistent cache test.
- The environment must contain Python 3.10-3.13, vLLM 0.19.1, LMCache 0.4.3,
  PyTorch 2.10.0+cu128, and CUDA runtime 12.8.

## Install and verify identity

```bash
CACHEBLEND_REPO=/path/to/cacheblend-gpt-oss
cd "$CACHEBLEND_REPO"
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade "pip==25.2"
.venv/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -e ".[gpu,test]"

nvidia-smi \
  --query-gpu=name,memory.total,driver_version \
  --format=csv,noheader
.venv/bin/python -c "import importlib.metadata as m, torch; print({'torch': torch.__version__, 'torch_cuda': torch.version.cuda, 'vllm': m.version('vllm'), 'lmcache': m.version('lmcache'), 'gpu': torch.cuda.get_device_name(0)})"
```

Run the environment and connector contract tests:

```bash
.venv/bin/python -m pytest \
  -m "gpu and integration and not model" \
  -vv
.venv/bin/python -c "from cacheblend_gpt_oss.vllm_compat.v0_19_1.connector import GptOssCacheBlendConnector; print(GptOssCacheBlendConnector.__module__)"
```

## Start the control-flow connector

Use separate variables rather than copying model paths into the repository:

```bash
CACHEBLEND_MODEL_PATH=/path/to/pinned/gpt-oss-20b
CACHEBLEND_SERVED_MODEL=openai/gpt-oss-20b
CACHEBLEND_KV_CONFIG='{"kv_connector":"GptOssCacheBlendConnector","kv_connector_module_path":"cacheblend_gpt_oss.vllm_compat.v0_19_1.connector","kv_role":"kv_both","kv_load_failure_policy":"fail"}'

export VLLM_USE_V2_MODEL_RUNNER=0

.venv/bin/vllm serve "$CACHEBLEND_MODEL_PATH" \
  --served-model-name "$CACHEBLEND_SERVED_MODEL" \
  --tensor-parallel-size 1 \
  --dtype auto \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 1 \
  --attention-backend TRITON_ATTN \
  --enable-auto-tool-choice \
  --tool-call-parser openai \
  --generation-config vllm \
  --no-disable-hybrid-kv-cache-manager \
  --kv-transfer-config "$CACHEBLEND_KV_CONFIG"
```

Expected startup evidence:

- vLLM logs creation of `GptOssCacheBlendConnector` in scheduler and worker
  roles.
- The connector receives a finalized multi-group `KVCacheConfig`.
- The model runner registers every GPT-OSS attention-layer KV tensor.
- Startup does not request `--disable-hybrid-kv-cache-manager`.
- No log claims a cache lookup, KV load, or saved prefill.

## Prepare the live-transfer services

Do this only after the control-flow smoke passes. The live mode still
recomputes the complete prompt and must not be described as acceleration.

Start the pinned LMCache Blend V2 server in a separate shell on `solab-g3`.
Use the repository wrapper, not the raw 0.4.3 module: the pinned server has a
documented store-completion race where its CUDA event can become visible before
the storage-index commit. The public direct-address matcher also missed the
first live offset-17 fixture after a successful store. The wrapper synchronizes
the server copy stream before publishing the write, records the client event
last, and substitutes a bounded exact-token candidate index. LMCache 0.4.3
still owns object storage, prefetch, retrieval, and server lifecycle.

```bash
export CUDA_VISIBLE_DEVICES=0
.venv/bin/python -m cacheblend_gpt_oss.storage.lmcache_server_v0_4_3 \
  --host 127.0.0.1 \
  --port 5556 \
  --chunk-size 256 \
  --hash-algorithm blake3 \
  --l1-size-gb 4 \
  --l1-init-size-gb 4 \
  --eviction-policy LRU \
  --max-workers 1
```

Precreate the SQLite sidecar before vLLM constructs its scheduler-role
read-only handle:

```bash
CACHEBLEND_SIDECAR=/absolute/path/to/cacheblend-sidecar.sqlite3
export CACHEBLEND_SIDECAR
.venv/bin/python -c "import os; from cacheblend_gpt_oss.storage.sidecar import SidecarMode, open_sidecar_index; index = open_sidecar_index(os.environ['CACHEBLEND_SIDECAR'], SidecarMode.WORKER_READ_WRITE); index.close()"
```

The live connector also requires the exact model and KV compatibility digests
derived from the finalized `VllmConfig` and `KVCacheConfig`. Do not substitute
arbitrary 64-hex values: startup intentionally rejects them, and a mislabeled
persistent namespace would make KV reuse unsafe.

Run the same intended transfer launch once with the explicit probe mode. It
always stops startup after printing the two finalized digests and never opens
LMCache or reuses KV:

```bash
CACHEBLEND_PROBE_CONFIG='{"kv_connector":"GptOssCacheBlendConnector","kv_connector_module_path":"cacheblend_gpt_oss.vllm_compat.v0_19_1.connector","kv_role":"kv_both","kv_load_failure_policy":"fail","kv_connector_extra_config":{"mode":"compatibility_probe"}}'

export VLLM_USE_V2_MODEL_RUNNER=0
.venv/bin/vllm serve "$CACHEBLEND_MODEL_PATH" \
  --served-model-name "$CACHEBLEND_SERVED_MODEL" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 4096 \
  --long-prefill-token-threshold 0 \
  --no-async-scheduling \
  --enforce-eager \
  --no-enable-prefix-caching \
  --kv-cache-dtype auto \
  --attention-backend TRITON_ATTN \
  --no-disable-hybrid-kv-cache-manager \
  --kv-transfer-config "$CACHEBLEND_PROBE_CONFIG"
```

Expected terminal evidence is one bounded message containing
`model_config_digest=<64 lowercase hex>` and
`kv_cache_config_digest=<64 lowercase hex>`, plus the statement that transfer
remained disabled. Record both values with the exact model/tokenizer revisions
and use them unchanged in the later `transfer_100pct` JSON. Any other startup
failure is not a successful probe.

## Exercise `/v1/responses`

From a second shell on `solab-g3`:

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/models

curl --fail-with-body http://127.0.0.1:8000/v1/responses \
  -H 'Authorization: Bearer EMPTY' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "openai/gpt-oss-20b",
    "input": [{"role": "user", "content": "Return exactly: connector loaded"}],
    "max_output_tokens": 64,
    "reasoning": {"effort": "low", "summary": "auto"}
  }'
```

Save the full server log, identity output, response JSON, and `/metrics`
snapshot. Return those artifacts before marking M1 as passed.

After at least one response, retain a filtered connector-metric view as well:

```bash
curl --fail-with-body http://127.0.0.1:8000/metrics \
  | grep 'vllm:cacheblend_'
```

For this 100%-recompute milestone, the scrape must show recomputed prompt
tokens and `vllm:cacheblend_effective_saved_prefill_fraction` equal to zero.
A miss may legitimately show zero requested/found/loaded rows. The metric must
not contain a request ID, prompt/token content, document identity, fingerprint,
path, or server URL as a label. TTFT and prefill latency remain vLLM-native
server metrics; do not estimate TTFT from the non-streaming client response.

## Required negative check

Stop the server and remove `--no-disable-hybrid-kv-cache-manager` from the same
command. Startup must reject the unsupported configuration; it must not fall
back to a single/uniform cache group or claim CacheBlend operation.

## Current boundary after a pass

A control-flow/model pass plus the non-model CUDA suite proves external loading,
scheduler/worker construction, hybrid metadata propagation, the unchanged
Responses path, production YaRN arithmetic, and real Torch scatter-gather. The
planner and `transfer_100pct` path are implemented, but M3 is not passed until a
moved-document run independently proves a nonzero load, full overwrite, and
deterministic logit/hidden-state equivalence.
