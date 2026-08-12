# solab-g3 connector-loading smoke test

This is a manual M1 runbook for the GPU host. The current connector is a
control-flow skeleton: it loads out of tree, validates hybrid groups, reports
zero external tokens, transfers no KV, and performs ordinary full prefill.
Passing this runbook does **not** demonstrate CacheBlend reuse or speedup.

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

## Start the no-transfer connector

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

## Required negative check

Stop the server and remove `--no-disable-hybrid-kv-cache-manager` from the same
command. Startup must reject the unsupported configuration; it must not fall
back to a single/uniform cache group or claim CacheBlend operation.

## Current boundary after a pass

A pass proves only external loading, scheduler/worker construction, hybrid
group metadata propagation, no-transfer hook execution, request completion, and
the unchanged Responses path. M2/M3 still need planner integration, LMCache
transfer, proof that KV was loaded and overwritten at 100% recomputation, and
deterministic logit/hidden-state equivalence.
