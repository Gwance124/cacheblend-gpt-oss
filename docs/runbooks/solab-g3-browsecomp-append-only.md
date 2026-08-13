# solab-g3 BrowseComp-Plus append-only transfer smoke

This gate runs one real `rag-system` BrowseComp-Plus agent trajectory through
the pinned GPT-OSS CacheBlend connector without modifying or importing the
workload repository. It proves that the transparent `/v1/responses` path can
find and load previously stored KV during an append-only trajectory.

This is a **100%-recomputation transfer smoke**, not a speedup experiment.
Every prompt token is still scheduled and recomputed, and
`prefill_tokens_avoided` must remain zero. A positive loaded-token count proves
that transfer occurred; it does not prove that model work was saved or that an
embedded document (rather than an unchanged prompt-aligned chunk) was reused.

No command in this runbook has been executed on the authoring workstation.
Only user-supplied `solab-g3`/`solab-p7` output can pass the gate.

## Preconditions and stop rules

Complete and preserve the passing artifacts from:

1. `solab-g3-moved-document-correctness.md` (M3/M4/M5 numerical and transfer
   evidence); and
2. `solab-g3-responses-contract.md` (M8 Harmony, tools, and append-only API
   evidence).

Use only vLLM `0.19.1`, LMCache `0.4.3`, PyTorch `2.10.0+cu128`, CUDA runtime
`12.8`, and the NVIDIA A100-SXM4-80GB. Keep the hybrid KV-cache manager enabled
and use `TRITON_ATTN`. Stop on any identity mismatch, fallback, store failure,
missing cache group, or prefix-cache credit.

The connector currently requires one complete prefill step. A request is
ineligible when its prompt exceeds either `staging_token_capacity` or
`max_num_batched_tokens`. The 512/1024-token settings used by the synthetic
gates therefore cannot exercise a real BrowseComp agent prompt. For this smoke,
both values are 131,072 and `max_num_seqs` is one. The registered BF16 staging
tensor is approximately 6 GiB:

```text
2 (K/V) * 24 layers * 131072 tokens * 512 values * 2 bytes
```

Use a fresh process with `--gpu-memory-utilization 0.80` and record peak memory.
If startup or the first request cannot fit, stop and return the OOM/startup
evidence. Do not lower the staging envelope and then call an ineligible request
a CacheBlend run.

## 1. Prepare a dedicated g3 deployment

Use a clean checkout and a new artifact directory:

```bash
cd /path/to/cacheblend-gpt-oss

CACHEBLEND_MODEL_PATH=/path/to/pinned/gpt-oss-20b
CACHEBLEND_MODEL_REVISION=replace-with-model-commit-or-manifest-sha
CACHEBLEND_TOKENIZER_REVISION=replace-with-tokenizer-commit-or-manifest-sha
CACHEBLEND_PLUGIN_COMMIT=$(git rev-parse HEAD)
CACHEBLEND_RUN_DIR=/absolute/path/to/new/browsecomp-append-only-smoke
CACHEBLEND_STAGING_TOKENS=131072

export CACHEBLEND_MODEL_PATH CACHEBLEND_MODEL_REVISION
export CACHEBLEND_TOKENIZER_REVISION CACHEBLEND_PLUGIN_COMMIT
export CACHEBLEND_RUN_DIR CACHEBLEND_STAGING_TOKENS

test "$(git branch --show-current)" = main
test -z "$(git status --porcelain)"
mkdir -p "$CACHEBLEND_RUN_DIR"

nvidia-smi \
  --query-gpu=name,memory.total,memory.free,driver_version \
  --format=csv,noheader | tee "$CACHEBLEND_RUN_DIR/gpu-before.txt"
.venv/bin/python -c "import importlib.metadata as m, torch; print({'torch': torch.__version__, 'torch_cuda': torch.version.cuda, 'vllm': m.version('vllm'), 'lmcache': m.version('lmcache'), 'gpu': torch.cuda.get_device_name(0)})" \
  | tee "$CACHEBLEND_RUN_DIR/runtime.txt"
```

Run compatibility-probe mode with the exact serving flags below. A successful
probe exits after printing the finalized model and KV-cache digests:

```bash
CACHEBLEND_PROBE_CONFIG='{"kv_connector":"GptOssCacheBlendConnector","kv_connector_module_path":"cacheblend_gpt_oss.vllm_compat.v0_19_1.connector","kv_role":"kv_both","kv_load_failure_policy":"fail","kv_connector_extra_config":{"mode":"compatibility_probe"}}'
export VLLM_USE_V2_MODEL_RUNNER=0

.venv/bin/vllm serve "$CACHEBLEND_MODEL_PATH" \
  --served-model-name openai/gpt-oss-20b \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.80 \
  --max-num-seqs 1 \
  --max-num-batched-tokens "$CACHEBLEND_STAGING_TOKENS" \
  --long-prefill-token-threshold 0 \
  --no-async-scheduling \
  --enforce-eager \
  --no-enable-prefix-caching \
  --kv-cache-dtype auto \
  --attention-backend TRITON_ATTN \
  --no-disable-hybrid-kv-cache-manager \
  --enable-auto-tool-choice \
  --tool-call-parser openai \
  --generation-config vllm \
  --kv-transfer-config "$CACHEBLEND_PROBE_CONFIG" \
  2>&1 | tee "$CACHEBLEND_RUN_DIR/compatibility-probe.log"
```

Copy the two emitted lowercase SHA-256 values exactly:

```bash
CACHEBLEND_MODEL_CONFIG_DIGEST=replace-with-probe-model-config-digest
CACHEBLEND_KV_CONFIG_DIGEST=replace-with-probe-kv-cache-config-digest
export CACHEBLEND_MODEL_CONFIG_DIGEST CACHEBLEND_KV_CONFIG_DIGEST
```

## 2. Start LMCache and vLLM on g3

In one shell, launch the repository's version-scoped LMCache server wrapper.
Do not launch LMCache's raw 0.4.3 module because it contains the audited store
completion race:

```bash
cd /path/to/cacheblend-gpt-oss
export CUDA_VISIBLE_DEVICES=0

.venv/bin/python -m cacheblend_gpt_oss.storage.lmcache_server_v0_4_3 \
  --host 127.0.0.1 \
  --port 5555 \
  --chunk-size 256 \
  --hash-algorithm blake3 \
  --l1-size-gb 16 \
  --l1-init-size-gb 16 \
  --eviction-policy LRU \
  --max-workers 1 \
  2>&1 | tee "$CACHEBLEND_RUN_DIR/lmcache-server.log"
```

In another shell, create a fresh sidecar and render the exact connector config:

```bash
cd /path/to/cacheblend-gpt-oss

CACHEBLEND_SIDECAR="$CACHEBLEND_RUN_DIR/sidecar.sqlite3"
export CACHEBLEND_SIDECAR
test ! -e "$CACHEBLEND_SIDECAR"
.venv/bin/python -c "import os; from cacheblend_gpt_oss.storage.sidecar import SidecarMode, open_sidecar_index; index = open_sidecar_index(os.environ['CACHEBLEND_SIDECAR'], SidecarMode.WORKER_READ_WRITE); index.close()"

CACHEBLEND_KV_CONFIG=$(
  .venv/bin/python scripts/render_transfer_config.py \
    --sidecar-path "$CACHEBLEND_SIDECAR" \
    --model-revision "$CACHEBLEND_MODEL_REVISION" \
    --tokenizer-revision "$CACHEBLEND_TOKENIZER_REVISION" \
    --model-config-digest "$CACHEBLEND_MODEL_CONFIG_DIGEST" \
    --kv-cache-config-digest "$CACHEBLEND_KV_CONFIG_DIGEST" \
    --adapter-revision "$CACHEBLEND_PLUGIN_COMMIT" \
    --staging-token-capacity "$CACHEBLEND_STAGING_TOKENS" \
    --request-timeout-seconds 300
)
export CACHEBLEND_KV_CONFIG
```

Start the dedicated endpoint:

```bash
.venv/bin/vllm serve "$CACHEBLEND_MODEL_PATH" \
  --served-model-name openai/gpt-oss-20b \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.80 \
  --max-num-seqs 1 \
  --max-num-batched-tokens "$CACHEBLEND_STAGING_TOKENS" \
  --long-prefill-token-threshold 0 \
  --no-async-scheduling \
  --enforce-eager \
  --no-enable-prefix-caching \
  --kv-cache-dtype auto \
  --attention-backend TRITON_ATTN \
  --no-disable-hybrid-kv-cache-manager \
  --enable-auto-tool-choice \
  --tool-call-parser openai \
  --generation-config vllm \
  --kv-transfer-config "$CACHEBLEND_KV_CONFIG" \
  2>&1 | tee "$CACHEBLEND_RUN_DIR/vllm-server.log"
```

Do not use `LMCacheMPConnector` for this gate. It is not this repository's
GPT-OSS CacheBlend connector and cannot supply the required 100%-recompute
transfer evidence.

Create the bounded runtime identity consumed by the offline validator. The
values must match the probe and the actual process:

```bash
jq -n \
  --arg model_revision "$CACHEBLEND_MODEL_REVISION" \
  --arg tokenizer_revision "$CACHEBLEND_TOKENIZER_REVISION" \
  --arg plugin_commit "$CACHEBLEND_PLUGIN_COMMIT" \
  --arg model_config_digest "$CACHEBLEND_MODEL_CONFIG_DIGEST" \
  --arg kv_cache_config_digest "$CACHEBLEND_KV_CONFIG_DIGEST" \
  '{
    model_id: "openai/gpt-oss-20b",
    model_revision: $model_revision,
    tokenizer_revision: $tokenizer_revision,
    plugin_commit: $plugin_commit,
    model_config_digest: $model_config_digest,
    kv_cache_config_digest: $kv_cache_config_digest,
    vllm_version: "0.19.1",
    lmcache_version: "0.4.3",
    torch_version: "2.10.0+cu128",
    cuda_runtime: "12.8",
    gpu_name: "NVIDIA A100-SXM4-80GB",
    dtype: "torch.bfloat16"
  }' > "$CACHEBLEND_RUN_DIR/runtime-identity.json"
```

## 3. Run one append-only BrowseComp query from p7

Read `rag-system/AGENTS.md` before running its existing code. Keep that
repository clean and unmodified. Use its prepared development split and the
preselected one-query gate ID; do not inspect held-out results.

Create a private, gitignored p7 artifact directory and transfer only the
bounded `runtime-identity.json` from g3 into it using the hosts' approved file
transfer mechanism:

```bash
cd /path/to/rag-system
source .env
test "$RAG_QUERY_ID" = 703

CACHEBLEND_P7_RUN_DIR=/absolute/private/path/to/new/browsecomp-append-only-smoke
export CACHEBLEND_P7_RUN_DIR
test ! -e "$CACHEBLEND_P7_RUN_DIR"
mkdir -p "$CACHEBLEND_P7_RUN_DIR/run"
chmod 700 "$CACHEBLEND_P7_RUN_DIR" "$CACHEBLEND_P7_RUN_DIR/run"

# After the approved transfer from g3:
test -s "$CACHEBLEND_P7_RUN_DIR/runtime-identity.json"
```

Take the before scrape while the dedicated endpoint is idle:

```bash
curl --fail-with-body "${RAG_GENERATOR_URL%/v1}/metrics" \
  > "$CACHEBLEND_P7_RUN_DIR/metrics-before.prom"
```

Run the existing agent entry point with the append-only contract. The
`cacheblend` value is workload metadata; the serving connector remains wholly
server-side:

```bash
cd /path/to/rag-system

PYTHONPATH=src python scripts/run_oss_standard_agent.py \
  --prepared-dir "$RAG_PREPARED_DIR" \
  --query-id "$RAG_QUERY_ID" \
  --search-url "$RAG_SEARCH_URL" \
  --generator-url "$RAG_GENERATOR_URL" \
  --model openai/gpt-oss-20b \
  --reasoning-effort "$AGENTIC_REASONING_EFFORT" \
  --max-output-tokens "$AGENTIC_MAX_OUTPUT_TOKENS" \
  --forced-decision-reasoning-effort "$AGENTIC_FORCED_DECISION_REASONING_EFFORT" \
  --forced-decision-max-output-tokens "$AGENTIC_FORCED_DECISION_MAX_OUTPUT_TOKENS" \
  --max-forced-decision-recoveries "$AGENTIC_MAX_FORCED_DECISION_RECOVERIES" \
  --max-iterations "$AGENTIC_MAX_ITERATIONS" \
  --max-search-calls "$AGENTIC_MAX_SEARCH_CALLS" \
  --context-budget-tokens "$AGENTIC_CONTEXT_BUDGET_TOKENS" \
  --context-strategy append_only \
  --cache-mode cacheblend \
  --no-deduplicate-retrieved-documents \
  --generator-timeout-seconds "$GENERATOR_TIMEOUT_SECONDS" \
  --trace-jsonl "$CACHEBLEND_P7_RUN_DIR/trace.jsonl" \
  --output-dir "$CACHEBLEND_P7_RUN_DIR/run"
```

Keep all decrypted questions, retrieved snippets, answers, traces, and run
records in the private gitignored artifact root on p7. Never copy them into
this repository.

After the run, wait for the connector request counter and every native timing
histogram to expose one observation per recorded generation request, then take
the final scrape:

```bash
curl --fail-with-body "${RAG_GENERATOR_URL%/v1}/metrics" \
  > "$CACHEBLEND_P7_RUN_DIR/metrics-after.prom"
```

## 4. Produce a sanitized evidence report

Run the validator on p7 from the CacheBlend checkout. It reads the private run
record but emits no query ID, document ID, prompt text, answer text, reasoning,
response ID, call ID, token sequence, or fingerprint:

```bash
cd /path/to/cacheblend-gpt-oss

python scripts/validate_browsecomp_append_only.py \
  --run-record "$CACHEBLEND_P7_RUN_DIR/run/run_${RAG_QUERY_ID}.json" \
  --metrics-before "$CACHEBLEND_P7_RUN_DIR/metrics-before.prom" \
  --metrics-after "$CACHEBLEND_P7_RUN_DIR/metrics-after.prom" \
  --runtime-identity "$CACHEBLEND_P7_RUN_DIR/runtime-identity.json" \
  --output "$CACHEBLEND_P7_RUN_DIR/browsecomp-append-only-evidence.json" \
  --require-passed
```

The report passes only when all of these facts reconcile:

- the workload completed with the unedited append-only Responses contract;
- connector requests equal recorded generation requests;
- at least one exact candidate chunk was loaded;
- found tokens equal loaded plus rejected tokens;
- every input token was recomputed and no prefill token was avoided;
- every eligible store token completed with zero store fallback;
- native vLLM prompt/source/prefill counters equal the workload usage totals;
- prefix-cache and external scheduler credit are both zero; and
- every native timing family has one observation per generation request.

If `kv_tokens_loaded` is zero, the CacheBlend arm was not reached. If
`prefill_tokens_avoided` is positive, the process is not running this
100%-recomputation milestone. If the after scrape is merely late, take a new
after scrape; never edit the report.

Preserve the sanitized report with the M3--M5 and M8 artifacts, the complete
g3 logs, both private Prometheus snapshots, the private p7 run record, and
before/after `nvidia-smi` output. Only the sanitized report should leave the
private artifact root.

## What this unlocks

A passing result establishes that one real BrowseComp-Plus append-only
trajectory is compatible with the custom endpoint and that KV transfer really
occurred. It does not establish non-prefix document reuse, because an
append-only request naturally repeats prompt-aligned chunks at their old
positions. It also does **not** authorize selective recomputation or a dev-100
performance claim. Reduced recomputation remains blocked until the required
M3--M5 GPU evidence passes and a sink-aware selective backend preserves
deterministic logits/hidden states. The append-only prefix-cache baseline must
remain a separate arm; ordinary vLLM prefix caching is expected to be a strong
baseline for byte-identical growing history.
