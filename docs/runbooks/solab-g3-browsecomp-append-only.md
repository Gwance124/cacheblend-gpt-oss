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

Do not start this smoke from the current diagnostic artifact. First complete
and preserve the formal artifacts from:

1. `solab-g3-moved-document-correctness.md` (a formal M3 numerical and
   all-layer transfer pass, followed by M4/M5 evidence); and
2. `solab-g3-responses-contract.md` (M8 Harmony, tools, and append-only API
   evidence).

The current user-supplied g3 evidence is not that precondition. Its schema-v2
transfer report passed for all 24 layers/groups, with 256 tokens loaded, 280
tokens recomputed, zero prefill tokens avoided, and artifact binding passed.
However, the old two-baseline numerical verdict failed: CacheBlend/reference
`max_abs_error=0.07559013366699219` and
`mean_abs_error=0.009438995041906416` exceeded frozen limits
`0.07054328918457031` and `0.006880644322352224`. A later five-control
1024-token comparison produced diagnostic `Umax=0.07423019409179688`,
`Umean=0.01318507041618522`, `Qmax=0.05914115905761719`, and
`Qmean=0.01111537456170986`, with sampled/top-token agreement. That candidate
is diagnostic-only because the five-control policy was defined after the
candidate was observed; it does not satisfy formal M3.

The later fresh candidate used for the formal strict-v1 response is an
immutable **`FAIL`**. Exactly one token, ID `71784`, violated the strict
full-vocabulary maximum: `Qmax=0.0984172821` versus
`Umax=0.0742301941`; its rank was `199583`, its probability was `5.0698e-8`,
and it was `0.0326042` outside the baseline range. The full-vocabulary mean
remained within its envelope (`Qmean=0.012911056652712563` versus
`Umean=0.01318507041618522`), sampled and top-token agreement remained true,
and transfer evidence passed. These v1 facts cannot be rewritten by a later
policy, and they do not establish v1 or M3 passage.

BrowseComp-Plus remains blocked pending a prospective probability-aware v2
response. V2 reuses the same five controls, freezes a separate probability-v2
manifest before one new candidate, and does not rerun the baselines. The
strict-v1 manifest remains immutable historical evidence and is not the v2
policy manifest. The failed strict-v1 candidate is digest-bound into v2 only as
an excluded pilot and cannot be reused as the prospective candidate.
It uses fixed `epsilon=1e-4` and requires the new candidate to be within the
empirical baseline envelope on full-vocabulary mean error, TV, and JS
divergence, with hard ceilings full mean `0.014`, TV `0.02`, and JS `0.001`.
The high-mass maximum remains a reported diagnostic only because the
connector-attached controls showed that its maximum over roughly 198k
coordinates is not stable on the pinned A100 stack. Sampled/top-token
agreement and independent transfer pass are also required. Do not claim this
policy or M3 passed until that prospective evidence exists.

The five controls are the existing Solab artifacts below. The final path is the
strict-v1 manifest retained only as historical evidence:

```text
/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m3-75bfe75-probe1-20260813/full-prefill-1024-control-1.json
/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m3-75bfe75-probe1-20260813/full-prefill-1024-control-2.json
/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m3-75bfe75-probe1-20260813/full-prefill-1024-control-3.json
/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m3-75bfe75-probe1-20260813/full-prefill-1024-control-4.json
/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m3-75bfe75-probe1-20260813/full-prefill-1024-control-5.json
/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m3-75bfe75-formal1-20260813/frozen-five-baseline-manifest.json
```

The eventual v2 candidate and verdict belong in a new create-only directory
under `/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/`, derived from the
actual checkout commit at run time. Do not document or invent an eventual new
plugin SHA. Conceptually, the command shell starts from:

```bash
cd /mnt/nvme3n1/mlee/cacheblend-gpt-oss
export CACHEBLEND_SERVING_COMMIT=75bfe75db794a77d305c495be5f8114e520d119f
export CACHEBLEND_V2_RUN_DIR=/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m3-75bfe75-formal-v2-20260813

test "$(git rev-parse HEAD)" = "$CACHEBLEND_SERVING_COMMIT"
```

The new candidate must be captured and evaluated under
`$CACHEBLEND_V2_RUN_DIR`; the v1 candidate and verdict remain in the existing
`solab-g3-m3-75bfe75-formal1-20260813` directory.

For the formal M3 rerun, capture five ordinary controls with the same
`--max-num-batched-tokens 1024` setting and source-warm-up-then-target ordering,
freeze their manifest and numerical envelope before capturing a fresh
candidate, and do not loosen the policy post hoc. For the prospective v2
response, use the already-frozen five controls above and capture one new
candidate without rerunning those controls. BrowseComp-Plus remains blocked
until the v2 response and the other M3--M8 preconditions pass.

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
cd /mnt/nvme3n1/mlee/cacheblend-gpt-oss

CACHEBLEND_MODEL_PATH=/mnt/nvme3n1/labuser/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee
CACHEBLEND_MODEL_REVISION=6cee5e81ee83917806bbde320786a8fb61efebee
CACHEBLEND_TOKENIZER_REVISION=6cee5e81ee83917806bbde320786a8fb61efebee
CACHEBLEND_PLUGIN_COMMIT=$(git rev-parse HEAD)
CACHEBLEND_RUN_DIR="/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/solab-g3-m8.5-${CACHEBLEND_PLUGIN_COMMIT:0:7}-browsecomp-append-only-20260813"
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

Verify that the two emitted lowercase SHA-256 values match these pinned values:

```bash
export CACHEBLEND_MODEL_CONFIG_DIGEST=1c69c7868c1206ea76c372df01e5baa2abcadcd2ca5b9f93b97d94fa6070aae0
export CACHEBLEND_KV_CONFIG_DIGEST=131eb7ec025bc9a4fa1dabd220bb41b75c7d8f921e537fd8be505e91c6850742
```

## 2. Start LMCache and vLLM on g3

In one shell, launch the repository's version-scoped LMCache server wrapper.
Do not launch LMCache's raw 0.4.3 module because it contains the audited store
completion race:

```bash
cd /mnt/nvme3n1/mlee/cacheblend-gpt-oss
export CUDA_VISIBLE_DEVICES=0

.venv/bin/python -m cacheblend_gpt_oss.storage.lmcache_server_v0_4_3 \
  --host 127.0.0.1 \
  --port 5556 \
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
cd /mnt/nvme3n1/mlee/cacheblend-gpt-oss

CACHEBLEND_SIDECAR="$CACHEBLEND_RUN_DIR/sidecar.sqlite3"
export CACHEBLEND_SIDECAR
test ! -e "$CACHEBLEND_SIDECAR"
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
cd /mnt/nvme3n1/mlee/cacheblend-gpt-oss

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
formal M3--M5 GPU evidence passes and a sink-aware selective backend preserves
deterministic logits/hidden states. The append-only prefix-cache baseline must
remain a separate arm; ordinary vLLM prefix caching is expected to be a strong
baseline for byte-identical growing history. The current all-layer transfer
report alone does not unlock this runbook.
