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
cd /path/to/cacheblend-gpt-oss

CACHEBLEND_MODEL_PATH=/path/to/pinned/gpt-oss-20b
CACHEBLEND_MODEL_REVISION=replace-with-model-commit-or-manifest-sha
CACHEBLEND_TOKENIZER_REVISION=replace-with-tokenizer-commit-or-manifest-sha
CACHEBLEND_PLUGIN_COMMIT=$(git rev-parse HEAD)
CACHEBLEND_RUN_DIR=/absolute/path/to/new/cacheblend-m3-run

export CACHEBLEND_MODEL_PATH CACHEBLEND_MODEL_REVISION
export CACHEBLEND_TOKENIZER_REVISION CACHEBLEND_PLUGIN_COMMIT
export CACHEBLEND_RUN_DIR

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
  --max-num-batched-tokens 512 \
  --long-prefill-token-threshold 0 \
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

A successful probe intentionally terminates startup after printing two
lowercase SHA-256 values. Copy them exactly:

```bash
CACHEBLEND_MODEL_CONFIG_DIGEST=replace-with-probe-model-config-digest
CACHEBLEND_KV_CONFIG_DIGEST=replace-with-probe-kv-cache-config-digest
export CACHEBLEND_MODEL_CONFIG_DIGEST CACHEBLEND_KV_CONFIG_DIGEST
```

Any other startup failure is not a successful probe.

## 3. Capture and freeze ordinary full-prefill behavior

Start a fresh server with no connector and otherwise identical flags:

```bash
export VLLM_USE_V2_MODEL_RUNNER=0

.venv/bin/vllm serve "$CACHEBLEND_MODEL_PATH" \
  --served-model-name openai/gpt-oss-20b \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 512 \
  --long-prefill-token-threshold 0 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --kv-cache-dtype auto \
  --attention-backend TRITON_ATTN \
  --no-disable-hybrid-kv-cache-manager \
  --generation-config vllm \
  --max-logprobs -1 \
  2>&1 | tee "$CACHEBLEND_RUN_DIR/full-prefill-server.log"
```

From a second shell, capture the target prompt twice:

```bash
cd /path/to/cacheblend-gpt-oss

.venv/bin/python scripts/capture_moved_document.py \
  --mode full_prefill \
  --model-revision "$CACHEBLEND_MODEL_REVISION" \
  --tokenizer-revision "$CACHEBLEND_TOKENIZER_REVISION" \
  --plugin-commit "$CACHEBLEND_PLUGIN_COMMIT" \
  --model-config-digest "$CACHEBLEND_MODEL_CONFIG_DIGEST" \
  --kv-cache-config-digest "$CACHEBLEND_KV_CONFIG_DIGEST" \
  --output "$CACHEBLEND_RUN_DIR/full-prefill-reference.json"

.venv/bin/python scripts/capture_moved_document.py \
  --mode full_prefill \
  --model-revision "$CACHEBLEND_MODEL_REVISION" \
  --tokenizer-revision "$CACHEBLEND_TOKENIZER_REVISION" \
  --plugin-commit "$CACHEBLEND_PLUGIN_COMMIT" \
  --model-config-digest "$CACHEBLEND_MODEL_CONFIG_DIGEST" \
  --kv-cache-config-digest "$CACHEBLEND_KV_CONFIG_DIGEST" \
  --output "$CACHEBLEND_RUN_DIR/full-prefill-repeat.json"

.venv/bin/python scripts/freeze_correctness_tolerance.py \
  --reference "$CACHEBLEND_RUN_DIR/full-prefill-reference.json" \
  --repeat "$CACHEBLEND_RUN_DIR/full-prefill-repeat.json" \
  --max-abs-floor 0 \
  --mean-abs-floor 0 \
  --multiplier 1 \
  --output "$CACHEBLEND_RUN_DIR/frozen-bf16-tolerance.json" \
  | tee "$CACHEBLEND_RUN_DIR/frozen-bf16-tolerance.txt"
```

This BF16 policy allows exactly the observed repeated-full-prefill maximum and
mean error, with zero added floor and no multiplier. Freeze the file before
starting CacheBlend. Do not relax it after seeing the CacheBlend result.

Stop the ordinary server before continuing.

## 4. Start LMCache and the 100% connector

Start the pinned public LMCache server in one shell, using the exact command in
`solab-g3-connector-smoke.md`. In another shell, create a new sidecar path. Do
not reuse a database from another model/config/plugin identity:

```bash
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
    --staging-token-capacity 512 \
    --request-timeout-seconds 120
)
export CACHEBLEND_KV_CONFIG
```

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
  --max-num-batched-tokens 512 \
  --long-prefill-token-threshold 0 \
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

.venv/bin/python scripts/evaluate_cacheblend_correctness.py \
  --reference "$CACHEBLEND_RUN_DIR/full-prefill-reference.json" \
  --cacheblend "$CACHEBLEND_RUN_DIR/cacheblend-100pct.json" \
  --tolerance "$CACHEBLEND_RUN_DIR/frozen-bf16-tolerance.json" \
  --transfer-evidence "$CACHEBLEND_RUN_DIR/transfer-evidence.json" \
  --output "$CACHEBLEND_RUN_DIR/cacheblend-verdict.json" \
  | tee "$CACHEBLEND_RUN_DIR/cacheblend-verdict.txt"

curl --fail-with-body http://127.0.0.1:8000/metrics \
  | grep 'vllm:cacheblend_' \
  > "$CACHEBLEND_RUN_DIR/cacheblend-metrics.txt"
```

If the worker-side probe has produced the required per-layer digest sidecar,
validate it independently before accepting the final-distribution result:

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

The evaluator exits nonzero unless sampled/top tokens agree and complete-vector
maximum and mean errors stay inside the already-frozen envelope. Preserve the
three artifacts, verdict, metrics, both server logs, compatibility probe, and
identity outputs.

## Required case matrix

The capture harness also has deterministic fixtures for the other required
correctness cases. Each case needs its own fresh baseline pair, frozen
tolerance, and CacheBlend artifact because the target prompt digest changes.
Keep the same server identity and run directory, but use distinct filenames:

```bash
for CASE in exact_prefix moved_document reordered_documents cache_miss; do
  .venv/bin/python scripts/capture_moved_document.py \
    --mode full_prefill \
    --case "$CASE" \
    --model-revision "$CACHEBLEND_MODEL_REVISION" \
    --tokenizer-revision "$CACHEBLEND_TOKENIZER_REVISION" \
    --plugin-commit "$CACHEBLEND_PLUGIN_COMMIT" \
    --model-config-digest "$CACHEBLEND_MODEL_CONFIG_DIGEST" \
    --kv-cache-config-digest "$CACHEBLEND_KV_CONFIG_DIGEST" \
    --output "$CACHEBLEND_RUN_DIR/${CASE}-full.json"
done
```

With the CacheBlend server and a clean sidecar state appropriate to the case,
repeat the same loop with `--mode cacheblend_100pct` and output names
`${CASE}-cacheblend.json`; then evaluate each pair with its own frozen
tolerance. The expected transfer evidence is:

| Case | Reusable segments | Expected loaded tokens | Position relation |
|---|---:|---:|---|
| `exact_prefix` | 1 | 256 | source and target start at 0 |
| `moved_document` | 1 | 256 | source 0, target 17 |
| `reordered_documents` | 2 | 512 | two 256-token documents swap order |
| `cache_miss` | 0 | 0 | target contains a new 256-token document |

Every case must still report full target-prompt recomputation and zero saved
prefill. A cache miss is a successful ordinary-prefill fallback only when its
artifact explicitly reports zero found/loaded tokens and the final distribution
passes the corresponding full-prefill comparison. Do not use one case's
tolerance or sidecar evidence to judge another case.

## Stop/go decision

Go to the `/v1/responses` Harmony/tool/multi-turn gate only if:

- the artifact reports `kv_tokens_loaded == 256`;
- `tokens_recomputed == 280` and `prefill_tokens_avoided == 0`;
- the evaluator reports `passed: true`;
- all identities/digests match; and
- no server log contains a fallback, rejected configuration, transfer error,
  correction error, or partial group/layer operation.

Stop on any failed invariant, timeout, fallback, distribution mismatch, or
identity drift. Do not lower recomputation or interpret generated text as
evidence. This gate demonstrates the final output distribution and the
connector's all-layer/group success accounting. A supplied transfer sidecar
must validate independently; raw per-layer evidence cannot be inferred from
the output artifact or connector counters.
