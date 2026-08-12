# solab-g3 GPT-OSS Responses contract gate

This manual gate verifies that the CacheBlend-enabled endpoint preserves the
GPT-OSS `/v1/responses` contract across Harmony reasoning, a forced function
call, append-only `function_call_output`, the tool continuation, and one later
user turn. It does not replace the moved-document numerical gate and does not
use natural-language fluency as KV correctness evidence.

No command in this runbook has been executed during local development. Only a
report and logs returned from `solab-g3` can pass this gate.

## Pinned source boundary

vLLM 0.19.1 accepts both response input items and prior output items in
[`ResponsesRequest.input`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/entrypoints/openai/responses/protocol.py#L119-L154).
The upstream named-tool test appends the emitted function call and a matching
`function_call_output` before requesting the continuation in
[`test_named_tool_use`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/tests/entrypoints/openai/responses/test_function_call.py#L111-L169).
The exact GPT-OSS suite separately checks completed Responses and low-effort
reasoning in
[`test_harmony.py`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/tests/entrypoints/openai/responses/test_harmony.py#L82-L105).

This repository's harness is stricter about append-only behavior: it carries
every output item, including Harmony reasoning, into the next input. It records
only structural counts, bounded type/name data, immutable runtime identity, and
aggregate connector counters. Response IDs, call IDs, argument values,
reasoning text, and message text are not written into the report.

## Preconditions

- First complete the environment, compatibility-digest, LMCache, sidecar, and
  transfer-config steps in
  `docs/runbooks/solab-g3-moved-document-correctness.md`.
- Use the same clean `main` commit, model/tokenizer revisions, compatibility
  digests, sidecar namespace, and `CACHEBLEND_KV_CONFIG`.
- The M3 moved-document verdict must already pass. This protocol gate cannot
  waive a numerical failure.
- Stop any earlier vLLM process; keep the pinned LMCache server running.

## Start the Responses-capable endpoint

The tool flags match the external RAG workload's current GPT-OSS serving
contract. All model/cache/attention flags remain the M3 values:

```bash
cd /path/to/cacheblend-gpt-oss
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
  --enable-auto-tool-choice \
  --tool-call-parser openai \
  --generation-config vllm \
  --max-logprobs -1 \
  --kv-transfer-config "$CACHEBLEND_KV_CONFIG" \
  2>&1 | tee "$CACHEBLEND_RUN_DIR/responses-server.log"
```

Do not continue if startup reports a config/digest mismatch, fallback, missing
tool parser, LMCache failure, or a different runtime identity.

## Run the three-turn contract harness

From a second shell with the same exported identity values:

```bash
cd /path/to/cacheblend-gpt-oss

.venv/bin/python scripts/check_responses_contract.py \
  --model-revision "$CACHEBLEND_MODEL_REVISION" \
  --tokenizer-revision "$CACHEBLEND_TOKENIZER_REVISION" \
  --plugin-commit "$CACHEBLEND_PLUGIN_COMMIT" \
  --model-config-digest "$CACHEBLEND_MODEL_CONFIG_DIGEST" \
  --kv-cache-config-digest "$CACHEBLEND_KV_CONFIG_DIGEST" \
  --output "$CACHEBLEND_RUN_DIR/responses-contract.json" \
  | tee "$CACHEBLEND_RUN_DIR/responses-contract.txt"

curl --fail-with-body http://127.0.0.1:8000/metrics \
  | grep 'vllm:cacheblend_' \
  > "$CACHEBLEND_RUN_DIR/responses-contract-metrics.txt"
```

The script performs exactly three non-streaming calls:

1. a low-effort Harmony turn forced to call `get_weather` for Paris;
2. a continuation containing the original user item, every emitted reasoning
   and function-call item, and a matching fixed local tool result; and
3. another append-only turn containing the complete continuation output plus a
   new user item.

It fails unless every response completes, every turn emits a Harmony reasoning
item, the first emits exactly the named function call with valid JSON
arguments, both later turns emit nonempty message text, the fixed city survives
both continuations, exactly three connector requests are observed, all
connector counter deltas reconcile, recomputed tokens are nonzero, and saved
prefill is zero.

## Stop/go decision

Go only when:

- `responses-contract.json` has `passed: true`;
- all three turn structures and append-only counts are present;
- the report's runtime identity exactly matches the M3 artifacts;
- the connector delta has `requests == 3`, positive recomputation, zero saved
  prefill, and `found == loaded + rejected`; and
- the server log contains no fallback, parser error, transfer/correction error,
  or partial group/layer operation.

Preserve the report, filtered metrics, complete server log, M3 verdict, and
runtime identity together. Stop on any mismatch. Do not infer numerical
equivalence from this API pass, and do not begin selective recomputation until
both this gate and all earlier GPU/model gates have user-supplied passing
evidence.
