# solab-g3 selective backend contract gate

This runbook checks the exact vLLM 0.19.1 Triton boundary and the first live
GPT-OSS selective execution seam. The structural test is still only an API
contract; the live smoke below loads GPT-OSS weights and must run on
`solab-g3`. A local skip is not a pass; only output returned from `solab-g3`
is evidence.

The test is grounded in the pinned source:

- [`TritonAttentionBackend`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/attention/backends/triton_attn.py#L257-L355)
  uses `forward_includes_kv_cache_update=False`, sink support, and the paged
  `[blocks, 2, block, 8, 64]` cache shape.
- [`TritonAttentionImpl.do_kv_cache_update`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/attention/backends/triton_attn.py#L575-L606)
  accepts `(layer, key, value, kv_cache, slot_mapping)` and writes every slot
  through `triton_reshape_and_cache_flash`.

## Structural contract command

Run on `solab-g3`, after synchronizing a clean `main` checkout:

```bash
export CACHEBLEND_REPO=/path/to/cacheblend-gpt-oss
cd "$CACHEBLEND_REPO"
uv sync --extra gpu --extra test

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
uv run python -c "import importlib.metadata as m, torch; print({'torch': torch.__version__, 'torch_cuda': torch.version.cuda, 'vllm': m.version('vllm'), 'lmcache': m.version('lmcache'), 'gpu': torch.cuda.get_device_name(0)})"

uv run pytest tests/gpu/test_selective_backend_contract.py \
  -m "gpu and integration and not model" -vv \
  | tee "$CACHEBLEND_REPO/selective-backend-contract.txt"
```

The output must show the exact versions and A100-SXM4-80GB identity. Preserve
the complete output. This gate only proves that the pinned stock backend has
the expected extension contract; it does not prove CacheBlend transfer,
position correction, sink parity, logits equivalence, or speedup.

## Evidence-hash handoff (after M3--M5 review)

Do not type five arbitrary hexadecimal values into a future registration
configuration. After the M3--M5 artifacts have been independently reviewed,
copy the exact files to one immutable handoff directory and derive the strict
digest bundle from their bytes:

```bash
export CACHEBLEND_GATE_DIR=/absolute/path/to/reviewed-m3-m5-artifacts
uv run python scripts/hash_selective_gate_artifacts.py \
  --runtime "$CACHEBLEND_GATE_DIR/runtime.txt" \
  --full-prefill "$CACHEBLEND_GATE_DIR/frozen-bf16-tolerance.json" \
  --transfer "$CACHEBLEND_GATE_DIR/transfer-evidence.json" \
  --yarn "$CACHEBLEND_GATE_DIR/yarn-correction.txt" \
  --hybrid-sink "$CACHEBLEND_GATE_DIR/hybrid-sink.txt" \
  --output "$CACHEBLEND_GATE_DIR/selective-gate-evidence.json" \
  | tee "$CACHEBLEND_GATE_DIR/selective-gate-evidence.txt"
```

The helper accepts only regular, non-symlink files with bounded nonzero size,
checks that each file is unchanged while read, and emits schema version 1 with
five SHA-256 digests. The bundle is an identity handoff, not semantic proof:
registration remains disabled until reviewers verify the contents and set all
four prerequisite results from the corresponding M3--M5 gates. The digest
JSON contains no prompt text, token IDs, request IDs, or document identifiers.

Before consuming a copied bundle, re-check that none of the five reviewed files
has changed since hashing:

```bash
uv run python scripts/verify_selective_gate_artifacts.py \
  --evidence "$CACHEBLEND_GATE_DIR/selective-gate-evidence.json" \
  --runtime "$CACHEBLEND_GATE_DIR/runtime.txt" \
  --full-prefill "$CACHEBLEND_GATE_DIR/frozen-bf16-tolerance.json" \
  --transfer "$CACHEBLEND_GATE_DIR/transfer-evidence.json" \
  --yarn "$CACHEBLEND_GATE_DIR/yarn-correction.txt" \
  --hybrid-sink "$CACHEBLEND_GATE_DIR/hybrid-sink.txt" \
  | tee "$CACHEBLEND_GATE_DIR/selective-gate-verify.txt"
```

This command verifies identity/freshness only; it does not approve artifact
semantics or enable selective registration.

## M6 matched CUSTOM backend control

The first concrete backend is now available as a matched control only. It does
not register a model override, does not enable `transfer_selective`, and does
not claim layer-token savings. With no row plan bound, it delegates to the
pinned Triton KV write path. Run this exact helper on `solab-g3` after pulling
the branch:

```bash
cd /mnt/nvme3n1/mlee/cacheblend-gpt-oss
bash ./local-m6-custom-backend-control.sh
```

The helper creates a new directory under
`/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/`, reinstalls the editable
project metadata so the new `vllm.general_plugins` entry point is visible,
starts `openai/gpt-oss-20b` with `--attention-backend CUSTOM`, and prints
`VLLM_READY=yes` if the API starts. A successful start validates registration,
sink/backend shape, and the pinned runtime only; it is not the selective speed
test.

The live selective smoke uses the same pinned launch shape with the explicit
`transfer_selective` connector mode and model opt-in. Do not infer that command
from a moving vLLM release or an unpublished CacheBlend image.

The full-plan model-wrapper control can be enabled for a separate short
forward test by exporting this flag before running the helper:

```bash
export CACHEBLEND_ENABLE_CUSTOM_MODEL=1
bash /mnt/nvme3n1/mlee/cacheblend-gpt-oss/local-m6-custom-backend-control.sh
```

This remains a 100%-recompute control and must not be reported as selective
speedup evidence.

## M7 first live selective mechanics smoke

After pulling the branch, run this exact helper on `solab-g3`:

```bash
cd /mnt/nvme3n1/mlee/cacheblend-gpt-oss
bash ./local-m7-selective-smoke-g3.sh
```

The helper creates a fresh directory under
`/mnt/nvme3n1/mlee/cacheblend-gpt-oss/artifacts/`, starts LMCache on `127.0.0.1:5556`,
starts the pinned GPT-OSS server on port `8000` with
`CACHEBLEND_ENABLE_CUSTOM_BACKEND=1` and `CACHEBLEND_ENABLE_CUSTOM_MODEL=1`,
then runs the synthetic moved-document request. It leaves the server running
after the smoke so the same process can be used for the next targeted test.

The required output is:

- `VLLM_READY=yes`;
- `SELECTIVE_CAPTURE_STATUS=0`;
- `connector.kv_tokens_loaded` equal to the reusable moved-document tokens;
- `selective_work.layer_token_rows_avoided` greater than zero; and
- `selective_work.layer_token_rows_recomputed + layer_token_rows_avoided`
  equal to `24 * target_prompt_tokens`.

This smoke deliberately does **not** pass a 100%-overwrite transfer-evidence
path: cached rows are supposed to remain untouched in selective mode. It proves
row-plan propagation and bounded work accounting only; it is not yet a
numerical-equivalence or BrowseComp speed claim. The output artifact is
`cacheblend-selective.json` in the printed run directory, and the raw metrics
are in `selective-metrics.prom`.

## Stop/go

Stop if the structural test reports a different backend shape/signature,
attention sink support, runtime version, or GPU identity. Do not add a vLLM
patch based on this test alone. A patch requires a later, user-supplied M6
model/backend failure after the public registries and dormant CPU contracts are
actually exercised.
