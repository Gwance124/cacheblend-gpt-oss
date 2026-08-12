# solab-g3 selective backend contract gate

This runbook checks the exact vLLM 0.19.1 Triton boundary that a future M6
selective backend must replace. It does not load GPT-OSS weights, register an
unimplemented model override, or claim selective execution. A local skip is not
a pass; only output returned from `solab-g3` is evidence.

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

## M6 model/backend command (not yet enabled)

Do not run a `CUSTOM` server command yet. This repository intentionally has no
`vllm.general_plugins` entry point or concrete selective model/backend class.
The registrar and row/update contracts are dormant until the M3--M5 GPU/model
artifacts are supplied and reviewed. Enabling `CUSTOM` now would either fail
startup or risk silently changing the serving path.

When those prerequisites and concrete classes exist, the first M6 command will
be a single-request, eager, TP=1 launch with `--attention-backend CUSTOM`,
`--no-disable-hybrid-kv-cache-manager`, V2 runner disabled, prefix caching off,
and the same pinned LMCache/sidecar configuration as
`solab-g3-moved-document-correctness.md`. The command must be added here with
the exact entry-point name and class paths before execution; do not infer them
from a moving vLLM release or an unpublished CacheBlend image.

## Stop/go

Stop if the structural test reports a different backend shape/signature,
attention sink support, runtime version, or GPU identity. Do not add a vLLM
patch based on this test alone. A patch requires a later, user-supplied M6
model/backend failure after the public registries and dormant CPU contracts are
actually exercised.
