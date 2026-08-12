# Manual GPU checks

These checks are authored locally but must be run by the user on `solab-g3`.
They validate the pinned environment, production BF16 YaRN correction, and the
real Torch/CUDA full/sliding gather-scatter path. They do not load model weights
yet.

The pinned attention-boundary check is separate:

```bash
uv run pytest tests/gpu/test_selective_backend_contract.py \
  -m "gpu and integration and not model" -vv
```

It inspects the exact sink-capable Triton backend and
`do_kv_cache_update` signature without loading GPT-OSS. See
`docs/runbooks/solab-g3-selective-contract.md`; this is not selective-cache
execution evidence.

After synchronizing this repository to a chosen path on `solab-g3`:

```bash
CACHEBLEND_REPO=/path/to/cacheblend-gpt-oss
cd "$CACHEBLEND_REPO"
uv sync --extra gpu --extra test
uv run pytest -m "gpu and integration and not model" -vv
```

For a pip-based environment, install the pinned CUDA wheel source explicitly:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade "pip==25.2"
.venv/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -e ".[gpu,test]"
.venv/bin/python -m pytest -m "gpu and integration and not model" -vv
```

Do not treat a local skip as a GPU pass. Preserve and return the full command
output before recording any `solab-g3` result.

Before any weight-loading or numerical gate, point the model-marked config
check at the exact local GPT-OSS-20B checkpoint. It uses
`local_files_only=True`, so it never downloads a model during the test:

```bash
export CACHEBLEND_MODEL_PATH=/absolute/path/to/pinned/gpt-oss-20b
uv run pytest tests/gpu/test_gpt_oss_model_config.py \
  -m "gpu and integration and model" -vv
```

The check must report the exact 24-layer alternating sliding/full layout,
128-token window, 131,072-token context, 64/8 heads with dimension 64, YaRN
factor/original context, 32 experts with 4 active, and 201,088-token
vocabulary. This is a configuration gate only; it is not logits, transfer, or
`/v1/responses` evidence.

The external connector import smoke check is separate and does not construct a
server or load weights:

```bash
uv run pytest tests/gpu/test_connector_loading.py \
  -m "gpu and integration and not model" -vv
```

It must import vLLM `0.19.1`, dynamically import
`cacheblend_gpt_oss.vllm_compat.v0_19_1.connector`, verify `KVConnectorBase_V1`
and `SupportsHMA`, and inspect the current three-argument constructor.
