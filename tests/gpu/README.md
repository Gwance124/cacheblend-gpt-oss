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
