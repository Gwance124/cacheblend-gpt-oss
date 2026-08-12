# solab-g3 selective-ratio sweep artifact

This runbook is for the future M7 GPU experiment. It does not enable selective
execution and does not turn a CPU-generated row plan into GPU evidence. The
concrete GPT-OSS model/backend remains gated by the M3--M6 stop/go criteria in
the feasibility plan.

## Host and runtime gate

Run the sweep only on `solab-g3`, using the pinned environment:

```bash
cd ~/Workspace/Github/work/cacheblend-gpt-oss
uv run --extra gpu python - <<'PY'
import torch, vllm, lmcache
print({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "vllm": vllm.__version__,
    "lmcache": getattr(lmcache, "__version__", "unknown"),
    "gpu": torch.cuda.get_device_name(0),
})
PY
```

The output must identify PyTorch `2.10.0+cu128`, CUDA `12.8`, vLLM `0.19.1`,
LMCache `0.4.3`, and an NVIDIA A100-SXM4-80GB. Preserve the output with the
experiment artifacts. A local skip or a different GPU is not a pass.

## Artifact contract

The future worker should write one JSON file per prompt case using the
`cacheblend_gpt_oss_selection_sweep` schema. The file contains no prompt text,
token IDs, fingerprints, request IDs, or document IDs. It must include a
descending ratio sweep and exact row ranges for all 24 layers. If the worker
does not yet have measured logits/hidden-state error and selective latency,
leave every point's `measurement` as `null`.

Copy the file to the authoring workstation and validate it read-only:

```bash
cd ~/Workspace/Github/work/cacheblend-gpt-oss
.venv/bin/python scripts/validate_selection_sweep.py \
  --input /path/to/moved-document-selection-sweep.json \
  --output /path/to/moved-document-selection-report.json
```

The report's `measured` and `passed` fields are `false` until every ratio has
explicit finite error and latency measurements. To require those measurements
for a final ratio comparison:

```bash
.venv/bin/python scripts/validate_selection_sweep.py \
  --input /path/to/moved-document-selection-sweep.json \
  --require-measurements
```

Record the printed `artifact_digest` alongside the exact model/config digests,
full-prefill baseline artifact, transfer-evidence sidecar, and complete server
log. Do not infer correctness from fluent text or from a lower recomputation
fraction alone. Stop at the first ratio whose deterministic logits/hidden-state
error exceeds the written accuracy budget.
