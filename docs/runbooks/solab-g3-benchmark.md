# solab-g3 controlled benchmark evidence

This runbook describes the M9 evidence hand-off. It does not run a benchmark
from the authoring workstation and does not claim any GPU result. Run the
serving arms only on `solab-g3` after the M3 moved-document correctness,
M4 YaRN correction, M5 hybrid/sink, and M8 Responses gates have supplied their
artifacts.

## Runtime and arm controls

Use the pinned environment and preserve the identity output:

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

The required identity is PyTorch `2.10.0+cu128`, CUDA `12.8`, vLLM `0.19.1`,
LMCache `0.4.3`, and NVIDIA A100-SXM4-80GB. Use TP=1/PP=1, block size 16,
131,072 maximum model length, hybrid KV-cache management enabled, Triton
attention with learned sinks, temperature 0, top-p 1, and seed 0. Warm kernels
separately from reusable cache content.

Run isolated arms with identical prompt token fixtures and runtime/config
digests:

1. `full_prefill`, prefix caching off.
2. `vllm_prefix_exact`, exact-prefix repeat.
3. `vllm_prefix_moved`, moved-document control (expected prefix miss).
4. `cacheblend_100pct`, moved-document transfer with all prompt rows
   recomputed and zero saved-prefill fraction.
5. `cacheblend_selective`, only at ratios that already passed the M7 numerical
   gate.
6. `prefix_plus_cacheblend`, only after the individual arms pass.

Repeat each arm enough to report confidence intervals. Keep lookup/transfer/
correction/selective/prefill/TTFT/end-to-end timings separate and preserve the
full correctness artifact and, for CacheBlend arms, the validated transfer
evidence sidecar for every trial. Use one exact prompt-fixture digest across all
arms in a case. Do not put
request IDs, prompt text, token IDs, fingerprints, or document IDs into the
benchmark JSON.

## Validate on the authoring workstation

Copy one case artifact from `solab-g3` and validate it read-only:

```bash
cd ~/Workspace/Github/work/cacheblend-gpt-oss
.venv/bin/python scripts/validate_benchmark.py \
  --input /path/to/moved-document-benchmark.json \
  --output /path/to/moved-document-benchmark-report.json
```

The report includes per-arm means, medians, and normal-approximation 95%
confidence intervals. `benchmark_ready` and `passed` remain false if a required
control arm is missing or any recorded trial lacks passing numerical
correctness evidence. A correctness artifact digest alone is not sufficient:
each passing trial must also carry finite maximum and mean absolute logit (or
adapter-produced hidden-state) error in its request metrics.
The derived report retains the artifact digest, prompt-fixture digest, one
uniform warm/cold cache state, and the complete pinned runtime/config identity;
do not detach the report from those fields when copying evidence between hosts.
For a final comparison, require readiness explicitly:

```bash
.venv/bin/python scripts/validate_benchmark.py \
  --input /path/to/moved-document-benchmark.json \
  --require-ready
```

Treat a missing or failed correctness artifact as a benchmark stop, not as a
slow trial. Compare ordinary full prefill, vLLM prefix caching, CacheBlend at
100%, and each approved selective ratio only when their runtime/config digests
and prompt-token fixtures match.
