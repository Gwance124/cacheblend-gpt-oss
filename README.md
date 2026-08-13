# GPT-OSS CacheBlend prototype

This repository is a focused research prototype for non-prefix KV-cache reuse
with **`openai/gpt-oss-20b` only**. It is not a drop-in installation for the
original public CacheBlend fork: that fork targets an old vLLM 0.4.1 Llama /
XFormers stack and does not support GPT-OSS's YaRN RoPE, learned sinks, MoE
model, or hybrid KV-cache groups.

The current implementation is an out-of-tree vLLM V1 connector and a
GPT-OSS-specific transfer/data-plane seam. The first milestone deliberately
loads candidate KV for instrumentation and then recomputes **100% of the
prompt**, reporting zero externally computed tokens and zero saved prefill.
Selective recomputation remains dormant until the required GPU correctness
gates pass.

## Supported envelope

Only this runtime is supported:

- vLLM `0.19.1`
- LMCache `0.4.3`
- PyTorch `2.10.0+cu128`, CUDA runtime `12.8`
- one NVIDIA A100-SXM4-80GB, TP=1/PP=1
- vLLM V1, hybrid KV-cache manager enabled, sink-capable `TRITON_ATTN`

Unsupported or unverified configurations fail startup or use ordinary full
prefill. The repository does not import `rag-system`; that workload remains a
separate client of a future validated `/v1/responses` endpoint.

## Local checks

The authoring workstation is CPU-only for this project. Run:

```bash
.venv/bin/python -m pytest -m "not gpu"
.venv/bin/ruff check src tests scripts
.venv/bin/mypy src --strict
UV_CACHE_DIR=/tmp/cacheblend-uv uv lock --check
```

The GPU-marked tests intentionally skip when Torch/vLLM are absent. A local
skip is not GPU evidence.

## `solab-g3` workflow

Run the manual gates in this order, from a clean `main` checkout on
`solab-g3`:

1. [connector loading and CUDA primitive smoke test](docs/runbooks/solab-g3-connector-smoke.md)
2. [moved-document 100%-recompute correctness](docs/runbooks/solab-g3-moved-document-correctness.md)
3. [GPT-OSS `/v1/responses` Harmony/tool/multi-turn contract](docs/runbooks/solab-g3-responses-contract.md)
4. [one-query BrowseComp-Plus append-only transfer smoke](docs/runbooks/solab-g3-browsecomp-append-only.md)
5. [selective backend contract](docs/runbooks/solab-g3-selective-contract.md), only after the earlier numerical gates
6. [controlled benchmark](docs/runbooks/solab-g3-benchmark.md), only after correctness artifacts exist

Return the complete command output and generated artifacts before recording a
GPU pass. The exact source boundary, lifecycle, metrics, and stop/go criteria
are in [the architecture](docs/architecture.md), [the feasibility plan](docs/plans/gpt-oss-cacheblend-feasibility.md), and [the pinned source audit](docs/source-audit.md).

## Current status

The pinned source audit shows that connector loading and the 100%-recompute
transfer path can remain out of tree; no vLLM patch is justified for those
milestones. CPU contracts cover position-independent fingerprints and lookup,
hybrid full/sliding groups, YaRN key correction, staging, sidecar integrity,
Responses parsing, and fail-closed metrics. No model/GPU result has been
claimed from this workstation.

LMCache 0.4.3's `CB_STORE_PRE_COMPUTED` protocol currently admits compact
complete 256-token prefix chunks. Lookup can find a moved chunk at an arbitrary
target position, but arbitrary embedded-document persistence requires a later
per-range gather/store design and is not silently inferred.
