# Repository instructions

These rules apply to the entire `cacheblend-gpt-oss` repository.

## Scope and support envelope

- This is a focused research prototype for `openai/gpt-oss-20b` only.
- The only supported serving stack is vLLM `0.19.1`, LMCache `0.4.3`,
  PyTorch `2.10.0+cu128`, CUDA runtime `12.8`, and an
  `NVIDIA A100-SXM4-80GB`.
- Do not claim support for other models, vLLM versions, LMCache versions,
  attention backends, GPUs, or CUDA versions.
- Preserve the vLLM `/v1/responses` contract, Harmony reasoning, tool calls,
  and append-only multi-turn inputs.
- Unsupported or unverified configurations must fail startup or fall back to a
  documented full-prefill path. Never silently reuse KV that has not passed all
  compatibility checks.

## Source and compatibility discipline

- Design against the exact pinned sources, never a moving `main` branch:
  - vLLM tag `v0.19.1`, commit
    `b1388b1fbf5aaef47937fabe98931211684666a6`.
  - LMCache tag `v0.4.3`, commit
    `7f326118a2f1afc7801988dd02e3055bdf21ef6b`.
  - The audited public CacheBlend reference snapshot is commit
    `55ad02675939f783a38d579393527d218a7fd581`.
- Cite a pinned source line for every use of a vLLM or LMCache internal API.
- Keep all vLLM imports and layout assumptions under
  `cacheblend_gpt_oss.vllm_compat.v0_19_1`.
- The connector must be importable in both scheduler and worker processes and
  must use the current three-argument V1 constructor, including
  `kv_cache_config`.
- GPT-OSS hybrid KV groups are part of the target. Do not work around them by
  disabling the hybrid KV-cache manager. The connector must implement vLLM's
  `SupportsHMA` contract and handle every cache group explicitly.
- Do not depend on `LMCacheMPCBConnector`, `CBKVConnector`,
  `lmcache_cacheblend`, or any unpublished image. Those implementations are not
  present in the pinned public releases.
- Do not vendor vLLM, LMCache, or the original CacheBlend fork into this
  repository. A small patch may be added under a version-named patch directory
  only after an evidence-backed stop/go gate shows public extension points are
  insufficient.
- Custom schedulers are not a supported integration boundary for this project.

## Architecture boundaries

- Keep the planner, storage/transport, connector, GPT-OSS adapter, metrics, and
  vLLM compatibility layers separate and dependency-injected.
- Generic planner interfaces are allowed, but a generic interface is not a
  support claim. Runtime support remains GPT-OSS-20B only.
- The first transfer milestone reports zero externally computed tokens to the
  scheduler and recomputes 100% of the prompt. Loaded KV is instrumentation data
  and is synchronously overwritten by ordinary prefill.
- Do not introduce selective recomputation until full-prefill equivalence,
  shifted-key correction, hybrid-group handling, and attention-sink tests pass.
- Keep cache identity separate from prompt position. Include the exact token
  sequence and all model/cache compatibility fields in verification; store old
  positions as correction metadata, not as fingerprint identity.
- Treat cache hits as candidates until a strong digest and exact token sequence
  check succeed. Hash-table or rolling-hash matches alone are insufficient.
- Preserve learned attention sinks as model parameters. They are not KV tokens
  and must never be serialized as document cache entries.

## Correctness and observability

- Fluent output is not correctness evidence. Compare deterministic logits or
  hidden states against ordinary full prefill.
- Required cases are exact prefix, moved document, reordered documents, cache
  miss, full-attention layers, and 128-token sliding-attention layers.
- At 100% recomputation, require close numerical agreement with a repeated
  full-prefill baseline and record the exact dtype-specific tolerances used.
- When recomputation is reduced, record approximation error rather than hiding
  it behind output-level agreement.
- Track requested reusable tokens, candidates found, loaded and rejected KV
  tokens, recomputed tokens, document/token hit fractions, effective saved
  prefill, lookup/transfer/correction/recomputation time, TTFT, prefill latency,
  and correctness error.
- Metrics and logs must not use request IDs, document IDs, prompt text, token
  sequences, or fingerprints as unbounded Prometheus labels.

## Code and tests

- Require Python `>=3.10,<3.14`, type hints, small modules, and explicit
  protocols or constructor injection at subsystem boundaries.
- Avoid importing vLLM, LMCache, CUDA, or PyTorch from package top-level code so
  CPU-only unit tests remain lightweight.
- Put CPU-only tests in `tests/unit`.
- Mark every GPU test with both `gpu` and `integration`. A test that loads
  GPT-OSS must also use the `model` marker.
- Run local checks with:

  ```bash
  python -m pytest -m "not gpu"
  ```

- GPU commands are for the user to run manually on `solab-g3`. Never claim a
  GPU test passed unless the user supplies its output.
- Prefer deterministic fixtures and fake planner/storage/connector interfaces
  for unit tests. No unit test may require model weights or network access.
- Preserve SPDX headers, notices, commit links, and license attribution for any
  code adapted from CacheBlend, vLLM, or LMCache.

## Development topology and repository separation

- Authoring and CPU source analysis happen on the local workstation.
- GPU execution happens separately on `solab-g3`.
- RAG orchestration happens separately on `solab-p7`.
- Do not assume this workstation can reach either host.
- The external workload repository is
  `~/Workspace/Github/work/rag-system`. Read its own `AGENTS.md` before any
  inspection.
- Treat `rag-system` as strictly read-only during this feasibility project. Do
  not create, edit, delete, format, commit, or generate files there.
- Do not import `rag-system` as a runtime dependency. It will eventually use the
  validated endpoint through configuration and record deployment metadata such
  as the CacheBlend commit SHA.
- Document any proposed future RAG changes here and wait for explicit
  authorization before editing the other repository.

