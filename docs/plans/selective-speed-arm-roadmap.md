# Selective-recompute feasibility arm roadmap (append-only dev-100)

> Status: **planning / not started.** Staged roadmap, authored 2026-08-14, revised
> 2026-08-14 after external review. It records decisions and the milestone dependency
> chain; it does not itself change code. All GPU/RAG steps run on `solab-g3` / `solab-p7`;
> the authoring workstation is CPU-only. See
> [the feasibility plan](gpt-oss-cacheblend-feasibility.md) for the source-pinned milestone
> definitions (M3–M9) referenced below.

## Goal

> Build and validate a **selective-recompute feasibility arm** that demonstrates positive
> **layer-token** compute savings on GPT-OSS-20B without unacceptable numerical or
> task-quality regression; promote it to BrowseComp-Plus dev-100 **only after** matched-backend
> and one-query gates pass.

Scope is BrowseComp-Plus **append-only** only (dynamic changing-summary + reordered-docs mode
is deferred as a later showcase). The comparison is against the existing *no-prefix-cache* and
*with-prefix-cache* append-only arms. **Speed is measured as layer-token rows recomputed vs.
avoided (see Phase E1 for why native prompt-token counters cannot show this out-of-tree), with
accuracy/recall tracked on p7.**

## Revision note (what changed after review)

1. Headline goal reframed from "enable selective recompute → dev-100" to a **feasibility arm**
   with explicit matched-backend and one-query gates before dev-100.
2. **M3 diagnostic reordered:** the cheap *scatter-disabled* control runs **first** to bisect
   connector/allocator drift from loaded-KV contamination, before any per-slot instrumentation.
3. **Metrics corrected:** the out-of-tree connector **cannot** make native vLLM report
   `external_kv_transfer` credit or reduced native prefill work — vLLM 0.19.1's connector API
   credits only one contiguous prefix (feasibility `:767`). Savings are reported as
   **CacheBlend-internal layer-token metrics**; native vLLM still reports the full prompt
   processed. A vLLM patch for non-prefix work accounting is an explicit **M6-gated** option.
4. **Selective policy gap acknowledged:** `check_layer` is currently inert — the same recompute
   ranges are replicated to all 24 layers (`selective_policy.py:386-390`). The CPU scaffold is
   **not** a complete CacheBlend layered plan; fixing it is a Phase-C prerequisite.
5. **Matched-backend control arm added** (CUSTOM backend at 100% recompute, no scatter).
6. **Calibration hygiene added:** a predeclared calibration set picks and freezes the recompute
   ratio; dev-100 is never used to choose it.

## Critical path (milestone dependency chain)

```
M3 numerical equivalence (FAILING)  ──►  M4 YaRN + M5 hybrid/sinks (CPU-done, GPU-pending)
        │                                          │
        │                          selective policy check-layer fix (CPU, prerequisite)
        │                                          │
        └──────────────────────────────►  M6 selective backend (UNBUILT)
                                                   │
                                          M7 recompute-ratio sweep + calibration freeze
                                                   │
              matched-backend + one-query gates  ──►  dev-100 arm  ──►  accuracy/recall on p7
```

Nothing downstream is trustworthy until **M3 is green**. The immediate next work is the
**scatter-disabled M3 diagnostic — not the selective backend.**

## Why this is a roadmap, not a one-shot change

The repo today runs a deliberately *dormant* CacheBlend: it transfers KV, then recomputes
**100%** of every prompt and overwrites the loaded KV, so `prefill_tokens_avoided` is pinned to
`0` at emission (`connector_metrics.py:227`). The selective machinery is a tested **CPU
scaffold** (row-plan invariant, selection policy, KV-write planner, ordering/forward bridges,
fail-closed registrar) with **no live activation seam** — no connector mode, no CUDA kernel, no
`CUSTOM` attention backend, no model override, no `vllm.general_plugins` entry point — and, per
the gap in item 4 above, its selection policy does not yet encode per-layer check-layer
execution. It is gated behind the **M3** numerical-equivalence gate, currently **FAILING** by
~3.6× the baseline envelope (candidate mean-abs-logprob 0.0478 vs envelope 0.0132). Per
`AGENTS.md`, selective recomputation must not be introduced until full-prefill equivalence,
shifted-key correction, hybrid-group handling, and attention-sink tests pass.

### Root-cause lead for M3 (the linchpin)

Loaded, **YaRN-corrected K is copied into vLLM's real paged KV-cache slots**
(`data_plane.py:505-526`, `:603-610`) at `physical_slot_start = block_id*block_size +
block_offset` (`gpt_oss/layout.py:461`). The "overwrite by ordinary prefill" is **implicit** —
nothing re-writes or verifies those slots; it relies on vLLM's prefill `reshape_and_cache`
hitting the *identical* physical slots. Candidate cause: in the hybrid **sliding-window (even)
layers**, vLLM's real `slot_mapping` may not overwrite exactly the slots scatter wrote, leaving
corrected K to contaminate attention broadly. **But this theory is not the first thing to
test** (see A1) — a cheaper control comes first.

## Phase A — Close the M3 numerical gate (immediate blocker)

**A1. GPU diagnostic on solab-g3, cheapest-bisection-first.** Using the moved-document capture
(`scripts/capture_moved_document.py`, runbook `docs/runbooks/solab-g3-moved-document-correctness.md`):

1. **Scatter-disabled control (do this first).** Run the connector attached but with KV scatter
   **disabled** (zero KV writes). This cleanly separates two causes:
   - discrepancy **persists** → connector/allocator/synchronization/numerical **drift** (no
     contamination); the resolution is envelope policy, not a code fix (see A2 / Risks).
   - discrepancy **disappears** → actual **loaded-KV contamination**; proceed to step 2.
2. **Group bisection.** Only if step 1 implicates contamination: test **full-attention-only**
   scatter vs. **sliding-only** scatter to localize the leak to a group.
3. **Per-slot instrumentation (last, most expensive).** Only if a group is implicated: compare
   the corrected-K slots scatter wrote (`gpt_oss/layout.py:428-465`) against vLLM's actual
   prefill `slot_mapping` for that group; check whether any scatter-written slot is left
   un-overwritten.
4. Secondary checks as needed: YaRN FP32→BF16 rounding and the `target−source` delta sign
   (`gpt_oss/torch_yarn.py:174-177`, `:263-292`); source-store side effects on the target
   request; `max_num_batched_tokens` equality with the baseline (config enforces only a lower
   bound, `config_validation.py:445-459`).

**A2. Targeted fix (location depends on A1):**
- **Contamination (overwrite gap):** fix the hybrid slot math / group handling in
  `gpt_oss/layout.py` (`_scatter_group`, `_validated_tables`) and/or add an **explicit
  post-prefill overwrite-verification guard** in the connector's `wait_for_save` /
  `save_kv_layer` path (`connector.py:639-674`) that fails closed if any scatter-written slot
  was not re-written by prefill. Add CPU unit tests alongside `tests/unit/test_gpt_oss_layout.py`
  and `test_vllm_data_plane.py`.
- **Drift (no contamination):** the fix is **envelope policy** — **decided (Risk 1): adopt the
  connector-attached, scatter-disabled run as the prospectively-frozen M3 baseline**, since it
  is the correct one-variable control and equalizes the allocator footprint. Guardrail: if that
  control's drift consumes a large fraction of the tolerance (near the hard ceilings), do **not**
  absorb it — switch to active drift-reduction so the gate keeps discriminating power. Keep the
  tight full-prefill envelope as a secondary reported diagnostic.

**A3. Re-freeze and re-run the v2 gate.** Re-capture 5 controls + 1 candidate and run
`scripts/evaluate_probability_ensemble.py` against a freshly frozen manifest
(`scripts/freeze_probability_ensemble.py`). Pass requires every `Q ≤ U` and all hard ceilings
(`correctness/probability_ensemble.py:35-39`, `:645-693`). Preserve artifacts in a new
create-only dir; **do not** reuse the failed v2 directory.

**Exit criterion:** formal M3 pass (full-prefill equivalence at 100% recompute), artifacts +
digests preserved.

## Phase B — Confirm M4 (YaRN) and M5 (hybrid groups + sinks) on GPU

Run the M4/M5 GPU gates so the four `SelectivePrerequisites` proof flags
(`selective_registry.py:265-273`) — `full_prefill_equivalence`, `transfer_100pct`,
`yarn_correction`, `hybrid_groups_and_sinks` — are all backed by reviewed artifacts, and
produce the five-digest `SelectiveGateEvidence` bundle via
`scripts/hash_selective_gate_artifacts.py` + `scripts/verify_selective_gate_artifacts.py`.

## Phase C — Fix the selective policy, then build the backend (M6)

**C0 (prerequisite, CPU, do before the CUDA work).** Correct the check-layer execution model in
`gpt_oss/selective_policy.py`. Today `select()` replicates one recompute-range set to all 24
layers (`:386-390`) and `check_layer` never shapes the plan. A CacheBlend-style plan needs:
- **full recomputation through the check-layer prefix** (layers `≤ check_layer`);
- importance computed **at** the check layer;
- **selective rows only in subsequent layers** (layers `> check_layer`);
- **layer-token** work accounting (rows × layers), not prompt-token accounting.
This makes `ForwardRowPlan` per-layer-differentiated. Extend CPU tests in
`tests/unit/test_gpt_oss_selective_policy.py` / `test_gpt_oss_selective.py`.

**C1–C6 (backend, currently entirely absent).** Under `src/cacheblend_gpt_oss/vllm_compat/v0_19_1/`
(the registrar's `_PACKAGE_PREFIX`, `selective_registry.py:276-293`):
1. A concrete **`CUSTOM` attention backend + `AttentionImpl`** honoring update-before-attention
   ordering (`gpt_oss/selective_attention.py`); `tests/gpu/test_selective_backend_contract.py`
   pins the stock Triton hook boundary to build against.
2. A **GPT-OSS model override** delegating to `GptOssSelectiveModelAdapter`
   (`gpt_oss/selective_runtime.py:148`).
3. A CUDA **`SelectiveCacheOps`** implementation of the protocol in `selective_kv.py:341-372`.
4. A **new connector mode** `transfer_selective` in `ConnectorTransferMode`
   (`transfer_config.py:69-74`) + config type paralleling `Transfer100PctConfig`, branched into
   the transfer runtime where code switches on `isinstance(..., Transfer100PctConfig)`
   (`connector.py:374,415,439,497,658,686,936`).
5. A **`vllm.general_plugins` entry point** in `pyproject.toml` calling
   `register_selective_extension` (`selective_registry.py:468`).
6. Emit honest savings as **CacheBlend-internal layer-token counters** (e.g.
   `layer_token_rows_recomputed` / `layer_token_rows_avoided`) — see Phase E1. Do **not** relabel
   internally-skipped layer rows as native prompt tokens avoided. A narrowly-pinned vLLM patch
   for non-prefix native accounting is optional and **gated at M6** (feasibility `:767`).

Launch with `--attention-backend CUSTOM` (runbook `docs/runbooks/solab-g3-selective-contract.md:82-96`).

## Phase D — Recompute-ratio sweep + calibration freeze (M7)

Use the corrected `CacheBlendSelectionPolicy` (Phase C0). Drive real GPU runs to produce
importance scores + **layer-token** work/error/latency curves, serialized via the sweep artifact
(`gpt_oss/selective_policy_io.py`, schema `cacheblend_gpt_oss_selection_sweep`). Pick the ratio
on a **predeclared calibration set** (a small held-out set of trajectories, *not* dev-100),
then **freeze** the ratio and thresholds before any final measurement. This is where the
speed/accuracy tradeoff is quantified.

## Phase E — Gates, then the dev-100 arm

**E1. Metrics contract (corrected).** vLLM 0.19.1's connector API credits only one contiguous
prefix, so the out-of-tree connector **cannot** make native vLLM report `external_kv_transfer`
credit or reduced native prefill work (feasibility `:767`; `AGENTS.md` "reports zero externally
computed tokens"). Therefore:
- Native vLLM still reports the **full prompt processed** — this is expected, not a failure.
- The savings signal is the **CacheBlend-internal layer-token metric**
  (`layer_token_rows_avoided` / total layer-token rows), plus the connector stage timers
  (`connector_metrics.py:47-53`) and wall-clock TTFT/prefill latency from native histograms
  (`correctness/capture.py:68,71`).
- Author a **sibling evidence contract**
  `browsecomp_plus_agentic_append_only_transfer_selective` in `benchmark/browsecomp.py` that
  asserts `layer_token_rows_avoided > 0` and internal reconciliation, and that native prompt
  tokens still equal the full prompt (the *opposite* of trying to prove native savings). Reuse
  all non-recompute config/workload gates unchanged; add a `--selective` mode to
  `scripts/validate_browsecomp_append_only.py`.

**E2. Matched-arm experiment (4 arms).** To attribute cost correctly, run:
1. ordinary **no-prefix** Triton (= existing no-prefix data);
2. ordinary **prefix-cached** Triton (= existing prefix data);
3. **CUSTOM backend at 100% recompute, no KV scatter** (isolates custom-backend + staging +
   sync overhead — motivated by the observed ~1095s vs ~600s overhead);
4. **selective CacheBlend CUSTOM** (the feasibility arm).
Serving config for arms 3–4 matches the append-only runbook with `--attention-backend CUSTOM`
and the appropriate connector mode.

**E3. Pre-dev-100 gates (do not skip).** In order: one **synthetic moved-document selective**
correctness test → one fresh **query-703 selective** trajectory → calibration freeze (Phase D).
Only after these pass, run **dev-100** append-only on p7 (`rag-system`, read-only,
`--context-strategy append_only --cache-mode cacheblend`), validating each trajectory with the
selective contract and capturing all four arms' metrics.

**E4. Accuracy/recall.** Not in this repo — scoring lives on the rag-system/p7 side
(`analysis/dev100`). Run p7's grader on the CacheBlend arm and compare to the no-prefix / prefix
arms. This repo can only certify layer-token/work reconciliation + the boolean
`final_answer_validation.valid`.

**E5. Compare arms.** Fold the four arms into the controlled-benchmark schema
(`benchmark/models.py`: `full_prefill`, `vllm_prefix_*`, `cacheblend_100pct`,
`cacheblend_selective`) with mean/median/95%-CI over every timer/counter (`:513-701`). Joining
the single-prompt schema to real dev-100 trajectories is a small adapter.

## Verification

- **CPU (authoring workstation), after every code change:** `.venv/bin/python -m pytest -m
  "not gpu"` (currently 842 pass), `ruff check src tests scripts`, `mypy src --strict` (green).
  Add unit tests for: the Phase-A overwrite guard/slot-math fix, the Phase-C0 per-layer
  check-layer plan, the selective connector mode, and the selective evidence contract.
- **M3 (g3):** `scripts/evaluate_probability_ensemble.py` passes; transfer evidence digest binds.
- **M6 (g3):** `--attention-backend CUSTOM` server starts, compatibility-probe digests match,
  `tests/gpu/test_selective_backend_contract.py` and selective GPU gates pass.
- **E2E (g3+p7):** all four arms complete; selective validator passes with
  `layer_token_rows_avoided > 0` while native prompt tokens remain the full prompt; the
  100%-no-scatter control quantifies backend overhead; p7 grader yields accuracy/recall within
  the accepted tolerance vs. the prefix arm.

## Key risks / decision points

1. **M3 may be reduction-order drift, not a leak** (settled early by the A1 scatter-disabled
   control). **Decided:** adopt the connector-attached, scatter-disabled run as the
   prospectively-frozen baseline (correct one-variable control); switch to active drift-reduction
   only if that control's drift consumes a large fraction of the tolerance. Enabler: the
   KV-scatter-disabled diagnostic mode (first implementation task).
2. **Out-of-tree metrics ceiling.** Native vLLM cannot show non-prefix savings; the honest
   signal is internal layer-token accounting. A vLLM patch is an M6-gated option only. Do not
   present internal skips as native tokens avoided.
3. **Append-only is CacheBlend's weakest case.** Prefix caching is exact and free on the
   byte-identical growing prefix; the selective arm's unique savings come only from
   re-retrieved / moved documents (`--no-deduplicate-retrieved-documents`, already set), and the
   store admits only complete prefix-aligned 256-token chunks (embedded-doc persistence
   deferred). Expect a modest, possibly near-break-even win; treat as the stress-test baseline.
4. **Error compounding across turns.** Selective approximation feeds the next turn's context;
   the frozen ratio (Phase D) must bound *accumulated* error, not just per-forward.
5. **Backend overhead can masquerade as CacheBlend cost/gain** — the 100%-no-scatter control arm
   (E2 arm 3) exists to isolate it. The current ~1095s vs ~600s gap shows this matters.
6. **Effort asymmetry.** Phase A (and the envelope decision) is the gate; Phase C0 is a required
   CPU fix; Phase C1–C6 is substantial vLLM-internal + CUDA-kernel work.
7. **O(N²) re-store — prerequisite for any append-only speed result (found 2026-08-14).**
   `transfer_runtime.py:_build_store_plan` (`:801-826`) re-gathers/re-transfers/re-hashes **every
   complete chunk of the whole prompt on every request** (`complete_store_token_count`), so on
   append-only the per-turn store work grows with accumulated context → cumulative O(turns²),
   synchronous, single-threaded (`--max-workers 1`). Evidence: query-703 100%-recompute run took
   **8252s vs ~600s baseline (~13.75×)** on ~1.4× the turns — the store tax, not compute or the
   extra turns, dominates. Fix in flight: **delta-store** (store only complete chunks not already
   in the sidecar, by exact-token identity). This would otherwise swamp any selective compute
   savings and make CacheBlend lose on append-only regardless of recompute ratio.
8. **Load-side re-fetch (deeper item, not yet scoped).** With prefix caching off, prefix KV is
   not resident across turns, so the load path likely re-retrieves the growing prefix each turn —
   also O(N²). This is entangled with the prefix-caching-off constraint; the principled fix is
   the prefix-cache-for-prefix + CacheBlend-for-moved-docs architecture, a larger design change.
9. **Trajectory divergence invalidates naive arm comparison.** The failing M3 numerics changed
   the agent's decisions (43 search calls vs the baseline's 32 on query 703), so the CacheBlend
   arm did *more work* than the baseline. Arms must run matched trajectories to be comparable —
   another reason M3 comes before any speed claim.
