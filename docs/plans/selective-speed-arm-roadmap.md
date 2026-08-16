# Selective-recompute feasibility arm roadmap (append-only dev-100)

> Status: **implementation in progress.** M3's connector-inclusive probability-v2 gate is
> green on solab-g3; C0 is CPU-complete; the pinned CUSTOM backend, model wrapper, and
> explicit `transfer_selective` row-plan path are implemented, with GPU verification next.
> Staged roadmap, authored 2026-08-14, revised 2026-08-15 after the g3 gate.
> It records decisions and the milestone dependency
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
4. **Selective policy gap closed on CPU:** `check_layer` now forces full recomputation through
   the check-layer prefix and applies selective ranges only to later layers. The plan reports
   layer-token work; the initial live model override now consumes that plan under
   the explicit selective opt-in.
5. **Matched-backend control arm added** (CUSTOM backend at 100% recompute, no scatter).
6. **Calibration hygiene added:** a predeclared calibration set picks and freezes the recompute
   ratio; dev-100 is never used to choose it.

## Critical path (milestone dependency chain)

```
M3 numerical equivalence (PASS, connector-inclusive v2 envelope)  ──►  M4 YaRN + M5 hybrid/sinks (GPU evidence review)
        │                                          │
        │                          selective policy check-layer fix (CPU, prerequisite)
        │                                          │
        └──────────────────────────────►  M6 selective backend (implemented; GPU pending)
                                                   │
                                          M7 recompute-ratio sweep + calibration freeze
                                                   │
              matched-backend + one-query gates  ──►  dev-100 arm  ──►  accuracy/recall on p7
```

M3 is now green under the connector-inclusive v2 policy. The matched CUSTOM backend control
and selective serving seams are implemented. The immediate next work is one synthetic g3
selective smoke/correctness run using measured check-layer scores, followed by the ratio sweep
and matched BrowseComp gates.

## Why this is a roadmap, not a one-shot change

The validated milestone still runs a deliberately conservative *100%* path: it transfers KV,
then recomputes **100%** of every prompt and overwrites the loaded KV, so
`prefill_tokens_avoided` remains pinned to `0` at emission (`connector_metrics.py:227`).
Alongside it, the selective arm now has a tested CPU row plan, an explicit
`transfer_selective` configuration, a pinned vLLM 0.19.1 CUSTOM attention backend, and a lazy
GPT-OSS model wrapper that skips MLP work for cached rows after the check layer. The selective
arm is still opt-in, measures loaded-versus-fresh value differences at the check layer, and has
**no GPU correctness or speed claim** until g3 evidence is captured.

### Root-cause lead for M3 (the linchpin)

Loaded, **YaRN-corrected K is copied into vLLM's real paged KV-cache slots**
(`data_plane.py:505-526`, `:603-610`) at `physical_slot_start = block_id*block_size +
block_offset` (`gpt_oss/layout.py:461`). The "overwrite by ordinary prefill" is **implicit** —
nothing re-writes or verifies those slots; it relies on vLLM's prefill `reshape_and_cache`
hitting the *identical* physical slots. Candidate cause: in the hybrid **sliding-window (even)
layers**, vLLM's real `slot_mapping` may not overwrite exactly the slots scatter wrote, leaving
corrected K to contaminate attention broadly. **But this theory is not the first thing to
test** (see A1) — a cheaper control comes first.

## Phase A — Close the M3 numerical gate (completed on g3)

The connector-attached scatter-disabled controls and the real 100%-transfer candidate were
captured on the pinned A100 environment. The prospective probability-v2 envelope reproduced
with `stable: true`, `status: PASS`, and the candidate evaluation returned `EVAL_STATUS=0`.
This clears the numerical gate for the selective implementation; it does not claim selective
compute or a BrowseComp quality result.

**A1. GPU diagnostic on solab-g3, cheapest-bisection-first (historical record).** Using the moved-document capture
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

**A2. Targeted fix (location depends on A1, not required after the observed drift result):**
- **Contamination (overwrite gap):** fix the hybrid slot math / group handling in
  `gpt_oss/layout.py` (`_scatter_group`, `_validated_tables`) and/or add an **explicit
  post-prefill overwrite-verification guard** in the connector's `wait_for_save` /
  `save_kv_layer` path (`connector.py:639-674`) that fails closed if any scatter-written slot
  was not re-written by prefill. Add CPU unit tests alongside `tests/unit/test_gpt_oss_layout.py`
  and `test_vllm_data_plane.py`.
- **Drift (no contamination):** the fix is **envelope policy** — **decided (Risk 1): adopt the
  connector-attached, scatter-disabled run as the prospectively-frozen M3 baseline**, since it
  is the correct one-variable control and equalizes the allocator footprint. Guardrail: if that
  control's drift consumes a large fraction of the mean/TV/JS tolerances (near the hard
  ceilings), do **not** absorb it — switch to active drift-reduction so the gate keeps
  discriminating power. Keep the high-mass maximum and tight full-prefill envelope as
  secondary reported diagnostics.

**A3. Re-freeze and re-run the versioned probability gate (completed).** Re-capture 5 controls + 1
candidate and run `scripts/evaluate_probability_ensemble.py` against a freshly frozen
manifest (`scripts/freeze_probability_ensemble.py`). Pass requires every `Q ≤ U` for
full-vocabulary mean, TV, and JS, all three hard ceilings, agreement checks, and bound
transfer evidence. The high-mass maximum remains a serialized diagnostic because the
connector-attached controls themselves were unstable on that maximum. Preserve artifacts
in a new create-only dir; **do not** reuse the failed v2 directory.

**Exit criterion:** formal M3 pass (full-prefill equivalence at 100% recompute), artifacts +
digests preserved.

## Phase B — Confirm M4 (YaRN) and M5 (hybrid groups + sinks) on GPU

Run the M4/M5 GPU gates so the four `SelectivePrerequisites` proof flags
(`selective_registry.py:265-273`) — `full_prefill_equivalence`, `transfer_100pct`,
`yarn_correction`, `hybrid_groups_and_sinks` — are all backed by reviewed artifacts, and
produce the five-digest `SelectiveGateEvidence` bundle via
`scripts/hash_selective_gate_artifacts.py` + `scripts/verify_selective_gate_artifacts.py`.

## Phase C — Fix the selective policy, then build the backend (M6)

**C0 (completed, CPU).** Corrected the check-layer execution model in
`gpt_oss/selective_policy.py`. A CacheBlend-style plan now provides:
- **full recomputation through the check-layer prefix** (layers `≤ check_layer`);
- importance computed **at** the check layer;
- **selective rows only in subsequent layers** (layers `> check_layer`);
- **layer-token** work accounting (rows × layers), not prompt-token accounting.
This makes `ForwardRowPlan` per-layer-differentiated. Extend CPU tests in
`tests/unit/test_gpt_oss_selective_policy.py` / `test_gpt_oss_selective.py`.

**C1 (initial seam implemented, GPU verification pending).** Under
`src/cacheblend_gpt_oss/vllm_compat/v0_19_1/`, `selective_backend.py` subclasses the pinned
sink-capable Triton backend. It preserves stock full-row writes without an active plan and has
a fail-closed, plan-aware selected-row write hook. `selective_plugin.py` registers it only for
the explicit CUSTOM opt-in; the matched control helper is
`local-m6-custom-backend-control.sh`. This is an API/control milestone, not a speed milestone.

**C2 (initial model seam implemented, GPU activation pending).**
`selective_model.py` provides a lazy subclass of the pinned GPT-OSS model that
binds the exact forward signature. It preserves full execution when no selective
plan is installed and, for `transfer_selective`, keeps full-shaped attention while
skipping MLP work for cached rows after the check layer. It is opt-in through
`CACHEBLEND_ENABLE_CUSTOM_MODEL=1`; this is an implementation seam, not a GPU
correctness or speed result.

**C3–C6 (initial serving path implemented; GPU validation pending).** Under
`src/cacheblend_gpt_oss/vllm_compat/v0_19_1/` (the registrar's `_PACKAGE_PREFIX`,
`selective_registry.py:276-293`):
1. The existing pinned CUDA data plane remains the KV transport boundary; the CUSTOM
   backend narrows the selected KV-cache write rows.
2. The **`transfer_selective`** connector mode and strict config are implemented in
   `transfer_config.py`, and the runtime emits a full-shaped row plan.
3. The explicit plugin opt-ins register the CUSTOM backend and GPT-OSS model wrapper.
4. Emit honest savings as **CacheBlend-internal layer-token counters** (e.g.
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
  "not gpu"` (currently 872 pass), `ruff check src tests scripts`, `mypy src --strict` (green).
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

1. **M3 is reduction-order DRIFT, not a leak — RESOLVED on g3.** The scatter-disabled control
   retained essentially all the discrepancy (max 0.082 vs 0.089 normal; mean 0.0103 vs 0.0104;
   all sampled/top tokens agree), so reused KV is NOT contaminating output — no KV-overwrite bug,
   no code fix. Drift lives in negligible-probability tail tokens (mean within the old envelope;
   only full-vocab max elevated), so it does not destroy the v2 probability-gate's discriminating
   power → **disciplined Option 1 adopted, drift-reduction not needed.** Path forward: capture ~5
   connector-attached scatter-disabled controls, freeze a connector-inclusive v2 envelope
   prospectively, capture one real candidate, evaluate. This validates the *reuse mechanism*
   against an envelope that accepts connector-presence drift — a legitimate but slightly weaker
   claim than bit-clean equivalence; label accordingly.
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
   **Status:** delta-store implemented + committed (`_build_store_plan` skips leading chunks the
   load-side lookup already exact-token-verified as present; fail-closed, so it can never drop KV
   or wrong-match — worst case a no-op). **Efficacy unverified on CPU:** LMCache hashes the passed
   tokens rooted at relative-0 (`lmcache_v0_4_3.py:537`), so whether delta-stored tail chunks are
   re-found by later full-prefix lookups depends on the server-side CB matcher's rooting, which
   only g3 can settle. **Required g3 validation:** run ~3 append-only turns and confirm per-turn
   store volume (`cacheblend_store_tokens_completed` delta) stays small/flat instead of growing
   with turn index. If it still grows, delta chunks aren't being re-found → redesign to preserve
   the full-prefix hash chain for the stored tail.
8. **Load-side re-fetch (deeper item, not yet scoped).** With prefix caching off, prefix KV is
   not resident across turns, so the load path likely re-retrieves the growing prefix each turn —
   also O(N²). This is entangled with the prefix-caching-off constraint; the principled fix is
   the prefix-cache-for-prefix + CacheBlend-for-moved-docs architecture, a larger design change.
9. **Trajectory divergence invalidates naive arm comparison.** The failing M3 numerics changed
   the agent's decisions (43 search calls vs the baseline's 32 on query 703), so the CacheBlend
   arm did *more work* than the baseline. Arms must run matched trajectories to be comparable —
   another reason M3 comes before any speed claim.
