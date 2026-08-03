# 66 — Bisect the TP=4 quality divergence across the #49–#61 execution-engine chain

Status: closed

## Parent

`.scratch/rocm-tensor-parallel/issues/63-tp4-quality-fixture-post-59.md`
`.scratch/rocm-tensor-parallel/issues/55-tp4-throughput-quality-revalidation.md`

## What to build

Split out of `#63` on human disposition 2026-08-03. #63's measurement mandate
is complete (arena load regression root-caused and fixed, NaN verified fixed,
quality number captured at avg_nll=13.10). The quality divergence itself is a
pre-existing regression from the #49–#61 execution-engine chain, not a #63
discovery — it belongs in its own investigation issue.

**The evidence:** At issue #48 (2026-07-31), TP=4 `score_official` avg_nll was
**0.377** — comfortably inside the PRD bar (0.370–0.378), with 68/100
first-token matches. Today (2026-08-03), on a NaN-free, clean-loading build,
case_000 scores avg_nll **13.10** (35× the bar), first_match=0/24,
api_top1_rate=0.0. This is not noise, not NaN — it's a systematic logit shift
(target_mean_delta ≈ −13 across all 129280 logits). The regression window is
the #49–#61 chain:

| Commit | Issue | What |
|---|---|---|
| `fa59d97` (prime suspect) | #49 | Attention-kernel replacement (spike→new kernel path) |
| … | #50–#54 | Persistent-thread engine, sync/dispatch restructuring |
| `1fe4829` | #60 | 43-layer persistent-thread rollout |
| `0cb9cf3` | #61 | Async all-reduce (eliminates host sync barriers) |

At `fe2900b` (08-02 baseline, pre-#55), the quality was **16.32** (NaN
present, but even the finite logits are 44× off). At `6f3171f` (HEAD with #55
NaN fix), with the arena load regression reverted (#63's fix), quality is
**13.10**. NaN elimination improved it from 16→13 but didn't close the 35×
gap — the core divergence is in the compute graph, not a NaN cascade.

**Router-broadcast negative result:** A WIP router-state broadcast
(`ds4_gpu_tensor_copy_xdev` for `router_selected_by_tier` /
`router_weights_by_tier` / `ffn_norm_by_tier` from tier 0 to tiers 1–3) was
tested and shifted avg_nll from 15.32→12.86 in smoke — not material, not the
fix. Preserved as git stash `stash@{0}` (2026-08-03).

**VRAM constraint:** The full 100-case fixture currently crashes on case_001
(`routed_moe x quantize launch failed`, free VRAM 0.42 GiB). This is the #65
structural-VRAM-headroom problem. The bisect can still proceed on case_000
smoke (single case, ~30s), since the quality signal is categorically
unambiguous (13 vs 0.37) and doesn't need 100 cases to detect.

## Acceptance criteria

- [x] Bisect across the #49–#61 commit range identifies the specific commit
      that regressed TP=4 quality from ~0.377 (in-bar) to ~13+ (35× bar) —
      **the persistent-thread full-token execution engine, introduced #50
      (`4a21216`) and made default-on by #53 (`04ec7be`), is the regression.
      Measured case_000 avg_nll: anchor `06fcf82` 0.407 (good), `04ec7be`
      with `DS4_TP4_THREADED_LAYERS=0` 0.442 (good), `04ec7be` default
      16.16 (bad), `1fe4829` 16.78 (bad), `0cb9cf3` 15.98 (bad), HEAD
      default 13.24 (bad), HEAD `DS4_TP4_THREADED_LAYERS=0` 0.398 (good).**
- [x] Root cause confirmed by reverting or patching that commit and verifying
      case_000 avg_nll returns to the ~0.37–0.76 range (or better) —
      **root cause is a cross-layer data race in the threaded engine: each
      rank's all-reduce reads peer partials from single per-tier buffers
      (`attn_out_by_tier`/`shared_out_by_tier`) that every rank overwrites
      once per layer, and the async overlap lets a fast rank overwrite a
      lagging rank's still-being-read buffer. Fix = default
      `metal_graph_tp4_spike_layer_enabled` back to 0 (legacy sequential
      path) when `DS4_TP4_THREADED_LAYERS` is unset. Case_000 avg_nll on
      the fix: 0.398 (in bar).**
- [x] `make -j8 test-rocm` passes on the fix branch — **exit 0, all 4
      targets (`test_rocm_tp_stubs`, `test_rocm_xdev` incl. async-stream
      event tests, `test_rocm_kernel_compare` 6/6,
      `test_engine_rocm_tp_refusal`). `make -j8 rocm` also clean.**
- [x] Pipeline quality verified not regressed (case_000 smoke at ~0.369;
      or, if the bisect implicates a commit that also affects the pipeline
      path, documented and dispositioned) — **pipeline case_000 avg_nll
      0.383 on the fix branch; the threaded path is TP=4-only and never
      touched the pipeline.**
- [x] Findings recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`
- [x] Once fixed, #55 AC4 (full 100-case TP=4 quality fixture) is unblocked
      and can proceed — **TP=4 quality is back in-bar (case_000 0.398); the
      full 100-case fixture itself still needs #65's structural-VRAM-headroom
      work to get past case_001 (`routed_moe x quantize launch failed`,
      free VRAM 0.42 GiB), which is unrelated to this fix.**

## Comments

### 2026-08-03 Bisect result and fix (closed)

Full bisect table, root-cause analysis, and verification are recorded in
`.scratch/rocm-tensor-parallel/experiment-log.md` (2026-08-03 issue 66 entry)
and in the artifacts under `.scratch/rocm-tensor-parallel/quality-out/66-bisect/`.

**Bisect:** The regression is not `fa59d97` (that was the #46/#47/#48
regression, already fixed and re-validated to 0.377 at #48). The #49–#61
window's regression is the persistent-thread full-token execution engine,
introduced in #50 (`4a21216`, opt-in) and enabled for all 43 layers by #53
(`04ec7be`). Every threaded-default commit measures case_000 avg_nll 13–17;
every legacy-path commit measures 0.40–0.44, including the good anchor `06fcf82`
and HEAD with `DS4_TP4_THREADED_LAYERS=0`. All commits share an identical load
state (same arena OOM + 40 q8-budget warnings), ruling out the VRAM/load path
as the cause.

**Root cause:** Cross-layer data race on the per-tier peer-partial buffers.
The threaded all-reduce reads `attn_out_by_tier[peer]` / `shared_out_by_tier[peer]`
through `ds4_rocm_xdev_copy`, whose event fence only orders each copy against
the peer's stream state at *issue* time — it cannot prevent the peer from
overwriting that single reused buffer with the next layer's partial before the
lagging rank's copy has read it. Result is a systematic logit shift
(target_mean_delta ≈ −13 across all 129280 logits) with run-to-run variance
(15.8 vs 16.2 on the identical 04ec7be binary). The legacy path cannot race
because it `sync_all_devices` + blocking all-reduces before any rank starts the
next layer.

**Fix:** `metal_graph_tp4_spike_layer_enabled` now defaults to `0` (legacy
sequential TP=4 path) when `DS4_TP4_THREADED_LAYERS` is unset. The threaded
path stays reachable via `DS4_TP4_THREADED_LAYERS=N` as a diagnostic switch.
Properly fixing the race (per-layer double-buffering of the peer-partial
tensors, or a per-layer handshake) is left as follow-up work; it is not a
one-line patch and does not belong in a bisect issue.

**Verification on the fix branch:**
- case_000 TP=4 avg_nll **0.398** (in bar), api_top1_rate 0.833, api_pair_rate 0.947.
- 5-case run crashes on case_001 with the known #65 VRAM-headroom issue
  (unrelated to this fix).
- Pipeline case_000 avg_nll **0.383** (not regressed).
- `make -j8 test-rocm` exit 0; `make -j8 rocm` clean.

**#55 AC4:** unblocked on quality. The full 100-case TP=4 fixture can proceed
once #65's structural-VRAM-headroom fix allows case_001+ to load.


## Blocked by

*(Nothing blocks this issue's machine dispatch: the bisect runs on case_000
smoke, which needs no VRAM headroom, so it starts immediately. Issue #65's
structural-VRAM-headroom work gates only the *full* 100-case fixture (and
#55 AC4) — it is ordered after this issue, not before it. Keep this section
free of `.scratch/…` path references, which the ralph parser reads as hard
machine blockers.)*
