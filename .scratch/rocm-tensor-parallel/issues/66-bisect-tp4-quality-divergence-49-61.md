# 66 — Bisect the TP=4 quality divergence across the #49–#61 execution-engine chain

Status: ready-for-agent

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

- [ ] Bisect across the #49–#61 commit range identifies the specific commit
      that regressed TP=4 quality from ~0.377 (in-bar) to ~13+ (35× bar)
- [ ] Root cause confirmed by reverting or patching that commit and verifying
      case_000 avg_nll returns to the ~0.37–0.76 range (or better)
- [ ] `make -j8 test-rocm` passes on the fix branch
- [ ] Pipeline quality verified not regressed (case_000 smoke at ~0.369;
      or, if the bisect implicates a commit that also affects the pipeline
      path, documented and dispositioned)
- [ ] Findings recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`
- [ ] Once fixed, #55 AC4 (full 100-case TP=4 quality fixture) is unblocked
      and can proceed

## Blocked by

`.scratch/rocm-tensor-parallel/issues/65-tp4-arena-structural-vram-headroom.md`

*(#65 blocks the *full* 100-case run, not the single-case bisect smoke;
the bisect itself can proceed immediately on case_000 since the signal is
categorically unambiguous at 35× bar.)*
