# 43 — Pipeline VRAM-accounting regression invalidates reference baseline

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/issues/32-tp4-quality-fixture.md`
`.scratch/rocm-tensor-parallel/issues/41-host-mapped-moe-weight-precision.md` (discovered during
this investigation)

## Problem

The pipeline (non-TP) reference baseline used throughout this project —
`q_pipeline_ref_tp4issue32.tsv`, avg_nll=0.374733, first_match=65/100,
captured 2026-07-29 07:37 — is **no longer reproducible on the current
tree**. Re-running the same cases in pipeline mode
(`./gguf-tools/quality-testing/score_official ... --gpu-devices 0,1,2,3`,
no `--cuda-tensor-parallel`) now gives avg_nll ≈ 1.56, essentially
indistinguishable from TP=4's broken score.

This was discovered while investigating issue #41 (host-mapped MoE weight
fallback). The instrumented run showed pipeline mode hitting the same
arena-exhaustion / host-register-PCIe-fallback signature that TP=4 hits
(see #41 for the full mechanism): ~987 weight lookups per 5-case run
fall through to PCIe-mapped host reads instead of the pre-loaded VRAM
cache, starting at layer 0.

## Root cause

Commit `414f9fc` ("unblock TP=4 quality fixture with memory accounting
fixes", 2026-07-29 16:21 — *after* the reference baseline was captured)
changed `engine_per_tier_graph_overhead_bytes` (ds4.c, "Fix 2") to gate
the batch-scratch line items (`batch_cur_hc_by_tier`,
`batch_next_hc_by_tier`, ... `batch_ffn_out_by_tier`, ~30 buffers) behind
`e->cuda_tensor_parallel`:

```c
if (e && e->cuda_tensor_parallel) {
    total += pc * hc_dim * sizeof(float);   /* batch_cur_hc_by_tier */
    ...
}
```

Intent (per the commit message): "Pipeline path no longer over-accounts
scratch on all 4 tiers, fixing OOM at layer 40." The problem is that the
*actual* allocation code, `metal_graph_alloc_raw_cap` (ds4.c ~17540),
allocates these same `batch_*_by_tier` buffers for every `used_tier`
**unconditionally** — there is no `cuda_tensor_parallel` gate on that
loop. So in pipeline mode the two sides of the budget now disagree:

- **Budget calculation** (`engine_per_tier_graph_overhead_bytes`):
  assumes 0 bytes of batch scratch needed → reserves less headroom →
  packs the primary selective weight cache closer to each GPU's VRAM
  limit.
- **Actual allocation** (`metal_graph_alloc_raw_cap`): still allocates
  the full batch-scratch set on every used tier.

The result: pipeline mode's primary weight cache now consumes nearly all
available VRAM (observed: GPU0 28.8/29.7 GiB with layers 0-14 +
embedding), leaving no headroom for `ds4_gpu_set_use_host_weights(1)`'s
arena mechanism (see #41) during batch prefill. The arena's first
`cudaMalloc` fails almost immediately (layer 0), and the
`g_model_cache_full` latch (global, not per-device) permanently routes
the rest of that run's weight lookups through PCIe host-register reads —
the same quality-destroying fallback TP=4 suffers from.

At larger context (ctx=4096, the quality-fixture default) the effect is
worse: session creation itself can fail
(`alloc_raw_cap check failed: class_p_ok=0`, missing `batch_*_by_tier`
fields) because the batch-scratch buffers don't fit at all once the
under-reserved weight cache has already claimed the VRAM.

**Severity update:** this is not limited to `score_official`'s ctx=4096.
Plain `./ds4 --rocm --gpu-devices 0,1,2,3 --model ... -p "..." -n 12`
(default ctx=32768, no `--cuda-tensor-parallel`) fails outright —
`ds4 debug: alloc_raw_cap check failed: state_init_ok=1 layer_cache_ok=1
class_p_ok=0 output_tp_ok=1` followed by `ds4: sampled CLI generation
requires a session backend` — no output at all. **Pipeline mode is
currently non-functional on this 4×R9700 host at default settings**,
which also means the PRD's mandatory fallback guarantee ("If tensor
parallelism is unavailable, refused, or fails to initialise, the engine
falls back to the existing pipeline layer-split path... the user must
never end up with no working inference because of this feature") does
not currently hold: TP=4 initializes but produces garbled output (the
quality gap), and pipeline — the fallback — fails to start at all. This
raises the priority of this issue above a simple baseline-regeneration
task.

## What to build

Reconcile the two sides of the pipeline VRAM budget. Two options:

### Option A: Un-gate the budget calculation (safest, matches actual allocation)

Remove the `e->cuda_tensor_parallel` gate around the batch-scratch total
in `engine_per_tier_graph_overhead_bytes`, restoring it to always count
those buffers (as it did before `414f9fc`). This makes the budget
pessimistic-but-correct for pipeline mode again. Re-check whether this
reintroduces the "OOM at layer 40" that Fix 2 was originally trying to
solve — if so, the real fix is elsewhere (e.g. only allocate
`batch_*_by_tier` for tiers that actually run batch prefill in pipeline
mode, rather than every `used_tier`), not in skipping the reservation.

### Option B: Gate the allocation to match the budget

Make `metal_graph_alloc_raw_cap`'s batch-scratch loop skip tiers when
`!cuda_tensor_parallel` in the same way the budget calculation now does
— but only if pipeline mode genuinely never uses `batch_*_by_tier` for
tiers other than the one running batch prefill. This needs verification;
if pipeline batch prefill touches per-tier batch buffers for reasons
unrelated to TP (e.g. shared workspace plumbing), this option is unsafe.

**Whichever option is chosen, the deliverable is a regenerated pipeline
reference TSV that reproduces (or explains any legitimate delta from)
the original avg_nll=0.374733 / first_match=65/100 baseline**, since that
number is the ground truth every TP=4 quality comparison in this project
depends on.

## Acceptance criteria

- [ ] Root cause reconciled (Option A, B, or a third approach) with reasoning recorded
- [ ] Pipeline mode (`--gpu-devices 0,1,2,3`, no `--cuda-tensor-parallel`) quality fixture reproduces avg_nll ≈ 0.375, first_match ≈ 65/100 (matching or explaining any delta from the original baseline)
- [ ] The original "OOM at layer 40" problem Fix 2 solved is confirmed still solved (or re-fixed correctly)
- [ ] Fresh pipeline reference TSV saved to `.scratch/rocm-tensor-parallel/quality-out/`
- [ ] `make -j8 test-rocm` passes
- [ ] Cross-reference noted in issue #42 (TP=4 VRAM budget), since both issues touch the same `engine_per_tier_graph_overhead_bytes` accounting

## Blocked by

- 4× R9700 GPU access
- Existing quality fixture infrastructure

## References

- [Issue #41 — Host-mapped MoE weight numerical impact](41-host-mapped-moe-weight-precision.md) — discovered this regression while instrumenting the weight-resolution fallback path; see its "Comments" section for the live trace data (`DS4_ROCM_WEIGHT_PATH_STATS=1`) that surfaced it.
- [Issue #42 — Free VRAM budget for TP=4](42-tp4-vram-budget.md) — same accounting function, TP=4 side of the same underlying problem.
- Commit `414f9fc` — introduced the accounting mismatch.
