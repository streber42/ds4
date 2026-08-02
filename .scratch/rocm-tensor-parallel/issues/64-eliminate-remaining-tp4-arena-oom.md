# 64 — Eliminate the remaining ROCm TP=4 model-arena OOM and its permanent-latch cascade

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/issues/55-tp4-throughput-quality-revalidation.md`

## What to build

`#59` implemented a bounded per-device reusable buffer for the batch-prefill
MoE gate/up/down fallback (`cuda_model_prefill_fallback_ptr`,
`rocm/ds4_rocm_runtime.cuh`), which eliminated the specific `moe_down` arena
OOM that used to fire before any layer completed. That measurably worked: TP=4
now traverses the full 43-layer model during prefill. But one `arena alloc
failed` still fires — now for a different, smaller tenant (`q8_0`, at offset
80.24 GiB of an 80.76 GiB production model, i.e. essentially the last tensor)
— and it still degrades the rest of the session, not just that one request.

Two things need fixing here, and they're separable:

1. **Structural headroom.** After the static 25.94 GiB/tier selective-weight
   slab + 2.11 GiB per-tier graph overhead, live free VRAM sits around ~0.9
   GiB (`ds4: ROCm q8 fp16 cache budget exhausted ... free=0.90 GiB` lines,
   consistent throughout a run). That's thin enough that essentially any
   single sub-1 GiB allocation request from any of the shared arena's many
   tenants (`compressor_ape`, `rms_weight`, `q8_0`, `f16`, `f16_pair0/1`, and
   others — see `#59`'s audit for the full tenant list) can be the one that
   loses the race, and which specific tensor trips it appears to depend on
   allocation order/fragmentation, not a fixed offender.
2. **The `g_model_cache_full` permanent latch.** `rocm/ds4_rocm_runtime.cuh:5749`
   checks it, `:5785` sets it (on the *first* `cudaMalloc` failure in
   `cuda_model_arena_alloc`), and nothing clears it short of full model
   teardown (`cuda_model_range_release_ranges_only`, `:6037`). One transient
   near-the-margin allocation failure therefore converts into a session-long
   cascade of `arena-full skip` → `host-register PCIe-map` fallbacks for
   *every* subsequent request from *every* tenant on *every* device, not just
   the one that failed — `#59`'s live run logged 1601 such skips off a single
   trigger. Even if (1) is fully solved, a single future transient failure
   (fragmentation, a slightly larger model, a slightly smaller GPU) would
   reproduce the entire cascade again from whatever tenant hits it next.

Fix candidates carried forward from `#59`'s audit (both still apply, not
mutually exclusive):

- Extend `#59`'s bounded-reusable-buffer pattern to other large, single-use,
  non-persistent-across-calls arena tenants, if any exist beyond
  moe_gate/up/down — needs the "audit every other caller" work `#59`
  explicitly deferred (confirm which of `compressor_ape`, `rms_weight`,
  `q8_0`, `f16`, `f16_pair0/1` are actually safe to bound this way vs. which
  have genuine cross-call reuse that a single-slot cache would thrash).
- Make `g_model_cache_full` recoverable instead of a permanent latch — e.g.
  retry once per request instead of latching globally, or scope the latch
  to "this arena chunk size class is full" rather than "no arena allocation
  will ever succeed again this session."
- Build a true 4-way TP-aware batched owned-expert prefill kernel (the TP=2
  CUDA-style path already has one, `ds4_gpu_routed_moe_batch_owned_tensor`,
  `ds4.c:65102/65130`) to remove the need for the full-256-expert home-tier
  fallback entirely — `ds4.c:30855-30860` documents a four-tier version of
  this was already tried and abandoned for FP-accumulation noise, so this
  means re-solving that, not avoiding it.

## Acceptance criteria

- [ ] Live run on the production model (`ds4 --rocm --gpu-devices 0,1,2,3
      --cuda-tensor-parallel`) produces zero `ds4: ROCm model arena alloc
      failed` warnings, not just a later/smaller one
- [ ] `DS4_ROCM_WEIGHT_PATH_STATS=1` shows no `arena-full skip` cascade
      following from a single failure (i.e. `g_model_cache_full`'s
      all-or-nothing latch behavior is either gone or provably not hit)
- [ ] `make -j8 test-rocm` passes
- [ ] Findings recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`

## Blocked by

*(nothing)*

## Comments

**2026-08-02 — Opened on `#59`'s disposition.** `#59` closed with its
candidate-1 fix landed and measured (moved the `arena alloc failed` failure
point from `moe_down`/layer-0 to `q8_0`/last-tensor, bounded the MoE
fallback's own VRAM growth), but AC3 ("verification run confirms `arena
alloc failed` warnings are gone") was not literally satisfied — one failure
remains, and the permanent-latch design means one failure is functionally
as bad as many. See `#59`'s final comment and the 2026-08-02 experiment-log
entry ("#59 candidate-1 fix") for the full before/after numbers and the
exact call sites involved.

`#58` and `#63` were both `Blocked by #59`; re-pointed to block on this
issue instead, since they need a TP=4 config that initializes without any
arena failure, which `#59` got much closer to but did not fully deliver.
