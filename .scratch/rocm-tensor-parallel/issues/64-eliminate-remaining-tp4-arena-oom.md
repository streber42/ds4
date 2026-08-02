# 64 — Eliminate the remaining ROCm TP=4 model-arena OOM and its permanent-latch cascade

Status: ready-for-human

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

- [x] Live run on the production model (`ds4 --rocm --gpu-devices 0,1,2,3
      --cuda-tensor-parallel`) produces zero `ds4: ROCm model arena alloc
      failed` warnings, not just a later/smaller one
- [ ] `DS4_ROCM_WEIGHT_PATH_STATS=1` shows no `arena-full skip` cascade
      following from a single failure (i.e. `g_model_cache_full`'s
      all-or-nothing latch behavior is either gone or provably not hit) —
      **the latch is gone (parenthetical satisfied), but the log still shows
      ~1038 `arena-full skip` lines** (down from baseline's 1438, same
      shape); see Comments for why this is left unchecked pending a human
      call on whether that clears the bar
- [x] `make -j8 test-rocm` passes
- [x] Findings recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`

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

**2026-08-02 — Latch removed and shipped; leaving closure to a human.**
`g_model_cache_full` (the permanent latch) and its three reset sites are
deleted entirely from `rocm/ds4_rocm_runtime.cuh`. `cuda_model_arena_alloc`
now does a fresh `cudaMemGetInfo` check before every `cudaMalloc` attempt —
if free VRAM (minus a 64 MiB margin) can't cover the chunk, it skips
immediately, exactly like the old latch's fast path, except re-evaluated on
every call instead of frozen forever by one historical failure.
`cuda_model_arena_chunk_bytes` is unchanged from its original
1-GiB-preferring form: a first attempt that shrank the chunk to exactly
what's needed was tried, measured live, and reverted — it caused *more*
real `cudaMalloc` failures (672 vs. baseline's 1) and ultimately broke
decode outright, because a right-sized chunk can never serve a later,
different tenant the way a spare 1 GiB chunk can. Full before/after
numbers, the reverted attempt, and the final two-run verification are in
the 2026-08-02 "Issue 64" entry in `experiment-log.md`.

**What this does and doesn't fix, stated plainly.** Two consecutive live
runs against the production model (`DS4_ROCM_WEIGHT_PATH_STATS=1
AMD_SERIALIZE_KERNEL=3 ./ds4 --rocm --gpu-devices 0,1,2,3
--cuda-tensor-parallel --model /home/murphy/src/ds4/ds4flash.gguf -c 64 -p
"The capital of France is" -n 20`, GPU-locked, VRAM confirmed idle
beforehand) both produced **zero** `arena alloc failed` warnings, exit code
0, and a completed generation (prefill 1.10 t/s, generation 0.59 t/s) —
meeting AC1 literally and beating the baseline's 1 failure and the reverted
attempt's 672-failure decode crash. `make -j8 test-rocm` passes clean on
the final version — but note it *also* passed clean on the broken
672-failure build, so it's near-zero evidence for this specific class of
regression; the live run is the only signal that caught it.

What it does **not** do: add headroom. The "structural headroom" prong from
this issue's own description (~0.9 GiB free after the 25.94 GiB/tier slab)
is untouched — the fix declines allocations it can already see won't fit,
it doesn't make more fit. `arena-full skip` → `host-register PCIe-map`
fallback volume went from baseline's 1438/1439 to 1036–1038 in these runs
(same shape, ~72% the count, likely just run-to-run variance rather than a
structural improvement) — the arena is still not serving the large majority
of these tenants, they're still on the slower PCIe path every time. AC2's
parenthetical ("latch behavior is either gone or provably not hit") is
satisfied: the variable is deleted, so there is no cascading multiplier
effect from one failure anymore, by construction. But AC2's main clause
("shows no `arena-full skip` cascade") is not literally true of the
artifact — the skips are still there, just no longer caused by a stuck
flag. Given this project's closure-history (`tp4-issue-closure-scope-creep`
memory: `#16`/`#53`/`#54`/`#57` were closed on unverified or
partially-true claims and had to be reopened), that literal gap is left for
a human to rule on rather than self-certified here.

**Recommendation.** The permanent-latch hazard this issue was really about
— one transient failure permanently downgrading the *entire rest of the
session* — is gone, verified, and safe to keep regardless of the AC2 call.
Whether the residual ~1038 skips/run still count as "an arena OOM problem"
worth reopening (vs. accepted as the ~0.9 GiB structural headroom being a
separate, harder problem — extending `#59`'s bounded-buffer pattern to more
tenants, or the four-way TP-aware batched owned-expert kernel, neither of
which was attempted here) is the human call this issue is left open for.
`#58`/`#63` both need "a TP=4 config that initializes without any arena
failure" — AC1 (zero `arena alloc failed`) is met literally, which may be
enough to unblock them even before AC2's literal wording is resolved; that
judgment is included in what's being handed back.
