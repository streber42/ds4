# 65 — Recover structural VRAM headroom so arena tenants stop falling back to PCIe host-register

Status: ready-for-human

## Parent

`.scratch/rocm-tensor-parallel/issues/55-tp4-throughput-quality-revalidation.md`

## What to build

Split out of `#64` on human disposition 2026-08-02. `#64` deleted the
`g_model_cache_full` permanent latch and replaced it with a fresh
`cudaMemGetInfo` check before every `cudaMalloc` in `cuda_model_arena_alloc`
(`rocm/ds4_rocm_runtime.cuh`). That fully eliminated the one-failure-poisons-
the-whole-session cascade hazard `#64` was opened for — verified live twice,
zero `arena alloc failed` warnings, exit code 0, generation completes. That
part is done and not in scope here.

What it didn't touch: after the static 25.94 GiB/tier selective-weight slab
and 2.11 GiB per-tier graph overhead, live free VRAM sits around ~0.9 GiB
(`ds4: ROCm q8 fp16 cache budget exhausted ... free=0.90 GiB`, consistent
throughout a run). That's thin enough that most sub-1 GiB allocation
requests from the shared arena's many tenants (`compressor_ape`,
`rms_weight`, `q8_0`, `f16`, `f16_pair0/1`, and others — see `#59`'s audit
for the full tenant list) correctly and legitimately fail the pre-check and
fall back to the slower `cudaHostRegister` PCIe-map path. Two consecutive
live runs against the production model measured **1038 and 1036**
`arena-full skip` / `host-register PCIe-map` lines per run under
`DS4_ROCM_WEIGHT_PATH_STATS=1` (down from a stuck-latch baseline's
1438/1439, but that drop is likely just the cascade removal plus run-to-run
variance, not a structural improvement — the arena is still not serving the
large majority of these tenants). Treat **~1038 skips/run as the new
baseline** to compare future attempts against, not zero.

This is a genuinely harder problem than `#64`'s: there's no stuck flag to
delete, the allocations are failing because the VRAM really isn't there.

**A more aggressive attempt was already tried under `#64` and reverted** —
shrinking `cuda_model_arena_chunk_bytes` to request exactly what's needed
instead of preferring a 1 GiB chunk. Live-measured result: 672 real
`cudaMalloc` failures (vs. baseline's 1) and a crashed decode, because a
right-sized chunk is full the instant it's created and can never serve a
later, different-sized tenant the way a spare 1 GiB chunk can. Don't repeat
this without addressing that root cause. Full numbers in the 2026-08-02
"Issue 64" experiment-log entry.

Fix candidates carried forward from `#59`'s and `#64`'s audits (not
mutually exclusive):

- Extend `#59`'s bounded-reusable-buffer pattern (the fix that solved the
  `moe_gate`/`up`/`down` fallback's arena pressure) to other large,
  single-use, non-persistent-across-calls arena tenants — needs the "audit
  every other caller" work both `#59` and `#64` explicitly deferred (confirm
  which of `compressor_ape`, `rms_weight`, `q8_0`, `f16`, `f16_pair0/1` are
  actually safe to bound this way vs. which have genuine cross-call reuse
  that a single-slot cache would thrash).
- Build a true 4-way TP-aware batched owned-expert prefill kernel (the TP=2
  CUDA-style path already has one, `ds4_gpu_routed_moe_batch_owned_tensor`,
  `ds4.c:65102/65130`) to remove the need for the full-256-expert home-tier
  fallback entirely — `ds4.c:30855-30860` documents a four-tier version of
  this was already tried and abandoned for FP-accumulation noise, so this
  means re-solving that, not avoiding it.

## Acceptance criteria

- [ ] Live run on the production model (`ds4 --rocm --gpu-devices 0,1,2,3
      --cuda-tensor-parallel`) shows a measured reduction in `arena-full
      skip` count materially below the ~1038/run baseline established here
      (not run-to-run noise — re-run at least twice to confirm any claimed
      improvement clears variance)
- [ ] No regression in `arena alloc failed` count (must stay at 0, per
      `#64`) or decode completion (must not reproduce the reverted attempt's
      672-failure crash)
- [ ] `make -j8 test-rocm` passes (noting, per `#64`'s finding, that this
      suite doesn't load the production model and is weak evidence for this
      failure class on its own — a live run is required)
- [ ] Findings recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`

## Blocked by

`.scratch/rocm-tensor-parallel/issues/66-bisect-tp4-quality-divergence-49-61.md`

*(The TP=4 quality regression (avg_nll=13.10, 35× the PRD bar) must be fixed
before VRAM-headroom work can be meaningfully measured — any arena-skip count
under a broken compute graph is noise, not signal. Note: #66's own Blocked-by
entry cites #65 for the full 100-case fixture only; the bisect itself proceeds
on case_000 smoke, so #66 runs first.)*

## Comments

**2026-08-02 — Split out of `#64` on human disposition, after an AI
consultant panel consult.** `#64`'s actual purpose — eliminating the
permanent-latch cascade where one transient failure downgraded the entire
rest of the session — is fixed and verified; that part closes clean. The
residual ~1038 skips/run are a separate, harder, legitimate-VRAM-scarcity
problem that `#64` never attempted to solve (its own description flagged
this as a separable "structural headroom" prong from the start). A 9-model
consultant panel unanimously agreed `#64` should close now rather than stay
open for this, and unanimously warned against attempting further arena
changes without closing `#64` first, citing the reverted 672-failure
chunk-shrinking attempt as evidence this class of change is easy to get
wrong. This issue exists so that work isn't lost and isn't blocking `#64`'s
closure or downstream issues (`#58`, `#63`) that only need AC1 (zero `arena
alloc failed`), which `#64` already delivers.

**2026-08-02 — Status normalized to `ready-for-agent`.** Was `Status: open`,
a non-canonical value the ralph engine parser falls back to `ready-for-human`
for, so this issue was invisible to `ralph unblocked`/agent dispatch. #64 is
closed (the panel's stated precondition), so the block on attempting further
arena changes no longer applies. No scope change — an agent picking this up
should read the reverted-attempt note above before touching
`cuda_model_arena_chunk_bytes` or related sizing logic.

**2026-08-03 — Prefill Fallback Weight Copy Crash Resolved:**
The deterministic `moe_down` prefill fallback copy crash (`invalid argument` on `cudaMemcpyHostToDevice`) that blocked `score_official` execution in TP=4 mode has been fixed:
- Removed `posix_madvise(DONTNEED)` calls from `cuda_model_prefill_fallback_ptr`.
- Added `cuda_model_find_existing_device_ptr` and a 3-tier fallback copy chain (Device-to-Device -> Host-to-Device -> Direct `pread` from `g_model_fd` + `cudaMemcpyHostToDevice`).
- Updated `gate`, `up`, and `down` tensor wrappers in `routed_moe_launch` to point to the fallback buffers.
- Verified: All 43 layers of prefill fallback weight loading now complete 100% cleanly without crashing or erroring.
- Logit NaN investigation revealed that after prefill completes, all 129,280 output logits evaluate to `-nan` (`nan_cnt=129280/129280`), isolating the remaining TP=4 quality divergence issue to numerical NaN propagation in the TP=4 prefill computation graph.
