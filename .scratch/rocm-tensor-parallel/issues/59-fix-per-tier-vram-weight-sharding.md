# 59 — Audit and fix per-tier VRAM weight sharding to eliminate 25.94 GiB load

Status: closed

## Parent

`.scratch/rocm-tensor-parallel/issues/55-tp4-throughput-quality-revalidation.md`

## What to build

Audit `ds4.c` selective weight placement and device caching to fix the per-tier
weight footprint in TP=4 mode.

Currently, selective weight caching loads **25.94 GiB per tier** (1328 ranges) on all
4 GPUs. For an 81 GiB production model, proper 4-way sharding (expert, head, and
vocab sharding) should reduce per-rank weight storage to ~20.2 GiB. The extra ~5.7 GiB
per tier indicates that certain sharded tensors are being fully replicated across all tiers
or cached improperly during model loading.

On 32GB GPUs (29.79 GiB available VRAM after 2.0 GiB scratch reservation), 25.94 GiB
of weight cache leaves only ~0.35 GiB free VRAM per GPU. This triggers
`ds4: ROCm model arena alloc failed for moe_*: out of memory` during model initialization
and forces MoE weights to fall back to host-mapped memory over PCIe.

Fixing per-tier sharding to reduce weight footprint to ~20.2 GiB will reclaim ~5.7 GiB
VRAM per GPU, eliminating model arena host fallbacks and ensuring all weights sit cleanly in
fast VRAM.

## Acceptance criteria

- [x] Audit tensor sharding logic in `ds4.c` (`engine_append_device_cache_span` /
      `cuda_tp4` placement paths) to ensure sharded weights are not replicated across tiers
      — audited; **no bug found**, see Comments/experiment-log 2026-08-01 entry
- [ ] Measured per-tier selective weight load in TP=4 mode reduced from 25.94 GiB to ~20.2 GiB
      — **unreachable without regressing #30/#32**; 4.26 of the 5.75 GiB gap is
      deliberately-replicated correctness-fixed tensors
- [ ] Verification run confirms `ds4: ROCm model arena alloc failed` warnings are gone
      — **improved, not eliminated.** Candidate-1 fix (bounded per-device 3-slot
      reusable buffer for the batch-prefill MoE gate/up/down fallback, replacing
      the unbounded shared-arena growth) landed and is measurably working: the
      `arena alloc failed` failure point moved from `moe_down` before any layer
      completed (1212 skips cascading for the rest of the session) to a single
      `q8_0` failure at offset 80.24 of 80.76 GiB — i.e. after the model's full
      43 layers traverse cleanly, only the last tensor doesn't fit. 129
      prefill-fallback reload events = 43 layers × 3 tensors exactly, confirming
      bounded (non-accumulating) reuse. One `arena alloc failed` is still one
      too many by this AC's own wording — remaining ~0.9 GiB structural headroom
      and the `g_model_cache_full` permanent latch that turns that one failure
      into a session-long PCIe-fallback cascade are split into `#64`.
- [x] `make -j8 test-rocm` passes against the patched tree (all suites, including
      the new code paths — not a clean-tree baseline this time)
- [x] Findings recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`

## Blocked by

*(nothing — `#62`'s HEAD re-measurement is done; human disposition is to
proceed directly to this issue)*

## Comments

**2026-08-01 — Promoted on audit: this is now the leading hypothesis for the
TP=4 quality gap.** (Human authorization given to override prior dispositions
and make issues reflect reality.)

**Dependency cycle broken.** This issue was `Blocked by #55` while #55 was
`Blocked by #58`/`#59` — a cycle making all three permanently undispatchable
under this project's literal-only `Blocked by` semantics (see
[[ralph-issue-blocked-by-must-be-explicit]]). #55 is this issue's **Parent**, not
its blocker. Replaced with `#62`.

**New supporting evidence.** Per-case distribution analysis of the existing
100-case artifacts (experiment-log, "Audit of the 0.7607 TP=4 quality number")
found the TP=4 degradation is a **uniform** shift of the whole distribution —
median 0.72 vs pipeline's 0.35, 21 cases below 0.5 vs 76, first-half/second-half
means flat at 0.7895/0.7697. That shape is a systematic per-token tax, which fits
a precision fallback and does *not* fit a race (episodic → bimodal) or
progressive VRAM exhaustion over a run (would skew second-half; it doesn't).

The warning counts point the same way and are strikingly asymmetric:

| log | `q8 fp16 cache budget exhausted` | `arena alloc failed` |
|---|---|---|
| `q_pipeline_51.log` | 1 | 0 |
| `q_tp4_51.log` | **4300** | 1 |

4300 / 100 cases = 43 per case = **once per layer, every case**. The model is
running its low-precision fallback path essentially all the time under TP=4,
which is exactly the mechanism this issue exists to remove: 25.94 GiB/tier
against 27.79 GiB post-overhead leaves ~0.34 GiB free, too little for the
Q8→F16 acceleration cache and the model arena.

**Suggested ordering: run this issue before `#58`.** `#58`'s premise (issue
#23's compressor-prefill race, provable via `AMD_SERIALIZE_KERNEL=3`) is
argued against by the distribution shape above, and serialized runs are
expensive. If reclaiming the ~5.7 GiB here removes the fallbacks, the quality
number may simply recover, making `#58`'s experiment moot — and if it doesn't,
`#58` at least gets to run against a config that isn't VRAM-starved.

**Caveat carried from `#62`:** the 25.94 GiB/tier figure and the warning counts
above come from logs timestamped 04:44–05:46, which predate commits `1fe4829`
and `0cb9cf3`. The per-tier figure is corroborated across both the full and
discriminator runs so is very likely still current, but re-confirm it on a HEAD
build before sizing the fix.

**2026-08-01 — Confirmed current on HEAD; escalated. Cleared to start, human
disposition (option 2 of #62's proposed next steps: proceed straight here,
no further TP=4 retries).** `#62` re-ran the fixture against the post-`1fe4829`/
post-`0cb9cf3` HEAD build and the 25.94 GiB/tier figure and per-tier VRAM
pressure are still live — confirmed via the same four `CUDA tier N ...
selective weights: 25.94 GiB in 1328 ranges` lines in both new TP=4 runs.

The situation is worse than the caveat above anticipated: this issue is no
longer just about reclaiming ~5.7 GiB for throughput/headroom. Both HEAD TP=4
runs hit `ds4: ROCm model arena alloc failed for moe_down (1024.00 MiB
chunk): out of memory` **before any case was scored** (same tensor
deterministically both times), and what happens after that failure varies:
one run crashed outright (`case_019 logits failed at target token 21`), the
other completed all 100 cases but with avg_nll 16.43 (median 16.36, every
case in the ≥2 bucket) — garbage output, not drift. `q8 fp16 cache budget
exhausted` warnings went 860 → 4300 between the two runs, consistent with
worsening fragmentation against the same ~0.35 GiB free-VRAM margin.

This means AC3 ("verification run confirms `arena alloc failed` warnings are
gone") is now the load-bearing acceptance criterion, not a nice-to-have — a
model that cannot reliably allocate `moe_down` is not a smaller quality gap,
it's an unusable TP=4 path. `#58` stays blocked on this issue per the human's
call: its race hypothesis can't be evaluated against a config that can't
reliably initialize in the first place.

**2026-08-01 — Audit complete on HEAD (`185a2ae`, unmodified). AC1 satisfied
(no bug); AC2 unreachable without regressing #30/#32; AC3's premise is
arithmetically false. `ready-for-human`.** Full numbers, per-category tensor
breakdown, live VRAM budget chain, and `DS4_ROCM_WEIGHT_PATH_STATS=1` trace
are in `.scratch/rocm-tensor-parallel/experiment-log.md`'s "#59 audit"
entry (2026-08-01). Summary:

- Parsed the production GGUF's tensor table directly (offset-delta byte
  sizes, no live GPU load needed) and re-derived `engine_tp4_shard_divisor`'s
  (ds4.c:56571) category rules: `sharded/4 + replicated = 73.0865/4 +
  7.6729 = 25.9446 GiB`, matching the observed `25.94 GiB` log line to two
  decimals, over exactly 1328 tensors (matches "1328 ranges"). **The
  sharding logic is correct** — nothing sharded is being replicated.
- The ~20.2 GiB AC2 target requires zero replication. 4.26 of the 7.67 GiB
  replicated tail is `attn_q_b`/`attn_output_a`/shared-expert, all `div=1`
  by **documented, deliberate** decision (issues #32 and #30 respectively,
  see ds4.c:56592-56613) — flipping those divisors to hit AC2 re-breaks
  closed correctness fixes and is out of bounds for this issue.
- Live run (`ds4 --rocm --gpu-devices 0,1,2,3 --cuda-tensor-parallel` on the
  production model) reproduces the `arena alloc failed for moe_down (1024
  MiB chunk)` failure, but it does **not** come from the static 25.94 GiB
  per-tier slab (one exact-sized `hipMalloc`). It comes from a separate,
  unbounded, session-lifetime VRAM cache (`cuda_model_arena_alloc`,
  ds4_rocm_runtime.cuh:5747) that ROCm TP=4's batch-prefill MoE fallback
  uses to promote each layer's *full, unsharded* 256-expert table into VRAM
  one layer at a time (documented at ds4.c:30845-30860: prefill runs the
  whole expert table on tier 0 only, by design, to match the pipeline
  reference bit-for-bit — it is not tensor-parallel during prefill at all).
  Each layer needs ~1.6–1.7 GiB for this and it is **never freed**
  (`g_model_arenas` accumulates for the process lifetime). Even a perfect
  zero-replication 20.19 GiB/tier buys only ~3 more layers on a 43-layer
  model before the same OOM recurs. **AC2, even fully achieved, would not
  satisfy AC3** — the issue's own causal claim ("reclaiming ~5.7 GiB...
  eliminate model arena host fallbacks") does not hold arithmetically.
  `DS4_ROCM_WEIGHT_PATH_STATS=1` confirms the zero-copy `cudaHostRegister`
  fallback (ds4_rocm_runtime.cuh:4837-4874) does successfully catch every
  subsequent request after the one hard arena failure (1212 skips, 1213
  successful host-register maps) — weights are read correctly but slowly
  over PCIe for the rest of the session, not silently dropped.
- Two real fix candidates for AC3, both new-issue-sized and both outside
  this audit's safe scope (see experiment-log for detail): (1) free each
  layer's fallback-loaded expert chunk right after that layer's prefill MoE
  call, since it's never reused; needs an audit of every other caller
  sharing the same cache (`compressor_ape`, `rms_weight`, `q8_0`, `f16`,
  `f16_pair0/1`) to confirm none expects persistence. (2) Build a true
  4-way TP-aware batched owned-expert prefill kernel (the TP=2 CUDA-style
  path already has one, `ds4_gpu_routed_moe_batch_owned_tensor`,
  ds4.c:65102/65130) — ds4.c:30855-30860 documents a four-tier version of
  this was already tried and abandoned for FP-accumulation noise, so this
  means re-solving that, not avoiding it.
- `make -j8 test-rocm` passes, but against an unmodified tree — reported as
  the clean-tree baseline, not as evidence of a fix.

No code was changed this session. Leaving `Status: ready-for-human` for a
call on which AC3 fix path to take (or whether to retarget AC2's number
given it's unreachable as written).

**2026-08-02 — Candidate 1 implemented per human disposition ("implement now, in
#59"). Real, measured improvement; AC3 not fully closed; residual split to `#64`.**

Replaced the two `cuda_resolve_weight_ptr(..., "moe_gate"/"moe_up"/"moe_down")`
call sites (`ds4_rocm_moe_launch.cuh:730-732`) — the home-tier-only, full
unsharded 256-expert table fallback used during ROCm TP=4 batch prefill — with
a new `cuda_model_prefill_fallback_ptr` (`ds4_rocm_runtime.cuh`, next to
`cuda_model_arena_alloc`): a fixed 3-slot (gate/up/down), per-device reusable
buffer that overwrites in place on a new offset instead of growing the shared
`g_model_arenas` bump allocator forever. This sidesteps rather than solves the
"audit every other caller" problem noted above — `compressor_ape`,
`rms_weight`, `q8_0`, `f16`, `f16_pair0/1` still go through the original
shared arena/range cache, untouched.

Live run (`ds4 --rocm --gpu-devices 0,1,2,3 --cuda-tensor-parallel` on the
production model, `DS4_ROCM_WEIGHT_PATH_STATS=1`), HEAD + this change:

| | before (this issue's earlier audit) | after (this session) |
|---|---|---|
| first `arena alloc failed` | `moe_down`, before any layer completes | `q8_0`, at offset 80.24 GiB of 80.76 GiB (last tensor) |
| downstream cascade | 1212 arena-full skips / 1213 host-register maps | 1601 arena-full skips (same permanent-latch cascade, just later) |
| MoE fallback behavior | unbounded — new ~1.6-1.7 GiB chunk per layer, never freed | bounded — 129 reload events = 43 layers × 3 tensors exactly, same 3 buffers reused |

So the specific failure this issue was opened to chase (`moe_down` OOM before
scoring starts) is gone, and the model now traverses its full 43 layers before
hitting any arena failure. But **AC3 as worded is not satisfied**: one
`arena alloc failed` still fires (now for `q8_0`, not `moe_down`), and after it
latches, everything downstream still falls back to host-register PCIe reads —
`g_model_cache_full` (`ds4_rocm_runtime.cuh:5749`, set at `:5785`) is a
process-lifetime latch with no reset path short of full model teardown, so one
transient near-the-margin failure still converts into a session-long cascade
regardless of which tenant trips it first. Remaining structural headroom is
~0.9 GiB free after the static 25.94 GiB/tier weight slab + 2.11 GiB overhead —
thin enough that almost any single-digit-hundred-MiB request can be the one
that loses the race.

Text-coherence smoke testing (`-p "The capital of France is" -n 20`, greedy)
turned out to be an invalid instrument for judging this: **pipeline mode
produces equally incoherent output on the identical unserialized raw-prompt
test**, confirming this project's already-documented pre-`AMD_SERIALIZE_KERNEL=3`
dispatch-race behavior (see `#58`), not a regression from this change. No
numerical/quality claim is made here one way or the other — that requires the
`score_official` NLL fixture, which is `#63`'s job, not this audit's.

Also tried (and reverted, no observable effect either way): an explicit
`cudaDeviceSynchronize()` before the new buffer's overwrite, to rule out a
stream-ordering hazard between one layer's kernels reading the slot and the
next layer's `cudaMemcpy` overwriting it. Arena-failure count/position were
identical with and without it. Did not ship since it was a no-op probe, not a
fix — worth re-litigating if `#64`'s work ever surfaces a live correctness bug
in this specific path.

`make -j8 rocm && make -j8 rocm-quality && make -j8 test-rocm` all pass against
the patched tree (`score_official` rebuilt so it isn't stale — see
[[ds4-build-targets-share-binary-name]]).

**Disposition:** closing `#59` on what's done and measured (AC1, AC4, AC5;
AC3 materially improved though not literally satisfied). Opened `#64` for the
residual — the `q8_0`-class arena failure, the ~0.9 GiB headroom, and the
`g_model_cache_full` permanent-latch design — and re-pointed `#58`/`#63`'s
`Blocked by` from `#59` to `#64`, since they need a TP=4 config that
initializes without any arena failure, which this issue got closer to but did
not fully deliver.
