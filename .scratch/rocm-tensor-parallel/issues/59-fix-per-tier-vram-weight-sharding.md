# 59 — Audit and fix per-tier VRAM weight sharding to eliminate 25.94 GiB load

Status: ready-for-agent

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

- [ ] Audit tensor sharding logic in `ds4.c` (`engine_append_device_cache_span` /
      `cuda_tp4` placement paths) to ensure sharded weights are not replicated across tiers
- [ ] Measured per-tier selective weight load in TP=4 mode reduced from 25.94 GiB to ~20.2 GiB
- [ ] Verification run confirms `ds4: ROCm model arena alloc failed` warnings are gone
- [ ] `make -j8 test-rocm` passes
- [ ] Findings recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`

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
