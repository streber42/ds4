# 58 — Re-validate quality fixture with AMD_SERIALIZE_KERNEL=3 & align prefill weight path

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/issues/55-tp4-throughput-quality-revalidation.md`

## What to build

Re-run the full 100-case `score_official` quality fixture under `AMD_SERIALIZE_KERNEL=3`
to isolate the root cause of the TP=4 `avg_nll` divergence (0.761 vs 0.369 pipeline baseline).

Historical passing quality fixture runs (#10, #32, #48) all executed under
`AMD_SERIALIZE_KERNEL=3`. Without serialization, open issue #23 documents an
intra-device race in `ds4_gpu_compressor_prefill_tensor` (`ds4_rocm_compressor.cuh`)
during compressed-KV cache prefill. The AI consultant panel confirmed that if
`score_official` under `AMD_SERIALIZE_KERNEL=3` restores `avg_nll` to ~0.370-0.378,
the quality divergence is proven to be from issue #23's compressor prefill race
(and VRAM starvation), not floating-point accumulation drift from the 86 all-reduces.

Also audit and align `g_use_host_weights` in `ds4.c` / `rocm/ds4_rocm_runtime.cuh`.
Currently `g_use_host_weights` is enabled during TP=4 decode but not during prefill,
causing prefill vs decode weight resolution discrepancies between host-mapped and
cached VRAM pointers.

## Acceptance criteria

- [ ] Full 100-case `score_official` quality fixture run under `AMD_SERIALIZE_KERNEL=3`
      for both pipeline and TP=4 paths
- [ ] Confirmed whether `avg_nll` under serialization returns to the ~0.370 band
      (isolating issue #23 compressor race)
- [ ] `g_use_host_weights` aligned across prefill and decode paths in `ds4.c`
- [ ] `make -j8 test-rocm` passes
- [ ] Findings recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`

## Blocked by

`.scratch/rocm-tensor-parallel/issues/59-fix-per-tier-vram-weight-sharding.md`

## Comments

**2026-08-01 — #62's HEAD re-measurement disposition: stay blocked, re-pointed
at #59.** (Human: proceed straight to #59, no further TP=4 retries on #62.)

`#62` re-ran the fixture on HEAD (post-`1fe4829`/`0cb9cf3`) and found neither
of the outcomes anyone expected. Two TP=4 attempts, both hitting `arena alloc
failed for moe_down` before any case scored: one crashed at case 19/100, one
completed all 100 cases at avg_nll 16.43 (median 16.36, **every case in the
≥2 bucket**) — ~44x the PRD bar, ~20x worse than the old 0.7607 figure this
issue's "What to build" cites. That number is now even less usable as this
issue's target than the stale 0.7607 was.

The distribution-shape argument against this issue's compressor-race premise
(recorded below, from the old artifacts) is *not* the whole story anymore:
the new result's determinism is split — same failing tensor both times
(favors #59: VRAM pressure), but a different consequence each time, crash vs.
silent corruption (the run-to-run variance that would favor this issue's
race hypothesis). Re-pointed `Blocked by` from `#62` to `#59` per the human's
decision: run `#59` first regardless, since the VRAM-driven alloc failure
gates everything downstream either way, and re-evaluate this issue's premise
once TP=4 can produce a stable run at all. AC3 (`g_use_host_weights`
alignment) remains valid and independent of this — worth splitting out if
picked up before the rest of this issue is unblocked.

**2026-08-01 — Re-scoped on audit; premise is now in doubt.** (Human
authorization given to override prior dispositions and make issues reflect
reality.)

Three corrections, in increasing order of importance.

**1. Dependency cycle broken.** This issue was `Blocked by #55` while #55 was
`Blocked by #58`/`#59` — a cycle that made all three permanently undispatchable
under the literal-only `Blocked by` semantics this project uses (see
[[ralph-issue-blocked-by-must-be-explicit]]). #55 is this issue's **Parent**, not
its blocker. Replaced with `#62`.

**2. The 0.761 figure in "What to build" does not measure HEAD.** It comes from
`q_tp4_51.log`, timestamped 05:46, while commits `1fe4829` (#60 rollout, 10:38)
and `0cb9cf3` (#61 async all-reduce, 11:38) landed hours later. This issue is
currently scoped to explain a number that does not refer to code in the tree.
`#62` exists to produce a real one first.

**3. The compressor-race hypothesis is disfavoured by evidence already in
hand.** Per-case distribution analysis of `q_tp4_51.log` vs `q_pipeline_51.log`
(experiment-log, "Audit of the 0.7607 TP=4 quality number") shows the TP=4
degradation is a **uniform** rightward shift of the entire distribution —
median 0.72 vs 0.35, 21 cases under 0.5 vs 76, and first-half/second-half means
flat at 0.7895/0.7697. A race is episodic: it would produce bimodality or
run-to-run variance, not a flat per-token tax. The hardest cases are also hard
on the pipeline path (case_094: 4.85 pipeline / 4.67 TP=4), i.e. intrinsic
prompt difficulty.

What the shape *does* fit is a systematic precision fallback:
`q8 fp16 cache budget exhausted` fires **1×** in the pipeline log and **4300×**
in TP=4 — 43 per case, once per layer — plus a TP=4-only `arena alloc failed`.
That is `#59`'s territory, not this issue's.

**Recommendation: run `#59` before the serialization experiment here.**
`AMD_SERIALIZE_KERNEL=3` makes runs dramatically slower, so this is an expensive
experiment to spend GPU-hours on against a hypothesis the distribution data
already argues against, on a config known to be VRAM-starved.

**Still valid and worth keeping regardless:** AC3, aligning `g_use_host_weights`
across the prefill and decode paths in `ds4.c`. That is a genuine correctness
discrepancy, independent of the race hypothesis and cheap to do. Consider
splitting it out rather than letting it sit behind a serialization run that is
now predicted to come back negative.

There is also an open methodological hole `#62` must close: the historical
passing runs (#10/#32/#48) all used `AMD_SERIALIZE_KERNEL=3`, and it was never
recorded which setting `q_pipeline_51` and `q_tp4_51` each used. If they
differed, the 0.37-vs-0.76 comparison was never valid to begin with — which
would dissolve this issue's premise entirely.
