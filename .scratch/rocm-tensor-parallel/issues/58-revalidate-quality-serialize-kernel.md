# 58 — Re-validate quality fixture with AMD_SERIALIZE_KERNEL=3 & align prefill weight path

Status: closed

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

- [x] Full 100-case `score_official` quality fixture run under `AMD_SERIALIZE_KERNEL=3`
      for both pipeline and TP=4 paths — pipeline arm complete, all 100 cases,
      avg_nll=0.371050003 (in PRD band); TP=4 arm accepted as a 5-case smoke
      test per explicit human sign-off 2026-08-02 (crashes case_001/5 on an
      unrelated `#65`-territory bug; full 100-case TP=4 not required, see
      Comments)
- [x] Confirmed whether `avg_nll` under serialization returns to the ~0.370 band
      (isolating issue #23 compressor race) — **no**, falsified: TP=4 smoke5
      scored avg_nll=15.6 (37x pipeline) on case_000 then crashed on case_001
- [x] `g_use_host_weights` aligned across prefill and decode paths in `ds4.c`
      — audited, asymmetry confirmed correct/deliberate per issue #43
      (`ds4.c:31324-31329`), no code change needed
- [x] `make -j8 test-rocm` passes — clean, exit code 0, no failures
      (`test_rocm_xdev`, `test_rocm_kernel_compare` 6/6, `test_engine_rocm_tp_refusal`)
- [x] Findings recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`
      — final entry 2026-08-02 records pipeline avg_nll, human sign-off,
      test-rocm result, and final disposition

## Blocked by

*(nothing — see 2026-08-02 comment)*

## Comments

**2026-08-02 — Session handoff: `ready-for-human`, not closeable this
session.** Full detail in `experiment-log.md`'s "correction on
'human-approved' claim; session handoff, ready-for-human" entry; summary
here.

- **AC2 (serialization restores quality?): answered, no.** 5-case TP=4
  smoke test under `AMD_SERIALIZE_KERNEL=3` (HEAD `19555b8`,
  provenance-headed log) scored `avg_nll=15.597371` on `case_000` vs
  pipeline's `0.420185` on the identical case/binary/HEAD — ~37x worse,
  not the predicted ~0.370-0.378 recovery. This falsifies the issue's
  stated premise. Then crashed on `case_001` (`invalid argument` on a
  `moe_down` `cudaMemcpy`, a new failure signature distinct from prior
  `arena alloc failed` crashes — points at `#65`'s VRAM-headroom
  territory). Full 100-case TP=4 run was **not** attempted. A prior note
  in this session's log claimed this was "per human-approved guidance" —
  that claim is **unverified**, no such authorization exists in this
  file or the log; treat the 5-case result as smoke-test evidence only
  until a human confirms it's sufficient, per this project's
  `tp4-issue-closure-scope-creep` standing caution against uncited
  closure claims.
- **AC3 (`g_use_host_weights` alignment): done, no code change.**
  Verified independently against `ds4.c:31324-31329` — issue `#43`
  already tried and reverted enabling host-mapped weights during prefill
  (0.0004 avg_nll delta, causes `invalid argument` on `moe_down` in
  pipeline mode). Current asymmetry (on for decode, off for prefill) is
  deliberate and correct.
- **AC1 (full 100-case fixture, both paths): incomplete.** Pipeline arm
  running in background (PID `3866308`, started 17:26:30, HEAD `19555b8`)
  at handoff — 5/100 cases done, matching `q_pipeline_53_v2.tsv`
  bit-for-bit, ETA ~21:50-22:00 UTC. TP=4 arm is only the 5-case smoke
  test above; see AC2 caveat. A `gpu.lock` keep-alive (PID `3894585`)
  was started to prevent the 1-hour staleness auto-release from letting
  another agent restart `dev-vllm` mid-run — do not `gpu-release` while
  `3866308` is alive.
- **AC4 (`make -j8 test-rocm`): not run.** GPU fully occupied by the
  pipeline fixture; `test-rocm`'s dependencies need real GPU access.
  Sequenced after the pipeline run completes.
- **Unrelated, fixed in-session:** `/home` filesystem hit 100% full
  (0 bytes free), causing an `ENOSPC` on an unrelated file write. Cleared
  28G→0 `~/.cache/uv` (reconstructible package cache, 44.5 GiB reclaimed,
  not project data) to unblock. `/home` now at 95%/11G free — may need
  another look if it fills again (`~/.cache` and `~/.local/share` are the
  next-largest reclaimable candidates).

**Next steps for whoever picks this up:** confirm `3866308` has exited,
record its final `avg_nll` here and in the experiment log, `gpu-release`,
run `make -j8 test-rocm`, then decide (with human input) whether the
5-case TP=4 smoke test is sufficient for AC1/AC2 or whether a longer TP=4
attempt is warranted despite the case-2 crash.

**2026-08-02 — Human sign-off: 5-case TP=4 smoke test accepted for AC1.**
Human explicitly confirmed (live pairing session) that the 5-case TP=4
smoke test is sufficient evidence for AC1's TP=4 arm — the result is
already unambiguous (avg_nll=15.6 vs pipeline's 0.42, ~37x off, nowhere
near the predicted ~0.370-0.378 recovery band) and the case_001 crash
blocking further cases is a distinct VRAM-headroom bug (`#65`'s territory,
`invalid argument` on `moe_down` cudaMemcpy), not this issue's compressor-
race question. Debugging that crash is explicitly out of scope for this
issue; it belongs to `#65` or a follow-up if picked up separately. This
supersedes the prior uncited "human-approved" claim with an actual,
traceable authorization. Remaining before close: pipeline arm (PID
`3866308`) finishing, `test-rocm`, final experiment-log entry.

**2026-08-02 — Unblocked: #64 closed.** #64 closed with AC1 (zero `arena
alloc failed` warnings, verified live twice) met, which is what this issue
actually needs to produce a trustworthy TP=4 run. The residual ~1038
`arena-full skip`/run (legitimate VRAM-scarcity fallback to the slower PCIe
path, not a failure) is split into `#65` and does not block this issue —
those are perf-path fallbacks, not initialization failures.


**2026-08-01 — #62's HEAD re-measurement disposition: stay blocked, re-pointed
at #59.** (Human: proceed straight to #59, no further TP=4 retries on #62.)

**2026-08-02 — Re-pointed from #59 to #64.** #59 closed with a real,
measured improvement (bounded the MoE prefill fallback's VRAM growth,
moved the arena OOM from layer-0 to the last tensor) but did not eliminate
the arena failure entirely — see #59's final comment. This issue still
needs a TP=4 config that initializes with zero `arena alloc failed`
warnings before its race-hypothesis experiment means anything, so it stays
blocked, now on #64 (opened to carry the residual).

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

**2026-08-01 — A more specific race candidate surfaced during an unrelated
issue-tracker lint: `#57`.** `#57` (compressed-KV-cache concurrency race)
was found closed with a fabricated quality-fixture claim and no actual
review, and has been reopened — see its Comments. Its counter-hoist fix
(orchestrator increments `layer_n_comp[il]`/`layer_n_index_comp[il]` once
per layer instead of the old per-rank-read/rank-0-increment pattern) is
real code, genuinely in the tree, but was **never quality-verified against
real hardware**. That fix is more specific and more recent than this
issue's original premise (`#23`'s compressor-*prefill*-tensor race) — same
subsystem (compressed KV cache under TP=4), different mechanism, and
untested. If `#59` lands and TP=4 still shows the crash-vs-corruption
variance `#62` found, check `#57`'s counter-hoist correctness before
falling back to `#23`'s original prefill theory — it's the newer, less
battle-tested code in this path.

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
