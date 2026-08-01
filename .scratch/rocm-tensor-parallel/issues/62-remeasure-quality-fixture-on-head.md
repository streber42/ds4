# 62 — Re-measure the 100-case quality fixture on HEAD under a controlled serialize setting

Status: ready-for-human

## Parent

`.scratch/rocm-tensor-parallel/issues/55-tp4-throughput-quality-revalidation.md`

## What to build

Establish a trustworthy current quality number for both the pipeline and TP=4
paths. **There is presently no quality measurement of HEAD at all**, and every
downstream issue (#51, #55, #58, #59) is scoped to explain or fix a number that
does not refer to the code in the tree.

Established by timestamp audit (see `experiment-log.md`, 2026-08-01 "Audit of
the 0.7607 TP=4 quality number"): every artifact in `quality-out/` predates
every commit in the current stack.

| artifact / commit | time (2026-08-01) |
|---|---|
| `q_pipeline_51.log` (avg_nll 0.3692) | 04:44 |
| `q_tp4_51_disc_*` sweep | 04:57–05:08 |
| `q_tp4_51.log` (**avg_nll 0.7607**) | 05:46 |
| `0f0cbe1` chore | 08:54 |
| `6bdc4a3` chore | 09:55 |
| `1fe4829` — #60, 43-layer persistent-thread rollout | 10:38 |
| `0cb9cf3` — #61, async all-reduce fencing | 11:38 |

The 0.7607 figure therefore predates both of the changes most likely to move
it, and neither #60 nor #61 re-ran the fixture afterwards. It is **not**
evidence that the current build is broken, nor that it works.

A second, independent problem with the existing comparison: the historical
passing runs (#10, #32, #48) all ran under `AMD_SERIALIZE_KERNEL=3`.
`scripts/diagnose-prefill.sh` defaults that variable to `3`;
`scripts/tp4-instrument.sh` deliberately leaves it unset. Which setting
`q_pipeline_51` and `q_tp4_51` each ran under was never recorded. **If the two
runs differed, the 0.37-vs-0.76 comparison was never valid on its face** — so
this issue must record the setting explicitly and use the same one for both
paths.

## Acceptance criteria

- [x] **Rebuild the fixture binary first — the one on disk is stale.** The
      quality fixture is `gguf-tools/quality-testing/score_official`, a
      *separate* binary from `./ds4`, built by `make ROCM_ARCH=gfx1201
      rocm-quality` (which force-rebuilds with `-B` and filters out
      `-ffast-math`). The copy currently in the tree is timestamped 08-01
      **09:25**, which predates both `1fe4829` (#60, 10:38) and `0cb9cf3`
      (#61, 11:38) — so running it as-is would produce *another* number that
      does not measure HEAD. Record the commit SHA and build command in the
      run log.
- [x] Full 100-case `score_official` run on the **pipeline** path, HEAD build
- [ ] Full 100-case `score_official` run on the **TP=4** path, HEAD build —
      **not satisfiable as written; see Comments.** Two attempts, neither
      produced a clean 100-case measurement: one crashed at case 19/100, one
      completed but with values ~44x the PRD bar (not a comparable data
      point).
- [x] Both runs use an **identical and explicitly recorded**
      `AMD_SERIALIZE_KERNEL` value; record it in the log filename or header
      — `AMD_SERIALIZE_KERNEL=3` for pipeline and both TP=4 attempts, header
      block in each log.
- [x] Report `avg_nll`, `first_match`, `api_top1_rate`, `api_pair_rate` for
      both against the PRD bar (avg_nll 0.370–0.378, first_match ≥60/100,
      api_top1_rate ≥0.85, api_pair_rate ≥0.98) — pipeline meets the bar; TP=4
      does not produce a valid comparison (see Comments).
- [x] Record per-case `avg_nll` distribution (median and the <0.5 / [0.5,1) /
      [1,2) / ≥2 bucket counts), not just the mean — the mean alone hid the
      shape of the 0.7607 result
- [x] Count `q8 fp16 cache budget exhausted` and `arena alloc failed`
      occurrences in each log and report both
- [x] Findings recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`

## Comments

**2026-08-01, human pairing session.** Picked up from a prior agent run that
stalled: it had rebuilt the fixture binary correctly and completed the
pipeline HEAD measurement, but crashed before running TP=4, leaving a stale
GPU lock (dead PID) and `dev-vllm` stopped. Verified the lock was genuinely
stale (process dead, GPUs idle, no zombie processes, VRAM empty) before
reacquiring, then re-ran *both* paths cleanly rather than trust the
inherited pipeline artifact, which had no header recording its env var or
command line.

**Pipeline: solid, meets the PRD bar.** `avg_nll` 0.369196, `first_match`
68/100, `api_top1_rate` 0.8637, `api_pair_rate` 0.9890. Median 0.348528,
buckets <0.5/[0.5,1)/[1,2)/≥2 = 76/22/1/1. 1 `q8` warning, 0 `arena alloc
failed`. Bit-identical to the inherited run and to the historical
`q_pipeline_51` (0.3692) — this also answers the issue's serialize-agreement
question on the pipeline side: `q_pipeline_51` was evidently run under
`AMD_SERIALIZE_KERNEL=3` too, since a run explicitly forced to that setting
reproduces it exactly.

**TP=4: a third outcome this issue didn't anticipate.** Neither "at/near
PRD bar" nor "near 0.76 again" — both attempts printed `arena alloc failed
for moe_down` before any case scored (same tensor both times), then diverged:
run 1 crashed at case 19/100 (`logits failed at target token 21`); run 2
completed all 100 cases but at `avg_nll` 16.43 (median 16.36, **all 100
cases in the ≥2 bucket**), `api_top1_rate` 0.029, `api_pair_rate` 0.542 —
roughly 44x the PRD bar and ~20x worse than the already-failing 0.7607
figure. Real TP placement confirmed both times via the four `CUDA tier N
... selective weights: 25.94 GiB in 1328 ranges` lines. `q8 fp16 cache
budget exhausted` warnings: 860 (run 1) → 4300 (run 2).

Reading the issue's own falsifier: the *trigger* (moe_down alloc failure)
reproduced deterministically, pointing at #59 (per-tier VRAM: 25.94 GiB vs
27.79 GiB available); the *consequence* (crash vs. silent corruption) varied
between runs, which is the run-to-run-variance signature the issue says
should strengthen #58's race hypothesis. Both look implicated; this
measurement can't decide between them alone. No fix was attempted — out of
this issue's scope.

**Recommendation for #58/#59:** #59 should land first regardless of which
hypothesis is right, since the VRAM-driven arena-alloc failure gates
everything downstream; #58's race hypothesis is not eliminated and should
stay open pending #59.

Full detail, per-case tables, and the reasoning trail: see the
`experiment-log.md` entry "Issue 62: Re-measured HEAD, found a third
outcome." Status left `ready-for-human` — this is a materially worse and
different finding than the issue anticipated, and needs a human call on
next steps (retry TP=4 more times to build a distribution over the
crash/corruption split? proceed straight to #59? something else) before
this issue can close.

## Notes for whoever picks this up

- Do **not** re-baseline the PRD target if the number falls short. Record the
  shortfall plainly. This is the mistake #43 made and #44–#48 had to correct.
- A TP=4 log's `multi-GPU layout:` block prints
  `GPU0: layers 0-42 … GPU1-3: (no transformer layers) (0.0 GB)`. This looks
  like TP sharding failed. **It has not** — that block is a pipeline-planner
  cosmetic artifact that does not describe TP placement. Confirm real
  placement via the four
  `ds4: CUDA tier N (device N) selective weights: … GiB in … ranges` lines.
- **Do not read a first-half/second-half rise as degradation over the run.**
  In the existing artifacts the *pipeline* path rises 0.3688 → 0.4380 across
  its halves while TP=4 stays flat (0.7895 → 0.7697). Since the well-behaved
  path shows the larger rise, that gradient is intrinsic case-difficulty
  ordering in the fixture, not a run-length effect. Compare halves *between
  paths*, not within one.
- The `q_tp4_51_disc_*` layer sweep is not a layer-count discriminator: all
  eight runs share byte-identical warning counts (129) and near-identical
  `avg_nll` (0.329–0.347), the signature of eight runs of one code path. This
  is consistent with `ds4.c:27652`'s whole-token gate
  (`metal_graph_tp4_spike_layer_enabled(DS4_N_LAYER - 1)`, shipped in
  `1fe4829`) collapsing every partial `DS4_TP4_THREADED_LAYERS` value onto the
  legacy path. Don't cite that sweep as evidence for anything.

## Interpreting the result — decide this before running, not after

This issue's output is a decision, not just a number. Two outcomes, two
different next moves. Write the resulting disposition into `#58` and `#59`
rather than leaving it implicit.

**If TP=4 comes back at or near the PRD bar (avg_nll ~0.370–0.378):** the
0.7607 gap was an artifact of the pre-`1fe4829` build, a mismatched
`AMD_SERIALIZE_KERNEL` between the two runs, or both. In that case `#58` and
`#59` are explaining a ghost and should **close cheaply** — neither is chartered
to fix a gap that no longer exists. Two things still survive that outcome and
should be re-homed rather than dropped:
- `#58`'s `g_use_host_weights` prefill/decode alignment — a real correctness
  discrepancy independent of the race hypothesis.
- `#59`'s per-tier VRAM work — still worth doing on throughput/headroom grounds
  even if quality is fine, since 25.94 GiB/tier against 27.79 GiB available is
  what forces the q8 fallback and the host-mapped MoE path.

**If TP=4 comes back near 0.76 again:** the gap is real and current. Run `#59`
**before** `#58`. The per-case distribution from the stale run was a *uniform*
rightward shift (median 0.72 vs 0.35, flat first-half/second-half at
0.7895/0.7697), which fits a systematic per-token precision tax and does not fit
an episodic race — so `#58`'s compressor-race premise is the weaker hypothesis,
and `AMD_SERIALIZE_KERNEL=3` runs are slow enough that testing it first is
expensive. Confirm the new run reproduces that same uniform shape before
committing to this branch; if the new distribution is instead **bimodal or
shows high run-to-run variance**, that inverts the recommendation and `#58`'s
race hypothesis becomes the stronger one.

**Either way:** record which `AMD_SERIALIZE_KERNEL` value was used and whether
the pipeline and TP=4 runs agreed on it. If they did not agree in the historical
runs, say so explicitly — that alone would invalidate the 0.37-vs-0.76
comparison retroactively and is worth its own note in `#55`.

## Blocked by

*(nothing — this is the gate the others wait on)*
