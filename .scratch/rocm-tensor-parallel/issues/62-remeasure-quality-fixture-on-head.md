# 62 — Re-measure the 100-case quality fixture on HEAD under a controlled serialize setting

Status: ready-for-agent

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

- [ ] Rebuild from HEAD and record the exact commit SHA and build command in
      the run log (the `make cpu` / `make rocm` clobber makes `./ds4`'s mtime
      unreliable as evidence of what is built — verify, don't assume)
- [ ] Full 100-case `score_official` run on the **pipeline** path, HEAD build
- [ ] Full 100-case `score_official` run on the **TP=4** path, HEAD build
- [ ] Both runs use an **identical and explicitly recorded**
      `AMD_SERIALIZE_KERNEL` value; record it in the log filename or header
- [ ] Report `avg_nll`, `first_match`, `api_top1_rate`, `api_pair_rate` for
      both against the PRD bar (avg_nll 0.370–0.378, first_match ≥60/100,
      api_top1_rate ≥0.85, api_pair_rate ≥0.98)
- [ ] Record per-case `avg_nll` distribution (median and the <0.5 / [0.5,1) /
      [1,2) / ≥2 bucket counts), not just the mean — the mean alone hid the
      shape of the 0.7607 result
- [ ] Count `q8 fp16 cache budget exhausted` and `arena alloc failed`
      occurrences in each log and report both
- [ ] Findings recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`

## Notes for whoever picks this up

- Do **not** re-baseline the PRD target if the number falls short. Record the
  shortfall plainly. This is the mistake #43 made and #44–#48 had to correct.
- A TP=4 log's `multi-GPU layout:` block prints
  `GPU0: layers 0-42 … GPU1-3: (no transformer layers) (0.0 GB)`. This looks
  like TP sharding failed. **It has not** — that block is a pipeline-planner
  cosmetic artifact that does not describe TP placement. Confirm real
  placement via the four
  `ds4: CUDA tier N (device N) selective weights: … GiB in … ranges` lines.
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
