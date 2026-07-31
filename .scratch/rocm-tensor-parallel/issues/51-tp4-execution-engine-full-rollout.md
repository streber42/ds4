# 51 — Roll out the persistent-thread/async-stream execution engine to all 43 layers

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/issues/33-tp4-throughput-measurement.md`

## What to build

Issue #50 spiked the persistent-per-rank-thread + async-stream/event dispatch
model on a subset of layers and reported the real achievable overhead
reduction. Assuming that spike showed a worthwhile win (if it didn't, stop
here and reassess rather than proceeding — see #50's honesty requirement),
extend the same execution engine to all 43 transformer layers plus the
output head and embedding paths, replacing the TP=4 decode loop's remaining
`hipSetDevice`/`hipDeviceSynchronize` call sites identified in #49.

As with #50, do not change the all-reduce implementation itself here — the
existing (blocking) all-reduce calls should simply run inside the new
persistent-thread/async-stream execution model. Swapping to an async
collective is #52's job, deliberately kept separate and HITL-gated.

## Acceptance criteria

- [ ] All 43 layers (plus output head/embedding) run under the persistent
      per-rank thread model with stream/event-based dispatch — no remaining
      `hipSetDevice` churn in steady-state decode
- [ ] Per-call-site sync counts (via #49's harness) confirm the reduction
      generalizes from the #50 spike to the full model, not just the spiked
      subset
- [ ] End-to-end TP=4 decode throughput measured and compared against both
      the pre-refactor baseline (issue #33: ~4.5 t/s) and the pipeline
      baseline (~22-28 t/s) — report the real number, whatever it is
- [ ] Full 100-case `score_official` quality fixture re-run (pipeline and
      TP=4)
- [ ] `make -j8 test-rocm` passes
- [ ] Findings recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`

## Blocked by

`.scratch/rocm-tensor-parallel/issues/50-tp4-execution-engine-spike.md`
