# 53 — Overlap layer N+1 compute with layer N's all-reduce

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/issues/33-tp4-throughput-measurement.md`

## What to build

With the async ring/tree all-reduce from #52 running on streams, overlap the
communication for layer N's all-reduce with compute for layer N+1 where data
dependencies allow, instead of waiting for each all-reduce to fully complete
before starting the next layer's compute.

This is the final-mile throughput step in the chain — expected to be a
smaller win than #50/#51 (dispatch/sync elimination) or #52 (async
collective), since most of the overhead this chain targets is host-side
blocking, not the underlying communication latency itself.

## Acceptance criteria

- [ ] Layer N+1 compute begins before layer N's all-reduce fully completes,
      for the dependency-safe portion of the computation
- [ ] Per-token decode throughput measured and compared against #51/#52's
      numbers — report the actual overlap win, which may be small
- [ ] Full 100-case `score_official` quality fixture re-run (pipeline and
      TP=4) — overlap logic is easy to get subtly wrong in the same
      "silent corruption" way as #52, so do not skip this
- [ ] Findings recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`

## Blocked by

`.scratch/rocm-tensor-parallel/issues/52-tp4-hand-rolled-async-allreduce.md`
`.scratch/rocm-tensor-parallel/issues/51-tp4-execution-engine-full-rollout.md`
