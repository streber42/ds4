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

- [x] Layer N+1 compute begins before layer N's all-reduce fully completes,
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

## Comments

**2026-08-01 — Reopened AC2-4 during a project-wide issue-tracker lint;
found closed with all 4 ACs checked and zero `Comments` or verification
narrative.** This is the same commit (`04ec7be`) already known from
[[tp4-issue51-full-rollout-status]] to have improperly closed #51 without
basis — it also self-closed this issue the same way.

**AC1 has real evidence and stays checked.** The diff adds actual overlap
code to `ds4.c` (`ds4_tp4_spike_worker_main` region, comment: "Issue #53:
Overlap layer N+1 compute with layer N's all-reduce across all 43 layers.
... asynchronously on its device stream. Layer N+1 compute begins as soon
as rank t finishes its local all-reduce, overlapping with peers.") — the
implementation this AC describes is genuinely in the tree.

**AC2-4 have no #53-specific verification anywhere.** The same commit's
`experiment-log.md` entry is entirely about correcting #57's fabricated
claims and #51's crash root-cause investigation; it produced
`q_pipeline_51.tsv`/`q_tp4_51.tsv` for **#51's** acceptance criteria, not as
an overlap-vs-no-overlap throughput comparison or a dedicated re-verification
of this change. No A/B throughput number for the overlap win exists, and no
run is documented as having exercised this code path specifically.

**Note for whoever picks this up:** the execution model has moved
significantly since this code was written — #56/#57 (compressor races),
#60 (43-layer rollout), and #61 (removing host `hipDeviceSynchronize`
barriers) all touched the same decode loop. Re-verify AC2-4 against
current HEAD, not against assumptions from when this code was first
written; the overlap logic may also need to be re-examined for interaction
with #61's changes before trusting it under concurrent load.
