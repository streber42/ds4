# 52 — Hand-rolled async P2P all-reduce (no RCCL)

Status: ready-for-human

## Parent

`.scratch/rocm-tensor-parallel/issues/33-tp4-throughput-measurement.md`

## HITL: design review required before implementation

This issue requires a human design review before an AFK agent implements it.
Do not proceed to implementation from this ticket alone — a human needs to
sign off on the collective algorithm design and the correctness-test plan
first.

**Why HITL:** an 8-consultant AI panel (Codex, Cursor, Gemini, Qwen3, Grok,
GLM, Mistral, MiniMax) reached high-consensus agreement that hand-rolled
async collective code is exactly the kind of change most likely to introduce
*silent* numerical corruption or deadlocks — not a crash, a wrong answer that
looks plausible. This codebase already spent five issues (#44-#48) bisecting
precisely that failure mode: commit `fa59d97` replaced a two-pass attention
kernel with a single-pass one that dropped a `warp_sum_f32` before a shuffle
broadcast, silently corrupting softmax output for weeks without crashing.
The panel was explicit that #52 carries the same risk profile and should not
be waved through on green quality-fixture numbers alone without a design
review first — the fixture is only 100 cases and didn't catch the fa59d97
regression's family of bugs immediately either.

## What to build (subject to the design review above)

Replace the blocking peer-copy all-reduce (`ds4_rocm_xdev_allreduce_f32`'s
current implementation, called from the boundary-hop / tier-switch logic
around `cur_hc_by_tier`) with an async collective, built on the persistent
per-rank thread + stream/event execution engine landed in #50/#51.

**Do not integrate RCCL.** The panel unanimously judged RCCL the wrong tool
here: it's designed for multi-process/multi-node bootstrapping (rank
discovery, communicator init, transport selection), which is unnecessary
complexity for a single-process C binary talking to exactly 4 fixed local
peers over PCIe. Instead, implement a hand-rolled async ring or tree
all-reduce using `hipMemcpyPeerAsync` queued on the per-rank streams from
#50/#51, synchronized via HIP events rather than `hipDeviceSynchronize`.

Leave a clean architectural seam so RCCL could be swapped in later as an
optional path if the hand-rolled version leaves meaningful performance on
the table at larger activation sizes (a risk Codex and Cursor flagged,
though judged low-severity for this model's per-token activation sizes).

## Acceptance criteria

- [ ] Design review completed and signed off by a human before implementation
      begins (algorithm choice — ring vs. tree — topology assumptions, and the
      correctness-test plan below)
- [ ] Standalone collective-correctness tests added that specifically probe
      the failure mode from #44-#48 (a reduction that silently drops or
      double-counts a partial sum/shuffle) — these must exist and pass
      *before* the quality fixture is treated as sufficient evidence
- [ ] Blocking peer-copy all-reduce replaced with the hand-rolled async
      ring/tree implementation, running on the #50/#51 stream/event model
- [ ] All-reduce count per token unchanged at ~86 (2 per layer) — this
      number was independently confirmed correct/expected by the panel and
      should not change here, only how each all-reduce is executed
- [ ] Per-call-site sync/cost breakdown (via #49's harness) shows the
      all-reduce path no longer blocks the host
- [ ] Full 100-case `score_official` quality fixture re-run (pipeline and
      TP=4), plus explicit note in the issue comments of what the standalone
      collective tests checked that the fixture alone would not have caught
- [ ] `make -j8 test-rocm` passes
- [ ] Findings recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`

## Blocked by

`.scratch/rocm-tensor-parallel/issues/51-tp4-execution-engine-full-rollout.md`
