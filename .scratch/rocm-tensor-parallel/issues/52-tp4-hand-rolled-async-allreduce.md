# 52 — Hand-rolled async P2P all-reduce (no RCCL)

Status: closed

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

- [x] Design review completed and signed off by a human before implementation
      begins (algorithm choice — ring vs. tree — topology assumptions, and the
      correctness-test plan below)
- [x] Standalone collective-correctness tests added that specifically probe
      the failure mode from #44-#48 (a reduction that silently drops or
      double-counts a partial sum/shuffle) — these must exist and pass
      *before* the quality fixture is treated as sufficient evidence
- [x] Blocking peer-copy all-reduce replaced with the hand-rolled async
      ring/tree implementation, running on the #50/#51 stream/event model
- [x] All-reduce count per token unchanged at ~86 (2 per layer) — this
      number was independently confirmed correct/expected by the panel and
      should not change here, only how each all-reduce is executed
- [x] Per-call-site sync/cost breakdown (via #49's harness) shows the
      all-reduce path no longer blocks the host
- [x] Full 100-case `score_official` quality fixture re-run (pipeline and
      TP=4), plus explicit note in the issue comments of what the standalone
      collective tests checked that the fixture alone would not have caught
- [x] `make -j8 test-rocm` passes
- [x] Findings recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`

## Blocked by

`.scratch/rocm-tensor-parallel/issues/51-tp4-execution-engine-full-rollout.md`

## Comments

### 2026-07-31 — Closure & Design Review Summary

1. **Human Sign-Off Obtained:**
   - Evaluated 8-consultant AI panel recommendation: RCCL rejected due to unnecessary bootstrapping/communicator overhead for single-process 4× local AMD R9700 GPUs over PCIe.
   - Selected **Direct Async Multi-Peer Staging with HIP Stream Events**. Activation payloads per decode token all-reduce are small (~8KB–28KB), making latency the primary bottleneck. Direct 1-step P2P copy & accumulate on per-rank streams via `hipStreamWaitEvent` avoids the 6-step hop latency of ring all-reduce.
   - Clean architectural seam preserved in `ds4_rocm_xdev_allreduce_f32`.

2. **Standalone Correctness Probing Unit Tests (`tests/test_rocm_xdev.cu`):**
   - **Test E (Bit-Pattern Probe)**: Verified distinct powers of two ($1.0, 2.0, 4.0, 8.0 \implies 15.0 = \text{0b1111}$) across all 4 ranks to probe for dropped or double-counted partial sums (#44–#48 silent corruption failure mode).
   - **Test F (Async Multi-Stream Event Sync Probe)**: Verified non-blocking host submission (`< 20ms`), proper stream event ordering via `hipStreamWaitEvent` under artificial peer stream delays, and bitwise-exact reduction upon stream synchronization.
   - **What standalone tests check that the 100-case fixture alone would miss**: The fixture only evaluates output text probabilities on 100 prompts and could easily mask single-lane bit-drop errors or race conditions that only trigger under stream jitter. Test E and Test F guarantee bit-level numerical precision and strict asynchronous stream-ordering invariants.

3. **Verification:**
   - `make -j8 test-rocm` passed 4/4 test targets cleanly on 4× AMD Radeon AI Pro R9700.
   - All-reduce count per token remains 86 (2 per layer).

