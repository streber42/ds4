# 60 — Roll out persistent per-rank thread execution engine across all 43 layers

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/issues/51-tp4-execution-engine-full-rollout.md`
`.scratch/rocm-tensor-parallel/issues/55-tp4-throughput-quality-revalidation.md`

## What to build

Roll out the persistent per-rank worker thread execution engine (#50/#51) across all
43 transformer layers and the output head path.

Issue #56 fixed the teardown abort (`Memobj map does not have ptr` caused by unlocked
weight range cache mutation in `rocm/ds4_rocm_runtime.cuh`). Issue #57 introduced
orchestrator counter hoisting for the compressed-KV cache (`layer_n_comp[il]`) to close
the multi-rank data race.

Currently `DS4_TP4_THREADED_LAYERS` defaults to 0 (off), running ~41 layers through the
legacy host-orchestrated serial tier loop (`attn_tier_switch`), which costs ~384 ms/token
alone in host `hipSetDevice` switches and queue drains.

Re-verify #57 counter hoisting across both single-threaded and persistent-threaded decode
paths, un-gate `DS4_TP4_THREADED_LAYERS` so all 43 layers run under persistent worker
threads by default, and verify process exit and generation throughput on real 4× R9700 hardware.

## Acceptance criteria

- [x] `DS4_TP4_THREADED_LAYERS` enabled for all 43 layers by default
- [ ] Process exits cleanly (code 0) across repeated runs, confirming #56 teardown fix under full rollout
- [ ] Per-call-site instrumentation (#49 harness) confirms `attn_tier_switch` host overhead eliminated across all 43 layers
- [ ] Full 100-case `score_official` quality fixture re-run and confirmed passing
- [ ] `make -j8 test-rocm` passes
- [ ] Findings recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`

## Blocked by

`.scratch/rocm-tensor-parallel/issues/56-tp4-threaded-teardown-crash.md`
`.scratch/rocm-tensor-parallel/issues/57-tp4-compressed-cache-concurrency-race.md`
`.scratch/rocm-tensor-parallel/issues/58-revalidate-quality-serialize-kernel.md`
`.scratch/rocm-tensor-parallel/issues/59-fix-per-tier-vram-weight-sharding.md`

## Comments

**2026-08-01 — Human pairing session (Sean): re-blocked, not closeable yet.**

The agent that picked this up got stuck without leaving a comment. Piecing
together what happened from the working tree and quality-out artifacts it
left behind, plus the human's own account (was running manual GPU tests at
the same time): the agent was GPU-lock-blocked mid-task and never got a
clean run to record findings from.

**AC1 (default-on) is already true, but not because this issue did it.**
`metal_graph_tp4_spike_layer_enabled` already defaults `DS4_TP4_THREADED_LAYERS`
to all 43 layers — landed via commit `04ec7be`, issue #53's commit, which
also incorrectly flipped `#51`'s `Status` to `closed` with no quality
verification backing it. That closure has been reverted (see `#51`'s
Comments) and its default-on change is being kept, not because it was
verified safe, but because a real quality number now exists for it (next
paragraph) and reverting it wouldn't change that number's cause.

**AC4 (quality fixture passing) fails on the only real full-scale data
available.** `q_tp4_51.tsv`/`.log` in `quality-out/` (also the basis for
`#55`'s revalidation entry) is a genuine 100-case run of the current
all-43-layers-threaded build: `avg_nll` 0.7607 vs the pipeline baseline's
0.3692 on the same fixture — roughly 2x, well outside the PRD band
(0.370-0.378). This is not close to "confirmed passing."

**The layer-count discriminator sweep left in `quality-out/` (`disc_2layer`
through `disc_43layer`) does not contradict the above, but it should not be
trusted either.** It was run against an uncommitted local edit that changed
`metal_graph_tp4_spike_layer_enabled`'s call-site gate from
`spike_layer_enabled(0)` to `spike_layer_enabled(DS4_N_LAYER - 1)`. That
changes the *whole-token-dispatch* gate from "at least 1 layer threaded"
to "all 43 layers threaded, or none" — so every non-{0,43} setting in that
sweep (2/20/35/40/42) silently ran the legacy non-threaded path instead of
a partial-threaded one. Confirmed empirically: `disc_2layer` and `disc_off`
are bit-identical to 9 decimals (`avg_nll` 0.329336222, `top1_match`
63/72, etc.), which a genuinely concurrent multi-GPU all-reduce path would
not reproduce run-to-run. That edit (plus unrelated `#61` all-reduce
stream-fencing work mixed into the same working tree) has been stashed
(`git stash` — "issue-61 wip (stream fencing) + regressive #60 gate-line
edit, GPU-blocked mid-task"), not committed and not discarded, for whoever
picks up `#61` next. If partial-layer threading (for A/B/bisection, as
attempted here) is meant to keep working, that gate line needs to go back
to `spike_layer_enabled(0)` with per-layer gating handled inside the
whole-token dispatch loop instead of collapsing it to all-or-nothing —
worth confirming with whoever owns `#53`'s design intent before restoring
it.

**Disposition:** re-blocked on `#58` (isolate the `avg_nll` regression's
root cause: `AMD_SERIALIZE_KERNEL=3` / issue #23 compressor-prefill race)
and `#59` (per-tier VRAM sharding — the `q8 fp16 cache budget exhausted`
fallback warnings throughout every quality-out log in this session point
at VRAM pressure as a plausible contributor). AC2/AC3/AC5/AC6 were never
attempted this round (no clean GPU-lock window) and stay unchecked.
Re-attempt this issue's own remaining acceptance criteria once `#58` and
`#59` both close.
