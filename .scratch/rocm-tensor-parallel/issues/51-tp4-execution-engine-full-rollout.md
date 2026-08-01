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
`.scratch/rocm-tensor-parallel/issues/56-tp4-threaded-teardown-crash.md`
`.scratch/rocm-tensor-parallel/issues/57-tp4-compressed-cache-concurrency-race.md`
`.scratch/rocm-tensor-parallel/issues/62-remeasure-quality-fixture-on-head.md`

## Comments

**2026-08-01 — Audit: the reopen rationale below rests on a stale measurement.**
(Human authorization given to override prior dispositions and make issues
reflect reality.)

The most recent entry below reopened this issue on the grounds that "a real full
100-case run now exists (`q_tp4_51.tsv`/`.log`) and shows `avg_nll` 0.7607
against pipeline's 0.3692, so the rollout is real but not yet passing quality."

**The reopen was the right call; that specific justification was not.** That run
is timestamped 05:46, while `1fe4829` (the 43-layer rollout) landed at 10:38 and
`0cb9cf3` (#61's async all-reduce) at 11:38. It cannot describe the rollout it
was cited to evaluate. The honest state is that the rollout's quality is
**unmeasured**, not measured-and-failing.

`Blocked by` has been simplified: `#58` and `#59` are replaced by `#62`, which
establishes a trustworthy HEAD number. That is what this issue actually needs to
re-attempt verification — #58 and #59 are diagnosis/fix issues for a gap that may
not survive re-measurement, and gating on them was over-constraining. (Under
this project's literal-only `Blocked by` semantics, they also formed part of a
dependency cycle with `#55`; see [[ralph-issue-blocked-by-must-be-explicit]].)

One correction to the entry below that matters for future work: it points at the
`q_tp4_51_disc_*` sweep as investigation carried forward. That sweep is **not a
layer-count discriminator** — all eight runs share byte-identical warning counts
(129) and near-identical `avg_nll` (0.329–0.347), the signature of eight runs of
a single code path, consistent with `ds4.c:27652`'s whole-token gate collapsing
partial `DS4_TP4_THREADED_LAYERS` values onto the legacy path. Don't build on it.

Evidence: `experiment-log.md`, "Audit of the 0.7607 TP=4 quality number".

**2026-08-01 — Human disposition (Sean):** left open rather than closed.
Status reset to `ready-for-agent`, but `Blocked by` now literally lists
`#56` and `#57` (in addition to `#50`) so the scheduler cannot re-dispatch
this until both of those close — per the literal-only `Blocked by`
semantics that already bit this issue once (see #53's missing dependency,
below). Once #56 and #57 both land, re-attempt the full 43-layer rollout
under this same issue rather than opening a new one.

**Status: ready-for-human (superseded above).** This issue was picked up in a tangled state:
the previous autonomous run (a different harness, `gemini-3.6-flash-high`
via "antigravity-cli") timed out mid-task and left uncommitted changes,
and issue #53 — which should have declared `Blocked by: #51` but didn't —
was dispatched concurrently by the same engine and also timed out editing
the same files. Full detail on untangling that, and everything below, is
in `.scratch/rocm-tensor-parallel/experiment-log.md`'s 2026-08-01 entry.
Summary:

- **#53's missing dependency is fixed**: its issue file now declares
  `Blocked by: #51` (in addition to `#52`), and its status was reset from
  `ready-for-human` (a findings-free crash, not real progress) back to
  `ready-for-agent`.
- **The BLAS-handle thread-safety fix #50 sized for this issue is
  implemented and retained** (`rocm/ds4_rocm_runtime.cuh`,
  `rocm/ds4_rocm_hipblaslt.cuh`): `g_cublas`/`g_hipblaslt`/
  `g_blas_active_tier` are `thread_local`; the owned per-tier handle
  arrays stay process-wide, now mutex-guarded on first creation; teardown
  destroys every tier ever created, not just the calling thread's active
  one. `make -j8 cpu`, `make -j8 rocm`, `make -j8 test-rocm` all pass.
- **But it does not fix the crash.** Re-running the same 2-layer threaded
  scope #50 spiked on real hardware reproduced the identical
  `Memobj map does not have ptr` abort on process teardown, after
  generation completed and printed correct output. A discriminator run
  (same binary, `DS4_TP4_THREADED_LAYERS=0`, threading fully off) exited
  clean — proving the crash is triggered by activating the persistent
  worker threads specifically, not by the BLAS fix or its teardown
  rewrite. **#50's root-cause diagnosis (found by reading, never
  repro'd) is therefore falsified as the sole cause.** An untested
  candidate is recorded in the experiment log: `ds4_engine_close` frees
  model weights (`weights_free`) *before* joining the worker threads
  (`metal_graph_tp4_spike_pool_shutdown`), which may unregister
  host-mapped ranges from the wrong thread's context.
- **A second, independent blocker was found**, unrelated to the crash:
  the compressed-KV-cache row counter (`layer_n_comp[il]`) is read by
  every rank but incremented only by rank 0, in the same function that
  now runs concurrently across ranks — a same-phase data race that #50's
  `ds4_layer_compress_ratio(il) == 0` gate (layers 0-1 only, for
  `DS4_VARIANT_FLASH`) was specifically excluding. This is why "all 43
  layers" cannot simply drop that gate; a discarded diff from the earlier
  stuck run had done exactly that, silently, which is part of why it was
  discarded rather than salvaged (see experiment log).
- **`metal_graph_tp4_spike_layer_enabled` was reverted to opt-in**
  (`DS4_TP4_THREADED_LAYERS` unset/0 = off, same default as before this
  issue). The BLAS fix ships anyway, since it's correct and inert while
  the threaded path stays opt-in.
- The 100-case quality fixture was **not run**, per #50's own reasoning
  (still valid): a build whose only tested threaded configuration
  reproducibly aborts would not produce a trustworthy signal.

**Real-hardware instrumentation data** (4×AMD Radeon AI Pro R9700,
production `DeepSeek-V4-Flash-IQ2XXS` 86GB model, `-c 64 -p "The capital
of France is"`, `DS4_TP4_INSTRUMENT=1`):

2-layer threaded run (`-n 40`, crash-terminated after 39 tokens generated
and the report printed — throughput and per-call-site numbers are real):
prefill 0.28 t/s, generation 1.52 t/s, total measured sync/dispatch
545.1 ms/token. `spike_attn_dispatch`/`spike_attn_barrier`/
`spike_moe_dispatch`/`spike_moe_barrier` (2 layers) totaled ~11.2 ms/token;
`attn_tier_switch` (the still-unthreaded 41 layers) alone cost
242.4 ms/token — confirming the 2-layer win is real but small against the
whole model's budget.

43-layer non-threaded discriminator run (`-n 20`, exited clean): 19 tokens,
total measured sync/dispatch 549.3 ms/token, `attn_tier_switch`
246.5 ms/token, `moe_tier_switch` 138.9 ms/token — consistent with #49's
original baseline, confirming the discriminator's non-crash was not from a
degraded/different code path.

**This issue cannot close as "rolled out" in any form yet.** Two follow-up
issues were opened to track the blockers found here, both scoped so they
don't require #51 itself to close first:

- `.scratch/rocm-tensor-parallel/issues/56-tp4-threaded-teardown-crash.md`
  — root-cause the "Memobj map does not have ptr" abort, starting from the
  `weights_free`-ordering hypothesis above.
- `.scratch/rocm-tensor-parallel/issues/57-tp4-compressed-cache-concurrency-race.md`
  — design and implement a concurrency-safe fix for the
  `layer_n_comp[il]` race, to unblock rollout past layers 0-1.

The BLAS thread-safety fix from this issue was committed as a standalone,
correct, tested contribution despite not unblocking the issue on its own.

**2026-08-01 — Human disposition (Sean): reopened.** Commit `04ec7be`
(issue #53's own commit — "Overlap layer N+1 compute with layer N's
all-reduce") flipped this issue's `Status` to `closed` and, in the same
change, un-gated `metal_graph_tp4_spike_layer_enabled`'s default from
opt-in to all 43 layers — i.e. it silently did this issue's rollout job
under a different issue's commit message, with no comment here and no
quality re-verification backing the close. That closure had no basis:
the disposition recorded immediately above this entry is still the real
one (open, blocked on `#56`/`#57`, re-attempt full rollout once both
close), and both of those closed 2026-08-01. The default-on code change
itself is not being reverted — a real full 100-case run now exists
(`q_tp4_51.tsv`/`.log`, also referenced from #55) and shows `avg_nll`
0.7607 against pipeline's 0.3692, so the rollout is real but not yet
passing quality. That gap is now tracked by `#58` (isolate root cause:
`AMD_SERIALIZE_KERNEL=3` / issue #23 compressor-prefill race) and `#59`
(per-tier VRAM sharding, likely contributor via the q8 fallback path).
Status reset to `ready-for-agent`; `Blocked by` below gets `#58` and
`#59` added so re-dispatch waits for both, same literal-only-dependency
pattern this issue already required once for `#56`/`#57`. See
`.scratch/rocm-tensor-parallel/issues/60-rollout-persistent-threads-all-layers.md`
for the issue that was carrying this same investigation forward and got
GPU-blocked (human doing manual testing on the same hardware) before it
could finish.
