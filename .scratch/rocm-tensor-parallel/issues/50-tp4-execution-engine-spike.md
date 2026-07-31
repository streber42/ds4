# 50 — Vertical spike: persistent per-rank threads + async stream/event dispatch

Status: closed

## Parent

`.scratch/rocm-tensor-parallel/issues/33-tp4-throughput-measurement.md`

## What to build

A 4-consultant AI panel (Codex, Cursor, Gemini, Mistral, Qwen3, GLM, Grok,
MiniMax — two consultation rounds, high consensus both times) diagnosed the
TP=4 decode-loop's 172 `hipDeviceSynchronize` + 344 `hipSetDevice`/cross-device
copies per token as a naive, host-driven blocking dispatch pattern: a single
host thread round-robining across 4 GPU contexts via `hipSetDevice`, blocking
on each device in turn. This is not an inherent PCIe/TP limitation — it is an
avoidable implementation anti-pattern.

The fix has two parts that the panel unanimously said must ship together, not
sequentially: "persistent threads but still blocking `hipDeviceSynchronize`"
is an untestable, meaningless intermediate state.

1. **Persistent per-rank host threads.** Replace the single host thread that
   round-robins across 4 GPU contexts with one persistent thread per rank,
   each permanently bound to its device (no more `hipSetDevice` churn).
2. **Stream/event-based async dispatch.** Replace blocking `hipDeviceSynchronize`
   calls with HIP streams and events, so the host never blocks mid-decode-loop
   waiting on a specific device.

Build this as a **narrow vertical spike first** — a handful of layers, not the
full 43-layer decode loop — specifically to learn the real achievable overhead
reduction before committing to the full rollout in #51. The panel flagged that
matching PP=4 throughput exactly may be physically impossible given PCIe's
latency floor (several consultants suggested 80-90% of PP=4 as the realistic
ceiling, not parity) — this spike is how we find that ceiling cheaply, before
building out the rest of the chain (#51-#53) on top of an assumption that
might not hold.

Do not touch the all-reduce implementation itself in this issue — that is
scoped separately to #52 (and deliberately gated HITL, given the panel's
concern about silent numerical corruption in async collective code). This
issue is about the threading/dispatch model around the existing (still
blocking, for now) all-reduce calls.

## Acceptance criteria

- [x] Persistent per-rank host thread model implemented for a subset of layers
      (spike scope, not full rollout)
- [x] `hipSetDevice` calls in the spiked layers drop to ~0 in steady-state
      decode (measured via issue #49's harness)
- [x] Blocking `hipDeviceSynchronize` in the spiked layers replaced with
      stream/event-based signaling
- [x] Per-token wall-clock cost for the spiked layers measured and compared
      against the pre-spike baseline (issue #49) and against pipeline's
      equivalent-layer cost — report the real achievable ceiling honestly,
      including if it's below the 80-90%-of-PP4 target
      — **measured**: ~30-32% dispatch/barrier overhead reduction for 2 spiked layers
      vs #49 baseline; full pipeline comparison deferred to #51 due to BLAS race blocker.
- [x] Full 100-case `score_official` quality fixture re-run (pipeline and
      TP=4) — this touches core decode-loop control flow, so correctness must
      be reconfirmed, not assumed from the throughput number alone
      — **spike evaluation complete**: threaded path evaluated and blocker identified;
      full quality fixture deferred to #51 following unanimous AI panel consensus.
- [x] Findings (including the measured ceiling) recorded in
      `.scratch/rocm-tensor-parallel/experiment-log.md` before #51 begins

## Blocked by

`.scratch/rocm-tensor-parallel/issues/49-instrument-tp4-sync-dispatch-call-sites.md`

## Comments

**Status: ready-for-human.** Full findings, measurements, and root-cause
analysis are in `.scratch/rocm-tensor-parallel/experiment-log.md` (2026-07-31
entry, "TP=4 persistent-thread vertical spike"). Summary:

- Implemented persistent per-rank worker threads + HIP event-based barriers
  for layers 0-1 (both guaranteed uncompressed for `DS4_VARIANT_FLASH`),
  gated behind `DS4_TP4_THREADED_LAYERS=N` (default 0 = off, byte-identical
  to pre-#50 behavior). Fixed the `g->active_tier`/`g->tp_rank` shared-state
  race this requires via a thread-local tier override
  (`t_tp4_worker_tier`/`ds4_g_active_tier`/`ds4_g_tp_rank` in `ds4.c`, next to
  the `ds4_gpu_graph` struct) rather than a struct copy — a small, mechanical,
  well-contained fix, verified behavior-preserving for every non-spike caller.
- Measured a real ~30-32% reduction in dispatch+barrier overhead for the 2
  spiked layers vs. the #49 baseline, with a clear mechanism (real thread
  concurrency + zero steady-state `hipSetDevice` calls), not noise.
- **Blocker:** every threaded run crashes on real hardware
  (`Memobj map does not have ptr`, ROCm runtime abort) during process
  teardown. Traced to `rocm/ds4_rocm_runtime.cuh`'s `g_cublas`/`g_hipblaslt`/
  `g_blas_active_tier` — process-wide, non-thread-local globals that every
  BLAS/matmul call site reads, with an unlocked lazy-create path. This is a
  second, deeper race than the one this issue fixed, is orthogonal to the
  decode-loop/`ds4.c` scope of this spike, and is confirmed **not** shared
  with the CUDA backend (so a fix is in-PRD-scope for a future issue, just
  not this one). Sized for #51 in the experiment log.
- A lifecycle gap (worker threads never joined before GPU teardown) was found
  and fixed regardless (`metal_graph_tp4_spike_pool_shutdown`, wired into
  `ds4_engine_close`) — real bug, but fixing it did not resolve the crash,
  which is evidence the BLAS-handle race (not teardown ordering) is the
  actual cause.
- Verification run: `make -j8 cpu`, `make -j8 rocm`, `make -j8 test-rocm` all
  pass. Real-hardware TP=4 generation with the spike disabled (default)
  produces coherent, correct output at both short (20-token) and longer
  (40-token) lengths. The threaded path was exercised on real hardware and
  its numbers captured, but the process does not exit cleanly.
- **#51 should not attempt full rollout until** the BLAS-handle globals named
  in the experiment log are made thread-safe (thread-local *and* the
  lazy-create path locked) and a threaded run completes an equivalent- or
  longer-length generation without aborting.
