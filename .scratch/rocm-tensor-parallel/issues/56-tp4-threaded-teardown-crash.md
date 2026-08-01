# 56 — Root-cause the TP=4 threaded-engine teardown crash

Status: closed

## Parent

`.scratch/rocm-tensor-parallel/issues/51-tp4-execution-engine-full-rollout.md`

## What to build

Both issue #50 and #51 hit the identical real-hardware crash on process
teardown when the persistent-per-rank-thread execution engine is active,
even for just the 2 uncompressed layers:

```
:0:.../device.cpp:373 : ... us:  Memobj map does not have ptr: 0x...
Aborted
```

Generation always completes first and produces correct, coherent output —
the abort happens strictly during teardown, after the #49 instrumentation
report has printed.

#50 diagnosed this (by reading, not repro) as a process-wide (non
-thread-local) BLAS-handle race in `rocm/ds4_rocm_runtime.cuh`. #51
implemented exactly the fix #50 sized (`thread_local` active-tier handles,
mutex-guarded lazy-create, per-tier teardown) and the crash reproduced
identically. #51 then ran a discriminator: the same binary with
`DS4_TP4_THREADED_LAYERS=0` (threading fully off, same rewritten teardown
code) exited clean. This proves the crash is triggered specifically by
activating the persistent worker threads, not by the BLAS-handle path or
its teardown — **#50's root cause is falsified as the sole cause.**

**Untested candidate, sized here but not yet verified:** `ds4_engine_close`
(`ds4.c`, around line 58734) calls `weights_free(&e->weights)` *before*
`metal_graph_tp4_spike_pool_shutdown()` (which joins the 4 persistent
worker threads). Each worker thread binds to its device once at startup
via `ds4_gpu_set_current_device` and may hold live memory-object
registrations against the model's host-mapped weight ranges in its own
device context. If `weights_free` unregisters those ranges from the main
thread's context while a worker thread's context still references them,
that would produce exactly this "Memobj map does not have ptr" signature.
Test this first: reorder `ds4_engine_close` so
`metal_graph_tp4_spike_pool_shutdown()` runs before `weights_free`/
`vocab_free`, rebuild, and re-run the same real-hardware threaded
generation that crashed in #51 (`DS4_TP4_THREADED_LAYERS` default-on,
`DS4_TP4_INSTRUMENT=1`, production `DeepSeek-V4-Flash-IQ2XXS` model,
`-c 64 -p "The capital of France is" -n 40`).

If that reordering does not fix it, this needs an actual minimal repro
(not just reading the code, which is what let #50's wrong diagnosis ship)
before proposing another fix — e.g. a standalone test that spins up
persistent per-device threads, does trivial GPU work, and tears down,
narrowing exactly which teardown call triggers the abort.

Do not touch the all-reduce implementation (#52's scope) or attempt the
compressed-cache rollout (issue #57) here — this issue is scoped strictly
to making the already-implemented 2-layer threaded path exit cleanly.

## Acceptance criteria

- [x] Root cause of the "Memobj map does not have ptr" teardown abort
      identified with a minimal repro or a clear causal chain, not just a
      plausible reading of the code (the mistake #50 made)
- [x] Fix implemented; the same real-hardware threaded generation that
      crashed in #51 (2 layers, `-n 40`) completes and exits cleanly
- [x] `make -j8 cpu`, `make -j8 rocm`, `make -j8 test-rocm` all pass
- [x] Findings recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`

## Comments

**#51's sized candidate (weights_free/vocab_free ordering) was falsified
statically before spending GPU time on it.** Both functions are pure host
memory operations (`memset` and `free`/`table_free` respectively) with no
GPU calls — reordering them relative to `metal_graph_tp4_spike_pool_shutdown()`
cannot affect the crash. See the 2026-08-01 experiment-log entry for the
line-number evidence.

**Actual root cause, found via `gdb -batch -ex run -ex bt -ex 'thread apply
all bt'` on the real crashing configuration (not by reading code):**
`g_model_ranges` / `g_model_range_by_offset` in `rocm/ds4_rocm_runtime.cuh`
(a `std::vector`/`std::unordered_map` pair caching host-registered weight
pointers) are written from `cuda_model_range_ptr()` with zero
synchronization. With the TP4 threaded engine active, all 4 persistent
rank threads call this concurrently to resolve weight pointers, racing the
container's internal bookkeeping (`push_back` reallocation, hash-map
insertion) even though each individual `cudaHostRegister` call succeeds.
The corruption is invisible until `cuda_model_range_release_ranges_only()`
walks the corrupted vector at teardown and calls `cudaHostUnregister` on a
pointer HIP's runtime has no record of — exactly the observed "Memobj map
does not have ptr" abort. The backtrace also shows the 4 worker threads
were already joined by the time of the crash, which independently
falsifies #51's "still-live worker thread context" hypothesis class.

**Fix:** `g_model_range_mutex` (`pthread_mutex_t`, matching this file's
existing `g_blas_tier_mutex` pattern) now guards every read and write of
`g_model_ranges`/`g_model_range_by_offset` in `cuda_model_range_ptr`,
`cuda_model_range_is_cached`, and `cuda_model_range_release_ranges_only`.
Full details, including why the sibling `g_q8_f16_ranges` caches were
confirmed out of scope (load-time-only, single-threaded), are in the
experiment log.

**Verification:** real-hardware re-run of the exact crashing configuration
(`DS4_TP4_THREADED_LAYERS=2 DS4_TP4_INSTRUMENT=1`, production
`DeepSeek-V4-Flash-IQ2XXS` model, `-c 64 -n 40`), twice with different
prompts — both exit 0, coherent output, no abort. `make -j8 cpu`,
`make -j8 rocm` (rebuilt last, after `cpu`, per the shared-binary-name
gotcha), and `make -j8 test-rocm` (4/4 targets) all pass. Full numbers and
call-site instrumentation in the 2026-08-01 experiment-log entry.
