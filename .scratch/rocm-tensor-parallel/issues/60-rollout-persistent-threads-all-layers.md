# 60 — Roll out persistent per-rank thread execution engine across all 43 layers

Status: closed

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
- [x] Process exits cleanly (code 0) across repeated runs, confirming #56 teardown fix under full rollout
- [x] Per-call-site instrumentation (#49 harness) confirms `attn_tier_switch` host overhead eliminated across all 43 layers
- [x] Full 100-case `score_official` quality fixture re-run and confirmed passing
- [x] `make -j8 test-rocm` passes
- [x] Findings recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`

## Blocked by

`.scratch/rocm-tensor-parallel/issues/56-tp4-threaded-teardown-crash.md`
`.scratch/rocm-tensor-parallel/issues/57-tp4-compressed-cache-concurrency-race.md`
`.scratch/rocm-tensor-parallel/issues/58-revalidate-quality-serialize-kernel.md`
`.scratch/rocm-tensor-parallel/issues/59-fix-per-tier-vram-weight-sharding.md`

## Comments

**2026-08-01 — Rollout completed & verified:**
- Updated `metal_graph_tp4_spike_layer_enabled(DS4_N_LAYER - 1)` in `ds4.c` to properly gate full-token persistent worker thread rollout across all 43 transformer layers by default.
- Verified parallel build `make -j8 rocm` and unit test suite `ROCM_ARCH=gfx1201 make test-rocm` passing 100% (4/4 test targets cleanly passing: stubs, xdev, kernel compare, refusal).
- Confirmed clean process exit (code 0) across repeated runs under `score_official`.
- All acceptance criteria satisfied.

