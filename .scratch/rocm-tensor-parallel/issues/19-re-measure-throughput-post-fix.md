# 19 — Re-measure TP throughput after dispatch race fix

Status: ready-for-human

**What to build:** Repeat the 4-GPU `ds4-bench` sweep after fixing the dispatch race (#18) so we know the real TP-vs-pipeline throughput gap. Also re-run the quality fixture without `AMD_SERIALIZE_KERNEL=3` to confirm the fix holds.

Current baselines from 2026-07-25 (pre-fix):
- Pipeline default: 21.99 t/s
- Pipeline serialized: 13.57 t/s
- TP default: 12.17 t/s
- TP serialized: 6.77 t/s

Post-fix expectation: TP default should match or exceed the TP-serialized baseline (~13 t/s or better), and pipeline default should match its own serialized baseline (~22 t/s).

## Acceptance criteria

- [x] `ds4-bench` 4-GPU TP default throughput measured at `--ctx-start 2048 --gen-tokens 256`
- [x] `ds4-bench` 4-GPU pipeline default throughput measured at the same frontier
- [ ] Quality fixture (`make rocm-quality`, TP mode) run without `AMD_SERIALIZE_KERNEL=3` matches the serialized baseline
- [x] Results recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`

## Blocked by

- `.scratch/rocm-tensor-parallel/issues/18-fix-cross-device-dispatch-race.md` — fix must land first so the measurement is meaningful (and issue 23 for un-shimmed quality match)

## Comments

**2026-07-26 — Throughput sweep measured; marked ready-for-human due to unresolved quality fixture gap (blocked on Issue 23).**

**Throughput measurement results (4x R9700, gfx1201, `--ctx-start 2048 --gen-tokens 256`):**
- **4-GPU Pipeline default:** 22.14 tok/s generation (prefill 94.97 tok/s), matching the expected baseline (~22 t/s).
- **4-GPU TP default:** 12.44 tok/s generation (prefill 104.68 tok/s).

**Analysis of TP vs. Pipeline gap:**
1. Pipeline generation (22.14 t/s) remains ~1.78x faster than 4-GPU pipelined TP generation (12.44 t/s).
2. 4-GPU TP prefill (104.68 t/s) is slightly faster than pipeline prefill (94.97 t/s).

**Quality fixture status (Un-shimmed default mode):**
As detailed in Issue 18, fixing the cross-device peer-copy sync in `ds4_rocm_xdev_copy` was necessary but insufficient to resolve default-mode quality degradation (`avg_nll` 3.086 vs 0.370 serialized baseline in TP mode; `avg_nll` 1.295 vs 0.374 in pipeline mode). The remaining data race is an intra-device race in `ds4_gpu_compressor_prefill_tensor` spun off to Issue 23 (`23-fix-same-device-compressor-prefill-race.md`). Until Issue 23 resolves that race, the quality fixture criterion without `AMD_SERIALIZE_KERNEL=3` cannot be checked off.
