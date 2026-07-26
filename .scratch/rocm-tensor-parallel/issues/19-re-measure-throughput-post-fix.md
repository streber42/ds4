# 19 — Re-measure TP throughput after dispatch race fix

**What to build:** Repeat the 4-GPU `ds4-bench` sweep after fixing the dispatch race (#18) so we know the real TP-vs-pipeline throughput gap. Also re-run the quality fixture without `AMD_SERIALIZE_KERNEL=3` to confirm the fix holds.

Current baselines from 2026-07-25 (pre-fix):
- Pipeline default: 21.99 t/s
- Pipeline serialized: 13.57 t/s
- TP default: 12.17 t/s
- TP serialized: 6.77 t/s

Post-fix expectation: TP default should match or exceed the TP-serialized baseline (~13 t/s or better), and pipeline default should match its own serialized baseline (~22 t/s).

## Blocked by

- #18 — fix must land first so the measurement is meaningful

## Status

ready-for-agent

- [ ] `ds4-bench` 4-GPU TP default throughput measured at `--ctx-start 2048 --gen-tokens 256`
- [ ] `ds4-bench` 4-GPU pipeline default throughput measured at the same frontier
- [ ] Quality fixture (`make rocm-quality`, TP mode) run without `AMD_SERIALIZE_KERNEL=3` matches the serialized baseline
- [ ] Results recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`
