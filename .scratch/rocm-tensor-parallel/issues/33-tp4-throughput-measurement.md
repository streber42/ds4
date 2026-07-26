# 33 — TP=4 throughput measurement and utilization

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/issues/25-widen-tp-to-4-rank.md`

## What to build

Measure TP=4 throughput and per-GPU utilization against the pipeline and TP=2 baselines. This is the proof-of-value gate: does TP=4 actually deliver the expected improvement over 2-pair pipelined TP?

**Benchmark:** `ds4-bench --rocm --gpu-devices 0,1,2,3 --cuda-tensor-parallel -m <production model> --prompt-file speed-bench/promessi_sposi.txt --ctx-start 2048 --ctx-max 2048 --step-incr 2048 --gen-tokens 256`

**Baselines to compare against:**
- 4-GPU pipeline layer-split: ~22.8 t/s generation, ~193 t/s prefill
- 4-GPU TP=2 (2-pair pipelined): ~12.3 t/s generation, ~206 t/s prefill

**Expected outcome:** TP=4 generation throughput should approach or exceed the pipeline baseline (~22.8 t/s) because all 4 GPUs compute on every token with no pipeline serialization. Per-GPU utilization should rise from ~25% (TP=2) toward >75%.

**If TP=4 doesn't beat pipeline:** Record the finding honestly in the experiment log with analysis of why. The PRD's secondary risk applies: "record the finding rather than bury it." This is not a failure — it is a measured data point.

## Acceptance criteria

- [ ] `ds4-bench` 4-GPU TP=4 throughput measured at `--ctx-start 2048 --gen-tokens 256`
- [ ] Generation throughput compared against pipeline (~22.8 t/s) and TP=2 (~12.3 t/s) baselines
- [ ] Prefill throughput measured and compared
- [ ] Per-GPU utilization measured via `rocm-smi` during steady-state decode
- [ ] All-reduce overhead measured as fraction of per-token time
- [ ] Results recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`
- [ ] If TP=4 generation < pipeline generation: analysis of bottleneck recorded
- [ ] Issue #25 parent updated with findings; closed if all criteria met

## Blocked by

- Issue #32: TP=4 quality fixture (correctness must be verified before trusting throughput numbers)
