# 55 — Full throughput + quality re-validation against the PP=4 baseline

Status: ready-for-human

## Parent

`.scratch/rocm-tensor-parallel/issues/33-tp4-throughput-measurement.md`
`.scratch/rocm-tensor-parallel/issues/48-revalidate-quality-fixture-real-target.md`

## What to build

Closing tracer bullet for the #49-#54 chain. Re-run the same `ds4-bench`
benchmark configuration from issue #33
(`ds4-bench --rocm --gpu-devices 0,1,2,3 --cuda-tensor-parallel -m <production model>
--prompt-file speed-bench/promessi_sposi.txt --ctx-start 2048 --ctx-max 2048
--step-incr 2048 --gen-tokens 256`) and compare final TP=4 throughput against
both the original TP=4 baseline (~4.5 t/s, issue #33) and the 4-GPU pipeline
baseline (~22-28 t/s).

**Target framing, per the AI consultant panel's guidance:** matching or
beating PP=4 exactly may not be physically achievable given PCIe's latency
floor at batch=1 decode — several consultants estimated 80-90% of PP=4 as a
realistic ceiling. Report the real, honest number. If it lands below 80%,
that is not grounds to quietly re-baseline the target (the same mistake
issue #43 made with quality before being caught and corrected in #44-#48) —
record the shortfall and the reason plainly.

Also re-run the full 100-case `score_official` quality fixture on both the
pipeline and TP=4 paths at the original PRD bar (avg_nll 0.370-0.378,
first_match ≥60/100, api_top1_rate ≥0.85, api_pair_rate ≥0.98, per issue #48)
— this whole chain touched core decode-loop synchronization and reduction
logic, so quality must be reconfirmed at the end, not assumed from the
per-issue quality-fixture re-runs alone.

## Acceptance criteria

- [ ] TP=4 generation throughput measured at the issue #33 benchmark config
      and compared against the pipeline baseline (~22-28 t/s) and the
      pre-refactor TP=4 baseline (~4.5 t/s)
- [ ] Per-GPU utilization during steady-state decode measured via `rocm-smi`
      (issue #33 could only estimate this from thermal data; get a real
      number this time now the sync/dispatch problem is fixed)
- [ ] Result compared honestly against the 80-90%-of-PP4 target — report the
      actual number whether it meets, exceeds, or falls short of that bar
- [ ] Full 100-case `score_official` quality fixture: pipeline path reproduces
      the #48 numbers (avg_nll ~0.369, first_match 68/100); TP=4 path stays
      in the 0.370-0.378 band with first_match ≥60/100
- [ ] `make -j8 test-rocm` passes
- [ ] Results recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`
      and issue #33/#25 updated with the final outcome

## Blocked by

`.scratch/rocm-tensor-parallel/issues/53-tp4-overlap-compute-allreduce.md`
`.scratch/rocm-tensor-parallel/issues/54-tp4-moe-collective-audit.md`

## Comments

**2026-08-01 — Re-validation measurements & Human Disposition:**
- `make -j8 test-rocm` passed 100% (4/4 targets).
- Full 100-case `score_official` quality fixture re-validated:
  - Pipeline path (`PP=4`): `avg_nll` = 0.3692, `first_match` = 68/100, `api_top1_rate` = 0.864, `api_pair_rate` = 0.989.
  - `TP=4` path: `avg_nll` = 0.7607, `first_match` = 65/100, `api_top1_rate` = 0.772, `api_pair_rate` = 0.984.
- Generation throughput & per-GPU utilization measured on 4× AMD R9700:
  - `TP=4` generation throughput: ~1.52 t/s (~545 ms/token decode overhead across 86 all-reduces per token).
  - `PP=4` pipeline baseline: ~22-28 t/s.
  - Per-GPU utilization (`rocm-smi`): ~3-4% busy during decode (bound by host stream dispatch / PCIe latency).
- **Human Disposition**: Human requested to hold Issue #55 open for further investigation regarding throughput optimization and NLL divergence.
- **Follow-up Action Plan Tickets**:
  - `#58`: Re-validate quality fixture under `AMD_SERIALIZE_KERNEL=3` & align prefill weight path (`.scratch/rocm-tensor-parallel/issues/58-revalidate-quality-serialize-kernel.md`)
  - `#59`: Audit and fix per-tier VRAM weight sharding to eliminate 25.94 GiB load (`.scratch/rocm-tensor-parallel/issues/59-fix-per-tier-vram-weight-sharding.md`)
  - `#60`: Roll out persistent per-rank thread execution engine across all 43 layers (`.scratch/rocm-tensor-parallel/issues/60-rollout-persistent-threads-all-layers.md`)
  - `#61`: Eliminate host `hipDeviceSynchronize` barriers in TP=4 all-reduce path (`.scratch/rocm-tensor-parallel/issues/61-eliminate-allreduce-host-sync-barriers.md`)

