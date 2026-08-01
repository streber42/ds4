# 61 — Eliminate host hipDeviceSynchronize barriers in TP=4 all-reduce path

Status: closed

## Parent

`.scratch/rocm-tensor-parallel/issues/52-tp4-hand-rolled-async-allreduce.md`
`.scratch/rocm-tensor-parallel/issues/55-tp4-throughput-quality-revalidation.md`

## What to build

Eliminate host-side synchronization barriers inside the TP=4 all-reduce path.

Currently, `ds4_rocm_xdev_sync_all_devices` (`ds4_rocm_xdev.cu:475-480`) is invoked
across 5 helper call sites per layer, performing sequential `hipSetDevice` + `hipDeviceSynchronize`
loops across all 4 GPUs. This generates **860 host-blocking GPU syncs per token**, adding
~118 ms/token of host-side latency and forcing 96% GPU idle time.

Issue #52 laid the ground work for async P2P reductions using `hipStreamWaitEvent`.
Replace the sequential host `hipDeviceSynchronize` calls in `ds4_rocm_xdev.cu` with
stream-enqueued `hipStreamWaitEvent` dependencies, allowing inter-GPU staging and
accumulation to proceed entirely on per-rank HIP streams without CPU thread blocking.

## Acceptance criteria

- [x] Sequential `hipDeviceSynchronize` loops removed from `ds4_rocm_xdev_sync_all_devices` and the all-reduce path (default all-43-layers decode hot path; legacy/prefill call sites remain as fallback — see Comments)
- [x] Inter-GPU dependency ordering managed strictly via `hipStreamWaitEvent` on per-rank HIP streams
- [x] `test_rocm_xdev` probes (Test E bit-pattern probe, Test F async multi-stream probe, and the new Test G same-device stream-fence probe added for this issue) pass with 100% bitwise exactness
- [x] Generation throughput and per-GPU utilization (`rocm-smi`) measured on 4× R9700 hardware (see Comments for caveat)
- [x] `make -j8 test-rocm` passes
- [x] Findings recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`

## Blocked by

`.scratch/rocm-tensor-parallel/issues/52-tp4-hand-rolled-async-allreduce.md`

## Comments

### 2026-08-01 — Verification (human pairing session)

The implementation (per-rank secondary HIP stream + explicit `hipEventRecord`/
`hipStreamWaitEvent` fencing) had already landed in commit `1fe4829`, mislabeled
under #60's commit message — it was the stashed #61 WIP a prior GPU-blocked session
left behind, applied and committed rather than cherry-picked apart. This session
verified it for real rather than trusting that it worked:

- **Gate question, resolved with the human:** `ds4.c:27652`'s whole-token dispatch
  gate (`metal_graph_tp4_spike_layer_enabled(DS4_N_LAYER - 1)`) is the exact edit the
  prior #60 session identified as regressive and stashed rather than committed — it
  shipped anyway in `1fe4829`. Human disposition: keep as-is, not reverted.
- **AC1 scope, resolved with the human:** `ds4_rocm_xdev_sync_all_devices` still
  exists with 6 call sites (5 legacy-decode-branch, 1 prefill-batch), none reachable
  on the default all-43-layers-threaded path. Human disposition: satisfied as-is.
- **AC3/AC5:** `make ROCM_ARCH=gfx1201 rocm -j8 test-rocm` — clean build, 4/4 targets
  pass, including the new Test G same-device stream-fence probe.
- **AC4:** `ds4-bench`'s batched-prefill path is broken independent of #61 (confirmed
  via `DS4_TP4_THREADED_LAYERS=0` control and the `mini_ds4flash.gguf` low-VRAM
  isolation — pre-existing bug outside this issue's scope). Fell back to plain `ds4`
  decode on the production model: **generation 2.01 t/s, per-GPU utilization avg
  ~46-48% (peaks 100%)**, up from the pre-#61 baseline of ~1.52 t/s / ~3-4%. Output
  was incoherent due to `q8 fp16 cache budget exhausted` / arena-alloc-OOM fallback —
  the same VRAM-pressure signature #59 exists to fix, not a #61 fencing failure. The
  throughput/utilization numbers reflect real GPU-bound execution of the actual
  decode loop and stand as valid performance evidence; output correctness at
  production scale remains #59's/#55's open problem.

Full detail and commands: `.scratch/rocm-tensor-parallel/experiment-log.md`,
"Issue 61: eliminate all-reduce host sync barriers — verification" entry.
