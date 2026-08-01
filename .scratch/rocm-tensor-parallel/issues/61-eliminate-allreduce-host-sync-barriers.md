# 61 — Eliminate host hipDeviceSynchronize barriers in TP=4 all-reduce path

Status: ready-for-human

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

- [ ] Sequential `hipDeviceSynchronize` loops removed from `ds4_rocm_xdev_sync_all_devices` and the all-reduce path
- [ ] Inter-GPU dependency ordering managed strictly via `hipStreamWaitEvent` on per-rank HIP streams
- [ ] `test_rocm_xdev` probes (Test E bit-pattern probe & Test F stream-wait probe) pass with 100% bitwise exactness
- [ ] Generation throughput and per-GPU utilization (`rocm-smi`) measured on 4× R9700 hardware
- [ ] `make -j8 test-rocm` passes
- [ ] Findings recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`

## Blocked by

`.scratch/rocm-tensor-parallel/issues/52-tp4-hand-rolled-async-allreduce.md`
