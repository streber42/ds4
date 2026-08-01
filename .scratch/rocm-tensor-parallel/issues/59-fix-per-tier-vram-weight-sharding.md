# 59 — Audit and fix per-tier VRAM weight sharding to eliminate 25.94 GiB load

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/issues/55-tp4-throughput-quality-revalidation.md`

## What to build

Audit `ds4.c` selective weight placement and device caching to fix the per-tier
weight footprint in TP=4 mode.

Currently, selective weight caching loads **25.94 GiB per tier** (1328 ranges) on all
4 GPUs. For an 81 GiB production model, proper 4-way sharding (expert, head, and
vocab sharding) should reduce per-rank weight storage to ~20.2 GiB. The extra ~5.7 GiB
per tier indicates that certain sharded tensors are being fully replicated across all tiers
or cached improperly during model loading.

On 32GB GPUs (29.79 GiB available VRAM after 2.0 GiB scratch reservation), 25.94 GiB
of weight cache leaves only ~0.35 GiB free VRAM per GPU. This triggers
`ds4: ROCm model arena alloc failed for moe_*: out of memory` during model initialization
and forces MoE weights to fall back to host-mapped memory over PCIe.

Fixing per-tier sharding to reduce weight footprint to ~20.2 GiB will reclaim ~5.7 GiB
VRAM per GPU, eliminating model arena host fallbacks and ensuring all weights sit cleanly in
fast VRAM.

## Acceptance criteria

- [ ] Audit tensor sharding logic in `ds4.c` (`engine_append_device_cache_span` /
      `cuda_tp4` placement paths) to ensure sharded weights are not replicated across tiers
- [ ] Measured per-tier selective weight load in TP=4 mode reduced from 25.94 GiB to ~20.2 GiB
- [ ] Verification run confirms `ds4: ROCm model arena alloc failed` warnings are gone
- [ ] `make -j8 test-rocm` passes
- [ ] Findings recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`

## Blocked by

`.scratch/rocm-tensor-parallel/issues/55-tp4-throughput-quality-revalidation.md`
