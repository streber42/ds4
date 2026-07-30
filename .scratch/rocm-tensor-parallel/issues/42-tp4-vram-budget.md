# 42 — Free VRAM budget for TP=4 to eliminate host-mapped MoE fallback

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/issues/32-tp4-quality-fixture.md`
`.scratch/rocm-tensor-parallel/issues/40-option-b-row-split-batch-prefill.md`

## Problem

Each R9700 has 29.79 GiB VRAM, reduced to 27.79 GiB post-overhead (2.00 GiB subtracted for per-tier graph scratch). The model loads 25.94 GiB of selective weights per GPU, leaving only ~1.85 GiB for runtime allocations. When `moe_gate` (1024 MiB) tries to allocate, it fails:

```
ds4: ROCm model arena alloc failed for moe_gate (1024.00 MiB chunk): out of memory
```

This forces MoE weight access through host-mapped fallback (`cuda_model_range_ptr_from_fd`), which is the leading hypothesis for the persistent ~1.72 avg_nll quality gap (see issue #41).

## VRAM budget (per GPU)

| Category | Size | Notes |
|----------|------|-------|
| Total VRAM | 29.79 GiB | |
| Per-tier overhead | -2.00 GiB | Graph scratch reserve |
| Available | 27.79 GiB | |
| Selective weights | -25.94 GiB | 1328 ranges per tier |
| Remaining | ~1.85 GiB | Insufficient for 1 GiB moe_gate |
| q8 fp16 cache | 0 | Exhausted immediately |

## Approaches to free VRAM

### Option A: Reduce per-tier graph scratch overhead (easiest)

The 2.00 GiB per-tier overhead is a conservative reservation. The actual graph scratch usage may be lower. If we can reduce this to 1.00 GiB, that frees ~1 GiB — enough to fit `moe_gate`.

**Risk**: Graph scratch OOM during long prefill.

### Option B: Reduce q8 fp16 cache reserve

The reserve defaults to 4.00 GiB per GPU. With the cache already exhausted immediately (requested 64 MiB, 0 cached), this is not consuming VRAM — but the reserve parameter may affect allocation decisions elsewhere.

### Option C: Reduce selective weight footprint

The 25.94 GiB of weights per tier is driven by full replication of all layers. If some tensors could be sharded (reverting to the earlier approach but only for specific VRAM-critical tensors), this frees budget.

**Risk**: Would reintroduce the all-reduce FP noise the project was trying to eliminate.

### Option D: Model quantization or reduced context

Switch to a smaller quant (IQ1_S instead of IQ2/Q2_K) or reduce max context to free VRAM.

**Risk**: Changes the reference baseline, may not be acceptable.

## What to build

1. **Audit actual graph scratch usage**: Run with `DS4_DEBUG_MEMORY=1` to see peak scratch consumption. Determine the minimum safe overhead value.
2. **Test reduction**: Reduce `per_tier_overhead` incrementally (2.00 → 1.50 → 1.00 GiB), verify TP=4 quality runs without OOM.
3. **If successful**: Re-run quality fixture and check whether host-mapped fallback is eliminated and scores improve.

## Acceptance criteria

- [ ] Actual per-tier scratch usage measured
- [ ] Per-tier overhead reduced to minimum safe value
- [ ] `moe_gate` allocation succeeds (no host-mapped fallback for MoE weights)
- [ ] Quality fixture re-run and scores compared to pipeline reference
- [ ] If quality gap closes, issues #40 and #32 can be resolved

## Blocked by

- 4× R9700 GPU access
- Quality fixture infrastructure
- Issue #41 creates diagnostic context if host-mapped vs cached numerical equivalence is confirmed
