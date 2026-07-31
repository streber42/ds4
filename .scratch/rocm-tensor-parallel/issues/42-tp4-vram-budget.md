# 42 — Free VRAM budget for TP=4 to eliminate host-mapped MoE fallback

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/issues/32-tp4-quality-fixture.md`
`.scratch/rocm-tensor-parallel/issues/40-option-b-row-split-batch-prefill.md`
`.scratch/rocm-tensor-parallel/issues/41-host-mapped-moe-weight-precision.md` (closed — root cause)

## Update (2026-07-31, from issue #41)

Issue #41's investigation confirmed the mechanism behind this OOM and
found it is larger than "moe_gate specifically doesn't fit": every
weight tensor resolved during batch prefill — attention, router, shared
expert, and MoE alike — goes through this same arena/host-mapped-fallback
path, because `ds4_gpu_set_use_host_weights(1)` (set unconditionally at
the top of `metal_graph_encode_layer_batch`) bypasses the primary,
already-populated per-device selective weight cache entirely during batch
prefill and re-resolves everything via this separate arena instead.
That arena duplicates VRAM already spent on the primary cache and
overflows almost immediately (observed: layer 0, not deep into the
network). Once it overflows, the failure is permanent for the rest of
the process (`g_model_cache_full` is a global latch, not per-device).

**Implication for this issue's fix options:** Option A (reduce per-tier
overhead) will buy some headroom but the real waste is the redundant
arena copy of data that's already cached — freeing ~1 GiB via overhead
tuning may just delay the OOM by a few layers rather than eliminate it.
A fix that makes `cuda_resolve_weight_ptr` check the primary selective
cache *before* falling through to the arena (i.e., don't set
`g_use_host_weights` blindly, or have `cuda_model_range_ptr` try
`ds4_gpu_lookup_cache_strict` even when the flag is set) would eliminate
the redundant allocation at the source rather than just buying more
headroom for it. See #41's Comments for the full trace and code path.

**Caution:** this is in tension with issue #37's stated intent. #37 set
`g_use_host_weights` specifically to force pipeline and TP=4 onto a
*consistent* resolution path after finding the primary per-device cache
puts the same weight bytes at *different* VRAM addresses in the two
configs (2.37e-4 Q8-matmul divergence per #37). #37's own fix comment
assumed this consistency came from a "model image pointer" shared across
configs — #41 found that path is dead code, so the real (accidental)
consistency #37 measured came from both configs draining into the same
arena/host-register fallback instead. Restoring primary-cache lookups
when the flag is set could resurrect #37's original divergence unless
the primary cache is made to put weights at consistent addresses across
configs too. Whoever implements this fix should re-run #37's per-layer
diff (`diagnose-prefill.sh`) to confirm no regression.

**Before implementing a VRAM-budget fix, run the falsifying test.** The
link between the fallback and the avg_nll gap is correlational, not yet
causally isolated — every configuration measured in #41 had the fallback
firing *and* a broken score, but no run with the fallback absent was
measured. Run `score_official` with `DS4_ROCM_WEIGHT_PATH_STATS=1` in a
configuration that reports **zero** "arena-full skip" / "host-register
PCIe-map" lines (e.g. cap `--gpu-vram` well below the model's per-tier
footprint so the packer leaves deliberate headroom, accepting SSD
streaming or a smaller effective model as the cost). If that run still
scores ≈1.5-2.0, this root cause is exonerated and a VRAM-budget fix here
will not close the gate — stop and re-open the investigation instead of
spending hours on this issue's implementation. This check costs one run,
not a redesign, and four prior "confirmed" fixes in this project (f16
cuBLAS, cache reserve, row-split, attention gate) each left the score
unmoved, so verifying before building is cheap insurance.

This VRAM pressure is not TP=4-specific — see
[issue #43](43-pipeline-vram-accounting-regression.md), which found the
same fallback now also fires in pipeline mode. Note: #43's regression
predates commit `414f9fc` (an earlier attribution to that commit was
tested on real hardware and disproven — see #43's Root Cause section);
the exact introduction point is still unbisected. Any fix here should be
validated against both TP=4 and pipeline configurations.

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
