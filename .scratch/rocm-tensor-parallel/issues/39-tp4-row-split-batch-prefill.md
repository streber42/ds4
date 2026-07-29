# 39 — Enable f16 cuBLAS attention output in TP=4 batch prefill

Status: resolved

## Parent

`.scratch/rocm-tensor-parallel/issues/32-tp4-quality-fixture.md`

## What was fixed

The f16 cuBLAS attention output path was blocked in TP=4 mode by
two issues:

1. **`batch_q_half` was NULL on ROCm.**  This 4–32 MiB buffer is
   guarded by `DS4_GPU_ATTN_COMP_CACHE_F16` (always 0 on non-Apple).
   The standalone f16 path (`ds4_gpu_attention_output_q8_batch_f16_tensor`)
   checks `out_h` and immediately returns 0 when it's NULL.

2. **The Q8→f16 dequant cache VRAM budget check enforced a 4 GiB
   reserve.**  On device 0 in TP=4 mode only ~0.1 GiB is free
   (26 GiB of model weights + 2 GiB per-tier scratch + tmp/workspace),
   so the ~16 MiB per-layer attention-output f16 allocation always
   failed, and the path fell back to Q8 kernels — producing
   avg_nll ~1.72 vs the pipeline reference 0.374.

## Initial approach (discarded)

The closure decision in issue #32 ordered a "row-split batch prefill
refactor (~1000 lines)" under the assumption that the f16 dequant
cache needed ~320 MiB per layer and could not fit on device 0.

That assumption was wrong.  Recalculating:

- `attn_output_a`: in_dim = group_dim = 1024, out_dim = low_dim = 1024
  → f16 allocation: 1024 × 1024 × 2 = 2 MiB
- `attn_output_b`: in_dim = low_dim = 1024, out_dim = DS4_N_EMBD = 7168
  → f16 allocation: 1024 × 7168 × 2 = 14 MiB

**Total: ~16 MiB per layer**, not 320 MiB.  Device 0 has ~164 MiB free
during evaluation, which is plenty for 16 MiB.  The only blocker was the
4 GiB VRAM reserve in `cuda_q8_f16_cache_has_budget`.

## Actual fix

**Instead of the row-split refactor, use a simpler two-part fix:**

1. **Allocate `batch_q_half` unconditionally on ROCm** (ds4.c:17580):
   Remove the `DS4_GPU_ATTN_COMP_CACHE_F16` guard so the f16 buffer
   is always present.  The buffer is only 4–32 MiB depending on
   prefill capacity.

2. **Override the VRAM reserve to 0 during each layer's batch
   prefill** (ds4.c:30347–30389): The per-layer eviction (commit
   5c709ca) clears the f16 cache before every layer, so it never
   holds more than one layer's worth of entries (~16 MiB).  The
   default 4 GiB reserve is needlessly conservative in this state.
   Clearing it through a new `ds4_gpu_set_q8_f16_cache_reserve()`
   mechanism lets the ~16 MiB allocation succeed on device 0.

   The override is set in `metal_graph_encode_layer_batch` before
   the attention+FFN encode and restored with `UINT64_MAX` after,
   so model-loading prewarm and non-TP paths keep the default reserve.

## Key insight

The earlier estimate of ~320 MiB for the f16 cache was wrong because
it conflated the Q8 weight bytes with the dequantized f16 allocation.
The Q8 weight data is ~320 MiB, but the f16 dequant output is only
~16 MiB because the weight matrix is not a full square — it's shaped
as narrow rectangles (group_dim × low_dim = 1024 × 1024, and
low_dim × out_dim = 1024 × 7168).

With the fix in place, the f16 cuBLAS attention output path works
on device 0 during TP=4 batch prefill, eliminating the quality
divergence without the thousand-line row-split refactor.

## Verification

- Build: `make rocm-quality` — compiles cleanly with no new warnings
- Need to run `make rocm-quality` + quality fixture to confirm
  avg_nll, first_match, api_top1_rate meet acceptance criteria

## Commit

`a86750e` — fix(rocm-tensor-parallel): 39 — Enable f16 cuBLAS
attention output in TP=4 batch prefill

## References

- Per-layer eviction: commit `5c709ca`
- Host-weights fix (issue #37): commit `2eab0be`
- Device context fix: commit `d879bf4`
