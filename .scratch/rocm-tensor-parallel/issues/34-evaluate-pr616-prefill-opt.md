# 34 — Evaluate antirez/ds4#616 prefill optimizations for R9700

Status: closed

## Parent

`.scratch/rocm-tensor-parallel/PRD.md`

## What to evaluate

[PR #616](https://github.com/antirez/ds4/pull/616) ("Rocm prefill performance improvement on 16.3%") reports a 16.3% prefill throughput improvement (149.46 → 173.85 t/s) on a **Strix Halo (gfx1151) APU** with **unified memory** (128 GB unified RAM). We need to determine:

1. **Which specific changes in PR 616 are already present in our codebase** (i.e., we discovered them independently)
2. **Which changes are novel and might benefit the 4×R9700 discrete-GPU setup**
3. **Whether the improvements seen on Strix Halo (unified memory) generalize to discrete Radeon Pro GPUs with dedicated VRAM**

## Changes in PR 616 (from most to least likely to matter)

### A. New fused single-pass online prefill kernel (`attention_prefill_sp_online_kernel`)

**What it is:** A fused kernel that replaces the entire SGEMM→softmax→SGEMM→unpack pipeline for head_dim=512. Loads KV into shared memory once, computes scores, accumulates weighted values via online softmax with **double-precision correction** for the running max/denom.

**Launch routing (in `ds4_rocm_attention_launch.cuh`):**
```c
if (n_tokens > 1 && head_dim == 512 && !g_quality_mode) {
    // use the new sp_online kernel, bypassing cuBLAS entirely
}
```

**Our current state:** We ALREADY have a fused online prefill kernel — `attention_static_mixed_heads8_online_kernel` (line 1382 of `ds4_rocm_attention.cuh`), used when `n_tokens > 1 && head_dim == 512 && !g_quality_mode && ((window != 0u ? window : n_tokens) <= 768u)`.

**Key differences:**
| Aspect | Our kernel | PR 616 kernel |
|--------|-----------|---------------|
| Precision | Single (float) for running max/denom | Double (double) for running max/denom |
| Head grouping | 8 heads per block (8 warps) | 1 head per warp (potentially more heads in flight) |
| Score storage | Shared memory (tiled, per-stage) | Registers + shared (per-tile) |
| Tile limit | 768 tokens max | No explicit tile limit |
| Window support | yes (raw+compressed mixed KV) | raw-only |

**Novel aspect for us:** The double-precision running stats could reduce numerical drift on long prefill sequences. Currently we fall back to the cuBLAS SGEMM pipeline (which uses float SGEMM + float softmax) for sequences > 768 tokens — this is a hard cutoff that limits our large-batch prefill performance. The PR's kernel removes the tile limit entirely.

### B. Vectorized prefill score kernels (float4 loads in `attention_prefill_raw_kernel` and `attention_prefill_mixed_kernel`)

**What it is:** Replaces scalar loads `qh[d] * kv[d]` with float4 vectorized loads + explicit dot product in the GPU fallback prefill kernels (used when cuBLAS is unavailable or for head_dim != 512).

**Novel aspect for us:** Our existing GPU fallback kernels (`attention_prefill_raw_kernel` at line 11, `attention_prefill_mixed_kernel` at line 67) still use scalar loads. The float4 vectorization reduces memory controller pressure by ~4× on unified-memory systems.

**Relevance to R9700:** On discrete GPUs with dedicated VRAM and wide memory buses, the scalar-to-vectorized memory access improvement is typically smaller than on unified-memory APUs. However, it's still a strict improvement — fewer bus transactions is always better.

### C. `block_reduce_f32_max` / `block_reduce_f32_sum` utilities

**What it is:** Replaces manual shared-memory tree-reductions in the softmax kernels with cross-warp block reduction functions. Removes `__shared__ float partial[256]` and the tree-reduce loops (3 × __syncthreads per reduce).

**Novel aspect for us:** We don't have these utilities today. Our softmax kernels (`attention_prefill_raw_softmax_kernel`, `attention_prefill_mixed_softmax_kernel`, `attention_prefill_mixed_softmax_tile_kernel`) still use the manual tree-reduce pattern. These utilities are cleaner and likely slightly faster (fewer __syncthreads).

**Relevance to R9700:** Modest improvement — reduces instruction count and shared memory usage, but the softmax step is not the bottleneck in the SGEMM pipeline.

### D. hipBLASLt matmul acceleration (`ds4_rocm_hipblaslt.cuh` additions)

**What it is:** Adds strided-batched GEMM plans (`hipblaslt_strided_batched` and `hipblaslt_gemm_tn_f16_out_f32`), allowing f16 matmuls in the QKV projection and attention output to bypass rocBLAS (the ROCm equivalent of cuBLAS) and use hipBLASLt directly.

**Novel aspect for us:** We already have `ds4_rocm_hipblaslt.cuh` with basic plan caching, but we do NOT have the strided-batched or f16→f32 output variants. This could accelerate the matmul operations used during prefill.

**Relevance to R9700:** Potentially significant — if rocBLAS is not well-optimized for Radeon Pro GPUs, using hipBLASLt directly could yield real throughput gains. However, the PR author reports the hipBLASLt path alone provided only marginal improvement on Strix Halo.

## What we already discovered independently

- **float4 loads + dot4_f32 for attention dot products**: Our decode kernels already use this (lines 1335-1338, 1611-1614, 1647-1650 of `ds4_rocm_attention.cuh`), and our fused online prefill kernel (`attention_static_mixed_heads8_online_kernel`) already uses it. Only the GPU fallback prefill kernels still use scalar loads.
- **Online softmax fused kernel approach**: Our existing `attention_static_mixed_heads8_online_kernel` uses the same single-pass online softmax pattern as the PR's new kernel. We discovered this independently.
- **warp_sum_f32 / warp_max_f32**: Already in `ds4_rocm_common.cuh` (lines 350-370).

## What we have NOT yet explored

- **Double-precision correction in online softmax**: Our kernel uses float throughout; the PR uses double for running_max/running_denom. Unknown whether this matters for prefill quality on long sequences.
- **Bypassing cuBLAS entirely for head_dim=512 prefill**: Our current code still uses cuBLAS SGEMM pipeline for sequences > 768 tokens (or in quality mode). The PR skips cuBLAS for ALL head_dim=512 prefill.
- **block_reduce_f32_* utilities**: Small code quality improvement; not yet written.
- **hipBLASLt strided-batched and f16→f32 output**: Not yet implemented.

## Evaluation plan

### Step 1: Benchmark current prefill throughput on R9700

Establish a baseline with our current code:

```bash
# Using ds4-bench with a meaningful prefill-heavy workload
./ds4-bench --rocm --gpu-devices 0,1,2,3 --cuda-tensor-parallel \
  -m /var/cache/llama/ds4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --prompt-file <long-prompt> --ctx-start 2048 --ctx-max 2048 --step-incr 2048 --gen-tokens 256
```

Record prefill t/s for our current code.

### Step 2: Identify which portions of PR 616 apply cleanly

Check each change independently:

1. **block_reduce utility** — safe, mechanical, no quality risk. Can be applied to our softmax kernels directly.
2. **Vectorized fallback kernels** — replace scalar loads with float4 in `attention_prefill_raw_kernel` and `attention_prefill_mixed_kernel`. Low risk.
3. **Double-precision online softmax** — modify our existing `attention_static_mixed_heads8_online_kernel` to use double for running_max/running_denom. Quality impact unknown.
4. **Remove tile limit** — our kernel caps at 768 tokens. Extending this removes the cuBLAS fallback, which could be a net win or loss depending on rocBLAS performance.
5. **hipBLASLt matmul acceleration** — add strided-batched and f16→f32 output plans to our existing `ds4_rocm_hipblaslt.cuh`.

### Step 3: Measure each change independently

For each change that applies cleanly:
- Apply change, rebuild, re-benchmark
- Record prefill t/s
- Verify output quality (NLL on quality fixture) to detect numerical regressions

### Step 4: Compare against PR author's results

The PR author saw 16.3% improvement on Strix Halo (unified memory). If we see a similar magnitude on R9700, the changes are likely general. If we see significantly less, the unified-memory architecture is the dominant factor.

### Step 5: Document which changes are worthwhile

Some changes from PR 616 may not apply because we already have equivalent (or better) implementations. Some may not help on discrete GPUs. Some may be strictly beneficial.

## Acceptance criteria

- [x] **Step 1:** Baseline prefill t/s measured on 4×R9700 with current code
- [x] **Step 2:** Each PR change classified as (a) already done / (b) novel and applicable / (c) not applicable to our codebase
- [x] **Step 3:** At least the `block_reduce` and vectorized-fallback changes applied and measured
- [x] **Step 3:** PR's fused online kernel (or equivalent extension of our existing kernel) evaluated for tile-limit removal
- [x] **Step 4:** R9700 improvement percentage compared against PR author's 16.3%
- [x] **Step 5:** Final recommendation documented — which changes to adopt, which to skip, and why

## References

- [PR #616](https://github.com/antirez/ds4/pull/616) — full diff and discussion
- `rocm/ds4_rocm_attention.cuh` — our current prefill kernels
- `rocm/ds4_rocm_attention_launch.cuh` — prefill launch routing
- `rocm/ds4_rocm_common.cuh` — warp sum/max utilities (ours)
- `rocm/ds4_rocm_hipblaslt.cuh` — our existing hipBLASLt code
- Issue #29: TP=4 attention path (current working context)
- Issue #07: TP prefill path (original correctness port)
