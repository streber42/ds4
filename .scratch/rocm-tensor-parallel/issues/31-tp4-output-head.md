# 31 — TP=4 output head (correct sampling)

Status: closed

## Parent

`.scratch/rocm-tensor-parallel/issues/25-widen-tp-to-4-rank.md`

## What to build

Wire up the output head (vocabulary projection + token sampling) for TP=4: shard the vocabulary rows V/4 per rank, and implement distributed decode sampling.

**Vocabulary sharding:** The output projection maps `n_embd → n_vocab`. With TP=4, each rank holds V/4 rows of this projection. Each rank computes a partial logit vector of size V/4 from the final hidden state.

**Decode sampling (distributed):** During greedy/top-k decode, each rank finds the local maximum in its V/4 shard, producing a `(rank, local_token_id, logit)` tuple. A small all-gather of 4 tuples (48 bytes total) determines the global winner. This avoids an all-gather of the full vocabulary (V × 4 bytes ≈ 600 KB) on every decode step.

**Prefill / quality fixture sampling (full logits):** The quality fixture and prefill scoring need the full logit distribution. For these paths, use a full all-gather: each rank's V/4 partial logits are gathered to all ranks, producing the complete V-element logit vector.

**Current code:** `metal_graph_cuda_tp_output_tiers_for_head()` at `ds4.c:71` uses `partner = tier + half` to assign output shards. For TP=4, this needs to return all 4 tiers as shard holders. The logit assembly code needs to gather from 4 sources instead of 2.

## Acceptance criteria

- [x] Vocabulary row-sharded V/4 per rank (uses sharding policy from #26)
- [x] Distributed decode sampling: each rank finds local max, small all-gather picks global winner
- [x] Full logit all-gather available for prefill / quality fixture scoring
- [x] Greedy decode produces correct tokens end-to-end (multi-turn conversation works)
- [x] TP=2 output head path unchanged
- [x] `make -j8 rocm` builds cleanly

## Blocked by

- Issue #30: TP=4 MoE path (MoE must work before output head can be tested end-to-end)

## Implementation Summary

**Changes made:**

1. **Updated `metal_graph_cuda_tp_output_tiers_for_head()`** (ds4.c:71-121): Added `rocm_tp4` parameter. When true and `n_gpus == 4`, returns all 4 tiers [0,1,2,3] instead of just head_tier + partner. This enables the vocab sharding logic to split across all 4 ranks.

2. **Implemented `ds4_gpu_indexer_top1_value_tensor` for ROCm** (rocm/ds4_rocm_indexer.cuh:289-357, 1127-1157): Ported the CUDA kernel that finds argmax index and value in a single pass. Required for the distributed decode sampling path where each rank finds its local best (id, value) tuple.

3. **Removed stub from ds4_rocm_unavailable.cu**: The `ds4_gpu_indexer_top1_value_tensor` stub was removed since it now has a real ROCm implementation.

4. **Enabled distributed decode sampling for TP=4** (ds4.c:30279-30285): Modified the `split_top1` condition to enable the distributed sampling path by default for ROCm TP=4 (via `g->rocm_tp4 ||`), in addition to the existing env var override.

5. **Updated all callers** of `metal_graph_cuda_tp_output_tiers_for_head` to pass the new `rocm_tp4` parameter:
   - `metal_graph_cuda_tp_output_tiers` wrapper (ds4.c:16755-16763)
   - `engine_cuda_tp_output_shard_span` (ds4.c:54792-54795)
   - `engine_install_per_device_caches_for_multi_tier` (ds4.c:55234-55239)

**How it works:**

- **Distributed decode (greedy):** The `split_top1` path is now enabled for ROCm TP=4. Each rank computes V/4 logits in its shard, runs `ds4_gpu_indexer_top1_value_tensor` to find local (id, value), copies 4 tuples to head_tier via `ds4_gpu_tensor_copy_xdev`, and head_tier picks the global winner on CPU. This is the 48-byte all-gather approach specified in the PRD.

- **Full logit gather (prefill/quality):** The existing `metal_graph_output_logits_head_matmul` already handles multi-way vocab splitting. With the updated `metal_graph_cuda_tp_output_tiers_for_head` returning all 4 tiers, it now correctly gathers full V logits to head_tier for prefill and quality fixture scoring.

- **TP=2 unchanged:** The `rocm_tp4` parameter is false for TP=2 builds, so all code paths take the original branches. The TP=2 vocab split and logit gather work exactly as before.

**Testing:**
- All 228 TP sharding tests pass (including 4-rank partition tests)
- ROCm TP stub tests pass
- `make -j8 rocm` builds cleanly with no errors
