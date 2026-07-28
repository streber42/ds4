# 37 — Fix the identified prefill divergence

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/issues/32-tp4-quality-fixture.md`

## What to build

Implement the minimum code change that makes the per-layer hidden states from the batch prefill path match between pipeline and TP=4 within 1e-3 floating-point tolerance for both the failing and passing diagnostic prompts (issue #36).

The exact fix location depends on issue #36's findings. The three most likely divergence points, in decreasing probability:

### Most likely: Batch FFN MoE path (`metal_graph_encode_layer_ffn_batch`, ds4.c:30085–30220)

The TP=4 batch FFN uses `ds4_gpu_routed_moe_batch_owned_tensor` which:
1. Runs `moe_filter_owned_pairs_kernel` to mark unowned experts as -1 (weight 0) in-place on the selected/weights arrays
2. Calls `routed_moe_launch` which skips pairs with expert=-1
3. The resulting `batch_routed_out_by_tier[t]` is all-reduced across tiers

Possible defects:
- `moe_filter_owned_pairs_kernel` may leave weight=0 but still contribute a non-zero expert output (the kernel skips the pair entirely — verify the skip logic is numerically equivalent to computing an expert with weight=0)
- The all-reduce may have a different accumulation order than the non-TP full-expert MoE, producing different floating-point rounding
- The shared expert combination after the all-reduce may differ from the non-TP path
- The WMMA vs scalar-tile8 kernel path selection may differ between the owned and non-owned code paths

### Less likely: Batch attention path

All attention weights are replicated (shard_divisor=1 for all attention tensors) and the batch attention runs on tier 0 only with the same full-64-head code as the non-TP path. Any divergence here would indicate a subtler issue (Q8 quantization block alignment, host-mapped vs device-cached weight reads).

### Least likely: Multi-layer accumulation

If the per-layer error is small but compounds across 43 layers, the fix may need a tighter tolerance on the all-reduce or HC expand, or a different reduction order.

**Regardless of the root cause**, the fix must:
- Be minimal — change only the code path that causes the divergence
- Not regress the decode path (which is already correct)
- Not increase VRAM usage beyond the current ~24 GiB per GPU
- Pass the existing test suite (`make -j8 test-rocm`)

## Acceptance criteria

- [ ] Per-layer diff between pipeline and TP=4 shows max error ≤ 1e-3 at all 43 layers for both case_094 and case_060
- [ ] `make -j8 test-rocm` passes (all 4 test targets)
- [ ] Pipeline path is not regressed (coherent output for `-p "Hello" -n 1`)
- [ ] TP=4 decode path is not regressed (coherent multi-token output)
- [ ] A comment is added to issue #32 with the fix description and per-layer diff table confirming convergence

## Blocked by

- `#36 — Run per-layer prefill diagnostic on failing vs passing prompts`
