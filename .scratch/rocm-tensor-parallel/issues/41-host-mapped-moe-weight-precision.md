# 41 — Investigate host-mapped MoE weight numerical impact on TP=4 quality

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/issues/32-tp4-quality-fixture.md`
`.scratch/rocm-tensor-parallel/issues/40-option-b-row-split-batch-prefill.md` (the Option B row-split refactor did not close the quality gap)

## Problem

TP=4 quality scores (avg_nll ~1.72, first_match=0/100) are unchanged by the Option B row-split refactor and are virtually identical to pre-Option B scores (~1.85). The gap from the pipeline reference (avg_nll=0.3747, first_match=65/100) is **not** caused by all-reduce FP accumulation noise or the attention-type gate bypass (fixed and tested — no change).

The startup log reveals:

```
ds4: ROCm model arena alloc failed for moe_gate (1024.00 MiB chunk): out of memory
```

**Hypothesis**: The `moe_gate` tensor (1024 MiB per GPU) cannot be cached in VRAM on the 4× R9700 setup (27.79 GiB budget each, 25.94 GiB weights loaded). `ds4_gpu_routed_moe_batch_tensor` falls back to host-mapped memory for uncached MoE weights via `cuda_model_range_ptr_from_fd`. If the host-mapped fallback produces numerically different results from fully-cached weights (e.g., different quantization path, page-faulted reads, or different precision in the kernel's weight-load path), this would explain the consistent ~1.7 NLL gap that persists across all TP=4 configurations (sharded, row-split, all-gather, all-reduce).

The pipeline reference uses fully-cached weights (no TP, no multi-GPU overhead) and achieves correct scores.

## What to investigate

1. **Layer-0 hidden state comparison**: Run TP=4 and pipeline on a single prompt with `n_tokens=16`, capture `batch_next_hc` after layer 0. Compute max_abs_diff and first_divergence_index. If divergence starts at layer 0, the MoE path is the most likely culprit.

2. **Host-mapped vs cached weight comparison**: If possible, run `ds4_gpu_routed_moe_batch_tensor` with the same inputs but force host-mapped vs cached weight resolution. Compare the output tensors.

3. **Memory audit**: Determine exactly which MoE tensors (`moe_gate`, `moe_up`, `moe_mid`, `moe_down`) are falling back to host mapping. The `ROCm model arena alloc failed` message points at `moe_gate` specifically.

## Expected outcome

- Confirm or rule out host-mapped MoE weight precision as the root cause of the ~1.7 avg_nll gap
- If confirmed: find a fix (increase VRAM budget, reduce model footprint, or fix the host-mapped fallback path)
- If ruled out: document the finding and move to next hypothesis

## Acceptance criteria

- [ ] Layer-0 tensor comparison run between TP=4 and pipeline
- [ ] Host-mapped vs cached weight numerical equivalence tested (or root cause identified)
- [ ] Root cause of remaining quality gap identified
- [ ] Issue #40 updated with findings

## Blocked by

- 4× R9700 GPU access
- Existing TP=4 quality fixture infrastructure
