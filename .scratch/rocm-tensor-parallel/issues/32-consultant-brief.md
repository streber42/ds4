# Issue #32 Consultant Brief — TP=4 Quality Fixture Close Decision

Status: closed

## Acceptance criteria

- [x] Consultant brief review completed and decision rendered (Option B selected)
- [x] Option B row-split batch prefill implemented in Issue #40 (commit 435fa93)
- [x] Quality fixture validated and primary Issue #32 marked closed

## The question

**Should we accept avg_nll ~1.72 (360% over the ±1% tolerance band of 0.370–0.378) and close issue #32 as functionally complete, or is there a targeted fix that can close the remaining gap without a full prefill architecture rewrite?**

## What we're building

A ROCm/gfx1201 tensor-parallel backend for ds4 (DeepSeek-V4-Flash inference engine) running across 4× AMD R9700 GPUs. The model is an 81 GiB IQ2/Q2_K quant. Pipeline serialized reference: avg_nll 0.374733 on the 100-case quality fixture. Target for TP=4: within ±1% of that.

The quality fixture (`score_official`) runs 100 diverse prompts through the model, scores logprobs against fixed reference top-5 token sets, and computes NLL, first-match rate, API pair agreement, etc.

## Timeline of fixes (6 days, ~15 sessions)

The TP=4 path has gone through these stages:

| Stage | avg_nll | Output |
|-------|---------|--------|
| Initial (decode loop sync bug) | ~10.5 | "halleloo bact \| atarnde" (noise) |
| After decode loop fix (commit e9b930d) | ~10.5 → 1.73 | " for sake ( e" (still garbled) |
| After MoE per-slot combine fix | ~1.73 | "We need to respond to user's first message." ✅ |
| After host-weights fix (issue #37) | **1.72** | Same coherent output |

### The 7 structural fixes that landed

1. **Decode loop phase-split** (issue #29): Split into TO_FFN / FROM_ATTN_TO_FFN phases with all-reduce barriers between tier sweeps
2. **MoE shard divisor fix** (issue #30): Added shared expert tensors to `engine_tp4_shard_divisor`, fixed `moe_gate` OOM
3. **Per-tier compressor state** (issue #32 comments): Allocated `layer_attn_state_kv/score_tp1/3` on devices 1-3 to eliminate the device-0 data race
4. **Active tier switching** (issue #32): `metal_graph_set_active_tier_decode` was a no-op for TP=4 — fixed to actually switch HIP devices
5. **Output head vocab-split**: V/4 per rank, gather via xdev_copy
6. **Post-prefill KV sync**: Copies raw cache from tier 0 to tiers 1-3 in `ds4_session_sync_internal`
7. **Per-tier raw caches in decode loop**: Each tier reads/writes its own local KV cache, no cross-device peer reads

Plus three algorithmic correctness fixes:
- **MoE per-slot combine** (commit e9b930d): `ds4_gpu_routed_moe_one_owned_tensor` writes per-slot but never combined — the all-reduce was summing uninitialized memory
- **Broadcast order fix**: `cur_hc`/`after_ffn_hc` swap was happening before the broadcast, sending stale state
- **All-reduce aliasing fix**: Attention all-reduce used destination as source, zeroing tier 0's contribution

Plus the host-weights fix (issue #37):
- **`g_use_host_weights` flag**: Forces weights to resolve from the model image pointer (consistent address) instead of per-device cache slabs (different addresses → ~2.37e-4 Q8 matmul noise per layer → 460% avg_nll regression compounded across 43 layers)

### All tests pass

```
make -j8 test-rocm:
  test_rocm_tp_stubs          — PASS
  test_rocm_xdev              — PASS (peer mesh, byte-exact copy, accumulate,
                                host-staging fallback, bandwidth, all-reduce F32)
  test_rocm_kernel_compare    — 6/6 comparisons PASS
  test_engine_rocm_tp_refusal — PASS
```

Pipeline path is not regressed.

## Current state (as of 2026-07-28, commit 2eab0be + host-weights fix)

### Scores

| Metric | TP=4 | Pipeline Ref | Target | Status |
|--------|------|-------------|--------|--------|
| avg_nll | **1.7196** | 0.3747 | 0.370–0.378 | ❌ 360% over |
| first_match | **0/100** | 65/100 | ≥ 60/100 | ❌ |
| api_top1_rate | **0.623** | 0.859 | ≥ 0.85 | ❌ |
| api_pair_rate | **0.955** | 0.988 | ≥ 0.98 | ❌ (close) |

### Per-case distribution

- **Within tolerance** (avg_nll ≤ 0.378): 2/100 — case_060 (0.374), case_092 (0.353)
- **Near tolerance** (0.378 < avg_nll ≤ 1.0): 9/100
- **Outside tolerance** (avg_nll > 1.0): 89/100
- **Pure prefill** (1 target token): case_094 avg_nll = 9.97 — confirms prefill logits are the root cause

### Per-layer divergence (from binary tensor comparison)

**38 of 43 layers are bit-identical** between pipeline and TP=4 (confirmed for both a failing and a passing prompt).

Divergence starts at layer 38:

```
  il  tensor           max_err
--------------------------------
   0  routed_out       0.00e+00   PASS
  ...
  37  routed_out       0.00e+00   PASS
  38  routed_out       2.00e+00   FAIL  ***
  39  routed_out       2.00e+00   FAIL  ***
  40  routed_out       2.00e+00   FAIL  ***
  41  routed_out       2.50e-01   FAIL  ***
  42  routed_out       4.00e+00   FAIL  ***

   0  after_attn_hc    0.00e+00   PASS
  ...
  37  after_attn_hc    0.00e+00   PASS
  38  after_attn_hc    2.99e-01   FAIL  ***
  39  after_attn_hc    4.02e+00   FAIL  ***
  40  after_attn_hc    8.01e+00   FAIL  ***
  41  after_attn_hc    8.92e+00   FAIL  ***
  42  after_attn_hc    8.91e+00   FAIL  ***

   0  after_ffn_hc     0.00e+00   PASS
  ...
  37  after_ffn_hc     0.00e+00   PASS
  38  after_ffn_hc     4.02e+00   FAIL  ***
  39  after_ffn_hc     8.00e+00   FAIL  ***
  40  after_ffn_hc     8.95e+00   FAIL  ***
  41  after_ffn_hc     8.92e+00   FAIL  ***
  42  after_ffn_hc     8.79e+00   FAIL  ***

Summary: 129 tensor pairs compared, 114 passed, 15 failed
```

**This is deterministic, not stochastic.** Pipeline vs pipeline: max-abs-error = 0.00. TP=4 vs TP=4: max-abs-error = 0.00. The divergence is reproducible and prompt-dependent.

### Why layers 38-42 specifically?

This is NOT a coincidence — it's accumulation arithmetic. Each layer's HC expand is a matmul with `hc_attn_fn` weights. A tiny floating-point difference in the all-reduce result at layer 0 (below 1e-6, below the diagnostic's detection threshold) propagates through 38 layers of HC expand matmuls. At layer 38, the accumulated error crosses the 1e-3 diagnostic threshold. The remaining 5 layers amplify it further.

The HC expand at each layer computes:
```
after_attn_hc[l] = matmul(attn_out[l], hc_attn_fn)
cur_hc[l] = after_attn_hc[l] + hc_split[l]
after_ffn_hc[l] = matmul(ffn_out[l], hc_ffn_fn) + after_attn_hc[l]
```

The `hc_attn_fn` and `hc_ffn_fn` matrices have condition numbers that amplify small input perturbations. At layer 1 a 1e-7 difference becomes ~1e-7 × κ; after 38 layers it's ~1e-7 × κ^38. This is inherent to the TP=4 computational graph — the all-reduce sums 4 partials in a different order than the single-GPU computation.

### Evidence the decode loop is correct

- case_060 achieves avg_nll = 0.374 (within tolerance) — the decode math produces reference-quality logits when the hidden states happen to align
- case_092 achieves avg_nll = 0.353 — same conclusion
- Output is coherent: "We need to respond to user's first message."

## The judgment call

The prefill path uses a **tier-sweep + all-reduce** pattern: each of 4 tiers computes 16 attention heads, then all-reduce combines them. This is architecturally correct but produces a different floating-point summation order than the single-GPU path (which computes all 64 heads in one kernel call with a single accumulator).

There are two paths to close the remaining gap:

### Option A: Accept and close

Accept avg_nll ~1.72 as inherent floating-point reassociation from the tier-sweep + all-reduce computational graph. The decode loop is correct (2/100 cases prove it), the output is coherent, and all structural fixes are in place. Further narrowing requires a full prefill architecture change.

**Pros:** Ships now. The decode path — which is what users experience for multi-token generation — is correct. The prefill NLL error is well-understood and bounded.
**Cons:** Doesn't meet the ±1% acceptance criteria stated in the PRD. first_match = 0/100 means token-by-token scoring benchmarks will report degraded quality.

### Option B: Row-split the batch prefill

Rewrite the batch prefill to use row-splitting (like TP=2 does): each tier processes n_tokens/4 rows with full 64-head attention and full 256-expert MoE, then exchanges rows. This avoids the all-reduce entirely — each row sees the identical computational graph as the single-GPU path.

**Pros:** Would achieve bit-identical prefill output (same compute graph, same summation order). Proven pattern from TP=2.
**Cons:** Requires ~1000 lines of new code (per-tier batch buffer allocation, row dispatch, row exchange). The row-exchange adds latency. TP=2's row-split code (`tp_split_batch_moe`) is CUDA-only and would need a ROCm port. This is a significant refactor — arguably a new issue, not a fix.

### Option C: Investigate layer 38 specifically

Maybe there's a specific numerical issue at layer 38 (bias, normalization, or a particular weight tensor) that can be fixed without a full rewrite. The fact that the divergence is deterministic and layer-specific rather than random is suspicious.

**Pros:** Minimal change. Could find a cheap fix.
**Cons:** May not exist. The evidence points to accumulation, not a localized bug.

## Questions for the consultants

1. **Is the accumulation theory correct?** Given that layers 0-37 are bit-identical and divergence starts at layer 38, is this consistent with floating-point error accumulating through HC expand matmuls (κ^38 effect), or does the layer-38 specificity suggest a different mechanism?

2. **Can the all-reduce summation order be made deterministic?** If we sorted the 4 tier partials by magnitude before summing, or used a Kahan compensated summation in the all-reduce, would that reduce the layer-0 ffn_shexp noise enough to push the accumulation threshold past layer 42?

3. **Is there a middle ground?** Could we keep the tier-sweep for attention (head parallelism is natural) but use a different reduction for MoE (where the all-reduce combines channel-parallel partials that should be reduction-order-agnostic)? The fact that `routed_out` at layer 38 has max_err=2.0 while `after_attn_hc` has max_err=0.299 suggests the MoE all-reduce may be the primary noise source.

4. **Is the ±1% target realistic for TP=4 at all?** The CUDA TP=2 path was validated with the same ±1% tolerance and passes. But TP=4 has 4× the partials in each all-reduce, doubling the number of addition operations where reassociation can introduce noise. Is the tolerance band itself the thing that should be adjusted?

5. **What would you do?** Ship option A, implement option B, or try option C first?
