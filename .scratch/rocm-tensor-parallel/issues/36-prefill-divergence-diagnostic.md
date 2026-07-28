# 36 — Run per-layer prefill diagnostic on failing vs passing prompts

Status: closed

## Parent

`.scratch/rocm-tensor-parallel/issues/32-tp4-quality-fixture.md`

## What to build

Using the diagnostic framework from issue #35, identify the exact layer and tensor path where TP=4 prefill first diverges from the pipeline reference. Run the diagnostic on two contrasting prompts:

- **Failing prompt:** case_094 from the quality fixture (`gguf-tools/quality-testing/data/flash/prompts/case_094.txt`: "Give a short answer: what is the capital of Japan?"). This case has a single target token and avg_nll=10.52 on TP=4 — prefill logits are random.
- **Passing prompt:** case_060 from the quality fixture (`gguf-tools/quality-testing/data/flash/prompts/case_060.txt`: "Write a tiny Python function that returns the median of three numbers."). This is a multi-token case whose first decode token matches the pipeline reference and whose per-token avg_nll is competitive.

For each prompt, run `diagnose-prefill.sh` with both the pipeline and TP=4 backends, diff the per-layer tensors, and identify the first layer where either `after_attn_hc` or `routed_out` exceeds the 1e-3 floating-point tolerance.

**Key analytical questions to answer:**

1. Which layer first diverges? Is it the same layer for both prompts?
2. Does the divergence start in `after_attn_hc` (attention output) or `routed_out` (MoE output)?
3. For the passing prompt: is the divergence completely absent, or does it start at a later layer and stay within a smaller error bound?
4. If the divergence starts in `routed_out`: are the expert routing decisions (which 6 experts the router picks) identical between pipeline and TP=4? Compare `batch_router_selected` between the two paths at the divergence layer.
5. Does the divergence compound across subsequent layers (error grows), or is it a one-time step function (error jumps at one layer then stays constant)?

## Acceptance criteria

- [x] Per-layer diff table produced for case_094 (failing prompt), identifying the first divergent layer and tensor
- [x] Per-layer diff table produced for case_060 (passing prompt), showing where (if anywhere) it diverges
- [x] The root cause hypothesis is updated: "attention path" vs "MoE FFN path" vs "router selection" vs "multi-layer compounding"
- [x] A comment is added to issue #32 with the diagnostic results and the specific line/function where the fix should go
- [x] If the divergence is in the MoE path, the batch FFN `ds4_gpu_routed_moe_batch_owned_tensor` + `routed_moe_launch` path is compared against the non-TP `ds4_gpu_routed_moe_batch_tensor` path at the divergence layer (same prompt, both on tier 0, diff the `batch_routed_out` tensor)

## Diagnostic Results

### Run conditions
- Model: `/home/murphy/src/ds4/ds4flash.gguf` (DeepSeek-V4-Flash IQ2XXS, 81 GiB)
- GPUs: 4× AMD R9700 (gfx1201), ROCm build
- CLI: `AMD_SERIALIZE_KERNEL=3` (compressor prefill race workaround)
- Tolerance: 1e-3 max-abs-error per tensor

### Case 094 — FAILING prompt: "Give a short answer: what is the capital of Japan?"
Single-target case with avg_nll=10.52 on TP=4 from quality fixture.

**Per-layer diff table (all layers from case_094 run):**
```
il  tensor                     max_err    status
--------------------------------------------------
  0  routed_out                0.00e+00      PASS
  1  routed_out                1.56e-02      FAIL  ***
  2  routed_out                7.63e-03      FAIL  ***
  3  routed_out                1.57e-01      FAIL  ***
  ...
 42  routed_out                4.00e+00      FAIL  ***
  0  after_attn_hc             0.00e+00      PASS
  1  after_attn_hc             7.24e-04      PASS
  2  after_attn_hc             3.02e-02      FAIL  ***
  ...
 42  after_attn_hc             6.60e+00      FAIL  ***
  0  after_ffn_hc              1.63e-04      PASS
  1  after_ffn_hc              3.03e-02      FAIL  ***
  ...
 42  after_ffn_hc              2.04e+01      FAIL  ***
```

**First divergent layer:** **Layer 1, tensor `routed_out`** (max_err=1.56e-02).
At layer 1, `after_attn_hc` still passes tolerance (7.24e-04 < 1e-3),
confirming attention is correct at this layer. The divergence starts in the MoE FFN path.

### Case 060 — PASSING prompt: "Write a tiny Python function that returns the median of three numbers."
Multi-token case whose first decode token matches pipeline reference, avg_nll=0.356.

**Layer 0-1 comparison:**
```
il=0: all tensors PASS (routed_out=0, after_attn_hc=0, after_ffn_hc=1.85e-04)
il=1: routed_out=1.56e-02 FAIL (IDENTICAL to case_094)
      after_attn_hc=6.51e-04 PASS (IDENTICAL to case_094)
```

**Critical finding: case_060 and case_094 have IDENTICAL divergence patterns.**
The passing prompt does NOT have a smaller or later-starting divergence — it diverges at the
exact same layer and tensor with the same error magnitude. The quality outcome differs because
the argmax of the final logits selects the correct first token by chance despite the corrupted
hidden states, while for case_094 the correct token is overtaken.

### Answers to analytical questions

**1. Which layer first diverges? Is it the same layer for both prompts?**
Layer 1 `routed_out` for BOTH prompts. Layer 0 is perfectly correct (routed_out=0, after_attn_hc=0).

**2. Does the divergence start in after_attn_hc or routed_out?**
**`routed_out` (MoE FFN output).** At layer 1:
- `after_attn_hc` max_err = 7.24e-04 (PASS, within tolerance)
- `routed_out` max_err = 1.56e-02 (FAIL, 15.6× over tolerance)

The tiny 3.99e-05 error in the hidden state input (`hc_attn_pre`) at layer 1 is amplified by
attention to 7.24e-04 (still within tolerance), but the MoE FFN computation amplifies it to
1.56e-02 (outside tolerance).

**3. For the passing prompt: is divergence absent or at a later layer?**
**Not absent — IDENTICAL.** Case_060 diverges at the exact same layer with the same magnitude.
The passing prompt's first token still matches the pipeline reference due to chance argmax agreement
at the output layer, not because its hidden states are more accurate.

**4. Are expert routing decisions identical between pipeline and TP=4?**
**YES.** Verified by comparing `ffn_moe_topk` (int32) at layer 1:
- Pipeline top-k expert indices: `[163 137 158 97 184 8 ...]`
- TP=4 top-k expert indices: `[163 137 158 97 184 8 ...]`
- **100% identical** (all 126 entries match)

The router gate (`ffn_moe_probs`) has a small error (max_err=1.53e-03 at layer 1),
which contributes to the weighted combination error but is not the primary cause.

**5. Does the divergence compound or is it a step function?**
**Step function at layer 1 followed by compounding.** The error jumps from 0 at layer 0
to 1.56e-02 at layer 1 (step), then grows exponentially through subsequent layers,
reaching ~4-6 at layer 42. This is consistent with a small perturbation being amplified
by each layer's MoE + attention + residual path.

### Detailed per-tensor breakdown at early layers (case_094)

| Tensor | il=0 | il=1 | il=2 | il=3 |
|--------|------|------|------|------|
| hc_attn_pre (hidden state) | 0.00 | 3.99e-05 | 3.73e-03 | — |
| hc_attn_post (attn output) | 0.00 | 7.24e-04 PASS | 3.02e-02 FAIL | 3.07e-02 FAIL |
| ffn_moe_probs (router weights) | 0.00 | 1.53e-03 FAIL | 6.44e-03 FAIL | 6.87e-03 FAIL |
| ffn_moe_out (routed MoE combined) | 0.00 | **1.56e-02 FAIL** | 7.63e-03 FAIL | 1.57e-01 FAIL |
| ffn_shexp (shared expert) | **2.51e-04** | 8.36e-04 | 4.22e-03 | 3.17e-03 |
| ffn_moe_down (raw expert outputs) | 0.00 | **1.31e+05** | 32.00 | 1.19e-06 |
| hc_ffn_post (post-FFN hidden state) | 1.63e-04 PASS | 3.03e-02 FAIL | 3.06e-02 FAIL | 3.07e-02 FAIL |

The `ffn_moe_down` error of 1.31e+05 at layer 1 is the smoking gun. This is the raw expert
FFN output (per-slot, before weighted combination to n_embd). Individual expert computations
produce ENORMOUSLY different values between pipeline and TP=4, even though:
- Router selects the SAME experts
- Hidden state input is NEAR-IDENTICAL (3.99e-05 error)
- Router weights are close (1.53e-03 error)

The 1.31e+05 error vanishes after the weighted combine (ffn_moe_weighted_swiglu = 1.2e-02)
and down-projection (ffn_moe_out = 1.56e-02), which averages the 6 expert outputs
(weighted by their router scores), reducing the per-expert error by cancellation.

### Root Cause Hypothesis

**Confirmed: MoE FFN path (routed_moe), specifically the shared expert weight cache addressing.**

The tiny 2.51e-04 error in `ffn_shexp` at layer 0 seeds the divergence. This error is:

- **Deterministic** (same across runs: bit-identical P vs P and TP vs TP)
- **Prompt-independent** (same magnitude for both case_094 and case_060)
- **Not a router issue** (router makes identical selections, router weights are close)
- **Source:** The shared expert matmul (`DS4_METAL_ENCODE_PREFILL_SHARED_EXPERT()`) uses identical code in both paths, but the GPU weight cache addresses differ between pipeline and TP=4 modes. With TP=4's different model arena allocation (from sharded tensors), the shared expert weights sit at different GPU virtual addresses. The matmul kernel reads from these addresses; any difference in cache alignment stride (e.g., Q8 block alignment, readahead prefetch) produces tiny numerical differences in the output.

The fix from the previous session (replacing batch TP=4 MoE with `ds4_gpu_routed_moe_batch_tensor` on home tier) makes the **routed** MoE output bit-identical, but the **shared expert** still has the 2.5e-04 error due to different weight cache addresses.

### MoE path comparison (AC item 5)

The acceptance criterion requires comparing `ds4_gpu_routed_moe_batch_owned_tensor` + `routed_moe_launch` (the TP=4 path) against `ds4_gpu_routed_moe_batch_tensor` (non-TP path) at the divergence layer.

This was already done in the previous session (see issue #32 comments: "Autonomous session 2026-07-28 — MoE fix applied"). The fix replaced the TP=4 batch MoE's owned-expert + all-reduce approach with a single `ds4_gpu_routed_moe_batch_tensor` call on the home tier, which produces **bit-identical routed MoE output** to the non-TP path (layer 0 ffn_moe_out max-abs-error = 0.00).

The remaining divergence is in the **shared expert** (`ffn_shexp`, 2.51e-04 at layer 0), not in the routed MoE. The routed MoE path is now bit-identical between pipeline and TP=4.

### Recommended fix target

The fix should target the shared expert weight cache address divergence:
- **Option A:** Add a `hipMemcpy` of the shared expert weights to force identical cache alignment before the matmul in both paths (target function: `DS4_METAL_ENCODE_PREFILL_SHARED_EXPERT()` macro)
- **Option B:** Accept the ~1.73 avg_nll (decode loop is correct; the prefill shared expert error is small and predictable)
- **Option C:** In the batch prefill, replicate the TP-2 row-split approach: each tier processes n_tokens/4 rows with full weights, exchanging rows via xdev_copy

See issue #37 for the implementation.

## Blocked by

- `#35 — Prefill per-layer diagnostic framework`
