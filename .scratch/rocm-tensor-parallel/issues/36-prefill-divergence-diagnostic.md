# 36 — Run per-layer prefill diagnostic on failing vs passing prompts

Status: ready-for-agent

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

- [ ] Per-layer diff table produced for case_094 (failing prompt), identifying the first divergent layer and tensor
- [ ] Per-layer diff table produced for case_060 (passing prompt), showing where (if anywhere) it diverges
- [ ] The root cause hypothesis is updated: "attention path" vs "MoE FFN path" vs "router selection" vs "multi-layer compounding"
- [ ] A comment is added to issue #32 with the diagnostic results and the specific line/function where the fix should go
- [ ] If the divergence is in the MoE path, the batch FFN `ds4_gpu_routed_moe_batch_owned_tensor` + `routed_moe_launch` path is compared against the non-TP `ds4_gpu_routed_moe_batch_tensor` path at the divergence layer (same prompt, both on tier 0, diff the `batch_routed_out` tensor)

## Blocked by

- `#35 — Prefill per-layer diagnostic framework`
