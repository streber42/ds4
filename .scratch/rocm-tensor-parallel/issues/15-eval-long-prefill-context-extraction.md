# Long prefill chunking and key extraction validation under tensor-parallelism

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/PRD.md`

## Problem

During evaluation with `tests/test_opencode_reference_eval.py`, the long prompt prefill benchmark case (`long_prefill`, ~1000 tokens context containing hidden metadata key `DS4-ROCM-8849`) resulted in output truncation on local tensor-parallel inference (`DS4-ROCM-` instead of `DS4-ROCM-8849`).

## Key Requirements

1. **Prefill Context Preservation**: Verify that long prompt activation prefill chunking preserves full context across all TP ranks without dropping tail context tokens.
2. **KV Cache Boundary Audit**: Audit KV cache indexing and RoPE position offsets during long context prefill under ROCm tensor-parallel mode.
3. **Exact Key Retrieval**: Validate 100% exact match key retrieval (`DS4-ROCM-8849`) on long prompts compared to single-GPU reference.

## Acceptance Criteria

- [ ] Long prefill test case in `tests/test_opencode_reference_eval.py` passes with exact key match `DS4-ROCM-8849`.
- [ ] Logits and attention outputs verified consistent across long prefill chunk boundaries.
