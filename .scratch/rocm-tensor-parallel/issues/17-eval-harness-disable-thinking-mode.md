# Evaluation harness option to disable reasoning/thinking mode for concise reference comparison

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/PRD.md`

## Problem

When evaluating benchmark prompts against DeepSeek-V4-Flash, thinking/reasoning blocks (`<think>...</think>`) introduce non-deterministic reasoning overhead and high completion token counts. This complicates direct text similarity comparison and exact match verification against reference API endpoints for short QA and code generation tasks.

## Key Requirements

1. **CLI Flag & Option**: Add `--disable-thinking` option to `tests/test_opencode_reference_eval.py`.
2. **Payload / System Directive**: Support passing reasoning control parameters (e.g., `thinking: {"type": "disabled"}` or system prompt instructions "Output answer directly without thinking") to bypass reasoning generation.
3. **Simplified Metrics**: Simplify semantic similarity and exact match metrics by evaluating concise direct answers without reasoning token variance.

## Acceptance Criteria

- [ ] `tests/test_opencode_reference_eval.py --disable-thinking` implemented and documented.
- [ ] Evaluation runs without producing `<think>` blocks when `--disable-thinking` is passed.
- [ ] Evaluation runtime and TTFT improve with reasoning disabled, producing direct text output comparisons.
