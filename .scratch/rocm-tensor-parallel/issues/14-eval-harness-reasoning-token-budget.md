# Evaluation harness reasoning token budget & prompt formatting for DeepSeek-V4-Flash

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/PRD.md`

## Problem

In `tests/test_opencode_reference_eval.py`, short factual QA (`short_qa`) and code generation (`code_gen`) cases fail when evaluated against reasoning models like DeepSeek-V4-Flash. The model emits extensive `<think>` reasoning tokens before producing the final content text, consuming the default token budget before emitting the target answer substring (e.g. "Paris" or C function signature).

## Key Requirements

1. **Adaptive Token Allocation**: Dynamically scale `max_tokens` for benchmark cases when querying reasoning models so `<think>` content does not starve final response generation.
2. **Explicit Format Directives**: Include clear system/user prompt formatting directives (e.g. "Answer concisely after thinking") to encourage concise final content after reasoning.
3. **Dual Content & Reasoning Matching**: Ensure validation checks evaluate combined reasoning and content streams so correct answers stated inside thinking blocks are recognized.

## Acceptance Criteria

- [ ] `tests/test_opencode_reference_eval.py` updated with adaptive token budgets per test category.
- [ ] Short QA and Code Generation benchmark cases pass reliably against reasoning endpoints.
- [ ] Evaluation harness correctly separates `<think>` blocks from final answers across all benchmark cases.
