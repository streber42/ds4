# Evaluation harness option to disable reasoning/thinking mode for concise reference comparison

Status: closed

## Parent

`.scratch/rocm-tensor-parallel/PRD.md`

## Problem

When evaluating benchmark prompts against DeepSeek-V4-Flash, thinking/reasoning blocks (`<think>...<\/think>`) introduce non-deterministic reasoning overhead and high completion token counts. This complicates direct text similarity comparison and exact match verification against reference API endpoints for short QA and code generation tasks.

## Key Requirements

1. **CLI Flag & Option**: Add `--disable-thinking` option to `tests/test_opencode_reference_eval.py`.
2. **Payload / System Directive**: Support passing reasoning control parameters (e.g., `thinking: {"type": "disabled"}` or system prompt instructions "Output answer directly without thinking") to bypass reasoning generation.
3. **Simplified Metrics**: Simplify semantic similarity and exact match metrics by evaluating concise direct answers without reasoning token variance.

## Acceptance Criteria

- [x] `tests/test_opencode_reference_eval.py --disable-thinking` implemented and documented.
- [x] Evaluation runs without producing `<think>` blocks when `--disable-thinking` is passed.
- [x] Evaluation runtime and TTFT improve with reasoning disabled, producing direct text output comparisons.

## Implementation Notes

- Added `SYSTEM_NO_THINKING_DIRECTIVE` constant that explicitly forbids `<think>` tags and requests direct concise output, replacing `SYSTEM_REASONING_DIRECTIVE` in all benchmark case messages when `--disable-thinking` is passed.
- `post_chat_stream()` accepts a new `disable_thinking: bool` parameter; when `True`, adds `"thinking": {"type": "disabled"}` to the API payload for endpoints that support it (ds4 server, some OpenAI-compatible APIs). Servers that don't recognise the field ignore it gracefully.
- `_swap_system_directive()` helper returns a new message list with the first system message replaced by the no-thinking directive, without mutating the original benchmark case list.
- When `--disable-thinking` is active, pass/fail evaluation uses only `content` (not the combined `content + reasoning` fallback), since the point is to get a clean direct answer with no reasoning leakage.
- `disable_thinking` is recorded in the JSON summary and surfaced in the Markdown report header.
- `--help` and module docstring both document the flag and its exact semantics.
- 3 new self-tests added (tests 6–8): `_swap_system_directive` replace-in-place, `_swap_system_directive` prepend-when-absent, and `SYSTEM_NO_THINKING_DIRECTIVE` content validation. All 8 self-tests pass (`--self-test`).

## Verification

```
python3 tests/test_opencode_reference_eval.py --self-test
# → All evaluation harness self-tests PASSED successfully!

python3 tests/test_opencode_reference_eval.py --help | grep disable-thinking
# → --disable-thinking    Suppress <think>...</think> reasoning blocks ...
```
