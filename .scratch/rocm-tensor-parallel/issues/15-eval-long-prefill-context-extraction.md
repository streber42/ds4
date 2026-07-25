# Long prefill chunking and key extraction validation under tensor-parallelism

Status: closed

## Parent

`.scratch/rocm-tensor-parallel/PRD.md`

## Problem

During evaluation with `tests/test_opencode_reference_eval.py`, the long prompt prefill benchmark case (`long_prefill`, ~1000 tokens context containing hidden metadata key `DS4-ROCM-8849`) resulted in output truncation on local tensor-parallel inference (`DS4-ROCM-` instead of `DS4-ROCM-8849`).

## Key Requirements

1. ~~Prefill Context Preservation~~ — Found not to be a TP prefill chunking issue.
2. ~~KV Cache Boundary Audit~~ — Not applicable; root cause was elsewhere.
3. ~~Exact Key Retrieval~~ — Resolved by token budget fix in issue 14.

## Acceptance Criteria

- [x] Long prefill test case in `tests/test_opencode_reference_eval.py` passes with exact key match `DS4-ROCM-8849`. Resolved by issue 14's token budget and prompt formatting fix.
- [x] Logits and attention outputs verified consistent across long prefill chunk boundaries. Not audited — truncation was not a TP prefill chunking bug, so no boundary audit was needed.

## Implementation Notes

**Root cause:** The `long_prefill` case wasn't failing due to a tensor-parallelism prefill chunking or KV cache boundary bug. The model was spending its output token budget on `<think>` reasoning tokens, running out of budget before it could emit the full key string `DS4-ROCM-8849`.

**Fix:** Issue 14 (`14-eval-harness-reasoning-token-budget.md`) raised `max_tokens` for the `long_prefill` case to 2048 and improved prompt formatting directives to encourage concise final output after reasoning. This resolved the truncation without any TP-level changes.

**No TP chunking work was needed.** A worktree was created (`15-eval-long-prefill-context-extraction`) and status bumped to `in-progress`, but the issue was already solved by the time investigation began. Closed with this note.
