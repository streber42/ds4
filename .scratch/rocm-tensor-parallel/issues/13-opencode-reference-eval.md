# OpenCode Go reference comparison evaluation harness

Status: closed

## Parent

`.scratch/rocm-tensor-parallel/PRD.md`

## What to build

Build an automated evaluation test script (`tests/test_opencode_reference_eval.py` or shell harness) that compares our local ROCm tensor-parallel server (`http://localhost:8000/v1`) against the official live OpenCode Go reference API (`https://opencode.ai/zen/go/v1`).

The harness should run a suite of benchmark prompts (short QA, code generation, reasoning/math, multi-turn chat, and long prefill prompts) side-by-side across both endpoints using credentials from `.env` (`OPENCODE_API_KEY` and `OPENCODE_BASE_URL`).

## Key Requirements

1. **Endpoint Resolution**: Automatically load `OPENCODE_API_KEY` and `OPENCODE_BASE_URL` from `.env`. Fallback/default to `https://opencode.ai/zen/go/v1` if `OPENCODE_BASE_URL` is omitted.
2. **Benchmark Prompt Suite**:
   - Short factual QA (e.g. "What is the capital of France?")
   - Code generation / C pointers (e.g. "Write a C function to swap two integers using pointers.")
   - Math & reasoning (e.g. "What is 2 + 2? Answer directly.")
   - Multi-turn conversation / context preservation
   - Long prompt prefill chunking validation
3. **Comparison Metrics**:
   - Compare final answer text similarity/exactness.
   - Compare reasoning content format (presence of thinking blocks, step-by-step logic structure).
   - Compare token generation throughput (tokens/sec) and time-to-first-token (TTFT).
4. **Report Output**:
   - Output structured JSON/Markdown results summarizing pass/fail ratios, latency comparisons, and semantic agreement scores.
   - Save test run artifacts under `.scratch/rocm-tensor-parallel/eval-out/`.

## Acceptance criteria

- [x] `tests/test_opencode_reference_eval.py` script implemented and runnable via `python3 tests/test_opencode_reference_eval.py`.
- [x] Test harness queries both local server (`http://localhost:8000/v1`) and OpenCode Go (`https://opencode.ai/zen/go/v1`).
- [x] Evaluates short QA, code generation, reasoning, and multi-turn prompt cases.
- [x] Generates a clean Markdown comparison report detailing agreement score, latency, and output diffs.
- [x] Included in test suite / Makefile test target or standalone evaluation command.
