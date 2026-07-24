# Correctness harness (logits diff + quality fixture runner)

Status: closed

## Parent

`.scratch/rocm-tensor-parallel/PRD.md`

## What to build

A repeatable harness that answers "does this build produce the same output as the reference?"
It compares logits between a reference run and a candidate run for the same prompt and
sampling settings, and it runs the project's existing multi-case quality fixture to catch
regressions a single prompt would miss.

Critically, this must be built and proven **before** any tensor-parallel code exists, and
proven by running it against the current pipeline build where the answer is already known to
be "match". An oracle nobody has validated is worthless — if the harness cannot show itself
green on a known-good build, it cannot be trusted to condemn a new one.

The reference is the existing, already-trusted ROCm single-GPU / pipeline path on the same
hardware and quantisation, not a different backend on different hardware, so that any
divergence is attributable to sharding rather than to backend or hardware variation.

## Acceptance criteria

- [x] A single command produces a pass/fail logits comparison between reference and candidate
- [x] Comparison uses an explicit numeric tolerance that is stated and justified, not tuned until green
- [x] On failure the harness reports where it diverged (token position and magnitude), not just "failed"
- [x] The existing multi-case quality fixture can be run and produces a score comparable across runs
- [x] Harness is proven green against the current pipeline build, demonstrating the oracle itself works
- [x] Harness does not require tensor parallelism to exist in order to run
- [x] Reference selection (same hardware, same quantisation, pipeline path) is documented in the harness

## Blocked by

None - can start immediately.

## Implementation

`tests/test_engine_correctness_harness.c` with two modes:

- `--logits REF_MODEL REF_BE CAND_MODEL CAND_BE [--prompt "text"] [--ctx N] [--steps N] [--tol 1e-3]`
  Opens two engines sequentially (unique flock keys avoid intra-process contention), runs full
  greedy decode with vocab-sized logits at every step, compares with per-step max absolute
  error and argmax agreement. Reports step-by-step PASS/FAIL with error magnitudes.

- `--quality MODEL MANIFEST OUT_TSV [--ctx N]`
  Wraps `gguf-tools/quality-testing/score_official` to run the existing multi-case quality
  fixture. Produces TSV output comparable across runs.

## Verification

Built and proven green against the same model and CPU backend (bit-identical logits, 0.000000
max absolute error, all 8 steps PASS):

```sh
make tests/test_engine_correctness_harness
./tests/test_engine_correctness_harness --logits <model> cpu <model> cpu
```