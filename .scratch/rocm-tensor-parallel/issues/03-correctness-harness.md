# Correctness harness (logits diff + quality fixture runner)

Status: ready-for-agent

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

- [ ] A single command produces a pass/fail logits comparison between reference and candidate
- [ ] Comparison uses an explicit numeric tolerance that is stated and justified, not tuned until green
- [ ] On failure the harness reports where it diverged (token position and magnitude), not just "failed"
- [ ] The existing multi-case quality fixture can be run and produces a score comparable across runs
- [ ] Harness is proven green against the current pipeline build, demonstrating the oracle itself works
- [ ] Harness does not require tensor parallelism to exist in order to run
- [ ] Reference selection (same hardware, same quantisation, pipeline path) is documented in the harness

## Blocked by

None - can start immediately.
