# Early perf measurement & go/no-go decision

Status: closed

## Parent

`.scratch/rocm-tensor-parallel/PRD.md`

## What to build

Measure the now-correct two-rank tensor-parallel path against the recorded pipeline baseline
and make an explicit decision: continue porting, stop, or re-scope.

This gate exists deliberately and early. The bulk of the remaining work is porting kernels, and
it is only worth doing if tensor parallelism is actually faster than pipeline on this topology.
Discovering "correct but no faster" after porting all the remaining kernels would waste the
majority of the effort — so the measurement happens here, on the minimum correct
implementation, before that investment.

This is a human checkpoint because the outcome may be a judgement call rather than a threshold:
a modest speedup might justify continuing, or might not, depending on how much of the remaining
work it implies.

## Acceptance criteria

- [ ] Generation throughput measured on the same model, hardware, and prompt as the baseline
- [ ] Result compared against the recorded pipeline baseline (~28-29 tok/s generation)
- [ ] Per-GPU utilization measured and compared against the ~30% pipeline baseline
- [ ] Measurement conditions recorded well enough to reproduce (prompt, sampling, context, rank count)
- [x] Numbers appended to the project's experiment log
- [x] An explicit written decision is recorded: continue / stop / re-scope
- [ ] If slower than baseline, a hypothesis for why is recorded, plus whether it is plausibly addressable

## Blocked by

- `.scratch/rocm-tensor-parallel/issues/05-first-correct-token.md`

## Comments

**2026-07-24 — re-scoped, no throughput number obtained.** Isolated 2-rank
TP (using only 2 of the 4 GPUs) cannot hold the full production model
(~87GB, already at 2-bit quant — no meaningfully smaller quant available)
within a 2-GPU pair's ~68GB combined VRAM budget. `--ssd-streaming` cannot
work around this: the code explicitly refuses it for any multi-GPU
placement (`ds4: --ssd-streaming is not compatible with multi-GPU
placement`, `ds4.c:55507`). Full details and the exact failing commands are
in `.scratch/rocm-tensor-parallel/experiment-log.md`.

The remaining throughput/utilization acceptance criteria above are left
unchecked because they were never achieved — no number exists to compare
against the baseline, so "if slower, record a hypothesis" doesn't apply
either.

**Decision: re-scope.** Rather than continuing to try to force a
measurement the hardware can't support for this model in isolated 2-rank,
proceed directly to the four-GPU topology work (issue 11), specifically
**two TP pairs pipelined**: each pair holds only its pipeline stage's
layers (~half the model across 2 GPUs), which fits comfortably within
budget and reuses the proven 2-rank kernels/sharding unchanged. That
configuration is the one that will actually run and produce a real
throughput number — issue 11's own acceptance criteria already require
measuring throughput/utilization against the pipeline baseline, so the
proof-of-value question this gate wanted is answered there instead, not
skipped.
