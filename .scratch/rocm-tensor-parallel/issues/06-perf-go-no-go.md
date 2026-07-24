# Early perf measurement & go/no-go decision

Status: ready-for-human

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
- [ ] Numbers appended to the project's experiment log
- [ ] An explicit written decision is recorded: continue / stop / re-scope
- [ ] If slower than baseline, a hypothesis for why is recorded, plus whether it is plausibly addressable

## Blocked by

- `.scratch/rocm-tensor-parallel/issues/05-first-correct-token.md`
