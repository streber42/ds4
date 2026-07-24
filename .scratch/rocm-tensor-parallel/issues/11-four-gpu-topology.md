# Four-GPU topology: decide & extend

Status: ready-for-human

## Parent

`.scratch/rocm-tensor-parallel/PRD.md`

## What to build

Decide how to use all four GPUs, then implement that decision.

The upstream tensor-parallel design is natively two-rank. Two credible options exist for four
cards: run two tensor-parallel pairs arranged in a pipeline (reuses the proven two-rank path,
keeps some pipeline serialisation), or widen the design to four tensor-parallel ranks (more
uniform utilisation, but changes sharding and the exchange pattern in ways the upstream code
does not currently implement).

This is a human checkpoint because it is a genuine architectural trade-off with different risk
and effort profiles, informed by the throughput and utilisation numbers measured at the earlier
gate. Picking wrong means either leaving performance on the table or taking on a substantially
larger change than the win justifies.

Peer transfer bandwidth is uniform across all pairs on this machine, so topology asymmetry is
not a constraint on the decision.

## Acceptance criteria

- [ ] Both options are written up with expected utilisation, implementation effort, and risk
- [ ] A decision is recorded with its rationale, informed by the measured two-rank numbers
- [ ] The chosen approach is implemented
- [ ] Correctness is re-validated on four GPUs — logits match reference and the quality fixture passes
- [ ] Throughput and per-GPU utilisation on four GPUs are measured against both the two-rank result and the pipeline baseline
- [ ] If the four-GPU result is not better than two-rank plus pipeline, that finding is recorded rather than buried
- [ ] Results appended to the project's experiment log

## Blocked by

- `.scratch/rocm-tensor-parallel/issues/10-quality-fixture-validation.md`
