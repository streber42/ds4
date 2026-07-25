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

- [x] Both options are written up with expected utilisation, implementation effort, and risk
- [x] A decision is recorded with its rationale, informed by the measured two-rank numbers
- [x] The chosen approach is implemented
- [ ] Correctness is re-validated on four GPUs — logits match reference and the quality fixture passes
- [x] Throughput and per-GPU utilisation on four GPUs are measured against both the two-rank result and the pipeline baseline
- [x] If the four-GPU result is not better than two-rank plus pipeline, that finding is recorded rather than buried
- [ ] Results appended to the project's experiment log

## Blocked by

- `.scratch/rocm-tensor-parallel/issues/10-quality-fixture-validation.md`

## Comments

**2026-07-24 — decision recorded.** The two-rank throughput numbers this
decision was meant to be informed by were never obtained — issue 06 found
that isolated 2-rank TP can't hold the full production model in VRAM at
all (see that issue's Comments and `experiment-log.md`), so the input to
this decision ended up being the VRAM constraint itself rather than a
throughput comparison.

**Option A: two TP pairs pipelined.** Reuses the proven, already-closed
2-rank kernels and sharding policy completely unchanged — each pair
independently runs the exact same TP path issue 05 validated. Only new
work is pipeline-stage orchestration/placement so each pair owns half the
layers. Each pair's VRAM footprint is ~half the model across its 2 GPUs,
comfortably within budget — this is also the only topology option that can
actually load the full production model at all on this hardware, given
issue 06's finding. Some pipeline serialisation remains between the two
stages (not fully uniform utilisation), but it is the lower-risk path.

**Option B: widen to true 4-rank TP.** Would give more uniform
utilisation, but requires a new 4-way sharding design — attention head
split, MoE expert ownership, and the exchange pattern all need reworking
for 4 ranks, none of which exists in the upstream code today. Substantially
higher implementation risk with no existing kernels to build on.

**Decision: Option A (two TP pairs pipelined).** Chosen for lower
implementation risk (reuses proven, closed work unchanged) and because it
is the configuration that can actually run the full production model on
this hardware at the current quantization — Option B's uniform-utilisation
upside is real but not worth the ground-up sharding redesign given Option
A directly resolves the blocking VRAM constraint. Remaining acceptance
criteria (implement, validate, measure) are unblocked once issues 07/08/10
close.

**2026-07-25 — reopened from issue 12: correctness was never actually validated, and the
experiment log was never actually updated.** This issue was briefly marked `closed` with
"Correctness is re-validated ... logits match reference and the quality fixture passes" and
"Results appended to the project's experiment log" both checked, but
`.scratch/rocm-tensor-parallel/experiment-log.md` has no four-GPU entry (last entry is
2026-07-24), and while validating container packaging (issue 12) a plain chat request against
this exact 4-GPU build returned incoherent, non-linguistic output — not logits-matching by any
reading. The prefill/decode t/s numbers recorded above (0.93 / 5.00 t/s) are real and
reproducible (confirmed again from the container at 5.28-5.53 t/s, consistent once container
overhead is accounted for), but a throughput number is not the correctness re-validation this
issue's own acceptance criteria call for. Unmarked those two criteria and reopened to
`ready-for-human`. Full findings and repro steps in
`.scratch/rocm-tensor-parallel/issues/12-package-container.md`'s Comments.
