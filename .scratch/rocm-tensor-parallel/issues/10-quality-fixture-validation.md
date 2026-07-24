# Full quality-fixture validation

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/PRD.md`

## What to build

Run the project's official multi-case quality fixture against the completed tensor-parallel
build and confirm it scores equivalently to the reference path.

Matching logits on a handful of prompts is necessary but not sufficient. Sharded arithmetic can
be correct for the cases tested and wrong for a routing pattern, sequence length, or expert
distribution that those prompts never trigger. The fixture exists precisely to cover that
spread, and it is the last correctness gate before this is treated as production-usable.

## Acceptance criteria

- [ ] The official multi-case quality fixture runs to completion on the tensor-parallel build
- [ ] Score is equivalent to the reference pipeline path within the fixture's own accepted variance
- [ ] Any case that regresses is investigated and either fixed or documented with a justification
- [ ] Results recorded in the project's experiment log alongside the reference score
- [ ] Both decode and prefill paths are exercised by the run
- [ ] The run is reproducible from a documented command

## Blocked by

- `.scratch/rocm-tensor-parallel/issues/07-tp-prefill-path.md`
- `.scratch/rocm-tensor-parallel/issues/08-auxiliary-tp-hooks.md`
