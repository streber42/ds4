# Kernel comparison scaffold

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/PRD.md`

## What to build

A reusable harness that runs a **single** tensor-parallel kernel with fixed inputs and compares
its output against a reference, reporting where and by how much it diverged.

This is a prefactor: build the comparison mechanism once rather than improvising it separately
for each kernel. It serves two purposes in the testing strategy:

1. **Pattern validation** — the first kernel ported in each subsystem uses it to prove the
   porting approach is numerically sound before that approach is repeated across the subsystem.
2. **On-demand localization** — when the end-to-end logits comparison fails, this is the tool
   that answers "which kernel?" without bisecting by hand across the whole tensor-parallel path.

The testing strategy deliberately does *not* require a standing per-kernel test for all
kernels; end-to-end logits is the gate. That strategy is only affordable because this scaffold
makes localization cheap, so it must be genuinely easy to point at an arbitrary kernel.

Follow the existing kernel-level numeric test already in the codebase as prior art rather than
inventing a parallel mechanism.

## Acceptance criteria

- [ ] A single entry point runs one named tensor-parallel kernel against fixed, reproducible inputs
- [ ] Output is compared against a reference with an explicit, stated tolerance
- [ ] On divergence it reports location and magnitude, not just pass/fail
- [ ] Pointing it at a different kernel requires no new bespoke harness
- [ ] Inputs are deterministic so results are reproducible across runs and machines
- [ ] It runs without loading the full model where the kernel permits
- [ ] Follows the existing kernel-level numeric test pattern in the codebase
- [ ] Demonstrated end-to-end on at least one already-working kernel, proving the scaffold itself is correct

## Blocked by

- `.scratch/rocm-tensor-parallel/issues/00-stub-inventory-and-loud-failure.md`
