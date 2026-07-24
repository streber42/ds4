# 2-rank TP plumbing runs without crashing

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/PRD.md`

## What to build

Wire the tensor-parallel path far enough that a two-rank run initialises, exchanges data
between ranks, and completes a full forward pass without crashing or hanging.

**Numerical correctness is explicitly out of scope for this slice.** The output is expected to
be wrong. The point is to separate integration failures from math failures: once plumbing is
known-green, a wrong answer in the next slice is unambiguously a kernel arithmetic problem
rather than a transport, ordering, or lifecycle problem. Debugging those two classes of
failure at the same time is what makes ports like this stall.

Use the cross-device module for every inter-rank transfer and the sharding policy module for
every ownership decision — do not re-derive either locally.

**This slice runs under the explicit bring-up mode** introduced by the stub-inventory slice.
Unimplemented kernels fail loudly by default, which would abort the forward pass before the
return path is ever exercised; bring-up mode restores neutral returns so the full round trip
can be observed exactly once, for this purpose. Bring-up mode must be off again by the end of
this slice — it exists to test plumbing, not to ship.

## Acceptance criteria

- [ ] A two-rank tensor-parallel session initialises and both ranks reach steady state
- [ ] Gate and synchronisation hooks fire in the expected order without deadlock
- [ ] Cross-device exchanges actually occur, with data volumes matching what the sharding implies
- [ ] A complete forward pass finishes without crash, hang, or unhandled device error
- [ ] The full round trip is exercised, including the return path back to the coordinating rank
- [ ] Runs clean under repeated invocation (no leak or state carried between runs that breaks a second run)
- [ ] If ranks cannot establish a session, this is reported clearly rather than hanging
- [ ] Numerically incorrect output is acceptable and explicitly noted as deferred to the next slice
- [ ] Bring-up mode is used only here, announces itself while active, and is not left enabled

## Blocked by

- `.scratch/rocm-tensor-parallel/issues/00-stub-inventory-and-loud-failure.md`
- `.scratch/rocm-tensor-parallel/issues/01-cross-device-transfer-module.md`
- `.scratch/rocm-tensor-parallel/issues/02-sharding-policy-module.md`
