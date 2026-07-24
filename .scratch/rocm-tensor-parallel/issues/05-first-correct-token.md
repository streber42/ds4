# First correct token on 2-rank TP (decode path)

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/PRD.md`

## What to build

The tracer bullet. Make the two-rank tensor-parallel **decode** path numerically correct for a
minimal single-token prompt, so that one token comes out matching the reference.

This is the narrowest complete path through every layer — transport, sharding policy, sharded
kernels, engine integration, and validated logits — and it is the real go/no-go for the whole
port. Scope is deliberately limited to the kernels the decode path actually exercises for a
one-token prompt; prefill-path kernels are a later slice.

End-to-end logits agreement is the gate. Per the testing strategy, the **first kernel ported in
each subsystem** additionally gets kernel-level numeric-equivalence evidence via the comparison
scaffold — enough to prove the porting approach is numerically sound before it is repeated —
rather than every kernel carrying a standing test. When end-to-end fails, the scaffold is the
localization tool.

## Acceptance criteria

- [ ] Logits for a single-token prompt match the same-hardware pipeline reference within the harness tolerance
- [ ] The correctness harness reports pass
- [ ] Under greedy sampling the generated token is identical to the reference
- [ ] The first kernel ported in each subsystem touched here has kernel-level numeric-equivalence evidence via the scaffold
- [ ] The tolerance used is stated and justified (float reassociation under a different sharding is expected; unexplained drift is not)
- [ ] Result is reproducible across repeated runs, not a one-off pass
- [ ] Prefill-path kernels remaining unported is explicitly noted as deferred

## Blocked by

- `.scratch/rocm-tensor-parallel/issues/03-correctness-harness.md`
- `.scratch/rocm-tensor-parallel/issues/03b-kernel-comparison-scaffold.md`
- `.scratch/rocm-tensor-parallel/issues/04-tp-plumbing-no-crash.md`
