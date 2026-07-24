# Remaining auxiliary TP hooks

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/PRD.md`

## What to build

Implement the remaining tensor-parallel entry points that are still no-op stubs after the
decode and prefill paths are working: device-cache handling, model-map registration, MoE
handoff packing, the indexer, and the KV and rope hooks used on the tensor-parallel path.

These are the long tail. Individually they are small, but each one that silently returns a
neutral value is a latent wrong-answer bug that only appears when a particular configuration
or code path is exercised — which may be long after this work is considered finished. The goal
is that no tensor-parallel entry point remains a silent no-op.

Any entry point deliberately left unimplemented must fail loudly rather than returning a
neutral value, so an unsupported path is a clear error instead of quietly wrong output.

## Acceptance criteria

- [ ] The inventory from the stub-inventory slice shows every entry point resolved: implemented, deliberately refused, or proven unreachable
- [ ] No tensor-parallel entry point remains a silent no-op returning a neutral value
- [ ] Each implemented hook has behavioural evidence that it does the right thing; the comparison scaffold is used where the hook is numeric
- [ ] Any hook intentionally left unimplemented still fails loudly when reached
- [ ] Decode and prefill correctness do not regress
- [ ] Ownership decisions come from the sharding policy module

## Blocked by

- `.scratch/rocm-tensor-parallel/issues/06-perf-go-no-go.md`
