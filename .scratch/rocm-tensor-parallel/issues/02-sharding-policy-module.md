# Sharding policy module + CPU unit tests

Status: closed

## Parent

`.scratch/rocm-tensor-parallel/PRD.md`

## What to build

A pure-logic module that decides ownership for tensor-parallel execution: given the model's
dimensions, the number of ranks, and a rank index, it reports which routed experts that rank
owns, which attention heads it computes, and which vocabulary rows it produces.

This must have no GPU dependency at all. Keeping the trickiest correctness logic in the port
free of device code means it can be exhaustively unit-tested on CPU, and gives every kernel a
single authoritative source of truth for ownership instead of each one re-deriving indices.

This is a deep module: simple inputs, simple outputs, and it absorbs all the sharding
arithmetic that would otherwise be smeared across kernels.

## Acceptance criteria

- [x] Ownership is reported for all three sharded dimensions (routed experts, attention heads, vocabulary rows)
- [x] Every expert, head, and vocabulary row is owned by exactly one rank — a complete partition with no gaps
- [x] No element is owned by more than one rank (no overlaps)
- [x] Uneven division across ranks is handled deterministically and documented
- [x] The single-rank degenerate configuration returns full ownership
- [x] Tests run on CPU with no GPU present and no model loaded
- [x] Kernels and engine code obtain ownership only from this module
- [x] Tests follow the existing multi-GPU placement test pattern already in the codebase

## Blocked by

None - can start immediately.
