# Cross-device transfer module + standalone tests

Status: closed

## Parent

`.scratch/rocm-tensor-parallel/PRD.md`

## What to build

A ROCm-backend module that owns every cross-GPU peer transfer concern behind a narrow
interface: establish the peer mesh, copy a buffer from one device to another, and accumulate a
buffer from another device into a local one.

Internally it must detect whether direct peer access is actually available between a given
pair of devices (rather than assuming it), enable access in both directions where supported,
and transparently fall back to host staging where it is not. After this lands, no kernel,
launcher, or engine code should call peer-transfer APIs directly — this module is the only
place transport behaviour lives, so it can change without touching compute.

This is a deep module: a lot of hardware-dependent complexity behind three operations that
should rarely change shape.

## Acceptance criteria

- [x] Module exposes exactly the three operations (establish mesh, copy, accumulate)
- [x] No peer-transfer API calls exist anywhere outside this module
- [x] Peer capability is detected per device pair, not assumed
- [x] Peer access is enabled in both directions where the pair supports it
- [x] Byte-exact copy correctness verified across every ordered device pair
- [x] Accumulate produces numerically correct sums against a CPU reference
- [x] Host-staging fallback engages when direct peer access is unavailable, and remains correct
- [x] Tests run standalone on plain device buffers without loading the model
- [x] A bandwidth floor is asserted so a future regression to a slow path is caught
- [x] Tests are runnable via the project's normal test entry point

## Blocked by

None - can start immediately.
