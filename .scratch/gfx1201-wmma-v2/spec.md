# Spec: port the real gfx12 WMMA v2 kernel for gfx1201 (RDNA4, R9700)

Status: ready-for-agent

## Problem statement

dev's current ROCm build for gfx1201 (`~/src/ds4` `make rdna4` →
`ROCM_ARCH=gfx1201 ROCM_EXTRA_CFLAGS=-DDS4_ROCM_NO_WMMA`) disables matrix-core
codegen entirely and falls back to a non-matrix-core kernel path. This was
introduced by commit `775ca6a` ("ROCm: support discrete RDNA4 (gfx1201) via
opt-in WMMA fallback and configurable VRAM reserve") because the existing
gfx11-style WMMA intrinsic (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`) does
not exist on gfx12 hardware. The same author's own upstream PR
(https://github.com/antirez/ds4/pull/599) states outright: *"This is a
compile/runtime workaround, not an RDNA4 WMMA v2 implementation... A real
RDNA4 WMMA v2 kernel would recover that performance; this PR intentionally
scopes to 'unbreak the build' rather than attempting the v2 intrinsic without
hardware to validate against."*

A real implementation already exists, unmerged, on a different branch in this
same repo: `origin/gfx1201-discrete-gpu`, commit `d9be29f` ("rocm: add gfx12
(RDNA 4) WMMA intrinsics for gfx1200/gfx1201", author Donato Capitella,
2026-07-06 — predates `775ca6a` by a week). It implements the actual gfx12
WMMA v2 encoding: 8-element `_Float16` input fragments (vs. gfx11's 16) and
the `_gfx12`-suffixed builtin `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`,
gated on a new `DS4_RDNA4` macro, leaving the existing gfx11 path untouched.
Independently corroborated as the correct approach: the identical builtin is
already shipping in `ggml-org/llama.cpp` (`ggml/src/ggml-cuda/mma.cuh`, per
`gh search code`).

It has never been merged anywhere (not into `kyuz0/ds4` main, not into
`antirez/ds4` upstream, not into dev's `~/src/ds4`) and never benchmarked on
real R9700 hardware. This is orthogonal to the parked tensor-parallel /
expert-parallel work in `../rocm-tensor-parallel/` — it's a possible free
decode-speed increment on top of whatever multi-GPU topology (pipeline, TP,
or future EP) is running, since it affects the per-GPU matrix-core kernel
itself, not the cross-GPU communication pattern.

Full research trail and citations:
`/home/murphy/src/dev_ds4/.scratch/deepseek-benchmark/artifacts/research-ds4-r9700-rocm.md`
(section 2).

## What to build

1. Port commit `d9be29f`'s WMMA v2 kernel (from `origin/gfx1201-discrete-gpu`
   in this repo) onto dev's current ROCm tree at `~/src/ds4`. Read the full
   diff first (`git show d9be29f -- rocm/ds4_rocm_q8.cuh` in this repo) —
   dev's tree has diverged since `d9be29f`'s branch point, so this is a port,
   not a clean cherry-pick; check for conflicts against dev's own gfx1201
   changes (`775ca6a` and anything after).
2. Rebuild with the `rdna4` target, replacing `-DDS4_ROCM_NO_WMMA` with
   whatever macro gates the new path (`DS4_RDNA4` in the source branch, or
   dev's own naming if it differs after the port).
3. A/B benchmark against the current `DS4_ROCM_NO_WMMA` fallback baseline
   using the existing `ds4-bench` harness (see `../rocm-tensor-parallel/POC.md`
   or the top-level `AGENT.md` for the standard invocation) — same model,
   same prompt depths, decode t/s comparison.
4. If it's a clear win with no correctness regression: leave both paths in
   the tree (matching this project's own precedent of validate-before-default
   from `docs/adr/0001-dense-tp4-parked-sequential-default.md`), document the
   result, and note it back to `/home/murphy/src/dev_ds4`'s benchmark
   campaign (`.scratch/deepseek-benchmark/`) since dev's baselines there
   (C1/C6) would inherit the speedup if adopted.
5. If it does not build, is not correct, or is not faster: document why and
   close as a documented negative result — this branch predates dev's own
   gfx1201 support by a week and was never validated on real hardware, so a
   non-trivial chance it needs further work before it's usable as-is.

## Acceptance criteria

- [ ] `d9be29f`'s WMMA v2 kernel ported onto dev's current ROCm tree, builds
      cleanly with `make rdna4` (or equivalent)
- [ ] Correctness check: output parity with the current `DS4_ROCM_NO_WMMA`
      baseline on a small fixture (reuse `../rocm-tensor-parallel`'s
      quality-fixture tooling if applicable)
- [ ] Decode t/s A/B measured against the current fallback baseline, same
      model/prompt/depth
- [ ] Written result (win/neutral/regression, with numbers) recorded in this
      issue's `## Answer`, and if it's a win, flagged back to the
      `dev_ds4` benchmark campaign as a candidate baseline update

## Blocked by

None — can start immediately. Real GPU work; agents have direct ROCm access
per this repo's `AGENTS.md` GPU locking protocol (acquire the GPU lock before
building/benchmarking).
