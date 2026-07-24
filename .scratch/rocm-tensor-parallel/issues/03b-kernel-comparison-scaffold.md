# Kernel comparison scaffold

Status: closed

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

- [x] A single entry point runs one named tensor-parallel kernel against fixed, reproducible inputs
- [x] Output is compared against a reference with an explicit, stated tolerance
- [x] On divergence it reports location and magnitude, not just pass/fail
- [x] Pointing it at a different kernel requires no new bespoke harness
- [x] Inputs are deterministic so results are reproducible across runs and machines
- [x] It runs without loading the full model where the kernel permits
- [x] Follows the existing kernel-level numeric test pattern in the codebase
- [x] Demonstrated end-to-end on at least one already-working kernel, proving the scaffold itself is correct

## Blocked by

- `.scratch/rocm-tensor-parallel/issues/00-stub-inventory-and-loud-failure.md`

## Comments

**Implementation:** `tests/test_rocm_kernel_compare.cu`, a standalone ROCm test binary built and
linked the same way as `tests/test_rocm_tp_stubs.cu` / `tests/test_rocm_xdev.cu` (links directly
against `ds4_rocm.o ds4_rocm_compat.o ds4_rocm_unavailable.o ds4_rocm_xdev.o`, no engine, no
model, no GGUF load).

- **Single entry point, generic mechanism.** `./tests/test_rocm_kernel_compare [--list] [--kernel
  NAME]`. A small registry (`KCMP_CASES`) maps a kernel name to one adapter function that builds
  fixed inputs, invokes the named GPU entry point, computes a CPU reference, and hands both to a
  shared `kcmp_compare_f32()`. The harness itself — determinism, comparison, tolerance, CLI,
  reporting — is written once; pointing it at a new kernel means adding one adapter + one registry
  row, not new comparison plumbing. Demonstrated concretely with two independent kernels
  (`rms_norm_plain`, `add`) sharing the exact same plumbing.
- **Deterministic inputs.** A fixed LCG (`kcmp_lcg_next`), the same scheme
  `tests/test_q4k_dot.c` uses for its block fixtures, so a given case produces bit-identical
  inputs on every run and every machine — no seeded system RNG.
- **Explicit, justified tolerance + divergence reporting.** Each case states its tolerance and why
  (float32 reduction reassociation vs. a double-precision linear reference for `rms_norm_plain`;
  near-zero for `add` since a single IEEE-754 op has nothing to reassociate). On failure the
  report gives the first index that exceeds tolerance, `ref` vs `got` at that index, and the
  global max absolute error — not just PASS/FAIL. Verified by rebuilding with an artificially
  tight tolerance (1e-9) and confirming the exact-index divergence report fires correctly, then
  discarding that build.
- **No model required.** Both demonstration kernels (`ds4_gpu_rms_norm_plain_tensor`,
  `ds4_gpu_add_tensor`) take plain `ds4_gpu_tensor` device buffers with no `model_map`/weight
  lookup, so the binary never touches a GGUF file.
- **Prior art followed.** Mirrors `tests/test_q4k_dot.c`'s pattern (fixed-seed fill, reference
  comparison with a stated tolerance, explicit PASS/FAIL reporting per case) rather than
  inventing a parallel mechanism.
- **Demonstrated on already-working kernels, not a TP kernel.** No ROCm tensor-parallel kernel is
  implemented yet — by design (issue 00) every TP entry point still aborts loudly. So "already
  working" here means `ds4_gpu_rms_norm_plain_tensor` and `ds4_gpu_add_tensor`, two kernels
  already used by the working non-tensor-parallel ROCm pipeline path. A pass proves the
  scaffold's own plumbing (input generation, invocation, comparison, reporting) is correct and
  ready to be pointed at tensor-parallel kernels as they land in later slices, per the PRD's
  "first kernel ported in each subsystem" rule.

**Verified on the real gfx1201 hardware present on this box (4x R9700):**

```
$ make -j8 tests/test_rocm_kernel_compare ROCM_ARCH=gfx1201   # clean build, no warnings
$ ./tests/test_rocm_kernel_compare
ds4: ROCm backend initialized on AMD Radeon AI Pro R9700 (sm_120)
[PASS] rms_norm_plain   max_abs_err=1.19209e-07 over 4096 elements -- ...
[PASS] add              max_abs_err=0 over 8192 elements -- ...

2/2 kernel comparisons passed
```

- `make -j8 test-rocm ROCM_ARCH=gfx1201` — all three ROCm standalone tests pass
  (`test_rocm_tp_stubs`, `test_rocm_xdev`, `test_rocm_kernel_compare`).
- `make -j8 rocm ROCM_ARCH=gfx1201` — full `ds4`/`ds4-server`/`ds4-bench`/`ds4-eval`/`ds4-agent`
  build cleanly, no warnings; only test-only files were added/changed (`tests/test_rocm_kernel_compare.cu`
  and `Makefile` build rules), so the production ROCm pipeline path is unaffected.
