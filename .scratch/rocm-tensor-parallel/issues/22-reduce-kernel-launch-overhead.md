# 22 — Reduce kernel launch overhead

Status: ready-for-agent

**What to build:** Profile data shows each decode token fires ~2000 kernel dispatches across 4 GPUs (529 per GPU). Many of these are micro-kernels (5-15μs) where the HIP launch overhead (~5-10μs per dispatch) is comparable to the kernel execution time. Opportunities include fusing sequences like RMS norm + attention QKV projection, or quantize + MoE gate sequence.

Post-fix profile (#18) may change the relative cost distribution, so this should be re-profiled first. The dispatch race fix may also reduce the effective number of sync barriers, which changes the launch pattern.

Suspected fusion candidates from the pre-fix profile:
- `rms_norm_plain_kernel` (~5914 dispatches, 30μs avg) could fuse with the following matmul
- `q8_K_quantize_kernel` (344 dispatches, 57μs avg) could fuse into the MoE gate launch
- `f32_to_f16_kernel` (1689 dispatches, 193μs avg) could be absorbed into its consumer

## Acceptance criteria

- [ ] Post-fix profile identifies the top kernel-launch-overhead cost sources
- [ ] At least one fusion implemented (norm+matmul, quant+gate, or f32_to_f16+consumer)
- [ ] Improvement measured against #19 baseline
- [ ] Quality fixture scores are not regressed

## Blocked by

- `.scratch/rocm-tensor-parallel/issues/19-re-measure-throughput-post-fix.md` — clean baseline needed to measure improvement

## Comments

**2026-07-26 — Marked `ready-for-human` due to missing ROCm toolchain and 4x AMD R9700 GPU hardware.**

### Findings & Environment Audit

1. **Missing ROCm Build & Profiling Toolchain:**
   - Compiler binaries (`cc`, `gcc`, `clang`, `hipcc`, `nvcc`) and profiling tools (`rocprof`, `rocm-smi`, `hipconfig`) are not installed/available in the sandbox execution environment (`which` checks returned not found).
   - `make -j8 rocm` fails immediately with `cc: No such file or directory`.

2. **Missing ROCm Hardware:**
   - 4× AMD Radeon AI PRO R9700 cards (`gfx1201`) required for 4-GPU TP profiling and quality scoring are not attached to this agent's sandbox VM (`SANDBOX_VM_ID=agy-src`).

3. **Technical Context & Recommended Next Steps for Human Operator:**
   - **Post-fix Re-profiling Needed:** Re-profile 4-GPU TP decode tokens using `rocprof` (e.g. 2048 prefill + 256 gen tokens) to identify the updated top kernel launch overheads following the dispatch race fix (#18) and WMMA hot-path always-on changes (#20).
   - **Fusion Candidates to Evaluate:**
     - `rms_norm_plain_kernel` (in `rocm/ds4_rocm_norm_rope.cuh`) + following matmul: Fuse RMS norm calculation directly into the input transformation phase of the subsequent GEMM kernel.
     - `q8_K_quantize_kernel` (in `rocm/ds4_rocm_moe_launch.cuh`) + MoE gate launch: Fuse quantization into expert routing/gate pre-processing.
     - `f32_to_f16_kernel` (in `rocm/ds4_rocm_norm_rope.cuh`, `ds4_rocm_matmul.cuh`) + consumer kernel: Absorb F32-to-F16 conversion into consumer compute kernels to eliminate standalone type conversion launch overhead.
   - **Verification:** Once implemented on hardware with ROCm installed, verify using `make -j8 rocm`, `make -j8 test-rocm`, measure throughput vs #19 baseline (12.44 t/s), and ensure `make rocm-quality` scores do not regress.

