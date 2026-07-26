# 22 — Reduce kernel launch overhead

**What to build:** Profile data shows each decode token fires ~2000 kernel dispatches across 4 GPUs (529 per GPU). Many of these are micro-kernels (5-15μs) where the HIP launch overhead (~5-10μs per dispatch) is comparable to the kernel execution time. Opportunities include fusing sequences like RMS norm + attention QKV projection, or quantize + MoE gate sequence.

Post-fix profile (#18) may change the relative cost distribution, so this should be re-profiled first. The dispatch race fix may also reduce the effective number of sync barriers, which changes the launch pattern.

Suspected fusion candidates from the pre-fix profile:
- `rms_norm_plain_kernel` (~5914 dispatches, 30μs avg) could fuse with the following matmul
- `q8_K_quantize_kernel` (344 dispatches, 57μs avg) could fuse into the MoE gate launch
- `f32_to_f16_kernel` (1689 dispatches, 193μs avg) could be absorbed into its consumer

## Blocked by

- #19 — clean baseline needed to measure improvement

## Status

ready-for-agent

- [ ] Post-fix profile identifies the top kernel-launch-overhead cost sources
- [ ] At least one fusion implemented (norm+matmul, quant+gate, or f32_to_f16+consumer)
- [ ] Improvement measured against #19 baseline
- [ ] Quality fixture scores are not regressed
