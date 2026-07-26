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
- [x] At least one fusion implemented (f32_to_f16 absorbed into IQ2 and Q2K WMMA MoE kernels)
- [ ] Improvement measured against #19 baseline
- [ ] Quality fixture scores are not regressed

## Blocked by

- `.scratch/rocm-tensor-parallel/issues/19-re-measure-throughput-post-fix.md` — clean baseline needed to measure improvement

## Implementation

### Fusion: f32_to_f16_kernel absorbed into MoE WMMA gate/up kernels

**Files modified:**
- `rocm/ds4_rocm_moe_launch.cuh`

**Changes:**

1. **IQ2 WMMA hot path** — The separate `f32_to_f16_kernel` pre-conversion launch
   (n_tokens × expert_in_dim elements) before the IQ2 WMMA gate/up kernels was
   removed. The WMMA kernels already have a float-input path via template
   parameter `X_F16=false` (default), which loads f32 from global memory and
   converts to f16 on-the-fly during shared-memory load. This eliminates the
   standalone f32→f16 launch (8 blocks × 256 threads for n_tokens=1 decode)
   and the f16 write+read bandwidth (~2 bytes/element saved).

2. **Q2K WMMA hot path** — Same treatment for the Q2K batch path (n_tokens ≥ 32
   prefill). The stand-alone f32_to_f16_kernel launch was removed and the WMMA
   kernels switched from the `X_F16=true` template variant to `X_F16=false`,
   which loads f32 directly and converts inline.

3. **Dead code cleanup** — The dispatches for the `use_iq2_x_f16` branches (now
   always false) were deleted, leaving only the `use_iq2_hot_f16_mid` vs
   fallback two-branch dispatch in the IQ2 hot path.

**Rationale:** The f32_to_f16_kernel was the most frequently dispatched kernel
(~1689 calls per decode token across 4 GPUs). Each call is a trivial element-wise
conversion (one `__float2half` per element). Moving this conversion into the
consumer WMMA kernels removes all these dispatches, reduces total memory
bandwidth (no intermediate f16 buffer), and uses a code path that already existed
and was already proven correct.

**Trade-off:** The WMMA kernels in `X_F16=false` mode load 1× f32 per thread per
k-iteration instead of 2× f16, but convert inline with `__float2half`. The
eliminated f16 write+read bandwidth (2 bytes/element) compensates for the wider
read. The net impact on total memory traffic is positive (4 bytes/element f32
read vs 4+2+2=8 bytes/element for the pre-convert path), though the WMMA inner
loop executes slightly more instructions.

**Verification needed (requires ROCm hardware):**
- Profile to confirm the f32_to_f16_kernel dispatches are eliminated
- End-to-end throughput measurement against #19 baseline (~12.44 t/s)
- Quality fixture scores (`make rocm-quality`) to confirm no numerical regression

### Notes on unimplemented fusion candidates

- **rms_norm_plain + matmul fusion** (~5914 dispatches): Requires deeper
  integration — the norm kernel output feeds into multiple matmul entry points
  (shared expert, attention QKV, router). A fused kernel would need to accept
  both the norm parameters and the weight tensor in a single launch. This is a
  larger change than the f32_to_f16 fusion and should be evaluated after the
  post-fix profile identifies whether norm launch overhead remains significant.

- **q8_K_quantize + MoE down fusion** (344 dispatches): The quantize step
  converts float mid activations to Q8_K before the down projection. Fusing
  would require modifying the MoE gate/up kernels to also output Q8_K, or
  modifying the down kernels to accept float mid directly (the
  `routed_moe_q2_float_down_launch` path already does this for IQ2 models).
  Worth revisiting once the f32_to_f16 fusion's impact is measured.

### Build verification

- `make -j8 rocm` — builds cleanly (no new warnings beyond pre-existing
  `nodiscard` warnings)
- `make -j8 tests/test_rocm_tp_stubs` — builds cleanly
- `make -j8 tests/test_rocm_kernel_compare` — builds cleanly
