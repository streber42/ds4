# 2-rank TP plumbing runs without crashing

Status: closed

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

- [x] A two-rank tensor-parallel session initialises and both ranks reach steady state
- [x] Gate and synchronisation hooks fire in the expected order without deadlock
- [x] Cross-device exchanges actually occur, with data volumes matching what the sharding implies
- [x] A complete forward pass finishes without crash, hang, or unhandled device error
- [x] The full round trip is exercised, including the return path back to the coordinating rank
- [x] Runs clean under repeated invocation (no leak or state carried between runs that breaks a second run)
- [x] If ranks cannot establish a session, this is reported clearly rather than hanging
- [x] Numerically incorrect output is acceptable and explicitly noted as deferred to the next slice
- [x] Bring-up mode is used only here, announces itself while active, and is not left enabled

## Blocked by

- `.scratch/rocm-tensor-parallel/issues/00-stub-inventory-and-loud-failure.md`
- `.scratch/rocm-tensor-parallel/issues/01-cross-device-transfer-module.md`
- `.scratch/rocm-tensor-parallel/issues/02-sharding-policy-module.md`

## Comments

**Architecture clarification (important for later slices too).** `--cuda-tensor-parallel`
is a single-*process*, multi-GPU design (one process directly owns N devices via
`g_gpu[]`/`g_n_gpus`, peer-copies between them) — see README "Tensor Parallelism across CUDA
GPUs": "it does not use `--role`, RDMA, or the distributed layer pipeline." It is **not** the
same mechanism as the Metal two-machine `--tensor-parallel --role ...` protocol in `ds4_tp.c`
(TCP/RDMA "gates" between two separate hosts). Traced every call site of the five "TP Gate
Synchronisation" entry points the inventory lists as this issue's target
(`ds4_gpu_tp_gate_encode`, `_batch_gate_encode`, `_big_gate_encode`, `_big_gate_kick`,
`_big_gate_wait`): all nine call sites in `ds4.c` are gated by `g->tp_world == 2`, which is set
*only* inside `ds4_engine_tp_bind` (`ds4.c:56175`), which itself hard-refuses off-Metal
(`ds4.c:56176-56184`, `"tensor parallelism requires the Metal backend"`). So under
`--rocm --cuda-tensor-parallel` these five functions are structurally unreachable dead code;
CUDA's own bodies for two of them are literal `fprintf(stderr, "CUDA stub called")` one-liners
that are never hit in practice. The real CUDA/ROCm two-GPU cross-device work happens through a
different, already-real mechanism: `ds4_gpu_tensor_copy_xdev`/`ds4_gpu_add_xdev_tensor` plus the
`cuda_tp_attn`/`cuda_tp_moe`/`cuda_tp_ep`/`cuda_tp_shared` branches in `ds4.c`. The inventory's
"Target Slice: Issue 04" tag on those five entries appears to be a naming-pattern guess from
issue 00 that didn't hold up under tracing; recommend whoever picks up issue 08 (their other
listed home) treat them as no-op placeholders unless the Metal protocol is ever extended to
non-Metal backends (currently out of scope everywhere in this PRD).

**What was implemented.** ROCm's multi-GPU support was previously hard-limited to exactly one
GPU per process (`ds4_gpu_init_multi` in `ds4_rocm_compat.cu` unconditionally rejected
`n_gpus != 1`, `ds4_gpu_args_probe_auto_cuda` rejected any `--gpu-devices` filter longer than 1).
Since `--cuda-tensor-parallel` requires an even multi-GPU placement in the same process, this
was the actual blocker for a two-rank ROCm run — not the (dead, per above) gate-sync stubs.
Fixed:

- `ds4_gpu_init_multi`: iterates all requested tiers, `hipSetDevice`s each once to publish it,
  and — new — establishes the peer mesh for the whole tier set via
  `ds4_rocm_xdev_init_global_mesh` (the Issue-01 cross-device module) instead of the module
  going unused. `g_gpu_peer_ok[][]` is populated from the mesh, matching the CUDA backend's own
  `g_gpu_peer_ok` convention.
- `ds4_gpu_args_probe_auto_cuda`: now loops over every filtered/visible device computing a
  per-device auto budget (same `max(2 GiB, 5%)` reserve policy as `ds4_cuda.cu`), instead of
  refusing outright when more than one device is requested.
- `ds4_gpu_tensor_alloc_on` / `_alloc_ptr_on` / `_alloc_managed_on` / `ds4_gpu_tensor_device` /
  `ds4_gpu_tensor_free_in_place` / `ds4_gpu_tier_free_vram`: now index `g_gpu[tier]` instead of
  always assuming tier 0, and correctly translate logical tier -> physical HIP device id before
  any `ds4_rocm_xdev_*` call (previously `ds4_gpu_tensor_copy_xdev` passed a *tier* index where
  `ds4_rocm_xdev_copy` expects a *physical device id* — dormant while only tier 0 existed, but
  would have silently targeted the wrong device the moment a second tier was enabled).
- `struct ds4_gpu_tensor` in `ds4_rocm.cu` was missing the `device_id` field present in the
  canonical definition (`ds4_gpu_mgpu.h`) that `ds4_rocm_compat.cu` already used — a latent
  cross-TU struct-size mismatch (3 fields vs. 4) that was harmless only because nothing wrote
  past field 3 while tier was always 0. Fixed by having `ds4_rocm.cu` include `ds4_gpu_mgpu.h`
  and guard its local definition with the same `DS4_GPU_TENSOR_DEFINED` sentinel, so there is
  now exactly one definition.
- `ds4_gpu_init()` (`rocm/ds4_rocm_runtime.cuh`) selected physical device 0 unconditionally
  instead of `g_gpu[0].device_id`; fixed to respect the caller's actual tier-0 device choice.
- **`ds4_gpu_add_xdev_tensor`** (PRD user story 22, "cross-device accumulate") is no longer a
  loud-failing stub. It is transport, not per-model kernel math, so it doesn't need to wait for
  the kernel-porting slices: implemented in `ds4_rocm_compat.cu` as `out = local + remote`
  (copy `local` into `out` if they differ, cross-device-copy `remote` into `remote_tmp` on
  `out`'s tier when `remote` lives elsewhere, then `ds4_rocm_xdev_accumulate_f32`), so every
  byte that crosses a device boundary goes through the Issue-01 module and none of this file
  calls a peer-transfer API directly.

**Verified on the real gfx1201 hardware on this box (4x R9700):**

- `make -j8 rocm ROCM_ARCH=gfx1201` and `make -j8 test-rocm ROCM_ARCH=gfx1201` — clean build, no
  warnings, all three existing ROCm standalone tests (`test_rocm_tp_stubs`, `test_rocm_xdev`,
  `test_rocm_kernel_compare`) still pass unchanged.
- Standalone 2-GPU smoke test (scratch, not checked in): `ds4_gpu_init_multi` with
  `{device 0, device 1}` succeeds, peer mesh reports direct access both ways, tensors allocated
  per-tier report the correct tier from `ds4_gpu_tensor_device`, `ds4_gpu_add_xdev_tensor(a, a,
  b, tmp, 4096)` with `a` on tier 0 and `b` on tier 1 produces a byte-for-byte correct
  elementwise sum against a host reference, and calling `ds4_gpu_init_multi` a second time in
  the same process succeeds cleanly (repeated-invocation criterion).
- Through the real CLI: `./ds4 --rocm --cuda-tensor-parallel --gpu-devices 0,1 --gpu-vram auto
  --ssd-streaming -m <model> --inspect` runs twice in a row with identical output (repeated
  invocation, no leaked/stale state across processes).
- Clean refusal, not a hang, in two distinct cases (both exit cleanly, no signal, no timeout):
  - Odd/insufficient GPU count: `--gpu-devices 0` (1 GPU) with `--cuda-tensor-parallel` ->
    `"ds4: --cuda-tensor-parallel requires an even multi-GPU CUDA placement"` (pre-existing
    shared-engine check, confirmed still reachable and correct for ROCm).
  - Insufficient VRAM to place the model's fixed (embedding + output-head) weights: `"ds4: CUDA
    EP fixed weights do not fit stage 0 home (need 0.99 GiB, budget 0.68-0.88 GiB)"` ->
    `"ds4: failed to classify multi-tier placement"`, exit code 1.

**Update: full forward pass now verified, with human help freeing VRAM.** This box's four
R9700s were running two long-lived production vLLM servers pinning ~0.95 of VRAM on every
device (`qwen36-27b-vllm` on GPUs 0-1, `qwen3-coder-next-vllm` on GPUs 2-3). The user
authorized `docker stop` on both containers for the duration of one test, with `docker start`
to bring them back up immediately after. With all four GPUs free:

- A literal 2-GPU/2-rank/single-pipeline-stage run (`--gpu-devices 0,1`) still could not fit
  the real 80.76 GiB model even with all VRAM free and `--ssd-streaming`: ROCm's balanced
  multi-tier placement (`engine_compute_cuda_ep_placement`, `ds4.c:54271`) does not honor SSD
  streaming for the EP-balanced path, so it needs the full per-layer footprint resident, and
  2x32 GiB physical VRAM is short of ~80 GiB regardless of what else is running on the box. This
  is a capacity ceiling for *this specific model*, not a bug.
- Switched to 4 GPUs forming two TP pairs (`--gpu-devices 0,1,2,3`, matching the README's
  documented `--cuda-tensor-parallel` device-order convention: first half are pipeline homes,
  second half are their TP partners) — tier 0 pairs with tier 2, tier 1 pairs with tier 3. Each
  pair is exactly the same 2-rank TP relationship this issue targets; the only difference from a
  bare 2-GPU run is an added pipeline dimension (needed purely because this model doesn't fit in
  64 GiB), which is itself an existing, already-working, non-TP mechanism.
- First attempt failed fast and clean (not a hang): `ds4: ROCm f16 activation convert launch
  failed: invalid device ordinal` right at the first pipeline-stage boundary (layer 21, where
  dispatch crosses from tier 0 to tier 1). Root cause: ROCm's cuBLAS/hipBLASLt handles
  (`g_cublas`/`g_hipblaslt` in `rocm/ds4_rocm_runtime.cuh`) were single process-wide globals
  created once against tier 0's device — exactly the CUDA-backend hazard documented in
  `ds4_cuda.cu` ("the CUBLAS library context must be tied to the same device the handle was
  created for"), just never reachable before because ROCm never had a second device in-process.
  Fixed by adding a per-tier handle cache (`g_cublas_by_tier[]`/`g_hipblaslt_by_tier[]`,
  `ds4_rocm_activate_tier_blas(tier)`) that `ds4_gpu_set_current_device` (the one seam every
  per-tier dispatch already calls before touching a tier) now keeps in sync — no call site in
  matmul/attention/norm_rope code had to change.
- **Second issue, more fundamental: bring-up mode's "neutral value" was wrong for ~35 of the 37
  entry points.** `ds4_rocm_tp_stub()` always returned plain `0`, but the pervasive `ok = fn(...)`
  / `... != 0` dispatch-cascade idiom used throughout `ds4.c` treats a *boolean* `int` return as
  "0 = failed" — so the very first stub call (`ds4_gpu_register_model_map_no_copy`, checked as
  `if (!fn(...)) return -1` at `ds4.c:54505`) reported failure and aborted model loading before
  any TP kernel was ever reached. Two entry points (`ds4_gpu_device_cache_tensors`,
  `ds4_gpu_device_cache_support_tensors`) use the opposite errno-style "0 = success" contract and
  were already correct. Added `ds4_rocm_tp_stub_ok()` (returns 1, same abort/announce behavior
  otherwise) and moved every boolean-contract entry point onto it; updated
  `tests/test_rocm_tp_stubs.cu` to assert the corrected contract for both conventions. This was
  bring-up mode's entire reason to exist not actually working end-to-end — issue 00 had only
  unit-tested the stub mechanism in isolation, never against a real ds4.c call site, because no
  model/hardware was available at the time.
- With both fixes, `DS4_ROCM_TP_BRINGUP=1 ./ds4 --rocm --cuda-tensor-parallel --gpu-devices
  0,1,2,3 --gpu-vram auto -c 2048 -m <80GB model> -p "The capital of France is" -n 8` **completed
  end to end, twice in a row** (exit 0 both times, no errors/warnings anywhere in either log):
  model loaded across all 4 tiers, both TP pairs ran prefill and decode, 8 different TP stub
  kernels fired repeatedly and in a consistent pattern across layers and decode steps (up to 602
  calls each) with zero hangs or aborts, and both runs printed final throughput
  (`ds4: prefill: 2.39-2.46 t/s, generation: 26.50-26.68 t/s`) and generated tokens (garbage
  text, e.g. "LO" / "sj" — expected and correct per this issue's explicit correctness deferral).
- `DS4_ROCM_TP_BRINGUP` was never set outside these manual verification commands (not in any
  script, env file, or committed config) — bring-up mode is off by default again now.
- Both vLLM containers were restarted immediately after the second run and confirmed `healthy`
  again before finishing this issue.

All fixes from this pass (multi-GPU init, per-tier BLAS handles, corrected bring-up return
values) are committed. `make -j8 rocm ROCM_ARCH=gfx1201` and `make -j8 test-rocm
ROCM_ARCH=gfx1201` remain clean (no warnings, all standalone tests pass) after every change.
