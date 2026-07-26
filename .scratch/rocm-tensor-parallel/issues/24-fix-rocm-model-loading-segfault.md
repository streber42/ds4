# 24 — Fix ROCm model loading segfault

Status: closed

**What to build:** All ROCm inference paths segfault during model loading on AMD Radeon AI Pro R9700 (gfx1201) hardware. The crash occurs after "ROCm preparing model tensor mappings" and during "loading model tensors into device cache", before any inference kernels execute.

This blocks quality verification for issue #22 (kernel launch overhead fusion) and all other ROCm quality/performance testing.

## Reproduction

```bash
# Any of these segfault with the same backtrace:
./ds4 -p "Hello" -n 5 --model ./tests/mini_ds4flash.gguf
./tests/test_engine_correctness_harness-rocm --quality ./tests/mini_ds4flash.gguf \
    gguf-tools/quality-testing/data/flash/manifest.tsv /tmp/out.tsv
./tests/test_rocm_kernel_compare

# All crash at the same point:
# ds4: ROCm backend initialized on AMD Radeon AI Pro R9700 (sm_120)
# ds4: ROCm preparing model tensor mappings
# ds4: ROCm loading model tensors into device cache
# Segmentation fault
```

## Context

- Hardware: 4× AMD Radeon AI Pro R9700 (gfx1201, 0x7551), ROCm 7.14
- The crash happens on the main branch (commit b62c4ba) and on the gfx1201_tp branch with issue #22 fusion changes
- `make -j8 rocm` builds successfully
- `tests/test_rocm_tp_stubs` passes (doesn't load a model)
- The crash is NOT caused by the issue #22 fusion — it's a pre-existing bug in model loading

## Investigation hints

- The crash happens in the tensor loading path, likely in `ds4_rocm.cu` or `rocm/ds4_rocm_runtime.cuh`
- Look for the code that prints "ROCm loading model tensors into device cache" and trace what it does next
- Check if there's a buffer allocation failure, invalid pointer, or GPU memory access error
- Try running with `HSA_ENABLE_SDMA=0` or other ROCm debug environment variables
- Consider adding error checking around hipMalloc/hipMemcpy calls in the loading path
- The mini test model (`tests/mini_ds4flash.gguf`) is small enough that OOM is unlikely

## Acceptance criteria

- [x] Root cause identified (which pointer/allocation/access is invalid)
- [x] Fix applied and model loading completes without segfault
- [x] `./ds4 -p "Hello" -n 5 --model ./tests/mini_ds4flash.gguf` runs successfully
- [x] Quality fixture can execute: `./tests/test_engine_correctness_harness-rocm --quality ./tests/mini_ds4flash.gguf gguf-tools/quality-testing/data/flash/manifest.tsv /tmp/out.tsv`
- [x] No regression in existing passing tests (`test_rocm_tp_stubs` still passes)

## Root cause

The Makefile's `ROCM_ARCH` defaulted to `gfx1151` (Strix Halo APU), but the hardware is `gfx1201` (R9700 discrete GPU). The resulting binary contained code objects only for gfx1151. When HIP attempted to launch the first kernel (`dequant_q8_0_to_f16_kernel`) on the gfx1201 device, the runtime could not find a compatible code object and segfaulted inside `libamdhip64.so` instead of returning a clean error.

Confirmed via `AMD_LOG_LEVEL=3`:
```
hip_module.cpp: hipLaunchKernel ( 0x245760, ...)
hip_fatbin.cpp: No compatible code objects found for: gfx1201
```

GDB backtrace placed the crash in `__device_stub__dequant_q8_0_to_f16_kernel` → `libamdhip64.so`.

## Fix

1. **Makefile `ROCM_ARCH` default changed to `gfx1151,gfx1201`** — the ROCm binary is now multi-arch and contains code objects for both Strix Halo (gfx1151) and R9700 (gfx1201). A single binary works on either hardware. The `strix-halo` target remains a gfx1151-only build for those who want it.
2. **Runtime arch-mismatch probe added to `ds4_gpu_init()`** — before any real kernel launch, the init path now calls `hipFuncGetAttributes` on a tiny probe kernel. If the binary has no code object for the active GPU, it fails loudly with a clear error message naming the device arch and the ROCM_ARCH to rebuild with, instead of crashing later.

## Comments

Verified behavior:

- `make rocm` (default) → multi-arch binary → runs cleanly on gfx1201, generates tokens correctly (~45 t/s prefill, ~530 t/s decode on mini model)
- `make rocm ROCM_ARCH=gfx1151` (wrong arch on purpose) → clear error at init:
  ```
  ds4: ROCm kernel code-object probe failed on AMD Radeon AI Pro R9700 (gfx1201): invalid kernel file
  ds4: ROCm this binary was not compiled for the active GPU architecture.
  ds4: ROCm rebuild with a matching ROCM_ARCH, for example: make rocm ROCM_ARCH=gfx1201
  ds4: ROCm backend unavailable; aborting startup
  ```
- All 4 `test-rocm` suites pass: `test_rocm_tp_stubs`, `test_rocm_xdev` (cross-device transfer, peer copy 23.6 GB/s, host-staging 13.7 GB/s), `test_rocm_kernel_compare` (6/6 kernel comparisons), `test_engine_rocm_tp_refusal`
- Quality fixture completes all 100 cases without segfault
