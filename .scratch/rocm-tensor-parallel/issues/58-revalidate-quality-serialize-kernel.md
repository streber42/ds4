# 58 — Re-validate quality fixture with AMD_SERIALIZE_KERNEL=3 & align prefill weight path

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/issues/55-tp4-throughput-quality-revalidation.md`

## What to build

Re-run the full 100-case `score_official` quality fixture under `AMD_SERIALIZE_KERNEL=3`
to isolate the root cause of the TP=4 `avg_nll` divergence (0.761 vs 0.369 pipeline baseline).

Historical passing quality fixture runs (#10, #32, #48) all executed under
`AMD_SERIALIZE_KERNEL=3`. Without serialization, open issue #23 documents an
intra-device race in `ds4_gpu_compressor_prefill_tensor` (`ds4_rocm_compressor.cuh`)
during compressed-KV cache prefill. The AI consultant panel confirmed that if
`score_official` under `AMD_SERIALIZE_KERNEL=3` restores `avg_nll` to ~0.370-0.378,
the quality divergence is proven to be from issue #23's compressor prefill race
(and VRAM starvation), not floating-point accumulation drift from the 86 all-reduces.

Also audit and align `g_use_host_weights` in `ds4.c` / `rocm/ds4_rocm_runtime.cuh`.
Currently `g_use_host_weights` is enabled during TP=4 decode but not during prefill,
causing prefill vs decode weight resolution discrepancies between host-mapped and
cached VRAM pointers.

## Acceptance criteria

- [ ] Full 100-case `score_official` quality fixture run under `AMD_SERIALIZE_KERNEL=3`
      for both pipeline and TP=4 paths
- [ ] Confirmed whether `avg_nll` under serialization returns to the ~0.370 band
      (isolating issue #23 compressor race)
- [ ] `g_use_host_weights` aligned across prefill and decode paths in `ds4.c`
- [ ] `make -j8 test-rocm` passes
- [ ] Findings recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`

## Blocked by

`.scratch/rocm-tensor-parallel/issues/55-tp4-throughput-quality-revalidation.md`
