Status: closed
# 01 — Port and A/B benchmark the real gfx12 WMMA v2 kernel

**What to build:** Port `d9be29f`'s gfx12 WMMA v2 kernel (from this repo's
`origin/gfx1201-discrete-gpu` branch) onto dev's current ROCm tree at
`~/src/ds4`, replacing the `DS4_ROCM_NO_WMMA` fallback introduced by
`775ca6a`. Rebuild, verify correctness against the current fallback on a
small fixture, then A/B decode throughput. Full context in `../spec.md`.

**Blocked by:** None — can start immediately.

**Status:** closed

- [x] Read `d9be29f`'s full diff (`git show d9be29f -- rocm/ds4_rocm_q8.cuh`
      in this repo) and port it onto dev's current `~/src/ds4` tree
- [x] Build with `make rdna4` (or equivalent) using the new WMMA v2 path in
      place of `-DDS4_ROCM_NO_WMMA`
- [x] Correctness: output parity check against the current fallback baseline
      on a small fixture (avg_nll or equivalent — reuse
      `../rocm-tensor-parallel`'s quality-fixture tooling if it fits)
- [x] A/B decode t/s: current `DS4_ROCM_NO_WMMA` baseline vs. the ported WMMA
      v2 path, same model/prompt/depth, via `ds4-bench`
- [x] Record the result (win/neutral/regression, with numbers) in `## Answer`
      below
- [x] If it's a win: flag it back to `/home/murphy/src/dev_ds4`'s
      `.scratch/deepseek-benchmark/` campaign as a candidate baseline update
      (their C1/C6 configs run on this same hardware)

## Answer

**Result: modest win on prefill throughput, no effect on decode, no
correctness regression — but the literal `d9be29f` port as written was
wrong on real gfx1201 hardware and had to be reimplemented via `rocwmma`
library fragments instead of the raw builtin intrinsic.**

### What shipped (in `~/src/ds4`, not this repo)

- `ds4_rocm.h`: added `DS4_RDNA4`/`DS4_RDNA3` arch-detection macros
  (`__GFX12__`/`__GFX11__`), exactly as in `d9be29f`.
- `rocm/ds4_rocm_q8.cuh`: `matmul_q8_0_f32_batch_wmma_4w_kernel` now has two
  bodies gated on `DS4_RDNA4`. The gfx11 body is untouched (still the raw
  `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` path). The gfx12 body does
  **not** match `d9be29f` — see "Why the literal port was replaced" below.
- `Makefile`: `rdna4` (the existing default, `-DDS4_ROCM_NO_WMMA`) is
  **unchanged**. Added a new `rdna4-wmma` target that builds the same
  `gfx1201` arch without that flag, so the WMMA v2 path compiles in but
  isn't the default — matching this project's own
  `docs/adr/0001-dense-tp4-parked-sequential-default.md` validate-before-default
  precedent cited in `spec.md`.

### Why the literal port was replaced

`d9be29f`'s diff (`git show d9be29f -- rocm/ds4_rocm_q8.cuh` in this repo)
halves the WMMA input fragment from `<16 x _Float16>` to `<8 x _Float16>`
and swaps in the `_gfx12`-suffixed builtin, keeping the same 2-calls-per-
32-wide-K-tile structure gfx11 uses. Ported literally and run against the
real 81 GiB DeepSeek-V4-Flash model on `~/src/ds4`'s single R9700 (via
`--ssd-streaming`, single-GPU — this dev tree has no multi-GPU sharding,
unlike this repo's TP4 fork), a full-model logits dump (`--dump-logits`)
showed the ported build's next-token argmax completely different from the
`DS4_ROCM_NO_WMMA` baseline (id 35716 "Kasarangang" vs. id 2581 "We"), with
mean absolute logit error ~4.45 across the 129,280-entry vocab (max 34.2) —
far beyond fp16 rounding. 30-token greedy generation from a >256-token
prompt was gibberish ("aisarvioitu, \n& 'B' ...") vs. the baseline's
coherent continuation.

The bug was reachability-confirmed first (not just a "never runs" issue):
`rocprofv3 --kernel-trace` showed 212 real dispatches of
`matmul_q8_0_f32_batch_wmma_4w_kernel` during that one run (avg 1.7ms each,
grid sizes matching the intended 64x64 tiling), and `llvm-objdump` on the
extracted gfx1201 code object confirmed the compiler emitted real
`v_wmma_f32_16x16x16_f16` matrix-core instructions — so the kernel was
compiling, launching, and running on real matrix-core hardware, just
producing wrong numbers.

I wrote a standalone `.hip` microbenchmark (small synthetic Q8_0 weight +
F32 activation buffers, `out_dim=1024, in_dim=256, n_tok=256`, CPU
reference in double precision) to iterate on fragment layouts without a
model load. Three literal-port-shaped hypotheses were tested and all
falsified against the CPU reference:

| candidate | structure | mean_abs err | verdict |
|---|---|---|---|
| A (`d9be29f` as-is) | 2 calls, first-8-of-16 truncated | 106.6 | wrong |
| B | 4 calls, contiguous 8+8 split of each 16-block | 130.4 | wrong |
| C | 4 calls, interleaved (even/odd) 8+8 split | 130.4 | wrong (identical to B — confirms B/C are the same dot-product sum reordered, both wrong) |
| D | `rocwmma::fragment`-based (library, not raw builtin) | **0.021** | correct |

Candidate D's per-element ratio to the CPU reference across the whole
output was ~1.000–1.003 throughout (fp16-precision-level noise), vs. wild,
element-dependent ratios for A/B/C (e.g. -33x, +14x, -0.2x on adjacent
elements) that ruled out "close but needs a scale fix." D's structure:
same 2-sub-block/K=32 outer shape as gfx11, but each 16x16x16 sub-block
uses `rocwmma::fragment<matrix_a, 16,16,16, half, row_major>` /
`fragment<matrix_b, ..., col_major>` (col_major requires no restaging —
it matches the existing per-token-K-contiguous `lds_x` layout directly)
loaded via `load_matrix_sync`/`mma_sync`/`store_matrix_sync`, the same API
this file already uses correctly on this exact hardware for the MoE
hotlist kernels (`moe_down_q2K_hotlist_wmma_n2_kernel` etc., which were
never touched by `775ca6a`'s `DS4_ROCM_NO_WMMA` guard and have been running
in production on gfx1201 the whole time). The raw builtin's true per-lane
operand layout for gfx12's halved fragment remains undocumented outside
AMD's compiler internals; rather than keep guessing, this reuses the
library abstraction that's already proven correct on this hardware.

After the swap, re-verified against the full model: argmax matches the
`DS4_ROCM_NO_WMMA` baseline exactly (id 2581 "We"), the top-5 logit set
matches in order, mean absolute logit error dropped to 0.51 (max 2.98,
consistent with fp16 accumulation noise across a 43-layer forward pass),
and 30-token greedy generation is byte-identical to baseline
("We need to answer the user's query. The user provided a long description
of the DwarfStar project. The query is not explicitly stated, but").

### A/B throughput (`ds4-bench`, `--ssd-streaming`, single GPU, same model/prompt)

| ctx frontier | metric | baseline (`NO_WMMA`) | WMMA v2 (fixed) | delta |
|---|---|---|---|---|
| 2048 | prefill t/s | 99.35, 97.56 (avg 98.46) | 110.39, 115.08 (avg 112.74) | **+14.5%** |
| 4096 | prefill t/s | 190.36 | 194.00 | +1.9% |
| 2048 | decode t/s | 1.54, 2.39 | 1.59, 2.25 | flat (run-to-run noise ≳ any delta) |
| 4096 | decode t/s | 1.54 | 1.54 | flat |

Decode is flat by design, not measurement noise alone: this kernel is only
called from `cuda_matmul_q8_0_tensor_labeled` (`rocm/ds4_rocm_matmul.cuh:277`)
under `n_tok >= 256`, i.e. large prefill batches. Autoregressive decode
calls that same matmul with `n_tok == 1`, which takes a completely
different code path (`matmul_q8_0_f32_sharedx_warp_rows_w32_kernel` /
`matmul_q8_0_f32_warp8_kernel`) untouched by this change. The issue and
`spec.md` both frame this as a "decode t/s" win; on this codebase's actual
call-site gating it is a **prefill-only** win. Both are measured above; the
decode number is reported for completeness and to make that gating
explicit rather than silently reporting a flat number without explanation.

Decode throughput itself (~1.5–2.4 t/s) is dominated by `--ssd-streaming`
expert-cache misses on this single-32GB-GPU host (86 GiB model, working set
capped at 25.49 GiB per the runtime's own log) — visible as "ROCm streaming
expert cache cannot keep ... while preserving 16.00 GiB free" messages that
vary run to run and are orthogonal to this kernel.

### Verification not run, and why

`make test` was not run: it unconditionally requires `nvcc`/CUDA
(`ds4_cuda.o` via `$(NVCC)`) regardless of backend, and this host has no
CUDA toolchain installed (ROCm-only). Confirmed pre-existing and unrelated
to this change — the `test:` target's CUDA dependency is untouched by this
diff (`git diff Makefile` here is 8 lines, all in the `rdna4`/`rdna4-wmma`
section and `.PHONY`/help text). In its place: `make rdna4` and
`make rdna4-wmma` both build cleanly with no warnings from the changed
files, and correctness was verified at three levels (standalone
microbenchmark vs. CPU reference, full-model logits vs. baseline, and
byte-identical greedy generation vs. baseline) as detailed above.

### Follow-up for `dev_ds4`

This is a genuine, if modest and prefill-scoped, win with verified
correctness. Flagging to `/home/murphy/src/dev_ds4/.scratch/deepseek-benchmark/`
as a candidate baseline update for their C1/C6 configs: swap
`ROCM_EXTRA_CFLAGS=-DDS4_ROCM_NO_WMMA` for the new `rdna4-wmma` target (or
just drop the flag) to pick up the prefill speedup on prompts/chunks
≥256 tokens. No decode-path change to account for.

## Comments

Both build targets (`rdna4`, `rdna4-wmma`) were rebuilt clean on this repo's
report-writing pass; `rdna4` (the existing safe default) is what's left
built in `~/src/ds4` at hand-off. Scratch A/B binaries (`*.no_wmma`,
`*.wmma_v2*`), the standalone microbenchmark, and rocprof output under
`/tmp` were not carried into either repo — they were throwaway
verification artifacts, not part of the deliverable. The GPU lock
(`gfx1201-wmma-v2`) was released after the last GPU-dependent step.
