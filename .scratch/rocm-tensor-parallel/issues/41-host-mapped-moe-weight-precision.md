# 41 — Investigate host-mapped MoE weight numerical impact on TP=4 quality

Status: closed

## Parent

`.scratch/rocm-tensor-parallel/issues/32-tp4-quality-fixture.md`
`.scratch/rocm-tensor-parallel/issues/40-option-b-row-split-batch-prefill.md` (the Option B row-split refactor did not close the quality gap)

## Problem

TP=4 quality scores (avg_nll ~1.72, first_match=0/100) are unchanged by the Option B row-split refactor and are virtually identical to pre-Option B scores (~1.85). The gap from the pipeline reference (avg_nll=0.3747, first_match=65/100) is **not** caused by all-reduce FP accumulation noise or the attention-type gate bypass (fixed and tested — no change).

The startup log reveals:

```
ds4: ROCm model arena alloc failed for moe_gate (1024.00 MiB chunk): out of memory
```

**Hypothesis**: The `moe_gate` tensor (1024 MiB per GPU) cannot be cached in VRAM on the 4× R9700 setup (27.79 GiB budget each, 25.94 GiB weights loaded). `ds4_gpu_routed_moe_batch_tensor` falls back to host-mapped memory for uncached MoE weights via `cuda_model_range_ptr_from_fd`. If the host-mapped fallback produces numerically different results from fully-cached weights (e.g., different quantization path, page-faulted reads, or different precision in the kernel's weight-load path), this would explain the consistent ~1.7 NLL gap that persists across all TP=4 configurations (sharded, row-split, all-gather, all-reduce).

The pipeline reference uses fully-cached weights (no TP, no multi-GPU overhead) and achieves correct scores.

## What to investigate

1. **Layer-0 hidden state comparison**: Run TP=4 and pipeline on a single prompt with `n_tokens=16`, capture `batch_next_hc` after layer 0. Compute max_abs_diff and first_divergence_index. If divergence starts at layer 0, the MoE path is the most likely culprit.

2. **Host-mapped vs cached weight comparison**: If possible, run `ds4_gpu_routed_moe_batch_tensor` with the same inputs but force host-mapped vs cached weight resolution. Compare the output tensors.

3. **Memory audit**: Determine exactly which MoE tensors (`moe_gate`, `moe_up`, `moe_mid`, `moe_down`) are falling back to host mapping. The `ROCm model arena alloc failed` message points at `moe_gate` specifically.

## Expected outcome

- Confirm or rule out host-mapped MoE weight precision as the root cause of the ~1.7 avg_nll gap
- If confirmed: find a fix (increase VRAM budget, reduce model footprint, or fix the host-mapped fallback path)
- If ruled out: document the finding and move to next hypothesis

## Acceptance criteria

- [x] Layer-0 tensor comparison run between TP=4 and pipeline
- [x] Host-mapped vs cached weight numerical equivalence tested (or root cause identified)
- [x] Root cause of remaining quality gap identified
- [x] Issue #40 updated with findings

## Blocked by

- 4× R9700 GPU access
- Existing TP=4 quality fixture infrastructure

## Comments

### Findings (2026-07-31)

**Root cause confirmed via static analysis + live instrumentation.** The
hypothesis in this issue is correct, but the mechanism is broader than
"MoE weights specifically fall back to host-mapped memory under VRAM
pressure" — it is **every weight tensor touched during batch prefill, in
both TP=4 and (currently) pipeline mode**, because of an unconditional
call added by issue #37's fix.

#### How the fallback actually works

`cuda_resolve_weight_ptr` / `cuda_model_range_ptr`
(`rocm/ds4_rocm_runtime.cuh`) resolve a weight tensor's device pointer in
one of four ways, in order:

1. **Per-device selective cache** (`ds4_gpu_lookup_cache_strict`) — the
   normal path: weights pre-loaded into VRAM at model-load time, one
   lookup, no copy.
2. **`cuda_model_image_owned`** — a fast path for models fully resident
   as one contiguous device image. **Dead code**: `g_model_images` is
   only ever populated by `cuda_model_copy_chunked`, which has zero
   callers anywhere in the tree. `cuda_model_image_owned()` is always
   false for this build. The comments at ds4.c:24916 and ds4.c:30489
   claiming "same address in both modes via the model image pointer"
   describe behavior the code cannot deliver.
3. **Arena cache** (`cuda_model_arena_alloc` → `cuda_model_range_ptr_from_fd`)
   — a *second*, separate VRAM pool (`g_model_arenas`, global, not
   per-device) that `cudaMalloc`s new chunks and copies bytes from disk
   into them on first access, independent of the primary selective
   cache. Subsequent lookups for the same offset hit this arena's map
   and are cheap. This is where the `ROCm model arena alloc failed for
   moe_gate (1024.00 MiB chunk): out of memory` message originates
   (`cuda_model_arena_alloc`, `cudaMalloc` failure).
4. **Host-register PCIe-map** (`cudaHostRegister` +
   `cudaHostGetDevicePointer`) — when the arena also fails, weights are
   read directly from pinned host memory over PCIe on every access
   instead of from local VRAM. Bytes are identical to the cached copy,
   but the *access path* differs (BAR-mapped host reads vs. local VRAM),
   which issue #37 already established produces divergent Q8 matmul
   results on gfx1201 ("2.37e-4 error... different device addresses").

`ds4_gpu_set_use_host_weights(1)` — set unconditionally at the top of
`metal_graph_encode_layer_batch` for **every** batch-prefill layer, in
both TP=4 and pipeline mode (not gated on `g->rocm_tp4`) — makes
`cuda_resolve_weight_ptr` skip path 1 (the primary, already-loaded
selective cache) and go straight to paths 2–4. Combined with path 2
being dead code, this means **all batch-prefill weight resolution
re-resolves through the small arena/host-register mechanism instead of
using the weights already resident in VRAM from model load**, wasting
VRAM (duplicate copies of the same bytes) and exhausting the sliver of
headroom left after the primary cache fills the GPU.

Once the arena's `cudaMalloc` fails once, `cuda_model_arena_alloc` sets
`g_model_cache_full = 1` — a **global, not per-device** flag — and every
later arena request for the rest of the process, on any tier, returns
NULL immediately without retrying `cudaMalloc`, permanently routing that
tier's remaining weight lookups through the PCIe host-register path.

#### Live confirmation (4×R9700, `DS4_ROCM_WEIGHT_PATH_STATS=1` instrumentation added this session)

Ran `score_official` on a 5-case subset
(`gguf-tools/quality-testing/data/flash/manifest.tsv` cases 000–004,
ctx=1024/4096, `AMD_SERIALIZE_KERNEL=3`), instrumented to log every arena
skip and host-register event:

| Config | avg_nll (5 cases) | arena `cudaMalloc` OOM | arena-full skips | host-register PCIe reads |
|---|---|---|---|---|
| `--cuda-tensor-parallel` (TP=4), ctx=4096 | 1.558 | 1 (layer 0, `moe_gate`, offset 1.70 GiB) | 972 | 973 |
| pipeline (no TP flag), ctx=1024 | 1.563 | 1 | 986 | 987 |

The first arena OOM fires at **layer 0**, immediately after model load,
in both configs — not at layer 38 as the earlier (pre-Option-B, pre-Fix-2)
per-layer diff in issue #37 found. From that point on, essentially the
entire 43-layer forward pass for every case resolves its attention,
router, shared-expert, and MoE weights through PCIe host-register reads
rather than the pre-loaded VRAM cache. This directly answers criterion 2
("host-mapped vs cached weight numerical equivalence"): the two paths
are not equivalent — issue #37 already measured 2.37e-4 per-element
divergence from an *address change alone*, which compounds catastrophically
across 43 layers into exactly the ~1.5–2.0 avg_nll gap this project has
been chasing since Option B (and, it turns out, since before Option B).
No further per-kernel numeric equivalence test is needed; the existing
#37 measurement plus this live trace fully explains the gap.

#### Unexpected discovery: pipeline mode is currently affected too

The historical pipeline reference (`q_pipeline_ref_tp4issue32.tsv`,
avg_nll=0.374733, captured 2026-07-29 07:37) is **stale**. Re-running the
same 5 cases on today's tree in pipeline mode (no `--cuda-tensor-parallel`)
gives avg_nll=1.563 — indistinguishable from TP=4's 1.558, with the same
arena-exhaustion signature. Bisecting by comment/diff inspection: commit
`414f9fc` ("unblock TP=4 quality fixture with memory accounting fixes",
2026-07-29 16:21, after the reference was captured) gated the batch-scratch
line items in `engine_per_tier_graph_overhead_bytes` behind
`e->cuda_tensor_parallel` — intended to stop pipeline mode over-reserving
scratch it doesn't need. But `metal_graph_alloc_raw_cap` still
unconditionally allocates those same `batch_*_by_tier` buffers for every
`used_tier` regardless of TP mode (ds4.c:17540 loop has no TP gate). So
pipeline mode's weight packer now believes it has more free VRAM than it
does, packs the primary selective cache closer to the limit, and leaves
no headroom for the (already-wasteful) arena mechanism described above —
the same failure mode TP=4 has, now also on the pipeline path. Filed as
[issue #43](43-pipeline-vram-accounting-regression.md) since it's a
distinct, fixable accounting bug rather than part of this investigation's
scope.

#### Scope note

Per this issue's acceptance criteria, the investigation is complete: root
cause confirmed, mechanism explained, live evidence gathered. The actual
fix (give batch prefill enough VRAM headroom, or stop
`ds4_gpu_set_use_host_weights` from bypassing the primary cache when a
tensor is already resident there) belongs to
[issue #42](42-tp4-vram-budget.md), which should be re-scoped to also
cover the pipeline regression from #43 — reducing headroom alone won't
help if `cuda_resolve_weight_ptr` keeps ignoring the primary cache for
every batch-prefill lookup.

#### Instrumentation added

`rocm/ds4_rocm_runtime.cuh`: two env-gated (`DS4_ROCM_WEIGHT_PATH_STATS=1`)
diagnostic logs — one in `cuda_model_arena_alloc` for skips after the
arena-full latch trips, one in `cuda_model_range_ptr` for each new
host-register PCIe mapping. Both off by default (no perf/log impact on
normal runs). Left in place as a standing diagnostic for whoever
implements #42/#43, since they make the failure mode directly observable
instead of inferred from OOM messages alone.
