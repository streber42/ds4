# 39 — TP=4 row-split batch prefill (Option B)

Status: planning

## Parent

`.scratch/rocm-tensor-parallel/issues/32-tp4-quality-fixture.md`

## What to build

Extend the existing TP=2 row-split attention mechanism to TP=4 so that
each of the 4 tiers processes n_tokens/4 rows with full weights (accessed
via host-mapped fallback) rather than sharded weights.  The f16 cuBLAS
attention output path requires enough VRAM to cache the dequantized
weight matrix (~320 MiB per layer for attn_output_a + attn_output_b).
On 32 GiB R9700 GPUs, device 0 has only ~100 MiB free in TP=4 mode
because it carries all layers' KV caches and temp buffers.  Spreading
the f16 cache across 4 tiers (each with ~5–6 GiB free) makes the f16
path viable.

## Motivation

Current TP=4: All 43 batch-prefill layers run on device 0 (home tier).
The f16 cuBLAS attention output path is blocked by the 4 GiB VRAM
reserve in `cuda_q8_f16_cache_has_budget` (device 0 has only ~100 MiB
free during evaluation).  The pure Q8 fallback produces avg_nll=1.72
vs the pipeline reference 0.374 (within ±1% tolerance).

Attempted alternative (per-layer cache eviction + budget bypass) failed
because:
- The 4 GiB reserve exists for good reason (prevents OOM on MoE gate
  allocations, cuBLAS workspaces, etc.)
- The OOM at model loading showed that bypassing the reserve is unsafe
- The f16 cache (320 MiB) simply doesn't fit in ~100 MiB free

## Approach

The existing TP=2 row-split code (`tp_row_split_attn` at ds4.c:27882)
already handles splitting tokens across 2 ranks with sharded weights.
Extending to TP=4 requires:

### 1. Gate expansion (ds4.c)

Change `g->tp_world == 2` to `g->tp_world >= 2` so TP=4 also enters
the row-split path.

### 2. Host-mapped full-weight attention output

In TP=4 row-split, each tier needs the FULL weight matrix (not its 1/4
shard) for the attention output projection.  The weights are accessed
via `model->map` (host-mapped pointer), and dequantized to f16 via
`cuda_q8_f16_ptr`.  The per-tier f16 cache only needs one layer's
worth of entries (~320 MiB), which fits in each tier's ~5–6 GiB free.

Files to modify:
- `ds4.c`: `metal_graph_encode_layer_attention_batch` — extend the
  standalone f16 path (`ds4_gpu_attention_output_q8_batch_f16_tensor`)
  to work with row-split tokens.  The standalone path currently fails
  on ROCm because `batch_q_half` is NULL (`DS4_GPU_ATTN_COMP_CACHE_F16`
  is 0 on non-Apple).  For row-split, use `tp_heads` and `tp_attn_out`
  tensors instead.
- `rocm/ds4_rocm_attention_launch.cuh`: The standalone f16 path
  `ds4_gpu_attention_output_q8_batch_f16_tensor` needs to accept
  arbitrary output tensors (not just `g->batch_q_half`).  Add a variant
  or parameterize the output tensor.

### 3. Multi-rank row exchange

Current TP=2 row-split uses `ds4_gpu_tp_big_gate_kick` (peer-to-peer
exchange between 2 ranks).  For TP=4, replace with all-gather: each
tier writes its n_tokens/4 output rows into a shared buffer on
device 0 via copy-engine, then device 0 has the full output.

Alternatively, use a ring exchange (tier 0→1, 1→2, 2→3, 3→0) to
distribute all rows to all tiers.

Files to modify:
- `ds4.c`: `metal_graph_encode_layer_attention_batch` — replace the
  2-rank gate exchange with 4-rank all-gather
- `rocm/ds4_rocm_xdev.cu` or `ds4_tp.c`: Add all-gather primitive or
  extend existing primitives

### 4. MoE row-split (stretch goal)

The MoE tier-sweep + all-reduce pattern also contributes to the quality
divergence.  Row-splitting the MoE computation would make each tier's
routed-expert output match the pipeline path for its token subset.
This is a separate concern from the attention output and can be
deferred if attention alone closes the quality gap.

## Implementation plan

### Phase 1: Standalone f16 path for arbitrary output tensors

Modify `ds4_gpu_attention_output_q8_batch_f16_tensor` to accept an
explicit output tensor parameter instead of hardcoding `g->batch_q_half`.
On ROCm, `batch_q_half` is NULL; the row-split path would pass
`tp_attn_out` (which is a f32 tensor).

Alternatively, parameterize the output element type (f16 for Apple,
f32 for ROCm) so the same function works for both backends.

### Phase 2: TP=4 row-split attention gate

Extend `tp_row_split_attn` to TP=4.  Key changes:
- `tp_half_rows` → `tp_tier_rows` (n_tokens/4 per tier)
- `tp_row0` per tier: `tier * tp_tier_rows`
- `tp_rows` per tier: `tp_tier_rows` (last tier gets remainder)
- `tp_heads` and `tp_attn_out` per tier: row-range views of
  `batch_heads` and `batch_attn_out`

### Phase 3: All-gather exchange

After each tier computes its n_tokens/4 rows of attention output,
gather the full output onto device 0.  Use copy-engine (P2P) writes
to device 0's buffer, then synchronize.

### Phase 4: Verification

- Run single-prompt coherence test: should produce fluent output
- Run per-layer prefill dump: compare pipeline vs TP=4 row-split
  for tensor-level bit-identity
- Run quality fixture: verify avg_nll within 0.370–0.378

## References

- TP=2 row-split implementation: ds4.c lines 29483–29582
- Row-split gate: ds4.c:27882 (`g->tp_world == 2`)
- Standalone f16 path: `ds4_gpu_attention_output_q8_batch_f16_tensor`
  in `rocm/ds4_rocm_attention_launch.cuh:1018`
- Internal f16 path in `ds4_gpu_attention_output_q8_batch_tensor`:
  `rocm/ds4_rocm_attention_launch.cuh:1213`
- Host-mapped weights fix (issue #37): commit `2eab0be`
- Device context fix (issue #32): commit `d879bf4`
