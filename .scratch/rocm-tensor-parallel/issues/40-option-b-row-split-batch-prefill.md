# 40 — Option B: row-split batch prefill refactor (close quality gate)

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/issues/32-tp4-quality-fixture.md`

## What to build

Restructure the TP=4 batch prefill path (`metal_graph_encode_layer_batch`)
so that each tier processes `n_tokens / 4` rows with full (replicated)
attention and FFN weights, then exchanges rows via all-gather.  This
eliminates the computational graph divergence between TP=4 batch prefill
and single-GPU pipeline prefill that produces avg_nll ~2.0 (target
0.370–0.378, first_match 0/100 vs target ≥60/100).

## Why the simpler fixes didn't work

A chain of four fixes was applied (sessions 2026-07-29) to unblock the
quality fixture and align the computational graphs:

| Fix | What | Effect |
|---|---|---|
| #39 (commit a86750e) | Enable f16 cuBLAS attention output in batch prefill | avg_nll ~1.72 → ~1.50 (marginal) |
| Fix 1 (uncommitted) | Disable f16 cache reserve clearing for TP=4 | No change (both paths already use Q8) |
| Fix 2 (uncommitted) | Gate batch scratch behind `cuda_tensor_parallel` | Unblocked pipeline OOM |
| Fix 3a/b (uncommitted) | Correct output weight accounting for TP=4 placement | Unblocked TP=4 fixture OOM |

After all four fixes the fixture runs cleanly (all 4 tiers active, 27.79 GiB
budget each, peer access validated) but quality scores are unchanged:

| Metric | Pipeline ref | TP=4 (2026-07-29) | Target |
|---|---|---|---|
| avg_nll | 0.3747 | 1.85 – 2.05 | 0.370–0.378 |
| first_match | 65/100 | 0/100 | ≥60/100 |
| api_top1_rate | 0.859 | 0.500 | ≥0.85 |
| api_pair_rate | 0.988 | 0.908 | ≥0.98 |

The consultant panel (7/7 respondents) and Gemini unanimously ordered
Option B.  The ~1.5–2.0 NLL gap is in the TP=4 batch prefill
**computational graph** — all-reduce of partial attention/FFN outputs
across 4 tiers produces different floating-point results than the
single-GPU pipeline path.  No further per-kernel precision fix will close
it; the arithmetic structure must change.

## Architecture

Current TP=4 batch prefill (per layer):

```
for tier in 0..3:
    set_active_tier(tier)
    encode_batch_layer_attention(tier)   ← partials (sharded heads or rows)
    encode_batch_layer_ffn(tier)         ← partials (sharded experts)
allreduce_attention()
allreduce_ffn()
```

Option B (target):

```
for tier in 0..3:
    set_active_tier(tier)
    # Each tier processes tokens[tier*chunk : (tier+1)*chunk]
    # with FULL replicated weights (no head/expert sharding)
    encode_batch_layer_attention_full(tier, my_rows)
    encode_batch_layer_ffn_full(tier, my_rows)
allgather_rows()   # exchange completed rows so all tiers have full hidden state
```

Key differences:
- **No weight sharding.** Each tier holds full attention + FFN weights
  (already replicated in cache install per TP=4 placement).  Head sharding
  and expert sharding are disabled during batch prefill.
- **Row partitioning.** `n_tokens` is divided 4 ways.  Each tier computes
  its rows' attention and FFN independently (embarrassingly parallel for
  the matmuls; KV cache reads may need adjustment).
- **All-gather instead of all-reduce.** After each tier has its rows'
  hidden state, exchange rows so all tiers hold the full hidden state for
  the next layer.  This is a deterministic operation — no floating-point
  accumulation across partial sums.
- **Decode path unchanged.** Decode is token-by-token (batch=1) and
  already produces correct quality when given correct hidden states.

## Scope

Estimated ~800–1200 lines of changes in `ds4.c`:

1. **Batch prefill attention path** (~400 lines): Add a `tp_row_split`
   branch in `metal_graph_encode_layer_batch` that computes attention on
   a row slice.  May reuse the existing `tp_row_split_attn` gate (line
   29570, currently TP=2-only).
2. **Batch prefill FFN path** (~300 lines): Same row-split treatment for
   shared experts and routed experts.  The MoE routing must run on the
   full hidden state (after all-gather) to get globally consistent
   top-k routing.
3. **All-gather primitive** (~150 lines): New `ds4_rocm_tp4_allgather_rows`
   that exchanges row slices across all 4 tiers.  Similar to the existing
   `ds4_rocm_xdev_allreduce_f32` but a copy (not accumulate) operation.
4. **Planner / shard divisor** (~100 lines): Ensure TP=4 batch prefill
   uses `shard_divisor=1` for attention and FFN weights (already correct
   per issue #37), and that scratch buffers are sized for `n_tokens/4`
   rows per tier.
5. **Tests / validation** (~100 lines): Layer-by-layer output comparison
   between pipeline and TP=4 at layers 0, 10, 20, 30, 40 to confirm
   bit-identical (or within ±1% NLL) hidden states.

## Acceptance criteria

- [ ] TP=4 batch prefill uses row-split architecture (each tier processes
      n_tokens/4 rows with full weights)
- [ ] All-gather primitive implemented and validated on 4× R9700
- [ ] Quality fixture scores meet tolerance:
      - avg_nll within ±1% of pipeline (0.370–0.378)
      - first_match ≥ 60/100
      - api_top1_rate ≥ 0.85
      - api_pair_rate ≥ 0.98
- [ ] Decode path unchanged and still correct
- [ ] Pipeline path (non-TP) unaffected
- [ ] TP=4 coherence test produces coherent output
- [ ] Issue #32 closed after scores verified

## Blocked by

- Memory accounting fixes from session 2026-07-29 (Fix 3a/b, uncommitted
  on `gfx1201_tp`) — must be committed first so the fixture can run.

## Key references in ds4.c

- `metal_graph_encode_layer_batch` — batch prefill entry point
- Line 29570: `tp_row_split_attn` gate (currently TP=2-only, needs
  extension to TP=4)
- Line 17580: `batch_q_half` allocation (f16 attention output buffer)
- Line 30333: `ds4_gpu_release_q8_f16_cache` per-layer eviction
- `ds4_rocm_xdev_allreduce_f32` — existing all-reduce primitive (model
  for all-gather)
- `engine_tp4_shard_divisor` — currently returns 4 for attention/FFN
  tensors during batch prefill; must return 1 for row-split mode
- `engine_compute_tp4_placement` — placement selection (unchanged)

## Context for ralph loop

- Branch: `gfx1201_tp` (uncommitted fixes from session 2026-07-29)
- Build: `make rocm-quality`
- Run: `AMD_SERIALIZE_KERNEL=3 ./gguf-tools/quality-testing/score_official /home/murphy/src/ds4/ds4flash.gguf gguf-tools/quality-testing/data/flash/manifest.tsv .scratch/rocm-tensor-parallel/quality-out/q_tp4_option_b.tsv 4096 --gpu-devices 0,1,2,3 --cuda-tensor-parallel`
- Pipeline reference: `.scratch/rocm-tensor-parallel/quality-out/q_pipeline_ref_tp4issue32.tsv` (avg_nll=0.374733, first_match=65/100)
- Previous TP=4 results: `.scratch/rocm-tensor-parallel/quality-out/q_tp4_current.tsv` (avg_nll=1.85)
- Fix 3a/b must be committed before starting Option B work
- GPUs must be free (no vLLM running): `sudo kill $(rocm-smi --showpids | awk '/VLLM/{print $1}')`
