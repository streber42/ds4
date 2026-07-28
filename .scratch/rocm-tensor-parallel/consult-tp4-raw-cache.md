# Consultant Query: TP=4 decode loop — raw KV cache device assignment

## Context

Issue #32 (TP=4 quality fixture, authoritative correctness gate) is at `Status: ready-for-human`. All prerequisite issues (#25, #29, #30, #31) are closed. The following structural fixes are committed at HEAD (`d04c4cc`):

- Per-tier compressor state arrays (`layer_attn_state_kv_tp1/3`, `layer_attn_state_score_tp1/3`), indexer state arrays, and per-tier raw KV cache tensors (`layer_raw_cache_tp1/3`) are allocated on their respective GPU devices.
- After batch prefill, `metal_graph_rocm_tp4_sync_kv_cache` replicates tier 0's KV cache data to all 4 tiers via xdev copy.
- Embedding is broadcast from tier 0 to tiers 1-3.
- `metal_graph_set_active_tier_decode` correctly switches the HIP device and sets `active_tier`.
- Output head splits vocabulary 4 ways.
- Shared expert counted only once, attention all-reduce no longer aliases, compressor counters guarded.

## Remaining symptom

First token matches the pipeline reference ("We" for prompt "Hello"). Subsequent decode tokens are garbled — non-linguistic noise like "Wealth, .   " instead of "We need to respond to the".

This is the *exact* same symptom as the earlier decode loop data race (first token OK, pos>0 fails), which was previously fixed by the compressor state arrays. But the symptom persists.

## Known-working reference

Pipeline path (no `--cuda-tensor-parallel`) produces correct output on the same binary. CPU backend produces correct output. The pipeline reference TSV (`q_pipeline_ref_tp4issue32.tsv`, avg_nll 0.3747) is valid and reproducible.

## Hypothesis we want verified

**The decode loop passes device-0's `g->layer_raw_cache[il]` to all 4 tiers for both phases, instead of each tier's local per-tier raw cache.**

Inside `metal_graph_encode_token_raw_swa` (ds4.c:27050), the TP=4 decode loop:

```c
// Line 27078 — Phase 1 (TO_FFN / attention)
for (int tier = 0; ok && tier < 4; tier++) {
    metal_graph_set_active_tier_decode(g, tier);
    g->tp_rank = tier;
    ok = metal_graph_encode_decode_layer_phase(
            g, model, &weights->layer[il],
            il, pos,
            g->layer_raw_cache[il],   // ← device 0 for ALL tiers
            g->raw_cap, raw_row, n_raw, token,
            METAL_DECODE_LAYER_TO_FFN);
}

// Line 27162 — Phase 2 (FROM_ATTN_TO_FFN / MoE)
for (int tier = 0; ok && tier < 4; tier++) {
    metal_graph_set_active_tier_decode(g, tier);
    g->tp_rank = tier;
    ok = metal_graph_encode_decode_layer_phase(
            g, model, &weights->layer[il],
            il, pos,
            g->layer_raw_cache[il],   // ← device 0 for ALL tiers
            g->raw_cap, raw_row, n_raw, token,
            METAL_DECODE_LAYER_FROM_ATTN_TO_FFN);
}
```

A per-tier raw cache accessor **exists**:

```c
static inline ds4_gpu_tensor *metal_graph_tp4_raw_cache(
        const ds4_gpu_graph *g, uint32_t il, int tier) {
    if (!g->rocm_tp4) return g->layer_raw_cache[il];
    switch (tier) {
        case 0: return g->layer_raw_cache[il];
        case 1: return g->layer_raw_cache_tp1[il];
        case 2: return g->layer_raw_cache_tp[il];
        case 3: return g->layer_raw_cache_tp3[il];
        default: return NULL;
    }
}
```

But it is **never called** from the decode loop. The phase function receives `g->layer_raw_cache[il]` (device 0) as the `raw_cache` parameter and:
1. Writes this tier's K,V to device 0's memory via the KV store kernel (peer write over xGMI)
2. Reads K,V from device 0's memory during the attention kernel (peer read over xGMI)

At pos=0 (first token), the KV cache starts empty from prefill (already synced to all tiers as identical data), so reading device 0's memory is fine — all tiers see the same prefill KV data and produce correct partials.

At pos>0 (decode tokens), each tier has written its *own* 32-head K,V to device 0's memory at the previous pos, and then reads the *full* 128-head K,V from device 0's memory. The KV store and attention kernel on device t both go through peer access to device 0's memory. The question is whether this cross-device peer read of the GPU framebuffer is consistent:

- **If peer reads from device 0's framebuffer are coherent on the gfx1201 xGMI fabric:** then all 4 tiers reading the same `g->layer_raw_cache[il]` (device 0) would work correctly — they'd see every tier's K,V written at pos-1.
- **If peer reads are not coherent** (e.g., device 0's write from device t doesn't invalidate device 0's L2 cache for device t+1's read): then device t reads stale K,V data, producing garbage attention output for that tier.

### Why first token works but subsequent tokens don't

At pos=0, the prefill has populated tier 0's KV cache with *all* 128-head K,V for every prefill token. The KV sync (`metal_graph_rocm_tp4_sync_kv_cache`) has replicated this to tiers 1-3. So at pos=0 attention, every tier correctly reads the full prefill history. Attention partials from all 4 tiers are correct → all-reduce is correct → first decode token is correct.

At pos=1, the attention kernel on each tier has to read the full KV cache (128 heads × pos tokens). The pos=0 K,V was written by each tier's own KV store kernel. If a tier reads its peer's data from device 0's memory and that read is stale, the attention output for those heads is garbage.

## The fix we propose

Replace `g->layer_raw_cache[il]` with `metal_graph_tp4_raw_cache(g, il, tier)` in both raw_cache references in the TP=4 decode loop. This makes each tier read/write its own local per-tier raw cache — no cross-device peer access needed during decode KV operations.

The cost: ~125 MiB per GPU for `raw_cap * DS4_N_HEAD_DIM * sizeof(float) * DS4_N_LAYER = 2289 * 4096 * 4 * 43 ≈ 1.6 GiB` total across 4 GPUs, or ~400 MiB per GPU. This VRAM was already allocated in commit `d04c4cc` — we just need to use it.

After the prefill sync, all 4 per-tier caches contain identical data (the prefill KV). During decode, each tier writes only to its own cache. The next token's attention kernel reads only from its own local cache — but each tier's local cache only has its *own* 32-head K,V from the previous decode step, not the full 128 heads.

**Wait — this doesn't work either!** Each tier's attention kernel needs the *full* 128-head K,V to compute its 32-head partial. If tier 1's local cache only contains the K,V for heads 32-63 (written by tier 1 at pos-1), the attention kernel can't compute its attention over heads 0-31 and 64-127.

So a naïve per-tier raw cache assignment without a full-cache replication step between tokens would *also* produce incorrect attention. The options are:

### Option A: Local caches (each tier reads only its own)

Each tier reads/writes its own per-tier raw cache. Between decode tokens, replicate each tier's writes to all other tiers (all-gather of raw KV rows). Each tier ends up with the full 128-head KV in its local cache for the next token's attention. Cost: one xdev broadcast of one KV row (4096 floats = 16 KiB) per tier per layer per token, but the all-reduce barrier already syncs all devices — we could batch the KV row replication there.

### Option B: Shared device-0 cache with guaranteed peer coherence

Keep using `g->layer_raw_cache[il]` (device 0) for all tiers. Add explicit `ds4_rocm_xdev_sync_all_devices` + GPU memory fence after each tier's KV store so that device 0's memory is coherent before the next tier's attention kernel reads it. This doesn't require extra data copies — just synchronization.

ROCm provides `hipDeviceSynchronize()`, and since all tiers write to device 0's memory (coherent on xGMI), this may suffice. We already call `ds4_rocm_xdev_sync_all_devices` after the phase 1 attention compute — but the KV store happens *during* the phase function, before the sync. The question: does the xGMI fabric provide read-after-write coherence for peer GPU framebuffer accesses from kernel code?

### Option C: Host-serialized tiers (slow, diagnostic only)

Serialize the tier loop so tier 0 finishes its entire decode layer (attention + KV store + MoE) before tier 1 starts. This eliminates all concurrency issues but defeats the purpose of TP (4× latency). Only useful as a diagnostic to confirm the data race hypothesis.

## Questions for the board

1. **Is the cross-device peer read of GPU framebuffer memory coherent on gfx1201 (Radeon PRO R9700) xGMI?** Specifically, if device 1 writes to device 0's memory via a kernel store, is that write visible to device 2's kernel reading from the same device-0 address, without an explicit inter-device synchronization?

2. **Which approach should we take?**
   - **A** — Per-tier local caches with inter-token KV replication (safe, ~cost of one xdev broadcast per token)
   - **B** — Shared device-0 cache with explicit peer coherence (no extra data movement, relies on ROCm peer coherence guarantees)
   - **C** — Serialize tiers (diagnostic only — 4× latency)
   - **Something else** (e.g., use device 0 globally but add a `hipDeviceSynchronize` + `__threadfence_system` after KV store on each tier before proceeding to the next tier's attention)

3. **Is there a subtler root cause we're missing?** E.g., the phase-split architecture itself has a race we haven't noticed, or the prefill KV sync runs only in the batch prefill path (ds4_session_slice) but the quality fixture uses the token-by-token eval path (ds4_session_eval → metal_graph_eval_token_raw_swa).
