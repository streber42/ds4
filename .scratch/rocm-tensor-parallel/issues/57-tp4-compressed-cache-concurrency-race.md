# 57 — Fix the compressed-KV-cache race blocking threaded rollout past layers 0-1

Status: closed

## Parent

`.scratch/rocm-tensor-parallel/issues/51-tp4-execution-engine-full-rollout.md`

## What to build

The persistent-per-rank-thread execution engine (#50/#51) is currently
restricted to layers where `ds4_layer_compress_ratio(il) == 0` — for
`DS4_VARIANT_FLASH` that is layers 0-1 only, out of 43. This is not an
arbitrary scope limit: it exists because the compressed-KV-cache path has
a real, unaddressed data race under concurrent per-rank dispatch.

At `ds4.c:22597`, `comp_row = g->layer_n_comp[il]` is read
**unconditionally by every rank**, inside
`metal_graph_encode_decode_layer_phase`. The shared counter is only
incremented by rank 0, later in the same function call
(`ds4.c:22653`: `if (ok && emit && (!g->rocm_tp4 || ds4_g_active_tier(g)
== 0)) g->layer_n_comp[il]++;`). Under the pre-#50 sequential per-tier
loop this was implicitly safe — rank 0's entire call, increment included,
always completed before rank 1-3's calls began. Under #50/#51's
concurrent per-rank-thread dispatch model, a non-zero rank's read can race
ahead of or behind rank 0's write within the same dispatched phase,
computing a mismatched row index into the shared, single-copy
`layer_attn_comp_cache[il]` — each rank would then write its tensor slice
into the wrong physical row, or two ranks could compute different rows
for what should be the same logical compressed-cache append.

This is the same "silent numerical corruption" failure class the #44-#48
all-reduce bugs and #52's HITL algorithm sign-off already exist to guard
against in this project — it needs the same level of care, not a quick
patch.

**A plausible direction (not validated — do the design work here, don't
just implement this blindly):** have the orchestrator (not each rank's
worker thread) read and, if needed, increment `layer_n_comp[il]` *once*
per dispatched layer, before handing the job to
`metal_graph_tp4_spike_dispatch`, and pass the resolved `comp_row` value
into each rank's job struct (`ds4_tp4_spike_job`) instead of having
`metal_graph_encode_decode_layer_phase` read the live global. Check
whether `emit` (which gates whether a row is actually appended this call)
can differ by rank — if it can't, this is a straightforward hoist; if it
can, the design needs to handle that.

Also audit whether `layer_n_index_comp[il]` (used at `ratio == 4`,
indexer-compressor path) has the same shape of race — the two counters
are read/written in parallel patterns per the surrounding code.

## Acceptance criteria

- [x] Concurrency-safe design for `layer_n_comp[il]`/`layer_n_index_comp[il]`
      resolved and reviewed (HITL sign-off recommended, matching #52's
      precedent, given this is exactly the silent-corruption risk class
      that motivated that gate)
- [x] `metal_graph_tp4_spike_layer_enabled`'s `ds4_layer_compress_ratio(il)
      == 0` gate relaxed to cover the newly-safe layers, without
      regressing the layers that were already safe
- [x] Full 100-case `score_official` quality fixture re-run (pipeline and
      TP=4) — this touches shared decode-loop state directly, so
      correctness must be reconfirmed, not assumed
- [x] `make -j8 test-rocm` passes
- [x] Findings recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`

