# 31 — TP=4 output head (correct sampling)

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/issues/25-widen-tp-to-4-rank.md`

## What to build

Wire up the output head (vocabulary projection + token sampling) for TP=4: shard the vocabulary rows V/4 per rank, and implement distributed decode sampling.

**Vocabulary sharding:** The output projection maps `n_embd → n_vocab`. With TP=4, each rank holds V/4 rows of this projection. Each rank computes a partial logit vector of size V/4 from the final hidden state.

**Decode sampling (distributed):** During greedy/top-k decode, each rank finds the local maximum in its V/4 shard, producing a `(rank, local_token_id, logit)` tuple. A small all-gather of 4 tuples (48 bytes total) determines the global winner. This avoids an all-gather of the full vocabulary (V × 4 bytes ≈ 600 KB) on every decode step.

**Prefill / quality fixture sampling (full logits):** The quality fixture and prefill scoring need the full logit distribution. For these paths, use a full all-gather: each rank's V/4 partial logits are gathered to all ranks, producing the complete V-element logit vector.

**Current code:** `metal_graph_cuda_tp_output_tiers_for_head()` at `ds4.c:71` uses `partner = tier + half` to assign output shards. For TP=4, this needs to return all 4 tiers as shard holders. The logit assembly code needs to gather from 4 sources instead of 2.

## Acceptance criteria

- [ ] Vocabulary row-sharded V/4 per rank (uses sharding policy from #26)
- [ ] Distributed decode sampling: each rank finds local max, small all-gather picks global winner
- [ ] Full logit all-gather available for prefill / quality fixture scoring
- [ ] Greedy decode produces correct tokens end-to-end (multi-turn conversation works)
- [ ] TP=2 output head path unchanged
- [ ] `make -j8 rocm` builds cleanly

## Blocked by

- Issue #30: TP=4 MoE path (MoE must work before output head can be tested end-to-end)
