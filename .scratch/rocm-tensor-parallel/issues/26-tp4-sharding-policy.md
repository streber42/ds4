# 26 — TP=4 sharding policy

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/issues/25-widen-tp-to-4-rank.md`

## What to build

A pure CPU function that computes per-rank shard boundaries for a given TP world size and rank. This is the foundational policy that every subsequent TP=4 slice queries to know which slice of a tensor each rank owns.

Given `tp_world` (1 or 4) and `tp_rank` (0..3), plus the model's total head count, expert count, vocabulary size, and embedding dimension, return a struct describing this rank's shard of each dimension:

- Attention heads: 128 total → 32 per rank (contiguous range)
- MoE routed experts: 256 total → 64 per rank (contiguous range)
- Output vocabulary: V total → V/4 per rank (contiguous range)
- FFN/embedding columns: n_embd total → n_embd/4 per rank (contiguous range)

When `tp_world == 1`, a single rank owns everything. When a dimension does not divide evenly by `tp_world`, the function should fail loudly (not silently mis-shard).

This is a pure function with no GPU code and no model dependency. It can be tested with a simple C unit test.

## Acceptance criteria

- [ ] `ds4_tp_compute_shard_config()` implemented (or equivalent function in the sharding module)
- [ ] Complete partition: union of all rank shards covers every head/expert/vocab row exactly once
- [ ] No gaps, no overlaps: verified programmatically for tp_world=4
- [ ] Even division verified: 128/4=32 heads, 256/4=64 experts per rank
- [ ] Degenerate case: tp_world=1 → single rank owns everything
- [ ] Boundary case: tp_world does not divide evenly → returns error (not silent mis-shard)
- [ ] Unit test added and passing (existing test scaffold or new standalone test)

## Blocked by

None — can start immediately.
