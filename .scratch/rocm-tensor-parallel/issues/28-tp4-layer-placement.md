# 28 — TP=4 layer placement (model loads on 4 GPUs)

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/issues/25-widen-tp-to-4-rank.md`

## What to build

Replace the current pipeline-split layer placement with a single-stage all-replicated placement for TP=4. All 4 GPUs load all 43 transformer layers, with sharded tensors split by `tp_rank` and replicated tensors loaded in full on every rank.

Currently, `n_stages = n_gpus / 2 = 2` creates two pipeline stages: lower-half tiers (0,1) hold contiguous layer ranges, upper-half tiers (2,3) hold no layers. For TP=4, `n_stages = 1` — every tier holds every layer.

**Sharded tensors** (loaded as the rank's shard only):
- Attention QKV projection weights: split by head count (32 of 128 heads per rank)
- Attention output projection weights: split by head count
- MoE routed expert weights (gate/up/down): split by expert count (64 of 256 per rank)
- Output head / vocabulary projection: split by row (V/4 per rank)
- Dense FFN columns (if applicable): split by column

**Replicated tensors** (loaded in full on every rank):
- RMS normalization weights
- MLA compressor weights (compressed KV is a single latent vector, not per-head)
- Shared expert weights (all ranks need it; rank 0 computes it during MoE)
- Embedding table

The `~15` sites in `ds4.c` that use `half = n_gpus / 2` and `partner = tier + half` need auditing. Each is either removed (no partner concept in TP=4) or replaced with `tp_world`-based N-way logic.

The TP=2 path must remain functional — this adds a parallel `tp_world == 4` code path.

## Acceptance criteria

- [ ] All `half = n_gpus / 2` sites in `ds4.c` audited and updated for TP=4 path
- [ ] Production 81 GiB model loads on 4 GPUs without crash (~23 GiB per GPU)
- [ ] VRAM usage confirmed via `rocm-smi` after model load
- [ ] `./ds4 -p "Hello" -n 1` reaches first kernel dispatch (output will be garbage — attention/MoE exchange not yet implemented)
- [ ] TP=2 path still works: `--gpu-devices 0,1 --cuda-tensor-parallel` with 2 GPUs produces correct output
- [ ] `make -j8 rocm` builds cleanly
- [ ] Existing `test-rocm` test suite still passes

## Blocked by

- Issue #26: TP=4 sharding policy (shard config function needed for tensor loading)
