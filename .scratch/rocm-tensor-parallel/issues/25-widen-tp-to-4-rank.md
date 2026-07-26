# 25 — Widen TP from 2-pair pipeline to true 4-rank tensor parallelism

Status: ready-for-agent

**What to build:** The current 4-GPU topology is "Option A" from issue #11 — two TP=2
pairs arranged in a pipeline. GPUs 0-1 form one TP pair processing layers 0-20, GPUs 2-3
form the second pair processing layers 21-42. Each "pair" uses the proven 2-rank TP kernels
from upstream unchanged, with the upper-half GPUs primarily holding replicated expert
weights for half-resident decode.

This topology leaves half the hardware idle during compute:

```
GPU0: layers 0-20 + embedding   (22.1 / 27.8 GB)  ← active
GPU1: layers 21-42 + output head (22.1 / 27.8 GB)  ← active
GPU2: (no transformer layers)    (0.0 / 27.8 GB)   ← expert weight storage only
GPU3: (no transformer layers)    (0.0 / 27.8 GB)   ← expert weight storage only
```

The inter-pair pipeline serialization means only one pair computes at any instant during
decode, giving 12.3 t/s generation — slower than 4-GPU pipeline layer-split (22.8 t/s).
Optimizations that reduce per-GPU compute (issue #20 WMMA, issue #22 fusion) cannot help
because the bottleneck is structural: 2 of 4 GPUs are always waiting.

**Option B (this issue): widen to true TP=4.** All 4 GPUs compute on every token
simultaneously. The 81 GiB model at IQ2/Q2K quantization fits easily in 4× 34 GiB VRAM
(~20 GiB per GPU with even sharding). No pipeline serialization. Every GPU is active on
every token.

## Architecture change required

The current code uses a `half = n_gpus / 2` pairing model throughout `ds4.c`. Each
lower-half tier `t` is paired with upper-half tier `t + half`. This is the fundamental
abstraction that needs replacing with an N-way group model.

### 1. Attention sharding
- **Current (2-way):** 128 attention heads split 64/64 between the two ranks in a pair.
  MLA (multi-latent attention) uses a single compressed KV cache per layer, so KV is
  replicated, not sharded.
- **Target (4-way):** 128 heads split 32/32/32/32 across 4 ranks. Divides evenly.
  MLA compressed KV remains replicated. The attention output projection gathers partial
  results from all 4 ranks.
- **Scope:** The head-split dispatch in `rocm/ds4_rocm_runtime.cuh` and the attention
  output accumulation need generalizing from 2-way to N-way. Moderate effort.

### 2. MoE expert sharding
- **Current (2-way):** 256 routed experts split 128/128 between the two ranks in a pair.
  "Half-resident decode" uses the partner tier to hold the other 128 experts. The gate/up
  projections compute partial results that are summed via cross-device peer copy.
- **Target (4-way):** 256 experts split 64/64/64/64 across 4 ranks. Each rank computes its
  quarter. Partial results are all-reduced across all 4 ranks.
- **Scope:** The expert ownership policy (`cuda_tp_ep`), the gate/up/down dispatch, and the
  partial-result accumulation all need N-way generalization. The WMMA kernels from issues
  #20/#22 remain valid per-rank — only the orchestration around them changes. Moderate effort.

### 3. Exchange / reduction pattern
- **Current (2-way):** Point-to-point peer copy between paired tiers. One send, one receive,
  one sum. Simple and proven correct (issues #01, #18).
- **Target (4-way):** All-reduce across 4 ranks. This is the largest change. Options:
  - Ring all-reduce (4 sends/receives around a ring)
  - Tree all-reduce (log₂(4) = 2 steps)
  - Butterfly / hypercube (2 steps, all-to-all)
  - Brute-force: each rank broadcasts its partial to all others (3 sends, 3 receives, 3 sums)
- **Scope:** New code. The peer-copy infrastructure from issue #01 is reusable, but the
  collective pattern is new. Peer bandwidth is 24 GB/s per link and all pairs have direct
  access (verified in issue #01), so any topology works. High effort, highest risk.

### 4. Output head / vocabulary sharding
- **Current (2-way):** Vocabulary is row-sharded across paired tiers. The `partner = tier +
  half` pattern in `metal_graph_cuda_tp_output_tiers_for_head()` assigns output shards.
- **Target (4-way):** Vocabulary row-sharded across all 4 ranks. Each rank holds 1/4 of the
  output projection. Logit gathering needs all-gather from 4 sources instead of 2.
- **Scope:** Generalize the `output_tiers_for_head` function and logit assembly. Low effort.

### 5. Layer placement / tier assignment
- **Current:** Lower-half tiers get contiguous layer ranges (pipeline split). Upper-half
  tiers get no layers.
- **Target:** All 4 tiers hold the same layers (true tensor parallelism). Each rank holds
  1/4 of every layer's sharded tensors (attention QKV/O weights, MoE experts, output head).
  Non-sharded tensors (RMS norm, MLA compressor) are replicated.
- **Scope:** The layer placement logic in `ds4.c` (`n_stages = n_gpus / 2`) needs replacing
  with a single-stage, all-replicate-layers model. Moderate effort.

### 6. Hardcoded 2-rank assumptions
- `ds4_tp.c` — the upstream transport is point-to-point (leader/worker). The ROCm path does
  not use this transport (it uses direct peer copies within a single host), so this is not
  a blocker.
- `half = n_gpus / 2` — appears in ~15 locations in `ds4.c`. Each site needs auditing.
- `partner = tier + half` — the pairing function. Needs replacement with a group membership
  function.
- `n_stages = n_gpus / 2` — the pipeline stage count. For TP=4 this becomes 1 (no pipeline).

## Expected outcome

- All 4 GPUs active on every token during both prefill and decode
- No inter-stage pipeline serialization
- Estimated decode throughput: should approach or exceed pipeline layer-split baseline
  (~22.8 t/s) since all 4 GPUs compute in parallel with only small all-reduce overhead
  (a few MB per token at 24 GB/s per link ≈ <1ms)
- Per-GPU utilization: should rise from ~25% (one of four active at a time) toward ~75%+
- VRAM per GPU: ~20 GiB (81 GiB / 4 + overhead), well within 34 GiB budget

## Acceptance criteria

- [ ] Sharding policy generalized to N-way (complete partition of heads, experts, vocab across 4 ranks)
- [ ] All-reduce collective implemented over existing peer-copy infrastructure
- [ ] Attention path works with 4-way head split (prefill + decode)
- [ ] MoE path works with 4-way expert split (prefill + decode)
- [ ] Output head works with 4-way vocabulary shard
- [ ] Layer placement: all 4 GPUs hold same layers (no pipeline split)
- [ ] Correctness: quality fixture (`make rocm-quality`, serialized) matches reference within ±1% avg_nll
- [ ] Throughput: decode generation measured against pipeline baseline (~22.8 t/s) and current TP (~12.3 t/s)
- [ ] Per-GPU utilization measured via `rocm-smi` during decode
- [ ] Results recorded in experiment log

## Blocked by

- Issue #23: same-device compressor prefill race (quality fixture needs this fixed for
  default-mode scoring; serialized mode can be used for initial validation)

## Risks

1. **All-reduce correctness.** The collective pattern is new code. A subtle bug (wrong
   accumulation order, missing barrier, race between sends) would produce plausible-looking
   but wrong output. Mitigate with the kernel comparison scaffold and serialized quality
   fixture.
2. **Performance may not beat pipeline.** If the all-reduce overhead per token is larger
   than expected, TP=4 could still be slower than 4-GPU pipeline layer-split. The PRD's
   secondary risk applies: record the finding rather than bury it.
3. **Scope creep.** Generalizing from 2-way to N-way touches many code paths. The change
   should be scoped to exactly N=4 (not arbitrary N) to minimize surface area.

## Implementation approach

Suggested vertical slice order (each step produces a runnable, testable state):

1. **Sharding policy first** — generalize expert/head/vocab ownership to N=4 ranks.
   Pure CPU unit tests (existing scaffold from issue #03). No GPU code yet.
2. **All-reduce primitive** — implement over peer-copy infrastructure. Standalone test
   on device buffers (existing scaffold from issue #03b). Bandwidth floor measurement.
3. **Layer placement** — switch from pipeline split to all-replicate. Model loads on 4 GPUs
   with every rank holding every layer's sharded tensors.
4. **Attention path** — 4-way head split + all-reduce of attention output. First correctness
   signal: generate a coherent sentence.
5. **MoE path** — 4-way expert split + all-reduce of down-projection output. Second
   correctness signal.
6. **Output head** — 4-way vocab shard + all-gather of logits. Third correctness signal.
7. **Quality fixture** — full 100-case run. Authoritative correctness gate.
8. **Throughput measurement** — `ds4-bench` sweep against baselines.
