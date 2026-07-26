# Spec: True 4-Rank Tensor Parallelism for ROCm

Status: draft
Issue: `.scratch/rocm-tensor-parallel/issues/25-widen-tp-to-4-rank.md`

## 1. Goal

Replace the current 2-pair pipelined topology with true TP=4: all four GPUs compute
on every token simultaneously, with weight tensors sharded 4 ways and partial results
combined via an all-reduce collective over the existing peer-copy mesh.

Non-goal: arbitrary N-way TP. This spec targets exactly N=4 to minimize surface area.

## 2. Current Architecture (what changes)

### 2.1 The pairing model

The current code in `ds4.c` uses a pervasive `half = n_gpus / 2` abstraction:

```
For 4 GPUs:  half = 2
  Lower half:  tiers 0, 1  → hold transformer layers (pipeline split)
  Upper half:  tiers 2, 3  → hold replicated expert weights (no layers)
  Partner of tier t:  t + half  (0↔2, 1↔3)
```

Every TP sharding decision, exchange, and placement function keys off `half` and
`partner`. This is the fundamental abstraction that gets replaced.

**What replaces it:** a single TP group of size 4, where every tier holds every layer
and every sharded dimension is divided by 4 instead of 2.

### 2.2 The TP gate mechanism

The upstream TP gate (`ds4_gpu_tp_gate_encode`) is a 2-rank synchronization barrier:

1. Each rank writes its partial into `tp_out[slot]` (a view into a transport slab)
2. The gate signals the peer, which copies `tp_out` → `tp_in` (over RDMA, TCP, or peer copy)
3. After the gate, both ranks have both partials
4. They are summed: `rank0_partial + rank1_partial` in canonical order

The ROCm build does **not** use the upstream `ds4_tp.c` transport (that is Metal-only,
guarded by `#ifdef __APPLE__`). Instead, the ROCm path uses the xdev peer-copy primitives
directly within the graph encoding.

**What replaces it:** an N-way all-reduce where each rank gathers partials from 3 peers
and accumulates them into its own buffer. No slab, no single-gate barrier — instead, a
sequence of peer-copy + accumulate operations.

### 2.3 Current sharding dimensions

| Subsystem | Dimension | 2-way split | 4-way split | Divisible? |
|---|---|---|---|---|
| Attention heads (Q/O) | 128 heads | 64/rank | 32/rank | Yes |
| MLA compressed KV | 1 latent head | replicated | replicated | N/A |
| MoE routed experts | 256 experts | 128/rank | 64/rank | Yes |
| MoE top-k selected | 6 per token | all 6 routed to owning rank | all 6 routed to owning rank | N/A |
| Shared expert | 1 per layer | replicated | replicated | N/A |
| Output vocabulary | V rows | V/2 per pair | V/4 per rank | V ≈ 152064, ÷4 = 38016 |
| Attention QKV weights | head_dim × n_heads | half the heads | quarter the heads | Yes |
| FFN gate/up (dense) | n_embd | half (column-split) | quarter (column-split) | Yes, 7168 ÷ 4 = 1792 |

### 2.4 The xdev transport primitives

`ds4_rocm_xdev.h` provides three primitives:

- `ds4_rocm_xdev_copy(mesh, dst_dev, dst_ptr, src_dev, src_ptr, bytes, stream)` — peer copy
- `ds4_rocm_xdev_accumulate_f32(mesh, dst_dev, dst_ptr, src_dev, src_ptr, count, stream)` — `dst += src`
- `ds4_rocm_xdev_accumulate_f16(...)` — same for f16

These use direct `hipMemcpyPeer` when available, host-staging fallback otherwise.
Stream ordering is via per-device producer events (`hipStreamWaitEvent`).

All 12 ordered device pairs on this hardware have direct peer access (verified in
issue #01, 24 GB/s per link). So host staging is never used in practice.

## 3. Design

### 3.1 TP group model

Replace the lower/upper half pairing with a flat TP group:

```c
// Current:
//   half = n_gpus / 2
//   partner(t) = t + half

// New:
//   tp_world = 4
//   tp_rank = tier_id  (each tier IS a rank)
//   All tiers hold all layers
//   No "partner" concept — all ranks are peers
```

The `tp_world` and `tp_rank` fields already exist in `ds4_gpu_graph` (line 15179-15180).
Currently `tp_world` is always 2 for TP and 1 for non-TP. Change to 4 for TP=4.

### 3.2 Layer placement

**Current:** `n_stages = n_gpus / 2 = 2`. Layers are split into contiguous ranges
assigned to lower-half tiers. Upper-half tiers hold no layers.

**New:** `n_stages = 1`. All 4 tiers hold all layers. Sharded tensors (attention weights,
MoE experts, output head) are split 4 ways. Non-sharded tensors (RMS norms, MLA compressor,
shared expert) are replicated.

```
GPU0: all layers, 1/4 of sharded tensors (~20 GiB)
GPU1: all layers, 1/4 of sharded tensors (~20 GiB)
GPU2: all layers, 1/4 of sharded tensors (~20 GiB)
GPU3: all layers, 1/4 of sharded tensors (~20 GiB)
```

VRAM: 81 GiB / 4 ≈ 20.25 GiB + replicated overhead (~2-3 GiB) ≈ 23 GiB per GPU.
Fits comfortably in 34 GiB.

**Placement code changes in `ds4.c`:**
- `n_stages = pcfg->n_gpus / 2` → `n_stages = 1`
- `tp_half = cuda_tp_decode ? n_gpus / 2 : 0` → `tp_world = cuda_tp_decode ? n_gpus : 1`
- Remove all `partner = tier + half` assignments
- Each tier loads its shard of every sharded tensor (by `tp_rank`-indexed offset)

### 3.3 Attention sharding (4-way head split)

**Current (2-way):**
- 128 attention heads split 64/64 between two ranks
- Each rank computes QKV projection for its 64 heads
- MLA attention operates on each head independently
- Attention output projection: each rank projects its 64 heads → partial `n_embd` vector
- The two partials are exchanged and summed → full `n_embd` output

**New (4-way):**
- 128 attention heads split 32/32/32/32 across 4 ranks
- Each rank computes QKV projection for its 32 heads
- MLA attention operates on each head independently (no change)
- Attention output projection: each rank projects its 32 heads → partial `n_embd` vector
- The four partials are all-reduced → full `n_embd` output

**Code path:** `metal_graph_attention_output_dense_quant_tp()` at line 22710 currently
takes `g->tp_rank * tp_groups, tp_groups` where `tp_groups = n_groups / 2`. Change to
`g->tp_rank * tp_groups, tp_groups` where `tp_groups = n_groups / tp_world`.

**MLA compressed KV:** remains replicated on all ranks (it is a single latent vector per
token, not per-head). No change needed.

### 3.4 MoE expert sharding (4-way)

**Current (2-way):**
- 256 routed experts split 128/128 between two ranks
- Router selects top-6 experts per token
- Each rank only processes the experts it owns (0-3 of the 6 selected)
- Each rank computes: shared_expert_output + sum(owned_routed_experts)
- The two rank partials are exchanged and summed → full FFN output

**New (4-way):**
- 256 routed experts split 64/64/64/64 across 4 ranks
- Router selects top-6 experts per token
- Each rank only processes the experts it owns (0-2 of the 6 selected, on average 1.5)
- Each rank computes: shared_expert_output / 4 + sum(owned_routed_experts)
  (shared expert output is divided by 4 since all ranks will sum it)
- The four rank partials are all-reduced → full FFN output

**Shared expert handling:** The shared expert is replicated on all ranks. Under 2-way TP,
each rank adds the full shared expert to its partial before exchange, so the sum includes
it exactly once (rank0's shared + rank0's routed) + (rank1's shared + rank1's routed) =
2×shared + routed, which is wrong — looking at the code at line 24058, the shared expert
output is added to `tp_out` on each rank, and the exchange sums them. So the shared expert
is double-counted? No — looking more carefully, the `tp_out` slot contains
`shared_out + routed_out`, and after exchange, rank0's `tp_out` and rank1's `tp_in` are
summed. This gives shared + routed0 + shared + routed1 = 2*shared + all_routed. That can't
be right.

Actually, re-reading: the shared expert is only computed by one rank in the TP path.
Looking at the MoE dispatch, the shared expert output goes into `shared_out`, the routed
output goes into `routed_out`, and then `tp_out = shared_out + routed_out`. The exchange
sums the two rank partials. So if both ranks compute the shared expert, it would be
double-counted. The fix is that only one rank computes the shared expert, or the shared
expert output is divided by tp_world.

For TP=4, the shared expert output should be computed by exactly one rank (rank 0) or
divided by 4 on all ranks before the all-reduce. The simplest approach: **rank 0 computes
the shared expert; other ranks contribute zero for the shared part.**

**Expert ownership policy:** The function that maps expert_id → owning rank needs to
change from `expert_id < 128 ? rank0 : rank1` to `expert_id / 64`. This is a pure CPU
function, testable without GPU.

### 3.5 All-reduce collective

**Current exchange (2-rank):**
```
rank0: tp_out = my_partial
       gate_encode()          // signals peer, copies peer's tp_out to my tp_in
       result = tp_out + tp_in  // rank0's + rank1's partials
```

**New all-reduce (4-rank):**

The all-reduce combines N partial vectors into one summed vector, available on all ranks.
For N=4 and small vectors (n_embd = 7168 floats = 28 KB), the simplest correct approach
is **brute-force all-gather + local sum:**

```
For each rank r in [0, 3]:
  1. Write my partial into xdev_buf[my_rank]
  2. For each peer p != my_rank:
       xdev_copy(peer p's partial → my xdev_buf[p])   // 3 peer copies
  3. Sum all 4 buffers: result = buf[0] + buf[1] + buf[2] + buf[3]
```

This is 3 peer copies per rank (96 KB total per rank), plus 3 accumulate kernels.

**Latency estimate:** 28 KB at 24 GB/s = ~1.2 μs per copy. Three serial copies = ~3.5 μs.
Three accumulate kernels at ~1 μs each = ~3 μs. Total: ~6.5 μs per all-reduce.
Two all-reduces per layer (attention + FFN) × 43 layers = ~560 μs per token.
Against a ~35 ms per-token compute budget, this is ~1.6% overhead. Negligible.

**Optimization (butterfly, optional):** A 4-rank butterfly all-reduce takes 2 steps instead
of 3 serial copies, but at 28 KB the copy latency is already sub-microsecond. The added
code complexity is not warranted. Stick with brute-force for correctness and simplicity.

**Synchronization:** Each rank's partial must be ready before peers read it. Use the
existing producer event mechanism (`ds4_rocm_xdev_wait_producer`) — each rank records
an event after writing its partial, then each peer waits on that event before reading.
This is the same stream-ordered dependency already proven correct in the xdev module.

**Buffer layout:**
```
xdev_allreduce_buf[DS4_MAX_GPUS][n_embd]  // one partial per rank, f32
  - buf[rank] = this rank's partial (written locally)
  - buf[peer] = peer's partial (received via xdev_copy)
  - After all-gather: accumulate buf[0..3] → result
```

### 3.6 Output head / vocabulary sharding

**Current (2-way):** Vocabulary rows split V/2 per pair. `metal_graph_cuda_tp_output_tiers_for_head()`
assigns output shards via `partner = tier + half`.

**New (4-way):** Vocabulary rows split V/4 per rank. Each rank holds its quarter of the
output projection weight. Logit computation:

1. Each rank computes partial logits: `partial = hidden × vocab_shard[rank]` → shape `[n_vocab/4]`
2. All-gather: each rank receives the other 3 quarters → full `[n_vocab]` logits

Alternatively (and cheaper for sampling): keep logits distributed, sample locally on each
rank's shard, then communicate only the selected token ID. This avoids the all-gather of
the full vocabulary. For greedy/top-k sampling, each rank finds the local max in its shard,
then a small all-gather of 4 (rank, token_id, logit) tuples determines the global max.

**Recommendation:** Use distributed sampling (no full vocab all-gather) for decode. For
prefill and quality fixture scoring (which needs full logits), use all-gather.

### 3.7 The `half = n_gpus / 2` sites to change

Every site in `ds4.c` that uses `half` or `partner` for TP decisions:

| Line | Pattern | Change |
|---|---|---|
| 78 | `half = n_gpus / 2` in output tiers | Generalize to N-way shard selection |
| 16745 | `half = g_n_gpus / 2`, `return tier + half` | Remove — no partner concept |
| 16956 | `half = g_n_gpus / 2`, decode TP init | `tp_world = n_gpus` |
| 16968 | `used_tier[t + half] = true` | All tiers are used |
| 54538 | `n_stages = pcfg->n_gpus / 2` | `n_stages = 1` |
| 54569 | `half = n_gpus / 2`, layer placement | All tiers get all layers |
| 54603 | head placement on partner tier | Head replicated or sharded 4-way |
| 54621 | `n_stages = n_gpus / 2` | `n_stages = 1` |
| 54855 | `tp_half = n_gpus / 2`, tensor sharding | `tp_world = n_gpus` |
| 54960 | `partner_tier = logical_tier + tp_half` | Remove — all tiers hold all tensors |
| 55103 | `tp_half = n_gpus / 2`, exec tier mapping | `exec_tier = logical_tier` (no offset) |
| 55138 | `if (t < tp_half) continue` | Remove — all tiers pack |
| 55819 | `half = n_gpus / 2` | Remove or replace with `tp_world` |

Each site needs individual review. The pattern is consistent: replace `half`-based binary
decisions with `tp_world`-based N-way decisions.

## 4. API Contracts

### 4.1 Sharding policy

```c
// New function (replaces implicit half/partner logic):
typedef struct {
    uint32_t tp_world;      // 4 for TP=4, 1 for no TP
    uint32_t tp_rank;       // 0..3
    uint32_t head_start;    // first head owned by this rank
    uint32_t head_count;    // heads owned by this rank (128/4 = 32)
    uint32_t expert_start;  // first expert owned by this rank
    uint32_t expert_count;  // experts owned by this rank (256/4 = 64)
    uint32_t vocab_start;   // first vocab row owned by this rank
    uint32_t vocab_count;   // vocab rows owned by this rank
    uint32_t ffn_col_start; // FFN column shard start
    uint32_t ffn_col_count; // FFN column shard width
} ds4_tp_shard_config;

// Pure function: given world size and rank, compute shard boundaries.
// Unit-testable with no GPU.
ds4_tp_shard_config ds4_tp_compute_shard_config(uint32_t tp_world, uint32_t tp_rank,
                                                 uint32_t n_heads, uint32_t n_experts,
                                                 uint32_t n_vocab, uint32_t n_embd);
```

### 4.2 All-reduce primitive

```c
// Initialize all-reduce buffers for a given element count.
// Called once during graph construction.
int ds4_rocm_xdev_allreduce_init(uint32_t n_elements);

// Perform all-reduce: each rank contributes its partial in `my_partial`,
// result (sum of all partials) is written to `result` on the calling rank's device.
// Stream-ordered: records producer event, waits on all peers, copies, accumulates.
int ds4_rocm_xdev_allreduce_f32(ds4_rocm_xdev_mesh *mesh,
                                 int my_rank,
                                 float *my_partial,    // on my_rank's device
                                 float *result,        // on my_rank's device
                                 uint32_t n_elements,
                                 hipStream_t stream);
```

### 4.3 Layer placement

```c
// New: every tier gets every layer, sharded by tp_rank.
// Replaces the lower/upper half split.
//
// For sharded tensors: load the tp_rank'th shard.
// For replicated tensors: load the full tensor on every rank.
```

## 5. Synchronization Model

### 5.1 Decode (single token)

Per layer, the decode path does:

1. **Attention compute** — each rank computes its 32 heads independently
2. **Attention all-reduce** — combine 4 partial `n_embd` vectors
   - Record producer event on own device
   - Wait on all 3 peer producer events
   - Copy 3 peer partials to local all-reduce buffer
   - Accumulate 4 buffers → result
3. **MoE compute** — each rank computes shared/4 + owned routed experts
4. **MoE all-reduce** — combine 4 partial `n_embd` vectors (same pattern)
5. **Output head** (final layer only) — distributed sampling

**Ordering invariant:** steps 1 and 3 are local compute (no cross-device dependency).
Steps 2 and 4 are all-reduces that depend on step 1/3 completing on ALL ranks. The
producer event mechanism ensures this: each rank records an event after its compute,
and the all-reduce waits on all peer events before reading.

**No global barrier needed.** The per-rank stream ordering via events is sufficient:
each rank proceeds independently, blocking only when it needs to read a peer's buffer.
This is the same model as the current 2-rank path, extended to 4 peers.

### 5.2 Prefill (batch of tokens)

Prefill processes a chunk of tokens (up to 2048) through all layers. The all-reduce
pattern is the same, but vectors are `n_tokens × n_embd` instead of `1 × n_embd`.

For a 2048-token prefill chunk: 2048 × 7168 × 4 bytes = 56 MB per all-reduce.
At 24 GB/s per link, 3 copies = 3 × 56 MB / 24 GB/s ≈ 7 ms per all-reduce.
Two per layer × 43 layers = ~600 ms. Against a prefill compute budget of
2048 / 206 t/s ≈ 10 seconds, this is ~6% overhead. Acceptable.

**Optimization for prefill:** use the "big gate" exchange path (already exists for
prefill in the 2-rank code) to amortize the exchange overhead across larger chunks.

### 5.3 Failure modes

- **One rank slower:** The all-reduce naturally waits (via producer events). No timeout
  needed for correctness; the gate timeout from the upstream TP code (300s) is generous.
- **Peer copy failure:** `ds4_rocm_xdev_copy` falls back to host staging. If both fail,
  the accumulate kernel reads garbage — detected by the quality fixture, not silently.
- **Partial all-reduce (one peer missing):** The result is the sum of 3 of 4 partials —
  output is plausible-looking but wrong. Same failure mode as the current 2-rank path.
  Mitigated by the serialized quality fixture as the reference.

## 6. Migration Strategy

### 6.1 Parallel path (recommended)

Add TP=4 as a new code path alongside the existing TP=2 pair path. Do not modify the
TP=2 path in place. The `tp_world` value selects which path runs:

- `tp_world == 1`: no TP (pipeline or single-GPU)
- `tp_world == 2`: existing 2-pair TP (preserved as fallback)
- `tp_world == 4`: new 4-rank TP

This allows:
- Incremental development (each subsystem can be tested independently)
- A/B comparison (same model, same hardware, different topology)
- Rollback (if TP=4 has issues, TP=2 still works)

### 6.2 Vertical slice order

1. **Sharding policy** — `ds4_tp_compute_shard_config()` + unit tests. No GPU code.
2. **All-reduce primitive** — `ds4_rocm_xdev_allreduce_f32()` + standalone device buffer
   test. No model code.
3. **Layer placement** — all 4 GPUs load all layers, sharded. Model loads without crash.
4. **Attention 4-way** — head split + all-reduce. Generate a coherent sentence.
5. **MoE 4-way** — expert split + all-reduce. Generate a longer paragraph.
6. **Output head 4-way** — vocab shard + distributed sampling. Sampling produces correct tokens.
7. **Quality fixture** — 100-case run. Authoritative gate.
8. **Throughput measurement** — `ds4-bench` sweep.

Each slice produces a runnable state that can be tested before proceeding.

## 7. Test Plan

### 7.1 Sharding policy (pure CPU)

- Complete partition: every head/expert/vocab row assigned to exactly one rank
- No gaps, no overlaps: union of all rank shards = full dimension
- Even division: 128/4 = 32 heads, 256/4 = 64 experts per rank
- Degenerate: tp_world=1 → single rank owns everything
- Boundary: tp_world does not divide evenly → graceful error (not silent mis-shard)

### 7.2 All-reduce (device buffers)

- 4-rank all-reduce of known vectors: sum matches CPU reference
- Byte-exact: f32 sum in canonical rank order (buf[0] + buf[1] + buf[2] + buf[3])
- Bandwidth: measure time for n_embd and prefill-chunk-sized all-reduces
- Fallback: force host-staging mode, verify correctness

### 7.3 End-to-end correctness

- Single-token generation: `"Hello"` → coherent continuation
- Multi-token: `"Explain C pointers"` → fluent paragraph
- Quality fixture: 100 cases, avg_nll within ±1% of serialized pipeline reference

### 7.4 Performance

- Decode throughput: `ds4-bench --ctx-start 2048 --gen-tokens 256` vs pipeline baseline
- Per-GPU utilization: `rocm-smi` during decode (target: >75% on all 4 GPUs)
- All-reduce overhead: measured as fraction of per-token time

## 8. Open Questions

1. **Shared expert in MoE:** Should rank 0 compute the full shared expert (others contribute
   zero), or should all ranks compute shared/4? The first is simpler; the second distributes
   compute but requires a division. Recommend: rank 0 computes full shared expert.

2. **Prefill exchange for large chunks:** The current "big gate" path uses CPU bounce
   buffers. For TP=4 prefill, the all-reduce vectors are 56 MB — too large for slab slots.
   Need to extend the big-gate path to 4-way, or use a separate prefill all-reduce that
   operates on larger buffers.

3. **MLA compressed KV cache:** Currently replicated. With TP=4, each rank needs the full
   compressed KV to compute attention on its heads. Replication is correct but uses 4× the
   memory. Alternative: shard the compressed KV and all-gather before attention. For now,
   replicate (simpler, and the compressed KV is small).

4. **Compatibility with existing TP=2 tests:** The `test_rocm_tp_stubs` and other TP tests
   currently assume 2-rank. They need to be updated or new 4-rank variants added.

5. **`ds4_tp.c` upstream transport:** Not used by ROCm (Metal-only). No changes needed.
   But if TP=4 is ever extended to multi-machine Metal, the upstream transport would need
   N-way support. Out of scope for this work.

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| All-reduce correctness bug | Medium | High (silent corruption) | Canonical sum order, unit tests, quality fixture gate |
| Performance doesn't beat pipeline | Low | High (wasted effort) | Early throughput check after slice 6 |
| Shared expert double-counting | Medium | Medium (quality regression) | Unit test: sum of all rank partials = 1× shared + all routed |
| VRAM OOM during model load | Low | Low (adjust shard sizes) | 23 GiB estimated vs 34 GiB budget |
| Prefill exchange too slow | Low | Medium (prefill regression) | Measure at slice 7; optimize big-gate if needed |
| Scope creep to arbitrary N | Low | High (delays delivery) | Hard-code N=4 throughout; parameterize later if needed |
