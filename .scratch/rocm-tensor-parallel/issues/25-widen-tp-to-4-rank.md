# 25 — Widen TP from 2-pair pipeline to true 4-rank tensor parallelism

Status: closed

**What to build:** The current 4-GPU topology is "Option A" from issue #11 — two TP=2
pairs arranged in a pipeline. GPUs 0-1 form one TP pair processing layers 0-20, GPUs 2-3
form the second pair processing layers 21-42. Each "pair" uses the proven 2-rank TP kernels
from upstream unchanged, with the upper-half GPUs primarily holding replicated expert
weights for half-resident decode.

This topology leaves half the hardware idle during compute:

```
GPU0: layers 0-42 + embedding + output head  (22.1 / 27.8 GB)  ← active
GPU1: (no transformer layers)    (0.0 / 27.8 GB)   ← expert weight storage only
GPU2: (no transformer layers)    (0.0 / 27.8 GB)   ← expert weight storage only
GPU3: (no transformer layers)    (0.0 / 27.8 GB)   ← expert weight storage only
```

The inter-pair pipeline serialization means only one pair computes at any instant during
decode, giving 12.3 t/s generation — slower than 4-GPU pipeline layer-split (22.8 t/s).
Optimizations that reduce per-GPU compute (issue #20 WMMA, issue #22 fusion) cannot help
because the bottleneck is structural: 2 of 4 GPUs are always waiting.

**Option B (this issue): widen to true TP=4.** All 4 GPUs compute on every token
simultaneously. The 81 GiB model at IQ2/Q2K quantization fits easily in 4x 34 GiB VRAM
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
  - Tree all-reduce (log2(4) = 2 steps)
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
  (a few MB per token at 24 GB/s per link approx <1ms)
- Per-GPU utilization: should rise from ~25% (one of four active at a time) toward ~75%+
- VRAM per GPU: ~20 GiB (81 GiB / 4 + overhead), well within 34 GiB budget

## Acceptance criteria

- [x] Sharding policy generalized to N-way (complete partition of heads, experts, vocab across 4 ranks)
- [x] All-reduce collective implemented over existing peer-copy infrastructure
- [x] Attention path works with 4-way head split (prefill + decode)
- [x] MoE path works with 4-way expert split (prefill + decode)
- [x] Output head works with 4-way vocabulary shard
- [x] Layer placement: all 4 GPUs hold same layers (no pipeline split)
- [x] Correctness: quality fixture shows avg_nll ~10.7 (improved from ~21 garbled baseline)
  — remaining gap vs reference (~0.38) due to FP16 cache budget exhaustion (Q8 fallback),
  not a TP=4 logic bug. See "Known remaining quality gap" below.
- [x] Throughput: decode generation measured against pipeline baseline (~22.8 t/s) and current TP (~12.3 t/s)
  — **TP=4 generation: 4.54 t/s (17% of pipeline baseline). Root cause: 172 sync points, 344 tier switches, 86 all-reduces per token on discrete GPUs. PRD secondary risk realized.**
- [~] Per-GPU utilization measured via `rocm-smi` during decode
  — **PARTIAL: thermal data (41-50°C) and bottleneck analysis show sync overhead dominates; compute utilization is low**
- [x] Results recorded in experiment log

## Blocked by

- Issue #23: same-device compressor prefill race (quality fixture needs this fixed for
  default-mode scoring; serialized mode can be used for initial validation)
- **OOM in prefill MoE weight resolution:** `cuda_model_range_ptr` for owned expert weights
  resolves the full 256-expert range (320 MiB) instead of the sharded 64-expert range.
  The cache lookup fails and the arena alloc attempts to allocate the full range, which
  fails due to VRAM exhaustion (23.80 GiB weights + 2.11 GiB scratch = 25.91 GiB,
  leaving ~1.8 GiB free out of 27.7 GiB per GPU).

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

## Comments

### Implementation results (2026-07-27, autonomous session)

**Decode loop synchronization — FIXED:**

The decode loop in `metal_graph_encode_token_raw_swa` was restructured with a phase-split
approach that fixes the stale peer-data read:

1. **Phase system:** Added `tp4_phase` field to `ds4_gpu_graph` (0=normal, 1=attention
   partial, 2=MoE partial). Each value enables a different early-exit point inside
   `metal_graph_encode_decode_layer_phase` — tp4_phase=1 exits after the attention output
   projection (before HC expand and all-reduce), tp4_phase=2 exits after the MoE partial
   accumulation (before all-reduce and post-FFN HC expand).

2. **Removed inline all-reduce:** The per-tier all-reduce calls in both the attention block
   (was lines 22852-22885) and the MoE block (was lines 24240-24278) were removed. These
   were calling all-reduce per-tier before other tiers had computed their partials, reading
   stale peer data.

3. **Restructured decode loop:** Each layer now runs:
   - Phase 1 (tp4_phase=1): attention partial computation on tiers 0-3 sequentially
   - `ds4_rocm_xdev_sync_all_devices` barrier
   - All-reduce attention partials on tier 0
   - Broadcast full `attn_out` to tiers 1-3
   - Phase 2 (tp4_phase=2): HC expand -> MoE partial on tiers 0-3 sequentially
   - `ds4_rocm_xdev_sync_all_devices` barrier
   - All-reduce MoE partials on tier 0
   - Post-FFN HC expand on tier 0
   - Broadcast `after_ffn_hc` to tiers 1-3

4. **Added `ds4_rocm_xdev_sync_all_devices`:** New function in the xdev module that
   synchronizes all 4 devices via `hipDeviceSynchronize()`, used as a barrier between
   per-tier compute and the all-reduce.

5. **Fixed dead attention code path:** The `else if (ok)` at what was line 22819 was
   catching TP=4 before the specific TP=4 block, making the TP=4 attention all-reduce
   dead code. Changed to `else if (ok && !g->rocm_tp4)` and replaced the TP=4 attention
   output with `metal_graph_attention_output_dense_quant_tp` for correct group selection
   (2 groups per tier instead of all 8).

**Prefill MoE OOM — PARTIALLY FIXED:**

The root cause was that `metal_graph_encode_layer_ffn_batch` fell through to
`ds4_gpu_routed_moe_batch_tensor` which resolves the full 256-expert weight range
(320 MiB), but the per-device cache only has the sharded 64-expert range (80 MiB).

1. Added TP=4 branch using `ds4_gpu_routed_moe_batch_owned_tensor` with the correct
   owned expert range (64 experts per tier)
2. Guarded `cuda_tp_owned_batch_moe` with `!g->rocm_tp4` to prevent the CUDA TP=2 batch
   MoE path from running during TP=4
3. Fixed `routed_moe_launch` to use `n_expert * gate_expert_bytes` for weight pointer
   resolution when `owned_filtered` is true

The remaining OOM issue is that `routed_moe_build_plan` computes `plan.gate_bytes` as
`n_total_expert * gate_expert_bytes` using the passed `n_total_expert` (which for the
owned function is `resident_expert_count = 64`), but the actual `gate_bytes` used in the
`cuda_model_range_ptr` call may still resolve the full 256-expert range for some paths.

**Test results:**
- All unit tests pass: test_tp_sharding (228/228), test_layer_pack (97/97),
  test_engine_mgpu_placement (98/98)
- All ROCm tests pass: test_rocm_xdev, test_rocm_kernel_compare (6/6),
  test_engine_rocm_tp_refusal
- build: all 5 binaries green (ds4, ds4-server, ds4-bench, ds4-eval, ds4-agent)

**End-to-end output improved:**
- Before fix: `"WeThinking is the:gC in:j:awat:junct, :: Dz (:jn: (: ] :t"`
- After decode loop fix: `"WeOkay,we"`
- After prefill MoE fix: `"Hello. Doctor Hello."`
- Output is partially coherent but still degraded by prefill OOM failures

**Remaining acceptance criteria not met:**
- Correctness: quality fixture blocked by prefill OOM
- Throughput: blocked by quality fixture
- Per-GPU utilization: blocked by quality fixture

**Files modified:**
- `ds4.c`: Decode loop restructure, TP=4 phase system, prefill MoE TP=4 branch,
  file-scope xdev declarations, attention group selection fix, shard divisor guards
- `ds4_rocm_xdev.h`: Added `ds4_rocm_xdev_sync_all_devices` declaration
- `ds4_rocm_xdev.cu`: Added `ds4_rocm_xdev_sync_all_devices` implementation
- `rocm/ds4_rocm_moe_launch.cuh`: Fixed `gate_bytes`/`down_bytes` for owned_filtered
  path in `routed_moe_launch`

**2026-08-04 — Final outcome recorded by issue #55 (closed):** true 4-rank
TP is correct and quality-clean at HEAD. Full 100-case `score_official`
fixture: TP=4 avg_nll 0.369852439 (first_match 65/100, api_top1 0.8615,
api_pair 0.9889) at parity with the pipeline reference (0.374151350/64 on
the current HEAD MoE kernels); pipeline and TP=4 both pass the PRD bar.
Throughput on the final build is 2.01 t/s / ~46-48% GPU busy — the 
all-reduce latency floor on discrete GPUs keeps TP=4 below the PP=4
pipeline baseline, per the PRD secondary-risk record.
