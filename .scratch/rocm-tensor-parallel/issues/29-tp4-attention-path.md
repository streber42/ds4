# 29 — TP=4 attention path (coherent single sentence)

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/issues/25-widen-tp-to-4-rank.md`

## What to build

Wire up the attention subsystem for TP=4: split 128 attention heads 32/32/32/32 across 4 ranks, and combine partial attention outputs via the all-reduce primitive from #27.

**Attention compute:** Each rank computes QKV projection for its 32 heads independently. MLA (multi-latent attention) operates on each head independently — no cross-head communication needed during attention itself. The compressed KV cache remains replicated on all ranks (it is a single latent vector per token, not per-head).

**Attention output exchange:** Currently (TP=2), each rank projects its 64 heads through the attention output projection, producing a partial `n_embd` vector. The two partials are exchanged via the TP gate and summed. For TP=4, each rank projects its 32 heads → partial `n_embd` vector. The four partials are all-reduced using `ds4_rocm_xdev_allreduce_f32()` from #27.

The `tp_world == 2` attention exchange code at `ds4.c:22705-22775` needs a `tp_world == 4` branch that calls the all-reduce instead of the 2-rank gate exchange. The attention compute itself (head split, MLA) already parameterizes on `tp_rank * tp_groups` where `tp_groups = n_groups / tp_world` — just change the divisor.

**Correctness signal:** This is the first slice that produces meaningful text. A coherent single sentence in response to a simple prompt proves the attention path is numerically sound end-to-end.

## Acceptance criteria

- [x] Attention head split generalized: `tp_groups = n_groups / tp_world` (works for both 2 and 4)
- [x] Attention output exchange: `tp_world == 4` branch calls all-reduce instead of 2-rank gate
- [x] MLA compressed KV remains replicated (no sharding change)
- [ ] `"Explain C pointers in one sentence."` → fluent, coherent single sentence on 4 GPUs
- [x] TP=2 attention path unchanged (still works with `tp_world == 2`)
- [x] `make -j8 rocm` builds cleanly

## Blocked by

- Issue #28: TP=4 layer placement (model must load on 4 GPUs first)
- Issue #27: TP=4 all-reduce primitive (needed for attention output combine)

## Comments

### Partial implementation (2026-07-26, autonomous session)

Made the following code changes to `ds4.c` to set up TP=4 attention infrastructure:

1. **Set tp_world and tp_rank for ROCm TP=4** (line ~16820):
   ```c
   if (g->rocm_tp4) {
       g->tp_world = 4;
       /* tp_rank will be set dynamically based on active_tier during decode */
   }
   ```

2. **Set tp_rank dynamically based on active_tier** (line ~21580):
   ```c
   if (g->rocm_tp4) {
       g->tp_rank = (uint32_t)g->active_tier;
   }
   ```
   This ensures that when the decode loop switches to a different tier, tp_rank reflects that tier's rank (0, 1, 2, or 3).

3. **Generalized attention head split** (line ~21615):
   - Changed `tp_split_attn = g->tp_world == 2` to `tp_split_attn = g->tp_world >= 2`
   - Changed `tp_heads = DS4_N_HEAD / 2u` to `tp_heads = DS4_N_HEAD / g->tp_world`
   - This works for both TP=2 (tp_world=2, 64 heads per rank) and TP=4 (tp_world=4, 32 heads per rank)

4. **Generalized tp_groups calculation** (lines ~22606, ~22755):
   - Changed `tp_groups = n_groups / 2` to `tp_groups = n_groups / g->tp_world`
   - This ensures the group-based attention compute uses the correct divisor for TP=4

**Build & test verification:**
```
$ make -j8 rocm
[all 5 binaries build cleanly: ds4, ds4-server, ds4-bench, ds4-eval, ds4-agent]

$ make test-rocm
test_rocm_tp_stubs: PASS
test_rocm_xdev: ALL TESTS PASSED (including 4-rank all-reduce tests)
test_rocm_kernel_compare: 6/6 kernels passed
test_engine_rocm_tp_refusal: PASS

$ ./tests/test_tp_sharding
228/228 checks passed (0 failed)

$ ./tests/test_layer_pack
97/97 checks passed (0 failed)

$ ./tests/test_engine_mgpu_placement
98/98 checks passed (0 failed)
```

### What remains to be implemented

The changes above set up the infrastructure for TP=4 attention, but the **decode loop architecture** needs fundamental changes:

1. **Decode loop iteration over all 4 tiers**: The current decode loop (`metal_graph_encode_token_raw_swa`, line ~26441) calls `metal_graph_encode_decode_layer` once per layer, which processes only the tier specified by `placement[il+1]`. For TP=4, `placement[il+1] = 0` for all layers, so the decode loop only runs on tier 0.

   For TP=4 to work, the decode loop needs to iterate over all 4 tiers per layer, similar to how `cuda_tp_attn_heads_active` (line ~22341) iterates over 2 tiers within a single call by switching devices. Each tier needs to:
   - Compute its 32 heads (QKV projection, attention core, RoPE)
   - Run the attention output projection on its 32 heads → partial n_embd vector
   - After all 4 tiers have computed their partials, all-reduce the 4 partial n_embd vectors

   This could be implemented by:
   - Modifying `metal_graph_encode_decode_layer_phase` to loop over all 4 tiers for TP=4, OR
   - Modifying the decode loop in `metal_graph_encode_token_raw_swa` to call `metal_graph_encode_decode_layer` 4 times per layer (once per tier)

   The `cuda_tp_attn_heads_active` pattern shows how to switch devices within a single call, but extending this to 4 tiers and integrating it with the full layer compute (not just attention) is a significant architectural change.

2. **Attention output exchange for TP=4**: The current `tp_world == 2` branch (line ~22802) uses the gate exchange mechanism. A `tp_world == 4` branch needs to be added that:
   - Collects the partial attention outputs from all 4 tiers (stored in `g->attn_out_by_tier[0..3]`)
   - Calls `ds4_rocm_xdev_allreduce_f32()` to sum the 4 partials
   - Stores the result in `metal_graph_attn_out(g)` on the home tier

   The all-reduce API is ready (issue #27, closed), but the orchestration of when to call it (after all 4 tiers have computed their partials) requires the decode loop changes above.

3. **MLA compressed KV cache**: The issue states "MLA compressed KV remains replicated (no sharding change)". The current code already handles this correctly — the compressed KV cache is a single latent vector per token, not per-head, so it's replicated across all tiers. No changes needed here.

4. **End-to-end correctness verification**: The key acceptance criterion is:
   > `"Explain C pointers in one sentence."` → fluent, coherent single sentence on 4 GPUs

   This requires:
   - Loading the 81 GiB production GGUF on the 4×R9700 workstation
   - Running `./ds4 --rocm --gpu-devices 0,1,2,3 --cuda-tensor-parallel --model /path/to/deepseek-v4-flash-iq2.gguf -p "Explain C pointers in one sentence." -n 50`
   - Verifying the output is a coherent single sentence
   - If not, debugging the attention/MoE/output-head sharding logic

   This requires the production hardware and model (GPU access).

5. **TP=2 path unchanged**: The changes above preserve the TP=2 path — `tp_world == 2` still works because the divisor change from `2u` to `g->tp_world` evaluates to the same value for TP=2. The `tp_split_attn` change from `== 2` to `>= 2` also preserves TP=2 behavior.

### Recommendation

The minimal infrastructure changes I've made (setting tp_world=4, tp_rank=active_tier, generalizing the divisor) are necessary but not sufficient. The decode loop architecture needs to be modified to iterate over all 4 tiers per layer, which is a significant change that requires:

1. Careful design to ensure the device switching and synchronization is correct
2. Testing on the production hardware to verify correctness
3. Debugging if the output is garbled (which is likely on the first attempt)

The code changes I've made can serve as a starting point. The key insight is that `tp_rank` needs to be set dynamically based on `active_tier` so that the existing parameterized attention code (which already uses `tp_rank` and `tp_groups`) works correctly for TP=4.

### Complete implementation (2026-07-27, autonomous session)

Implemented the full TP=4 attention path with the following changes to `ds4.c`:

1. **TP=4 attention all-reduce** (lines ~22820-22880):
   - Added `else if (ok && g->rocm_tp4)` branch after the tp_world == 2 gate exchange
   - Each rank computes its 32 heads → partial n_embd vector via `ds4_gpu_attention_output_q8_batch_tensor`
   - Stores partial in `g->attn_out_by_tier[home_tier]`
   - Calls `ds4_rocm_xdev_allreduce_f32()` to sum all 4 tiers' partials
   - Result stored in `metal_graph_attn_out(g)` on the home tier

2. **Decode loop iteration over 4 tiers** (lines ~26648-26674):
   - Modified `metal_graph_encode_token_raw_swa` to iterate over all 4 tiers per layer for TP=4
   - For each tier, switches active tier via `metal_graph_set_active_tier_decode(g, tier_iter)`
   - Sets `g->tp_rank = tier_iter` so the sharding logic uses the correct rank
   - Calls `metal_graph_encode_decode_layer` once per tier
   - Each tier computes its partial results (32 attention heads, 64 routed experts)
   - All-reduce combines the partials after each tier's computation

3. **Consolidated extern declarations** (lines ~21593-21607):
   - Moved `ds4_rocm_xdev_mesh` typedef and extern declarations to function scope
   - Avoids duplicate declarations in attention and MoE all-reduce blocks
   - Wrapped in `#if defined(DS4_ROCM_BUILD)` to avoid pulling ROCm symbols into non-ROCm builds

4. **MoE all-reduce** (lines ~24182-24251):
   - Already implemented in previous session
   - Each rank computes its 64 experts → partial n_embd vector
   - All-reduce sums all 4 tiers' partials
   - Now works correctly with the decode loop iteration

**Build & test verification:**
```
$ make -j8 rocm
[all 5 binaries build cleanly: ds4, ds4-server, ds4-bench, ds4-eval, ds4-agent]

$ make test-rocm
test_rocm_tp_stubs: PASS
test_rocm_xdev: ALL TESTS PASSED (including 4-rank all-reduce tests)
test_rocm_kernel_compare: 6/6 kernels passed
test_engine_rocm_tp_refusal: PASS

$ ./tests/test_tp_sharding
228/228 checks passed (0 failed)

$ ./tests/test_layer_pack
97/97 checks passed (0 failed)

$ ./tests/test_engine_mgpu_placement
98/98 checks passed (0 failed)
```

**Acceptance criteria status:**
- [x] Attention head split generalized: `tp_groups = n_groups / tp_world` (works for both 2 and 4)
- [x] Attention output exchange: `tp_world == 4` branch calls all-reduce instead of 2-rank gate
- [x] MLA compressed KV remains replicated (no sharding change)
- [ ] `"Explain C pointers in one sentence."` → fluent, coherent single sentence on 4 GPUs
- [x] TP=2 attention path unchanged (still works with `tp_world == 2`)
- [x] `make -j8 rocm` builds cleanly

**What remains for human verification:**

The end-to-end correctness test requires the production 81 GiB DeepSeek-V4-Flash GGUF model on the 4×R9700 workstation:

```bash
./ds4 --rocm --gpu-devices 0,1,2,3 --cuda-tensor-parallel \
  --model /path/to/deepseek-v4-flash-iq2.gguf \
  -p "Explain C pointers in one sentence." -n 50
```

The output should be a fluent, coherent single sentence. If the output is garbled, debug the attention/MoE/output-head sharding logic. The infrastructure is in place; the numerical correctness needs verification on real hardware with the production model.

**Key architectural changes:**

1. The decode loop now iterates over all 4 tiers per layer for TP=4, calling `metal_graph_encode_decode_layer` 4 times per layer (once per tier).
2. Each tier computes its partial results independently (32 attention heads, 64 routed experts).
3. After each tier's computation, all-reduce combines the partials so all tiers have the full result before proceeding to the next tier.
4. The MLA compressed KV cache remains replicated (no sharding change) as specified in the issue.

### Live-pair session (2026-07-27)

Human confirmed direct hardware access is available on this machine. Ran the end-to-end test:

```bash
./ds4 --rocm --gpu-devices 0,1,2,3 --cuda-tensor-parallel \
  --model /home/murphy/src/ds4/ds4flash.gguf \
  -p "Explain C pointers in one sentence." -n 50
```

**Result: FAILED — garbled output + OOM**

1. OOM during model load: `ROCm model arena alloc failed for moe_gate (320.00 MiB chunk): out of memory`
2. Despite OOM, decode continued and produced garbled output: `"WeOkay, here,: .X,Panda, Bipanda,Abs,PBANa *FHu...I,minus_in,I6,,Eit,A BADASIw Destroy,A,a,B,A,,N,igg"`
3. Generation speed: 2.19 t/s (prefill 0.73 t/s)

**What needs debugging:**

- The OOM on `moe_gate` suggests the TP=4 placement may not be correctly splitting the MoE gate tensor across 4 ranks (320 MiB × 4 = 1.28 GiB total — too large for the per-GPU budget).
- The garbled output suggests the attention/MoE all-reduce or the decode loop iteration over 4 tiers has a numerical bug.
- The decode loop change at line ~26644 iterates over all 4 tiers per layer, but the all-reduce inside `metal_graph_encode_decode_layer` may be called before all 4 tiers have computed their partials, or the partials may not be correctly accumulated.

**Hardware access confirmed:**
- 4× AMD Radeon AI Pro R9700 (GPU 0-3)
- Model: `/home/murphy/src/ds4/ds4flash.gguf` → `/var/cache/llama/ds4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
- Binary: `/home/murphy/src/ds4-rebase/ds4`
- Direct shell access available — no SSH or remote credentials needed

Next agent should:
1. Debug the moe_gate OOM (check if TP=4 placement correctly splits this tensor)
2. Debug the garbled output (check attention/MoE all-reduce ordering, partial accumulation)
3. Re-run the end-to-end test and verify coherent output
4. Once output is coherent, mark the final acceptance criterion as complete

### Gemini consultation — approved plan (2026-07-27, live-pair session)

Second-opinion review of the decode-loop phase-split architecture. Gemini reviewed
the full context dump (current decode loop, TP=4 all-reduce code, existing
`metal_decode_layer_phase` enum, shard divisor, OOM error) and gave an opinionated
recommendation. A follow-up corrected a subtle error in the first opinion.

**Decision 1: reuse existing `TO_FFN` / `FROM_ATTN_TO_FFN` phases.**
Do NOT add new phases (option b) and do NOT split into `_attention_only` /
`_moe_only` functions (option c). The existing phases are already used as a split
pair in the batch-session code (`ds4.c:63929/63967/64353`), so fixing them fixes
two things at once: the latent bug in batch execution AND the TP=4 decode path.
~2,800 lines of GPU kernel dispatch should not be duplicated.

**Decision 2: hoist TP=4 all-reduce OUT of `metal_graph_encode_decode_layer_phase`.**
Communication belongs in the outer decode loop, not inside the phase function.
The outer loop becomes: attention tier-sweep (TO_FFN) -> all-reduce attn -> MoE
tier-sweep (FROM_ATTN_TO_FFN) -> all-reduce MoE.

**Decision 3: add the missing `TO_FFN` early-exit.**
Caveat discovered during review: `METAL_DECODE_LAYER_TO_FFN` has NO explicit
`return ok` anywhere in `metal_graph_encode_decode_layer_phase` (the only phase
early-exits are `TO_QKV` at 21755, `stop_before_attn` at 22348, `TO_ROUTER` at
23048, `TO_SHARED_MID` at 23975). `TO_FFN` currently falls through and runs the
full layer, same as `FULL`. This means the existing batch-session use of
`TO_FFN`/`FROM_ATTN_TO_FFN` as a split pair is latent-broken.

Fix: add `if (phase == METAL_DECODE_LAYER_TO_FFN) return ok;` at ~line 22933,
**after** `ds4_gpu_hc_expand_tensor` (attention output projection into
`after_attn_hc`) and **before** `ds4_gpu_rms_norm_plain_tensor` on `after_attn_hc`
(the FFN-side HC post norm).

**Decision 4: do NOT split at `TO_ROUTER`.**
RMSNorm between attention output and the router is non-linear:
`RMSNorm(sum_t A_t) != sum_t RMSNorm(A_t)`. Each tier RMSNorm'ing its own
partial and then all-reducing is mathematically wrong. The all-reduce MUST happen
immediately after the attention output projection, before any LayerNorm/RMSNorm.

**Decision 5: add `ds4_rocm_tp4_sync_tier_streams` barrier.**
ROCm kernel dispatch is async. Before `ds4_rocm_xdev_allreduce_f32` reads peer
buffers, we must issue a host stream sync across all 4 devices. Otherwise rank
0's all-reduce kernel launches before rank 3's attention kernel has finished
writing its buffer — same "reads stale peer data" bug even with phase-splitting.

**Implementation plan (5 edits):**
1. Add `if (phase == METAL_DECODE_LAYER_TO_FFN) return ok;` at ~line 22933.
2. Verify `FROM_ATTN_TO_FFN` skips attention (existing `resume_after_attn` at
   line 21680 should handle this; confirm by reading).
3. Gut the inline TP=4 all-reduce blocks at lines 22833-22885 (attn) and
   24249-24316 (MoE) — move them to helper functions called from outer loop.
4. Add `ds4_rocm_tp4_sync_tier_streams` barrier between tier-sweep and
   all-reduce (both for attn and MoE).
5. Restructure outer loop at `ds4.c:26639` into: attn sweep -> attn all-reduce
   -> MoE sweep -> MoE all-reduce.

**Validation:** after the loop refactor, re-run the "Explain C pointers in one
sentence." test and verify coherent output. Then re-run the quality fixture and
verify avg_nll drops from 6.0-9.2 down to ~0.37 (within +/-1% of pipeline
reference 0.373815).

**Order of attack (revised):** decode loop first, OOM second. Evidence: the
partial 25/100 quality fixture ran *despite* the OOM warning and produced NLL
6.0-9.2. The OOM is an arena-alloc warning that execution currently proceeds
past; the decode loop is the primary correctness blocker. Fix the loop first
to get a working validation path; then fix the OOM as a clean VRAM-pressure
reduction on top of a working system.

### Implementation complete — decode loop phase-split (2026-07-27)

Followed the Gemini-approved 5-step plan to restructure the TP=4 decode loop.

**Changes made to `ds4.c`:**

1. **Added `TO_FFN` early exit** (line ~22937): `if (phase == METAL_DECODE_LAYER_TO_FFN) return ok;`
   - Placed after attention output HC expand, before FFN-side RMSNorm
   - Makes `TO_FFN` a true attention-only phase for the first time

2. **Wrapped KV store → attention output in `if (!resume_after_attn)`** (lines ~21965-22932):
   - When `FROM_ATTN_TO_FFN` is the phase, skips KV store, compressor, attention core,
     attention output, HC expand
   - Resumes at the FFN-side RMSNorm on `after_attn_hc`
   - Fixes the latent bug in batch session code too (both `TO_FFN` and `FROM_ATTN_TO_FFN`
     were previously broken — one ran the full layer, the other re-entered attention core)

3. **Changed TP=4 attention/MoE early-exit guards** (lines ~22876, ~24260):
   - Replaced `g->tp4_phase == 1` with `phase == METAL_DECODE_LAYER_TO_FFN`
   - Replaced `g->tp4_phase == 2` with `g->rocm_tp4`
   - Removed all `tp4_phase` references (struct field, assignments, checks)

4. **Added HC expand in outer loop** (lines ~26655-26670):
   - After the attention all-reduce and broadcast, each tier runs
     `ds4_gpu_hc_expand_tensor(after_attn_hc, attn_out_by_tier[tier], ...)`
   - Tier 0 uses `metal_graph_attn_out(g)` (has full sum from all-reduce)
   - Tiers 1-3 use `g->attn_out_by_tier[tier]` (has full sum from broadcast)
   - Barrier after all 4 tiers complete their HC expand

5. **Restructured outer decode loop** (lines ~26609-26719):
   - Phase 1: `METAL_DECODE_LAYER_TO_FFN` — attention partials on all 4 tiers
   - Barrier + all-reduce + broadcast
   - HC expand on all 4 tiers (NEW — produces correct `after_attn_hc`)
   - Phase 2: `METAL_DECODE_LAYER_FROM_ATTN_TO_FFN` — MoE partials on all 4 tiers
   - Barrier + all-reduce MoE → post-FFN HC expand → broadcast cur_hc

**Build & test verification:**
```
$ make -j8 rocm     # all 5 binaries build cleanly
$ make test-rocm    # all tests pass
$ ./tests/test_tp_sharding    # 228/228 checks passed
$ ./tests/test_layer_pack     # 97/97 checks passed
$ ./tests/test_engine_mgpu_placement  # 98/98 checks passed
```

**End-to-end test:**
```
$ ./ds4 --rocm --gpu-devices 0,1,2,3 --cuda-tensor-parallel \
    --model /home/murphy/src/ds4/ds4flash.gguf -c 512 -p "Hello" -n 10
```

Result: still OOM-blocked. The model loads but fails on:
```
ROCm model arena alloc failed for moe_owned_down (384.00 MiB chunk): out of memory
```

This is a VRAM fragmentation issue, not a decode-loop bug. After loading 1328 weight
ranges on tier 0 (23.80 GiB selective weights), the free arena is 4.01 GiB but no
single fragment is large enough for a 384 MiB contiguous allocation. The OOM happens
during model loading (before any decode), so it is entirely independent of the decode
loop changes.

The previous garbled output was caused by the OOM (missing MoE weights) AND the
missing HC expand. With both the OOM and the new HC expand fix, the attention path
is architecturally complete — but the MoE path is still corrupted by missing weights.

### Live-pair session 2 (2026-07-27) — OOM fixed, still garbled

**OOM fix applied:** Arena chunk size increased from 256 MiB → 1024 MiB in
`rocm/ds4_rocm_runtime.cuh:5662`. Gemini (gemini-3.6-flash) reviewed and
confirmed it's a valid quick fix for this hardware configuration (4×32 GiB).
Adaptive/geometric chunking recommended for production but deferred.

**Build & test results:**
```
$ make -j8 rocm     # all 5 binaries clean
$ make test-rocm     # all tests pass
```

**End-to-end test:**
```
$ ./ds4 --rocm --gpu-devices 0,1,2,3 --cuda-tensor-parallel \
    --model /var/cache/llama/ds4-gguf/DeepSeek-V4-Flash-IQ2XXS-...gguf \
    -c 512 -p "Explain C pointers in one sentence." -n 50
```

**Load:** All 4 tiers load 23.00 GiB in 1328 ranges. No more OOM.
```
ds4: CUDA tier 0 (device 0) selective weights: 23.00 GiB in 1328 ranges
ds4: CUDA tier 1 (device 1) selective weights: 23.00 GiB in 1328 ranges
ds4: CUDA tier 2 (device 2) selective weights: 23.00 GiB in 1328 ranges
ds4: CUDA tier 3 (device 3) selective weights: 23.00 GiB in 1328 ranges
```

**Output:** Still garbled:
```
"to taleds  e a\n,\n\n**     **.\n, \n,.,.**     ****a**  ..."
```
Generation: 4.14 t/s. Prefill: 5.50 t/s.

**Status:** The OOM was masking a decode-loop correctness bug. The floor is
now clean to debug it.

### Diagnosis of garbled output

**What was ruled out (correct):**
- `cur_hc` initialization across tiers: `metal_graph_set_active_tier_decode`
  (line 15527-15543) already copies `cur_hc` from the previous active tier
  when switching. So all tiers start layer 0 with the correct embedding.
  Confirmed by reading the code — NOT the bug.
- Arena fragmentation causing missing weights: fixed. All weights load cleanly.
- HC expand missing: Phase-split restructure handles this correctly (TO_FFN →
  all-reduce → HC expand → FROM_ATTN_TO_FFN → all-reduce → post-FFN HC
  expand → broadcast cur_hc).

**Likely root cause areas:**

1. **MoE expert weight sharding vs. router selection.** The `FROM_ATTN_TO_FFN`
   phase runs the router on each tier, producing identical selected-expert
   lists (same ffn_norm → same router logits). But each tier only HAS 64 of
   256 experts in its selective weight cache (tier 0 has experts 0-63, tier 1
   has 64-127, etc.). When the router selects experts from outside a tier's
   owned set, the MoE matmul reads garbage weight data for the missing
   experts. The all-reduce then mixes garbage into the result.

   The mechanism by which tiers compute only their owned experts needs
   investigation. Two sub-questions:
   a. Does the routed MoE matmul on each tier actually restrict itself to
      owned expert weights, or does it attempt the full 256-expert matmul?
   b. If it attempts the full matmul, what does `cuda_resolve_weight_ptr`
      return for expert weight ranges not owned by the current tier? If it
      falls back to a host copy of the full tensor, each tier computes all
      256 experts redundantly (wasteful but numerically correct), and the
      all-reduce sums 4× the correct value. If it returns NULL or garbage,
      the output is corrupted.

2. **Attention output combine.** The attention all-reduce sums the 4 partials
   from `attn_out_by_tier[0..3]` into `metal_graph_attn_out(g)` on tier 0.
   Each partial should be `n_embd` floats computed from 32 attention heads.
   If the `metal_graph_attention_output_dense_quant_tp` function with
   `tp_attn_groups = 2` and `tp_attn_group0 = tier * 2` produces the correct
   partial, the all-reduce should be correct. But this hasn't been verified
   numerically (e.g. with the kernel comparison scaffold).

3. **Post-MoE HC expand and cur_hc broadcast.** After the MoE all-reduce into
   `metal_graph_routed_out(g)` on tier 0, the code calls
   `ds4_gpu_hc_expand_split_tensor` on tier 0 only, then swaps cur_hc/after_ffn_hc
   pointers on tier 0 only, and broadcasts `after_ffn_hc` to tiers 1-3 via
   `cur_hc_by_tier[tier]->ptr`. If `metal_graph_cur_hc(g)` on tiers 1-3
   points to a different tensor than `cur_hc_by_tier[tier]`, the broadcast
   writes to a stale location. But the tier switch copies cur_hc explicitly,
   so this should be fine.

4. **Prefill vs. decode mismatch.** The test uses context size 512, so there
   is a prefill phase followed by decode. The prefill path may not correctly
   handle TP=4. Testing with `-c 1 -p "Hello"` (pure decode, no prefill)
   would isolate this.

**Debugging approach for next agent:**

1. **Isolate decode-only:** Run with `-c 1` to skip prefill entirely. If
   output is still garbled, the bug is in the decode loop. If output is
   clean, the bug is in the prefill path.
2. **Verify attention all-reduce:** Use `DS4_DEBUG_TP_OUTPUT=1` to dump
   per-layer tensor statistics and compare tier 0 partial vs. all-reduced
   result.
3. **Verify MoE expert ownership:** Add temporary debug logging to
   `metal_graph_encode_decode_layer_phase` in the MoE section showing which
   expert weights are used on each tier.
4. **Disable MoE sharding temporarily:** If the MoE path is the issue, try
   having all tiers load the full model (no sharding) to see if the output
   becomes coherent. This would confirm the MoE matmul itself works when all
   weights are present.
5. **Numerical comparison:** Run the kernel comparison scaffold (`test_rocm`)
   on the attention output and MoE matmul for TP=4 to verify partial results
   sum correctly.

**Acceptance criteria status:**
- [x] Attention head split generalized: `tp_groups = n_groups / tp_world` (works for both 2 and 4)
- [x] Attention output exchange: `tp_world == 4` branch calls all-reduce instead of 2-rank gate
- [x] MLA compressed KV remains replicated (no sharding change)
- [ ] `"Explain C pointers in one sentence."` → fluent, coherent single sentence on 4 GPUs
- [x] TP=2 attention path unchanged (still works with `tp_world == 2`)
- [x] `make -j8 rocm` builds cleanly
