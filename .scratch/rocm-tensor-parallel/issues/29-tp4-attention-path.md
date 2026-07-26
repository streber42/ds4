# 29 — TP=4 attention path (coherent single sentence)

Status: ready-for-human

## Parent

`.scratch/rocm-tensor-parallel/issues/25-widen-tp-to-4-rank.md`

## What to build

Wire up the attention subsystem for TP=4: split 128 attention heads 32/32/32/32 across 4 ranks, and combine partial attention outputs via the all-reduce primitive from #27.

**Attention compute:** Each rank computes QKV projection for its 32 heads independently. MLA (multi-latent attention) operates on each head independently — no cross-head communication needed during attention itself. The compressed KV cache remains replicated on all ranks (it is a single latent vector per token, not per-head).

**Attention output exchange:** Currently (TP=2), each rank projects its 64 heads through the attention output projection, producing a partial `n_embd` vector. The two partials are exchanged via the TP gate and summed. For TP=4, each rank projects its 32 heads → partial `n_embd` vector. The four partials are all-reduced using `ds4_rocm_xdev_allreduce_f32()` from #27.

The `tp_world == 2` attention exchange code at `ds4.c:22705-22775` needs a `tp_world == 4` branch that calls the all-reduce instead of the 2-rank gate exchange. The attention compute itself (head split, MLA) already parameterizes on `tp_rank * tp_groups` where `tp_groups = n_groups / tp_world` — just change the divisor.

**Correctness signal:** This is the first slice that produces meaningful text. A coherent single sentence in response to a simple prompt proves the attention path is numerically sound end-to-end.

## Acceptance criteria

- [ ] Attention head split generalized: `tp_groups = n_groups / tp_world` (works for both 2 and 4)
- [ ] Attention output exchange: `tp_world == 4` branch calls all-reduce instead of 2-rank gate
- [ ] MLA compressed KV remains replicated (no sharding change)
- [ ] `"Explain C pointers in one sentence."` → fluent, coherent single sentence on 4 GPUs
- [ ] TP=2 attention path unchanged (still works with `tp_world == 2`)
- [ ] `make -j8 rocm` builds cleanly

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

The changes above set up the infrastructure for TP=4 attention, but the **decode loop architecture** needs fundamental changes that require human judgment and hardware access:

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

   This cannot be done without the production hardware and model.

5. **TP=2 path unchanged**: The changes above preserve the TP=2 path — `tp_world == 2` still works because the divisor change from `2u` to `g->tp_world` evaluates to the same value for TP=2. The `tp_split_attn` change from `== 2` to `>= 2` also preserves TP=2 behavior.

### Recommendation

The minimal infrastructure changes I've made (setting tp_world=4, tp_rank=active_tier, generalizing the divisor) are necessary but not sufficient. The decode loop architecture needs to be modified to iterate over all 4 tiers per layer, which is a significant change that requires:

1. Careful design to ensure the device switching and synchronization is correct
2. Testing on the production hardware to verify correctness
3. Debugging if the output is garbled (which is likely on the first attempt)

Given that issue #28 (layer placement) is marked `ready-for-human` for the same reason (cannot verify without hardware), and this issue requires even more complex changes, I recommend marking this as `ready-for-human` as well.

The code changes I've made can serve as a starting point for the human implementer. The key insight is that `tp_rank` needs to be set dynamically based on `active_tier` so that the existing parameterized attention code (which already uses `tp_rank` and `tp_groups`) works correctly for TP=4.
