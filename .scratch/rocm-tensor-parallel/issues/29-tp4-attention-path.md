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

- [ ] Attention head split generalized: `tp_groups = n_groups / tp_world` (works for both 2 and 4)
- [ ] Attention output exchange: `tp_world == 4` branch calls all-reduce instead of 2-rank gate
- [ ] MLA compressed KV remains replicated (no sharding change)
- [ ] `"Explain C pointers in one sentence."` → fluent, coherent single sentence on 4 GPUs
- [ ] TP=2 attention path unchanged (still works with `tp_world == 2`)
- [ ] `make -j8 rocm` builds cleanly

## Blocked by

- Issue #28: TP=4 layer placement (model must load on 4 GPUs first)
- Issue #27: TP=4 all-reduce primitive (needed for attention output combine)
