# 30 — TP=4 MoE path (coherent paragraph)

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/issues/25-widen-tp-to-4-rank.md`

## What to build

Wire up the MoE (mixture of experts) subsystem for TP=4: split 256 routed experts 64/64/64/64 across 4 ranks, and combine partial FFN outputs via all-reduce.

**Expert ownership:** Each rank owns 64 contiguous experts (rank 0: 0-63, rank 1: 64-127, etc.). The router selects top-6 experts per token; each rank only computes gate/up/mid/down for the experts it owns (0-2 of the 6 selected, averaging 1.5 per rank).

**Shared expert:** The shared expert is replicated on all ranks. Under TP=2, both ranks compute the shared expert and the exchange sums them — but the code at `ds4.c:24058` adds `shared_out + routed_out` into `tp_out` on each rank, and the exchange sums the two rank partials. For TP=4, only rank 0 computes the shared expert output; ranks 1-3 contribute zero for the shared part. This avoids 4× over-counting after the all-reduce sums all partials.

**FFN exchange:** Currently (TP=2), each rank's partial is `shared_out + sum(owned_routed_experts)`. The two partials are exchanged and summed. For TP=4, each rank's partial is `shared_out/4 (or zero for non-rank-0) + sum(owned_routed_experts)`. The four partials are all-reduced. The canonical sum yields exactly 1× shared expert + all routed experts.

**Correctness signal:** A multi-sentence coherent paragraph proves the MoE path is numerically sound and the shared/routed expert accounting is correct.

## Acceptance criteria

- [ ] Expert ownership: 256 experts split 64/64/64/64 (uses sharding policy from #26)
- [ ] Shared expert: only rank 0 computes it; other ranks contribute zero for shared part
- [ ] FFN exchange: `tp_world == 4` branch calls all-reduce instead of 2-rank gate
- [ ] `"Write a paragraph explaining how recursion works."` → coherent multi-sentence paragraph
- [ ] Shared expert accounting verified: sum of all 4 rank partials = 1× shared + all routed (not 4× shared)
- [ ] TP=2 MoE path unchanged
- [ ] `make -j8 rocm` builds cleanly

## Blocked by

- Issue #29: TP=4 attention path (attention must work before MoE can be tested end-to-end)
