# ROCm Tensor Parallel Entry Point Inventory

Status: `resolved` (15 / 37 implemented; remaining 22 all deliberately refused or proven unreachable for this model/config; none are silent no-ops)

This inventory enumerates all 37 GPU entry points for the ROCm tensor-parallelism port.
It serves as the machine-checkable and human-readable progress tracker and the guard against quietly forgetting any required stub.

Every entry below is resolved in one of three ways:

- **Implemented** — a real kernel/logic path exists and is reachable.
- **Deliberately refused** — reachable only through an out-of-scope feature (DSpark
  speculative decoding, session/continuous batching, or an off-by-default optimization
  flag); intentionally left unported and fails loudly if a future config reaches it.
- **Proven unreachable** — structurally dead code for this model/config (a type check,
  forced-off flag, or backend guard makes the call site impossible to hit).

No entry point returns a silent neutral value: every stubbed entry aborts by default via
`ds4_rocm_tp_stub`/`ds4_rocm_tp_stub_ok` (see `ds4_rocm_tp_bringup.h`), announcing itself and
naming the caller, unless `DS4_ROCM_TP_BRINGUP=1` is explicitly set for plumbing bring-up.

Machine-checkable representation: [inventory.json](file:///home/murphy/src/ds4-rebase/.scratch/rocm-tensor-parallel/inventory.json)

## Entry Point Status Checklist

| # | Entry Point Name | Category | Status | Resolution | Target Slice |
|---|------------------|----------|--------|------------|--------------|
| 1 | `ds4_gpu_add_xdev_tensor` | Cross-Device Accumulate | ✅ Implemented | — | Issue 04 |
| 2 | `ds4_gpu_attention_decode_rows_rope_tensor` | Attention | 🚫 Refused | proven unreachable (session-batch only) | Issue 05 |
| 3 | `ds4_gpu_attention_noncausal_raw_batch_heads_tensor` | Attention | 🚫 Refused | deliberately refused (DSpark-only) | Issue 07 |
| 4 | `ds4_gpu_attention_output_low_q4_K_slice_tensor` | Attention | 🚫 Refused | proven unreachable (dead for Q8_0 model) | Issue 05 |
| 5 | `ds4_gpu_attention_output_low_q8_rows_exact_tensor` | Attention | 🚫 Refused | proven unreachable (session-batch only) | Issue 05 |
| 6 | `ds4_gpu_attention_output_q4_K_batch_tensor` | Attention | 🚫 Refused | proven unreachable (dead for Q8_0 model) | Issue 07 |
| 7 | `ds4_gpu_attention_output_q8_tp_tensor` | Attention | ✅ Implemented | — | Issue 05 |
| 8 | `ds4_gpu_attention_prefill_raw_heads_range_tensor` | Attention | ✅ Implemented | scaffold evidence, max_abs_err=2.98e-8 | Issue 07 |
| 9 | `ds4_gpu_attention_prefill_static_mixed_heads_range_tensor` | Attention | ✅ Implemented | gated on E2E logits | Issue 07 |
| 10 | `ds4_gpu_device_cache_support_tensors` | Device Cache & Registration | 🚫 Refused | deliberately refused (DSpark-only) | Issue 08 |
| 11 | `ds4_gpu_device_cache_tensors` | Device Cache & Registration | ✅ Implemented | — | Issue 08 |
| 12 | `ds4_gpu_dspark_markov_argmax_tensor` | DSpark | 🚫 Refused | deliberately refused (DSpark itself out of scope) | Issue 08 |
| 13 | `ds4_gpu_hc_expand_add_tensor` | Attention | ✅ Implemented | — | Issue 05 |
| 14 | `ds4_gpu_indexer_top1_value_tensor` | Indexer & RoPE | 🚫 Refused | deliberately refused (off-by-default greedy shortcut) | Issue 08 |
| 15 | `ds4_gpu_kv_fp8_store_raw_decode_rows_tensor` | KV Store | 🚫 Refused | proven unreachable (session-batch only) | Issue 08 |
| 16 | `ds4_gpu_matmul_q8_0_kslice_hc_expand_add_tensor` | Matmul | 🚫 Refused | deliberately refused (off-by-default fusion) | Issue 05 |
| 17 | `ds4_gpu_matmul_q8_0_kslice_rows_tensor` | Matmul | ✅ Implemented | scaffold evidence, max_abs_err=3.5e-3 | Issue 07 |
| 18 | `ds4_gpu_matmul_q8_0_kslice_tensor` | Matmul | ✅ Implemented | delegation to kslice_rows (n_tokens==1) | Issue 08 |
| 19 | `ds4_gpu_matmul_q8_0_top1_tensor` | Matmul | 🚫 Refused | deliberately refused (paired with indexer_top1) | Issue 05 |
| 20 | `ds4_gpu_matmul_quant_kslice_tensor` | Matmul | 🚫 Refused | proven unreachable (dead for Q8_0 model) | Issue 05 |
| 21 | `ds4_gpu_moe_handoff_pack_tensor` | MoE Handoff | ✅ Implemented | scaffold evidence, byte-exact (max_abs_err=0) | Issue 08 |
| 22 | `ds4_gpu_register_model_map_no_copy` | Device Cache & Registration | ✅ Implemented | — | Issue 08 |
| 23 | `ds4_gpu_register_support_map` | Device Cache & Registration | 🚫 Refused | deliberately refused (DSpark-only) | Issue 08 |
| 24 | `ds4_gpu_rope_tail_decode_rows_tensor` | Indexer & RoPE | 🚫 Refused | proven unreachable (session-batch only) | Issue 08 |
| 25 | `ds4_gpu_routed_moe_batch_owned_tensor` | Routed MoE | ✅ Implemented | — | Issue 07 |
| 26 | `ds4_gpu_routed_moe_one_owned_tensor` | Routed MoE | ✅ Implemented | — | Issue 05 |
| 27 | `ds4_gpu_routed_moe_owned_packed_combine_tensor` | Routed MoE | 🚫 Refused | proven unreachable (`cuda_tp_ep_pack_exact` forced false on ROCm) | Issue 05 |
| 28 | `ds4_gpu_routed_moe_owned_slots_combine_rows_tensor` | Routed MoE | ✅ Implemented | — | Issue 07 |
| 29 | `ds4_gpu_routed_moe_owned_slots_combine_tensor` | Routed MoE | ✅ Implemented | — | Issue 05 |
| 30 | `ds4_gpu_shared_down_hc_expand_add_q8_0_tensor` | Shared Expert | 🚫 Refused | proven unreachable (requires `tp_world < 2`) | Issue 05 |
| 31 | `ds4_gpu_shared_down_hc_expand_owned_q8_0_tensor` | Shared Expert | 🚫 Refused | proven unreachable (`cuda_tp_ep_pack_exact` forced false on ROCm) | Issue 05 |
| 32 | `ds4_gpu_shared_mid_swiglu_q8_0_decode_exact_tensor` | Shared Expert | ✅ Implemented | — | Issue 05 |
| 33 | `ds4_gpu_tp_batch_gate_encode` | TP Gate Synchronisation | 🚫 Refused | proven unreachable (Metal-only gate protocol) | Issue 04 |
| 34 | `ds4_gpu_tp_big_gate_encode` | TP Gate Synchronisation | 🚫 Refused | proven unreachable (Metal-only gate protocol) | Issue 04 |
| 35 | `ds4_gpu_tp_big_gate_kick` | TP Gate Synchronisation | 🚫 Refused | proven unreachable (Metal-only gate protocol) | Issue 04 |
| 36 | `ds4_gpu_tp_big_gate_wait` | TP Gate Synchronisation | 🚫 Refused | proven unreachable (Metal-only gate protocol) | Issue 04 |
| 37 | `ds4_gpu_tp_gate_encode` | TP Gate Synchronisation | 🚫 Refused | proven unreachable (Metal-only gate protocol) | Issue 04 |

## Category Summary

- **Attention**: 9 entry points (4 implemented, 5 refused)
- **Routed MoE**: 5 entry points (4 implemented, 1 refused)
- **Matmul**: 5 entry points (2 implemented, 3 refused)
- **Shared Expert**: 3 entry points (1 implemented, 2 refused)
- **TP Gate Synchronisation**: 5 entry points (0 implemented, 5 refused -- Metal-only protocol, unreachable under ROCm's `--cuda-tensor-parallel`)
- **Cross-Device Accumulate**: 1 entry point (implemented)
- **DSpark**: 1 entry point (refused -- out of scope per PRD)
- **Indexer & RoPE**: 2 entry points (0 implemented, 2 refused)
- **KV Store**: 1 entry point (refused -- session-batch only)
- **MoE Handoff**: 1 entry point (implemented)
- **Device Cache & Registration**: 4 entry points (2 implemented, 2 refused -- DSpark-only)

## Notes on issue 08 (auxiliary TP hooks)

Issue 08's originally-tagged 9 entry points (Device Cache & Registration ×4, DSpark ×1,
Indexer & RoPE ×2, KV Store ×1, MoE Handoff ×1) are all resolved: 3 implemented
(`ds4_gpu_device_cache_tensors`, `ds4_gpu_register_model_map_no_copy` were already real;
`ds4_gpu_moe_handoff_pack_tensor` ported this slice), 6 deliberately refused or proven
unreachable with an explanatory comment at each stub site.

Issue 04's Comments section explicitly traced the five "TP Gate Synchronisation" entries and
found their "Issue 04" tag was a naming-pattern guess that didn't hold up — all five are
structurally unreachable under ROCm's `--cuda-tensor-parallel` (they belong to the Metal
two-machine gate protocol, gated on `g->tp_world == 2` which hard-refuses off-Metal) — and
recommended issue 08 document them as no-op placeholders. Done as part of this slice.

The remaining refused/unreachable entries outside issue 08's own tag (Attention, Matmul,
Shared Expert, Routed MoE categories, owned by issues 05/07) already carried their own
reachability comments from those slices; one gap
(`ds4_gpu_routed_moe_owned_packed_combine_tensor`, which had inherited a stale comment
describing a different, since-implemented entry point) was corrected as part of this
inventory refresh so every stubbed entry in `ds4_rocm_unavailable.cu`/`ds4_rocm.cu` now
carries its own accurate rationale.
