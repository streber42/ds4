# ROCm Tensor Parallel Entry Point Inventory

Status: `in-progress` (0 / 37 implemented)

This inventory enumerates all 37 GPU entry points for the ROCm tensor-parallelism port.
It serves as the machine-checkable and human-readable progress tracker and the guard against quietly forgetting any required stub.

Machine-checkable representation: [inventory.json](file:///home/murphy/src/ds4-rebase/.scratch/rocm-tensor-parallel/inventory.json)

## Entry Point Status Checklist

| # | Entry Point Name | Category | Status | Target Slice |
|---|------------------|----------|--------|--------------|
| 1 | `ds4_gpu_add_xdev_tensor` | Cross-Device Accumulate | ❌ Stubbed | Issue 01 / Issue 08 |
| 2 | `ds4_gpu_attention_decode_rows_rope_tensor` | Attention | ❌ Stubbed | Issue 05 |
| 3 | `ds4_gpu_attention_noncausal_raw_batch_heads_tensor` | Attention | ❌ Stubbed | Issue 07 |
| 4 | `ds4_gpu_attention_output_low_q4_K_slice_tensor` | Attention | ❌ Stubbed | Issue 05 |
| 5 | `ds4_gpu_attention_output_low_q8_rows_exact_tensor` | Attention | ❌ Stubbed | Issue 05 |
| 6 | `ds4_gpu_attention_output_q4_K_batch_tensor` | Attention | ❌ Stubbed | Issue 07 |
| 7 | `ds4_gpu_attention_output_q8_tp_tensor` | Attention | ❌ Stubbed | Issue 05 |
| 8 | `ds4_gpu_attention_prefill_raw_heads_range_tensor` | Attention | ❌ Stubbed | Issue 07 |
| 9 | `ds4_gpu_attention_prefill_static_mixed_heads_range_tensor` | Attention | ❌ Stubbed | Issue 07 |
| 10 | `ds4_gpu_device_cache_support_tensors` | Device Cache & Registration | ❌ Stubbed | Issue 08 |
| 11 | `ds4_gpu_device_cache_tensors` | Device Cache & Registration | ❌ Stubbed | Issue 08 |
| 12 | `ds4_gpu_dspark_markov_argmax_tensor` | DSpark | ❌ Stubbed | Issue 08 |
| 13 | `ds4_gpu_hc_expand_add_tensor` | Attention | ❌ Stubbed | Issue 05 |
| 14 | `ds4_gpu_indexer_top1_value_tensor` | Indexer & RoPE | ❌ Stubbed | Issue 08 |
| 15 | `ds4_gpu_kv_fp8_store_raw_decode_rows_tensor` | KV Store | ❌ Stubbed | Issue 08 |
| 16 | `ds4_gpu_matmul_q8_0_kslice_hc_expand_add_tensor` | Matmul | ❌ Stubbed | Issue 05 |
| 17 | `ds4_gpu_matmul_q8_0_kslice_rows_tensor` | Matmul | ❌ Stubbed | Issue 07 |
| 18 | `ds4_gpu_matmul_q8_0_kslice_tensor` | Matmul | ❌ Stubbed | Issue 05 |
| 19 | `ds4_gpu_matmul_q8_0_top1_tensor` | Matmul | ❌ Stubbed | Issue 05 |
| 20 | `ds4_gpu_matmul_quant_kslice_tensor` | Matmul | ❌ Stubbed | Issue 05 |
| 21 | `ds4_gpu_moe_handoff_pack_tensor` | MoE Handoff | ❌ Stubbed | Issue 08 |
| 22 | `ds4_gpu_register_model_map_no_copy` | Device Cache & Registration | ❌ Stubbed | Issue 08 |
| 23 | `ds4_gpu_register_support_map` | Device Cache & Registration | ❌ Stubbed | Issue 08 |
| 24 | `ds4_gpu_rope_tail_decode_rows_tensor` | Indexer & RoPE | ❌ Stubbed | Issue 08 |
| 25 | `ds4_gpu_routed_moe_batch_owned_tensor` | Routed MoE | ❌ Stubbed | Issue 07 |
| 26 | `ds4_gpu_routed_moe_one_owned_tensor` | Routed MoE | ❌ Stubbed | Issue 05 |
| 27 | `ds4_gpu_routed_moe_owned_packed_combine_tensor` | Routed MoE | ❌ Stubbed | Issue 05 |
| 28 | `ds4_gpu_routed_moe_owned_slots_combine_rows_tensor` | Routed MoE | ❌ Stubbed | Issue 07 |
| 29 | `ds4_gpu_routed_moe_owned_slots_combine_tensor` | Routed MoE | ❌ Stubbed | Issue 05 |
| 30 | `ds4_gpu_shared_down_hc_expand_add_q8_0_tensor` | Shared Expert | ❌ Stubbed | Issue 05 |
| 31 | `ds4_gpu_shared_down_hc_expand_owned_q8_0_tensor` | Shared Expert | ❌ Stubbed | Issue 05 |
| 32 | `ds4_gpu_shared_mid_swiglu_q8_0_decode_exact_tensor` | Shared Expert | ❌ Stubbed | Issue 05 |
| 33 | `ds4_gpu_tp_batch_gate_encode` | TP Gate Synchronisation | ❌ Stubbed | Issue 04 |
| 34 | `ds4_gpu_tp_big_gate_encode` | TP Gate Synchronisation | ❌ Stubbed | Issue 04 |
| 35 | `ds4_gpu_tp_big_gate_kick` | TP Gate Synchronisation | ❌ Stubbed | Issue 04 |
| 36 | `ds4_gpu_tp_big_gate_wait` | TP Gate Synchronisation | ❌ Stubbed | Issue 04 |
| 37 | `ds4_gpu_tp_gate_encode` | TP Gate Synchronisation | ❌ Stubbed | Issue 04 |

## Category Summary

- **Attention**: 9 entry points
- **Routed MoE**: 5 entry points
- **Matmul**: 5 entry points
- **Shared Expert**: 3 entry points
- **TP Gate Synchronisation**: 5 entry points
- **Cross-Device Accumulate**: 1 entry point
- **DSpark**: 1 entry point
- **Indexer & RoPE**: 2 entry points
- **KV Store**: 1 entry point
- **MoE Handoff**: 1 entry point
- **Device Cache & Registration**: 4 entry points
