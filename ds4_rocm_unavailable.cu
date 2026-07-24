/* Tensor-parallel GPU compute has no ROCm implementation yet (see
 * .scratch/rocm-tensor-parallel/PRD.md). Keep the unimplemented surface in
 * one place. These C-linkage stubs intentionally accept any argument list
 * because they never inspect their arguments -- by default they fail loudly
 * and name themselves via ds4_rocm_tp_stub() rather than silently returning
 * a neutral value that would corrupt tensor-parallel output. */

#include <stdint.h>

#include "ds4_rocm_tp_bringup.h"

#define ROCM_UNAVAILABLE_INT(name) extern "C" int name(...) { return ds4_rocm_tp_stub(#name); }

/* ds4_gpu_add_xdev_tensor is implemented in ds4_rocm_compat.cu using the
 * cross-device module (ds4_rocm_xdev.h) -- it is transport, not per-model
 * kernel math, so it is not gated behind bring-up mode. */
ROCM_UNAVAILABLE_INT(ds4_gpu_attention_decode_rows_rope_tensor)
ROCM_UNAVAILABLE_INT(ds4_gpu_attention_noncausal_raw_batch_heads_tensor)
ROCM_UNAVAILABLE_INT(ds4_gpu_attention_output_low_q4_K_slice_tensor)
ROCM_UNAVAILABLE_INT(ds4_gpu_attention_output_low_q8_rows_exact_tensor)
ROCM_UNAVAILABLE_INT(ds4_gpu_attention_output_q4_K_batch_tensor)
ROCM_UNAVAILABLE_INT(ds4_gpu_attention_prefill_raw_heads_range_tensor)
ROCM_UNAVAILABLE_INT(ds4_gpu_attention_prefill_static_mixed_heads_range_tensor)
ROCM_UNAVAILABLE_INT(ds4_gpu_device_cache_support_tensors)
ROCM_UNAVAILABLE_INT(ds4_gpu_device_cache_tensors)
ROCM_UNAVAILABLE_INT(ds4_gpu_dspark_markov_argmax_tensor)
ROCM_UNAVAILABLE_INT(ds4_gpu_indexer_top1_value_tensor)
ROCM_UNAVAILABLE_INT(ds4_gpu_kv_fp8_store_raw_decode_rows_tensor)
ROCM_UNAVAILABLE_INT(ds4_gpu_matmul_q8_0_kslice_hc_expand_add_tensor)
ROCM_UNAVAILABLE_INT(ds4_gpu_matmul_q8_0_kslice_rows_tensor)
ROCM_UNAVAILABLE_INT(ds4_gpu_matmul_q8_0_top1_tensor)
ROCM_UNAVAILABLE_INT(ds4_gpu_matmul_quant_kslice_tensor)
ROCM_UNAVAILABLE_INT(ds4_gpu_moe_handoff_pack_tensor)
ROCM_UNAVAILABLE_INT(ds4_gpu_register_model_map_no_copy)
ROCM_UNAVAILABLE_INT(ds4_gpu_register_support_map)
ROCM_UNAVAILABLE_INT(ds4_gpu_rope_tail_decode_rows_tensor)
ROCM_UNAVAILABLE_INT(ds4_gpu_routed_moe_batch_owned_tensor)
ROCM_UNAVAILABLE_INT(ds4_gpu_routed_moe_one_owned_tensor)
ROCM_UNAVAILABLE_INT(ds4_gpu_routed_moe_owned_packed_combine_tensor)
ROCM_UNAVAILABLE_INT(ds4_gpu_routed_moe_owned_slots_combine_rows_tensor)
ROCM_UNAVAILABLE_INT(ds4_gpu_routed_moe_owned_slots_combine_tensor)
ROCM_UNAVAILABLE_INT(ds4_gpu_shared_down_hc_expand_add_q8_0_tensor)
ROCM_UNAVAILABLE_INT(ds4_gpu_shared_down_hc_expand_owned_q8_0_tensor)
ROCM_UNAVAILABLE_INT(ds4_gpu_shared_mid_swiglu_q8_0_decode_exact_tensor)

extern "C" uint64_t ds4_gpu_tp_big_gate_kick(...) { return (uint64_t)ds4_rocm_tp_stub("ds4_gpu_tp_big_gate_kick"); }
ROCM_UNAVAILABLE_INT(ds4_gpu_tp_big_gate_wait)
