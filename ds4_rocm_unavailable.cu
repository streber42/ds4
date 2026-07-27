/* Tensor-parallel GPU compute has no ROCm implementation yet (see
 * .scratch/rocm-tensor-parallel/PRD.md). Keep the unimplemented surface in
 * one place. These C-linkage stubs intentionally accept any argument list
 * because they never inspect their arguments -- by default they fail loudly
 * and name themselves via ds4_rocm_tp_stub() rather than silently returning
 * a neutral value that would corrupt tensor-parallel output. */

#include <stdint.h>

#include "ds4_rocm_tp_bringup.h"

#define ROCM_UNAVAILABLE_INT(name) extern "C" int name(...) { return ds4_rocm_tp_stub(#name); }
/* Boolean-contract entry points (ds4.c checks the return as
 * `ok = fn(...)` / `... != 0`): bring-up must report success (1), not 0 --
 * see ds4_rocm_tp_stub_ok's comment in ds4_rocm_tp_bringup.h. This is nearly
 * every entry point in this file. */
#define ROCM_UNAVAILABLE_INT_OK(name) extern "C" int name(...) { return ds4_rocm_tp_stub_ok(#name); }

/* ds4_gpu_add_xdev_tensor is implemented in ds4_rocm_compat.cu using the
 * cross-device module (ds4_rocm_xdev.h) -- it is transport, not per-model
 * kernel math, so it is not gated behind bring-up mode. */
/* Only called from metal_graph_encode_attention_session_batch (ds4.c), the
 * multi-session continuous-batching decode path -- session batching for
 * ROCm is out of scope for this PRD (see PRD.md Out of Scope). The
 * single-stream TP decode path (issue 05) applies RoPE via the already-real,
 * non-TP-specific ds4_gpu_rope_tail_tensor instead. Same category as the
 * other "_rows_"-suffixed session-batch-only hooks
 * (ds4_gpu_kv_fp8_store_raw_decode_rows_tensor, ds4_gpu_rope_tail_decode_
 * rows_tensor below). Deliberately refused, not deferred. */
ROCM_UNAVAILABLE_INT_OK(ds4_gpu_attention_decode_rows_rope_tensor)
/* DSpark-only (single-node speculative-decode verification, noncausal
 * softmax over draft-window rows); silently ignored by the distributed
 * coordinator per the PRD problem statement, unreached by a plain 2-rank
 * TP prefill/decode session. Deferred to issue 08 (auxiliary TP hooks). */
ROCM_UNAVAILABLE_INT_OK(ds4_gpu_attention_noncausal_raw_batch_heads_tensor)
/* Both q4_K attention-output-low variants are only called when
 * layer->attn_output_a->type == DS4_TENSOR_Q4_K (ds4.c,
 * metal_graph_attention_output_dense_quant_low/_batch) -- DeepSeek-V4-Flash's
 * attn_output_a is Q8_0 (the "AProjQ8" quant in this project's target GGUF,
 * see PRD.md Out of Scope), so this branch is dead for the model this port
 * targets. */
ROCM_UNAVAILABLE_INT_OK(ds4_gpu_attention_output_low_q4_K_slice_tensor)
/* Session-batch only (metal_graph_encode_attn_post_session_batch, ds4.c,
 * count>=2 concurrent decode items) -- continuous/session batching is out
 * of scope for this PRD (see PRD.md Out of Scope), unreached by the
 * single-stream prefill/decode path issues 05/07 target. */
ROCM_UNAVAILABLE_INT_OK(ds4_gpu_attention_output_low_q8_rows_exact_tensor)
ROCM_UNAVAILABLE_INT_OK(ds4_gpu_attention_output_q4_K_batch_tensor)
/* device_cache_support_tensors uses an errno-style "0 = success" contract
 * (ds4.c checks `if (rc != 0) fail`) -- 0 already is the correct neutral
 * value here. It is DSpark-only (engine_install_dspark_support_cache
 * returns early when e->dspark is unset, ds4.c) so it is unreached by a
 * plain 2-rank TP decode session and keeps the plain stub. Its sibling
 * ds4_gpu_device_cache_tensors IS reached (engine_install_per_device_caches
 * runs for every multi-tier session) and has a real implementation in
 * ds4_rocm.cu next to the other TP entry points. */
ROCM_UNAVAILABLE_INT(ds4_gpu_device_cache_support_tensors)
/* DSpark-only (the decode-time Markov-bias-plus-argmax fast path over draft
 * logits, ds4.c: guarded by g->dspark_draft_tokens). DSpark speculative
 * decoding in distributed/TP mode is out of scope for this PRD (see
 * PRD.md Out of Scope; the distributed coordinator already silently ignores
 * DSpark per the PRD problem statement), so this is unreached by any TP
 * session regardless of rank count. Deliberately refused, not deferred. */
ROCM_UNAVAILABLE_INT_OK(ds4_gpu_dspark_markov_argmax_tensor)
/* ds4_gpu_indexer_top1_value_tensor has a real implementation in
 * rocm/ds4_rocm_indexer.cuh (issue 31: TP=4 output head).  Reached by the
 * TP=4 distributed decode sampling path (metal_graph_encode_output_head_
 * split_top1, ds4.c) -- each rank finds its local best (id, value) in its
 * V/4 shard, then a small all-gather picks the global winner. */
/* Only called from metal_graph_encode_qkv_session_batch (ds4.c), the
 * multi-session continuous-batching decode path -- session batching for
 * ROCm is out of scope for this PRD (see PRD.md Out of Scope). The
 * single-stream decode/prefill path this port targets stores KV via the
 * already-real, non-TP-specific ds4_gpu_kv_fp8_store_raw_tensor instead.
 * Deliberately refused, not deferred. */
ROCM_UNAVAILABLE_INT_OK(ds4_gpu_kv_fp8_store_raw_decode_rows_tensor)
/* fuse_tp_attn_out_hc / DS4_CUDA_TP_ATTN_OUT_HC_FUSE default off (ds4.c) --
 * an optional decode-path fusion of the TP attention-output projection with
 * the HC expand, not required for correctness. Deferred to issue 08. */
ROCM_UNAVAILABLE_INT_OK(ds4_gpu_matmul_q8_0_kslice_hc_expand_add_tensor)
ROCM_UNAVAILABLE_INT_OK(ds4_gpu_matmul_q8_0_top1_tensor)
/* Only called from metal_graph_matmul_dense_quant_kslice's non-Q8_0 output-
 * head fallback (ds4.c), itself only reached when out_a/out_b aren't both
 * Q8_0 -- dead for DeepSeek-V4-Flash's Q8_0 attn_output/output weights. */
ROCM_UNAVAILABLE_INT_OK(ds4_gpu_matmul_quant_kslice_tensor)
/* ds4_gpu_moe_handoff_pack_tensor has a real implementation in
 * rocm/ds4_rocm_moe_launch.cuh (issue 08: auxiliary TP hooks) -- reached
 * when DS4_CUDA_TP_MOE_PACK=1 (default off) selects the packed-handoff MoE
 * mode instead of the default three-copy handoff. */
/* ds4_gpu_register_model_map_no_copy has a real implementation in
 * ds4_rocm.cu (delegates to the already-real ds4_gpu_set_model_map) --
 * engine_install_per_device_caches calls it unconditionally for every
 * multi-tier session, TP included. ds4_gpu_register_support_map is
 * DSpark-only (see device_cache_support_tensors above) and stays a stub. */
ROCM_UNAVAILABLE_INT_OK(ds4_gpu_register_support_map)
/* Only called from metal_graph_encode_qkv_session_batch (ds4.c), the
 * multi-session continuous-batching decode path -- out of scope for this
 * PRD (see PRD.md Out of Scope). The single-stream decode/prefill path
 * applies RoPE via the already-real, non-TP-specific ds4_gpu_rope_tail_tensor
 * instead. Deliberately refused, not deferred. */
ROCM_UNAVAILABLE_INT_OK(ds4_gpu_rope_tail_decode_rows_tensor)
/* ds4_gpu_routed_moe_batch_owned_tensor (formerly stubbed here) has a real
 * implementation in ds4_rocm_moe_launch.cuh (issue 07: TP prefill-path
 * kernels) -- reached unconditionally for the prefill/batch routed-MoE TP
 * path (metal_graph_encode_mixed_routed_rows, ds4.c, cuda_tp_owned_batch_moe). */
/* Gated on g->cuda_tp_ep_pack_exact (ds4.c:21385), which
 * metal_graph_cuda_tp_ep_pack_exact_requested() forces false on ROCm builds
 * -- the packed-4-slot layout is a CUDA-only perf optimization over the
 * plain 6-slot owned combine (see ds4_gpu_routed_moe_owned_slots_combine_
 * tensor, already real, issue 05). Confirmed unreachable under ROCm. */
ROCM_UNAVAILABLE_INT_OK(ds4_gpu_routed_moe_owned_packed_combine_tensor)
/* Both are decode-only (metal_graph_encode_decode_layer_phase, ds4.c).
 * _owned_ needs cuda_tp_ep_fused_hc_reduce, which ds4.c ties to
 * cuda_tp_ep_pack_exact -- forced false for ROCm (see issue 05 handoff /
 * ds4_rocm_moe_launch.cuh's ds4_gpu_routed_moe_one_owned_tensor comment) --
 * confirmed unreachable. _add_ needs fuse_shared_down_hc, which requires
 * g->tp_world < 2 -- also unreachable under any TP session. */
ROCM_UNAVAILABLE_INT_OK(ds4_gpu_shared_down_hc_expand_add_q8_0_tensor)
ROCM_UNAVAILABLE_INT_OK(ds4_gpu_shared_down_hc_expand_owned_q8_0_tensor)

/* ds4_gpu_tp_big_gate_kick returns a sequence number (data, not a boolean
 * flag), so the plain 0-returning stub is left as its neutral value.
 * ds4_gpu_tp_big_gate_wait IS boolean (`... != 0` at its call site) so it
 * gets the _OK variant. Both are part of the Metal two-machine gate
 * protocol and unreachable under --cuda-tensor-parallel (see issue 04
 * Comments), but there is no reason to leave a known trap for later. */
extern "C" uint64_t ds4_gpu_tp_big_gate_kick(...) { return (uint64_t)ds4_rocm_tp_stub("ds4_gpu_tp_big_gate_kick"); }
ROCM_UNAVAILABLE_INT_OK(ds4_gpu_tp_big_gate_wait)
