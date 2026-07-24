#ifdef __HIP_PLATFORM_AMD__
#include "ds4_rocm.h"
#include <hipblaslt/hipblaslt.h>

#define FULL_WARP_MASK 0xFFFFFFFFFFFFFFFFULL
#define MASK_T uint64_t
#define DS4_GPU_BACKEND_NAME "ROCm"
#define DS4_GPU_LOG_PREFIX "ds4: ROCm "
#define DS4_GPU_BLAS_NAME "hipBLAS"
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cub/block/block_radix_sort.cuh>

#define FULL_WARP_MASK 0xFFFFFFFFu
#define MASK_T uint32_t
#define DS4_GPU_BACKEND_NAME "CUDA"
#define DS4_GPU_LOG_PREFIX "ds4: CUDA "
#define DS4_GPU_BLAS_NAME "cuBLAS"
#endif

#include <stdint.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <unordered_map>
#include <vector>

#include "ds4_gpu.h"
#include "ds4_gpu_mgpu.h"
#include "ds4_rocm_tp_bringup.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_QK_K 256
#define DS4_ROCM_UNUSED __attribute__((unused))

enum {
    /* attention_decode_mixed_kernel stores raw-window scores plus visible
     * compressed scores in shared memory.  The host routes larger unmasked
     * decode calls to the online attention kernel so this fixed buffer never
     * becomes an out-of-bounds write at long context. */
    DS4_ROCM_ATTENTION_SCORE_CAP = 8192u,
    DS4_ROCM_ATTENTION_RAW_SCORE_CAP = 256u
};

#ifndef DS4_GPU_TENSOR_DEFINED
#define DS4_GPU_TENSOR_DEFINED
struct ds4_gpu_tensor {
    void *ptr;
    uint64_t bytes;
    int owner;
    int device_id;
};
typedef struct ds4_gpu_tensor ds4_gpu_tensor;
#endif

typedef struct {
    uint8_t scales[CUDA_QK_K / 16];
    uint8_t qs[CUDA_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} cuda_block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[CUDA_QK_K / 2];
} cuda_block_q4_K;

typedef struct {
    float d;
    int8_t qs[CUDA_QK_K];
    int16_t bsums[CUDA_QK_K / 16];
} cuda_block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[CUDA_QK_K / 8];
} cuda_block_iq2_xxs;

#include "ds4_iq2_tables_cuda.inc"

#include "rocm/ds4_rocm_runtime.cuh"

#include "rocm/ds4_rocm_common.cuh"

#include "rocm/ds4_rocm_q8.cuh"

#include "rocm/ds4_rocm_norm_rope.cuh"

#include "rocm/ds4_rocm_fp8_kv.cuh"

#include "rocm/ds4_rocm_attention.cuh"

#include "rocm/ds4_rocm_hc.cuh"

#include "rocm/ds4_rocm_output.cuh"

#include "rocm/ds4_rocm_indexer.cuh"

#include "rocm/ds4_rocm_embedding_launch.cuh"

#include "rocm/ds4_rocm_matmul.cuh"

#include "rocm/ds4_rocm_fp8_kv_launch.cuh"

#include "rocm/ds4_rocm_compressor.cuh"

#include "rocm/ds4_rocm_attention_launch.cuh"

#include "rocm/ds4_rocm_shared_expert.cuh"

#include "rocm/ds4_rocm_misc_launch.cuh"
#include "rocm/ds4_rocm_router.cuh"

#include "rocm/ds4_rocm_moe.cuh"

#include "rocm/ds4_rocm_moe_launch.cuh"

#include "rocm/ds4_rocm_glm.cuh"

#include "rocm/ds4_rocm_hc_output_launch.cuh"

#include "rocm/ds4_rocm_current_api_compat.cuh"

/* engine_install_per_device_caches (ds4.c) calls this unconditionally for
 * every multi-tier session -- pipeline layer-split and TP alike -- as a
 * prerequisite before it builds per-device weight caches. It only needs
 * g_model_host_base pointed at the model mmap, which is exactly what the
 * already-real ds4_gpu_set_model_map does; "no copy" just means we must not
 * let that call eagerly stage a full-model device copy, which
 * ds4_gpu_set_model_map already doesn't do. */
extern "C" int ds4_gpu_register_model_map_no_copy(const void *model_map, uint64_t model_size) {
    return ds4_gpu_set_model_map(model_map, model_size);
}

/* Real per-device selective weight caching (the VRAM/perf optimization
 * engine_install_per_device_caches is building toward) is not ported yet.
 * Always-succeed no-op is correctness-safe rather than a bring-up-only
 * bypass: every weight read goes through cuda_model_range_ptr, which
 * already falls back to a host-mapped/uncached read when no device range
 * was cached for it (see ds4_rocm_tp_bringup.h's comment on this same
 * safety property for the other errno-contract entry point,
 * ds4_gpu_device_cache_support_tensors). Slower, not wrong -- performance
 * is out of scope for issue 05. */
extern "C" int ds4_gpu_device_cache_tensors(int device_id, const ds4_tensor_range *ranges, int n_ranges) {
    (void)device_id; (void)ranges; (void)n_ranges;
    return 0;
}

/* Tensor-parallel GPU compute is not yet ported to ROCm (see
 * .scratch/rocm-tensor-parallel/PRD.md). These entry points fail loudly and
 * name themselves by default via ds4_rocm_tp_stub(); DS4_ROCM_TP_BRINGUP=1
 * is the sole opt-in escape hatch for plumbing bring-up. */
extern "C" int ds4_gpu_tp_gate_encode(uint32_t layer, uint32_t gate) {
    (void)layer; (void)gate;
    return ds4_rocm_tp_stub_ok("ds4_gpu_tp_gate_encode");
}

extern "C" void ds4_gpu_tp_set_batch_exchange(ds4_gpu_tp_batch_exchange_fn fn) {
    (void)fn;
}

extern "C" void ds4_gpu_tp_suspend_expert_sharding(int suspend) {
    (void)suspend;
}

extern "C" void ds4_gpu_tp_keepalive_pause(int paused) {
    (void)paused;
}

extern "C" void ds4_gpu_tp_set_attn_head_split(int enabled) {
    (void)enabled;
}

extern "C" void ds4_gpu_model_residency_skip(int skip) {
    (void)skip;
}

extern "C" void ds4_gpu_tp_set_big_exchange(ds4_gpu_tp_big_exchange_fn fn) {
    (void)fn;
}

extern "C" int ds4_gpu_tp_big_gate_encode(uint32_t layer, uint32_t rows,
                                          const ds4_gpu_tensor *out_t,
                                          ds4_gpu_tensor *in_t,
                                          uint64_t bytes) {
    (void)layer; (void)rows; (void)out_t; (void)in_t; (void)bytes;
    return ds4_rocm_tp_stub_ok("ds4_gpu_tp_big_gate_encode");
}

extern "C" int ds4_gpu_tp_batch_gate_encode(uint32_t layer, uint32_t rows) {
    (void)layer; (void)rows;
    return ds4_rocm_tp_stub_ok("ds4_gpu_tp_batch_gate_encode");
}

extern "C" int ds4_gpu_matmul_q8_0_kslice_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t full_in_dim, uint64_t k_off,
        uint64_t k_cnt, uint64_t out_dim, const ds4_gpu_tensor *x,
        uint64_t x_elem_off) {
    (void)out; (void)model_map; (void)model_size; (void)weight_offset;
    (void)full_in_dim; (void)k_off; (void)k_cnt; (void)out_dim; (void)x;
    (void)x_elem_off;
    return ds4_rocm_tp_stub_ok("ds4_gpu_matmul_q8_0_kslice_tensor");
}

/* Group-sliced attention-output pair for 2-rank TP decode (n_tokens == 1
 * only): the low-rank A projection for this rank's owned head groups
 * [group0, group0+group_cnt), followed by the matching k-slice of the B
 * expand projection. Ported from the CUDA TP path (ds4_cuda.cu, same
 * function name) -- both halves reduce to already-real ROCm kernels, since
 * offsetting the A-weight pointer and heads view by group0 makes the owned
 * slice look like a complete (non-sliced) low-rank projection to
 * ds4_gpu_attention_output_low_q8_tensor. */
extern "C" int ds4_gpu_attention_output_q8_tp_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low, const void *model_map,
        uint64_t model_size, uint64_t out_a_offset, uint64_t out_b_offset,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups_total,
        uint32_t group0, uint32_t group_cnt, uint64_t out_dim,
        const ds4_gpu_tensor *heads) {
    if (!out || !low || !heads || !model_map ||
        group_dim == 0 || rank == 0 || n_groups_total == 0 ||
        group_cnt == 0 || group0 > n_groups_total ||
        group_cnt > n_groups_total - group0 || out_dim == 0) {
        return 0;
    }
    const uint64_t blocks_a = (group_dim + 31u) / 32u;
    const uint64_t row_a_bytes = blocks_a * 34u;
    const uint64_t low_dim_total = (uint64_t)n_groups_total * rank;
    const uint64_t k_off = (uint64_t)group0 * rank;
    const uint64_t k_cnt = (uint64_t)group_cnt * rank;
    if ((k_off % 32u) != 0 || (k_cnt % 32u) != 0) return 0;
    if (heads->bytes < (uint64_t)(group0 + group_cnt) * group_dim * sizeof(float) ||
        low->bytes < k_cnt * sizeof(float) ||
        out->bytes < out_dim * sizeof(float)) {
        return 0;
    }

    ds4_gpu_tensor heads_slice = *heads;
    heads_slice.ptr = (char *)heads->ptr + (uint64_t)group0 * group_dim * sizeof(float);
    heads_slice.bytes = (uint64_t)group_cnt * group_dim * sizeof(float);
    heads_slice.owner = 0;

    const uint64_t a_off = out_a_offset + (uint64_t)group0 * rank * row_a_bytes;
    return ds4_gpu_attention_output_low_q8_tensor(low,
                                                  model_map,
                                                  model_size,
                                                  a_off,
                                                  group_dim,
                                                  rank,
                                                  group_cnt,
                                                  &heads_slice) &&
           ds4_gpu_matmul_q8_0_kslice_rows_tensor(out,
                                             model_map,
                                             model_size,
                                             out_b_offset,
                                             low_dim_total,
                                             out_dim,
                                             k_off,
                                             k_cnt,
                                             low,
                                             1);
}

/* Real implementation lives in rocm/ds4_rocm_hc_output_launch.cuh (included
 * above), next to the sibling ds4_gpu_hc_expand_tensor it was ported
 * alongside -- see that file for the port note. */
