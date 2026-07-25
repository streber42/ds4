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

/* Real per-device selective weight caching for multi-GPU ROCm sessions.
 * Allocates a per-device VRAM slab for device_id, copies selective weight
 * ranges from host mmap to device_id's slab, and populates g_cache_ranges. */
extern "C" int ds4_gpu_device_cache_tensors(int device_id,
                                            const ds4_tensor_range *ranges,
                                            int n_ranges) {
    if (device_id < 0 || device_id >= DS4_MAX_GPUS) return 1;
    if (n_ranges < 0 || (!ranges && n_ranges > 0)) return 2;
    if (n_ranges == 0) return 0;

    if (!g_model_host_base || g_model_registered_size == 0) return 3;

    uint64_t want_bytes = 0;
    for (int i = 0; i < n_ranges; i++) {
        if (ranges[i].target_device != device_id) continue;
        const uint64_t off = ranges[i].source_offset;
        const uint64_t nb  = ranges[i].bytes;
        if (nb == 0) continue;
        if (off > g_model_registered_size) return 8;
        if (nb > g_model_registered_size - off) return 9;
        if (want_bytes > UINT64_MAX - nb) return 10;
        want_bytes += nb;
    }
    if (want_bytes == 0) return 0;

    rocm_device_cache &c = g_dev_cache[device_id];

    int prev_device = -1;
    if (hipGetDevice(&prev_device) != hipSuccess) prev_device = -1;
    if (hipSetDevice(device_id) != hipSuccess) return 4;

    void *new_base = NULL;
    size_t new_bytes = c.bytes + want_bytes;

    {
        size_t free_b = 0, total_b = 0;
        if (hipMemGetInfo(&free_b, &total_b) == hipSuccess) {
            const size_t safety = (size_t)2ull * 1024ull * 1024ull * 1024ull;
            const size_t need = new_bytes + safety;
            if (need > free_b) {
                fprintf(stderr,
                        "ds4: ROCm device cache slab needs %.2f GiB on device %d "
                        "but only %.2f GiB free (slab=%.2f GiB + %.2f GiB safety). "
                        "Refusing upfront to avoid late OOM at hipMalloc.\n",
                        (double)need / 1073741824.0,
                        device_id,
                        (double)free_b / 1073741824.0,
                        (double)new_bytes / 1073741824.0,
                        (double)safety / 1073741824.0);
                if (prev_device >= 0) (void)hipSetDevice(prev_device);
                return 5;
            }
        }
    }

    if (hipMalloc(&new_base, new_bytes) != hipSuccess) {
        if (prev_device >= 0) (void)hipSetDevice(prev_device);
        return 5;
    }
    if (c.present && c.bytes > 0) {
        hipError_t e = hipMemcpy(new_base, c.base, c.bytes, hipMemcpyDeviceToDevice);
        if (e != hipSuccess) {
            (void)hipFree(new_base);
            if (prev_device >= 0) (void)hipSetDevice(prev_device);
            return 6;
        }
        char *old_base = (char *)c.base;
        char *grown    = (char *)new_base;
        for (size_t k = 0; k < g_cache_ranges.size(); k++) {
            if (g_cache_ranges[k].device_id == device_id) {
                g_cache_ranges[k].device_ptr =
                    grown + ((char *)g_cache_ranges[k].device_ptr - old_base);
            }
        }
        (void)hipFree(c.base);
    }
    c.base = new_base;
    c.bytes = new_bytes;
    c.present = 1;

    const char *host_base = (const char *)g_model_host_base;
    size_t write_off = c.bytes - want_bytes;
    for (int i = 0; i < n_ranges; i++) {
        if (ranges[i].target_device != device_id) continue;
        char *dev_ptr = (char *)c.base + write_off;
        hipError_t e = hipMemcpy(dev_ptr,
                                 host_base + ranges[i].source_offset,
                                 (size_t)ranges[i].bytes,
                                 hipMemcpyHostToDevice);
        if (e != hipSuccess) {
            if (prev_device >= 0) (void)hipSetDevice(prev_device);
            return 7;
        }
        cache_range_entry ent;
        ent.source_offset = ranges[i].source_offset;
        ent.bytes         = ranges[i].bytes;
        ent.device_id     = device_id;
        ent.device_ptr    = dev_ptr;
        g_cache_ranges.push_back(ent);
        write_off += ranges[i].bytes;
    }

    std::sort(g_cache_ranges.begin(), g_cache_ranges.end(),
              [](const cache_range_entry &a, const cache_range_entry &b) {
                  if (a.source_offset != b.source_offset)
                      return a.source_offset < b.source_offset;
                  return a.device_id < b.device_id;
              });

    if (prev_device >= 0) (void)hipSetDevice(prev_device);
    return 0;
}

/* Tensor-parallel GPU compute is not yet ported to ROCm (see
 * .scratch/rocm-tensor-parallel/PRD.md). These entry points fail loudly and
 * name themselves by default via ds4_rocm_tp_stub(); DS4_ROCM_TP_BRINGUP=1
 * is the sole opt-in escape hatch for plumbing bring-up. */
/* ds4_gpu_tp_gate_encode / _batch_gate_encode / _big_gate_encode below (plus
 * ds4_gpu_tp_big_gate_kick / _big_gate_wait in ds4_rocm_unavailable.cu) are
 * the "TP Gate Synchronisation" category (issue 08, auxiliary TP hooks --
 * per issue 04's trace, their listed "Issue 04" target was a naming-pattern
 * guess that didn't hold up). All nine ds4.c call sites for these five are
 * gated by g->tp_world == 2, set only inside ds4_engine_tp_bind, which hard-
 * refuses off-Metal ("tensor parallelism requires the Metal backend",
 * ds4.c). They belong to the Metal two-machine --tensor-parallel --role gate
 * protocol (ds4_tp.c), a different mechanism from --cuda-tensor-parallel's
 * single-process multi-GPU design. So under ROCm's --cuda-tensor-parallel
 * these five are structurally unreachable dead code -- the real ROCm
 * cross-device work happens through the already-real
 * ds4_gpu_tensor_copy_xdev / ds4_gpu_add_xdev_tensor primitives and the
 * cuda_tp_attn/cuda_tp_moe/cuda_tp_ep/cuda_tp_shared branches in ds4.c.
 * Deliberately left as no-op placeholders, not deferred. */
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

/* Single-row (n_tokens == 1) special case of ds4_gpu_matmul_q8_0_kslice_
 * rows_tensor (already real, issue 05/07): slices x to its owned K range and
 * delegates. Not reached from ds4.c directly -- its only caller is
 * ds4_gpu_matmul_quant_kslice_tensor's non-Q8_0 output-head fallback, itself
 * dead for this model (see that stub's comment in ds4_rocm_unavailable.cu)
 * -- but a two-line delegation to an already-proven kernel is zero
 * incremental risk, so it is ported for real rather than left stubbed.
 * Ported line-for-line from CUDA's ds4_gpu_matmul_q8_0_kslice_tensor
 * (ds4_cuda.cu). */
extern "C" int ds4_gpu_matmul_q8_0_kslice_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t full_in_dim, uint64_t k_off,
        uint64_t k_cnt, uint64_t out_dim, const ds4_gpu_tensor *x,
        uint64_t x_elem_off) {
    if (!x || x_elem_off > x->bytes / sizeof(float) ||
        k_cnt > x->bytes / sizeof(float) - x_elem_off) {
        return 0;
    }
    ds4_gpu_tensor x_slice = *x;
    x_slice.ptr = (char *)x->ptr + x_elem_off * sizeof(float);
    x_slice.bytes = k_cnt * sizeof(float);
    x_slice.owner = 0;
    return ds4_gpu_matmul_q8_0_kslice_rows_tensor(
            out, model_map, model_size, weight_offset,
            full_in_dim, out_dim, k_off, k_cnt, &x_slice, 1u);
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
