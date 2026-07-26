#include <hip/hip_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_rocm_xdev.h"
#include "ds4_gpu_mgpu.h"
#include "ds4_gpu.h"
#include "ds4_gpu_args.h"

ds4_gpu_ctx g_gpu[DS4_MAX_GPUS] = {};
int g_n_gpus = 1;
int g_gpu_peer_ok[DS4_MAX_GPUS][DS4_MAX_GPUS] = {{1}};

/* Defined in rocm/ds4_rocm_runtime.cuh (compiled into ds4_rocm.o): swaps the
 * process-wide cuBLAS/hipBLASLt handles to the ones matching `tier`,
 * creating them lazily on first use. Must be called after hipSetDevice
 * whenever the active tier changes -- see ds4_gpu_set_current_device below
 * and its comment in ds4_rocm_runtime.cuh. */
extern "C" void ds4_rocm_activate_tier_blas(int tier);

static int rocm_tier_valid(int tier) {
    return tier >= 0 && tier < g_n_gpus;
}

/* Logical tier -> physical HIP device id. Every cross-device call in this
 * file goes through this so ds4_rocm_xdev.h (which speaks physical device
 * ids, see ds4_rocm_xdev.cu) never has to know about tiers, and callers
 * here never have to know that a tier and its physical id can differ
 * (e.g. --gpu-devices 2,3). */
static int rocm_tier_device(int tier) {
    return rocm_tier_valid(tier) ? g_gpu[tier].device_id : g_gpu[0].device_id;
}

extern "C" int ds4_gpu_init_multi(const ds4_gpu_config *cfg) {
    if (!cfg || cfg->n_gpus < 1 || cfg->n_gpus > DS4_MAX_GPUS) {
        fprintf(stderr, "ds4: ROCm: invalid GPU config (n_gpus=%d)\n",
                cfg ? cfg->n_gpus : -1);
        return 0;
    }

    int dev_ids[DS4_MAX_GPUS];
    for (int i = 0; i < cfg->n_gpus; i++) {
        ds4_gpu_ctx *c = &g_gpu[i];
        c->device_id = cfg->device_indices[i];
        if (c->device_id < 0) return 0;
        dev_ids[i] = c->device_id;
        /* Publish incrementally, same discipline as the CUDA backend: a
         * later failure still lets tier-aware cleanup/logging see how far
         * init got. */
        g_n_gpus = i + 1;
        if (hipSetDevice(c->device_id) != hipSuccess) {
            fprintf(stderr, "ds4: ROCm: hipSetDevice(%d) (tier %d) failed\n",
                    c->device_id, i);
            return 0;
        }
        hipDeviceProp_t prop;
        if (hipGetDeviceProperties(&prop, c->device_id) == hipSuccess) {
            fprintf(stderr,
                    "ds4: ROCm backend initialized on %s dev=%d (tier %d)\n",
                    prop.name, c->device_id, i);
        }
        c->budget_bytes = cfg->vram_bytes[i];
        c->used_bytes = 0;
        c->scratch = NULL;
        c->scratch_bytes = 0;
    }

    /* Establish the peer mesh once for the whole tier set through the
     * cross-device module -- this is the only place peer access is
     * requested; no kernel or engine code touches hipDeviceEnablePeerAccess
     * directly. A single tier is trivially self-peer via the {{1}}
     * initializer above, so the mesh is only needed once tp/mgpu actually
     * spans more than one device. */
    if (g_n_gpus > 1) {
        if (!ds4_rocm_xdev_init_global_mesh(dev_ids, g_n_gpus)) {
            fprintf(stderr,
                    "ds4: ROCm: cross-device mesh init failed for %d GPUs\n",
                    g_n_gpus);
            return 0;
        }
        ds4_rocm_xdev_mesh *mesh = ds4_rocm_xdev_get_global_mesh();
        for (int i = 0; i < g_n_gpus; i++) {
            for (int j = 0; j < g_n_gpus; j++) {
                g_gpu_peer_ok[i][j] = mesh->peer_ok[i][j];
            }
            if (i > 0) {
                fprintf(stderr, "ds4: ROCm peer access tier 0<->%d: %s\n", i,
                        (mesh->peer_ok[0][i] && mesh->peer_ok[i][0]) ?
                            "direct" : "host-staging fallback");
            }
        }
    }

    if (hipSetDevice(g_gpu[0].device_id) != hipSuccess) return 0;
    return ds4_gpu_init();
}

extern "C" int ds4_gpu_set_current_device(int tier) {
    if (!rocm_tier_valid(tier)) return 1;
    if (hipSetDevice(rocm_tier_device(tier)) != hipSuccess) return 1;
    ds4_rocm_activate_tier_blas(tier);
    return 0;
}

extern "C" int ds4_gpu_set_current_device_fenced(int tier) {
    return ds4_gpu_set_current_device(tier);
}

extern "C" int ds4_gpu_tensor_alloc_on(ds4_gpu_tensor *t, int tier,
                                         uint64_t bytes) {
    if (!t) return 1;
    if (!rocm_tier_valid(tier)) return 2;
    if (bytes == 0) bytes = 1;
    if (hipSetDevice(rocm_tier_device(tier)) != hipSuccess) return 2;
    if (hipMalloc(&t->ptr, (size_t)bytes) != hipSuccess) return 3;
    t->bytes = bytes;
    t->owner = 1;
    t->device_id = tier;
    g_gpu[tier].used_bytes += bytes;
    return 0;
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_ptr_on(int tier,
                                                         uint64_t bytes) {
    if (!rocm_tier_valid(tier)) return NULL;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    if (ds4_gpu_tensor_alloc_on(t, tier, bytes) != 0) {
        free(t);
        return NULL;
    }
    return t;
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed_on(int tier,
                                                             uint64_t bytes) {
    if (!rocm_tier_valid(tier)) return NULL;
    if (tier == 0) return ds4_gpu_tensor_alloc_managed(bytes);
    /* HIP managed memory is not device-bound (pages on first touch), but
     * stamping the home tier keeps free-time accounting and the device
     * used for the allocation call itself consistent with the CUDA
     * backend's convention (see ds4_gpu_tensor_alloc_managed_on in
     * ds4_cuda.cu). */
    if (bytes == 0) bytes = 1;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    if (hipSetDevice(rocm_tier_device(tier)) != hipSuccess ||
        hipMallocManaged(&t->ptr, (size_t)bytes) != hipSuccess) {
        free(t);
        return NULL;
    }
    t->bytes = bytes;
    t->owner = 1;
    t->device_id = tier;
    return t;
}

extern "C" void ds4_gpu_tensor_free_in_place(ds4_gpu_tensor *t) {
    if (!t) return;
    if (t->owner && t->ptr) {
        (void)hipSetDevice(rocm_tier_device(t->device_id));
        (void)hipFree(t->ptr);
    }
    memset(t, 0, sizeof(*t));
}

extern "C" int ds4_gpu_tensor_device(const ds4_gpu_tensor *t) {
    return t ? t->device_id : -1;
}

extern "C" int ds4_gpu_tensor_copy_async(ds4_gpu_tensor *dst,
                                           const ds4_gpu_tensor *src,
                                           uint64_t bytes) {
    if (!dst || !src || bytes > dst->bytes || bytes > src->bytes) return 0;
    if (bytes == 0) return 1;
    return hipMemcpyAsync(dst->ptr, src->ptr, (size_t)bytes,
                          hipMemcpyDeviceToDevice, 0) == hipSuccess;
}

extern "C" int ds4_gpu_tensor_copy_xdev(ds4_gpu_tensor *dst,
                                          const ds4_gpu_tensor *src,
                                          uint64_t bytes) {
    if (!dst || !src) return 0;
    ds4_rocm_xdev_mesh *mesh = ds4_rocm_xdev_get_global_mesh();
    int dst_dev = rocm_tier_device(ds4_gpu_tensor_device(dst));
    int src_dev = rocm_tier_device(ds4_gpu_tensor_device(src));
    return ds4_rocm_xdev_copy(mesh, dst_dev, dst->ptr, src_dev, src->ptr, (size_t)bytes, 0);
}

extern "C" int ds4_gpu_tensor_copy_xdev_default(ds4_gpu_tensor *dst,
                                                  const ds4_gpu_tensor *src,
                                                  uint64_t bytes) {
    return ds4_gpu_tensor_copy_xdev(dst, src, bytes);
}

extern "C" int ds4_gpu_tensor_copy_xdev_ordered(ds4_gpu_tensor *dst,
                                                  const ds4_gpu_tensor *src,
                                                  uint64_t bytes) {
    return ds4_gpu_tensor_copy_xdev(dst, src, bytes);
}

extern "C" int ds4_gpu_tensor_copy_xdev3(
        ds4_gpu_tensor *dst0, const ds4_gpu_tensor *src0, uint64_t bytes0,
        ds4_gpu_tensor *dst1, const ds4_gpu_tensor *src1, uint64_t bytes1,
        ds4_gpu_tensor *dst2, const ds4_gpu_tensor *src2, uint64_t bytes2) {
    return (bytes0 == 0 || ds4_gpu_tensor_copy_xdev(dst0, src0, bytes0)) &&
           (bytes1 == 0 || ds4_gpu_tensor_copy_xdev(dst1, src1, bytes1)) &&
           (bytes2 == 0 || ds4_gpu_tensor_copy_xdev(dst2, src2, bytes2));
}

extern "C" int ds4_gpu_tensor_copy_xdev3_default_dst(
        ds4_gpu_tensor *dst0, const ds4_gpu_tensor *src0, uint64_t bytes0,
        ds4_gpu_tensor *dst1, const ds4_gpu_tensor *src1, uint64_t bytes1,
        ds4_gpu_tensor *dst2, const ds4_gpu_tensor *src2, uint64_t bytes2) {
    return ds4_gpu_tensor_copy_xdev3(dst0, src0, bytes0, dst1, src1, bytes1,
                                     dst2, src2, bytes2);
}

extern "C" int ds4_gpu_tensor_wait_xdev(const ds4_gpu_tensor *src,
                                          int dst_tier) {
    /* dst_tier is about to read src via direct peer access rather than an
     * explicit xdev copy; src's writer may still have kernels in flight on
     * its own default stream, which has no implicit ordering against
     * dst_tier's queue. A device-wide hipDeviceSynchronize() is not a
     * reliable fence on gfx1201's hardware scheduler (see the peer-copy
     * comment in ds4_rocm_xdev.cu), so this routes through the same
     * event-based happens-before edge ds4_rocm_xdev_copy uses. */
    if (!src || !rocm_tier_valid(dst_tier)) return 0;
    const int src_dev = rocm_tier_device(ds4_gpu_tensor_device(src));
    const int dst_dev = rocm_tier_device(dst_tier);
    return ds4_rocm_xdev_wait_producer(dst_dev, src_dev);
}

/* Cross-device accumulate: out = local + remote, where remote may live on
 * a different tier than out/local. Every byte that crosses a device
 * boundary goes through ds4_rocm_xdev_copy/accumulate_f32 (ds4_rocm_xdev.h)
 * -- this function does no peer-transfer API calls of its own. Mirrors the
 * CUDA backend's ds4_gpu_add_xdev_tensor (ds4_cuda.cu) semantics: out and
 * local must already share a tier; remote_tmp is required (and must live
 * on out's tier) whenever remote is on a different tier. */
extern "C" int ds4_gpu_add_xdev_tensor(ds4_gpu_tensor *out,
                                        const ds4_gpu_tensor *local,
                                        const ds4_gpu_tensor *remote,
                                        ds4_gpu_tensor *remote_tmp,
                                        uint32_t n) {
    if (!out || !local || !remote ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        local->bytes < (uint64_t)n * sizeof(float) ||
        remote->bytes < (uint64_t)n * sizeof(float)) {
        return 0;
    }
    if (n == 0) return 1;

    const int out_tier = ds4_gpu_tensor_device(out);
    const int local_tier = ds4_gpu_tensor_device(local);
    if (!rocm_tier_valid(out_tier) || out_tier != local_tier) return 0;

    if (out->ptr != local->ptr &&
        !ds4_gpu_tensor_copy_xdev(out, local, (uint64_t)n * sizeof(float))) {
        return 0;
    }

    const ds4_gpu_tensor *rhs = remote;
    const int remote_tier = ds4_gpu_tensor_device(remote);
    if (remote_tier != out_tier) {
        if (!remote_tmp ||
            remote_tmp->bytes < (uint64_t)n * sizeof(float) ||
            ds4_gpu_tensor_device(remote_tmp) != out_tier) {
            return 0;
        }
        if (!ds4_gpu_tensor_copy_xdev(remote_tmp, remote,
                                      (uint64_t)n * sizeof(float))) {
            return 0;
        }
        rhs = remote_tmp;
    }

    const int dev = rocm_tier_device(out_tier);
    ds4_rocm_xdev_mesh *mesh = ds4_rocm_xdev_get_global_mesh();
    if (hipSetDevice(dev) != hipSuccess) return 0;
    return ds4_rocm_xdev_accumulate_f32(mesh, dev, (float *)out->ptr,
                                        dev, (const float *)rhs->ptr,
                                        n, 0) != 0;
}

extern "C" int ds4_gpu_tensor_wait_xdev_default(const ds4_gpu_tensor *src,
                                                  int dst_tier) {
    return ds4_gpu_tensor_wait_xdev(src, dst_tier);
}

extern "C" uint64_t ds4_gpu_tier_free_vram(int tier) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (!rocm_tier_valid(tier) ||
        hipSetDevice(rocm_tier_device(tier)) != hipSuccess ||
        hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) {
        return 0;
    }
    return (uint64_t)free_bytes;
}

extern "C" int ds4_gpu_args_probe_auto_cuda(
        const int *device_filter, int filter_len, ds4_gpu_config *out,
        size_t safety_margin_bytes, char *errbuf, size_t errbuflen) {
    if (!out) {
        if (errbuf && errbuflen) snprintf(errbuf, errbuflen, "internal: NULL out");
        return 1;
    }
    int visible = 0;
    hipError_t rc = hipGetDeviceCount(&visible);
    if (rc != hipSuccess || visible <= 0) {
        if (errbuf && errbuflen) {
            snprintf(errbuf, errbuflen, "hipGetDeviceCount failed: %s",
                     rc == hipSuccess ? "no devices" : hipGetErrorString(rc));
        }
        return 1;
    }
    if (filter_len > DS4_MAX_GPUS) {
        if (errbuf && errbuflen) {
            snprintf(errbuf, errbuflen,
                     "--gpu-devices filter has %d entries (max %d)",
                     filter_len, DS4_MAX_GPUS);
        }
        return 1;
    }
    int devs[DS4_MAX_GPUS];
    int n_dev = 0;
    if (device_filter && filter_len > 0) {
        for (int i = 0; i < filter_len; i++) {
            int d = device_filter[i];
            if (d < 0 || d >= visible) {
                if (errbuf && errbuflen) {
                    snprintf(errbuf, errbuflen,
                             "--gpu-devices: device %d not in 0..%d",
                             d, visible - 1);
                }
                return 1;
            }
            devs[n_dev++] = d;
        }
    } else {
        int cap = visible < DS4_MAX_GPUS ? visible : DS4_MAX_GPUS;
        for (int i = 0; i < cap; i++) devs[n_dev++] = i;
    }
    memset(out, 0, sizeof(*out));
    out->n_gpus = n_dev;
    out->safety_margin_bytes = safety_margin_bytes;
    for (int i = 0; i < n_dev; i++) {
        const int device = devs[i];
        if (hipSetDevice(device) != hipSuccess) {
            if (errbuf && errbuflen) {
                snprintf(errbuf, errbuflen, "hipSetDevice(%d) failed", device);
            }
            return 1;
        }
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        rc = hipMemGetInfo(&free_bytes, &total_bytes);
        if (rc != hipSuccess) {
            if (errbuf && errbuflen) {
                snprintf(errbuf, errbuflen,
                         "hipMemGetInfo(device %d) failed: %s",
                         device, hipGetErrorString(rc));
            }
            return 1;
        }
        /* Same reserve policy as the CUDA auto-probe (ds4_cuda.cu): max(2
         * GiB, 5% of free). Explicit --gpu-vram budgets bypass this path
         * entirely and are unaffected. */
        const size_t reserve_floor = (size_t)2ull * 1024ull * 1024ull * 1024ull;
        const size_t reserve_pct = free_bytes / 20u;
        const size_t reserve = reserve_floor > reserve_pct ? reserve_floor : reserve_pct;
        out->device_indices[i] = device;
        out->vram_bytes[i] = free_bytes > reserve ? free_bytes - reserve : 0;
    }
    return 0;
}

extern "C" void ds4_gpu_enable_q8_dequant_gemm(void) {
}

static int g_rocm_q8_cache_suppressed = 0;

extern "C" int ds4_gpu_q8_cache_suppressed(void) {
    return g_rocm_q8_cache_suppressed;
}

extern "C" void ds4_gpu_set_q8_cache_suppressed(int suppressed) {
    g_rocm_q8_cache_suppressed = suppressed != 0;
}

extern "C" int ds4_gpu_set_decode_fast_attention(int enabled) {
    (void)enabled;
    return 0;
}

extern "C" int ds4_gpu_set_decode_score_vec4(int enabled) {
    (void)enabled;
    return 0;
}

extern "C" int ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint32_t n_rows) {
    return ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size,
                                      weight_offset, in_dim, out_dim, x,
                                      n_rows);
}

extern "C" int ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor(
        ds4_gpu_tensor *out0, ds4_gpu_tensor *out1, const void *model_map,
        uint64_t model_size, uint64_t weight0_offset,
        uint64_t weight1_offset, uint64_t in_dim, uint64_t out0_dim,
        uint64_t out1_dim, const ds4_gpu_tensor *x, uint32_t n_rows) {
    return ds4_gpu_matmul_q8_0_pair_tensor(
            out0, out1, model_map, model_size, weight0_offset, weight1_offset,
            in_dim, out0_dim, out1_dim, x, n_rows);
}

extern "C" int ds4_gpu_matmul_f16_router_rows_exact_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, const ds4_gpu_tensor *x, uint32_t n_rows) {
    return ds4_gpu_matmul_f16_tensor(out, model_map, model_size, weight_offset,
                                     4096u, 256u, x, n_rows);
}

extern "C" int ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
        ds4_gpu_tensor *q_out, const ds4_gpu_tensor *q,
        const void *model_map, uint64_t model_size,
        uint64_t q_weight_offset, uint32_t q_n,
        ds4_gpu_tensor *kv_out, const ds4_gpu_tensor *kv,
        uint64_t kv_weight_offset, uint32_t kv_n, uint32_t rows,
        uint32_t kv_n_head, uint32_t kv_head_dim, uint32_t n_rot,
        uint32_t pos0, uint32_t n_ctx_orig, bool inverse,
        float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow, float eps) {
    return ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
                   q_out, q, model_map, model_size, q_weight_offset, q_n,
                   kv_out, kv, kv_weight_offset, kv_n, rows, eps) != 0 &&
           ds4_gpu_rope_tail_tensor(
                   kv_out, rows, kv_n_head, kv_head_dim, n_rot, pos0,
                   n_ctx_orig, inverse, freq_base, freq_scale, ext_factor,
                   attn_factor, beta_fast, beta_slow) != 0;
}

extern "C" int ds4_gpu_embed_token_quant_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t weight_type, uint32_t n_vocab,
        uint32_t token, uint32_t n_embd) {
    if (weight_type != 8u) return 0;
    return ds4_gpu_embed_token_q8_0_tensor(out, model_map, model_size,
                                           weight_offset, n_vocab, token,
                                           n_embd);
}

extern "C" int ds4_gpu_embed_tokens_quant_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *tokens,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t weight_type, uint32_t n_vocab, uint32_t n_tokens,
        uint32_t n_embd) {
    if (weight_type != 8u) return 0;
    return ds4_gpu_embed_tokens_q8_0_tensor(out, tokens, model_map, model_size,
                                            weight_offset, n_vocab, n_tokens,
                                            n_embd);
}

extern "C" int ds4_gpu_matmul_quant_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t weight_type, uint64_t in_dim,
        uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (weight_type == 8u) {
        return ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size,
                                          weight_offset, in_dim, out_dim, x,
                                          n_tok);
    }
    if (weight_type == 1u) {
        return ds4_gpu_matmul_f16_tensor(out, model_map, model_size,
                                         weight_offset, in_dim, out_dim, x,
                                         n_tok);
    }
    return 0;
}

extern "C" int ds4_gpu_matmul_quant_decode_mpp_model_view_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t weight_type, uint64_t in_dim,
        uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (weight_type == 8u) {
        return ds4_gpu_matmul_q8_0_decode_mpp_model_view_tensor(
                out, model_map, model_size, weight_offset, in_dim, out_dim,
                x, n_tok);
    }
    return ds4_gpu_matmul_quant_tensor(out, model_map, model_size,
                                       weight_offset, weight_type, in_dim,
                                       out_dim, x, n_tok);
}

extern "C" int ds4_gpu_glm_k_b_project_typed_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *kv_norm,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t weight_type, uint32_t n_tokens, uint32_t kv_lora_dim,
        uint32_t qk_nope, uint32_t n_head) {
    if (weight_type != 8u) return 0;
    return ds4_gpu_glm_k_b_project_tensor(out, kv_norm, model_map, model_size,
                                           weight_offset, n_tokens,
                                           kv_lora_dim, qk_nope, n_head);
}

extern "C" int ds4_gpu_glm_qk_lowrank_typed_tensor(
        ds4_gpu_tensor *qk_low, const ds4_gpu_tensor *q,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t weight_type, uint32_t n_head, uint32_t kv_lora_dim,
        uint32_t qk_nope, uint32_t qk_dim) {
    if (weight_type != 8u) return 0;
    return ds4_gpu_glm_qk_lowrank_q8_0_tensor(
            qk_low, q, model_map, model_size, weight_offset, n_head,
            kv_lora_dim, qk_nope, qk_dim);
}

extern "C" int ds4_gpu_glm_qk_lowrank_typed_batch_tensor(
        ds4_gpu_tensor *qk_low, const ds4_gpu_tensor *q,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t weight_type, uint32_t n_tokens, uint32_t n_head,
        uint32_t kv_lora_dim, uint32_t qk_nope, uint32_t qk_dim) {
    if (weight_type != 8u) return 0;
    return ds4_gpu_glm_qk_lowrank_q8_0_batch_tensor(
            qk_low, q, model_map, model_size, weight_offset, n_tokens, n_head,
            kv_lora_dim, qk_nope, qk_dim);
}

extern "C" int ds4_gpu_glm_value_project_typed_batch_heads_tensor(
        ds4_gpu_tensor *heads, const ds4_gpu_tensor *lora,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t weight_type, uint32_t n_tokens, uint32_t n_head,
        uint32_t kv_lora_dim, uint32_t value_dim) {
    if (weight_type != 8u) return 0;
    return ds4_gpu_glm_value_project_q8_0_batch_heads_tensor(
            heads, lora, model_map, model_size, weight_offset, n_tokens,
            n_head, kv_lora_dim, value_dim);
}

extern "C" int ds4_gpu_glm_attention_indexed_decode_typed_tensor(
        ds4_gpu_tensor *heads, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low, const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache, const void *model_map,
        uint64_t model_size, uint64_t value_weight_offset,
        uint32_t value_weight_type, const ds4_gpu_tensor *selected,
        uint32_t n_selected, uint32_t cache_cap, bool cache_f16,
        uint32_t n_head, uint32_t kv_lora_dim, uint32_t qk_nope,
        uint32_t qk_rope, uint32_t value_dim, uint32_t n_ctx_orig,
        float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow) {
    if (value_weight_type != 8u) return 0;
    return ds4_gpu_glm_attention_indexed_decode_tensor(
            heads, q, qk_low, kv_lora_cache, k_rope_cache, model_map,
            model_size, value_weight_offset, selected, n_selected, cache_cap,
            cache_f16, n_head, kv_lora_dim, qk_nope, qk_rope, value_dim,
            n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow);
}

extern "C" int ds4_gpu_glm_attention_indexed_batch_typed_tensor(
        ds4_gpu_tensor *heads, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low, const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache, const void *model_map,
        uint64_t model_size, uint64_t value_weight_offset,
        uint32_t value_weight_type, const ds4_gpu_tensor *selected,
        uint32_t n_tokens, uint32_t n_selected, uint32_t cache_cap,
        bool cache_f16, uint32_t n_head, uint32_t kv_lora_dim,
        uint32_t qk_nope, uint32_t qk_rope, uint32_t value_dim,
        uint32_t n_ctx_orig, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast,
        float beta_slow) {
    if (value_weight_type != 8u) return 0;
    return ds4_gpu_glm_attention_indexed_batch_tensor(
            heads, q, qk_low, kv_lora_cache, k_rope_cache, model_map,
            model_size, value_weight_offset, selected, n_tokens, n_selected,
            cache_cap, cache_f16, n_head, kv_lora_dim, qk_nope, qk_rope,
            value_dim, n_ctx_orig, freq_base, freq_scale, ext_factor,
            attn_factor, beta_fast, beta_slow);
}
