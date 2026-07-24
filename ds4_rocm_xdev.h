#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#else
typedef void* hipStream_t;
#endif

#define DS4_ROCM_XDEV_MAX_DEVICES 16

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int n_devices;
    int device_ids[DS4_ROCM_XDEV_MAX_DEVICES];
    int peer_ok[DS4_ROCM_XDEV_MAX_DEVICES][DS4_ROCM_XDEV_MAX_DEVICES];
    bool force_host_staging;
    void *host_staging_buf;
    size_t host_staging_cap;
} ds4_rocm_xdev_mesh;

/* 
 * 1. Establish Mesh
 * Detects peer capability per device pair, enables peer access in both directions
 * where supported, and initializes host staging resources.
 */
int ds4_rocm_xdev_init_mesh(const int *device_ids, int n_devices, ds4_rocm_xdev_mesh *mesh);
void ds4_rocm_xdev_destroy_mesh(ds4_rocm_xdev_mesh *mesh);
void ds4_rocm_xdev_set_force_host_staging(ds4_rocm_xdev_mesh *mesh, bool force);

/* 
 * 2. Copy Buffer
 * Copies bytes from src_ptr on src_dev to dst_ptr on dst_dev.
 * Uses direct peer transfer if peer_ok, otherwise falls back to host staging.
 */
int ds4_rocm_xdev_copy(ds4_rocm_xdev_mesh *mesh,
                       int dst_dev, void *dst_ptr,
                       int src_dev, const void *src_ptr,
                       size_t bytes, hipStream_t stream);

/* 
 * 3. Accumulate Buffer (F32)
 * Accumulates (dst[i] += src[i]) count float elements from src_ptr on src_dev
 * into dst_ptr on dst_dev.
 * Uses direct peer transfer if peer_ok, otherwise falls back to host staging.
 */
int ds4_rocm_xdev_accumulate_f32(ds4_rocm_xdev_mesh *mesh,
                                 int dst_dev, float *dst_ptr,
                                 int src_dev, const float *src_ptr,
                                 size_t count, hipStream_t stream);

/* Accumulate Buffer (F16 / Half) */
int ds4_rocm_xdev_accumulate_f16(ds4_rocm_xdev_mesh *mesh,
                                 int dst_dev, void *dst_ptr,
                                 int src_dev, const void *src_ptr,
                                 size_t count, hipStream_t stream);

/* Global mesh access convenience functions */
int ds4_rocm_xdev_init_global_mesh(const int *device_ids, int n_devices);
ds4_rocm_xdev_mesh *ds4_rocm_xdev_get_global_mesh(void);

/*
 * Tensor-parallel transport reachability.
 *
 * Tensor parallelism needs to move data between `half` home/partner tier
 * pairs (i, i+half). A working host-staging bounce buffer is a universal
 * fallback that covers every pair by itself (this is the path already
 * proven correct and fast enough -- see the PRD's feasibility notes), so it
 * alone is sufficient; only when host-staging is unavailable does every
 * individual pair need bidirectional direct peer access.
 *
 * ds4_rocm_xdev_tp_transport_ok is pure decision logic (no device I/O), so
 * it can be unit-tested with fabricated inputs without hardware.
 * ds4_rocm_xdev_tp_transport_probe gathers the real inputs via lightweight
 * capability queries -- no hipSetDevice, no peer-access enable calls, safe
 * to call before ds4_gpu_init_multi's mesh actually establishes the peer
 * mesh -- so callers can decide whether to attempt tensor parallelism at
 * all before any placement or device-init work happens.
 */
bool ds4_rocm_xdev_tp_transport_ok(bool host_staging_available,
                                   const bool *pair_peer_ok, int half);
bool ds4_rocm_xdev_tp_transport_probe(const int *device_ids, int n_devices, int half);

#ifdef __cplusplus
}
#endif
