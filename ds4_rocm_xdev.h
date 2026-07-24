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

#ifdef __cplusplus
}
#endif
