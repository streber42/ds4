#include "ds4_rocm_xdev.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

static ds4_rocm_xdev_mesh g_global_mesh = {};
static bool g_global_mesh_initialized = false;
static pthread_mutex_t g_xdev_mutex = PTHREAD_MUTEX_INITIALIZER;

__global__ static void rocm_xdev_accumulate_f32_kernel(float *dst, const float *src, size_t count) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] += src[idx];
    }
}

__global__ static void rocm_xdev_accumulate_f16_kernel(uint16_t *dst, const uint16_t *src, size_t count) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        __half d = __ushort_as_half(dst[idx]);
        __half s = __ushort_as_half(src[idx]);
        dst[idx] = __half_as_ushort(__hadd(d, s));
    }
}

extern "C" int ds4_rocm_xdev_init_mesh(const int *device_ids, int n_devices, ds4_rocm_xdev_mesh *mesh) {
    if (!mesh || n_devices <= 0 || n_devices > DS4_ROCM_XDEV_MAX_DEVICES) return 0;
    
    pthread_mutex_lock(&g_xdev_mutex);
    memset(mesh, 0, sizeof(*mesh));
    mesh->n_devices = n_devices;
    
    for (int i = 0; i < n_devices; i++) {
        mesh->device_ids[i] = device_ids ? device_ids[i] : i;
    }

    if (getenv("DS4_ROCM_FORCE_HOST_STAGING") != NULL) {
        mesh->force_host_staging = true;
    }

    mesh->host_staging_cap = 64u * 1024u * 1024u;
    hipError_t err = hipHostMalloc(&mesh->host_staging_buf, mesh->host_staging_cap, hipHostMallocDefault);
    if (err != hipSuccess) {
        mesh->host_staging_buf = NULL;
        mesh->host_staging_cap = 0;
    }

    // Detect capability per device pair and enable peer access in both directions
    for (int i = 0; i < n_devices; i++) {
        for (int j = 0; j < n_devices; j++) {
            if (i == j) {
                mesh->peer_ok[i][j] = 1;
                continue;
            }
            int dev_i = mesh->device_ids[i];
            int dev_j = mesh->device_ids[j];
            int can_i_to_j = 0;
            int can_j_to_i = 0;
            (void)hipDeviceCanAccessPeer(&can_i_to_j, dev_i, dev_j);
            (void)hipDeviceCanAccessPeer(&can_j_to_i, dev_j, dev_i);

            if (!can_i_to_j || !can_j_to_i) {
                mesh->peer_ok[i][j] = 0;
                continue;
            }

            // Enable access in direction i -> j
            (void)hipSetDevice(dev_i);
            hipError_t e1 = hipDeviceEnablePeerAccess(dev_j, 0);
            (void)hipGetLastError();
            bool ok1 = (e1 == hipSuccess || e1 == hipErrorPeerAccessAlreadyEnabled);

            // Enable access in direction j -> i
            (void)hipSetDevice(dev_j);
            hipError_t e2 = hipDeviceEnablePeerAccess(dev_i, 0);
            (void)hipGetLastError();
            bool ok2 = (e2 == hipSuccess || e2 == hipErrorPeerAccessAlreadyEnabled);

            if (ok1 && ok2) {
                mesh->peer_ok[i][j] = 1;
            } else {
                mesh->peer_ok[i][j] = 0;
            }
        }
    }
    pthread_mutex_unlock(&g_xdev_mutex);
    return 1;
}

extern "C" void ds4_rocm_xdev_destroy_mesh(ds4_rocm_xdev_mesh *mesh) {
    if (!mesh) return;
    pthread_mutex_lock(&g_xdev_mutex);
    if (mesh->host_staging_buf) {
        (void)hipHostFree(mesh->host_staging_buf);
        mesh->host_staging_buf = NULL;
    }
    mesh->host_staging_cap = 0;
    mesh->n_devices = 0;
    pthread_mutex_unlock(&g_xdev_mutex);
}

extern "C" void ds4_rocm_xdev_set_force_host_staging(ds4_rocm_xdev_mesh *mesh, bool force) {
    if (mesh) {
        mesh->force_host_staging = force;
    }
}

extern "C" int ds4_rocm_xdev_copy(ds4_rocm_xdev_mesh *mesh,
                                   int dst_dev, void *dst_ptr,
                                   int src_dev, const void *src_ptr,
                                   size_t bytes, hipStream_t stream) {
    if (!dst_ptr || !src_ptr || bytes == 0) return 1;

    int src_idx = -1, dst_idx = -1;
    if (mesh) {
        for (int i = 0; i < mesh->n_devices; i++) {
            if (mesh->device_ids[i] == src_dev) src_idx = i;
            if (mesh->device_ids[i] == dst_dev) dst_idx = i;
        }
    }

    if (src_dev == dst_dev) {
        (void)hipSetDevice(dst_dev);
        if (stream) {
            return hipMemcpyAsync(dst_ptr, src_ptr, bytes, hipMemcpyDeviceToDevice, stream) == hipSuccess;
        } else {
            return hipMemcpy(dst_ptr, src_ptr, bytes, hipMemcpyDeviceToDevice) == hipSuccess;
        }
    }

    bool use_peer = false;
    if (mesh && !mesh->force_host_staging && src_idx >= 0 && dst_idx >= 0) {
        if (mesh->peer_ok[src_idx][dst_idx]) {
            use_peer = true;
        }
    }

    if (use_peer) {
        (void)hipSetDevice(dst_dev);
        hipError_t err = hipMemcpyPeerAsync(dst_ptr, dst_dev, src_ptr, src_dev, bytes, stream);
        if (err == hipSuccess) {
            return 1;
        }
        // Fall back to host staging if peer memcpy returned error
    }

    // Host Staging Fallback
    pthread_mutex_lock(&g_xdev_mutex);
    ds4_rocm_xdev_mesh local_mesh;
    if (!mesh) {
        mesh = &local_mesh;
        memset(mesh, 0, sizeof(*mesh));
    }

    if (bytes > mesh->host_staging_cap || !mesh->host_staging_buf) {
        if (mesh->host_staging_buf) (void)hipHostFree(mesh->host_staging_buf);
        mesh->host_staging_cap = (bytes > 64u * 1024u * 1024u) ? bytes * 2 : 64u * 1024u * 1024u;
        if (hipHostMalloc(&mesh->host_staging_buf, mesh->host_staging_cap, hipHostMallocDefault) != hipSuccess) {
            mesh->host_staging_buf = NULL;
            mesh->host_staging_cap = 0;
            pthread_mutex_unlock(&g_xdev_mutex);
            return 0;
        }
    }

    (void)hipSetDevice(src_dev);
    if (hipMemcpyAsync(mesh->host_staging_buf, src_ptr, bytes, hipMemcpyDeviceToHost, stream) != hipSuccess) {
        pthread_mutex_unlock(&g_xdev_mutex);
        return 0;
    }
    if (stream) {
        (void)hipStreamSynchronize(stream);
    } else {
        (void)hipDeviceSynchronize();
    }

    (void)hipSetDevice(dst_dev);
    if (hipMemcpyAsync(dst_ptr, mesh->host_staging_buf, bytes, hipMemcpyHostToDevice, stream) != hipSuccess) {
        pthread_mutex_unlock(&g_xdev_mutex);
        return 0;
    }
    if (stream) {
        (void)hipStreamSynchronize(stream);
    } else {
        (void)hipDeviceSynchronize();
    }

    pthread_mutex_unlock(&g_xdev_mutex);
    return 1;
}

extern "C" int ds4_rocm_xdev_accumulate_f32(ds4_rocm_xdev_mesh *mesh,
                                             int dst_dev, float *dst_ptr,
                                             int src_dev, const float *src_ptr,
                                             size_t count, hipStream_t stream) {
    if (!dst_ptr || !src_ptr || count == 0) return 1;
    size_t bytes = count * sizeof(float);

    if (src_dev == dst_dev) {
        (void)hipSetDevice(dst_dev);
        int threads = 256;
        int blocks = (int)((count + threads - 1) / threads);
        rocm_xdev_accumulate_f32_kernel<<<blocks, threads, 0, stream>>>(dst_ptr, src_ptr, count);
        return hipGetLastError() == hipSuccess;
    }

    (void)hipSetDevice(dst_dev);
    float *temp_dst = NULL;
    if (hipMalloc(&temp_dst, bytes) != hipSuccess) return 0;

    int copy_ok = ds4_rocm_xdev_copy(mesh, dst_dev, temp_dst, src_dev, src_ptr, bytes, stream);
    if (!copy_ok) {
        (void)hipFree(temp_dst);
        return 0;
    }

    (void)hipSetDevice(dst_dev);
    int threads = 256;
    int blocks = (int)((count + threads - 1) / threads);
    rocm_xdev_accumulate_f32_kernel<<<blocks, threads, 0, stream>>>(dst_ptr, temp_dst, count);
    hipError_t err = hipGetLastError();

    if (stream) {
        (void)hipStreamSynchronize(stream);
    } else {
        (void)hipDeviceSynchronize();
    }

    (void)hipFree(temp_dst);
    return err == hipSuccess;
}

extern "C" int ds4_rocm_xdev_accumulate_f16(ds4_rocm_xdev_mesh *mesh,
                                             int dst_dev, void *dst_ptr,
                                             int src_dev, const void *src_ptr,
                                             size_t count, hipStream_t stream) {
    if (!dst_ptr || !src_ptr || count == 0) return 1;
    size_t bytes = count * sizeof(uint16_t);

    if (src_dev == dst_dev) {
        (void)hipSetDevice(dst_dev);
        int threads = 256;
        int blocks = (int)((count + threads - 1) / threads);
        rocm_xdev_accumulate_f16_kernel<<<blocks, threads, 0, stream>>>((uint16_t*)dst_ptr, (const uint16_t*)src_ptr, count);
        return hipGetLastError() == hipSuccess;
    }

    (void)hipSetDevice(dst_dev);
    uint16_t *temp_dst = NULL;
    if (hipMalloc(&temp_dst, bytes) != hipSuccess) return 0;

    int copy_ok = ds4_rocm_xdev_copy(mesh, dst_dev, temp_dst, src_dev, src_ptr, bytes, stream);
    if (!copy_ok) {
        (void)hipFree(temp_dst);
        return 0;
    }

    (void)hipSetDevice(dst_dev);
    int threads = 256;
    int blocks = (int)((count + threads - 1) / threads);
    rocm_xdev_accumulate_f16_kernel<<<blocks, threads, 0, stream>>>((uint16_t*)dst_ptr, temp_dst, count);
    hipError_t err = hipGetLastError();

    if (stream) {
        (void)hipStreamSynchronize(stream);
    } else {
        (void)hipDeviceSynchronize();
    }

    (void)hipFree(temp_dst);
    return err == hipSuccess;
}

extern "C" int ds4_rocm_xdev_init_global_mesh(const int *device_ids, int n_devices) {
    if (g_global_mesh_initialized) {
        ds4_rocm_xdev_destroy_mesh(&g_global_mesh);
    }
    int ok = ds4_rocm_xdev_init_mesh(device_ids, n_devices, &g_global_mesh);
    if (ok) {
        g_global_mesh_initialized = true;
    }
    return ok;
}

extern "C" ds4_rocm_xdev_mesh *ds4_rocm_xdev_get_global_mesh(void) {
    if (!g_global_mesh_initialized) {
        int visible = 0;
        if (hipGetDeviceCount(&visible) == hipSuccess && visible > 0) {
            ds4_rocm_xdev_init_global_mesh(NULL, visible);
        }
    }
    return &g_global_mesh;
}
