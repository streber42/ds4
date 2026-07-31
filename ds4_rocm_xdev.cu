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

/* Every producer kernel in this codebase launches on its device's default
 * (null) stream, so "has src_dev's writer drained" is really "has src_dev's
 * default stream reached this point". One persistent, lazily-created event
 * per physical device publishes that point; the peer-copy stream then waits
 * on it via hipStreamWaitEvent, which is a real stream-ordered dependency
 * edge rather than the device-wide hipDeviceSynchronize() flush that
 * gfx1201's hardware scheduler can reorder past. Mirrors the
 * g_shared_gate_up_ready_event pattern in ds4_rocm_shared_expert.cuh. */
#define DS4_ROCM_XDEV_MAX_PHYSICAL_DEVICES 64
static pthread_mutex_t g_xdev_event_mutex = PTHREAD_MUTEX_INITIALIZER;
static hipEvent_t g_xdev_producer_event[DS4_ROCM_XDEV_MAX_PHYSICAL_DEVICES];
static bool g_xdev_producer_event_ready[DS4_ROCM_XDEV_MAX_PHYSICAL_DEVICES];

static hipEvent_t rocm_xdev_producer_event(int device_id) {
    if (device_id < 0 || device_id >= DS4_ROCM_XDEV_MAX_PHYSICAL_DEVICES) return NULL;
    if (!g_xdev_producer_event_ready[device_id]) {
        pthread_mutex_lock(&g_xdev_event_mutex);
        if (!g_xdev_producer_event_ready[device_id]) {
            int cur_dev = 0;
            (void)hipGetDevice(&cur_dev);
            (void)hipSetDevice(device_id);
            if (hipEventCreateWithFlags(&g_xdev_producer_event[device_id], hipEventDisableTiming) == hipSuccess) {
                g_xdev_producer_event_ready[device_id] = true;
            }
            (void)hipSetDevice(cur_dev);
        }
        pthread_mutex_unlock(&g_xdev_event_mutex);
    }
    return g_xdev_producer_event_ready[device_id] ? g_xdev_producer_event[device_id] : NULL;
}

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
        /* The writing kernels on src_dev may still be in flight on src_dev's
         * own default stream: hipMemcpyPeerAsync below is enqueued on
         * dst_dev and has no implicit ordering against src_dev's queue, so
         * without an explicit edge the peer copy can race ahead of
         * src_dev's last write and read stale / partially-written memory.
         * A device-wide hipDeviceSynchronize() here is not a reliable fence
         * on gfx1201's hardware scheduler, which can reorder across it;
         * recording an event on src_dev's producer stream and making the
         * peer-copy stream wait on it is a real stream-ordered
         * happens-before edge the HWS must respect. */
        hipEvent_t producer_ready = rocm_xdev_producer_event(src_dev);
        bool ordered = false;
        if (producer_ready) {
            (void)hipSetDevice(src_dev);
            if (hipEventRecord(producer_ready, 0) == hipSuccess) {
                (void)hipSetDevice(dst_dev);
                ordered = hipStreamWaitEvent(stream, producer_ready, 0) == hipSuccess;
            }
        }
        if (ordered) {
            (void)hipSetDevice(dst_dev);
            hipError_t err = hipMemcpyPeerAsync(dst_ptr, dst_dev, src_ptr, src_dev, bytes, stream);
            if (err == hipSuccess) {
                if (!stream) (void)hipDeviceSynchronize();
                return 1;
            }
        }
        // Fall back to host staging if the ordering edge or peer memcpy failed
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

extern "C" int ds4_rocm_xdev_wait_producer(int dst_dev, int src_dev) {
    if (src_dev == dst_dev) return 1;
    hipEvent_t producer_ready = rocm_xdev_producer_event(src_dev);
    if (!producer_ready) return 0;
    (void)hipSetDevice(src_dev);
    if (hipEventRecord(producer_ready, 0) != hipSuccess) return 0;
    (void)hipSetDevice(dst_dev);
    return hipStreamWaitEvent(0, producer_ready, 0) == hipSuccess;
}

/* All-reduce staging cache. A single lazy-grown device buffer per
 * participating device, so an N-way all-reduce uses one allocation instead
 * of N. The buffer is keyed on (device, capacity); when a call needs more
 * capacity than the cached buffer holds, the old buffer is freed and a new,
 * larger one is installed. Protected by g_xdev_mutex (shared with the mesh
 * and host-staging buffer). */
#define DS4_ROCM_XDEV_ALLREDUCE_MAX_STAGES 16
static struct {
    int device;
    float *buf;
    size_t cap; /* float count */
} g_xdev_allreduce_stage[DS4_ROCM_XDEV_ALLREDUCE_MAX_STAGES];
static int g_xdev_allreduce_stage_count = 0;

static float *rocm_xdev_allreduce_get_stage(int device, size_t count) {
    pthread_mutex_lock(&g_xdev_mutex);
    /* Find an existing stage for this device. */
    for (int i = 0; i < g_xdev_allreduce_stage_count; i++) {
        if (g_xdev_allreduce_stage[i].device == device) {
            if (g_xdev_allreduce_stage[i].cap >= count) {
                float *b = g_xdev_allreduce_stage[i].buf;
                pthread_mutex_unlock(&g_xdev_mutex);
                return b;
            }
            /* Grow: free the too-small buffer and fall through to realloc. */
            (void)hipSetDevice(device);
            (void)hipFree(g_xdev_allreduce_stage[i].buf);
            g_xdev_allreduce_stage[i].buf = NULL;
            g_xdev_allreduce_stage[i].cap = 0;
            float *nb = NULL;
            (void)hipSetDevice(device);
            if (hipMalloc(&nb, count * sizeof(float)) != hipSuccess) {
                pthread_mutex_unlock(&g_xdev_mutex);
                return NULL;
            }
            g_xdev_allreduce_stage[i].buf = nb;
            g_xdev_allreduce_stage[i].cap = count;
            pthread_mutex_unlock(&g_xdev_mutex);
            return nb;
        }
    }
    /* No existing stage for this device -- add one. */
    if (g_xdev_allreduce_stage_count >= DS4_ROCM_XDEV_ALLREDUCE_MAX_STAGES) {
        pthread_mutex_unlock(&g_xdev_mutex);
        return NULL;
    }
    int idx = g_xdev_allreduce_stage_count++;
    g_xdev_allreduce_stage[idx].device = device;
    g_xdev_allreduce_stage[idx].buf = NULL;
    g_xdev_allreduce_stage[idx].cap = 0;
    (void)hipSetDevice(device);
    float *nb = NULL;
    if (hipMalloc(&nb, count * sizeof(float)) != hipSuccess) {
        g_xdev_allreduce_stage_count--;
        pthread_mutex_unlock(&g_xdev_mutex);
        return NULL;
    }
    g_xdev_allreduce_stage[idx].buf = nb;
    g_xdev_allreduce_stage[idx].cap = count;
    pthread_mutex_unlock(&g_xdev_mutex);
    return nb;
}

/* Zero kernel used by all-reduce to initialize the result buffer. */
__global__ static void rocm_xdev_zero_f32_kernel(float *dst, size_t count) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = 0.0f;
    }
}

extern "C" int ds4_rocm_xdev_allreduce_f32(ds4_rocm_xdev_mesh *mesh,
                                            int my_dev, float *result_ptr,
                                            const float *my_partial,
                                            const int *peer_devs,
                                            const float *const *peer_partials,
                                            int n_peers,
                                            size_t count, hipStream_t stream) {
    if (!result_ptr || !my_partial || count == 0) return 1;
    if (n_peers < 0) return 0;
    if (n_peers > 0 && (!peer_devs || !peer_partials)) return 0;

    /* Result = 0, then accumulate every contribution (local + peers) into it.
     * The zero is required because accumulate_f32 adds into the destination;
     * without zeroing, leftover bits from a previous call would corrupt the
     * sum. */
    (void)hipSetDevice(my_dev);
    int threads = 256;
    int blocks = (int)((count + threads - 1) / threads);
    rocm_xdev_zero_f32_kernel<<<blocks, threads, 0, stream>>>(result_ptr, count);
    if (hipGetLastError() != hipSuccess) return 0;

    /* Local partial: same-device accumulate avoids any copy. */
    {
        int t = 256;
        int b = (int)((count + t - 1) / t);
        rocm_xdev_accumulate_f32_kernel<<<b, t, 0, stream>>>(result_ptr, my_partial, count);
        if (hipGetLastError() != hipSuccess) return 0;
    }

    if (n_peers == 0) {
        if (!stream) (void)hipDeviceSynchronize();
        return 1;
    }

    /* One cached staging buffer on my_dev for peer partials -- avoids N
     * malloc/free cycles per collective call. The buffer grows on demand
     * and persists across calls targeting the same device. */
    float *stage = rocm_xdev_allreduce_get_stage(my_dev, count);
    if (!stage) return 0;

    for (int p = 0; p < n_peers; p++) {
        int peer_dev = peer_devs[p];
        const float *peer_buf = peer_partials[p];
        if (!peer_buf) return 0;

        size_t bytes = count * sizeof(float);
        if (!ds4_rocm_xdev_copy(mesh, my_dev, stage, peer_dev, (void *)peer_buf, bytes, stream)) {
            return 0;
        }

        /* Accumulate the staged peer partial into the running result. */
        (void)hipSetDevice(my_dev);
        int t = 256;
        int b = (int)((count + t - 1) / t);
        rocm_xdev_accumulate_f32_kernel<<<b, t, 0, stream>>>(result_ptr, stage, count);
        if (hipGetLastError() != hipSuccess) return 0;
    }

    if (!stream) (void)hipDeviceSynchronize();
    return 1;
}

extern "C" int ds4_rocm_xdev_sync_all_devices(const int *device_ids, int n_devices) {
    for (int i = 0; i < n_devices; i++) {
        if (hipSetDevice(device_ids[i]) != hipSuccess) return 0;
        if (hipDeviceSynchronize() != hipSuccess) return 0;
    }
    return 1;
}

/* Issue #50 (TP=4 persistent per-rank threads, vertical spike): one barrier
 * event per device, lazily created, completely separate from
 * g_xdev_producer_event[] above -- that array orders cross-device copies and
 * the all-reduce internals, and #50 explicitly does not touch the all-reduce.
 * A persistent per-rank worker thread calls ds4_rocm_xdev_spike_record_event
 * after issuing its phase's kernels, on its own device's default stream; the
 * orchestrator thread calls ds4_rocm_xdev_spike_sync_event once per rank as
 * the replacement for ds4_rocm_xdev_sync_all_devices's hipSetDevice +
 * hipDeviceSynchronize loop. Neither call does hipSetDevice: the worker
 * thread is already current on its own device (set once at thread start),
 * and hipEventRecord/hipEventSynchronize do not require the calling thread
 * to be current on the event's device. */
#define DS4_TP4_SPIKE_MAX_DEVICES 8
static pthread_mutex_t g_tp4_spike_event_mutex = PTHREAD_MUTEX_INITIALIZER;
static hipEvent_t g_tp4_spike_barrier_event[DS4_TP4_SPIKE_MAX_DEVICES];
static bool g_tp4_spike_barrier_event_ready[DS4_TP4_SPIKE_MAX_DEVICES];

extern "C" int ds4_rocm_xdev_spike_record_event(int device_id) {
    if (device_id < 0 || device_id >= DS4_TP4_SPIKE_MAX_DEVICES) return 0;
    if (!g_tp4_spike_barrier_event_ready[device_id]) {
        pthread_mutex_lock(&g_tp4_spike_event_mutex);
        if (!g_tp4_spike_barrier_event_ready[device_id]) {
            /* Created while the calling (worker) thread is current on
             * device_id, so the event is correctly bound to that device. */
            if (hipEventCreateWithFlags(&g_tp4_spike_barrier_event[device_id],
                                         hipEventDisableTiming) == hipSuccess) {
                g_tp4_spike_barrier_event_ready[device_id] = true;
            }
        }
        pthread_mutex_unlock(&g_tp4_spike_event_mutex);
    }
    if (!g_tp4_spike_barrier_event_ready[device_id]) return 0;
    return hipEventRecord(g_tp4_spike_barrier_event[device_id], 0) == hipSuccess;
}

extern "C" int ds4_rocm_xdev_spike_sync_event(int device_id) {
    if (device_id < 0 || device_id >= DS4_TP4_SPIKE_MAX_DEVICES) return 0;
    if (!g_tp4_spike_barrier_event_ready[device_id]) return 0;
    return hipEventSynchronize(g_tp4_spike_barrier_event[device_id]) == hipSuccess;
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

extern "C" bool ds4_rocm_xdev_tp_transport_ok(bool host_staging_available,
                                               const bool *pair_peer_ok, int half) {
    if (half <= 0) return false;
    if (host_staging_available) return true;
    if (!pair_peer_ok) return false;
    for (int i = 0; i < half; i++) {
        if (!pair_peer_ok[i]) return false;
    }
    return true;
}

extern "C" bool ds4_rocm_xdev_tp_transport_probe(const int *device_ids, int n_devices, int half) {
    if (!device_ids || half <= 0 || half > DS4_ROCM_XDEV_MAX_DEVICES ||
        n_devices < half * 2) {
        return false;
    }

    /* Host-staging availability: a tiny pinned allocation is representative
     * of whether the real (larger, lazily-grown) staging buffer in
     * ds4_rocm_xdev_init_mesh will succeed -- both go through the same
     * hipHostMalloc path, and pinned-memory exhaustion is a host-wide
     * condition, not a per-size one. */
    void *probe_buf = NULL;
    bool host_staging_available =
        hipHostMalloc(&probe_buf, 4096, hipHostMallocDefault) == hipSuccess;
    if (probe_buf) (void)hipHostFree(probe_buf);

    bool pair_ok[DS4_ROCM_XDEV_MAX_DEVICES];
    for (int i = 0; i < half; i++) {
        int home = device_ids[i];
        int partner = device_ids[i + half];
        int can_home_to_partner = 0, can_partner_to_home = 0;
        (void)hipDeviceCanAccessPeer(&can_home_to_partner, home, partner);
        (void)hipDeviceCanAccessPeer(&can_partner_to_home, partner, home);
        pair_ok[i] = can_home_to_partner && can_partner_to_home;
    }

    return ds4_rocm_xdev_tp_transport_ok(host_staging_available, pair_ok, half);
}
