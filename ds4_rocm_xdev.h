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

/*
 * 4. All-Reduce (F32) -- TP=4 collective
 *
 * Combines partial float vectors from `1 + n_peers` devices into the
 * element-wise sum and writes it to `result_ptr` on `my_dev`. This is the
 * primitive TP=4 uses to merge per-rank partial attention outputs, per-rank
 * partial MoE down-projections, and per-rank partial output-head logits into
 * the complete result.
 *
 * `my_dev` is the device whose result buffer is being populated. `my_partial`
 * is that device's own contribution (count floats); it is read-only and is
 * NOT overwritten -- the caller keeps the partial and a separate result
 * buffer. `peer_devs[0..n_peers-1]` and `peer_partials[0..n_peers-1]` name
 * each peer's device and its partial buffer. `n_peers` is the world size
 * minus one (3 for TP=4). `result_ptr` lives on `my_dev` and is filled with
 * sum(my_partial, peer_partials[0..n_peers-1]) -- it is overwritten, not
 * added into, so the caller need not zero it.
 *
 * Implementation: brute-force all-gather + local accumulate. A single local
 * staging buffer of `count` floats is allocated lazily (cached across calls
 * that share `my_dev` and `count`, grown on demand). For each peer: wait on
 * the peer's producer event so the peer kernel has drained, copy the peer
 * partial into the staging buffer via ds4_rocm_xdev_copy, then accumulate
 * the staging buffer into `result_ptr`. After the peer loop, accumulate the
 * local `my_partial` as the final contribution (the local->local path uses
 * the existing same-device kernel -- no temp allocation).
 *
 * Ordering: each peer copy is stream-ordered against that peer's default
 * stream via ds4_rocm_xdev_wait_producer (mirroring the fence used by
 * ds4_rocm_xdev_copy's direct-peer path), so the collective is safe to call
 * immediately after each peer's compute kernel returns -- the caller does
 * not need to synchronize.
 *
 * Returns 1 on success, 0 on any failure.
 */
int ds4_rocm_xdev_allreduce_f32(ds4_rocm_xdev_mesh *mesh,
                                int my_dev, float *result_ptr,
                                const float *my_partial,
                                const int *peer_devs,
                                const float *const *peer_partials,
                                int n_peers,
                                size_t count, hipStream_t stream);

/* Global mesh access convenience functions */
int ds4_rocm_xdev_init_global_mesh(const int *device_ids, int n_devices);
ds4_rocm_xdev_mesh *ds4_rocm_xdev_get_global_mesh(void);

/*
 * Synchronize all devices in the mesh. Issues a hipDeviceSynchronize()
 * on each device in order. Use this as a barrier between per-tier compute
 * (which runs asynchronously on each device's default stream) and a
 * multi-device all-reduce that reads peer buffers, ensuring every device's
 * kernels have completed before any peer's data is read.
 * Returns 1 on success, 0 if any device's sync failed.
 */
int ds4_rocm_xdev_sync_all_devices(const int *device_ids, int n_devices);

/*
 * Issue #50 (TP=4 persistent per-rank threads, vertical spike) barrier
 * events. One lazily-created event per device, independent of the
 * producer-event mesh used internally by ds4_rocm_xdev_copy /
 * ds4_rocm_xdev_allreduce_f32 -- kept separate so this barrier cannot
 * perturb all-reduce ordering, which issue #50 explicitly does not touch.
 *
 * ds4_rocm_xdev_spike_record_event(device_id): call from a thread whose
 * current device is already device_id (e.g. a persistent per-rank worker
 * thread that called hipSetDevice once at startup). Records an event on
 * that device's default stream marking everything queued so far. Does not
 * call hipSetDevice.
 *
 * ds4_rocm_xdev_spike_sync_event(device_id): call from any thread (no
 * hipSetDevice needed) to block until device_id's queued work up to its
 * last recorded event has completed -- the event/stream-scoped replacement
 * for ds4_rocm_xdev_sync_all_devices's hipSetDevice + hipDeviceSynchronize
 * loop.
 *
 * Both return 1 on success, 0 on failure (including "never recorded").
 */
int ds4_rocm_xdev_spike_record_event(int device_id);
int ds4_rocm_xdev_spike_sync_event(int device_id);

/*
 * Issue #61: same-device stream fencing so a worker thread's all-reduce can
 * run on its own secondary stream (see ds4_rocm_xdev_stream_create) without
 * a host-blocking hipDeviceSynchronize, while staying correctly ordered
 * against the default-stream compute kernels that produce its input and
 * consume its output. Both reuse the same per-device barrier event as
 * ds4_rocm_xdev_spike_record_event/spike_sync_event above -- only the
 * stream each call targets differs. Both assume the calling thread's
 * current device is already device_id (same invariant as the two calls
 * above).
 *
 * ds4_rocm_xdev_spike_record_event_on(device_id, stream): record
 *   device_id's barrier event on `stream` instead of the default stream.
 * ds4_rocm_xdev_spike_stream_wait(device_id, stream): make `stream` (NULL
 *   selects the default stream) wait on device_id's most recently recorded
 *   barrier event.
 */
int ds4_rocm_xdev_spike_record_event_on(int device_id, void *stream);
int ds4_rocm_xdev_spike_stream_wait(int device_id, void *stream);

/*
 * Issue #61: create/destroy a per-rank persistent stream for the all-reduce
 * path, used together with the two fences above. Created with
 * hipStreamNonBlocking so it has no implicit ordering against the default
 * stream -- every ordering edge is the explicit fence pair instead, since
 * that is the only cross-stream ordering primitive this codebase has
 * actually verified on gfx1201 (see ds4_rocm_xdev_copy's peer-copy path);
 * implicit legacy-stream ordering has not been exercised here. Must be
 * created by the thread that will use it, while that thread is current on
 * the target device (same invariant as the persistent worker threads use
 * for ds4_gpu_set_current_device).
 */
void *ds4_rocm_xdev_stream_create(void);
void ds4_rocm_xdev_stream_destroy(void *stream);

/*
 * Fence a producer device against a consumer that will read its memory
 * directly via a peer-mapped pointer (no explicit xdev copy). Records an
 * event on src_dev's producer (default) stream and makes dst_dev's default
 * stream wait on it -- the same stream-ordered happens-before edge
 * ds4_rocm_xdev_copy uses internally, exposed here for callers (e.g.
 * ds4_gpu_tensor_wait_xdev) that only need the ordering, not a copy.
 */
int ds4_rocm_xdev_wait_producer(int dst_dev, int src_dev);

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
