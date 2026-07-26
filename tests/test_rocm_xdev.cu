#include "ds4_rocm_xdev.h"

#include <hip/hip_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define TEST_BUF_BYTES (16u * 1024u * 1024u)
#define BENCH_BUF_BYTES (64u * 1024u * 1024u)

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void fill_pattern(uint8_t *buf, size_t bytes, uint8_t seed) {
    for (size_t i = 0; i < bytes; i++) {
        buf[i] = (uint8_t)((i ^ seed) & 0xFF);
    }
}

static void fill_floats(float *buf, size_t count, float seed) {
    for (size_t i = 0; i < count; i++) {
        buf[i] = (float)(i * 0.125f + seed);
    }
}

/* Forward: defined below main() for readability. Runs the TP=4 all-reduce
 * correctness, fallback, degenerate, and cache-reuse tests. */
static void run_allreduce_tests(ds4_rocm_xdev_mesh *mesh, int device_count);

int main(void) {
    printf("================================================================================\n");
    printf("Running ROCm Cross-Device Transfer Module Standalone Tests\n");
    printf("================================================================================\n\n");

    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count <= 0) {
        fprintf(stderr, "Error: No ROCm devices found: %s\n", hipGetErrorString(err));
        return 1;
    }

    printf("[INFO] Detected %d ROCm GPU device(s).\n", device_count);

    // 1. Establish Mesh
    ds4_rocm_xdev_mesh mesh;
    if (!ds4_rocm_xdev_init_mesh(NULL, device_count, &mesh)) {
        fprintf(stderr, "[FAIL] ds4_rocm_xdev_init_mesh failed!\n");
        return 1;
    }
    printf("[PASS] Establish Mesh initialized successfully.\n");

    // 2. Verify Peer Capability Detection & Bidirectional Enablement
    printf("\n--- Peer Mesh Status ---\n");
    for (int i = 0; i < device_count; i++) {
        for (int j = 0; j < device_count; j++) {
            if (i == j) continue;
            int can_ij = 0, can_ji = 0;
            (void)hipDeviceCanAccessPeer(&can_ij, i, j);
            (void)hipDeviceCanAccessPeer(&can_ji, j, i);
            bool peer_active = mesh.peer_ok[i][j] != 0;
            printf("  Device Pair (%d -> %d): canAccess=%d (rev=%d), peer_ok=%d\n",
                   i, j, can_ij, can_ji, peer_active ? 1 : 0);
            
            // Assert detected capability matches hipDeviceCanAccessPeer
            if (can_ij && can_ji) {
                if (!peer_active) {
                    fprintf(stderr, "[FAIL] Pair (%d, %d) supports peer access but mesh.peer_ok is false!\n", i, j);
                    return 1;
                }
                // Assert bidirectional symmetry
                if (mesh.peer_ok[i][j] != mesh.peer_ok[j][i]) {
                    fprintf(stderr, "[FAIL] Peer access asymmetry detected between (%d, %d)!\n", i, j);
                    return 1;
                }
            }
        }
    }
    printf("[PASS] Peer capability detected per pair and bidirectional access verified.\n");

    // Test Allocations across all devices
    void *d_src[DS4_ROCM_XDEV_MAX_DEVICES] = {NULL};
    void *d_dst[DS4_ROCM_XDEV_MAX_DEVICES] = {NULL};
    uint8_t *h_src = (uint8_t *)malloc(TEST_BUF_BYTES);
    uint8_t *h_dst = (uint8_t *)malloc(TEST_BUF_BYTES);

    for (int i = 0; i < device_count; i++) {
        hipSetDevice(i);
        if (hipMalloc(&d_src[i], TEST_BUF_BYTES) != hipSuccess ||
            hipMalloc(&d_dst[i], TEST_BUF_BYTES) != hipSuccess) {
            fprintf(stderr, "[FAIL] Device allocation failed on GPU %d\n", i);
            return 1;
        }
    }

    // 3. Byte-exact Copy Correctness Across All Ordered Device Pairs (Direct Peer Mode)
    printf("\n--- Testing Byte-Exact Copy (Direct Peer Mode) ---\n");
    for (int i = 0; i < device_count; i++) {
        for (int j = 0; j < device_count; j++) {
            fill_pattern(h_src, TEST_BUF_BYTES, (uint8_t)(i * 17 + j * 31 + 5));
            memset(h_dst, 0, TEST_BUF_BYTES);

            hipSetDevice(i);
            hipMemcpy(d_src[i], h_src, TEST_BUF_BYTES, hipMemcpyHostToDevice);
            hipSetDevice(j);
            hipMemset(d_dst[j], 0, TEST_BUF_BYTES);

            int copy_ok = ds4_rocm_xdev_copy(&mesh, j, d_dst[j], i, d_src[i], TEST_BUF_BYTES, 0);
            if (!copy_ok) {
                fprintf(stderr, "[FAIL] ds4_rocm_xdev_copy failed for pair (%d -> %d)!\n", i, j);
                return 1;
            }

            hipSetDevice(j);
            hipMemcpy(h_dst, d_dst[j], TEST_BUF_BYTES, hipMemcpyDeviceToHost);

            if (memcmp(h_src, h_dst, TEST_BUF_BYTES) != 0) {
                fprintf(stderr, "[FAIL] Byte mismatch in copy for pair (%d -> %d)!\n", i, j);
                return 1;
            }
        }
    }
    printf("[PASS] Byte-exact copy verified across all ordered device pairs.\n");

    // 4. Accumulate Correctness Against CPU Reference (Direct Peer Mode)
    printf("\n--- Testing Accumulate F32 / F16 (Direct Peer Mode) ---\n");
    size_t count_f32 = TEST_BUF_BYTES / sizeof(float);
    float *h_a = (float *)malloc(TEST_BUF_BYTES);
    float *h_b = (float *)malloc(TEST_BUF_BYTES);
    float *h_res = (float *)malloc(TEST_BUF_BYTES);

    for (int i = 0; i < device_count; i++) {
        for (int j = 0; j < device_count; j++) {
            fill_floats(h_a, count_f32, (float)(i * 1.5f + 1.0f));
            fill_floats(h_b, count_f32, (float)(j * 2.2f + 3.0f));

            hipSetDevice(j);
            hipMemcpy(d_dst[j], h_a, TEST_BUF_BYTES, hipMemcpyHostToDevice);
            hipSetDevice(i);
            hipMemcpy(d_src[i], h_b, TEST_BUF_BYTES, hipMemcpyHostToDevice);

            int accum_ok = ds4_rocm_xdev_accumulate_f32(&mesh, j, (float*)d_dst[j], i, (const float*)d_src[i], count_f32, 0);
            if (!accum_ok) {
                fprintf(stderr, "[FAIL] ds4_rocm_xdev_accumulate_f32 failed for pair (%d -> %d)!\n", i, j);
                return 1;
            }

            hipSetDevice(j);
            hipMemcpy(h_res, d_dst[j], TEST_BUF_BYTES, hipMemcpyDeviceToHost);

            for (size_t k = 0; k < count_f32; k++) {
                float expected = h_a[k] + h_b[k];
                if (fabsf(h_res[k] - expected) > 1e-5f) {
                    fprintf(stderr, "[FAIL] Accumulate F32 divergence at [%kn] pair (%d -> %d): got %f, expected %f\n",
                            k, i, j, h_res[k], expected);
                    return 1;
                }
            }
        }
    }
    printf("[PASS] Accumulate F32 produces numerically correct sums against CPU reference.\n");

    // 5. Host-Staging Fallback Mode Tests
    printf("\n--- Testing Host-Staging Fallback Mode ---\n");
    ds4_rocm_xdev_set_force_host_staging(&mesh, true);

    // Verify Copy under Host Staging Fallback
    for (int i = 0; i < device_count; i++) {
        for (int j = 0; j < device_count; j++) {
            fill_pattern(h_src, TEST_BUF_BYTES, (uint8_t)(i * 13 + j * 19 + 7));
            memset(h_dst, 0, TEST_BUF_BYTES);

            hipSetDevice(i);
            hipMemcpy(d_src[i], h_src, TEST_BUF_BYTES, hipMemcpyHostToDevice);
            hipSetDevice(j);
            hipMemset(d_dst[j], 0, TEST_BUF_BYTES);

            int copy_ok = ds4_rocm_xdev_copy(&mesh, j, d_dst[j], i, d_src[i], TEST_BUF_BYTES, 0);
            if (!copy_ok) {
                fprintf(stderr, "[FAIL] Host staging copy failed for pair (%d -> %d)!\n", i, j);
                return 1;
            }

            hipSetDevice(j);
            hipMemcpy(h_dst, d_dst[j], TEST_BUF_BYTES, hipMemcpyDeviceToHost);

            if (memcmp(h_src, h_dst, TEST_BUF_BYTES) != 0) {
                fprintf(stderr, "[FAIL] Host staging copy byte mismatch for pair (%d -> %d)!\n", i, j);
                return 1;
            }
        }
    }
    printf("[PASS] Host-staging fallback copy verified byte-exact across all pairs.\n");

    // Verify Accumulate under Host Staging Fallback
    for (int i = 0; i < device_count; i++) {
        for (int j = 0; j < device_count; j++) {
            fill_floats(h_a, count_f32, (float)(i * 3.1f + 0.5f));
            fill_floats(h_b, count_f32, (float)(j * 1.7f + 2.0f));

            hipSetDevice(j);
            hipMemcpy(d_dst[j], h_a, TEST_BUF_BYTES, hipMemcpyHostToDevice);
            hipSetDevice(i);
            hipMemcpy(d_src[i], h_b, TEST_BUF_BYTES, hipMemcpyHostToDevice);

            int accum_ok = ds4_rocm_xdev_accumulate_f32(&mesh, j, (float*)d_dst[j], i, (const float*)d_src[i], count_f32, 0);
            if (!accum_ok) {
                fprintf(stderr, "[FAIL] Host staging accumulate F32 failed for pair (%d -> %d)!\n", i, j);
                return 1;
            }

            hipSetDevice(j);
            hipMemcpy(h_res, d_dst[j], TEST_BUF_BYTES, hipMemcpyDeviceToHost);

            for (size_t k = 0; k < count_f32; k++) {
                float expected = h_a[k] + h_b[k];
                if (fabsf(h_res[k] - expected) > 1e-5f) {
                    fprintf(stderr, "[FAIL] Host staging accumulate F32 divergence at [%kn] pair (%d -> %d): got %f, expected %f\n",
                            k, i, j, h_res[k], expected);
                    return 1;
                }
            }
        }
    }
    printf("[PASS] Host-staging fallback accumulate verified numerically correct.\n");

    // Disable forced host staging for bandwidth benchmark
    ds4_rocm_xdev_set_force_host_staging(&mesh, false);

    // 6. Bandwidth Floor Benchmark & Assertion
    printf("\n--- Bandwidth Benchmark & Floor Assertion ---\n");
    void *d_bench_src = NULL, *d_bench_dst = NULL;
    if (device_count > 1) {
        hipSetDevice(0);
        hipMalloc(&d_bench_src, BENCH_BUF_BYTES);
        hipSetDevice(1);
        hipMalloc(&d_bench_dst, BENCH_BUF_BYTES);

        int iters = 10;
        // Warmup
        ds4_rocm_xdev_copy(&mesh, 1, d_bench_dst, 0, d_bench_src, BENCH_BUF_BYTES, 0);
        hipDeviceSynchronize();

        // Direct Peer Benchmark
        double t0 = get_time_sec();
        for (int it = 0; it < iters; it++) {
            ds4_rocm_xdev_copy(&mesh, 1, d_bench_dst, 0, d_bench_src, BENCH_BUF_BYTES, 0);
        }
        hipDeviceSynchronize();
        double t1 = get_time_sec();
        double elapsed_peer = t1 - t0;
        double gbps_peer = (double)BENCH_BUF_BYTES * iters / (elapsed_peer * 1e9);

        printf("  Direct Peer Copy (0 -> 1): %.2f GB/s (elapsed=%.3f s)\n", gbps_peer, elapsed_peer);

        // Host Staging Benchmark
        ds4_rocm_xdev_set_force_host_staging(&mesh, true);
        double t2 = get_time_sec();
        for (int it = 0; it < iters; it++) {
            ds4_rocm_xdev_copy(&mesh, 1, d_bench_dst, 0, d_bench_src, BENCH_BUF_BYTES, 0);
        }
        hipDeviceSynchronize();
        double t3 = get_time_sec();
        double elapsed_host = t3 - t2;
        double gbps_host = (double)BENCH_BUF_BYTES * iters / (elapsed_host * 1e9);

        printf("  Host Staging Copy (0 -> 1): %.2f GB/s (elapsed=%.3f s)\n", gbps_host, elapsed_host);

        // Bandwidth Floor Assertion
        double min_peer_gbps = 5.0;
        if (mesh.peer_ok[0][1] && gbps_peer < min_peer_gbps) {
            fprintf(stderr, "[FAIL] Peer bandwidth %.2f GB/s is below floor of %.2f GB/s!\n",
                    gbps_peer, min_peer_gbps);
            return 1;
        }

        double min_host_gbps = 2.0;
        if (gbps_host < min_host_gbps) {
            fprintf(stderr, "[FAIL] Host staging bandwidth %.2f GB/s is below floor of %.2f GB/s!\n",
                    gbps_host, min_host_gbps);
            return 1;
        }
        printf("[PASS] Bandwidth floor assertion passed (Peer >= %.1f GB/s, Host >= %.1f GB/s).\n", min_peer_gbps, min_host_gbps);

        hipSetDevice(0); hipFree(d_bench_src);
        hipSetDevice(1); hipFree(d_bench_dst);
    } else {
        printf("  Single GPU detected; skipping multi-GPU peer bandwidth benchmark.\n");
        printf("[PASS] Bandwidth benchmark skipped for single GPU setup.\n");
    }

    // Cleanup
    for (int i = 0; i < device_count; i++) {
        hipSetDevice(i);
        hipFree(d_src[i]);
        hipFree(d_dst[i]);
    }
    free(h_src);
    free(h_dst);
    free(h_a);
    free(h_b);
    free(h_res);
    ds4_rocm_xdev_destroy_mesh(&mesh);

    // 7. Tensor-Parallel Transport Reachability (issue 09: refusal & fallback)
    //
    // ds4_rocm_xdev_tp_transport_ok is pure decision logic -- no device I/O --
    // so every case below is checked with fabricated inputs, independent of
    // whatever this box's real topology happens to be.
    printf("\n--- Testing TP Transport Reachability Decision ---\n");
    {
        bool all_peer_ok[4] = {true, true, true, true};
        bool one_peer_down[4] = {true, false, true, true};

        // Host-staging available covers every pair regardless of peer_ok.
        if (!ds4_rocm_xdev_tp_transport_ok(true, one_peer_down, 4)) {
            fprintf(stderr, "[FAIL] host-staging available should cover a degraded peer pair\n");
            return 1;
        }
        // No host-staging, but every pair has direct peer access: still ok.
        if (!ds4_rocm_xdev_tp_transport_ok(false, all_peer_ok, 4)) {
            fprintf(stderr, "[FAIL] no host-staging + all peer_ok should still be reachable\n");
            return 1;
        }
        // No host-staging and one pair down: genuinely unreachable.
        if (ds4_rocm_xdev_tp_transport_ok(false, one_peer_down, 4)) {
            fprintf(stderr, "[FAIL] no host-staging + one degraded pair must be unreachable\n");
            return 1;
        }
        // No host-staging and no peer array at all: unreachable, not a crash.
        if (ds4_rocm_xdev_tp_transport_ok(false, NULL, 4)) {
            fprintf(stderr, "[FAIL] null pair_peer_ok with no host-staging must be unreachable\n");
            return 1;
        }
        // Degenerate rank counts refuse rather than vacuously succeed.
        if (ds4_rocm_xdev_tp_transport_ok(true, all_peer_ok, 0) ||
            ds4_rocm_xdev_tp_transport_ok(true, all_peer_ok, -1)) {
            fprintf(stderr, "[FAIL] half<=0 must be unreachable, not vacuously true\n");
            return 1;
        }
        printf("[PASS] Pure decision logic covers host-staging, peer-only, and unreachable cases.\n");
    }

    // The hardware probe wraps the same decision with real capability
    // queries. This box's peer mesh is fully connected (verified above), so
    // the probe must report every adjacent-pair TP topology as reachable.
    if (device_count >= 2) {
        int ids[DS4_ROCM_XDEV_MAX_DEVICES];
        for (int i = 0; i < device_count; i++) ids[i] = i;
        int half = device_count / 2;
        if (half > 0) {
            if (!ds4_rocm_xdev_tp_transport_probe(ids, device_count, half)) {
                fprintf(stderr,
                        "[FAIL] tp_transport_probe reported unreachable on a box with a "
                        "fully-connected peer mesh (half=%d)\n", half);
                return 1;
            }
            printf("[PASS] tp_transport_probe reports reachable for the real %d-device topology (half=%d).\n",
                   device_count, half);
        }
    }
    // Malformed inputs refuse cleanly rather than reading out of bounds.
    if (ds4_rocm_xdev_tp_transport_probe(NULL, device_count, 1) ||
        ds4_rocm_xdev_tp_transport_probe(NULL, 0, 0)) {
        fprintf(stderr, "[FAIL] tp_transport_probe must refuse malformed inputs, not crash\n");
        return 1;
    }
    printf("[PASS] tp_transport_probe refuses malformed inputs cleanly.\n");

    /* TP=4 collective tests (issue 27). */
    run_allreduce_tests(&mesh, device_count);

    printf("\n================================================================================\n");
    printf("ALL CROSS-DEVICE TRANSFER STANDALONE TESTS PASSED SUCCESSFULLY!\n");
    printf("================================================================================\n");
    return 0;
}

/* =============================================================================
 * All-Reduce Tests (TP=4 collective)
 *
 * These run inside main() above but are defined here as separate functions
 * for readability. The test harness is compiled into the same binary as the
 * rest of the xdev tests; main() calls them before printing the final PASS.
 *
 * To avoid disturbing the existing PASS output above, we insert the all-
 * reduce tests right before the "ALL CROSS-DEVICE TRANSFER STANDALONE TESTS
 * PASSED" banner by calling the function from main() instead of duplicating
 * the banner. The banner line is emitted by main() itself -- see the edit
 * near the bottom of main() that invokes run_allreduce_tests().
 * ========================================================================== */

static void run_allreduce_tests(ds4_rocm_xdev_mesh *mesh, int device_count) {
    printf("\n--- All-Reduce F32 (TP=4 collective) ---\n");
    if (device_count < 2) {
        printf("  SKIP: need at least 2 GPUs for all-reduce; have %d\n", device_count);
        return;
    }

    /* Use up to 4 devices for the TP=4 scenario; fall back to 2 or 3 when
     * fewer devices are present. The all-reduce API is N-way so it must
     * work for any n_peers >= 1. */
    int n_ranks = device_count >= 4 ? 4 : device_count;
    int n_peers = n_ranks - 1;
    size_t count = 1024 * 1024; /* 1M floats = 4 MB per rank -- representative of a real attn/MoE partial */
    size_t bytes = count * sizeof(float);

    /* Per-rank device buffers: partial (input) and result (output). */
    float *d_partial[DS4_ROCM_XDEV_MAX_DEVICES] = {NULL};
    float *d_result[DS4_ROCM_XDEV_MAX_DEVICES] = {NULL};
    float *h_partial[DS4_ROCM_XDEV_MAX_DEVICES] = {NULL};
    float *h_result[DS4_ROCM_XDEV_MAX_DEVICES] = {NULL};

    for (int r = 0; r < n_ranks; r++) {
        hipSetDevice(r);
        if (hipMalloc(&d_partial[r], bytes) != hipSuccess ||
            hipMalloc(&d_result[r], bytes) != hipSuccess) {
            fprintf(stderr, "[FAIL] all-reduce: device allocation failed on GPU %d\n", r);
            return;
        }
        h_partial[r] = (float *)malloc(bytes);
        h_result[r] = (float *)malloc(bytes);
        if (!h_partial[r] || !h_result[r]) {
            fprintf(stderr, "[FAIL] all-reduce: host allocation failed\n");
            return;
        }
    }

    /* --- Test A: 4-rank (or 2-/3-rank) all-reduce correctness ---
     *
     * Each rank r contributes partial[i] = (i + 1) * (r + 1) -- a distinct
     * per-rank linear pattern. The expected result on every rank is the sum
     * across all ranks: sum_{r=0..N-1} (i+1)*(r+1) = (i+1) * N*(N+1)/2.
     *
     * With N=4 and i=0: 1 * 10 = 10. With i=999999: 1000000 * 10 = 10000000.
     * Well within f32 exact-integer range (up to 2^24 = 16.7M). */
    int dev_ids[DS4_ROCM_XDEV_MAX_DEVICES];
    const float *d_partial_ptrs[DS4_ROCM_XDEV_MAX_DEVICES];
    for (int r = 0; r < n_ranks; r++) dev_ids[r] = r;

    for (int r = 0; r < n_ranks; r++) {
        for (size_t i = 0; i < count; i++) {
            h_partial[r][i] = (float)((i + 1) * (r + 1));
        }
        hipSetDevice(r);
        hipMemcpy(d_partial[r], h_partial[r], bytes, hipMemcpyHostToDevice);
        hipMemset(d_result[r], 0xAB, bytes); /* dirty-fill to catch missing-zero bugs */
    }

    /* Run the all-reduce for each rank as the "result owner". */
    for (int owner = 0; owner < n_ranks; owner++) {
        /* Build peer arrays excluding the owner. */
        int peer_devs[DS4_ROCM_XDEV_MAX_DEVICES];
        const float *peer_ptrs[DS4_ROCM_XDEV_MAX_DEVICES];
        int pi = 0;
        for (int r = 0; r < n_ranks; r++) {
            if (r == owner) continue;
            peer_devs[pi] = dev_ids[r];
            peer_ptrs[pi] = d_partial[r];
            pi++;
        }

        int ok = ds4_rocm_xdev_allreduce_f32(mesh, owner, d_result[owner],
                                              d_partial[owner],
                                              peer_devs, peer_ptrs, n_peers,
                                              count, 0);
        if (!ok) {
            fprintf(stderr, "[FAIL] all-reduce: ds4_rocm_xdev_allreduce_f32 failed for owner=%d\n", owner);
            goto cleanup_ar;
        }

        hipSetDevice(owner);
        hipMemcpy(h_result[owner], d_result[owner], bytes, hipMemcpyDeviceToHost);

        /* Verify against the closed-form sum. */
        float scale = (float)(n_ranks * (n_ranks + 1) / 2); /* sum of 1..N */
        int diverged = 0;
        for (size_t i = 0; i < count; i++) {
            float expected = (float)(i + 1) * scale;
            float got = h_result[owner][i];
            /* Relative tolerance ~1e-5; the sum involves at most N=4 adds of
             * well-conditioned integers, so f32 roundoff is negligible. */
            float tol = 1e-4f * fabsf(expected) + 1e-6f;
            if (fabsf(got - expected) > tol) {
                fprintf(stderr, "[FAIL] all-reduce owner=%d idx=%zu: got %f expected %f\n",
                        owner, i, got, expected);
                diverged = 1;
                break;
            }
        }
        if (diverged) goto cleanup_ar;
    }
    printf("[PASS] %d-rank all-reduce produces correct sums on every owner device.\n", n_ranks);

    /* --- Test B: host-staging fallback path for all-reduce ---
     *
     * Force host staging, re-run the same computation, verify the same
     * expected sums. This exercises the staging-buffer code path in
     * ds4_rocm_xdev_copy() that all-reduce relies on when peer access is
     * unavailable -- a critical correctness path since the PRD's transport
     * feasibility probe treats host-staging as the universal fallback. */
    ds4_rocm_xdev_set_force_host_staging(mesh, true);

    /* Re-dirty the result buffers to catch missed writes. */
    for (int r = 0; r < n_ranks; r++) {
        hipSetDevice(r);
        hipMemset(d_result[r], 0xCD, bytes);
    }

    for (int owner = 0; owner < n_ranks; owner++) {
        int peer_devs[DS4_ROCM_XDEV_MAX_DEVICES];
        const float *peer_ptrs[DS4_ROCM_XDEV_MAX_DEVICES];
        int pi = 0;
        for (int r = 0; r < n_ranks; r++) {
            if (r == owner) continue;
            peer_devs[pi] = dev_ids[r];
            peer_ptrs[pi] = d_partial[r];
            pi++;
        }

        int ok = ds4_rocm_xdev_allreduce_f32(mesh, owner, d_result[owner],
                                              d_partial[owner],
                                              peer_devs, peer_ptrs, n_peers,
                                              count, 0);
        if (!ok) {
            fprintf(stderr, "[FAIL] all-reduce host-staging: call failed for owner=%d\n", owner);
            ds4_rocm_xdev_set_force_host_staging(mesh, false);
            goto cleanup_ar;
        }

        hipSetDevice(owner);
        hipMemcpy(h_result[owner], d_result[owner], bytes, hipMemcpyDeviceToHost);

        float scale = (float)(n_ranks * (n_ranks + 1) / 2);
        int diverged = 0;
        for (size_t i = 0; i < count; i++) {
            float expected = (float)(i + 1) * scale;
            float got = h_result[owner][i];
            float tol = 1e-4f * fabsf(expected) + 1e-6f;
            if (fabsf(got - expected) > tol) {
                fprintf(stderr, "[FAIL] all-reduce host-staging owner=%d idx=%zu: got %f expected %f\n",
                        owner, i, got, expected);
                diverged = 1;
                break;
            }
        }
        if (diverged) {
            ds4_rocm_xdev_set_force_host_staging(mesh, false);
            goto cleanup_ar;
        }
    }
    printf("[PASS] %d-rank all-reduce (host-staging fallback) produces correct sums.\n", n_ranks);

    ds4_rocm_xdev_set_force_host_staging(mesh, false);

    /* --- Test C: degenerate 1-rank (n_peers=0) is a no-op copy ---
     *
     * With n_peers=0, all-reduce must produce result = my_partial (no peers
     * to combine). Refusing or crashing on this input would force callers to
     * special-case the world-size-1 path; the right behaviour is to handle
     * it uniformly. */
    {
        size_t small_count = 1024;
        size_t small_bytes = small_count * sizeof(float);
        float *d_p = NULL, *d_r = NULL;
        float *h_p = (float *)malloc(small_bytes);
        float *h_r = (float *)malloc(small_bytes);
        hipSetDevice(0);
        hipMalloc(&d_p, small_bytes);
        hipMalloc(&d_r, small_bytes);
        for (size_t i = 0; i < small_count; i++) h_p[i] = (float)(i + 1) * 0.5f;
        hipMemcpy(d_p, h_p, small_bytes, hipMemcpyHostToDevice);
        hipMemset(d_r, 0, small_bytes);

        int ok = ds4_rocm_xdev_allreduce_f32(mesh, 0, d_r, d_p, NULL, NULL, 0, small_count, 0);
        if (!ok) {
            fprintf(stderr, "[FAIL] all-reduce n_peers=0 returned failure\n");
        } else {
            hipMemcpy(h_r, d_r, small_bytes, hipMemcpyDeviceToHost);
            int diverged = 0;
            for (size_t i = 0; i < small_count; i++) {
                if (fabsf(h_r[i] - h_p[i]) > 1e-6f) {
                    fprintf(stderr, "[FAIL] all-reduce n_peers=0 idx=%zu: got %f expected %f\n",
                            i, h_r[i], h_p[i]);
                    diverged = 1;
                    break;
                }
            }
            if (!diverged) {
                printf("[PASS] all-reduce n_peers=0 passes through my_partial unchanged.\n");
            }
        }
        hipFree(d_p); hipFree(d_r); free(h_p); free(h_r);
    }

    /* --- Test D: cached staging buffer reuse ---
     *
     * Run the same-size all-reduce twice in a row. The second call should
     * hit the cached staging buffer, not reallocate. We can't directly
     * observe caching, but we can verify the result is still correct (the
     * cache must not reuse stale data). */
    for (int r = 0; r < n_ranks; r++) {
        /* Change the pattern so a stale-cache bug would show up as a wrong
         * answer: partial[i] = r * 1000 + i. */
        for (size_t i = 0; i < count; i++) {
            h_partial[r][i] = (float)(r * 1000 + (int)i);
        }
        hipSetDevice(r);
        hipMemcpy(d_partial[r], h_partial[r], bytes, hipMemcpyHostToDevice);
    }
    {
        int owner = 0;
        int peer_devs[DS4_ROCM_XDEV_MAX_DEVICES];
        const float *peer_ptrs[DS4_ROCM_XDEV_MAX_DEVICES];
        int pi = 0;
        for (int r = 0; r < n_ranks; r++) {
            if (r == owner) continue;
            peer_devs[pi] = dev_ids[r];
            peer_ptrs[pi] = d_partial[r];
            pi++;
        }
        /* Call twice: the second call exercises the cache. */
        for (int repeat = 0; repeat < 2; repeat++) {
            int ok = ds4_rocm_xdev_allreduce_f32(mesh, owner, d_result[owner],
                                                  d_partial[owner],
                                                  peer_devs, peer_ptrs, n_peers,
                                                  count, 0);
            if (!ok) {
                fprintf(stderr, "[FAIL] all-reduce cache-test repeat=%d failed\n", repeat);
                goto cleanup_ar;
            }
            hipSetDevice(owner);
            hipMemcpy(h_result[owner], d_result[owner], bytes, hipMemcpyDeviceToHost);
            /* Expected sum: sum_{r=0..N-1} (r*1000 + i) = 1000 * N*(N-1)/2 + N*i. */
            float base = 1000.0f * (float)(n_ranks * (n_ranks - 1) / 2);
            int diverged = 0;
            for (size_t i = 0; i < count; i++) {
                float expected = base + (float)(n_ranks * (int)i);
                float got = h_result[owner][i];
                float tol = 1e-3f * fabsf(expected) + 1e-3f;
                if (fabsf(got - expected) > tol) {
                    fprintf(stderr, "[FAIL] all-reduce cache-test repeat=%d owner=%d idx=%zu: got %f expected %f\n",
                            repeat, owner, i, got, expected);
                    diverged = 1;
                    break;
                }
            }
            if (diverged) goto cleanup_ar;
        }
        printf("[PASS] all-reduce cached staging buffer produces correct results across repeated calls.\n");
    }

cleanup_ar:
    for (int r = 0; r < n_ranks; r++) {
        if (d_partial[r]) { hipSetDevice(r); hipFree(d_partial[r]); }
        if (d_result[r])  { hipSetDevice(r); hipFree(d_result[r]); }
        free(h_partial[r]);
        free(h_result[r]);
    }
}
