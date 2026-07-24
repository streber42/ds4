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

    printf("\n================================================================================\n");
    printf("ALL CROSS-DEVICE TRANSFER STANDALONE TESTS PASSED SUCCESSFULLY!\n");
    printf("================================================================================\n");
    return 0;
}
