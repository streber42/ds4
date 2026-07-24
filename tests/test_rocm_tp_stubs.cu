#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <assert.h>

#include "ds4_rocm.h"
#include "ds4_gpu.h"

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

static void test_loud_failure_by_default(void) {
    unsetenv("DS4_ROCM_TP_BRINGUP");
    pid_t pid = fork();
    if (pid == 0) {
        // Child process: call unimplemented stub
        ds4_gpu_tp_gate_encode(0, 0);
        exit(0); // Should not be reached
    }
    int status = 0;
    waitpid(pid, &status, 0);
    // Child should have aborted (SIGABRT or exited non-zero)
    CHECK(WIFSIGNALED(status) || (WIFEXITED(status) && WEXITSTATUS(status) != 0),
          "Unimplemented stub should fail loudly (abort) by default");
    printf("[PASS] Loud failure by default verified.\n");
}

/* ds4_gpu_tp_gate_encode has a BOOLEAN "0 = failed, nonzero = succeeded"
 * contract at its ds4.c call site (`ok = ds4_gpu_tp_gate_encode(...) != 0`).
 * A neutral bring-up return must therefore be 1, not 0 -- returning 0 would
 * report failure and abort the very first forward pass it's supposed to let
 * through. Discovered by actually running a two-rank forward pass under
 * bring-up (see issue 04 Comments); ds4_rocm_tp_stub_ok is the fixed helper
 * used for every boolean-contract entry point. */
static void test_bringup_mode_escape_hatch(void) {
    setenv("DS4_ROCM_TP_BRINGUP", "1", 1);
    int rc = ds4_gpu_tp_gate_encode(0, 0);
    CHECK(rc == 1, "Bring-up mode should return 1 (neutral == reports success)");
    printf("[PASS] Bring-up mode escape hatch verified (boolean-contract entry point).\n");
}

/* ds4_gpu_device_cache_tensors has the opposite, errno-style "0 = success"
 * contract (`if (rc != 0) fail` at its call site) -- 0 is already the
 * correct neutral value there, so it stays on the plain stub and must NOT
 * be flipped to 1. */
static void test_bringup_mode_errno_style_stays_zero(void) {
    setenv("DS4_ROCM_TP_BRINGUP", "1", 1);
    int rc = ds4_gpu_device_cache_tensors(0, NULL, 0);
    CHECK(rc == 0, "Bring-up mode should keep 0 for errno-style (0 == success) entry points");
    printf("[PASS] Bring-up mode escape hatch verified (errno-contract entry point).\n");
}

int main(void) {
    printf("Running ROCm TP stub loud failure and bring-up tests...\n");
    test_loud_failure_by_default();
    test_bringup_mode_escape_hatch();
    test_bringup_mode_errno_style_stays_zero();
    printf("All ROCm TP stub tests passed!\n");
    return 0;
}
