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

static void test_bringup_mode_escape_hatch(void) {
    setenv("DS4_ROCM_TP_BRINGUP", "1", 1);
    int rc = ds4_gpu_tp_gate_encode(0, 0);
    CHECK(rc == 0, "Bring-up mode should return 0 (neutral return)");
    printf("[PASS] Bring-up mode escape hatch verified.\n");
}

int main(void) {
    printf("Running ROCm TP stub loud failure and bring-up tests...\n");
    test_loud_failure_by_default();
    test_bringup_mode_escape_hatch();
    printf("All ROCm TP stub tests passed!\n");
    return 0;
}
