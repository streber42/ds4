/* ROCm-build regression test for issue 09 (graceful refusal & pipeline
 * fallback): --cuda-tensor-parallel must refuse cleanly (nonzero rc, NULL
 * engine, no crash/hang) on configurations it cannot support, and the
 * refusal message must name the actual constraint rather than a generic
 * failure. Modeled on tests/test_engine_mgpu_refusal.c (the CUDA
 * multi-tier-refusal precedent this PRD asks new refusal tests to follow).
 *
 * Two refusals are covered:
 *
 *   1. Odd GPU count with --cuda-tensor-parallel requested. This check
 *      fires before the model is ever opened (ds4_engine_open_internal
 *      validates rank count immediately), so it needs no real GGUF -- a
 *      nonexistent path is enough to prove the refusal is reached first.
 *      Always runs when >= 1 ROCm device is visible.
 *
 *   2. Model-shape mismatch (--cuda-tensor-parallel against a non-DeepSeek
 *      model). This check runs after model_open + config_validate_model, so
 *      it needs a real, loadable, non-DeepSeek-family GGUF (e.g. GLM 5.2).
 *      Per the PRD's testing decisions, refusal tests should not require
 *      the full 80 GiB DeepSeek quant; this case instead requires
 *      DS4_TEST_NON_DEEPSEEK_MODEL to point at a small non-DeepSeek GGUF
 *      and skips cleanly (not a failure) when that is unset, since no such
 *      fixture ships in this repo.
 *
 * The cross-device-transport fallback path (peer + host-staging both
 * unavailable -> disable TP, keep pipeline layer-split) is not exercised
 * here: on real hardware neither failure mode is reachable without
 * corrupting driver state, so it is covered at the pure-decision-logic
 * level instead (ds4_rocm_xdev_tp_transport_ok in tests/test_rocm_xdev.cu),
 * consistent with the PRD's stance that the host-staging fallback path is
 * "unreachable from higher seams" by design.
 *
 * Both refusals checked here fire on plain configuration/shape validation
 * before any ROCm device or HIP API is touched, so this test links against
 * the ROCm-built engine objects but never calls a HIP API itself and needs
 * no GPU to be physically present -- it exercises the same code that would
 * run on real hardware, just the part of it that runs before hardware
 * matters. */

#include "ds4.h"
#include "ds4_gpu_mgpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);       \
            return 1;                                                       \
        }                                                                   \
    } while (0)

static int read_file_to_buf(const char *path, char **out_buf, long *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return 1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return 1; }
    long n = ftell(f);
    if (n < 0) { fclose(f); return 1; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return 1; }
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return 1; }
    size_t r = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[r] = '\0';
    *out_buf = buf;
    *out_len = (long)r;
    return 0;
}

/* Redirects stderr to cap_path, runs engine creation, restores stderr, and
 * reads the captured output back. Shared by both sub-tests below. */
static int run_capturing_stderr(
        const ds4_engine_options *opt,
        const ds4_gpu_config *cfg,
        int *out_rc,
        ds4_engine **out_engine,
        char **out_cap) {
    const char *cap_path = "/tmp/ds4_rocm_tp_refusal_stderr.log";
    (void)unlink(cap_path);
    fflush(stderr);
    int saved_stderr = dup(fileno(stderr));
    if (saved_stderr < 0) return 1;
    FILE *redir = freopen(cap_path, "w+", stderr);
    if (!redir) return 1;

    *out_rc = ds4_engine_create_with_gpu_config(out_engine, opt, cfg);

    fflush(stderr);
    FILE *sink = freopen("/dev/null", "w", stderr);
    if (!sink) return 1;
    int err_fd = fileno(stderr);
    if (err_fd >= 0) {
        (void)dup2(saved_stderr, err_fd);
        (void)close(saved_stderr);
    }

    long cap_len = 0;
    int read_rc = read_file_to_buf(cap_path, out_cap, &cap_len);
    (void)unlink(cap_path);
    return read_rc;
}

/* Sub-test 1: odd GPU count refuses before any model I/O happens. */
static int test_rank_count_refusal(void) {
    fprintf(stderr, "-- rank-count refusal --\n");

    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 3;
    cfg.device_indices[0] = 0;
    cfg.device_indices[1] = 1;
    cfg.device_indices[2] = 2;
    cfg.vram_bytes[0] = (size_t)8ull * 1024u * 1024u * 1024u;
    cfg.vram_bytes[1] = (size_t)8ull * 1024u * 1024u * 1024u;
    cfg.vram_bytes[2] = (size_t)8ull * 1024u * 1024u * 1024u;

    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    /* Deliberately nonexistent: the rank-count refusal must fire before
     * model_open ever touches this path. */
    opt.model_path = "/nonexistent/ds4-rocm-tp-refusal-test.gguf";
    opt.backend = DS4_BACKEND_CUDA; /* ROCm builds still identify as CUDA at runtime */
    opt.n_threads = 1;
    opt.cuda_tensor_parallel = true;

    int rc = 0;
    ds4_engine *engine = NULL;
    char *cap = NULL;
    CHECK(run_capturing_stderr(&opt, &cfg, &rc, &engine, &cap) == 0,
          "could not capture stderr");

    fprintf(stderr, "  engine_create_with_gpu_config -> rc=%d, engine=%p\n",
            rc, (void *)engine);
    fprintf(stderr, "  captured stderr:\n----\n%s\n----\n", cap ? cap : "(null)");

    CHECK(rc != 0, "odd GPU count + --cuda-tensor-parallel must refuse (nonzero rc)");
    CHECK(engine == NULL, "engine pointer must be NULL on refusal");
    CHECK(cap && strstr(cap, "--cuda-tensor-parallel") != NULL,
          "refusal message must name the flag responsible");
    CHECK(cap && strstr(cap, "even number of GPUs") != NULL,
          "refusal message must name the actual constraint (even GPU count), "
          "not a generic failure");
    /* Never opened the (nonexistent) model: proves the refusal fires ahead
     * of any model I/O, so no unsupported configuration can reach far
     * enough to silently compute wrong output. */
    CHECK(cap && strstr(cap, "nonexistent") == NULL &&
          strstr(cap, "No such file") == NULL,
          "refusal must fire before model_open, not after a failed model open");

    free(cap);
    fprintf(stderr, "-- rank-count refusal: PASS --\n");
    return 0;
}

/* Sub-test 2: model-shape mismatch. Requires DS4_TEST_NON_DEEPSEEK_MODEL to
 * point at a small, loadable, non-DeepSeek-family GGUF (e.g. GLM 5.2).
 * Skips cleanly (PASS, not FAIL) when unset, since no such fixture is
 * checked into this repo and downloading one is out of scope for a single
 * test run. */
static int test_model_shape_refusal(void) {
    fprintf(stderr, "-- model-shape refusal --\n");

    const char *model_path = getenv("DS4_TEST_NON_DEEPSEEK_MODEL");
    if (!model_path || !model_path[0]) {
        fprintf(stderr,
                "  skipping: DS4_TEST_NON_DEEPSEEK_MODEL not set (need a small "
                "non-DeepSeek-family GGUF, e.g. GLM 5.2, to exercise this path)\n");
        return 0;
    }

    ds4_gpu_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_gpus = 2;
    cfg.device_indices[0] = 0;
    cfg.device_indices[1] = 1;
    cfg.vram_bytes[0] = (size_t)8ull * 1024u * 1024u * 1024u;
    cfg.vram_bytes[1] = (size_t)8ull * 1024u * 1024u * 1024u;

    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = model_path;
    opt.backend = DS4_BACKEND_CUDA;
    opt.n_threads = 1;
    opt.cuda_tensor_parallel = true;

    int rc = 0;
    ds4_engine *engine = NULL;
    char *cap = NULL;
    CHECK(run_capturing_stderr(&opt, &cfg, &rc, &engine, &cap) == 0,
          "could not capture stderr");

    fprintf(stderr, "  engine_create_with_gpu_config -> rc=%d, engine=%p\n",
            rc, (void *)engine);
    fprintf(stderr, "  captured stderr:\n----\n%s\n----\n", cap ? cap : "(null)");

    CHECK(rc != 0, "non-DeepSeek model + --cuda-tensor-parallel must refuse");
    CHECK(engine == NULL, "engine pointer must be NULL on refusal");
    CHECK(cap && strstr(cap, "DeepSeek-V4-Flash") != NULL,
          "refusal message must name the specific supported model, not a "
          "generic failure");

    free(cap);
    fprintf(stderr, "-- model-shape refusal: PASS --\n");
    return 0;
}

int main(void) {
    if (test_rank_count_refusal() != 0) return 1;
    if (test_model_shape_refusal() != 0) return 1;

    fprintf(stderr, "test_engine_rocm_tp_refusal PASS\n");
    return 0;
}
