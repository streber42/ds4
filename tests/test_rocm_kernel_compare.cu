/* Kernel comparison scaffold (rocm-tensor-parallel issue 03b).
 *
 * A single entry point that runs one named GPU kernel against fixed,
 * reproducible inputs and compares its output to a CPU reference, reporting
 * where and by how much it diverged -- not just pass/fail.
 *
 * This is a prefactor for the ROCm tensor-parallel port
 * (.scratch/rocm-tensor-parallel/PRD.md): the end-to-end logits comparison
 * (tests/test_engine_correctness_harness.c) is the correctness gate, and
 * per-kernel tests are deliberately NOT a standing obligation for all ~31
 * tensor-parallel entry points. This scaffold is what makes that trade
 * affordable -- it is the tool used to (a) prove the porting approach sound
 * on the first kernel ported in each subsystem, and (b) localize an
 * end-to-end failure down to a single kernel on demand, without inventing a
 * bespoke comparison harness each time.
 *
 * Usage:
 *   ./tests/test_rocm_kernel_compare              run every registered case
 *   ./tests/test_rocm_kernel_compare --list        list registered cases
 *   ./tests/test_rocm_kernel_compare --kernel NAME run only NAME
 *
 * Adding a new kernel means adding one adapter function (build inputs,
 * invoke the kernel, compute the reference) and one line in KCMP_CASES
 * below. The harness itself -- input determinism, comparison, tolerance
 * reporting, CLI -- does not change per kernel.
 *
 * No tensor-parallel kernel is implemented yet (they fail loudly by design;
 * see ds4_rocm_tp_bringup.h), so this build demonstrates the scaffold on
 * ds4_gpu_rms_norm_plain_tensor and ds4_gpu_add_tensor -- two kernels
 * already used by the working non-tensor-parallel ROCm pipeline path. A
 * pass here proves the scaffold's own plumbing is correct and ready to be
 * pointed at tensor-parallel kernels as they land.
 *
 * Both demo kernels take plain device buffers with no model_map/weight
 * lookup, so this binary never loads a model or GGUF file.
 *
 * Inputs are generated with a fixed linear congruential generator (the same
 * scheme as tests/test_q4k_dot.c's fill_q4_K/fill_q8_K) rather than a
 * seeded system RNG, so a given case produces bit-identical inputs on every
 * run and every machine.
 */

#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Deterministic input generation ---- */

static uint32_t kcmp_lcg_next(uint32_t *state) {
    *state = *state * 1103515245u + 12345u;
    return *state;
}

static void kcmp_fill_f32(float *out, uint64_t n, uint32_t seed, float lo, float hi) {
    uint32_t s = seed;
    for (uint64_t i = 0; i < n; i++) {
        uint32_t r = kcmp_lcg_next(&s);
        float u = (float)(r >> 8) / (float)(1u << 24); /* [0, 1) */
        out[i] = lo + u * (hi - lo);
    }
}

/* ---- Comparison result: where and by how much a kernel diverged ---- */

typedef struct {
    int      ok;               /* GPU call + allocation plumbing succeeded */
    int      pass;              /* numeric comparison passed within tol */
    uint64_t n;
    uint64_t divergent_at;       /* index of first |ref-got| > tol, or n if none */
    float    max_abs_err;
    float    ref_at_divergence;
    float    got_at_divergence;
} kcmp_result;

static kcmp_result kcmp_compare_f32(const float *ref, const float *got, uint64_t n, float tol) {
    kcmp_result r;
    memset(&r, 0, sizeof(r));
    r.ok = 1;
    r.n = n;
    r.divergent_at = n;
    for (uint64_t i = 0; i < n; i++) {
        float e = fabsf(ref[i] - got[i]);
        if (e > r.max_abs_err) r.max_abs_err = e;
        if (e > tol && r.divergent_at == n) {
            r.divergent_at = i;
            r.ref_at_divergence = ref[i];
            r.got_at_divergence = got[i];
        }
    }
    r.pass = (r.divergent_at == n);
    return r;
}

static kcmp_result kcmp_fail(const char *why) {
    kcmp_result r;
    memset(&r, 0, sizeof(r));
    fprintf(stderr, "kcmp: %s\n", why);
    return r; /* ok=0, pass=0 */
}

/* ============================================================================
 * Kernel cases. Each adapter builds fixed inputs, invokes the named GPU
 * kernel, computes the CPU reference, and hands both to kcmp_compare_f32.
 * ============================================================================
 */

/* ds4_gpu_rms_norm_plain_tensor: out[i] = x[i] / sqrt(mean(x^2) + eps). */
static kcmp_result kcmp_run_rms_norm_plain(void) {
    const uint64_t n = 4096;
    const float eps = 1e-5f;
    /* The kernel reduces sum-of-squares with a 256-wide float32 tree
     * reduction; the reference below sums in double precision, linearly.
     * Different reassociation means exact bit-match is not expected.
     * Output here is RMS-normalized (O(1) magnitude), so 1e-3 absolute is
     * comfortably above float32 rounding noise while still catching a real
     * formula error -- the same tolerance philosophy used by the end-to-end
     * logits harness (tests/test_engine_correctness_harness.c) relative to
     * its own output range. */
    const float tol = 1e-3f;

    float *x   = (float *)malloc(n * sizeof(float));
    float *ref = (float *)malloc(n * sizeof(float));
    float *got = (float *)malloc(n * sizeof(float));
    if (!x || !ref || !got) {
        free(x); free(ref); free(got);
        return kcmp_fail("out of memory");
    }

    kcmp_fill_f32(x, n, /*seed=*/0x5eed0001u, -2.0f, 2.0f);

    double sumsq = 0.0;
    for (uint64_t i = 0; i < n; i++) sumsq += (double)x[i] * (double)x[i];
    double scale = 1.0 / sqrt(sumsq / (double)n + (double)eps);
    for (uint64_t i = 0; i < n; i++) ref[i] = (float)((double)x[i] * scale);

    ds4_gpu_tensor *in_t  = ds4_gpu_tensor_alloc(n * sizeof(float));
    ds4_gpu_tensor *out_t = ds4_gpu_tensor_alloc(n * sizeof(float));
    kcmp_result r;
    if (!in_t || !out_t) {
        r = kcmp_fail("tensor alloc failed");
    } else if (!ds4_gpu_tensor_write(in_t, 0, x, n * sizeof(float))) {
        r = kcmp_fail("tensor write failed");
    } else if (!ds4_gpu_rms_norm_plain_tensor(out_t, in_t, (uint32_t)n, eps)) {
        r = kcmp_fail("ds4_gpu_rms_norm_plain_tensor launch failed");
    } else if (!ds4_gpu_tensor_read(out_t, 0, got, n * sizeof(float))) {
        r = kcmp_fail("tensor read failed");
    } else {
        r = kcmp_compare_f32(ref, got, n, tol);
    }

    if (in_t)  ds4_gpu_tensor_free(in_t);
    if (out_t) ds4_gpu_tensor_free(out_t);
    free(x); free(ref); free(got);
    return r;
}

/* ds4_gpu_add_tensor: out[i] = a[i] + b[i]. A second, unrelated kernel
 * reusing the exact same plumbing as the case above -- concrete proof that
 * pointing this scaffold at a different kernel needs only a new adapter,
 * never a new comparison mechanism. */
static kcmp_result kcmp_run_add(void) {
    const uint64_t n = 8192;
    /* A single IEEE-754 float addition has nothing to reassociate, so the
     * GPU and CPU results are expected to match exactly; the tiny nonzero
     * tolerance only guards against a differing FMA-contraction default
     * between hipcc and the host compiler. */
    const float tol = 1e-6f;

    float *a   = (float *)malloc(n * sizeof(float));
    float *b   = (float *)malloc(n * sizeof(float));
    float *ref = (float *)malloc(n * sizeof(float));
    float *got = (float *)malloc(n * sizeof(float));
    if (!a || !b || !ref || !got) {
        free(a); free(b); free(ref); free(got);
        return kcmp_fail("out of memory");
    }

    kcmp_fill_f32(a, n, 0xa11ce001u, -100.0f, 100.0f);
    kcmp_fill_f32(b, n, 0xb0b00002u, -100.0f, 100.0f);
    for (uint64_t i = 0; i < n; i++) ref[i] = a[i] + b[i];

    ds4_gpu_tensor *a_t   = ds4_gpu_tensor_alloc(n * sizeof(float));
    ds4_gpu_tensor *b_t   = ds4_gpu_tensor_alloc(n * sizeof(float));
    ds4_gpu_tensor *out_t = ds4_gpu_tensor_alloc(n * sizeof(float));
    kcmp_result r;
    if (!a_t || !b_t || !out_t) {
        r = kcmp_fail("tensor alloc failed");
    } else if (!ds4_gpu_tensor_write(a_t, 0, a, n * sizeof(float)) ||
               !ds4_gpu_tensor_write(b_t, 0, b, n * sizeof(float))) {
        r = kcmp_fail("tensor write failed");
    } else if (!ds4_gpu_add_tensor(out_t, a_t, b_t, (uint32_t)n)) {
        r = kcmp_fail("ds4_gpu_add_tensor launch failed");
    } else if (!ds4_gpu_tensor_read(out_t, 0, got, n * sizeof(float))) {
        r = kcmp_fail("tensor read failed");
    } else {
        r = kcmp_compare_f32(ref, got, n, tol);
    }

    if (a_t)   ds4_gpu_tensor_free(a_t);
    if (b_t)   ds4_gpu_tensor_free(b_t);
    if (out_t) ds4_gpu_tensor_free(out_t);
    free(a); free(b); free(ref); free(got);
    return r;
}

/* ---- Registry: the one place a new kernel case gets wired in ---- */

typedef struct {
    const char *name;
    const char *description;
    kcmp_result (*run)(void);
} kcmp_case;

static const kcmp_case KCMP_CASES[] = {
    { "rms_norm_plain",
      "ds4_gpu_rms_norm_plain_tensor vs double-precision RMSNorm reference (n=4096)",
      kcmp_run_rms_norm_plain },
    { "add",
      "ds4_gpu_add_tensor vs elementwise float addition (n=8192)",
      kcmp_run_add },
};
#define KCMP_N_CASES (sizeof(KCMP_CASES) / sizeof(KCMP_CASES[0]))

static void kcmp_print_report(const kcmp_case *c, const kcmp_result *r) {
    if (!r->ok) {
        printf("[FAIL] %-16s %s -- kernel call failed (see stderr)\n", c->name, c->description);
        return;
    }
    if (r->pass) {
        printf("[PASS] %-16s max_abs_err=%.6g over %llu elements -- %s\n",
               c->name, (double)r->max_abs_err, (unsigned long long)r->n, c->description);
    } else {
        printf("[FAIL] %-16s %s\n", c->name, c->description);
        printf("       diverged at index %llu / %llu: ref=%.6f got=%.6f |diff|=%.6f (max_abs_err=%.6g)\n",
               (unsigned long long)r->divergent_at, (unsigned long long)r->n,
               (double)r->ref_at_divergence, (double)r->got_at_divergence,
               (double)fabsf(r->ref_at_divergence - r->got_at_divergence),
               (double)r->max_abs_err);
    }
}

static void kcmp_list(void) {
    for (size_t i = 0; i < KCMP_N_CASES; i++) {
        printf("%-16s %s\n", KCMP_CASES[i].name, KCMP_CASES[i].description);
    }
}

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s [--list] [--kernel NAME]\n"
        "  (no args)      run every registered kernel case\n"
        "  --list         list registered kernel cases and exit\n"
        "  --kernel NAME  run only the named case\n",
        prog);
}

int main(int argc, char **argv) {
    const char *only = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--list")) { kcmp_list(); return 0; }
        if (!strcmp(argv[i], "--kernel") && i + 1 < argc) { only = argv[++i]; continue; }
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        usage(argv[0]);
        return 2;
    }

    if (!ds4_gpu_init()) {
        fprintf(stderr, "kcmp: ds4_gpu_init failed\n");
        return 1;
    }

    int ran = 0, failed = 0;
    for (size_t i = 0; i < KCMP_N_CASES; i++) {
        if (only && strcmp(only, KCMP_CASES[i].name) != 0) continue;
        kcmp_result r = KCMP_CASES[i].run();
        kcmp_print_report(&KCMP_CASES[i], &r);
        ran++;
        if (!r.ok || !r.pass) failed++;
    }

    ds4_gpu_cleanup();

    if (only && ran == 0) {
        fprintf(stderr, "kcmp: no kernel case named '%s' (see --list)\n", only);
        return 2;
    }

    printf("\n%d/%d kernel comparisons passed\n", ran - failed, ran);
    return failed ? 1 : 0;
}
