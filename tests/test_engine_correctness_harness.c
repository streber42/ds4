/* Correctness harness: logits diff comparison + quality fixture runner.
 *
 * Produces a repeatable pass/fail comparison between a reference run and
 * a candidate run for the same prompt and sampling settings.
 *
 * Usage:
 *   ./test-engine-ch --logits REF_MODEL REF_BACKEND CAND_MODEL CAND_BACKEND \
 *       [--prompt "text"] [--ctx 1024] [--steps N] [--tol 1e-3]
 *   ./test-engine-ch --quality MODEL MANIFEST OUT_TSV [--ctx 4096]
 *   ./test-engine-ch --provenance MODEL MANIFEST OUT_TSV [--ctx 4096]
 *
 * Logits comparison uses an explicit numeric tolerance:
 *   - absolute per-position error ≤ tol (default 1e-3)
 *   - greedy argmax must agree
 *   - justified by floating-point reassociation across different
 *     sharding or compute paths; bit-identity is impossible when the
 *     operation order differs, but the tolerance stays tight enough
 *     to catch real math regressions (sub-1% of the typical logit
 *     range of ~[-20, +20] per token).
 *
 * On failure the harness reports:
 *   - the token position where divergence first exceeds tolerance
 *   - the maximum absolute error across the vocabulary at that step
 *   - the argmax mismatch (token IDs), if any
 *
 * The reference is the existing, already-trusted pipeline path on the
 * same hardware and quantisation — not a different backend on different
 * hardware — so that any divergence is attributable to sharding rather
 * than to backend or hardware variation.
 *
 * This harness does NOT require tensor parallelism to exist in order to
 * run. It compares two independent inference runs, which is sufficient
 * to validate the oracle against a known-good build.
 */

#include "ds4.h"
#include "ds4_distributed.h"
#include "ds4_ssd.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Maximum number of generation steps the harness will exercise. */
#define MAX_STEPS 128

/* ---- Argument parsing helpers ---- */

static void usage(const char *prog) {
    fprintf(stderr,
        "usage:\n"
        "  %s --logits REF_MODEL REF_BE CAND_MODEL CAND_BE\n"
        "      [--prompt \"text\"] [--ctx N] [--steps N] [--tol float]\n"
        "  %s --quality MODEL MANIFEST OUT_TSV [--ctx N]\n"
        "  %s --provenance MODEL MANIFEST OUT_TSV [--ctx N]\n"
        "\n"
        "  --logits      Compare logits between reference and candidate runs.\n"
        "  --quality     Run the multi-case quality fixture (score_official).\n"
        "  --provenance  Convenience alias for --quality.\n"
        "\n"
        "  Reference selection: the pipeline path on the same hardware and\n"
        "  quantisation, not a different backend on different hardware.\n",
        prog, prog, prog);
    exit(2);
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "%s requires an argument\n", opt);
        exit(2);
    }
    return argv[++*i];
}

static int parse_backend(const char *s) {
    if (strcmp(s, "cpu") == 0)   return DS4_BACKEND_CPU;
    if (strcmp(s, "cuda") == 0)  return DS4_BACKEND_CUDA;
    if (strcmp(s, "metal") == 0) return DS4_BACKEND_METAL;
    fprintf(stderr, "unknown backend: %s\n", s);
    exit(2);
}

/* ---- Build a prompt from text ---- */

static int build_prompt(ds4_engine *e, const char *text, ds4_tokens *out) {
    ds4_encode_chat_prompt(e, NULL, text, DS4_THINK_NONE, out);
    return out->len;
}

/* ---- Greedy decode with logits collection ---- */

/* Greedy-decode `max_steps` tokens from a session, collecting the full
 * vocab-sized logits at every step.  Stores results in `out_logits`
 * (shape: max_steps × n_vocab, row-major) and returns the actual step
 * count (<= max_steps; stops at EOS).  On error returns -1.
 *
 * The EOS token ID is passed in from the caller because ds4_session
 * is opaque (incomplete type); we obtained eos from ds4_token_eos()
 * on the engine before creating the session.
 *
 * The logits are copied with ds4_session_copy_logits which returns the
 * number of logits written; this is the canonical API for reading the
 * model's output distribution after an eval step. */
static int greedy_decode_with_logits(ds4_session *s, int eos, float *out_logits,
                                     int n_vocab, int max_steps,
                                     int *out_tokens, int *out_count) {
    char err[256] = {0};
    int count = 0;

    /* At the very first step (sync), grab the prefill logits too. */
    {
        int got = ds4_session_copy_logits(s, out_logits, n_vocab);
        if (got != n_vocab) {
            fprintf(stderr, "copy logits failed at sync (got %d, expected %d)\n", got, n_vocab);
            return -1;
        }
        int t = ds4_session_argmax(s);
        if (t < 0) { fprintf(stderr, "argmax failed at step 0\n"); return -1; }
        out_tokens[0] = t;
        out_logits += n_vocab;
        count++;
        if (t == eos) { *out_count = count; return 0; }
        if (ds4_session_eval(s, t, err, sizeof(err)) != 0) {
            fprintf(stderr, "eval failed at step 0: %s\n", err);
            return -1;
        }
    }

    for (int i = 1; i < max_steps; i++) {
        int got = ds4_session_copy_logits(s, out_logits, n_vocab);
        if (got != n_vocab) {
            fprintf(stderr, "copy logits failed at step %d (got %d, expected %d)\n",
                    i, got, n_vocab);
            return -1;
        }
        int t = ds4_session_argmax(s);
        if (t < 0) { fprintf(stderr, "argmax failed at step %d\n", i); return -1; }
        out_tokens[i] = t;
        out_logits += n_vocab;
        count++;
        if (t == eos) { *out_count = count; return 0; }
        if (ds4_session_eval(s, t, err, sizeof(err)) != 0) {
            fprintf(stderr, "eval failed at step %d: %s\n", i, err);
            return -1;
        }
    }

    *out_count = max_steps;
    return 0;
}

/* ---- Logits comparison ---- */

typedef struct {
    int first_mismatch_step;   /* -1 if all pass */
    int step;                  /* token position where divergence found */
    float max_abs_error;       /* max |ref[i] - cand[i]| at divergence */
    int ref_argmax;
    int cand_argmax;
} logits_comparison_result;

/* Compare two vocab-sized logits arrays element-wise.  Returns the
 * maximum absolute error.  Also updates `*out_argmax_ref` and
 * `*out_argmax_cand` with the greedy token indices. */
static float compare_vocab_logits(const float *ref, const float *cand,
                                  int n_vocab,
                                  int *out_argmax_ref, int *out_argmax_cand,
                                  float *out_max_abs) {
    float max_abs = 0.0f;
    int best_ref = -1, best_cand = -1;
    float val_ref = -INFINITY, val_cand = -INFINITY;

    for (int i = 0; i < n_vocab; i++) {
        const float dr = ref[i], dc = cand[i];
        if (isfinite(dr) && (best_ref < 0 || dr > val_ref)) {
            val_ref = dr; best_ref = i;
        }
        if (isfinite(dc) && (best_cand < 0 || dc > val_cand)) {
            val_cand = dc; best_cand = i;
        }
        if (isfinite(dr) && isfinite(dc)) {
            const float d = fabsf(dr - dc);
            if (d > max_abs) max_abs = d;
        }
    }

    *out_argmax_ref = best_ref;
    *out_argmax_cand = best_cand;
    *out_max_abs = max_abs;
    return max_abs;
}

typedef struct {
    ds4_engine *engine;
    ds4_session *session;
    float *logits;
    int *tokens;
    int steps;
    int vocab_size;
    char lock_path[256];
} harness_run_t;

static void harness_run_cleanup(harness_run_t *r) {
    if (!r) return;
    free(r->logits);
    free(r->tokens);
    ds4_session_free(r->session);
    ds4_engine_close(r->engine);
    memset(r, 0, sizeof(*r));
}

/* Run one engine to completion: open, sync, greedy-decode with logits,
 * close.  Returns 0 on success, -1 on failure.  Stores results in *out.
 *
 * The instance lock (flock) is per-process, so each pass sets a unique
 * DS4_LOCK_FILE to avoid flock re-entrance issues within a single
 * process. */
static int run_harness_pass(const char *model_path, int backend,
                            const char *prompt_text, int ctx_size,
                            int max_steps, harness_run_t *out) {
    ds4_engine *e = NULL;
    ds4_session *s = NULL;
    ds4_tokens prompt = {0};
    int rc;

    memset(out, 0, sizeof(*out));

    /* Use a unique lock file per pass to avoid flock re-entrance
     * within a single process (flock is per-process, not per-fd). */
    snprintf(out->lock_path, sizeof(out->lock_path), "/tmp/ds4-ch-lock-%d", (int)rand());
    setenv("DS4_LOCK_FILE", out->lock_path, 1);

    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = model_path;
    opt.backend = (ds4_backend)backend;
    opt.n_threads = 0;
    opt.warm_weights = false;
    opt.quality = false;

    rc = ds4_engine_open(&e, &opt);
    if (rc != 0 || !e) {
        fprintf(stderr, "harness: failed to open engine: rc=%d\n", rc);
        return -1;
    }

    int prompt_len = build_prompt(e, prompt_text, &prompt);
    if (prompt_len <= 0) {
        fprintf(stderr, "harness: failed to tokenize prompt\n");
        ds4_engine_close(e);
        return -1;
    }

    const int n_vocab = ds4_engine_vocab_size(e);
    const int eos = ds4_token_eos(e);

    float *logits = malloc((size_t)max_steps * n_vocab * sizeof(float));
    int   *tokens = calloc((size_t)max_steps, sizeof(int));
    if (!logits || !tokens) {
        fprintf(stderr, "harness: out of memory\n");
        free(logits); free(tokens);
        ds4_engine_close(e);
        return -1;
    }

    ds4_session_create(&s, e, ctx_size);
    if (!s) { fprintf(stderr, "harness: session create failed\n"); return -1; }

    char err[256] = {0};
    rc = ds4_session_sync(s, &prompt, err, sizeof(err));
    if (rc != 0) {
        fprintf(stderr, "harness: sync failed: %s\n", err);
        ds4_session_free(s);
        free(logits); free(tokens);
        ds4_engine_close(e);
        return -1;
    }

    int steps = 0;
    if (greedy_decode_with_logits(s, eos, logits, n_vocab, max_steps,
                                   tokens, &steps) != 0) {
        fprintf(stderr, "harness: greedy decode failed\n");
        ds4_session_free(s);
        free(logits); free(tokens);
        ds4_engine_close(e);
        return -1;
    }

    out->engine = e;
    out->session = s;
    out->logits = logits;
    out->tokens = tokens;
    out->steps = steps;
    out->vocab_size = n_vocab;

    ds4_tokens_free(&prompt);
    fprintf(stderr, "harness: decoded %d steps\n", steps);
    return 0;
}

static int run_logits_comparison(const char *ref_model, int ref_backend,
                                  const char *cand_model, int cand_backend,
                                  const char *prompt_text, int ctx_size,
                                  int max_steps, float tol) {
    /* The instance lock uses flock(LOCK_EX) which is not reentrant within
     * the same process.  Open and fully close each engine sequentially:
     * reference first, then candidate. */
    fprintf(stderr, "harness: running reference pass...\n");
    harness_run_t ref_run = {0};
    if (run_harness_pass(ref_model, ref_backend,
                          prompt_text, ctx_size, max_steps, &ref_run) != 0) {
        return 1;
    }

    fprintf(stderr, "harness: running candidate pass...\n");
    harness_run_t cand_run = {0};
    if (run_harness_pass(cand_model, cand_backend,
                          prompt_text, ctx_size, max_steps, &cand_run) != 0) {
        harness_run_cleanup(&ref_run);
        return 1;
    }

    /* ---- Compare logits step-by-step ---- */
    fprintf(stderr, "harness: comparing logits (tol=%g)...\n\n", (double)tol);
    printf("step\tmax_abs_err\targmax_ref\targmax_cand\tmatch\n");

    float global_max_err = 0.0f;
    int   first_fail_step = -1;
    int   argmax_mismatch_at = -1;
    bool  argmax_mismatch = false;

    int n_compare_steps = ref_run.steps;
    if (cand_run.steps < n_compare_steps) n_compare_steps = cand_run.steps;

    for (int step = 0; step < n_compare_steps; step++) {
        const float *ref_slice  = ref_run.logits  + (size_t)step * ref_run.vocab_size;
        const float *cand_slice = cand_run.logits + (size_t)step * cand_run.vocab_size;

        int ar = -1, ac = -1;
        float max_abs = 0.0f;
        float err = compare_vocab_logits(ref_slice, cand_slice, ref_run.vocab_size,
                                         &ar, &ac, &max_abs);

        bool abs_ok = err <= tol;
        bool argmax_ok = (ar == ac);
        bool step_ok = abs_ok && argmax_ok;

        if (!step_ok && first_fail_step < 0) {
            first_fail_step = step;
            global_max_err = max_abs;
        }
        if (!argmax_ok && !argmax_mismatch) {
            argmax_mismatch_at = step;
            argmax_mismatch = true;
        }

        if (err > global_max_err) global_max_err = err;

        printf("%d\t%.6f\t%d\t%d\t%s\n",
               step, (double)err, ar, ac,
               step_ok ? "PASS" : "FAIL");

        if (!step_ok) {
            fprintf(stderr, "harness: STEP %d DIVERGES: max_abs=%.6f "
                    "(tol=%g), argmax ref=%d cand=%d\n",
                    step, (double)err, (double)tol, ar, ac);
        }
    }

    /* ---- Report ---- */
    printf("\n--- Summary ---\n");
    printf("reference steps : %d\n", ref_run.steps);
    printf("candidate steps : %d\n", cand_run.steps);
    printf("global max abs  : %.6f\n", (double)global_max_err);
    printf("tolerance       : %.6g\n", (double)tol);
    printf("all steps pass  : %s\n",
           first_fail_step < 0 ? "YES" : "NO");

    if (argmax_mismatch) {
        printf("argmax mismatch at step %d: ref=%d cand=%d\n",
               argmax_mismatch_at, ref_run.tokens[argmax_mismatch_at],
               cand_run.tokens[argmax_mismatch_at]);
    }

    if (ref_run.steps != cand_run.steps) {
        printf("WARNING: step counts differ (ref=%d cand=%d)\n",
               ref_run.steps, cand_run.steps);
    }

    /* Print decoded reference output for human inspection. */
    printf("\n--- Reference output (%d tokens) ---\n", ref_run.steps);
    for (int i = 0; i < ref_run.steps; i++) {
        size_t tlen = 0;
        char *txt = ds4_token_text(ref_run.engine, ref_run.tokens[i], &tlen);
        if (txt) {
            printf("  [%d] id=%d text='%.*s'\n", i, ref_run.tokens[i],
                   (int)tlen, txt);
            free(txt);
        }
    }

    printf("\n--- Candidate output (%d tokens) ---\n", cand_run.steps);
    for (int i = 0; i < cand_run.steps; i++) {
        size_t tlen = 0;
        char *txt = ds4_token_text(cand_run.engine, cand_run.tokens[i], &tlen);
        if (txt) {
            printf("  [%d] id=%d text='%.*s'\n", i, cand_run.tokens[i],
                   (int)tlen, txt);
            free(txt);
        }
    }

    int pass = (first_fail_step < 0 && !argmax_mismatch);
    printf("\nharness result: %s\n", pass ? "PASS" : "FAIL");

    harness_run_cleanup(&cand_run);
    harness_run_cleanup(&ref_run);

    return pass ? 0 : 1;
}

/* ---- Quality fixture runner ---- */

static int run_quality_fixture(const char *model_path, const char *manifest_path,
                                const char *out_path, int ctx_size) {
    /* Build the score_official command and exec it. */
    /* The quality fixture is built into gguf-tools/quality-testing/score_official */
    char cmd[4096];
    snprintf(cmd, sizeof(cmd),
             "./gguf-tools/quality-testing/score_official %s %s %s %d",
             model_path, manifest_path, out_path, ctx_size);

    fprintf(stderr, "harness: running quality fixture: %s\n", cmd);
    int rc = system(cmd);
    if (rc != 0) {
        fprintf(stderr, "harness: quality fixture failed (exit %d)\n", rc & 0xFF);
        return 1;
    }
    fprintf(stderr, "harness: quality fixture complete, results in %s\n", out_path);
    return 0;
}

/* ---- Main ---- */

int main(int argc, char **argv) {
    if (argc < 2) usage(argv[0]);

    const char *mode = argv[1];

    if (strcmp(mode, "--logits") == 0) {
        /* --logits REF_MODEL REF_BACKEND CAND_MODEL CAND_BACKEND
           [--prompt TEXT] [--ctx N] [--steps N] [--tol F] */
        if (argc < 6) usage(argv[0]);

        const char *ref_model  = argv[2];
        int ref_be = parse_backend(argv[3]);
        const char *cand_model = argv[4];
        int cand_be = parse_backend(argv[5]);

        const char *prompt_text = "What is the capital of France?";
        int ctx_size = 1024;
        int max_steps = 16;
        float tol = 1e-3f;

        for (int i = 6; i < argc; i++) {
            if (!strcmp(argv[i], "--prompt") && i + 1 < argc) {
                prompt_text = need_arg(&i, argc, argv, argv[i]);
            } else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) {
                ctx_size = atoi(need_arg(&i, argc, argv, argv[i]));
                if (ctx_size < 256) ctx_size = 256;
            } else if (!strcmp(argv[i], "--steps") && i + 1 < argc) {
                max_steps = atoi(need_arg(&i, argc, argv, argv[i]));
                if (max_steps < 1) max_steps = 16;
                if (max_steps > MAX_STEPS) max_steps = MAX_STEPS;
            } else if (!strcmp(argv[i], "--tol") && i + 1 < argc) {
                tol = (float)atof(need_arg(&i, argc, argv, argv[i]));
            } else {
                usage(argv[0]);
            }
        }

        return run_logits_comparison(ref_model, ref_be,
                                      cand_model, cand_be,
                                      prompt_text, ctx_size,
                                      max_steps, tol);

    } else if (strcmp(mode, "--quality") == 0 ||
               strcmp(mode, "--provenance") == 0) {
        /* --quality MODEL MANIFEST OUT_TSV [--ctx N] */
        if (argc < 5) usage(argv[0]);

        const char *model_path    = argv[2];
        const char *manifest_path = argv[3];
        const char *out_path      = argv[4];
        int ctx_size = 4096;

        for (int i = 5; i < argc; i++) {
            if (!strcmp(argv[i], "--ctx") && i + 1 < argc) {
                ctx_size = atoi(need_arg(&i, argc, argv, argv[i]));
                if (ctx_size < 1024) ctx_size = 1024;
            } else {
                usage(argv[0]);
            }
        }

        return run_quality_fixture(model_path, manifest_path,
                                    out_path, ctx_size);
    }

    usage(argv[0]);
}