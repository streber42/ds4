/* test_tp_sharding — CPU unit tests for the tensor-parallel sharding policy.
 *
 * Tests the pure-logic ownership functions in ds4_tp_shard.h with no GPU
 * present and no model loaded.  Follows the pattern of test_engine_mgpu_placement.c.
 *
 * Scenarios:
 *   1. complete partition: every expert, head, and vocab row is owned by
 *      exactly one rank (union covers [0, N), intersection is empty).
 *   2. no gaps: iterating all elements finds exactly one owner per element.
 *   3. no overlaps: no element appears in more than one rank's range.
 *   4. uneven division: n_expert not divisible by rank_count; the last rank
 *      gets the remainder; all others get floor(N/R).
 *   5. single-rank degenerate: rank_count == 1 returns full ownership for all
 *      three dimensions.
 *   6. validation: ds4_tp_shard_valid rejects invalid configurations.
 *   7. two-rank exact: the two-rank (Flash) configuration matches the
 *      constants used by the engine (n_expert=256, n_head=64, n_vocab=129280).
 *  10. monotonic ownership: higher ranks always start higher.
 *  11. TP=4 aggregate config (Flash shape): ds4_tp_compute_shard_config()
 *      returns 32 heads, 64 experts, 32320 vocab, 1024 embd per rank and
 *      produces complete partitions across all four dimensions.
 *  12. TP=4 aggregate config (Pro shape): 32 heads, 96 experts, 32320 vocab,
 *      1792 embd per rank.
 *  13. TP=1 degenerate config: single rank owns every dimension fully.
 *  14. TP=4 strict-even division: every uneven-division variant returns -1.
 *  15. TP=4 invalid inputs: rank/world==0/rank>=world/zero dims all rejected.
 */

#include "../ds4_tp_shard.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, msg) do { \
    g_checks++; \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (line %d)\n", (msg), __LINE__); \
        g_failures++; \
    } \
} while (0)

/* ── Helper: verify that ranges for all ranks cover [0, n_total) exactly ──
 *
 * Allocates a uint8_t bitmap and marks each owned element once.  Fails if
 * any element is marked twice (overlap) or not at all (gap). */
static void check_complete_partition(
    uint32_t rank_count,
    uint32_t n_total,
    ds4_tp_shard_range (*fn)(uint32_t rank, uint32_t rank_count, uint32_t n),
    const char *dim_name)
{
    if (n_total == 0) return;

    uint8_t *seen = calloc(n_total, 1);
    if (!seen) {
        fprintf(stderr, "  SKIP: %s partition check — malloc failed\n", dim_name);
        return;
    }

    int overlap = 0;
    for (uint32_t r = 0; r < rank_count; r++) {
        ds4_tp_shard_range rng = fn(r, rank_count, n_total);

        /* Range must be within bounds. */
        char msg[128];
        snprintf(msg, sizeof(msg), "%s rank %u start in bounds", dim_name, r);
        CHECK(rng.start < n_total || rng.count == 0, msg);
        snprintf(msg, sizeof(msg), "%s rank %u end in bounds", dim_name, r);
        CHECK(rng.count == 0 || rng.start + rng.count <= n_total, msg);

        for (uint32_t i = rng.start; i < rng.start + rng.count; i++) {
            if (seen[i]) {
                overlap = 1;
            }
            seen[i]++;
        }
    }

    /* Check no overlaps. */
    char msg_overlap[64];
    snprintf(msg_overlap, sizeof(msg_overlap), "%s: no overlaps", dim_name);
    CHECK(!overlap, msg_overlap);

    /* Check no gaps: every element must be owned by exactly one rank. */
    int gap = 0;
    for (uint32_t i = 0; i < n_total; i++) {
        if (seen[i] != 1) { gap = 1; break; }
    }
    char msg_gap[64];
    snprintf(msg_gap, sizeof(msg_gap), "%s: no gaps (complete cover)", dim_name);
    CHECK(!gap, msg_gap);

    free(seen);
}

/* ── Test 1 + 2 + 3: two-rank complete partition (Flash model shape) ── */
static void test_two_rank_partition_flash(void) {
    fprintf(stderr, "RUN: test_two_rank_partition_flash\n");

    /* Flash model: n_expert=256, n_head=64, n_vocab=129280, rank_count=2. */
    const uint32_t n_expert = 256;
    const uint32_t n_head   = 64;
    const uint32_t n_vocab  = 129280;
    const uint32_t nranks   = 2;

    CHECK(ds4_tp_shard_valid(0, nranks, n_expert, n_head, n_vocab),
          "rank 0 config valid");
    CHECK(ds4_tp_shard_valid(1, nranks, n_expert, n_head, n_vocab),
          "rank 1 config valid");

    check_complete_partition(nranks, n_expert, ds4_tp_shard_experts, "experts(2-rank)");
    check_complete_partition(nranks, n_head,   ds4_tp_shard_heads,   "heads(2-rank)");
    check_complete_partition(nranks, n_vocab,  ds4_tp_shard_vocab_rows, "vocab(2-rank)");
}

/* ── Test 4: uneven expert division ── */
static void test_uneven_expert_division(void) {
    fprintf(stderr, "RUN: test_uneven_expert_division\n");

    /* 7 experts across 3 ranks: floor(7/3)=2, remainder=1 → [0,2), [2,4), [4,7). */
    const uint32_t n_expert = 7;
    const uint32_t nranks   = 3;

    /* Validate with dummy head/vocab counts that satisfy constraints. */
    CHECK(ds4_tp_shard_valid(0, nranks, n_expert, /*n_head=*/3, /*n_vocab=*/1),
          "uneven expert config valid");

    ds4_tp_shard_range r0 = ds4_tp_shard_experts(0, nranks, n_expert);
    ds4_tp_shard_range r1 = ds4_tp_shard_experts(1, nranks, n_expert);
    ds4_tp_shard_range r2 = ds4_tp_shard_experts(2, nranks, n_expert);

    /* Ranks 0 and 1 get floor(7/3)=2 each. */
    CHECK(r0.start == 0 && r0.count == 2, "uneven: rank 0 gets [0,2)");
    CHECK(r1.start == 2 && r1.count == 2, "uneven: rank 1 gets [2,4)");
    /* Last rank absorbs remainder: 7 - 4 = 3. */
    CHECK(r2.start == 4 && r2.count == 3, "uneven: rank 2 gets [4,7) — remainder");

    /* Verify complete partition. */
    check_complete_partition(nranks, n_expert, ds4_tp_shard_experts,
                             "experts(uneven 7/3)");
}

/* ── Test 4b: uneven vocab division ── */
static void test_uneven_vocab_division(void) {
    fprintf(stderr, "RUN: test_uneven_vocab_division\n");

    /* 10 vocab rows across 3 ranks: floor(10/3)=3, remainder=1 → [0,3), [3,6), [6,10). */
    const uint32_t n_vocab = 10;
    const uint32_t nranks  = 3;

    ds4_tp_shard_range r0 = ds4_tp_shard_vocab_rows(0, nranks, n_vocab);
    ds4_tp_shard_range r1 = ds4_tp_shard_vocab_rows(1, nranks, n_vocab);
    ds4_tp_shard_range r2 = ds4_tp_shard_vocab_rows(2, nranks, n_vocab);

    CHECK(r0.start == 0 && r0.count == 3, "uneven vocab: rank 0 gets [0,3)");
    CHECK(r1.start == 3 && r1.count == 3, "uneven vocab: rank 1 gets [3,6)");
    CHECK(r2.start == 6 && r2.count == 4, "uneven vocab: rank 2 gets [6,10) — remainder");

    check_complete_partition(nranks, n_vocab, ds4_tp_shard_vocab_rows,
                             "vocab(uneven 10/3)");
}

/* ── Test 5: single-rank degenerate (full ownership) ── */
static void test_single_rank_full_ownership(void) {
    fprintf(stderr, "RUN: test_single_rank_full_ownership\n");

    const uint32_t n_expert = 256;
    const uint32_t n_head   = 64;
    const uint32_t n_vocab  = 129280;

    CHECK(ds4_tp_shard_valid(0, 1, n_expert, n_head, n_vocab),
          "single-rank config valid");

    ds4_tp_shard_range re = ds4_tp_shard_experts(0, 1, n_expert);
    CHECK(re.start == 0 && re.count == n_expert,
          "single-rank: full expert ownership");

    ds4_tp_shard_range rh = ds4_tp_shard_heads(0, 1, n_head);
    CHECK(rh.start == 0 && rh.count == n_head,
          "single-rank: full head ownership");

    ds4_tp_shard_range rv = ds4_tp_shard_vocab_rows(0, 1, n_vocab);
    CHECK(rv.start == 0 && rv.count == n_vocab,
          "single-rank: full vocab ownership");
}

/* ── Test 5b: four-rank complete partition ── */
static void test_four_rank_partition(void) {
    fprintf(stderr, "RUN: test_four_rank_partition\n");

    /* Pro model: n_expert=384, n_head=128, n_vocab=129280, 4 ranks. */
    const uint32_t n_expert = 384;
    const uint32_t n_head   = 128;
    const uint32_t n_vocab  = 129280;
    const uint32_t nranks   = 4;

    for (uint32_t r = 0; r < nranks; r++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "rank %u of 4 valid", r);
        CHECK(ds4_tp_shard_valid(r, nranks, n_expert, n_head, n_vocab), msg);
    }

    check_complete_partition(nranks, n_expert, ds4_tp_shard_experts, "experts(4-rank)");
    check_complete_partition(nranks, n_head,   ds4_tp_shard_heads,   "heads(4-rank)");
    check_complete_partition(nranks, n_vocab,  ds4_tp_shard_vocab_rows, "vocab(4-rank)");
}

/* ── Test 6: ds4_tp_shard_valid rejection cases ── */
static void test_validation(void) {
    fprintf(stderr, "RUN: test_validation\n");

    /* rank_count == 0 */
    CHECK(!ds4_tp_shard_valid(0, 0, 256, 64, 129280),
          "validation: rank_count 0 rejected");

    /* rank >= rank_count */
    CHECK(!ds4_tp_shard_valid(2, 2, 256, 64, 129280),
          "validation: rank == rank_count rejected");
    CHECK(!ds4_tp_shard_valid(3, 2, 256, 64, 129280),
          "validation: rank > rank_count rejected");

    /* n_expert == 0 */
    CHECK(!ds4_tp_shard_valid(0, 2, 0, 64, 129280),
          "validation: n_expert 0 rejected");

    /* n_head == 0 */
    CHECK(!ds4_tp_shard_valid(0, 2, 256, 0, 129280),
          "validation: n_head 0 rejected");

    /* n_vocab == 0 */
    CHECK(!ds4_tp_shard_valid(0, 2, 256, 64, 0),
          "validation: n_vocab 0 rejected");

    /* n_head not divisible by rank_count — uneven head split is illegal */
    CHECK(!ds4_tp_shard_valid(0, 3, 256, 64, 129280),
          "validation: n_head (64) not divisible by rank_count (3) rejected");

    /* Legal cases. */
    CHECK(ds4_tp_shard_valid(0, 2, 256, 64, 129280),
          "validation: standard Flash 2-rank config accepted");
    CHECK(ds4_tp_shard_valid(1, 2, 256, 64, 129280),
          "validation: rank 1 Flash 2-rank config accepted");
    CHECK(ds4_tp_shard_valid(0, 1, 256, 64, 129280),
          "validation: single-rank config accepted");
}

/* ── Test 7: two-rank expert split matches Metal-side convention ──
 *
 * In weights_model_map_sharded_spans (ds4.c) the two-rank expert split is:
 *
 *   low_experts = x->dim[2] / 2  (floor division)
 *   rank 0: [0, low_experts)
 *   rank 1: [low_experts, n_expert)
 *
 * Verify ds4_tp_shard_experts produces the same ranges. */
static void test_two_rank_expert_matches_engine_convention(void) {
    fprintf(stderr, "RUN: test_two_rank_expert_matches_engine_convention\n");

    const uint32_t n_expert = 256; /* Flash model */
    const uint32_t low_experts = n_expert / 2u; /* 128 */

    ds4_tp_shard_range r0 = ds4_tp_shard_experts(0, 2, n_expert);
    ds4_tp_shard_range r1 = ds4_tp_shard_experts(1, 2, n_expert);

    CHECK(r0.start == 0 && r0.count == low_experts,
          "rank 0 owns [0, 128) matching engine convention");
    CHECK(r1.start == low_experts && r1.count == n_expert - low_experts,
          "rank 1 owns [128, 256) matching engine convention");
}

/* ── Test 8: two-rank head split matches engine convention ──
 *
 * In the Metal decode graph (ds4.c line ~21460):
 *
 *   tp_heads = DS4_N_HEAD / 2u  (when tp_world == 2)
 *   tp_head0 = g->tp_rank * tp_heads
 *
 * So rank k owns [k * (n_head/2), (k+1) * (n_head/2)).
 * Verify ds4_tp_shard_heads produces the same layout. */
static void test_two_rank_head_matches_engine_convention(void) {
    fprintf(stderr, "RUN: test_two_rank_head_matches_engine_convention\n");

    const uint32_t n_head = 64; /* Flash model */
    const uint32_t tp_heads = n_head / 2u; /* 32 */

    ds4_tp_shard_range r0 = ds4_tp_shard_heads(0, 2, n_head);
    ds4_tp_shard_range r1 = ds4_tp_shard_heads(1, 2, n_head);

    CHECK(r0.start == 0 && r0.count == tp_heads,
          "rank 0 heads [0, 32) matching engine convention");
    CHECK(r1.start == tp_heads && r1.count == tp_heads,
          "rank 1 heads [32, 64) matching engine convention");
}

/* ── Test 9: two-rank vocab split matches engine convention ──
 *
 * The Metal output head (ds4.c line ~24040):
 *
 *   tp_vhalf = vocab_dim / 2u
 *   rank k reads from g->tp_rank * tp_vhalf
 *
 * Verify ds4_tp_shard_vocab_rows produces the same layout. */
static void test_two_rank_vocab_matches_engine_convention(void) {
    fprintf(stderr, "RUN: test_two_rank_vocab_matches_engine_convention\n");

    const uint32_t n_vocab  = 129280; /* Flash / Pro */
    const uint32_t tp_vhalf = n_vocab / 2u; /* 64640 */

    ds4_tp_shard_range r0 = ds4_tp_shard_vocab_rows(0, 2, n_vocab);
    ds4_tp_shard_range r1 = ds4_tp_shard_vocab_rows(1, 2, n_vocab);

    CHECK(r0.start == 0 && r0.count == tp_vhalf,
          "rank 0 vocab [0, 64640) matching engine convention");
    CHECK(r1.start == tp_vhalf && r1.count == n_vocab - tp_vhalf,
          "rank 1 vocab [64640, 129280) matching engine convention");
}

/* ── Test 10: monotonic ownership — higher ranks always start higher ── */
static void test_monotonic_ownership(void) {
    fprintf(stderr, "RUN: test_monotonic_ownership\n");

    /* Use a range of configurations to verify rank k's start is always
     * strictly less than rank k+1's start (for count > 0). */
    struct { uint32_t n; uint32_t nranks; } cases[] = {
        {256, 2}, {256, 4}, {384, 4}, {7, 3}, {100, 3}, {1, 1}
    };

    for (size_t ci = 0; ci < sizeof(cases)/sizeof(cases[0]); ci++) {
        uint32_t n      = cases[ci].n;
        uint32_t nranks = cases[ci].nranks;

        for (uint32_t r = 0; r + 1 < nranks; r++) {
            ds4_tp_shard_range ra = ds4_tp_shard_experts(r,   nranks, n);
            ds4_tp_shard_range rb = ds4_tp_shard_experts(r+1, nranks, n);
            char msg[128];
            snprintf(msg, sizeof(msg),
                "experts monotonic: rank %u start < rank %u start (n=%u nranks=%u)",
                r, r+1, n, nranks);
            CHECK(ra.start < rb.start, msg);
        }

        /* Heads — use n as n_head only when n % nranks == 0. */
        if (n % nranks == 0) {
            for (uint32_t r = 0; r + 1 < nranks; r++) {
                ds4_tp_shard_range ra = ds4_tp_shard_heads(r,   nranks, n);
                ds4_tp_shard_range rb = ds4_tp_shard_heads(r+1, nranks, n);
                char msg[128];
                snprintf(msg, sizeof(msg),
                    "heads monotonic: rank %u start < rank %u start (n=%u nranks=%u)",
                    r, r+1, n, nranks);
                CHECK(ra.start < rb.start, msg);
            }
        }
    }
}

/* ── Test 11: TP=4 aggregate config — Flash-shape model ──
 *
 * Exercises ds4_tp_compute_shard_config() with the shape the issue
 * specifies (n_head=128, n_expert=256, n_vocab=129280, n_embd=4096).
 * Each rank should own 32 heads, 64 experts, 32320 vocab rows, and
 * 1024 embd columns.  Also verifies complete partition for all four
 * dimensions, including the new embd dimension. */
static void test_tp4_shard_config_flash(void) {
    fprintf(stderr, "RUN: test_tp4_shard_config_flash\n");

    const uint32_t n_head   = 128;
    const uint32_t n_expert = 256;
    const uint32_t n_vocab  = 129280;
    const uint32_t n_embd   = 4096;
    const uint32_t nranks   = 4;

    /* Expected per-rank counts (all even). */
    const uint32_t exp_heads   = 32;   /* 128 / 4 */
    const uint32_t exp_experts = 64;   /* 256 / 4 */
    const uint32_t exp_vocab   = 32320; /* 129280 / 4 */
    const uint32_t exp_embd    = 1024; /* 4096 / 4 */

    for (uint32_t r = 0; r < nranks; r++) {
        ds4_tp_shard_config cfg;
        int rc = ds4_tp_compute_shard_config(
            nranks, r, n_head, n_expert, n_vocab, n_embd, &cfg);
        char msg[128];
        snprintf(msg, sizeof(msg), "flash tp=4: rank %u config ok", r);
        CHECK(rc == 0, msg);

        snprintf(msg, sizeof(msg), "flash tp=4: rank %u heads count", r);
        CHECK(cfg.heads.count == exp_heads, msg);
        snprintf(msg, sizeof(msg), "flash tp=4: rank %u heads start", r);
        CHECK(cfg.heads.start == r * exp_heads, msg);

        snprintf(msg, sizeof(msg), "flash tp=4: rank %u experts count", r);
        CHECK(cfg.experts.count == exp_experts, msg);
        snprintf(msg, sizeof(msg), "flash tp=4: rank %u experts start", r);
        CHECK(cfg.experts.start == r * exp_experts, msg);

        snprintf(msg, sizeof(msg), "flash tp=4: rank %u vocab count", r);
        CHECK(cfg.vocab.count == exp_vocab, msg);
        snprintf(msg, sizeof(msg), "flash tp=4: rank %u vocab start", r);
        CHECK(cfg.vocab.start == r * exp_vocab, msg);

        snprintf(msg, sizeof(msg), "flash tp=4: rank %u embd count", r);
        CHECK(cfg.embd.count == exp_embd, msg);
        snprintf(msg, sizeof(msg), "flash tp=4: rank %u embd start", r);
        CHECK(cfg.embd.start == r * exp_embd, msg);
    }

    /* Complete partition for every dimension including embd. */
    check_complete_partition(nranks, n_head,   ds4_tp_shard_heads,     "heads(tp=4 flash)");
    check_complete_partition(nranks, n_expert, ds4_tp_shard_experts,   "experts(tp=4 flash)");
    check_complete_partition(nranks, n_vocab,  ds4_tp_shard_vocab_rows, "vocab(tp=4 flash)");
    check_complete_partition(nranks, n_embd,   ds4_tp_shard_embd_cols, "embd(tp=4 flash)");
}

/* ── Test 12: TP=4 aggregate config — Pro-shape model ── */
static void test_tp4_shard_config_pro(void) {
    fprintf(stderr, "RUN: test_tp4_shard_config_pro\n");

    const uint32_t n_head   = 128;
    const uint32_t n_expert = 384;
    const uint32_t n_vocab  = 129280;
    const uint32_t n_embd   = 7168;
    const uint32_t nranks   = 4;

    const uint32_t exp_heads   = 32;
    const uint32_t exp_experts = 96;   /* 384 / 4 */
    const uint32_t exp_vocab   = 32320;
    const uint32_t exp_embd    = 1792; /* 7168 / 4 */

    for (uint32_t r = 0; r < nranks; r++) {
        ds4_tp_shard_config cfg;
        int rc = ds4_tp_compute_shard_config(
            nranks, r, n_head, n_expert, n_vocab, n_embd, &cfg);
        char msg[128];
        snprintf(msg, sizeof(msg), "pro tp=4: rank %u config ok", r);
        CHECK(rc == 0, msg);
        snprintf(msg, sizeof(msg), "pro tp=4: rank %u experts=%u expected=%u",
                 r, cfg.experts.count, exp_experts);
        CHECK(cfg.experts.count == exp_experts, msg);
        snprintf(msg, sizeof(msg), "pro tp=4: rank %u heads=%u expected=%u",
                 r, cfg.heads.count, exp_heads);
        CHECK(cfg.heads.count == exp_heads, msg);
        snprintf(msg, sizeof(msg), "pro tp=4: rank %u vocab=%u expected=%u",
                 r, cfg.vocab.count, exp_vocab);
        CHECK(cfg.vocab.count == exp_vocab, msg);
        snprintf(msg, sizeof(msg), "pro tp=4: rank %u embd=%u expected=%u",
                 r, cfg.embd.count, exp_embd);
        CHECK(cfg.embd.count == exp_embd, msg);
    }
}

/* ── Test 13: TP=1 degenerate — single rank owns everything ──
 *
 * The issue explicitly calls out tp_world==1 as the degenerate case:
 * every dimension must have start==0 and count==N. */
static void test_tp1_shard_config_degenerate(void) {
    fprintf(stderr, "RUN: test_tp1_shard_config_degenerate\n");

    const uint32_t n_head   = 128;
    const uint32_t n_expert = 256;
    const uint32_t n_vocab  = 129280;
    const uint32_t n_embd   = 4096;

    ds4_tp_shard_config cfg;
    int rc = ds4_tp_compute_shard_config(
        1, 0, n_head, n_expert, n_vocab, n_embd, &cfg);
    CHECK(rc == 0, "tp=1: config ok");

    CHECK(cfg.heads.start == 0 && cfg.heads.count == n_head,
          "tp=1: full head ownership");
    CHECK(cfg.experts.start == 0 && cfg.experts.count == n_expert,
          "tp=1: full expert ownership");
    CHECK(cfg.vocab.start == 0 && cfg.vocab.count == n_vocab,
          "tp=1: full vocab ownership");
    CHECK(cfg.embd.start == 0 && cfg.embd.count == n_embd,
          "tp=1: full embd ownership");
}

/* ── Test 14: TP=4 strict-even division rejects uneven dimensions ──
 *
 * The issue's boundary case: when any dimension does not divide evenly
 * by tp_world, ds4_tp_compute_shard_config() must return -1 (not
 * silently mis-shard).  Tests each of the four dimensions independently,
 * plus the NULL-out sentinel. */
static void test_tp4_shard_config_uneven_division_error(void) {
    fprintf(stderr, "RUN: test_tp4_shard_config_uneven_division_error\n");

    ds4_tp_shard_config cfg;

    /* n_head not divisible by 4 (129 / 4 = 32 remainder 1). */
    CHECK(ds4_tp_compute_shard_config(4, 0, 129, 256, 129280, 4096, &cfg) == -1,
          "tp=4: n_head=129 rejected (not divisible by 4)");

    /* n_expert not divisible by 4 (257 / 4 = 64 remainder 1). */
    CHECK(ds4_tp_compute_shard_config(4, 0, 128, 257, 129280, 4096, &cfg) == -1,
          "tp=4: n_expert=257 rejected (not divisible by 4)");

    /* n_vocab not divisible by 4 (129281 / 4 = 32320 remainder 1). */
    CHECK(ds4_tp_compute_shard_config(4, 0, 128, 256, 129281, 4096, &cfg) == -1,
          "tp=4: n_vocab=129281 rejected (not divisible by 4)");

    /* n_embd not divisible by 4 (4097 / 4 = 1024 remainder 1). */
    CHECK(ds4_tp_compute_shard_config(4, 0, 128, 256, 129280, 4097, &cfg) == -1,
          "tp=4: n_embd=4097 rejected (not divisible by 4)");

    /* NULL out pointer. */
    CHECK(ds4_tp_compute_shard_config(4, 0, 128, 256, 129280, 4096, NULL) == -1,
          "tp=4: NULL out pointer rejected");
}

/* ── Test 15: TP=4 invalid rank / world combinations ── */
static void test_tp4_shard_config_invalid_inputs(void) {
    fprintf(stderr, "RUN: test_tp4_shard_config_invalid_inputs\n");

    ds4_tp_shard_config cfg;

    /* tp_world == 0. */
    CHECK(ds4_tp_compute_shard_config(0, 0, 128, 256, 129280, 4096, &cfg) == -1,
          "tp=4: world=0 rejected");

    /* tp_rank == tp_world (off the end). */
    CHECK(ds4_tp_compute_shard_config(4, 4, 128, 256, 129280, 4096, &cfg) == -1,
          "tp=4: rank==world rejected");

    /* tp_rank > tp_world. */
    CHECK(ds4_tp_compute_shard_config(4, 5, 128, 256, 129280, 4096, &cfg) == -1,
          "tp=4: rank>world rejected");

    /* n_head == 0. */
    CHECK(ds4_tp_compute_shard_config(4, 0, 0, 256, 129280, 4096, &cfg) == -1,
          "tp=4: n_head=0 rejected");

    /* n_expert == 0. */
    CHECK(ds4_tp_compute_shard_config(4, 0, 128, 0, 129280, 4096, &cfg) == -1,
          "tp=4: n_expert=0 rejected");

    /* n_vocab == 0. */
    CHECK(ds4_tp_compute_shard_config(4, 0, 128, 256, 0, 4096, &cfg) == -1,
          "tp=4: n_vocab=0 rejected");

    /* n_embd == 0. */
    CHECK(ds4_tp_compute_shard_config(4, 0, 128, 256, 129280, 0, &cfg) == -1,
          "tp=4: n_embd=0 rejected");
}

int main(void) {
    test_two_rank_partition_flash();
    test_uneven_expert_division();
    test_uneven_vocab_division();
    test_single_rank_full_ownership();
    test_four_rank_partition();
    test_validation();
    test_two_rank_expert_matches_engine_convention();
    test_two_rank_head_matches_engine_convention();
    test_two_rank_vocab_matches_engine_convention();
    test_monotonic_ownership();
    test_tp4_shard_config_flash();
    test_tp4_shard_config_pro();
    test_tp1_shard_config_degenerate();
    test_tp4_shard_config_uneven_division_error();
    test_tp4_shard_config_invalid_inputs();

    fprintf(stderr, "\ntest_tp_sharding: %d/%d checks passed (%d failed)\n",
            g_checks - g_failures, g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
