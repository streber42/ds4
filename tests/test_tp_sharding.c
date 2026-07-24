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

    fprintf(stderr, "\ntest_tp_sharding: %d/%d checks passed (%d failed)\n",
            g_checks - g_failures, g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
