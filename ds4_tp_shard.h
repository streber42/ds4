/* ds4_tp_shard.h — tensor-parallel sharding policy.
 *
 * Pure-logic ownership decisions for the ROCm tensor-parallel port.
 * No GPU dependency, no model state: all functions are deterministic
 * pure functions of the model dimensions, rank count, and rank index.
 *
 * Four sharded dimensions:
 *
 *   Routed experts — the full set of n_expert routed experts is split
 *     across ranks in contiguous ranges.  Rank 0 takes the lower
 *     experts; the last rank absorbs any remainder from uneven division
 *     in the per-dimension helper.  The aggregate
 *     ds4_tp_compute_shard_config() is stricter: it rejects uneven
 *     division of any dimension outright (see below).
 *
 *   Attention heads — the n_head query heads are divided evenly in half
 *     for the two-rank design, rank 0 owns [0, n_head/2), rank 1 owns
 *     the upper half.  Heads must be divisible by rank_count (asserted
 *     by ds4_tp_shard_valid).
 *
 *   Vocabulary rows — the output head vocab rows (n_vocab) are split
 *     across ranks in contiguous ranges.  Same remainder-to-last-rank
 *     policy as routed experts for the per-dimension helper; the
 *     aggregate config rejects uneven splits.
 *
 *   Embedding / FFN columns — the n_embd shared projection dimension
 *     (token embedding rows, shared-expert columns, output-projection
 *     rows) is row-sharded across ranks.  Same remainder-to-last-rank
 *     policy in the per-dimension helper; the aggregate config rejects
 *     uneven splits.
 *
 * Uneven division: the per-dimension helpers use floor(N/R) with the
 * remainder absorbed by the last rank.  The aggregate
 * ds4_tp_compute_shard_config() is stricter and requires every
 * dimension to divide evenly; this matches the TP=4 policy where
 * every target model shape divides without remainder.
 *
 * Single-rank degenerate case: rank_count == 1 returns full ownership for
 * all four dimensions (start == 0, count == N).
 *
 * Kernels and engine code must obtain ownership bounds only through
 * the shard helpers; they must not re-derive these indices inline.
 */

#ifndef DS4_TP_SHARD_H
#define DS4_TP_SHARD_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Ownership range [start, start + count).  All three query functions fill
 * one of these.  Callers must treat start and count as the authoritative
 * source of truth for what a rank computes; never derive them elsewhere. */
typedef struct {
    uint32_t start; /* first owned index (expert id / head index / vocab row) */
    uint32_t count; /* number of owned items; 0 is valid only for n_total == 0 */
} ds4_tp_shard_range;

/* Validate a sharding configuration.  Returns true when the parameters are
 * consistent and can be sharded without data loss:
 *
 *   rank_count >= 1
 *   rank < rank_count
 *   n_head divisible by rank_count (heads must be evenly divided;
 *     an uneven head split would produce incorrect attention output)
 *   n_expert >= 1 and n_vocab >= 1
 *
 * Call this once at engine init before invoking the range functions. */
static inline bool ds4_tp_shard_valid(
    uint32_t rank,
    uint32_t rank_count,
    uint32_t n_expert,
    uint32_t n_head,
    uint32_t n_vocab)
{
    if (rank_count == 0) return false;
    if (rank >= rank_count) return false;
    if (n_expert == 0) return false;
    if (n_head == 0) return false;
    if (n_vocab == 0) return false;
    /* Attention heads must divide evenly: an uneven split would leave some
     * heads uncomputed on the local rank while still reserving output space
     * for them, silently zeroing partial attention output. */
    if (n_head % rank_count != 0) return false;
    return true;
}

/* Routed-expert ownership for one rank.
 *
 * Division policy: each rank gets floor(n_expert / rank_count) experts;
 * the last rank absorbs the remainder so the partition is always complete.
 * Rank 0 owns the lowest expert ids; rank (rank_count - 1) owns the highest.
 *
 * Returns the range [start, start + count) of expert ids owned by `rank`.
 * For rank_count == 1 the range is [0, n_expert). */
static inline ds4_tp_shard_range ds4_tp_shard_experts(
    uint32_t rank,
    uint32_t rank_count,
    uint32_t n_expert)
{
    /* floor division gives the base per-rank quota. */
    const uint32_t base = n_expert / rank_count;
    /* Remainder experts go to the last rank so lower ranks are never larger
     * than the last, keeping the hot path (rank 0) lean. */
    const uint32_t start = rank * base;
    const uint32_t count = (rank == rank_count - 1u)
        ? n_expert - start   /* last rank takes all remaining */
        : base;
    ds4_tp_shard_range r;
    r.start = start;
    r.count = count;
    return r;
}

/* Attention-head ownership for one rank.
 *
 * Heads are always evenly divided (ds4_tp_shard_valid enforces
 * n_head % rank_count == 0).  Rank 0 owns [0, heads_per_rank);
 * rank k owns [k * heads_per_rank, (k+1) * heads_per_rank).
 *
 * Returns the range [start, start + count) of head indices for `rank`.
 * For rank_count == 1 the range is [0, n_head). */
static inline ds4_tp_shard_range ds4_tp_shard_heads(
    uint32_t rank,
    uint32_t rank_count,
    uint32_t n_head)
{
    const uint32_t heads_per_rank = n_head / rank_count;
    ds4_tp_shard_range r;
    r.start = rank * heads_per_rank;
    r.count = heads_per_rank;
    return r;
}

/* Vocabulary-row ownership for one rank (output head split).
 *
 * Division policy: same floor/remainder scheme as routed experts.
 * Each rank gets floor(n_vocab / rank_count) rows; the last rank absorbs
 * the remainder.  Rank 0 produces rows [0, low_rows); the last rank
 * produces [start, n_vocab).
 *
 * Returns the range [start, start + count) of vocab rows for `rank`.
 * For rank_count == 1 the range is [0, n_vocab). */
static inline ds4_tp_shard_range ds4_tp_shard_vocab_rows(
    uint32_t rank,
    uint32_t rank_count,
    uint32_t n_vocab)
{
    const uint32_t base = n_vocab / rank_count;
    const uint32_t start = rank * base;
    const uint32_t count = (rank == rank_count - 1u)
        ? n_vocab - start
        : base;
    ds4_tp_shard_range r;
    r.start = start;
    r.count = count;
    return r;
}

/* Embedding / FFN column ownership for one rank.
 *
 * Row-shards the shared n_embd dimension (token embedding, shared-expert
 * columns, output projection) across ranks.  Same floor/remainder scheme
 * as routed experts and vocab rows: each rank gets floor(n_embd / rank_count)
 * columns; the last rank absorbs the remainder.
 *
 * Returns the range [start, start + count) of column indices for `rank`.
 * For rank_count == 1 the range is [0, n_embd). */
static inline ds4_tp_shard_range ds4_tp_shard_embd_cols(
    uint32_t rank,
    uint32_t rank_count,
    uint32_t n_embd)
{
    const uint32_t base = n_embd / rank_count;
    const uint32_t start = rank * base;
    const uint32_t count = (rank == rank_count - 1u)
        ? n_embd - start
        : base;
    ds4_tp_shard_range r;
    r.start = start;
    r.count = count;
    return r;
}

/* Aggregate per-rank shard configuration for all four sharded dimensions.
 *
 * Filled by ds4_tp_compute_shard_config().  Each field is the ownership
 * range for that dimension on the calling rank. */
typedef struct {
    ds4_tp_shard_range heads;   /* attention query heads */
    ds4_tp_shard_range experts; /* routed MoE experts */
    ds4_tp_shard_range vocab;   /* output head vocab rows */
    ds4_tp_shard_range embd;    /* embedding / FFN columns */
} ds4_tp_shard_config;

/* Compute the full per-rank shard configuration in one call.
 *
 * Unlike the per-dimension helpers (which absorb remainders into the
 * last rank), this function enforces strict even division across all
 * four dimensions and fails loudly when any dimension does not divide
 * evenly by tp_world.  This matches the TP=4 policy: the target model
 * shapes (Flash: 256 experts, 64 heads, 129280 vocab, 4096 embd;
 * Pro: 384 experts, 128 heads, 129280 vocab, 7168 embd) all divide
 * evenly by 4, so uneven-division is treated as a configuration error
 * rather than silently mis-sharding.
 *
 * Returns 0 and fills *out on success.
 * Returns -1 and leaves *out untouched on any of:
 *   - tp_world == 0
 *   - tp_rank >= tp_world
 *   - any of n_head, n_expert, n_vocab, n_embd is 0
 *   - any of n_head, n_expert, n_vocab, n_embd is not divisible by tp_world
 *   - out == NULL */
static inline int ds4_tp_compute_shard_config(
    uint32_t tp_world,
    uint32_t tp_rank,
    uint32_t n_head,
    uint32_t n_expert,
    uint32_t n_vocab,
    uint32_t n_embd,
    ds4_tp_shard_config *out)
{
    if (out == NULL) return -1;
    /* Reuse the per-dimension validator; it covers rank/count invariants
     * and the even-heads requirement. */
    if (!ds4_tp_shard_valid(tp_rank, tp_world, n_expert, n_head, n_vocab)) {
        return -1;
    }
    /* The validator does not know about the embd dimension — check its
     * non-zeroness here before the divisibility test, since 0 % N == 0
     * would otherwise pass silently. */
    if (n_embd == 0) return -1;
    /* The validator intentionally allows uneven expert/vocab division for
     * the 2-rank case.  For the aggregate config we require strict even
     * division of every dimension — fail loudly instead of silently
     * mis-sharding. */
    if (n_expert % tp_world != 0) return -1;
    if (n_vocab   % tp_world != 0) return -1;
    if (n_embd    % tp_world != 0) return -1;

    out->heads   = ds4_tp_shard_heads(tp_rank, tp_world, n_head);
    out->experts = ds4_tp_shard_experts(tp_rank, tp_world, n_expert);
    out->vocab   = ds4_tp_shard_vocab_rows(tp_rank, tp_world, n_vocab);
    out->embd    = ds4_tp_shard_embd_cols(tp_rank, tp_world, n_embd);
    return 0;
}

#endif /* DS4_TP_SHARD_H */
