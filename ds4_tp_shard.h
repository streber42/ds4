/* ds4_tp_shard.h — tensor-parallel sharding policy.
 *
 * Pure-logic ownership decisions for the ROCm tensor-parallel port.
 * No GPU dependency, no model state: all functions are deterministic
 * pure functions of the model dimensions, rank count, and rank index.
 *
 * Three sharded dimensions:
 *
 *   Routed experts — the full set of n_expert routed experts is split
 *     across ranks in contiguous ranges.  Rank 0 takes the lower
 *     experts; the last rank absorbs any remainder from uneven division.
 *     This mirrors the Metal-side convention in weights_model_map_sharded_spans:
 *     rank 0 owns experts [0, low_experts), rank 1 owns the rest.
 *
 *   Attention heads — the n_head query heads are divided evenly in half
 *     for the two-rank design, rank 0 owns [0, n_head/2), rank 1 owns
 *     the upper half.  Heads must be divisible by rank_count (asserted
 *     by ds4_tp_shard_valid).
 *
 *   Vocabulary rows — the output head vocab rows (n_vocab) are split
 *     across ranks in contiguous ranges.  Rank 0 owns [0, low_rows);
 *     the last rank takes any remainder.  This matches the two-rank
 *     Metal decode-side tp_vhalf = vocab_dim / 2u convention.
 *
 * Uneven division: when n_expert or n_vocab is not exactly divisible by
 * rank_count, the first (rank_count - 1) ranks each get floor(N/R) items
 * and the last rank gets the remainder.  This is deterministic and the
 * only source of uneven ranges; attention heads are always evenly split.
 *
 * Single-rank degenerate case: rank_count == 1 returns full ownership for
 * all three dimensions (start == 0, count == N).
 *
 * Kernels and engine code must obtain ownership bounds only through
 * ds4_tp_shard_experts / ds4_tp_shard_heads / ds4_tp_shard_vocab_rows;
 * they must not re-derive these indices inline.
 */

#ifndef DS4_TP_SHARD_H
#define DS4_TP_SHARD_H

#include <stdbool.h>
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

#endif /* DS4_TP_SHARD_H */
