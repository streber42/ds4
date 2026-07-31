# 43 — Pipeline reference baseline is not reproducible on the current tree

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/issues/32-tp4-quality-fixture.md`
`.scratch/rocm-tensor-parallel/issues/41-host-mapped-moe-weight-precision.md` (discovered during
this investigation)

## Problem

The pipeline (non-TP) reference baseline used throughout this project —
`q_pipeline_ref_tp4issue32.tsv`, avg_nll=0.374733, first_match=65/100 —
is **not reproducible on the current tree**. Re-running the same cases in
pipeline mode (`./gguf-tools/quality-testing/score_official ...
--gpu-devices 0,1,2,3`, no `--cuda-tensor-parallel`) gives avg_nll ≈ 1.56,
indistinguishable from TP=4's broken score, with the same
arena-exhaustion / host-register-PCIe-fallback signature TP=4 hits (see
[issue #41](41-host-mapped-moe-weight-precision.md) for that mechanism):
~970-987 weight lookups per 5-case run fall through to PCIe-mapped host
reads instead of the pre-loaded VRAM cache, starting at layer 0.

**Severity:** this is not limited to `score_official`'s ctx=4096. Plain
`./ds4 --rocm --gpu-devices 0,1,2,3 --model ... -p "..." -n 12` (default
ctx=32768, no `--cuda-tensor-parallel`) fails outright —
`ds4 debug: alloc_raw_cap check failed: state_init_ok=1 layer_cache_ok=1
class_p_ok=0 output_tp_ok=1` followed by `ds4: sampled CLI generation
requires a session backend` — no output at all. **Pipeline mode is
currently non-functional on this 4×R9700 host at default settings**,
which also means the PRD's mandatory fallback guarantee ("If tensor
parallelism is unavailable, refused, or fails to initialise, the engine
falls back to the existing pipeline layer-split path... the user must
never end up with no working inference because of this feature") does
not currently hold: TP=4 initializes but produces garbled output (the
quality gap), and pipeline — the fallback — fails to start at all.

## Root cause: NOT bisected — an earlier attribution to commit 414f9fc was tested and disproven

An earlier version of this issue attributed the regression to commit
`414f9fc` ("unblock TP=4 quality fixture with memory accounting fixes"),
based on a diff read: it gates the batch-scratch line items in
`engine_per_tier_graph_overhead_bytes` behind `e->cuda_tensor_parallel`
while the actual allocation in `metal_graph_alloc_raw_cap` (~ds4.c:17540)
has no such gate, so pipeline mode's budget calculation under-reserves
relative to what it actually allocates. That mismatch is real (see code
excerpt below) but **direct measurement shows it is not what broke the
baseline**:

| Commit | Date | 5-case pipeline avg_nll (ctx=4096, cases 000-004) |
|---|---|---|
| `4b40c5d` ("29 — Fix TP=4 attention output heads slice offset") | 2026-07-27 18:40 | **1.558** |
| `414f9fc^` = `2a558ea` ("32 — Record f16 cuBLAS fix...") | 2026-07-29 03:53 | **1.559** |
| `414f9fc` ("unblock TP=4 quality fixture...") | 2026-07-29 16:21 | (session-creation fails at ctx=4096, class_p_ok=0) |
| current HEAD (`gfx1201_tp`) | — | **1.558** |

The broken score is *already present* at `4b40c5d`, a commit that
predates `414f9fc` by two days and ~30 commits, and whose own
commit-adjacent issue-file note ("Pipeline reference re-validated,
2026-07-27... avg_nll 0.3747... Fresh reference TSV saved") explicitly
claims the pipeline path was healthy *on that same day*. So either:

1. The regression was introduced earlier the same day (2026-07-27),
   between whatever commit the "re-validated 0.3747" run used and
   `4b40c5d` — the note doesn't pin an exact hash, so this window is
   unbisected; or
2. The difference is not a code regression at all — e.g. a run-condition
   difference (VRAM fragmentation from a long-lived process, GPU
   firmware/driver state, thermal throttling affecting `cudaMemGetInfo`
   results, or some other non-deterministic factor) between whatever
   session produced the reference TSV and a fresh single-shot run today.

**This needs a proper `git bisect`** starting from a commit confirmed to
reproduce ~0.37 (not yet found — `4b40c5d` does not) down to a commit
confirmed to reproduce ~1.56 (`4b40c5d` qualifies), rather than further
reasoning from diffs. Budget real GPU time for this: each bisect step is
a full rebuild (`make -j8 rocm-quality`, ~2-3 min) plus a 5-case
`score_official` run (~1-2 min) — a `git worktree` per candidate commit,
as done for the two data points above, avoids disturbing the main tree.

The `414f9fc` accounting mismatch (budget vs. actual allocation
disagreeing on whether to count `batch_*_by_tier`) is still real and
still worth fixing — it's why `class_p_ok=0` shows up at ctx=4096 at
`414f9fc` and later, which is a *second*, additive failure on top of
whatever caused the pre-existing avg_nll≈1.56 regression. Fixing it will
not by itself restore the 0.375 baseline, per the table above.

```c
// engine_per_tier_graph_overhead_bytes, ds4.c (post-414f9fc)
if (e && e->cuda_tensor_parallel) {
    total += pc * hc_dim * sizeof(float);   /* batch_cur_hc_by_tier */
    ...                                       /* ~30 more buffers */
}
// metal_graph_alloc_raw_cap, ds4.c ~17540 — no such gate:
for (int t = 0; t < DS4_MAX_GPUS; t++) {
    if (!used_tier[t]) continue;
    g->batch_cur_hc_by_tier[t] = ds4_gpu_tensor_alloc_ptr_on(t, ...);
    ...
}
```

## What to build

1. **Bisect** the avg_nll≈1.56 regression to a specific commit (or rule
   out code entirely and identify the run-condition difference). Start
   by finding a commit that still reproduces ~0.37 — this issue does not
   currently have one on record; the earliest candidate to try is
   whatever commit produced the "Pipeline reference re-validated,
   2026-07-27" result referenced in `32-tp4-quality-fixture.md`.
2. **Separately**, fix the `414f9fc` budget/allocation mismatch (Option A
   or B below) so ctx=4096 pipeline runs can at least create a session,
   which is a prerequisite for testing bisect candidates at the
   quality-fixture's actual context size.
3. Once both are resolved, regenerate the pipeline reference TSV and
   confirm it reproduces (or explains any legitimate delta from) the
   original avg_nll=0.374733 / first_match=65/100.

### Option A: Un-gate the budget calculation (safest, matches actual allocation)

Remove the `e->cuda_tensor_parallel` gate around the batch-scratch total
in `engine_per_tier_graph_overhead_bytes`, restoring it to always count
those buffers (as it did before `414f9fc`). This makes the budget
pessimistic-but-correct for pipeline mode again. Re-check whether this
reintroduces the "OOM at layer 40" that `414f9fc` was originally trying
to solve — if so, the real fix is elsewhere (e.g. only allocate
`batch_*_by_tier` for tiers that actually run batch prefill in pipeline
mode, rather than every `used_tier`), not in skipping the reservation.

### Option B: Gate the allocation to match the budget

Make `metal_graph_alloc_raw_cap`'s batch-scratch loop skip tiers when
`!cuda_tensor_parallel`, matching what the budget calculation now
assumes — but only if pipeline mode genuinely never uses
`batch_*_by_tier` for tiers other than the one running batch prefill.
Needs verification; if pipeline batch prefill touches per-tier batch
buffers for reasons unrelated to TP (e.g. shared workspace plumbing),
this option is unsafe.

## Acceptance criteria

- [ ] Regression bisected to a specific commit, or confirmed to be a run-condition difference rather than code (with the actual differing condition identified)
- [ ] `414f9fc` budget/allocation mismatch reconciled (Option A, B, or a third approach) with reasoning recorded
- [ ] Pipeline mode (`--gpu-devices 0,1,2,3`, no `--cuda-tensor-parallel`) quality fixture at ctx=4096 creates a session and completes without `class_p_ok=0`
- [ ] Pipeline mode quality fixture reproduces avg_nll ≈ 0.375, first_match ≈ 65/100 (matching or explaining any delta from the original baseline)
- [ ] Fresh pipeline reference TSV saved to `.scratch/rocm-tensor-parallel/quality-out/`
- [ ] `make -j8 test-rocm` passes
- [ ] Cross-reference noted in issue #42 (TP=4 VRAM budget), since both issues touch the same `engine_per_tier_graph_overhead_bytes` accounting

## Blocked by

- 4× R9700 GPU access
- Existing quality fixture infrastructure

## References

- [Issue #41 — Host-mapped MoE weight numerical impact](41-host-mapped-moe-weight-precision.md) — discovered this regression while instrumenting the weight-resolution fallback path; see its Comments section for the live trace data (`DS4_ROCM_WEIGHT_PATH_STATS=1`) that surfaced it.
- [Issue #42 — Free VRAM budget for TP=4](42-tp4-vram-budget.md) — same accounting function, TP=4 side of the same underlying problem.
- `.scratch/rocm-tensor-parallel/issues/32-tp4-quality-fixture.md` — original "Pipeline reference re-validated (this session, 2026-07-27)" note this baseline traces back to (itself referencing an even earlier "issue #20 ref" of 0.3738).
