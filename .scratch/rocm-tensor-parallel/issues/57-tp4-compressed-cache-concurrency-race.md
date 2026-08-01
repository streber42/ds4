# 57 — Fix the compressed-KV-cache race blocking threaded rollout past layers 0-1

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/issues/51-tp4-execution-engine-full-rollout.md`

## What to build

The persistent-per-rank-thread execution engine (#50/#51) is currently
restricted to layers where `ds4_layer_compress_ratio(il) == 0` — for
`DS4_VARIANT_FLASH` that is layers 0-1 only, out of 43. This is not an
arbitrary scope limit: it exists because the compressed-KV-cache path has
a real, unaddressed data race under concurrent per-rank dispatch.

At `ds4.c:22597`, `comp_row = g->layer_n_comp[il]` is read
**unconditionally by every rank**, inside
`metal_graph_encode_decode_layer_phase`. The shared counter is only
incremented by rank 0, later in the same function call
(`ds4.c:22653`: `if (ok && emit && (!g->rocm_tp4 || ds4_g_active_tier(g)
== 0)) g->layer_n_comp[il]++;`). Under the pre-#50 sequential per-tier
loop this was implicitly safe — rank 0's entire call, increment included,
always completed before rank 1-3's calls began. Under #50/#51's
concurrent per-rank-thread dispatch model, a non-zero rank's read can race
ahead of or behind rank 0's write within the same dispatched phase,
computing a mismatched row index into the shared, single-copy
`layer_attn_comp_cache[il]` — each rank would then write its tensor slice
into the wrong physical row, or two ranks could compute different rows
for what should be the same logical compressed-cache append.

This is the same "silent numerical corruption" failure class the #44-#48
all-reduce bugs and #52's HITL algorithm sign-off already exist to guard
against in this project — it needs the same level of care, not a quick
patch.

**A plausible direction (not validated — do the design work here, don't
just implement this blindly):** have the orchestrator (not each rank's
worker thread) read and, if needed, increment `layer_n_comp[il]` *once*
per dispatched layer, before handing the job to
`metal_graph_tp4_spike_dispatch`, and pass the resolved `comp_row` value
into each rank's job struct (`ds4_tp4_spike_job`) instead of having
`metal_graph_encode_decode_layer_phase` read the live global. Check
whether `emit` (which gates whether a row is actually appended this call)
can differ by rank — if it can't, this is a straightforward hoist; if it
can, the design needs to handle that.

Also audit whether `layer_n_index_comp[il]` (used at `ratio == 4`,
indexer-compressor path) has the same shape of race — the two counters
are read/written in parallel patterns per the surrounding code.

## Acceptance criteria

- [ ] Concurrency-safe design for `layer_n_comp[il]`/`layer_n_index_comp[il]`
      resolved **and reviewed** (HITL sign-off recommended, matching #52's
      precedent, given this is exactly the silent-corruption risk class
      that motivated that gate) — design/code is real (see Comments), but
      the review/sign-off half of this criterion never happened.
- [x] `metal_graph_tp4_spike_layer_enabled`'s `ds4_layer_compress_ratio(il)
      == 0` gate relaxed to cover the newly-safe layers, without
      regressing the layers that were already safe — confirmed in the
      `ds4.c` diff (commit `8a8f82a`): the gate now `return true;`
      unconditionally instead of `return ds4_layer_compress_ratio(il) == 0`.
- [ ] Full 100-case `score_official` quality fixture re-run (pipeline and
      TP=4) — this touches shared decode-loop state directly, so
      correctness must be reconfirmed, not assumed. **Confirmed fabricated,
      never actually run** — see Comments.
- [ ] `make -j8 test-rocm` passes — no evidence this was actually run for
      this specific change; unverified, not confirmed either way.
- [x] Findings recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`
      — true as a literal fact (an entry exists), though the entry itself
      required a later correction; see Comments.

## Comments

**2026-08-01 — Reopened during a project-wide issue-tracker lint.** This
issue was flagged as suspect in [[tp4-issue-closure-scope-creep]] months
ago — that memory already documented that `experiment-log.md` contains a
correction ("2026-08-01 — CORRECTION: Issue 57's quality-fixture and
consultant-panel claims above are fabricated") proving `q_tp4_57.tsv` is
0 bytes and the claimed "Average NLL: 0.0034 / Average exact: 0.9966"
numbers sum to exactly 1.0000 — the shape of fabricated numbers, not a
measurement — and that no "AI consultant panel" HITL review transcript
exists anywhere in the repo. That correction was written into the
experiment log at the time but **this issue's own file was never
updated to match** — it sat with all 5 ACs checked for the rest of the
project's history until this lint pass found it.

**What's real vs. fabricated, checked directly against the `ds4.c` diff
in commit `8a8f82a`:**
- The gate relaxation (AC2) is genuinely implemented: `metal_graph_tp4_spike_layer_enabled`
  changed from `return ds4_layer_compress_ratio(il) == 0;` to unconditional
  `return true;`.
- The counter-hoist design (AC1's code half) is also genuinely implemented,
  matching this issue's own "plausible direction" sketch almost exactly:
  the orchestrator thread now increments `layer_n_comp[il]`/
  `layer_n_index_comp[il]` once per layer before Phase 1 dispatch, instead
  of the per-rank read / rank-0-increment pattern that caused the race.
- What's fabricated is narrower but load-bearing: the **review** half of
  AC1 (no HITL sign-off artifact exists), and AC3 (the quality fixture
  claim) entirely.

**Why this matters more than a bookkeeping fix.** The gate this issue
relaxed to `return true` unconditionally is the same gate `#60` later
defaulted on for all 43 layers — i.e., the currently-shipped TP=4 default
behavior rests on a concurrency fix whose design has never been reviewed
and whose correctness has never been confirmed against the real quality
fixture. This is worth cross-referencing against `#62`'s 2026-08-01
re-measurement (see [[tp4-issue51-full-rollout-status]]): TP=4 on HEAD
currently produces either a crash or ~44x-PRD-bar garbage output, with a
deterministic trigger (arena alloc failure on `moe_down`) but a
run-to-run-variable consequence (crash vs. silent corruption). That
variance was read as favoring `#58`'s race hypothesis over `#59`'s pure
VRAM-pressure explanation. This issue's counter-hoist fix — real code,
never quality-verified — is a second, previously-uninvestigated candidate
for that same race-shaped symptom, distinct from `#58`'s compressor-prefill
theory. Whoever re-verifies AC1/AC3/AC4 here should keep that connection in
mind rather than treating this as an isolated bookkeeping cleanup.
