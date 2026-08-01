# 57 — Fix the compressed-KV-cache race blocking threaded rollout past layers 0-1

Status: ready-for-human

## Parent

`.scratch/rocm-tensor-parallel/issues/51-tp4-execution-engine-full-rollout.md`

## Blocked by

`.scratch/rocm-tensor-parallel/issues/59-fix-per-tier-vram-weight-sharding.md`

(AC3's TP=4 fixture half only — see Comments, 2026-08-01 verification pass.
AC1's review/sign-off half needs a human, not another issue.)

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

- [x] Concurrency-safe design for `layer_n_comp[il]`/`layer_n_index_comp[il]`
      resolved **and reviewed** (HITL sign-off recommended, matching #52's
      precedent, given this is exactly the silent-corruption risk class
      that motivated that gate) — **signed off by human 2026-08-01**, after
      the non-transactional increment-then-maybe-fail hazard flagged in the
      2026-08-01 verification pass was fixed (pre-increment preserved,
      rollback added on dispatch/capacity failure) and independently
      reviewed across three rounds of AI-consultant panel review plus three
      green `make -j8 test-rocm` runs. Full detail in Comments and
      `experiment-log.md`.
- [x] `metal_graph_tp4_spike_layer_enabled`'s `ds4_layer_compress_ratio(il)
      == 0` gate relaxed to cover the newly-safe layers, without
      regressing the layers that were already safe — confirmed in the
      `ds4.c` diff (commit `8a8f82a`): the gate now `return true;`
      unconditionally instead of `return ds4_layer_compress_ratio(il) == 0`.
- [ ] Full 100-case `score_official` quality fixture re-run (pipeline and
      TP=4) — this touches shared decode-loop state directly, so
      correctness must be reconfirmed, not assumed. **Pipeline half
      satisfied** by `#62`'s HEAD re-measurement, `quality-out/q_pipeline_head62.log`
      (HEAD `498a39d`, which has `8a8f82a` — this issue's fix commit — as an
      ancestor; `avg_nll` 0.369196, at the PRD bar). **TP=4 half blocked by
      `#59`** — see Comments; do not retry TP=4 fixture runs until `#59`
      lands, per the human disposition recorded in `#62`/`#58`.
- [x] `make -j8 test-rocm` passes — re-run 2026-08-01, all suites green
      (`test_rocm_tp_stubs`, `test_rocm_xdev`, `test_rocm_kernel_compare`
      6/6, `test_engine_rocm_tp_refusal`). Log: `/tmp/test-rocm-57.log`
      (not committed; ephemeral). This AC only covers the non-GPU-hardware
      concurrency-race path indirectly — none of these suites drive the
      spike-worker counter-hoist path directly (see Comments).
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

**2026-08-01 — Verification pass: AC4 closed, AC3 pipeline half closed,
AC1 design note written, issue stays open (`ready-for-human`).**

*AC4.* Ran `make -j8 test-rocm` fresh (GPU lock acquired/released around
the run). All four suites pass, including `test_rocm_kernel_compare` (6/6)
and the full `test_rocm_xdev` all-reduce/cross-device suite. None of these
suites exercise `ds4_tp4_spike_worker_main` / the counter-hoist path
directly — they test kernels and cross-device transport in isolation, not
the persistent-thread decode loop — so this AC is satisfied as literally
written but is not, by itself, evidence the race is fixed. That evidence
has to come from AC1's review and AC3's quality numbers.

*AC3, pipeline half.* Did not re-run GPU hours for this: `8a8f82a` (this
issue's fix) is an ancestor of `498a39d`, the exact commit `#62`'s pipeline
HEAD run was built at (`git merge-base --is-ancestor 8a8f82a 498a39d`
confirms; the log header at `quality-out/q_pipeline_head62.log` records
`HEAD=498a39d9a...`). That run already measures a build containing this
fix, at `avg_nll` 0.369196 — PRD bar. Citing it directly rather than
reproducing it.

*AC3, TP=4 half.* Deliberately **not** attempted. `#62` already spent two
GPU runs establishing that TP=4 currently fails deterministically before
any case scores (`arena alloc failed for moe_down`, `#59`'s VRAM budget),
with the human disposition "proceed to `#59`, no further TP=4 retries."
Running it again here would reproduce the same failure regardless of
whether this issue's fix is correct, burning GPU-hours to relearn a known
fact. Added a `## Blocked by` pointing at `#59` below so this doesn't get
re-derived by the next pass.

*AC1, design review (not a HITL sign-off — an agent cannot provide that;
flagging for human review).* Read the hoisted counter logic end to end
(`ds4.c:27652-27676` for the all-43-layer dispatch path, `ds4.c:27690-27717`
for the `DS4_TP4_THREADED_LAYERS`-scoped path, `ds4.c:22606`/`22662` and
`22736`/`22801` for the worker-side read-only consumption gated by
`!g->rocm_tp4`). Two things worth a human's five minutes:

1. **The load-bearing assumption holds.** The issue's own "plausible
   direction" flagged a risk: *"Check whether `emit` ... can differ by
   rank."* It cannot — `emit = ((pos + 1) % ratio) == 0` (`ds4.c:27662`,
   `27699`) is a function of `pos` and `ratio` alone, both identical across
   all 4 ranks for a given layer/token. Since `emit` can't diverge by rank,
   pre-computing it once on the orchestrator and hoisting the counter
   mutation ahead of dispatch is sound — every rank would have computed the
   same `emit` had it still done so locally.

2. **New finding, not previously flagged: increment/dispatch is not
   transactional.** The old code incremented only `if (ok && emit)`
   (`ds4.c:22662`) — *after* the phase succeeded, so a failed phase never
   advanced the counter and was safe to retry. The hoisted code
   (`ds4.c:27659-27676`) increments `layer_n_comp`/`layer_n_index_comp` for
   *every* layer with `emit` true, unconditionally, in a pass that
   completes *before* `metal_graph_tp4_spike_dispatch` even runs. If the
   dispatch or barrier subsequently fails (`ok = false` from
   `metal_graph_tp4_spike_dispatch`/`metal_graph_tp4_spike_barrier`,
   `ds4.c:27684-27687`), or the capacity check itself `break`s partway
   through the pre-increment loop, the counters for layers whose rows were
   never actually written are left permanently advanced. If the caller
   treats that as fatal and aborts the whole process (most call sites do:
   `metal_graph_eval_token_raw_swa` and friends just `fprintf` and
   `return false`, unwound by session/request teardown), this is
   harmless — the corrupted graph state dies with the process. But if any
   caller retries the same graph/session after a transient failure (I did
   not find one on the decode hot path, but did not exhaustively check the
   session-batch and disk-checkpoint-resume paths at `ds4.c:62736`/`66315`),
   a subsequent emit at that layer would append at the wrong physical row —
   the exact silent-corruption class this issue exists to prevent, just
   relocated from "concurrent read/write race" to "non-transactional
   increment-then-maybe-fail." Separately: `#62` observed TP=4 runs that
   hit `arena alloc failed for moe_down` and *did not abort* — one run
   completed all 100 cases at 44x the PRD bar. That failure is in the
   weight-arena path (`rocm/ds4_rocm_runtime.cuh:5778`, `#59`'s VRAM
   budget), which falls back to host-mapped weights rather than setting
   `ok = false`, so it likely does not trigger this specific hazard — but
   it demonstrates the codebase does have failure modes on this hot path
   that don't hard-abort, which is exactly the precondition this hazard
   needs. Not fixing this here per the issue's own instruction not to
   patch quickly; flagging for the human reviewer to decide whether it's
   in scope for this issue's sign-off or a follow-up.

Also checked the prefill/batch counter-mutation pattern at `ds4.c:29126-
29652` (`metal_graph_encode_layer_attention_batch`,
`metal_graph_encode_layer_batch`) that the original issue asked to audit:
confirmed unreachable from the spike-worker threads. Its TP=4 row-split
call sites (e.g. `ds4.c:31288`, `for (int tier = 0; tier < 4; tier++)`)
iterate all 4 tiers sequentially on the single orchestrator thread, the
same pattern as pre-#50 — no concurrent dispatch, so no race, orthogonal
to this issue's fix.

**Disposition: `ready-for-human`, not `closed`.** AC1's review/sign-off
half genuinely requires a human (an agent restating its own analysis is
not a second opinion). AC3's TP=4 half is blocked on `#59` by explicit
prior human decision. Everything else achievable without those two is
done and cited above.

**2026-08-01 — Human/agent pairing session: non-transactional hazard
fixed via pre-increment + rollback (after a reverted first attempt).**
Presented the two open items above (AC1 review + the non-transactional
increment hazard) to a human directly. Disposition: **block AC1 sign-off
until the hazard is fixed**, not spun out as a follow-up.

**First attempt (reverted).** Deferred the counter commit to after
dispatch success, and simplified the `comp_row`/`index_row`/capacity-check
reads in `metal_graph_encode_decode_layer_phase` to drop their
pre-increment compensation, on the theory that the race-safety property
only needs "not mutated during dispatch," not "pre-incremented." `test-
rocm` passed. Before asking for sign-off, ran it by the AI-consultants
panel (`/ai-consultants:consult`, 7/10 responded) as a second opinion —
Cursor caught, and the synthesis confirmed, that the theory was wrong: the
same function also reads the counters *directly, uncompensated* at several
points during Phase 1 (sparse-threshold gate, indexer scoring/top-k,
selected-row count — `ds4.c:22802-22804`, `22853`, `22860`, `22869`,
`22874`, `22882`, `22920-22922`), and those reads need the *post*-increment
value. Deferring the commit silently excluded each TP4 token's own
freshly-written compressed row from its own indexer/attention computation
— a real correctness regression that `test-rocm` did not catch (it doesn't
exercise this path). Full detail in `experiment-log.md`'s 2026-08-01
entry.

**Actual fix (kept).** Reverted the read-side back to byte-identical
`8a8f82a`/HEAD (verified via `git show HEAD:ds4.c` diff). The counter is
still pre-incremented before dispatch, preserving the reads above. The
real fix is: **roll the increment back if dispatch subsequently fails**,
instead of leaving it stranded (the original hazard) or never incrementing
at all (the reverted attempt).
- Per-layer path (`ds4.c:~27742-28085`): a `tp4_counter_incremented_this_layer`
  flag gates a decrement just before `continue`, firing only on `!ok`.
- Full-token overlap path (`ds4.c:~27667-27722`): a `pre_loop_stopped_at`
  bound (see Round 2 note below for why this replaced an earlier
  `full_token_dispatched` flag) gates a rollback pass after
  `metal_graph_tp4_spike_barrier()`, mirroring the pre-loop's increment
  logic with `--`, firing on `!ok`.

Re-ran `make -j8 test-rocm` under the GPU lock against this corrected
version (the earlier green run doesn't count — it validated the reverted
deferred-commit version). All four suites pass. Log:
`/tmp/test-rocm-57-followup-v2.log` (ephemeral).

**Round 2 consultant review found one more real gap (fixed) and two false
positives (checked, not acted on).** Ran this version by the panel again
(7/10 responded). Both Cursor and Qwen3 independently confirmed the Round-1
bug does not reoccur, and both flagged: the full-token path's rollback was
gated on `full_token_dispatched`, so a capacity-check `break` partway
through the pre-loop (layers `0..k-1` already incremented before layer `k`
fails its check) never got rolled back — same hazard class, pre-existing
in `8a8f82a`/HEAD, not introduced this session. **Fixed:** replaced
`full_token_dispatched` with `pre_loop_stopped_at` (defaults to
`DS4_N_LAYER`, set to the failing `il` on a capacity break); rollback is
now a single `if (!ok)` bounded to `[0, pre_loop_stopped_at)`, correctly
covering both the capacity-break case and the dispatch/barrier-failure
case. Re-ran `make -j8 test-rocm` a third time against this version, still
green (`/tmp/test-rocm-57-followup-v3.log`, ephemeral).

Two other panel claims were checked directly against the code and are
false positives (full reasoning in `experiment-log.md`'s Round 2 entry):
Qwen3's claimed counter-underflow from asymmetric comp/index increments
(both capacity checks run before either counter is incremented, in both
paths, unchanged from `8a8f82a`), and Qwen3's claimed data race on barrier
failure (`metal_graph_tp4_spike_dispatch` blocks on every worker's
`job_done`, set unconditionally after each worker finishes its
synchronous host-side counter reads, before `barrier()` — which only
gates GPU-side event completion — ever runs).

Same standing caveat as every prior pass on this issue: none of
`test-rocm`'s suites drive the spike-worker counter-commit/rollback path
directly, so passing it is necessary, not sufficient.

**Round 3 consultant review (final): fix confirmed, no new bugs.** Ran
the `pre_loop_stopped_at` version by the panel a third time (7/10
responded). Cursor and Qwen3 both independently confirmed the fix closes
the Round 2 gap completely, both Round 2 false-positive re-checks hold up,
and the per-layer path has no equivalent gap. No new correctness bugs
found — the only two notes raised (full-token "blunt all-or-nothing"
rollback on a post-dispatch barrier failure; a defensive-programming
suggestion around the `ok` invariant) are both already covered: the first
is the same documented, deliberate tradeoff from Round 2; the second isn't
a bug, since the enclosing `if (ok && g->rocm_tp4 && ...)` guard already
structurally guarantees `ok == true` on entry. Full detail in
`experiment-log.md`'s Round 3 entry. Three independent consultant rounds
plus three green `test-rocm` runs (one per version) is the evidence this
session is presenting for AC1 sign-off below.

Scope note carried into the write-up so it isn't lost: this makes the
*counter* transactional w.r.t. dispatch failure. It does not make full
session/request retry after a failed token safe in general —
`ds4_gpu_compressor_update_tensor` mutates recurrent compressor state in
place, non-idempotently, independent of this fix. Pre-existing, out of
scope here.

AC1 is now ready for the human's actual sign-off on this fix (design +
diff, above and in the `ds4.c` diff directly) — pending in this session.

**2026-08-01 — Human sign-off recorded.** The human reviewed this fix's
disposition (pre-increment preserved, rollback added, three rounds of
AI-consultant review, three green `test-rocm` runs) and signed off on AC1.
AC1 checked off above.

**Closure disposition.** AC1/AC2/AC4/AC5 are now satisfied. AC3's TP=4
half remains genuinely blocked on `#59` (the arena-OOM root cause), which
has not landed — the human explicitly chose to leave this issue open
rather than rule the TP=4 fixture out of scope. `Status` stays
`ready-for-human` with the existing `Blocked by: #59` link; this issue
closes once `#59` lands and the TP=4 quality fixture can actually run
against a build containing this fix.
