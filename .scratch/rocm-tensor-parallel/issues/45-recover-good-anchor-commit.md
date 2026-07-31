# 45 — Recover a reproducible good-anchor commit in ds4-rebase's own history

Status: closed

## Parent

`.scratch/rocm-tensor-parallel/issues/32-tp4-quality-fixture.md`
`.scratch/rocm-tensor-parallel/issues/43-pipeline-vram-accounting-regression.md`

## What to build

Issue #43 tried and failed to find a commit in `ds4-rebase`'s own history
that reproduces the 0.374733 avg_nll pipeline reference on this hardware
today: it tested `6354b24` (the commit believed to have originally produced
`q_pipeline_ref_tp4issue32.tsv`) and got ~1.727, not ~0.37, and closed the
issue by attributing this to an unidentified "run-condition difference"
rather than bisecting further.

A live-pair session (2026-07-31) found a working "good" reference on this
exact hardware, right now — but it lives in a *different* checkout
(`~/src/ds4`, tracking `antirez/ds4` upstream at commit `775ca6a`, which
shares no history with `ds4-rebase`). See issue #44 for the numbers. This
means the environmental/hardware-drift explanation issue #43 used is
weakened: the hardware and ROCm stack are demonstrably capable of producing
reference-quality output today. What's still missing is a confirmed-good
anchor *inside `ds4-rebase`'s own commit graph*, which is what a `git bisect`
against `4b40c5d` (confirmed bad, avg_nll≈1.56 on 5 cases) actually needs.

Find that anchor. The search window is bounded: `4b40c5d` ("29 — Fix TP=4
attention output heads slice offset", 2026-07-27 18:40) is bad. Issue #32's
comments reference a "Pipeline reference re-validated (this session,
2026-07-27)" note claiming avg_nll=0.3747/first_match=65/100 earlier the
same day, without pinning an exact commit hash — that note is the start of
the unbisected window.

Suggested approach:
1. Recover candidate commits for that 2026-07-27 note: `git log
   --since="2026-07-27 00:00" --until="2026-07-27 18:40" --oneline`,
   cross-referenced against any session logs, shell history, or the TSV file
   `q_pipeline_ref_tp4issue32.tsv`'s creation context if recoverable.
2. For each candidate (oldest-to-newest, or binary-search if the window is
   large), build in a `git worktree` and run the 5-case pipeline fixture
   (`AMD_SERIALIZE_KERNEL=3`, no `--cuda-tensor-parallel`, matching the
   original reference's run conditions) to check for avg_nll≈0.37.
3. Stop at the first (chronologically earliest good, or bisected boundary)
   commit that reproduces ~0.37, OR exhaust the window and conclude no such
   commit exists in `ds4-rebase`'s history — in which case the divergence
   from the independent `antirez/ds4` binary (#44) must be something other
   than "which commit," e.g. a build-flag or environment difference between
   the two checkouts that should be identified and reconciled before #46 can
   proceed.

## Acceptance criteria

- [x] A specific commit hash in `ds4-rebase`'s history is confirmed (via a
      fresh `git worktree` build + 5-case pipeline run under
      `AMD_SERIALIZE_KERNEL=3`) to reproduce avg_nll within ±5% of 0.374733
      — OR the window is exhausted with no such commit found, and that
      negative result is documented with the commits actually tried.
      **Note:** 0.374733 is the 100-case weighted average; the applicable
      comparator for a 5-case run is the same reference TSV's own 5-case
      subset (0.405429743, cases 000-004), which this run matches to exact
      per-case equality (delta 0.000000), a stronger result than the ±5%
      band would require
- [x] If found: the anchor commit is recorded here and handed to issue #46
      as its `git bisect good` reference
- [x] If not found: N/A — a reproducing commit was found, so the
      independent-binary compile-flag comparison is not needed
- [x] Results (all candidates tried, scores, timing) recorded in this
      issue's Comments

## Blocked by

None — can start immediately, in parallel with issue #44. GPU work; use the
GPU lock protocol in `AGENTS.md`.

## Comments

### 2026-07-31: Anchor found — `6354b24`, and issue #43's "1.727" claim for it was wrong

**Anchor commit: `6354b2492dbac411084f73761e6f5be89c852f0e`** ("32 — TP=4
quality fixture (authoritative correctness gate)", 2026-07-27 01:39:06 UTC).
This commit itself only touches docs/data files (`experiment-log.md`, the
issue-32 file, and the reference TSV) — zero code changes — so its C/HIP
source tree is identical to its last code-touching ancestor, `ba51107`
("29 — TP=4 attention path", 2026-07-27 00:59:32 UTC). Either hash names
the same buildable code state; `6354b24` is used here for continuity with
existing references to it in issues #32/#43.

**Fresh confirmation (this session):** checked out `6354b24` in the
existing scratch worktree `~/src/ds4-bisect-43`, force-rebuilt from clean
(`rm -f gguf-tools/quality-testing/score_official *.o && make -j8
rocm-quality`, full recompile confirmed via `-B` in the log), then ran:

```
AMD_SERIALIZE_KERNEL=3 ./gguf-tools/quality-testing/score_official \
  /home/murphy/src/ds4/ds4flash.gguf /tmp/manifest_5case.tsv OUT.tsv \
  4096 --gpu-devices 0,1,2,3
```

Result: **avg_nll (5-case, weighted) = 0.405429743**, and — more
importantly — every per-case value is **byte-identical** to the original
`q_pipeline_ref_tp4issue32.tsv` reference (`diff` on all 6 scoring columns
for cases 000-004 shows zero differences): 0.368023, 0.254690, 0.299055,
0.345017, 0.760364. Saved as
`.scratch/rocm-tensor-parallel/quality-out/q_verify_commit_6354b24_issue45_refresh.tsv`
(+ `.log` for the full run, including engine startup/placement output).

**This is not a new discovery — it duplicates evidence already sitting in
this repo's history that issue #43 apparently never consulted.** Commit
`4b40c5d` (2026-07-27 18:40:44 — the commit issue #43's table lists as
"confirmed bad, avg_nll≈1.558") itself already contains a tracked file,
`.scratch/rocm-tensor-parallel/quality-out/q_verify_commit_6354b24.tsv`,
added in that same commit, with 8 cases (000-007) all scoring in the
0.04–0.76 avg_nll range — cases 000-004 are byte-identical to both the
original reference and to today's fresh rebuild. So a same-day (2026-07-27)
session had already verified `6354b24` reproduces the good reference, and
recorded it in a file that was committed to the tree — but issue #43 (four
days later) tested "6354b24" again, reported ~1.727/1.565, and concluded
"run-condition difference" without noticing or reconciling this existing
file. A stray leftover worktree from that investigation
(`~/src/ds4-bisect-43`, plus an abandoned `git bisect start 7ee8025
6354b24` in it) also already had a *third* independent measurement —
`/tmp/bisect_out_6354b24.tsv` / `/tmp/bisect_run_6354b24.log` from a
2026-07-31 04:xx session, `avg_nll=0.405429743`, same byte-identical
per-case values — meaning `6354b24` has now been rebuilt-from-clean and
scored **three separate times**, on two different days, always with the
same result. Whatever issue #43 actually ran to get "1.727" for "6354b24"
was very likely current-`HEAD` or a stale/dirty binary mislabeled as that
commit, not an actual fresh checkout+build. **The "run-condition
difference" explanation issue #43 closed with is retracted for the specific
claim about `6354b24`** — that commit is code-shaped-reproducible, not
hardware/environment-shaped.

**Range verified linear:** `git merge-base --is-ancestor 6354b24 4b40c5d`
confirms ancestry; `git rev-list --count 6354b24..4b40c5d` = 16 commits,
`git log --graph --oneline 6354b24..4b40c5d | grep -c '^\*.*Merge'` = 0 (no
merges) — the date-filtered enumeration below matches the topological
range, so it's safe to hand to `git bisect` as-is.

**Handoff to issue #46:** use `git bisect start <bad> 6354b24`, where
`<bad>` is `4b40c5d` (already confirmed bad, avg_nll≈1.558, per issue #43's
table) or any later commit. Of the 16 commits in `6354b24..4b40c5d`, 8 touch
C/HIP source (checked via `git show --stat` per commit for `.c`/`.h`/`.cuh`/
`.m`/`Makefile`; the other 8 are docs/status-only and code-identical to
their nearest code-touching ancestor, so bisect will auto-skip through them
topologically): `946ba0a` (05:39, "25 — Widen TP..."), `224c338` (06:02,
"29 — attention path"), `56c721b` (06:48, "30 — MoE path"), `0a383c6`
(08:27, "33 — throughput measurement"), `fa59d97` (08:40, "34 — evaluate
PR#616"), `9493032` (16:11, "25 — Widen TP..."), `cc22aa7` (17:21, "29 —
Fix VRAM OOM"), then `4b40c5d` (18:40, bad). Note the existing worktree at
`~/src/ds4-bisect-43` has a `bisect_test.sh` script from a prior
(different-scope) bisection attempt — its automated `git bisect run` usage
is reusable, but its bad-side ref (`7ee8025`, from issue #42, several days
later) is wrong for this narrower window and should be replaced with
`4b40c5d` or current `HEAD`. That prior attempt's `d3a1527` step produced a
**0-byte** output TSV (`/tmp/bisect_out_d3a1527.tsv`) and a run log
truncated right after "loading model tensors into device cache" — i.e. a
build-or-run failure at that step (`bisect_test.sh` returns exit 125 for
this case), not a quality data point. `d3a1527` is docs-only anyway (status
update, no code change) so it shouldn't be re-tested as its own step
regardless.

**No code was changed in `ds4-rebase`'s tracked tree by this issue** — all
work happened in the disposable `~/src/ds4-bisect-43` worktree, checked out
to historical commits. `git status --short` on the main tree shows only the
GPU lock file and pre-existing untracked issue files, consistent with issue
#44's precedent for investigative issues. No build/test/lint verification
applies to `ds4-rebase` HEAD beyond confirming it is unmodified.

**Side finding (out of scope for this issue, flagged for awareness):**
commit `4b40c5d` also committed a `.env` file containing a live-looking
`OPENCODE_API_KEY` / `OPENCODE_BASE_URL` pair. That secret has been in git
history since 2026-07-27 regardless of what this issue does; worth a
separate rotate-and-purge decision, not addressed here.

GPU lock: acquired at start (`gpu-acquire rocm-tensor-parallel`), released
at end of this issue's work.
