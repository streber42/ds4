Status: ready-for-agent
# 01 — Merge `upstream/main` (antirez) into `gfx1201_tp`

**What to build:** Bring this branch up to date with upstream. As of 2026-08-11
`gfx1201_tp` is **112 commits behind `upstream/main`** with **189 commits of its
own**. Merge — do not rebase.

**Blocked by:** None mechanically. **Must not run concurrently with any GPU or
benchmark issue** — see "Isolation" below.

## Which remote

`upstream/main` (`antirez/ds4`), not `origin/main` (`kyuz0/ds4`). Verified
2026-08-11: `origin/main` is a strict *ancestor* of `upstream/main` — 61 commits
behind it, with **zero** commits of its own. Merging `upstream/main` therefore
subsumes `origin/main` entirely, and there is nothing to merge separately from
kyuz0.

## Merge, not rebase

189 ours-only commits replayed autonomously is 189 opportunities for an agent to
resolve a conflict by picking a side and silently dropping TP work. This project
already has a documented history of exactly that failure mode
(`tp4-issue-closure-scope-creep`). A merge produces one reviewable diff and one
`git merge --abort`. If you find yourself running `git rebase`, stop.

## Conflict surface (measured 2026-08-11)

Favorable. Upstream's 112 commits touch **zero** of this project's ROCm hot files:

| file | upstream commits |
|---|---|
| `rocm/ds4_rocm_q8.cuh` | 0 |
| `rocm/ds4_rocm_matmul.cuh` | 0 |
| `ds4_rocm.h` | 0 |
| `rocm/ds4_rocm_runtime.cuh` | 0 |

Expect conflicts mainly in `ds4.c` (17 upstream commits) and `Makefile` (9).

## The test-vector trap — read before writing any baseline AC

The diffstat shows the quality test vectors were **restructured and modified**
upstream: the existing set moved to `flash-pre-0731/` and a new `flash-0731/` was
added, and the moved files are not identical (`official.vec | 11 +-`,
`local-golden.vec | 2 +-`).

So an acceptance criterion of the form *"confirm avg_nll is still 0.36985"* is
**unsatisfiable as written**, and forcing it is precisely how this project
previously produced fabricated closures. Handle it as the AC below specifies:
establish whether the vectors changed, and if they did, record a fresh baseline
and say explicitly that pre-merge numbers are not comparable.

## Acceptance criteria

- [ ] Pre-merge `HEAD` tagged with a recoverable name (e.g.
      `pre-upstream-merge-20260811`) and the tag reported in `## Answer`, before
      any merge command runs.
- [ ] `upstream/main` merged into `gfx1201_tp` via `git merge`. Every conflict
      resolution that touches a `ds4_tp*`, `ds4_rocm_xdev*`, or `rocm/` file is
      listed individually in `## Answer` with a one-line justification. "Took
      theirs" on a TP file is a red flag and must be explained.
- [ ] `make rocm` builds cleanly for `gfx1151,gfx1201` post-merge.
- [ ] `make -j8 test-rocm` passes, or each failure is triaged as pre-existing vs
      merge-introduced with evidence (e.g. the same test run at the pre-merge tag).
- [ ] Test-vector status determined and recorded: is
      `gguf-tools/quality-testing/test-vectors/flash-pre-0731/official.vec`
      content-identical to the pre-merge `official.vec`? Report the answer.
- [ ] A post-merge `score_official` baseline recorded on the pipeline path, with
      the manifest/vector set used named explicitly. If the vectors changed,
      state in `## Answer` that pre-merge figures (pipeline avg_nll 0.3742,
      TP=4 0.36985) are **not** comparable to it, and do not present a
      pass/fail against the old PRD numbers as if it were like-for-like.
- [ ] A short smoke generation on the pipeline path is coherent (guards against
      a merge that builds and passes tests but breaks inference).

## Isolation

Run this as its own `ralph.sh -f upstream-sync` invocation. It rewrites the
working tree; if it shares a loop with the `gfx1201-wmma-v2` GPU issues, an agent
will be benchmarking against a tree another agent is mid-merge on. Do not run the
two features concurrently.

## Notes for the agent

- Requires the GPU lock for the build/test/baseline steps (`AGENTS.md`), but not
  for the merge itself. Acquire it after conflicts are resolved.
- `docs/adr/0001-dense-tp4-parked-sequential-default.md` records why TP=4 is off
  by default. A merge must not change that default. If it does, that is a
  conflict resolution error, not an upstream feature.
- If the merge turns out to be genuinely large or ambiguous, stop and emit
  `READY-FOR-HUMAN` with the conflict list. Do not guess at TP semantics.

## Answer

## Comments

Filed 2026-08-11 from an interactive human session.
