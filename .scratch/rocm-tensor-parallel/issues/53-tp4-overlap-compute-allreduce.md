# 53 — Overlap layer N+1 compute with layer N's all-reduce

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/issues/33-tp4-throughput-measurement.md`

## What to build

With the async ring/tree all-reduce from #52 running on streams, overlap the
communication for layer N's all-reduce with compute for layer N+1 where data
dependencies allow, instead of waiting for each all-reduce to fully complete
before starting the next layer's compute.

This is the final-mile throughput step in the chain — expected to be a
smaller win than #50/#51 (dispatch/sync elimination) or #52 (async
collective), since most of the overhead this chain targets is host-side
blocking, not the underlying communication latency itself.

## Acceptance criteria

- [x] Layer N+1 compute begins before layer N's all-reduce fully completes,
      for the dependency-safe portion of the computation
- [ ] Per-token decode throughput measured and compared against #51/#52's
      numbers — report the actual overlap win, which may be small
- [ ] Full 100-case `score_official` quality fixture re-run (pipeline and
      TP=4) — overlap logic is easy to get subtly wrong in the same
      "silent corruption" way as #52, so do not skip this
- [ ] Findings recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`

## Blocked by

`.scratch/rocm-tensor-parallel/issues/52-tp4-hand-rolled-async-allreduce.md`
`.scratch/rocm-tensor-parallel/issues/51-tp4-execution-engine-full-rollout.md`

## Comments

**2026-08-01 — Reopened AC2-4 during a project-wide issue-tracker lint;
found closed with all 4 ACs checked and zero `Comments` or verification
narrative.** This is the same commit (`04ec7be`) already known from
[[tp4-issue51-full-rollout-status]] to have improperly closed #51 without
basis — it also self-closed this issue the same way.

**AC1 has real evidence and stays checked.** The diff adds actual overlap
code to `ds4.c` (`ds4_tp4_spike_worker_main` region, comment: "Issue #53:
Overlap layer N+1 compute with layer N's all-reduce across all 43 layers.
... asynchronously on its device stream. Layer N+1 compute begins as soon
as rank t finishes its local all-reduce, overlapping with peers.") — the
implementation this AC describes is genuinely in the tree.

**AC2-4 have no #53-specific verification anywhere.** The same commit's
`experiment-log.md` entry is entirely about correcting #57's fabricated
claims and #51's crash root-cause investigation; it produced
`q_pipeline_51.tsv`/`q_tp4_51.tsv` for **#51's** acceptance criteria, not as
an overlap-vs-no-overlap throughput comparison or a dedicated re-verification
of this change. No A/B throughput number for the overlap win exists, and no
run is documented as having exercised this code path specifically.

**Note for whoever picks this up:** the execution model has moved
significantly since this code was written — #56/#57 (compressor races),
#60 (43-layer rollout), and #61 (removing host `hipDeviceSynchronize`
barriers) all touched the same decode loop. Re-verify AC2-4 against
current HEAD, not against assumptions from when this code was first
written; the overlap logic may also need to be re-examined for interaction
with #61's changes before trusting it under concurrent load.

**2026-08-02 — Work in progress, stashed pending GPU availability. Pop
`stash@{0}` (SHA `e4f265c758ff40c80571566e0b7fa8ae57ecc326`, message
"issue-53: AC1 re-confirmed, AC2 throughput measured...") FIRST, before
doing anything else — it already contains real, verified progress on this
issue and re-doing it from scratch would duplicate work:**
- AC1 re-confirmed solid (the `#57`-followup consultant panel already
  traced this exact overlap path in detail; see the stashed Comments
  entry for why).
- AC2 done: a dedicated 3-run throughput measurement on current HEAD
  (`79e8181`) found 0.83/0.90/0.90 t/s generation (mean 0.88) — no
  detectable win over `#51`'s own recorded 0.52-1.52 t/s spread. Log at
  `quality-out/tp4_53_overlap_throughput.log` (already on disk,
  untracked, not stashed).
- AC3's TP=4 half formally deferred to `#63` (already scoped for it,
  blocked on `#64`) — a 9-model AI-consultant panel (7/9 converged) plus
  human sign-off backed this; don't re-litigate it, don't attempt a TP=4
  quality run here.
- `experiment-log.md` has a matching 2026-08-02 entry (also in the
  stash) with full detail and an **interim, not final** disposition.

**What's still genuinely outstanding after popping the stash:** AC3's
pipeline half needs one more fresh `score_official` 100-case run with a
proper provenance header — the existing `quality-out/q_pipeline_53.log`
(healthy, avg_nll=0.371, already on disk) predates the `#64` fix commit
by ~22 min and has no provenance header, so it was declined as evidence
by human direction; don't cite it as-is. Two other untracked files in
`quality-out/` are known-stale and should be disregarded, not cited, and
not staged into any commit: `q_tp4_51_full.{log,tsv}` (crashed after 1
case, pre-`#64`-fix) and the undated `q_pipeline_53.{log,tsv}` pair
itself (superseded once the fresh rerun below lands).

**GPU-lock protocol reminder:** acquire the lock
(`ralph_engine.py gpu-acquire rocm-tensor-parallel --agent-id "$$"`)
before any of this; if `LOCKED`, this issue stays `ready-for-agent` for
the next dispatch rather than blocking synchronously — don't loop
forever inside one session waiting for it.

**Remaining steps once the GPU is free** (after popping the stash):
1. Confirm VRAM idle on all 4 GPUs (`rocm-smi --showmeminfo vram`,
   expect tens of MB, not GB, per GPU).
2. Rebuild `score_official` fresh: `make ROCM_ARCH=gfx1201 rocm-quality`.
3. Run the 100-case pipeline fixture (see
   [[score-official-quality-fixture-invocation]] memory for the exact
   command/model/manifest paths and the `AMD_SERIALIZE_KERNEL=3`
   requirement) to `quality-out/q_pipeline_53_v2.{log,tsv}`, with a
   provenance header (HEAD SHA, build command, full invocation) written
   at the top of the log first.
4. Release the GPU lock (`ralph_engine.py gpu-release rocm-tensor-parallel`).
5. Compare the fresh `avg_nll` against the PRD bar (0.369-0.378) and
   against the existing 0.371 as a sanity check — flag clearly, don't
   paper over it, if they diverge meaningfully.
6. Finalize the `experiment-log.md` 2026-08-02 entry (it's currently
   marked interim) with this result and a final disposition.
7. Check off the remaining AC3/AC4 boxes below with the real result, and
   change `Status: ready-for-agent` to `Status: closed`.
8. One commit for everything (the popped stash's changes plus this
   session's): `feat(rocm-tensor-parallel): 53 — Overlap layer N+1
   compute with layer N's all-reduce`.
