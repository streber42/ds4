# 53 — Overlap layer N+1 compute with layer N's all-reduce

Status: closed

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
- [x] Per-token decode throughput measured and compared against #51/#52's
      numbers — report the actual overlap win, which may be small.
      **Measured 2026-08-02**: 3 GPU-locked runs, current HEAD (`79e8181`),
      `DS4_TP4_INSTRUMENT=1 ./ds4 --rocm --gpu-devices 0,1,2,3
      --cuda-tensor-parallel --model /home/murphy/src/ds4/ds4flash.gguf -c
      64 -p "The capital of France is" -n 40` — generation 0.83/0.90/0.90
      t/s (mean 0.88 t/s). This sits squarely inside `#51`'s own recorded
      0.52-1.52 t/s spread: **no detectable overlap win**, matching this
      issue's own prediction going in. No runtime toggle exists for a
      clean overlap-on/off A/B, so this is a before/after-in-time
      comparison, not an isolated ablation. Full detail and per-run table
      in `experiment-log.md`'s 2026-08-02 entry; log:
      `quality-out/tp4_53_overlap_throughput.log`.
- [x] Full 100-case `score_official` quality fixture re-run (pipeline and
      TP=4) — overlap logic is easy to get subtly wrong in the same
      "silent corruption" way as #52, so do not skip this.
      **TP=4 half deferred to #63** (2026-08-02, human disposition,
      AI-consultant-panel-reviewed — 7/9 consulted converged on this):
      `#63` already exists with the identical scope, split out of `#57`
      for the same reason, and is blocked on `#64`'s still-open human
      judgment call on the residual arena-full-skip rate. Running a fresh
      TP=4 fixture here would duplicate `#63`'s job or pre-empt `#64`'s
      decision. **Pipeline half done 2026-08-02**: fresh 100-case run on
      current HEAD, `avg_nll=0.371050003` — inside the PRD bar
      (0.369-0.378) and consistent with the prior (uncitable,
      no-provenance) 0.371 figure. Log: `quality-out/q_pipeline_53_v2.log`
      / `.tsv`, provenance header included. See `experiment-log.md`'s
      2026-08-02 follow-up entry for full detail.
- [x] Findings recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`
      — AC1/AC2 entry plus the 2026-08-02 follow-up entry with the
      pipeline rerun result and final disposition.

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

**2026-08-02 — AC1 re-confirmed, AC2 measured, AC3 split, closure held
open pending one more GPU run.** AC1's "re-examine against #61" concern
is discharged: the `#57` followup's Round 2/3 AI-consultant panel review
already traced the full-token overlap path in detail (Cursor explicitly
confirmed the "workers aren't synchronized per-layer" property is this
issue's/`#60`'s deliberate design, not a bug). Ran the AC3/AC2 disposition
below past `/ai-consultants:consult` (9/10 responded) before acting on it;
7/9 substantive responses converged on deferring AC3's TP=4 arm to `#63`
(already scoped for exactly this) and not citing `#64`'s incidental 0.59
t/s figure for AC2 since it falls inside `#51`'s own noise band — ran a
dedicated throughput measurement instead (see AC2 above). Two stale
artifacts sitting in `quality-out/` should be disregarded: `q_tp4_51_full.*`
crashed after 1 case and predates the `#64` fix; `q_pipeline_53.*` is
healthy but never exercises this issue's code path (pipeline mode has
`cuda_tensor_parallel=0`) and also predates `#64` without a provenance
header. Human directed holding this issue open (rather than closing on
existing evidence) until a fresh, properly-provenanced pipeline rerun
lands — GPUs were busy at decision time, so that rerun is queued to run
automatically once the GPU lock is free; see `experiment-log.md` for the
live status.

**2026-08-02 (cont'd) — this progress had been stashed (`e4f265c`,
"issue-53: AC1 re-confirmed, AC2 throughput measured...") pending GPU
availability, then partially reconciled into the tree by a follow-up
session that also attempted the pipeline rerun but crashed mid-model-load
(stale `gpu.lock` held by a dead PID, `dev-vllm` left stopped, no `.tsv`
produced). Reconciled the remaining stash content into this file by hand
during a live human-paired session; the stash object itself is now
unreachable (dropped from `git stash list`, superseded by unrelated work)
and is not needed further. Pipeline rerun retried below with the same
GPU-lock/build/provenance protocol, this time backgrounded via `nohup` to
avoid the foreground-timeout kill that likely caused the earlier crash.

**2026-08-02 (final) — Pipeline rerun complete, issue closed.** Rebuilt
`score_official` fresh against current HEAD and ran the 100-case pipeline
fixture backgrounded (~3h20m wall-clock — slow but not stuck; pipeline
mode's sequential 4-GPU-hop-per-token under `AMD_SERIALIZE_KERNEL=3` is
simply slow at this scale, confirmed via steady per-case progress
throughout). Result: 100/100 cases, exit 0, `avg_nll=0.371050003`
(token-weighted, hand-verified against the raw per-case columns) —
inside the PRD bar and matching the prior uncitable 0.371 figure with no
meaningful divergence. AC3's TP=4 half stays deferred to `#63` per the
existing human-approved disposition above. All four ACs now satisfied;
see `experiment-log.md`'s matching final entry for full detail.
