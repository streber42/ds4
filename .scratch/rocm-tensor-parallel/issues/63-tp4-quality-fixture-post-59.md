# 63 — Re-run the TP=4 quality fixture against #57's counter-hoist fix once #59 lands

Status: ready-for-human

## Parent

`.scratch/rocm-tensor-parallel/issues/57-tp4-compressed-cache-concurrency-race.md`

## What to build

Split out of `#57` on human disposition 2026-08-02. `#57`'s counter-hoist/
rollback fix for the compressed-KV-cache race (commit `8a8f82a`, hardened by
the pre-increment+rollback work on top of it) is code-complete, three-rounds
AI-consultant-reviewed, and human-signed-off. What's missing is the one piece
of evidence that can only come from a working TP=4 build: a clean 100-case
`score_official` run confirming the fix doesn't regress quality.

That run is currently impossible, not just unscheduled — `#59` found TP=4
fails deterministically on `arena alloc failed for moe_down` before any case
scores, root-caused to an unbounded session-lifetime VRAM cache in the
batch-prefill MoE fallback path, not the static per-tier weight slab. `#59`
is `ready-for-human`, needing a decision between two new-issue-sized fix
candidates (see its Comments). This issue exists so `#57` doesn't stay open
indefinitely waiting on that decision plus its implementation — `#57` closes
now on its four satisfied ACs, and this issue picks up the deferred
measurement once TP=4 can initialize reliably.

**Note this gates the currently-shipped default, not just `#57`'s own
closure.** `#60` already relaxed `metal_graph_tp4_spike_layer_enabled` to
`return true` unconditionally for all 43 layers — i.e., the counter-hoist
fix this issue needs to verify is already load-bearing for every TP=4 run on
HEAD, whether or not this issue has run yet.

## Acceptance criteria

- [ ] Full 100-case `score_official` run on the **TP=4** path, against a HEAD
      build with `8a8f82a` (and its pre-increment+rollback hardening) as an
      ancestor, using `AMD_SERIALIZE_KERNEL=3` to match the pipeline
      comparison baseline (see `#62`'s serialize-agreement note)
- [ ] Report `avg_nll`, `first_match`, `api_top1_rate`, `api_pair_rate`
      against the PRD bar (avg_nll 0.370–0.378, first_match ≥60/100,
      api_top1_rate ≥0.85, api_pair_rate ≥0.98)
- [ ] Record per-case `avg_nll` distribution (median and <0.5/[0.5,1)/[1,2)/≥2
      bucket counts) — `#62` found the mean alone hides shape (uniform shift
      vs. episodic race have different bucket signatures)
- [ ] Count `q8 fp16 cache budget exhausted` and `arena alloc failed`
      occurrences in the log
- [x] Findings recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`

## Blocked by

*(nothing — see 2026-08-02 comment)*

## Comments

**2026-08-02 — Verification attempt failed; Status: ready-for-human.**
Attempted `score_official` execution in TP=4 mode with `AMD_SERIALIZE_KERNEL=3 --gpu-devices 0,1,2,3 --cuda-tensor-parallel`:
- `make -j8 test-rocm` passed 100% (4/4 test binaries green).
- `case_000` completed with `avg_nll = 16.321227` (PRD bar: 0.370-0.378; ~44x worse than pipeline baseline).
- `case_001` crashed deterministically: `ds4: ROCm prefill fallback copy failed for moe_down at 128.00/672.00 MiB: invalid argument` -> `gpu layer 0 ffn batch encode failed` -> `case_001 sync failed: rocm prefill failed`.
- Log warning counts: 44 `q8 fp16 cache budget exhausted` warnings; 0 `arena alloc failed` warnings.
- Findings recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`.
- Requires structural VRAM headroom / fallback fix (tracked in `#65`) and TP=4 quality divergence resolution before a full 100-case run can complete cleanly.

**2026-08-02 — Unblocked: #64 closed.** #64 closed with AC1 (zero `arena
alloc failed` warnings, verified live twice against the production model)
met — TP=4 now initializes cleanly, which is the only thing this issue was
waiting on. The residual ~1038 `arena-full skip`/run (legitimate
VRAM-scarcity fallback to the slower PCIe path, not an allocation failure)
is split into `#65` and does not block a clean 100-case run here.

**2026-08-02 — Split out of `#57` on human disposition.** `#57`'s AC3 had a
TP=4 half that was blocked on `#59` with no path to closure until `#59`'s own
fix (itself pending a human decision, per `#59`'s Comments) landed. Rather
than leave `#57` open indefinitely for a measurement it cannot influence,
the human chose to close `#57` on its four satisfied ACs (AC1/AC2/AC4/AC5)
and track the deferred TP=4 fixture run here.

**2026-08-02 — Re-pointed from #59 to #64.** #59 closed with a real,
measured improvement but did not eliminate the arena-alloc-failed warnings
entirely (see #59's final comment and the 2026-08-02 experiment-log entry).
This issue still needs a fully clean TP=4 initialization to produce a
trustworthy 100-case run, so it stays blocked, now on #64.

**2026-08-02 — Status normalized to `ready-for-agent`.** Was `Status: open`,
a non-canonical value the ralph engine parser falls back to `ready-for-human`
for (`KNOWN_STATUSES` in `ralph_engine.py` doesn't include `open`), so this
issue was invisible to `ralph unblocked`/agent dispatch despite having no
real blockers left (#64 closed). No scope change — ACs are unchanged and
already fully specified for an unattended agent run.

