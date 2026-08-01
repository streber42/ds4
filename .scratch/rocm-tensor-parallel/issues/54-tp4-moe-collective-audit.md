# 54 — Audit MoE routed/shared-expert dispatch for extra collectives

Status: closed

## Parent

`.scratch/rocm-tensor-parallel/issues/33-tp4-throughput-measurement.md`

## What to build

The expected TP=4 all-reduce count is ~86/token (2 per layer: post-attention,
post-MoE/FFN), and the AI consultant panel confirmed this is architecturally
normal and should not be reduced. However, two consultants (Mistral, Cursor)
separately raised an unconfirmed suspicion: DeepSeek-V4-Flash's MoE routing
(routed experts + shared expert, per issue #30/#41) might be generating
collectives beyond that 2-per-layer baseline — e.g. if routing/gating or the
shared-expert path triggers its own reduction rather than folding into the
single per-layer MoE all-reduce.

Using issue #49's per-call-site instrumentation, audit the MoE dispatch path
specifically to confirm or refute this. If extra collectives are found,
determine whether they can be batched into the standard 2-per-layer pattern
or whether they're a genuine architectural requirement of the routed+shared
expert split.

This is an independent track — it does not block or get blocked by #50-#53,
since it's investigating a different question (collective *count*, not
collective *execution model*). It only depends on #49's instrumentation
existing.

## Acceptance criteria

- [x] MoE routed/shared-expert dispatch path's collective count measured via
      #49's harness, per layer
- [x] Confirmed either: (a) MoE stays at exactly 1 all-reduce per layer (2
      total with attention), matching the expected baseline — report this
      and close, or (b) extra collectives found, with a fix that batches
      them into the standard pattern if possible, or documents why they're
      structurally required if not
- [x] Full 100-case `score_official` quality fixture re-run if any change was
      made to the MoE collective path — **n/a, no change was made** (outcome
      (a) below); nothing to re-verify.
- [x] Findings recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`
      — cites #49's pre-existing entry rather than a new run (see Comments).

## Blocked by

`.scratch/rocm-tensor-parallel/issues/49-instrument-tp4-sync-dispatch-call-sites.md`

## Comments

**2026-08-01 — Retroactively documented during a project-wide issue-tracker
lint; found closed with 0/4 ACs checked and no explanation.** Git blame
traces this file's closure to commit `4a21216`, whose message
("feat: 50 — Vertical spike...") is entirely about unrelated issue #50 and
never mentions #54 — this issue was created already marked `closed` with no
audit ever performed, the same class of mistake recorded in
[[tp4-issue-closure-scope-creep]].

The good news: #49's per-call-site instrumentation (`experiment-log.md`,
"2026-07-31 — TP=4 decode-loop sync/dispatch instrumented per call site
(issue 49)") already contains the exact data this issue asked for — it just
was never cross-referenced here. The 17-site breakdown shows exactly one
MoE-related collective site, `moe_allreduce`, at 43 calls/token (once per
layer), with no separate routed-expert or shared-expert collective site
anywhere in the table. Combined with `attn_allreduce` (also 43/token), that
is 86/token total — matching the expected baseline this issue's own "What
to build" section cites, with no extra collectives from the routed+shared
expert split. **Outcome (a): confirmed, no extra collectives, no fix
needed.** The two consultants' suspicion that motivated this issue is not
borne out by the instrumentation data.

Closing for real this time, on the evidence above, rather than re-running
#49's instrumentation from scratch — the data already answers the question
and re-collecting it would not change the answer.
