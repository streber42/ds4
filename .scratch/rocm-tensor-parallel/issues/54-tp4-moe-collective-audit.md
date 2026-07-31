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

- [ ] MoE routed/shared-expert dispatch path's collective count measured via
      #49's harness, per layer
- [ ] Confirmed either: (a) MoE stays at exactly 1 all-reduce per layer (2
      total with attention), matching the expected baseline — report this
      and close, or (b) extra collectives found, with a fix that batches
      them into the standard pattern if possible, or documents why they're
      structurally required if not
- [ ] Full 100-case `score_official` quality fixture re-run if any change was
      made to the MoE collective path
- [ ] Findings recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`

## Blocked by

`.scratch/rocm-tensor-parallel/issues/49-instrument-tp4-sync-dispatch-call-sites.md`
