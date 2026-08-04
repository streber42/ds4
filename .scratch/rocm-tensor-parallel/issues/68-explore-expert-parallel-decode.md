# 68 — Explore an expert-parallel (EP) decode path as the alternative to dense TP=4

Status: ready-for-human

## Parent

`.scratch/rocm-tensor-parallel/PRD.md`

## What to build

Research/scoping issue, filed at project park (2026-08-04) — see ADR
`docs/adr/0001-dense-tp4-parked-sequential-default.md` for the full context.
This is a direction to *evaluate*, not committed work.

**Motivating evidence:** hipfire (https://github.com/warpfront/hipfire) claims
25.6 t/s decode for DeepSeek V4 Flash (82 GB MQ2-Lloyd quant) on the same
hardware as this project — 4× AMD R9700 — using `serve --tp 4`, which its
README describes as expert-parallel (EP) sharding, not dense tensor
parallelism. That is parity with our PP=4 pipeline baseline (~22–28 t/s) and
~9× our dense TP=4 result (2.87 t/s).

**Why EP plausibly avoids our bottleneck:** dense TP=4 pays 86 hidden-state
all-reduces per token — the measured PCIe latency floor that parked this
project. EP instead assigns whole routed experts to GPUs and moves only the
per-token activations for the experts actually selected (a dispatch/combine
exchange per MoE layer, with attention replicated or data-parallel rather
than head-sharded). The synchronization count per token drops roughly from 86
collectives to ~43 small routed exchanges, and each exchange moves less data.
Notably, this project already owns the building blocks: the sharding-policy
module (`ds4_tp_shard.h`) already expresses routed-expert ownership as pure
logic, and the cross-device module (`ds4_rocm_xdev`) already does verified
peer copies at ~24 GB/s.

Open questions this issue exists to answer before any implementation is
approved:

1. Verify the hipfire claim on our own hardware (build/run it against our
   quant or theirs; confirm batch size 1 and measure honestly).
2. If EP only matches PP=4 (~25 vs ~28 t/s), is there any reason to build it
   at all? The PRD's bar was *beating* pipeline. Utilization alone is not a
   win at equal throughput. State the go/no-go case explicitly.
3. What would EP mean for ds4's existing distributed session protocol —
   is this a new sharding mode inside the TP surface, or a different
   coordinator topology?
4. Where does attention live under EP for MLA (one shared KV latent,
   64 heads)? Replication interacts with the VRAM budget that #59/#64/#65
   fought for.

## Acceptance criteria

- [ ] hipfire's 25.6 t/s claim independently reproduced (or refuted) on this
      machine, with the config recorded
- [ ] A written go/no-go recommendation: expected decode t/s for an ds4 EP
      path, the per-token communication budget backing that estimate, and the
      implementation scope — or a documented decision not to proceed
- [ ] If go: a PRD-level plan (this issue does not authorize implementation)

## Blocked by

*(None mechanically. Held at `ready-for-human`: research direction, requires
a human decision to invest.)*

## Comments

**2026-08-04 — Filed as part of the project park (wrap-up of #49–#66).**
