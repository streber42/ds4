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

- [x] hipfire's 25.6 t/s claim independently reproduced (or refuted) on this
      machine, with the config recorded — **refuted**, see 2026-08-07 comment
- [ ] A written go/no-go recommendation: expected decode t/s for an ds4 EP
      path, the per-token communication budget backing that estimate, and the
      implementation scope — or a documented decision not to proceed
- [ ] If go: a PRD-level plan (this issue does not authorize implementation)

## Blocked by

*(None mechanically. Held at `ready-for-human`: research direction, requires
a human decision to invest.)*

## Comments

**2026-08-04 — Filed as part of the project park (wrap-up of #49–#66).**

**2026-08-07 — AC1 resolved: the motivating 25.6 t/s number is refuted, not
reproduced.** A sibling benchmark campaign in `/home/murphy/src/dev_ds4`
(`.scratch/deepseek-benchmark/`) independently tested hipfire's `serve --tp 4`
EP path on this exact 4× R9700 hardware, for DeepSeek V4 Flash, one day after
this project's park (2026-08-05, issue T5/C3,
`.scratch/deepseek-benchmark/issues/05-t5-c3-hipfire-mq2-tp4.md`, commit
`264d09d` in that repo). Measured **prefill 15.7 t/s, decode 4.0 t/s** at
pp=2048 (3/3 runs) — not 25.6 t/s, and **6.6× slower** than this project's own
PP=4 pipeline baseline (~26.5–28 t/s), not parity with it. At pp≥8192,
hipfire's EP decode loop deadlocked the daemon (`ep_serve_ds4`'s watchdog never
fires); reproduced twice. Root cause per that issue: hipfire's EP path runs
plain autoregressive decode with **no MTP speculative decode** (unlike its
non-EP/pipeline path), and per-step cost grows with KV length — the 25.6 t/s
figure is very likely a short-context/small-batch number, or reflects a
hipfire build/config that campaign didn't reproduce. hipfire is under active
development (pushed 2026-08-07, per its GitHub metadata), so this is a
point-in-time result, not a permanent verdict on EP as a technique — but the
specific empirical claim that made this issue's priority ordering (#68 before
#67) no longer holds as measured.

**Recommendation on AC2 (informational, not a go/no-go decision — that stays
with a human per this issue's `ready-for-human` status):** don't commission
new ds4-native EP implementation work on the strength of the hipfire number
alone; it's gone. If EP is still wanted, the honest next step is either (a)
re-test hipfire's current `main` (it has moved since 2026-08-05) with a
longer, context-controlled protocol to see if 25.6 t/s ever reappears under
conditions the sibling campaign's harness didn't hit, or (b) treat this issue
as unblocked-but-unpromising and deprioritize it below the WMMA v2 lead in
`../gfx1201-wmma-v2/` (cheaper, orthogonal to TP/EP, never tested). Full
citations in `/home/murphy/src/dev_ds4/.scratch/deepseek-benchmark/artifacts/research-ds4-r9700-rocm.md`.
