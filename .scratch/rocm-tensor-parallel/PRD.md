# PRD: ROCm/gfx1201 Tensor Parallelism for ds4

Status: ready-for-agent

## Problem Statement

I run DeepSeek-V4-Flash (an 80 GiB IQ2/Q2_K quant) locally on a workstation with four AMD
Radeon AI PRO R9700 cards (gfx1201, 32 GiB each, 128 GiB total VRAM) under ROCm. The model
does not fit on a single card, so ds4 runs it in distributed mode, splitting the 43 layers
into four contiguous ranges — one per GPU — and streaming activations between them.

This works and produces correct output, but it is **pipeline parallelism**: at batch size 1,
only one GPU is computing at any instant while the other three wait their turn. The result is
~28–29 tok/s single-stream generation and roughly **30% per-GPU utilization**. I am paying for
four GPUs and getting approximately one GPU's worth of compute, plus inter-stage latency.

Every other throughput lever in ds4 is unavailable to me because they were implemented for
other backends:

- **Continuous batching / session batching** — CUDA and Metal only. Concurrent requests on my
  setup serialize; measured 4 concurrent requests take 3.96× the wall time of one, and GPU
  utilization does not rise. Extra concurrency buys nothing.
- **MTP and DSpark speculative decoding** — single-node only. Silently ignored by the
  distributed coordinator (verified: no acknowledgement, no draft/accept activity even with
  debug logging enabled, decode speed unchanged). Since my model must be split across GPUs,
  these cannot help.
- **Tensor parallelism** — implemented only as CUDA (`--cuda-tensor-parallel`, kernels live in
  the CUDA translation unit) and Metal (two-machine RDMA). The ROCm build links no-op stubs
  for the entire tensor-parallel GPU surface. There is no ROCm tensor-parallel path anywhere
  upstream.

So the ~28 tok/s pipeline ceiling is not a tuning problem I can configure my way out of. It is
a missing implementation.

## Solution

Implement the tensor-parallel GPU compute path for the ROCm backend so that DeepSeek-V4-Flash
can run **tensor-parallel** across the R9700s — every GPU computing on every token — instead of
pipeline layer-split where they take turns.

Tensor parallelism shards each layer *across* ranks rather than assigning whole layers to
ranks: routed experts are split between ranks, attention heads are divided, and the vocabulary
output head is row-sharded. Ranks exchange partial results with an all-reduce-style bulk
transfer each layer. Because all ranks work on the same token simultaneously, per-GPU
utilization rises and single-stream latency falls.

The feasibility question that gates this — whether ROCm cross-GPU peer transfer works
correctly and fast enough on gfx1201 — has already been answered affirmatively (see Further
Notes). The transport and protocol layer is backend-agnostic and already compiles for ROCm.
What is missing is the GPU-side compute: 31 tensor-parallel kernel entry points currently
stubbed as no-op `return 0` in the ROCm build.

Success means, in order of priority:

1. **Correct** — tensor-parallel output matches single-GPU/pipeline reference logits.
2. **Faster** — beats the ~28–29 tok/s pipeline baseline on the same hardware and model.
3. **Utilized** — per-GPU utilization materially above the current ~30%.

Correctness strictly precedes speed. A fast wrong answer is worthless, and subtly wrong
sharded math is the primary risk in this work.

## User Stories

1. As a local LLM operator, I want DeepSeek-V4-Flash to run tensor-parallel across my four
   R9700s, so that all four GPUs compute on every token instead of taking turns.
2. As a local LLM operator, I want single-stream generation faster than the ~28 tok/s pipeline
   baseline, so that interactive use feels responsive.
3. As a local LLM operator, I want per-GPU utilization well above 30%, so that the hardware I
   bought is actually doing work.
4. As a local LLM operator, I want tensor-parallel output to be textually and numerically
   equivalent to the pipeline path, so that I can trust the faster mode.
5. As a local LLM operator, I want to enable tensor parallelism with an explicit flag, so that
   I can A/B it against the pipeline path without rebuilding.
6. As a local LLM operator, I want to fall back to pipeline layer-split if tensor parallelism
   fails or is unsupported, so that a regression never leaves me with no working inference.
7. As a local LLM operator, I want the OpenAI-compatible server to work unchanged in
   tensor-parallel mode, so that my existing tooling keeps pointing at the same endpoint and
   model name.
8. As a local LLM operator, I want tensor-parallel mode to respect my context-size setting, so
   that long-output workloads still work.
9. As a local LLM operator, I want a clear error when my GPU count or model shape is
   incompatible with tensor parallelism, so that I am not left guessing why it refused.
10. As a local LLM operator, I want the ported build packaged into the same container workflow
    I already use, so that bringing the server up stays a single command.
11. As a porting engineer, I want a single cross-device transfer module that hides peer-access
    setup, peer copies, and the host-staging fallback, so that no kernel has to reason about
    transport details.
12. As a porting engineer, I want the cross-device module to detect whether direct peer access
    is available and transparently fall back to host staging, so that the port still functions
    on topologies where peer transfer is unavailable or unreliable.
13. As a porting engineer, I want the cross-device module tested standalone with plain buffers,
    so that I can prove transport correctness without loading an 80 GiB model.
14. As a porting engineer, I want the sharding policy expressed as pure logic separate from GPU
    code, so that I can unit-test ownership maps on CPU with no hardware.
15. As a porting engineer, I want the sharding policy to cover routed experts, attention heads,
    and vocabulary rows, so that every sharded dimension has one authoritative source of truth.
16. As a porting engineer, I want each ported kernel numerically compared against a reference
    implementation, so that I catch math errors at the kernel level rather than debugging
    garbled text.
17. As a porting engineer, I want kernels ported in coherent waves grouped by subsystem, so
    that each wave is reviewable and independently landable.
18. As a porting engineer, I want the routed-MoE tensor-parallel kernels implemented, so that
    expert computation is split between ranks.
19. As a porting engineer, I want the attention tensor-parallel kernels implemented, so that
    attention heads are divided across ranks.
20. As a porting engineer, I want the matmul and shared-expert tensor-parallel kernels
    implemented, so that the remaining per-layer compute is sharded.
21. As a porting engineer, I want the tensor-parallel gate/synchronisation hooks implemented,
    so that ranks stay in lockstep across the pipeline of exchanges.
22. As a porting engineer, I want the cross-device accumulate operation implemented, so that
    partial results from each rank combine into the correct whole result.
23. As a porting engineer, I want an end-to-end logits comparison against the reference path,
    so that I have a single authoritative go/no-go correctness signal.
24. As a porting engineer, I want the existing quality fixture run against the tensor-parallel
    build, so that I detect quality regressions that a single prompt would miss.
25. As a porting engineer, I want to validate on two GPUs before extending to four, so that I
    debug the simplest possible sharded configuration first.
26. As a porting engineer, I want a documented plan for going from the native two-rank design
    to four GPUs, so that the extension is a deliberate decision rather than an accident.
27. As a porting engineer, I want throughput and utilization measured against the recorded
    pipeline baseline, so that I can prove the port actually delivers a win.
28. As a porting engineer, I want to know early if tensor parallelism is *slower* than
    pipeline, so that I can stop or re-scope rather than finishing a port that does not help.
29. As a reviewer, I want the discrete-GPU behaviour preserved from the existing branch, so
    that the rebase onto upstream does not silently regress cards with separate VRAM.
30. As a reviewer, I want changes isolated to the ROCm backend, so that CUDA and Metal users
    are unaffected by this work.
31. As an AFK agent, I want each issue to state its own verification command, so that I can
    confirm my change works without human context.
32. As an AFK agent, I want issues ordered by dependency, so that I never pick up work whose
    prerequisites do not exist yet.

## Implementation Decisions

**Baseline already established.** The branch carrying discrete-GPU memory management and
gfx12 (RDNA 4) WMMA intrinsics has been rebased onto current upstream, which contains CUDA and
Metal tensor parallelism, DSpark, and session batching. The one merge conflict — competing
rewrites of the streaming expert-cache allocator — was resolved by taking upstream's slab-pool
allocator while preserving the discrete-GPU detection guards that matter for full residency.
The rebased tree builds for gfx1201, exposes the tensor-parallel and DSpark flags, and was
verified to still produce correct output. All subsequent work builds on this.

**Scope of the missing surface.** 31 tensor-parallel GPU entry points are currently stubbed as
no-op integer returns in the ROCm build. They break down as: attention (7), routed MoE (5),
matmul (4), shared expert (3), tensor-parallel gate synchronisation (2), DSpark (1), and
assorted device-cache, model-map registration, MoE handoff, KV, rope, and indexer hooks. Each
has a CUDA counterpart to port from. The ROCm kernels are hand-written HIP in modular headers
rather than generated from the CUDA source, so each port is a deliberate adaptation, not a
mechanical copy.

**Deep module: cross-device transfer.** A new ROCm module owns all peer-to-peer concerns
behind a narrow interface — establish the peer mesh, copy a buffer between devices, and
accumulate a buffer from another device. It internally handles peer-access capability
detection, enabling access in both directions, and falling back to host staging when direct
peer transfer is unavailable. No kernel or engine code calls peer-transfer APIs directly. This
is the single place transport behaviour can change without touching compute.

**Deep module: sharding policy.** Ownership decisions are pure functions of model dimensions,
rank count, and rank index, with no GPU dependency: which routed experts a rank owns, which
attention heads it computes, and which vocabulary rows it produces. Keeping this free of
device code makes the trickiest correctness logic testable on CPU and gives kernels a single
authoritative source for ownership.

**Transport and protocol are reused, not rewritten.** The tensor-parallel session protocol
(session create/destroy, sync, eval dispatch, logits-half exchange) is backend-agnostic,
already compiles for ROCm, and is not part of this work. Only GPU compute and the cross-device
primitive are being implemented.

**Two ranks first, then four.** Upstream's tensor parallelism is natively a two-rank design
(50/50 expert sharding, attention head split, row-sharded vocabulary head). The port targets
two ranks first because it is the configuration the upstream code actually implements and the
simplest to debug. Extending to four GPUs is a separate, explicit decision — either two
tensor-parallel pairs arranged in a pipeline, or widening the design to four ranks — and is
deferred until two-rank correctness is proven.

**Correctness gates speed, always.** No performance work begins until end-to-end logits match
the reference. Each kernel wave lands only with its numeric-equivalence evidence.

**Fallback is mandatory.** If tensor parallelism is unavailable, refused, or fails to
initialise, the engine falls back to the existing pipeline layer-split path. The user must
never end up with no working inference because of this feature.

**Changes stay inside the ROCm backend.** CUDA and Metal paths are not modified. The shared
engine is touched only where a ROCm code path must be selected.

**Ordering of work.** Cross-device primitive and sharding policy first (they are dependencies
of everything and are independently testable), then the correctness harness, then kernel waves
grouped by subsystem, then measurement, then the four-GPU extension, then packaging.

## Testing Decisions

**What makes a good test here.** Tests assert externally observable behaviour — bytes that
arrive, ownership maps produced, numeric outputs of a kernel, logits from the engine — never
internal structure. A test should fail for exactly one reason and say plainly what diverged.
For numeric work, tests compare against a reference with an explicit tolerance rather than
asserting bit-identity, because floating-point reassociation across a different sharding is
expected; the tolerance is stated and justified per test rather than loosened until green.
Tests must not require the full 80 GiB model unless they are explicitly the end-to-end tier,
so that the fast tiers stay fast.

**Seam selection.** Testing is organised around as few seams as possible, placed as high as
possible, preferring seams that already exist. Three seams are mandatory; a fourth exists as a
diagnostic rather than a standing obligation.

1. **End-to-end logits (highest seam, already exists) — the gate.** The whole engine in
   tensor-parallel mode compared against the reference path, plus the project's existing
   multi-case quality fixture. This is the authoritative correctness signal and everything else
   is justified only by what it catches that this cannot.
2. **Cross-device primitive (new seam).** Justified because the host-staging fallback path is
   *unreachable from higher seams* — there is no way to force it end-to-end — and because
   silent transport corruption would be invisible until it surfaced as garbage text. Standalone
   tests over plain device buffers: peer mesh establishment, byte-exact copy across every
   ordered device pair, accumulate correctness, staging-fallback correctness, and a bandwidth
   floor. No model required. Productionises the throwaway feasibility probe already written.
3. **Sharding policy (new seam, placed at the highest point ownership logic exists).** Justified
   because it is a pure function whose tests cost almost nothing, and because an ownership
   partition bug silently corrupts everything downstream while being expensive to localize from
   the top. Pure CPU unit tests: complete partition, no gaps, no overlaps, uneven division, and
   the single-rank degenerate case.
4. **Kernel numeric equivalence (lowest seam) — deliberately NOT a per-kernel obligation.**
   Requiring a standing test for each of the ~31 kernels would mean 31 low seams, which is the
   opposite of the discipline above. Instead: the **first kernel ported in each subsystem** gets
   equivalence evidence, which is enough to prove the porting approach is numerically sound
   before it is repeated across that subsystem; thereafter end-to-end logits is the gate and the
   comparison scaffold is used **on demand to localize** a failure. Detection stays at the high
   seam; the low seam becomes a debugging tool. This trade is only affordable because the
   scaffold makes pointing at an arbitrary kernel cheap, which is why building it is its own
   prefactor.

**Prefactoring before porting.** Two changes land before kernel work begins, on the principle
of making the change easy before making the change. First, unimplemented tensor-parallel entry
points are switched from returning a neutral value to failing loudly and naming themselves —
otherwise every not-yet-ported kernel silently corrupts output for the whole duration of the
port, which is precisely the primary risk this project is trying to avoid. A single explicit,
self-announcing bring-up mode restores neutral returns for plumbing validation only. Second,
the kernel comparison scaffold is built once rather than improvised per kernel.

**Prior art in this codebase to follow.** There is an existing kernel-level numeric dot-product
test to model the kernel-equivalence tier on. There are existing multi-GPU placement, refusal,
and runtime tests that establish the pattern for testing ownership and for asserting graceful
refusal on unsupported configurations. There is a built-in diagnostic that resets and replays a
prompt then compares logits, which is the natural backbone of the end-to-end tier. There is an
existing evaluation harness with an official multi-case fixture for quality scoring. New tests
should extend these patterns rather than invent parallel infrastructure.

**Reference selection.** For kernel and end-to-end tiers the reference is the existing,
already-trusted ROCm single-GPU / pipeline path on the same hardware and quantisation — not the
CUDA implementation on different hardware — so that differences are attributable to sharding
rather than to backend or hardware variation.

## Out of Scope

- **Continuous batching / session batching for ROCm.** A separate CUDA/Metal-only feature. This
  PRD targets single-stream latency and utilization, not multi-request throughput.
- **MTP and DSpark speculative decoding in distributed mode.** These are single-node features;
  making them work across split models is a different problem.
- **Metal two-machine RDMA tensor parallelism.** Different backend, different transport.
- **CUDA or Metal changes of any kind**, including refactoring shared code for elegance.
- **Multi-machine tensor parallelism.** This work is intra-host across local GPUs only.
- **Upstreaming.** Getting these changes accepted by the upstream project is a possible later
  goal, not a requirement here.
- **Supporting models other than DeepSeek-V4-Flash**, and quantisations other than the ones
  currently in use. GLM and the PRO variant are not targets.
- **Beating a tensor-parallel CUDA or vLLM deployment.** The bar is this machine's own pipeline
  baseline.
- **Restoring the host-mapped streaming expert-cache fallback** that was dropped during
  conflict resolution. It only affects SSD-streaming mode on VRAM-constrained discrete cards,
  which this deployment does not use.

## Further Notes

**The feasibility gate already passed.** Before committing to this work, the load-bearing
assumption was tested directly: a standalone probe enabled peer access across all four cards
and exercised peer copies. Results — a full peer mesh (every pair reports direct access
capability), **all twelve ordered device pairs byte-correct with no corruption**, ~24–25 GB/s
uniformly across pairs, and a host-staging fallback measured at 13.5 GB/s. Tensor parallelism
moves only a few MB per token, so at 24 GB/s inter-GPU transfer costs well under a millisecond
against a ~35 ms per-token compute budget. **Transport bandwidth is not a bottleneck, and
silent data corruption — the failure mode that would have killed this project — does not
occur.**

**Why this succeeds where another engine's tensor parallelism failed on the same cards.** The
widely-reported multi-GPU failures on this hardware come from a collective-communications
library that relies on IPC handles and advanced peer features which are unreliable on this
architecture. ds4 does not use that library. It uses its own session protocol plus plain peer
copies — precisely the mechanism proven correct above. Avoiding the collective library is a
structural advantage, not a coincidence.

**Baseline to beat, measured on this machine.** Pipeline layer-split across four R9700s with
the 80 GiB quant: ~28–29 tok/s generation, ~10 t/s prefill on short prompts, ~30% per-GPU
utilization, four concurrent requests taking 3.96× single-request wall time. Single-GPU with
SSD streaming: ~4.9 tok/s. These are the numbers any claimed improvement is measured against.

**Primary risk.** Subtly incorrect sharded mathematics that produces plausible-looking but
wrong output. This is why all four test tiers were chosen, why correctness strictly precedes
performance, and why the reference is the same-hardware pipeline path.

**Secondary risk.** That correct tensor parallelism turns out no faster than pipeline on this
topology. The two-rank milestone is deliberately positioned as an early measurement point so
this can be discovered before the remaining kernels are ported.
