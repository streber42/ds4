# Dense TP=4 is parked: correct but sync-bound; decode defaults to the legacy sequential path

The ROCm/gfx1201 tensor-parallel port (PRD:
`.scratch/rocm-tensor-parallel/PRD.md`, issues #00–#66) delivered its
correctness goal and missed its performance goal by an order of magnitude. As
of 2026-08-04 (tag `tp-parked-v1`) the project is deliberately parked: TP=4
stays in the tree, fully validated and off by default, and the pipeline
layer-split path remains the daily driver.

**The facts behind the decision.** Dense TP=4 output quality is at parity
with the pipeline reference (full 100-case fixture: TP=4 avg_nll 0.3699 vs
pipeline 0.3742, both passing every PRD bar; measured under
`AMD_SERIALIZE_KERNEL=3`). But decode throughput is ~2.87 t/s against the
PP=4 baseline of ~22–28 t/s. The bound is architectural, not hardware: dense
TP=4 performs **86 hidden-state all-reduces per token**, and at batch=1 each
is a synchronous PCIe round-trip; even at an optimistic ~1 ms per exchange
that is ~86 ms/token — 2.5× over the ~35 ms budget that 28 t/s implies —
before any compute. Supporting evidence that the hardware is not the limit:
hipfire reports 25.6 t/s for the same model on the same 4× R9700 using
*expert-parallel* sharding, which avoids per-token hidden-state collectives
(explored in issue #68).

**Why the sequential path is the default.** The persistent-thread execution
engine built to hide the all-reduce latency (#50/#53/#60) has a cross-layer
data race on its per-tier peer-partial buffers (bisected in #66: a lagging
rank's peer read can be clobbered by the owner's next-layer write; systematic
logit shift ≈ −13). It is disabled by default
(`metal_graph_tp4_spike_layer_enabled` = 0 unless `DS4_TP4_THREADED_LAYERS`
is set) because correctness gates speed, and even when racing it measured
only 2.01 t/s. The proper fix is sketched in issue #67.

**Consequences.**

- A future reader will find a complete, quality-validated TP=4 implementation
  that is ~10× slower than pipeline and off by default. That is intentional,
  not neglect.
- Do not resume dense-TP=4 perf work by "just fixing" #67; the parked
  analysis says the win is capped unless the per-token synchronization
  *count* collapses. Re-entry points, in order of expected value: #68
  (expert-parallel decode), then #67 (threaded-engine race fix).
- The port paid for itself outside TP: the shared-MoE unselected-expert fix
  (`0725d69`) corrected pipeline-path math, and the slab/arena work
  (#59/#64/#65) removed ~1019 per-token PCIe host-register fallbacks.

**Considered alternatives.** (a) Keep optimizing dense TP=4 — rejected: no
written design gets 86 synchronizations to a number that beats 28 t/s on
PCIe. (b) Delete the TP=4 path — rejected: it is correct, validated, inert
when unflagged, and the only A/B reference for any future sharding work.
(c) Park with the fast-but-racing engine as default — rejected outright: a
fast wrong answer is worthless (PRD, "Correctness gates speed, always").
