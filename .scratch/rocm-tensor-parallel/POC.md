# ROCm/gfx1201 Tensor Parallelism — Proof of Concept

Tag: `tp-poc-v1` (at commit `59ba21f`)

Status: **proof of concept, not production-ready.** The tensor-parallel GPU compute path
ports cleanly and is numerically correct — but only when kernel dispatch is serialized. The
default (fast) path has a known, unresolved multi-GPU dispatch-ordering bug (also present in
plain pipeline mode) that produces incoherent output. See [Known limitations](#known-limitations).

This is the shareable output of the PRD at [`PRD.md`](PRD.md): implementing the 31 stubbed
ROCm tensor-parallel GPU entry points so DeepSeek-V4-Flash can run tensor-parallel across 4x
AMD R9700 (gfx1201) instead of pipeline layer-split only.

## What works

- All GPU compute entry points required for two-rank tensor parallelism are ported and
  reachable (see [What was ported](#what-was-ported)).
- Four-GPU topology (Option A: two TP pairs pipelined across 4 GPUs, see
  [`issues/11-four-gpu-topology.md`](issues/11-four-gpu-topology.md)) loads the full 81 GiB
  production quant and produces correct output — under `AMD_SERIALIZE_KERNEL=3`.
- Graceful refusal and pipeline fallback for unsupported configurations
  ([`issues/09-refusal-and-fallback.md`](issues/09-refusal-and-fallback.md)).
- Container packaging works end-to-end for both TP and pipeline profiles
  ([`issues/12-package-container.md`](issues/12-package-container.md)).

## What does not work

- **Default (unserialized) TP and pipeline dispatch produce incoherent output.** This is the
  fast path — the one this whole PRD exists to make usable — and it is currently broken. See
  [Known limitations](#known-limitations).
- Post-dispatch-race-fix throughput has not been re-measured
  ([`issues/19-re-measure-throughput-post-fix.md`](issues/19-re-measure-throughput-post-fix.md)
  is still open); the numbers below are the last confirmed pre-fix baselines.
- MoE hot-path optimization (#20) and kernel-launch-overhead reduction (#22) are blocked on
  #19 and have not started.

## Correctness evidence

Authoritative signal: the official 100-case quality fixture (`score_official`,
`gguf-tools/quality-testing/data/flash`), scored as `avg_nll` (lower is better) against the
same-hardware ROCm pipeline reference. Full detail in
[`experiment-log.md`](experiment-log.md).

| config | avg_nll | first_match | avg_lcp | notes |
|---|---|---|---|---|
| 4-GPU pipeline, `AMD_SERIALIZE_KERNEL=3` | 0.373815 | 64/100 | 5.810 | reference baseline |
| **4-GPU TP, `AMD_SERIALIZE_KERNEL=3`** | **0.369930** | **68/100** | **6.700** | **TP 1.04% better than pipeline — numerically sound** |
| 4-GPU pipeline, default (unserialized), 2026-07-25 | 1.355060 | 1/100 | 0.010 | dispatch race — 3.6x worse |
| 4-GPU TP, default (unserialized), 2026-07-25 | 3.076559 | 0/100 | 0.000 | dispatch race — 8.3x worse |
| 4-GPU pipeline, default, post `xdev` fix, 2026-07-26 | 1.295131 | 0/100 | — | ~4% better, still broken |
| 4-GPU TP, default, post `xdev` fix, 2026-07-26 | 3.086362 | 0/100 | — | unchanged — fix did not close the gap |

**Reading this table:** under `AMD_SERIALIZE_KERNEL=3` (which forces every kernel launch to
fully drain before the next starts), tensor-parallel output is quality-equivalent to — very
slightly better than — the existing pipeline reference on the same hardware and quantization.
This is the proof that the ported TP kernel math is correct: the per-case score spread is
symmetric (largest TP win `case_052` -4.90 nll, largest TP loss `case_091` +3.30 nll), the
signature of floating-point reassociation across a different sharding, not of a sharded-math
bug. Without serialization, both multi-GPU paths are corrupted by a still-unresolved dispatch
race (see below); a first attempted fix ([#18](issues/18-fix-cross-device-dispatch-race.md))
closed one confirmed instance of the bug pattern but did not close the quality gap.

Raw per-case TSVs and logs: [`quality-out/`](quality-out/).

Plain-chat spot check (same conclusion, qualitative): under serialization,
`"Explain C pointers in one sentence."` at `temp 0` produces fluent, reference-matching output
in 4-GPU TP mode. Without serialization, both TP and pipeline mode produce sustained
mixed-script, non-linguistic noise. See [`issues/11-four-gpu-topology.md`](issues/11-four-gpu-topology.md)
and [`issues/12-package-container.md`](issues/12-package-container.md) for the full repro history.

## Performance numbers

Two independent measurements exist and have not been reconciled — recorded honestly rather
than picking one:

**`ds4-bench` sweep** (`--ctx-start 2048 --gen-tokens 256`, pre-fix, 2026-07-25, from
[`issues/19-re-measure-throughput-post-fix.md`](issues/19-re-measure-throughput-post-fix.md)):

| config | tok/s |
|---|---|
| Pipeline, default | 21.99 |
| Pipeline, `AMD_SERIALIZE_KERNEL=3` | 13.57 |
| TP, default | 12.17 |
| TP, `AMD_SERIALIZE_KERNEL=3` | 6.77 |

**Live server chat completion** (Option A, two TP pairs pipelined, 4-GPU, real 81 GiB model,
from [`experiment-log.md`](experiment-log.md) 2026-07-25 container run):

| config | decode tok/s | prefill tok/s |
|---|---|---|
| Pipeline (4-way layer split) | 28.0–28.5 | — |
| TP (2 pairs pipelined) | 5.0–5.5 | 0.93 |

The bench-tool number (~12 t/s default TP) and the live-server number (~5 t/s) were measured
with different harnesses, likely different prompt/output lengths, and were never reconciled
against each other — that reconciliation, plus a clean post-#18 baseline, is exactly what
issue #19 (open) exists to produce. Both agree on the qualitative finding: **TP is currently
slower than plain pipeline layer-split on this hardware**, the opposite of the PRD's stated
goal, and PRD success criterion 2 ("faster than pipeline") is not yet met by either
measurement.

**Why the ~12 t/s (or ~5 t/s) ceiling.** Profiling ([#20](issues/20-moe-hot-path-optimization.md),
[#22](issues/22-reduce-kernel-launch-overhead.md)) attributes this to two compounding costs
specific to TP's finer-grained, more cross-device-heavy execution:

- **MoE dominates GPU time (~72%)**: 57% IQ2 hot-WMMA path, 15% f32 cold-path fallback. The
  single most expensive kernel, `moe_gate_up_mid_expert_tile8_rowspan_kernel`, averages 33ms
  per call and fires 344 times per run.
- **Kernel launch overhead is significant relative to kernel size**: each decode token fires
  ~2000 kernel dispatches across 4 GPUs (529 per GPU), many of them 5–15μs micro-kernels
  where the ~5–10μs HIP launch overhead is comparable to the compute itself. TP's extra
  cross-device synchronization and combine steps add to this dispatch count relative to
  pipeline mode, which only synchronizes at stage boundaries.

Neither optimization has landed yet (#20, #22 both blocked on #19's clean baseline).

## Known limitations

1. **Default (unserialized) dispatch produces incoherent output — open, unresolved.** This is
   the primary blocker to calling this PoC production-ready. `AMD_SERIALIZE_KERNEL=3` /
   `HIP_LAUNCH_BLOCKING=1` is currently the *only* way to get trustworthy output from either TP
   or pipeline mode on this hardware, at a measured ~40–60% throughput cost. Root cause: a
   dispatch-ordering race on gfx1201's hardware scheduler (HWS), which can reorder kernel
   execution across what looks like a proper stream/device barrier. One confirmed instance —
   `hipDeviceSynchronize()` used instead of explicit `hipEventRecord`/`hipStreamWaitEvent`
   ordering at the cross-device peer-copy sync point in `ds4_rocm_xdev.cu` and
   `ds4_gpu_tensor_wait_xdev` in `ds4_rocm_compat.cu` — was found and fixed
   ([#18](issues/18-fix-cross-device-dispatch-race.md)), and verified correct in isolation
   (`make ROCM_ARCH=gfx1201 test-rocm` passes with no serialization env vars). But it did not
   close the end-to-end quality gap: post-fix `avg_nll` is statistically unchanged (TP
   3.086362 vs pre-fix 3.076559). **The actual remaining corruption is a different,
   still-unlocated race**, most plausibly a same-device (not cross-device) missing dependency
   that the blunt serialization flags happen to also mask. Full diagnosis and a recommended
   `rocgdb` watchpoint / sync-bisect debugging approach for whoever picks this up next: see
   [issue #18's Comments](issues/18-fix-cross-device-dispatch-race.md).
2. **TP is currently slower than pipeline layer-split**, not faster — the inverse of the PRD's
   primary success criterion. See [Performance numbers](#performance-numbers) above. This may
   improve once #19 (clean post-fix baseline), #20 (MoE hot-path), and #22 (launch-overhead
   fusion) land, but none of that work is meaningful until limitation 1 is resolved, since a
   fast wrong answer is worthless.
3. **Four-GPU topology is two pipelined TP pairs (Option A), not native 4-rank TP.** Upstream's
   TP design is natively two-rank (50/50 expert split, head split, row-sharded vocab). Four
   GPUs run two independent TP pairs with a pipeline hop between them, which keeps some
   pipeline serialization rather than achieving uniform 4-way utilization. Rationale and the
   rejected alternative (Option B, true 4-rank sharding) are recorded in
   [`issues/11-four-gpu-topology.md`](issues/11-four-gpu-topology.md).
4. **Isolated 2-rank TP cannot hold the full production model.** The 81 GiB IQ2/Q2_K quant
   does not fit in a 2-GPU pair's ~68 GiB combined VRAM budget, with or without SSD streaming
   (multi-GPU placement disables streaming). This is why Option A above uses all 4 GPUs even
   though the native design is two-rank. See
   [`experiment-log.md`](experiment-log.md) (2026-07-24 entries).
5. **22 of 37 GPU entry points are deliberately unported**, not silently stubbed: DSpark
   speculative decoding, continuous/session batching, and a handful of dead-for-this-model
   code paths (Q4_K attention output, non-Q8_0 matmul quant paths) are out of scope per the PRD
   and fail loudly (`ds4_rocm_tp_stub`) rather than silently no-op if a future config reaches
   them. Full breakdown: [`inventory.md`](inventory.md).
6. **Continuous batching, DSpark, and Metal RDMA TP remain out of scope**, unaffected by this
   work (see PRD "Out of Scope"). Concurrent requests still serialize on this backend.

## How to reproduce

All commands assume 4x AMD Radeon AI Pro R9700 (gfx1201), ROCm, and the production model at
`/var/cache/llama/ds4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`.
Build with `ROCM_ARCH=gfx1201` throughout.

**Build:**

```sh
make ROCM_ARCH=gfx1201 rocm -j8
```

**Unit / correctness test suite** (cross-device transfer, kernel numeric equivalence, TP
refusal — no model required, no serialization flags):

```sh
make ROCM_ARCH=gfx1201 test-rocm -j8
```

**Quality fixture** (the authoritative correctness gate — 100-case fixture, `avg_nll`):

```sh
M=/var/cache/llama/ds4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
make ROCM_ARCH=gfx1201 rocm-quality -j8

# Trustworthy output currently REQUIRES serialization (see Known limitations #1):
./gguf-tools/quality-testing/score_official $M \
  gguf-tools/quality-testing/data/flash/manifest.tsv /tmp/out_tp.tsv 4096 \
  --gpu-devices 0,1,2,3 --cuda-tensor-parallel
AMD_SERIALIZE_KERNEL=3 ./gguf-tools/quality-testing/score_official $M \
  gguf-tools/quality-testing/data/flash/manifest.tsv /tmp/out_tp_serialized.tsv 4096 \
  --gpu-devices 0,1,2,3 --cuda-tensor-parallel
```

**Throughput bench:**

```sh
./ds4-bench -m "$MODEL" --rocm --gpu-devices 0,1,2,3 --cuda-tensor-parallel \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 2048 --step-incr 2048 --gen-tokens 256
```

**Plain chat test** (single-prompt repro used throughout this PRD's debugging):

```sh
./ds4 -m "$MODEL" --rocm --gpu-devices 0,1,2,3 --cuda-tensor-parallel \
  --ctx 4096 --temp 0 -n 20 -p "Explain C pointers in one sentence."
```

**Eval harness** (OpenCode-style reference comparison, requires a running server):

```sh
./ds4-server --rocm --gpu-devices 0,1,2,3 --cuda-tensor-parallel \
  --model "$MODEL" --ctx 4096 --host 0.0.0.0 --port 8000 &
# then run the eval harness against http://localhost:8000/v1 — see issues 13-17
```

## What was ported

- **Inventory of all 37 GPU entry points**, their port status, and why each unported one is
  either deliberately refused or structurally unreachable for this model/config:
  [`inventory.md`](inventory.md) / [`inventory.json`](inventory.json).
- **Cross-device transfer module** (deep module owning all peer-to-peer transport — mesh
  setup, peer copy, accumulate, host-staging fallback): [`ds4_rocm_xdev.h`](../../ds4_rocm_xdev.h),
  [`ds4_rocm_xdev.cu`](../../ds4_rocm_xdev.cu), tested standalone in
  [`tests/test_rocm_xdev.cu`](../../tests/test_rocm_xdev.cu).
- **Sharding policy module** (pure-logic ownership of routed experts, attention heads, and
  vocabulary rows — no GPU dependency, unit-testable on CPU):
  [`ds4_tp_shard.h`](../../ds4_tp_shard.h).
- **Kernel waves and correctness harness**: see issues
  [00](issues/00-stub-inventory-and-loud-failure.md) through
  [08](issues/08-auxiliary-tp-hooks.md) for the bring-up mode, kernel-comparison scaffold, and
  the attention / routed-MoE / matmul / shared-expert / gate-sync kernel ports.
- **Four-GPU extension, refusal/fallback, and container packaging**: issues
  [09](issues/09-refusal-and-fallback.md), [11](issues/11-four-gpu-topology.md),
  [12](issues/12-package-container.md).

## Full history

Detailed session-by-session findings, dead ends, and root-cause diagnoses:
[`experiment-log.md`](experiment-log.md) and the individual issue files under
[`issues/`](issues/).
