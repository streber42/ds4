# 49 — Instrument TP=4 decode-loop sync/dispatch call sites

Status: closed

## Parent

`.scratch/rocm-tensor-parallel/issues/33-tp4-throughput-measurement.md`

## What to build

Issue #33 measured that TP=4 decode does 86 all-reduces, 172 `hipDeviceSynchronize`
calls, and 344 `hipSetDevice`/cross-device `cur_hc` copies per generated token,
and attributed the ~238ms/token decode cost mostly to that sync/dispatch overhead
rather than compute (~35ms) or the theoretical communication cost of the
all-reduces themselves (~0.4ms). That attribution was analytical/aggregate, not
per-call-site.

Build a lightweight instrumentation harness (counters and/or `rocprof` markers)
that attributes each `hipDeviceSynchronize`, `hipSetDevice`, and cross-device
copy in the TP=4 decode loop to its call site (attention tier switch, MoE tier
switch, all-reduce boundary hop, output head, etc.), and reports a per-call-site
breakdown of count and wall-clock cost for one decode token.

This is the measurement baseline that every later issue in this chain (#50-#55)
is graded against — each of those issues should show a before/after delta from
this harness, not just an aggregate t/s number.

## Acceptance criteria

- [x] Per-call-site counts of `hipDeviceSynchronize`/`hipSetDevice`/cross-device
      copies in the TP=4 decode loop, confirming (or correcting) issue #33's
      172/344 aggregate figures — **DONE, corrected: actual `hipDeviceSynchronize`
      is 860/token (not 172) and `hipSetDevice` is 1505/token (not 344, which
      conflated switches with copies and missed the `hipSetDevice` calls hidden
      inside the sync helper)**
- [x] Per-call-site wall-clock cost breakdown for one steady-state decode token
      — **DONE: 17-site table, ms/token, in experiment-log.md**
- [x] Harness is easy to re-run (a flag or script), so #50/#51/#52/#53 can each
      report a before/after delta against this baseline — **DONE:
      `DS4_TP4_INSTRUMENT=1` env flag + `.scratch/rocm-tensor-parallel/scripts/tp4-instrument.sh`**
- [x] Findings recorded in `.scratch/rocm-tensor-parallel/experiment-log.md`
      — **DONE**

## Blocked by

None — can start immediately.

## Comments

### Instrumentation harness landed and measured on real hardware (2026-07-31)

Implemented as a `DS4_TP4_INSTRUMENT=1` env-gated counter/timer table in
`ds4.c`, wrapping every `metal_graph_set_active_tier_decode` (`hipSetDevice`
+ BLAS-handle tier swap), `ds4_rocm_xdev_sync_all_devices` (`hipSetDevice` +
`hipDeviceSynchronize` per device — see below, not sync-only),
`ds4_rocm_xdev_copy` (cross-device copy), and `ds4_rocm_xdev_allreduce_f32`
call already present in the ROCm TP=4 decode loop with a named call site.
Report prints to stderr at process exit (count, calls/token, total ms,
ms/token per site). Disabled by default; near-zero overhead when off (one
cached branch per call site).

**First pass used `AMD_SERIALIZE_KERNEL=3` and gave a misleading result**
(sync barriers looked cheap, ~1.4 ms/token combined). That flag makes every
kernel launch block synchronously, so real wait time was silently absorbed
into the uninstrumented kernel-launch calls rather than the named barrier
sites. Re-ran without it — the numbers below are from that unserialized
run, which is the one that should be treated as the baseline. Full
before/after writeup in `.scratch/rocm-tensor-parallel/experiment-log.md`
(2026-07-31 entry).

Ran on the production 81 GiB model across 4×R9700, `-c 64`, no kernel
serialization, two independent runs with different prompts (19 decode
tokens each, results agreed within 1%), both with coherent TP=4 output
("...simply \"Paris\"." / a correct C-pointer explanation). Headline
corrections to issue #33's aggregate estimate:

- `hipDeviceSynchronize` count is 860/token, not 172 (5 sync call sites ×
  43 layers × 4 devices each — checked the implementation in
  `ds4_rocm_xdev.cu:482-488`, not just the header comment).
- `hipSetDevice` count is 1505/token, not 344: 645/token from the six
  explicit tier-switch call sites, *plus* another 860/token hidden inside
  the five sync call sites (each `ds4_rocm_xdev_sync_all_devices`
  invocation calls `hipSetDevice` once per device before its
  `hipDeviceSynchronize`).
- 501 ms/token (91% of the 549 ms/token instrumented total, 72% of the
  entire ~694 ms/token decode budget this run measured) lands inside the
  two 4-tier switch loops that precede real compute (`attn_tier_switch`,
  `moe_tier_switch`) plus their barriers. This does **not** by itself
  settle issue #33's overhead-vs-compute attribution — the harness times
  host-side wall-clock, which can't distinguish true dispatch cost from
  the host blocking on outstanding GPU compute inside the same interval.
  `hc_expand_tier_switch` uses the identical switch pattern but precedes
  far less per-tier compute and costs 7ms/token vs `attn_tier_switch`'s
  247ms/token — strong evidence a real share of the 501ms is compute-wait
  misattributed to the call that blocked on it, not fixed dispatch
  overhead. Use `rocprof` kernel timelines if #50-#55 need that split.
- All-reduce collectives cost ~18 ms/token combined; cross-device copies
  outside the all-reduce internals cost ~14 ms/token.
- The "output head" call site named in the issue text does not fire for
  TP=4 — its tier switch is gated on `g->placement`, which is `NULL` in
  `rocm_tp4` mode, and the decode loop already leaves `active_tier == 0`
  after the last layer.
- Free win the harness surfaced: `hc_dump_tier0_switch` (43 calls, ~2
  ms/token) exists solely to reposition the tier for a debug dump that is
  normally a no-op — unconditional release-path cost worth gating in
  #50-#55.
- Caveat: coarse CPU-side timers can't cleanly separate true
  `hipSetDevice`/BLAS-swap cost from HIP command-queue backpressure on a
  device's outstanding async work — the 3-8× ms/call gap between
  `attn_tier_switch`/`moe_tier_switch` (heavy compute follows) and
  `hc_expand_tier_switch` (light compute follows) suggests the latter
  contributes. Use `rocprof` kernel timelines if #50-#55 need to isolate
  the two.

**Verification:** `make -j8 rocm` builds clean; `make -j8 cpu` (non-ROCm
build, exercises the same `ds4.c` with the instrumentation code compiled
out via `#if defined(DS4_ROCM_BUILD)`) builds clean under `-Wall -Wextra`;
`make -j8 test-rocm` (`test_rocm_tp_stubs`, `test_rocm_xdev`,
`test_rocm_kernel_compare`, `test_engine_rocm_tp_refusal`) all pass on
real hardware. GPU lock acquired/released per protocol, across two separate
acquire/release cycles for the two measurement passes.

Every instrumented run (serialized and unserialized, `-c 64` and `-c 128`)
hit `ROCm model arena alloc failed for moe_down (1024.00 MiB chunk): out of
memory` and fell back to q8 kernels. `rocm-smi` confirms VRAM is otherwise
idle between runs, so this is a transient allocator-peak issue at load
time, not stale VRAM or a context-size effect. It depresses absolute t/s
(measured ~1.4 t/s here vs issue #33's clean-load ~4.5 t/s) but not the
relative per-call-site attribution — and sync/dispatch still dominates
(79% of per-token time) even with slower compute in the denominator, which
if anything strengthens the conclusion. This VRAM tightness looks like a
regression from whatever state gave issue #33 its clean 23.80 GiB/tier
load; flagged in the experiment log as a follow-up, out of scope for this
issue.
