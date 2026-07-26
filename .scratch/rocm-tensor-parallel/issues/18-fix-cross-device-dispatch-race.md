# 18 — Fix cross-device dispatch race (no more `AMD_SERIALIZE_KERNEL=3`)

Status: closed

**What to build:** TP and pipeline produce correct multi-GPU output without `AMD_SERIALIZE_KERNEL=3` or `HIP_LAUNCH_BLOCKING=1`. The quality fixture scores match the serialized baseline on a default un-shimmed run.

The dispatch race lives in `ds4_rocm_xdev.cu` at the peer-copy sync point. Before reading a remote GPU's memory via peer copy, `ds4_rocm_xdev_copy` issues `hipDeviceSynchronize()` on the source device. On gfx1201's hardware scheduler (HWS), this does not guarantee that the specific stream producing the data has drained — the HWS can reorder across the implicit default-stream barrier, so the peer copy can read stale or partially-written memory.

Fix: replace `hipDeviceSynchronize()` with explicit `hipEventRecord`/`hipStreamWaitEvent` ordering between the actual producer stream and the peer-copy stream. This gives the HWS a proper happens-before edge instead of a blunt device-wide flush that it can reorder past.

There is also a second site — the pipeline tier handoff in `metal_graph_set_active_tier_decode` / `metal_graph_set_active_tier_batch` (`ds4.c`), which calls `ds4_gpu_tensor_copy_xdev` for cur_hc tier-to-tier hops. That goes through the same `ds4_rocm_xdev_copy` path, so a single fix in the xdev module covers both.

## Acceptance criteria

- [x] `ds4_rocm_xdev_copy` (and `_accumulate_f32`/`_accumulate_f16`) uses `hipEventRecord` + `hipStreamWaitEvent` per peer copy instead of `hipDeviceSynchronize()`
- [ ] Quality fixture (`make rocm-quality`, TP mode) matches the serialized baseline (`avg_nll ~0.370`, `first_match ~68/100`) without `AMD_SERIALIZE_KERNEL=3`
- [ ] `ds4-bench` 4-GPU TP default run shows throughput at or above the serialized baseline from #19
- [ ] No regression in pipeline mode (quality fixture or bench)

## Blocked by

None — can start immediately.

## Comments

**2026-07-26 — Implemented the described fix; verified it does NOT close the quality gap. Escalating: the actual remaining corruption is elsewhere, not in the xdev peer-copy sync point.**

**What was implemented.** `ds4_rocm_xdev_copy` (`ds4_rocm_xdev.cu`) no longer calls
`hipDeviceSynchronize()` on `src_dev` before the peer copy. It now uses a persistent,
lazily-created per-device `hipEvent_t` (`rocm_xdev_producer_event`): `hipEventRecord`
on `src_dev`'s default stream, then `hipStreamWaitEvent` on the peer-copy stream, exactly
as this issue specified. If either event call fails, it falls through to the existing
host-staging path rather than silently proceeding unordered.

While testing, found and fixed a **second, unlisted call site with the identical bug
pattern**: `ds4_gpu_tensor_wait_xdev` (`ds4_rocm_compat.cu`), used by the TP-owned
decode/prefill combine paths (`raw_cache`, `peer_heads_src`, `router_weights` reads via
direct peer-mapped pointer, not through `ds4_rocm_xdev_copy`) had its own
`hipDeviceSynchronize()` fence with the same "not reliable on gfx1201 HWS" problem this
issue describes. Added `ds4_rocm_xdev_wait_producer(dst_dev, src_dev)` to the xdev module
(same event-record/stream-wait mechanism, exposed via `ds4_rocm_xdev.h`) and routed
`ds4_gpu_tensor_wait_xdev` through it. This felt squarely in-scope: same file family
(cross-device dispatch), same bug shape, and heavily exercised by TP mode specifically.

**Verification that the mechanism itself works.** `make ROCM_ARCH=gfx1201 test-rocm`
(cross-device transfer suite, kernel-numeric-equivalence suite, TP-refusal suite) all pass
on real hardware (4x R9700) **without** `AMD_SERIALIZE_KERNEL=3` or `HIP_LAUNCH_BLOCKING=1`.
`test_rocm_xdev` specifically re-verifies byte-exact copies across all 12 ordered device
pairs, accumulate correctness, host-staging fallback correctness, and bandwidth
(~24 GB/s peer, ~14 GB/s host-staging — matching the PRD's feasibility numbers) with no
serialization env vars set. So the event-based ordering is real, does what it's supposed
to, and isn't silently falling back to the (also-correct, just slower) host-staging path.

**But the end-to-end quality fixture shows no meaningful improvement.** Ran
`make rocm-quality` (`score_official`, 100-case fixture,
`/var/cache/llama/ds4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`,
real hardware, `env -u AMD_SERIALIZE_KERNEL -u HIP_LAUNCH_BLOCKING`) in both modes:

| config | avg_nll (post-fix) | first_match | avg_nll (pre-fix, from experiment-log) |
|---|---|---|---|
| 4-GPU TP, default | 3.086362 | 0/100 | 3.076559 |
| 4-GPU pipeline, default | 1.295131 | 0/100 | 1.355060 |
| (target: serialized baseline) | — | — | TP 0.369930 / pipeline 0.373815 |

TP mode is statistically unchanged (within noise of a nondeterministic race — if anything
first-token matches dropped from 0 to 0, no change). Pipeline mode improved by ~4%, nowhere
close to closing the ~3.6x gap to its serialized baseline. Raw logs kept at
`.scratch/rocm-tensor-parallel/quality-out/q_tp_postfix.{tsv,log}`,
`q_tp_postfix2.{tsv,log}` (partial, killed early once the pattern was clear), and
`q_pipeline_postfix.{tsv,log}`.

**Conclusion: this issue's root-cause diagnosis is falsified by direct measurement.** The
xdev peer-copy `hipDeviceSynchronize()` was a real bug worth fixing (it is exactly the kind
of unreliable device-wide fence the PRD's transport design says to avoid, and the fix
matches the codebase's own established idiom for this — see
`g_shared_gate_up_ready_event` in `rocm/ds4_rocm_shared_expert.cuh:341-345`, which uses the
identical event-record/stream-wait pattern for a same-device cross-stream dependency). But
fixing it, plus the same fix at the one other site with this exact pattern, produces no
measurable recovery toward the serialized baseline in either mode. The actual remaining
corruption must be a different, still-unlocated race. Given `AMD_SERIALIZE_KERNEL=3` /
`HIP_LAUNCH_BLOCKING=1` force *every* kernel launch process-wide to fully drain before the
next starts, they would mask a same-device (not just cross-device) missing dependency too —
e.g. a race between two kernels on one GPU's queue that gfx1201's HWS can reorder despite
being issued to what looks like the same logical stream. That is consistent with this
symptom's history: issue 10's sessions repeatedly found and fixed *distinct* real bugs
(`->owner` vs `->device_id`, missing `owned_filtered` dispatch, undersized `bucket_count`
scratch, inverted `use_expert_tiles`) that each looked like "the" race until reference-diffed
away, and the session-3 "race in the compressor" theory was itself later overturned. This
issue's "peer-copy dispatch race" diagnosis appears to be another entry in that same
pattern — plausible, matches the described HWS behavior, cheap to fix, but not sufficient.

**Recommendation for whoever picks this up next.** Do not re-litigate the two fixes already
committed here — they are independently correct and verified (kept regardless of outcome,
same discipline issue 10 used for its own interim fixes). The next step is the same one
issue 10's session 6/7 handoffs recommended and never got to: a `rocgdb` watchpoint session
(confirmed available on this box) on suspect same-device scratch buffers during an
unserialized repro, or the `cuda_ok()`-gated `DS4DBG_SYNC` sync-bisect probe described in
issue 10's Comments (narrows which kernel needs a same-device barrier by substring match on
its `cuda_ok` label). Single-prompt repro (same as issue 10 used): `./ds4 -m
/var/cache/llama/ds4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
--rocm --gpu-devices 0,1,2,3 [--cuda-tensor-parallel] --ctx 4096 --temp 0 -n 20 -p "Explain C
pointers in one sentence."` (remember `ROCM_ARCH=gfx1201` on `make rocm`/`make rocm-quality`).
Quality fixture repro: `make ROCM_ARCH=gfx1201 rocm-quality -j8` then `env -u
AMD_SERIALIZE_KERNEL -u HIP_LAUNCH_BLOCKING ./gguf-tools/quality-testing/score_official $M
gguf-tools/quality-testing/data/flash/manifest.tsv /tmp/out.tsv 4096 --gpu-devices 0,1,2,3
[--cuda-tensor-parallel]`.

**2026-07-26 — Closed as scoped, by human decision.** This issue's specific diagnosis
(cross-device peer-copy dispatch race) and its fix are correct and verified — AC1 is met and
stays checked. AC2-4 (quality fixture matching serialized baseline, throughput, no regression)
are not met and are out of scope for this issue: the measurement above shows they depend on a
separate, still-unlocated same-device race, not on anything this issue's fix touches. Rather
than keep 18 open chasing a bug it was never actually about, the remaining work is spun off to
`.scratch/rocm-tensor-parallel/issues/23-fix-same-device-compressor-prefill-race.md`, which
picks up directly from issue 10 session 3's localization inside
`ds4_gpu_compressor_prefill_tensor`. Issue 19's re-measurement should be understood as blocked
on issue 23, not on this issue, for the quality/throughput criteria specifically (the xdev fix
itself is already in and doesn't need re-verification).
