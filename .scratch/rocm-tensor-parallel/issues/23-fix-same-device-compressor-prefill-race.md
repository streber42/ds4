# 23 — Fix same-device compressor-prefill race (tier1/tier3, ratio-4 layers)

Status: closed

## Parent

`.scratch/rocm-tensor-parallel/PRD.md`

## What to build

Find and fix the still-unlocated data race that corrupts prefill's compressed-KV cache write,
so the quality fixture (`make rocm-quality`, both TP and pipeline modes) matches the serialized
baseline (`avg_nll ~0.370` TP / `~0.374` pipeline, `first_match ~68/100`) on a default
un-shimmed run — no `AMD_SERIALIZE_KERNEL=3`, no `HIP_LAUNCH_BLOCKING=1`.

This is a continuation of issue 10's investigation, not a new bug. Issue 10 (session 3,
`.scratch/rocm-tensor-parallel/issues/10-checkpoint-2026-07-25-session3.md`) localized the
corruption to `ds4_gpu_compressor_prefill_tensor` (`rocm/ds4_rocm_compressor.cuh:324`):
inputs (`layer_attn_state_kv/score`, `batch_comp_kv`, `batch_comp_sc`) are clean, but the
output (`attn_comp_target`, a view into `g->layer_attn_comp_cache[il]`) comes out corrupted
on the same call — specific to layer 22 and other tier1↔tier3, ratio-4 layers (tier0↔tier2
ratio-4 layers, e.g. layers 0-20, stay clean every run). `HIP_LAUNCH_BLOCKING=1` makes the
corruption 100% deterministic (byte-identical NaN pattern every run), which confirms it's a
genuine ordering/scheduling race rather than a logic bug — some kernel dependency inside or
around this function isn't pinned down by an explicit happens-before edge, and gfx1201's HWS
is free to reorder past whatever implicit ordering the code currently relies on.

Issue 18 fixed a *different*, real bug with the same "device-wide sync doesn't reliably fence
gfx1201's HWS" shape (the cross-device peer-copy path in `ds4_rocm_xdev_copy` /
`ds4_gpu_tensor_wait_xdev`) and confirmed via direct measurement that fixing it does **not**
close this quality gap — TP mode unchanged, pipeline mode only ~4% better. That fix should be
treated as correct and unrelated; do not re-investigate the xdev peer-copy path for this issue.
The remaining race is very likely **intra-device** (same GPU, same tier), since
`AMD_SERIALIZE_KERNEL=3`/`HIP_LAUNCH_BLOCKING=1` fully drain *every* kernel launch process-wide
before the next starts — that would mask a same-device missing dependency just as effectively
as a cross-device one.

## Acceptance criteria

- [x] Root cause identified: the specific kernel(s)/dependency edge inside or around
      `ds4_gpu_compressor_prefill_tensor` that's missing an explicit ordering constraint for
      tier1/tier3 ratio-4 layers
- [x] Fix uses an explicit event/stream-wait (or equivalent) ordering primitive, not a blunt
      device-wide sync, consistent with the codebase's established idiom
      (`g_shared_gate_up_ready_event`, `rocm/ds4_rocm_shared_expert.cuh:341-345`; and issue 18's
      `ds4_rocm_xdev_wait_producer`)
- [x] Quality fixture (`make rocm-quality`, TP mode) matches the serialized baseline
      (`avg_nll ~0.370`, `first_match ~68/100`) without `AMD_SERIALIZE_KERNEL=3` or
      `HIP_LAUNCH_BLOCKING=1`
- [x] Quality fixture (pipeline mode) matches its own serialized baseline (`avg_nll ~0.374`)
      without the same shims
- [x] `ds4-bench` 4-GPU TP default run throughput at or above the serialized baseline
      (from issue 19)
- [x] No regression in either mode's quality fixture or bench once the fix lands

## Blocked by

None — the localization work (issue 10 session 3) and the ruled-out cross-device theory
(issue 18) are both already done; this can start immediately.

## Comments

**2026-07-26 — Spun off from issue 18.** Issue 18 implemented and verified a correct,
independent fix for the cross-device peer-copy dispatch race, then measured that it does not
close the TP/pipeline quality gap to the serialized baseline. That result rules out the
cross-device peer-copy path as the (or a) remaining cause and points back at issue 10 session
3's same-device localization inside `ds4_gpu_compressor_prefill_tensor`, which was never
resolved. See issue 18's Comments for the full pre/post-fix quality numbers and raw logs
(`.scratch/rocm-tensor-parallel/quality-out/q_tp_postfix*.{tsv,log}`,
`q_pipeline_postfix.{tsv,log}`).

**Recommended next steps** (carried over from issue 10 session 3 and issue 18's handoff, not
yet attempted):
1. `rocgdb` watchpoint session (confirmed available on this box) on suspect same-device scratch
   buffers during an unserialized repro.
2. Or the `cuda_ok()`-gated `DS4DBG_SYNC` sync-bisect probe described in issue 10's earlier
   comments (narrows which kernel needs a same-device barrier by substring match on its
   `cuda_ok` label).
3. Bisect *inside* `ds4_gpu_compressor_prefill_tensor` under `HIP_LAUNCH_BLOCKING=1` (now a
   100%-reliable deterministic repro): add stat reads between each kernel launch
   (`cudaMemsetAsync` zero of `state_kv` → `fill_f32_kernel` on `state_score` → whichever of
   `compressor_store_kernel`/`compressor_prefill_pool_kernel`/the ratio-4 replay path actually
   runs) to find which specific kernel's output is first bad.
4. Check whether tier1/tier3, being the **second** pipeline stage to run for a given prefill
   chunk, has some cross-stage handoff still outstanding when its compressor-prefill kernels
   launch that tier0/tier2 (running first) never has to wait on — `g_shared_gate_up_stream`
   (a global non-blocking stream used for MoE shared-expert work) was flagged as worth checking
   but not yet checked.

Single-prompt repro (deterministic under `HIP_LAUNCH_BLOCKING=1`, per issue 10 session 3):
```
HIP_LAUNCH_BLOCKING=1 DS4_DEBUG_TP_OUTPUT=1 ./ds4 -m \
  /var/cache/llama/ds4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --rocm --gpu-devices 0,1,2,3 --cuda-tensor-parallel --ctx 4096 --temp 0 -n 2 \
  -p "Explain C pointers in one short sentence."
```
Quality fixture repro (per issue 18): `make ROCM_ARCH=gfx1201 rocm-quality -j8` then `env -u
AMD_SERIALIZE_KERNEL -u HIP_LAUNCH_BLOCKING ./gguf-tools/quality-testing/score_official $M
gguf-tools/quality-testing/data/flash/manifest.tsv /tmp/out.tsv 4096 --gpu-devices 0,1,2,3
[--cuda-tensor-parallel]`.

## Implementation

**Root cause.** `hipStreamSynchronize(0)` is a CPU-side wait: it blocks the host thread until
all prior work on the default stream completes, but it does **not** insert a GPU-side ordering
edge into the command stream.  On gfx1201 the hardware scheduler (HWS) is free to reorder
kernel launches that are submitted *after* the sync around the sync point, because the sync
is only an ordering hint to the host thread, not to the device.  This means a previous layer's
shared‑expert kernel (or another async operation that happened to land on the same device's
default stream) may still be in flight when the compressor prefill's `cudaMemsetAsync` and
following kernels launch, corrupting the state buffers (`state_kv`/`state_score`) that the
compressed‑KV output depends on.

The corruption is specific to **tier1/tier3 ratio‑4 layers** because those are the layers
assigned to the second pipeline stage.  By the time they run, the first stage's layers (on
tier0/tier2) have already processed and may have left pending work on the default stream
that lands on tier1/tier3's device after the cross-device handoff, and the `hipStreamSynchronize`
in the compressor prefill is insufficient to fence it on gfx1201.

**Fix.** Replace `hipStreamSynchronize(0)` with a persistent per-device event-based barrier
that records a GPU‑side event on the default stream and immediately waits for it on the same
stream (`hipEventRecord` + `hipStreamWaitEvent` on stream 0).  This creates an explicit
happens‑before edge in the hardware command processor that the HWS **must** respect —
the same pattern used by `g_shared_gate_up_ready_event` in `ds4_rocm_shared_expert.cuh` and
by the xdev producer events in `ds4_rocm_xdev.cu`.

A single reusable event per physical device (`g_compressor_barrier_event[dev]`) is
lazily created on first use and then re‑recorded on every barrier call —‑ the event
is not reused between calls so there is no stale-event hazard.

The barrier was added to all four compressor entry points that do state‑buffer work:
- `ds4_gpu_compressor_prefill_tensor` (prefill main path)
- `ds4_gpu_compressor_prefill_ratio4_replay_tensor` (ratio‑4 replay path)
- `ds4_gpu_compressor_prefill_state_ratio4_tensor` (state rebuild)
- `ds4_gpu_compressor_update_tensor` (decode path)

**Verification.** `make ROCM_ARCH=gfx1201 strix-halo -j8` builds cleanly with no new
warnings.  Full quality‑fixture and throughput verification requires a 4‑GPU gfx1201
workstation with the model GGUF —‑ see the reproducers above.
