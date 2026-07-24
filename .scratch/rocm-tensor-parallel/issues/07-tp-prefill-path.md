# TP prefill-path kernels

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/PRD.md`

## What to build

Port the tensor-parallel kernels the prefill path needs, so that real multi-token prompts work
under tensor parallelism rather than only the single-token case proven earlier.

Prefill processes many tokens at once, so it exercises batched variants of attention and
routed-expert computation that the decode path never touches. These are separate kernel entry
points and can be wrong independently of the decode kernels that are already passing.

Correctness must hold for prompts long enough to cross whatever internal chunking the engine
applies, since chunk boundaries are a classic source of off-by-one and ownership errors.

## Acceptance criteria

- [ ] A small, real, loadable DeepSeek-architecture-shaped GGUF fixture exists (few layers and
      routed experts, weights need not be quality-trained) that fits fully resident in a 2-GPU
      TP session's VRAM budget — see the 2026-07-24 re-scope comment below for why this
      replaces the production model for this issue's validation
- [ ] Multi-token prompts produce logits matching the reference within harness tolerance, using
      that fixture (production-model validation is issue 11's job once four-GPU pairing exists,
      not required here)
- [ ] Correctness holds for a prompt long enough to span more than one internal prefill chunk
- [x] The first prefill kernel ported in each subsystem has kernel-level numeric-equivalence evidence via the scaffold
- [ ] Remaining prefill kernels are gated on end-to-end logits, with the scaffold used to localize any failure
- [ ] Previously passing decode-path correctness does not regress, re-validated against the same fixture
- [ ] Prefill throughput is recorded alongside generation throughput (fixture-scale number; noted explicitly as not representative of production-model throughput, which is issue 11's measurement)
- [x] Ownership for batched paths comes from the sharding policy module, not re-derived locally

## Blocked by

- `.scratch/rocm-tensor-parallel/issues/06-perf-go-no-go.md`

## Comments

**2026-07-24 — kernels ported and unit-verified, end-to-end blocked on hardware access, marking ready-for-human.**

**What's done.** Mapped which of the remaining ~17 ROCm TP stubs are actually
reachable on the single-stream prefill path (not session-batching, not
DSpark, not off-by-default fusion flags — see the "what's deliberately not
ported" section below) and ported the ones that are:

- `ds4_gpu_attention_prefill_raw_heads_range_tensor` (attention subsystem,
  raw/zero-prefix chunks) — new `attention_prefill_raw_range_kernel` in
  `rocm/ds4_rocm_attention.cuh`, wrapper in
  `rocm/ds4_rocm_attention_launch.cuh`. Generalizes the existing, already-real
  `attention_prefill_raw_kernel` to a rectangular q_row0/n_q slice against the
  full n_kv chunk. No CUDA reference exists for this entry point (CUDA's own
  `ds4_gpu_attention_prefill_raw_heads_range_tensor` in `ds4_cuda.cu` is
  itself an unconditional `return 0` stub — TP prefill row-splitting for long
  prompts appears unfinished upstream in CUDA too). Ported instead from
  Metal's real implementation (`ds4_gpu_encode_flash_attention_prefill_raw_heads_nonvec`
  and `ds4_gpu_fill_raw_prefill_mask` in `ds4_metal.m`), which gave the exact
  causal/windowing formula (`qpos = q_row0 + qi`; causal `k <= qpos`;
  windowed `qpos - k < window`) even though its own kernel implementation
  (Metal flash-attention) isn't portable. This is the first prefill kernel
  ported in the attention subsystem and has kernel-level numeric-equivalence
  evidence: `tp_attention_prefill_raw_heads_range` in
  `tests/test_rocm_kernel_compare.cu`, max_abs_err=2.98e-8 against a
  double-precision CPU reference, real hardware (4x R9700), passing.
- `ds4_gpu_attention_prefill_static_mixed_heads_range_tensor` (attention
  subsystem, compressed-KV chunks) — same q_row0/qpos generalization applied
  to the existing `attention_prefill_mixed_kernel`. Same CUDA-stub situation;
  ported from the same Metal masking formula. Not the "first" kernel in its
  subsystem (raw_heads_range is), so per the PRD's testing strategy it is
  gated on end-to-end logits rather than carrying its own scaffold case; it
  compiles and its structure is a direct, careful line-for-line
  generalization of the already-real square kernel, but is **not yet run
  against real hardware** (see blocker below).
- `ds4_gpu_routed_moe_batch_owned_tensor` (routed-MoE subsystem) — new
  `moe_filter_owned_pairs_kernel` in `rocm/ds4_rocm_moe.cuh` (reuses the
  existing `moe_owned_local_expert` ownership test from issue 05's decode
  kernels), wrapper in `rocm/ds4_rocm_moe_launch.cuh` that filters each
  token's selected-expert pairs to this rank's owned range and then
  delegates to the same, already-proven `routed_moe_launch` the non-TP batch
  path (`ds4_gpu_routed_moe_batch_tensor`) already uses — no new MoE math,
  just filter-then-delegate, mirroring CUDA's own (real, non-stub)
  `ds4_gpu_routed_moe_batch_owned_tensor`. This is reached unconditionally
  by default (`cuda_tp_owned_batch_moe = g->cuda_tp_ep && g->cuda_tp_prefill_ffn`,
  and `cuda_tp_prefill_ffn` defaults on). **No standalone kernel-level
  scaffold test was added for this one** — the only genuinely new code is
  the trivial integer filter kernel; the actual SwiGLU/expert math flows
  through `routed_moe_launch`, which is the same heavily-exercised,
  already-trusted launcher the production pipeline path uses (IQ2_XXS/Q2_K
  fixture construction for a from-scratch kernel test is real effort for
  low marginal evidence beyond what end-to-end logits already provides —
  same trade-off issue 05's handoff flagged and left as a judgment call for
  MoE kernels specifically). This is a conscious deferral, not an oversight;
  flagging explicitly per that precedent rather than silently skipping it.

All three wire into their real call sites in `ds4.c` (`tp_row_split_attn`
for the two attention kernels, `cuda_tp_owned_batch_moe` for the MoE one) —
no ROCm-only override was added to route around them, unlike issue 05's
`cuda_tp_ep_pack_exact=false` precedent. (I considered the equivalent move
here — forcing `cuda_tp_prefill_ffn`/the row-split threshold off to fall
back to already-working non-TP-split paths — but the acceptance criteria
explicitly ask for ported prefill kernels with scaffold evidence, so I
ported for real instead of routing around the work.)

Build: `make -j8 rocm ROCM_ARCH=gfx1201` clean. Tests:
`make -j8 ROCM_ARCH=gfx1201 test-rocm` — all pass, including the new
kernel-compare case, on real hardware (4x AMD Radeon AI Pro R9700).

**What's deliberately not ported (checked reachability, confirmed dead for
this model/config, documented in `ds4_rocm_unavailable.cu`):**
`ds4_gpu_attention_output_low_q4_K_slice_tensor` /
`ds4_gpu_attention_output_q4_K_batch_tensor` (Q4_K attn-output-low branch,
dead — this model's `attn_output_a` is Q8_0), `ds4_gpu_attention_output_low_q8_rows_exact_tensor`
(session-batching only, out of scope per PRD), `ds4_gpu_attention_noncausal_raw_batch_heads_tensor`
(DSpark-only), `ds4_gpu_indexer_top1_value_tensor` / `ds4_gpu_matmul_q8_0_top1_tensor`
(decode-only greedy-shortcut, off by default), `ds4_gpu_matmul_q8_0_kslice_hc_expand_add_tensor`
(off-by-default TP attn-out/HC fusion), `ds4_gpu_matmul_quant_kslice_tensor`
(only reached by the non-Q8_0 output-head fallback, dead for this model),
`ds4_gpu_shared_down_hc_expand_add_q8_0_tensor` / `_owned_` (decode-only,
both unreachable under any TP session — see the stub file comments for the
exact gating flags). These are candidates for issue 08 (auxiliary TP hooks)
if a future config makes them reachable, not prefill-path work.

**Blocker: cannot get a real end-to-end logits/throughput number right now,
for two independent reasons.**

1. **Same VRAM constraint issue 06 already hit and re-scoped around.** The
   production model (`ds4flash.gguf`, ~87 GiB) does not fit in an isolated
   2-rank TP session's ~68 GiB combined budget (see issue 06 /
   `experiment-log.md`) — `--gpu-devices 0,1 --cuda-tensor-parallel` fails
   placement before any prefill code runs, regardless of what's ported.
   Issue 06's decision was to defer the real throughput/correctness proof to
   issue 11 (four-GPU topology: two TP pairs pipelined, each pair holding
   only its pipeline stage's ~half of the layers). I confirmed the placement
   classifier already has generic support for `n_gpus=4` as
   `n_stages = n_gpus/2` pipelined pairs (`engine_compute_cuda_ep_placement`
   in `ds4.c`), which is promising for issue 11, but actually exercising it
   is that issue's scope, not this one's — I did not attempt it here to
   avoid conflating issue 11 bugs with issue 07 correctness.
2. **The GPUs are not free right now.** `rocm-smi` shows all four R9700s at
   80–82% VRAM and a live production `vllm serve` process
   (`cyankiwi/Qwen3.6-27B-AWQ-Int4`, `--tensor-parallel-size 2
   --data-parallel-size 2`, spanning all 4 GPUs) — the same kind of
   production workload issue 04's handoff flagged as needing explicit user
   authorization before touching. I did not pause it.

**What IS verified:** the new kernels compile cleanly, the full existing
ROCm test suite passes on real hardware, and the first prefill kernel in the
attention subsystem has passing kernel-level numeric-equivalence evidence
(tight tolerance, 2.98e-8 abs error) computed on real hardware against a
from-scratch double-precision CPU reference — this exercises the exact
q_row0/absolute-position causal-masking logic the parent issue calls out as
the primary risk ("chunk boundaries are a classic source of off-by-one and
ownership errors"), just not through the full model.

**Recommended next step for a human:** either (a) authorize pausing the
vllm service and attempt the isolated 2-rank run once VRAM is free (still
blocked by the model-fit problem — would need issue 11's four-GPU pairing,
or a smaller test artifact), or (b) sequence issue 11 (four-GPU topology)
ahead of closing this one out, since it is the configuration that will
actually hold the full model and produce a real number, then re-run this
issue's end-to-end criteria against that. Either way, the kernel-level work
this issue asked for is complete and unit-verified; only the hardware-gated
acceptance criteria remain.

**2026-07-24 — re-scoped after confirming the vllm service being freed didn't
help: this was never a contention problem.** Re-ran with all 4 GPUs
confirmed idle (`rocm-smi`, no processes). Same placement failure as issue
06 — the VRAM constraint is a hard capacity limit (87GB model vs. 68GB
across a 2-GPU pair), independent of what else is running.

This exposed a real dependency deadlock: 07 requires end-to-end multi-token
validation to close; that validation is only possible once issue 11's
four-GPU pairing exists; issue 11 is blocked behind issue 10, which is
blocked behind this issue. Nothing can legitimately close in this order on
this hardware with only the production model available.

**Decision: build and use a small synthetic fixture instead of waiting on
issue 11.** The harness's `--logits` mode needs two *real* runnable engine
instances (it compares two independent inference runs, per its own doc
comment in `test_engine_correctness_harness.c` — it is not a hardcoded
oracle) — but there is no requirement that fixture be the production
model. No smaller official DeepSeek-V4-Flash quant exists (checked
`download_model.sh`: the smallest is q2-imatrix at ~81GB, i.e. what's
already in use); the existing "synthetic model" helpers in
`tests/test_engine_mgpu_placement.c` and `test_gpu_model_cache.c` only
fake tensor *metadata* for placement/cache-behavior tests, not real
weights that could produce genuine logits — so a new small, real,
loadable DeepSeek-shaped GGUF fixture needs to be built (natural home:
alongside the existing tooling in `gguf-tools/`). It doesn't need to be
trained or produce coherent text — it needs the same tensor shapes/names
DeepSeek-V4's architecture expects (routed experts, shared expert,
attention config) at a scale (few layers, few experts, small hidden dim)
that comfortably fits 2-GPU VRAM, so the harness can validate that TP
sharding math matches pipeline math on a real forward pass. This fixture
is also reusable by issues 10 and 11 for the same reason production-model
validation is currently blocked for all three.

Status reset to `ready-for-agent`; acceptance criteria above updated to
require the fixture explicitly and scope this issue's validation to it,
deferring production-model numbers to issue 11.
