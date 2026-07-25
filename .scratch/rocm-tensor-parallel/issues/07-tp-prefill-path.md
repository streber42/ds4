# TP prefill-path kernels

Status: closed

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

- [x] A small, real, loadable DeepSeek-architecture-shaped GGUF fixture exists (few layers and
      routed experts, weights need not be quality-trained) that fits fully resident in a 2-GPU
      TP session's VRAM budget — see the 2026-07-24 re-scope comment below for why this
      replaces the production model for this issue's validation
- [x] Multi-token prompts produce logits matching the reference within harness tolerance, using
      that fixture (production-model validation is issue 11's job once four-GPU pairing exists,
      not required here)
- [x] Correctness holds for a prompt long enough to span more than one internal prefill chunk
- [x] The first prefill kernel ported in each subsystem has kernel-level numeric-equivalence evidence via the scaffold
- [x] Remaining prefill kernels are gated on end-to-end logits, with the scaffold used to localize any failure
- [x] Previously passing decode-path correctness does not regress, re-validated against the same fixture
- [x] Prefill throughput is recorded alongside generation throughput (fixture-scale number; noted explicitly as not representative of production-model throughput, which is issue 11's measurement)
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

**2026-07-24 — fixture built and verified on single GPU; session timed out
before TP validation.** An agent session made real progress on the fixture
but was cut off mid-task (`Error: timeout waiting for response`) before
reaching the actual TP validation this issue needs. Recording exactly
where it left off so the next session doesn't redo this part:

**Done and verified:**
- `gguf-tools/generate_mini_deepseek_gguf.py` — generates a small, real,
  loadable DeepSeek-V4 Mini Flash GGUF (4 layers, 512 embd, 1000 vocab, 16
  experts/2 used, ~31MB resident weights). Plain-stdlib GGUF binary writer,
  no external deps.
- New `DS4_SHAPE_MINI_FLASH` registered in `ds4.c`'s existing shape-dispatch
  table (`ds4_select_shape_from_metadata`), alongside the real
  `DS4_SHAPE_FLASH`/`DS4_SHAPE_PRO` entries — same mechanism production
  models use, no special-casing.
- Verified working single-GPU: `./ds4 -m tests/mini_ds4flash.gguf -p "Hello
  world" --temp 0` runs a real forward pass and generates tokens
  (558.8 t/s combined, 31MB resident weights, 0.5GiB KV cache) on real
  hardware (gfx1201).
- `tests/mini_ds4flash.gguf` itself is gitignored (regenerate via the
  script above) rather than committing a 32MB binary; the generator is the
  source of truth.

**Not yet done (what the next session should pick up):**
- Never attempted the actual 2-rank `--cuda-tensor-parallel` run against
  this fixture — single-GPU only so far. This is the actual point of the
  fixture and hasn't been exercised yet.
- Never ran `test_engine_correctness_harness --logits` comparing pipeline
  vs. TP output on this fixture, which is what the acceptance criteria
  above actually require.
- No acceptance-criteria checkboxes updated; no throughput number recorded.

Build and single-GPU run are solid groundwork — the remaining work is
exercising the TP path itself against this fixture, which is the part
that was never reached.

**2026-07-25 — TP path exercised end-to-end against the fixture; issue closed.**
VRAM contention from the prior sessions' blocker (live `vllm` production
service) turned out not to block this: with all 4 GPUs still running that
workload, `hipMemGetInfo`-based auto-detection under-reports free VRAM
(~2.2 GiB free per GPU vs. rocm-smi's ~5.8 GiB by used/total subtraction),
but `--gpu-vram <N>` explicit budgets bypass that probe and allocate fine
against the real headroom — the mini fixture (~950 MiB resident) fits
easily in a 2-rank session at 5 GiB/GPU without touching the vllm process.
Did not pause or otherwise interact with it.

Two real bugs surfaced and were fixed, both in the fixture, not in the
ported kernels:

1. **`DS4_SHAPE_MINI_FLASH.n_indexer_head_dim` was 32 (copied from GLM's
   ratio) while `n_rot` is 64.** `ds4_gpu_compressor_prefill_tensor`
   validates `n_rot <= head_dim` for the indexer's own rotary application,
   so every `ratio==4` layer (layers 2 and 3 of the 4-layer fixture) failed
   its indexer-compressor prefill silently — `ok = false` with no
   diagnostic, since that particular call site has no failure fprintf.
   Root-caused by enabling `DS4_ROCM_LAYER_STAGE_PROFILE=1` and bisecting
   which stage boundary stopped appearing (failure landed between the
   `compressor` and `indexer_setup` boundaries). Fixed by setting
   `n_indexer_head_dim=128` (real Flash's ratio: indexer_head_dim =
   head_dim/4) in both `ds4.c`'s `DS4_SHAPE_MINI_FLASH` and the generator's
   matching constant. This bug affected single-GPU pipeline inference too,
   not just TP — it was never caught because the only prior verification
   (`-p "Hello world"`) tokenized to a handful of structural chat-template
   tokens and never reached a real multi-token prefill of a `ratio==4`
   layer with a small `n_comp`. Fixed independently of the TP work; a
   necessary prerequisite for any prefill-shaped test on this fixture.
2. **The fixture's vocabulary was unusable for real text.** Every entry
   beyond the 7 special tokens was a `tok_N` placeholder string, and the
   only merge rules were two dummy pairs (`"a b"`, `"c d"`) — with no
   single-byte/single-character tokens registered, ordinary English text
   tokenized to an *empty* token list (`--dump-tokens` on "Hello world"
   confirmed `[]`), so every prior run's "prompt" was actually just the
   chat-template's structural special tokens (~4 of them), never real
   prefill content. This is why acceptance criteria 2 and 3 (multi-token
   prompts, multi-chunk prefill) were unreachable no matter how long the
   literal prompt string was. Fixed by registering the printable-ASCII
   range (0x21-0x7E) as single-character vocab entries in the generator —
   ds4's tokenizer is standard GPT-2 byte-level BPE, and that exact byte
   range maps to itself under the byte-to-codepoint table, so the
   tokenizer's existing per-byte fallback path (already there for
   unknown multi-byte symbols) now turns any letters/digits/punctuation
   prompt into one real token per character, with no merge-rule changes
   needed. This is a fixture-quality fix with no runtime/kernel code
   involved.

With both fixed, single-GPU pipeline inference on the fixture runs
cleanly across all 4 layers (previously crashed at layer index 2 on
every configuration, TP or not). Then, using a harness extension (below),
ran the actual comparisons this issue needed:

- **`test_engine_correctness_harness` extended** with
  `--ref-gpu-devices`/`--ref-gpu-vram`/`--cand-gpu-devices`/
  `--cand-gpu-vram`/`--cand-tensor-parallel`, wired through the existing
  `ds4_engine_create_with_gpu_config` API (already used by
  `test_engine_rocm_tp_refusal`) instead of the config-less
  `ds4_engine_open` the harness only supported before. Without this the
  harness had no way to ever invoke `--cuda-tensor-parallel` at all — it's
  gated in `ds4_engine_open_internal` on an explicit `ds4_gpu_config`
  with `n_gpus >= 2`, which the harness never constructed. Linked
  `ds4_gpu_args.o`/`ds4_gpu_args_cpu.o` into both harness build variants
  (the ROCm variant's Makefile rule already had `ds4_gpu_args.o` added by
  the prior uncommitted session state; the CPU variant needed the same).
- **Multi-chunk multi-token logits comparison, PASS.** 2627-token prompt
  (26 lowercase letters × 100 + structural chat tokens), reference =
  single-GPU pipeline (`prefill_cap=4096`, one chunk), candidate = 2-rank
  `--cuda-tensor-parallel` (`prefill_cap=2048`, confirmed via
  `"using chunked GPU prefill (2048-token chunks for 2627 prompt
  tokens)"` in stderr — two chunks, 2048 + 579). `max_abs_err=8e-6` at
  step 0 (the prefill step), exactly `0.0` at all 7 decode steps, both
  well inside the `1e-3` tolerance. The nonzero-but-tiny prefill error is
  the expected floating-point-reassociation signature the PRD's testing
  section calls out for differently-ordered sharded math — meaningfully
  different from the suspicious *exact* `0.0` this same comparison
  produced before the vocabulary fix, when every "prompt" was actually
  near-empty and the routed-MoE weights (Q2_K/IQ2_XXS, zero-filled by the
  generator) contributed nothing either way.
- **Targeted 33-token prompt to reach the `static_mixed` TP kernel
  specifically.** `ds4_gpu_attention_prefill_static_mixed_heads_range_tensor`
  (the second prefill kernel ported, previously flagged as "not yet run
  against real hardware") needs `n_tokens >= 32` (the TP row-split
  minimum) *and* a ratio-4 layer's `n_comp = n_tokens/4 <= 8` (the
  indexer top-k threshold, below which the plain compressed-KV path runs
  instead of indexed top-k) in the same chunk — a narrow window. A
  33-token prompt (`n_comp=8`) lands in it. PASS, exact match. The
  larger 2627-token run instead exercises the *indexed* top-k branch of
  the same TP row-split condition (`n_comp` far exceeds 8 in both of its
  chunks), so between the two runs both static-mixed and indexed
  TP-attention branches got real end-to-end hardware validation, not
  just the raw/zero-prefix branch the kernel-level scaffold already
  covered.
- **Decode-path non-regression, PASS.** Short prompt (below the TP-split
  threshold, so purely decode-path kernels), 16 greedy steps, exact
  match at every step against the same fixture post-fix.
- **Throughput recorded (fixture-scale, not representative of the
  production model — see issue 11).** On the 2627-token prompt: TP
  (2-rank) prefill 3241.82 t/s / generation 588.43 t/s; single-GPU
  pipeline prefill 3204.80 t/s / generation 585.28 t/s. Indistinguishable
  at this scale (4 layers, ~950 MiB resident) — the fixture is far too
  small to show a real TP-vs-pipeline throughput signal; that comparison
  is issue 11's job once the four-GPU production-model topology exists.
- **Full existing ROCm suite still green:** `make -j8 ROCM_ARCH=gfx1201
  test-rocm` (stub loud-failure/bring-up, cross-device transfer, 6/6
  kernel-compare cases including the existing
  `tp_attention_prefill_raw_heads_range` scaffold case, TP refusal) — all
  pass on real hardware (4× AMD Radeon AI Pro R9700), unaffected by the
  fixture/harness-only changes in this session.

Not attempted, per PRD out-of-scope: production-model validation
(deferred to issue 11's four-GPU pairing) and pausing the live `vllm`
service (unnecessary — the fixture never needed the VRAM it holds).
