# Checkpoint — 2026-07-25 (session 3) Issue 10: TP Output Divergence (BOS Loop)

Status: closed

This session picked up from
`.scratch/rocm-tensor-parallel/issues/10-checkpoint-2026-07-25-session2.md` (session 2's
checkpoint). Session 2 concluded the bug was **not** a race ("two plausible synchronization
bugs were fixed defensively... neither changed the failure point by even one layer, which is
strong evidence the real bug is not a race condition at all, but a deterministic logic/state
bug") and pointed at layer 22's attention core (`metal_graph_heads(g)` already 100% NaN before
any cross-device code runs). **That "not a race" conclusion is wrong** — see §1. The rest of
session 2's localization (layer 22, tier1↔tier3, ratio-4) still holds and this session narrowed
it much further: into prefill, into a specific cache write, with a reliable deterministic
repro command.

---

## 1. Session 2's "not a race" conclusion is disproven

Ran the **exact same command** (`DS4_DEBUG_TP_OUTPUT=1 ./ds4 ... -n 2 -p "Explain C pointers in
one short sentence."`, same binary, same weights, same hardware, `--temp 0`) back to back,
five times in a row, changing nothing between runs:

| Run | `il=22 comp_cache` (n_comp=4) | `il=22 heads (post attn-core, pre-RoPE)` |
|---|---|---|
| 1 | clean, spread of real values (min=-5, max=4.5) | clean (nan=0) |
| 2 | clean, spread of real values (min=-5, max=4.5) | clean (nan=0) |
| 3 | **flat, n=2048 all ≈ -0.000106812 or 0, nan=128** | **100% NaN (32768/32768)** |
| (repeated across many more runs) | alternates unpredictably | alternates unpredictably |

Identical binary, identical inputs, identical `--temp 0` — different results between runs.
**This is definitionally a race condition**, not a deterministic logic bug. Session 2's
instrumentation additions (`ds4_debug_tp_output_stat_f32`, which calls
`ds4_gpu_synchronize()` before every read) apparently perturbed timing enough to *reduce* but
not eliminate the race window, which is presumably why session 2's own defensive sync fixes
"didn't move the failure point" — the race was still there, just hitting less often, and
session 2 happened to interpret consistent-*looking* failures across a couple of runs as
determinism rather than checking run-to-run variance directly.

**Confirmation via `HIP_LAUNCH_BLOCKING=1`**: this HIP env var forces every kernel launch to
block until complete (fully serializes the GPU timeline). With it set, the exact same command
was run 5 times: **100% failure, byte-identical corruption every time** (same flat
`-0.000106812`/`0` pattern, same 128 NaNs, same all-NaN `heads`). This is the standard way to
confirmed the race-condition hypothesis — this doesn't just say "still fails sometimes," it
became **fully deterministic once you removed the raciness**, which is the signature of a real
race. It also means the corrupted result is not "we occasionally read stale/half-written
memory and got lucky the other times" in the intuitive sense — under full serialization it
*always* corrupts, meaning whatever is racing wins deterministically once ordering is pinned
down. See §4 for what that implies.

**Practical upshot**: `HIP_LAUNCH_BLOCKING=1` gives a **100%-reliable deterministic repro** for
this bug. Use it for all further bisection instead of the flaky default (no more "run it 5
times and see").

```bash
HIP_LAUNCH_BLOCKING=1 DS4_DEBUG_TP_OUTPUT=1 ./ds4 \
  -m /var/cache/llama/ds4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --rocm --gpu-devices 0,1,2,3 --cuda-tensor-parallel \
  --ctx 4096 --temp 0 -n 2 \
  -p "Explain C pointers in one short sentence."
```

---

## 2. New instrumentation added this session (all gated on `DS4_DEBUG_TP_OUTPUT`, all still in
the working tree, uncommitted — same pattern as session 2)

Extended `ds4.c`'s decode-path stat dumps (session 2's) with matching stats along the
**prefill** path, since that's where corruption actually originates (see §3):

- `ds4.c` decode attention phase (`metal_graph_encode_decode_layer_phase`): added stats for
  `attn_norm`, `q_lora`, `Qcur` (post-RoPE), `KVrope`, `layer_attn_state_kv/score` (post
  `compressor_update`), `layer_index_state_kv/score` (post indexer `compressor_update`),
  `raw_cache`/`comp_cache` (right before the attention-core call), and `heads` both
  pre-RoPE and post-RoPE (right after the attention-core call, before any TP-specific
  branching) — this is what let this session confirm heads is already NaN pre-RoPE, i.e. the
  attention-core kernel itself outputs NaN when fed a corrupted `comp_cache`, not that RoPE or
  the TP output path introduces it.
- `ds4.c` prefill attention batch (`metal_graph_encode_layer_attention_batch`, around
  ds4.c:27380-27680): added stats on `batch_comp_kv`/`batch_comp_sc` (post F16 projection,
  pre-compressor) and `batch_attn_norm` (compressor input), plus stats on `attn_comp_target`
  (== the view into `layer_attn_comp_cache[il]` that `ds4_gpu_compressor_prefill_tensor`
  writes) immediately after that write, in both the `zero_prefix` and `aligned_chunk` code
  paths. **This is what pinpointed the corruption to originate inside/around
  `ds4_gpu_compressor_prefill_tensor`'s write during prefill**, not in decode at all.

All additions follow the existing `if (getenv("DS4_DEBUG_TP_OUTPUT") && ok) { ...
ds4_debug_tp_output_stat_f32(...); }` pattern session 2 established. Cheap, should stay in
place for continued bisection (or get formalized behind a real `#ifdef DS4_DEBUG` — still
recommended, not yet done, three sessions in a row have left this as a TODO).

---

## 3. Where the corruption actually originates: prefill's compressor-KV write, tier 1, layer 22

Trace (from a failing run, `DS4_DEBUG_TP_OUTPUT=1`, no `HIP_LAUNCH_BLOCKING`):

```
il=22 PREFILL batch_comp_kv (post proj) tier=1 n_tokens=17     -> clean (nan=0, real spread)
il=22 PREFILL batch_comp_sc (post proj) tier=1 n_tokens=17     -> clean (nan=0, real spread)
il=22 PREFILL batch_attn_norm (compressor input) tier=1        -> clean (nan=0, real spread)
il=22 PREFILL zero_prefix attn_comp_target n_comp=4 tier=1     -> CORRUPTED (nan=128, flat ~0)
```

The three direct inputs to `ds4_gpu_compressor_prefill_tensor` (`layer_attn_state_kv/score`
freshly zero/-inf-filled at the top of that function, `batch_comp_kv`, `batch_comp_sc`) are
clean. The output (`attn_comp_target`, a view into `g->layer_attn_comp_cache[il]`) comes out
corrupted **on the same call, same tier, same layer** — i.e., something inside
`ds4_gpu_compressor_prefill_tensor` (`rocm/ds4_rocm_compressor.cuh:324`) or the immediate
device-context surrounding it, is racy specifically for this layer/tier combination.

This happens entirely during **prefill**, before any decode step runs. Session 2 characterized
this as a decode-time bug because that's where they saw NaN heads first — but decode is just
consuming a cache that prefill already corrupted. `n_comp=4` at decode time (from the earlier
decode trace) matches exactly what prefill computed (`n_comp = n_tokens/ratio = 17/4 = 4`), so
this is the same cache, not re-derived at decode time.

**Why only tier1↔tier3, only ratio-4, still holds**: layers 0-20 (tier0↔tier2, ratio-4 for
even il) go through the exact same `ds4_gpu_compressor_prefill_tensor` call and are clean every
time (checked again this session, still holds). So this is not "the kernel is broken" in
general — it's specific to whatever is different about being tier1 (the **second** pipeline
stage to run for a given prefill chunk) vs tier0 (the first).

**What was ruled out as the cause**:
- Not bad inputs (checked immediately upstream, clean).
- Not the per-layer device-switch bookkeeping: `metal_graph_set_active_tier_batch`
  (ds4.c:15510) correctly calls `ds4_gpu_set_current_device(tier)` before doing anything when
  switching tiers, and correctly no-ops (keeps current device) for consecutive layers already
  on the same tier — inspected this session, looks correct as written.
- Not literally random memory: the corrupted pattern is byte-identical across different
  failing runs (`-0.000106812` repeated, `nan=128` at the same relative positions), which
  reads as "this specific region of tier-1 VRAM, as left over from something else," not
  arbitrary garbage — consistent with a read that raced ahead of the write rather than the
  write itself computing wrong math.

---

## 4. Working hypothesis for next session (not yet verified)

`ds4_gpu_synchronize()` (`rocm/ds4_rocm_runtime.cuh:6225`) is
`cudaDeviceSynchronize()` — synchronizes **whatever device is "current" for the calling
thread**, not a specific tier. `ds4_gpu_tensor_read()`
(`rocm/ds4_rocm_runtime.cuh:6161`) is a plain `cudaMemcpy` (D2H) with **no `cudaSetDevice`
call of its own** — it relies entirely on ambient "current device" state and on UVA-addressed
pointers to make the copy "just work" regardless of which device is current. Neither of these
two primitives takes or verifies a device/tier argument. That means: any code path that reads
or syncs a tensor without having *just* called `ds4_gpu_set_current_device(that tensor's tier)`
immediately before is trusting ambient state that something else could have changed in
between.

This session did **not** find the specific place where that ambient-state assumption breaks
for layer 22 tier 1 — `metal_graph_set_active_tier_batch` looked correct on inspection (§3).
The `HIP_LAUNCH_BLOCKING=1` determinism (§1) narrows the search: it means the race is a genuine
ordering/scheduling race on the GPU timeline (two operations whose relative order isn't pinned
by a dependency edge), not a host-side "wrong current device" mistake that would reproduce
identically regardless of `HIP_LAUNCH_BLOCKING` (a wrong-device bug would still be wrong under
blocking launches, which is consistent with what we saw — but a host-side wrong-device mistake
would likely *also* show up on tier0/tier2's ratio-4 layers, which stay clean, so pure
wrong-device selection seems less likely than a missing dependency edge specific to whatever
tier1/tier3 do differently as the second stage).

**Concrete next steps**:
1. Bisect *inside* `ds4_gpu_compressor_prefill_tensor` (`rocm/ds4_rocm_compressor.cuh:324-420`)
   under `HIP_LAUNCH_BLOCKING=1` (now deterministic — no more flaky repros). Add stat reads
   between each kernel launch in that function (`cudaMemsetAsync` zero of `state_kv` →
   `fill_f32_kernel` on `state_score` → whichever of `compressor_store_kernel` /
   `compressor_prefill_pool_kernel` / the ratio-4 replay path actually runs for a 17-token,
   pos0=0 zero-prefix chunk with n_comp=4) to see which specific kernel's output is first bad.
2. Check whether tier1/tier3, being the **second** stage in the pipeline, has some
   cross-stage handoff (e.g. a `hipStream_t` or event from stage 0's work) still outstanding
   when stage 1's compressor-prefill kernels launch, that stage 0 (tier0/tier2, running first)
   never has to wait on. Grep for any explicit `cudaStreamCreate`/event usage that's stage- or
   tier-conditional (this session found `g_shared_gate_up_stream`, a global non-blocking
   stream used for MoE shared-expert work — worth checking whether it's shared/reused across
   tiers in a way that could interleave with the attention-compressor kernels; not checked
   yet).
3. Once the exact missing dependency is found, the fix is almost certainly adding one
   `cudaStreamSynchronize`/`cudaDeviceSynchronize`/event-wait at the right point — this class
   of bug is usually a one-line fix once located precisely. Don't broaden scope beyond that.

---

## 5. Things confirmed / re-confirmed this session (don't re-litigate)

- The bug reproduces on **prefill**, not just decode — session 2 only looked at decode.
- It is a genuine data race (§1), contradicting session 2's write-up. `HIP_LAUNCH_BLOCKING=1`
  gives a reliable, deterministic repro for future bisection — **use it**.
- Corruption is present in `layer_attn_comp_cache[il]` (the dense f32 compressed-KV cache) for
  layer 22 immediately after prefill's compressor write, before decode ever runs.
- `layer_attn_state_kv[il]`/`layer_attn_state_score[il]` and `layer_index_state_kv[il]`/
  `layer_index_state_score[il]` (the rolling compressor/indexer state, separate from the dense
  cache) were clean in the traces gathered this session — the corruption looks specific to the
  dense `layer_attn_comp_cache[il]` write path, not the rolling state.
- Per-layer tier switching (`metal_graph_set_active_tier_batch`) is correct as written; not the
  cause.
- Ratio-4 layers on tier0↔tier2 (layers 0, 2, 4, ..., 20) remain clean every run checked this
  session — the bug is specific to being the *second* stage tier pair.

## 5b. Aside: plain pipeline mode (no `--cuda-tensor-parallel`) currently hard-errors, unrelated bug

Checked whether the original (pre-session-1) note about garbling "in pipeline mode too (no TP
involved)" still applies with the current build, by dropping `--cuda-tensor-parallel`:

```bash
./ds4 -m ... --rocm --gpu-devices 0,1,2,3 --ctx 4096 --temp 0 -n 2 -p "..."
```

This **hard-errors** at layer 0 (`ds4: ROCm routed_moe iq2/q2 float-down counts copy failed:
invalid argument`, `gpu layer 0 ffn batch encode failed`, `multi-tier prefill failed`) — it
doesn't even reach layer 22, and the symptom (a copy-argument error, not silent NaN
corruption) is clearly a different bug from the one this checkpoint is about. Not
investigated further — out of scope for this issue's TP-specific focus (matches session 1/2's
framing) and would need its own bisection. Worth a human decision on whether it deserves its
own issue, since it currently means the non-TP 4-GPU pipeline path is *also* broken, just
differently.

## 6. Environment (unchanged from sessions 1-2)

- Hardware: 4x AMD Radeon AI Pro R9700 (gfx1201)
- Model: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf` at
  `/var/cache/llama/ds4-gguf/`
- Build: `make rocm ROCM_ARCH=gfx1201 -j$(nproc)` in `/home/murphy/src/ds4-rebase/` (default
  `ROCM_ARCH` in the Makefile is `gfx1151` — must be overridden for this hardware)
- New this session: `HIP_LAUNCH_BLOCKING=1` for deterministic repro (see §1)
