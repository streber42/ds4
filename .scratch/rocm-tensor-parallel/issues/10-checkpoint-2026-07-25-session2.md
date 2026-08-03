# Checkpoint — 2026-07-25 (session 2) Issue 10: TP Output Divergence (BOS Loop)

Status: closed

This session picked up from `.scratch/rocm-tensor-parallel/issues/10-checkpoint-2026-07-25.md`
(session 1's checkpoint, written by agy before it stopped). Session 1 proposed three
candidate root causes (A: matmul size-guard silent failure, B: cross-pair HC reduction
missing, C: `tp_ok` guard silently rejecting TP). **All three are now disproven** via live
instrumentation on the actual 4x gfx1201 hardware. A much more precise root cause has been
located. This doc supersedes session 1's candidates A/B/C.

---

## 1. How to reproduce

```bash
DS4_DEBUG_TP_OUTPUT=1 ./ds4 -m /var/cache/llama/ds4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --rocm --gpu-devices 0,1,2,3 --cuda-tensor-parallel \
  --ctx 4096 --temp 0 -n 2 \
  -p "Explain C pointers in one short sentence."
```

`DS4_DEBUG_TP_OUTPUT=1` is a new debug env var added this session (see §4) that dumps
NaN-count/sum/min/max/argmax stats for key tensors at each pipeline stage. It is currently
wired into `ds4.c` at every location described below. `-n 2` is enough to exercise one full
decode step through the instrumented per-layer path (the very first generated token comes
from a separate prefill/batch code path that isn't instrumented the same way, but shows the
identical NaN symptom).

Hardware/model unchanged from session 1: 4x AMD Radeon AI Pro R9700 (gfx1201),
`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`, placement is
GPU0 layers 0-20 + embedding, GPU1 layers 21-42 + output head, GPU2/GPU3 hold routed-expert
half-resident shards for the (0,2) and (1,3) TP pairs respectively.

---

## 2. Candidates A/B/C from session 1 — all disproven

- **Candidate A (matmul size-guard silent failure)**: Added an `fprintf` at the guard in
  `rocm/ds4_rocm_matmul.cuh` (`cuda_matmul_q8_0_tensor_labeled`). It **never fires** across
  the whole run. Ruled out.
- **Candidate C (`tp_ok` silently false)**: Added `fprintf` logging `tp_ok`/`tp_ways`/
  `head_rows`/`home` in `ds4.c` (`metal_graph_output_logits_head_matmul`). `tp_ok=1` and all
  4 per-tier shard matmuls report `ok=1` on every single call, every token. Ruled out.
- **Candidate B (cross-pair HC reduce not covering all 4 tiers)**: Added stats on
  `metal_graph_routed_out(g)` before/after every `ds4_gpu_add_xdev_tensor` /
  `metal_graph_cuda_tp_ep_finish_reduce` call (129 call sites logged across one decode
  step). **Zero NaNs** in any of them — the routed-MoE cross-device reduce is numerically
  clean on every layer. Ruled out.

So the output-TP mechanics (the vocabulary-shard matmul/gather machinery that session 1's
checkpoint focused on) are all working correctly. The bug is upstream of all of that.

---

## 3. What's actually broken: pinpointed to a single layer, single sub-step

Added stat dumps at every stage of the pipeline (see §4 for exact helper). Key trace from a
clean (uncorrupted) decode step, `DS4_DEBUG_TP_OUTPUT=1 -n 2`:

| Checkpoint | il=21 (tier1, first layer after tier0→tier1 hop) | il=22 (tier1, second layer, no hop) |
|---|---|---|
| `after_ffn_hc` (residual after full layer) | clean, sum=2973 | **all-NaN** |
| `after_attn_hc` (residual after attention only, before FFN/MoE) | clean, sum=7213-8016 across runs | **all-NaN** |
| `metal_graph_heads(g)` — the Q/K/V→attention-output intermediate, **read on the home device (tier1) before any cross-device/peer-tier code runs at all** | clean, sum≈51.6, no NaN | **all-NaN, nan_count=32768/32768 (100%)** |

**This is the critical finding**: `metal_graph_heads(g)` is already 100% NaN *before* the
code even switches to the partner tier (tier3) for the cross-device attention-TP step. That
means:

- This is **not** a cross-device / peer-access / synchronization bug. Two plausible
  synchronization bugs were found and fixed defensively this session (see §5) — neither
  changed the failure point by even one layer, which is strong evidence the real bug is not
  a race condition at all, but a deterministic logic/state bug.
- The corruption happens in plain single-device attention math (RMS-norm → Q/KV low-rank
  projection → RoPE → KV-cache lookup → softmax → weighted-V) for layer 22 on tier1, before
  the tier1↔tier3 pairing code is even reached.

### Why layer 22 and not layer 21?

`ds4_layer_compress_ratio(il)` (ds4.c:1097, and the expected-ratio table at
`ds4_expected_layer_compress_ratio`, ds4.c:1103) alternates by layer parity for the Flash
variant:

```c
case DS4_VARIANT_FLASH:
    if (il < 2) return 0;
    return (il & 1u) == 0 ? 4u : 128u;
```

So **il=21 (odd) → compress ratio 128, il=22 (even) → compress ratio 4**. These are
different attention code paths (the `compressed` bool at ds4.c:1545 gates RoPE frequency
scaling and, more importantly, which KV-cache/index-cache representation is used — see
`layer_index_state_kv`/`layer_index_state_score`, allocated per-layer around ds4.c:17037-17050).

Layers 0-20 (tier0↔tier2 pair) go through the **exact same alternating ratio pattern**
(il=0,2,4,...,20 all have ratio 4) and are **completely clean** — verified via
`after_ffn_hc` stats for every layer 0-20 (all finite, no NaN, matches expected magnitude
growth pattern layer over layer).

**So the bug requires BOTH conditions simultaneously: (a) being a compress-ratio-4
("densely compressed") layer, AND (b) being on the tier1↔tier3 TP pair specifically.**
Ratio-4 layers on tier0↔tier2 work fine; ratio-128 layers on tier1↔tier3 work fine
(il=21). Only the combination fails.

This strongly suggests some resource used by the compressed-attention (ratio=4) code path —
most likely the index/compressed-KV cache scratch tensors, or a fixed/hardcoded tier
assumption in whatever kernel handles the ratio=4 case — is not properly separated per TP
pair, and something about being the **second** pair processed in the same token's forward
pass (tier0↔tier2 runs first for layers 0-20, tier1↔tier3 runs second for layers 21-42)
causes a collision specific to the ratio=4 path.

---

## 4. Debug instrumentation added this session (all gated on `DS4_DEBUG_TP_OUTPUT` env var)

A helper `ds4_debug_tp_output_stat_f32(label, tensor, n_f32)` was added in `ds4.c` (just
above `metal_graph_output_logits_head_matmul`, forward-declared near
`metal_graph_set_active_tier_decode`). It does a synchronous device→host readback and
prints `n`, `sum`, `mean`, `min`, `max`, `argmax`, `nan` count, and the first 4 values.
Wired in at:

- `rocm/ds4_rocm_matmul.cuh` — size-guard failure fprintf (Candidate A check).
- `ds4.c` `metal_graph_output_logits_head_matmul` — `tp_ok`/per-tier-ok fprintf (Candidate C
  check), plus stat dumps of `output_norm`, each `shard_out`, and gathered `dst_logits`.
- `ds4.c` `metal_graph_set_active_tier_decode` — tier-hop fprintf + `cur_hc` src/dst stats
  (confirms the tier0→tier1 boundary copy is byte-exact and NaN-free).
- `ds4.c` `metal_graph_cuda_tp_ep_finish_reduce` (both call sites, threaded an `il`
  parameter through) — stats on partner `routed_down`, `tp_peer_tmp`, and combined
  `routed_out` (Candidate B check, all clean).
- `ds4.c` decode per-layer loop (`ds4_gpu_embed_token_hc_tensor` caller, ~line 26315) —
  per-layer tier-match assertion (`expected_tier == g->active_tier`, always OK, 43/43) plus
  `after_ffn_hc` stat right before the cur_hc pointer swap — **this is what pinpointed
  il=22 as first-NaN**.
- `ds4.c` `metal_graph_encode_output_head` — stats at every stage (`cur_hc` at entry,
  `flat_hc` after rms_norm, `output_pre` after hc_pre matmul, `output_weights`,
  `output_embd` after weighted sum) — confirmed NaN is already present at entry, i.e.
  upstream of all output-head-specific math.
- `ds4.c` attention TP block (~line 22537, inside `metal_graph_encode_decode_layer_phase`)
  — `after_attn_hc` stat right after the attention residual-add, and a stat on
  `metal_graph_heads(g)` **before** the switch to the partner tier — **this is what proved
  the bug is single-device, pre-cross-device**.

All of these are cheap to re-enable (`DS4_DEBUG_TP_OUTPUT=1`) for continued bisection. They
should probably be left in place (or converted to a proper `#ifdef DS4_DEBUG`-gated block)
rather than stripped, since they'll be needed again to narrow further inside the ratio=4
attention path.

---

## 5. Two real (but not-the-cause) synchronization bugs fixed defensively

While chasing the (ultimately wrong) cross-device-race hypothesis, two genuine bugs were
found and fixed. Both are correct fixes for real issues, but neither changed the il=22
failure point at all, confirming the actual bug is not a race:

1. **`ds4_rocm_xdev_copy` peer-copy path** (`ds4_rocm_xdev.cu:143-149`): called
   `hipSetDevice(dst_dev)` then `hipMemcpyPeerAsync` then `hipDeviceSynchronize()` — but
   that synchronize targets the *destination* device, not the *source* device whose
   kernels wrote the data being copied. Fixed by synchronizing `src_dev` before issuing the
   peer copy.
2. **`ds4_gpu_tensor_wait_xdev`** (`ds4_rocm_compat.cu:221`): this was a **no-op stub**
   (`return src && rocm_tier_valid(dst_tier);` — validates pointers, does no actual wait/
   sync). It's used as the synchronization point before direct peer *reads* (as opposed to
   explicit copies) in the attention TP-peer-read fast path
   (`DS4_CUDA_TP_ATTN_PEER_READ`, **enabled by default** on non-Apple builds — see
   `metal_graph_cuda_tp_attn_peer_read_requested`, ds4.c:16282-16288). Fixed to actually
   `hipSetDevice(src_dev); hipDeviceSynchronize();`.

Both fixes are real correctness improvements (a stub synchronization primitive is a latent
bug waiting to bite under different timing/scheduling) and should be kept regardless of
whether they resolve issue 10. They add a small latency cost (device sync) but should not
be removed.

---

## 6. Next steps (not yet done)

1. Bisect **inside** layer 22's attention computation to find exactly which sub-step
   produces the all-NaN `metal_graph_heads(g)`: RMS-norm → Q/KV-A projection → RoPE →
   KV-cache append/lookup → attention score/softmax → weighted-V. Add
   `ds4_debug_tp_output_stat_f32` calls at each intermediate (the tensors are all local to
   `metal_graph_encode_decode_layer_phase`, upstream of the ~line 22461 TP block already
   instrumented this session).
2. Specifically check the **compressed-KV/index-cache path** (`compressed` bool,
   ds4.c:1545; `layer_index_state_kv`/`layer_index_state_score`, ds4.c:17037-17050) for
   anything tier/pair-specific — e.g. a buffer allocated assuming it's only ever used by
   the tier0↔tier2 pair, a hardcoded device id, or a global (non-per-tier) scratch tensor
   that tier0↔tier2's ratio-4 layers (0,2,4,...,20) and tier1↔tier3's ratio-4 layers
   (22,24,...,42) both write to without proper separation.
3. Confirm the "second pair to run" hypothesis directly by testing tier1↔tier3 (or
   tier0↔tier2) as the *only*/*first* pair processed. **Not possible with this 87GB model
   on this hardware** — any 2-GPU subset (`--gpu-devices 0,1`, `1,3`, etc.) has only
   ~54-60GB total VRAM budget, well under the ~87GB the weights need, so placement
   classification fails outright regardless of which 2 tiers are chosen (confirmed this
   session). Would need either a smaller compressed-attention DeepSeek4-Flash GGUF that
   fits in 2×27GB, or a way to force the full 43-layer network onto a single TP pair on
   this same 4-GPU box (if the placement/layout code supports that — not yet checked).

## 7. Things ruled out / confirmed correct this session (don't re-check these)

- Output-TP vocab-shard matmul/gather mechanics (all of session 1's candidates A/B/C).
- Cross-device `cur_hc` boundary-hop copy (tier0→tier1) — byte-exact, verified.
- Per-layer active-tier bookkeeping in the decode loop swap (43/43 layers correct).
- Routed-MoE cross-tier reduce (`ds4_gpu_add_xdev_tensor` /
  `metal_graph_cuda_tp_ep_finish_reduce`) — clean on every layer.
- 1-GPU and 2-GPU repro attempts are **not viable** on this hardware for this specific
  87GB quantization: 1-GPU has only 25.8GB free (needs CPU-spill, not implemented yet);
  2-GPU budget (~54GB total) is still too small, placement classification fails outright.
  Only 4-GPU fits.

## 8. Environment (unchanged from session 1)

- Hardware: 4x AMD Radeon AI Pro R9700 (gfx1201)
- Model: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf` at
  `/var/cache/llama/ds4-gguf/`
- Build: `make rocm ROCM_ARCH=gfx1201 -j$(nproc)` in `/home/murphy/src/ds4-rebase/`
  (default `ROCM_ARCH` in the Makefile is `gfx1151` — must be overridden for this
  hardware, `make rocm` alone will build for the wrong arch).
