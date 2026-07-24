# Handoff: Issue 05 — First correct token on 2-rank TP (decode path)

Written by an agent session that ran out of turn budget partway through. Read
this alongside `05-first-correct-token.md` (the issue) and `PRD.md` (the
parent feature). This doc is a handoff, not a memory record — delete it once
issue 05 is closed.

## CRITICAL: where this work lives

**All work described below is uncommitted, in the working tree at
`/home/murphy/src/ds4-rebase` on branch `gfx1201_tp`.** It is NOT on any
commit, NOT stashed. A fresh `git worktree add` from this branch's tip
(`38d87e3`) will **not** contain any of it.

The continuing agent must operate directly in `/home/murphy/src/ds4-rebase`
(this exact directory), not a new worktree, until this work is committed.
Run `git status` / `git diff --stat` there first thing to confirm you're
seeing the state this doc describes before doing anything else.

Current `git diff --stat` against HEAD (38d87e3):

```
 .../issues/05-first-correct-token.md               |   2 +-
 ds4.c                                              |   8 +
 ds4_rocm.cu                                        |  92 +++++++++--
 ds4_rocm_unavailable.cu                            |  21 ++-
 ralph.sh                                           |  15 +-   (pre-existing, unrelated to this work — do not touch/revert)
 rocm/ds4_rocm_hc_output_launch.cuh                 |  31 ++++
 rocm/ds4_rocm_matmul.cuh                           |  66 ++++++++
 rocm/ds4_rocm_moe.cuh                              | 170 +++++++++++++++++++
 rocm/ds4_rocm_moe_launch.cuh                       | 183 +++++++++++++++++++++
 rocm/ds4_rocm_q8.cuh                               |  39 +++++
 rocm/ds4_rocm_shared_expert.cuh                    | 143 ++++++++++++++++
 tests/test_rocm_kernel_compare.cu                  |  78 +++++++++  (IN PROGRESS, see below)
 12 files changed, 822 insertions(+), 26 deletions(-)
```

Also present but pre-existing/unrelated, do not touch: modified `ralph.sh`,
untracked `tests/test_engine_correctness_harness` (a stale build artifact —
just a binary, safe to rebuild/overwrite).

The build is currently **green**: `make rocm ROCM_ARCH=gfx1201 -j8` compiles
clean, and `make ROCM_ARCH=gfx1201 test-rocm -j8` passes (existing xdev/stub
tests + the 2 pre-existing kernel-compare cases). Hardware is real: 4x AMD
Radeon AI Pro R9700 (gfx1201) visible via `rocm-smi`/`rocminfo`.

## What issue 05 actually needs (recap)

Make the two-rank tensor-parallel **decode** path (single-token prompt)
numerically correct, gated by the existing correctness harness
(`tests/test_engine_correctness_harness.c`, from issue 03) plus
kernel-level equivalence evidence via the comparison scaffold
(`tests/test_rocm_kernel_compare.cu`, from issue 03b) for the first kernel
ported in each subsystem. Prefill kernels are explicitly out of scope
(deferred to issue 07).

## Scoping work already done (read this before re-deriving it)

I spent a large fraction of this session's budget mapping which of the ~31
ROCm TP stub entry points are actually reachable on the **decode** path for
**2-rank** TP with the default flag values, using an Explore agent plus
direct reading of `ds4.c`'s TP decode branch (search for `cuda_tp_attn`,
`cuda_tp_ep`, `cuda_tp_ep_pack_exact`, `cuda_tp_ep_fused_hc_reduce` in
`ds4.c` around lines 22350–23850). Conclusion: only **6 real kernels**
plus **2 one-time registration hooks** gate correctness for decode; the
rest of the ~31 stubs are either prefill-only, DSpark-only, dead
(`g->tp_world==2` Metal-only gate, unreachable on ROCm/CUDA), or
optional-fusion paths that are off by default. Do not re-derive this from
scratch — see the completed/skipped task list below for the final
breakdown, and re-verify against current `ds4.c` if anything seems off
(code may have moved).

## Completed (all verified compiling; NOT yet run against real weights)

1. **`ds4_gpu_attention_output_q8_tp_tensor`** — `ds4_rocm.cu` (was a stub
   in the TP block ~line 199, now real). Thin wrapper exactly matching
   CUDA's `ds4_cuda.cu` version: group-sliced attention-output A-projection
   + B-projection k-slice, reusing the already-real ROCm
   `ds4_gpu_attention_output_low_q8_tensor` and the newly-ported
   `ds4_gpu_matmul_q8_0_kslice_rows_tensor` (see next item). No new
   low-level math — this one is a pure composition.

2. **`ds4_gpu_matmul_q8_0_kslice_rows_tensor`** — new kernel
   `matmul_q8_0_kslice_preq_warp8_kernel` in `rocm/ds4_rocm_q8.cuh`
   (inserted after `matmul_q8_0_preq_warp8_kernel`), wrapper in
   `rocm/ds4_rocm_matmul.cuh` (after `ds4_gpu_matmul_q8_0_tensor`). Ported
   directly from `ds4_cuda.cu`'s function of the same name, adapted to
   ROCm's simpler (single-tier, no `logical_tier` param) `cuda_model_range_ptr`
   / `cuda_tmp_alloc` conventions and hardcoded `use_dp4a=1` (matches how
   every other ROCm Q8_0 kernel in this file does it — CUDA calls
   `cuda_q8_use_dp4a()`, ROCm doesn't have that helper and always uses 1).

3. **`ds4_gpu_hc_expand_add_tensor`** — real impl in
   `rocm/ds4_rocm_hc_output_launch.cuh` (after `ds4_gpu_hc_expand_tensor`),
   removed the stub definition that lived directly in `ds4_rocm.cu`.
   Reuses the *already-existing* ROCm `hc_expand_kernel` (12-param
   signature, no `block_add2`/`has_add2`) — confirmed this is exactly
   equivalent to CUDA's 14-param version called with `has_add2=0`, so no
   new kernel was needed, just a new host-side wrapper with the right
   validation (modeled on the existing `ds4_gpu_hc_expand_tensor`
   validation style in the same file).

4. **`ds4_gpu_routed_moe_one_owned_tensor`** — the big one. See "Key design
   decision" below for why this ended up much smaller in scope than CUDA's
   version. New device helper `moe_owned_local_expert` + two new kernels
   `moe_gate_up_mid_owned_f32_kernel` / `moe_down_owned_f32_kernel` in
   `rocm/ds4_rocm_moe.cuh` (appended after the existing, **already
   production-used** `moe_gate_up_mid_f32_kernel` / `moe_down_f32_kernel`
   — confirmed these are live call sites in `routed_moe_launch`'s fallback
   branch in `rocm/ds4_rocm_moe_launch.cuh` around line 2229–2275, not dead
   code, so building on them is low-risk). Wrapper in
   `rocm/ds4_rocm_moe_launch.cuh` after `ds4_gpu_routed_moe_one_tensor`.
   Only supports `gate_type==16` (IQ2_XXS) / `down_type==10` (Q2_K), which
   is confirmed (via `rocm/ds4_rocm_moe_launch.cuh`'s own `plan->iq2_path`
   check) to be what this model's routed experts actually use. `pack_fixed3`
   always returns 0 (see design decision below — this is intentional, not
   a TODO).

5. **`ds4_gpu_routed_moe_owned_slots_combine_tensor`** (+
   `..._rows_tensor`) — new kernel `moe_owned_slots_combine_kernel` +
   wrappers in `rocm/ds4_rocm_moe_launch.cuh`, right after item 4's
   wrapper. Ported near-verbatim from CUDA's
   `moe_owned_slots_combine_fixed3_kernel`.

6. **`ds4_gpu_shared_mid_swiglu_q8_0_decode_exact_tensor`** — new kernel
   `shared_mid_q8_0_preq_warp8_exact_kernel` + wrapper in
   `rocm/ds4_rocm_shared_expert.cuh` (after
   `ds4_gpu_shared_gate_up_swiglu_q8_0_tensor`). Ported verbatim from CUDA
   including the exact same-count tie-break rule for which rank "wins" the
   whole-kernel assignment — this must match the reference bit-for-bit or
   the wrong rank computes shared-expert output for a whole layer.

7. **Two one-time registration/cache hooks made real** (`ds4_rocm.cu`,
   right before the TP entry-point block):
   - `ds4_gpu_register_model_map_no_copy` → delegates to the already-real
     `ds4_gpu_set_model_map`.
   - `ds4_gpu_device_cache_tensors` → real, permanent no-op (always
     returns 0/success). This is NOT a bring-up bypass; it's documented as
     correctness-safe because every weight read goes through
     `cuda_model_range_ptr`, which already has a host-fallback path when no
     device range was cached. Confirmed these two ARE reached on the
     decode-TP model-load path (`engine_install_per_device_caches` in
     `ds4.c`, called unconditionally for every multi-tier session).
     Confirmed `ds4_gpu_register_support_map` /
     `ds4_gpu_device_cache_support_tensors` are DSpark-only
     (`engine_install_dspark_support_cache` early-returns 0 when
     `e->dspark` is unset — true for a plain TP decode session) and can
     stay stubs.

8. **Narrow shared-engine override in `ds4.c`**:
   `metal_graph_cuda_tp_ep_pack_exact_requested()` now returns `false`
   unconditionally under `#elif defined(DS4_ROCM_BUILD)`. This is the key
   design decision — see below. Does not touch the CUDA (`#else`) or Metal
   (`#if defined(__APPLE__)`) branches at all.

## Key design decision: forcing `cuda_tp_ep_pack_exact = false` for ROCm

CUDA's `ds4_gpu_routed_moe_one_owned_tensor` supports a "packed" 4-slot
output layout (`pack_fixed3=true`) that's a pure VRAM/perf optimization
(folds 2 always-co-owned slots of the 6 selected experts into 1). It
defaults ON (`DS4_CUDA_TP_EP_PACK_EXACT` env default `true`), and — this
is the important part — `cuda_tp_ep_fused_hc_reduce` (which gates whether
`ds4_gpu_shared_down_hc_expand_owned_q8_0_tensor` gets called at all) is
computed in `ds4.c` as `g->cuda_tp_ep_pack_exact && ...`, i.e. it's ANDed
with pack_exact.

By forcing `cuda_tp_ep_pack_exact = false` on ROCm, `cuda_tp_ep_fused_hc_reduce`
becomes unconditionally false too, which means:
- `ds4_gpu_routed_moe_one_owned_tensor` only ever needs to support the
  plain unpacked 6-slot layout (much simpler than CUDA's packed
  `moe_owned_packed_component` indexing scheme).
- `ds4_gpu_shared_down_hc_expand_owned_q8_0_tensor` (a fused
  shared-expert-down + HC-expand + owned-MoE-reduce kernel) becomes
  **entirely unreachable** on ROCm — traced this through `ds4.c`'s
  `metal_graph_cuda_tp_ep_finish_reduce` and the branch at ~line 23750
  (`else if (ok && cuda_tp_ep_fused_hc_reduce)` → false → falls through to
  the already-real, non-owned `ds4_gpu_shared_down_hc_expand_q8_0_tensor`
  / `_add_q8_0_tensor` instead). This eliminated what would have been the
  single hardest remaining kernel to port.
- Also eliminated the need for `ds4_gpu_routed_moe_owned_packed_combine_tensor`
  (still a stub, confirmed unreachable).

Verify at `ds4.c`'s `metal_graph_cuda_tp_ep_pack_exact_requested` (search
for it) that this override is still there and still reads correctly if
you're re-deriving state — this is load-bearing for the whole MoE port
being as small as it is. If you ever see
`ds4_gpu_shared_down_hc_expand_owned_q8_0_tensor` or
`ds4_gpu_routed_moe_owned_packed_combine_tensor` actually getting called
(process aborts there), this override has regressed or something upstream
changed the flag wiring — re-check the AND condition on
`cuda_tp_ep_fused_hc_reduce` in `ds4.c`.

## IN PROGRESS — do not treat as done

`tests/test_rocm_kernel_compare.cu` — I was mid-way adding kernel-level
numeric-equivalence test cases (acceptance criterion: "the first kernel
ported in each subsystem touched here has kernel-level numeric-equivalence
evidence via the scaffold"). So far only added **helper functions** (not
yet wired into `KCMP_CASES` or even a full test case):
- `kcmp_f32_to_half_bits` / `kcmp_half_to_f32` — binary32↔binary16
  round-trip helpers for building Q8_0 fixtures on the host.
- `kcmp_pack_q8_0_block` — packs a 34-byte Q8_0 block (matches the layout
  every Q8_0 kernel in this codebase reads: 2-byte half scale + 32 int8
  values), using the same amax/127 quantization scheme
  `quantize_q8_0_f32_kernel` uses.

**Not yet done**: an actual `kcmp_run_*` test function using these helpers
(intended target: `ds4_gpu_matmul_q8_0_kslice_rows_tensor`, representing
the attention/matmul subsystem — build a small fake model_map via plain
`malloc` + `ds4_gpu_matmul_q8_0_kslice_rows_tensor` called directly with
that pointer as `model_map`, since `cuda_model_range_ptr`'s
`cuda_model_range_copy_uncached` fallback handles an unregistered host
pointer with no setup required — confirmed by reading
`rocm/ds4_rocm_runtime.cuh`'s `cuda_model_range_ptr`), plus one entry in
`KCMP_CASES`, plus (acceptance criterion wants "first kernel in each
subsystem") likely two more small cases for the routed-MoE and
shared-expert subsystems. The MoE case is harder because IQ2_XXS/Q2_K
fixture construction (lookup-table grid decode, 2-bit packed scales) is
much more involved than Q8_0 — budget real time for that if you take it
on, or consider whether the end-to-end harness (task 9) already exercising
these kernels is judged sufficient supporting evidence alongside the Q8_0
scaffold case, and note that tradeoff explicitly in the issue file's
Comments section rather than silently skipping it.

## NOT STARTED

- **Run the actual correctness harness** (`tests/test_engine_correctness_harness`,
  from issue 03) against the real 2-rank TP decode path with the real
  80GB DeepSeek-V4-Flash GGUF, comparing logits to the same-hardware
  pipeline reference, checking greedy-token match, and checking
  reproducibility across repeated runs (all explicit acceptance criteria
  in `05-first-correct-token.md`). **This has not been attempted yet** —
  everything above is "should be correct by construction and by careful
  reading," not verified against real hardware output. Given the
  complexity of what was ported (especially the MoE owned/combine path and
  the shared-mid load-balance tie-break), budget for this to surface bugs.
  Rebuild first (`make rocm ROCM_ARCH=gfx1201 -j8`) since nothing has been
  rebuilt since the last edits to `ds4.c` / `ds4_rocm_moe_launch.cuh` /
  `ds4_rocm_shared_expert.cuh`.
- Two production vLLM services were mentioned as running on this box in
  prior issue commits (see `04-tp-plumbing-no-crash.md` comments) —
  running a real 2-GPU 80GB-model TP session may need those paused with
  user authorization first, same as issue 04 did. Check current GPU
  utilization (`rocm-smi`) before assuming all 4 GPUs are free.
- If the harness fails, the natural debugging path is: (a) narrow to
  `DS4_ROCM_TP_BRINGUP=1` to confirm the plumbing itself still doesn't
  crash (should already hold, nothing here touched the bring-up-mode
  behavior itself), then (b) use `tests/test_rocm_kernel_compare.cu`
  pointed at individual kernels once the scaffold cases above exist, per
  the PRD's testing strategy ("first correct token" issue's own guidance:
  "When end-to-end fails, the scaffold is the localization tool").
- Update `.scratch/rocm-tensor-parallel/issues/05-first-correct-token.md`:
  check off acceptance criteria as they're verified, change `Status:
  in-progress` to `Status: closed` only once the harness genuinely passes
  reproducibly — do not close on a clean build alone.
- Commit with message
  `feat(rocm-tensor-parallel): First correct token on 2-rank TP (decode path)`
  once verified. Review `git status`/`git diff` before staging — this repo
  has an unrelated pre-existing modification to `ralph.sh` sitting in the
  working tree; don't fold that into this commit unless the user asks.

## Things I'd flag if I kept going

- I have NOT verified the `moe_owned_slots_combine_kernel`'s float
  reassociation (home = slot0+slot1+slot2, peer = slot3+slot4+slot5, out =
  home+peer) actually matches what the reference pipeline path produces
  bit-for-bit-enough to pass the harness tolerance — I ported it verbatim
  from CUDA's `moe_owned_slots_combine_fixed3_kernel` on the assumption
  CUDA's own choice of reassociation was itself already validated against
  the *Metal* reference at some point upstream, which I did not confirm.
  If the harness tolerance is tight and this specific sum order turns out
  to matter, this is the first place to look.
- The whole MoE owned/combine design leans on a documented invariant (a
  slot is written by at most one rank, read from the other rank's copy
  when not locally owned) that is enforced purely by both kernels
  independently re-deriving ownership from the same `selected[]` array via
  `moe_owned_local_expert`. I did not add an assertion or test proving
  `selected` is bit-identical between ranks at the point these kernels
  run (it's supposed to be, via the earlier cross-device copy of
  `router_selected`/`peer_selected` in `ds4.c`) — if it ever isn't, both
  ranks could silently double-count or drop a slot with no loud failure.
  Worth a debug-mode assertion if this becomes a real bug.
