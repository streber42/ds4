Status: ready-for-agent
# 03 — Replace this repo's broken raw-builtin gfx12 WMMA kernel with the verified rocwmma one

**What to build:** This repo (`ds4-rebase`) still carries the *literal* `d9be29f`
gfx12 WMMA port — the exact implementation that issue 01 empirically falsified and
threw away. Issue 01's fix shipped only to the **other** tree (`~/src/ds4`, commit
`3d7693f`). Port the verified rocwmma body here, add the missing
`DS4_ROCM_NO_WMMA` escape hatch so a correctness reference exists, and measure how
much traffic this kernel actually sees on this tree's TP=4 / pipeline prefill path.

**Blocked by:** None. Do this before any TP=4 or prefill measurement work in this
tree — see "Why this blocks" below.

## The defect

`rocm/ds4_rocm_q8.cuh`, `matmul_q8_0_f32_batch_wmma_4w_kernel`, gfx12 branch at
lines 786–814. A Q8_0 block is 34 bytes (2 scale + 32 int8 weights), with
`w0 = bp + 2` and `w1 = bp + 18` (lines 782–783):

| build | A operand | B operand | K covered |
|---|---|---|---|
| gfx11 (`#else`, line 816) | `w0[0..15]`, `w1[0..15]` | `xb`, `xb + 16` | all 32 |
| gfx12 (`DS4_RDNA4`, line 786) | `w0[0..7]`, `w1[0..7]` | `xb`, `xb + 8` | **16 of 32** |

The gfx12 path silently drops weights 8–15 and 24–31 — block bytes 10–17 and
26–33 — so half of every dot product is missing. This is **data loss, not a
layout permutation**, which is why issue 01's candidate A measured mean_abs error
106.6 against a double-precision CPU reference, and why the full model emitted
gibberish with next-token argmax 35716 ("Kasarangang") instead of 2581 ("We").

It is live in this tree: `DS4_RDNA4` is auto-defined from `__GFX12__`
(`ds4_rocm.h:14`), and unlike `~/src/ds4` this tree has **no `DS4_ROCM_NO_WMMA`
macro anywhere** — so every gfx1201 build compiles and dispatches it.

Provenance: commit `79f09d4` ("rocm: add gfx12 (RDNA 4) WMMA intrinsics for
gfx1200/gfx1201"), which predates all TP work. **This is ours alone** — verified
that `upstream/main` and `origin/main` contain 8 raw gfx11 builtins and zero
`_gfx12` occurrences in this file, i.e. upstream has no gfx12 WMMA path at all
and cannot compile this kernel for gfx1201. No upstream merge will fix it.

## Why this blocks

The only thing containing the blast radius is `!g_quality_mode` at
`rocm/ds4_rocm_matmul.cuh:404`, which turns this kernel off inside the fixture.
Two consequences, and **both matter**:

1. **The fixture's numbers are clean.** `avg_nll 0.3699` and the entire #32–#48
   quality investigation never dispatched this kernel. Do not write a narrative
   connecting this bug to those issues — it is ruled out by that gate.
2. **Nothing else is.** Production-mode inference on gfx1201 reaches it whenever
   `n_tok >= 256 && out_dim >= 1024 && in_dim % 32 == 0`, and callers include the
   attention Q/K/V/O projections (`rocm/ds4_rocm_attention_launch.cuh:1220,1443`).
   Any TP=4 or prefill throughput/quality number measured in this tree in
   non-quality mode has a half-computed matmul in the path.

## Acceptance criteria

- [ ] The `DS4_RDNA4` branch of `matmul_q8_0_f32_batch_wmma_4w_kernel` in
      `rocm/ds4_rocm_q8.cuh` is replaced by the rocwmma-fragment implementation
      from `~/src/ds4` commit `3d7693f` (`git show 3d7693f -- rocm/ds4_rocm_q8.cuh`
      in that repo). Keep its explanatory comment — it records why the raw
      builtin was abandoned. The gfx11 `#else` body must be left byte-identical.
- [ ] A `DS4_ROCM_NO_WMMA` guard is introduced in this tree, matching
      `3d7693f`'s structure, so the kernel can be compiled out. **This is
      required, not optional**: without it there is no in-tree reference build
      to A/B correctness against, and no fallback if the WMMA path regresses.
- [ ] Both configurations build cleanly for `gfx1201` (no new warnings from the
      changed files).
- [ ] Correctness, production mode (i.e. **not** under `score_official`, which
      gates this kernel off): with a `>256`-token prompt, `--dump-logits` argmax
      and top-5 match the `DS4_ROCM_NO_WMMA` reference build, and a 30-token
      greedy generation is byte-identical between the two builds. Record the
      mean/max absolute logit error. Issue 01's reference figures for the fixed
      kernel were mean 0.51 / max 2.98 on the other tree.
- [ ] Dispatch traffic measured, same instrument issue 01 used: `rocprofv3
      --kernel-trace` on a short **non-quality-mode** prefill run *in this tree*,
      reporting the dispatch count of `matmul_q8_0_f32_batch_wmma_4w_kernel`.
      This tells us whether the bug affected the TP=4/pipeline path specifically
      or all gfx1201 prefill — #39/#40 moved attention output to cuBLAS and MoE
      goes through the Q2_K rocwmma hotlist kernels, so real traffic is an open
      question. **Report the count honestly, including if it is zero** — a zero
      is a valuable result, not a failed AC.
- [ ] Result written to `## Answer`: what changed, the parity numbers, the
      dispatch count, and whether this bug had any measurable blast radius.

## Notes for the agent

- The port is small and self-contained (`3d7693f` is 3 files / ~136 insertions),
  but this tree's `ds4_rocm_q8.cuh` has diverged from `~/src/ds4`'s — it is a
  port, not a cherry-pick. Read both bodies before editing.
- Requires the GPU lock (see `AGENTS.md`). Acquire once and do the build,
  parity check, and `rocprofv3` trace in a single session — the model load is
  expensive, so don't split them across lock cycles.
- Do **not** flip any default build target to the WMMA path in this issue.
  Validate-before-default, per `docs/adr/0001-dense-tp4-parked-sequential-default.md`.

## Answer

## Comments

Filed 2026-08-11 from an interactive human session, after auditing why issue 01's
"+14.5% prefill" result did not appear in this tree. Root cause of the confusion:
issue 01 and 02 both ran entirely in `~/src/ds4` (single R9700, 86 GiB model,
`--ssd-streaming`); this repo never received the fix.
