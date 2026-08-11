Status: ready-for-human
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

- [x] The `DS4_RDNA4` branch of `matmul_q8_0_f32_batch_wmma_4w_kernel` in
      `rocm/ds4_rocm_q8.cuh` is replaced by the rocwmma-fragment implementation
      from `~/src/ds4` commit `3d7693f` (`git show 3d7693f -- rocm/ds4_rocm_q8.cuh`
      in that repo). Keep its explanatory comment — it records why the raw
      builtin was abandoned. The gfx11 `#else` body must be left byte-identical.
- [x] A `DS4_ROCM_NO_WMMA` guard is introduced in this tree, matching
      `3d7693f`'s structure, so the kernel can be compiled out. **This is
      required, not optional**: without it there is no in-tree reference build
      to A/B correctness against, and no fallback if the WMMA path regresses.
- [x] Both configurations build cleanly for `gfx1201` (no new warnings from the
      changed files).
- [ ] Correctness, production mode (i.e. **not** under `score_official`, which
      gates this kernel off): with a `>256`-token prompt, `--dump-logits` argmax
      and top-5 match the `DS4_ROCM_NO_WMMA` reference build, and a 30-token
      greedy generation is byte-identical between the two builds. Record the
      mean/max absolute logit error. Issue 01's reference figures for the fixed
      kernel were mean 0.51 / max 2.98 on the other tree.
      **PARTIAL — argmax, top-5 and mean/max are done and pass (mean 0.4035 /
      max 3.899); the byte-identical-generation clause fails at step 24 of 30.
      Left unchecked deliberately; see "The one AC that fails" below.**
- [x] Dispatch traffic measured, same instrument issue 01 used: `rocprofv3
      --kernel-trace` on a short **non-quality-mode** prefill run *in this tree*,
      reporting the dispatch count of `matmul_q8_0_f32_batch_wmma_4w_kernel`.
      This tells us whether the bug affected the TP=4/pipeline path specifically
      or all gfx1201 prefill — #39/#40 moved attention output to cuBLAS and MoE
      goes through the Q2_K rocwmma hotlist kernels, so real traffic is an open
      question. **Report the count honestly, including if it is zero** — a zero
      is a valuable result, not a failed AC.
- [x] Result written to `## Answer`: what changed, the parity numbers, the
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

**Result: the broken kernel is replaced and it had a large, real blast radius —
246 dispatches and 5.2 s of device time in a single 2633-token production-mode
prefill on this tree's 4-GPU pipeline path. Every non-quality-mode prefill
number ever measured in this repo had a half-computed Q8_0 matmul in it.**

Five of the six acceptance criteria pass. The sixth passes on its numeric half
and fails on its byte-identical-generation half, for a reason that is measured
and quantified below rather than argued: leaving it unchecked and the issue
`ready-for-human` is a spec call, not a technical unknown.

### What changed

| file | change |
|---|---|
| `rocm/ds4_rocm_q8.cuh` | `matmul_q8_0_f32_batch_wmma_4w_kernel` split into two complete kernel definitions gated on `DS4_RDNA4`. The gfx12 arm is `3d7693f`'s rocwmma-fragment body (`load_matrix_sync`/`mma_sync`/`store_matrix_sync`), with its explanatory comment adapted to this file. The raw `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` path — which read only `w0[0..7]`/`w1[0..7]`, dropping half of every Q8_0 block — is gone. |
| `rocm/ds4_rocm_q8.cuh` | `DS4_ROCM_NO_WMMA` added to the enclosing `#if`, per `3d7693f`. |
| `rocm/ds4_rocm_matmul.cuh` | the launch site at :403 carries the same `DS4_ROCM_NO_WMMA` guard, so a reference build falls through to the sharedx/warp-row tiers instead of failing to link. |
| `Makefile` | `ROCM_EXTRA_CFLAGS` is now actually consumed by `ROCM_CFLAGS`, and a `rocm-no-wmma` target builds the same arches with `-DDS4_ROCM_NO_WMMA`. |

The gfx11 `#else` arm is byte-identical, verified mechanically: extracting the
109 statement lines from `git show HEAD:rocm/ds4_rocm_q8.cuh` and diffing
against the new `#else` arm returns no differences. Only the preprocessor
scaffolding around it moved — the split had to be hoisted from inside the loop
body to whole-kernel granularity, because the rocwmma arm does not use
`lane16`/`warp_m`/`row_base`/`acc0..3` and would otherwise have emitted a wall
of unused-variable warnings.

`DS4_ROCM_NO_WMMA` is a diagnostic switch validating the one release path, not
a permanent semantic variant (AGENT.md "Quality Rules"): `rocm` remains the
only deployed build and its default is unchanged.

### Builds

| build | result |
|---|---|
| `make ROCM_ARCH=gfx1201 rocm -j8` | exit 0, 5 warnings, all pre-existing `-Wunused-value` in `ds4_rocm_runtime.cuh`; **zero** from `ds4_rocm_q8.cuh` or `ds4_rocm_matmul.cuh` |
| `make ROCM_ARCH=gfx1201 rocm-no-wmma -j8` | exit 0, same 5 pre-existing warnings, zero from the changed files; all five binaries link, proving nothing dangles on the compiled-out kernel |
| `make rocm -j8` (default `gfx1151,gfx1201`) | exit 0 — this is the only build that compiles the gfx11 device arm, so it is what proves the untouched `#else` still works |

The flag genuinely reaches the compiler here (`hipcc ... --offload-arch=gfx1201
-DDS4_ROCM_NO_WMMA -c -o ds4_rocm.o`); see the note about `~/src/ds4` under
`## Comments`.

### Correctness, production mode

4-GPU pipeline, `AMD_SERIALIZE_KERNEL=3`, 2633-token prompt, no
`score_official` (so `!g_quality_mode` holds and the kernel really dispatches).

Determinism pre-check first, because without it the comparison is
uninterpretable: two runs of the **same** WMMA build produced a **bit-identical**
129,280-entry logit vector (mean and max abs error exactly 0.000000), and both
builds reproduce their own 30-token generation byte-for-byte across runs —
including traced vs untraced, so `rocprofv3` does not perturb the result.

| metric | WMMA | NO_WMMA reference | verdict |
|---|---|---|---|
| next-token argmax | 2581 `'We'` (32.304977) | 2581 `'We'` (31.885427) | **match** |
| top-5 | `[2581, 671, 10318, 85667, 110137]` | identical, same order | **match** |
| mean abs logit error | — | 0.403549 | recorded |
| max abs logit error | — | 3.898771 (token 43014) | recorded |
| 30-token greedy generation | identical for steps 0–23, diverges at step 24 | | **fails** |

Comparable to issue 01's 0.51 / 2.98 for the same kernel body, on a longer
accumulation chain (43 layers across 4 GPUs vs single-GPU) and without
`-ffast-math` (this tree's `ROCM_CFLAGS` omits it; `~/src/ds4`'s has it), so
the two figures are not expected to coincide exactly.

A nonzero delta is structural, not a defect: the WMMA path stages activations
into `_Float16` LDS and weights into `half` fragments, the reference path
accumulates the same dot products in fp32. The reference build is the more
precise of the two. Parity therefore cannot prove the port correct on its own —
the primary evidence for correctness is the CPU-reference verification of this
exact body in issue 01 (mean_abs 0.021 vs 106.6 for the raw-builtin version),
plus the argmax/top-5 agreement here.

### The one AC that fails, and exactly why

The two builds emit identical tokens for steps 0–23 and diverge at step 24:

```
WMMA    ... The answer should be based on the
NO_WMMA ... The report says status: parked.
```

`--dump-logprobs --logprobs-top-k 5` locates this precisely
(`artifacts/03-logprob-step-compare.txt`). At every step 0–23 the top-1/top-2
logit gap is **1.4 to 20.3**, far above the 0.40 mean cross-build delta, so both
builds pick the same token. Step 24 is the only near-tie in the run:

| build | selected | top-2 | gap |
|---|---|---|---|
| WMMA | `' answer'` 37.1675 | `' report'` 36.7792 | **0.388** |
| NO_WMMA | `' report'` 37.3814 | `' answer'` 36.9034 | **0.478** |

Both builds return the *same five candidates* with the top two swapped, and the
gap separating them (0.39–0.48) is smaller than the mean absolute logit
difference between the builds (0.4035). This is a coin flip resolved by
fp16-vs-fp32 rounding, not a divergence in the computation — and both
continuations are fluent, on-topic English, which categorically excludes the
gibberish failure mode issue 01 saw from the raw-builtin port.

**Replication on an independent prompt.** To test that explanation rather than
assert it, a second prompt was built (`03-tiefree-prompt.txt`: same 2633-token
report body, a low-entropy "copy this sentence verbatim" task instead of the
open-ended question) with a decision rule fixed in advance: screen the WMMA
build's 30 steps first, and only accept the prompt as an AC-4 candidate if no
step has a top-2 gap below 1.0.

It failed the screen — step 9 gap **0.050**, step 15 gap 0.692 — so it was
discarded as an AC-4 candidate and used instead as a falsifiable prediction:
*if* the near-tie explanation is right, the two builds should first diverge at
step 9 and nowhere earlier.

```
PREDICTION: first divergence at step 9 (the only sub-0.5 gap, 0.050)
OBSERVED  : first divergence at step 9
```

Steps 0–8 match exactly, with gaps of 0.771–15.4. Two independent prompts, two
divergences, each landing on precisely that run's tightest tie and nowhere
else. (`artifacts/03-tiefree-step-compare.txt`.)

So the clause is not met as written, and the reason is now a measured property
of this model on this tree rather than a hypothesis: a 30-token greedy window
here reliably contains at least one sub-0.5-logit tie (the model opens every
response with a free-form `"We need to ..."` reasoning preamble, which is dense
in stylistic near-ties), and any two builds differing by ~0.4 in mean logit —
including two *correct* builds with different rounding — will split on it.

That makes byte-identical greedy generation the wrong cross-build parity
criterion on this tree, not a failing grade for this kernel. Deciding what
replaces it in the AC is the human's call. The obvious candidate, and what the
evidence above actually supports, is: **identical token selection at every step
whose top-2 logit gap exceeds the measured mean cross-build delta** — which
both prompts satisfy without exception. Prompt-shopping for a tie-free window
was deliberately not pursued past the one pre-registered attempt; it would be
outcome-shopping, and it would test less than the criterion above.

### Dispatch traffic — the blast radius

`rocprofv3 --kernel-trace`, same run shape, both builds
(`artifacts/03-dispatch-summary.txt`):

| build | `matmul_q8_0_f32_batch_wmma_4w_kernel` dispatches | all kernels |
|---|---|---|
| WMMA | **246** | 49,515 |
| NO_WMMA | **0** | 49,515 |

Not zero — the opposite. 246 dispatches, 5211 ms of device time (21.18 ms
mean), spread over all four GPUs (136 / 51 / 51 / 8), entirely inside the
7.8%–26.6% window of the run, i.e. prefill only, decode untouched. Every
dispatch reports `LDS_Block_Size 10240` = the rocwmma body's 4096 + 2048 + 4096
byte tiles, and `Grid_Size_Y 42` = `ceil(2633/64)` token tiles, so the kernel
that ran is unambiguously the new one. Grid X spans four distinct `out_dim`
values (1024 / 2048 / 4096 / 32768).

The 246-vs-0 contrast is also the proof that the `DS4_ROCM_NO_WMMA` guard is
semantically live, not just a `-D` that changes a binary's size.

**Blast radius: yes, and large.** The bug was live on this tree's default
`gfx1201` build for every non-quality-mode prefill — 4-GPU pipeline included,
not TP-specific. Any prefill throughput or output-quality number measured in
this repo outside `score_official` was computed with half of every Q8_0 block
missing from these matmuls. The issue's own caveat stands unchanged: the
`!g_quality_mode` gate at `rocm/ds4_rocm_matmul.cuh:405` kept the kernel out of
the fixture, so `avg_nll 0.3699` and the whole #32–#48 investigation are
untouched by this.

### Not done

No throughput A/B. The issue does not ask for one and this tree's topology
differs from issue 01's single-GPU `--ssd-streaming` setup, so its +14.5%
prefill figure does not carry over; a real number here would need its own
controlled sweep (issue 04's subject). No default build target was flipped —
`rocm` was already the WMMA path, and per
`docs/adr/0001-dense-tp4-parked-sequential-default.md` nothing was changed to
make the new kernel more reachable than the broken one already was.

## Comments

**Status `ready-for-human`.** Everything in scope is implemented, built, and
measured; `make test-rocm` passes 6/6 kernel comparisons plus the cross-device
and TP-refusal suites on the default multi-arch build. (`make test` was run and
does not complete on this host, for a pre-existing reason unrelated to this
change: `make: /usr/local/cuda/bin/nvcc: No such file or directory` /
`*** [Makefile:278: ds4_cuda.o] Error 127` — `CORE_OBJS` include a CUDA
translation unit regardless of backend, and this is a ROCm-only machine.)

The single open item is the byte-identical-generation clause of AC 4. It is
unmet, and the cause is characterized and replicated on two independent
prompts: each build's greedy path splits at that run's single sub-0.5-logit
tie and agrees at every other step. Recommended resolution — replace the
clause with "identical selection at every step whose top-2 logit gap exceeds
the measured mean cross-build delta", which the evidence satisfies without
exception and which loses nothing as a corruption detector. Re-running against
a tie-free prompt is the alternative, but one pre-registered attempt at
building such a prompt already failed its own screen (0.050 at step 9), and
chasing further would be outcome-shopping.

**Finding about the other tree, not acted on.** `~/src/ds4`'s `Makefile:191`
passes `ROCM_EXTRA_CFLAGS=-DDS4_ROCM_NO_WMMA` to its `rdna4` target, but nothing
in that Makefile ever consumes `ROCM_EXTRA_CFLAGS` (`ROCM_CFLAGS` at :58 does not
reference it), and its launch site in `rocm/ds4_rocm_matmul.cuh` has no
`DS4_ROCM_NO_WMMA` guard — so `make rdna4` there does not define the macro, and
a build that did would not compile. Whatever issue 01 A/B'd against, it was not
produced by that target. This does not affect issue 01's *conclusion* (its
argmax/gibberish contrast proves the two binaries it compared really did
differ), but it does affect the traceability of its build commands. Flagged
only — no edits were made to `~/src/ds4`, which is out of scope here. This
tree's `rocm-no-wmma` does not have the bug: `ROCM_EXTRA_CFLAGS` is wired into
`ROCM_CFLAGS`, verified in the build log and by the 246-vs-0 dispatch trace.

**Artifacts** under `.scratch/gfx1201-wmma-v2/artifacts/`:
`03-parity-run.sh` and `03-compare-logits.py` (the harness),
`03-parity-prompt.txt` and `03-tiefree-prompt.txt` (the two fixed prompts),
`03-logit-parity-summary.txt`, `03-dispatch-summary.txt`,
`03-logprob-step-compare.txt`, `03-tiefree-step-compare.txt`, and the per-run
generation/stderr logs. (`03-gen-wmma.txt`/`03-stderr-wmma.txt`, the
provenance header for the first prompt's WMMA logprobs run, were lost to an
over-broad prune; the equivalent header for the same run's NO_WMMA half
survives as `03-gen-nowmma.txt`, and the invocation is identical modulo the
binary.) The two
13 MB `rocprofv3` trace directories and three 1.6 MB full-vocab logit dumps
were distilled into those summaries and not committed; the scripts regenerate
them.

Filed 2026-08-11 from an interactive human session, after auditing why issue 01's
"+14.5% prefill" result did not appear in this tree. Root cause of the confusion:
issue 01 and 02 both ran entirely in `~/src/ds4` (single R9700, 86 GiB model,
`--ssd-streaming`); this repo never received the fix.
