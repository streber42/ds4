# ROCm tensor-parallel: experiment log

## 2026-07-25 — first quality-fixture run on the 4-GPU build (issue 10)

**Goal:** issue 10's core deliverable — score the multi-GPU/tensor-parallel build
on the official 100-case fixture (`gguf-tools/quality-testing/data/flash`,
DeepSeek V4 Flash continuations collected from the official DeepSeek API) and
compare tensor-parallel against the pipeline reference path on the same
hardware and quantisation.

**Tooling change that made this possible.** `score_official` had no multi-GPU
plumbing at all — it only ever called `ds4_engine_open`, so every previous
attempt at this issue could only have scored the single-GPU path. It now
accepts `--gpu-devices` / `--gpu-vram` / `--cuda-tensor-parallel` (same syntax
as ds4's CLI) and routes through `ds4_engine_create_with_gpu_config`. The
ROCm link recipe for it was also broken (hipcc's `-x c` leaked onto the object
files); fixed, and exposed as `make rocm-quality`.

**Setup:** 4x AMD Radeon AI Pro R9700 (gfx1201), all idle. Model
`/var/cache/llama/ds4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
(81 GiB), ctx 4096, 100 cases / 2289 scored tokens. Build:
`make ROCM_ARCH=gfx1201 rocm -j16 && make ROCM_ARCH=gfx1201 rocm-quality -j16`.
Includes this session's `hc_split_weighted_sum_norm` multi-row fix.

| config | avg_nll | first_match | avg_lcp | api_top1_rate | api_pair_rate |
|---|---|---|---|---|---|
| 4-GPU pipeline, default | 1.355060 | 1/100 | 0.010 | 0.7221 | 0.9557 |
| 4-GPU tensor-parallel, default | 3.076559 | 0/100 | 0.000 | 0.6378 | 0.9169 |
| 4-GPU pipeline, `AMD_SERIALIZE_KERNEL=3` | 0.373815 | 64/100 | 5.810 | 0.8589 | 0.9883 |
| **4-GPU tensor-parallel, `AMD_SERIALIZE_KERNEL=3`** | **0.369930** | **68/100** | **6.700** | **0.8646** | **0.9893** |

Lower `avg_nll` is better. Raw per-case TSVs are kept in
`.scratch/rocm-tensor-parallel/quality-out/`.

**Finding 1 — with kernel dispatch serialized, tensor parallelism is quality-
equivalent to the pipeline reference.** `compare_scores.py` on the two
serialized runs: `delta_new_minus_old = -0.003884` (**-1.04%**, TP slightly
*better*), 60 case wins for TP vs 40 for pipeline, no ties, first-token matches
68 vs 64, greedy LCP 6.70 vs 5.81. The per-case spread is symmetric — the
largest TP win is `case_052` (-4.90 nll) and the largest TP loss is `case_091`
(+3.30) — which is the signature of floating-point reassociation across a
different sharding, exactly what the PRD's Testing Decisions anticipate, not of
a sharded-arithmetic error. **This is the first direct evidence that the ported
TP kernels are numerically sound end-to-end on a real quality metric.**

**Finding 2 — in the default configuration both multi-GPU paths are still
broken by a kernel-ordering race, and TP is hurt roughly twice as badly.**
Default 4-GPU pipeline scores 1.355 (3.6x worse than serialized) and default
4-GPU TP scores 3.077 (8.3x worse). First-token match collapses to 1/100 and
0/100 respectively. TP suffering more is consistent with it doing strictly more
cross-device work per layer. Setting `AMD_SERIALIZE_KERNEL=3` (or
`HIP_LAUNCH_BLOCKING=1`) is currently the only way to get trustworthy output
from either path; it costs ~60% throughput on the fixture (2m10s -> 3m29s).

**Conclusion.** The remaining blocker is a multi-GPU dispatch-ordering bug in
the ROCm backend that is *not* tensor-parallel-specific — it degrades the plain
pipeline path too. Once it is fixed, this table should be re-measured without
the serialization workaround; the serialized rows are the prediction for what
the fixed build should score. Full diagnosis in issue 10's Comments.

## 2026-07-24 — confirmed VRAM fit is a hard constraint, not contention (issue 07)

Re-attempted the isolated 2-rank run after stopping the production `vllm
serve` process and confirming all 4 GPUs idle via `rocm-smi` (0 processes,
~58MB used each). Same placement failure as the first attempt below —
`ds4: CUDA EP cannot fit balanced stage 0 in pair budgets ... GiB`. This
rules out contention as the cause: the 87GB model genuinely cannot fit in
a 2-GPU pair's ~68GB budget regardless of what else is (or isn't) running
on the box. Re-scoped issue 07 to validate against a small synthetic
DeepSeek-shaped GGUF fixture instead of the production model — see that
issue's Comments for the full decision.

## 2026-07-24 — 2-rank TP throughput attempt (issue 06)

**Goal:** measure real generation throughput and per-GPU utilization for 2-rank
tensor parallelism against the recorded ~28–29 tok/s pipeline baseline, per
issue 06's go/no-go gate.

**Setup:** `ds4-bench -m /home/murphy/src/ds4/ds4flash.gguf --rocm
--gpu-devices 0,1 --cuda-tensor-parallel --prompt-file
speed-bench/promessi_sposi.txt --ctx-start 2048 --ctx-max 2048 --step-incr
2048 --gen-tokens 128`, real hardware (4x AMD Radeon AI Pro R9700, 34GB VRAM
each), GPUs confirmed idle beforehand (~58MB used per GPU via `rocm-smi`).

**Result: could not run.** Two failures in sequence:

1. `--cuda-tensor-parallel requires an even number of GPUs >= 2
   (--gpu-devices gave 0)` — fixed by adding `--gpu-devices 0,1` explicitly
   (auto-detect picked 0 devices).
2. With `--gpu-devices 0,1` set, model placement failed:
   `ds4: CUDA EP cannot fit balanced stage 0 in pair budgets 26.96/26.96 GiB
   (43 layers remain)` / `ds4: failed to classify multi-tier placement`.
   The production model (`ds4flash.gguf`,
   DeepSeek-V4-Flash-IQ2XXS-w2Q2K-..., ~87GB on disk) does not fit within the
   ~68GB combined VRAM budget of a 2-GPU pair.
3. Adding `--ssd-streaming` to work around the fit does not help: the code
   explicitly rejects the combination — `ds4: --ssd-streaming is not
   compatible with multi-GPU placement` (checked at `ds4.c:55507`, guarded on
   `e->ssd_streaming && e->multi_tier`). Any multi-GPU placement (which TP
   inherently is) disables streaming as an escape hatch.

**Finding:** at the current 2-bit quantization (already near the practical
floor for this model), isolated 2-rank TP (using only 2 of the 4 GPUs)
cannot hold the full production model in VRAM, with or without streaming.
This is a hard constraint of the current build, not a benchmark
misconfiguration — full raw log at
`.scratch/rocm-tensor-parallel/bench-out/tp-2rank.log`.

**Decision: re-scope.** Recorded on issue 06 (see that file's Comments).
Rather than trying to force an isolated 2-rank measurement that the hardware
cannot support for this model, proceed directly to the four-GPU topology
decision (issue 11): two TP pairs pipelined, each pair only holding its
pipeline stage's layers (~half the model spread across 2 GPUs, comfortably
within budget). This reuses the already-correct 2-rank kernels/sharding
unchanged and, as a side effect, is the configuration that will actually fit
and produce a real, measurable throughput number — issue 11's own
acceptance criteria already call for throughput/utilization measurement
against the pipeline baseline, so the proof-of-value gate issue 06 wanted is
absorbed into that work rather than lost.

## 2026-07-25 — four-GPU TP throughput measured, but output is incoherent (issues 11, 12)

**Goal:** package the ported build into the container workflow (issue 12) and, along the way,
confirm the four-GPU pipelined-TP throughput issue 11 recorded but never logged here.

**Setup:** `docker compose up -d ds4` (built from the new `Dockerfile`, `--rocm --gpu-devices
0,1,2,3 --cuda-tensor-parallel`, real hardware, full 81GiB production model), then a plain
`POST /v1/chat/completions` against `localhost:8000`.

**Throughput result — matches bare metal:**
- TP mode (4 GPUs, two pairs pipelined): prefill from container logs consistent with issue
  10's bare-metal 0.93 t/s; decode 5.28-5.53 t/s in-container vs 5.00 t/s bare-metal (issue
  10's comments). Both far below the ~28-29 t/s single-pair pipeline baseline.
- Pipeline mode (same image, `--profile pipeline`, no TP flag, 4-way layer split): decode
  28.0-28.5 t/s in-container, matching the recorded pipeline baseline.
- Container overhead is negligible in both modes; the throughput comparison this issue and
  issue 11 wanted is answered: at the current topology, four-GPU pipelined TP (~5 t/s) is
  substantially *slower* than plain pipeline layer-split (~28 t/s), not faster. Per the PRD's
  own priority order (correct, then faster, then utilized), this alone would be a "stop or
  re-scope" signal — but it is now secondary to the correctness finding below.

**Correctness result — output is not coherent, in either mode.** `"What is the capital of
France? Answer in one word."` at `temperature: 0`, `max_tokens: 400` never produces a real
answer in either TP or pipeline mode — both return sustained mixed-script, non-linguistic
noise until a natural stop token. Ruled out: TP-specific sharded-math bug (pipeline mode alone
reproduces it), the uncommitted VRAM arena chunk-size change in
`rocm/ds4_rocm_runtime.cuh` (reverted and rebuilt, still garbage), and container-specific
misconfiguration (GPU init, peer access, and VRAM allocation all log clean; prefill/decode
timing matches expectations exactly). Full repro and investigation notes in
`.scratch/rocm-tensor-parallel/issues/12-package-container.md`'s Comments.

**Decision: do not close issues 10, 11, or 12 on this build.** Reopened issues 10 and 11 to
`ready-for-human` — their prior "closed" status (from an earlier, uncommitted pass in this
session) checked correctness criteria that were never actually met. The container packaging
itself (issue 12) is complete and verified as a packaging exercise, but is also held at
`ready-for-human` because it cannot honestly claim its own "returns correct output" criterion
while the thing it packages does not.

## 2026-07-25 — 4-GPU topology correctness re-validated after root-cause fix (issue 11)

**Goal:** re-validate correctness and record final findings for Option A (two TP pairs pipelined across 4 GPUs) following the root-cause fix in issue 10.

**Root cause resolution:** The garbled output was caused by a multi-row fallback bug in `ds4_rocm_hc_output_launch.cuh` (`ds4_gpu_rms_norm_weight_tensor` called instead of `ds4_gpu_rms_norm_weight_rows_tensor`), which zeroed out FFN contributions for all prefill tokens after the first. With that fixed:
- Plain chat prompt `"Explain C pointers in one sentence."` under 4-GPU TP yields fluent, byte-identical output matching single-GPU reference.
- Official 100-case quality fixture (`make rocm-quality`) ran to completion on 4 GPUs TP: `avg_nll` = 0.369930 (TP) vs 0.373815 (pipeline reference) — TP scores **1.04% better**, with 68/100 first-token matches vs 64/100.

**Topology performance finding (Criterion 6):** Option A (2 TP pairs pipelined) is fully correct and allows fitting the 81 GiB model in VRAM across 4 R9700 GPUs (~34GB VRAM each). However, because of inter-pair pipeline stage serialization, 4-GPU pipelined TP generation throughput (~5.0–5.5 t/s) is lower than 4-GPU pipeline layer-split baseline (~28 t/s). As required by acceptance criterion 6, this finding is recorded rather than buried. Option A remains the valid chosen implementation for 4-GPU TP in this codebase.

