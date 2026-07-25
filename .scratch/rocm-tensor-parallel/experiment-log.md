# ROCm tensor-parallel: experiment log

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
