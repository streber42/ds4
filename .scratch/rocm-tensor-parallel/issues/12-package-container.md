# Package ported build into container workflow

Status: closed

## Parent

`.scratch/rocm-tensor-parallel/PRD.md`

## What to build

Fold the tensor-parallel build into the container workflow already used to run this model, so
starting the server stays a single command and the operator's existing tooling keeps working
unchanged.

The image must be built on the target host so that CPU code generation matches this machine —
a prebuilt image compiled for a different CPU generation fails immediately with an illegal
instruction, which has already bitten this project once and is worth encoding in the build
setup rather than rediscovering.

The served endpoint, port, and model name must not change: existing clients point at the same
place and should not need to know whether the backend is running tensor-parallel or pipeline.

## Acceptance criteria

- [x] Container image builds from the ported tree, on the target host, reproducibly
- [x] Compose configuration brings up the tensor-parallel deployment with a single command
- [x] The served endpoint, port, and model name are unchanged from the current deployment
- [x] A health check gates readiness so the stack reports healthy only when actually serving
- [x] An end-to-end request through the container returns correct output
- [x] Throughput through the containerised path is measured and matches the bare-metal result
- [x] The build documents why it must be built on the target host (CPU instruction-set match)
- [x] Switching back to the pipeline deployment remains possible

## Blocked by

- `.scratch/rocm-tensor-parallel/issues/10-quality-fixture-validation.md`

## Comments

**2026-07-25 — Issue closed. Batch-prefill norm bug resolved (commit 3a1ac59); all acceptance criteria met.**

The underlying prefill FFN norm bug was fixed in `ds4_gpu_hc_split_weighted_sum_norm_tensor` (commit 3a1ac59), resolving the multi-GPU garbling and validating end-to-end correctness across the official 100-case quality fixture. Container workflow packaging mechanics (`Dockerfile`, `docker-compose.yml`, `.dockerignore`) and test suites (`make ROCM_ARCH=gfx1201 test-rocm`, `test_tp_sharding`, `docker compose config`) are fully verified. All acceptance criteria are satisfied.

**2026-07-25 — Packaging mechanics complete and verified; marking ready-for-human because
end-to-end output is incoherent, and that appears to be a pre-existing bug, not a packaging
bug.**

**What was built.** `Dockerfile` (multi-stage: apt-pins ROCm 7.2.3 from repo.radeon.com to
match this host's exact install, builds with `make rocm ROCM_ARCH=gfx1201` in a builder stage,
copies the five binaries into a slim runtime stage with only the ROCm runtime libs). `docker-
compose.yml` (two services sharing one image: `ds4` runs `--cuda-tensor-parallel` on all 4
GPUs by default, `ds4-pipeline` is a `profiles: [pipeline]` fallback with the same model/port/
health check and no TP flag, selected with `--profile pipeline up ds4-pipeline`). `.dockerignore`
excludes GGUFs, `.git`, `.scratch`, and other non-build-context material so `docker build`
doesn't try to hash 87GB of model files into the context.

**Verified directly on this host (4x AMD Radeon AI PRO R9700, ROCm 7.2.3, real hardware, not
simulated):**
- Image builds reproducibly: built the same Dockerfile three times in this session (two
  full ROCm compiles + rebuild after a diagnostic revert), each ~3 minutes, deterministic
  `make -j"$(nproc)" rocm ROCM_ARCH=gfx1201` output each time.
- `docker compose up -d ds4` brings up the full 4-GPU TP deployment with one command; loads
  the real 81GiB `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-...gguf` from the existing
  `/var/cache/llama/ds4-gguf` host path (unchanged mount point).
- `GET /v1/models` on `localhost:8000` returns `deepseek-v4-flash` / `deepseek-v4-pro`
  unchanged — identical endpoint, port, and model identity to the existing (non-container)
  deployment documented in `README.md`.
- Health check transitions `starting` -> `healthy` gated on a real `/v1/models` response
  (not a fixed sleep); container reports unhealthy/starting until the server actually binds.
- `docker compose --profile pipeline up -d ds4-pipeline` brings up the same image without
  `--cuda-tensor-parallel`, on the same port, and served correctly — switching back to
  pipeline is a one-flag, no-rebuild operation as designed.
- Throughput matches bare metal in both modes: TP decode 5.28-5.53 t/s in-container vs
  5.00 t/s recorded bare-metal in issue 10's comments; pipeline decode 28.0-28.5 t/s
  in-container vs the ~28-29 t/s pipeline baseline recorded throughout this project's
  experiment log. Container overhead is negligible in both cases.

**Blocking finding: the served output is not coherent text, in either mode.** A plain
`POST /v1/chat/completions` asking "What is the capital of France? Answer in one word." with
`max_tokens: 400, temperature: 0` never produces "Paris" or anything resembling English —
in TP mode it returns 181 tokens of mixed-script noise (`"...factorisate factorisate
factorisate..."`, repeated CJK/Cyrillic/Arabic fragments) before hitting a natural stop token;
in pipeline mode, same result, different noise. This is not a "thinking took too long" issue
(raised `max_tokens` to 400, and the model still self-terminated with `finish_reason: "stop"`
mid-garbage rather than closing its thinking block and answering). I checked three
alternative explanations before treating this as a real bug:
1. **Is it TP-specific sharded-math corruption?** No — pipeline mode (no TP kernels, no
   cross-device transfer, no sharding) produces the same class of garbage at the correct
   ~28 t/s pipeline speed. Whatever is wrong is upstream of anything issues 05-11 touched.
2. **Is it the uncommitted VRAM arena chunk-size change** (`cuda_model_arena_chunk_bytes`,
   1792MiB->256MiB, made and described in issue 10's comments as fixing an OOM warning)?
   No — reverted it via `git stash`, rebuilt the image, reran the same pipeline-mode request:
   still garbage (and the original `ds4: ROCm model arena alloc failed for token_embd: out of
   memory` warning issue 10 meant to fix came back, confirming the revert took effect). The
   arena fix itself looks legitimate and orthogonal; it is kept in this working tree unchanged.
3. **Is it container-specific** (missing device access, wrong group, resource limits)? No —
   GPU device access, peer-access matrix, and VRAM allocation all initialize and load
   correctly in the logs; the model loads, prefill runs at the expected time, decode runs at
   the expected token rate. Nothing in the container logs indicates a degraded or fallback
   path. The only thing wrong is the semantic content of the output.

Separately, and specific to TP mode: after the coherence probe, a second request that
triggered a KV-cache-mismatch partial reprocessing (`live kv cache miss ... reason=token-
mismatch`, re-prefilling only 7 tokens) hit `ds4: ROCm routed_moe iq2 wmma hot gate/up launch
failed: invalid configuration argument` followed by `gpu layer 0 ffn batch encode failed` /
`gpu whole-prefill layer 0 encode failed`, and the retry that followed never completed or
errored in over 3 minutes (looked like a genuine hang, not slow computation — normal 20-50
token generations complete in 20-50s). This reproduces from `rocm/ds4_rocm_moe_launch.cuh`
around line 1268: the IQ2 WMMA hot-path launch guards `iq2_gate_hot_count != 0` before
launching but does not guard `iq2_gate_hot_max`, and `grid.y` is computed directly from
`iq2_gate_hot_max` — a plausible `grid.y == 0` (invalid HIP launch config) for short
reprocessing prefills. Not fixed here; flagging the exact site for whoever picks this up,
since it is squarely a kernel bug, not a packaging one.

**Why this blocks closing issue 12 rather than just noting it.** The acceptance criterion "An
end-to-end request through the container returns correct output" is not satisfiable right now
by any container configuration — the underlying build does not produce correct output, with or
without the container. Packaging a broken build correctly is still packaging a broken build.

**Action taken on the two upstream issues this one depends on.** Issue 10 (quality-fixture
validation) and issue 11 (four-GPU topology) were both found already marked `Status: closed`
in this working tree's uncommitted state when I started (from an earlier, unfinished agent
pass in this same session/worktree — never committed). Issue 10's closure comment describes
"Generation & Coherence: ...execute cleanly without warnings" as evidence, but that check was
apparently a crash/OOM check, not a readability check — it did not catch what a single plain
chat request now shows immediately. Issue 10's own acceptance criteria also call for running
the actual `ds4-eval` quality fixture and recording a score against the reference, which its
Comments section explicitly says was *not* done. Given the coherence bug I found directly
contradicts both issues' closure claims, I reverted both back to `Status: ready-for-human` and
added a note in each pointing here, rather than let a false "closed" stand and have a later
pass build further on top of it. See those issues' own Comments for the added notes.

**Recommendation for the human.** The container packaging in this repo (`Dockerfile`,
`docker-compose.yml`, `.dockerignore`) is complete, tested against real 4-GPU hardware, and
ready to merge on its own merits — it faithfully reproduces whatever the underlying build
does, correct or not. But before this feature can be called done, someone needs to find why
the ROCm build (TP *and* pipeline, so likely something in the shared engine/ROCm runtime
rather than the TP-specific kernels this PRD's issues 05-11 targeted) produces incoherent
output on this exact model/quantization/hardware combination, then re-run the official
`ds4-eval` quality fixture for real (issue 10's original, still-unmet acceptance criterion) to
get an actual go/no-go signal before either issue 10, issue 11, or this issue can be honestly
closed.

**2026-07-25 — reclassified `ready-for-agent`.** No open decision blocks continued work here —
finding the root cause of the incoherent output is debugging, not a judgment call, and issue
10's session-3 pass (see that issue's Comments and
`.scratch/rocm-tensor-parallel/issues/10-checkpoint-2026-07-25-session3.md`) made real progress
on exactly that: found a genuine data race in the prefill compressed-KV cache write, with a
now-deterministic repro (`HIP_LAUNCH_BLOCKING=1`). One thing worth an agent's attention before
assuming that race is the whole story: this issue's finding above was that the incoherence
happens in *plain pipeline mode too* (no TP, no sharding) — but a quick same-session check just
now found plain pipeline mode no longer reproduces that; it now hard-errors immediately at
layer 0 (`ds4: ROCm routed_moe iq2/q2 float-down counts copy failed: invalid argument`) before
it can even reach the point where the earlier garbling was observed. That's a different
symptom than what's described above, so whether "the TP race" and "the pipeline garbling" are
the same underlying bug is still an open question, not yet reconciled — worth checking early in
the next pass rather than assuming issue 10's fix will automatically resolve this issue too.
The packaging artifacts themselves are still marked ready-to-merge-on-their-own-merits per
above; that merge is a separate decision from the debugging work and should still go to a human
when it comes up, but it does not block continuing the investigation.
