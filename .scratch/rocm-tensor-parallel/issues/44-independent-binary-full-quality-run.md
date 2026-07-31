# 44 — Extend independent-binary validation to a full 100-case quality run

Status: closed

## Parent

`.scratch/rocm-tensor-parallel/issues/32-tp4-quality-fixture.md`
`.scratch/rocm-tensor-parallel/issues/43-pipeline-vram-accounting-regression.md`

## What to build

Issue #43 closed the pipeline avg_nll regression (0.374733 → ~1.7 on GPU) as
"a run-condition difference, not a code regression," but never found any
commit or configuration that reproduces ~0.37 on this hardware today. A
live-pair session (2026-07-31) found a strong counter-signal: an independent,
already-built binary at `~/src/ds4` (git origin `antirez/ds4` upstream,
commit `775ca6a`, dated 2026-07-13 — predates this branch's entire TP4 effort
and shares no commit history with it) reproduces near-reference quality on
this exact hardware right now:

- Single-GPU, SSD-streaming, `score_official` against the same
  `manifest.tsv` and same GGUF: case_000 avg_nll=0.417, case_001 avg_nll=0.237
  (both in-band with the 0.374733 reference; nowhere near the ~1.72 seen on
  `ds4-rebase`'s current pipeline/TP=4 paths).
- The same binary via its `docker-compose.yml` (true 4-GPU distributed
  layer-split pipeline, no TP): 5/5 fixture prompts (case_000–004) produced
  coherent, correctly on-topic completions via raw `/v1/completions` at
  temperature=0, semantically matching the recorded reference continuations,
  no VRAM errors across all 4 GPUs.

Only 2 of 100 manifest cases were scored before hitting a hardcoded VRAM
wall: under `g_ssd_streaming_mode`, `cuda_q8_f16_cache_reserve_bytes()` in
`rocm/ds4_rocm_runtime.cuh` (~line 3572) delegates to
`cuda_stream_resident_free_reserve_bytes()`, which reserved exactly 50% of
total VRAM (16.00 GiB of 31.86 GiB) for the Q8→f16 dequant cache — leaving no
room for the streaming expert cache to grow past a couple of cases.
`ds4_engine_options.ssd_streaming_cache_experts` does not affect this reserve
(confirmed: setting it to 1200 made no difference to the wall). This is a
budget-tuning problem in the independent binary, not a correctness bug — but
2/100 cases is a weak statistical anchor for a decision this consequential.

Get a full (or much larger, e.g. 20-30 case) run out of this binary so the
"is this hardware/environment healthy" signal is solid before betting a
bisect on it (issues #45/#46). Options, in rough order of effort:

1. Patch `cuda_stream_resident_free_reserve_bytes()` (or add an env-var
   override) in the local `~/src/ds4` checkout to reserve less — this is a
   throwaway/diagnostic checkout, not upstream code that needs to stay clean.
2. Reduce `ctx_size` or otherwise shrink the per-session working set so the
   existing 50% reserve leaves enough headroom.
3. Run the fixture in smaller batches (restart the process every N cases) if
   the wall is caused by monotonic cache growth across cases within one
   engine lifetime rather than a true one-shot budget shortfall — confirm
   which it is first, since this changes the fix.

## Acceptance criteria

- [x] At least 20 cases (ideally all 100) of the manifest scored on the
      independent `~/src/ds4` binary via `score_official`, same GGUF, same
      manifest.tsv as `ds4-rebase`'s fixture
- [x] avg_nll and first_match reported and compared against both the
      0.374733/65-per-100 original reference and the ~1.72/0-per-100 numbers
      currently seen on `ds4-rebase`
- [x] Root cause of the 2-case wall documented (one-shot budget shortfall vs.
      monotonic per-case growth) so issue #45/#46 know whether "run more
      cases in one process" is even valid methodology
- [x] Raw per-case TSV saved under `.scratch/rocm-tensor-parallel/quality-out/`
- [x] Any patch made to `~/src/ds4` is noted in this issue's Comments (that
      checkout is outside this repo; do not expect it to be committed here)

## Blocked by

None — can start immediately. GPU work; use the GPU lock protocol in
`AGENTS.md` (stops `dev-vllm` automatically).

## Comments

**Result: independent binary reproduces near-reference quality on this
hardware today, at full 100/100 cases.** This closes the "is the
hardware/environment healthy" question that #45/#46 depend on.

### Runs and provenance

Two TSVs under `.scratch/rocm-tensor-parallel/quality-out/`, both against
`~/src/ds4/gguf-tools/quality-testing/data/flash/manifest.tsv` (byte-identical
md5 to `ds4-rebase`'s own `gguf-tools/quality-testing/data/flash/manifest.tsv`)
and the same GGUF
(`/var/cache/llama/ds4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`):

- `q_independent_binary_100case.tsv` — **merged, two-segment** run: cases
  000-058 from the first backgrounded process (which died at case_058 when
  its parent agent session was torn down, not from any wall), cases 059-099
  from a second process resuming from case_059, detached via `setsid`.
- `q_independent_binary_100case_singleproc.tsv` — **single unbroken
  process**, all 100 cases, run specifically to settle the "does restarting
  the process matter" methodology question for #46.

The two files are **byte-identical** (`diff -q` confirms), and spot-checked
independently against the raw stdout logs at the process boundary
(`case_059`, `case_099` match exactly between `/tmp/q_independent_100_part2.log`
and `/tmp/q_independent_100_singleproc.log`). Both used
`DS4_ROCM_FREE_RESERVE_MB=2048`; neither used `AMD_SERIALIZE_KERNEL` (this is
a single-GPU SSD-streaming run, so the multi-GPU cross-device kernel race
that motivates serialization in the `ds4-rebase` pipeline/TP4 reference
doesn't apply here — noted as a configuration asymmetry, not a like-for-like
run).

### Scores (token-weighted avg_nll = sum(nll)/sum(target_tokens), the
convention the 0.374733/0.370-0.378 references use; unweighted mean of
per-case avg_nll given alongside since it reads differently)

| Run | avg_nll (weighted) | avg_nll (unweighted mean) | first_match |
|---|---|---|---|
| Independent binary, 100/100 cases | **0.370493** | 0.403377 | **64/100** |
| Original reference (`q_pipeline_ref_tp4issue32.tsv`) | 0.374733 | 0.406944 | 65/100 |
| `ds4-rebase` current pipeline (`q_pipeline_quality.tsv`) | 1.727047 | 1.848420 | 0/100 |
| `ds4-rebase` current TP=4 (`q_tp4_option_b.tsv`) | 1.719620 | 1.836588 | 0/100 |

The independent binary's weighted avg_nll (0.370493) is inside the PRD's
0.370-0.378 band and within ~1.1% of the 0.374733 anchor, with first_match
64/100 vs the anchor's 65/100 — noise-level difference. This is ~4.6x better
than the ~1.72 currently seen on `ds4-rebase`'s own pipeline and TP=4 paths.
Caveat: the independent-binary run is single-GPU SSD-streaming, not 4-GPU
pipeline/TP — it answers "is this hardware/model/manifest capable of 0.37
today" (yes), not "is the 4-GPU pipeline path itself healthy" (still open,
per #45/#46).

### Root cause of the 2-case wall: one-shot fixed reserve, not monotonic growth

At the default reserve, the run reproducibly died at `case_002` with the
streaming expert cache already at `cached=0.00 GiB` — i.e. free VRAM itself
had dipped under the fixed 16 GiB (`cuda_stream_resident_free_reserve_bytes()`,
50% of 31.86 GiB total) reserve threshold *before* the streaming cache had
grown at all. That rules out runaway/monotonic per-case cache growth as the
mechanism — an empty cache can't be "too big."

Lowering the reserve via the existing `DS4_ROCM_FREE_RESERVE_MB=2048` env var
override (already present in upstream commit `775ca6a`, not added by this
session) cleared the wall immediately: 5/5 cases at reserve=2048MB, then
confirmed at 100/100 cases in both the merged-segments run and the single
unbroken process. **Verdict for #45/#46: "run more cases in one process" is
valid methodology, provided `DS4_ROCM_FREE_RESERVE_MB` is set low enough
(2048 tested clean) — no per-case state accumulates that would make a longer
single-process run less reliable than a freshly restarted one.**

### Patch made to `~/src/ds4` (outside this repo, not committed here)

Working-tree-only, uncommitted change to
`gguf-tools/quality-testing/score_official.c` (in the `~/src/ds4` checkout,
commit `775ca6a`, `pr-558` branch): added `.ssd_streaming = true,
.ssd_streaming_cache_experts = 1200` to the engine options struct, to enable
SSD-streaming mode for the scoring binary. `DS4_ROCM_FREE_RESERVE_MB=2048`
was passed as a runtime env var, not a code patch — the override already
existed in `rocm/ds4_rocm_runtime.cuh` from upstream commit `775ca6a`.

### Verification

No `ds4-rebase` source was touched by this issue (`git status --short` shows
no tracked-file changes; `git diff --stat HEAD` is empty for `.c`/`.cuh`/`.h`/
`Makefile`). The deliverable is data (two TSVs) plus this writeup, so
`ds4-rebase`'s build/test/lint suite is not applicable here — nothing to
compile or lint.

GPU lock: released (`dev-vllm` restarted successfully). The stale lock this
session inherited was from PID `1864420` (a prior session's crashed/detached
process); confirmed dead before releasing.
