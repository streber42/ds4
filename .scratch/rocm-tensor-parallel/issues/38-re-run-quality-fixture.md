# 38 — Re-run quality fixture and close issue #32

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/issues/32-tp4-quality-fixture.md`

## What to build

Run the full 100-case quality fixture on the TP=4 build (after the prefill fix from issue #37 lands) and verify all acceptance criteria from issue #32 are met. If scores pass, close issue #32. If they fail, identify specific failing cases and determine whether the remaining divergence is floating-point reassociation (acceptable) or a sharding/arithmetic bug (must fix).

**Setup:** `AMD_SERIALIZE_KERNEL=3` to avoid the compressor prefill race (issue #23), production 81 GiB model, 4× R9700 GPUs, `make rocm-quality` build.

**Command:**
```bash
make -j8 rocm-quality
AMD_SERIALIZE_KERNEL=3 \
  ./gguf-tools/quality-testing/score_official \
    /home/murphy/src/ds4/ds4flash.gguf \
    gguf-tools/quality-testing/data/flash/manifest.tsv \
    .scratch/rocm-tensor-parallel/quality-out/q_tp4_final.tsv \
    4096 --gpu-devices 0,1,2,3 --cuda-tensor-parallel
```

**Reference baseline:** `q_pipeline_ref_tp4issue32.tsv` (avg_nll 0.374733, first_match 65/100) from the 2026-07-27 pipeline serialized run. If the build has changed materially since then, re-run the pipeline reference first.

**Acceptable range:** avg_nll within ±1% of the pipeline serialized reference (0.370–0.378). First-token match ≥ 60/100. api_top1_rate ≥ 0.85. api_pair_rate ≥ 0.98.

**If scores pass:** Change `Status: ready-for-human` to `Status: closed` in issue #32, add a closing comment with the final comparison table against the pipeline reference, and commit with message `feat(rocm-tensor-parallel): 32 — TP=4 quality fixture (authoritative correctness gate)`.

**If scores still fail:** For each failing metric, list the specific cases that are outside tolerance. For any case with avg_nll > 1.0, dump the per-layer hidden states using the diagnostic framework from issue #35 and identify the remaining divergence point. Determine whether it's floating-point reassociation (acceptable — document and justify the deviation from the ±1% target) or a real bug (create a new follow-up issue).

## Acceptance criteria

- [ ] Quality fixture runs to completion on TP=4 (100 cases, 2289 tokens)
- [ ] avg_nll within ±1% of pipeline serialized reference (0.373815) — or if outside, documented with root cause analysis
- [ ] first_match ≥ 60/100 — or if below, documented with root cause analysis
- [ ] api_top1_rate ≥ 0.85 — or if below, documented with root cause analysis
- [ ] api_pair_rate ≥ 0.98 — or if below, documented with root cause analysis
- [ ] Results recorded in experiment log with comparison table against pipeline reference
- [ ] Raw per-case TSV saved in `.scratch/rocm-tensor-parallel/quality-out/`
- [ ] Issue #32 status updated (closed if passing, or `ready-for-human` with detailed failure analysis if not)

## Blocked by

- `#37 — Fix the identified prefill divergence`
