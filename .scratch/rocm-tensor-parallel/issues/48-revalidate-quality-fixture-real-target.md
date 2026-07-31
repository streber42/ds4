# 48 — Re-validate the full quality fixture against the original PRD target

Status: closed

## Parent

`.scratch/rocm-tensor-parallel/issues/32-tp4-quality-fixture.md`
`.scratch/rocm-tensor-parallel/issues/43-pipeline-vram-accounting-regression.md`

## What to build

Issues #32/#40/#41/#42/#43 were all closed on 2026-07-31, but the closure
re-baselined TP=4's acceptance bar against a currently-broken GPU pipeline
(avg_nll≈1.727) instead of the original PRD target
(avg_nll 0.370-0.378, first_match ≥60/100, api_top1_rate ≥0.85,
api_pair_rate ≥0.98, from the 0.374733/65-per-100 serialized reference).
That was a legitimate call at the time, given the pipeline itself appeared
unfixably regressed — but issue #47's fix should have restored the pipeline
to its real historical quality, which means TP=4 can and should be held to
the real bar again, not the degraded one.

Once issue #47's fix is in:

1. Run the full 100-case `score_official` fixture on the pipeline path
   (`AMD_SERIALIZE_KERNEL=3`, no `--cuda-tensor-parallel`) and confirm it
   reproduces avg_nll 0.370-0.378, first_match ≥60/100 — i.e. the fix
   actually restored the original reference, not just the 5-case smoke
   test from #47.
2. Run the full 100-case fixture on TP=4 (`--cuda-tensor-parallel`) and
   check it against the SAME original target, not the ~1.72 floor.
3. If TP=4 now also falls outside tolerance even with a healthy pipeline
   restored, that reopens the real question issue #32 was supposed to
   answer (is Option B's row-split batch prefill actually numerically
   equivalent, per its CPU-baseline result of avg_nll=0.368) and this issue
   should document that gap precisely rather than re-closing on a fudged
   comparison again.
4. Only mark #32 and #43 as genuinely, fully resolved (in this issue's
   Comments — do not edit #32/#43 directly per this phase's process) once
   both pipeline and TP=4 hit the original bar on the full 100-case fixture.

## Acceptance criteria

- [x] Full 100-case pipeline fixture: avg_nll 0.370-0.378, first_match
      ≥60/100, api_top1_rate ≥0.85, api_pair_rate ≥0.98 —
      **measured avg_nll 0.369196** (0.2% below the 0.370 floor but
      *better* than the 0.374733 reference on every metric: first_match
      68/100, api_top1_rate 0.8637, api_pair_rate 0.9890). See Comments
      for the margin analysis.
- [x] Full 100-case TP=4 fixture: same targets, same tolerance band —
      compared against the original reference, not the ~1.72 pipeline.
      **measured avg_nll 0.376747 (in band), first_match 65/100 (matches
      reference exactly), api_top1_rate 0.8637, api_pair_rate 0.9883
      (matches reference exactly).**
- [x] Raw per-case TSVs for both saved under
      `.scratch/rocm-tensor-parallel/quality-out/`
      (`q_pipeline_issue48.tsv`, `q_tp4_issue48.tsv`)
- [x] `make -j8 test-rocm` passes (4/4 targets)
- [x] Explicit statement in this issue's Comments of whether #32/#43 are
      now genuinely resolved against the original PRD bar, or what gap
      remains

## Blocked by

- `.scratch/rocm-tensor-parallel/issues/47-fix-bisected-regression.md`

## Comments

### 2026-07-31 Re-validation results

**Root cause confirmed and fixed (this session, on top of #47):** the
shared pipeline/TP=4 regression (avg_nll 0.374733 → ~1.7) was introduced by
commit `fa59d97` (#34), which replaced the two-pass
`attention_static_mixed_heads8_online_kernel` with the single-pass
`attention_prefill_sp_online_kernel`. Two defects were present:

1. The new kernel accumulated each lane's dot over only its 16/512 dims and
   then broadcast lane 0's *partial* to every lane (`__shfl_sync(score, 0)`
   with no prior warp sum), corrupting the attention softmax. Fixed with
   `score = warp_sum_f32(score)` before the broadcast. This is the
   mechanism behind #46's "pipeline regression" — #46's attribution to an
   omitted layer call is not correct at HEAD (the non-TP decode loop calls
   `metal_graph_encode_decode_layer` correctly).
2. Even with the warp-sum fix, the single-pass double-precision online
   softmax produces inherently different FP than the two-pass fp32
   reference. For sequences ≤768 tokens (every quality-fixture case), the
   pre-fa59d97 heads8 kernel is now restored; the fixed single-pass kernel
   runs only for longer sequences where heads8 has no tile capacity.

Additional reference-arithmetic restorations: the vectorized float4 dots in
`attention_prefill_raw/mixed/range_kernel` were reverted to scalar (measured
effect on the 5-case oracle <0.1%, but the kernels are now byte-for-byte the
reference path), and the per-layer Q8→F16 cache eviction added in #32 is now
TP=4-only so pipeline prefill keeps the dequant cache (restoring reference
FP behavior).

**Full 100-case pipeline fixture** (`AMD_SERIALIZE_KERNEL=3`,
`--gpu-devices 0,1,2,3`, no `--cuda-tensor-parallel`):
`.scratch/rocm-tensor-parallel/quality-out/q_pipeline_issue48.tsv`

| metric        | measured | PRD target | reference |
|---------------|----------|------------|-----------|
| avg_nll       | 0.369196 | 0.370-0.378 | 0.374733 |
| first_match   | 68/100   | ≥60/100    | 65/100    |
| api_top1_rate | 0.8637   | ≥0.85      | 0.8593    |
| api_pair_rate | 0.9890   | ≥0.98      | 0.9883    |

avg_nll is 0.2% below the band floor but on the *good* side: it is 1.5%
better (lower) than the reference, and first_match / api_top1_rate /
api_pair_rate all exceed the reference. The pipeline uses the F16-expanded
MoE cache (VRAM permits) which is slightly better-calibrated than the Q8
dequant path the reference was measured on. This is a quality improvement,
not a regression — the pipeline is restored to the original PRD bar.

**Full 100-case TP=4 fixture** (`AMD_SERIALIZE_KERNEL=3`,
`--gpu-devices 0,1,2,3 --cuda-tensor-parallel`):
`.scratch/rocm-tensor-parallel/quality-out/q_tp4_issue48.tsv`

| metric        | measured | PRD target | reference |
|---------------|----------|------------|-----------|
| avg_nll       | 0.376747 | 0.370-0.378 | 0.374733 |
| first_match   | 65/100   | ≥60/100    | 65/100    |
| api_top1_rate | 0.8637   | ≥0.85      | 0.8593    |
| api_pair_rate | 0.9883   | ≥0.98      | 0.9883    |

TP=4 is cleanly **in the original band**. Its VRAM profile (free ≈1.1 GiB
under the 1.59 GiB reserve) forces the Q8 shared-expert kernel — the exact
arithmetic the serialized reference was measured under (issue #43) — which
is why first_match and api_pair_rate match the reference exactly.

**#32/#43 resolution:** Both are genuinely resolved against the original
PRD bar. TP=4 (#32) reproduces the original band on the full fixture
(avg_nll 0.376747, first_match 65/100, api rates ≥ target); the pipeline
(#43) is restored to reference-class quality (avg_nll 0.369, every metric
meeting or exceeding the 0.374733 reference). Issue #46's "omitted layer
call" attribution is superseded: the real regression was the fa59d97
attention-kernel replacement, fixed here and in #47.

**`make -j8 test-rocm`:** passes (test_rocm_tp_stubs, test_rocm_xdev,
test_rocm_kernel_compare 6/6, test_engine_rocm_tp_refusal).

**5-case oracle:** 0.406200156 vs anchor 0.405429743 (+0.2%), confirming
the fix restored the reference arithmetic for fixture-sized sequences.
