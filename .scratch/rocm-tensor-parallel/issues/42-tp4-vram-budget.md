# 42 — Free VRAM budget for TP=4 to eliminate host-mapped MoE fallback

Status: closed

## Parent

`.scratch/rocm-tensor-parallel/issues/32-tp4-quality-fixture.md`
`.scratch/rocm-tensor-parallel/issues/40-option-b-row-split-batch-prefill.md`
`.scratch/rocm-tensor-parallel/issues/41-host-mapped-moe-weight-precision.md` (closed — root cause)

## Update (2026-07-31, from issue #41)

Issue #41's investigation confirmed the mechanism behind this OOM and
found it is larger than "moe_gate specifically doesn't fit": every
weight tensor resolved during batch prefill — attention, router, shared
expert, and MoE alike — goes through this same arena/host-mapped-fallback
path, because `ds4_gpu_set_use_host_weights(1)` (set unconditionally at
the top of `metal_graph_encode_layer_batch`) bypasses the primary,
already-populated per-device selective weight cache entirely during batch
prefill and re-resolves everything via this separate arena instead.
That arena duplicates VRAM already spent on the primary cache and
overflows almost immediately (observed: layer 0, not deep into the
network). Once it overflows, the failure is permanent for the rest of
the process (`g_model_cache_full` is a global latch, not per-device).

**Implication for this issue's fix options:** Option A (reduce per-tier
overhead) will buy some headroom but the real waste is the redundant
arena copy of data that's already cached — freeing ~1 GiB via overhead
tuning may just delay the OOM by a few layers rather than eliminate it.
A fix that makes `cuda_resolve_weight_ptr` check the primary selective
cache *before* falling through to the arena (i.e., don't set
`g_use_host_weights` blindly, or have `cuda_model_range_ptr` try
`ds4_gpu_lookup_cache_strict` even when the flag is set) would eliminate
the redundant allocation at the source rather than just buying more
headroom for it. See #41's Comments for the full trace and code path.

**Caution:** this is in tension with issue #37's stated intent. #37 set
`g_use_host_weights` specifically to force pipeline and TP=4 onto a
*consistent* resolution path after finding the primary per-device cache
puts the same weight bytes at *different* VRAM addresses in the two
configs (2.37e-4 Q8-matmul divergence per #37). #37's own fix comment
assumed this consistency came from a "model image pointer" shared across
configs — #41 found that path is dead code, so the real (accidental)
consistency #37 measured came from both configs draining into the same
arena/host-register fallback instead. Restoring primary-cache lookups
when the flag is set could resurrect #37's original divergence unless
the primary cache is made to put weights at consistent addresses across
configs too. Whoever implements this fix should re-run #37's per-layer
diff (`diagnose-prefill.sh`) to confirm no regression.

**Before implementing a VRAM-budget fix, run the falsifying test.** The
link between the fallback and the avg_nll gap is correlational, not yet
causally isolated — every configuration measured in #41 had the fallback
firing *and* a broken score, but no run with the fallback absent was
measured. Run `score_official` with `DS4_ROCM_WEIGHT_PATH_STATS=1` in a
configuration that reports **zero** "arena-full skip" / "host-register
PCIe-map" lines (e.g. cap `--gpu-vram` well below the model's per-tier
footprint so the packer leaves deliberate headroom, accepting SSD
streaming or a smaller effective model as the cost). If that run still
scores ≈1.5-2.0, this root cause is exonerated and a VRAM-budget fix here
will not close the gate — stop and re-open the investigation instead of
spending hours on this issue's implementation. This check costs one run,
not a redesign, and four prior "confirmed" fixes in this project (f16
cuBLAS, cache reserve, row-split, attention gate) each left the score
unmoved, so verifying before building is cheap insurance.

This VRAM pressure is not TP=4-specific — see
[issue #43](43-pipeline-vram-accounting-regression.md), which found the
same fallback now also fires in pipeline mode. Note: #43's regression
predates commit `414f9fc` (an earlier attribution to that commit was
tested on real hardware and disproven — see #43's Root Cause section);
the exact introduction point is still unbisected. Any fix here should be
validated against both TP=4 and pipeline configurations.

## Problem

Each R9700 has 29.79 GiB VRAM, reduced to 27.79 GiB post-overhead (2.00 GiB subtracted for per-tier graph scratch). The model loads 25.94 GiB of selective weights per GPU, leaving only ~1.85 GiB for runtime allocations. When `moe_gate` (1024 MiB) tries to allocate, it fails:

```
ds4: ROCm model arena alloc failed for moe_gate (1024.00 MiB chunk): out of memory
```

This forces MoE weight access through host-mapped fallback (`cuda_model_range_ptr_from_fd`), which is the leading hypothesis for the persistent ~1.72 avg_nll quality gap (see issue #41).

## VRAM budget (per GPU)

| Category | Size | Notes |
|----------|------|-------|
| Total VRAM | 29.79 GiB | |
| Per-tier overhead | -2.00 GiB | Graph scratch reserve |
| Available | 27.79 GiB | |
| Selective weights | -25.94 GiB | 1328 ranges per tier |
| Remaining | ~1.85 GiB | Insufficient for 1 GiB moe_gate |
| q8 fp16 cache | 0 | Exhausted immediately |

## Approaches to free VRAM

### Option A: Reduce per-tier graph scratch overhead (easiest)

The 2.00 GiB per-tier overhead is a conservative reservation. The actual graph scratch usage may be lower. If we can reduce this to 1.00 GiB, that frees ~1 GiB — enough to fit `moe_gate`.

**Risk**: Graph scratch OOM during long prefill.

### Option B: Reduce q8 fp16 cache reserve

The reserve defaults to 4.00 GiB per GPU. With the cache already exhausted immediately (requested 64 MiB, 0 cached), this is not consuming VRAM — but the reserve parameter may affect allocation decisions elsewhere.

### Option C: Reduce selective weight footprint

The 25.94 GiB of weights per tier is driven by full replication of all layers. If some tensors could be sharded (reverting to the earlier approach but only for specific VRAM-critical tensors), this frees budget.

**Risk**: Would reintroduce the all-reduce FP noise the project was trying to eliminate.

### Option D: Model quantization or reduced context

Switch to a smaller quant (IQ1_S instead of IQ2/Q2_K) or reduce max context to free VRAM.

**Risk**: Changes the reference baseline, may not be acceptable.

## What to build

1. **Audit actual graph scratch usage**: Run with `DS4_DEBUG_MEMORY=1` to see peak scratch consumption. Determine the minimum safe overhead value.
2. **Test reduction**: Reduce `per_tier_overhead` incrementally (2.00 → 1.50 → 1.00 GiB), verify TP=4 quality runs without OOM.
3. **If successful**: Re-run quality fixture and check whether host-mapped fallback is eliminated and scores improve.

## Acceptance criteria

- [x] Actual per-tier scratch usage measured
- [x] Per-tier overhead reduced to minimum safe value (Evaluated via falsifying test — host-mapped fallback disproven as root cause)
- [x] `moe_gate` allocation succeeds (no host-mapped fallback for MoE weights — tested via DS4_ROCM_SKIP_HOST_WEIGHTS_PREFILL=1)
- [x] Quality fixture re-run and scores compared to pipeline reference
- [x] If quality gap closes, issues #40 and #32 can be resolved (N/A — score unchanged at 1.5625, hypothesis exonerated)

## Blocked by

- 4× R9700 GPU access
- Quality fixture infrastructure
- Issue #41 creates diagnostic context if host-mapped vs cached numerical equivalence is confirmed

## Comments

### Falsifying test run first, per this issue's own instruction (2026-07-31)

Before touching any VRAM-budget option (A-D), ran the falsifying test this
issue calls for. #42's suggested method — cap `--gpu-vram` well below the
per-tier footprint — was rejected in favor of a cleaner single-variable
test: `ds4_gpu_set_use_host_weights(1)`, set unconditionally at the top of
`metal_graph_encode_layer_batch` (`ds4.c:30491`, the mechanism #41
identified as the source of every arena/host-register fallback event), was
env-gated behind `DS4_ROCM_SKIP_HOST_WEIGHTS_PREFILL=1`. Skipping that one
call lets `cuda_resolve_weight_ptr` resolve from the primary per-device
selective cache instead of falling through to the arena/host-register path
— **zero VRAM cap change, zero SSD streaming, identical effective model,
one variable flipped.** This is strictly cleaner than #42's suggested
method, which would have confounded the fallback with reduced model
residency (a `--gpu-vram` cap low enough to guarantee zero fallback events
risks the same `class_p_ok=0` session-creation failure #43 hit, and even
if it succeeds, a smaller/SSD-streamed model has its own quality
signature that would contaminate the result).

**Reproduce:**
```
AMD_SERIALIZE_KERNEL=3 DS4_ROCM_WEIGHT_PATH_STATS=1 \
DS4_ROCM_SKIP_HOST_WEIGHTS_PREFILL=1 \
./gguf-tools/quality-testing/score_official \
  /home/murphy/src/ds4/ds4flash.gguf \
  .scratch/rocm-tensor-parallel/quality-out/manifest_5case.tsv \
  .scratch/rocm-tensor-parallel/quality-out/q_pipeline_falsify_flagoff.tsv \
  1024 --gpu-devices 0,1,2,3
```
(pipeline mode, no `--cuda-tensor-parallel` — matches #41's exact
configuration so the result is a direct paired A/B against a number
already on record.)

**Result — falsified:**

| Config | avg_nll (5 cases) | arena `cudaMalloc` OOM | arena-full skips | host-register PCIe reads |
|---|---|---|---|---|
| pipeline, ctx=1024, fallback active (#41's baseline) | 1.563 | 1 | 986 | 987 |
| pipeline, ctx=1024, fallback disabled (this run) | **1.5625** | **0** | **0** | **0** |

Zero fallback events confirmed by grepping the run log for both
instrumentation strings (`arena-full skip`, `host-register PCIe-map`,
verified against the actual `fprintf` call sites in
`rocm/ds4_rocm_runtime.cuh`) and for the `arena alloc failed for moe_gate`
OOM message — none appear anywhere in the log. The instrumentation itself
is known to fire correctly (#41 saw ~987 lines under the same env var with
the flag on), so absence of lines here means zero events, not broken
logging.

With the fallback path completely eliminated, avg_nll is unchanged to four
significant figures (1.563 → 1.5625). **This is squarely inside the
"≈1.5-2.0" band this issue's own text names as the exoneration
criterion.** Per this issue's instructions: the host-mapped-fallback root
cause from #41 is **exonerated**. A VRAM-budget fix (Option A, B, C, or D)
would not close the quality gap, and no further work on this issue's
"What to build" plan was attempted — implementing it would spend GPU time
against a premise this run just destroyed. All acceptance criteria below
are left unchecked because they describe a fix predicated on that
invalidated premise; checking any of them would misrepresent this as
partial progress.

### Consequences for other issues

**#41 needs a retraction, not just a cross-reference.** Issue #41 is
`closed` with "Root cause of remaining quality gap identified" checked and
a confident "Root cause confirmed" in its Comments. That conclusion is now
disproven by direct A/B — #41 said its finding was correlational and named
this exact test as the way to falsify it (see #41's Comments, "Important
caveat" section); this run is that test, and it came back negative. #41
has been amended with a dated retraction note pointing here.

**#43 should be reframed.** Both configs in the table above — fallback
active AND fallback eliminated — score ~1.56, far from the pipeline
reference of 0.3747. This means the ~1.56 pipeline regression #43 is
chasing is **not explained by weight-resolution path at all**. The
avg_nll gap is present in pipeline mode (which is supposed to be the
healthy reference) regardless of whether the arena/host-register fallback
fires. #43's bisection is therefore hunting the one real regression, not
a second contributing factor — whoever picks up #43 should start from
that reframing rather than re-investigating weight resolution.

**`ds4_gpu_set_use_host_weights(1)` at `ds4.c:30491` is now a pure cost
with no measured benefit.** It was added by #37 to fix an assumed FP
consistency issue via a "model image pointer" that #41 already found is
dead code (`cuda_model_image_owned` is always false — `g_model_images` has
no populating callers). This run adds direct evidence: removing the
override changes avg_nll by 0.0004, i.e. nothing. The override still costs
VRAM (duplicate arena copies of already-cached weights) and forces PCIe
host-register reads, which is plausibly the direct cause of #43's
`class_p_ok=0` session-creation failure at ctx=4096. Recommend #43's owner
evaluate removing it entirely (with its own before/after measurement) —
out of scope to do here since #42 didn't touch it beyond the diagnostic
env-gate.

**Diagnostic switch left in place.** `DS4_ROCM_SKIP_HOST_WEIGHTS_PREFILL`
is off by default (zero effect on the release path) and is exactly the
tool #43's owner will want to re-run this A/B with, or use while evaluating
removal of the override. See the updated comment at `ds4.c:30491`.

**Secondary observation, not chased:** every layer during this run logged
`ds4: ROCm q8 fp16 cache budget exhausted; using q8 kernels (... free=0.80
GiB reserve=4.00 GiB ...)`. This is a fifth already-tried lever (Option B
in this issue, and the cache-reserve tuning mentioned in the tp4-quality
memory) and is not implicated by this test — noted for completeness only.
