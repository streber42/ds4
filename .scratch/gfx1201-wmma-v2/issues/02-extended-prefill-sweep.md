Status: closed
# 02 — Extended prefill sweep: WMMA v2 vs NO_WMMA across full context range

**What to build:** A/B benchmark the WMMA v2 (`rdna4-wmma`) vs baseline (`rdna4`)
builds across a wider context sweep (2048–32768 tokens) to characterize where the
prefill win holds, fades, or inverts. Issue 01 showed +14.5% at 2048 ctx but only
+1.9% at 4096 — this run will resolve whether that's a fluke, a crossover, or a
consistent taper.

**Blocked by:** Issue 01 (closed). Both build targets already exist in `~/src/ds4`.

- [x] Acquire GPU lock
- [x] Run baseline sweep: `rdna4` build, ctx 2048→32768, step 2048, pure prefill (`--gen-tokens 0`)
- [x] Run WMMA v2 sweep: `rdna4-wmma` build, same sweep
- [x] Record per-frontier t/s in `## Answer`
- [x] Release GPU lock

## Answer

> **SUPERSEDED 2026-08-11 by issue 04
> (`04-cold-start-controlled-prefill-sweep.md`). Do not cite the headline or the
> table below as a measurement of the WMMA kernel.**
>
> Issue 04 established two things that invalidate this result:
>
> 1. **The A/B was null.** This sweep's two columns were produced by
>    `~/src/ds4`'s `make rdna4` and `make rdna4-wmma` targets, per this issue's
>    own acceptance criteria. Those two targets are provably the same build:
>    `rdna4` sets `ROCM_EXTRA_CFLAGS=-DDS4_ROCM_NO_WMMA` (`Makefile:191`) but
>    nothing in that Makefile ever consumes `ROCM_EXTRA_CFLAGS` — `ROCM_CFLAGS`
>    (`:58`) does not reference it and the compile rules (`:340–346`) use only
>    `$(ROCM_CFLAGS)`. And a build that *did* define the macro would not compile
>    at all: that tree guards the kernel's definition (`ds4_rocm_q8.cuh:672`)
>    but not its launch site (`ds4_rocm_matmul.cuh:398`), which issue 04
>    confirmed with `hipcc -fsyntax-only -DDS4_ROCM_NO_WMMA` →
>    *"use of undeclared identifier 'matmul_q8_0_f32_batch_wmma_4w_kernel'"*.
>    So the "baseline" column below is a second WMMA run, not a baseline. The
>    +22.8% and the "+0.5% to +1.0% at all sixteen frontiers" are both
>    same-binary run-to-run variation, not kernel effects.
> 2. **The effect size was wrong in both directions.** Re-measured on this
>    tree's 4× R9700 against a reference build that really does compile the
>    kernel out, the warm-prefill win is **+20.5%** (+21.7% at 4096 to +19.4%
>    at 32768, with a mild KV-cache-pressure taper) — it does not drop to ~1%.
>    Separately, the first prefill in any process pays a fixed **+5.4 s**
>    cold-start cost, identical in both builds, which is what makes a 2048-frontier
>    row look slow and produced this table's rising 101 → 166 → 181 baseline curve.
>
> The interpretation below — "the win is only active on the first 2048-token
> chunk", "both builds converge to the same memory-bandwidth ceiling" — is
> therefore withdrawn. The `n_tok >= 256` gate does hold; the claim that the
> win fades above it does not.

**Result (WITHDRAWN — see above): WMMA v2 win is real but narrow-context-only. The kernel's n_tok≥256 guard
means the speedup is only active on the first 2048-token prefill chunk; all subsequent
chunks at higher ctx frontiers run at the same per-chunk throughput as baseline.**

### Per-frontier prefill t/s (2048-token chunk at each frontier, pure prefill)

| ctx frontier | baseline (`NO_WMMA`) | WMMA v2 | delta | delta % |
|---|---|---|---|---|
| 2048 | 101.16 | 124.28 | +23.12 | **+22.8%** |
| 4096 | 166.67 | 168.28 | +1.61 | +1.0% |
| 6144 | 181.85 | 183.04 | +1.19 | +0.7% |
| 8192 | 181.53 | 183.41 | +1.88 | +1.0% |
| 10240 | 181.97 | 182.88 | +0.91 | +0.5% |
| 12288 | 181.56 | 183.41 | +1.85 | +1.0% |
| 14336 | 182.11 | 183.49 | +1.38 | +0.8% |
| 16384 | 181.94 | 182.82 | +0.88 | +0.5% |
| 18432 | 181.66 | 183.34 | +1.68 | +0.9% |
| 20480 | 182.13 | 183.14 | +1.01 | +0.6% |
| 22528 | 182.12 | 183.26 | +1.14 | +0.6% |
| 24576 | 181.95 | 183.10 | +1.15 | +0.6% |
| 26624 | 181.82 | 182.73 | +0.91 | +0.5% |
| 28672 | 181.55 | 182.42 | +0.87 | +0.5% |
| 30720 | 181.26 | 183.07 | +1.81 | +1.0% |
| 32768 | 180.52 | 182.20 | +1.68 | +0.9% |

### Interpretation

The ds4-bench `--step-incr 2048` sweep measures the last 2048 tokens of each
frontier (i.e. the final chunk at that ctx depth), not the total time to prefill
from zero. Both builds plateau at ~181–183 t/s from 6k ctx onward — within
run-to-run noise of each other (~0.5–1.0%). The **+22.8% spike at 2048 ctx is
real**: it represents a cold prefill of exactly 2048 tokens in a single chunk,
which is the best case for the WMMA kernel (n_tok=2048 >> 256 gate, no prior
context, entire sequence fits in one kernel call).

At 4096+ ctx the bench measures a fresh 2048-token chunk appended after the
prior context is already cached. The WMMA kernel still fires (n_tok=2048 ≥ 256),
but the KV-cache prefill overhead and SSD expert-cache pressure dominate — the
matmul is no longer the bottleneck. Both builds converge to the same
memory-bandwidth ceiling (~182 t/s).

### Conclusion for `dev_ds4` campaign

The **first-chunk prefill win is larger than issue 01 measured** (+22.8% vs the
prior +14.5% — the prior run had two samples with variance). It is genuinely
useful for interactive single-turn prompts ≤ 2048 tokens. For long-context use
(RAG, document QA, chat with large history) the gain is noise-level once KV
cache is warm. The `rdna4-wmma` Makefile target remains the right recommendation
for `dev_ds4`'s C1/C6 benchmark configs; the win profile is just more precisely
characterised now.

## Comments

Initiated from interactive human session 2026-08-07.
Previous data points (from issue 01): baseline 98.46 t/s vs 112.74 t/s at 2048,
190.36 t/s vs 194.00 t/s at 4096.
