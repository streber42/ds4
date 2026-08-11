Status: ready-for-agent
# 69 — Characterize the real TP=4 per-token synchronization budget

## Parent

`docs/adr/0001-dense-tp4-parked-sequential-default.md`

## What to build

**Research and measurement only. This issue authorizes no implementation, no
kernel changes, and no default flips.** Its output is a number and a written
verdict.

ADR 0001 parked dense TP=4 on a specific causal claim:

> dense TP=4 performs **86 hidden-state all-reduces per token**, and at batch=1
> each is a synchronous PCIe round-trip; even at an optimistic ~1 ms per
> exchange that is ~86 ms/token — 2.5× over the ~35 ms budget that 28 t/s
> implies — before any compute.

That is an *estimate*, and this project's own measurement does not agree with it.
Issue #49's instrumented run attributes:

| call-site class | measured cost |
|---|---|
| all-reduce collectives (combined) | **~18 ms/token** |
| cross-device copies outside all-reduce | ~14 ms/token |
| `attn_tier_switch` | ~247 ms/token |
| `hc_expand_tier_switch` | ~7 ms/token |
| sync/dispatch total | **79% of per-token time** |

If all 86 all-reduces cost ~18 ms combined, that is ~0.21 ms each — roughly **5×
cheaper** than the ADR's "optimistic" 1 ms, and the collectives are then *not* the
dominant term. What dominates #49's table is **tier switching**
(`hipSetDevice` / BLAS handle swaps), by more than an order of magnitude.

#49 flagged its own caveat honestly: coarse CPU-side wall-clock timers cannot
separate true dispatch cost from the host blocking on outstanding GPU compute
inside the same interval. Its evidence that misattribution is real —
`hc_expand_tier_switch` uses an identical switch pattern but precedes far less
compute and costs 7 ms/token vs `attn_tier_switch`'s 247 — is suggestive but not
conclusive. **#49 explicitly recommended `rocprof` kernel timelines to split the
two, and that was never done.**

So the premise the project was parked on has never been directly measured. This
issue measures it.

## Why this matters

ADR 0001's re-entry guidance says not to resume dense-TP=4 perf work by "just
fixing" #67, because the win is capped unless the per-token synchronization
*count* collapses. That guidance is sound **if** collectives are the wall. If
#49's numbers are closer to the truth, the wall is tier-switch/dispatch overhead
— a different problem with a different (and possibly much cheaper) set of fixes,
and one that does not require an EP rewrite or a sharding redesign.

Either way the answer determines whether #67, #68, or neither is worth
re-opening. Resolve it before spending on any of them.

## Acceptance criteria

- [ ] `rocprof` (or `rocprofv3`) kernel-timeline capture of a TP=4 decode run,
      producing a per-token breakdown that separates: time in compute kernels,
      time in cross-device copy/collective kernels, and host-side gaps where no
      GPU work is in flight. This is the split #49 could not make and explicitly
      deferred.
- [ ] The literal all-reduce count per token confirmed or corrected against the
      ADR's "86" figure, by counting from the trace rather than from source
      reading. State the method.
- [ ] The ~247 ms/token `attn_tier_switch` figure resolved into its true
      components: how much is genuine `hipSetDevice`/BLAS-swap cost, and how much
      is compute-wait misattributed by #49's CPU-side timers.
- [ ] A written verdict in `## Answer` naming **the** dominant per-token cost,
      with the measured number: collectives, tier-switch/dispatch, compute, or
      host-side idle. Rank the top 3 with figures.
- [ ] An explicit statement of whether ADR 0001's causal claim ("86 all-reduces
      ≈ 86 ms ≈ the wall") is **supported, partially supported, or contradicted**
      by the measurement. If contradicted, say so plainly — the ADR is a document
      to be amended, not defended.
- [ ] A go/no-go recommendation for each of #67 and #68, grounded in the measured
      breakdown rather than the estimate. "Neither is worth re-opening" is a
      valid and welcome outcome.

## Guardrails

- **Do not amend ADR 0001 in this issue.** Record findings here; amending the ADR
  is a human decision, taken after reading this.
- **Do not change `metal_graph_tp4_spike_layer_enabled`'s default**, or any other
  default. Measurement only.
- Use the legacy sequential path (the current default) as the subject unless
  measuring the threaded path is specifically needed for a comparison — and if
  you do use `DS4_TP4_THREADED_LAYERS`, note that per #66 that path is *known
  incorrect* and its timings describe a racing engine, not a shippable one.
- Reuse #49's existing `DS4_GLM_SYNC_TRACE` instrumentation in `ds4.c` rather
  than building new timers. The gap to fill is the `rocprof` timeline, not more
  CPU-side counters.
- Beware the arena OOM that contaminated #49's absolute numbers (it measured
  ~1.4 t/s vs issue #33's clean-load ~4.5 t/s). #64/#65 addressed this; confirm
  a clean load before trusting absolute figures, and report whether you got one.
- **Keep the measurement prompt under 256 tokens**, unless
  `.scratch/gfx1201-wmma-v2/issues/03-fix-broken-gfx12-wmma-in-this-repo.md` has
  already landed. Reason: `rocm/ds4_rocm_matmul.cuh:404` dispatches the currently
  *broken* gfx12 WMMA kernel at `n_tok >= 256`, so a longer prompt prefills a
  numerically corrupt KV cache that every decode step then attends over. It
  should not distort the *timing* ranking this issue is after — a wrong matmul
  costs about what a right one does — but the generated text will be garbage.
  **Do not burn a GPU session debugging that**; if you see incoherent output with
  a ≥256-token prompt, this is the known cause. Say which prompt length you used.

## Blocked by

*(None — but see the prompt-length guardrail above, which is what keeps this
independent of issue 03 rather than blocked on it.)*

## Answer

## Comments

**2026-08-11 — Filed from an interactive human session**, as the scoping issue
ADR 0001 called for but never got. Prompted by noticing that #49's measured
all-reduce cost (~18 ms/token) is ~5× cheaper per exchange than the ADR's
estimate, and that tier-switching — not collectives — tops #49's table.
