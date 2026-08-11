Status: ready-for-agent
# 04 — Re-run the prefill sweep with cold-start controlled, on this tree's 4× R9700

**What to build:** Issue 02 concluded the WMMA v2 win is "first-block only"
(+22.8% at ctx 2048, ~+1% everywhere above). That conclusion is **not
established** — issue 02 asserted one of two readings its own data supports, and
it measured the wrong machine configuration for the question we now care about.
Re-measure here, with a warm-up discriminator, and amend issue 02's headline to
whatever the data actually says.

**Blocked by:** Issue 03 (this tree's WMMA kernel must be correct before its
throughput means anything).

## Why issue 02's conclusion is suspect

Its own table shows baseline throughput **rising** 101 → 166 → 181 t/s as context
grows, while work per chunk also grows (same `n_tok=2048` GEMMs, plus twice the KV
to attend over). Throughput going up as work goes up is the signature of a fixed
overhead being amortized away — model load, first-touch page faults, SSD
expert-cache warm-up. Issue 01 explicitly blamed that same variance source for its
1.5–2.4 t/s decode spread on that host.

So both readings fit the table:

- **(a)** +22.8% is the WMMA kernel's genuine best case (one cold 2048-token chunk,
  `n_tok` far above the 256 gate, no prior context).
- **(b)** +22.8% is cold-start noise, and the honest effect size is the plateau.

Issue 02 asserted (a) without excluding (b).

Note also that the plateau delta is **positive at all sixteen frontiers**
(+0.5% to +1.0%, never negative). That consistency is hard to explain as noise —
it may be the true warm-prefill win, and it is small.

Finally: issues 01 and 02 both ran on `~/src/ds4` — a **single** R9700 with an
86 GiB model under `--ssd-streaming`, memory-starved in a way this tree's 4×
R9700 is not. "Both builds converge to a memory-bandwidth ceiling at ~182 t/s"
may be a property of that host, not of the kernel. This tree measured 228 t/s
prefill after issue #20.

## Acceptance criteria

- [ ] The discriminating experiment: run the 2048-token prefill **twice within a
      single process** (or with an explicit discarded warm-up pass) and report
      first-pass vs second-pass t/s for both the WMMA and `DS4_ROCM_NO_WMMA`
      builds. State plainly which reading — (a) or (b) — the data supports.
- [ ] Context sweep repeated on this tree's 4-GPU configuration, WMMA vs
      `DS4_ROCM_NO_WMMA`, with cold-start excluded. Reuse issue 02's frontier
      grid (2048→32768, step 2048) so the tables are comparable.
- [ ] ≥3 runs per configuration at the 2048 frontier, reported with spread, not
      a single sample. Issue 01's headline moved 14.5% → 22.8% between two runs
      of the same comparison; single samples are not trustworthy here.
- [ ] Issue 02's `## Answer` amended with a pointer to this issue's result and a
      correction if reading (b) wins. Do not silently leave a superseded
      headline in place.
- [ ] Result recorded in `## Answer`: the honest warm-prefill effect size, with
      the cold-start contribution separated out.

## Notes for the agent

- Requires the GPU lock (`AGENTS.md`). Plan the sweep as one locked session.
- A finding of "the win is ~1% once warm" is a **perfectly good result** and must
  be reported as such. Do not reach for the larger number because issue 02 did.
- This kernel only fires for `n_tok >= 256`, so this is a prefill-only question.
  Decode is untouched by design — don't re-litigate it.

## Blocked by

`.scratch/gfx1201-wmma-v2/issues/03-fix-broken-gfx12-wmma-in-this-repo.md`

## Answer

## Comments

Filed 2026-08-11 from an interactive human session.
