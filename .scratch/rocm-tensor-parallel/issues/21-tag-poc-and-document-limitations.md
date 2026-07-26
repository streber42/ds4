# 21 — Tag PoC and document known limitations

Status: closed

**What to build:** A clean git tag (`tp-poc-v1`) plus a one-page summary at `.scratch/rocm-tensor-parallel/POC.md` capturing what works, what does not, and how to run it. This is the shareable proof-of-concept output of the entire TP porting effort.

The doc should cover:
- **Correctness evidence:** quality fixture scores, eval harness results, known-correct prompts
- **Performance numbers:** current TP vs pipeline throughput (from #19, or the pre-fix numbers if #18 not yet done)
- **Known limitations:** the dispatch race (and workaround if not yet fixed), the ~12 t/s ceiling and why
- **How to reproduce:** exact commands for quality fixture, bench, eval harness, and a plain chat test
- **What was ported:** link to the inventory, the sharding module, the xdev module

## Acceptance criteria

- [x] `.scratch/rocm-tensor-parallel/POC.md` written with all sections above
- [x] `tp-poc-v1` tag created at the current commit
- [x] README.md or a pointer from the PRD notes the tag and the doc location

## Blocked by

None — can start immediately. Links to issue 19's numbers when available, otherwise uses current known values.

## Comments

**2026-07-26 — closed.** Wrote `.scratch/rocm-tensor-parallel/POC.md` covering correctness
evidence, performance numbers, known limitations, reproduction commands, and what was ported.
Issue #19 (post-fix throughput re-measurement) is still open, so the performance section uses
the last confirmed pre-fix `ds4-bench` baselines and the separate live-server container
measurement from issue #11, both labeled by source and date, with the discrepancy between them
called out explicitly rather than papered over — reconciling them is exactly what #19 is for.
The document's headline finding is that this PoC is **not** production-ready: default
(unserialized) dispatch is still broken per issue #18's own escalation to `ready-for-human`
(the attempted fix did not close the quality gap), so `AMD_SERIALIZE_KERNEL=3` remains
required for trustworthy output. Documented this prominently rather than only in a limitations
subsection, since it's the single fact most likely to matter to anyone deciding whether to use
this build. Added a pointer to `POC.md` from the top of `PRD.md` (not README.md — the PRD's own
acceptance criteria allow either, and README.md documents shipped, working features; advertising
an admittedly-broken-by-default mode there would be misleading). Tagged `tp-poc-v1` at commit
`59ba21f` (the commit this issue's own prior work — docs for issues 18-22 and the issue-18 raw
logs — already landed on).
