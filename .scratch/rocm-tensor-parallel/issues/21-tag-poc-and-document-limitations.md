# 21 — Tag PoC and document known limitations

**What to build:** A clean git tag (`tp-poc-v1`) plus a one-page summary at `.scratch/rocm-tensor-parallel/POC.md` capturing what works, what does not, and how to run it. This is the shareable proof-of-concept output of the entire TP porting effort.

The doc should cover:
- **Correctness evidence:** quality fixture scores, eval harness results, known-correct prompts
- **Performance numbers:** current TP vs pipeline throughput (from #19, or the pre-fix numbers if #18 not yet done)
- **Known limitations:** the dispatch race (and workaround if not yet fixed), the ~12 t/s ceiling and why
- **How to reproduce:** exact commands for quality fixture, bench, eval harness, and a plain chat test
- **What was ported:** link to the inventory, the sharding module, the xdev module

## Blocked by

None — can start immediately. Links to #19 numbers when available otherwise uses current known values.

## Status

ready-for-agent

- [ ] `.scratch/rocm-tensor-parallel/POC.md` written with all sections above
- [ ] `tp-poc-v1` tag created at the current commit
- [ ] README.md or a pointer from the PRD notes the tag and the doc location
