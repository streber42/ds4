Status: ready-for-agent
# 01 — Port and A/B benchmark the real gfx12 WMMA v2 kernel

**What to build:** Port `d9be29f`'s gfx12 WMMA v2 kernel (from this repo's
`origin/gfx1201-discrete-gpu` branch) onto dev's current ROCm tree at
`~/src/ds4`, replacing the `DS4_ROCM_NO_WMMA` fallback introduced by
`775ca6a`. Rebuild, verify correctness against the current fallback on a
small fixture, then A/B decode throughput. Full context in `../spec.md`.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] Read `d9be29f`'s full diff (`git show d9be29f -- rocm/ds4_rocm_q8.cuh`
      in this repo) and port it onto dev's current `~/src/ds4` tree
- [ ] Build with `make rdna4` (or equivalent) using the new WMMA v2 path in
      place of `-DDS4_ROCM_NO_WMMA`
- [ ] Correctness: output parity check against the current fallback baseline
      on a small fixture (avg_nll or equivalent — reuse
      `../rocm-tensor-parallel`'s quality-fixture tooling if it fits)
- [ ] A/B decode t/s: current `DS4_ROCM_NO_WMMA` baseline vs. the ported WMMA
      v2 path, same model/prompt/depth, via `ds4-bench`
- [ ] Record the result (win/neutral/regression, with numbers) in `## Answer`
      below
- [ ] If it's a win: flag it back to `/home/murphy/src/dev_ds4`'s
      `.scratch/deepseek-benchmark/` campaign as a candidate baseline update
      (their C1/C6 configs run on this same hardware)
