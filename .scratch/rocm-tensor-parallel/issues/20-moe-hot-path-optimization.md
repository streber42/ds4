# 20 — MoE hot-path optimization

**What to build:** Profile-guided improvement of the dominant compute cost. Profile data shows MoE consumes ~72% of all GPU time (57% IQ2 hot-path, 15% f32 cold-path fallback). The `moe_gate_up_mid_expert_tile8_rowspan_kernel` averages 33ms per call and fires 344 times per run — it is the single most expensive operation.

Reduce either the cold-path fallback frequency (more experts stay hot) or the hot-path kernel execution time. This is the highest-ROI compute optimization target after the dispatch race fix.

Profile context (4-GPU TP, 64 decode tokens, pre-fix):
- MoE (IQ2 hot WMMA): 15.1s total, 57.0% of GPU time
- MoE (f32 cold fallback): 3.88s total, 14.6% of GPU time
- Hot kernel: avg 33.0ms per call on IQ2 path

## Blocked by

- #19 — clean post-fix baseline needed to measure improvement

## Status

ready-for-agent

- [ ] Post-fix profile identifies whether IQ2 hot or f32 cold path is the bigger remaining cost
- [ ] Either hot-path IQ2 kernel improved or cold-path fallback frequency reduced
- [ ] Improvement measured against #19 baseline (target: >10% TP throughput gain)
- [ ] Quality fixture scores are not regressed
