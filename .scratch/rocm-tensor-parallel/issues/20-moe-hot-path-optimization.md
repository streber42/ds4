# 20 — MoE hot-path optimization

Status: ready-for-human

**What to build:** Profile-guided improvement of the dominant compute cost. Profile data shows MoE consumes ~72% of all GPU time (57% IQ2 hot-path, 15% f32 cold-path fallback). The `moe_gate_up_mid_expert_tile8_rowspan_kernel` averages 33ms per call and fires 344 times per run — it is the single most expensive operation.

Reduce either the cold-path fallback frequency (more experts stay hot) or the hot-path kernel execution time. This is the highest-ROI compute optimization target after the dispatch race fix.

Profile context (4-GPU TP, 8-gen-token rocprof, post-fix):
- IQ2 hot path (tile8_rowspan gate/up + batch_sharedmid down): 15.1s total, 71.9% of GPU time
- f32 owned kernels (decode cold path): 0.48s total, 2.3% of GPU time
- Hot kernel: avg 33.0ms per call on IQ2 path (unchanged from pre-fix — dispatch race did not affect MoE kernel cost)

## Acceptance criteria

- [x] Post-fix profile identifies whether IQ2 hot or f32 cold path is the bigger remaining cost
- [x] Hot-path IQ2 kernel improved: WMMA hotlist kernels enabled by default (DS4_ROCM_MOE_WMMA gate removed); drops per-layer MoE cost from ~43.9ms to ~8.87ms
- [ ] Improvement measured against #19 baseline (target: >10% TP throughput gain) — prefill improved 2.19×, decode unchanged at 12.44 t/s; need quality fixture and full throughput re-measurement
- [ ] Quality fixture scores are not regressed

## Comments

**2026-07-26 — Post-fix profile complete; WMMA hot-path enabled by default.**

### Post-fix profile results

rocprof kernel trace on 4-GPU TP, 2048 prefill + 8 gen tokens:

**Baseline (no WMMA):**
| Kernel | Avg | Calls | % GPU |
|---|---|---|---|
| moe_gate_up_mid_expert_tile8_rowspan_kernel | 33.0 ms | 344 | 54.0% |
| moe_down_q2K_expert_batch_sharedmid_kernel | 10.9 ms | 344 | 17.9% |
| f32 owned decode kernels | ~0.35 ms | 1376 | ~2.3% |

**WMMA enabled (DS4_ROCM_MOE_WMMA=1):**
| Kernel | Avg | Calls | % GPU |
|---|---|---|---|
| moe_gate_up_mid_iq2_hotlist_wmma_n2_kernel | 4.08 ms | 344 | 15.8% |
| moe_gate_up_mid_expert_tile8_rowspan_kernel | 2.30 ms | 344 | 8.9% |
| moe_down_q2K_hotlist_wmma_n2_kernel | 1.91 ms | 344 | 7.4% |
| moe_down_q2K_expert_batch_sharedmid_kernel | 0.58 ms | 344 | 2.3% |
| f32 owned decode kernels | ~0.35 ms | 1376 | ~5.4% |

**Throughput comparison (256 gen tokens):**
| Metric | Baseline | WMMA On | Δ |
|---|---|---|---|
| Prefill | 104.12 t/s | 227.85 t/s | **+2.19×** |
| Generation | 12.43 t/s | 12.42 t/s | ~0% |

### Key findings

1. **The tile8_rowspan kernel (54%) is confirmed as the dominant cost** — matches pre-fix profile. The "f32 cold path fallback" from the issue description (claimed 15%) is only ~2.3% in post-fix data; it was likely a pre-fix artifact of the dispatch race forcing serialization.

2. **WMMA gives ~5× per-layer MoE speedup for prefill** (43.9ms → 8.87ms per layer). The IQ2 hot-path tile8_rowspan drops from 33ms → 2.30ms (for non-hot experts only), with WMMA WMMA-N2 kernels handling the majority of work at 4.08ms + 1.91ms.

3. **Decode throughput is unchanged** — the decode path uses separate `_owned_f32_kernel` variants for TP that are independent of the env-var-gated WMMA path. Decode MoE is only ~2.3% of GPU time; the bottleneck is cross-device TP handoff overhead, not compute. Improving decode throughput needs a different approach (reducing sync points, batching copies, fusing layers).

4. **The WMMA gate was removed** — `ds4_rocm_moe_wmma_enabled()` now returns 1 unconditionally (escape hatch: DS4_ROCM_MOE_WMMA=0). The ~5× per-layer prefill speedup was too large to leave gated.

### Remaining work before closing

- Build and run quality fixture (`make rocm-quality`) to verify no regression with WMMA always-on
- Full throughput re-measurement against #19 baseline (12.44 t/s) — prefill already validated at 227.85 t/s
- The >10% TP throughput AC is partially met: prefill +2.19×, but generation unchanged
