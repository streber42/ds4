# Issue 20 — MoE hot-path optimization runbook

## Plan

1. **Phase 1: Post-fix profile** — collect fresh MoE timing breakdown
   - `DS4_ROCM_MOE_DECODE_PROFILE=1` (built-in decode-phase timers)
   - `rocprof` kernel-level GPU timing (all kernels, all phases)
   - Compare against pre-fix numbers: 57% IQ2 hot (15.1s), 15% f32 cold (3.88s)

2. **Phase 2: Analysis** — identify whether IQ2 hot or f32 cold is the bigger cost
   - Hot path: `moe_gate_up_mid_expert_tile8_rowspan_kernel` (33ms × 344 calls)
   - Cold path: f32 fallback frequency

3. **Phase 3: Optimize** — based on Phase 2 findings
   - Option A: Enable WMMA hot-path (`DS4_ROCM_MOE_WMMA=1`) and measure
   - Option B: Reduce cold-path fallback (tune expert caching / streaming thresholds)
   - Option C: Tune gate_row_span (1024→2048 or 512) for better occupancy
   - Option D: Profile-guided kernel tuning (block size, shared mem, tile size)

4. **Phase 4: Validate** — quality fixture and throughput re-measurement

## Commands

### Build
```bash
make ROCM_ARCH=gfx1201 rocm -j16
```

### MoE decode profile (built-in)
```bash
DS4_ROCM_MOE_DECODE_PROFILE=1 ./ds4-bench -m /var/cache/llama/ds4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf --rocm --gpu-devices 0,1,2,3 --cuda-tensor-parallel --prompt-file speed-bench/promessi_sposi.txt --ctx-start 2048 --ctx-max 2048 --step-incr 2048 --gen-tokens 256 2>&1 | tee bench-out/issue20-moe-profile-1.log
```

### MoE decode profile with WMMA enabled
```bash
DS4_ROCM_MOE_DECODE_PROFILE=1 DS4_ROCM_MOE_WMMA=1 ./ds4-bench ... 2>&1 | tee bench-out/issue20-moe-profile-wmma.log
```

### rocprof kernel-level profile
```bash
rocprof -i rocprof_input.txt -o rocprof-out/results.csv -- ./ds4-bench ... 2>&1 | tee bench-out/issue20-rocprof.log
```

### Quality fixture
```bash
DS4_ROCM_MOE_DECODE_PROFILE=1 make ROCM_ARCH=gfx1201 rocm-quality -j16 && DS4_ROCM_MOE_DECODE_PROFILE=1 ./score_official --rocm --gpu-devices 0,1,2,3 --cuda-tensor-parallel 2>&1 | tee bench-out/issue20-quality-baseline.log
```
