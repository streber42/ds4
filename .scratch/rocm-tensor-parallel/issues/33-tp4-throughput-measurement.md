# 33 — TP=4 throughput measurement and utilization

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/issues/25-widen-tp-to-4-rank.md`

## What to build

Measure TP=4 throughput and per-GPU utilization against the pipeline and TP=2 baselines. This is the proof-of-value gate: does TP=4 actually deliver the expected improvement over 2-pair pipelined TP?

**Benchmark:** `ds4-bench --rocm --gpu-devices 0,1,2,3 --cuda-tensor-parallel -m <production model> --prompt-file speed-bench/promessi_sposi.txt --ctx-start 2048 --ctx-max 2048 --step-incr 2048 --gen-tokens 256`

**Baselines to compare against:**
- 4-GPU pipeline layer-split: ~22.8 t/s generation, ~193 t/s prefill
- 4-GPU TP=2 (2-pair pipelined): ~12.3 t/s generation, ~206 t/s prefill

**Expected outcome:** TP=4 generation throughput should approach or exceed the pipeline baseline (~22.8 t/s) because all 4 GPUs compute on every token with no pipeline serialization. Per-GPU utilization should rise from ~25% (TP=2) toward >75%.

**If TP=4 doesn't beat pipeline:** Record the finding honestly in the experiment log with analysis of why. The PRD's secondary risk applies: "record the finding rather than bury it." This is not a failure — it is a measured data point.

## Acceptance criteria

- [ ] `ds4-bench` 4-GPU TP=4 throughput measured at `--ctx-start 2048 --gen-tokens 256` — **BLOCKED: prefill fails with OOM (moe_gate 320 MiB alloc fails)**
- [ ] Generation throughput compared against pipeline (~22.8 t/s) and TP=2 (~12.3 t/s) baselines — **BLOCKED: TP=4 produces garbled output, throughput not meaningful**
- [ ] Prefill throughput measured and compared — **BLOCKED: prefill fails**
- [ ] Per-GPU utilization measured via `rocm-smi` during steady-state decode — **BLOCKED: no steady-state decode possible**
- [ ] All-reduce overhead measured as fraction of per-token time — **BLOCKED: no successful decode**
- [ ] Results recorded in `.scratch/rocm-tensor-parallel/experiment-log.md` — **DONE: findings recorded**
- [ ] If TP=4 generation < pipeline generation: analysis of bottleneck recorded — **DONE: root cause analysis in Comments below**
- [ ] Issue #25 parent updated with findings; closed if all criteria met — **BLOCKED: cannot close until TP=4 is functional**

## Blocked by

- Issue #32: TP=4 quality fixture (correctness must be verified before trusting throughput numbers) — **Status: ready-for-human** (garbled output persists, needs debug)
- Issue #29: TP=4 attention path (decode loop synchronization) — **Status: implemented** (committed in `224c338`)
- Issue #30: TP=4 MoE path (decode loop synchronization) — **Status: implemented** (committed in `56c721b`)

**Note:** Issues #29 and #30 have been implemented (decode loop phase-split + shard divisor fix + cuda_tp_ep disable + FROM_ATTN_TO_FFN fix). The TP=4 path still produces garbled output — the remaining bug is in the phase-split logic or prefill MoE TP=4 path. Issue #32 documents the quality fixture failure.

## Comments

### Throughput measurement attempt (2026-07-27, autonomous session)

**Status: ready-for-human.** TP=4 throughput cannot be measured because the path
is broken by two independent bugs.

**Evidence 1 — coherence test:**
```
$ AMD_SERIALIZE_KERNEL=3 ./ds4 --rocm --gpu-devices 0,1,2,3 --cuda-tensor-parallel \
    -p "The capital of France is" -n 30
ds4: ROCm TP=4 placement: all 4 tiers hold every layer, sharded tensors split 4-way per rank
ds4: ROCm model arena alloc failed for moe_gate (320.00 MiB chunk): out of memory
We know a lot about the nature of the6 |.. Wester! one of ( as? working? My??  for? Logical
ds4: prefill: 0.64 t/s, generation: 1.22 t/s
```
Output is non-linguistic noise. Generation speed 1.22 t/s is not meaningful
because the output is incoherent.

**Evidence 2 — benchmark failure:**
```
$ AMD_SERIALIZE_KERNEL=3 ./ds4-bench --rocm --gpu-devices 0,1,2,3 --cuda-tensor-parallel \
    --prompt-file speed-bench/promessi_sposi.txt \
    --ctx-start 2048 --ctx-max 2048 --step-incr 2048 --gen-tokens 256
ds4-bench: prefill to 2048 failed: rocm prefill failed
```
The benchmark cannot run at all.

**Root cause 1 — decode loop synchronization (from issues #29/#30):**
The all-reduce primitive reads stale peer data because tiers compute partials
sequentially within a single tier's iteration rather than all 4 tiers computing
before any all-reduce fires. This produces garbled output.

**Root cause 2 — model arena OOM:**
Each tier loads 23.80 GiB of weights (1328 ranges), which is nearly the full
model size. With 4-way sharding, the 81 GiB model should shard to ~20 GiB per
rank (81 / 4 ≈ 20). The extra 3.8 GiB suggests some tensors are being
replicated instead of sharded, or the sharded offset calculation is not
reducing per-rank weight bytes as expected. Each GPU has ~31.86 GiB usable
(after 2.00 GiB scratch reservation), so 23.80 GiB weights leaves only ~6 GiB
for KV cache, activations, and model arena — insufficient for the 320 MiB
moe_gate allocation plus runtime overhead.

**What needs to be fixed:**
1. Issues #29/#30: restructure decode loop to separate attention and MoE phases
   so all 4 tiers compute partials BEFORE any all-reduce fires
2. Audit per-tier weight loading to ensure proper 4-way sharding reduces
   per-GPU VRAM from 23.80 GiB to ~20 GiB

**Baselines for comparison (from earlier experiments):**

| config | prefill (t/s) | generation (t/s) | per-GPU util |
|---|---|---|---|
| 4-GPU pipeline layer-split | 192.83 | 22.81 | ~30% |
| 4-GPU TP=2 pipelined | 206.09 | 12.27 | ~25% |
| **4-GPU TP=4 (this session)** | **FAIL** | **FAIL** | N/A |

Raw benchmark log: `.scratch/rocm-tensor-parallel/bench-out/tp4-issue33.log`

**Recommendation:** resolve issues #29/#30 (decode loop sync) and audit weight
sharding, then re-run this benchmark. Issue #32 (quality fixture) must also
pass before TP=4 throughput numbers are meaningful.

### Pipeline regression fix + VRAM block (2026-07-27, Ralph Loop session)

**Pipeline regression fixed.** The non-TP pipeline path had a latent bug introduced
in commit `946ba0a` (feat: 25 — Widen TP): the `metal_graph_encode_decode_layer`
call was accidentally removed when the decode loop was restructured for TP=4.
The non-TP fallthrough code (line ~26758 in `ds4.c`) only did post-layer
processing (cur_hc swap, dspark capture) without actually computing the layer.
This caused `ds4 --rocm --gpu-devices 0,1,2,3` (no `--cuda-tensor-parallel`)
to produce garbled output, making the pipeline reference baseline unusable.

**Fix applied:** Added `metal_graph_encode_decode_layer(g, model, &weights->layer[il], ...)`
call before the non-TP post-layer processing block. Verified correct output:
```
$ ./ds4 --rocm --gpu-devices 0,1,2,3 --model ds4flash.gguf -c 512 -p "The capital of France is" -n 30
We need to answer: "The capital of France is" and then provide the correct answer...
ds4: prefill: 4.46 t/s, generation: 14.35 t/s
```

**VRAM blocked — cannot test TP=4.**
After the fix was verified, subsequent test runs were killed by `timeout 120`,
which left ~29 GiB of stale VRAM on GPUs 0-2 (confirmed via DRM sysfs:
`mem_info_vram_used` shows 29010 MiB on GPU 0, 29447 MiB on GPU 1,
28359 MiB on GPU 2). Only GPU 3 is free (337 MiB used).

Attempted cleanup methods (all failed):
- `hipDeviceReset()` on each device — only affects the calling process's context
- Allocating+freed all available free VRAM — stale allocations are from exited process
- Kernel cache drop (`echo 3 > /proc/sys/vm/drop_caches`) — doesn't affect GPU VRAM
- KFD topology write (`tee /sys/class/kfd/kfd_topology/reset`) — triggered a GPU reset that recovered from a hang but didn't free VRAM

**Required:** A GPU reset or reboot to clear the stale VRAM before TP=4 can be tested.

**Recommended next steps (for human):**
1. Reboot or `sudo rocm-smi --reset-gpu` to free VRAM on GPUs 0-2
2. Rebuild with the pipeline fix (already applied at current HEAD)
3. Run pipeline reference: `AMD_SERIALIZE_KERNEL=3 ./ds4 --rocm --gpu-devices 0,1,2,3 --model ... -p "Hello" -n 10` to confirm pipeline still works
4. Run TP=4 coherence test: `AMD_SERIALIZE_KERNEL=3 ./ds4 --rocm --gpu-devices 0,1,2,3 --cuda-tensor-parallel --model ... -p "Explain C pointers in one sentence." -n 50`
5. If TP=4 output is still garbled (as documented in issue #32), debug the remaining decode loop correctness bug
6. Once TP=4 output is coherent, run the benchmark: `ds4-bench --rocm --gpu-devices 0,1,2,3 --cuda-tensor-parallel -m <model> --prompt-file speed-bench/promessi_sposi.txt --ctx-start 2048 --ctx-max 2048 --step-incr 2048 --gen-tokens 256`
7. Record throughput measurements in experiment-log.md

### VRAM cleared by reboot (2026-07-27, human action)

GPUs 0-2 had ~29 GiB stale VRAM from a killed test process. User rebooted the
machine, clearing all GPU VRAM. TP=4 testing is no longer VRAM-blocked.

**Remaining bugs (from comments above):**
1. Decode loop synchronization (issues #29/#30): tiers compute partials
   sequentially, then all-reduce — needs all 4 tiers to compute before
   any all-reduce fires
2. Weight sharding overshoot: each rank loads 23.80 GiB vs expected ~20 GiB —
   audit per-tier weight loading for proper 4-way sharding
3. Prefill OOM on moe_gate (320 MiB alloc) — downstream of issue 2

**Next actions for agent:**
1. Rebuild from current HEAD (pipeline fix already applied)
2. Verify pipeline reference still works
3. Debug TP=4 decode loop correctness
4. Once coherent, run TP=4 benchmark
5. Record throughput in experiment-log.md
