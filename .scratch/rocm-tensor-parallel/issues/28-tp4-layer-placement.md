# 28 — TP=4 layer placement (model loads on 4 GPUs)

Status: ready-for-human

## Parent

`.scratch/rocm-tensor-parallel/issues/25-widen-tp-to-4-rank.md`

## What to build

Replace the current pipeline-split layer placement with a single-stage all-replicated placement for TP=4. All 4 GPUs load all 43 transformer layers, with sharded tensors split by `tp_rank` and replicated tensors loaded in full on every rank.

Currently, `n_stages = n_gpus / 2 = 2` creates two pipeline stages: lower-half tiers (0,1) hold contiguous layer ranges, upper-half tiers (2,3) hold no layers. For TP=4, `n_stages = 1` — every tier holds every layer.

**Sharded tensors** (loaded as the rank's shard only):
- Attention QKV projection weights: split by head count (32 of 128 heads per rank)
- Attention output projection weights: split by head count
- MoE routed expert weights (gate/up/down): split by expert count (64 of 256 per rank)
- Output head / vocabulary projection: split by row (V/4 per rank)
- Dense FFN columns (if applicable): split by column

**Replicated tensors** (loaded in full on every rank):
- RMS normalization weights
- MLA compressor weights (compressed KV is a single latent vector, not per-head)
- Shared expert weights (all ranks need it; rank 0 computes it during MoE)
- Embedding table

The `~15` sites in `ds4.c` that use `half = n_gpus / 2` and `partner = tier + half` need auditing. Each is either removed (no partner concept in TP=4) or replaced with `tp_world`-based N-way logic.

The TP=2 path must remain functional — this adds a parallel `tp_world == 4` code path.

## Acceptance criteria

- [x] All `half = n_gpus / 2` sites in `ds4.c` audited and updated for TP=4 path
- [ ] Production 81 GiB model loads on 4 GPUs without crash (~23 GiB per GPU)
- [ ] VRAM usage confirmed via `rocm-smi` after model load
- [ ] `./ds4 -p "Hello" -n 1` reaches first kernel dispatch (output will be garbage — attention/MoE exchange not yet implemented)
- [ ] TP=2 path still works: `--gpu-devices 0,1 --cuda-tensor-parallel` with 2 GPUs produces correct output
- [x] `make -j8 rocm` builds cleanly
- [x] Existing `test-rocm` test suite still passes

## Blocked by

- Issue #26: TP=4 sharding policy (shard config function needed for tensor loading)

## Comments

### Implementation summary (2026-07-26, autonomous session)

Implemented TP=4 single-stage all-replicated layer placement in `ds4.c`. The TP=4 path runs alongside (not replacing) the existing TP=2 CUDA path.

**New helpers added:**
- `engine_rocm_tp4_requested(e)` — detects ROCm TP=4 setup (DS4_ROCM_BUILD + cuda_tensor_parallel + exactly 4 GPUs + DeepSeek family + N_EXPERT divisible by 4)
- `engine_tp4_shard_divisor(e, t, entry)` — returns 4 for sharded tensors (routed experts, attn_q_b, attn_output/attn_output_a, output head), 1 for replicated tensors
- `engine_compute_tp4_placement(entry_bytes, n_entries, pcfg, placement)` — sets all entries to tier 0 as canonical home (cache install handles replication to all 4 tiers)
- `g->rocm_tp4` field on `ds4_gpu_graph` to propagate TP=4 flag through graph allocation

**Sites audited and updated:**
- `engine_compute_entry_bytes` — TP=4 branch: sharded tensors count as bytes/4 per entry, replicated as full bytes; refuses uneven division
- `engine_classify_multi_tier` — routes to `engine_compute_tp4_placement` for TP=4; skips `engine_adjust_output_head_for_cuda_tp` (TP=2-only); forces `multi_tier=1` even though placement[] is homogeneous
- `engine_install_per_device_caches` — TP=4 branch: replicates every tensor to all 4 tiers with appropriate sharding (sharded: offset = abs_offset + rank * bytes/4; replicated: full copy on each tier)
- `metal_graph_alloc_raw_cap` — TP=4 branch: activates all 4 tiers (no lower-half/upper-half partner model); skips the cuda_tp_partner_tier check
- `engine_install_dspark_support_cache` — TP=4 branch: picks tier with most free VRAM across all 4 (no home/partner bias)
- Transport probe (DS4_ROCM_BUILD block before engine_classify_multi_tier) — skipped for TP=4 (the half-pair probe only covers 2 of 12 ordered pairs; TP=4's all-reduce primitive has its own tests, and global mesh init handles all-to-all via direct peer or host-staging fallback)

**Build & test verification:**
```
$ make -j8 rocm
[all 5 binaries build cleanly: ds4, ds4-server, ds4-bench, ds4-eval, ds4-agent; only pre-existing nodiscard warnings from rocm runtime headers]

$ make test-rocm
test_rocm_tp_stubs: PASS (DS4_ROCM_TP_BRINGUP=1)
test_rocm_xdev: ALL TESTS PASSED (including 4-rank all-reduce tests)
test_rocm_kernel_compare: 6/6 kernels passed
test_engine_rocm_tp_refusal: PASS

$ ./tests/test_tp_sharding
228/228 checks passed (0 failed)

$ ./tests/test_layer_pack
97/97 checks passed (0 failed)

$ ./tests/test_engine_mgpu_placement
98/98 checks passed (0 failed)
```

**Acceptance criteria not verifiable without hardware + 81 GiB model:**
- Production 81 GiB model load on 4 GPUs
- rocm-smi VRAM confirmation
- First kernel dispatch reached via `./ds4 -p "Hello" -n 1`
- TP=2 path end-to-end with 2 GPUs

These require the actual production GGUF on the 4×R9700 workstation. The implementation follows the sharding policy from closed issue #26 and uses the all-reduce primitive from closed issue #27. The code paths are isolated (TP=4 runs only when `engine_rocm_tp4_requested` returns true, which requires exactly 4 GPUs + DeepSeek family), so TP=2 should be unaffected.

**Key design decision:** placement[] is homogeneous (all entries on tier 0) for TP=4. The `multi_tier=1` flag is forced explicitly so the multi-GPU init and per-device cache install actually run — without them the other 3 GPUs would get nothing loaded. The actual replication happens in `engine_install_per_device_caches`, mirroring how TP=2's cache install replicates each layer to its partner tier (extended here to all 4 tiers).

**Sharded tensor identification:** routed experts (gate/up/down via `engine_deepseek_routed_expert_tensor`), per-head attention projections (`attn_q_b`, `attn_output` for GLM-style, `attn_output_a` for Flash-style low-rank split), and the output head. Everything else (RMS norms, MLA compressor, shared expert, embedding, low-rank Q/KV stages) is replicated. The sharding is contiguous byte ranges starting at `abs_offset + rank * (bytes/4)`, which aligns with the sharding policy's contiguous ownership ranges.

**Next steps for human:** load the 81 GiB production GGUF on the 4×R9700 workstation and run:
```
./ds4 --rocm --gpu-devices 0,1,2,3 --cuda-tensor-parallel \
      --model /path/to/deepseek-v4-flash-iq2.gguf \
      -p "Hello" -n 1
```
Confirm model loads without crash, `rocm-smi` shows ~20-23 GiB per GPU, and first kernel dispatch is reached (output will be garbage until issues #29/#30 port the attention and MoE TP kernels).
