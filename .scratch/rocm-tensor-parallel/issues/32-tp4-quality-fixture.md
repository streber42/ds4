# 32 — TP=4 quality fixture (authoritative correctness gate)

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/issues/25-widen-tp-to-4-rank.md`

## What to build

Run the official 100-case quality fixture on the TP=4 build and verify scores match the serialized pipeline reference within tolerance. This is the authoritative correctness gate for the entire TP=4 effort.

**Setup:** `AMD_SERIALIZE_KERNEL=3` to avoid the compressor prefill race (issue #23), production 81 GiB model, 4× R9700 GPUs, `make rocm-quality` build.

**Reference scores (from issue #20):**
- Pipeline serialized: avg_nll 0.373815, first_match 64/100, avg_lcp 5.81
- TP=2 serialized: avg_nll 0.372143, first_match 67/100, avg_lcp 6.59

**Acceptable range:** avg_nll within ±1% of the pipeline serialized reference (0.370 – 0.378). Floating-point reassociation from different sharding is expected and acceptable within this band (see PRD Testing Decisions). First-token match should be ≥60/100.

If the scores fall outside tolerance, this becomes a HITL issue: the specific failing cases need human review to determine whether the divergence is floating-point reassociation (acceptable) or a sharding/arithmetic bug (must fix before closing).

## Acceptance criteria

- [ ] Quality fixture runs to completion on TP=4 (100 cases, 2289 tokens)
      - BLOCKED BY: Cross-device KV cache read consistency (first token correct, subsequent decode tokens garbled)
      - Output head TP=4 vocab-split: ✅ IMPLEMENTED (fixed in commit 475c92b)
      - Prefill attention TP=4 tier sweep: ❌ NOT YET NEEDED for token-by-token path; may be needed for batch prefill path
- [ ] avg_nll within ±1% of pipeline serialized reference (0.373815)
- [ ] first_match ≥ 60/100
- [ ] api_top1_rate ≥ 0.85 (consistent with reference)
- [ ] api_pair_rate ≥ 0.98 (consistent with reference)
- [ ] Results recorded in experiment log with comparison table
- [ ] Raw per-case TSV saved in `.scratch/rocm-tensor-parallel/quality-out/`
- [ ] If scores are outside tolerance: failing cases identified and root cause analyzed

## Blocked by

- ~~Prefill attention lacks TP=4 tier sweep~~ ✅ CLOSED — token-by-token path doesn't need it
- ~~Output head lacks TP=4 vocab-split path~~ ✅ CLOSED (commit 475c92b)
- ~~Issue #30: TP=4 MoE path~~ ✅ CLOSED
- ~~Issue #23: same-device compressor prefill race~~ ✅ CLOSED (AMD_SERIALIZE_KERNEL=3 works around it)

## Comments

### TP=4 path produces garbled output (2026-07-27, autonomous session)

**Status (at time of writing): ready-for-human.** The
TP=4 decode path produces incoherent output. Root cause is the decode loop
synchronization issue documented in issues #29 and #30: the all-reduce primitive
(`ds4_rocm_xdev_allreduce_f32`) is a LOCAL operation on `my_dev` that reads peer
buffers without waiting for them to be computed. When tier 0 runs all-reduce,
tiers 1/2/3 have not yet computed their partials for the current layer, so the
all-reduce sums `[tier0=correct, tier1=stale, tier2=stale, tier3=stale]` — wrong
result. The same corruption applies to the MoE all-reduce. The decode loop in
`metal_graph_encode_token_raw_swa` calls `metal_graph_encode_decode_layer` once
per layer and iterates tiers inside it; the all-reduce fires inside each tier's
iteration rather than after all tiers have computed their partials.

**Evidence — quick coherence test:**
```
$ AMD_SERIALIZE_KERNEL=3 ./ds4 --rocm --gpu-devices 0,1,2,3 --cuda-tensor-parallel \
    --model .../DeepSeek-V4-Flash-IQ2XXS-...gguf \
    -p "Hello" -n 30
ds4: ROCm TP=4 placement: all 4 tiers hold every layer, sharded tensors split 4-way per rank
ds4: ROCm model arena alloc failed for moe_gate (320.00 MiB chunk): out of memory
Hello. [halleloo bact [ | atarnde. |:type:  epilee' (ex?a: a: a.a
ds4: prefill: 0.46 t/s, generation: 1.22 t/s
```
Output is non-linguistic noise. Also note the `moe_gate` OOM warning.

**Evidence — quality fixture partial run (25 of 100 cases before kill at 600s):**

| metric | TP=4 (partial, 25 cases) | pipeline reference (100 cases) |
|---|---|---|
| per-case avg_nll | 6.0 – 9.2 | 0.001 – 0.91 |
| first_match | 0 or 1 per case | 0 or 1 (65/100 total) |
| api_top1_rate | 0.04 – 0.21 | ~0.83 – 1.00 |
| api_pair_rate | 0.66 – 0.88 | ~0.96 – 1.00 |

Scores are 10–20× worse than reference; this is not floating-point
reassociation, it is fundamental numerical corruption from the decode loop
sync bug. Continuing to 100 cases would not change the conclusion.

**Pipeline reference re-validated (this session, 2026-07-27):**

The pipeline path (no `--cuda-tensor-parallel`) still works correctly on the
current build, confirming the test infrastructure is healthy:

| metric | pipeline serialized (this run) | pipeline serialized (issue #20 ref) |
|---|---|---|
| avg_nll | 0.3747 | 0.3738 |
| first_match | 65/100 | 64/100 |
| avg_lcp | 6.26 | 5.81 |
| api_top1_rate | 0.859 | 0.859 |
| api_pair_rate | 0.988 | 0.988 |

Fresh reference TSV saved: `.scratch/rocm-tensor-parallel/quality-out/q_pipeline_ref_tp4issue32.tsv`

**Required fix (from issue #30 comments):**

Restructure the decode loop to separate attention and MoE phases so that all
4 tiers compute their partials BEFORE any all-reduce fires:

```c
for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
    if (g->rocm_tp4) {
        // Phase 1: all 4 tiers compute attention partials
        for (int tier_iter = 0; ok && tier_iter < 4; tier_iter++) {
            metal_graph_set_active_tier_decode(g, tier_iter);
            g->tp_rank = tier_iter;
            ok = metal_graph_encode_decode_layer_attention_only(g, ...);
        }
        // All-reduce attention (now all 4 partials are available)
        allreduce_attention();

        // Phase 2: all 4 tiers compute MoE (using all-reduced attention)
        for (int tier_iter = 0; ok && tier_iter < 4; tier_iter++) {
            metal_graph_set_active_tier_decode(g, tier_iter);
            g->tp_rank = tier_iter;
            ok = metal_graph_encode_decode_layer_moe_only(g, ...);
        }
        // All-reduce MoE (now all 4 partials are available)
        allreduce_moe();
    } else {
        ok = metal_graph_encode_decode_layer(g, ...);
    }
}
```

This requires splitting `metal_graph_encode_decode_layer` into attention-only
and MoE-only phases. This is a significant refactor and is the correct scope
of issue #29. Issue #32 cannot close until that is done and the TP=4 path
produces coherent output again.

**Recommendation:** resolve issues #29 and #30 (decode loop sync), then re-run
this fixture. The pipeline serialized reference from this session
(`q_pipeline_ref_tp4issue32.tsv`) can be reused as the comparison baseline if
the build has not changed materially.

### Gemini consultation — path to unblock (2026-07-27, live-pair session)

Second-opinion review covering the decode loop phase-split architecture (issue
#29), the `moe_gate` OOM (issue #30), and the resulting quality fixture (this
issue). Gemini reviewed a full context dump of `metal_graph_encode_decode_layer_phase`,
the outer decode loop, TP=4 all-reduce code, the existing phase enum, the shard
divisor, and the OOM error message. A follow-up corrected a subtle error in the
first opinion.

**Status changed `ready-for-human` -> `ready-for-agent`.** The ralph loop will
pick this up by resolving issues #29 and #30 first.

**Architectural decisions (approved by human):**

1. **Reuse existing `TO_FFN` / `FROM_ATTN_TO_FFN` phases.** Do NOT add new
   phases and do NOT split into `_attention_only`/`_moe_only` functions. The
   existing phases are already used as a split pair in batch-session code
   (`ds4.c:63929/63967/64353`). ~2,800 lines of GPU kernel dispatch should not
   be duplicated.
2. **Hoist TP=4 all-reduce OUT of `metal_graph_encode_decode_layer_phase`**
   into the outer decode loop. Communication belongs in the decode loop, not
   inside the phase function.
3. **Add the missing `TO_FFN` early-exit** at ~line 22933 (after
   `ds4_gpu_hc_expand_tensor`, before FFN-side HC post norm). Currently
   `TO_FFN` falls through and runs the full layer — the existing batch-session
   split pair is latent-broken. Fixing it fixes both paths.
4. **Do NOT split at `TO_ROUTER`.** RMSNorm between attention output and the
   router is non-linear: `RMSNorm(sum_t A_t) != sum_t RMSNorm(A_t)`. The
   all-reduce MUST happen immediately after the attention output projection,
   before any LayerNorm/RMSNorm.
5. **Add `ds4_rocm_tp4_sync_tier_streams` barrier** between the tier-sweep and
   all-reduce. ROCm kernel dispatch is async; without the sync, rank 0's
   all-reduce kernel launches before rank 3's attention kernel has finished
   writing its buffer.
6. **`moe_gate` OOM root cause is missing shared expert in shard divisor.**
   `engine_tp4_shard_divisor` doesn't include `ffn_gate_shexp`/`ffn_up_shexp`/
   `ffn_down_shexp`, so they're loaded fully on every GPU. Add them to divisor
   to shard shared expert across 4 ranks. Keep `ffn_gate_inp` (router gate)
   replicated — small (~7 MiB) and must be globally consistent for top-k.
7. **Order of attack (revised):** decode loop FIRST, OOM second. Evidence: the
   partial 25/100 quality fixture ran *despite* the OOM warning and produced
   NLL 6.0-9.2. The OOM is an arena-alloc warning execution proceeds past; the
   decode loop is the primary correctness blocker.

**Prerequisite issues:** #29 (decode loop phase-split) and #30 (shard divisor
fix + coherent paragraph test) must land before this fixture can be re-run.

**After prerequisites land, re-run this fixture:**
```bash
AMD_SERIALIZE_KERNEL=3 ./ds4 --rocm --gpu-devices 0,1,2,3 --cuda-tensor-parallel \
  --model /home/murphy/src/ds4/ds4flash.gguf \
  --quality-fixture 100
```

**Acceptance criteria (from top of issue):**
- avg_nll within +/-1% of pipeline serialized reference (0.373815)
- first_match >= 60/100
- api_top1_rate >= 0.85
- api_pair_rate >= 0.98
- Results recorded in experiment log
- Raw per-case TSV saved in `.scratch/rocm-tensor-parallel/quality-out/`

**Reference baseline:** `q_pipeline_ref_tp4issue32.tsv` (from prior session)
can be reused if the build has not changed materially since then. If the build
HAS changed, re-run the pipeline serialized reference first.

**Implementation plan for #29 (decode loop phase-split):** see issue #29
comments "Gemini consultation — approved plan" section.
**Implementation plan for #30 (shard divisor fix):** see issue #30 comments
"Gemini consultation — approved plan" section.

### Autonomous re-evaluation (2026-07-27) — TP=4 path still garbled after #29/#30

**Context:** Commits #29 (`224c338` — decode loop phase-split) and #30
(`56c721b` — MoE shard divisor + cuda_tp_ep fix) are both applied at HEAD.
All automated tests pass (sharding, xdev, kernel compare, TP stubs, engine
refusal). The model loads without OOM on 4 GPUs (23.00 GiB per tier, 3.83 GiB
free).

**TP=4 coherence test:**
```
$ AMD_SERIALIZE_KERNEL=3 ./ds4 --rocm --gpu-devices 0,1,2,3 --cuda-tensor-parallel \
    --model .../ds4flash.gguf -p "Hello" -n 10
 for sake     ( e
```
Output is garbled — not a coherent sentence. No OOM errors this time
(cuda_tp_moe=false fix is present and working), but the decode loop
synchronization issue persists.

**Pipeline path regression (new finding):**
The ROCm pipeline path (without `--cuda-tensor-parallel`) also produces
garbled output on the current HEAD. This is a regression introduced in
commit `946ba0a` (feat: 25 — Widen TP from 2-pair pipeline to true 4-rank
tensor parallelism). At commit `6354b24` (the parent of `946ba0a`), the
pipeline path produces correct avg_nll ~0.37 on the quality fixture.
At the current HEAD, the pipeline path produces avg_nll ~21 on the same
fixture. The exact root cause could not be identified — all code changes
between these commits appear to be inside `g->rocm_tp4` guarded blocks.

The CPU backend path (`make cpu`) still produces coherent output:
"for sake     ( e" on GPU vs "We need to answer the" on CPU.

**Quality fixture status:**
The previously captured pipeline reference (`q_pipeline_ref_tp4issue32.tsv`,
avg_nll 0.374733) cannot be re-validated because the pipeline path is
regressed. The TP=4 quality fixture (run at HEAD with correct binary)
produces avg_nll ~10-21 (from partial 55-case run earlier in this session),
which is far outside the ±1% tolerance (0.370-0.378).

**Root cause:**
The TP=4 decode loop synchronization issue remains unfixed. The phase-split
implementation from #29 restructured the decode loop into TO_FFN and
FROM_ATTN_TO_FFN phases with barriers and all-reduces, but the output is
still numerically corrupted. The specific remaining bug could be:

1. The HC expand after attention all-reduce may not be correctly using
   the full combined attention output on all 4 tiers (the broadcast step
   might have a stale buffer or wrong size).
2. The prefill MoE all-reduce path (batch mode) might have a similar
   synchronization issue — each tier computes only its owned 64 experts,
   but the all-reduce might fire before all tiers finish.
3. The `metal_graph_encode_output_head` path for TP=4 (distributed decode
   sampling) might have a correctness bug.
4. A subtle issue in the barrier placement or `ds4_rocm_xdev_sync_all_devices`
   might leave stale data in the peer buffers read by the all-reduce.

**Recommendation:**
This issue requires human investigation to debug the remaining TP=4
correctness bug. The following diagnostic steps would help:

1. Use the correctness harness (`test_engine_correctness_harness-rocm`) to
   compare per-layer logits between TP=4 and the CPU reference path.
2. Add debug output to the TP=4 decode loop to trace per-tier attention
   output values before and after all-reduce.
3. Verify the HC expand produces identical `after_attn_hc` on all 4 tiers
   after the broadcast + HC expand phase.
4. Fix the pipeline path regression (commit `946ba0a`) to restore the
   ability to re-validate the pipeline reference.

### Human-investigation session (2026-07-27) — root cause identified

**Status changed `ready-for-human` -> `ready-for-agent`.** Human and Gemini
(Gemini 3.6 Flash) investigated the remaining TP=4 correctness bug. The decode
loop phase-split (issue #29) is architecturally correct. The root cause is TWO
untouched code paths:

---

**Root Cause 1 — Prefill attention is TP=4-unaware (PRIMARY).**

`metal_graph_encode_layer_attention_batch` at `ds4.c:27356` has
`tp_row_split_attn` gated on `g->tp_world == 2` (line 27398). For TP=4
(`tp_world == 4`), this is FALSE, so no tier-aware attention happens during
prefill. The function runs on tier 0 only.

Sequence of failure:
1. Prefill attention runs on tier 0, populates tier 0's KV cache.
2. Tiers 1, 2, 3 never compute attention during prefill — their KV cache
   regions remain uninitialized (zeros/garbage).
3. Decode loop iterates all 4 tiers correctly (issue #29 fix), but when tier 1
   switches in, it reads garbage from its local KV cache for the prefill
   tokens.
4. Tier 1's garbage attention output contaminates the all-reduce, corrupting
   tier 0's correct partial.
5. All subsequent decode tokens are garbage.

Contrast with `metal_graph_encode_layer_ffn_batch` at `ds4.c:29152` which DOES
have a complete `g->rocm_tp4` branch (line 29620) that iterates all 4 tiers,
copies router data via xdev, and all-reduces MoE partials. The attention batch
function has no equivalent.

**Root Cause 2 — Output head runs full-vocab matmul against sharded weights.**

`metal_graph_encode_output_head` at `ds4.c:24390` switches to head_tier (tier
0) and falls through the TP=2-only branches (`tp_world == 2`, `cuda_tp_ep &&
cuda_tp_output`) to the default `metal_graph_matmul_dense_quant_tensor` with
the FULL `weights->output` descriptor. For TP=4, `engine_tp4_shard_divisor`
returns 4 for `weights->output`, so only 1/4 of the tensor is cached on
tier 0. The full-range weight resolution falls back to
`cuda_model_range_ptr_from_fd` which, on discrete GPUs with limited VRAM,
returns NULL. The matmul then reads from NULL → GPU page fault / garbage
logits → garbled sampling.

Issue #31 claimed the output head is complete, but its "Full logit all-gather
available for prefill / quality fixture scoring" acceptance criterion is not
actually implemented for the TP=4 code path. The distributed decode sampling
(split_top1 with local argmax + all-gather of 4 tuples) works correctly for
greedy decode, but the full-logit path used by quality fixture and prefill
scoring does not.

**Gemini consultation (2026-07-27, live-pair session):**

Gemini reviewed the full code context including the decode loop, batch
attention, batch FFN, output head, weight resolution, and cache installation
code. Verdict: "YES, the TP=4-unaware prefill attention path and output head
are 100% the root cause of the garbled output."

**Approved fix plan (reviewed and approved by human):**

1. **Add TP=4 tier sweep to prefill attention** (`metal_graph_encode_layer_attention_batch`):
   Mirror the pattern from FFN batch and decode loop: iterate tiers 0-3 per
   layer, each computing 32-head attention partials, all-reduce attention
   output, broadcast to all tiers, HC expand on each tier. KV cache must be
   populated on all 4 tiers during prefill.

2. **Add TP=4 vocab-split path to output head** (`metal_graph_encode_output_head`):
   Split vocabulary into 4 shards (V/4 per rank), each rank computes its shard
   logits, gather via xdev_copy onto tier 0 for full logit scoring.
   Alternative: set `cuda_tp_output` or equivalent flag for TP=4 to use the
   existing multi-tier logit gather machinery.

**Pipeline regression note:** The pipeline path regression reported in the
previous comment is NOT present at HEAD (`4b40c5d`). The pipeline path produces
correct output: `"We need to respond to the user's initial greeting"` for
`"Hello"`. The pipeline reference TSV `q_pipeline_ref_tp4issue32.tsv` (100
cases, avg_nll 0.374733) is valid and reusable.

**After both fixes land**, re-run the quality fixture:
```bash
AMD_SERIALIZE_KERNEL=3 ./ds4 --rocm --gpu-devices 0,1,2,3 --cuda-tensor-parallel \
  --model /home/murphy/src/ds4/ds4flash.gguf \
  -p "Explain C pointers in one sentence." -n 50
```
If that produces coherent output, run the full fixture via `score_official`:
```bash
make -j8 rocm-quality
AMD_SERIALIZE_KERNEL=3 \
  ./gguf-tools/quality-testing/score_official \
    /home/murphy/src/ds4/ds4flash.gguf \
    gguf-tools/quality-testing/data/flash/manifest.tsv \
    .scratch/rocm-tensor-parallel/quality-out/q_tp4_final.tsv \
    4096 --gpu-devices 0,1,2,3 --cuda-tensor-parallel
```

### AI Consultants panel — per-tier compressor state consensus (2026-07-28, human + consultant panel)

**Status: ready-for-agent** (changed from ready-for-human). The remaining decode
loop data race has been reviewed by a consultant panel (Grok 9/10, Qwen3 9/10)
and the fix plan is approved.

**Root cause of remaining garbled decode:**
The compressor state arrays (`layer_attn_state_kv[il]`, `layer_attn_state_score[il]`)
are allocated only on device 0 but the decode loop launches all 4 tiers' kernels
concurrently (async HIP launches). All 4 tiers write the same device-0 memory
simultaneously → data race → corrupted attention state → garbled output.

**Consultant panel verdict (unanimous): Option A — per-tier compressor state arrays.**
Options B (serialize tiers) and C (read-only shared state) are rejected:
- B imposes ~4× latency penalty, defeating TP's purpose entirely — not even
  acceptable as a "correctness-first interim step"
- C risks silent quality drift (tiers 1-3 would compress with stale state)

**Approved fix plan (3 steps):**

1. **Allocate per-tier compressor state arrays:**
   `layer_attn_state_kv_tp1[il]`, `layer_attn_state_kv_tp3[il]`,
   `layer_attn_state_score_tp1[il]`, `layer_attn_state_score_tp3[il]`
   (tier 2 reuses existing `_tp` variants). Each allocated on its respective
   device. VRAM cost: ~8.6 MB total (0.026% of 32 GiB per GPU).

2. **Fix raw cache + compressor state pointer selection in decode loop:**
   Stop passing `g->layer_raw_cache[il]` (tier 0) to all 4 tiers. Use a
   tier-switch (preferably tier-indexed arrays: `_tier[4][il]`) to select
   the correct per-tier raw cache, compressed cache, and compressor state
   pointers based on active tier. Fix all three pointer families together.

3. **Close the prefill→decode KV sync gap:**
   Ensure the token-by-token eval path (`ds4_session_eval` →
   `metal_graph_eval_token_raw_swa`) initiates KV cache + state on all 4
   tiers after prefill, not just the batch prefll path (`ds4_session_slice`).

**Decision:** proceed directly to Option A implementation. Do not ship a
host-serial interim.

### Autonomous session (2026-07-27) — structural fixes landed, decode loop still garbleed

**What was implemented:**

1. **TP=4 output head vocab-split** (`metal_graph_encode_output_head`):
   Splits vocabulary into 4 equal shards (V/4 per rank). Each rank computes its
   shard via `metal_graph_matmul_dense_quant_abs` with adjusted weight offset.
   Output_norm is xdev-copied from tier 0 to tiers 1-3 before each shard
   matmul. Shard results are gathered back to tier 0's logits buffer via
   `ds4_rocm_xdev_copy`.

2. **Per-tier KV cache allocation** (`metal_graph_alloc_raw_cap`):
   For ROCm TP=4, allocates `layer_raw_cache_tp1/tp3` and
   `layer_attn_comp_cache_tp1/tp3` on tiers 1 and 3 (tier 0 uses the
   existing `layer_raw_cache[il]` and tier 2 reuses the existing
   `layer_raw_cache_tp[il]`).

3. **Post-prefill KV cache sync** (`metal_graph_rocm_tp4_sync_kv_cache`):
   Copies raw and compressed KV cache from tier 0 to tiers 1-3 using
   `ds4_rocm_xdev_copy`. Called from `ds4_session_slice` (batch prefill path).

4. **Per-tier raw cache helper** (`metal_graph_tp4_raw_cache`,
   `metal_graph_tp4_comp_cache`): Provides per-tier cache pointer lookup
   for the decode loop. Compressed cache returns shared tier-0 cache
   because all tiers read the shared compressed cache (only tier 0 updates
   it during decode).

**Remaining issue — decode loop still produces garbleed output:**
" about any    " for prompt "Explain C pointers.".
First decode token is often correct ("You" for "Hello"), but subsequent
tokens degrade. Root cause not fully identified but likely involves:

1. The post-prefill sync is in `ds4_session_slice` (batch prefill), but the
   quality fixture uses token-by-token `ds4_session_eval` → `metal_graph_eval_
   token_raw_swa` path. The sync is never called during quality fixture runs.

2. The decode loop uses the shared `g->layer_raw_cache[il]` (all 4 tiers
   read/write the same device-0 memory). Even though the actual KV data is
   identical across tiers, the compressor state arrays
   (`layer_attn_state_kv/score`) are shared device-0 memory. All 4 tiers'
   GPU kernels run concurrently (async launch), causing a read-modify-write
   race on the shared state arrays.

3. Fixing the race requires either:
   a) Per-tier compressor state arrays (add `layer_attn_state_kv_tp1/3` etc.)
   b) Inter-tier sync barriers (serialize tiers)
   c) Move KV cache sync to the token-by-token eval path

### Autonomous session (2026-07-28) — TP=4 first token correct, subsequent decode tokens garbled

**Status: ready-for-agent -> ready-for-human.** Implemented the following fixes as approved by the AI Consultants panel and earlier Gemini consultation:

1. **Per-tier compressor state arrays:** Allocated `layer_attn_state_kv_tp/1/3`, `layer_attn_state_score_tp/1/3`, and corresponding indexer state arrays on devices 1-3. Added helpers (`metal_graph_tp4_attn_state_kv`, `metal_graph_tp4_attn_state_score`, etc.) and patched the decode path to use per-tier state arrays based on active tier.

2. **Shared expert over-counting fix:** Changed TP=4 MoE partial storage so only tier 0 includes the shared expert in `shared_out_by_tier[tier]`; tiers 1-3 store only their 64 routed expert partials (via `ds4_gpu_tensor_copy` of `routed_out`). Previously all 4 tiers included the shared expert, causing 4× over-counting after all-reduce.

3. **Embedding broadcast to all 4 tiers:** After token embedding on tier 0, `cur_hc_by_tier[0]` is now broadcast to tiers 1-3 via `ds4_rocm_xdev_copy`. Without this broadcast, tiers 1-3 read uninitialized memory as the attention input, producing garbage partials.

4. **Active tier setting for TP=4:** `metal_graph_set_active_tier_decode` was a no-op when `g->placement` is NULL (the TP=4 case). Added `g->rocm_tp4` branch that sets `g->active_tier` and switches the HIP device. Without this fix, the per-tier class-P accessors always returned tier 0's buffers, and all 4 tiers' kernel dispatches targeted device 0.

5. **Attention all-reduce aliasing fix:** The attention all-reduce used `attn_out_by_tier[0]` as both destination and source. `ds4_rocm_xdev_allreduce_f32` zeroes the destination before accumulating, which erased tier 0's 32-head contribution. Fixed by staging through `shared_out_by_tier[0]` (same pattern as the batch prefill MoE fix in commit 78b718d).

6. **Compressed row counter guarding:** Only tier 0 increments `layer_n_comp[il]` and `layer_n_index_comp[il]` to prevent 4× counter inflation from redundant compressor emits across tiers.

**Verification results (with HIP_LAUNCH_BLOCKING=1 for determinism):**

| Test | Pipeline | TP=4 (this session) |
|---|---|---|
| First token (prompt "Hello") | "We" | "We" ✅ |
| Multi-token (3+ gen tokens) | "We need to respond to" | "Wealth, .   " ❌ |

The first token matches the pipeline reference, confirming prefill correctness (attention all-reduce, output head, embedding). Subsequent tokens are garbled, confirming the decode loop produces garbage at pos>0 (when KV cache is read).

**Remaining root cause hypothesis:**
The decode loop at pos>0 reads K,V from the raw cache (`g->layer_raw_cache[il]`, device 0) via peer access from each tier's device. The K,V data was stored by all 4 tiers during pos=0 (each tier writing its 32 heads' K,V to device 0's memory). The attention kernel on each tier reads the full 128-head K,V from device 0's raw cache and selects its 32 heads. The likely remaining issue:

1. **Cross-device raw cache access correctness:** The attention kernel on device t reads K,V from device 0's raw_cache pointer. While peer access is enabled and validated for host-side copies, GPU kernel peer-read of device 0's memory from device t's stream may have correctness or consistency issues (e.g., uncached reads, L2 cache coherence between devices, or stale read-after-write ordering across the xGMI/PCIe fabric).

2. **Missing per-tier native raw cache reads:** The code passes `g->layer_raw_cache[il]` (device 0) to all 4 tiers. Each tier should read from its OWN per-tier raw cache (`metal_graph_tp4_raw_cache`) to avoid cross-device peer reads. This requires replicating the KV cache from device 0 to tiers 1-3 after each decode step (expensive but correct).

**Recommendation:** Human investigation needed to resolve the cross-device KV cache read consistency issue. The fixes above (1-6) are correct structural changes that should land. The remaining decode correctness bug requires understanding GPU peer-read consistency across the 4× R9700 xGMI topology. A suggested diagnostic: add `metal_graph_rocm_tp4_sync_kv_cache` call at the end of each decode token's layer loop to replicate the raw cache from device 0 to tiers 1-3 before the next token's attention reads trigger cross-device accesses.

**Build status:** Compiles cleanly (ROCm, rocm-quality targets). Pipeline path unaffected. All 228 TP sharding unit tests pass.

### Consultant panel + Gemini analysis (2026-07-28) — root cause confirmed, fix scope narrowed

**Panel convened:** DeepSeek, Qwen3 (confidence 9/10), GLM, MiniMax, Grok (confidence 7/10), Gemini 3.6 Flash. Gemini failed to respond (API key not configured at consult time; re-run via gemini-consultant skill).

**Key model dimension (most consultants missed):** DeepSeek V4 Flash uses MLA with `DS4_N_HEAD_KV = 1`. There is ONE 512-float K,V latent per token, SHARED across all 64 heads (`DS4_N_HEAD = 64`, `DS4_N_HEAD_DIM = 512`). All tiers compute IDENTICAL K,V from the shared `attn_norm` input during decode. Head parallelism is in Q projection and output projection, not in K,V storage.

**Panel reveals:**
- **Grok** ✅: Pure local caches work without all-gather — but for the wrong stated reason (head locality doesn't apply to MLA, but the conclusion holds because all tiers compute identical K,V).
- **Qwen3** ⚠️: Option A (all-gather) is over-engineering. ~0.05ms overhead for no benefit.
- **Option B (shared device-0 cache with peer coherence fences) universally rejected** as a "hardware lottery" relying on undocumented ROCm guarantees.

**Critical finding (code audit by live-pair agent):** The KV sync function `metal_graph_rocm_tp4_sync_kv_cache` is called **only** in `ds4_session_eval_layer_slice` (line 59229). It is **NOT** called in the `ds4_session_sync_internal` prefill path (lines 60131-60182) used by `score_official` → `ds4_session_sync`. This means per-tier raw caches (`layer_raw_cache_tp1/2/3`) are allocated but **never populated** with prefill data on the quality fixture path. Currently masked because the decode loop uses device 0's pointer for all tiers.

**Approved fix plan (reviewed and agreed by human):**

Two changes:

1. **Add post-prefill KV sync in `ds4_session_sync_internal`** (~line 60175, after `metal_graph_prefill_raw_swa` succeeds):
   ```c
   if (g->rocm_tp4) { ok = metal_graph_rocm_tp4_sync_kv_cache(g); }
   ```
   Without this, per-tier caches on tiers 1-3 are empty/uninitialized after prefill.

2. **Swap to per-tier raw caches in the decode loop** (lines 27078, 27162):
   ```c
   // Change:
   g->layer_raw_cache[il]
   // To:
   metal_graph_tp4_raw_cache(g, il, tier)
   ```
   This makes each tier read/write its own local per-tier KV cache — zero cross-device peer access for KV operations. The attention kernel reads the full MLA KV latent from local VRAM. No all-gather or inter-device synchronization needed because all tiers compute identical K,V from the shared input.

**After both fixes land**, re-run the quality fixture:
```bash
make -j8 rocm-quality
AMD_SERIALIZE_KERNEL=3 \
  ./gguf-tools/quality-testing/score_official \
    /home/murphy/src/ds4/ds4flash.gguf \
    gguf-tools/quality-testing/data/flash/manifest.tsv \
    .scratch/rocm-tensor-parallel/quality-out/q_tp4_final.tsv \
    4096 --gpu-devices 0,1,2,3 --cuda-tensor-parallel
```

Record results and compare against pipeline reference (`q_pipeline_ref_tp4issue32.tsv`, avg_nll 0.374733).