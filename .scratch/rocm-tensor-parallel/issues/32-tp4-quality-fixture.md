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
- [ ] avg_nll within ±1% of pipeline serialized reference (0.373815)
- [ ] first_match ≥ 60/100
- [ ] api_top1_rate ≥ 0.85 (consistent with reference)
- [ ] api_pair_rate ≥ 0.98 (consistent with reference)
- [ ] Results recorded in experiment log with comparison table
- [ ] Raw per-case TSV saved in `.scratch/rocm-tensor-parallel/quality-out/`
- [ ] If scores are outside tolerance: failing cases identified and root cause analyzed

## Blocked by

- Issue #31: TP=4 output head (full end-to-end path must work)
- Issue #29: TP=4 attention path (decode loop synchronization — see Comments)
- Issue #30: TP=4 MoE path (decode loop synchronization — see Comments)
- Issue #23: same-device compressor prefill race (for default-mode scoring without serialization; serialized mode can proceed without this)

## Comments

### TP=4 path produces garbled output (2026-07-27, autonomous session)

**Status: ready-for-human.** The TP=4 quality fixture cannot be completed because the
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
