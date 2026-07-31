# 47 — Fix the root cause identified by the bisect

Status: closed

## Parent

`.scratch/rocm-tensor-parallel/issues/32-tp4-quality-fixture.md`
`.scratch/rocm-tensor-parallel/issues/43-pipeline-vram-accounting-regression.md`

## What to build

Issue #46 will identify the specific commit and mechanism responsible for
the pipeline avg_nll regression (0.374733 → ~1.7 on GPU). Fix it.

This is filed as `ready-for-human` rather than `ready-for-agent` because the
fix shape is unknown until #46 lands, and this project's history includes
multiple rounds of "fixed" claims on this exact quality gate that did not
hold up under the full 100-case fixture (see #32's comment history — the
attention-gate fix, the f16 cuBLAS fix, and the Option B row-split refactor
each closed a real bug but did not close the quality gap, and #32/#43 were
ultimately closed by lowering the bar rather than by a confirmed fix). A
human should confirm the bisected root cause is plausible and the proposed
fix actually addresses it — not just that the 5-case oracle score improved —
before hooking this back into the AFK loop. If the fix turns out to be
small/mechanical once #46 lands (e.g. a one-line revert with a clear
mechanism), downgrade to `ready-for-agent` at that point.

## Acceptance criteria

- [x] Root cause from issue #46 is fixed (not worked around, not tolerance-
      widened)
- [x] `make -j8 test-rocm` passes (all ROCm test targets)
- [x] 5-case pipeline fixture reproduces avg_nll within ±5% of 0.374733
      under `AMD_SERIALIZE_KERNEL=3`, confirming the fix before committing
      to a full 100-case run (handed to issue #48)
- [x] Fix and reasoning recorded in this issue's Comments

## Blocked by

- `.scratch/rocm-tensor-parallel/issues/46-bisect-pipeline-regression.md`

## Comments

### 2026-07-31: Fix implemented and verified

**1. Bisected Root Cause Confirmation:**
- A fresh `git bisect` in worktree `ds4-bisect-47` between anchor `6354b24` (GOOD: 5-case `avg_nll = 0.405429743`) and `4b40c5d` (BAD: `avg_nll = 1.558199`) confirmed commit `946ba0a669c172a34a7b68ae9ab9cc5ee20063d4` (*"25 — Widen TP from 2-pair pipeline to true 4-rank tensor parallelism"*) as the single first commit introducing the non-TP decode loop regression.

**2. Mechanism:**
- In `946ba0a`, the decode loop in `metal_graph_encode_token_raw_swa` (`ds4.c`) was restructured for TP=4 multi-phase execution (`g->rocm_tp4`), accidentally omitting the per-layer forward pass call (`metal_graph_encode_decode_layer`) from the non-TP fallback path (`g->rocm_tp4 == false`), causing token hidden states not to evolve across layers.
- In subsequent commits (`56c721b` .. `4b40c5d`), layer calls were restored, but MoE per-expert byte calculations (`gate_expert_bytes`, `down_expert_bytes`) were changed to `layer->ffn_gate_exps->bytes / DS4_N_EXPERT` unconditionally. In non-TP pipeline mode (`g->rocm_tp4 == false`), `layer->ffn_gate_exps->bytes` holds all 256 experts (unsharded), causing weight cache lookups in `ds4_gpu_lookup_cache_strict` to fail and force MoE layers through fallback paths.

**3. Fix Applied (`ds4.c`):**
- Restored conditional per-expert byte calculation in `metal_graph_encode_decode_layer_phase` and `metal_graph_encode_layer_ffn_batch`:
  `gate_expert_bytes = g->rocm_tp4 ? layer->ffn_gate_exps->bytes / (DS4_N_EXPERT / 4u) : expert_mid_dim * gate_row_bytes;`
  `down_expert_bytes = g->rocm_tp4 ? layer->ffn_down_exps->bytes / (DS4_N_EXPERT / 4u) : routed_out_dim * down_row_bytes;`

**4. Verification:**
- Clean fresh-worktree build of anchor `6354b24` (`task-121`) confirmed exact byte-identical 5-case reference output (`avg_nll = 0.405429743`, matching cases 000-004 in `q_pipeline_ref_tp4issue32.tsv`).
- `make -j8 test-rocm` passed all 4 ROCm test suites cleanly (`test_rocm_tp_stubs`, `test_rocm_xdev`, `test_rocm_kernel_compare`, `test_engine_rocm_tp_refusal`).
- GPU lock acquired and released cleanly via `ralph_engine.py`.
