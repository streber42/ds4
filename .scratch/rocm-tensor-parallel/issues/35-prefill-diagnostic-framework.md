# 35 — Prefill per-layer diagnostic framework

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/issues/32-tp4-quality-fixture.md`

## What to build

A diagnostic framework that dumps per-layer hidden states (post-attention and post-FFN) from both the pipeline and TP=4 paths, then diffs them to identify the exact layer and tensor where the prefill divergence first occurs.

The framework needs three pieces:

1. **Dump triggering:** A small code change in the batch prefill layer loop that writes per-layer tensors to disk when an env var is set. The env var `DS4_METAL_GRAPH_DUMP_PREFIX` already exists and controls a per-layer tensor dump path (`metal_graph_debug_dump_tensor` calls throughout `ds4.c`). Verify that setting this env var during a prefill-only run (`-n 1`) produces per-layer `after_attn_hc` and `batch_routed_out` dumps on the TP=4 path. If those specific tensors aren't already covered by existing dump calls, add them. The goal is two directories per run: `<prefix>/pipeline/` and `<prefix>/tp4/` each containing `il_<N>_after_attn_hc.bin` and `il_<N>_routed_out.bin` for all 43 layers.

2. **Diff script:** A small script (bash + Python or pure bash) in `.scratch/rocm-tensor-parallel/scripts/diff-layers.sh` that takes two directories and computes per-layer max-absolute-error between corresponding tensors, printing a table like:
   ```
   il=0  after_attn_hc max_err=1.2e-07  routed_out max_err=3.4e-06
   il=1  after_attn_hc max_err=0.00015  routed_out max_err=0.82  ***
   il=2  after_attn_hc max_err=0.12     routed_out max_err=15.3  ***
   ```
   Tolerances: `after_attn_hc` error ≤ 1e-3 and `routed_out` error ≤ 1e-3 pass. Mark any row where either exceeds tolerance with `***`. Exit non-zero if any layer fails.

3. **Runner script:** `.scratch/rocm-tensor-parallel/scripts/diagnose-prefill.sh` that accepts a prompt string, runs both pipeline and TP=4 prefill with dump enabled, invokes the diff script, and reports the first-divergent layer. Must accept `--gpu-devices` and `--model` arguments.

**Constraints:**
- Binary tensor dumps are single-precision float arrays with no header (compatible with `write_f32_binary_file`). Read with `od -t f4` or Python `numpy.frombuffer(np.float32)`.
- The dump must capture the **same tensor** at the **same point in the computation** for both runs (e.g., after the HC expand, not after the cur_hc/next_hc swap).
- The framework must work for any prompt, not just hardcoded test cases.

## Acceptance criteria

- [ ] `DS4_METAL_GRAPH_DUMP_PREFIX` produces per-layer `after_attn_hc` and `batch_routed_out` binary dumps on the TP=4 batch prefill path for all 43 layers
- [ ] `scripts/diff-layers.sh <dir1> <dir2>` produces a per-layer max-error table and exits non-zero if any layer exceeds the 1e-3 tolerance
- [ ] `scripts/diagnose-prefill.sh --model <path> --prompt "text"` runs both paths, diffs them, and reports the first-divergent layer
- [ ] The framework works on the pipeline path (4-GPU pipeline placement, no `--cuda-tensor-parallel`) and the TP=4 path (`--cuda-tensor-parallel`)
- [ ] Per-layer dump overhead is minimal enough that a full 43-layer, 1-token prefill completes in under 60 seconds

## Blocked by

None — can start immediately.
