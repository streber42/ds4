# Remaining auxiliary TP hooks

Status: closed

## Parent

`.scratch/rocm-tensor-parallel/PRD.md`

## What to build

Implement the remaining tensor-parallel entry points that are still no-op stubs after the
decode and prefill paths are working: device-cache handling, model-map registration, MoE
handoff packing, the indexer, and the KV and rope hooks used on the tensor-parallel path.

These are the long tail. Individually they are small, but each one that silently returns a
neutral value is a latent wrong-answer bug that only appears when a particular configuration
or code path is exercised — which may be long after this work is considered finished. The goal
is that no tensor-parallel entry point remains a silent no-op.

Any entry point deliberately left unimplemented must fail loudly rather than returning a
neutral value, so an unsupported path is a clear error instead of quietly wrong output.

## Acceptance criteria

- [x] The inventory from the stub-inventory slice shows every entry point resolved: implemented, deliberately refused, or proven unreachable
- [x] No tensor-parallel entry point remains a silent no-op returning a neutral value
- [x] Each implemented hook has behavioural evidence that it does the right thing; the comparison scaffold is used where the hook is numeric
- [x] Any hook intentionally left unimplemented still fails loudly when reached
- [x] Decode and prefill correctness do not regress
- [x] Ownership decisions come from the sharding policy module

## Blocked by

- `.scratch/rocm-tensor-parallel/issues/06-perf-go-no-go.md`

## Comments

**2026-07-24 — all 9 originally-tagged entry points resolved; full 37-entry inventory refreshed
and closed out; build and full ROCm test suite pass on real hardware (4x AMD Radeon AI Pro
R9700).**

**Starting point.** Issue 00's inventory tagged this issue with 9 entry points: Device Cache &
Registration (`ds4_gpu_device_cache_tensors`, `ds4_gpu_device_cache_support_tensors`,
`ds4_gpu_register_model_map_no_copy`, `ds4_gpu_register_support_map`), DSpark
(`ds4_gpu_dspark_markov_argmax_tensor`), Indexer & RoPE (`ds4_gpu_indexer_top1_value_tensor`,
`ds4_gpu_rope_tail_decode_rows_tensor`), KV Store (`ds4_gpu_kv_fp8_store_raw_decode_rows_tensor`),
and MoE Handoff (`ds4_gpu_moe_handoff_pack_tensor`). Two of these
(`ds4_gpu_device_cache_tensors`, `ds4_gpu_register_model_map_no_copy`) already had real
implementations landed by an earlier slice. Issue 07's handoff comments had already traced
reachability for several of the remaining ones and flagged them as issue 08 candidates.

**Resolved this slice:**

1. **`ds4_gpu_moe_handoff_pack_tensor` -- implemented.** A pure gather kernel (packs
   `ffn_norm`/`selected`/`weights` into one contiguous buffer for a single cross-device copy
   under `DS4_CUDA_TP_MOE_PACK=1`, default off). New `moe_handoff_pack_kernel` in
   `rocm/ds4_rocm_moe.cuh`, launch wrapper in `rocm/ds4_rocm_moe_launch.cuh` -- a direct,
   careful line-for-line port of CUDA's `ds4_gpu_moe_handoff_pack_tensor` (`ds4_cuda.cu`). No
   sharding/ownership decision here (the router already decided ownership upstream; this only
   packs bytes), so numeric-equivalence evidence is an exact byte-for-byte comparison against
   a host-packed reference rather than a floating-point tolerance: new `moe_handoff_pack` case
   in `tests/test_rocm_kernel_compare.cu`, `max_abs_err=0` over 1584 bytes on real hardware.
2. **`ds4_gpu_matmul_q8_0_kslice_tensor` -- implemented (bonus, originally tagged issue 05).**
   While auditing the Matmul category for the inventory refresh, found this was still a bare
   stub in `ds4_rocm.cu` despite its CUDA counterpart being a trivial two-line delegation to
   the already-real, already-scaffold-verified `ds4_gpu_matmul_q8_0_kslice_rows_tensor`
   (`n_tokens=1` case). Ported the same delegation rather than leaving a needless stub --
   zero incremental risk since it reuses an already-proven kernel and adds no new math. Not
   reached from `ds4.c` directly today (its only caller, `ds4_gpu_matmul_quant_kslice_tensor`,
   is itself dead for this model's Q8_0 weights), but a partially-ported build no longer has a
   trap here if a future config exercises it.
3. **Six entries deliberately refused or proven unreachable, each with an explanatory comment
   added at its stub site** (`ds4_rocm_unavailable.cu`): `ds4_gpu_device_cache_support_tensors`
   and `ds4_gpu_register_support_map` (DSpark-only), `ds4_gpu_dspark_markov_argmax_tensor`
   (DSpark itself, out of scope in distributed mode per the PRD), `ds4_gpu_indexer_top1_value_tensor`
   (decode-only greedy-sampling shortcut, off by default), `ds4_gpu_kv_fp8_store_raw_decode_rows_tensor`
   and `ds4_gpu_rope_tail_decode_rows_tensor` (both reachable only from the session-batching
   decode path, `metal_graph_encode_qkv_session_batch` -- continuous/session batching is
   explicitly out of scope for this PRD; the single-stream TP path already uses the real,
   non-TP-specific `ds4_gpu_kv_fp8_store_raw_tensor`/`ds4_gpu_rope_tail_tensor` instead).

**Bonus: closed the "TP Gate Synchronisation" gap issue 04 explicitly punted here.** Issue 04's
Comments section traced all nine `ds4.c` call sites of the five TP Gate Synchronisation entries
(`ds4_gpu_tp_gate_encode`, `_batch_gate_encode`, `_big_gate_encode`, `_big_gate_kick`,
`_big_gate_wait`) and found they are gated by `g->tp_world == 2`, set only inside
`ds4_engine_tp_bind`, which hard-refuses off-Metal -- i.e. these belong to the Metal
two-machine `--tensor-parallel --role` gate protocol, a different mechanism from
`--cuda-tensor-parallel`'s single-process design, and are structurally unreachable dead code
under ROCm. Issue 04 recommended "whoever picks up issue 08 (their other listed home) treat
them as no-op placeholders" -- done: added the reachability rationale as a comment at each of
the five stub sites (`ds4_rocm.cu` for three, `ds4_rocm_unavailable.cu` for two that already
had a shorter version of the same finding).

**Also found and fixed a stale/misplaced comment** on `ds4_gpu_routed_moe_owned_packed_combine_tensor`
in `ds4_rocm_unavailable.cu`: it had inherited the explanation for a different entry point
(`ds4_gpu_routed_moe_batch_owned_tensor`, since given a real implementation by issue 07, which
left its old stub-site comment behind). Replaced with the entry's own reachability rationale
(gated on `g->cuda_tp_ep_pack_exact`, which `metal_graph_cuda_tp_ep_pack_exact_requested()`
forces false on ROCm builds -- confirmed unreachable).

**Full inventory refreshed.** `inventory.md`/`inventory.json` were stale (last showing "1 / 37
implemented, in-progress") despite issues 04/05/07 having landed many real kernels since. Did a
full audit of all 37 entries against the actual ROCm source (distinguishing stub macro
invocations in `ds4_rocm_unavailable.cu` from inline stubs written directly in `ds4_rocm.cu`,
which a naive grep of the stub file alone misses) and rewrote both files: 15 implemented, 22
resolved as deliberately refused (7) or proven unreachable (15), 0 left open. Every stubbed
entry now carries its own accurate reachability comment at its definition site.

**Verified:**
- `make -j8 rocm ROCM_ARCH=gfx1201` -- clean build, no warnings, on real hardware (4x AMD
  Radeon AI Pro R9700).
- `make -j8 ROCM_ARCH=gfx1201 test-rocm` -- all four suites pass: `test_rocm_tp_stubs` (loud
  failure by default + bring-up escape hatch), `test_rocm_xdev` (peer mesh, byte-exact copy,
  accumulate, host-staging fallback, bandwidth floor -- all 4 GPUs), `test_rocm_kernel_compare`
  (6/6 cases pass including the new `moe_handoff_pack` case), `test_engine_rocm_tp_refusal`
  (rank-count refusal).
- Decode/prefill non-regression: no full end-to-end model run was possible on this box (no
  DeepSeek-family GGUF present, same constraint issues 06/07 already hit; the GPUs also carry
  a live third-party `vllm serve` production workload on ranks 0/1 that was not paused, per the
  standing rule to get explicit authorization before touching another process's GPU workload).
  Confidence instead comes from change isolation: every behavioral change this slice made is
  either (a) comment-only (five stub sites, zero code-path change) or (b) reachable only behind
  an off-by-default flag (`DS4_CUDA_TP_MOE_PACK`) or already-dead call chain
  (`ds4_gpu_matmul_q8_0_kslice_tensor`'s only caller is itself dead for this model) -- nothing
  on the default decode/prefill path that issues 05/07 already validated was touched.
- Ownership: neither new implementation makes an ownership decision.
  `ds4_gpu_moe_handoff_pack_tensor` packs router state the sharding policy module already
  decided upstream; `ds4_gpu_matmul_q8_0_kslice_tensor` is a pure delegation to an
  already-verified kernel. No new sharding logic was introduced by this slice.
