# 18 — Fix cross-device dispatch race (no more `AMD_SERIALIZE_KERNEL=3`)

**What to build:** TP and pipeline produce correct multi-GPU output without `AMD_SERIALIZE_KERNEL=3` or `HIP_LAUNCH_BLOCKING=1`. The quality fixture scores match the serialized baseline on a default un-shimmed run.

The dispatch race lives in `ds4_rocm_xdev.cu` at the peer-copy sync point. Before reading a remote GPU's memory via peer copy, `ds4_rocm_xdev_copy` issues `hipDeviceSynchronize()` on the source device. On gfx1201's hardware scheduler (HWS), this does not guarantee that the specific stream producing the data has drained — the HWS can reorder across the implicit default-stream barrier, so the peer copy can read stale or partially-written memory.

Fix: replace `hipDeviceSynchronize()` with explicit `hipEventRecord`/`hipStreamWaitEvent` ordering between the actual producer stream and the peer-copy stream. This gives the HWS a proper happens-before edge instead of a blunt device-wide flush that it can reorder past.

There is also a second site — the pipeline tier handoff in `metal_graph_set_active_tier_decode` / `metal_graph_set_active_tier_batch` (`ds4.c`), which calls `ds4_gpu_tensor_copy_xdev` for cur_hc tier-to-tier hops. That goes through the same `ds4_rocm_xdev_copy` path, so a single fix in the xdev module covers both.

## Blocked by

None — can start immediately.

## Status

ready-for-agent

- [ ] `ds4_rocm_xdev_copy` (and `_accumulate_f32`/`_accumulate_f16`) uses `hipEventRecord` + `hipStreamWaitEvent` per peer copy instead of `hipDeviceSynchronize()`
- [ ] Quality fixture (`make rocm-quality`, TP mode) matches the serialized baseline (`avg_nll ~0.370`, `first_match ~68/100`) without `AMD_SERIALIZE_KERNEL=3`
- [ ] `ds4-bench` 4-GPU TP default run shows throughput at or above the serialized baseline from #19
- [ ] No regression in pipeline mode (quality fixture or bench)
