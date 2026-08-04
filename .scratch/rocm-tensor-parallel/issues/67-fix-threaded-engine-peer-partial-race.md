# 67 — Fix the threaded-engine cross-layer peer-partial buffer race, re-enable `DS4_TP4_THREADED_LAYERS`

Status: ready-for-human

## Parent

`.scratch/rocm-tensor-parallel/issues/66-bisect-tp4-quality-divergence-49-61.md`

## What to build

This is the parked re-entry point for dense-TP=4 performance work. Do not pick
this up without first reading ADR `docs/adr/0001-dense-tp4-parked-sequential-default.md`
— the project was deliberately parked on 2026-08-04 with correctness delivered
and throughput ~10× short of the PP=4 baseline, and the analysis there says
fixing this race alone is unlikely to close that gap (the racy engine only
measured 2.01 t/s vs 28 t/s pipeline). Reopen only alongside a plan that
attacks the 86-all-reduces-per-token synchronization count itself.

The defect, per the #66 bisect: the persistent-thread execution engine
(#50, rolled out by #53/#60) has a cross-layer data race on the per-tier
peer-partial buffers. The threaded all-reduce reads
`attn_out_by_tier[peer]` / `shared_out_by_tier[peer]` through
`ds4_rocm_xdev_copy`, whose event fence only orders each copy against the
peer's stream state at *issue* time — it cannot prevent the peer from
overwriting that single reused buffer with the next layer's partial before a
lagging rank's copy has read it. Result: systematic logit shift
(target_mean_delta ≈ −13 across all 129280 logits) with run-to-run variance.
The mitigation shipped by #66 is `metal_graph_tp4_spike_layer_enabled`
defaulting to 0 (legacy sequential path); the threaded path remains reachable
via `DS4_TP4_THREADED_LAYERS=N` as a diagnostic switch.

Sketched fixes from #66, in preference order:

1. **Per-layer double-buffering** of the peer-partial tensors (each layer
   writes to `buf[layer & 1]`, so a lagging peer's read of layer N cannot be
   clobbered by the owner's write of layer N+1).
2. **Per-layer handshake** — a rank may not begin writing its layer-N+1
   partial until all peers acknowledge having consumed its layer-N partial.

## Acceptance criteria

- [ ] Root-cause fix landed (not a serialization workaround); the mechanism
      that prevents the cross-layer overwrite is documented in the code
- [ ] With `DS4_TP4_THREADED_LAYERS=43`, case_000 smoke avg_nll is in the
      0.37–0.44 band that the legacy path measures (not 13–17), across ≥3
      consecutive runs (the race showed run-to-run variance, so one clean run
      proves nothing)
- [ ] Full 100-case `score_official` TP=4 fixture under the threaded engine
      passes the PRD bar (avg_nll 0.370–0.378 band / parity with pipeline,
      first_match ≥60/100, api_top1_rate ≥0.85, api_pair_rate ≥0.98)
- [ ] Threaded-vs-sequential decode throughput A/B measured at the issue #33
      bench config and recorded honestly in `experiment-log.md`
- [ ] Decision recorded: flip the default to threaded, or keep sequential —
      with the measured numbers as justification (supersede or amend ADR 0001)
- [ ] `make -j8 test-rocm` passes

## Blocked by

*(None mechanically. Held at `ready-for-human` deliberately: the project is
parked per ADR 0001 and this must not enter the unattended queue without a
human first deciding the sync-count analysis question documented there.)*

## Comments

**2026-08-04 — Filed as part of the project park (wrap-up of #49–#66).**
Carries forward the follow-up work item explicitly left open by #66's closure.
