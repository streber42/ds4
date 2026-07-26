## ROCm/gfx1201 findings

- [ROCm 7.14 does not fix the dispatch race](memory/roc-7.14-dispatch-race.md) — HWS issue on gfx1201, not a HIP runtime bug
- [gfx1201 TP decode profiling](memory/gfx1201-profiling.md) — MoE hot path dominates at 57%, ~2000 kernel dispatches/token, 2-pair pipeline caps TP at ~50%
- [Ticket plan](memory/ticket-plan.md) — #18 Fix dispatch race, #19 Re-measure throughput, #20 MoE optimization, #21 PoC doc/tag, #22 Launch overhead reduction
- [Sandbox GPU access limitation](memory/sandbox-gpu-access-limitation.md) — Failed sandbox experiment; now running outside sandbox with full ROCm/GPU access