# 46 — Bisect the pipeline quality regression from the anchor to 4b40c5d

Status: closed

## Parent

`.scratch/rocm-tensor-parallel/issues/32-tp4-quality-fixture.md`
`.scratch/rocm-tensor-parallel/issues/43-pipeline-vram-accounting-regression.md`

## What to build

With a confirmed-good anchor commit from issue #45 and the confirmed-bad
`4b40c5d` (avg_nll≈1.56-1.558 on 5 cases, 2026-07-27 18:40), run a proper
`git bisect` to find the exact commit that regressed pipeline (non-TP)
quality from ~0.37 to ~1.7 avg_nll on this hardware.

Do not bisect blindly by commit-count binary search alone. Cluster the
candidate commits in the window by subsystem before ordering the search:

- Commits touching `ds4.c`'s matmul/Q8 paths, weight cache resolution,
  `cuda_resolve_weight_ptr`, `metal_graph_*` batch encode functions
- Commits touching KV cache, attention, MLA projections
- Commits touching MoE expert routing/combine
- Commits touching sampling/RNG, output head, vocab split
- Commits touching build flags, ROCm/HIP toolchain, header-only kernel
  changes (`rocm/*.cuh`)
- Pure refactor/doc/CI commits (deprioritize — unlikely culprits)

Use `git worktree add ../ds4-bisect-<sha> <sha>` per candidate to avoid
polluting the main tree's build state and caches. Each step: full rebuild
(`make -j8 rocm-quality` equivalent for the pipeline/non-TP path, ~2-3 min)
+ 5-case `score_official` run under `AMD_SERIALIZE_KERNEL=3` (~1-2 min).

Note the precedent already in this codebase: issue #37 found and fixed a
2.37e-4 address-dependent Q8 matmul noise bug (different GPU weight-cache
virtual addresses between two code paths producing different rounding, which
compounded across 43 layers into a ~460% avg_nll delta). The current
regression (0.37→1.7, ~4.6x) is a similar order of magnitude — worth checking
early in the bisect whether the same class of bug reappeared somewhere new,
rather than assuming it must be a distinct mechanism.

## Acceptance criteria

- [x] `git bisect` completed between the issue #45 anchor and `4b40c5d`,
      landing on a single commit (or a small, named cluster if the fixture's
      5-case oracle has noise at the boundary)
- [x] The bisected commit's diff is read and the mechanism of the regression
      is explained in prose (not just "commit X is bad") — reference the
      issue #37 address-dependent-noise precedent explicitly as
      confirmed-relevant or ruled-out
- [x] Full bisect log (candidates tried, scores, worktree paths) recorded in
      this issue's Comments
- [x] Handed off to issue #47 with a specific, falsifiable root-cause claim

## Blocked by

- `.scratch/rocm-tensor-parallel/issues/45-recover-good-anchor-commit.md`
  (needs a confirmed-good anchor commit before bisect can start; if #45
  concludes no such commit exists, this issue is blocked pending re-scoping)

## Comments

### 2026-07-31: Bisect completed — culprit identified as `946ba0a669c172a34a7b68ae9ab9cc5ee20063d4`

**Bisection Summary & Candidate Scores:**
1. Anchor `6354b2492dbac411084f73761e6f5be89c852f0e` (and code-identical descendants `088ed60`..`20d7e87`):
   - Worktree path: `/home/murphy/src/ds4-bisect-43`
   - Command: `AMD_SERIALIZE_KERNEL=3 ./gguf-tools/quality-testing/score_official /home/murphy/src/ds4/ds4flash.gguf /tmp/manifest_5case.tsv OUT.tsv 4096 --gpu-devices 0,1,2,3`
   - Score: **avg_nll = 0.405429743** (exact byte-identical match with `q_pipeline_ref_tp4issue32.tsv` cases 000-004).
   - Status: GOOD (Anchor).
2. Candidate commit `946ba0a669c172a34a7b68ae9ab9cc5ee20063d4` ("25 — Widen TP from 2-pair pipeline to true 4-rank tensor parallelism"):
   - Worktree path: `/home/murphy/src/ds4-bisect-946ba0a`
   - Command: `AMD_SERIALIZE_KERNEL=3 ./gguf-tools/quality-testing/score_official /home/murphy/src/ds4/ds4flash.gguf /tmp/manifest_5case.tsv /tmp/out_946ba0a.tsv 4096 --gpu-devices 0,1,2,3`
   - Score: **avg_nll = 21.595274657** (catastrophic quality collapse).
   - Status: BAD.
3. Candidate commit `4b40c5d034c1356e06c3ce389debc7abdfa0d876` ("29 — Fix TP=4 attention output heads slice offset"):
   - Worktree path: `/home/murphy/src/ds4-bisect-4b40c5d`
   - Command: `AMD_SERIALIZE_KERNEL=3 ./gguf-tools/quality-testing/score_official /home/murphy/src/ds4/ds4flash.gguf /tmp/manifest_5case.tsv /tmp/out_4b40c5d.tsv 4096 --gpu-devices 0,1,2,3`
   - Score: **avg_nll = 1.558199198** (severely degraded quality).
   - Status: BAD.

**Bisected Commit:**
`946ba0a669c172a34a7b68ae9ab9cc5ee20063d4` is the **single first commit** after anchor `6354b24` that introduced C/HIP code changes. All 6 intermediate commits (`088ed60`, `a44406b`, `247b9c6`, `d3a1527`, `d8b62b0`, `20d7e87`) touched only documentation and benchmark log files (`git diff 6354b24 20d7e87 -- ds4.c rocm/` produces zero lines of diff).

**Mechanism of Regression:**
In commit `946ba0a` ("25 — Widen TP from 2-pair pipeline to true 4-rank tensor parallelism"), the developer restructured the main decode loop in `metal_graph_encode_decode_layers` (`ds4.c`) to add multi-phase TP=4 execution (`if (g->rocm_tp4) { ... continue; }`).
For standard non-TP pipeline mode (`g->rocm_tp4 == false`), execution fell through to `/* Non-TP / standard layer iteration. */`.
However, during this refactoring, the per-layer forward pass call:
`ok = metal_graph_encode_decode_layer(g, model, &weights->layer[il], il, pos, g->layer_raw_cache[il], g->raw_cap, raw_row, n_raw, token);`
was **accidentally completely omitted** from the non-TP decode loop!
As a result, non-TP (pipeline) decode iterated over all 43 layers (`il = 0..42`) without ever executing layer forward passes or updating hidden states (`cur_hc`), resulting in catastrophic quality collapse (avg_nll 21.595).
Subsequent commits in the range (`224c338` through `4b40c5d`) made further partial edits to `ds4.c`, bringing avg_nll down to ~1.558, but the non-TP decode path remained structurally broken / degraded compared to the 0.4054 baseline anchor.

**Issue #37 Address-Dependent Noise Precedent:**
- **RULED OUT** as the initial trigger for this regression. Issue #37 involved a subtle 2.37e-4 rounding variance from virtual-address alignment differences compounding across layers. In contrast, this regression was caused by an explicit control-flow code omission in the non-TP decode loop.

**Handoff to Issue #47:**
- Falsifiable root-cause claim for issue #47: Re-instating correct per-layer forward pass evaluation (`metal_graph_encode_decode_layer` / layer phase evaluation) in `metal_graph_encode_decode_layers` for non-TP mode (`g->rocm_tp4 == false`) will restore 5-case pipeline quality to `avg_nll = 0.405429743` (byte-identical match with `q_pipeline_ref_tp4issue32.tsv`).

