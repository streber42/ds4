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
- Issue #23: same-device compressor prefill race (for default-mode scoring without serialization; serialized mode can proceed without this)
