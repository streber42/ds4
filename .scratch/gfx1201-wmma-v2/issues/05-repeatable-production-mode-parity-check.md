Status: ready-for-agent
# 05 — Make the production-mode parity check a repeatable target

**What to build:** Turn the one-off correctness check from issue 03 into a
script/make target anyone can re-run. Narrow scope by design — see "Scope" below.

**Blocked by:** Issue 03 (this packages the check that issue 03 performs by hand).

## Why

Issue 03 exposed a structural gap: `matmul_q8_0_f32_batch_wmma_4w_kernel` shipped
wrong in this tree from commit `79f09d4` and no test caught it, because the
authoritative correctness gate (`score_official`) sets `g_quality_mode`, and
`rocm/ds4_rocm_matmul.cuh:404` gates this kernel on `!g_quality_mode`. The fixture
structurally cannot see the kernel it needs to validate.

The same is true of a handful of neighbouring fast paths selected by the same
flag (`g_rocm_cfg.attention_output_cublas_all`, `shared_down_cublas`,
`disable_splitk_attn_out_low`, and others set at
`rocm/ds4_rocm_runtime.cuh:5061-5125`).

## Scope

**In scope:** a repeatable check that a production-mode (`g_quality_mode = 0`)
forward pass agrees with a trusted reference on the same prompt.

**Explicitly out of scope:** auditing or changing the ~20 `g_quality_mode` call
sites. Most are legitimate precision knobs — TF32 off, `CUBLAS_DEFAULT_MATH` — and
that is standard, deliberate practice for a reference path, not a bug. Do not
turn this into a `g_quality_mode` refactor. If the audit looks tempting, file a
separate issue and leave it for a human.

## Acceptance criteria

- [ ] A script or make target (e.g. `make rocm-parity-prod`) that, for a fixed
      prompt of >256 tokens, runs a production-mode forward pass under both the
      WMMA and `DS4_ROCM_NO_WMMA` builds and reports argmax agreement, top-5
      agreement, and mean/max absolute logit error.
- [ ] It exits non-zero on divergence beyond a documented threshold, with the
      threshold justified from issue 03's measured fp16 accumulation noise (not
      picked arbitrarily).
- [ ] Verified to **fail** if pointed at a deliberately broken kernel — e.g. by
      temporarily reintroducing the `79f09d4` gfx12 body. A parity check never
      demonstrated to fail is not evidence of anything.
- [ ] Documented in `## Answer`: how to run it, what it covers, and — stated
      plainly — which production fast paths it does *not* cover.

## Notes for the agent

- Requires the GPU lock for the verification runs (`AGENTS.md`).
- Reuse issue 03's harness rather than inventing a new one.
- Keep it cheap enough to actually get run: one prompt, one comparison. This is
  a smoke gate, not a second quality fixture.

## Blocked by

`.scratch/gfx1201-wmma-v2/issues/03-fix-broken-gfx12-wmma-in-this-repo.md`

## Answer

## Comments

Filed 2026-08-11 from an interactive human session.
