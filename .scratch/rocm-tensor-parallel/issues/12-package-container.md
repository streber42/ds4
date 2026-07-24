# Package ported build into container workflow

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/PRD.md`

## What to build

Fold the tensor-parallel build into the container workflow already used to run this model, so
starting the server stays a single command and the operator's existing tooling keeps working
unchanged.

The image must be built on the target host so that CPU code generation matches this machine —
a prebuilt image compiled for a different CPU generation fails immediately with an illegal
instruction, which has already bitten this project once and is worth encoding in the build
setup rather than rediscovering.

The served endpoint, port, and model name must not change: existing clients point at the same
place and should not need to know whether the backend is running tensor-parallel or pipeline.

## Acceptance criteria

- [ ] Container image builds from the ported tree, on the target host, reproducibly
- [ ] Compose configuration brings up the tensor-parallel deployment with a single command
- [ ] The served endpoint, port, and model name are unchanged from the current deployment
- [ ] A health check gates readiness so the stack reports healthy only when actually serving
- [ ] An end-to-end request through the container returns correct output
- [ ] Throughput through the containerised path is measured and matches the bare-metal result
- [ ] The build documents why it must be built on the target host (CPU instruction-set match)
- [ ] Switching back to the pipeline deployment remains possible

## Blocked by

- `.scratch/rocm-tensor-parallel/issues/10-quality-fixture-validation.md`
