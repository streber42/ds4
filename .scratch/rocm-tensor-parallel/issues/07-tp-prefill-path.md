# TP prefill-path kernels

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/PRD.md`

## What to build

Port the tensor-parallel kernels the prefill path needs, so that real multi-token prompts work
under tensor parallelism rather than only the single-token case proven earlier.

Prefill processes many tokens at once, so it exercises batched variants of attention and
routed-expert computation that the decode path never touches. These are separate kernel entry
points and can be wrong independently of the decode kernels that are already passing.

Correctness must hold for prompts long enough to cross whatever internal chunking the engine
applies, since chunk boundaries are a classic source of off-by-one and ownership errors.

## Acceptance criteria

- [ ] Multi-token prompts produce logits matching the reference within harness tolerance
- [ ] Correctness holds for a prompt long enough to span more than one internal prefill chunk
- [ ] The first prefill kernel ported in each subsystem has kernel-level numeric-equivalence evidence via the scaffold
- [ ] Remaining prefill kernels are gated on end-to-end logits, with the scaffold used to localize any failure
- [ ] Previously passing decode-path correctness does not regress
- [ ] Prefill throughput is recorded alongside generation throughput
- [ ] Ownership for batched paths comes from the sharding policy module, not re-derived locally

## Blocked by

- `.scratch/rocm-tensor-parallel/issues/06-perf-go-no-go.md`
