# 27 — TP=4 all-reduce primitive

Status: ready-for-agent

## Parent

`.scratch/rocm-tensor-parallel/issues/25-widen-tp-to-4-rank.md`

## What to build

A 4-rank all-reduce collective built on top of the existing `ds4_rocm_xdev` peer-copy infrastructure. Each rank contributes a partial float vector; the result (element-wise sum of all 4 partials) is available on every rank's device.

The implementation should be brute-force all-gather + local accumulate:

1. Each rank writes its partial into a local buffer
2. Record a producer event on the local device (stream ordering)
3. Wait on all 3 peer producer events (`ds4_rocm_xdev_wait_producer`)
4. Peer-copy 3 remote partials into local staging buffers (`ds4_rocm_xdev_copy`)
5. Accumulate 4 buffers (local + 3 remote) into the result (`rocm_xdev_accumulate_f32_kernel`)

Canonical sum order: `buf[0] + buf[1] + buf[2] + buf[3]` on every rank, so floating-point results are bit-identical across ranks.

Buffer layout: a persistent `xdev_allreduce_buf[4][n_elements]` per rank, allocated once during initialization. Sized for `n_embd` (7168 floats = 28 KB) for decode, and a separate larger allocation for prefill chunks.

The existing xdev module (`ds4_rocm_xdev.h/.cu`) provides all the needed primitives: mesh init, peer copy, accumulate kernels, and producer event fencing. This is composition, not new transport code.

## Acceptance criteria

- [ ] `ds4_rocm_xdev_allreduce_init()` and `ds4_rocm_xdev_allreduce_f32()` implemented
- [ ] 4-rank all-reduce of known vectors: result matches CPU reference sum
- [ ] Bit-exact canonical order: f32 sum in rank order (buf[0]+buf[1]+buf[2]+buf[3])
- [ ] Bandwidth measured for n_embd-sized vectors (28 KB): expected ~6.5 μs total
- [ ] Host-staging fallback verified: force `DS4_ROCM_FORCE_HOST_STAGING`, confirm correctness
- [ ] Standalone test added to `tests/test_rocm_xdev` or new test file
- [ ] Builds cleanly with `make -j8 rocm` and `make -j8 test-rocm`

## Blocked by

None — can start immediately (parallel with #26).
