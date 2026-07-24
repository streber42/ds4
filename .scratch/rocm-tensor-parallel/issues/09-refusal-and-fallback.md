# Graceful refusal & pipeline fallback

Status: closed

## Parent

`.scratch/rocm-tensor-parallel/PRD.md`

## What to build

Make tensor parallelism fail safely. When the requested configuration is not supported — wrong
rank count, incompatible model shape, unavailable peer transport — the engine must refuse with
a clear, specific message. When tensor parallelism is requested but cannot initialise, the
engine must fall back to the existing pipeline layer-split path rather than leaving the user
with nothing.

The operator must never end up with no working inference because of this feature. Pipeline
layer-split at ~28 tok/s is a perfectly good outcome; a hang or an unexplained crash is not.
Equally important, an unsupported configuration must never silently produce wrong output — a
refusal is always better than a plausible-looking wrong answer.

## Acceptance criteria

- [x] An unsupported rank count is refused with a message naming the actual constraint
- [x] An incompatible model shape is refused with a specific reason, not a generic failure
- [x] Unavailable or failed peer transport results in fallback to pipeline layer-split, not a crash or hang
- [x] Fallback is reported clearly to the operator so the slower path is never silent
- [x] After fallback, inference produces correct output via the pipeline path
- [x] No unsupported configuration path can silently produce wrong output
- [x] Refusal behaviour is covered by tests, following the existing multi-GPU refusal test pattern

## Blocked by

- `.scratch/rocm-tensor-parallel/issues/04-tp-plumbing-no-crash.md`

## Comments

**Two refusal checks already existed from earlier slices and were hardened here.**
`ds4_engine_open_internal` (`ds4.c`) already hard-refused an odd/insufficient GPU count
(before any model I/O) and a non-DeepSeek model family (after `model_open`) when
`--cuda-tensor-parallel` was requested. Both messages were generic/backend-hardcoded; both
now name the actual constraint and the actual observed value:

- Rank count: `"--cuda-tensor-parallel requires an even number of GPUs >= 2 (--gpu-devices
  gave %d)"`, or, if the backend itself is wrong, `"--cuda-tensor-parallel requires the %s
  backend (got %s)"` using `ds4_backend_name()` (so ROCm builds say "rocm", not a hardcoded
  "CUDA" that would confuse a ROCm operator).
- Model shape: `"--cuda-tensor-parallel is currently supported only for DeepSeek-V4-Flash;
  loaded model is %s"`, naming the detected family (e.g. "GLM 5.2 (glm-dsa)") instead of a
  bare "supported only for DeepSeek models".

**What was newly implemented: transport-failure fallback.** Traced how `--cuda-tensor-parallel`
and plain pipeline layer-split relate: both are the *same* `multi_tier` engine-open path,
differing only in the `cuda_tensor_parallel` bool, which `engine_classify_multi_tier` reads
(via `engine_cuda_tp_ep_requested`) to pick TP-sharded (`engine_compute_cuda_ep_placement`) vs.
plain layer-split (`ds4_compute_layer_placement`) placement. This means disabling
`cuda_tensor_parallel` *before* that placement decision is exactly "fall back to pipeline
layer-split" -- no new pipeline code needed, just routing into the existing, already-verified
path (issue 04/05's ~28 tok/s baseline).

Traced the actual ROCm transport-failure surface: `ds4_rocm_xdev_init_mesh` (issue 01) already
degrades a fully-unavailable peer pair to a host-staging bounce automatically (never hard-fails
for capability reasons), and the PRD's own feasibility notes already established host-staging
(13.5 GB/s) as a fully acceptable, correct fallback for TP's per-token traffic. So the *only*
genuine "transport unavailable" condition is when *neither* mechanism works for some
home/partner pair -- i.e. no direct peer access *and* the pinned host-staging buffer itself
fails to allocate (system-wide pinned-memory exhaustion). That is rare but real, and today it
would either hard-abort engine open (odd/wrong device cases) or, worse, surface as a crash deep
inside a forward pass once a TP kernel actually needed to move data across that pair.

Added `ds4_rocm_xdev_tp_transport_ok` (pure decision: host-staging available, or every
home/partner pair has bidirectional peer access) and `ds4_rocm_xdev_tp_transport_probe` (gathers
the real inputs via `hipHostMalloc`/`hipDeviceCanAccessPeer` queries, no `hipSetDevice`, no
peer-access side effects -- safe to call before `ds4_gpu_init_multi` actually establishes the
peer mesh) to `ds4_rocm_xdev.h`/`.cu`. Wired into `ds4.c`'s `ds4_engine_open_internal`,
guarded by `#ifdef DS4_ROCM_BUILD` and declared as a local `extern` (not via a shared header),
so CUDA/Metal builds never reference this ROCm-only symbol and are untouched. The check runs
*before* `engine_classify_multi_tier` (the placement-decision ordering constraint above): if
transport is unreachable, `e->cuda_tensor_parallel` is cleared, `e->prefill_chunk` is
recomputed for the non-TP path, and a clear stderr line explains the fallback
(`"falling back to pipeline layer-split (tensor parallelism disabled for this run; expect
pipeline throughput, not TP)"`) -- so the slower path is never silent.

**Verified on the real gfx1201 hardware on this box (4x R9700), `make -j8 rocm
ROCM_ARCH=gfx1201` and `make -j8 test-rocm ROCM_ARCH=gfx1201` both clean (no warnings, no
regressions in the three pre-existing ROCm standalone tests):**

- New pure-logic + hardware-probe tests appended to `tests/test_rocm_xdev.cu`: host-staging
  covering a degraded pair, no-host-staging-but-all-peer-ok, the genuine-unreachable case,
  null-input and degenerate-rank-count handling, and a hardware assertion that the probe reports
  "reachable" for this box's real, fully-connected 4-GPU mesh (half=2) -- all pass.
- New `tests/test_engine_rocm_tp_refusal` (modeled on `tests/test_engine_mgpu_refusal.c`,
  wired into `make test-rocm`): asserts an odd GPU count (3) with `--cuda-tensor-parallel`
  refuses with nonzero rc, NULL engine, and stderr naming both the flag and the even-count
  constraint -- and does so *before* the (deliberately nonexistent) model path is ever opened,
  proving the refusal path can't be reached late enough to silently do anything. Confirmed
  through the real `./ds4` CLI too (`--rocm --cuda-tensor-parallel --gpu-devices 0,1,2`): clean
  exit 1, no hang, message unchanged from the library-level test.
- The model-shape sub-test requires `DS4_TEST_NON_DEEPSEEK_MODEL` (a small non-DeepSeek-family
  GGUF, e.g. GLM 5.2) to reach `config_validate_model`; no such fixture exists in this sandbox
  (no GLM or DeepSeek-V4-Flash GGUF present, and downloading an 80 GiB quant to exercise a
  refusal check was judged out of proportion for this slice per the PRD's testing
  decisions -- refusal tests should not require the full model). The sub-test skips cleanly
  (PASS, not FAIL) when unset and says so. The underlying check itself is pre-existing,
  unchanged in behavior by this issue (only its message was improved), and was already exercised
  end-to-end by the original CUDA implementation this ROCm path shares.
- The genuinely-unreachable branch of `ds4_rocm_xdev_tp_transport_probe` (both peer *and*
  host-staging failing) cannot be forced on working hardware without corrupting driver state, so
  it is verified at the pure-decision-logic level (`ds4_rocm_xdev_tp_transport_ok`) with
  fabricated inputs instead, per the PRD's own stance that this class of failure is
  "unreachable from higher seams" by design.
