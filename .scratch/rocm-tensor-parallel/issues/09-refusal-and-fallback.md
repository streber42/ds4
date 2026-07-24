# Graceful refusal & pipeline fallback

Status: ready-for-agent

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

- [ ] An unsupported rank count is refused with a message naming the actual constraint
- [ ] An incompatible model shape is refused with a specific reason, not a generic failure
- [ ] Unavailable or failed peer transport results in fallback to pipeline layer-split, not a crash or hang
- [ ] Fallback is reported clearly to the operator so the slower path is never silent
- [ ] After fallback, inference produces correct output via the pipeline path
- [ ] No unsupported configuration path can silently produce wrong output
- [ ] Refusal behaviour is covered by tests, following the existing multi-GPU refusal test pattern

## Blocked by

- `.scratch/rocm-tensor-parallel/issues/04-tp-plumbing-no-crash.md`
