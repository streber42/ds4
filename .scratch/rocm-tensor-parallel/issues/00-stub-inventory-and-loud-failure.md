# TP entry-point inventory + loud failure for unimplemented stubs

Status: closed

## Parent

`.scratch/rocm-tensor-parallel/PRD.md`

## What to build

A prefactor that runs **before any kernel porting**. Two parts:

**Inventory.** Enumerate every tensor-parallel GPU entry point the ROCm build currently stubs,
and record it as a machine-checkable checklist with each entry marked implemented or not. This
becomes the progress tracker for the whole port and the guard against quietly forgetting one.

**Loud failure by default.** Today an unimplemented entry point returns a neutral value. That
means for the entire duration of this port, any kernel not yet reached **silently corrupts
output instead of failing** — the single worst failure mode in this project, and the one the
spec names as the primary risk. Flip the default: an unimplemented entry point must fail
loudly and name itself, so a partially-ported build produces an obvious, localized error
rather than plausible-looking wrong numbers.

**Bring-up escape hatch.** Plumbing bring-up (see the plumbing slice) legitimately needs to run
a full forward pass before any kernel is ported, which loud failure would abort. Provide one
explicit, opt-in mode that restores neutral returns for that purpose. It must announce itself
unmistakably whenever active and must be impossible to enable by accident or leave on without
noticing — a silent neutral-return mode is exactly what this ticket exists to eliminate.

"Make the change easy, then make the easy change": every later slice is safer and easier to
debug once a missing kernel screams instead of whispering.

## Acceptance criteria

- [x] Every tensor-parallel entry point stubbed in the ROCm build is enumerated in the inventory
- [x] The inventory records implemented / not-implemented status per entry point and is machine-checkable
- [x] By default, calling an unimplemented entry point fails loudly and identifies which entry point it was
- [x] Default behaviour cannot silently return a neutral value
- [x] An explicit opt-in bring-up mode restores neutral returns for plumbing work only
- [x] Bring-up mode announces itself unmistakably whenever it is active
- [x] Bring-up mode cannot be enabled accidentally, and its being left on is obvious
- [x] Existing pipeline (non-tensor-parallel) inference is completely unaffected by this change
- [x] The inventory is referenced by later slices as the definition of "done" for kernel coverage

## Blocked by

None - can start immediately.

## Comments

Inventory (`inventory.json` / `inventory.md`) and the stub test (`tests/test_rocm_tp_stubs.cu`)
already existed on this branch from a prior pass (commit de5701f); this slice implements the
loud-failure behaviour the test asserts, which had not landed yet.

**Audit found one gap:** cross-checking every neutral-`return 0` stub actually linked into the
ROCm build (`ds4_rocm_unavailable.cu`'s `ROCM_UNAVAILABLE_INT` macros + the ROCm-only stubs in
`ds4_rocm.cu`) against the inventory turned up `ds4_gpu_hc_expand_add_tensor` — called from the
TP decode path in `ds4.c` (attention→HC combine step) but missing from the 36-entry list. Added
as entry #13 (category Attention, target slice 05), bringing the total to 37. Total/stubbed
counts and category summary updated in both `inventory.json` and `inventory.md`.

**Implementation:** new shared header `ds4_rocm_tp_bringup.h` defines `ds4_rocm_tp_stub(name)`,
used by every stub in both `ds4_rocm_unavailable.cu` and `ds4_rocm.cu`. Default: `fprintf` to
stderr naming the entry point and `abort()`. `DS4_ROCM_TP_BRINGUP=1` (exact match only, so it
can't be tripped by an empty or stray value) restores the neutral 0 return, printing an
unmistakable banner on every stub call so the mode can never be left on unnoticed.

**Verified:**
- `make -j8 test-rocm ROCM_ARCH=gfx1201` — `tests/test_rocm_tp_stubs` passes both cases (default
  abort via fork/SIGABRT, and `DS4_ROCM_TP_BRINGUP=1` returning the neutral value).
- `make -j8 rocm ROCM_ARCH=gfx1201` — full `ds4`/`ds4-server`/`ds4-bench`/`ds4-eval`/`ds4-agent`
  build cleanly against the real gfx1201 hardware present on this box (4x R9700), no warnings.
- Pipeline-unaffected: every touched entry point is only reached from TP-gated branches in
  `ds4.c` (`g->tp_world == 2`, `cuda_tp_*` conditionals) — non-TP pipeline inference never calls
  these stubs, so loud failure cannot fire outside tensor-parallel mode.
- `tests/test_rocm_xdev` segfaults on this box, but it only links `ds4_rocm_xdev.o`/`.cu`, files
  untouched by this change (confirmed via the Makefile's dependency list and by diffing against
  a stash of this slice's changes) — pre-existing, out of scope for this issue (belongs to
  01-cross-device-transfer-module), left for whichever slice owns that module next.
- No model GGUF is present on this box, so an end-to-end pipeline inference run was not
  exercised; scope here is a prefactor with no kernel changes, and the code-path audit above
  establishes non-TP inference cannot reach the changed stubs.
