# TP entry-point inventory + loud failure for unimplemented stubs

Status: ready-for-agent

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

- [ ] Every tensor-parallel entry point stubbed in the ROCm build is enumerated in the inventory
- [ ] The inventory records implemented / not-implemented status per entry point and is machine-checkable
- [ ] By default, calling an unimplemented entry point fails loudly and identifies which entry point it was
- [ ] Default behaviour cannot silently return a neutral value
- [ ] An explicit opt-in bring-up mode restores neutral returns for plumbing work only
- [ ] Bring-up mode announces itself unmistakably whenever it is active
- [ ] Bring-up mode cannot be enabled accidentally, and its being left on is obvious
- [ ] Existing pipeline (non-tensor-parallel) inference is completely unaffected by this change
- [ ] The inventory is referenced by later slices as the definition of "done" for kernel coverage

## Blocked by

None - can start immediately.
