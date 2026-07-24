#pragma once

/* Every unimplemented ROCm tensor-parallel entry point routes its neutral
 * fallback through here. Default: fail loudly and name the caller, so a
 * partially-ported TP build produces an obvious, localized abort instead of
 * plausible-looking wrong numbers -- the primary risk this port guards
 * against (see .scratch/rocm-tensor-parallel/PRD.md).
 *
 * DS4_ROCM_TP_BRINGUP=1 is the single sanctioned escape hatch: it restores
 * the old neutral-return behaviour for plumbing bring-up (running a full
 * forward pass before any kernel is ported) and announces itself loudly on
 * every stub call so it can never be left on unnoticed. It requires an exact
 * "1" so it cannot be tripped by an empty or stray env var value.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline int ds4_rocm_tp_bringup_active(void) {
    const char *v = getenv("DS4_ROCM_TP_BRINGUP");
    return v != NULL && strcmp(v, "1") == 0;
}

/* Returns the neutral value (0) only when bring-up mode is active; aborts
 * the process otherwise. `name` should be the stub's own function name so
 * the error pinpoints exactly which entry point was reached. */
static inline int ds4_rocm_tp_stub(const char *name) {
    if (!ds4_rocm_tp_bringup_active()) {
        fprintf(stderr,
            "ds4: FATAL: ROCm tensor-parallel entry point '%s' is not implemented.\n"
            "ds4:        Tensor-parallel inference cannot proceed past this kernel.\n"
            "ds4:        Set DS4_ROCM_TP_BRINGUP=1 to bypass with a neutral return\n"
            "ds4:        for plumbing bring-up only -- output will NOT be correct.\n",
            name);
        fflush(stderr);
        abort();
    }
    fprintf(stderr,
        "ds4: #### DS4_ROCM_TP_BRINGUP=1 ACTIVE -- '%s' returning a NEUTRAL "
        "(no-op) value. Output is NOT correct. Bring-up mode only. ####\n",
        name);
    return 0;
}
