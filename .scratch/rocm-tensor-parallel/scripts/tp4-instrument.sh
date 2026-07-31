#!/usr/bin/env bash
#
# tp4-instrument.sh -- Re-run the TP=4 decode-loop sync/dispatch instrumentation
# harness (issue #49) and print a per-call-site count + wall-clock breakdown.
#
# This is the baseline every later issue in the chain (#50-#55) should report
# a before/after delta against, not just an aggregate t/s number.
#
# Usage:
#   ./tp4-instrument.sh --model <path> [--prompt "text"] [--gen-tokens N] \
#       [--gpu-devices 0,1,2,3] [--ctx N]
#
# Output:
#   The instrumentation report (per-call-site calls, calls/token, total ms,
#   ms/token) is printed to stderr by the `ds4` binary itself at exit; this
#   wrapper just sets DS4_TP4_INSTRUMENT=1 and forwards it.
#
# Does NOT set AMD_SERIALIZE_KERNEL by default. That flag makes every
# kernel launch block synchronously, which hides real wait time inside the
# (uninstrumented) kernel-launch calls instead of the named sync/dispatch
# sites this harness measures -- it produced misleadingly low numbers for
# hipDeviceSynchronize barriers in an earlier pass (see the 2026-07-31
# experiment-log entry for issue #49). Only set it if you specifically need
# to debug under serialization; the unserialized run is the baseline.
set -euo pipefail

MODEL=""
PROMPT="The capital of France is"
GEN_TOKENS=20
GPU_DEVICES="0,1,2,3"
CTX=64
DS4_BINARY="./ds4"

while [ $# -gt 0 ]; do
    case "$1" in
        --model)       MODEL="$2";       shift 2 ;;
        --prompt)      PROMPT="$2";      shift 2 ;;
        --gen-tokens)  GEN_TOKENS="$2";  shift 2 ;;
        --gpu-devices) GPU_DEVICES="$2"; shift 2 ;;
        --ctx)         CTX="$2";         shift 2 ;;
        *) echo "error: unknown argument $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "error: --model <path> is required"
    exit 1
fi
if [ ! -f "$DS4_BINARY" ]; then
    echo "error: ds4 binary not found at $DS4_BINARY (run 'make rocm' first)"
    exit 1
fi

# A few early tokens run cold caches; the report averages over all generated
# tokens, so ask for enough tokens that steady-state dominates the average.
DS4_TP4_INSTRUMENT=1 AMD_SERIALIZE_KERNEL="${AMD_SERIALIZE_KERNEL:-}" \
    "$DS4_BINARY" --rocm --gpu-devices "$GPU_DEVICES" --cuda-tensor-parallel \
        --model "$MODEL" -c "$CTX" -p "$PROMPT" -n "$GEN_TOKENS"
