#!/usr/bin/env bash
#
# diagnose-prefill.sh -- Run prefill on both pipeline and TP=4 paths with
# per-layer tensor dumps, then diff them to identify the first divergent
# layer.
#
# Usage:
#   ./diagnose-prefill.sh --model <path> --prompt "prompt text" [--gpu-devices 0,1,2,3]
#
# Environment variables:
#   AMD_SERIALIZE_KERNEL  Set to 3 to work around compressor prefill race (issue #23)
#
# Output:
#   Prints per-layer diff table and identifies first divergent layer.
#   Temporary dump files are left in /tmp/diagnose-prefill-* for inspection.
#   Exits 0 if all layers pass tolerance, 1 otherwise.
set -euo pipefail

MODEL=""
PROMPT=""
GPU_DEVICES="0,1,2,3"
TOLERANCE="1e-3"
DS4_BINARY="./ds4"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIFF_SCRIPT="${SCRIPT_DIR}/diff-layers.py"

# --- Parse arguments ---
while [ $# -gt 0 ]; do
    case "$1" in
        --model)       MODEL="$2";       shift 2 ;;
        --prompt)      PROMPT="$2";      shift 2 ;;
        --gpu-devices) GPU_DEVICES="$2"; shift 2 ;;
        --tolerance)   TOLERANCE="$2";   shift 2 ;;
        *) echo "error: unknown argument $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "error: --model <path> is required"
    exit 1
fi
if [ -z "$PROMPT" ]; then
    echo "error: --prompt \"text\" is required"
    exit 1
fi
if [ ! -f "$DS4_BINARY" ]; then
    echo "error: ds4 binary not found at $DS4_BINARY (run 'make rocm' first)"
    exit 1
fi
if [ ! -f "$DIFF_SCRIPT" ]; then
    echo "error: diff-layers.py not found at $DIFF_SCRIPT"
    exit 1
fi

SERIALIZE="${AMD_SERIALIZE_KERNEL:-3}"

# --- Create temporary directories ---
PIPELINE_DIR=$(mktemp -d /tmp/diagnose-prefill-pipeline-XXXXXX)
TP4_DIR=$(mktemp -d /tmp/diagnose-prefill-tp4-XXXXXX)
PIPELINE_PREFIX="${PIPELINE_DIR}/dump"
TP4_PREFIX="${TP4_DIR}/dump"

cleanup() {
    rm -rf "$PIPELINE_DIR" "$TP4_DIR"
}
trap cleanup EXIT

# --- Common ds4 args ---
COMMON_ARGS=(
    --rocm
    --gpu-devices "$GPU_DEVICES"
    --model "$MODEL"
    -p "$PROMPT"
    -n 1
)

# --- Step 1: Run pipeline path ---
echo "=== Step 1/3: Running PIPELINE prefill (no --cuda-tensor-parallel) ==="
echo "  Prefix: ${PIPELINE_PREFIX}"
echo "  Prompt: \"${PROMPT}\""
echo ""

# Set dump prefix for per-layer tensor dumps
export DS4_METAL_GRAPH_DUMP_PREFIX="${PIPELINE_PREFIX}"
export AMD_SERIALIZE_KERNEL="${SERIALIZE}"

if ! DS4_METAL_GRAPH_DUMP_PREFIX="${PIPELINE_PREFIX}" \
      AMD_SERIALIZE_KERNEL="${SERIALIZE}" \
      "$DS4_BINARY" "${COMMON_ARGS[@]}" > /dev/null 2>&1; then
    echo "error: pipeline prefill run failed"
    exit 1
fi

echo "  Pipeline dumps written to ${PIPELINE_DIR}/"
echo ""

# Count dumps
PIPELINE_COUNT=$(find "$PIPELINE_DIR" -name '*.bin' 2>/dev/null | wc -l)
echo "  Found ${PIPELINE_COUNT} tensor dump files."

# --- Step 2: Run TP=4 path ---
echo ""
echo "=== Step 2/3: Running TP=4 prefill (--cuda-tensor-parallel) ==="
echo "  Prefix: ${TP4_PREFIX}"
echo ""

if ! DS4_METAL_GRAPH_DUMP_PREFIX="${TP4_PREFIX}" \
      AMD_SERIALIZE_KERNEL="${SERIALIZE}" \
      "$DS4_BINARY" "${COMMON_ARGS[@]}" --cuda-tensor-parallel > /dev/null 2>&1; then
    echo "error: TP=4 prefill run failed"
    exit 1
fi

echo "  TP=4 dumps written to ${TP4_DIR}/"
echo ""

TP4_COUNT=$(find "$TP4_DIR" -name '*.bin' 2>/dev/null | wc -l)
echo "  Found ${TP4_COUNT} tensor dump files."

# --- Step 3: Diff ---
echo ""
echo "=== Step 3/3: Comparing per-layer tensors ==="
echo ""

"$DIFF_SCRIPT" "$PIPELINE_DIR" "$TP4_DIR" --tolerance "$TOLERANCE"
DIFF_EXIT=$?

# --- Report first divergent layer ---
echo ""
echo "=== Summary ==="
echo "  Pipeline dir: ${PIPELINE_DIR}/"
echo "  TP=4 dir:     ${TP4_DIR}/"
echo "  Pipeline dumps: ${PIPELINE_COUNT} files"
echo "  TP=4 dumps:     ${TP4_COUNT} files"

if [ "$DIFF_EXIT" -eq 0 ]; then
    echo "  Result: ALL LAYERS PASS (max error <= ${TOLERANCE})"
else
    echo "  Result: DIVERGENCE DETECTED (some layers exceed tolerance ${TOLERANCE})"

    # Find the first layer with a failed tensor
    echo ""
    echo "  First divergent layer(s):"
    # Re-run the diff script silently and parse output
    "$DIFF_SCRIPT" "$PIPELINE_DIR" "$TP4_DIR" --tolerance "$TOLERANCE" 2>&1 \
        | grep -E '^\s+\d+\s+' \
        | grep -i 'FAIL' \
        | head -5
fi

exit "$DIFF_EXIT"
