#!/usr/bin/env bash
# Issue 04 prefill-sweep harness: cold-start-controlled A/B of the gfx12 WMMA
# prefill matmul on this tree's 4x R9700.
#
#   usage: 04-sweep-run.sh <wmma|nowmma> <tag> <short|full|trace> [passes]
#
# short : ctx 2048..4096, the AC-1/AC-3 discriminator. One invocation gives a
#         cold pass-1 and (passes-1) warm re-prefills of the same 2048 chunk.
# full  : ctx 2048..32768 step 2048, issue 02's grid, for AC 2.
# trace : short shape under rocprofv3, to prove the kernel under test really
#         dispatches through ds4-bench (issue 03 only proved it through ./ds4).
#
# Pure prefill (--gen-tokens 0): this kernel is gated on n_tok >= 256 and never
# fires during decode, so decode time would only add noise. No --quality: that
# sets g_quality_mode, which gates the kernel off at ds4_rocm_matmul.cuh:404 and
# would make both builds measure identically. No --ssd-streaming: this tree
# holds the model in VRAM across 4 GPUs, which is the whole point of re-running
# issues 01/02 here.
set -u
BUILD=$1; TAG=$2; SHAPE=$3; PASSES=${4:-3}
M=/var/cache/llama/ds4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
ART=.scratch/gfx1201-wmma-v2/artifacts
BIN=./ds4-bench.$BUILD
P=$ART/04-sweep-prompt.txt

ARGS=(-m "$M" --rocm --gpu-devices 0,1,2,3 --prompt-file "$P"
      --gen-tokens 0 --step-incr 2048 --warm-weights --passes "$PASSES")
case $SHAPE in
  short|trace) ARGS+=(--ctx-start 2048 --ctx-max 4096) ;;
  full)        ARGS+=(--ctx-start 2048 --ctx-max 32768) ;;
  *) echo "bad shape $SHAPE"; exit 2 ;;
esac
ARGS+=(--csv "$ART/04-csv-$TAG.csv")

# `env` is a no-op prefix that keeps the array non-empty, so the expansion below
# is uniform whether or not rocprofv3 is in front.
PRE=(env)
if [ "$SHAPE" = trace ]; then
    rm -rf "$ART/04-trace-$TAG"
    PRE=(rocprofv3 --kernel-trace --output-format csv -d "$ART/04-trace-$TAG" -o trace -- env)
fi

# argv is not otherwise recorded anywhere; header every run log with it plus the
# binary's md5, because `make rocm` and `make rocm-no-wmma` write the same
# ./ds4-bench path and measuring one binary twice is the easy silent failure.
{
  echo "# tag=$TAG build=$BUILD shape=$SHAPE passes=$PASSES"
  echo "# binary=$BIN md5=$(md5sum "$BIN" | cut -d' ' -f1)"
  echo "# prompt=$P md5=$(md5sum "$P" | cut -d' ' -f1)"
  echo "# head=$(git rev-parse --short HEAD) date=$(date -Is)"
  echo "# ${PRE[*]} $BIN ${ARGS[*]}"
} > "$ART/04-run-$TAG.txt"

start=$(date +%s)
"${PRE[@]}" "$BIN" "${ARGS[@]}" >> "$ART/04-run-$TAG.txt" 2> "$ART/04-stderr-$TAG.txt"
rc=$?
echo "# exit=$rc wall_sec=$(( $(date +%s) - start ))" >> "$ART/04-run-$TAG.txt"
echo "exit=$rc wall=$(( $(date +%s) - start ))s"
