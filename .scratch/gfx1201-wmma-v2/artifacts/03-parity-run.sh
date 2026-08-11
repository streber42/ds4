#!/usr/bin/env bash
# Issue 03 parity harness: run one ds4 build in production mode (NOT under
# score_official, so the WMMA prefill matmul is not gated off by
# g_quality_mode) on a >256-token prompt.
#
#   usage: 03-parity-run.sh <ds4-binary> <tag> <logits|gen|gen-trace>
#
# logits    : --dump-logits (short-circuits generation; used for numeric parity)
# gen       : 30-token greedy generation (--temp 0)
# gen-trace : same as gen, under `rocprofv3 --kernel-trace` for dispatch counts
set -u
BIN=$1; TAG=$2; MODE=$3
M=/var/cache/llama/ds4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
ART=.scratch/gfx1201-wmma-v2/artifacts
P=$ART/03-parity-prompt.txt

ARGS=(-m "$M" --rocm --gpu-devices 0,1,2,3 --ctx 4096 --temp 0 --prompt-file "$P")
case $MODE in
  logits)    ARGS+=(-n 1 --dump-logits "$ART/03-logits-$TAG.json") ;;
  logprobs)  ARGS+=(-n 30 --logprobs-top-k 5 --dump-logprobs "$ART/03-logprobs-$TAG.json") ;;
  gen|gen-trace) ARGS+=(-n 30) ;;
  *) echo "bad mode $MODE"; exit 2 ;;
esac

PRE=(env AMD_SERIALIZE_KERNEL=3)
if [ "$MODE" = gen-trace ]; then
    rm -rf "$ART/03-trace-$TAG"
    PRE=(rocprofv3 --kernel-trace --output-format csv -d "$ART/03-trace-$TAG" -o trace --
         env AMD_SERIALIZE_KERNEL=3)
fi

{
  echo "# tag=$TAG mode=$MODE binary=$BIN md5=$(md5sum "$BIN" | cut -d' ' -f1)"
  echo "# head=$(git rev-parse --short HEAD) date=$(date -Is)"
  echo "# ${PRE[*]} $BIN ${ARGS[*]}"
} > "$ART/03-gen-$TAG.txt"

"${PRE[@]}" "$BIN" "${ARGS[@]}" >> "$ART/03-gen-$TAG.txt" 2> "$ART/03-stderr-$TAG.txt"
echo "exit=$?"
