#!/usr/bin/env bash
#
# diff-layers.sh — Compare per-layer binary float32 tensor dumps from two
# directories and report max-absolute-error between corresponding tensors.
#
# Usage:
#   ./diff-layers.sh <dir1> <dir2> [--tolerance TOL]
#
# This is a thin wrapper around diff-layers.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/diff-layers.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "error: ${PYTHON_SCRIPT} not found"
    exit 1
fi

exec python3 "$PYTHON_SCRIPT" "$@"
