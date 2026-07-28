#!/usr/bin/env python3
"""
diff-layers.py <dir1> <dir2> [--tolerance TOL]

Compare per-layer binary float32 tensor dumps from two directories and report
max-absolute-error between corresponding tensors.  The dump files follow the
ds4 naming convention:

    <prefix>_<tensor_name>-<layer>_pos<pos>.bin

e.g. "mydump_hc_attn_post-5_pos0.bin"

Exits with code 0 if all layers pass tolerance (default 1e-3), 1 otherwise.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

# Tensor names we know about and the tolerance label to display
KNOWN_TENSORS = {
    "hc_attn_post": "after_attn_hc",
    "ffn_moe_out":  "routed_out",
    "hc_ffn_post":  "after_ffn_hc",
}

# Default per-tensor tolerance
DEFAULT_TOLERANCE = 1e-3

# Pattern for ds4 dump filenames: <prefix>_<name>-<layer>_pos<pos>.bin
FILENAME_RE = re.compile(
    r"(.*)_(hc_attn_post|ffn_moe_out|hc_ffn_post)-(\d+)_pos(\d+)\.bin$"
)


def parse_dumps(directory):
    """Scan `directory` for dump files and return a dict:
        {(tensor_name, layer, pos): filepath}
    """
    dumps = {}
    for fname in os.listdir(directory):
        m = FILENAME_RE.match(fname)
        if not m:
            continue
        prefix, name, layer_str, pos_str = m.groups()
        layer = int(layer_str)
        pos = int(pos_str)
        key = (name, layer, pos)
        dumps[key] = os.path.join(directory, fname)
    return dumps


def load_tensor(filepath):
    """Load a raw float32 binary tensor from file. Returns numpy array."""
    data = np.fromfile(filepath, dtype=np.float32)
    if len(data) == 0:
        print(f"  error: {filepath} is empty")
        return None
    return data


def max_abs_error(a, b):
    """Compute max-abs-error between two numpy arrays. Returns (float, bool)."""
    if a.shape != b.shape:
        return float("inf"), False
    err = np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))
    return float(err), True


def main():
    parser = argparse.ArgumentParser(
        description="Compare per-layer binary float32 tensor dumps"
    )
    parser.add_argument("dir1", help="First dump directory (pipeline ref)")
    parser.add_argument("dir2", help="Second dump directory (TP=4)")
    parser.add_argument(
        "--tolerance", type=float, default=DEFAULT_TOLERANCE,
        help=f"Max-abs-error tolerance per tensor (default {DEFAULT_TOLERANCE})"
    )
    args = parser.parse_args()

    dump1 = parse_dumps(args.dir1)
    dump2 = parse_dumps(args.dir2)

    if not dump1:
        print(f"error: no dump files found in {args.dir1}")
        sys.exit(1)
    if not dump2:
        print(f"error: no dump files found in {args.dir2}")
        sys.exit(1)

    # Collect all unique keys
    all_keys = sorted(set(list(dump1.keys()) + list(dump2.keys())))

    # Print table header
    header = f"{'il':>4s}  {'tensor':20s}  {'max_err':>12s}  {'status':>8s}"
    sep = "-" * len(header)
    print("Per-layer max-abs-error comparison")
    print(f"  Reference dir: {args.dir1}")
    print(f"  Test dir:      {args.dir2}")
    print(f"  Tolerance:     {args.tolerance}")
    print()
    print(header)
    print(sep)

    any_fail = False
    total_pairs = 0
    fail_pairs = 0

    for key in all_keys:
        name, layer, pos = key
        f1 = dump1.get(key)
        f2 = dump2.get(key)
        label = KNOWN_TENSORS.get(name, name)

        if f1 is None:
            print(f"{layer:4d}  {label:20s}  {'MISSING':>12s}  {'dir1':>8s}")
            continue
        if f2 is None:
            print(f"{layer:4d}  {label:20s}  {'MISSING':>12s}  {'dir2':>8s}")
            continue

        tensor1 = load_tensor(f1)
        tensor2 = load_tensor(f2)
        if tensor1 is None or tensor2 is None:
            any_fail = True
            continue

        err, shape_ok = max_abs_error(tensor1, tensor2)
        total_pairs += 1

        if not shape_ok:
            print(
                f"{layer:4d}  {label:20s}  {'SHAPE':>12s}  "
                f"dir1={tensor1.shape} dir2={tensor2.shape}  ***"
            )
            any_fail = True
            fail_pairs += 1
            continue

        pass_fail = "PASS" if err <= args.tolerance else "FAIL"
        marker = "  ***" if err > args.tolerance else ""
        if err > args.tolerance:
            any_fail = True
            fail_pairs += 1

        print(f"{layer:4d}  {label:20s}  {err:>12.2e}  {pass_fail:>8s}{marker}")

    print(sep)
    print(f"\nSummary: {total_pairs} tensor pairs compared, "
          f"{total_pairs - fail_pairs} passed, "
          f"{fail_pairs} failed")

    if any_fail:
        sys.exit(1)
    else:
        print("All layers pass tolerance.")
        sys.exit(0)


if __name__ == "__main__":
    main()
