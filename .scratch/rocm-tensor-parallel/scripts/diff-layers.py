#!/usr/bin/env python3
"""
diff-layers.py <dir1> <dir2> [--tolerance TOL] [--full-precision]

Compare per-layer binary float32 tensor dumps from two directories and report
max-absolute-error between corresponding tensors.  The dump files follow the
ds4 naming convention:

    <prefix>_<tensor_name>-<layer>_pos<pos>.bin

e.g. "mydump_hc_attn_post-5_pos0.bin"

Exits with code 0 if all layers pass tolerance (default 1e-3), 1 otherwise.

With --full-precision, prints the full FP64 bit-accurate error rather than
the default 2-digit scientific notation.
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
    "ffn_shexp":    "shared_out",
}

# Default per-tensor tolerance
DEFAULT_TOLERANCE = 1e-3

# Pattern for ds4 dump filenames: <prefix>_<name>-<layer>_pos<pos>.bin
FILENAME_RE = re.compile(
    r"(.*)_(hc_attn_post|ffn_moe_out|hc_ffn_post|ffn_shexp)-(\d+)_pos(\d+)\.bin$"
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


def analyze_bit_identical(tensor1, tensor2):
    """Check if two tensors are bit-identical (same float32 representation)."""
    if tensor1.shape != tensor2.shape:
        return False, "shape mismatch"
    # Check if every float32 element is exactly equal
    diff = tensor1.view(np.uint32) ^ tensor2.view(np.uint32)
    different_bits = np.count_nonzero(diff)
    if different_bits == 0:
        return True, "bit-identical"
    return False, f"{different_bits} differing uint32 words"


def format_error(err, full_precision):
    """Format error value for display."""
    if full_precision:
        return f"{err:.6e}"
    return f"{err:>12.2e}"


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
    parser.add_argument(
        "--full-precision", action="store_true",
        help="Print full-precision error values (not abbreviated)"
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
    header = f"{'il':>4s}  {'tensor':20s}  {'max_err':>12s}  {'status':>8s}  {'bit_id':>8s}"
    sep = "-" * len(header)
    print("Per-layer max-abs-error comparison")
    print(f"  Reference dir: {args.dir1}")
    print(f"  Test dir:      {args.dir2}")
    print(f"  Tolerance:     {args.tolerance}")
    print(f"  Precision:     {'full (%.6e)' if args.full_precision else 'default (%.2e)'}")
    print()
    print(header)
    print(sep)

    any_fail = False
    total_pairs = 0
    fail_pairs = 0
    bit_identical_count = 0

    for key in all_keys:
        name, layer, pos = key
        f1 = dump1.get(key)
        f2 = dump2.get(key)
        label = KNOWN_TENSORS.get(name, name)

        if f1 is None:
            print(f"{layer:4d}  {label:20s}  {'MISSING':>12s}  {'dir1':>8s}  {'':>8s}")
            continue
        if f2 is None:
            print(f"{layer:4d}  {label:20s}  {'MISSING':>12s}  {'dir2':>8s}  {'':>8s}")
            continue

        tensor1 = load_tensor(f1)
        tensor2 = load_tensor(f2)
        if tensor1 is None or tensor2 is None:
            any_fail = True
            continue

        err, shape_ok = max_abs_error(tensor1, tensor2)
        bit_id, bit_info = analyze_bit_identical(tensor1, tensor2)
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

        bit_label = "BIT_ID" if bit_id else "DIFF"
        if bit_id:
            bit_identical_count += 1

        err_str = format_error(err, args.full_precision)
        print(f"{layer:4d}  {label:20s}  {err_str:>12s}  {pass_fail:>8s}  {bit_label:>8s}{marker}")

    print(sep)
    print(f"\nSummary: {total_pairs} tensor pairs compared, "
          f"{total_pairs - fail_pairs} passed, "
          f"{fail_pairs} failed")
    print(f"Bit-identical: {bit_identical_count}/{total_pairs}")

    # Compute cumulative drift: find the max error across all passing layers
    # to check if there's a smooth ramp
    all_errors = {}
    for key in all_keys:
        name, layer, pos = key
        f1 = dump1.get(key)
        f2 = dump2.get(key)
        if f1 is None or f2 is None:
            continue
        tensor1 = load_tensor(f1)
        tensor2 = load_tensor(f2)
        if tensor1 is None or tensor2 is None:
            continue
        err, shape_ok = max_abs_error(tensor1, tensor2)
        if shape_ok:
            if layer not in all_errors:
                all_errors[layer] = {}
            all_errors[layer][label] = err

    if all_errors:
        print(f"\nPer-layer max errors (all tensors):")
        print(f"{'il':>4s}  {'max_err':>12s}  {'bit_id?':>8s}")
        print("-" * 28)
        for layer in sorted(all_errors.keys()):
            layer_vals = list(all_errors[layer].values())
            max_layer_err = max(layer_vals) if layer_vals else 0
            # Check if ALL tensors at this layer are bit-identical
            all_bit_id = all(
                np.max(np.abs(
                    load_tensor(dump1.get((k[0], k[1], k[2]))).astype(np.float64) -
                    load_tensor(dump2.get((k[0], k[1], k[2]))).astype(np.float64)
                )) == 0.0
                for k in all_keys
                if k[1] == layer
                for _ in [1]  # just to make the comprehension work
            )
            # Simplified bit-id check
            layer_bit_id = True
            for key in all_keys:
                k_name, k_layer, k_pos = key
                if k_layer != layer:
                    continue
                t1 = load_tensor(dump1.get(key))
                t2 = load_tensor(dump2.get(key))
                if t1 is None or t2 is None:
                    continue
                if not np.array_equal(t1.view(np.uint32), t2.view(np.uint32)):
                    layer_bit_id = False
                    break

            err_str = format_error(max_layer_err, args.full_precision)
            bid_str = "BIT_ID" if layer_bit_id else "diff"
            print(f"{layer:4d}  {err_str:>12s}  {bid_str:>8s}")

    if any_fail:
        sys.exit(1)
    else:
        print("All layers pass tolerance.")
        sys.exit(0)


if __name__ == "__main__":
    main()
