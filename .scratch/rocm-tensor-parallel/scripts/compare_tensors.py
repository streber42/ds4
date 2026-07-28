#!/usr/bin/env python3
"""Compare two binary float32 tensors and report max-abs-error and bit identity."""
import sys
import numpy as np

a = np.fromfile(sys.argv[1], dtype=np.float32)
b = np.fromfile(sys.argv[2], dtype=np.float32)
if a.shape != b.shape:
    print(f"SHAPE MISMATCH: {a.shape} vs {b.shape}")
    sys.exit(1)
max_err = np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))
bit_id = np.array_equal(a.view(np.uint32), b.view(np.uint32))
print(f"max_err={max_err:.6e}  bit_id={bit_id}")
