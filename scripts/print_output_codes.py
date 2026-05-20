#!/usr/bin/env python3
"""Print output_codes.bin content (static inspection only)."""

import json
from pathlib import Path

import numpy as np


def main():
    script_dir = Path(__file__).resolve().parent
    meta_path = script_dir / "output_meta.json"
    bin_path = script_dir / "output_codes.bin"

    if not meta_path.exists():
        raise FileNotFoundError(f"missing {meta_path}")
    if not bin_path.exists():
        raise FileNotFoundError(f"missing {bin_path}")

    meta = json.loads(meta_path.read_text())
    shape = tuple(int(x) for x in meta["shape"])
    dtype_str = meta.get("dtype", "int32")
    print(f"meta: shape={shape}, dtype={dtype_str}")

    np_dtype = np.dtype(dtype_str)
    raw = bin_path.read_bytes()
    expected = int(np.prod(shape)) * np_dtype.itemsize
    if len(raw) != expected:
        raise ValueError(f"{bin_path}: size mismatch, got {len(raw)}, expected {expected}")

    arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
    arr = arr.copy()
    print(f"loaded array shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"min={arr.min()}, max={arr.max()}, mean={arr.mean():.2f}")
    print("first 50 frames:")
    print(arr[:50])
    print("last 50 frames:")
    print(arr[-50:])

    npy_path = bin_path.with_suffix(".npy")
    np.save(npy_path, arr)
    print(f"saved npy: {npy_path}")


if __name__ == "__main__":
    main()
