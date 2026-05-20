#!/usr/bin/env python3
"""Print debug prefill_embeds_bf16.bin content (static inspection only)."""

import json
from pathlib import Path

import numpy as np


def _load_bf16_bin(path: Path, shape: tuple) -> np.ndarray:
    """Load raw bfloat16 data and convert to float32 numpy array."""
    raw = path.read_bytes()
    expected = int(np.prod(shape)) * 2
    if len(raw) != expected:
        raise ValueError(f"{path}: size mismatch, got {len(raw)}, expected {expected}")
    u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
    fp32 = (u16.astype(np.uint32) << 16).view(np.float32)
    return fp32.copy()


def main():
    debug_dir = Path(__file__).resolve().parent.parent / "debug_bin"
    meta_path = debug_dir / "meta.json"
    bin_path = debug_dir / "prefill_embeds_bf16.bin"

    if not meta_path.exists():
        raise FileNotFoundError(f"missing {meta_path}")
    if not bin_path.exists():
        raise FileNotFoundError(f"missing {bin_path}")

    meta = json.loads(meta_path.read_text())
    S = int(meta["S"])
    hidden_size = int(meta["hidden_size"])
    print(f"meta: S={S}, hidden_size={hidden_size}")

    arr = _load_bf16_bin(bin_path, (S, hidden_size))
    print(f"loaded array shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}")
    print("first row (first 16 values):")
    print(arr[0, :16])
    print("last row (first 16 values):")
    print(arr[-1, :16])

    npy_path = bin_path.with_suffix(".npy")
    np.save(npy_path, arr)
    print(f"saved npy: {npy_path}")


if __name__ == "__main__":
    main()
