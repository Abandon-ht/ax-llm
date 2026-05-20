#!/usr/bin/env python3
"""
Compare C++ vs Python prefill checkpoints for Qwen3-TTS talker.

Usage:
    python3 scripts/compare_prefill_checkpoints.py \
        --cpp_dir ./debug_bin/cpp_dump \
        --py_dir ./debug_bin/python_prefill_dump \
        [--hidden_size 1024] [--vocab_size 3072] [--S 8] [--num_layers 28] [--kv_dim 256]

Checkpoints compared:
    1. prefill input          (talker_prefill_input.bin vs prefill_embeds.bin)
    2. layer0 output          (talker_layer0_prefill_output.bin vs python_layer0_prefill_output.bin)
    3. KV cache (all layers)  (debug_talker_kvcache_ax vs python_kvcache)
    4. last raw hidden        (talker_prefill_last_raw_hidden.bin vs python_prefill_last_raw_hidden.bin)
    5. last normed hidden     (talker_prefill_last_normed_hidden.bin vs python_prefill_last_normed_hidden.bin)
    6. logits                 (debug_talker_prefill_logits_ax.bin vs python_prefill_logits.bin)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_fp32_bin(path: Path, shape: tuple = None) -> np.ndarray:
    if not path.exists():
        return None
    raw = path.read_bytes()
    arr = np.frombuffer(raw, dtype=np.float32)
    if shape is not None:
        expected = int(np.prod(shape))
        if arr.size != expected:
            print(f"[WARN] {path.name}: size mismatch got {arr.size}, expected {expected} for shape {shape}")
            # try to auto-reshape if one dim is -1
            if shape.count(-1) == 1:
                arr = arr.reshape(shape)
            else:
                return arr
        else:
            arr = arr.reshape(shape)
    return arr


def load_bf16_bin(path: Path, shape: tuple = None) -> np.ndarray:
    if not path.exists():
        return None
    raw = path.read_bytes()
    u16 = np.frombuffer(raw, dtype=np.uint16)
    fp32 = (u16.astype(np.uint32) << 16).view(np.float32)
    if shape is not None:
        expected = int(np.prod(shape))
        if fp32.size != expected:
            print(f"[WARN] {path.name}: size mismatch got {fp32.size}, expected {expected} for shape {shape}")
            if shape.count(-1) == 1:
                fp32 = fp32.reshape(shape)
            else:
                return fp32
        else:
            fp32 = fp32.reshape(shape)
    return fp32


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def argmax_topk(arr: np.ndarray, k: int = 5):
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    k = min(max(1, int(k)), flat.shape[0])
    idx = np.argsort(flat)[-k:][::-1]
    return [(int(i), float(flat[i])) for i in idx]


def report(name: str, cpp_arr: np.ndarray, py_arr: np.ndarray):
    if cpp_arr is None or py_arr is None:
        return
    if cpp_arr.shape != py_arr.shape:
        print(f"\n[{name}] SHAPE MISMATCH cpp={cpp_arr.shape} py={py_arr.shape}")
        return
    cos = cosine_sim(cpp_arr, py_arr)
    max_d = max_abs_diff(cpp_arr, py_arr)
    mean_d = mean_abs_diff(cpp_arr, py_arr)
    print(f"\n[{name}] shape={cpp_arr.shape}")
    print(f"  cosine={cos:.8f}  max_diff={max_d:.8f}  mean_diff={mean_d:.8f}")
    if cpp_arr.size > 0 and cpp_arr.ndim >= 1 and cpp_arr.shape[-1] > 1:
        cpp_argmax = int(np.argmax(cpp_arr.reshape(-1)))
        py_argmax = int(np.argmax(py_arr.reshape(-1)))
        match = "✓" if cpp_argmax == py_argmax else "✗"
        print(f"  cpp_argmax={cpp_argmax}  py_argmax={py_argmax}  {match}")
        print(f"  cpp_top5={argmax_topk(cpp_arr, 5)}")
        print(f"  py_top5={argmax_topk(py_arr, 5)}")


def compare_checkpoint(name: str, cpp_path: Path, py_path: Path, shape_hint: tuple = None, cpp_is_bf16: bool = False, py_is_bf16: bool = False):
    if not cpp_path.exists() and not py_path.exists():
        return
    if not cpp_path.exists():
        print(f"\n[{name}] SKIP: C++ file missing: {cpp_path}")
        return
    if not py_path.exists():
        print(f"\n[{name}] SKIP: Python file missing: {py_path}")
        return

    if cpp_is_bf16:
        cpp_arr = load_bf16_bin(cpp_path, shape_hint)
    else:
        cpp_arr = load_fp32_bin(cpp_path, shape_hint)

    if py_is_bf16:
        py_arr = load_bf16_bin(py_path, shape_hint)
    else:
        py_arr = load_fp32_bin(py_path, shape_hint)

    report(name, cpp_arr, py_arr)


def compare_kv_cache(cpp_dir: Path, py_dir: Path, num_layers: int, valid_len: int, kv_dim: int):
    cpp_kvcache_dir = cpp_dir / "debug_talker_kvcache_ax"
    py_kvcache_dir = py_dir / "python_kvcache"

    for layer_idx in range(num_layers):
        name = f"3_kv_cache_layer_{layer_idx:02d}"
        cpp_k = cpp_kvcache_dir / f"layer_{layer_idx:02d}_k.bin"
        cpp_v = cpp_kvcache_dir / f"layer_{layer_idx:02d}_v.bin"
        py_k = py_kvcache_dir / f"layer_{layer_idx:02d}_k.bin"
        py_v = py_kvcache_dir / f"layer_{layer_idx:02d}_v.bin"

        if cpp_k.exists() and py_k.exists():
            cpp_arr = load_fp32_bin(cpp_k, (valid_len, kv_dim))
            py_arr = load_fp32_bin(py_k, (valid_len, kv_dim))
            report(f"{name}_k", cpp_arr, py_arr)
        if cpp_v.exists() and py_v.exists():
            cpp_arr = load_fp32_bin(cpp_v, (valid_len, kv_dim))
            py_arr = load_fp32_bin(py_v, (valid_len, kv_dim))
            report(f"{name}_v", cpp_arr, py_arr)


def main():
    parser = argparse.ArgumentParser(description="Compare C++ vs Python prefill checkpoints")
    parser.add_argument("--cpp_dir", type=Path, required=True, help="C++ dump directory")
    parser.add_argument("--py_dir", type=Path, required=True, help="Python dump directory")
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--vocab_size", type=int, default=3072)
    parser.add_argument("--S", type=int, default=None, help="Prefill sequence length")
    parser.add_argument("--num_layers", type=int, default=28)
    parser.add_argument("--kv_dim", type=int, default=256)
    args = parser.parse_args()

    # Read meta.json from either dir
    meta_path = args.py_dir / "meta.json"
    if not meta_path.exists():
        meta_path = args.cpp_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        S = args.S or meta.get("S", 8)
        hidden_size = meta.get("hidden_size", args.hidden_size)
        vocab_size = meta.get("vocab_size", args.vocab_size)
    else:
        S = args.S or 8
        hidden_size = args.hidden_size
        vocab_size = args.vocab_size

    print(f"=== Prefill Checkpoint Comparison ===")
    print(f"S={S} hidden_size={hidden_size} vocab_size={vocab_size}")
    print(f"cpp_dir={args.cpp_dir}")
    print(f"py_dir={args.py_dir}")

    # 1. prefill input
    compare_checkpoint(
        "1_prefill_input",
        args.cpp_dir / "talker_prefill_input.bin",
        args.py_dir / "prefill_embeds.bin",
        shape_hint=(S, hidden_size),
        cpp_is_bf16=False,
        py_is_bf16=True,
    )

    # 2. layer0 output
    compare_checkpoint(
        "2_layer0_output",
        args.cpp_dir / "talker_layer0_prefill_output.bin",
        args.py_dir / "python_layer0_prefill_output.bin",
        shape_hint=(S, hidden_size),
        cpp_is_bf16=False,
        py_is_bf16=True,
    )

    # 3. KV cache
    compare_kv_cache(args.cpp_dir, args.py_dir, args.num_layers, S, args.kv_dim)

    # 4. last raw hidden (pre-RMSNorm)
    compare_checkpoint(
        "4_last_raw_hidden",
        args.cpp_dir / "debug_talker_prefill_last_hidden_ax.bin",
        args.py_dir / "prefill_last_raw_hidden.bin",
        shape_hint=(hidden_size,),
    )

    # 4b. last normed hidden (post-RMSNorm)
    compare_checkpoint(
        "4_last_normed_hidden",
        args.cpp_dir / "prefill_last_normed_hidden_ax.bin",
        args.py_dir / "prefill_last_normed_hidden.bin",
        shape_hint=(hidden_size,),
    )

    # 5. logits
    compare_checkpoint(
        "5_prefill_logits",
        args.cpp_dir / "debug_talker_prefill_logits_ax.bin",
        args.py_dir / "python_prefill_logits.bin",
        shape_hint=(vocab_size,),
    )

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
