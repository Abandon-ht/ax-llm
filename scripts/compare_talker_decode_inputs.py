#!/usr/bin/env python3
"""
Compare C++ vs Python Talker decode step 0 inputs (L3 deep-dive).

Usage:
    python3 scripts/compare_talker_decode_inputs.py \
        --cpp-dir ./debug_bin/cpp_dump \
        --py-dir ./debug_bin/py_dump \
        [--step 0] \
        [-v]

Checkpoints compared:
    1. k_cache_l00   (prefill后的layer 0 K cache)
    2. v_cache_l00   (prefill后的layer 0 V cache)
    3. indices       (decode indices)
    4. mask          (decode mask)
    5. layer outputs (逐层输出 hidden state, 支持 --scan-layers 自动二分定位)
"""

import argparse
import sys
from pathlib import Path

import numpy as np


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


def load_fp32_bin(path: Path, expected_elems: int = None) -> np.ndarray:
    if not path.exists():
        return None
    arr = np.fromfile(path, dtype=np.float32)
    if expected_elems is not None and arr.size != expected_elems:
        print(f"[WARN] {path.name}: size mismatch got {arr.size}, expected {expected_elems}")
    return arr


def report(name: str, cpp_arr: np.ndarray, py_arr: np.ndarray, verbose: bool = False):
    if cpp_arr is None or py_arr is None:
        status = "MISSING"
        detail = ""
        if cpp_arr is None:
            detail += " cpp_missing"
        if py_arr is None:
            detail += " py_missing"
        print(f"  [{name}] {status}{detail}")
        return "MISSING"

    if cpp_arr.shape != py_arr.shape:
        print(f"  [{name}] SHAPE_MISMATCH cpp={cpp_arr.shape} py={py_arr.shape}")
        return "MISMATCH"

    cos = cosine_sim(cpp_arr, py_arr)
    max_d = max_abs_diff(cpp_arr, py_arr)
    mean_d = mean_abs_diff(cpp_arr, py_arr)

    match_str = "MATCH" if max_d < 1e-4 else "DIVERGE"
    print(f"  [{name}] cos={cos:.6f} max_diff={max_d:.6f} mean_diff={mean_d:.6f} {match_str}")

    if verbose and max_d > 1e-6 and cpp_arr.size > 0:
        diff = np.abs(cpp_arr - py_arr)
        first_idx = int(np.argmax(diff > 1e-6)) if np.any(diff > 1e-6) else -1
        if first_idx >= 0:
            print(f"           first_diff_idx={first_idx} "
                  f"cpp={cpp_arr.flat[first_idx]:.6f} py={py_arr.flat[first_idx]:.6f}")

    return "MATCH" if max_d < 1e-4 else "MISMATCH"


def main():
    parser = argparse.ArgumentParser(description="Compare C++ vs Python Talker decode inputs")
    parser.add_argument("--cpp-dir", type=Path, required=True, help="C++ dump directory")
    parser.add_argument("--py-dir", type=Path, required=True, help="Python dump directory")
    parser.add_argument("--step", type=int, default=0, help="Decode step index to compare (default 0)")
    parser.add_argument("--layer", type=int, default=None, help="Compare a specific layer output (e.g. 14)")
    parser.add_argument("--scan-layers", action="store_true", help="Auto-scan all layer outputs to find first divergence")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print first diff index")
    args = parser.parse_args()

    step = args.step
    print(f"=== Talker Decode Inputs Comparison (step {step}) ===")
    print(f"cpp_dir={args.cpp_dir}")
    print(f"py_dir={args.py_dir}")

    # Python files may live in a subdirectory
    py_subdir = args.py_dir / "python_talker_decode"

    # 1. K cache (layer 0)
    cpp_k = args.cpp_dir / "debug_talker_kvcache_ax" / "layer_00_k.bin"
    py_k = py_subdir / f"python_talker_decode_step{step:03d}_k_cache_l00.bin"
    if not py_k.exists():
        py_k = args.py_dir / f"python_talker_decode_step{step:03d}_k_cache_l00.bin"
    result_k = report("k_cache_l00", load_fp32_bin(cpp_k), load_fp32_bin(py_k), args.verbose)

    # 2. V cache (layer 0)
    cpp_v = args.cpp_dir / "debug_talker_kvcache_ax" / "layer_00_v.bin"
    py_v = py_subdir / f"python_talker_decode_step{step:03d}_v_cache_l00.bin"
    if not py_v.exists():
        py_v = args.py_dir / f"python_talker_decode_step{step:03d}_v_cache_l00.bin"
    result_v = report("v_cache_l00", load_fp32_bin(cpp_v), load_fp32_bin(py_v), args.verbose)

    # 3. indices
    cpp_indices = args.cpp_dir / f"cpp_talker_decode_step{step:03d}_indices.bin"
    py_indices = py_subdir / f"python_talker_decode_step{step:03d}_indices.bin"
    if not py_indices.exists():
        py_indices = args.py_dir / f"python_talker_decode_step{step:03d}_indices.bin"
    result_indices = report("indices", load_fp32_bin(cpp_indices), load_fp32_bin(py_indices), args.verbose)

    # 4. mask
    cpp_mask = args.cpp_dir / f"cpp_talker_decode_step{step:03d}_mask.bin"
    py_mask = py_subdir / f"python_talker_decode_step{step:03d}_mask.bin"
    if not py_mask.exists():
        py_mask = args.py_dir / f"python_talker_decode_step{step:03d}_mask.bin"
    result_mask = report("mask", load_fp32_bin(cpp_mask), load_fp32_bin(py_mask), args.verbose)

    # 5. layer output(s)
    result_l0 = None
    if args.layer is not None:
        cpp_l = args.cpp_dir / f"cpp_talker_decode_step{step:03d}_layer{args.layer:02d}_output.bin"
        py_l = py_subdir / f"python_talker_decode_step{step:03d}_layer{args.layer:02d}_output.bin"
        if not py_l.exists():
            py_l = args.py_dir / f"python_talker_decode_step{step:03d}_layer{args.layer:02d}_output.bin"
        result_l0 = report(f"layer{args.layer:02d}_output", load_fp32_bin(cpp_l), load_fp32_bin(py_l), args.verbose)
    elif args.scan_layers:
        # auto-scan all available layers
        first_diverge_layer = None
        for layer_idx in range(100):
            cpp_l = args.cpp_dir / f"cpp_talker_decode_step{step:03d}_layer{layer_idx:02d}_output.bin"
            py_l = py_subdir / f"python_talker_decode_step{step:03d}_layer{layer_idx:02d}_output.bin"
            if not py_l.exists():
                py_l = args.py_dir / f"python_talker_decode_step{step:03d}_layer{layer_idx:02d}_output.bin"
            if not cpp_l.exists() or not py_l.exists():
                continue
            status = report(f"layer{layer_idx:02d}_output", load_fp32_bin(cpp_l), load_fp32_bin(py_l), args.verbose)
            if status != "MATCH" and first_diverge_layer is None:
                first_diverge_layer = layer_idx
        if first_diverge_layer is None:
            print("\n✅ All scanned layer outputs MATCH")
        else:
            print(f"\n❌ FIRST DIVERGENCE at layer {first_diverge_layer}")
    else:
        # default: compare layer 0 only if file exists
        cpp_l0 = args.cpp_dir / f"cpp_talker_decode_step{step:03d}_layer00_output.bin"
        py_l0 = py_subdir / f"python_talker_decode_step{step:03d}_layer00_output.bin"
        if not py_l0.exists():
            py_l0 = args.py_dir / f"python_talker_decode_step{step:03d}_layer00_output.bin"
        if cpp_l0.exists() or py_l0.exists():
            result_l0 = report("layer00_output", load_fp32_bin(cpp_l0), load_fp32_bin(py_l0), args.verbose)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    results = {
        "k_cache_l00": result_k,
        "v_cache_l00": result_v,
        "indices": result_indices,
        "mask": result_mask,
        "layer00_output": result_l0,
    }
    all_match = all(r == "MATCH" for r in results.values())
    if all_match:
        print("✅ All decode inputs MATCH")
        if args.scan_layers:
            # scan result already printed above
            pass
        elif results.get("layer00_output") == "MATCH":
            print("\n→ layer 0 output 也匹配 → 问题在 layer 1 ~ N-1 之间")
            print("→ 重新编译后使用 --scan-layers 自动定位第一个发散层")
        elif results.get("layer00_output") == "MISMATCH":
            print("\n→ layer 0 output 已发散 → 问题锁定在 layer 0 axmodel 内部")
            print("→ 检查 layer 0 weights / bias / RoPE / attention 实现差异")
        else:
            print("\n→ layer output 未 dump，无法进一步定位")
            print("→ 需重新跑 Python + C++ 生成 layerXX_output.bin")
    else:
        for name, status in results.items():
            if status != "MATCH":
                print(f"❌ {name}: {status}")
        print("\nDiagnosis hints:")
        if results["k_cache_l00"] != "MATCH" or results["v_cache_l00"] != "MATCH":
            print("  → KV cache 不一致: 检查 prefill 后 state 的保存逻辑")
            print("  → 检查 C++ prefill KV cache 更新位置 vs Python")
        if results["indices"] != "MATCH":
            print("  → indices 不一致: 检查 _last_static_position_id() 返回值 vs C++ decode_start")
            print("  → 检查 position_ids / cache_position 传递链条")
        if results["mask"] != "MATCH":
            print("  → mask 不一致: 检查 _build_decode_mask_cache() 的 fp32 数值 vs C++ mask vector")
            print("  → 特别关注最后一个元素是否为 0")


if __name__ == "__main__":
    main()
