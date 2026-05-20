#!/usr/bin/env python3
"""
Coupling flow validation: verify Talker-CP data construction at each decode step.

This script focuses on the critical coupling logic:
    inputs_embeds = codec_sum + trailing_text_hidden[step]

Usage:
    python3 scripts/compare_coupling_flow.py \
        --cpp-dir ./debug_bin/cpp_dump \
        --py-dir ./debug_bin/py_dump \
        [--max-steps 10] [--hidden-size 1024] [-v]
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


def load_fp32_bin(path: Path, expected_elems: int = None) -> np.ndarray:
    if not path.exists():
        return None
    arr = np.fromfile(path, dtype=np.float32)
    if expected_elems is not None and arr.size != expected_elems:
        print(f"[WARN] {path.name}: size mismatch got {arr.size}, expected {expected_elems}")
    return arr


def report_vec(name: str, cpp_arr: np.ndarray, py_arr: np.ndarray):
    if cpp_arr is None or py_arr is None:
        return None, "MISSING"
    if cpp_arr.shape != py_arr.shape:
        print(f"  [{name}] SHAPE_MISMATCH cpp={cpp_arr.shape} py={py_arr.shape}")
        return None, "MISMATCH"
    cos = cosine_sim(cpp_arr, py_arr)
    max_d = max_abs_diff(cpp_arr, py_arr)
    status = "MATCH" if max_d < 1e-4 else "DIVERGE"
    print(f"  [{name}] cos={cos:.6f} max_diff={max_d:.6f} {status}")
    return max_d, status


def main():
    parser = argparse.ArgumentParser(description="Coupling flow validation")
    parser.add_argument("--cpp-dir", type=Path, required=True, help="C++ dump directory")
    parser.add_argument("--py-dir", type=Path, required=True, help="Python dump directory")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("Coupling Flow Validation (L3)")
    print("=" * 70)
    print(f"cpp_dir={args.cpp_dir}")
    print(f"py_dir={args.py_dir}")

    first_diverge = None

    for step in range(args.max_steps):
        cpp_prefix = f"cpp_talker_decode_step{step:03d}"
        py_prefix = f"python_talker_decode_step{step:03d}"

        cpp_inputs = args.cpp_dir / f"{cpp_prefix}_inputs_embeds.bin"
        py_inputs = args.py_dir / f"{py_prefix}_inputs_embeds.bin"
        if not cpp_inputs.exists() and not py_inputs.exists():
            if step == 0:
                print("\nERROR: No decode step files found.")
                sys.exit(1)
            break

        print(f"\n[Step {step}]")

        cpp_codec_sum = load_fp32_bin(args.cpp_dir / f"{cpp_prefix}_codec_sum.bin", args.hidden_size)
        py_codec_sum = load_fp32_bin(args.py_dir / f"{py_prefix}_codec_sum.bin", args.hidden_size)
        _, st_codec = report_vec("codec_sum", cpp_codec_sum, py_codec_sum)

        cpp_inputs_embeds = load_fp32_bin(args.cpp_dir / f"{cpp_prefix}_inputs_embeds.bin", args.hidden_size)
        py_inputs_embeds = load_fp32_bin(args.py_dir / f"{py_prefix}_inputs_embeds.bin", args.hidden_size)
        _, st_inputs = report_vec("inputs_embeds", cpp_inputs_embeds, py_inputs_embeds)

        # If both codec_sum and inputs_embeds are present, compute residual
        if cpp_codec_sum is not None and cpp_inputs_embeds is not None:
            cpp_residual = cpp_inputs_embeds - cpp_codec_sum
            print(f"  [cpp_residual] mean={np.mean(cpp_residual):.6f} max={np.max(np.abs(cpp_residual)):.6f}")
        if py_codec_sum is not None and py_inputs_embeds is not None:
            py_residual = py_inputs_embeds - py_codec_sum
            print(f"  [py_residual]  mean={np.mean(py_residual):.6f} max={np.max(np.abs(py_residual)):.6f}")

        # Compare residuals (trailing_text / tts_pad)
        if (cpp_codec_sum is not None and cpp_inputs_embeds is not None and
            py_codec_sum is not None and py_inputs_embeds is not None):
            cpp_residual = cpp_inputs_embeds - cpp_codec_sum
            py_residual = py_inputs_embeds - py_codec_sum
            _, st_residual = report_vec("residual(trail)", cpp_residual, py_residual)
        else:
            st_residual = "MISSING"

        if first_diverge is None:
            for field, st in [("codec_sum", st_codec), ("inputs_embeds", st_inputs), ("residual", st_residual)]:
                if st in ("MISMATCH", "DIVERGE", "MISSING"):
                    first_diverge = (step, field)
                    break

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    if first_diverge is None:
        print("✅ All coupling flow steps MATCH")
    else:
        step, field = first_diverge
        print(f"❌ FIRST DIVERGENCE at step {step}, field={field}")
        if field == "codec_sum":
            print("  → Check CP embedding accumulation (fp32 sum) and primary/sub-code embed lookups.")
            print("  → Run L2 validation: compare_cp_dumps.py")
        elif field == "inputs_embeds":
            print("  → Check trailing_text injection index or tts_pad_embed usage.")
            print("  → Verify generation_step indexing matches Python.")
        elif field == "residual":
            print("  → trailing_text_hidden or tts_pad_embed values differ.")
            print("  → Check L0: trailing_text_hiddens.bin and tts_pad_vec.bin alignment.")


if __name__ == "__main__":
    main()
