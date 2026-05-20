#!/usr/bin/env python3
"""
Compare C++ vs Python Talker decode steps for Qwen3-TTS (L3 validation).

Usage:
    python3 scripts/compare_talker_decode_full.py \
        --cpp-dir ./debug_bin/cpp_dump \
        --py-dir ./debug_bin/py_dump \
        [--max-steps 10] [--hidden-size 1024] [--vocab-size 3072] [-v]

Checkpoints compared per step:
    1. codec_sum          (cpp_talker_decode_step{NNN}_codec_sum.bin)
    2. inputs_embeds      (cpp_talker_decode_step{NNN}_inputs_embeds.bin)
    3. raw_hidden         (cpp_talker_decode_step{NNN}_raw_hidden.bin)
    4. logits             (cpp_talker_decode_step{NNN}_logits.bin)
    5. next_token         (cpp_talker_decode_step{NNN}_next_token.bin)
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
        # Try subdirectory
        path = path.parent / "python_talker_decode" / path.name
        if not path.exists():
            return None
    arr = np.fromfile(path, dtype=np.float32)
    if expected_elems is not None and arr.size != expected_elems:
        print(f"[WARN] {path.name}: size mismatch got {arr.size}, expected {expected_elems}")
    return arr


def load_int32_bin(path: Path) -> int:
    if not path.exists():
        path = path.parent / "python_talker_decode" / path.name
        if not path.exists():
            return None
    arr = np.fromfile(path, dtype=np.int32)
    return int(arr[0]) if arr.size > 0 else None


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

    # For logits, check argmax; for vectors, just check cosine + max_diff
    has_argmax = cpp_arr.size > 1 and cpp_arr.ndim == 1
    argmax_match = None
    if has_argmax:
        cpp_argmax = int(np.argmax(cpp_arr))
        py_argmax = int(np.argmax(py_arr))
        argmax_match = cpp_argmax == py_argmax

    if argmax_match is not None:
        match_str = "MATCH" if argmax_match else "DIVERGE"
        print(f"  [{name}] cos={cos:.6f} max_diff={max_d:.6f} mean_diff={mean_d:.6f} "
              f"cpp_argmax={np.argmax(cpp_arr)} py_argmax={np.argmax(py_arr)} {match_str}")
    else:
        match_str = "MATCH" if max_d < 1e-4 else "DIVERGE"
        print(f"  [{name}] cos={cos:.6f} max_diff={max_d:.6f} mean_diff={mean_d:.6f} {match_str}")

    if verbose and max_d > 1e-6 and cpp_arr.size > 0:
        diff = np.abs(cpp_arr - py_arr)
        first_idx = int(np.argmax(diff > 1e-6)) if np.any(diff > 1e-6) else -1
        if first_idx >= 0:
            print(f"           first_diff_idx={first_idx} "
                  f"cpp={cpp_arr.flat[first_idx]:.6f} py={py_arr.flat[first_idx]:.6f}")

    if argmax_match is not None:
        return "MATCH" if argmax_match and max_d < 1e-3 else "MISMATCH"
    return "MATCH" if max_d < 1e-4 else "MISMATCH"


def compare_step(cpp_dir: Path, py_dir: Path, step: int, hidden_size: int, vocab_size: int, verbose: bool = False):
    prefix_cpp = f"cpp_talker_decode_step{step:03d}"
    prefix_py = f"python_talker_decode_step{step:03d}"

    print(f"\n[Talker Decode Step {step}]")

    results = {}

    # 1. codec_sum
    results["codec_sum"] = report(
        "codec_sum",
        load_fp32_bin(cpp_dir / f"{prefix_cpp}_codec_sum.bin", hidden_size),
        load_fp32_bin(py_dir / f"{prefix_py}_codec_sum.bin", hidden_size),
        verbose,
    )

    # 2. inputs_embeds
    results["inputs_embeds"] = report(
        "inputs_embeds",
        load_fp32_bin(cpp_dir / f"{prefix_cpp}_inputs_embeds.bin", hidden_size),
        load_fp32_bin(py_dir / f"{prefix_py}_inputs_embeds.bin", hidden_size),
        verbose,
    )

    # 3. raw_hidden
    results["raw_hidden"] = report(
        "raw_hidden",
        load_fp32_bin(cpp_dir / f"{prefix_cpp}_raw_hidden.bin", hidden_size),
        load_fp32_bin(py_dir / f"{prefix_py}_raw_hidden.bin", hidden_size),
        verbose,
    )

    # 4. logits
    results["logits"] = report(
        "logits",
        load_fp32_bin(cpp_dir / f"{prefix_cpp}_logits.bin", vocab_size),
        load_fp32_bin(py_dir / f"{prefix_py}_logits.bin", vocab_size),
        verbose,
    )

    # 5. next_token
    cpp_tok = load_int32_bin(cpp_dir / f"{prefix_cpp}_next_token.bin")
    # Python dump writes float32 (e.g. 1174.0); read as float32 then truncate
    py_tok_path = py_dir / f"{prefix_py}_next_token.bin"
    if not py_tok_path.exists():
        py_tok_path = py_tok_path.parent / "python_talker_decode" / py_tok_path.name
    if py_tok_path.exists():
        py_tok_raw = np.fromfile(py_tok_path, dtype=np.float32)
        py_tok = int(py_tok_raw[0]) if py_tok_raw.size > 0 else None
    else:
        py_tok = None
    if cpp_tok is None or py_tok is None:
        print(f"  [next_token] MISSING cpp={cpp_tok} py={py_tok}")
        results["next_token"] = "MISSING"
    else:
        match = "MATCH" if cpp_tok == py_tok else "DIVERGE"
        print(f"  [next_token] cpp={cpp_tok} py={py_tok} {match}")
        results["next_token"] = "MATCH" if cpp_tok == py_tok else "MISMATCH"

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare C++ vs Python Talker decode steps")
    parser.add_argument("--cpp-dir", type=Path, required=True, help="C++ dump directory")
    parser.add_argument("--py-dir", type=Path, required=True, help="Python dump directory")
    parser.add_argument("--max-steps", type=int, default=20, help="Max decode steps to compare")
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=3072)
    parser.add_argument("-v", "--verbose", action="store_true", help="Print first diff index")
    args = parser.parse_args()

    print(f"=== Talker Decode Full Comparison ===")
    print(f"cpp_dir={args.cpp_dir}")
    print(f"py_dir={args.py_dir}")
    print(f"max_steps={args.max_steps} hidden_size={args.hidden_size} vocab_size={args.vocab_size}")

    first_diverge_step = None
    first_diverge_field = None

    for step in range(args.max_steps):
        # Stop if C++ has no more files for this step
        cpp_exists = (args.cpp_dir / f"cpp_talker_decode_step{step:03d}_inputs_embeds.bin").exists()
        py_exists = (args.py_dir / f"python_talker_decode_step{step:03d}_inputs_embeds.bin").exists()
        if not cpp_exists and not py_exists:
            if step == 0:
                print("\nERROR: No decode step files found in either directory.")
                print("Expected files like: cpp_talker_decode_step000_inputs_embeds.bin")
                sys.exit(1)
            break

        results = compare_step(args.cpp_dir, args.py_dir, step, args.hidden_size, args.vocab_size, args.verbose)

        for field, status in results.items():
            if status in ("MISMATCH", "MISSING") and first_diverge_step is None:
                first_diverge_step = step
                first_diverge_field = field

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    if first_diverge_step is None:
        print("✅ All compared steps MATCH")
    else:
        print(f"❌ FIRST DIVERGENCE at step {first_diverge_step}, field={first_diverge_field}")
        print("\nDiagnosis hints:")
        if first_diverge_field == "codec_sum":
            print("  → CP embedding accumulation or primary_embed lookup differs. Check L2 (CP independent).")
        elif first_diverge_field == "inputs_embeds":
            print("  → trailing_text injection or codec_sum transfer differs. Check L0 trailing_text / tts_pad.")
        elif first_diverge_field == "raw_hidden":
            print("  → Talker decode KV cache / indices / mask differs. Check L1 KV cache consistency.")
        elif first_diverge_field == "logits":
            print("  → Post head / codec_head differs. Check talker post norm / lm head weights.")
        elif first_diverge_field == "next_token":
            print("  → Sampling logic differs (temperature/top_k/top_p) or earlier hidden divergence.")


if __name__ == "__main__":
    main()
