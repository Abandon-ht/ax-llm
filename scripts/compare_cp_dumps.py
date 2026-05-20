#!/usr/bin/env python3
"""Compare C++ vs Python CP debug dumps.

Supports both legacy flat layouts and new cpp_cp_dump/ python_cp_dump/ subdirs.

Usage:
    python compare_cp_dumps.py --cpp-dir ./cpp_dump --py-dir ./py_dump
    python compare_cp_dumps.py --cpp-dir ./cpp_dump --py-dir ./py_dump --frame 0 -v
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def cosine(a, b):
    a = a.flatten()
    b = b.flatten()
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def max_diff(a, b):
    return np.max(np.abs(a - b))


def find_first_diff_index(a, b, tol=1e-6):
    diff = np.abs(a - b) > tol
    idx = np.where(diff)[0]
    return int(idx[0]) if len(idx) > 0 else -1


def resolve_cp_dir(base_dir: Path) -> Path:
    """Return cpp_cp_dump/ if it exists, else base_dir itself."""
    sub = base_dir / "cpp_cp_dump"
    return sub if sub.exists() else base_dir


def resolve_py_dir(base_dir: Path) -> Path:
    sub = base_dir / "python_cp_dump"
    return sub if sub.exists() else base_dir


def compare_cp_lm_head_logits(cpp_dir: Path, py_dir: Path, frame_idx: int = 0, verbose: bool = False):
    print("=" * 70)
    print(f"CP lm_head logits comparison (frame {frame_idx})")
    print("=" * 70)
    first_diverge = None
    for j in range(15):
        # New naming
        cpp = cpp_dir / f"cpp_cp_frame_{frame_idx:03d}_lm_head_{j:03d}_logits.bin"
        py = py_dir / f"python_cp_frame_{frame_idx:03d}_lm_head_{j:03d}_logits.bin"
        # Legacy fallback
        if not cpp.exists():
            cpp = cpp_dir / f"cpp_cp_lm_head_{j:03d}_logits.bin"
        if not py.exists():
            py = py_dir / f"python_cp_frame_001_lm_head_{j:03d}_logits.bin"

        if not cpp.exists() or not py.exists():
            print(f"  step {j}: missing files (cpp={cpp.exists()}, py={py.exists()})")
            continue
        cpp_logits = np.fromfile(cpp, dtype=np.float32)
        py_logits = np.fromfile(py, dtype=np.float32)
        if len(cpp_logits) != len(py_logits):
            print(f"  step {j}: size mismatch cpp={len(cpp_logits)} py={len(py_logits)}")
            continue
        cos = cosine(cpp_logits, py_logits)
        md = max_diff(cpp_logits, py_logits)
        cpp_argmax = int(np.argmax(cpp_logits))
        py_argmax = int(np.argmax(py_logits))
        match = "MATCH" if cpp_argmax == py_argmax else "DIVERGE"
        if cpp_argmax != py_argmax and first_diverge is None:
            first_diverge = j
        print(f"  step {j}: cos={cos:.6f} max_diff={md:.6f} cpp_argmax={cpp_argmax} py_argmax={py_argmax} {match}")
        if verbose and md > 1e-6:
            first_diff = find_first_diff_index(cpp_logits, py_logits)
            print(f"           first_diff_idx={first_diff} cpp_val={cpp_logits[first_diff]:.6f} py_val={py_logits[first_diff]:.6f}")
    if first_diverge is not None:
        print(f"\n  >>> FIRST DIVERGENCE at step {first_diverge}")
    else:
        print(f"\n  >>> ALL 15 STEPS MATCH")


def compare_cp_hidden(cpp_dir: Path, py_dir: Path, frame_idx: int = 0, verbose: bool = False):
    print("=" * 70)
    print(f"CP hidden state comparison (frame {frame_idx})")
    print("=" * 70)
    for j in range(15):
        # New naming
        cpp_pre = cpp_dir / f"cpp_cp_frame_{frame_idx:03d}_hidden_pre_norm_{j:03d}.bin"
        py_pre = py_dir / f"python_cp_frame_{frame_idx:03d}_hidden_pre_norm_{j:03d}.bin"
        cpp_post = cpp_dir / f"cpp_cp_frame_{frame_idx:03d}_hidden_post_norm_{j:03d}.bin"
        py_post = py_dir / f"python_cp_frame_{frame_idx:03d}_hidden_post_norm_{j:03d}.bin"
        # Legacy fallback
        if not cpp_pre.exists():
            cpp_pre = cpp_dir / f"cpp_cp_frame_000_hidden_pre_norm_{j:03d}.bin"
        if not py_pre.exists():
            py_pre = py_dir / f"python_cp_frame_001_hidden_pre_norm_{j:03d}.bin"
        if not cpp_post.exists():
            cpp_post = cpp_dir / f"cpp_cp_frame_000_hidden_post_norm_{j:03d}.bin"
        if not py_post.exists():
            py_post = py_dir / f"python_cp_frame_001_hidden_post_norm_{j:03d}.bin"

        if cpp_pre.exists() and py_pre.exists():
            cpp_h = np.fromfile(cpp_pre, dtype=np.float32)
            py_h = np.fromfile(py_pre, dtype=np.float32)
            cos = cosine(cpp_h, py_h)
            md = max_diff(cpp_h, py_h)
            print(f"  step {j} pre_norm:  cos={cos:.6f} max_diff={md:.6f}")
            if verbose and md > 1e-6:
                first_diff = find_first_diff_index(cpp_h, py_h)
                print(f"           first_diff_idx={first_diff}")

        if cpp_post.exists() and py_post.exists():
            cpp_h = np.fromfile(cpp_post, dtype=np.float32)
            py_h = np.fromfile(py_post, dtype=np.float32)
            cos = cosine(cpp_h, py_h)
            md = max_diff(cpp_h, py_h)
            print(f"  step {j} post_norm: cos={cos:.6f} max_diff={md:.6f}")
            if verbose and md > 1e-6:
                first_diff = find_first_diff_index(cpp_h, py_h)
                print(f"           first_diff_idx={first_diff}")


def compare_cp_input_embeds(cpp_dir: Path, py_dir: Path, frame_idx: int = 0):
    print("=" * 70)
    print(f"CP prefill input embeds comparison (frame {frame_idx})")
    print("=" * 70)
    cpp = cpp_dir / f"cpp_cp_frame_{frame_idx:03d}_input_embeds.bin"
    py = py_dir / f"python_cp_frame_{frame_idx:03d}_input_embeds.bin"
    if not cpp.exists() or not py.exists():
        print(f"  missing files (cpp={cpp.exists()}, py={py.exists()}), skip")
        return
    cpp_arr = np.fromfile(cpp, dtype=np.float32)
    py_arr = np.fromfile(py, dtype=np.float32)
    if len(cpp_arr) != len(py_arr):
        print(f"  size mismatch cpp={len(cpp_arr)} py={len(py_arr)}")
        return
    cos = cosine(cpp_arr, py_arr)
    md = max_diff(cpp_arr, py_arr)
    print(f"  cos={cos:.6f} max_diff={md:.6f} {'MATCH' if md < 1e-4 else 'DIVERGE'}")


def compare_cp_sampled_tokens(cpp_dir: Path, py_dir: Path, frame_idx: int = 0):
    print("=" * 70)
    print(f"CP sampled tokens comparison (frame {frame_idx})")
    print("=" * 70)
    first_diverge = None
    for j in range(15):
        cpp = cpp_dir / f"cpp_cp_frame_{frame_idx:03d}_sampled_token_{j:03d}.bin"
        py = py_dir / f"python_cp_frame_{frame_idx:03d}_sampled_token_{j:03d}.bin"
        if not cpp.exists() or not py.exists():
            print(f"  step {j}: missing files (cpp={cpp.exists()}, py={py.exists()})")
            continue
        cpp_tok = int(np.fromfile(cpp, dtype=np.int32)[0])
        py_tok = int(np.fromfile(py, dtype=np.int32)[0])
        match = "MATCH" if cpp_tok == py_tok else "DIVERGE"
        if cpp_tok != py_tok and first_diverge is None:
            first_diverge = j
        print(f"  step {j}: cpp={cpp_tok} py={py_tok} {match}")
    if first_diverge is not None:
        print(f"\n  >>> FIRST DIVERGENCE at step {first_diverge}")
    else:
        print(f"\n  >>> ALL 15 STEPS MATCH")


def compare_cp_codec_sum(cpp_dir: Path, py_dir: Path, frame_idx: int = 0):
    print("=" * 70)
    print(f"CP codec_sum comparison (frame {frame_idx})")
    print("=" * 70)
    cpp = cpp_dir / f"cpp_cp_frame_{frame_idx:03d}_codec_sum.bin"
    py = py_dir / f"python_cp_frame_{frame_idx:03d}_codec_sum.bin"
    if not cpp.exists() or not py.exists():
        print(f"  missing files (cpp={cpp.exists()}, py={py.exists()}), skip")
        return
    cpp_arr = np.fromfile(cpp, dtype=np.float32)
    py_arr = np.fromfile(py, dtype=np.float32)
    if len(cpp_arr) != len(py_arr):
        print(f"  size mismatch cpp={len(cpp_arr)} py={len(py_arr)}")
        return
    cos = cosine(cpp_arr, py_arr)
    md = max_diff(cpp_arr, py_arr)
    print(f"  cos={cos:.6f} max_diff={md:.6f} {'MATCH' if md < 1e-4 else 'DIVERGE'}")


def compare_talker_decode_inputs(cpp_dir: Path, py_dir: Path):
    print("=" * 70)
    print("Talker decode step 1 input comparison")
    print("=" * 70)
    names = ["k_cache", "v_cache", "indices", "mask"]
    all_match = True
    for name in names:
        cpp = cpp_dir / f"talker_decode_step1_{name}.bin"
        py = py_dir / f"python_talker_decode_step1_{name}.bin"
        if not cpp.exists() or not py.exists():
            print(f"  {name}: missing files (cpp={cpp.exists()}, py={py.exists()})")
            all_match = False
            continue
        if name == "indices":
            cpp_data = np.fromfile(cpp, dtype=np.uint32)
            py_data = np.fromfile(py, dtype=np.uint32)
            match = np.array_equal(cpp_data, py_data)
            print(f"  {name}: cpp={cpp_data} py={py_data} match={match}")
            if not match:
                all_match = False
        else:
            cpp_data = np.fromfile(cpp, dtype=np.float32)
            py_data = np.fromfile(py, dtype=np.float32)
            if len(cpp_data) != len(py_data):
                print(f"  {name}: size mismatch cpp={len(cpp_data)} py={len(py_data)}")
                all_match = False
                continue
            cos = cosine(cpp_data, py_data)
            md = max_diff(cpp_data, py_data)
            match = md < 1e-6
            print(f"  {name}: cos={cos:.6f} max_diff={md:.6f} match={match}")
            if not match:
                all_match = False
    if all_match:
        print(f"\n  >>> TALKER DECODE INPUTS ALL MATCH")
    else:
        print(f"\n  >>> TALKER DECODE INPUTS HAVE DIVERGENCE")


def compare_prefill_last_hidden(cpp_dir: Path, py_dir: Path):
    print("=" * 70)
    print("Talker prefill last hidden comparison (optional)")
    print("=" * 70)

    # Compare raw hidden (pre-RMSNorm): C++ embed vs Python prefill_last_raw_hidden
    cpp_raw = cpp_dir / "debug_talker_prefill_last_hidden_ax.bin"
    py_raw = py_dir / "prefill_last_raw_hidden.bin"
    if cpp_raw.exists() and py_raw.exists():
        cpp_h = np.fromfile(cpp_raw, dtype=np.float32)
        py_h = np.fromfile(py_raw, dtype=np.float32)
        cos = cosine(cpp_h, py_h)
        md = max_diff(cpp_h, py_h)
        print(f"  [raw hidden]  cos={cos:.6f} max_diff={md:.6f}")
    else:
        print(f"  [raw hidden]  missing (cpp={cpp_raw.exists()}, py={py_raw.exists()}), skip")

    # Compare normed hidden (post-RMSNorm): C++ all_prefill_hidden[-1] vs Python prefill_last_normed_hidden
    cpp_normed = cpp_dir / "prefill_last_normed_hidden_ax.bin"
    py_normed = py_dir / "prefill_last_normed_hidden.bin"
    if cpp_normed.exists() and py_normed.exists():
        cpp_h = np.fromfile(cpp_normed, dtype=np.float32)
        py_h = np.fromfile(py_normed, dtype=np.float32)
        cos = cosine(cpp_h, py_h)
        md = max_diff(cpp_h, py_h)
        print(f"  [normed hidden] cos={cos:.6f} max_diff={md:.6f}")
    else:
        print(f"  [normed hidden] missing (cpp={cpp_normed.exists()}, py={py_normed.exists()}), skip")


def main():
    parser = argparse.ArgumentParser(description="Compare C++ vs Python CP debug dumps")
    parser.add_argument("dump_dir", nargs="?", default="/tmp/cp_debug", help="Directory containing both C++ and Python dumps")
    parser.add_argument("--cpp-dir", default=None, help="Directory containing C++ dumps")
    parser.add_argument("--py-dir", default=None, help="Directory containing Python dumps")
    parser.add_argument("--frame", type=int, default=0, help="Talker decode frame index to compare")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print first differing index")
    args = parser.parse_args()

    base_cpp = Path(args.cpp_dir) if args.cpp_dir else Path(args.dump_dir)
    base_py = Path(args.py_dir) if args.py_dir else Path(args.dump_dir)

    cpp_dir = resolve_cp_dir(base_cpp)
    py_dir = resolve_py_dir(base_py)

    print(f"C++ dump dir:  {cpp_dir}")
    print(f"Python dump dir: {py_dir}")
    print()

    compare_cp_input_embeds(cpp_dir, py_dir, frame_idx=args.frame)
    print()
    compare_cp_hidden(cpp_dir, py_dir, frame_idx=args.frame, verbose=args.verbose)
    print()
    compare_cp_lm_head_logits(cpp_dir, py_dir, frame_idx=args.frame, verbose=args.verbose)
    print()
    compare_cp_sampled_tokens(cpp_dir, py_dir, frame_idx=args.frame)
    print()
    compare_cp_codec_sum(cpp_dir, py_dir, frame_idx=args.frame)
    print()
    compare_talker_decode_inputs(base_cpp, base_py)
    print()
    compare_prefill_last_hidden(base_cpp, base_py)


if __name__ == "__main__":
    main()
