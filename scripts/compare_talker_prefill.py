#!/usr/bin/env python3
"""
对比 AX Talker 和 ONNX Talker 的 prefill 输出

用法:
    python3 compare_talker_prefill.py <npy_dir>

输入文件（由 qwen3_tts_ablation 生成）:
    <npy_dir>/debug_talker_prefill_last_hidden_ax.bin   [hidden_size] float32
    <npy_dir>/debug_talker_prefill_logits_ax.bin        [vocab_size] float32
    <npy_dir>/debug_talker_prefill_last_hidden_onnx.bin [hidden_size] float32
    <npy_dir>/debug_talker_prefill_logits_onnx.bin      [vocab_size] float32
"""

import sys
import struct
import numpy as np
from pathlib import Path


def load_float32_bin(path, expected_elems=None):
    data = np.fromfile(path, dtype=np.float32)
    if expected_elems is not None and len(data) != expected_elems:
        print(f"WARNING: {path} has {len(data)} elements, expected {expected_elems}")
    return data


def compare_vectors(ax, gt, name):
    if len(ax) != len(gt):
        print(f"[{name}] SHAPE MISMATCH: AX={len(ax)} vs ONNX={len(gt)}")
        return

    cos_sim = float(np.dot(ax, gt) / (np.linalg.norm(ax) * np.linalg.norm(gt) + 1e-12))
    mse = float(np.mean((ax - gt) ** 2))
    max_diff = float(np.max(np.abs(ax - gt)))
    mean_diff = float(np.mean(np.abs(ax - gt)))
    ax_argmax = int(np.argmax(ax))
    gt_argmax = int(np.argmax(gt))
    top5_match = len(set(np.argsort(ax)[-5:]) & set(np.argsort(gt)[-5:]))

    print(f"\n{'='*60}")
    print(f"[{name}] 对比结果")
    print(f"{'='*60}")
    print(f"  长度          : {len(ax)}")
    print(f"  Cosine Sim    : {cos_sim:.8f}")
    print(f"  MSE           : {mse:.8f}")
    print(f"  Max Abs Diff  : {max_diff:.8f}")
    print(f"  Mean Abs Diff : {mean_diff:.8f}")
    print(f"  AX argmax     : {ax_argmax} (value={ax[ax_argmax]:.6f})")
    print(f"  ONNX argmax   : {gt_argmax} (value={gt[gt_argmax]:.6f})")
    print(f"  Top-5 overlap : {top5_match}/5")

    if cos_sim > 0.99999 and max_diff < 1e-4:
        print(f"  判定          : ✅ 高度一致")
    elif cos_sim > 0.999 and max_diff < 1e-3:
        print(f"  判定          : ⚠️  轻微偏差（可能为 BF16 量化误差）")
    elif cos_sim > 0.99 and max_diff < 1e-2:
        print(f"  判定          : ⚠️  中等偏差（需进一步排查）")
    else:
        print(f"  判定          : ❌ 严重偏差（实现或模型转换有问题）")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <npy_dir>")
        sys.exit(1)

    npy_dir = Path(sys.argv[1])
    print(f"对比目录: {npy_dir}")

    files = {
        "last_hidden": (
            npy_dir / "debug_talker_prefill_last_hidden_ax.bin",
            npy_dir / "debug_talker_prefill_last_hidden_onnx.bin",
        ),
        "logits": (
            npy_dir / "debug_talker_prefill_logits_ax.bin",
            npy_dir / "debug_talker_prefill_logits_onnx.bin",
        ),
    }

    for name, (ax_path, gt_path) in files.items():
        if not ax_path.exists():
            print(f"ERROR: AX file not found: {ax_path}")
            continue
        if not gt_path.exists():
            print(f"ERROR: ONNX file not found: {gt_path}")
            continue

        ax = load_float32_bin(ax_path)
        gt = load_float32_bin(gt_path)
        compare_vectors(ax, gt, name)

    print("\n" + "="*60)
    print("诊断建议")
    print("="*60)
    print("""
1. 如果 last_hidden 和 logits 都 ✅ 高度一致：
   → AX Talker 的 prefill 阶段无问题，偏差来源于 decode 阶段
   → 下一步：对比 decode 单步输出（Phase 1.2）

2. 如果 last_hidden ⚠️/❌ 但 logits ✅：
   → LM head（post.axmodel）可能补偿了 hidden 偏差，或 hidden 偏差在最后一层被放大
   → 需排查 transformer 层输出

3. 如果 logits ⚠️/❌：
   → prefill 阶段即存在精度/实现问题
   → 可能原因：BF16 量化损失、模型权重加载错误、layer norm 差异、KV cache 初始化差异
""")


if __name__ == "__main__":
    main()
