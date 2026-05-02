#!/usr/bin/env python3
"""
对比 AX Talker 和 ONNX Talker 的 prefill KV cache

用法:
    python3 compare_talker_kvcache.py <npy_dir>

输入文件（由 qwen3_tts_ablation 生成）:
    <npy_dir>/debug_talker_kvcache_ax/layer_{00..NN}_{k,v}.bin   [seq_len * kv_cache_size] float32
    <npy_dir>/debug_talker_kvcache_onnx/layer_{00..NN}_{k,v}.bin [seq_len * kv_cache_size] float32
"""

import sys
import struct
import numpy as np
from pathlib import Path


def load_float32_bin(path, expected_elems=None):
    if not path.exists():
        return None
    data = np.fromfile(path, dtype=np.float32)
    if expected_elems is not None and len(data) != expected_elems:
        print(f"WARNING: {path} has {len(data)} elements, expected {expected_elems}")
    return data


def compare_vectors(ax, gt, name):
    if ax is None or gt is None:
        return None
    if len(ax) != len(gt):
        print(f"[{name}] SHAPE MISMATCH: AX={len(ax)} vs ONNX={len(gt)}")
        return None

    cos_sim = float(np.dot(ax, gt) / (np.linalg.norm(ax) * np.linalg.norm(gt) + 1e-12))
    mse = float(np.mean((ax - gt) ** 2))
    max_diff = float(np.max(np.abs(ax - gt)))
    mean_diff = float(np.mean(np.abs(ax - gt)))
    ax_argmax = int(np.argmax(ax))
    gt_argmax = int(np.argmax(gt))

    return {
        "name": name,
        "len": len(ax),
        "cos_sim": cos_sim,
        "mse": mse,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "ax_argmax": ax_argmax,
        "gt_argmax": gt_argmax,
    }


def print_result(r):
    if r is None:
        return
    print(f"  [{r['name']}] len={r['len']} cos_sim={r['cos_sim']:.8f} mse={r['mse']:.8f} "
          f"max_diff={r['max_diff']:.8f} mean_diff={r['mean_diff']:.8f} "
          f"argmax(AX={r['ax_argmax']}, ONNX={r['gt_argmax']})")


def judge(cos_sim, max_diff):
    if cos_sim > 0.99999 and max_diff < 1e-4:
        return "✅ 高度一致"
    elif cos_sim > 0.999 and max_diff < 1e-3:
        return "⚠️  轻微偏差（可能为 BF16 量化误差）"
    elif cos_sim > 0.99 and max_diff < 1e-2:
        return "⚠️  中等偏差（需进一步排查）"
    else:
        return "❌ 严重偏差"


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <npy_dir>")
        sys.exit(1)

    npy_dir = Path(sys.argv[1])
    ax_dir = npy_dir / "debug_talker_kvcache_ax"
    onnx_dir = npy_dir / "debug_talker_kvcache_onnx"

    print(f"对比目录: {npy_dir}")
    print(f"AX    KV cache: {ax_dir}")
    print(f"ONNX  KV cache: {onnx_dir}")

    if not ax_dir.exists():
        print(f"ERROR: AX KV cache dir not found: {ax_dir}")
        sys.exit(1)
    if not onnx_dir.exists():
        print(f"ERROR: ONNX KV cache dir not found: {onnx_dir}")
        sys.exit(1)

    # Discover layers
    ax_files = sorted(ax_dir.glob("layer_*_k.bin"))
    layers = []
    for f in ax_files:
        # Extract layer index from filename, e.g., "layer_00_k.bin" -> 0
        parts = f.stem.split("_")
        if len(parts) >= 3 and parts[0] == "layer":
            layers.append(int(parts[1]))
    layers = sorted(set(layers))

    if not layers:
        print("ERROR: No layer files found in AX KV cache dir")
        sys.exit(1)

    print(f"发现层数: {len(layers)} (layer {min(layers)} ~ {max(layers)})")
    print("=" * 80)

    bad_layers = []
    all_k_cos = []
    all_v_cos = []

    for layer in layers:
        ax_k = ax_dir / f"layer_{layer:02d}_k.bin"
        ax_v = ax_dir / f"layer_{layer:02d}_v.bin"
        onnx_k = onnx_dir / f"layer_{layer:02d}_k.bin"
        onnx_v = onnx_dir / f"layer_{layer:02d}_v.bin"

        k_ax = load_float32_bin(ax_k)
        k_onnx = load_float32_bin(onnx_k)
        v_ax = load_float32_bin(ax_v)
        v_onnx = load_float32_bin(onnx_v)

        if k_ax is None or k_onnx is None or v_ax is None or v_onnx is None:
            print(f"\n[Layer {layer:02d}] SKIP: missing file")
            continue

        res_k = compare_vectors(k_ax, k_onnx, f"L{layer:02d}_K")
        res_v = compare_vectors(v_ax, v_onnx, f"L{layer:02d}_V")

        print(f"\n[Layer {layer:02d}]")
        print_result(res_k)
        print_result(res_v)

        if res_k:
            all_k_cos.append(res_k["cos_sim"])
            if res_k["cos_sim"] < 0.999 or res_k["max_diff"] >= 1e-3:
                bad_layers.append((layer, "K", res_k["cos_sim"], res_k["max_diff"]))
        if res_v:
            all_v_cos.append(res_v["cos_sim"])
            if res_v["cos_sim"] < 0.999 or res_v["max_diff"] >= 1e-3:
                bad_layers.append((layer, "V", res_v["cos_sim"], res_v["max_diff"]))

    print("\n" + "=" * 80)
    print("汇总")
    print("=" * 80)

    if all_k_cos:
        print(f"K-cache: min_cos_sim={min(all_k_cos):.8f}  mean_cos_sim={sum(all_k_cos)/len(all_k_cos):.8f}")
    if all_v_cos:
        print(f"V-cache: min_cos_sim={min(all_v_cos):.8f}  mean_cos_sim={sum(all_v_cos)/len(all_v_cos):.8f}")

    if bad_layers:
        print(f"\n⚠️  偏差层数: {len(bad_layers)}")
        for layer, kv_type, cos_sim, max_diff in bad_layers[:100]:
            print(f"  Layer {layer:02d} {kv_type}: cos_sim={cos_sim:.8f}, max_diff={max_diff:.8f}  {judge(cos_sim, max_diff)}")
        if len(bad_layers) > 100:
            print(f"  ... and {len(bad_layers) - 10} more")
    else:
        print("\n✅ 所有层 KV cache 高度一致")

    print("\n" + "=" * 80)
    print("诊断建议")
    print("=" * 80)
    print("""
1. 如果所有层 K/V cache 都 ✅ 高度一致：
   → prefill 阶段的 attention 实现无问题
   → 偏差可能来源于 layer norm / RMSNorm 或 post-processing

2. 如果某层 K-cache 或 V-cache 首先出现 ⚠️/❌：
   → 该层（或前一层）的 attention 实现或权重加载有问题
   → 建议对比该层的输入 embed（逐层 hidden state 对比）

3. 如果只有最后几层偏差：
   → 可能是量化误差在深层累积
   → 或 decode group 的 KV cache 同步有问题
""")


if __name__ == "__main__":
    main()
