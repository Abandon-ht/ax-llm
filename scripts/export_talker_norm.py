#!/usr/bin/env python3
"""
Export talker.model.norm.weight to bfloat16 raw binary from safetensors.
"""
import sys
from pathlib import Path
import numpy as np


def fp32_to_bf16_raw(arr: np.ndarray) -> bytes:
    arr = np.asarray(arr, dtype=np.float32)
    u32 = arr.view(np.uint32)
    bf16 = (u32 >> 16).astype(np.uint16)
    return bf16.tobytes()


def main():
    hf_path = Path.home() / "rsp" / "Qwen3-TTS-12Hz-0.6B-Base"
    out_dir = Path.home() / "rsp" / "Qwen3-TTS-12Hz-0.6B-Base-AX650" / "talker"
    out_dir.mkdir(parents=True, exist_ok=True)

    safetensors_path = hf_path / "model.safetensors"
    if not safetensors_path.exists():
        print(f"[error] {safetensors_path} not found")
        sys.exit(1)

    print(f"[load] {safetensors_path}")
    try:
        from safetensors.torch import load_file
        state_dict = load_file(str(safetensors_path), device="cpu")
    except Exception as e:
        print(f"[error] failed to load safetensors: {e}")
        sys.exit(1)

    # Try possible keys for talker norm weight
    candidate_keys = [
        "model.talker.model.norm.weight",
        "model.talker.norm.weight",
        "talker.model.norm.weight",
    ]

    norm_weight = None
    chosen_key = None
    for k in candidate_keys:
        if k in state_dict:
            norm_weight = state_dict[k]
            chosen_key = k
            break

    if norm_weight is None:
        print("[error] Cannot find talker norm weight. Available keys containing 'norm.weight':")
        for k in sorted(state_dict.keys()):
            if "norm.weight" in k:
                print(f"  {k}: {state_dict[k].shape}")
        sys.exit(1)

    print(f"[found] key={chosen_key}, shape={tuple(norm_weight.shape)}, dtype={norm_weight.dtype}")

    # Convert to float32 numpy
    np_weight = norm_weight.detach().cpu().float().numpy()
    out_path = out_dir / "talker.model.norm.weight.bfloat16.bin"
    out_path.write_bytes(fp32_to_bf16_raw(np_weight))
    print(f"[save] {out_path} ({out_path.stat().st_size} bytes, hidden_size={np_weight.shape[0]})")
    print("[done]")


if __name__ == "__main__":
    main()
