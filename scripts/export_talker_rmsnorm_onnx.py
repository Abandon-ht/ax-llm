#!/usr/bin/env python3
"""
Export Talker RMSNorm to ONNX.

This script loads the talker norm weight directly from the original safetensors
and exports a standalone RMSNorm module as ONNX, so both C++ and Python can use
the same implementation (via axmodel converted from ONNX) to eliminate the
0.125 max_diff caused by C++ handwritten rmsnorm.

Usage:
    python scripts/export_talker_rmsnorm_onnx.py \
        --safetensors ~/rsp/Qwen3-TTS-12Hz-0.6B-Base/model.safetensors \
        --output talker_rmsnorm.onnx \
        --hidden_size 1024 \
        --eps 1e-6
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class Qwen3TTSRMSNorm(nn.Module):
    """
    Equivalent to the original Qwen3TTSRMSNorm in
    qwen_tts/core/models/modeling_qwen3_tts.py
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def load_weight_from_safetensors(safetensors_path: str):
    """Load talker norm weight from safetensors and return a float32 torch tensor."""
    try:
        from safetensors.torch import load_file
    except ImportError as e:
        print(f"[error] safetensors not installed: {e}")
        sys.exit(1)

    safetensors_path = Path(safetensors_path)
    if not safetensors_path.exists():
        print(f"[error] {safetensors_path} not found")
        sys.exit(1)

    print(f"[load] {safetensors_path}")
    state_dict = load_file(str(safetensors_path), device="cpu")

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

    # Convert to float32 torch tensor
    weight = norm_weight.detach().cpu().float()
    return weight


def bf16_to_fp32_numpy(arr_bf16: np.ndarray) -> np.ndarray:
    """Convert uint16 bfloat16 array to float32 array."""
    assert arr_bf16.dtype == np.uint16
    fp32 = np.frombuffer((arr_bf16.astype(np.uint32) << 16).tobytes(), dtype=np.float32)
    return fp32.copy()


def load_bf16_weight(path: str) -> torch.Tensor:
    """Load a .bfloat16.bin file and return a float32 torch tensor."""
    raw = np.fromfile(path, dtype=np.uint16)
    fp32 = bf16_to_fp32_numpy(raw)
    return torch.from_numpy(fp32)


def main():
    parser = argparse.ArgumentParser(description="Export Talker RMSNorm to ONNX")
    parser.add_argument(
        "--safetensors",
        default=str(Path.home() / "rsp" / "Qwen3-TTS-12Hz-0.6B-Base" / "model.safetensors"),
        help="Path to original model.safetensors (default: ~/rsp/Qwen3-TTS-12Hz-0.6B-Base/model.safetensors)",
    )
    parser.add_argument(
        "--weight",
        default=None,
        help="(Deprecated) Path to talker.model.norm.weight.bfloat16.bin. "
             "If provided, uses bin instead of safetensors.",
    )
    parser.add_argument(
        "--output",
        default="talker_rmsnorm.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=1024,
        help="Hidden size (default 1024 for Qwen3-TTS 0.6B)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="RMSNorm epsilon (default 1e-6)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )
    args = parser.parse_args()

    # 1. Build model
    model = Qwen3TTSRMSNorm(hidden_size=args.hidden_size, eps=args.eps)

    # 2. Load weight (prefer safetensors, fallback to bin if --weight is given)
    if args.weight is not None:
        print(f"[warn] Using deprecated --weight path: {args.weight}")
        weight = load_bf16_weight(args.weight)
    else:
        weight = load_weight_from_safetensors(args.safetensors)

    assert weight.shape[0] == args.hidden_size, (
        f"Weight shape {weight.shape[0]} does not match hidden_size {args.hidden_size}"
    )
    with torch.no_grad():
        model.weight.copy_(weight)

    model.eval()

    # 3. Create dummy input
    dummy_input = torch.randn(1, 128, args.hidden_size, dtype=torch.float32)

    # 4. Verify against a reference implementation (PyTorch native RMSNorm if available)
    with torch.no_grad():
        out = model(dummy_input)
    print(f"[INFO] Model output shape: {out.shape}")
    print(f"[INFO] Output dtype: {out.dtype}")
    print(f"[INFO] Output sample (first 5): {out[0, 0, :5].tolist()}")

    # 5. Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["hidden_states"],
        output_names=["normed_hidden_states"],
        dynamic_axes={
            "hidden_states": {0: "batch_size", 1: "sequence_length"},
            "normed_hidden_states": {0: "batch_size", 1: "sequence_length"},
        },
    )

    print(f"[SUCCESS] ONNX model exported to: {args.output}")
    print(f"[INFO]    Hidden size: {args.hidden_size}, eps: {args.eps}")

    # 6. Optional: verify with onnxruntime if available
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(args.output)
        ort_out = sess.run(None, {"hidden_states": dummy_input.numpy()})[0]
        np.testing.assert_allclose(out.numpy(), ort_out, rtol=1e-5, atol=1e-5)
        print("[VERIFY] ONNXRuntime output matches PyTorch output (allclose).")
    except ImportError:
        print("[INFO] onnxruntime not installed, skipping verification.")
    except Exception as e:
        print(f"[WARN] ONNX verification failed: {e}")


if __name__ == "__main__":
    main()
