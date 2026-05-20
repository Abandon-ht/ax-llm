#!/usr/bin/env python3
"""
Compare C++ embed vs Python hidden_token at bf16 byte level.
"""
import numpy as np
import ml_dtypes
from pathlib import Path

# Load C++ embed as fp32, convert to bf16
cpp_embed_fp32 = np.fromfile("./debug_bin/cpp_dump/debug_talker_prefill_last_hidden_ax.bin", dtype=np.float32)
cpp_embed_bf16 = cpp_embed_fp32.astype(ml_dtypes.bfloat16)

# Load Python raw hidden as fp32, convert to bf16
py_raw_fp32 = np.fromfile("./debug_bin/python_prefill_last_raw_hidden.bin", dtype=np.float32)
if not Path("./debug_bin/python_prefill_last_raw_hidden.bin").exists():
    # fallback to new naming
    py_raw_fp32 = np.fromfile("./debug_bin/prefill_last_raw_hidden.bin", dtype=np.float32)
py_hidden_bf16 = py_raw_fp32.astype(ml_dtypes.bfloat16)

print(f"C++ embed fp32 shape={cpp_embed_fp32.shape} first5={cpp_embed_fp32[:5]}")
print(f"Python raw fp32 shape={py_raw_fp32.shape} first5={py_raw_fp32[:5]}")

# Compare fp32 values
fp32_diff = np.max(np.abs(cpp_embed_fp32 - py_raw_fp32))
print(f"\nfp32 max_diff={fp32_diff:.8f}")

# Compare bf16 bytes
cpp_bytes = cpp_embed_bf16.tobytes()
py_bytes = py_hidden_bf16.tobytes()

if cpp_bytes == py_bytes:
    print("bf16 bytes: EXACT MATCH")
else:
    print("bf16 bytes: DIFFERENT")
    # Find first differing byte
    for i, (a, b) in enumerate(zip(cpp_bytes, py_bytes)):
        if a != b:
            print(f"First byte diff at offset {i}: cpp=0x{a:02x} py=0x{b:02x}")
            break
    # Count diffs
    diff_count = sum(1 for a, b in zip(cpp_bytes, py_bytes) if a != b)
    print(f"Total differing bytes: {diff_count} / {len(cpp_bytes)}")

    # Compare bf16 values as uint16
    cpp_u16 = np.frombuffer(cpp_bytes, dtype=np.uint16)
    py_u16 = np.frombuffer(py_bytes, dtype=np.uint16)
    diff_indices = np.where(cpp_u16 != py_u16)[0]
    print(f"Total differing elements: {len(diff_indices)} / {len(cpp_u16)}")
    for idx in diff_indices[:10]:
        print(f"  idx={idx}: cpp_u16=0x{cpp_u16[idx]:04x} py_u16=0x{py_u16[idx]:04x}  cpp_fp32={cpp_embed_fp32[idx]:.8f} py_fp32={py_raw_fp32[idx]:.8f}")

# Also check: if we convert C++ bf16 back to fp32, does it match original?
cpp_back_fp32 = cpp_embed_bf16.astype(np.float32)
py_back_fp32 = py_hidden_bf16.astype(np.float32)
print(f"\nC++ fp32->bf16->fp32 roundtrip max_diff={np.max(np.abs(cpp_embed_fp32 - cpp_back_fp32)):.8f}")
print(f"Python fp32->bf16->fp32 roundtrip max_diff={np.max(np.abs(py_raw_fp32 - py_back_fp32)):.8f}")

# Check if C++ embed bf16 bytes match what Python runtime would produce from its own hidden_token
# (This tells us if the runtime sees different inputs)
print(f"\nC++ bf16 vs Python bf16 cosine={np.dot(cpp_embed_bf16.astype(np.float32), py_hidden_bf16.astype(np.float32)) / (np.linalg.norm(cpp_embed_bf16.astype(np.float32)) * np.linalg.norm(py_hidden_bf16.astype(np.float32)) + 1e-12):.8f}")
