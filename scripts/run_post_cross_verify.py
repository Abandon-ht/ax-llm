#!/usr/bin/env python3
"""
Cross-verify: Run C++ embed through Python post_session on the device.
"""
import numpy as np
from pathlib import Path
from axengine import InferenceSession

# Load C++ embed (input to llama_post)
cpp_embed = np.fromfile("./debug_bin/cpp_dump/debug_talker_prefill_last_hidden_ax.bin", dtype=np.float32)
print(f"[C++ embed] shape={cpp_embed.shape}, first5={cpp_embed[:5]}")

# Load C++ original logits
cpp_logits = np.fromfile("./debug_bin/cpp_dump/debug_talker_prefill_logits_ax.bin", dtype=np.float32)
print(f"[C++ logits] argmax={np.argmax(cpp_logits)} max={np.max(cpp_logits):.4f} top5={sorted([(i,float(cpp_logits[i])) for i in range(len(cpp_logits))], key=lambda x:-x[1])[:5]}")

# Load Python original logits (need to scp from PC if not on device)
py_logits_path = Path("./debug_bin/python_prefill_dump/python_prefill_logits.bin")
if py_logits_path.exists():
    py_logits = np.fromfile(py_logits_path, dtype=np.float32)
    print(f"[Python logits] argmax={np.argmax(py_logits)} max={np.max(py_logits):.4f} top5={sorted([(i,float(py_logits[i])) for i in range(len(py_logits))], key=lambda x:-x[1])[:5]}")
else:
    py_logits = None
    print("[Python logits] NOT FOUND on device, skipping comparison with Python logits")

# Run C++ embed through Python post_session on device
post_path = "rsp/Qwen3-TTS-12Hz-0.6B-Base-AX650/talker/qwen3_tts_talker_post.axmodel"
print(f"\n[Loading post.axmodel] {post_path}")
post_session = InferenceSession(post_path)

# Prepare input: [1, 1, 1024] bfloat16 (axengine expects bf16)
import ml_dtypes
embed_input = cpp_embed.reshape(1, 1, 1024).astype(ml_dtypes.bfloat16)
print(f"[Input to post_session] shape={embed_input.shape}, dtype={embed_input.dtype}")

raw_outputs = post_session.run(None, {"input": embed_input})
# axengine returns list, map to dict using get_outputs()
output_names = [o.name for o in post_session.get_outputs()]
outputs = dict(zip(output_names, raw_outputs))
out_key = "output" if "output" in outputs else "logits"
logits_from_cpp_embed = outputs[out_key].astype(np.float32).reshape(-1)

print(f"\n[Output from C++ embed via Python runtime]")
print(f"argmax={np.argmax(logits_from_cpp_embed)} max={np.max(logits_from_cpp_embed):.4f}")
print(f"top5={sorted([(i,float(logits_from_cpp_embed[i])) for i in range(len(logits_from_cpp_embed))], key=lambda x:-x[1])[:5]}")

# Triangular comparison
print(f"\n[Triangular comparison]")
if py_logits is not None:
    cos_cpp_embed_vs_py = np.dot(logits_from_cpp_embed, py_logits) / (np.linalg.norm(logits_from_cpp_embed) * np.linalg.norm(py_logits) + 1e-12)
    diff_cpp_embed_vs_py = np.max(np.abs(logits_from_cpp_embed - py_logits))
    print(f"C++ embed via Python  vs  Python original: cosine={cos_cpp_embed_vs_py:.8f} max_diff={diff_cpp_embed_vs_py:.8f}")

cos_cpp_embed_vs_cpp = np.dot(logits_from_cpp_embed, cpp_logits) / (np.linalg.norm(logits_from_cpp_embed) * np.linalg.norm(cpp_logits) + 1e-12)
diff_cpp_embed_vs_cpp = np.max(np.abs(logits_from_cpp_embed - cpp_logits))
print(f"C++ embed via Python  vs  C++ original:    cosine={cos_cpp_embed_vs_cpp:.8f} max_diff={diff_cpp_embed_vs_cpp:.8f}")

# Also dump the raw output bytes for byte-level comparison
raw_out_path = Path("./debug_bin/cpp_dump/python_runtime_from_cpp_embed_raw.bin")
raw_out_path.parent.mkdir(parents=True, exist_ok=True)
raw_out_path.write_bytes(outputs[out_key].tobytes())
print(f"\n[Saved raw output bytes to {raw_out_path}]")
