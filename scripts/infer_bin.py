#!/usr/bin/env python3
"""
Read output_codes.bin from AX650 board and run ONNX decoder to generate WAV.

Usage:
    conda activate qwen3-tts
    python infer_bin.py

Model:
    qwen3_tts_12hz_decoder_clean.onnx
    Input:  codes  int64[1, 16, 300]
    Output: wav    float32[1, 1, 576000]
"""

import os
import struct
import json
import numpy as np
import onnxruntime as ort
import soundfile as sf


def read_codes_bin(bin_path: str, meta_path: str = None):
    """Read AX650 output_codes.bin (int32 [num_frames, 16])."""
    if meta_path and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        num_frames = meta["num_frames"]
        num_codebooks = meta["num_codebooks"]
        dtype = meta.get("dtype", "int32")
        print(f"[META] {meta}")
    else:
        # Infer from file size: each frame = 16 * 4 bytes
        file_size = os.path.getsize(bin_path)
        num_codebooks = 16
        num_frames = file_size // (num_codebooks * 4)
        dtype = "int32"
        print(f"[INFER] file_size={file_size}, num_frames={num_frames}, num_codebooks={num_codebooks}")

    codes = np.fromfile(bin_path, dtype=np.int32)
    codes = codes.reshape(num_frames, num_codebooks)
    print(f"[LOAD] codes shape: {codes.shape}, dtype: {codes.dtype}")
    print(f"[STATS] primary codes range: [{codes[:,0].min()}, {codes[:,0].max()}]")
    return codes


def prepare_onnx_input(codes: np.ndarray, target_len: int = 300):
    """
    Convert [T, 16] -> [1, 16, T], then pad/crop to target_len.
    ONNX expects int64[1, 16, 300].
    """
    # [T, 16] -> [16, T]
    codes = codes.transpose(1, 0)
    # [16, T] -> [1, 16, T]
    codes = np.expand_dims(codes, axis=0)

    current_len = codes.shape[2]
    if current_len > target_len:
        print(f"[CROP] {current_len} -> {target_len}")
        codes = codes[:, :, :target_len]
    elif current_len < target_len:
        print(f"[PAD] {current_len} -> {target_len}")
        pad_width = ((0, 0), (0, 0), (0, target_len - current_len))
        codes = np.pad(codes, pad_width, mode='constant', constant_values=0)

    codes = codes.astype(np.int64)
    print(f"[ONNX INPUT] shape: {codes.shape}, dtype: {codes.dtype}")
    return codes


def run_onnx(onnx_path: str, codes: np.ndarray, output_wav_path: str,
             output_npy_path: str = None, sample_rate: int = 24000):
    """Run ONNX decoder and save WAV."""
    print(f"[ONNX] Loading model: {onnx_path}")
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    input_name = session.get_inputs()[0].name
    print(f"[ONNX] Input '{input_name}' shape: {session.get_inputs()[0].shape}")

    print("[ONNX] Running inference...")
    outputs = session.run(["wav"], {input_name: codes})

    wav_tensor = outputs[0]
    print(f"[ONNX] Output wav shape: {wav_tensor.shape}, dtype: {wav_tensor.dtype}")

    if output_npy_path:
        np.save(output_npy_path, wav_tensor)
        print(f"[SAVE] NPY: {output_npy_path}")

    wav_1d = wav_tensor.flatten()
    print(f"[SAVE] WAV: {output_wav_path} ({sample_rate} Hz, {len(wav_1d)/sample_rate:.2f}s)")
    sf.write(output_wav_path, wav_1d, sample_rate, format='WAV', subtype='PCM_16')
    print("[DONE] Inference finished!")


def main():
    # Paths
    work_dir = os.path.dirname(os.path.abspath(__file__))
    bin_path = os.path.join(work_dir, "output_codes.bin")
    meta_path = os.path.join(work_dir, "output_meta.json")
    onnx_path = os.path.join(work_dir, "qwen3_tts_12hz_0.6B-Base-decoder_static.onnx")

    # 1. Read bin
    codes = read_codes_bin(bin_path, meta_path)

    # 2. Split into batches of max 300 frames
    total_frames = codes.shape[0]
    target_len = 300
    num_batches = (total_frames + target_len - 1) // target_len

    for i in range(num_batches):
        start = i * target_len
        end = min(start + target_len, total_frames)
        chunk = codes[start:end]
        print(f"[BATCH {i:02d}] frames [{start}:{end}] / {total_frames}")

        onnx_input = prepare_onnx_input(chunk, target_len=target_len)

        output_wav = os.path.join(work_dir, f"output_ax650_{i:02d}.wav")
        output_npy = os.path.join(work_dir, f"output_ax650_{i:02d}.npy")

        # 3. Run ONNX decoder
        run_onnx(onnx_path, onnx_input, output_wav, output_npy)


if __name__ == "__main__":
    main()
