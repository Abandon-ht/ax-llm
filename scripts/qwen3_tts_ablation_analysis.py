#!/usr/bin/env python3
"""
Qwen3-TTS 消融实验分析脚本

用法:
    python3 qwen3_tts_ablation_analysis.py \
        --codes-dir <dir_with_output_codes_*.bin> \
        --tokenizer-decode <path_to_tokenizer12hz_decode.onnx> \
        --output-dir <output_dir>

输入文件（由 qwen3_tts_ablation 在板子上生成后 scp 回 Host）：
    output_codes_0.bin / output_meta_0.json   (AX Talker + AX CP)
    output_codes_1.bin / output_meta_1.json   (ONNX Talker + AX CP)
    output_codes_2.bin / output_meta_2.json   (AX Talker + ONNX CP)
    output_codes_3.bin / output_meta_3.json   (ONNX Talker + ONNX CP, Golden)

输出:
    mode_0.wav ~ mode_3.wav        统一解码后的音频
    ablation_report.md             量化对比报告
    codes_diff.txt                 逐帧 token 差异明细
"""

import os
import sys
import json
import argparse
import struct
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is required")
    sys.exit(1)

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime is required")
    sys.exit(1)

MODE_NAMES = {
    0: "AX_Talker + AX_CP",
    1: "ONNX_Talker + AX_CP",
    2: "AX_Talker + ONNX_CP",
    3: "ONNX_Talker + ONNX_CP (Golden)",
}


def load_codes_bin(path: str, meta: dict):
    """Load output_codes.bin as numpy array [num_frames, 16] int32."""
    num_frames = meta.get("num_frames", 0)
    num_codebooks = meta.get("num_codebooks", 16)
    dtype = meta.get("dtype", "int32")
    expected_bytes = num_frames * num_codebooks * 4  # int32
    with open(path, "rb") as f:
        data = f.read()
    if len(data) != expected_bytes:
        print(f"WARNING: {path} size mismatch: {len(data)} vs expected {expected_bytes}")
    arr = np.frombuffer(data, dtype=np.int32).reshape(num_frames, num_codebooks)
    return arr


def decode_audio(codes: np.ndarray, decoder_path: str) -> np.ndarray:
    """Decode codec codes [N, 16] to audio waveform using tokenizer12hz_decode.onnx."""
    # codes shape: [N, 16] -> add batch dim -> [1, N, 16]
    batch_codes = np.expand_dims(codes.astype(np.int64), axis=0)
    sess = ort.InferenceSession(decoder_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: batch_codes})
    audio = outputs[0]  # [batch, samples]
    if len(outputs) > 1 and outputs[1] is not None:
        lengths = outputs[1]  # [batch]
        valid_len = int(lengths[0])
        audio = audio[0, :valid_len]
    else:
        audio = audio[0]
    return audio.astype(np.float32)


def save_wav(path: str, audio: np.ndarray, sample_rate: int = 24000):
    """Save float32 audio [-1, 1] to 16-bit PCM WAV."""
    # Simple WAV header writer
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(audio_int16) * 2

    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # Subchunk1Size
        f.write(struct.pack("<H", 1))   # AudioFormat PCM
        f.write(struct.pack("<H", num_channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits_per_sample))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(audio_int16.tobytes())


def compute_frame_match_rate(codes_a: np.ndarray, codes_b: np.ndarray) -> float:
    """Compute frame-level exact match rate (%)."""
    min_frames = min(len(codes_a), len(codes_b))
    if min_frames == 0:
        return 0.0
    matches = 0
    for i in range(min_frames):
        if np.array_equal(codes_a[i], codes_b[i]):
            matches += 1
    return matches / min_frames * 100.0


def compute_codebook_accuracy(codes_a: np.ndarray, codes_b: np.ndarray) -> dict:
    """Compute per-codebook accuracy."""
    min_frames = min(len(codes_a), len(codes_b))
    result = {}
    for cb in range(16):
        if min_frames == 0:
            result[cb] = 0.0
        else:
            matches = np.sum(codes_a[:min_frames, cb] == codes_b[:min_frames, cb])
            result[cb] = matches / min_frames * 100.0
    return result


def compute_token_edit_distance(codes_a: np.ndarray, codes_b: np.ndarray) -> int:
    """Compute total number of differing tokens across all codebooks."""
    min_frames = min(len(codes_a), len(codes_b))
    diff = 0
    for i in range(min_frames):
        diff += np.sum(codes_a[i] != codes_b[i])
    # Penalize length difference
    diff += abs(len(codes_a) - len(codes_b)) * 16
    return int(diff)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Ablation Analysis")
    parser.add_argument("--codes-dir", required=True, help="Directory containing output_codes_*.bin")
    parser.add_argument("--tokenizer-decode", required=True, help="Path to tokenizer12hz_decode.onnx")
    parser.add_argument("--output-dir", required=True, help="Output directory for reports and audio")
    args = parser.parse_args()

    codes_dir = Path(args.codes_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all modes
    all_codes = {}
    all_metas = {}
    for mode in range(4):
        bin_path = codes_dir / f"output_codes_{mode}.bin"
        meta_path = codes_dir / f"output_meta_{mode}.json"
        if not bin_path.exists() or not meta_path.exists():
            print(f"WARNING: Mode {mode} files not found, skipping")
            continue
        with open(meta_path, "r") as f:
            meta = json.load(f)
        all_metas[mode] = meta
        all_codes[mode] = load_codes_bin(str(bin_path), meta)
        print(f"Mode {mode}: {MODE_NAMES[mode]} -> {len(all_codes[mode])} frames")

    if 3 not in all_codes:
        print("WARNING: Golden reference (Mode 3) not found. Comparisons will be skipped.")

    # Decode audio for each mode
    for mode, codes in all_codes.items():
        wav_path = output_dir / f"mode_{mode}.wav"
        print(f"Decoding Mode {mode} audio -> {wav_path} ...")
        audio = decode_audio(codes, args.tokenizer_decode)
        save_wav(str(wav_path), audio)
        print(f"  Saved {len(audio)} samples @ 24kHz ({len(audio)/24000:.2f}s)")

    # Generate comparison report
    report_lines = []
    report_lines.append("# Qwen3-TTS Ablation Experiment Report\n")
    report_lines.append(f"\nGolden Reference: Mode 3 (ONNX Talker + ONNX CP)\n")

    # Frame match rates vs golden
    if 3 in all_codes:
        golden = all_codes[3]
        report_lines.append("\n## Frame Match Rate vs Golden\n")
        report_lines.append("| Mode | Description | Frames | Match Rate | Total Diff Tokens |\n")
        report_lines.append("|------|-------------|--------|-----------:|------------------:|\n")
        for mode in sorted(all_codes.keys()):
            if mode == 3:
                continue
            codes = all_codes[mode]
            match_rate = compute_frame_match_rate(codes, golden)
            diff_tokens = compute_token_edit_distance(codes, golden)
            report_lines.append(
                f"| {mode} | {MODE_NAMES[mode]} | {len(codes)} | {match_rate:.2f}% | {diff_tokens} |\n"
            )

        # Per-codebook accuracy
        report_lines.append("\n## Per-Codebook Accuracy vs Golden\n")
        report_lines.append("| Mode | cb0 | cb1 | cb2 | cb3 | cb4 | cb5 | cb6 | cb7 | cb8 | cb9 | cb10 | cb11 | cb12 | cb13 | cb14 | cb15 |\n")
        report_lines.append("|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|------|------|------|------|------|\n")
        for mode in sorted(all_codes.keys()):
            if mode == 3:
                continue
            acc = compute_codebook_accuracy(all_codes[mode], golden)
            vals = " | ".join(f"{acc[cb]:.1f}" for cb in range(16))
            report_lines.append(f"| {mode} | {vals} |\n")

    # Pairwise diff matrix
    report_lines.append("\n## Pairwise Token Edit Distance\n")
    report_lines.append("|     | " + " | ".join(f"M{m}" for m in sorted(all_codes.keys())) + " |\n")
    report_lines.append("|-----|" + "|".join("-----" for _ in all_codes) + "|\n")
    for m_a in sorted(all_codes.keys()):
        row = [f"M{m_a}"]
        for m_b in sorted(all_codes.keys()):
            if m_a == m_b:
                row.append("0")
            else:
                row.append(str(compute_token_edit_distance(all_codes[m_a], all_codes[m_b])))
        report_lines.append("| " + " | ".join(row) + " |\n")

    # Diagnosis
    report_lines.append("\n## Diagnosis\n")
    if 3 in all_codes and 0 in all_codes and 1 in all_codes:
        match_0_vs_3 = compute_frame_match_rate(all_codes[0], all_codes[3])
        match_1_vs_3 = compute_frame_match_rate(all_codes[1], all_codes[3])
        if match_1_vs_3 > match_0_vs_3 + 10:
            report_lines.append("- **Talker issue detected**: Mode 1 (ONNX Talker + AX CP) is significantly closer to Golden than Mode 0 (AX/AX).\n")
            report_lines.append("  => AXModel Talker may have precision/implementation differences vs ONNX.\n")
        elif match_0_vs_3 > 95:
            report_lines.append("- Mode 0 is very close to Golden. AXModel Talker + AX CP are consistent with ONNX reference.\n")
    if 3 in all_codes and 0 in all_codes and 2 in all_codes:
        match_0_vs_3 = compute_frame_match_rate(all_codes[0], all_codes[3])
        match_2_vs_3 = compute_frame_match_rate(all_codes[2], all_codes[3])
        if match_2_vs_3 > match_0_vs_3 + 10:
            report_lines.append("- **CP issue detected**: Mode 2 (AX Talker + ONNX CP) is significantly closer to Golden than Mode 0 (AX/AX).\n")
            report_lines.append("  => AXModel CP may have precision/implementation differences vs ONNX.\n")
        elif match_0_vs_3 > 95:
            report_lines.append("- Mode 0 is very close to Golden. AXModel CP is consistent with ONNX reference.\n")

    report_path = output_dir / "ablation_report.md"
    with open(report_path, "w") as f:
        f.writelines(report_lines)
    print(f"\nReport saved to: {report_path}")

    # Save detailed diff
    diff_path = output_dir / "codes_diff.txt"
    with open(diff_path, "w") as f:
        f.write("# Per-frame code diff vs Golden (Mode 3)\n\n")
        if 3 in all_codes:
            golden = all_codes[3]
            for mode in sorted(all_codes.keys()):
                if mode == 3:
                    continue
                codes = all_codes[mode]
                f.write(f"\n## Mode {mode} ({MODE_NAMES[mode]})\n")
                min_f = min(len(codes), len(golden))
                for i in range(min_f):
                    if not np.array_equal(codes[i], golden[i]):
                        f.write(f"frame {i:4d}: golden={list(golden[i])}  mode{mode}={list(codes[i])}\n")
                if len(codes) != len(golden):
                    f.write(f"Length diff: mode{mode}={len(codes)} golden={len(golden)}\n")
    print(f"Diff detail saved to: {diff_path}")


if __name__ == "__main__":
    main()
