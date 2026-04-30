#!/bin/bash
# ============================================================
# Qwen3-TTS 消融实验 Host 侧分析脚本
# ============================================================
# 用法: 在 Host (x86) 上执行
#   1. 先从板子 scp 回结果文件
#   2. 再运行本脚本
# ============================================================

set -e

RESULTS_DIR="./ablation_results"
ONNX_DECODE="/home/m5stack/rsp/Qwen3-TTS-ONNX-DLL/onnx_kv_06b/tokenizer12hz_decode.onnx"
mkdir -p "${RESULTS_DIR}"

echo "========================================"
echo "Step 1: scp results from board"
echo "========================================"
# 假设板子别名为 pyramid，根据实际情况修改
scp pyramid:/root/tts_embeds/output_codes_*.bin "${RESULTS_DIR}/"
scp pyramid:/root/tts_embeds/output_meta_*.json "${RESULTS_DIR}/"

echo ""
echo "========================================"
echo "Step 2: Run analysis"
echo "========================================"
python3 scripts/qwen3_tts_ablation_analysis.py \
  --codes-dir "${RESULTS_DIR}" \
  --tokenizer-decode "${ONNX_DECODE}" \
  --output-dir "${RESULTS_DIR}"

echo ""
echo "========================================"
echo "Step 3: View report"
echo "========================================"
cat "${RESULTS_DIR}/ablation_report.md"

echo ""
echo "Audio files:"
ls -la "${RESULTS_DIR}"/*.wav
