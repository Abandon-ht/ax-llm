#!/bin/bash
# ============================================================
# Qwen3-TTS 消融实验板端运行脚本
# ============================================================
# 用法: 复制到板子上执行
#   chmod +x run_ablation_on_board.sh
#   ./run_ablation_on_board.sh
#
# 注意:
#   - Mode 1/3 使用 ONNX Talker (~1.7GB), 在板子 CPU 上推理极慢。
#     建议先用少量帧(如 10)快速验证流程, 确认无误后再跑完整 128 帧。
#   - Mode 0/2 使用 AXModel Talker (NPU), 速度正常。
# ============================================================

set -e

ABlation_BIN="/root/qwen3_tts_ablation"
MODEL_DIR="/root/rsp/Qwen3-TTS-12Hz-0.6B-Base-AX650/talker/"
ONNX_DIR="/root/rsp/Qwen3-TTS-ONNX-DLL/onnx_kv_06b/"
NPY_DIR="/root/tts_embeds/"
MAX_FRAMES="${1:-128}"   # 默认 128 帧, 可传第一个参数改为 10 等

echo "========================================"
echo "Qwen3-TTS Ablation Experiment"
echo "MAX_FRAMES = ${MAX_FRAMES}"
echo "========================================"

# Mode 0: AX Talker + AX CP (基准, NPU 推理, 快)
echo ""
echo ">>> [Mode 0] AX Talker + AX CP (baseline)"
time LD_LIBRARY_PATH=/usr/lib "${ABlation_BIN}" "${MODEL_DIR}" "${ONNX_DIR}" "${NPY_DIR}" --mode=0 "${MAX_FRAMES}"

# Mode 1: ONNX Talker + AX CP (验证 Talker, CPU 推理, 很慢)
echo ""
echo ">>> [Mode 1] ONNX Talker + AX CP (ablate talker) -- WARNING: very slow on CPU"
time LD_LIBRARY_PATH=/usr/lib "${ABlation_BIN}" "${MODEL_DIR}" "${ONNX_DIR}" "${NPY_DIR}" --mode=1 "${MAX_FRAMES}"

# Mode 2: AX Talker + ONNX CP (验证 CP, NPU+CPU 混合, CP 部分较慢)
echo ""
echo ">>> [Mode 2] AX Talker + ONNX CP (ablate CP)"
time LD_LIBRARY_PATH=/usr/lib "${ABlation_BIN}" "${MODEL_DIR}" "${ONNX_DIR}" "${NPY_DIR}" --mode=2 "${MAX_FRAMES}"

# Mode 3: ONNX Talker + ONNX CP (Golden Reference, CPU 推理, 很慢)
echo ""
echo ">>> [Mode 3] ONNX Talker + ONNX CP (golden reference) -- WARNING: very slow on CPU"
time LD_LIBRARY_PATH=/usr/lib "${ABlation_BIN}" "${MODEL_DIR}" "${ONNX_DIR}" "${NPY_DIR}" --mode=3 "${MAX_FRAMES}"

echo ""
echo "========================================"
echo "All modes finished. Results:"
ls -la "${NPY_DIR}"/output_codes_*.bin
ls -la "${NPY_DIR}"/output_meta_*.json
echo "========================================"
