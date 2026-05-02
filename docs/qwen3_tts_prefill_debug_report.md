# Qwen3-TTS Talker Prefill 阶段 Debug 报告

> **文档目的**：供后续独立会话直接参考，无需依赖前文上下文。  
> **测试时间**：2026-05-01  
> **测试方法**：对比 AX Talker（AX650 NPU）与 ONNX Talker（CPU FP32）在相同 prefill 输入下的 `last_hidden` 和 `logits` 输出。

---

## 1. 测试方法

### 1.1 代码修改

修改了以下文件以导出 prefill 阶段的中间张量：

| 文件 | 修改内容 |
|------|----------|
| `src/runner/LLM.hpp` | 新增 `SetDebugDumpDir(const std::string&)` 接口 |
| `src/runner/LLM.cpp` | 实现 debug dump 目录传递 |
| `src/runner/LLM_cp_tts_insert.inc` | 在 prefill 完成后导出 `last_hidden` [1024] 和 `logits` [3072]（均为 FP32） |
| `tools/qwen3_tts_ablation.cpp` | 调用 `llm.SetDebugDumpDir()`；ONNX Talker prefill 后导出相同数据 |

### 1.2 运行方式

在板子上分别运行 Mode 0（AX Talker）和 Mode 1（ONNX Talker），各只跑 1 帧（`max_new_tokens=1`），因为 prefill 在 AR loop 之前即完成：

```bash
# 板子上执行
./qwen3_tts_ablation <talker_dir> <onnx_dir> tts_embeds/ --mode=0 1
./qwen3_tts_ablation <talker_dir> <onnx_dir> tts_embeds/ --mode=1 1
```

生成文件：
```
tts_embeds/debug_talker_prefill_last_hidden_ax.bin   # [1024] float32
tts_embeds/debug_talker_prefill_logits_ax.bin        # [3072] float32
tts_embeds/debug_talker_prefill_last_hidden_onnx.bin # [1024] float32
tts_embeds/debug_talker_prefill_logits_onnx.bin      # [3072] float32
```

### 1.3 对比脚本

使用 `scripts/compare_talker_prefill.py` 自动对比：

```bash
python3 scripts/compare_talker_prefill.py ablation_results/
```

对比维度：
- Cosine Similarity
- MSE / Max Abs Diff / Mean Abs Diff
- Argmax 一致性
- Top-5 overlap

---

## 2. 测试结果

### 2.1 `last_hidden` 对比（prefill 最后一帧的 hidden state）

| 指标 | 数值 | 说明 |
|------|------|------|
| 长度 | 1024 | hidden_size |
| **Cosine Sim** | **0.96646011** | 显著低于 0.999 阈值 |
| **MSE** | **23.27842331** | 均方误差极大 |
| **Max Abs Diff** | **77.20493317** | 最大绝对差接近 80 |
| **Mean Abs Diff** | **3.13586569** | 平均每个维度差 3+ |
| AX argmax | 635 (value=20.500000) | ONNX value=97.704933，同维度但数值差 4.7x |
| ONNX argmax | 635 (value=97.704933) | |
| Top-5 overlap | 3/5 | 只有 3 个维度一致 |

**判定：❌ 严重偏差**

### 2.2 `logits` 对比（prefill 最后一帧的 LM head 输出）

| 指标 | 数值 | 说明 |
|------|------|------|
| 长度 | 3072 | talker_vocab_size |
| **Cosine Sim** | **0.99769276** | 相对较高，但仍低于 0.999 阈值 |
| **MSE** | **0.17180772** | |
| **Max Abs Diff** | **1.68999004** | 最大差约 1.7 |
| **Mean Abs Diff** | **0.30996758** | 平均差约 0.31 |
| AX argmax | **1995** (value=27.000000) | **与 ONNX 一致** |
| ONNX argmax | **1995** (value=27.648647) | |
| Top-5 overlap | 4/5 | |

**判定：⚠️ 中等偏差（但 argmax 碰巧一致）**

---

## 3. 核心分析

### 3.1 为什么 frame=0 的 primary token 相同（1995），但 hidden state 已严重偏离？

```
AX  last_hidden[635] = 20.5
ONNX last_hidden[635] = 97.7   ← 相差 4.7 倍

AX  logits[1995] = 27.0        ← argmax 位置
ONNX logits[1995] = 27.6
```

**关键发现：**
- `last_hidden` 的 **argmax 维度相同（635）**，但数值差 4.7x
- LM head（`talker_post.axmodel`）是一个线性投影，将这个偏差较大的 hidden 映射到 logits 空间时，**恰好让 1995 维度的 logit 仍然是最大值**
- 这是**巧合**，不是正确。因为其他维度的 logits 已经有显著差异（top-5 只有 4/5 重合，max diff 1.69）

### 3.2 为什么偏差在 frame=1 彻底爆发？

Prefill 阶段的 hidden 偏差具有**累加性**：
1. AX Talker prefill 输出的 `last_hidden` 已经偏离 ONNX 约 3.1（mean abs diff）
2. 这个偏离的 `last_hidden` 传给 CP → CP 生成错误的残差 token → `codec_sum` 错误
3. `next_embed = codec_sum + tts_pad_vec` 偏离正确值
4. AX Talker decode 步接收错误的 `next_embed`，加上 KV cache 状态也是基于偏离的 prefill 建立的
5. **误差在 decode 循环中指数级放大** → frame=1 的 logits 完全崩坏 → 输出 `2149`（接近 EOS 的异常值）

### 3.3 根因定位

| 假设 | 验证状态 | 可能性 |
|------|----------|--------|
| BF16 量化误差 | `last_hidden` mean diff=3.1，对 BF16 来说偏大 | ⚠️ 中等 |
| 某层 transformer 权重加载错误 | 需要逐层对比确认 | ❓ 待排查 |
| Layer Norm / RMS Norm 实现差异 | AX650 的 NPU 实现可能与 ONNX CPU 有差异 | ❓ 待排查 |
| Attention Mask 构造差异 | prefill 的 mask 构造逻辑需要对比 | ❓ 待排查 |
| KV Cache 初始化/布局差异 | 虽然 prefill 不依赖外部 KV，但内部 KV 写入可能有问题 | ❓ 待排查 |

---

## 4. 下一步排查方案（供新会话使用）

### 方案 A：逐层 hidden state 对比（最直接）

在 `LLM_cp_tts_insert.inc` 的 prefill 循环中，每跑完一层 transformer 就保存该层的 `embed_tmp` 输出（最后一帧的 hidden），与 ONNX 对应层的输出对比。

**ONNX 侧如何获取每层输出？**
- ONNX 的 `talker_prefill.onnx` 是一个整体模型，无法直接拿到中间层
- 可以用 ONNX Runtime 的 `IOBinding` 或**将 ONNX 拆分为 28 个独立层**来对比
- 更实际的方法：用 Python + PyTorch 加载原始 checkpoint，复现 prefill 过程，导出每层 hidden

**判定标准：**
- 找到 **cos_sim 第一次跌破 0.999 的层号**
- 该层即为偏差引入点

### 方案 B：BF16 量化精度验证

将 ONNX FP32 权重逐层转换为 BF16，再转回 FP32，与原始 FP32 计算 hidden state，对比量化损失是否足以解释 3.1 的 mean diff。

```python
# Python 验证思路
import torch

# 加载 ONNX 权重（或原始 PyTorch checkpoint）
# 模拟 BF16 量化：float32 -> bfloat16 -> float32
quantized = torch.tensor(weight, dtype=torch.bfloat16).float()
output_quantized = layer(input, quantized_weight=quantized)
output_fp32 = layer(input, weight=weight)
# 对比 output_quantized vs output_fp32
```

### 方案 C：Attention Mask 对比

打印 AX650 侧 `build_prefill_mask` 生成的 mask 张量，与 ONNX 侧的 `attention_mask` 逐元素对比。

**关键检查点：**
- mask 的 causal 结构是否正确（下三角）
- padding 位置是否一致
- 数值精度（BF16 的 -inf 表示是否与 ONNX FP32 的 -inf 等价）

### 方案 D：KV Cache 写入验证

在 prefill 循环中，每层的 `K_cache_out` / `V_cache` 写入后，立即读取回 host 内存并保存。与 ONNX 的 `past_key_values` 对比。

---

## 5. 关键数据速查

| 参数 | 值 |
|------|-----|
| hidden_size | 1024 |
| talker_vocab_size | 3072 |
| prefill_len (S) | 93 |
| first_primary_code | 1995 (AX=ONNX) |
| last_hidden cos_sim | 0.96646 ❌ |
| last_hidden max_diff | 77.20 ❌ |
| logits cos_sim | 0.99769 ⚠️ |
| logits max_diff | 1.69 ⚠️ |
| logits argmax | 1995 ✅ |

---

## 6. 相关文件

| 文件 | 作用 |
|------|------|
| `docs/qwen3_tts_prefill_debug_report.md` | 本文档 |
| `docs/qwen3_tts_ablation_analysis.md` | 消融实验整体分析与排查方案 |
| `scripts/compare_talker_prefill.py` | Prefill 输出对比脚本 |
| `tools/qwen3_tts_ablation.cpp` | 消融实验入口（已增加 debug dump） |
| `src/runner/LLM_cp_tts_insert.inc` | TTS decode 核心逻辑（已增加 debug dump） |

---

*报告生成时间：2026-05-01*  
*基于：talker_* 日志 + ablation_results/ debug bin 对比*
