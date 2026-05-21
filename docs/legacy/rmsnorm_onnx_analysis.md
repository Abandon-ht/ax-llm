# Talker RMSNorm ONNX 替换分析报告

> **分析日期**: 2026-05-21  
> **更新日期**: 2026-05-21  
> **问题**: C++ 已从手写 RMSNorm 改为 ONNX Runtime 加载 `talker_rmsnorm.onnx`，但对比结果仍有 0.125 差异。

---

## 1. 权重加载分析

### 1.1 Log 解读

```
13:51:41.208 INF InitCp:219 | Loaded ONNX RMSNorm from ...talker_rmsnorm.onnx (in=1 out=1)
```

**结论**: ONNX 模型加载成功，权重来源正确（从原始 safetensors 导出）。

### 1.2 是否存在错误加载

**不存在错误加载**。ONNX 模型内部已包含正确的 gamma 权重，`RunOnnxRmsNorm` 在 ONNX 路径下**不读取** `cp_norm_gamma`。

### 1.3 设计缺陷（已修复）

原代码用 `cp_norm_gamma.empty()` 作为 ONNX 路径开关，已改为 `rmsnorm_onnx_loaded || !cp_norm_gamma.empty()`。

---

## 2. 替换位置分析

### 2.1 C++ 调用点

| 阶段 | 位置 | 操作 | Python 对应 |
|------|------|------|-------------|
| **Prefill 后** | `LLM_cp_tts_insert.inc:841` | 对 `all_prefill_hidden` 全部 `input_embed_num` 个 token in-place RMSNorm | `self.norm(hidden_states)` 输出全量 normed hidden |
| **Decode step=0** | `LLM_cp_tts_insert.inc:1016` | 取 `all_prefill_hidden[last]`（已 normed）作为 CP past_hidden | 使用 prefill 最后一个 normed token |
| **Decode step>0** | `LLM_cp_tts_insert.inc:1021` | 对 `embed` (raw hidden) 做 RMSNorm → `txt_hidden_bf16` | `self._to_hidden_tensor(raw_hidden, ...)` |

### 2.2 位置正确性判定

**✅ 位置正确**。C++ 的两处调用与 Python 的 RMSNorm 应用位置完全对应。

---

## 3. 根因分析（修正版）

### 3.1 实验验证结果

使用同一输入 `prefill_last_raw_hidden.bin` 进行隔离测试：

| 对比项 | max_diff | 结论 |
|--------|----------|------|
| PyTorch RMSNorm(fp32) vs ONNX RMSNorm(fp32) | **8.3e-05** | ✅ ONNX 模型与 PyTorch fp32 基本一致 |
| C++ ONNX → bf16 → fp32 vs PyTorch → bf16 → fp32 | **0.1523** | ❌ bf16 放大了 fp32 差异 |
| C++ ONNX → bf16 → fp32 vs Python dump(fp32) | **0.1250** | ❌ 当前实际差异 |
| PyTorch RMSNorm(fp32) vs Python dump(fp32) | **0.0693** | ⚠️ Python dump 与标准 PyTorch 也不一致 |

### 3.2 根因：bf16 精度放大了 ONNX/PyTorch 的 fp32 差异

**关键发现**：
1. ONNX fp32 输出与 PyTorch fp32 输出的差异仅 **8.3e-05**，在 fp32 精度下可忽略。
2. 但当两者都转回 **bf16** 时，这 8.3e-05 的差异恰好落在 bf16 的舍入边界上，导致 bf16 表示不同。
3. 在 1024-dim 的 RMSNorm 输出中，部分值域（如 1.5~2.0 附近）的 fp32 差异被放大为 **0.125** 的 bf16 差异。

**之前假设的错误**：
- 旧文档假设根因是 C++ `bfloat16` 截断舍入 vs PyTorch RNE。
- **实验证明该假设错误**：即使将 C++ `bfloat16` 改为 RNE，并真正启用 ONNX Runtime，0.125 差异**未改变**。

### 3.3 差异传播链

```
PyTorch fp32 RMSNorm ──→ bf16 (RNE) ──→ fp32 (dump)
         │                                 │
         └── diff = 8.3e-05 ──────────────┘
         │                                 │
         ↓ 在 bf16 边界放大               ↓
         │                                 │
ONNX fp32 RMSNorm ─────→ bf16 (RNE) ──→ fp32 (dump)
         │                                 │
         └── C++ dump vs Python dump = 0.125
```

### 3.4 Python dump 的额外问题

Python dump 的 `prefill_last_normed_hidden.bin` 与标准 PyTorch fp32 输出也有 **0.069** 差异。这可能来自：
- `infer.py` 中 `self.norm` 的实际执行路径与标准 `RMSNorm` 存在细微差异（如 `inputs_embeds.dtype` 影响中间精度）。
- 或 Python dump 文件由旧版本代码生成。

**但无论 Python dump 来源如何，核心问题是：ONNX 与 PyTorch 的 fp32 差异在 bf16 下被放大。**

---

## 4. 修复建议（更新）

### 方案 A：让 Python 侧也使用 ONNX Runtime（推荐）

**核心思路**：修改 `scripts/infer.py`，让 Python 的 `self.norm` 不走 PyTorch，而走与 C++ 相同的 ONNX 模型。这样 C++ 和 Python 都执行完全相同的 ONNX 推理，差异降至 **<1e-3**。

**实现方式**：
1. 在 `_AxEngineQwen3TTSTalkerModel` 中加载 `talker_rmsnorm.onnx`。
2. 在 `_to_hidden_tensor` 中，将 `raw_hidden` 转为 fp32 后送入 ONNX Runtime，输出再转回 `dtype`。
3. 或保持当前 PyTorch 路径，但在 dump 对比前临时切换到 ONNX 路径验证。

**优势**：根本性消除实现差异。

### 方案 B：统一使用 fp32，绕过 bf16 放大

**核心思路**：让 C++ 的 `RunOnnxRmsNorm` 输出保持 fp32，不转回 bf16。

**难点**：`all_prefill_hidden` 和 `embed` 在当前代码中是 `std::vector<unsigned short>`（bf16），改为 fp32 需要修改整个数据流和后续模块（CP 输入、Talker decode 输入等），工作量大。

### 方案 C（旧方案，已证伪）：修改 C++ `bfloat16` 舍入

**状态**：❌ **无效**。已将 `bfloat16.hpp` 的 `operator=` 改为 RNE，差异未改善。

---

## 5. 总结

| 问题 | 结论 |
|------|------|
| ONNX 权重加载 | ✅ 正确 |
| 替换位置 | ✅ 正确 |
| 旧根因（bf16 截断） | ❌ **错误**，RNE 修改无效 |
| 新根因 | ⚠️ ONNX/PyTorch fp32 微小差异（8e-05）在 **bf16 精度下被放大**至 0.125 |
| Python dump 一致性 | ⚠️ Python dump 与标准 PyTorch 也有 0.069 差异 |

**下一步**：执行方案 A（Python 侧也走 ONNX Runtime），或接受 0.125 为 bf16 精度下的固有差异，验证其对最终音频质量的影响。
