# Qwen3-TTS Talker + CP 深度对齐验证报告

> **日期**: 2026-05-20  
> **依据文档**: `docs/qwen3_tts_validation_strategy.md`, `docs/qwen3_tts_talker_cp_coupling_spec.md`  
> **验证范围**: L0 输入对齐 ~ L3 耦合数据流（L4 端到端受采样差异影响，未纳入数值对比）

---

## 一、本次验证执行摘要

按照 `docs/qwen3_tts_validation_strategy.md` 设计的 L0~L4 分层验证体系，本次执行完成了以下工作：

1. **C++ 侧 dump 能力补齐**：在 `LLM_cp_tts_insert.inc` 中新增 CP 内部 6 类 dump、Talker decode 5 类 dump，支持按 `frame_idx/step` 维度命名。
2. **Python 侧 dump 能力补齐**：在 `modeling_qwen3_tts.py` 的 Talker/CP `forward()` 中插入 dump 逻辑，支持 `QWEN3_TTS_DUMP_DIR` 环境变量控制。
3. **对比脚本矩阵就绪**：`compare_cp_dumps.py`（增强版）、`compare_talker_decode_full.py`、`compare_coupling_flow.py`、`validate_tts_pipeline.py` 全部可用。
4. **首轮对比执行**：分别生成了 Python（原始 PyTorch）dump 和 C++（AX650）dump，并执行了 L1~L3 对比。

---

## 二、关键发现（按严重优先级排序）

### 🔴 发现 1：C++ CP 采样策略与 Python 不一致（已定位并验证）

**现象**：
- C++ 侧 `CpSampleFromLogits` 读取 `post_config.json` 中的 `temperature=0.9` 执行采样。
- Python 侧 `subtalker_dosample=False`，即 greedy argmax。
- 导致 CP 生成的 sub-codes 从 step 0 起就完全不同，codec_sum cosine 低至 **0.368**。

**验证**：
- 临时强制 C++ CP greedy（`temperature=0.0f`）后，frame 0 的 step 0~1 的 lm_head argmax 完全匹配，codec_sum cosine 提升至 **0.869**。

**结论**：
- **采样策略不一致是 CP 发散的表面原因**，必须在 C++ 侧增加 `subtalker_dosample` 的等效控制（例如当 `cp_temperature < 1e-6` 或配置项显式关闭采样时走 greedy）。

---

### 🔴 发现 2：Talker Prefill 阶段即存在 ~6% 偏差（cos=0.938）

**现象**：
- C++ `debug_talker_prefill_last_hidden_ax.bin` vs Python `python_prefill_last_hidden.bin`：
  - **cosine = 0.937883**
  - **max_diff = 75.125**
- C++ `debug_talker_prefill_logits_ax.bin` vs Python `python_prefill_logits.bin`：
  - **cosine = 0.997828**
  - **argmax DIVERGE**（1130 vs 609）

**影响**：
- Prefill 的 `past_hidden` 是 CP 的输入条件之一。该偏差会传导到 CP 的每一步，导致 CP hidden 从 step 0 起就有 ~3% 偏差（cos=0.969）。
- 同时导致 Talker decode 的 primary token 从 step 0 就不一致（C++=1737, Python=1174，greedy 下）。

**根因假设（待进一步验证）**：
1. **输入 embeds 构造差异**：C++ 读取的 `prefill_embeds.bin` 是由 `infer.py`（AXEngine Python 替换）dump 的 BF16 文件；而本次 Python dump 来自原始 PyTorch 模型 `test_model_12hz_base_single_batch.py`。两者的输入构造路径可能不同。
2. **模型权重加载差异**：C++ 使用 axmodel（BF16 静态图），Python 原始模型使用 safetensors（BF16）。虽然格式相同，但量化/转换过程可能引入差异。
3. **Attention mask / indices 差异**：C++ prefill 使用 `build_prefill_mask` 构造的 square mask，Python 使用 `create_causal_mask`。虽然逻辑应等价，但实现细节（如 padding mask 处理、sliding window 等）可能有差异。
4. **KV cache 初始化差异**：C++ 的 KV cache 在 prefill 后从 decode group 读取，而 Python 使用 DynamicCache。

**关键待验证项**：
- 需使用 **`infer.py` 的 AXEngine Python 替换实现**作为 Golden（而非原始 PyTorch），因为 C++ 的输入 `prefill_embeds.bin` 正是由 `infer.py` 生成。只有使用相同的输入源，L0 才能保证一致。

---

### 🟡 发现 3：Talker Decode 阶段 raw_hidden 完全发散（cos≈0.03）

**现象**：
- 即使强制 CP greedy 后，Talker decode step 0 的 `raw_hidden` cosine 仅 **0.031**，`logits` cosine 仅 **0.159**。
- 这说明 Talker decode 的 Transformer 前向传播本身存在严重问题，与 CP 无关。

**根因假设**：
1. **Prefill 偏差累积**：由于 prefill 的 `past_hidden` 已不一致，Talker decode 的初始状态就不同。
2. **KV cache 更新不一致**：C++ decode 时 KV cache 的更新逻辑（`memcpy(in_k_ptr + indices * kv_cache_size, ...)`）与 Python DynamicCache 的 `update()` 可能在 index 计算、linear layer 处理上有差异。
3. **Decode indices / mask 不一致**：C++ decode 使用固定的 `indices = decode_start + step`，而 Python 使用 `cache_position`。需对比两者的具体数值。

---

### 🟡 发现 4：Python next_token dump 格式错误（int32 被存为 float32）

**现象**：
- `_dump_fp32` 函数把 `torch.tensor(..., dtype=torch.int32)` 强制转成了 float32，导致读取时 `np.fromfile(..., dtype=np.int32)` 得到垃圾值（如 1150468096）。

**修复**：
- 已在 `modeling_qwen3_tts.py` 中改为对 int32 使用 `_dump_int32` 或直接用 `np.array(..., dtype=np.int32).tofile()`。

---

## 三、L0~L3 各层当前状态

| 层级 | 状态 | 说明 |
|------|------|------|
| **L0 输入对齐** | ⚠️ 待重新验证 | 当前 Python dump 来自原始 PyTorch，与 C++ 输入源（`infer.py`）不一致，需重新用 `infer.py` 生成 Golden |
| **L1 Talker Prefill** | ❌ 未通过 | prefill last_hidden cos=0.938，logits argmax 不一致 |
| **L2 CP 独立** | ❌ 未通过 | 采样策略不一致 + prefill 输入偏差导致级联发散 |
| **L3 耦合数据流** | ❌ 未通过 | codec_sum/inputs_embeds 均发散，根因回溯至 L1/L2 |
| **L4 端到端** | ⏸️ 未验证 | 需先解决 L1~L3 |

---

## 四、问题诊断决策树（基于本次数据更新）

```
开始验证
│
├─ L0 输入是否一致？
│  ├─ 否 → 检查 infer.py 与 C++ 的 prefill_embeds 构造路径是否相同
│  └─ 是 → 进入 L1
│
├─ L1 Talker Prefill 是否一致？
│  ├─ 否 → 
│  │    ├─ layer0 输出即发散 → 输入 embed 或 weights 加载错误
│  │    ├─ 某层 KV cache 发散 → attention mask / indices 有误
│  │    └─ 仅 last hidden/logits 发散 → RMSNorm gamma 或 post head 问题
│  └─ 是 → 进入 L2
│
├─ L2 CP 独立是否一致？（需统一为 greedy）
│  ├─ 否 → 
│  │    ├─ j=0 (prefill) 即发散 → CP prefill input [past_hidden+primary_embed] 不对
│  │    ├─ j=1 发散但 j=0 对 → CP decode embed lookup（table index）错误
│  │    └─ hidden 对但 logits 错 → lm_head[j] 选择或 weights 错误
│  └─ 是 → 进入 L3
│
└─ L3 耦合是否一致？
   ├─ 否 → 
   │    ├─ codec_sum 发散 → CP embedding accumulation 逻辑错误
   │    ├─ trailing_text 发散 → step 索引越界或 tts_pad_embed 未正确填充
   │    └─ inputs_embeds 对但 raw_hidden 错 → Talker decode KV cache / indices / mask 错误
   └─ 是 → L4 端到端验证
```

---

## 五、下一步修复建议（按优先级）

### Priority 1：统一 Golden 生成方式（L0）

**当前问题**：Python dump 来自原始 PyTorch `test_model_12hz_base_single_batch.py`，而 C++ 输入来自 `infer.py` 的 dump。两者输入构造路径可能不同。

**建议**：
1. 在 `infer.py` 的 `_AxEngineQwen3TTSTalkerModel.forward()` 和 `_AxEngineQwen3TTSTalkerCodePredictorModelForConditionalGeneration.generate()` 中添加与 C++ 对齐的 dump 逻辑。
2. 运行 `./scripts/infer.sh` 生成 `py_dump/`，确保 `prefill_embeds.bin`、`trailing_text_hiddens.bin`、`tts_pad_vec.bin` 与 C++ 输入完全一致。
3. 重新对比 L0，确认 `cosine > 0.9999`。

### Priority 2：修复 C++ CP 采样策略（L2）

**当前问题**：C++ 侧 CP 固定使用 `post_config.json` 的 `temperature=0.9`，无法与 Python `subtalker_dosample=False` 对齐。

**建议**：
1. 在 `qwen3_tts_infer.cpp` 或 `RunCpFrame` 中增加 `--cp-greedy` 命令行参数。
2. 或者在 `post_config.json` 中增加 `cp_temperature` / `cp_do_sample` 字段，由 `load_llm_config` 读取并传给 `_attr.cp_temperature`。
3. 临时验证方案：保持 `temperature=0.0f` 的强制 greedy 修改，用于对齐验证。

### Priority 3：排查 Talker Prefill 偏差（L1）

**当前问题**：prefill last_hidden cosine=0.938，偏差较大。

**建议**：
1. 使用 `compare_prefill_checkpoints.py` 逐层对比 KV cache、layer0 输出。
2. 检查 C++ 的 `build_prefill_mask` 与 Python `create_causal_mask` 的等价性。
3. 检查 C++ prefill 的 `indices` 构造（`_position_ids_to_static_indices`）与 Python `position_ids` 是否一致。

### Priority 4：排查 Talker Decode KV Cache（L3）

**当前问题**：decode raw_hidden 完全发散。

**建议**：
1. 在 C++ 侧 dump decode 每一步的 `indices`、`mask`、`K_cache`、`V_cache`。
2. 与 Python `DynamicCache` 的逐层张量对比。
3. 特别关注 `is_linear_layer` 的 KV cache 更新逻辑（`llm_d2d` 全量复制 vs 增量复制）。

---

## 六、已修改的文件清单

| 文件 | 修改内容 | 状态 |
|------|----------|------|
| `src/runner/LLM.hpp` | `RunCpFrame` 增加 `frame_idx` 参数 | ✅ 已编译 |
| `src/runner/LLM.cpp` | 转发 `frame_idx` | ✅ 已编译 |
| `src/runner/LLM_cp_tts_insert.inc` | 新增 CP/Talker decode dump 逻辑 | ✅ 已编译 |
| `tools/qwen3_tts_infer.cpp` | 新增 `--debug-dump-dir` 参数 | ✅ 已编译 |
| `scripts/compare_cp_dumps.py` | 支持新命名规范、sampled_token、codec_sum | ✅ 已测试 |
| `scripts/compare_talker_decode_full.py` | 新建：逐 step 对比 Talker decode | ✅ 已测试 |
| `scripts/compare_coupling_flow.py` | 新建：耦合数据流专项验证 | ✅ 已测试 |
| `scripts/validate_tts_pipeline.py` | 新建：L0~L3 自动化主控 | ✅ 已测试 |
| `modeling_qwen3_tts.py` | 插入 Talker/CP dump 逻辑 | ✅ 已运行 |
| `test_model_12hz_base_single_batch.py` | 改为本地模型路径、启用 dump | ✅ 已运行 |

---

## 七、修订记录

| 日期 | 版本 | 说明 |
|------|------|------|
| 2026-05-20 | v1.0 | 首次执行 L0~L3 分层验证，发现 CP 采样不一致、Talker prefill 偏差、decode KV cache 待排查等关键问题，输出修复建议。 |
