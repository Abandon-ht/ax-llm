# Qwen3-TTS C++ vs Python 推理一致性对比报告（ONNX RMSNorm 修正后）

> **分析日期**: 2026-05-21  
> **修正内容**: `scripts/infer.py` Talker RMSNorm 改为 ONNX Runtime 执行  
> **数据来源**: `debug_bin/cpp_dump/` (C++) vs `debug_bin/py_dump/` (Python)  
> **参考文档**: `docs/rmsnorm_onnx_analysis.md`, `docs/qwen3_tts_cpp_python_debug_summary.md`

---

## 1. 执行脚本总览

| 脚本 | 状态 | 关键结论 |
|------|------|----------|
| `compare_prefill_checkpoints.py` | ✅ 已执行 | prefill_logits bit-exact; prefill_last_raw/normed Python 文件缺失 |
| `compare_cp_dumps.py` | ✅ 已执行 | **CP 15 步全部 MATCH** |
| `compare_coupling_flow.py` | ⚠️ 脚本路径问题 | 未在子目录找到 codec_sum，结论不可信 |
| `compare_talker_decode_full.py` | ✅ 已执行 | **Talker decode inputs_embeds step 0 MATCH; raw_hidden step 0 DIVERGE** |

---

## 2. 核心结论（本次修正后的新发现）

### 2.1 已确认修复：CP 完全对齐

Python 侧 Talker RMSNorm 改用 ONNX Runtime（与 C++ 同一 `talker_rmsnorm.onnx`）后：

| 对比项 | 结果 |
|--------|------|
| CP hidden state (pre/post norm, 15 steps) | **cos=1.000000, max_diff=0.000000** |
| CP lm_head logits (15 steps) | **cos=1.000000, max_diff=0.000000** |
| CP sampled tokens (15 steps) | **全部 MATCH** |

**结论**：RMSNorm 差异是此前 CP step 2 发散的唯一根因。ONNX 统一后，CP 独立一致性完全解决。

### 2.2 仍未解决：Talker Decode 发散

Talker decode 呈现**新的发散模式**：

| Step | codec_sum | inputs_embeds | raw_hidden | logits | next_token |
|------|-----------|---------------|------------|--------|------------|
| 0 | ✅ cos=1.000 | ✅ cos=1.000 | ❌ cos=0.032, max_diff=102.5 | ❌ cos=0.19 | ❌ 1737 vs 1174 |
| 1+ | — | — | 持续发散 | 持续发散 | 全部不一致 |

**关键发现**：
- `codec_sum` 和 `inputs_embeds` **完全匹配** → Talker decode 的输入嵌入绝对正确
- `raw_hidden` **从 step 0 就彻底错误** → 问题在 Talker decode axmodel 的**内部状态**

---

## 3. L0/L1 — Talker Prefill 阶段

### 3.1 Prefill Logits

| 指标 | 数值 |
|------|------|
| cosine | 0.99999994 |
| max_diff | 0.00000000 |
| argmax | cpp=1130, py=1130 ✅ |

### 3.2 缺失的中间文件

Python 侧未生成以下文件（可能是 dump 逻辑未触发或路径错误）：
- `py_dump/prefill_last_raw_hidden.bin`
- `py_dump/prefill_last_normed_hidden.bin`
- `py_dump/python_kvcache/` 目录

---

## 4. L2 — CP (Code Predictor) 独立一致性

### 4.1 Hidden State 传播链（frame 0）

| Step | pre_norm cos | pre_norm max_diff | post_norm cos | post_norm max_diff |
|------|-------------|-------------------|---------------|--------------------|
| 0~14 | **1.000000** | **0.000000** | **1.000000** | **0.000000** |

### 4.2 LM Head Logits & Sampled Tokens（Greedy）

| Step | C++ Token | Python Token | lm_head cos | lm_head max_diff | 判定 |
|------|-----------|--------------|-------------|------------------|------|
| 0~14 | 完全一致 | 完全一致 | 1.000000 | 0.000000 | ✅ **ALL MATCH** |

---

## 5. L3 — Talker+CP 耦合流与 Talker Decode

### 5.1 耦合流 Step 0（关键数据）

| 字段 | cos | max_diff | 判定 |
|------|-----|----------|------|
| codec_sum | 1.000000 | 0.000000 | ✅ MATCH |
| inputs_embeds | 1.000000 | 0.000000 | ✅ MATCH |
| raw_hidden | 0.032391 | 102.457031 | ❌ DIVERGE |
| logits | 0.188359 | 38.312500 | ❌ DIVERGE |
| next_token | — | — | ❌ 1737 vs 1174 |

### 5.2 发散演进（Step 0~19）

- `codec_sum` / `inputs_embeds` 仅 step 0 匹配，step 1+ 因 token 不同导致输入不同而自然发散
- `raw_hidden` 从 step 0 起全程与 C++ 几乎无相关性（cos ≈ -0.09 ~ 0.11）

---

## 6. 排除法结论

| 假设 | 验证结果 | 状态 |
|------|----------|------|
| C++ axmodel runtime 有 bug | CP 15 步 bit-exact，prefill logits bit-exact | ❌ 排除 |
| ONNX RMSNorm 模型有 bug | Python 单独运行 ONNX RMSNorm 可生成正常音频 | ❌ 排除 |
| 精度（bf16/fp32）问题 | CP 完全匹配，说明 bf16 未导致发散 | ❌ 排除 |
| Talker decode 输入嵌入错误 | codec_sum / inputs_embeds step 0 完全匹配 | ❌ 排除 |
| **Talker decode 内部状态异常** | KV cache / indices / mask / state 传递未验证 | ⚠️ **唯一剩余假设** |

---

## 7. 根因假设：Talker Decode 内部状态不一致

在 `inputs_embeds` 已确认一致的前提下，Talker decode axmodel 输出 `raw_hidden` 发散的唯一解释是以下**四个内部输入**存在差异：

```
inputs_embeds (MATCHED)
    ├── K_cache  ← 假设 1: prefill 后的初始 KV cache 内容与 C++ 不同
    ├── V_cache  ← 假设 1
    ├── indices  ← 假设 2: decode indices 不等于 C++ 的 decode_start
    ├── mask     ← 假设 3: decode mask 内容与 C++ 不等价
    └── axmodel  ← 已排除
         ↓
    raw_hidden (DIVERGE)
```

此外：
- **假设 4**: Python `StaticTalkerState` 的 `current_len` 或 `prompt_len` 与 C++ 的 `input_embed_num` / `decode_start` 不一致，导致 KV cache 更新位置或 mask 索引错位。

---

## 8. 验证方案实现状态 ✅ 已完成

### 8.1 目标

Dump Talker decode **step 0** 的完整输入状态，与 C++ 进行逐字节对比。

### 8.2 Python 侧 dump 逻辑

✅ 已修改 `scripts/infer.py`，在 `StaticTalkerLayerRunner.decode_one()` 执行前增加 dump：
- `python_talker_decode_step{NNN}_k_cache_l00.bin`
- `python_talker_decode_step{NNN}_v_cache_l00.bin`
- `python_talker_decode_step{NNN}_indices.bin`
- `python_talker_decode_step{NNN}_mask.bin`

### 8.3 C++ 侧 dump 逻辑

✅ 已修改 `src/runner/LLM_cp_tts_insert.inc`，在 decode 循环开头增加 dump：
- `cpp_talker_decode_step{NNN}_indices.bin`
- `cpp_talker_decode_step{NNN}_mask.bin`

C++ 已有的相关文件：

| 文件 | 说明 |
|------|------|
| `cpp_talker_decode_step000_inputs_embeds.bin` | 已确认与 Python 匹配 |
| `cpp_talker_decode_step000_raw_hidden.bin` | 已确认发散 |
| `debug_talker_kvcache_ax/layer_00_k.bin` | prefill 后 layer 0 K cache |
| `debug_talker_kvcache_ax/layer_00_v.bin` | prefill 后 layer 0 V cache |

### 8.4 对比脚本

✅ 已新增 `scripts/compare_talker_decode_inputs.py`，专门对比 decode inputs：

```bash
python3 scripts/compare_talker_decode_inputs.py \
    --cpp-dir ./debug_bin/cpp_dump \
    --py-dir ./debug_bin/py_dump \
    [--step 0] \
    [-v]
```

对比项：
- `k_cache_l00`（C++: `debug_talker_kvcache_ax/layer_00_k.bin` vs Python: `python_talker_decode_step000_k_cache_l00.bin`）
- `v_cache_l00`（C++: `debug_talker_kvcache_ax/layer_00_v.bin` vs Python: `python_talker_decode_step000_v_cache_l00.bin`）
- `indices`
- `mask`

### 8.5 判定树

```
对比 decode step 0 输入
    ├── K_cache / V_cache 不一致
    │       → 检查 prefill 后 state 的保存逻辑
    │       → 检查 C++ prefill KV cache 更新位置 vs Python
    ├── indices 不一致
    │       → 检查 _last_static_position_id() 返回值 vs C++ decode_start
    │       → 检查 position_ids / cache_position 传递链条
    ├── mask 不一致
    │       → 检查 _build_decode_mask_cache() 的 fp32 数值 vs C++ mask vector
    │       → 特别关注最后一个元素是否为 0
    └── 四个输入全部一致
            → 说明问题在 axmodel 内部（层间 buffer、device 状态等）
            → 需 dump 逐层中间输出（layer0 output, layer1 input...）
```

---

## 9. 历史记录

### 2026-05-21 之前的状态

- RMSNorm 差异（PyTorch vs C++ 手写）导致 CP step 2 发散
- 本次更新：Python 侧改用 ONNX Runtime 执行 RMSNorm，CP 完全修复

### 本次更新后的状态

- **CP**: ✅ 15/15 完全匹配
- **Talker Prefill Logits**: ✅ 匹配
- **Talker Decode Inputs (codec_sum / inputs_embeds)**: ✅ step 0 匹配
- **Talker Decode Outputs (raw_hidden)**: ❌ step 0 即发散
- **根因**: 锁定在 Talker decode 内部状态（KV cache / indices / mask / state 传递）

### 验证方案实现

- **Python dump 逻辑**: `scripts/infer.py` `decode_one()` 前已增加 KV cache / indices / mask dump
- **C++ dump 逻辑**: `src/runner/LLM_cp_tts_insert.inc` decode 循环已增加 indices / mask dump
- **对比脚本**: 已新增 `scripts/compare_talker_decode_inputs.py`
- **使用文档**: 已更新 `docs/scripts_comparison_guide.md`

---

*报告基于 `scripts/compare_*.py` 输出及 `scripts/infer.py` ONNX RMSNorm 修正后的新 dump 数据生成。*
