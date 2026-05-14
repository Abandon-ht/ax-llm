# Qwen3-TTS 消融实验结果分析与排查方案

## 1. 实验结果概述

四种组合的运行结果如下：

| 模式 | Talker | CP | 结果 | 状态 |
|------|--------|-----|------|------|
| Mode 0 | AXModel | AXModel | frame=0 primary=1995 ✓, 但 frame=1 起大量 2149 | ❌ 异常 |
| Mode 1 | ONNX | AXModel | frame=0 primary=1995 ✓, Talker 正常, 但残差 token 全错 | ❌ 异常 |
| Mode 2 | AXModel | ONNX | frame=0 primary=1995 ✓, 但 frame=1 起大量 2149 | ❌ 异常 |
| Mode 3 | ONNX | ONNX | 全部正常, 可合成音频 | ✅ Golden |

**核心结论：AX Talker 和 AX CP 两个组件各自独立存在问题。**

---

## 2. 关键数据分析

### 2.1 AX Talker 问题（Mode 0 / Mode 2）

**现象：** 两个使用 AX Talker 的模式，均在 frame=1 开始输出异常的 primary token `2149`（非常接近 `codec_eos_token_id=2150`），并持续多帧。

```
Mode 3 (Golden):  frame0=1995 → frame1=215  → frame2=294  → ...
Mode 0 (AX/AX):   frame0=1995 → frame1=2149 → frame2=2149 → ...
Mode 2 (AX/ONNX): frame0=1995 → frame1=2149 → frame2=2149 → ...
```

**推断：** AX Talker 的 `prefill` 或 `decode` 输出存在系统性偏差。即使 frame=0 采样出的 primary token 相同（1995），其内部的 `last_hidden` 已经偏离了正确值。这导致：
- 传给 CP 的 `last_hidden` 错误 → CP 残差 token 错误（Mode 2 中 ONNX CP 的输出与 Golden 完全不同）
- `codec_sum` 错误 → `next_embed` 错误 → Talker decode 下一帧时输入错误 → 输出 `2149` 这种异常值

### 2.2 AX CP 问题（Mode 1）

**现象：** ONNX Talker 正常，但 AX CP 预测的 15 个残差 token 与 Golden 几乎完全不同。

```
Mode 3 frame0: 1995 1642 530 1703 149 1888 1776 653  948 962 1691 2024 1043 976 1006 380
Mode 1 frame0: 1995 112  189 1912 153 760  1645 1142 19  396 1538 1042 616  10  882  20
```

根据 `ablation_report.md`，Mode 1 与 Golden 的 **frame match rate = 0.00%**，cb1~cb15 的准确率几乎为 0。

**推断：** AX CP 在接收到**正确**的 `last_hidden` 和 `primary_code` 时，仍然输出错误的残差 token。说明 CP 内部存在独立的实现或精度问题，与 Talker 无关。

### 2.3 问题独立性验证

| 对比项 | 说明 |
|--------|------|
| Mode 0 vs Mode 2 | 两者 Talker 相同（AX），CP 不同（AX vs ONNX），但均在 frame=1 出现 2149 → **证明 Talker 问题是根因** |
| Mode 1 vs Mode 3 | 两者 Talker 相同（ONNX），CP 不同（AX vs ONNX），Talker 输出正常但残差全错 → **证明 CP 问题独立存在** |

---

## 3. 根因假设

### 3.1 AX Talker 根因假设（按可能性排序）

1. **Prefill 阶段 last_hidden 已有偏差**
   - ONNX Talker prefill 输出 `last_hidden[84]`（最后一帧）与 AX Talker 的对应值存在 cosine similarity < 0.99 的偏差
   - 可能原因：BF16 量化损失、模型权重加载错误、layer norm 计算差异

2. **Decode 阶段 KV Cache 管理错误**
   - AX Talker 在 decode 步更新 KV cache 时，与 ONNX Talker 的显式 past_key/value 传递不等价
   - 可能原因：mask 构造错误、indices 计算错误、KV cache 内存布局问题

3. **next_embed 构造链路缺失或错误**
   - `codec_sum + tts_pad_vec` 的构造在 AXModel 侧可能存在问题
   - 从日志看 `qwen3_tts_debug.md` 曾修复过 "Talker decode 后 embed 未更新" 的 bug，可能还有残余问题

4. **BF16 ↔ FP32 转换精度问题**
   - AXModel 内部使用 BF16，ONNX 使用 FP32。在关键位置（如 logits 计算）的精度损失可能导致 greedy decode 时 argmax 跳变

### 3.2 AX CP 根因假设（按可能性排序）

1. **CP Transformer（5层）隐藏层输出偏差**
   - ONNX CP 是单个模型，内部有 `generation_step` 参数控制不同残差 codebook 的路径
   - AX CP 拆分为 5 层独立 axmodel + 15 个 lm_head，缺少 `generation_step` 输入
   - **关键怀疑**：如果原始 ONNX 中 `generation_step` 不仅用于选择 head，还影响 transformer 内部的某些计算（如 positional embedding、layer-specific routing 等），那么简单的模型拆分就会导致偏差

2. **lm_head 权重加载错误或错位**
   - 15 个 `code_predictor_lm_head_*.axmodel` 可能加载了错误的权重，或者输入输出维度/格式不匹配
   - `docs/qwen3_tts_debug.md` 已确认 lm_head 输出是 FP32 [1,1,2048]，但需验证每个 head 的实际输出是否与 ONNX 对应 step 的 logits 一致

3. **CP Embedding 查表错误**
   - 15 个 `talker.code_predictor.model.codec_embedding.*.weight.bfloat16.bin` 可能加载了错误的索引
   - 或者 BF16 查表后的累加顺序/精度与 ONNX 不同

4. **CP KV Cache 或 mask 问题**
   - AX CP 的 KV cache 在 prefill → decode 切换时可能存在状态错误
   - `docs/qwen3_tts_debug.md` 曾修复过 `io_count >= 2` 的 bug，但可能还有其他 KV cache 相关问题

---

## 4. 排查方案：隐藏层相似度对比

### 4.1 Phase 1: Talker 隐藏层对比（优先级：P0）

**目标：** 定位 AX Talker 的 `last_hidden` 和 `logits` 从哪一步开始偏离 ONNX Talker。

#### 步骤 1.1：Prefill 输出对比

修改 `qwen3_tts_ablation.cpp`，在 Mode 0/2 和 Mode 1/3 的 prefill 完成后，保存以下数据到二进制文件：

```cpp
// 保存路径: tts_embeds/debug_talker_prefill_last_hidden_<mode>.bin
// shape: [S, hidden_size], dtype: float32
fwrite(last_hidden_fp32.data(), sizeof(float), S * hidden_size, fp);

// 保存路径: tts_embeds/debug_talker_prefill_logits_<mode>.bin
// shape: [S, vocab_size], dtype: float32
fwrite(logits_fp32.data(), sizeof(float), S * talker_vocab_size, fp);
```

**对比脚本（Python）：**

```python
import numpy as np

def compare_tensor(ax_path, onnx_path, shape, name):
    ax = np.fromfile(ax_path, np.float32).reshape(shape)
    gt = np.fromfile(onnx_path, np.float32).reshape(shape)
    
    cos_sim = np.sum(ax * gt) / (np.linalg.norm(ax) * np.linalg.norm(gt))
    mse = np.mean((ax - gt) ** 2)
    max_diff = np.max(np.abs(ax - gt))
    argmax_match = np.mean(np.argmax(ax, axis=-1) == np.argmax(gt, axis=-1))
    
    print(f"[{name}] cos_sim={cos_sim:.6f} mse={mse:.6f} max_diff={max_diff:.6f} argmax_match={argmax_match:.2%}")
    return ax, gt

# 对比 last_hidden (position 84 最关键)
ax_h, gt_h = compare_tensor("debug_talker_prefill_last_hidden_0.bin", 
                            "debug_talker_prefill_last_hidden_3.bin",
                            (93, 1024), "prefill_last_hidden")

# 对比 logits (position 84 最关键)
ax_l, gt_l = compare_tensor("debug_talker_prefill_logits_0.bin",
                            "debug_talker_prefill_logits_3.bin",
                            (93, 3072), "prefill_logits")
```

**判定标准：**
- `cos_sim > 0.999` 且 `argmax_match = 100%` → Prefill 无问题，问题在 Decode
- `cos_sim < 0.99` 或 `argmax_match < 100%` → Prefill 就存在精度问题

#### 步骤 1.2：Decode 单步对比

使用 Golden（Mode 3）生成的正确 `next_embed`（`codec_sum + tts_pad_vec`），固定作为输入，分别送入 AX Talker decode 和 ONNX Talker decode，对比输出：

```cpp
// 固定输入：使用 Mode 3 保存的 next_embed_fp32
// AX Talker: 需要转换为 bf16 后送入 LLM::RunDecodeStep
// ONNX Talker: 直接送入 onnx_talker->Decode()
// 保存输出：last_hidden [1024], logits [3072]
```

**关键检查点：**
1. 输入 `next_embed` 相同时，AX decode 的 `last_hidden` 是否与 ONNX decode 的 `cos_sim > 0.999`
2. `logits` 的 argmax 是否一致
3. 如果不一致，逐层导出 AX Talker 28 层的每层输出，与 ONNX 的对应层对比

#### 步骤 1.3：Talker 逐层中间结果对比（如需要）

如果步骤 1.2 发现 decode 输出不一致，需要在 AX Talker 的 decode 循环中，每跑完一层 transformer 就保存中间 hidden state，与 ONNX 对应层的输出对比。

> 注：ONNX 的 `talker_decode.onnx` 是一个整体，无法直接拿到中间层输出。可以用 ONNX Runtime 的 `IOBinding` 或者将 ONNX 拆分为 28 个独立层来对比。更实际的做法是：先确认是 prefill 还是 decode 阶段出问题，再决定是否需要逐层对比。

### 4.2 Phase 2: CP 隐藏层对比（优先级：P0）

**目标：** 定位 AX CP 的偏差来源（Transformer 5层 vs 15个 lm_head vs embedding 查表）。

#### 步骤 2.1：固定输入下的 CP 单帧对比

使用 ONNX Talker（Mode 3）在 frame=0 产生的正确 `last_hidden_fp32` 和 `primary_code=1995`，分别输入 ONNX CP 和 AX CP，保存以下中间结果：

**ONNX CP 侧：**
由于 ONNX CP 是单个模型，无法直接拿到中间层。可以用 Python + ONNX Runtime 跑 `code_predictor.onnx`，并尝试用 `onnx.numpy_helper` 或自定义推理来导出每步的 hidden state。

更实际的做法是：先用 Python 重写 ONNX CP 的推理逻辑，与 `qwen3_tts_ablation.cpp` 中的 `OnnxCp::RunFrame` 对齐，但增加中间结果保存。

**AX CP 侧：**
修改 `LLM::RunCpFrame`（或相关实现），在每层 CP transformer 输出后保存 hidden state：

```cpp
// 在 CP 的每层推理后保存 output
for (int layer = 0; layer < 5; ++layer) {
    // run cp layer
    // save output to file: debug_cp_layer_<layer>_output.ax.bin
}
// 在 CP post 后保存 output_norm
// save to file: debug_cp_post_output.ax.bin
// 在每个 lm_head 后保存 logits
for (int j = 0; j < 15; ++j) {
    // save logits to file: debug_cp_lm_head_<j>_logits.ax.bin
}
```

**对比维度：**

| 对比项 | ONNX CP | AX CP | 判定 |
|--------|---------|-------|------|
| CP Transformer 输出 hidden (step 0) | `code_predictor.onnx` 第0步输出 | 5层 axmodel 输出 | 检查是否一致 |
| CP Post 输出 `output_norm` | 同上，取最后一步 | `cp_post.axmodel` 输出 | 检查是否一致 |
| lm_head_0 logits | `code_predictor.onnx` step0 logits | `lm_head_0.axmodel` 输出 | 检查是否一致 |
| residual embed_0 | `code_predictor_embed.onnx` step0 | `codec_embedding.0.bin` 查表 | 检查是否一致 |
| ... | ... | ... | ... |

**关键发现预期：**
- 如果 CP Transformer 输出的 hidden 就不同 → 问题在 **5层模型转换**（可能是 `generation_step` 的缺失导致）
- 如果 Transformer 对但 lm_head 错 → 问题在 **15个独立 head 的权重**
- 如果 logits 对但 embed 错 → 问题在 **embedding 查表**

#### 步骤 2.2：CP 的 `generation_step` 影响验证

**核心疑问：** ONNX CP 的 `generation_step` 参数除了选择 head/embedding 外，是否还影响 transformer 的前向传播？

**验证方法：**
1. 在 Python 中加载 ONNX `code_predictor.onnx`
2. 固定输入 `cp_ctx`（last_hidden + primary_embed），但改变 `generation_step` 的值（0~14）
3. 观察输出的 hidden state 是否随 `generation_step` 变化
   - 如果变化 → 说明 `generation_step` 确实影响 transformer 计算，AX CP 的拆分方案需要重新设计
   - 如果不变 → 说明 `generation_step` 仅用于 head/embedding 选择，AX CP 的 transformer 部分应该是正确的

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("code_predictor.onnx")

# 固定 cp_ctx
D = 1024
cp_ctx = np.random.randn(1, 2, D).astype(np.float32)

for gen_step in range(15):
    gs = np.array([gen_step], dtype=np.int64)
    logits = sess.run(None, {"inputs_embeds": cp_ctx, "generation_step": gs})[0]
    print(f"gen_step={gen_step}: logits_mean={logits.mean():.6f} std={logits.std():.6f}")
```

如果不同 `gen_step` 下 logits 不同，则 AX CP 的 5层 transformer 必须为每个 `gen_step` 使用不同的模型权重（或必须在输入中嵌入 `gen_step`），当前的简单拆分是有缺陷的。

### 4.3 Phase 3: 端到端单帧精准对比（优先级：P1）

在 Phase 1/2 定位到具体偏差层后，进行更细粒度的对比：

1. **BF16 转换精度验证**
   - 将 ONNX 的 FP32 权重转换为 BF16，再转回 FP32，对比与原始 FP32 的 max diff
   - 验证 `fp32_vec_to_bf16` / `bf16_vec_to_fp32` 的实现是否正确

2. **KV Cache 状态对比**
   - 在 Talker decode 和 CP decode 的每一步，保存 KV cache 的内容
   - 对比 AXModel 和 ONNX 的 KV cache 在相同输入下是否一致

3. **Mask 和 Indices 对比**
   - 打印 AX Talker 和 AX CP 的 `mask`、`indices` 张量
   - 验证是否与 ONNX 侧的 `attention_mask`、position ids 等价

---

## 5. 快速验证清单（可先执行）

在启动完整的隐藏层对比前，可以先验证以下简单假设：

### 5.1 AX Talker 快速验证

- [ ] **Prefill 的 last token logits argmax 是否为 1995？**
  - Mode 0/2 的日志显示 `first_primary_code=1995`，说明 prefill 的 argmax 是对的
  - 但需要验证 `last_hidden` 的数值精度

- [ ] **Decode 第一步的输入 `next_embed` 是否正确？**
  - 在 Mode 2 中，ONNX CP 生成 `codec_sum`，加上 `tts_pad_vec` 后作为 decode 输入
  - 打印这个 `next_embed` 的值，与 Mode 3 的对应值对比
  - 如果输入就不同，问题在 CP；如果输入相同但输出不同，问题在 Talker decode

- [ ] **tts_pad_vec 是否加载正确？**
  - 验证 `tts_pad_vec.bin` 的内容与 ONNX 侧使用的是否一致

### 5.2 AX CP 快速验证

- [ ] **CP 第一层输出是否为 0 或异常值？**
  - 参考 `docs/qwen3_tts_debug.md`，如果 `max_logit=0.000000` 说明 CP 推理失败
  - 检查当前 AX CP 是否存在类似的 `inference ret=-1` 问题

- [ ] **CP lm_head 输出维度是否为 [1,1,2048] FP32？**
  - 已确认，但需验证实际输出的数值范围（正常应有正有负，最大 logit 约 0.5+）

- [ ] **CP 的 codec_embedding 查表是否越界？**
  - 如果查表时 token_id 超过了 2048，可能会读到错误数据
  - 但当前 CP vocab=2048，需确认是否有 token_id >= 2048 的情况

---

## 6. 建议的下一步行动

1. **立即执行 Phase 1.1**：保存并对比 prefill 的 `last_hidden` 和 `logits`
   - 如果 prefill 就偏差大 → 重点排查 Talker 模型转换和 BF16 精度
   - 如果 prefill 一致 → 重点排查 decode 循环和 KV cache

2. **立即执行 Phase 2.2**：验证 ONNX CP 的 `generation_step` 是否影响 transformer 计算
   - 如果影响 → 这是 AX CP 设计上的根本缺陷，需要重新考虑模型拆分方案
   - 如果不影响 → 重点排查 AX CP 的 5层 transformer 和 15个 lm_head 的权重加载

3. **修改 ablation 工具增加 debug 导出**
   - 为 Talker 和 CP 的关键中间结果添加 `.bin` 导出功能
   - 避免每次都从日志中人工提取数字

4. **建立自动化对比流水线**
   - 用 Python 脚本自动加载 AX/ONNX 的 debug 输出，计算 cos_sim / mse / max_diff
   - 生成可视化报告（如每层偏差的热力图）

---

## 7. 附录：关键常量与文件路径

| 常量/文件 | 值/路径 | 说明 |
|-----------|---------|------|
| `talker_vocab_size` | 3072 | Talker 输出维度 |
| `code_predictor_vocab_size` | 2048 | CP 输出维度 |
| `hidden_size` | 1024 | 隐藏层维度 |
| `codec_eos_token_id` | 2150 | EOS token |
| `prefill_len` | 93 | 当前测试的 prefill 长度 |
| `talker_prefill.onnx` | `onnx_kv_06b/talker_prefill.onnx` | ONNX Talker prefill |
| `talker_decode.onnx` | `onnx_kv_06b/talker_decode.onnx` | ONNX Talker decode |
| `code_predictor.onnx` | `onnx_kv_06b/code_predictor.onnx` | ONNX CP 主体 |
| `code_predictor_embed.onnx` | `onnx_kv_06b/code_predictor_embed.onnx` | ONNX CP embed |
| `codec_embed.onnx` | `onnx_kv_06b/codec_embed.onnx` | ONNX codec embed |

---

*分析日期：2026-05-01*
*基于文件：talker_*_code_predictor_*.txt, ablation_results/, tools/qwen3_tts_ablation.cpp*
