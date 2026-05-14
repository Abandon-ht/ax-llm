# Sherpa-ONNX (ONNX 推理) 与 Python/AXModel 推理流程对比分析

> 基于 `/home/m5stack/Workspace/kaldi/sherpa-onnx/sherpa-onnx/csrc/offline-tts-qwen3-impl.cc`

---

## 一、Sherpa-ONNX 架构总览

Sherpa-ONNX 采用**端到端 ONNX 模型封装**策略，与 AXModel 的"逐层拆解"方式截然不同：

| 模块 | Sherpa-ONNX (ONNX) | Python/C++ (AXModel) |
|------|-------------------|---------------------|
| **Talker Prefill** | 单个 `talker_prefill.onnx`（含 embed→layers→norm→lm_head） | 逐层 axmodel + post axmodel |
| **Talker Decode** | 单个 `talker_decode.onnx`（含 KV-cache 管理） | 逐层 axmodel decode group |
| **Code Predictor** | 单个 `code_predictor.onnx`（**无 KV cache**，全 attention 重算） | 逐层 axmodel + lm_head axmodel |
| **Code Embed** | 独立 `code_predictor_embed.onnx`（input_ids + generation_step → embed） | 直接从 weight bin 查表 |
| **Text/Codec Embed** | 独立 `text_project.onnx`, `codec_embed.onnx` | 原始模型嵌入层或外部查表 |

---

## 二、Sherpa-ONNX 核心推理流程

### 2.1 Talker Prefill

```cpp
// 输入
inputs_embeds  [1, T, D]  float32  // 手动构造的 prefill embeddings
attention_mask [1, T]     int64    // 全 1

// 输出
logits      [1, T, V]  float32  // 每个位置的 logits
last_hidden [1, T, D]  float32  // 每层经过 RMSNorm 后的 hidden states
KV-cache   (多个 tensor)         // 供 decode 复用
```

**关键特点**：
- `position_ids` / `rope` / `indices` **完全内嵌在 ONNX 模型内部**，外部无需传入
- `last_hidden` 已经是 **经过 RMSNorm 的**（ONNX 从 PyTorch trace 了完整 forward）
- 一次性输出所有位置的 logits 和 normed hidden

### 2.2 Talker Decode

```cpp
// 输入
inputs_embeds  [1, 1, D]      float32  // next_embed (codec_sum + txt_hidden)
attention_mask [1, total_len] int64    // 全 1，长度随 step 递增
KV-cache...                           // 来自 prefill 或上一步 decode

// 输出
logits      [1, 1, V]  float32
last_hidden [1, 1, D]  float32  // 已 normed
updated KV-cache
```

**关键特点**：
- 同样，position 计算完全内嵌
- KV cache 由 ONNX Runtime 在模型内部管理
- `attention_mask` 只需传全 1 的 1D mask（长度=total_seq_len），模型内部自行处理 causal mask

### 2.3 Code Predictor

```cpp
// 输入
inputs_embeds  [1, T, D]  float32  // cp_ctx = [last_hidden, primary_embed, res_embed_0, ...]
generation_step [1]       int64    // 当前 step j (0~14)

// 输出
logits [1, V_cp]  float32  // V_cp = 2048
```

**关键特点**：
- **无 KV cache**，每次传入完整上下文（max 17 tokens）
- 模型内部自己计算 full causal attention
- `generation_step` 用于区分当前预测第几个 residual codebook

### 2.4 Code Predictor Embed

```cpp
// 输入
input_ids       [1, 1]  int64  // 上一个生成的 residual code
generation_step [1]     int64  // step j

// 输出
embed [1, 1, D]  float32
```

---

## 三、Sherpa-ONNX 的 Prefill 构造细节

Sherpa-ONNX **完全手动构造 prefill embeddings**，逻辑与 Python `infer.py` 原始 pipeline 一致：

### 3.1 Non-Streaming Mode

```
prefill_len = 6 + body_text_len + 1 + 1

Pos 0-2: role_embed (3 tokens)
Pos 3:   tts_pad + codec_nothink
Pos 4:   tts_pad + codec_think_bos
Pos 5:   tts_pad + codec_think_eos
Pos 6..: text[i] + codec_pad  (每个 body text token)
Pos N-2: tts_eos + codec_pad
Pos N-1: tts_pad + codec_bos
```

### 3.2 Streaming Mode

```
prefill_len = 8

Pos 0-2: role_embed
Pos 3:   tts_pad + codec_nothink
Pos 4:   tts_pad + codec_think_bos
Pos 5:   tts_pad + codec_think_eos
Pos 6:   tts_bos + codec_pad
Pos 7:   text[0] + codec_bos
```

### 3.3 Voice Clone (ICL) Mode

```
prefill_len = 9 + ref_num_frames + 1

Pos 0-2:  role_embed
Pos 3-5:  tts_pad + codec_{nothink, think_bos, think_eos}
Pos 6:    tts_pad + speaker_embed
Pos 7:    tts_bos + codec_pad
Pos 8:    text[0] + codec_bos
Pos 9..:  ICL reference frames (text_embed + codec_embed_sum)
```

### 3.4 Trailing 构造（Streaming）

```cpp
// trailing = body_text[1:] 的 text_embed + tts_eos_embed
trailing[0] = text_embed[1]
trailing[1] = text_embed[2]
...
trailing[last] = tts_eos_embed
```

Decode 时：`next_in = codec_sum + trailing[step]`，若 step >= trailing.size() 则使用 `tts_pad_vec`。

### 3.5 Debug Dump

Sherpa-ONNX 在 `debug=true` 时会输出：
- `prefill_embeds_bf16.bin` → 供 axmodel C++ 使用
- `tts_pad_vec_bf16.bin` → 非流式模式必需
- `trailing_text_hiddens_bf16.bin` → 流式模式使用
- `meta.json`

**这与 `qwen3_tts_infer.cpp` 的预期输入完全一致。**

---

## 四、Sherpa-ONNX 如何处理"无法对齐"的问题

### 4.1 Position IDs / Indices → ONNX 内部封装

**问题**：Python transformers 内部使用 4D mRoPE position_ids，AXModel 需要外部传入静态 indices。

**Sherpa-ONNX 的处理**：
- ONNX 模型在 trace 时已经将 position_ids 的计算逻辑固化在模型内部
- 外部调用者只需传 `attention_mask`（甚至只是全 1 的 mask）
- **完全规避了 indices/position_ids 的外部管理问题**

**对 AXModel 的启示**：
- AXModel 由于逐层部署的架构限制，无法内嵌 position 计算，必须外部构造 indices
- 这是 AXModel 与 ONNX 的**架构级差异**，无法完全消除
- 对齐目标应为：根据 axmodel 输入 tensor 的 shape，尽可能复现 Python 的 indices 填充逻辑

### 4.2 CP KV Cache → 直接不用 KV Cache

**问题**：Python CP 使用 KV cache 的 prefill+decode 模式来加速，但管理复杂（区分 group、mask 形状、KV 更新位置）。

**Sherpa-ONNX 的处理**：
- CP 上下文极短（max 17 tokens = last_hidden + primary + 15 residuals）
- **直接放弃 KV cache，每次传入完整上下文做 full attention**
- 模型输出只有 logits，没有 KV cache tensors
- 代码注释明确说明：`"No KV cache: re-runs full attention each step (max 17 tokens)"`

**对 AXModel 的启示**：
- 如果 AXModel CP 的 prefill group 支持任意长度输入（<=17），可以仿照 sherpa-onnx 的简化策略
- 但当前 Python 实现使用 decode group 来对齐 HuggingFace 行为，若要求与 Python 对齐，AXModel C++ 仍需支持 decode 模式
- **或者**：确认 CP axmodel 是否可以把 decode 步骤也当作短序列 prefill 处理（即忽略 KV cache，每次重新计算）

### 4.3 Talker Norm → ONNX 内部包含

**问题**：Python 中 talker 返回的 hidden_states 是经过 RMSNorm 的，AXModel C++ 中 talker 只输出 raw hidden。

**Sherpa-ONNX 的处理**：
- `talker_prefill.onnx` 和 `talker_decode.onnx` 在 trace 时包含了完整的 `model.norm` + `lm_head`
- 输出的 `last_hidden` 已经是 normed 的
- CP 可以直接使用，无需额外 norm

**对 AXModel 的启示**：
- AXModel C++ 必须在 talker prefill/decode 后手动做 RMSNorm
- 当前 C++ 代码只在 CP 输入前做 norm，且仅对 `txt_hidden` 做，对 decode 循环中的 `embed` 不做，这是不一致的

### 4.4 采样逻辑 → 完整手动实现

**问题**：Python 有自定义的 `_select_next_code_from_logits`。

**Sherpa-ONNX 的处理**：
- 在 C++ 端完整实现了 `SampleFromLogits`，支持 temperature/top_k/top_p/repetition_penalty/suppress
- 实现逻辑与 Python numpy 版本几乎完全一致：
  - temperature scaling
  - partial_sort 实现 top_k 阈值
  - softmax + cumsum 实现 top_p
  - `std::discrete_distribution` 做随机采样

**对 AXModel 的启示**：
- 当前 AXModel C++ 的 CP 采样使用 `postprocess.apply(logits, {})`，history 为空，无法启用 repetition_penalty
- 应仿照 sherpa-onnx 重写 CP 采样逻辑，或至少正确透传采样参数

### 4.5 Attention Mask → 极简 1D Mask

**问题**：Python/AXModel 需要构造复杂的 2D causal mask（prefill）或预计算的 decode mask cache。

**Sherpa-ONNX 的处理**：
- Talker prefill: `attention_mask = [1, 1, 1, ..., 1]` (1D，长度=T)
- Talker decode: `attention_mask = [1, 1, ..., 1]` (1D，长度=total_seq_len)
- 模型内部自己构造 causal mask

**对 AXModel 的启示**：
- AXModel 需要外部传入 2D/4D mask（因为 NPU kernel 的静态 shape 要求），这是架构差异
- 但 C++ 当前的 mask 构造逻辑（`build_prefill_mask`）是正确的，只需确保 decode mask 也正确预计算

---

## 五、三端差异总表

| 维度 | Python (transformers + axmodel) | Sherpa-ONNX (ONNX) | AXModel C++ (当前) |
|------|--------------------------------|-------------------|-------------------|
| **模型粒度** | 逐层 axmodel + post/lm_head | 端到端 ONNX (prefill/decode/CP) | 逐层 axmodel + post/lm_head |
| **Talker Position** | 外部构造 3D indices | 模型内部计算 | 仅填单行 indices |
| **Talker Norm** | Python 端 PyTorch RMSNorm | ONNX 内部包含 | 不做或延迟到 CP 前 |
| **Talker Mask** | 外部构造 2D causal mask | 传 1D 全 1 mask | `build_prefill_mask` |
| **CP KV Cache** | prefill+decode KV cache | **无 KV cache**，全 attention | 手动管理但全用 prefill group |
| **CP Mask** | decode mask cache (预计算) | 模型内部处理 | `build_prefill_mask` |
| **CP Embed** | PyTorch Embedding 层 | `code_predictor_embed.onnx` | 直接从 bf16 bin 查表 |
| **Prefill 构造** | transformers 内部 / 手动 | 完全手动构造 | 读取 dump 的 bin |
| **Trailing** | transformers generate 驱动 | 预计算 trailing 列表 | 从 `all_prefill_hidden` 取 |
| **采样** | numpy 实现 | C++ `SampleFromLogits` | `postprocess.apply` (greedy) |

---

## 六、对 AXModel C++ 重写的关键建议

基于 Sherpa-ONNX 的"简化哲学"，AXModel C++ 的重写应聚焦以下优先级：

### P0（必须对齐）

1. **Talker prefill indices 多行填充**
   - 参考 Sherpa-ONNX，虽然它不需要外部传 indices，但 AXModel 必须传。应检测 indices tensor 的 `nSize`，若支持多行则按 Python 逻辑填充 3 行 repeat。

2. **CP 区分 prefill/decode group**
   - Sherpa-ONNX 放弃了 KV cache，但 AXModel 不能简单放弃（因为模型已编译为 prefill/decode 分组）。
   - 必须仿照 Python，j=0 用 prefill_gid，j>0 用 decode_gid。

3. **Talker 输出后做 RMSNorm**
   - Sherpa-ONNX 的 ONNX 模型输出已 normed。AXModel C++ 必须在 talker prefill 结束后对 hidden 做 norm，才能作为 CP 输入对齐。

### P1（强烈建议对齐）

4. **CP 采样逻辑重写**
   - 直接参考 Sherpa-ONNX 的 `SampleFromLogits` 实现，替换 `postprocess.apply`。

5. **CP decode KV cache 按位置更新**
   - 使用 decode group 后，只更新 `current_len` 位置的 KV，而非整体拷贝。

### P2（可选优化）

6. **CP 是否可放弃 KV cache？**
   - 若 CP axmodel 的 prefill group 支持短序列（如 T<=17）且性能可接受，可参考 Sherpa-ONNX 的简化策略，每次重新跑 prefill。
   - 但这需要与 Python 行为确认一致性，因为 Python 使用了 KV cache 的 decode。
