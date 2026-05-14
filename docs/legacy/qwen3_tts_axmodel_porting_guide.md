# Qwen3-TTS：ONNX → AX650 AXModel 移植开发指南

> 基于 sherpa-onnx ONNX 推理源码与 AX650 已部署 axmodel 的对比梳理，供后续 AX650 完整 pipeline 开发参考。

---

## 1. 模型资产清单（AX650 侧）

### 1.1 Talker（主自回归 LLM）

路径：`talker/`

| 文件 | 作用 | 对应 ONNX |
|------|------|-----------|
| `config.json` | 配置：`axmodel_num=28`, `tokens_embed_num=3072`, `tokens_embed_size=1024` | — |
| `qwen3_tts_talker_p128_l0~l27_together.axmodel` | Talker 28 层 Transformer | `talker_prefill.onnx` + `talker_decode.onnx` |
| `qwen3_tts_talker_post.axmodel` | Talker 后处理，输出 logits | talker post-process |
| `talker.model.text_embedding.weight.bfloat16.bin` | 文本词表 embedding（151936 × 1024） | `text_project.onnx` 权重 |
| `talker.model.codec_embedding.weight.bfloat16.bin` | Codec 词表 embedding（3072 × 1024） | `codec_embed.onnx` 权重 |
| `qwen3_tokenizer.txt` | Qwen3 文本 tokenizer | `tokenizer_dir` |

### 1.2 Code Predictor（残差 codebook 预测）

路径：`code-predictor/`

| 文件 | 作用 | 对应 ONNX |
|------|------|-----------|
| `qwen3_tts_talker_code_predictor_p64_l0~l4_together.axmodel` | Code Predictor 5 层 Transformer | `code_predictor.onnx` 主体 |
| `qwen3_tts_talker_code_predictor_post.axmodel` | CP 后处理，输出 logits [1, 2048] | `code_predictor.onnx` 输出头 |
| `code_predictor_lm_head_0~14.axmodel` | 15 个独立 lm_head（残差 embedding 投影） | `code_predictor_embed.onnx` |
| `talker.code_predictor.model.codec_embedding.0~14.weight.bfloat16.bin` | 15 个残差 codec embedding 表（2048 × 1024） | `code_predictor_embed.onnx` 权重 |

### 1.3 其他（前端/后端，待接入）

| ONNX 模块 | 当前 AX650 状态 | 说明 |
|-----------|-----------------|------|
| `tokenizer12hz_encode.onnx` | ❌ 未提供 | 参考音频 → codec tokens（可选） |
| `speaker_encoder.onnx` | ❌ 未提供 | 参考音频 → speaker embedding（可选） |
| `tokenizer12hz_decode.onnx` | ❌ 未提供 | codec tokens → waveform（必需，后续接入） |
| `tokenizer12hz_decode_stream.onnx` | ❌ 未提供 | 流式小块解码（可选） |

---

## 2. ONNX 推理流程回顾（sherpa-onnx）

### 2.1 整体数据流

```
文本 text
  │
  ▼
Tokenizer → text token IDs (int64)
  │
  ▼
text_project.onnx ──► text embeddings [1, T, 1024] float32
  │
  ├─► role(3) + special(3) + codec_prefix(5) + body_text + trailing
  │     │
  │     ▼
  │  拼接成 prefill_embeds [1, 85, 1024]
  │     │
  │     ▼
  │  talker_prefill.onnx ──► logits [1, 85, 3072], last_hidden [1, 85, 1024], KV-cache(28×2)
  │     │
  │     ▼
  │  AR Loop (per frame):
  │    ├─ Sample primary_code (codebook 0) from logits
  │    ├─ codec_embed.onnx(primary_code) ──► primary_embed [1, 1, 1024]
  │    ├─ Code Predictor Loop (j = 0..14):
  │    │    cp_ctx = concat(last_hidden[-1], primary_embed, prev_residual_embeds...)
  │    │    code_predictor.onnx(cp_ctx, gen_step=j) ──► logits [1, 2048]
  │    │    Sample res_code
  │    │    code_predictor_embed.onnx(res_code, gen_step=j) ──► res_embed [1, 1, 1024]
  │    │    cp_ctx += res_embed; codec_sum += res_embed
  │    ├─ next_in = codec_sum + txt_hidden[step]
  │    └─ talker_decode.onnx(next_in, mask, KV) ──► logits [1, 1, 3072], last_hidden [1, 1, 1024], KV'
  │
  ▼
累计 all_codes [N_frames, 16] int64
  │
  ▼
tokenizer12hz_decode.onnx ──► waveform float32 @ 24kHz
```

### 2.2 Prefill Embedding 构造细节（固定 85 tokens）

```
位置 0~2:   role_embed[0..2]                    (3 tokens)
位置 3~5:   tts_pad_embed + codec_prefix[0..2]  (3 tokens)
位置 6:     tts_bos_embed + codec_prefix[3]     (1 token)
位置 7:     text[0] + codec_bos                 (1 token)
位置 8~84:  text[1..] + tts_eos                 (trailing，长度可变，不足补 tts_pad)
```

注意：ONNX 源码中 `prefill_len = 85` 是硬编码的，与 `qwen3_tts_debug.cpp` 的 `meta.json` 中 `S=85` 一致。

### 2.3 Code Predictor 关键特性

ONNX 源码注释明确说明：
> **"No KV cache: re-runs full attention each step (max 17 tokens)"**

- 输入 `cp_ctx` 初始长度为 2（last_hidden + primary_embed）。
- 每生成一个残差 token，就把其 embedding append 到 `cp_ctx`，因此长度 = 2 + j。
- 最大长度 = 2 + 15 = 17，极小，**不需要 KV cache**，每步全量推理即可。
- `generation_step`（即 `j`）作为额外输入传入，用于模型内部区分当前预测第几个残差 codebook。

---

## 3. ONNX vs AXModel 核心差异

| 维度 | ONNX（sherpa-onnx） | AX650 AXModel（ax-llm） |
|------|---------------------|------------------------|
| **推理运行时** | ONNX Runtime（CPU） | AX Engine（NPU） |
| **数据精度** | float32 | bfloat16（NPU 原生） |
| **Talker 模型形态** | 2 个 ONNX：`prefill` + `decode` | 28 层独立 axmodel + post axmodel，由 `LLM` 类统一管理 prefill/decode 分组 |
| **Talker KV Cache** | 显式传入/传出 `past_key/value_0~27` tensors | `LLM` 类内部通过 `K_cache`/`V_cache` 输入和 mask 管理，上层无感知 |
| **Talker 采样** | CPU 端 `SampleFromLogits`（temperature/top_k/top_p/rep_penalty） | `LLM::post_process`（当前配置 greedy decode，temperature/top_k/top_p 均关闭） |
| **Text Embedding** | `text_project.onnx` 内部权重 | 外部 `talker.model.text_embedding.weight.bfloat16.bin`，`LLaMaEmbedSelector` mmap 加载 |
| **Codec Embedding** | `codec_embed.onnx` 内部权重 | 外部 `talker.model.codec_embedding.weight.bfloat16.bin`（3072 × 1024） |
| **Code Predictor 形态** | 2 个 ONNX：`code_predictor` + `code_predictor_embed` | 5 层 axmodel + post + **15 个独立 lm_head axmodel** + 15 个 embedding bin |
| **CP KV Cache** | 无（全量 re-run，max 17 tokens） | **待确认**：axmodel 是否内部已处理，或也需全量？ |
| **CP generation_step** | 作为 `int64[batch]` input tensor 传入 | **待确认**：axmodel 如何传入该参数？input tensor 还是常量折叠？ |
| **Prefill 长度** | 硬编码 85 | `meta.json` / `config.json` 中 `prefill_token_num=128`（模型文件名 `p128`），但实际输入 `S=85` |

---

## 4. AX650 AXModel 推理流程规划

### 4.1 当前已完成的 Talker 单步

`qwen3_tts_debug.cpp` + `LLM::Run(embed)` 已实现：

```
prefill_embeds.bin (bfloat16 [85, 1024])
  │
  ▼
LLM::Run(combined_embed, max_new_tokens)
  ├─ Prefill: qwen3_tts_talker_p128_l* (28 layers) + KV cache init
  ├─ Post: qwen3_tts_talker_post.axmodel → next_token (0~3071)
  ├─ Decode loop:
  │    embed_selector.getByIndex(next_token) → embed (查 codec_embedding.bin)
  │    qwen3_tts_talker_l* decode 组 → next_token
  │    ...
  └─ Return token_ids vector (当前返回空字符串，已注释 decode)
```

**缺失**：
- 没有接入 code predictor，每帧只有 codebook 0（主 token）。
- 没有 trailing text hidden 叠加到 next input embedding。
- 没有收集 `all_codes` 矩阵。
- 没有音频解码后端。

### 4.2 完整 AXModel Pipeline 架构设计

建议新增 `Qwen3TtsEngine` 类，分层封装：

```
┌─────────────────────────────────────────────────────────────┐
│                    Qwen3TtsEngine                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Frontend  │  │    Talker   │  │  CodePredictor      │  │
│  │  (tokenizer │  │  (ax-llm    │  │  (5-layer axmodel   │  │
│  │   + embed   │  │   LLM class)│  │   + 15 lm_head)     │  │
│  │   build)    │  │             │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│           │              │                    │              │
│           └──────────────┴────────────────────┘              │
│                          │                                   │
│                          ▼                                   │
│                   FrameBuffer [N, 16]                        │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           AudioDecoder (tokenizer12hz_decode)        │   │
│  │              ONNX / 后续转 AXModel                    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 Talker 侧改动点（基于现有 ax-llm）

1. **Embedding 来源**：当前 `qwen3_tts_debug.cpp` 直接喂入预计算的 `prefill_embeds.bin`。后续若走完整前端，需：
   - 文本 tokenizer → text token IDs（用 `qwen3_tokenizer.txt`）
   - 查 `talker.model.text_embedding.weight.bfloat16.bin` 得 text embed
   - 查 `talker.model.codec_embedding.weight.bfloat16.bin` 得 codec embed
   - 按 ONNX 的 85-token layout 拼接成 bfloat16 prefill_embeds

2. **Decode 循环改造**：
   当前 `LLM::Run(embed)` 内部循环是：
   ```cpp
   for (each step):
       next_token = post_process(logits)
       embed = embed_selector.getByIndex(next_token)  // 查 codec embed 表
       run_decode(embed)
   ```
   需要改为：
   ```cpp
   for (each step):
       primary_code = post_process(talker_logits)     // codebook 0
       
       // ── Code Predictor ──
       codec_sum = codec_embed(primary_code)            // [1, 1024]
       cp_ctx = concat(last_hidden[-1], codec_sum)      // [2, 1024]
       for (j = 0..14):
           cp_logits = run_code_predictor(cp_ctx, generation_step=j)
           res_code = argmax(cp_logits)                 // 0~2047
           res_embed = code_predictor_lm_head_embed(j, res_code)  // 查表
           cp_ctx += res_embed
           codec_sum += res_embed
           frame_codes[j+1] = res_code
       
       // ── Next Talker Input ──
       txt_hidden = (step < trailing.size()) ? trailing[step] : tts_pad_vec
       next_embed = codec_sum + txt_hidden              // [1, 1024]
       
       // ── Talker Decode ──
       run_talker_decode(next_embed)
       
       all_codes[step] = [primary_code, res_code_1, ..., res_code_15]
   ```

3. **`LLM` 类接口适配**：
   - 当前 `LLM::Run(std::vector<unsigned short>& embed)` 只能返回 `std::string`。
   - 需要扩展接口或新增 `GenerateTtsTokens()`，能**每步暴露 `last_hidden` 和 `logits`**，供外部 code predictor 使用。
   - 或者将 code predictor 封装进 `LLM` 类内部，但这会增加 `LLM` 类的复杂度。

### 4.4 Code Predictor AXModel 推理设计（重点）

根据模型文件拆分，CP 的 AX650 推理应分为：

**A. CP Transformer 主体**（`qwen3_tts_talker_code_predictor_p64_l0~l4` + `post`）
- 输入：`inputs_embeds` [batch, steps, 1024] bfloat16 + `generation_step` [batch] int64
- 输出：`logits` [batch, 2048] float32（或 bfloat16，需确认）
- **无 KV cache**：每步全量推理，steps 从 2 增长到 17。
- 可用 `axcl` / `AX_ENGINE` 直接跑 5 层模型序列，无需 KV cache 管理。

**B. CP Embedding 查表**（15 个独立子图）
- 每个 `code_predictor_lm_head_%d.axmodel` 对应一个残差 codebook。
- 输入：`input_ids` [batch, 1] int64
- 输出：`embeds` [batch, 1, 1024] bfloat16
- 或者更直接：如果 `lm_head` 只是 `nn.Embedding(2048, 1024)`，可以直接用 `talker.code_predictor.model.codec_embedding.%d.weight.bfloat16.bin` 查表，**不需要跑 axmodel**。
- **建议**：先确认 `code_predictor_lm_head_%d.axmodel` 是否包含额外计算（如投影层）。如果只是 embedding lookup，直接用 bin 文件查表更快。

### 4.5 关键数据结构

```cpp
// 每帧的 16 个 codebook token
struct CodecFrame {
    int16_t codes[16];  // codebook 0~15
};

// CP 上下文（每帧内维护）
struct CodePredictorCtx {
    std::vector<float> cp_ctx;      // concat(last_hidden + primary + residuals), size = (2+j)*1024
    std::vector<float> codec_sum;   // sum of all embeddings, size = 1024
    int generation_step = 0;        // 当前残差索引 0~14
};

// Talker decode 状态（每帧间维护）
struct TalkerDecodeState {
    std::vector<unsigned short> last_hidden;  // [1, 1024] bfloat16
    // KV cache 由 ax-llm LLM 类内部维护，上层不需要显式持有
};
```

---

## 5. 待向用户确认的问题

在继续编码前，以下问题会直接影响接口设计和实现方案：

### 5.1 Code Predictor 的 KV Cache 问题 ✅ 部分自解

**分析结果**：从 `qwen3_tts_talker_code_predictor_p64_l0_together.axmodel` 的二进制元数据中发现：
- decode 模式输出：`K_cache_out` / `V_cache_out` [1, 1, 1024]
- prefill 模式输出：`K_cache_out_1` / `V_cache_out_1` [1, 64, 1024]

**结论**：CP axmodel **确实带有 KV cache 接口**（和 Talker 一样支持 prefill/decode 两种模式）。但由于 CP 序列极短（max 17 tokens），**建议每步使用 prefill 模式全量推理**，不维护 KV state，与 ONNX 侧行为一致，实现最简单。

### 5.2 `generation_step` 的传入方式 ✅ 部分自解

**分析结果**：CP axmodel 的输入结构（`input`、`mask`、`K_cache`、`V_cache`、`indices`）和 Talker 类似，**没有显式的 `generation_step` 输入 tensor**。

**推断**：原始 ONNX 中 `generation_step` 用于在**单个模型内**选择 15 个 codebook 对应的 head/embedding。AX650 侧已将其拆分为：
- 15 个独立的 `code_predictor_lm_head_*.axmodel`
- 15 个独立的 `codec_embedding.*.bin`

因此 **`generation_step` 已隐含在模型选择中**（第 j 步就用 `lm_head_j` + `embedding_j`），**不需要作为 tensor 传入**。

### 5.3 `code_predictor_lm_head_%d.axmodel` 的真实作用 ✅ 已确认

**分析结果**：从 `code_predictor_lm_head_0.axmodel` 元数据确认：
- 输入：`input` [1, 1, 1024] BF16（来自 CP transformer 的 hidden state）
- 输出：`output` [1, 1, 2048] FP32（logits）

**结论**：`code_predictor_lm_head_*.axmodel` 是 **Linear 投影层**，将 CP transformer 输出的 hidden state [1,1,1024] 投影到对应 codebook 的 logits [1,1,2048]。

15 个独立文件对应 15 个残差 codebook，各有一组独立的投影权重。

> **注意**：`talker.code_predictor.model.codec_embedding.*.bin`（2048×1024×2 = 4MB）才是 embedding 查表权重，与 lm_head 是分开的。

### 5.4 `talker_post.axmodel` 的输出维度

`config.json` 中 `tokens_embed_num=3072`。ONNX 的 `talker_decode` 输出 `logits [batch, tokens, 3072]`。
**确认 AX650 侧 `qwen3_tts_talker_post.axmodel` 的输出张量形状是否为 `[batch, tokens, 3072]`？**

### 5.5 `talker.model.text_embedding.weight.bfloat16.bin` 的用途

当前 `qwen3_tts_debug.cpp` 直接喂入 `prefill_embeds.bin`。若后续要走完整前端：
- 文本 tokenizer 得到 token IDs 后，**是否用 `talker.model.text_embedding.weight.bfloat16.bin` 查表**（151936 × 1024）？
- 还是 `text_project.onnx` 在 AX650 侧有其他对应实现？

### 5.6 音频解码后端计划 ✅ 已确认：Host 侧 ONNX 解码

用户确认：**音频解码在 Host 侧完成**，AX650 只负责 Talker + Code Predictor 的 NPU 推理。

这意味着 AX650 侧输出为 `all_codes [N_frames, 16] int64`，通过某种方式（如文件、RPC、共享内存）传给 Host 侧，由 Host 侧的 `tokenizer12hz_decode.onnx` 生成最终波形。

---

## 6. 建议的下一步开发顺序

1. **回答并确认第 5 节的问题**，确定 CP 的 axmodel 推理方式。
2. **先实现 Code Predictor 的最小推理 demo**：
   - 输入：固定的 `last_hidden` + `primary_code`
   - 输出：15 个残差 token
   - 验证 logits 分布和 embedding 求和是否与 ONNX 对齐。
3. **改造 `qwen3_tts_debug.cpp`**：
   - 在现有 Talker decode loop 中，每步插入 CP 调用。
   - 收集 `all_codes` 并打印每帧的 16 个 token。
4. **接入音频解码**：
   - 先以 ONNX Runtime 或 PyTorch 脚本验证 `all_codes` 能正确出声音。
   - 再考虑 AX650 化。
5. **前端补齐**（低优先级）：
   - text tokenizer + embedding 拼接。
   - 参考音频 + speaker encoder（可选）。

---

*文档基于：*
- *sherpa-onnx `offline-tts-qwen3-impl.cc` / `offline-tts-qwen3-model.cc`*
- *AX650 模型目录 `rsp/Qwen3-TTS-12Hz-0.6B-Base-AX650/`*
- *ax-llm `src/runner/LLM.cpp` / `tools/qwen3_tts_debug.cpp`*
