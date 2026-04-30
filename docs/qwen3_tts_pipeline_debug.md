# Qwen3-TTS ONNX Pipeline 调试文档

> 基于 `qwen3-tts.png` 流程图梳理，供 AX650 部署调试参考。

---

## 1. 整体架构

Qwen3-TTS 分为 **前端** → **Talker（LLM）** → **Codec 后处理** 三大部分。

```
文本/参考音频 ──► 前端 Tokenizer / Encoder ──► Prefill Embeddings
                                                      │
                                                      ▼
                                    ┌─────────────────────────────┐
                                    │     Talker (LLM core)       │
                                    │  prefill ► decode loop      │
                                    │  KV-cache + codebook pred   │
                                    └─────────────────────────────┘
                                                      │
                                                      ▼
                                            帧级 Codec Codes
                                            (16 codebooks × N帧)
                                                      │
                                                      ▼
                                    ┌─────────────────────────────┐
                                    │   tokenizer12hz_decode      │
                                    │   (非流式 / 流式可选)        │
                                    └─────────────────────────────┘
                                                      │
                                                输出音频 24kHz
```

---

## 2. 前端模块（输入处理）

### 2.1 文本分支
| 模块 | 文件 | 输入 | 输出 | 说明 |
|------|------|------|------|------|
| Tokenizer | `tokenizer_dir` | 文本 `text` | `text_token IDs` | 与 LLM 共用 Qwen3 tokenizer |
| Text Projection | `text_project.onnx` | `text_token IDs` | 文本 embedding | 将词表 ID 映射到 hidden_size |

### 2.2 参考音频分支（可选）
| 模块 | 文件 | 输入 | 输出 | 说明 |
|------|------|------|------|------|
| Audio Codec Encoder | `tokenizer12hz_encode.onnx` | 参考音频 waveform | `codec tokens` | 12Hz 采样，生成离散 codec |
| Speaker Encoder | `speaker_encoder.onnx` | 参考音频 mel | `speaker embedding` | 说话人音色嵌入 |

> **调试注意**：`qwen3_tts_debug.cpp` 当前跳过前端，直接加载 `prefill_embeds.bin`（即预计算好的 embeddings）。若需端到端验证，需补充 tokenizer + text_project + audio encoder 链路。

---

## 3. Embedding 构造（Prefill 输入）

在送入 Talker 之前，需将多种 embedding **拼接/组合** 成完整的 prefill 序列：

```
[codec special tokens]  ──► code_embed.onnx ──► codec embedding
[text tokens]           ──► text_project.onnx ──► text embedding
```

### 3.1 Codec 特殊 Token IDs
图中标注包含：
- `codec_nothink`
- `think_bos`
- `think_eos`
- `pad`
- `bos`

这些通过 `code_embed.onnx` 映射为 codec embedding。

### 3.2 组合规则（Prefill Embeddings）
```
prefill_embeddings = concat(
    role/text/special embeddings,   <-- 来自 text_project / code_embed
    codec prefix embeddings         <-- 参考音频或占位 pad
)
```

> **关键尺寸**：`[S, hidden_size]`，其中 `S = 85` 为当前调试配置中的 prefill token 数。

---

## 4. Talker（LLM Core）

Talker 本质上是一个自回归 LLM，但输出不是文本词表，而是 **Codec Codebook Token**。

### 4.1 Prefill 阶段
| 模块 | 文件 | 输入 | 输出 |
|------|------|------|------|
| Talker Prefill | `talker_prefill.onnx` | `prefill_embeddings` + `attention_mask` | `logits` + `last_hidden` + **KV Cache** |

- 一次性处理整段输入（S tokens）。
- 初始化 KV Cache。
- 输出 **主 codec token（当前帧 codebook 0）**。

### 4.2 Decode 循环（单步）
| 模块 | 文件 | 输入 | 输出 |
|------|------|------|------|
| Talker Decode | `talker_decode.onnx` | `last token embedding` + `KV Cache` | `logits` + 更新后的 **KV Cache** |

- 每步生成 **一帧**的 codec 信息。
- 与标准文本 LLM 不同：decode 输出需经 **Code Predictor** 扩展为 16 个 codebook。

> **与 ax-llm 对应关系**：
> - `talker_prefill.onnx` + `talker_decode.onnx` 对应 `ax-llm` 中的 `LLM::Run(embed)`。
> - 当前 `qwen3_tts_debug.cpp` 将 talker 视为普通 LLM，输出维度已改为 3072（即 codec vocab）。
> - 但 **code predictor 链路尚未接入**，当前仅能得到 codebook 0 的 token，缺失残差 codebook 1~15。

---

## 5. Codebook 预测（核心差异点）

这是 Qwen3-TTS 与常规文本 LLM 的最大区别：

### 5.1 单帧结构
每帧音频由 **16 个 codebook** 组成：
- `codebook 0`：主 token（由 Talker 直接输出）
- `codebook 1~15`：残差 token（由 Code Predictor 迭代生成）

### 5.2 残差生成链路
```
主 codec token (codebook 0)
         │
         ▼
┌──────────────────────┐
│ code_predictor.onnx  │  <-- 输入：主 token / 上一层残差 + layer_idx
│ 预测残差 codebook    │
└──────────────────────┘
         │
         ▼
采样 residual token j
         │
         ▼
┌──────────────────────────────┐
│ code_predictor_embed.onnx    │  <-- 输入：residual token + layer_idx
│ residual token -> embedding  │
└──────────────────────────────┘
         │
         ▼
加入 codebook embedding 累加池
```

### 5.3 Embedding 求和
```
next_input_embedding = sum(
    code_embed(主 token),          # codebook 0
    code_predictor_embed(residual_1, layer=1),
    code_predictor_embed(residual_2, layer=2),
    ...,
    code_predictor_embed(residual_15, layer=15)
)
```

> **调试注意**：当前 `qwen3_tts_debug.cpp` **缺失此链路**。若直接拿 Talker 输出的单个 token 去解码音频，音质会严重下降（仅有 codebook 0 信息）。

---

## 6. 音频解码（后处理）

### 6.1 非流式路径
| 模块 | 文件 | 输入 | 输出 |
|------|------|------|------|
| Codec Decoder | `tokenizer12hz_decode.onnx` | 累计所有帧的 16-codebook tokens | `waveform` @ 24kHz |

### 6.2 流式路径（streaming 时替代）
| 模块 | 文件 | 输入 | 输出 |
|------|------|------|------|
| Stream Decoder | `tokenizer12hz_decode_stream.onnx` | 小块 codec tokens（流式缓冲） | 增量 waveform |

> **当前状态**：`qwen3_tts_debug.cpp` 只走到 Talker 输出 token ID，未接入 `tokenizer12hz_decode`。后续需补充：
> 1. 收集完整 `token_ids` 序列；
> 2. 按帧组织为 `[N_frames, 16]` 的 codebook 矩阵；
> 3. 送入 `tokenizer12hz_decode.onnx` 生成音频。

---

## 7. 与当前代码的映射 & TODO

### 7.1 已完成的调试链路
```
prefill_embeds.bin ──► qwen3_tts_debug ──► LLM::Run(embed)
                                               │
                                               ▼
                                        Talker decode loop
                                        (输出 0~3071 token IDs)
```

### 7.2 待补充链路（高优先级）
1. **Code Predictor**：`talker_decode` 输出后，需循环调用 `code_predictor.onnx` + `code_predictor_embed.onnx` 生成 16 个 codebook。
2. **帧组织**：维护 `vector<vector<int>> codec_codes`，每帧 16 个 token。
3. **Audio Decoder**：帧数足够后，调用 `tokenizer12hz_decode.onnx` 生成波形。
4. **前端补齐**：text tokenizer → text_project；可选参考音频 → speaker_encoder + tokenizer12hz_encode。

### 7.3 关键调试打印建议
在 `qwen3_tts_debug.cpp` 或后续 wrapper 中建议增加：
```cpp
// 每帧主 token
printf("frame=%d codebook0=%d\n", frame_idx, main_token);

// 残差 codebook
for (int cb = 1; cb < 16; ++cb) {
    printf("  cb%d=%d\n", cb, residual_tokens[cb]);
}

// 最终帧统计
printf("total_frames=%d total_codec_tokens=%zu\n",
       frame_count, all_tokens.size());
```

---

## 8. 维度速查表

| 名称 | 图中标注 | 当前调试值 | 说明 |
|------|----------|-----------|------|
| Prefill tokens | S | 85 | prefill_embeds.bin 的序列长度 |
| Hidden size | hidden_size | 1024 | Talker LLM 隐藏层维度 |
| Audio token ID | audio_token_id | 151676 | 图中未显式标注，来自 meta.json |
| Audio slots | audio_slots | 81 | 实际填充的音频帧槽位数 |
| Codebook 数 | 16 | 16 | 每帧由 16 个 codebook 组成 |
| Output vocab | 3072 | 3072 | Talker 输出维度（已修改） |
| Audio sample rate | 24 kHz | 24 kHz | 最终输出音频采样率 |

---

## 9. 参考文件清单

| 文件 | 作用 |
|------|------|
| `tools/qwen3_tts_debug.cpp` | 当前 Talker 调试入口 |
| `src/runner/LLM.cpp` | LLM prefill/decode 引擎（已去文本 decode） |
| `docs/vision_encoder_patterns.md` | 项目中已有的编码器模式文档（可参考结构） |
| `tokenizer12hz_encode.onnx` | 音频 → codec tokens（待接入） |
| `tokenizer12hz_decode.onnx` | codec tokens → waveform（待接入） |
| `code_predictor.onnx` | 残差 codebook 预测（待接入） |
| `code_predictor_embed.onnx` | 残差 token → embedding（待接入） |
| `talker_prefill.onnx` / `talker_decode.onnx` | 对应 ax-llm 中加载的 axmodel 组 |

---

*文档生成时间：2026-04-30*
*基于：Qwen3-TTS ONNX Pipeline 流程图（sherpa-onnx）*
