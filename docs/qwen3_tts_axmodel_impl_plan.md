# Qwen3-TTS AX650 实现方案：Talker + Code Predictor 逐帧推理

> **自包含文档**：供后续独立会话直接参考，无需依赖前文上下文。  
> **当前目标**：实现 Talker 1 个主 token + CP 15 个残差 token = 每帧 16 个 codebook token 的完整生成，输出为 `.npy`/`.bin` 文件，暂不做音频解码。

---

## 1. 模型资产速查

### 1.1 Talker（主自回归模型）
路径：`talker/`

| 文件 | 作用 |
|------|------|
| `qwen3_tts_talker_p128_l0~l27_together.axmodel` (28 个) | Talker Transformer 28 层 |
| `qwen3_tts_talker_post.axmodel` | Talker 后处理，输出 logits [1, tokens, **3072**] |
| `talker.model.text_embedding.weight.bfloat16.bin` | 文本 embedding 表，151936 × 1024 |
| `talker.model.codec_embedding.weight.bfloat16.bin` | Codec embedding 表，3072 × 1024 |
| `qwen3_tokenizer.txt` | Qwen3 tokenizer |

### 1.2 Code Predictor（残差 codebook 预测）
路径：`code-predictor/`

| 文件 | 作用 |
|------|------|
| `qwen3_tts_talker_code_predictor_p64_l0~l4_together.axmodel` (5 个) | CP Transformer 5 层（支持 prefill + decode） |
| `qwen3_tts_talker_code_predictor_post.axmodel` | CP 后处理，输出 `output_norm` [1, 1, **1024**] BF16 |
| `code_predictor_lm_head_0~14.axmodel` (15 个) | 15 个独立投影头，输入 [1,1,1024]，输出 [1,1,**2048**] FP32 logits |
| `talker.code_predictor.model.codec_embedding.0~14.weight.bfloat16.bin` (15 个) | 15 个残差 embedding 表，各 2048 × 1024 |

> **关键结论**：ONNX 中的 `generation_step` 参数在 AX650 侧已**隐含在模型选择中**。第 `j` 个残差 token 直接用 `lm_head_j.axmodel` + `codec_embedding.j.bin`，无需显式传入 `generation_step`。

---

## 2. 每帧数据流（Frame-wise Pipeline）

Talker 和 CP 不是先后独立跑完再组合，而是**每帧内紧密耦合**：

```
Frame 0:
  Talker Prefill (85 tokens) ──► logits[85, 3072], last_hidden[85, 1024], KV_cache
  Sample primary_code_0 (codebook 0)
  CP 生成 15 个残差 token ──► frame_0 = [cb0, cb1, ..., cb15]
  codec_sum = embed(cb0) + Σ embed(cbj)
  next_talker_input = codec_sum + txt_hidden[0]
  Talker Decode(next_talker_input) ──► logits[1, 3072], last_hidden[1, 1024], KV_cache'

Frame 1:
  Sample primary_code_1 (codebook 0)
  CP 生成 15 个残差 token ──► frame_1 = [cb0, cb1, ..., cb15]
  next_talker_input = codec_sum + txt_hidden[1]
  Talker Decode ──► ...

...直到 primary_code == codec_eos_token_id (2150)
```

> **与 ONNX 侧差异**：ONNX 中 CP 无 KV cache，每步把完整 `cp_ctx` 全量重跑（max 17 tokens）。AX650 CP 模型**支持 KV cache**，因此每帧内 CP 应该用 **prefill → decode** 模式，避免重复计算。
>
> ✅ **已确认**：CP 的 KV cache **每帧开始时重置**（即 talker 输出新的 primary token 后，CP 的 KV 重新计算，不跨帧复用）。

---

## 3. Code Predictor 详细推理步骤（每帧内）

### 3.1 初始状态

```cpp
// 来自 Talker 上一步的输出
std::vector<float> last_hidden_1x1024;   // talker last_hidden 最后一个位置
std::vector<float> primary_embed_1x1024; // 查 talker.model.codec_embedding.bin (3072×1024)

// CP 内部状态（每帧开始时重置）
std::vector<std::vector<float>> cp_kv_cache;  // 由 axmodel 输出/输入，层数=5
int32_t cp_seq_len = 0;
```

### 3.2 CP Prefill（第 0 步，送入 2 个 token）

构造 `cp_ctx_0`：
```
cp_ctx_0 = concat(last_hidden_1x1024, primary_embed_1x1024)
         // shape: [2, 1024]
```

送入 CP 5 层 axmodel（**prefill 模式**）：
- 输入：`input` [1, 2, 1024] BF16, `mask` [1, 2] int64, `indices` [1, 2] int64, `K_cache`/`V_cache` 为空或 zero
- 输出：`output_1` [1, 2, 1024] BF16, `K_cache_out_1` [1, 64, 1024] BF16, `V_cache_out_1` [1, 64, 1024] BF16
- 取 `hidden_0 = output_1[0, -1, :]` → [1, 1024]

> **注意**：prefill 模式输出名带 `_1` 后缀（`output_1`, `K_cache_out_1`），对应 `p64` 的 prefill trace。decode 模式输出名不带后缀（`output`, `K_cache_out`）。

送入 `code_predictor_lm_head_0.axmodel`：
- 输入：`input` [1, 1, 1024] BF16
- 输出：`output` [1, 1, 2048] FP32
- `logits_0 = output[0, 0, :]`
- `res_code_0 = argmax(logits_0)`（或 sampling）

查表得 embedding：
```cpp
res_embed_0 = lookup(
    talker.code_predictor.model.codec_embedding.0.weight.bfloat16.bin,
    res_code_0
); // [1, 1024] bfloat16
```

初始化累加器：
```cpp
codec_sum = primary_embed_1x1024 + res_embed_0;  // [1, 1024]
frame_codes[0] = primary_code;      // codebook 0 来自 talker
frame_codes[1] = res_code_0;        // codebook 1 来自 CP step 0
cp_seq_len = 2;                     // 已处理 2 个 token 的 KV
```

### 3.3 CP Decode Loop（第 j 步，j = 1 .. 14）

每步只送入**单个新 token**（上一步的残差 embedding），利用 KV cache：

```cpp
for (int j = 1; j <= 14; ++j) {
    // 上一步的残差 embedding 作为当前步的输入
    std::vector<float> step_input = res_embed_{j-1};  // [1, 1024]
    
    // 送入 CP（decode 模式）
    // 输入：
    //   input    [1, 1, 1024]      BF16  ← step_input
    //   mask     [1, cp_seq_len+1] int64 ← 全 1
    //   indices  [1, 1]            int64 ← [cp_seq_len]
    //   K_cache  [1, cp_seq_len, 1024]   BF16 ← 上一步 K_cache_out
    //   V_cache  [1, cp_seq_len, 1024]   BF16 ← 上一步 V_cache_out
    // 输出：
    //   output       [1, 1, 1024]  BF16
    //   K_cache_out  [1, cp_seq_len+1, 1024] BF16
    //   V_cache_out  [1, cp_seq_len+1, 1024] BF16
    
    auto cp_result = RunCpDecode(step_input, cp_kv_cache);
    std::vector<float> hidden_j = cp_result.output;          // [1, 1024]
    cp_kv_cache = {cp_result.k_cache, cp_result.v_cache};    // 更新
    cp_seq_len++;
    
    // 送入对应 lm_head
    auto logits_j = RunLmHead(j, hidden_j);  // code_predictor_lm_head_j.axmodel
    int32_t res_code_j = ArgMax(logits_j);    // [2048] → int
    
    // 查表
    res_embed_j = LookupCodecEmbedding(j, res_code_j);
    
    // 累加
    codec_sum += res_embed_j;
    frame_codes[j + 1] = res_code_j;  // frame_codes[0]=primary, [1]=res_0, [2]=res_1...
}
```

### 3.4 构造 Next Talker Input

```cpp
// txt_hidden 来自 prefill 阶段计算的 trailing text embeddings
// 如果 step < trailing.size()，用 trailing[step]
// 否则用 tts_pad_vec（全 1 的 attention mask 对应的 padding embed）
std::vector<float> txt_hidden = (step < trailing.size()) ? trailing[step] : tts_pad_vec;

std::vector<float> next_embed(D);
for (int d = 0; d < 1024; ++d) {
    next_embed[d] = codec_sum[d] + txt_hidden[d];
}
// next_embed [1, 1024] → 作为 Talker Decode 的输入
```

---

## 4. 采样策略（Talker + CP）

Talker 和 CP 各自独立采样，但**复用同一套采样逻辑**（`LLMPostprocess.hpp` 中的实现）。

### 4.1 复用 `LLMPostprocess`

`src/runner/LLMPostprocess.hpp` 已提供完整采样方法：
- `apply_temperature(logits, temperature)`
- `apply_repetition_penalty(logits, history, penalty, window)`
- `top_k_sampling(logits, k)` — partial_sort + softmax + discrete_distribution
- `faster_top_p_sampling(logits, top_p)` — heap + softmax + discrete_distribution
- `apply(logits, history)` — 按配置顺序组合上述操作

### 4.2 主参数 vs Subtalker 参数

官方配置中 Talker 和 CP 使用**同一套采样参数**，但 AX650 侧应支持独立配置：

| 参数 | Talker (主) | CP (subtalker) | 说明 |
|------|------------|----------------|------|
| `do_sample` | `talker_do_sample` | `subtalker_do_sample` | 是否启用采样（false=greedy） |
| `temperature` | `talker_temperature` | `subtalker_temperature` | 温度缩放 |
| `top_k` | `talker_top_k` | `subtalker_top_k` | top-k 截断 |
| `top_p` | `talker_top_p` | `subtalker_top_p` | top-p 核采样 |
| `repetition_penalty` | `talker_rep_penalty` | — | CP 通常不需要 rep penalty |

**默认配置**（对齐官方）：
```json
{
    "talker_do_sample": true,
    "talker_temperature": 0.9,
    "talker_top_k": 50,
    "talker_top_p": 1.0,
    "talker_repetition_penalty": 1.05,
    
    "subtalker_do_sample": true,
    "subtalker_temperature": 0.9,
    "subtalker_top_k": 50,
    "subtalker_top_p": 1.0
}
```

> **注意**：当前 `post_config.json` 中所有 `enable_*` 为 `false`，是 greedy decode 测试配置。实际部署应通过外部 JSON 传入完整采样参数。

### 4.3 每帧采样流程

```cpp
// Talker 采样（主 codebook 0）
std::vector<float> talker_logits_3072 = ...;  // 来自 talker_post.axmodel
int primary_code = talker_postprocess.apply(talker_logits_3072, generated_primary_history);

// CP 采样（残差 codebook 1~15）
std::vector<float> cp_logits_2048 = ...;      // 来自 code_predictor_lm_head_j.axmodel
int res_code_j = cp_postprocess.apply(cp_logits_2048, {});  // CP 无 rep penalty history
```

---

## 5. 与 ax-llm LLM 类的对接改造

当前 `LLM::Run(embed)` 内部循环：
```cpp
for (each step) {
    next_token = post_process(logits);                // ← primary_code (codebook 0)
    embed = embed_selector.getByIndex(next_token);    // ← primary_embed (查 codec_embedding.bin)
    run_decode(embed);                                // ← Talker 单步 decode
}
```

**需要扩展为**：
```cpp
for (int step = 0; step < max_new_tokens; ++step) {
    // 1. Talker 输出 logits，采样主 token
    int32_t primary_code = PostProcess(talker_logits);  // 0~3071
    if (primary_code == codec_eos_token_id) break;
    
    // 2. 查主 codec embedding
    auto primary_embed = LookupTalkerCodecEmbedding(primary_code);  // [1024]
    
    // 3. CP 生成 15 个残差 token（每帧内串行）
    auto frame = RunCodePredictorFrame(last_hidden, primary_embed, step);
    // frame.codes[16] = {primary_code, res_0, ..., res_14}
    
    // 4. 累加所有 embedding
    auto codec_sum = primary_embed + Sum(frame.residual_embeds);
    
    // 5. 叠加文本 hidden，构造 next talker input
    auto next_embed = codec_sum + GetTxtHidden(step);
    
    // 6. 送入 Talker decode，得到下一步 logits
    auto dr = RunTalkerDecode(next_embed);
    talker_logits = dr.logits;
    last_hidden = dr.last_hidden;
    
    // 7. 保存本帧
    all_codes.push_back(frame.codes);
}
```

### 5.1 需要 LLM 类暴露的接口

当前 `LLM::Run(std::vector<unsigned short>& embed, int output_max_token)` 是黑盒，无法拿到每步的 `last_hidden` 和 `logits`。

**建议新增接口**：
```cpp
struct TalkerDecodeStepResult {
    int next_token;                       // primary_code (codebook 0)
    std::vector<unsigned short> last_hidden;  // [1, 1024] bfloat16，最后一个位置
    // KV cache 仍由 LLM 内部管理
};

// 单步执行：输入 embed [1, 1024]，输出 token + last_hidden
TalkerDecodeStepResult LLM::RunDecodeStep(std::vector<unsigned short>& embed);

// 重新运行 prefill（如果需要在帧之间重置）
TalkerPrefillResult LLM::RunPrefill(std::vector<unsigned short>& prefill_embeds);
```

或者，如果改动 `LLM` 类风险太大，可以：
- 在 `qwen3_tts_debug.cpp` 中**绕过 `LLM::Run()`**，直接调用 `LLM` 内部的 `llama_layers` 和 `llama_post` 做逐帧控制。
- 但这需要把 `LLM` 类的私有成员暴露出来，或新增 `friend` 类。

### 5.2 最小侵入方案

在 `LLM` 类中新增一个专门的 TTS decode 接口：
```cpp
// 新增于 LLM.hpp
struct TtsDecodeCallback {
    // 每生成一个 primary_code 时回调，由外部实现 CP 逻辑
    virtual std::vector<int> OnPrimaryCode(
        int primary_code,
        const std::vector<unsigned short>& last_hidden_bf16
    ) = 0;
};

// LLM::RunTts 内部循环中，每步采样完 primary_code 后调用 callback
// callback 返回本帧完整的 16 个 codebook token
// LLM 内部负责构造 next_embed 并继续 decode
```

> **推荐**：先实现一个独立的 `Qwen3TtsEngine` 类，内部持有 `LLM` 实例和 CP 相关模型，通过新增 `LLM::RunDecodeStep()` 接口获取 `last_hidden`，再由 `Qwen3TtsEngine` 驱动 CP。

---

## 6. CP 模型加载与推理封装

### 6.1 CP Transformer（5 层 + post）

可以直接复用 ax-llm 中加载 `axmodel` 的方式（`AX_ENGINE` API），但因为 CP 只有 5 层且结构简单，也可以直接用 `axcl` 低级 API 逐个加载和推理。

不过，为了复用现有基础设施，**建议复用 `LLM` 类的分组机制**：
- CP 5 层相当于一个 mini LLM
- `p64` 表示 prefill 最大 64 tokens，decode 最大 64 tokens
- 可以用类似 `init_groups_from_model` 的方式初始化 CP 模型组

### 6.2 CP 输入输出张量规格（decode 模式）

基于二进制元数据分析：

| 输入名 | 形状 | 类型 | 说明 |
|--------|------|------|------|
| `input` | [1, 1, 1024] | BF16 | 当前 step 的 embedding |
| `mask` | [1, seq_len] | INT64 | attention mask，全 1 |
| `indices` | [1, 1] | INT64 | 当前位置索引 = seq_len |
| `K_cache` | [1, prev_seq, 1024] | BF16 | 上一步输出的 K_cache_out |
| `V_cache` | [1, prev_seq, 1024] | BF16 | 上一步输出的 V_cache_out |

| 输出名 | 形状 | 类型 | 说明 |
|--------|------|------|------|
| `output` | [1, 1, 1024] | BF16 | 当前 step 的 hidden state |
| `K_cache_out` | [1, seq_len, 1024] | BF16 | 更新后的 K cache |
| `V_cache_out` | [1, seq_len, 1024] | BF16 | 更新后的 V cache |

> **seq_len 关系**：
> - Prefill 第 0 步：input [1,2,1024] → output_1 [1,2,1024], K/V_cache_out_1 [1,64,1024]
> - Decode 第 j 步：input [1,1,1024] → output [1,1,1024], K/V_cache_out [1,2+j,1024]

### 6.3 15 个 lm_head 的加载与调用

每个 `code_predictor_lm_head_j.axmodel` 是独立单输入单输出模型：
- 输入：`input` [1, 1, 1024] BF16
- 输出：`output` [1, 1, 2048] FP32

可以用 `AX_ENGINE` 直接创建 15 个 session，或者复用 `axcl` 的模型加载器。

### 6.4 15 个 Embedding 查表

直接用 `mmap` 加载 15 个 `talker.code_predictor.model.codec_embedding.j.weight.bfloat16.bin`：
```cpp
std::vector<std::vector<unsigned short>> cp_codec_embeds;  // [15][2048 * 1024]

// 查表
void LookupCpEmbedding(int codebook_idx, int token_id, unsigned short* out_embed) {
    const auto& table = cp_codec_embeds[codebook_idx];  // 2048 × 1024
    memcpy(out_embed, &table[token_id * 1024], 1024 * sizeof(unsigned short));
}
```

---

## 7. 输出格式

当前阶段**不做音频解码**，只保存生成的 codec tokens。

### 7.1 建议输出格式

**`.npy` 文件**（最方便 Python 后处理读取）：
```python
# Python 侧读取示例
import numpy as np
codes = np.load("output_codes.npy")  # shape: [N_frames, 16], dtype=int64
```

或 **`.bin` 原始二进制** + `meta.json`：
```cpp
// C++ 侧保存
FILE* fp = fopen("output_codes.bin", "wb");
fwrite(all_codes.data(), sizeof(int32_t), all_codes.size(), fp);
fclose(fp);

// meta.json
{
    "num_frames": 128,
    "num_codebooks": 16,
    "dtype": "int32",
    "shape": [128, 16]
}
```

### 7.2 Debug 打印（每帧）

```cpp
printf("frame=%d primary=%d", frame_idx, primary_code);
for (int j = 0; j < 15; ++j) {
    printf(" res_%d=%d", j, frame.residual_codes[j]);
}
printf("\n");
```

---

## 8. 开发顺序建议

| 步骤 | 任务 | 验证标准 |
|------|------|----------|
| 1 | 实现 CP Transformer 单模型推理封装（复用 AX_ENGINE API） | 能成功加载 `code_predictor_p64_l0` 并跑通一次推理 |
| 2 | 实现 CP Prefill → Decode 循环（一帧内 15 步） | 输入固定 dummy data，输出 15 个 int token，数值与 ONNX 对齐 |
| 3 | 接入 15 个 lm_head + 15 个 embedding 查表 | 每步输出 logits [2048] + embed [1024] |
| 4 | 改造 `LLM` 类或新增接口，暴露 `last_hidden` | `qwen3_tts_debug.cpp` 能拿到 talker 每步的 last_hidden |
| 5 | Talker + CP 联调（一帧） | primary_code + 15 残差 → 16 个 token，构造 next_embed 继续 talker decode |
| 6 | 完整 AR loop（多帧） | 生成全部帧，遇到 eos 停止，保存 output_codes.npy |
| 7 | 与 Host 侧 ONNX decoder 对齐 | npy 文件能在 Python 侧通过 `tokenizer12hz_decode.onnx` 出声音 |

---

## 9. 关键常量汇总

| 常量 | 值 | 来源 |
|------|-----|------|
| Talker vocab size | 3072 | `config.json` tokens_embed_num |
| CP vocab size | 2048 | `code_predictor_lm_head` 输出维度 |
| Hidden size | 1024 | `config.json` tokens_embed_size |
| Num code groups | 16 | 每帧 16 个 codebook |
| Prefill len (talker) | 85 | 硬编码，与 ONNX 一致 |
| Prefill max (CP) | 64 | 模型文件名 `p64` |
| codec_eos_token_id | 2150 | `talker_config.codec_eos_token_id` |
| tts_pad_token_id | 151671 | `config.json` 顶层 |
| Audio sample rate | 24000 Hz | 最终输出（Host 侧解码） |

---

*文档版本：v1.0*  
*基于：AX650 模型文件元数据分析 + sherpa-onnx `offline-tts-qwen3-impl.cc` 源码*  
*文档版本：v1.1（已整合用户确认：CP KV 每帧重置、可修改任意代码、采样复用 LLMPostprocess）*  
*目标读者：后续独立开发会话*
