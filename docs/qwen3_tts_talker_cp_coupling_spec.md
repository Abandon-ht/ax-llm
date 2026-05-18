# Qwen3-TTS Talker 与 Code Predictor 耦合逻辑详细规范

> **版本**: v1.0  
> **目的**: 为 C++ 侧实现 Talker + Code Predictor (CP) 的端到端耦合推理提供精确的逻辑规范。  
> **依据**:  
> - `Qwen3-TTS/qwen_tts/core/models/modeling_qwen3_tts.py` (原始 PyTorch 模型)  
> - `scripts/infer.py` (AXEngine Python 替换推理)  
> - `src/runner/LLM_cp_tts_insert.inc` / `src/runner/LLM.cpp` (现有 C++ 实现参考)

---

## 一、总体架构

```
Qwen3TTSForConditionalGeneration
    └── talker: Qwen3TTSTalkerForConditionalGeneration
            ├── model: Qwen3TTSTalkerModel                ← Talker Transformer 主干
            ├── codec_head: nn.Linear(H, vocab)           ← 预测 next primary codec token
            ├── text_projection: ResizeMLP                ← text embedding → hidden_size
            └── code_predictor: Qwen3TTSTalkerCodePredictorModelForConditionalGeneration
                    ├── model: Qwen3TTSTalkerCodePredictorModel       ← CP Transformer
                    ├── small_to_mtp_projection: Linear/Identity      ← Talker H → CP H
                    └── lm_head: ModuleList[Linear]                   ← 每 sub-code 一个 head
```

**核心原则**：Talker 与 CP 不是独立运行的两个模型，而是**在 Talker 的每个 Decode 步中嵌套调用 CP 生成 sub-codes，再将 sub-codes 的 embedding 拼回 Talker 下一步输入**的耦合结构。

---

## 二、Talker Prefill 阶段

### 2.1 触发条件

**Python 源码** (`modeling_qwen3_tts.py:1665`):
```python
if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
    generation_step = -1
    codec_ids = None
```

### 2.2 输入

| 字段 | 形状 | 说明 |
|------|------|------|
| `inputs_embeds` | `[B, S_prompt, H]` | 由上层 `generate()` 构造，包含 text_projection(embed) + codec_embed + speaker_embed + ICL embed 的拼接结果 |
| `attention_mask` | `[B, S_prompt]` | padding mask |
| `position_ids` | `[3, B, S_prompt]` | 3D RoPE index (temporal/height/width)，纯文本时三行相同 |

**输入构造逻辑** (`modeling_qwen3_tts.py:2068-2233`):
上层 `Qwen3TTSForConditionalGeneration.generate()` 负责把 prompt 的各部分编码为 embedding 并拼接：

```python
# 伪代码：上层构造 talker_input_embeds 的关键步骤
# 1. 角色前缀：text_projection(text_embed(input_id[:, :3]))
# 2. Codec 前缀：codec_think_id + think_bos + language_id + think_eos + speaker_embed + codec_pad + codec_bos
# 3. 文本内容：text_projection(text_embed(input_id[:, 3:-5]))
# 4. ICL (可选)：generate_icl_prompt() 返回的 ref_text + ref_code 混合 embed
# 5. Trailing text hidden：后续每一步 decode 时按 step 索引叠加的文本信息
```

### 2.3 内部处理

```python
# modeling_qwen3_tts.py:1713-1724
outputs: BaseModelOutputWithPast = self.model(
    input_ids=None,
    attention_mask=attention_mask,
    position_ids=position_ids,
    past_key_values=past_key_values,   # None (Prefill)
    inputs_embeds=inputs_embeds,
    use_cache=True,
    ...
)
hidden_states = outputs.last_hidden_state        # [B, S_prompt, H]
logits = self.codec_head(hidden_states)          # [B, S_prompt, vocab]
```

### 2.4 输出

```python
# modeling_qwen3_tts.py:1734-1744
return Qwen3TTSTalkerOutputWithPast(
    logits=logits,                                 # [B, S_prompt, vocab]
    past_key_values=outputs.past_key_values,       # DynamicCache
    past_hidden=hidden_states[:, -1:, :],          # [B, 1, H] ← 下一步 CP 的输入之一
    generation_step=generation_step + 1,           # 0
    trailing_text_hidden=trailing_text_hidden,     # [B, T, H]
    tts_pad_embed=tts_pad_embed,                   # [B, 1, H]
)
```

### 2.5 C++ 侧对应实现

C++ 侧在 `RunTtsWithCpCallback` 中完成等效 Prefill：
- 输入 `talker_input_embeds` 由外部构造后直接传入。
- 输出 `all_prefill_hidden`（全部 token 的 raw hidden）和 `embed`（末帧 raw hidden）。
- `past_hidden` 取 `embed` 经 RMSNorm 后的结果（或直接取 raw hidden，取决于 CP 输入约定）。

---

## 三、Talker Decode 阶段（核心耦合逻辑）

### 3.1 触发条件

**Python 源码** (`modeling_qwen3_tts.py:1669`):
```python
else:  # Generate stage (seq_len == 1)
    last_id_hidden = self.get_input_embeddings()(input_ids)
    ...
```

### 3.2 输入

| 字段 | 形状 | 说明 |
|------|------|------|
| `input_ids` | `[B, 1]` | **上一个 Talker `codec_head` 采样出的 primary codec token** |
| `past_key_values` | `Cache` | Talker KV Cache（DynamicCache 或 StaticTalkerState） |
| `past_hidden` | `[B, 1, H]` | **上一步 Talker 最后一帧 hidden state**，作为 CP 的 condition |
| `generation_step` | `int` | 当前 Talker 生成步数（0-based） |
| `trailing_text_hidden` | `[B, T, H]` | 待合成的文本 hidden，按 `generation_step` 索引叠加到输入 |
| `tts_pad_embed` | `[B, 1, H]` | `tts_pad_token_id` 经 `text_projection` 后的向量，用于越界填充 |

### 3.3 完整耦合流程（逐步拆解）

#### Step A：将上一个 primary codec token 转为 embedding

**源码** (`modeling_qwen3_tts.py:1670`):
```python
last_id_hidden = self.get_input_embeddings()(input_ids)   # [B, 1, H]
```
- `get_input_embeddings()` 返回 `self.codec_embedding`，即 `nn.Embedding(vocab_size, H)`。
- 在 `infer.py` 替换后，等价于 `axengine_talker_model.codec_embedding(input_ids)`。

#### Step B：拼接 Talker last_hidden + last_id_hidden，调用 CP

**源码** (`modeling_qwen3_tts.py:1671-1680`):
```python
predictor_result = self.code_predictor.generate(
    inputs_embeds=torch.cat((past_hidden, last_id_hidden), dim=1),  # [B, 2, H]
    max_new_tokens=self.config.num_code_groups - 1,                # e.g. 15
    do_sample=subtalker_dosample,
    top_p=subtalker_top_p,
    top_k=subtalker_top_k,
    temperature=subtalker_temperature,
    output_hidden_states=True,
    return_dict_in_generate=True,
)
```

**关键约定**：
- CP 的 `inputs_embeds` 长度固定为 **2**：
  - Token 0: `past_hidden`（Talker 上一步末帧 hidden state）
  - Token 1: `last_id_hidden`（刚预测的 primary codec token 的 embedding）
- CP 通过 `GenerationMixin.generate()` 自回归生成 `num_code_groups - 1` 个 sub-code tokens。

#### Step C：CP 内部处理（Prefill + Decode）

**CP Prefill（第 0 个 sub-code）** (`modeling_qwen3_tts.py:1277-1299`):
```python
# CP.forward()
if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
    generation_steps = inputs_embeds.shape[1] - 2   # = 0 for [B,2,H]

# 投影：Talker hidden_size → CP hidden_size
inputs_embeds = self.small_to_mtp_projection(inputs_embeds)   # [B, 2, cp_H]

# CP Transformer
outputs = self.model(inputs_embeds=inputs_embeds, past_key_values=None, ...)
hidden_states = outputs.last_hidden_state                      # [B, 2, cp_H]

# 取对应 step 的 lm_head
logits = self.lm_head[generation_steps](hidden_states)         # [B, 2, vocab]
```

**CP Decode（第 1~N 个 sub-code）** (`modeling_qwen3_tts.py:1281-1299`):
```python
else:  # generation stage (seq_len == 1)
    # 用前一步生成的 token 查对应 code_group 的 embedding table
    inputs_embeds = self.model.get_input_embeddings()[generation_steps - 1](input_ids)

inputs_embeds = self.small_to_mtp_projection(inputs_embeds)
outputs = self.model(inputs_embeds=inputs_embeds, past_key_values=past_key_values, ...)
hidden_states = outputs.last_hidden_state
logits = self.lm_head[generation_steps](hidden_states)
```

**`_update_model_kwargs_for_generation`** (`modeling_qwen3_tts.py:1314-1319`):
```python
model_kwargs["generation_steps"] = outputs.generation_steps   # generation_steps + 1
```
确保每生成一个 sub-code，`generation_steps` 递增，从而切换到下一个 `lm_head` 和 `codec_embedding`。

#### Step D：将 CP 输出的 sub-codes 拼回 Talker 输入

**源码** (`modeling_qwen3_tts.py:1681-1692`):
```python
# 1) 组合完整 codec_ids：primary + sub-codes
codec_ids = torch.cat((input_ids, predictor_result.sequences), dim=-1)   # [B, num_code_groups]

# 2) 拼接所有 codebook 的 embedding
#    - last_id_hidden: primary code embed (Talker 的 codec_embedding)
#    - predictor_result.sequences[..., i]: CP 生成的第 i 个 sub-code
#    - cp_embed_i: CP 的第 i 个 codec_embedding table
codec_hiddens = torch.cat(
    [last_id_hidden] +
    [self.code_predictor.get_input_embeddings()[i](predictor_result.sequences[..., i:i+1])
     for i in range(self.config.num_code_groups - 1)],
    dim=1,
)   # [B, num_code_groups, H]

# 3) 多码本 embedding 求和，得到 Talker 下一步的单个输入 token
inputs_embeds = codec_hiddens.sum(1, keepdim=True)   # [B, 1, H]
```

**关键约定**：多个 codebook 的 embedding 是**逐元素相加**（`sum(1)`），不是拼接。这要求所有 codebook 的 embedding 维度与 Talker hidden_size 一致。

#### Step E：叠加 trailing_text_hidden（文本信息注入）

**源码** (`modeling_qwen3_tts.py:1689-1692`):
```python
if generation_step < trailing_text_hidden.shape[1]:
    inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step].unsqueeze(1)
else:
    inputs_embeds = inputs_embeds + tts_pad_embed
```

- 如果当前 step 仍在文本长度范围内，叠加对应位置的文本 hidden。
- 否则叠加 `tts_pad_embed`（避免模型看到随机值）。

#### Step F：Talker Model 单步 Decode

**源码** (`modeling_qwen3_tts.py:1713-1727`):
```python
outputs = self.model(
    input_ids=None,
    inputs_embeds=inputs_embeds,           # [B, 1, H]
    past_key_values=past_key_values,       # KV Cache
    attention_mask=attention_mask,
    position_ids=position_ids,
    ...
)
hidden_states = outputs.last_hidden_state    # [B, 1, H]
logits = self.codec_head(hidden_states)      # [B, 1, vocab_size]
```

### 3.4 Talker Decode 输出

```python
return Qwen3TTSTalkerOutputWithPast(
    logits=logits,                              # [B, 1, vocab_size]
    past_key_values=outputs.past_key_values,
    past_hidden=hidden_states[:, -1:, :],       # [B, 1, H] → 下一步 CP 的 condition
    generation_step=generation_step + 1,
    trailing_text_hidden=trailing_text_hidden,
    tts_pad_embed=tts_pad_embed,
)
```

### 3.5 上层 `_update_model_kwargs_for_generation`

**源码** (`modeling_qwen3_tts.py:1802-1810`):
```python
def _update_model_kwargs_for_generation(self, outputs, model_kwargs, ...):
    model_kwargs = super()._update_model_kwargs_for_generation(...)
    model_kwargs["past_hidden"] = outputs.past_hidden
    model_kwargs["generation_step"] = outputs.generation_step
    model_kwargs["trailing_text_hidden"] = outputs.trailing_text_hidden
    model_kwargs["tts_pad_embed"] = outputs.tts_pad_embed
    return model_kwargs
```

这些字段被 HuggingFace `GenerationMixin` 保留并传入下一步 `forward()`，形成自回归循环。

---

## 四、CP 推理阶段接口规范

### 4.1 CP 输入

| 字段 | 形状 | 说明 |
|------|------|------|
| `inputs_embeds` | `[B, 2, H_talker]` | 仅 Prefill 时传入。Token 0 = `past_hidden` (Talker 末帧 hidden)，Token 1 = `last_id_hidden` (primary codec embed) |
| `input_ids` | `[B, 1]` | Decode 时传入，上一步生成的 sub-code token id |
| `generation_steps` | `int` | 由 `model_kwargs` 维护。Prefill 时自动计算为 `inputs_embeds.shape[1] - 2` (=0)，此后每步 +1 |
| `past_key_values` | `Cache` | CP 自身的 KV Cache |

### 4.2 CP 内部数据流

```
inputs_embeds [B,2,H_talker]
    │
    ▼
small_to_mtp_projection ──► [B, 2, H_cp]
    │
    ▼
CP Model (Transformer) ──► hidden_states [B, 2, H_cp], past_key_values
    │
    ▼
lm_head[generation_steps] ──► logits [B, 2, vocab]
    │
    ▼
采样 (greedy/sample) ──► next_token_id [B, 1]
```

### 4.3 CP 输出

```python
return Qwen3TTSTalkerCodePredictorOutputWithPast(
    logits=logits,                           # [B, seq_len, vocab]
    past_key_values=outputs.past_key_values,
    hidden_states=outputs.hidden_states,
    generation_steps=generation_steps + 1,   # 下一步使用下一个 lm_head
)
```

### 4.4 CP 的 `forward_finetune`（训练时全景视角）

**源码** (`modeling_qwen3_tts.py:1197-1247`):
```python
def forward_finetune(self, inputs_embeds, labels):
    inputs_embeds = self.small_to_mtp_projection(inputs_embeds)
    outputs = self.model(inputs_embeds=inputs_embeds, ...)
    hidden_states = outputs.last_hidden_state

    # 一次性预测所有 sub-codes（训练用）
    logits = []
    for i in range(1, self.config.num_code_groups):
        logits.append(self.lm_head[i-1](hidden_states[:, i]))
    logits = torch.stack(logits, dim=1)
    ...
```

**注意**：训练时 `forward_finetune` 不走自回归，而是对每个位置用对应的 `lm_head` 一次性出所有 sub-code logits。但**推理时只走 `forward` + `GenerationMixin.generate()`**。

---

## 五、infer.py 替换后的等效接口

`scripts/infer.py` 把原始 PyTorch 模型替换为 AXEngine 静态推理模块后，必须**精确复现上述耦合数据流**。

### 5.1 Talker Model 替换 (`_AxEngineQwen3TTSTalkerModel`)

**源码位置**: `infer.py:639-870`

```python
class _AxEngineQwen3TTSTalkerModel(torch_module.nn.Module):
    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None, ...):
        # Prefill: past_key_values is None and seq_len > 1
        if is_prefill:
            self._static_state, raw_hidden = self.runner.prefill(
                active_embeds, prefill_indices, valid_len
            )
            past = self._static_state
        else:
            # Decode: 复用 StaticTalkerState
            state = past_key_values if isinstance(past_key_values, StaticTalkerState) else self._static_state
            raw_hidden = self.runner.decode_one(
                inputs_embeds[:, -1:, :], state, position_index=position_index
            )
            past = state

        hidden_states = self._to_hidden_tensor(raw_hidden, device, dtype)   # RMSNorm
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past if use_cache else None,
            ...
        )
```

**关键约束**：
- `batch_size` 强制为 1。
- Prefill 时返回完整序列的 `last_hidden_state`。
- Decode 时返回 `[1, 1, H]`，并更新 `StaticTalkerState`。
- `_last_raw_hidden` 被缓存，供 `codec_head` 读取。

### 5.2 Talker Codec Head 替换 (`_AxEngineQwen3TTSTalkerPostHead`)

**源码位置**: `infer.py:873-886`

```python
class _AxEngineQwen3TTSTalkerPostHead(torch_module.nn.Module):
    def forward(self, hidden_states):
        # 隐式状态传递：不读取传入的 hidden_states，而是读取 talker model 缓存的 raw_hidden
        raw_hidden = self.axengine_talker_model._last_raw_hidden
        logits = self.axengine_talker_model.runner.run_post_logits(raw_hidden)
        return torch_module.from_numpy(logits).to(device=hidden_states.device, dtype=torch.float32)
```

### 5.3 CP 替换 (`_AxEngineQwen3TTSTalkerCodePredictorModelForConditionalGeneration`)

**源码位置**: `infer.py:1246-1435`

```python
class _AxEngineQwen3TTSTalkerCodePredictorModelForConditionalGeneration(torch_module.nn.Module):
    def generate(self, inputs_embeds=None, max_new_tokens=None, ...):
        # 注意：small_to_mtp_projection 已合并到 CP layer0 axmodel，此处跳过
        projected_inputs = inputs_embeds

        ids = self.runner.generate_from_inputs_embeds(
            inputs_embeds=projected_inputs,
            embedding_tables=self.codec_embedding,      # ModuleList[Embedding]
            projection_module=None,                      # 已合并
            torch_module=torch_module,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature,
        )
        return torch_module.tensor([ids], device=inputs_embeds.device, dtype=torch_module.long)
```

**CP 内部生成逻辑** (`infer.py:1127-1243`):
```python
def generate_from_inputs_embeds(self, inputs_embeds, embedding_tables, projection_module, ...):
    # 1. Prefill: 把 inputs_embeds [1, valid_len, H] padding 到 prefill_len
    #    逐层 AxEngineSession.run(shape_group=1)
    #    保存 KV cache

    # 2. Decode 循环 (offset = 0 .. num_to_generate-1)
    for offset in range(num_to_generate):
        lm_step = start_lm_step + offset
        if offset > 0:
            # 用上一个生成的 id 查 embedding table
            prev_tensor = torch_module.tensor([[prev_id]], device=device)
            decode_embed = embedding_tables[lm_step - 1](prev_tensor)
            if projection_module is not None:
                decode_embed = projection_module(decode_embed)
            last_hidden_raw = self._decode_one(decode_embed, k_caches, v_caches, current_len)
            current_len += 1

        # Post Norm + LM Head
        hidden_norm = self._run_post_norm(last_hidden_raw)
        logits = self._run_lm_head_logits(lm_step, hidden_norm)
        next_id = _select_next_code_from_logits(logits, do_sample, top_k, top_p, temperature)
        generated_ids.append(next_id)
```

---

## 六、C++ 实现关键检查清单

基于上述分析，C++ 侧实现时必须确保以下数据流和接口完全一致：

### 6.1 Talker 侧

| # | 检查项 | 依据源码 | 说明 |
|---|--------|----------|------|
| 1 | Prefill 输入支持 `[B, S, H]` 且 `S > 1` | `modeling_qwen3_tts.py:1665` | 首次调用必须是 prefill |
| 2 | Decode 输入支持 `inputs_embeds[:, -1:, :]` 或 `input_ids` | `modeling_qwen3_tts.py:1669` | 单 token 输入 |
| 3 | 输出必须携带 `past_hidden`（末帧 raw/normed hidden） | `modeling_qwen3_tts.py:1740` | 供下一步 CP 使用 |
| 4 | `past_key_values` 必须可复用（KV Cache） | `modeling_qwen3_tts.py:1737` | Talker 自回归关键 |
| 5 | `codec_head` 必须能读取最近一次 forward 的 raw hidden | `infer.py:880` | 隐式状态传递 |
| 6 | `generation_step` 必须自增并回传 | `modeling_qwen3_tts.py:1741` | 供 trailing_text_hidden 索引 |

### 6.2 CP 侧

| # | 检查项 | 依据源码 | 说明 |
|---|--------|----------|------|
| 1 | Prefill 输入长度固定为 2（past_hidden + last_id_hidden） | `modeling_qwen3_tts.py:1672` | CP condition + primary embed |
| 2 | `small_to_mtp_projection` 必须在 CP 输入端执行 | `modeling_qwen3_tts.py:1282` | Talker H → CP H |
| 3 | `generation_steps` 必须从 0 开始，每 sub-code +1 | `modeling_qwen3_tts.py:1278` | 用于选择 lm_head 和 embedding table |
| 4 | Decode 时必须用 `generation_steps - 1` 索引 embedding table | `modeling_qwen3_tts.py:1281` | CP 有多个 codec_embedding |
| 5 | `lm_head` 必须用 `generation_steps` 索引 | `modeling_qwen3_tts.py:1299` | 每 sub-code 一个 head |
| 6 | 输出 `generation_steps` 必须 = 输入 + 1 | `modeling_qwen3_tts.py:1311` | 供 `GenerationMixin` 传递 |

### 6.3 耦合侧（Talker ↔ CP）

| # | 检查项 | 依据源码 | 说明 |
|---|--------|----------|------|
| 1 | Talker 每 Decode 一步必须调用一次 CP.generate | `modeling_qwen3_tts.py:1671` | 核心耦合频率 |
| 2 | CP 输入 = `cat(past_hidden, last_id_hidden)` | `modeling_qwen3_tts.py:1672` | dim=1，长度=2 |
| 3 | Talker 下一步输入 = 多码本 embedding 求和 + trailing_text | `modeling_qwen3_tts.py:1682-1692` | `sum(1, keepdim=True)` |
| 4 | `trailing_text_hidden` 按 `generation_step` 索引 | `modeling_qwen3_tts.py:1689` | 越界用 `tts_pad_embed` |
| 5 | `last_id_hidden` 来自 Talker 的 `get_input_embeddings()` | `modeling_qwen3_tts.py:1670` | 不是 CP 的 embedding |
| 6 | Sub-code embeds 来自 CP 的 `get_input_embeddings()[i]` | `modeling_qwen3_tts.py:1684` | CP 的 ModuleList[Embedding] |

---

## 七、端到端伪代码（C++ 视角）

```cpp
// ========== Talker Prefill ==========
// 输入: talker_input_embeds [B, S_prompt, H]
// 输出: talker_kv_cache, all_prefill_hidden [B, S_prompt, H], past_hidden [B, 1, H]
TalkerPrefill(talker_input_embeds, attention_mask);
past_hidden = all_prefill_hidden[:, -1:, :];   // 或取末帧 raw hidden

// ========== Talker Decode Loop ==========
for (int step = 0; step < max_new_tokens; ++step) {
    // 1. Talker codec_head 出 logits，采样得 primary_token_id
    logits = TalkerPost(past_hidden);            // [B, 1, vocab]
    primary_token_id = Sample(logits);           // [B, 1]

    // 2. primary_token_id → embedding
    last_id_hidden = TalkerCodecEmbedding(primary_token_id);   // [B, 1, H]

    // 3. 拼接 CP 输入
    cp_input = Concat(past_hidden, last_id_hidden, dim=1);     // [B, 2, H]

    // 4. CP 生成 sub-codes
    //    CP 内部：prefill(cp_input) → decode N-1 步
    cp_sequences = CPGenerate(cp_input, max_new_tokens=num_code_groups - 1);  // [B, N]

    // 5. 构造 Talker 下一步输入
    //    5a. 多码本 embedding 求和
    codec_hiddens = {last_id_hidden};            // primary
    for (int i = 0; i < num_code_groups - 1; ++i) {
        sub_embed = CpCodecEmbedding(i)(cp_sequences[:, i:i+1]);  // [B, 1, H]
        codec_hiddens.push_back(sub_embed);
    }
    inputs_embeds = Sum(codec_hiddens, dim=1);   // [B, 1, H]

    //    5b. 叠加 trailing text
    if (step < trailing_text_hidden.shape[1]) {
        inputs_embeds += trailing_text_hidden[:, step, :].unsqueeze(1);
    } else {
        inputs_embeds += tts_pad_embed;
    }

    // 6. Talker 单步 Decode
    outputs = TalkerDecode(inputs_embeds, talker_kv_cache);
    past_hidden = outputs.last_hidden_state;     // [B, 1, H]
    talker_kv_cache = outputs.past_key_values;
}

// ========== CP Generate 内部 ==========
CPGenerate(cp_input, max_new_tokens) {
    // cp_input: [B, 2, H], 已包含 past_hidden + last_id_hidden
    // CP Prefill (step 0)
    cp_input_proj = SmallToMtpProjection(cp_input);   // [B, 2, H_cp]
    cp_outputs = CPModelPrefill(cp_input_proj);       // hidden [B, 2, H_cp]
    logits_0 = CPLmHead(0)(cp_outputs.hidden);        // [B, 2, vocab]
    token_0 = Argmax(logits_0[:, -1, :]);
    tokens = {token_0};

    // CP Decode (step 1 .. N-1)
    for (int s = 1; s < max_new_tokens; ++s) {
        embed = CpCodecEmbedding(s - 1)(tokens.back());   // [B, 1, H]
        embed_proj = SmallToMtpProjection(embed);         // [B, 1, H_cp]
        cp_outputs = CPModelDecode(embed_proj, cp_kv_cache);
        logits_s = CPLmHead(s)(cp_outputs.hidden);        // [B, 1, vocab]
        token_s = Argmax(logits_s);
        tokens.push_back(token_s);
    }
    return Stack(tokens, dim=1);   // [B, N]
}
```

---

## 八、代码索引速查

### 8.1 Qwen3-TTS 原始模型 (`modeling_qwen3_tts.py`)

| 逻辑 | 行号 |
|------|------|
| `Qwen3TTSForConditionalGeneration.generate()` (上层入口) | `2022` |
| `generate_icl_prompt()` (ICL prompt 构造) | `1968` |
| `Qwen3TTSTalkerForConditionalGeneration.forward()` (Talker 核心) | `1636` |
| Talker Prefill 分支 | `1665` |
| Talker Decode 分支 + CP 调用 | `1669-1680` |
| Sub-code embed 拼接 + trailing text 叠加 | `1681-1692` |
| Talker 输出封装 | `1734-1744` |
| `_update_model_kwargs_for_generation` | `1802` |
| `Qwen3TTSTalkerCodePredictorModelForConditionalGeneration.forward()` | `1249` |
| CP Prefill (`generation_steps = len - 2`) | `1277` |
| CP Decode (`embed[step-1](input_ids)`) | `1281` |
| CP `small_to_mtp_projection` | `1282` |
| CP `lm_head[generation_steps]` | `1299` |
| `_update_model_kwargs_for_generation` (CP) | `1314` |
| `forward_sub_talker_finetune` (训练全景) | `1197` |
| `Qwen3TTSTalkerModel.forward()` (Talker Transformer) | `1457` |
| `Qwen3TTSTalkerCodePredictorModel.forward()` (CP Transformer) | `1044` |

### 8.2 infer.py AXEngine 替换实现

| 逻辑 | 行号 |
|------|------|
| `replace_talker_model()` | `889` |
| `StaticTalkerLayerRunner.__init__()` | `451` |
| `StaticTalkerLayerRunner.prefill()` | `518` |
| `StaticTalkerLayerRunner.decode_one()` | `588` |
| `StaticTalkerLayerRunner.run_post_logits()` | `628` |
| `_build_axengine_talker_module()` | `639` |
| Talker `forward()` (Prefill/Decode 路由) | `777` |
| `_build_axengine_talker_post_head()` | `873` |
| `replace_code_predictor_model()` | `1438` |
| `StaticCodePredictorRunner.__init__()` | `971` |
| `StaticCodePredictorRunner.generate_from_inputs_embeds()` | `1127` |
| `StaticCodePredictorRunner._decode_one()` | `1091` |
| `_build_axengine_code_predictor_module()` | `1246` |
| CP `generate()` | `1392` |
| `_select_next_code_from_logits()` (采样) | `936` |

---

## 九、修订记录

| 日期 | 版本 | 说明 |
|------|------|------|
| 2026-05-18 | v1.0 | 初始版本，基于 `modeling_qwen3_tts.py` 和 `infer.py` 源码分析，梳理 Talker-CP 耦合逻辑、输入输出、数据流及 C++ 实现检查清单。 |
