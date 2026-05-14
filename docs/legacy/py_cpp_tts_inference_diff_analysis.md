# Python vs C++ TTS 推理流程差异分析

> 基于 `scripts/infer.py` 与 `tools/qwen3_tts_infer.cpp` (+ `src/runner/LLM.cpp` / `LLM_cp_tts_insert.inc`)

---

## 一、Python (infer.py) 推理流程

### 1.1 Talker (StaticTalkerLayerRunner)

#### Prefill
- **输入**: `prefill_embed` [1, valid_len, H]、`_position_ids_to_static_indices` 生成的 indices、valid_len
- **Indices 构造**: 若 `position_ids` 为 None，生成 `np.arange(valid_len)` 并 **repeat 3 次** → shape `(3, prefill_len)`；否则从 `position_ids` 切片（支持 3D/4D mRoPE，若 shape[0]==4 则取 `pos[1:]`）
- **Mask 构造**: 2D causal mask，`mask[:, row, :row+1] = 0`，其余为 `-65536`
- **Chunk 化**: 支持多 chunk prefill（shape_group = chunk_idx + 1）。第 0 个 chunk 的 K/V cache 输入为零，后续 chunk 使用之前累积的 KV cache
- **层间传递**: 每层的 output hidden 作为下一层 input
- **输出**: `StaticTalkerState`（含每层 K/V cache）+ `raw_hidden` [1, valid_len, H]

#### Decode
- **输入**: `decode_embed` [1, 1, H]、`state`、`position_index`
- **Indices**: `[[position_index]]`（scalar）
- **Mask**: 使用预计算的 `decode_mask_cache[state.current_len]`（一维 causal mask，最后元素为 0）
- **KV 更新**: 输出 K/V 写入 `state.k_caches[layer][:, state.current_len, :]`
- **输出**: `raw_hidden` [1, 1, H]

#### Post / Logits
- 取 `raw_hidden[:, -1:, :]` 输入 talker_post axmodel
- Python 在 `forward` 返回前对 `raw_hidden` 调用 **`self.norm(hidden)`**（PyTorch RMSNorm），得到 `hidden_states` 返回给上层；但 **post logits 仍使用未经 norm 的 `raw_hidden`**

---

### 1.2 Code Predictor (StaticCodePredictorRunner)

#### Prefill
- **输入**: `inputs_embeds` [1, valid_len, H]（来自 Talker 的 **normed** `hidden_states`）
- **Indices**: `np.arange(prefill_len)`，无效位置填 0 → shape `(1, prefill_len)`
- **Mask**: 2D causal mask，同 Talker prefill
- **执行**: 一次性跑完所有 CP 层（shape_group=1），K/V cache 输出保存到 `k_caches[:, :copy_len, :]`
- **输出**: `last_hidden_raw` = `data[:, valid_len-1:valid_len, :]`

#### Decode / Generate
- **`start_lm_step = max(0, valid_len - 2)`**
- `num_to_generate = min(max_new_tokens, num_sub_codes - start_lm_step)`
- **Step 0** (offset=0): 直接使用 prefill 的 `last_hidden_raw`，过 post_norm + lm_head(start_lm_step) → logits → sample
- **Step >0** (offset>0):
  - 用前一个生成的 token ID，通过 `embedding_tables[lm_step - 1]` 查表得到 `decode_embed`
  - 调用 `_decode_one()`：**使用 decode group (shape_group=0)**，indices=`[[current_len]]`，mask 为预计算 decode mask
  - KV 输出写入 `k_caches[:, current_len, :]`
  - 过 post_norm + lm_head(lm_step) → logits → sample
- **采样**: 自定义 `_select_next_code_from_logits`，支持 `do_sample` / `top_k` / `top_p` / `temperature`

---

## 二、C++ (qwen3_tts_infer.cpp + LLM) 推理流程

### 2.1 Talker (`RunTtsWithCpCallback`)

#### Prefill
- 读取 `prefill_embeds.bin`（bf16 raw [S, H]），直接作为 talker layer 输入
- **Indices**: 仅填充 **单行** `history_len + i`（1D），**未处理 mRoPE 多行结构**
- **Mask**: `build_prefill_mask`（2D causal mask）
- 分 chunk 执行，保存所有 hidden 到 `all_prefill_hidden`（**未经 norm 的 raw hidden**）
- 最后一个 token 的 hidden 保存到 `embed`，输入 talker_post 得到 primary_code

#### Decode
- 使用 LLM 通用 decode 循环（`decode_grpid`）
- **Indices**: 单元素 `indices`（current_len）
- 得到 next_token (primary_code)，循环执行直到 EOS 或 `max_new_tokens`

#### Post / Logits
- Talker post 直接消费 `embed`（raw hidden）
- **未在 talker 输出后应用 RMSNorm**；norm 被延迟到 CP 输入前手动做

---

### 2.2 Code Predictor (`RunCpFrame`)

- 输入: `last_hidden_bf16` + `primary_code`
- 构建 context: `[last_hidden, primary_embed]`
- 对 j=0..14 循环生成 sub-codes：
  - **j=0**: seq_len=2, history_len=0, 输入为 `[last_hidden, primary_embed]`
  - **j>0**: seq_len=1, history_len=j+1, 输入为最后一个 residual embed
  - **全部使用 `cp_prefill_gid`**（即总是用 prefill group，而非 decode group）
  - KV cache: j==0 时清零；j>0 时把上一步 dump 出来的 full KV cache 整体拷贝回输入
  - 取 last hidden → cp_post（若存在）→ `cp_lm_heads[j]` → **greedy argmax**（`postprocess.apply(logits, {})`，history 为空）
  - residual embed 通过 `cp_embed_tables[j]` 查表，累加到 `codec_sum_bf16`

---

## 三、核心差异对比

| 维度 | Python (infer.py) | C++ (LLM.cpp) | 影响程度 |
|------|-------------------|---------------|----------|
| **Talker Prefill Indices** | 3 行 repeat 的 position ids (mRoPE)，支持 4D→3D slice | 仅单行顺序索引 | **高** |
| **Talker Decode Indices** | 支持从 `position_ids` / `cache_position` 提取 | 简单 scalar current_len | **中** |
| **Talker Norm** | `forward` 返回前对 raw hidden 做 PyTorch RMSNorm；CP 输入为 **normed hidden** | Talker 不输出 normed hidden；CP 输入前手动做 `rmsnorm_bf16` | **中** |
| **CP Prefill/Decode Group** | 明确区分：prefill 用 shape_group=1，decode 用 shape_group=0 | **全部使用 cp_prefill_gid**，无 decode group 概念 | **高** |
| **CP Decode KV Cache** | decode 时仅写入 `[:, current_len, :]`，mask 用预计算 decode mask | 每次把整段 KV cache 来回拷贝，mask 用 `build_prefill_mask` | **高** |
| **CP start_lm_step** | `max(0, valid_len - 2)`，决定从哪个 lm_head/embedding table 开始 | 固定从 j=0 开始（单 frame 场景下等价于 valid_len=2） | **低**（单 frame 下一致） |
| **CP 采样** | 完整 `do_sample` / `top_k` / `top_p` / `temperature` | `postprocess.apply(logits, {})`，实际为 greedy + 可能带 temperature（取决于 post_config.json） | **中** |
| **next_embed 构造 (streaming)** | Python generate 循环由 transformers 处理，decode 输入为 `codec_sum + txt_hidden[step]` | C++ 手动构造 `next_embed = codec_sum + txt_hidden`（streaming）或 `codec_sum + tts_pad_vec`（non-streaming） | **中** |
| **CP 输入 normed/raw** | CP 接收 **normed** hidden（PyTorch RMSNorm） | CP 接收 raw hidden + 手动 `rmsnorm_bf16` | **中** |

---

## 四、可对齐 vs 无法完全对齐

### 4.1 可以且应该对齐的项（重写目标）

1. **Talker Prefill Indices 的多行填充**
   - C++ `RunTtsWithCpCallback` 中只填充了 indices 的第一行。若模型输入 tensor `nSize` 对应多行（如 `3 * prefill_token_num * sizeof(uint32)`），需要按 Python 逻辑填充 3 行相同的递增 position。
   - **可行**：可读取 indices tensor 的 shape，计算 `idx_rows = nSize / prefill_token_num / sizeof(uint32)`，然后逐行填充。

2. **CP 区分 Prefill / Decode Group**
   - 需要在 `InitCp` 中检测 decode gid（indices 元素数为 1 的 group），`j=0` 用 prefill_gid，`j>0` 改用 decode_gid。
   - **可行**：仿照 LLM `init_groups_from_model` 的逻辑，为 CP 也解析出 decode_gids。

3. **CP Decode 的 KV Cache 更新方式**
   - 使用 decode group 后，KV cache 更新应与 Python 一致：只写 `current_len` 位置，而非整体拷贝。
   - **可行**：decode 输出 K/V 的形状是 `[1, 1, kv_dim]`，直接拷贝到 KV cache 的 `current_len` 偏移处。

4. **CP 的 Mask 构建**
   - prefill 保持 `build_prefill_mask`；decode 应使用预计算的 **decode mask**（同 Python 的 `decode_mask_cache`）。
   - **可行**：仿照 Python `_build_decode_mask_cache` 预计算 decode mask buffer。

5. **CP 采样参数透传**
   - Python 支持 temperature/top_k/top_p/do_sample；C++ 当前在 `RunCpFrame` 中只调用 `postprocess.apply(logits, {})`，且 history 为空，导致 repetition_penalty 也不生效。
   - **可行**：将 `LLMPostprocess` 的采样配置（或单独实现 `_select_next_code_from_logits`）应用到 CP 的 logits 上。

6. **Talker / CP 的 Norm 时机**
   - Python 中 talker 返回 normed hidden，CP 直接使用。C++ 中 talker 不 norm，CP 前手动 norm。虽然数学上等价，但为了逐层 hidden 对比一致，应在 C++ talker prefill 后也对 `all_prefill_hidden` 做 RMSNorm（与 PyTorch 实现对齐 eps=1e-6）。
   - **可行**：在 `RunTtsWithCpCallback` prefill 结束后，对 `all_prefill_hidden` 逐 token 应用 `rmsnorm_bf16`。

### 4.2 可能无法完全对齐的项（需要确认）

1. **mRoPE Position IDs 的精确值**
   - Python 中的 `position_ids` 由 transformers / Qwen3-TTS 模型内部根据 `attention_mask` 和 `cache_position` 动态计算，可能包含 4D 结构（如文本、音频等不同模态的独立 position）。C++ 若不重新实现完整的 position_ids 计算逻辑，只能用"顺序递增"近似。
   - **当前方案**: C++ 已采用 Python dump 的 `prefill_embeds.bin`，跳过了 tokenizer/embedding 预处理；但 indices 仍可能需要与 dump 时的 `position_ids` 对齐。由于 `prefill_embeds.bin` 只是 embedding，不包含 indices，C++ 只能根据模型输入 shape 推断 indices 维度。
   - **结论**: 若模型编译时 indices 固定为 3 行且值相同，可完全对齐；若需要真正的 4D mRoPE slice，则 C++ 无法在无额外 dump 数据的情况下完全对齐。

2. **PyTorch RMSNorm vs C++ 手写 `rmsnorm_bf16` 的数值精度**
   - 两者均使用 eps=1e-6，但 bf16↔fp32 的转换顺序、累加精度可能存在微小差异（<1e-4）。这在多层传递后可能导致 logits 级别的差异。
   - **结论**: 流程可以对齐，但无法保证 bit-wise 一致。属于可接受的浮点误差范围。

3. **Postprocess / 采样逻辑的完全等价**
   - Python 的 `_select_next_code_from_logits` 是显式 numpy 实现；C++ 的 `LLMPostprocess` 内部逻辑可能不完全一致（如 top_p 的阈值处理、random 种子机制）。
   - **结论**: 若用户要求 greedy 对齐，可关闭 `do_sample` 保证一致；若要求 sample 对齐，需要额外确认 `LLMPostprocess` 的实现细节，或重写一个与 Python 一致的 C++ 采样函数。

4. **Talker Decode 的完整循环结构**
   - Python 中 talker decode 由 `transformers` 库的 `generate()` 驱动，包含完整的 `past_key_values` 管理、attention_mask 更新、cache_position 维护等。C++ 的 `RunTts` 是手写循环，虽然逻辑等价，但边界条件（如 EOS 判断、max_new_tokens 计数）可能不完全一致。
   - **结论**: 核心计算可以对齐，但框架级行为（如 callback、token 计数方式）无法 1:1 复刻 transformers。

5. **Code Predictor 的 `small_to_mtp_projection`**
   - Python 注释说明该 projection 权重已合并到 CP layer0 axmodel，因此跳过。C++ 也直接传入 last_hidden，两者一致。
   - **结论**: 已对齐，无需修改。

---

## 五、重写计划建议

基于以上分析，建议按以下顺序重写 C++ 代码：

1. **Talker Prefill Indices**: 修改 `RunTtsWithCpCallback` 的 prefill 循环，检测 indices tensor 的 `nSize`，按 `idx_rows` 多行填充。
2. **Talker Norm**: prefill 结束后，对 `all_prefill_hidden` 的每个 token 应用 RMSNorm（使用已加载的 `cp_norm_gamma` 或单独加载 talker norm weight），使 CP 输入与 Python 一致。
3. **CP Group 检测**: 在 `InitCp` 中增加 decode gid 检测（indices 元素数为 1 的 group）。
4. **CP Decode 重构**:
   - `j=0` 继续使用 prefill_gid（seq_len=2）
   - `j>0` 切换到 decode_gid（seq_len=1）
   - 使用预计算的 decode_mask_cache
   - KV cache 按 `current_len` 位置更新，而非整体拷贝
5. **CP 采样**: 在 `RunCpFrame` 中支持完整的采样参数（temperature/top_k/top_p/do_sample），或至少在 CLI 参数与 `LLMPostprocess` 之间正确透传。
6. **非流式/流式 next_embed**: 确认 `txt_pos = trailing_start + step` 的取值与 Python 一致（已读取 meta.json 的 `trailing_start`）。

---

## 六、需要用户确认的问题

1. **Talker indices 的多行结构**: 你的 talker axmodel 的 `indices` 输入在 prefill group 下是多大？（例如 `nSize = 3 * prefill_token_num * 4` 字节？）C++ 是否需要严格复现 Python 的 3 行 repeat 逻辑？
2. **CP 采样**: 当前 C++ CP 使用 greedy argmax。你是否需要完整支持 `do_sample/top_k/top_p` 与 Python 一致？如果是，建议重写 C++ 采样逻辑以精确匹配 Python numpy 实现。
3. **精度容忍度**: RMSNorm 和 bf16/fp32 转换存在微小误差，逐层 cosine 可能无法达到 1.000。你期望的对齐标准是什么（如 cosine > 0.999 / greedy token 一致）？
4. **Talker decode 的 position/index**: C++ talker decode 当前使用简单递增 `indices`。你是否需要像 Python 一样支持从外部传入或计算复杂的 `cache_position`？
