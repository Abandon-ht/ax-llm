# Qwen3-TTS Python 与 C++ 推理逻辑详细对比

> 基于 `scripts/infer.py`（Python）与 `src/runner/LLM_cp_tts_insert.inc`（C++，已修复后）的代码级逐模块对比。
> 目的：确认 Talker 和 Code Predictor 的 AXModel 推理逻辑是否完全一致。

---

## 一、总体结论

| 模块 | 状态 | 说明 |
|------|------|------|
| **Talker Prefill** | ✅ 一致 | Mask、indices、KV cache、layer 计算逻辑对齐 |
| **Talker Decode** | ✅ 一致 | Mask 更新、KV cache 写入、层间数据流对齐 |
| **Talker Post / Logits** | ✅ 一致 | 输入都是 raw hidden，输出 logits |
| **Talker 采样（Greedy）** | ✅ 一致 | Argmax 逻辑相同 |
| **Talker 采样（Sample）** | ⚠️ 有差异 | Repetition penalty 公式、top_k/top_p 组合策略、随机数生成器不同 |
| **CP 输入构造** | ✅ 一致（已修复） | 第 0 帧取 prefill last normed hidden，第 1+ 帧取 decode 后 rmsnorm |
| **CP Prefill (j=0)** | ✅ 一致（已修复） | 输入仅 `last_hidden`（1 token），history_len=0 |
| **CP Decode (j>0)** | ✅ 一致（已修复） | seq_len=1, history_len=j，KV cache 传完整 buffer |
| **CP Post Norm** | ✅ 一致 | 都取 `output_norm`，取最后一个 token |
| **CP LM Head** | ✅ 一致 | 输入 hidden_norm，输出 logits |
| **CP 采样（Greedy）** | ✅ 一致 | Argmax 逻辑相同 |
| **CP 采样（Sample）** | ⚠️ 有差异 | C++ 采样参数硬编码（0.9/50/1.0），随机数生成器不同 |
| **Next Embed（Non-streaming）** | ✅ 一致 | `codec_sum + tts_pad_vec` |
| **Next Embed（Streaming）** | ⚠️ 待确认 | C++ 使用 `all_prefill_hidden[trailing_start+step]`，Python 逻辑由原始模型控制 |

---

## 二、Talker 部分详细对比

### 2.1 Prefill

| 步骤 | Python (`StaticTalkerLayerRunner.prefill`) | C++ (`RunTtsWithCpCallback` prefill 循环) | 是否一致 |
|------|-------------------------------------------|------------------------------------------|----------|
| **输入 embed** | `active_embeds[:, :valid_len, :]` 填入 `data[:, :valid_len, :]`，其余补 0 | `embed_tmp` 初始化为 0，`memcpy` 当前 chunk 的 tokens | ✅ |
| **Indices** | `_position_ids_to_static_indices` 生成 `[3, padded_len]`，3 行重复 `history_len+i` | `idx_rows = idx_elems / prefill_token_num`，多行填充 `history_len+i` | ✅ |
| **Mask** | `[1, padded_len, padded_len]` causal：`-65536`，`[:, row, :row+1] = 0` | `build_prefill_mask`：`row[j < history_len] = 0; row[kv_cache_num .. kv_cache_num+r] = 0` | ✅ |
| **KV Cache 初始值** | `np.zeros((1, kv_cache_len, kv_dim))` | Device buffer 分配时清零（runtime 保证） | ✅ |
| **Layer 计算顺序** | 外层 layer → 内层 chunk | 外层 chunk → 内层 layer | ⚠️ 仅当 `prefill_split_num > 1` 时有差异。但 Qwen3-TTS 的 `input_embed_num=85 <= 128`，`prefill_split_num=1`，**实际等价** |
| **Chunk 间传递** | `np.concatenate(layer_outputs, axis=1)` | `embed_tmp` 被 `llm_d2h` 回写，下一层 `llm_h2d` 读入 | ✅ 数值等价 |
| **输出** | `data[:, :valid_len, :]` (raw hidden) | `all_prefill_hidden` + `embed` (last token raw hidden) | ✅ |

### 2.2 Prefill 后 RMSNorm

| 步骤 | Python | C++ | 是否一致 |
|------|--------|-----|----------|
| **时机** | 每次 `forward()` 返回前，通过 `_to_hidden_tensor(raw_hidden)` 调用 `self.norm(hidden)` | Prefill 结束后，对 `all_prefill_hidden` **所有 token** 调用 `rmsnorm_bf16`；Decode 后单独对 `embed` 做 rmsnorm | ✅ |
| **输入** | `raw_hidden` (last token 或 full seq) | `all_prefill_hidden[t]` 逐 token | ✅ |
| **Gamma** | PyTorch `RMSNorm` 内置 weight | `cp_norm_gamma` (从 `talker.model.norm.weight.bfloat16.bin` 加载的 fp32 数组) | ⚠️ 数值上可能有 `<1e-4` 的 bf16↔fp32 截断差异，cosine 通常 >0.999 |

### 2.3 Decode

| 步骤 | Python (`decode_one`) | C++ (decode 循环) | 是否一致 |
|------|----------------------|-------------------|----------|
| **输入 embed** | `decode_embed[:, -1:, :]` (bf16 numpy) | `next_embed` (bf16 vector) | ✅ |
| **Indices** | `np.array([[position_index]], dtype=np.uint32)`，默认 `state.current_len` | `unsigned int indices = decode_start + step` | ✅ |
| **Mask** | 预计算 `decode_mask_cache[state.current_len]`：`[kv_cache_len+1, 1, 1, kv_cache_len+1]`，允许看到 `0..t-1` 和 `last` | `mask` 数组：`mask[0..t-1]=0`（由前序步骤设置），`mask[last]=0`，当前位置仍为 `-65536` | ✅ |
| **KV Cache 更新** | `state.k_caches[layer][:, current_len, :] = K_cache_out.reshape(1, kv_dim)` | `memcpy(in_k_ptr + indices * kv_cache_size, out_k.pVirAddr, kv_cache_size * 2)` | ✅ |
| **层间传递** | NumPy 数组赋值 | `memcpy(embed.data(), t_out.pVirAddr, ...)` | ✅ |
| **输出** | `data_decode` (raw hidden) | `embed` (raw hidden) | ✅ |

#### 关于 Decode Mask 的详细说明

Python 的 `_build_decode_mask_cache`：
```python
row_ids = np.arange(kv_cache_len + 1, dtype=np.int32)[:, None]
col_ids = np.arange(kv_cache_len + 1, dtype=np.int32)[None, :]
mask_2d = np.where(col_ids < row_ids, 0.0, -65536.0).astype(np.float32)
mask_2d[:, -1] = 0.0
```

当 decode step `t` 时，取 `decode_mask_cache[t]`：
- `col < t`（历史）：**0**（可见）
- `col == t`（自己）：**-65536**（不可见）— 因为自己的 KV 还未写入 cache
- `col == last`：**0**（特殊处理）
- `col > t` 且不是 last：**-65536**（不可见）

C++ 的 `mask` 数组在 step `t` 时：
- `mask[0..t-1] = 0`（之前步骤设置）
- `mask[t] = -65536`（当前 step，还未写入 KV cache）
- `mask[last] = 0`

两者**完全一致**。Decode 时当前 token 的 query 不应该看到自己，因为自己的 K/V 是在当前 layer inference **结束后**才被写入 KV cache 的。这是标准的 decode causal mask。

### 2.4 Talker Post → Primary Code

| 步骤 | Python (`run_post_logits`) | C++ (`llama_post`) | 是否一致 |
|------|---------------------------|--------------------|----------|
| **输入** | `raw_hidden[:, -1:, :]` (bf16) | `embed` (bf16, last token raw hidden) | ✅ |
| **Post axmodel** | `post_session.run({"input": hidden_token})` | `llama_post.inference()` | ✅ |
| **输出** | logits `[1, 1, vocab_size]` (float32) | `post_out` (bf16 raw → float32 via `proc << 16`) | ✅ |

### 2.5 Talker 采样

| 步骤 | Python (`_select_next_code_from_logits`) | C++ (`LLMPostprocess::apply`) | 是否一致 |
|------|-----------------------------------------|------------------------------|----------|
| **Greedy** | `np.argmax(scores)` | `std::max_element` | ✅ |
| **Temperature** | `scores / temp` | `logit /= temperature` | ✅ |
| **Top-K** | `np.partition(scores, -top_k)[-top_k]`，低于阈值的设 `-1e30`，然后 softmax + top-p | `partial_sort` 取 top_k，对 top_k logits 单独 softmax 采样，**不经过 top-p** | ❌ **不一致** |
| **Top-P** | 对 full logits softmax，按 prob 排序，cumsum > top_p 的 drop，重新 normalize | `faster_top_p_sampling`：max heap 按 prob 排序，cumsum >= top_p 时 break，重新 normalize | ✅ 基本一致 |
| **Top-K + Top-P** | 先 top-k 过滤，再 top-p，最后采样 | **互斥**：`set_top_p_sampling` 会关闭 `enable_top_k_sampling` | ❌ **不一致** |
| **Repetition Penalty** | `logit < 0 ? logit * penalty : logit / penalty`（作用于**全部历史**） | `logit < 0 ? logit * sqrt(penalty) : logit / sqrt(penalty)`（作用于**最近 20 个 token**，`penalty_window=20`） | ❌ **不一致** |
| **随机数** | `np.random.choice` (受 `np.random.seed` 控制) | `std::mt19937` (受 `set_seed` 控制) | ❌ 生成器不同，sample 结果不同 |

**关键结论**：在 **Greedy 模式**（`do_sample=False` 或 `temperature=0`）下，Talker 采样**完全一致**。在 Sample 模式下，由于 penalty 公式、top_k/top_p 组合策略、随机数生成器的差异，结果**可能不同**。建议对齐验证时使用 greedy。

---

## 三、Code Predictor 部分详细对比

### 3.1 CP 输入构造（每帧）

| 步骤 | Python | C++（已修复） | 是否一致 |
|------|--------|---------------|----------|
| **第 0 帧来源** | Talker prefill 输出的 `hidden_states[:, -1:, :]`（已 RMSNorm） | `all_prefill_hidden[input_embed_num - 1]`（已 RMSNorm） | ✅ |
| **第 1+ 帧来源** | Talker decode 输出的 `hidden_states[:, -1:, :]`（已 RMSNorm） | 对 `embed`（decode raw hidden）调用 `rmsnorm_bf16` | ✅ |
| **是否包含 primary_embed** | ❌ **不包含** | ❌ **不包含**（已移除） | ✅ |

### 3.2 CP Prefill (j=0，每帧内)

| 步骤 | Python (`generate_from_inputs_embeds` prefill 段) | C++ (`RunCpFrame` j=0) | 是否一致 |
|------|---------------------------------------------------|------------------------|----------|
| **输入序列** | `inputs_embeds` = `[last_hidden]`，`valid_len=1` | `cp_ctx[0:D]` = `last_hidden_bf16` | ✅ |
| **seq_len** | 隐式 1（`data[:, :valid_len, :] = inputs_embeds`，`valid_len=1`） | `seq_len = 1` | ✅ |
| **history_len** | 0 | `history_len = j = 0` | ✅ |
| **Indices** | `indices[:, 0] = 0`，其余为 0 | `idx_tmp[0] = 0` | ✅ |
| **Mask** | `[1, prefill_len, prefill_len]`，row 0 的前 1 个为 0 | `build_prefill_mask(mask_tmp, 0, cp_prefill_token_num, 0, 1)` → `row[0]=0` | ✅ |
| **KV Cache 输入** | 完整的 `k_caches[layer]`（`[1, kv_cache_len, kv_dim]`） | 完整的 `cp_k_cache[m].data()`（已修复） | ✅ |
| **KV Cache 更新** | `k_caches[layer][:, :copy_len, :] = k_prefill[:, :1, :]` | `cp_k_cache[m][0:1*kv_dim] = K_cache_out` | ✅ |
| **Layer 输出** | `data[:, 0:1, :]` | `embed_tmp[0:D]` | ✅ |

### 3.3 CP Decode (j=1..14，每帧内)

| 步骤 | Python (`_decode_one`) | C++ (`RunCpFrame` j>0) | 是否一致 |
|------|------------------------|------------------------|----------|
| **输入** | `embedding_tables[lm_step-1](prev_code)` 的 embed | `cp_ctx` 最后一个 res_embed（上一步生成的 code 的 embedding） | ✅ |
| **seq_len** | 1 | `seq_len = 1` | ✅ |
| **history_len** | `current_len`（从 `valid_len=1` 开始递增） | `history_len = j`（j=1 时为 1） | ✅ |
| **current_len** | `history_len + 1` | `current_len = j + 1` | ✅ |
| **Indices** | `np.array([[current_len]], dtype=np.uint32)` | `idx_tmp[0] = history_len`（即 j） | ✅ |
| **Mask** | `decode_mask_cache[current_len]`：允许看到 `0..t-1` 和 `last` | `mask_tmp[i] = (i <= history_len) ? 0 : -65536` | ✅ |
| **KV Cache 输入** | 完整的 `k_caches[layer]` | 完整的 `cp_k_cache[m].data()`（已修复） | ✅ |
| **KV Cache 更新** | `k_caches[layer][:, current_len, :] = K_cache_out` | `cp_k_cache[m][(current_len-1)*kv_dim] = K_cache_out` | ✅ |

### 3.4 CP Post Norm

| 步骤 | Python (`_run_post_norm`) | C++ (`RunCpFrame` post) | 是否一致 |
|------|--------------------------|-------------------------|----------|
| **输入** | `hidden_token` (bf16) | `hidden_step` (bf16) | ✅ |
| **Post axmodel** | `post_session.run({"input": hidden_token})` | `cp_post.inference()` | ✅ |
| **输出 key** | `output_norm` 或 `output` | `output_norm` | ✅ |
| **输出处理** | `out[:, -1:, :]` (取最后一个位置) | `memcpy(hidden_step, post_buf, copy_words * 2)` | ✅ |

### 3.5 CP LM Head

| 步骤 | Python (`_run_lm_head_logits`) | C++ (`RunCpFrame` lm_head) | 是否一致 |
|------|-------------------------------|----------------------------|----------|
| **输入** | `lm_input[:, -1:, :] = hidden_norm`（其余为 0） | `hidden_step`（直接传入，大小为 D） | ✅ 等价 |
| **LM Head axmodel** | `lm_head_sessions[step].run({"input": lm_input})` | `cp_lm_heads[j].inference()` | ✅ |
| **输出** | `logits[:, -1, :].reshape(-1)` (float32) | `logits_fp32` (float32) | ✅ |

### 3.6 CP 采样

| 步骤 | Python (`_select_next_code_from_logits`) | C++ (`CpSampleFromLogits`) | 是否一致 |
|------|-----------------------------------------|---------------------------|----------|
| **Greedy** | `np.argmax(scores)` | `std::max_element` | ✅ |
| **Temperature** | `scores / temp` | `buf[i] /= temperature` | ✅ |
| **Top-K** | `np.partition(scores, -top_k)[-top_k]`，低于阈值设 `-1e30` | `partial_sort` + `std::greater`，低于阈值设 `-1e9f` | ✅ 基本一致 |
| **Top-P** | `argsort` + `cumsum` + drop | `sort` + `cumsum` + cut | ✅ 基本一致 |
| **随机数** | `np.random.choice` | `std::discrete_distribution` + `std::mt19937` | ❌ 生成器不同 |
| **参数来源** | CLI 透传（temperature/top_k/top_p/do_sample） | **硬编码** `cp_temperature=0.9f, cp_top_k=50, cp_top_p=1.0f` | ❌ **不一致** |

**关键结论**：CP 在 greedy 模式下完全一致。在 sample 模式下，C++ 的 CP 采样参数**未从 CLI 透传**（始终使用 0.9/50/1.0），且随机数生成器不同。

---

## 四、Next Embed 构造（Talker 的下一步输入）

| 模式 | Python | C++ | 是否一致 |
|------|--------|-----|----------|
| **Non-streaming** | `next_embed = codec_sum + tts_pad_embed` | `next_embed[d] = bf16(codec_sum[d] + tts_pad_vec[d])` | ✅（前提是 `tts_pad_vec.bin` 是从 Python 同一模型导出的） |
| **Streaming** | 由 Qwen3TTSModel 内部控制，通常为 `codec_sum + txt_hidden[step]` | `codec_sum + all_prefill_hidden[trailing_start + step]` | ⚠️ **待确认**。C++ 使用 prefill 历史中的 text token hidden 作为 `txt_hidden`。如果 Python 的 streaming 模式也是从 prefill 历史中取对应位置的 hidden，则一致；否则可能不一致。 |

---

## 五、仍未修复 / 无法确认的差异清单

| 编号 | 差异点 | 影响 | 建议 |
|------|--------|------|------|
| 1 | **Talker Sample 模式**：repetition penalty 公式不同（`penalty` vs `sqrt(penalty)`），window 策略不同 | Sample 模式下 primary_code 可能不同 | Greedy 验证时无影响；如需 sample 对齐，需统一 penalty 实现 |
| 2 | **Talker Top-K/Top-P 互斥**：Python 支持组合，C++ 互斥 | Sample 模式下可能不同 | Greedy 时无影响 |
| 3 | **CP 采样参数硬编码**：C++ `RunCpFrame` 中写死 0.9/50/1.0，不从 CLI 透传 | Sample 模式下 residual codes 可能不同 | Greedy 时无影响；如需 sample 对齐，需将 CLI 参数传入 `RunCpFrame` |
| 4 | **Streaming `txt_hidden` 来源**：C++ 从 `all_prefill_hidden[txt_pos]` 取，Python 逻辑由原始模型控制 | Streaming 模式下 talker decode 输入可能不同 | 建议用 non-streaming 模式做 greedy 对齐验证 |
| 5 | **数值精度**：PyTorch RMSNorm vs C++ `rmsnorm_bf16` 的 bf16↔fp32 截断顺序 | hidden state 有 <1e-4 误差，logits 可能有微小偏差 | 通常 cosine > 0.999 即认为对齐；greedy 下 argmax 通常不受影响 |

---

## 六、Greedy 对齐验证步骤（推荐）

要确认两端完全一致，建议关闭所有随机性，用 **greedy + non-streaming** 模式对比：

**Python**：
```bash
python scripts/infer.py \
    --do_sample False \
    --temperature 1.0 \
    --non_streaming_mode \
    --dump_cpp_input_dir ./debug_bin \
    ...
```

**C++**：
```bash
./build/install/bin/qwen3_tts_infer \
    <talker_dir> ./debug_bin \
    --max_new_tokens 128 \
    --temperature 0.0 \
    --streaming false
```

然后对比：
1. `debug_bin/output_codes.bin`（Python） vs `debug_bin/output_codes.bin`（C++）
2. 或使用 `scripts/compare_talker_prefill.py` 对比 intermediate tensors

如果 greedy 下仍不一致，请提供**第 0 帧的 primary_code 和前 5 个 residual_code** 的对比，可继续定位。
