# Qwen3-TTS Python 与 C++ 推理逻辑详细对比

> 基于 `scripts/infer.py`（Python）与 `src/runner/LLM_cp_tts_insert.inc`（C++）的代码级逐模块对比。
> 目的：确认 Talker 和 Code Predictor 的 AXModel 推理逻辑是否完全一致。

---

## 一、总体结论

| 模块 | 状态 | 说明 |
|------|------|------|
| **Talker Prefill** | ✅ 一致 | Mask、indices、KV cache、layer 计算逻辑对齐 |
| **Talker Decode** | ✅ 一致（已修复） | P0 Bug 已修复：mask 现已标记 prefill 位置为可见（`mask[0..input_embed_num-1]=0`） |
| **Talker Post / Logits** | ✅ 一致 | 输入都是 raw hidden，输出 logits |
| **Talker 采样（Greedy）** | ✅ 一致 | Argmax 逻辑相同 |
| **Talker 采样（Sample）** | ✅ 一致（已修复） | Repetition penalty 公式已对齐（`< 0 ? * penalty : / penalty`，作用于全部历史）；Top-K + Top-P 组合策略已对齐（顺序：top_k → softmax → top_p） |
| **CP 输入构造** | ✅ 一致（已修复） | 第 0 帧取 prefill last normed hidden，第 1+ 帧取 decode 后 rmsnorm |
| **CP Prefill (j=0)** | ✅ 一致（已修复） | 输入仅 `last_hidden`（1 token），history_len=0 |
| **CP Decode (j>0)** | ✅ 一致（已修复） | seq_len=1, history_len=j，KV cache 传完整 buffer |
| **CP Post Norm** | ✅ 一致 | 都取 `output_norm`，取最后一个 token |
| **CP LM Head** | ✅ 一致 | 输入 hidden_norm，输出 logits |
| **CP 采样（Greedy）** | ✅ 一致 | Argmax 逻辑相同 |
| **CP 采样（Sample）** | ✅ 一致（已修复） | CLI 参数 `--cp_temperature`/`--cp_top_k`/`--cp_top_p` 透传到 `RunCpFrame`，不再硬编码 |
| **Next Embed（Non-streaming）** | ⚠️ 微小差异 | C++ bf16 逐步截断累加 codec_sum vs Python torch.sum（可能 fp32 中间值），有微小精度差异 |
| **Next Embed（Streaming）** | ⚠️ 待确认 | C++ 使用 `all_prefill_hidden[trailing_start+step]`，Python 逻辑由原始模型控制 |

---

## 二、Talker 部分详细对比

### 2.1 Prefill

| 步骤 | Python (`StaticTalkerLayerRunner.prefill`) | C++ (`RunTtsWithCpCallback` prefill 循环) | 是否一致 |
|------|-------------------------------------------|------------------------------------------|----------|
| **输入 embed** | `active_embeds[:, :valid_len, :]` 填入 `data[:, :valid_len, :]`，其余补 0 | `embed_tmp` 初始化为 0，`memcpy` 当前 chunk 的 tokens | ✅ |
| **Indices** | `_position_ids_to_static_indices` 生成 `[3, padded_len]`，3 行重复 `history_len+i`，**padding 值 = 1**（`np.ones` 初始化） | `idx_rows = idx_elems / prefill_token_num`，多行填充 `history_len+i`，**padding 值 = 0**（`memset(idx_ptr, 0)` 初始化） | ⚠️ padding 值不同（1 vs 0），但 padding 位置被 causal mask 遮蔽，不影响推理结果 |
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
| **Mask** | 预计算 `decode_mask_cache[state.current_len]`：`mask[0..t-1]=0`（包括所有 prefill 位置），`mask[t]=-65536`，`mask[last]=0` | `mask` 数组：`mask[0..input_embed_num-1]=0`（**已修复**），`mask[decode_start..t-1]=0`，`mask[last]=0` | ✅ **已修复** |
| **KV Cache 更新** | `state.k_caches[layer][:, current_len, :] = K_cache_out.reshape(1, kv_dim)` | `memcpy(in_k_ptr + indices * kv_cache_size, out_k.pVirAddr, kv_cache_size * 2)` | ✅ |
| **层间传递** | NumPy 数组赋值 | `memcpy(embed.data(), t_out.pVirAddr, ...)` | ✅ |
| **输出** | `data_decode` (raw hidden) | `embed` (raw hidden) | ✅ |

#### 关于 Decode Mask 的修复（P0 Bug）

**已修复**：在 `LLM_cp_tts_insert.inc` 的 `RunTtsWithCpCallback` 中，mask 初始化段后添加了：
```cpp
for (int i = 0; i < input_embed_num && i < (int)mask.size(); i++) mask[(size_t)i] = 0;
```

对照通用 `LLM.cpp` 的 `Run` 方法（line 1170）：
```cpp
for (int i = 0; i < precompute_len + input_embed_num && i < (int)mask.size(); i++) mask[(size_t)i] = 0;
```
TTS 场景下 `precompute_len=0`，所以等价于 `mask[0..input_embed_num-1] = 0`。

### 2.4 Talker Post → Primary Code

| 步骤 | Python (`run_post_logits`) | C++ (`llama_post`) | 是否一致 |
|------|---------------------------|--------------------|----------|
| **输入** | `raw_hidden[:, -1:, :]` (bf16) | `embed` (bf16, last token raw hidden) | ✅ |
| **Post axmodel** | `post_session.run({"input": hidden_token})` | `llama_post.inference()` | ✅ |
| **输出** | logits `[1, 1, vocab_size]` (float32) | `post_out` (bf16 raw → float32 via `proc << 16`) | ✅ |

### 2.5 Talker 采样

| 步骤 | Python (`_select_next_code_from_logits`) | C++ (`LLMPostprocess::sample_from_logits`) | 是否一致 |
|------|-----------------------------------------|------------------------------------------|----------|
| **Greedy** | `np.argmax(scores)` | `std::max_element` | ✅ |
| **Temperature** | `scores / temp` | `logit /= temperature` | ✅ |
| **Top-K** | `np.partition(scores, -top_k)[-top_k]`，低于阈值设 `-1e30`，然后 softmax + top-p | `partial_sort` 取 top_k，低于阈值设 `-1e9f`，然后 softmax + top-p | ✅ **已对齐** |
| **Top-P** | 对 full logits softmax，按 prob 排序，cumsum > top_p 的 drop，重新 normalize | `sort` + `cumsum` + cut + renormalize | ✅ **已对齐** |
| **Top-K + Top-P** | 先 top-k 过滤，再 top-p，最后采样 | 先 top-k 过滤，再 softmax，再 top-p，最后采样 | ✅ **已对齐**（均为顺序组合） |
| **Repetition Penalty** | `logit < 0 ? logit * penalty : logit / penalty`（作用于**全部历史**） | `logit >= 0 ? logit / penalty : logit * penalty`（作用于**全部历史**） | ✅ **已对齐**（公式等价：`< 0 → * penalty`，`>= 0 → / penalty`） |
| **随机数** | `np.random.choice` (受 `np.random.seed` 控制) | `std::mt19937` (受 `set_seed` 控制) | ❌ 生成器不同，sample 结果不同 |

**关键结论**：在 **Greedy 模式**（`do_sample=False` 或 `temperature=0`）下，Talker 采样**完全一致**。在 **Sample 模式**下，采样策略（repetition penalty / top_k / top_p 组合）**已对齐**，仅随机数生成器不同。

**修复说明**：
- 编号1（repetition penalty）：移除了使用 `sqrt(penalty)` + `penalty_window=20` 的死代码变体，确保唯一路径 `sample_from_logits` 使用与 Python 一致的公式（`< 0 → * penalty`，`>= 0 → / penalty`，作用于全部历史）。
- 编号2（Top-K/Top-P 组合）：`sample_from_logits` 已支持 top_k + top_p 顺序组合（top_k → softmax → top_p），与 Python 策略一致。`set_top_p_sampling` **不会**关闭 `enable_top_k_sampling`。

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
| **参数来源** | CLI 透传（`--subtalker_temperature`/`--subtalker_top_k`/`--subtalker_top_p`/`--subtalker_dosample`） | CLI 透传（`--cp_temperature`/`--cp_top_k`/`--cp_top_p`），通过 `LLMAttrType` 传入 `RunCpFrame` | ✅ **已对齐** |

**关键结论**：CP 在 greedy 模式下完全一致。在 sample 模式下，CP 采样参数**已从 CLI 透传**（编号3已修复），仅随机数生成器不同。

**修复说明**：
- 编号3（CP 采样参数硬编码）：`RunCpFrame` 中的 `cp_temperature=0.9f, cp_top_k=50, cp_top_p=1.0f` 硬编码已替换为从 `LLMAttrType` 读取（`_attr.cp_temperature`/`_attr.cp_top_k`/`_attr.cp_top_p`）。`qwen3_tts_infer.cpp` 新增 `--cp_temperature`、`--cp_top_k`、`--cp_top_p` CLI 参数，通过 `LLMAttrType` 透传到内部 `RunCpFrame`。默认值与 Python `--subtalker_temperature`/`--subtalker_top_k`/`--subtalker_top_p` 默认值一致（0.9/50/1.0）。

---

## 四、Next Embed 构造（Talker 的下一步输入）

| 模式 | Python | C++ | 是否一致 |
|------|--------|-----|----------|
| **Non-streaming** | `next_embed = codec_sum + tts_pad_embed` | `next_embed[d] = bf16(codec_sum[d] + tts_pad_vec[d])` — **但 codec_sum 构造方式不同**：Python `torch.sum(codec_hiddens)` 可能使用 fp32 中间值累加，C++ 逐个 bf16→fp32 累加并截断回 bf16 | ⚠️ 微小精度差异 |
| **Streaming** | 由 Qwen3TTSModel 内部控制，通常为 `codec_sum + txt_hidden[step]` | `codec_sum + all_prefill_hidden[trailing_start + step]` | ⚠️ **待确认** |

---

## 五、仍存在的差异清单

| 编号 | 差异点 | 影响 | 说明 |
|------|--------|------|------|
| 4 | **Prefill Indices Padding**：Python padding 值=1（`np.ones`），C++ padding 值=0（`memset(0)`） | padding 位置被 causal mask 遮蔽，不影响推理结果 | 无需修复，但需注意 |
| 5 | **codec_sum bf16 累加精度**：C++ 逐步 bf16→fp32 累加截断回 bf16，Python `torch.sum` 可能 fp32 中间值 | next_embed 有微小精度差异（<1e-4），对 greedy argmax 通常无影响 | 如需完全对齐，可改为 fp32 累加后截断 |
| 6 | **Streaming `txt_hidden` 来源**：C++ 从 `all_prefill_hidden[txt_pos]` 取，Python 逻辑由原始模型控制 | Streaming 模式下 talker decode 输入可能不同 | 建议用 non-streaming 模式做 greedy 对齐验证 |
| 7 | **数值精度**：PyTorch RMSNorm vs C++ `rmsnorm_bf16` 的 bf16↔fp32 截断顺序 | hidden state 有 <1e-4 误差，logits 可能有微小偏差 | 通常 cosine > 0.999 即认为对齐；greedy 下 argmax 通常不受影响 |
| 8 | **随机数生成器**：Python `np.random.choice` vs C++ `std::mt19937` | Sample 模式下采样结果不同；Greedy 模式无影响 | 不同随机数生成器，给定相同 seed 也无法产生完全相同的采样序列 |

---

## 六、已修复的差异清单

| 编号 | 差异点 | 修复方案 | 修复文件 |
|------|--------|----------|----------|
| **P0** | **Talker Decode Mask 缺少 prefill 可见标记** | 在 `RunTtsWithCpCallback` mask 初始化后添加 `for (int i = 0; i < input_embed_num && i < (int)mask.size(); i++) mask[(size_t)i] = 0;` | `src/runner/LLM_cp_tts_insert.inc:513` |
| 1 | **Talker Repetition Penalty 公式/窗口不一致** | 移除使用 `sqrt(penalty)` + `penalty_window=20` 的死代码变体；确保唯一路径 `sample_from_logits` 使用 `buf[id] >= 0 ? buf[id] / penalty : buf[id] * penalty`（与 Python `< 0 ? * penalty : / penalty` 等价），作用于全部历史 | `src/runner/LLMPostprocess.hpp` |
| 2 | **Talker Top-K/Top-P 组合策略不一致** | `sample_from_logits` 已支持 top_k + top_p 顺序组合（top_k → softmax → top_p），`set_top_p_sampling` 不会关闭 `enable_top_k_sampling` | 无需代码修改（已有逻辑对齐） |
| 3 | **CP 采样参数硬编码** | `RunCpFrame` 中硬编码 `cp_temperature=0.9f, cp_top_k=50, cp_top_p=1.0f` 替换为从 `_attr.cp_temperature`/`_attr.cp_top_k`/`_attr.cp_top_p` 读取；`qwen3_tts_infer.cpp` 新增 `--cp_temperature`/`--cp_top_k`/`--cp_top_p` CLI 参数；`LLMAttrType` 新增对应字段 | `src/runner/LLM_cp_tts_insert.inc`, `src/runner/LLM.hpp`, `tools/qwen3_tts_infer.cpp` |

---

## 七、Greedy 对齐验证步骤（推荐）

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
    --cp_temperature 0.0 \
    --streaming false
```

然后对比：
1. `debug_bin/output_codes.bin`（Python） vs `debug_bin/output_codes.bin`（C++）
2. 或使用 `scripts/compare_talker_prefill.py` 对比 intermediate tensors

如果 greedy 下仍不一致，请提供**第 0 帧的 primary_code 和前 5 个 residual_code** 的对比，可继续定位。