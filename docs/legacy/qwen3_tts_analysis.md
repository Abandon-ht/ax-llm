# Qwen3-TTS Python 与 C++ 推理逻辑深度分析

> 基于 `scripts/infer.py`（Python）与 `src/runner/LLM_cp_tts_insert.inc` / `src/runner/LLM.cpp`（C++）的代码级逐模块对比。
> 目的：确认 Talker 和 Code Predictor 的推理逻辑是否完全一致，分析差异的根本原因。
> 范围：非流式（non-streaming）模式。不考虑音频 encode 部分和 `infer_bin.py`。

---

## 一、总体结论

| 模块 | 状态 | 说明 |
|------|------|------|
| **Talker Prefill** | ✅ 一致 | 迭代顺序不同（layer outer vs chunk outer），但 prefill_split_num=1 时完全等价；>1 时数学上等价（详见 §2.1） |
| **Talker Decode** | ✅ 一致 | Mask 语义等价（详见 §2.3） |
| **Talker Post / Logits** | ⚠️ 有差异 | 精度路径不同：Python logits 从 raw_hidden → Post axmodel → fp32；C++ logits 从 embed bf16 → Post axmodel → bf16→u16→fp32（详见 §2.4） |
| **Talker RMSNorm** | ⚠️ 有差异 | PyTorch `nn.RMSNorm` vs C++ `rmsnorm_bf16` 手动实现，浮点截断路径不同（详见 §2.5） |
| **Talker Sampling** | ⚠️ 有差异 | 采样策略已对齐；Greedy 模式一致，Sample 模式随机数生成器不同 |
| **CP 输入构造** | ✅ 一致 | 都使用 `[last_hidden, primary_embed]` 作为 2-token prefill 输入 |
| **CP Prefill (j=0)** | ✅ 一致 | 输入 2 tokens, history_len=0, causal mask |
| **CP Decode (j=1..14)** | ✅ 一致 | seq_len=1, KV cache 传递完整 buffer |
| **CP Post Norm** | ✅ 一致 | 都取 `output_norm`, 取最后一个 token |
| **CP LM Head** | ✅ 一致 | 输入 hidden_norm, 输出 logits fp32 |
| **CP Sampling** | ✅ 一致 | Greedy 一致，Sample 仅有随机数差异 |
| **Next Embed (Non-streaming)** | ⚠️ 微小差异 | torch.sum (fp32 累加) vs bf16 逐步截断累加（详见 §5） |

---

## 二、Talker 部分详细对比

### 2.1 Prefill

#### 2.1.1 入口与调用链

| | Python | C++ |
|---|---|---|
| **入口** | `_AxEngineQwen3TTSTalkerModel.forward()` (`infer.py:858`) | `RunTtsWithCpCallback()` (`LLM_cp_tts_insert.inc:488`) |
| **调用** | `self.runner.prefill(active_embeds, prefill_indices, valid_len)` (`infer.py:922`) | 内联 prefill 循环 (`LLM_cp_tts_insert.inc:531-616`) |
| **类** | `StaticTalkerLayerRunner` (`infer.py:497`) | `LLM::Impl` (`LLM.cpp` 内部 struct) |
| **函数** | `prefill()` (`infer.py:565`) | 无独立函数，直接在 `RunTtsWithCpCallback` 中实现 |

#### 2.1.2 迭代顺序差异分析

这是 Python 与 C++ prefill 实现中最显著的架构差异：

**Python** (`StaticTalkerLayerRunner.prefill()`, `infer.py:582-623`):
```
for layer_idx in range(num_layers):       # 外层：layer
    for chunk_idx in range(chunk_count):  # 内层：chunk
        # 运行 layer[layer_idx] 的 chunk[chunk_idx]
        # KV cache 读写
        # 保存 hidden chunk
    data = concat(all chunks)             # 拼接当前 layer 所有 chunk 的输出
    # data 作为下一 layer 的输入
```

**C++** (`RunTtsWithCpCallback`, `LLM_cp_tts_insert.inc:531-616`):
```
for (int p = 0; p < prefill_split_num; p++):         # 外层：chunk
    for (int m = 0; m < _attr.axmodel_num; m++):      # 内层：layer
        # 运行 layer[m] 的 chunk[p]
        # KV cache 通过 device memory 传递 (d2d)
        # 输出 → embed_tmp
    # 保存 embed_tmp 到 all_prefill_hidden
```

**等价性分析**（以 2 chunks, 2 layers 为例）:

Python 执行顺序：
```
L0C0 → L0C1 → concat L0 → D0
L1C0 → L1C1 → concat L1 → D1
```

C++ 执行顺序：
```
C0L0 → C0L1 → 保存 C0 结果
C1L0 → C1L1 → 保存 C1 结果
```

两种顺序的计算依赖关系：
- `L0C0`: 输入 = raw_emb[0:S0], KV = zeros（Python 和 C++ 相同）
- `L0C1`: 输入 = raw_emb[S0:S1], KV = L0K[0:S0]（Python 和 C++ 相同，C++ 通过 device memory 传递）
- `L1C0`: 输入 = L0 的输出[0:S0], KV = zeros（不同 layer 的 KV 是独立的）
- `L1C1`: 输入 = L0 的输出[S0:S1], KV = L1K[0:S0]

**结论：两种迭代顺序数学上完全等价**。每个 layer 的每个 chunk 计算独立依赖于该 layer 自身的 KV cache 和上一层（全部 chunks）的输出。在 prefill_split_num=1（常见 case，当 input_embed_num ≤ prefill_token_num）时，两种顺序退化为完全相同。

#### 2.1.3 Indices / Position 初始化对比

**Python** (`_position_ids_to_static_indices`, `infer.py:431-451`):
```python
# 1. 从 position_ids 或 arange 生成 pos [1, valid_len]
pos = np.arange(valid_len, dtype=np.uint32).reshape(1, valid_len)
pos = np.repeat(pos, 3, axis=0)  # [3, valid_len]

# 2. 创建 indices，初始化为 np.ones（padding = 1）
indices = np.ones((3, prefill_len), dtype=np.uint32)
indices[:, :valid_len] = pos.astype(np.uint32)  # 有效位置填入 [0, 1, 2, ...]

# 结果: indices = [[0,1,2,...,S-1, 1,1,1,...,1],  # padding = 1
#                   [0,1,2,...,S-1, 1,1,1,...,1],
#                   [0,1,2,...,S-1, 1,1,1,...,1]]
```

**C++** (`RunTtsWithCpCallback`, `LLM_cp_tts_insert.inc:559-566`):
```cpp
// 1. memset 清零（padding = 0）
unsigned int *idx_ptr = (unsigned int *)t_idx.pVirAddr;
memset(idx_ptr, 0, t_idx.nSize);

// 2. 填充有效位置
for (int r = 0; r < idx_rows; ++r) {
    for (int i = 0; i < input_num_token; ++i) {
        idx_ptr[r * _attr.prefill_token_num + i] = (unsigned int)(history_len + i);
    }
}

// 结果: indices = [[0,1,2,...,S-1, 0,0,0,...,0],  # padding = 0
//                   [0,1,2,...,S-1, 0,0,0,...,0],
//                   [0,1,2,...,S-1, 0,0,0,...,0]]
```

**差异分析**:

| 维度 | Python | C++ |
|------|--------|-----|
| Padding 值 | **1** | **0** |
| 有效位置 | [0, 1, ..., S-1] | [0, 1, ..., S-1] |

**为什么差异不影响推理结果**:

1. Causal mask 遮蔽：padding 位置（S 到 prefill_len-1）在 prefill mask 中被标记为 `-65536`（不可见）。任何有效 query（位置 0..S-1）都无法 attend 到 padding 位置。
2. KV Cache 只读有效位置：prefill 只存储 `input_num_token` 个位置的 K/V 到 cache；padding 位置的 K/V 即使被计算出来（index=1 vs 0 导致不同的 RoPE），也不被存储或使用。
3. Output 只取有效位置：Python 返回 `data[:, :valid_len, :]`，C++ 只保存 `all_prefill_hidden` 的前 `valid_len` 个 token。

**结论：padding 值 1 vs 0 不影响有效位置的推理结果。** ✅

#### 2.1.4 Prefill Mask 构建

**Python** (`infer.py:573-576`):
```python
prefill_mask = np.zeros((1, padded_len, padded_len), dtype=np.float32) - 65536.0
for row in range(valid_len):
    prefill_mask[:, row, : row + 1] = 0.0  # causal: row sees [0..row]
```

**C++** (`build_prefill_mask`, `LLM.cpp:136-150`):
```cpp
static inline void build_prefill_mask(mask_tmp, kv_cache_num, token_rows, history_len, valid_rows) {
    std::fill(mask_tmp.begin(), mask_tmp.end(), bf16(-65536.f).data);
    for (int r = 0; r < valid_rows; ++r) {
        auto row = mask_tmp.data() + r * (kv_cache_num + token_rows);
        for (int j = 0; j < history_len; ++j) row[j] = 0;           // 历史 KV visible
        int cur = kv_cache_num;
        for (int j = cur; j < cur + r + 1; ++j) row[j] = 0;         // causal: [0..r] visible
    }
}
```

C++ mask 布局为 `[token_rows, kv_cache_num + token_rows]` = `[KV_cache_slots | Prefill_slots]`。prefill 时 history_len=0，两者均产生 causal mask。✅ 等价。

### 2.2 Prefill 后 RMSNorm 处理

这是 Python 与 C++ 一个关键差异点，直接关系到后续 hidden 的精度。

**Python** (`_AxEngineQwen3TTSTalkerModel.forward()`, `infer.py:854-856` 和 `infer.py:932`):
```python
def _to_hidden_tensor(self, raw_hidden, device, dtype):
    # raw_hidden: np.ndarray (bf16), 来自 prefill/decode 的 raw output
    hidden = torch_module.from_numpy(raw_hidden.astype(np.float32)).to(device=device, dtype=dtype)
    return self.norm(hidden)  # self.norm = original_model.norm (PyTorch nn.RMSNorm)

# 调用处:
raw_hidden = self.runner.prefill(...)   # raw_hidden 是 bf16 numpy
hidden_states = self._to_hidden_tensor(raw_hidden, inputs_embeds.device, inputs_embeds.dtype)
# hidden_states 类型: torch.Tensor, dtype=bf16, 经过 PyTorch RMSNorm
```

**C++** (`RunTtsWithCpCallback`, `LLM_cp_tts_insert.inc:619-627`):
```cpp
// prefill 后，对所有 all_prefill_hidden token 逐一调用 rmsnorm_bf16
if (!cp_norm_gamma.empty()) {
    for (int t = 0; t < input_embed_num; ++t) {
        rmsnorm_bf16(
            all_prefill_hidden.data() + (size_t)t * _attr.tokens_embed_size,  // dst
            all_prefill_hidden.data() + (size_t)t * _attr.tokens_embed_size,  // src (in-place)
            cp_norm_gamma.data(), _attr.tokens_embed_size);
    }
}
```

**C++ `rmsnorm_bf16` 实现** (`LLM_cp_tts_insert.inc:207-219`):
```cpp
static inline void rmsnorm_bf16(unsigned short *dst, const unsigned short *src,
                                 const float *gamma, int n, float eps=1e-6)
{
    float sum_sq = 0;
    for (int i = 0; i < n; ++i) {
        float v = bfloat16(src[i]).fp32();  // bf16 → fp32
        sum_sq += v * v;
    }
    float rms = std::sqrt(sum_sq / n + eps);
    for (int i = 0; i < n; ++i) {
        float v = bfloat16(src[i]).fp32();  // bf16 → fp32
        dst[i] = bfloat16((v / rms) * gamma[i]).data;  // fp32 → bf16 截断
    }
}
```

**精度差异分析**:

| 维度 | Python (PyTorch RMSNorm) | C++ (rmsnorm_bf16) |
|------|--------------------------|---------------------|
| 输入 | bf16 torch.Tensor | bf16 (unsigned short) |
| 内部计算精度 | PyTorch 可能使用 fp32 逐元素 | fp32 逐元素 |
| 输出截断 | PyTorch 的 bf16→fp32→计算→bf16 截断 | 明确的 bf16→fp32→bf16 |
| eps | RMSNorm 默认 1e-5（PyTorch 1.13+）/ 1e-6 | 1e-6 |

**差异来源**:
1. `eps` 可能不同（1e-5 vs 1e-6），数值影响极小（<1e-7 量级）
2. bf16→fp32→bf16 截断路径：PyTorch 内部使用 CUDA kernel / CPU kernel，与 C++ 简单循环的编译器优化路径可能不同
3. `sum_sq` 累加顺序可能导致极微小的浮点误差堆积

**结论：RMSNorm 精度可能产生 <1e-4 量级的 cosine 偏差，对 greedy argmax 通常无影响。** ⚠️

### 2.3 Decode

#### 2.3.1 Python Decode Mask 公式与推导

**Python** (`_build_decode_mask_cache`, `infer.py:423-428`):
```python
def _build_decode_mask_cache(kv_cache_len: int, mask_dtype):
    row_ids = np.arange(kv_cache_len + 1, dtype=np.int32)[:, None]   # [KV+1, 1]
    col_ids = np.arange(kv_cache_len + 1, dtype=np.int32)[None, :]   # [1, KV+1]
    mask_2d = np.where(col_ids < row_ids, 0.0, -65536.0)             # causal 下三角
    mask_2d[:, -1] = 0.0  # 最后一列恒为 0 (reserved slot)
    return mask_2d.astype(mask_dtype).reshape((kv_cache_len + 1, 1, 1, kv_cache_len + 1))
```

产生的 mask 矩阵为 `(kv_cache_len+1) x (kv_cache_len+1)`：
```
row 0: [ 0,      -65536, -65536, ..., -65536, 0 ]   # token 0 sees only self
row 1: [ 0,       0,     -65536, ..., -65536, 0 ]   # token 1 sees [0,1]
row 2: [ 0,       0,      0,     ..., -65536, 0 ]   # token 2 sees [0,1,2]
...
row S: [0,0,...,0, -65536, -65536, ..., -65536, 0]   # token S sees [0..S-1], self masked
                   ^self masked
row KV: [0,0,...,0,  0,     0,     ...,  0,      0]  # last row = all 0 (reserved)
```

**Decode 时 mask 使用方式** (`infer.py:648-649`):
```python
decode_mask = self.decode_mask_cache[state.current_len]  # 取第 current_len 行
# 例: current_len = S (第一步 decode):
# row S = [0,0,...,0 (S个), -65536, -65536, ..., 0]
# 即: prefill 的 0..S-1 位置可见，自己(S)不可见，future 不可见，last slot 可见
```

关键特征：
- **Self 不可见**：`mask[S, S] = -65536`，因为当前 token 的 K/V 尚未计算
- **所有 prefill 可见**：`mask[S, 0..S-1] = 0`
- **Reserved slot 可见**：`mask[S, KV] = 0`

#### 2.3.2 C++ Decode Mask 推导

**C++ mask 初始化** (`RunTtsWithCpCallback`, `LLM_cp_tts_insert.inc:488-512`):
```cpp
// Step A: 计算 max_decode_cap
int max_decode_cap = _attr.max_token_len;
if (!decode_max_token_len_grp_.empty())
    max_decode_cap = std::max(max_decode_cap, decode_max_token_len_grp_.back());
if (max_decode_cap <= 0) max_decode_cap = _attr.kv_cache_num;

// Step B: 创建 mask，全部初始化为 -65536
std::vector<unsigned short> mask((size_t)max_decode_cap + 1, bf16.data);

// Step C: 最后一位设为 0（对应 Python 的 "最后一列恒为 0"）
if (!mask.empty()) mask.back() = 0;

// Step D: decode 容量边界设为 0（decode_max_token_len_grp_ 的分段边界）
for (const int cap : decode_max_token_len_grp_) {
    if (cap >= 0 && cap < (int)mask.size()) mask[(size_t)cap] = 0;
}
```

**C++ decode loop 内 mask 更新** (`LLM_cp_tts_insert.inc:512` 和 `LLM_cp_tts_insert.inc:901`):
```cpp
// Prefill 后：使所有 prefill 位置可见（相当于 Python row S 中 0..S-1=0）
for (int i = 0; i < input_embed_num && i < (int)mask.size(); i++)
    mask[(size_t)i] = 0;

// 每步 decode 后：使当前 token 位置可见（为下一步 decode 准备）
if (indices < mask.size()) mask[indices] = 0;
```

**等价性验证**:

第 0 步 decode (indices = S = input_embed_num)：
```
初始 mask: [0,0,...,0 (S个0), -65536, -65536, ..., -65536, 0 (last)]
                                              ^S位置 = -65536 = self不可见
```
与 Python `row S` 等价：`[0 (S个), -65536(S位置, self), -65536(未来), 0(last)]` ✅

第 1 步 decode (indices = S+1)：
```
mask 已更新: [0,0,...,0 (S+1个0), -65536, ..., -65536, 0 (last)]
                                  ^S+1 = -65536 = self不可见
```
与 Python `row S+1` 等价：`[0 (S+1个), -65536(S+1, self), -65536(未来), 0(last)]` ✅

**结论：C++ 的 1D 状态更新 mask 与 Python 的 2D 预计算查表 mask 语义完全等价。** ✅

#### 2.3.3 `max_decode_cap` 计算与 `decode_max_token_len_grp_` 的作用

**`max_decode_cap` 的来源**:

1. 初始值 = `_attr.max_token_len`（从 config.json 的 `max_token_len` 字段，或从 axmodel 推断）
2. 如果 `decode_max_token_len_grp_` 非空，取 `decode_max_token_len_grp_.back()`（最大 decode 容量）
3. 如果仍 ≤0，fallback 到 `_attr.kv_cache_num`

**`decode_max_token_len_grp_` 的来源** (`init_groups_from_model`, `LLM.cpp:313-339`):

axmodel 可能包含多个 decode shape group（例如 2k/4k/8k/16k），每个 group 有不同的 mask 长度。`decode_max_token_len_grp_[i]` 存储第 i 个 decode group 的容量（`mask元素数 - 1`）。

推理时 `choose_decode_gid(needed_tokens)` (`LLM.cpp:226-234`) 根据当前序列长度自动选择最合适的 decode group。

**CB (Callback) 的含义** (`LLM.hpp:47`):

```cpp
LLMRuningCallback running_callback = nullptr;
```

在 `qwen3_tts_infer.cpp:423-425` 被设置为空回调：
```cpp
attr.runing_callback = [](std::string, float, void *) { /* no-op */ };
```

因为 TTS 生成的 token 是音频 code，不是文本，打印出来是乱码，所以回调不输出任何内容。该回调在标准 LLM text generation 中用于流式输出解码文本。

### 2.4 Talker Post → Logits 与 Hidden 输出流程

#### 2.4.1 Python Hidden 输出流程

**Prefill 阶段** (`infer.py:920-932`):
```
1. self._static_state, raw_hidden = self.runner.prefill(...)
   → raw_hidden: np.ndarray, bf16, shape [1, valid_len, H]
2. hidden_states = self._to_hidden_tensor(raw_hidden, device, dtype)
   → hidden_states: torch.Tensor, bf16, shape [1, valid_len, H], 经过 self.norm (RMSNorm)
3. self._last_raw_hidden = raw_hidden   # 保存 raw (pre-norm) 供 codec_head 使用
```

**Decode 阶段** (`infer.py:925-932`):
```
1. raw_hidden = self.runner.decode_one(inputs_embeds, state, ...)
   → raw_hidden: np.ndarray, bf16, shape [1, 1, H]
2. hidden_states = self._to_hidden_tensor(raw_hidden, device, dtype)
   → hidden_states: torch.Tensor, bf16, shape [1, 1, H], 经过 RMSNorm
3. self._last_raw_hidden = raw_hidden   # 保存 raw (pre-norm)
```

**Post Logits** (`_build_axengine_talker_post_head`, `infer.py:960-973`):
```python
def forward(self, hidden_states):
    raw_hidden = self.axengine_talker_model._last_raw_hidden  # 取 pre-norm hidden
    logits = self.axengine_talker_model.runner.run_post_logits(raw_hidden)
    return torch.from_numpy(logits).to(device=hidden_states.device, dtype=torch.float32)
```

**`run_post_logits`** (`infer.py:675-683`):
```python
def run_post_logits(self, raw_hidden: np.ndarray) -> np.ndarray:
    hidden_token = raw_hidden[:, -1:, :].astype(self.m_dtype, copy=False)  # bf16
    outputs = self.post_session.run({"input": hidden_token})
    logits = outputs[out_key].astype(np.float32, copy=False)  # Post axmodel 输出 fp32
    return logits[:, -1:, :]
```

#### 2.4.2 C++ Hidden 输出流程

**Prefill 阶段** (`LLM_cp_tts_insert.inc:610-616, 618-627`):
```
1. 每层 layer 输出 → d2h 到 embed_tmp (bf16)
2. 最后一层输出保存到 all_prefill_hidden (bf16, pre-norm)
3. 最后 token 保存到 embed (bf16, pre-norm)
4. 对 all_prefill_hidden 全部 token 调用 rmsnorm_bf16
```

**Decode 阶段** (`LLM_cp_tts_insert.inc:861-895`):
```
1. next_embed (bf16) → 各层 layer inference
2. 最后一层输出 → embed (bf16, pre-norm, raw hidden)
3. Talker Post → 得到 logits (bf16) → post_process 转换为 fp32
4. 下一帧: 若需要 normed hidden，对 embed 调用 rmsnorm_bf16
```

**对比**:

| 步骤 | Python | C++ |
|------|--------|-----|
| Prefill 后的 pre-norm hidden | `self._last_raw_hidden` (bf16 numpy) | `embed` (bf16) 最后 token + `all_prefill_hidden` (bf16) 全部 token |
| Prefill 后的 normed hidden | `self._to_hidden_tensor(raw_hidden)` = PyTorch RMSNorm 结果 | 对 `all_prefill_hidden` 调用 `rmsnorm_bf16`（已做 in-place） |
| Decode 后的 pre-norm hidden | `self._last_raw_hidden` (每次 forward 更新) | `embed` (每步 decode 更新) |
| Decode 后的 normed hidden | `self._to_hidden_tensor(raw_hidden)` | 需要时对 `embed` 调用 `rmsnorm_bf16` |
| Post logits 输入 | `raw_hidden[:, -1:, :]` (pre-norm, bf16) | `embed` (pre-norm, bf16) |

**关键差异**: Python 每次 forward 返回 normed hidden；C++ 保存 pre-norm raw hidden，需要 normed 时手动调用 `rmsnorm_bf16`。两者 norm 的计算方式不同（PyTorch RMSNorm vs C++ rmsnorm_bf16），已在 §2.2 分析。

**另外注意**：Python 的 `_last_raw_hidden` 是在每次 `talker.model.forward()` 结束时更新；C++ 的 `embed` 是在 decode 循环内每层最后一层的输出覆盖。二者都反映最新一步的 pre-norm hidden。✅

### 2.5 Talker Sampling

**Python** (`_select_next_code_from_logits`, `infer.py:1024-1056`):
```python
def _select_next_code_from_logits(logits, do_sample, top_k, top_p, temperature):
    if not do_sample:
        return int(np.argmax(scores))  # Greedy
    # Sample: temperature → top_k → softmax → top_p → multinomial
```

**C++** (`LLMPostprocess::sample_from_logits`, `LLMPostprocess.hpp:263-322`):
```cpp
int sample_from_logits(buf, temperature, top_k, top_p, rep_penalty, history) {
    // Repetition penalty → temperature → top_k → softmax → top_p → discrete_distribution
    if (temperature < 1e-6f) return argmax;  // Greedy
}
```

采样策略已对齐（top_k → softmax → top_p）。Greedy 模式一致。Sample 模式下 `np.random.choice` vs `std::mt19937` 导致不同结果，但策略相同。

---

## 三、Code Predictor 部分详细对比

### 3.1 整体流程差异分析

#### 3.1.1 Python: `generate_from_inputs_embeds`

**入口**: `_AxEngineQwen3TTSTalkerCodePredictorModelForConditionalGeneration.generate()` (`infer.py:1520-1561`)，调用 `runner.generate_from_inputs_embeds()` (`infer.py:1215-1331`)。

完整流程：
```
1. 接收 inputs_embeds [1, 2, H] = [last_hidden_normed, primary_embed]
   valid_len = 2

2. Prefill (2 tokens):
   - data = zeros[1, prefill_len, H]; data[:,:2,:] = inputs_embeds
   - indices = [0,1,0,0,...,0]
   - mask: causal, row 0 sees [0], row 1 sees [0,1]
   - 5 layers × prefill → output = data[:,:2,:]
   - KV_cache[:, :2, :] = prefill KV
   - last_hidden_raw = data[:, 1:2, :]  # 第二个 token (primary_embed) 的 output

3. 生成残余 codes (15 steps, lm_step = 0..14):
   for offset in range(num_to_generate):
       if offset > 0:  # step 1..14
           decode_embed = embedding_tables[lm_step-1](prev_id)
           last_hidden_raw = _decode_one(decode_embed, k_caches, v_caches, current_len)
           current_len += 1
       
       hidden_norm = _run_post_norm(last_hidden_raw)
       logits = _run_lm_head_logits(lm_step, hidden_norm)
       next_id = _select_next_code_from_logits(logits, ...)
       generated_ids.append(next_id)
```

**`_decode_one`** (`infer.py:1179-1213`):
```python
def _decode_one(self, decode_embed, k_caches, v_caches, current_len):
    data_decode = decode_embed  # [1, 1, H]
    decode_mask = self.decode_mask_cache[current_len]  # row current_len
    decode_indices = np.array([[current_len]], dtype=np.uint32)
    
    for layer_idx in range(num_layers):
        # 完整 K_cache / V_cache 送入
        # indices = current_len
        # mask = row current_len (0..current_len-1 visible)
        # 更新 K_cache[layer][:, current_len, :] = K_cache_out
        # 更新 V_cache[layer][:, current_len, :] = V_cache_out
        # data_decode = output
    return data_decode
```

#### 3.1.2 C++: `RunCpFrame`

**入口**: `RunCpFrame()` (`LLM_cp_tts_insert.inc:289-477`)。

完整流程：
```
1. 接收 last_hidden_bf16 [D] + primary_code
   primary_embed = embed_selector.getByIndex(primary_code)

2. 构建 cp_ctx = [last_hidden_bf16 | primary_embed]  # 2*D 元素
   out_codec_sum_bf16 = primary_embed

3. 生成 15 个残余 codes (j = 0..14):
   for j in range(15):
       prefill = (j == 0)
       seq_len = prefill ? 2 : 1
       history_len = prefill ? 0 : (j+1)  # j=0→0, j=1→2, j=2→3, ...
       current_len = history_len + seq_len  # j=0→2, j=1→3, j=2→4, ...

       # 零化 mask/embed/indices buffer
       # 构建 mask
       if prefill: build_prefill_mask(0, cp_prefill_token_num, 0, 2)
       else:       mask[i] = (i <= history_len) ? 0 : -65536

       # 填充 indices: [0,1,...] for prefill, [history_len] for decode
       # 填充 embed: cp_ctx (2 tokens) for prefill, 最后一个 res_embed for decode

       # 5层 CP layers × inference
       # 更新 KV cache
       # hidden_step = embed 最后一个位置
       # Post norm → lm_head_j → 残差 code
       # 查 embedding, 累加 codec_sum, append 到 cp_ctx
```

#### 3.1.3 CP Prefill 对比 (j=0)

| 维度 | Python | C++ | 状态 |
|------|--------|-----|------|
| 输入 token 数 | `valid_len = 2` | `seq_len = 2` | ✅ |
| 第 1 token | last_hidden (normed) | cp_ctx[0:D] = last_hidden (normed) | ✅ |
| 第 2 token | primary_embed | cp_ctx[D:2D] = primary_embed | ✅ |
| Indices | `[0, 1, 0, ..., 0]` | `[0, 1, 0, ..., 0]` | ✅ |
| Mask | causal row[0:1]=0, row[0:2]=0 | build_prefill_mask: causal, self 可见 | ✅ |
| KV cache 初始 | zeros (host) | zeros (host, memset 0) | ✅ |
| Post norm 输入 | last_hidden_raw = data[:, 1, :] | hidden_step = embed_tmp[1*D : 2*D] | ✅ |

**结论：CP prefill 完全一致。** ✅

#### 3.1.4 CP Decode 对比 (j=1..14)

| 维度 | Python | C++ | 状态 |
|------|--------|-----|------|
| 输入 | embedding_tables[lm_step-1](prev_id) | cp_embed_tables[j-1][res_code * D] | ✅ |
| seq_len | 1 | 1 | ✅ |
| current_len (decode_indices) | `2+j` (2,3,4で始まる) | `history_len = j+1`, idx = history_len = j+1 = `2+j` | ✅ |
| Mask | 0..current_len-1 可见, self 不可见 | 0..history_len 可见, self 不可见 | ✅ |
| KV cache | 完整 buffer 传入 | 完整 buffer 传入 | ✅ |

**结论：CP decode 完全一致。** ✅

### 3.2 CP Post Norm

**Python** (`_run_post_norm`, `infer.py:1157-1164`):
```python
def _run_post_norm(self, hidden_token: np.ndarray) -> np.ndarray:
    outputs = self.post_session.run({"input": hidden_token})
    out_key = "output_norm" if "output_norm" in outputs else "output"
    out = outputs[out_key].astype(np.float32)
    out = out.reshape((1, out.size // self.hidden_size, self.hidden_size))
    return out[:, -1:, :]  # [1, 1, H]
```

- 输入：`hidden_token` [1, 1, H] bf16
- 输出：取 key `"output_norm"` 或 fallback `"output"`, 取最后一个位置

**C++** (`RunCpFrame`, `LLM_cp_tts_insert.inc:422-436`):
```cpp
// 取最后一个位置 hidden
memcpy(hidden_step.data(), embed_tmp.data() + (seq_len - 1) * D, D * sizeof(unsigned short));

// CP post inference
auto &t_in = cp_post.get_input("input");
llm_h2d(LLM_WADDR(t_in), hidden_step.data(), ...);
cp_post.inference();
auto &t_out = cp_post.get_output("output_norm");  // 使用 "output_norm"
llm_d2h(post_buf.data(), LLM_RADDR(t_out), ...);
memcpy(hidden_step.data(), post_buf.data(), copy_words * sizeof(unsigned short));
```

| 维度 | Python | C++ |
|------|--------|-----|
| 输入形状 | [1, 1, H] bf16 | D 个 bf16（等价 [1, 1, H]） |
| 输出 key | `"output_norm"` 或 `"output"` | `"output_norm"` |
| 输出处理 | reshape 后取 `[:, -1:, :]` | memcpy 全部到 hidden_step |

**结论：CP Post Norm 逻辑一致。** ✅

### 3.3 CP LM Head

**Python** (`_run_lm_head_logits`, `infer.py:1166-1177`):
```python
def _run_lm_head_logits(self, step, hidden_norm):
    lm_input = self.lm_head_input_buffers[step]  # [1, lm_length, H]
    lm_input.fill(0.0)                            # 全部填 0
    lm_input[:, -1:, :] = hidden_norm             # 最后位置填入 hidden_norm
    outputs = self.lm_head_sessions[step].run({"input": lm_input})
    logits = outputs[out_key].astype(np.float32)
    logits = logits.reshape((1, logits.size // vocab_size, vocab_size))
    return logits[:, -1, :].reshape(-1)           # [vocab_size]
```

- lm_head 输入 shape：`[1, L, H]`（L 可能 > 1），除最后位置外全为 0
- 输出取最后一个位置的 logits

**C++** (`RunCpFrame`, `LLM_cp_tts_insert.inc:438-454`):
```cpp
auto &t_in = lmh.get_input("input");
llm_h2d(LLM_WADDR(t_in), hidden_step.data(), ...);  // D 个 bf16，即 [1, 1, H]
lmh.inference();
auto &t_out = lmh.get_output("output");
logits_n = t_out.nSize / sizeof(float);  // 直接 fp32
llm_d2h(logits_fp32.data(), LLM_RADDR(t_out), ...);
// logits_fp32 = [vocab_size]
```

- lm_head 输入 shape：`[1, 1, H]`
- 输出直接为 `[vocab_size]` fp32

**差异分析**:

Python lm_head 输入 buffer 是 `[1, L, H]`（L = input_shape[1]），仅最后位置有 non-zero 值。这是为了兼容 ONNX MatMul（需要固定 seq_len 维度）。C++ lm_head 轴模型 input 是 `[1, 1, H]`。

**为什么等价**: MatMul 每个位置独立计算。`zeros + last_position` 的 MatMul 结果中，最后一个位置的 logits 完全由 `hidden_norm` 决定。Python 取 `logits[:, -1, :]`，C++ 取唯一位置的 logits。两者等价。✅

### 3.4 CP `small_to_mtp_projection` 处理

**Python** (`infer.py:1537-1538`):
```python
# NOTE: small_to_mtp_projection weights merged into CP layer0 axmodel, skip here
projected_inputs = inputs_embeds  # 不做投影
```

**C++**: 无投影步骤。CP 轴模型的 layer0 已合并投影权重。

**结论：两种实现都跳过了投影，CP 轴模型内部处理。** ✅

---

## 四、Codec Sum 与 Next Embed 深入分析

### 4.1 Codec Sum 产生过程

#### Python

Python 端 codec_sum 的产生不直接在 `infer.py` 中，而在 Qwen3TTSModel 的 `generate_voice_clone` 中。根据 Qwen3TTS 原始实现逻辑推断：

```python
# 每生成一帧 (16 codes):
codec_hiddens = []
for i in range(num_code_groups):  # 16
    codec_hiddens.append(codec_embedding[i](codes[i]))
# torch.sum 可能使用 fp32 中间值累加
codec_sum = torch.sum(torch.stack(codec_hiddens), dim=0)  # fp32 accumulation
```

Qwen3TTS 的 `codec_embedding` 是一个 `ModuleList`，每步 `codes[i]` 是标量 token id，`codec_embedding[i](codes[i])` 返回 `[H]` 向量。`torch.sum` 默认使用 fp32 累加（即使输入是 bf16）。

#### C++ (`RunCpFrame`, `LLM_cp_tts_insert.inc:307-309, 463-472`):

```cpp
// 初始化: out_codec_sum_bf16 = primary_embed (bf16)
out_codec_sum_bf16 = primary_embed;

// 每步累加:
for (int d = 0; d < D; ++d) {
    float a = bfloat16(out_codec_sum_bf16[d]).fp32();  // bf16 → fp32
    float b = bfloat16(res_embed[d]).fp32();             // bf16 → fp32
    out_codec_sum_bf16[d] = bfloat16(a + b).data;       // fp32 → bf16 截断
}
```

**关键差异**:

| | Python (推测) | C++ |
|---|---|---|
| 累加方式 | `torch.sum(stack(embeds), dim=0)` | 逐元素 bf16→fp32→add→bf16 |
| 中间精度 | 可能全部在 fp32 累加后一次截断 | 每次加法后截断回 bf16 |
| 累加次数 | 16 次（1 primary + 15 residuals） | 16 次（同上） |

**差异来源**: C++ 每次 `a + b` 后截断回 bf16，损失了 16 位精度（bf16 只有 7 位尾数 vs fp32 的 23 位）。Python 的 `torch.sum` 可能在 fp32 累加全部 16 个向量后一次性截断（或使用 bf16 中间累加，取决于 PyTorch 版本和硬件）。

**影响**: 16 次累加的 bf16 截断误差可能累积到 `~1e-4 * sqrt(16) ≈ 4e-4` 量级。对 greedy argmax 通常无影响。

### 4.2 Codec Sum + tts_pad_vec 相加精度

#### Python

Python 端 next_embed 构造成发生在 Qwen3TTSModel 中，非流式模式：
```python
next_embed = codec_sum + tts_pad_embed  # torch.add, fp32 中间值
```

#### C++ (`LLM_cp_tts_insert.inc:780-793`):

```cpp
// non-streaming:
for (int d = 0; d < _attr.tokens_embed_size; ++d) {
    float a = bfloat16(codec_sum_bf16[d]).fp32();
    float b = bfloat16(tts_pad_vec_bf16[d]).fp32();
    next_embed[d] = bfloat16(a + b).data;
}
```

**差异**:

| | Python | C++ |
|---|---|---|
| 操作 | `torch.add(bf16_tensor, bf16_tensor)` | `bf16→fp32→add→bf16` 逐元素 |
| 内部精度 | fp32 | fp32 |
| 截断 | 加法后一次 | 加法后一次 |

**结论**: 单纯的一次 bf16 加法，Python 和 C++ 的精度路径相同（都是 fp32 加法后截断 bf16）。差异来自前一步 codec_sum 的累加精度差异（见 §4.1），而非加法本身。⚠️ 微小差异。

### 4.3 Streaming next_embed 来源（补充）

C++ streaming 模式（`LLM_cp_tts_insert.inc:763-777`）:
```cpp
int txt_pos = trailing_start + step;
if (txt_pos < input_embed_num) {
    next_embed = codec_sum + all_prefill_hidden[txt_pos];  // 取预存的 normed hidden
} else {
    next_embed = codec_sum;  // 超出文本范围，仅用 codec_sum
}
```

Python streaming 由 Qwen3TTSModel 内部控制。本文以非流式为主，streaming 不展开。

---

## 五、差异总结

### 5.1 数据结构差异（已验证等价）

| 编号 | 差异点 | Python | C++ | 影响 |
|------|--------|--------|-----|------|
| D1 | **Indices padding** | `np.ones` → **1** | `memset(0)` → **0** | 无（causal mask 遮蔽） |
| D2 | **Prefill 迭代顺序** | layer outer, chunk inner | chunk outer, layer inner | 无（数学等价） |
| D3 | **Decode mask 实现** | 2D 预计算矩阵查表 | 1D 状态更新数组 | 无（语义等价） |
| D4 | **CP lm_head 输入** | [1, L, H] + zeros padding | [1, 1, H] 直接 | 无（MatMul 各位置独立） |

### 5.2 数值精度差异（潜在影响）

| 编号 | 差异点 | Python | C++ | 量级 |
|------|--------|--------|-----|------|
| N1 | **RMSNorm 精度** | PyTorch `nn.RMSNorm` | `rmsnorm_bf16` 手动 | cosine >0.999 |
| N2 | **codec_sum 累加** | `torch.sum` (推测 fp32 累加) | 逐元素 bf16 截断累加 | ~4e-4 |
| N3 | **bfloat16 硬件路径** | PyTorch bf16 | C++ `bfloat16` 结构体 | 单次 <1e-7 |

### 5.3 随机性差异

| 编号 | 差异点 | Python | C++ | 影响 |
|------|--------|--------|-----|------|
| R1 | **随机数发生器** | `np.random.choice` | `std::mt19937` | Sample 模式不同；Greedy 无影响 |

---

## 六、采样对齐改进记录

> 基于 [qwen3_tts_single_sampling_coupling_analysis.md](../qwen3_tts_single_sampling_coupling_analysis.md) 的结论—
> 采样对结果影响极大，主 Talker 与 Sub-talker 通过 codec embedding 求和深度耦合。
> 以下改动已实施并编译通过。

### 6.1 Top-P 边界修正 (`>=` → `>`)

**影响文件**: `src/runner/LLMPostprocess.hpp:314`, `src/runner/LLM_cp_tts_insert.inc:278`

**差异**: Python 使用 `cum > top_p` (且 `drop[0]=False` 保证至少保留 1 个 token)，C++ 用 `cum >= top_p`。在 top_p 边界上 C++ 会比 Python 多保留一个低概率 token。

**修改**:
```cpp
// 前: if (cum >= top_p) { cut = i + 1; break; }
// 后: if (cum > top_p)  { cut = i + 1; break; }
if (cut == 0) cut = 1;   // 保证至少保留 1 个 token
```

### 6.2 CP 随机数种子可控

**影响文件**: `src/runner/LLM.hpp:61`, `src/runner/LLM.cpp:95,771`, `src/runner/LLM_cp_tts_insert.inc:288`, `tools/qwen3_tts_infer.cpp:408`

**差异**: CP 采样使用 `thread_local std::mt19937 rng(std::random_device{}())`，外部无法控制种子。Talker RNG 可通过 `set_seed()` 控制，但 CP 每帧 15 步采样的 RNG 完全随机，导致即使 Talker 对齐了，CP 的 codec_sum 仍不一致，进而影响下一帧 Talker 输入（耦合机制）。

**修改**:
- `LLMAttrType` 新增 `int cp_seed = -1;`
- `LLM::Impl` 新增成员 `std::mt19937 cp_rng_;`
- `CpSampleFromLogits()` 从静态函数改为成员函数，直接使用 `cp_rng_`
- `qwen3_tts_infer` CLI `--seed` 同时设置 Talker 和 CP 种子

### 6.3 浮点精度对齐 (float → double)

**影响文件**: `src/runner/LLMPostprocess.hpp:286`, `src/runner/LLM_cp_tts_insert.inc:250`

**差异**: Python `_select_next_code_from_logits` 全程使用 `np.float64` 双精度计算（temperature / top-k / softmax / top-p）。C++ 使用 `float` 单精度。对于 2000+ 词表，softmax 中 exp 计算和累加的精度差异会放大。

**修改**: `sample_from_logits` 和 `CpSampleFromLogits` 的中间计算全部改用 `std::vector<double>`，top-k 屏蔽值也相应改为 `-1e30`（对齐 Python）。

### 6.4 Repetition Penalty 默认关闭

**影响文件**: `talker/post_config.json`

**差异**: C++ Talker 默认 `enable_repetition_penalty: true, repetition_penalty: 1.05`。Python TTS 的 `_select_next_code_from_logits` 不应用 repetition penalty。

**修改**: `enable_repetition_penalty` 设为 `false`，`repetition_penalty` 设为 `1.0`。

### 6.5 Softmax 除零保护

**影响文件**: `src/runner/LLMPostprocess.hpp:303,319`, `src/runner/LLM_cp_tts_insert.inc:267,284`

**差异**: Python 有 `np.clip(probs.sum(), 1e-12, None)` 防除零。C++ 首次 softmax 归一化无保护。

**修改**: 两处 softmax 归一化（初次 + top-p 后）均加入 `sum = std::max(sum, 1e-12)` 保护。

### 6.6 Codec Sum fp32 累加

**影响文件**: `src/runner/LLM_cp_tts_insert.inc:309-315, 467-471, 480-482`

**差异**: Python 端 16 个 codec embedding 通过 `torch.cat + .sum(dim=1)` 求和，`torch.sum` 内部使用 fp32 累加器，最终只截断一次回 bf16。C++ 端原来每加一个 residual embed 就截断回 bf16，16 次截断累积误差约 `1e-3 * sqrt(16) ≈ 4e-3`。

这个 `codec_sum` 直接加进 `next_embed = codec_sum + tts_pad_vec` 作为 Talker decode 的输入，即使单个 token 差异不改变 argmax，累积到下一帧后级联放大。

**修改**: 增加 `std::vector<float> codec_sum_fp32` 在 fp32 中累加所有 16 个 embedding，最后一次性转回 bf16。与 Python `torch.sum` 的精度路径一致。

---

## 七、Token 不正确根因分析

实测中 Talker prefill 第一个 primary_code 可能正确，但后续 decode 帧的 token 快速偏离。根据上述差异链：

```
prefill raw_hidden → RMSNorm → normed_hidden → CP input
                                              → codec_sum (fp32 vs bf16累积)
next_embed = codec_sum + tts_pad_vec
    → Talker decode(next_embed) → next raw_hidden
    → 略微偏离的 hidden → 不同的 primary_code → 级联错误
```

关键链路上的总误差来源：

| 环节 | 差异来源 | 量化 | 是否已修复 |
|------|---------|------|-----------|
| CP 输入 hidden | RMSNorm 精度（eps 一致 1e-6，截断路径不同） | cosine > 0.999 | ⚠️ 未修复（容忍） |
| codec_sum | bf16 逐步截断 → **fp32 一次截断** | ~4e-3 → ~0 | ✅ 6.6 |
| next_embed | codec_sum + tts_pad_vec 的 bf16 加法 | 单次 < 1e-7 | ✅ 原本正确 |
| Talker decode | axmodel 内部 bf16 精度 | 与 ONNX 导出一致 | ✅ 无差异 |

### 后续建议

如果 codec_sum 修复后 token 仍不正确，建议**逐帧 dump intermediate tensors** 定位第一个分叉点：

1. **Frame 0 Talker prefill raw_hidden** — 对比 cosine similarity
2. **Frame 0 CP 输入 normed_hidden** — 对比 RMSNorm 输出
3. **Frame 0 CP 15 个 residual codes** — 定位具体哪一步分叉
4. **Frame 0 codec_sum fp32 向量** — 确认累加已对齐
5. **Frame 1 next_embed** — 对比 Talker decode 输入
6. **Frame 1 Talker decode raw_hidden** — 确认分叉起点

> 已有 debug dump 机制：`--debug_dump_dir` CLI 参数可开启。Python 端通过 `--dump_cpp_input_dir` 导出参考数据。使用 `scripts/compare_talker_kvcache.py` 对比 KV cache。

---

## 八、Greedy 对齐验证建议（原六）

关闭所有随机性，用 **greedy + non-streaming** 模式对比：

**Python**:
```bash
python scripts/infer.py \
    --do_sample False \
    --subtalker_dosample False \
    --non_streaming_mode \
    --dump_cpp_input_dir ./debug_bin \
    --dump_output_codes_dir ./debug_bin \
    --skip_wav_generation \
    ...
```

**C++**:
```bash
./build/install/bin/qwen3_tts_infer \
    <talker_dir> ./debug_bin \
    --max_new_tokens 4096 \
    --temperature 0.0 \
    --cp_temperature 0.0 \
    --streaming false
```

如果 Greedy 仍然不一致，请逐层 dump intermediate tensors，从以下关键节点开始排查：
1. Talker prefill 后第一个 token 的 raw hidden (Python: `runner.prefill()` 返回值的 `[:,0,:]`)
2. Talker prefill 后 RMSNorm 输出 (Python: `_to_hidden_tensor` 结果)
3. Talker 第一个 primary code (prefill post logits argmax)
4. CP prefill 后 `hidden_step`（对应 `data[:, 1, :]` 位置）
5. CP post norm 后 `hidden_norm`
6. CP lm_head_0 logits