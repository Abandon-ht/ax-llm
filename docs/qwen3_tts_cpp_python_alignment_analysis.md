# Qwen3-TTS C++ 与 Python 推理对齐分析报告

> 基于 `scripts/infer.py` + `scripts/infer.sh` 与 `tools/qwen3_tts_infer.cpp` + `src/runner/LLM.cpp` / `LLM_cp_tts_insert.inc` 的代码级对比分析。
> 输入来源：Sherpa-ONNX `offline-tts-qwen3-impl.cc` non-streaming 模式 dump 的 `debug_bin/`（`prefill_embeds_bf16.bin`, `meta.json`, `tts_pad_vec.bin`）。

---

## 一、问题现象

使用 C++ 工具 `qwen3_tts_infer` 加载 Sherpa-ONNX dump 的 `debug_bin` 输入，执行 talker + code predictor 推理后，生成的 codes 转换为音频不正确。需要从代码层面找出 C++ 实现与 Python 推理的差异。

---

## 二、两端推理流程概览

### 2.1 Python 端（`infer.py` → `infer.sh`）

```
Talker Prefill  →  normed hidden (last token)  →  Talker Post  →  primary_code_0
       ↓
CP Prefill (inputs_embeds = normed hidden[1,T,H])
       ↓
CP Step 0: last_hidden_raw (prefill 输出最后一个位置) → post_norm → lm_head(start_lm_step) → residual_code_0
CP Step 1: embed_table[0](prev_code) → CP Decode → post_norm → lm_head(1) → residual_code_1
... (共 15 个 residual codes)

Talker Decode Step 1:
  next_embed = codec_sum + tts_pad_vec (non-streaming)
  → Talker Decode → normed hidden → primary_code_1
  → 再次调用 CP（inputs_embeds = 当前 decode 输出的 normed hidden [1,1,H]）
```

**关键特征**：
- CP 的 `inputs_embeds` **仅来自 talker 当前 step 输出的 `hidden_states`**（已 RMSNorm），**不包含 `primary_embed`**。
- 当 talker 处于 decode step 时，CP 接收到的 `inputs_embeds` 形状为 `[1, 1, H]`，`valid_len = 1`。
- `start_lm_step = max(0, valid_len - 2)`，当 `valid_len = 1` 时，`start_lm_step = 0`。

### 2.2 C++ 端（`qwen3_tts_infer.cpp` → `LLM.cpp` / `LLM_cp_tts_insert.inc`）

```
Talker Prefill  →  all_prefill_hidden (逐 token RMSNorm 后)
       ↓
Talker Post (使用 embed/raw hidden) → primary_code_0
       ↓
RunCpFrame(txt_hidden_bf16=all_prefill_hidden[trailing_start+step], primary_code)
  → cp_ctx = [last_hidden, primary_embed]  (错误地包含 primary_embed)
  → j=0 (prefill, seq_len=2) → j>0 (decode, seq_len=1)
```

---

## 三、未对齐项详细分析

### P0-1：CP 输入 `last_hidden` 的来源错误（最高优先级）

| 项目 | Python | C++ 当前实现 | 影响 |
|------|--------|-------------|------|
| **CP 输入来源** | Talker 当前 step 输出的 `hidden_states`（已 normed）。Prefill 时取最后一个 token；Decode 时取当前 decode step 输出。 | 取自 `all_prefill_hidden[trailing_start + step]`，即 prefill 历史序列中第 `trailing_start + step` 个 token。 | **极高**。CP attention 的 query 根基完全错误。 |

**根因定位**：
- C++ 文件：`src/runner/LLM_cp_tts_insert.inc`，函数 `RunTtsWithCpCallback`
- 问题代码（约第 741-755 行）：

```cpp
int txt_pos = trailing_start + step;
std::vector<unsigned short> txt_hidden_bf16(_attr.tokens_embed_size);
if (txt_pos < input_embed_num) {
    for (int d = 0; d < _attr.tokens_embed_size; ++d) {
        txt_hidden_bf16[d] = all_prefill_hidden[(size_t)txt_pos * _attr.tokens_embed_size + d];
    }
}
```

- 这里混淆了 **"talker decode 的输入构造"**（`next_embed = codec_sum + txt_hidden[step]`，这是流式模式下 talker 自回归的输入）与 **"CP 的输入"**（CP 只需要 talker 当前 step 输出的 hidden state）。
- 在非流式模式下，Python 每一帧的 CP 输入都是 **当前 talker step 刚生成的 hidden**（prefill 最后一个 token 或 decode 输出），而不是 prefill 历史中的不同 token。

**修复方向**：
- 第 0 帧（来自 talker prefill）：CP 输入应为 `all_prefill_hidden` 的最后一个 token（`input_embed_num - 1`），即 `S-1`（`S=197`）。
- 第 1+ 帧（来自 talker decode）：应在 talker decode 生成 `embed` 后，对其做 RMSNorm，然后将 normed hidden 传给 CP。

---

### P0-2：CP 输入错误地混入了 `primary_embed`（最高优先级）

| 项目 | Python | C++ 当前实现 | 影响 |
|------|--------|-------------|------|
| **CP prefill 输入序列** | 仅 `[last_hidden]`（1 个 token）。`primary_code` 仅作为 frame 的第 0 个 code 保存，**不进入 CP 模型输入**。 | `cp_ctx = [last_hidden_bf16, primary_embed]`（2 个 token），把 primary code 的 embedding 也作为 CP layer 的输入。 | **极高**。导致 CP 序列长度、KV cache、mask、lm_head step 全部错位。 |

**根因定位**：
- C++ 文件：`src/runner/LLM_cp_tts_insert.inc`，函数 `RunCpFrame`
- 问题代码（约第 311-315 行）：

```cpp
std::vector<unsigned short> cp_ctx;
cp_ctx.reserve(17 * D);
cp_ctx.insert(cp_ctx.end(), last_hidden_bf16.begin(), last_hidden_bf16.end());
cp_ctx.insert(cp_ctx.end(), primary_embed.begin(), primary_embed.end());  // ← 不应加入
```

- Python 端：`StaticCodePredictorRunner.generate_from_inputs_embeds` 的输入 `inputs_embeds` 仅来自 talker 的 `hidden_states`，不包含 `primary_embed`。
- 在 Python 中，`primary_code` 仅用于：
  1. 保存为 `frame_codes[0]`
  2. 通过 talker 的 codec embedding 表构造下一帧 talker decode 的 `next_embed`（`codec_sum + txt_hidden` / `codec_sum + tts_pad_vec`）
  3. **不进入 CP 模型**

**修复方向**：
- 移除 `primary_embed` 进入 `cp_ctx` 的逻辑。
- `cp_ctx` 应仅包含 `last_hidden_bf16`。
- `j=0` 时 `seq_len = 1`（而非 2），`history_len = 0`。
- `j>0` 时 `seq_len = 1`，`history_len = j`（而非 `j+1`）。

---

### P0-3：Talker decode 后缺少 RMSNorm（最高优先级）

| 项目 | Python | C++ 当前实现 | 影响 |
|------|--------|-------------|------|
| **Talker decode 输出** | `forward` 返回前调用 `self.norm(hidden)`，得到 normed hidden。 | `embed` 是最后一层输出的 raw hidden，**未做 RMSNorm**。 | **高**。CP 输入数值分布与 Python 不一致。 |

**根因定位**：
- Python 文件：`infer.py`，`_AxEngineQwen3TTSTalkerModel._to_hidden_tensor`

```python
def _to_hidden_tensor(self, raw_hidden, device, dtype):
    hidden = torch_module.from_numpy(raw_hidden.astype(np.float32)).to(device=device, dtype=dtype)
    return self.norm(hidden)
```

- C++ 文件：`src/runner/LLM_cp_tts_insert.inc`
- C++ 仅在 prefill 结束后对 `all_prefill_hidden` 做了 `rmsnorm_bf16`（约第 628-637 行）。
- 但在 decode 循环中，`embed` 取自 `lyr.layer.get_output(decode_grpid, "output")`，是 raw hidden，**未 norm**。
- 当后续帧需要把当前 decode 的 hidden 传给 CP 时，如果直接传 `embed`，数值与 Python 不对齐。

**修复方向**：
- 在 talker decode 生成 `embed` 后，增加 RMSNorm：

```cpp
std::vector<unsigned short> normed_embed(_attr.tokens_embed_size);
if (!cp_norm_gamma.empty()) {
    rmsnorm_bf16(normed_embed.data(), embed.data(), cp_norm_gamma.data(), _attr.tokens_embed_size);
} else {
    normed_embed = embed;
}
```

- 第一帧 CP 输入使用 `all_prefill_hidden[S-1]`（已 normed）。
- 第 1+ 帧 CP 输入使用 `normed_embed`。

---

### P1-4：CP decode 的 KV cache 输入拷贝方式错误（高优先级）

| 项目 | Python | C++ 当前实现 | 影响 |
|------|--------|-------------|------|
| **CP decode KV cache 输入** | 传入完整的 `k_caches[layer_idx]`（`[1, kv_cache_len, kv_dim]`），模型通过 `indices`/`mask` 自行定位。 | 只拷贝 `cp_k_cache[m].data() + history_len * cp_kv_dim`（仅 1 个 token）到 device buffer 开头。 | **高**。模型在 decode step 看不到完整的历史 KV。 |

**根因定位**：
- C++ 文件：`src/runner/LLM_cp_tts_insert.inc`，函数 `RunCpFrame`
- 问题代码（约第 388-401 行）：

```cpp
// j>0 (decode) 时：
llm_h2d(LLM_WADDR(tk), cp_k_cache[m].data() + history_len * cp_kv_dim,
        std::min((size_t)tk.nSize, (size_t)cp_kv_dim * sizeof(unsigned short)), devid);
llm_h2d(LLM_WADDR(tv), cp_v_cache[m].data() + history_len * cp_kv_dim,
        std::min((size_t)tv.nSize, (size_t)cp_kv_dim * sizeof(unsigned short)), devid);
```

- Python 端：`_decode_one` 中：

```python
"K_cache": k_caches[layer_idx],   # 完整的 [1, kv_cache_len, kv_dim]
"V_cache": v_caches[layer_idx],
```

- AXModel 的 decode group 期望的 K_cache tensor 形状是整个 KV cache buffer（`[1, cp_kv_cache_num, cp_kv_dim]`），模型内部根据 `indices`（`current_len`）和 `mask` 来读取对应位置。C++ 代码错误地只拷贝了 `history_len` 偏移处的 1 个 token 到 buffer 开头。

**修复方向**：
- CP decode（`j>0`）时，与 prefill 一样传入完整的 KV cache buffer：

```cpp
llm_h2d(LLM_WADDR(tk), cp_k_cache[m].data(),
        std::min((size_t)tk.nSize, cp_k_cache[m].size() * sizeof(unsigned short)), devid);
llm_h2d(LLM_WADDR(tv), cp_v_cache[m].data(),
        std::min((size_t)tv.nSize, cp_v_cache[m].size() * sizeof(unsigned short)), devid);
```

- KV cache 更新逻辑（`llm_d2h` 到 `cp_k_cache[m].data() + (current_len-1) * cp_kv_dim`）保持不变，只更新新增位置即可。

---

### P1-5：CP decode mask 的 `history_len` 因 P0-2 而错位（中优先级）

| 项目 | Python | C++ 当前实现 | 影响 |
|------|--------|-------------|------|
| **CP decode mask history_len** | `current_len = valid_len + offset`。当 `valid_len=1` 时，step 1 的 `current_len=1`，mask 允许看到位置 0。 | 因 `primary_embed` 被加入输入，j=0 prefill 后 KV 中有 2 个 token，j=1 时 `history_len=2`。 | **中**。mask 允许长度比 Python 多 1。 |

**根因定位**：
- 这是 P0-2 的连锁反应。一旦 P0-2 修复（`cp_ctx` 只有 1 个 token），`history_len` 的自然定义应恢复为 `j`（而非 `j+1`）。
- 修复 P0-2 后，需同步检查 `RunCpFrame` 中的 `history_len` 和 `current_len` 计算：

```cpp
const int history_len = j;          // j=0 时为 0，j=1 时为 1
const int current_len = history_len + seq_len;  // seq_len=1，所以 current_len = j + 1
```

---

### P2-6：CP 采样参数未从 CLI 透传（低优先级）

| 项目 | Python | C++ 当前实现 | 影响 |
|------|--------|-------------|------|
| **CP 采样参数** | 从 CLI 透传 `temperature/top_k/top_p/do_sample`。 | `RunCpFrame` 中写死 `cp_temperature=0.9f, cp_top_k=50, cp_top_p=1.0f`，与 CLI 默认值相同，但未显式透传。 | **低**。当前默认值下不影响 greedy 对齐验证。 |

**修复方向**：
- 可选：将 `qwen3_tts_infer.cpp` 的 CLI 采样参数通过 `LLM::TtsCpCallback` 或新增接口传入 `RunCpFrame`。
- 在 greedy 对齐阶段可暂不处理。

---

## 四、修复步骤（建议执行顺序）

### Step 1：修复 CP 输入构造（解决 P0-1、P0-2）

**文件**：`src/runner/LLM_cp_tts_insert.inc`，函数 `RunCpFrame`

1. 修改 `cp_ctx` 构造，移除 `primary_embed`：

```cpp
// 修改前
std::vector<unsigned short> cp_ctx;
cp_ctx.reserve(17 * D);
cp_ctx.insert(cp_ctx.end(), last_hidden_bf16.begin(), last_hidden_bf16.end());
cp_ctx.insert(cp_ctx.end(), primary_embed.begin(), primary_embed.end());

// 修改后
std::vector<unsigned short> cp_ctx = last_hidden_bf16;  // 仅保留 last_hidden
```

2. 调整 `history_len` / `seq_len` 定义：

```cpp
const bool is_prefill = (j == 0);
const int gid = is_prefill ? cp_prefill_gid : (cp_decode_gid >= 0 ? cp_decode_gid : cp_prefill_gid);
const int seq_len = 1;  // 全部改为 1
const int history_len = j;  // j=0 时为 0，j=1 时为 1
const int current_len = history_len + seq_len;  // = j + 1
```

3. embed 拷贝逻辑简化：

```cpp
// 修改前（区分 is_prefill 和 !is_prefill）
if (is_prefill) {
    memcpy(embed_tmp.data(), cp_ctx.data(), seq_len * D * sizeof(unsigned short));
} else {
    memcpy(embed_tmp.data(), cp_ctx.data() + (cp_ctx.size() - D), D * sizeof(unsigned short));
}

// 修改后（始终只有 1 个 token）
memcpy(embed_tmp.data(), cp_ctx.data(), D * sizeof(unsigned short));
```

### Step 2：修复 Talker decode 后的 RMSNorm（解决 P0-3）

**文件**：`src/runner/LLM_cp_tts_insert.inc`，函数 `RunTtsWithCpCallback`

在 talker decode 生成 `embed` 后（`#else // AX650` 分支约第 905 行，或 `#ifdef USE_AXCL` 分支约第 857 行），增加 norm：

```cpp
// 在 talker decode 结束、得到 embed 后：
std::vector<unsigned short> normed_embed(_attr.tokens_embed_size);
if (!cp_norm_gamma.empty()) {
    rmsnorm_bf16(normed_embed.data(), embed.data(), cp_norm_gamma.data(), _attr.tokens_embed_size);
} else {
    normed_embed = embed;
}
```

然后修改 CP 输入传递逻辑：

```cpp
// 修改前（第 741-755 行附近）
int txt_pos = trailing_start + step;
std::vector<unsigned short> txt_hidden_bf16(_attr.tokens_embed_size);
if (txt_pos < input_embed_num) {
    txt_hidden_bf16 = ... all_prefill_hidden[txt_pos] ...
} else {
    ...
}

// 修改后
std::vector<unsigned short> txt_hidden_bf16;
if (step == 0) {
    // 第一帧：使用 talker prefill 最后一个 token 的 normed hidden
    txt_hidden_bf16.resize(_attr.tokens_embed_size);
    memcpy(txt_hidden_bf16.data(),
           all_prefill_hidden.data() + (size_t)(input_embed_num - 1) * _attr.tokens_embed_size,
           _attr.tokens_embed_size * sizeof(unsigned short));
} else {
    // 后续帧：使用 talker decode 输出的 normed hidden
    txt_hidden_bf16 = normed_embed;
}
```

### Step 3：修复 CP decode KV cache 输入（解决 P1-4）

**文件**：`src/runner/LLM_cp_tts_insert.inc`，函数 `RunCpFrame`

将 `j>0` 时的 KV cache 输入改为完整 buffer：

```cpp
// K/V cache input
{
    auto &tk = lyr.layer.get_input(gid, "K_cache");
    llm_h2d(LLM_WADDR(tk), cp_k_cache[m].data(),
            std::min((size_t)tk.nSize, cp_k_cache[m].size() * sizeof(unsigned short)), devid);
}
{
    auto &tv = lyr.layer.get_input(gid, "V_cache");
    llm_h2d(LLM_WADDR(tv), cp_v_cache[m].data(),
            std::min((size_t)tv.nSize, cp_v_cache[m].size() * sizeof(unsigned short)), devid);
}
```

**注意**：`is_prefill` 和 `!is_prefill` 的分支可以合并，两者都传入完整 KV cache。

### Step 4：检查并修复 CP decode mask（解决 P1-5）

修复 P0-2 后，`history_len = j`。检查 mask 构造逻辑：

```cpp
// 当前代码（约第 346-355 行）
if (is_prefill || cp_decode_gid < 0) {
    build_prefill_mask(mask_tmp, 0, cp_prefill_token_num, history_len, seq_len);
} else {
    const int mask_cap = cp_kv_cache_num + 1;
    const int mask_fill = std::min((int)mask_tmp.size(), mask_cap);
    for (int i = 0; i < mask_fill; ++i) {
        mask_tmp[i] = (i <= history_len) ? bfloat16(0.f).data : bfloat16(-65536.f).data;
    }
}
```

- 若 `cp_decode_gid >= 0`，确认 decode group 的 mask tensor `nSize` 是否等于 `mask_cap * sizeof(unsigned short)`。
- 如果是，上述 1D 填充逻辑在 `history_len = j` 时是正确的（允许看到前 `j+1` 个位置）。
- 如果 decode group 的 mask tensor 是 2D 的（`[1, cp_kv_cache_num + 1]`），需要确认数据布局是否匹配。
- **建议**：在 `InitCp` 的日志输出中，打印 decode group 的 `mask` tensor shape，确认 `nSize`。

---

## 五、验证与调试方案

### 5.1 逐模块 greedy 对齐验证

关闭所有随机采样，使用 greedy（argmax）模式对比 Python 与 C++ 的 intermediate tensors 和最终 tokens。

#### 阶段 A：Talker Prefill 对齐

**Python 侧**：
```bash
python3 scripts/infer.py \
    --compare_talker_frames 1 \
    --do_sample False \
    --temperature 1.0 \
    --dump_cpp_input_dir ./debug_bin \
    --non_streaming_mode \
    ...
```

**C++ 侧**：
- 开启 `SetDebugDumpDir`（在 `qwen3_tts_infer.cpp` 中调用 `llm.SetDebugDumpDir("./debug_ax");`）。
- 对比文件：
  - `debug_bin/prefill_embeds.bin` vs C++ 加载的 `prefill_embeds`
  - Python 打印的 `[compare][talker][frame=1][prefill] hidden_cos=... logits_cos=...` 应接近 1.0。
  - C++ dump 的 `debug_talker_prefill_last_hidden_ax.bin` 与 Python 的 `hidden_states[:, -1, :]` 做 cosine。

#### 阶段 B：Talker Decode 第一帧对齐

- 确认 C++ talker decode 后的 `normed_embed` 与 Python 对应位置的 `hidden_states` cosine > 0.999。
- 对比第一帧的 `primary_code` 是否一致（greedy 下必须相同）。

#### 阶段 C：CP Step 0 对齐

**Python 侧**：
```bash
python3 scripts/infer.py \
    --compare_code_predictor_frames 1 \
    --do_sample False \
    --non_streaming_mode \
    ...
```

观察输出：
```
[compare][code_predictor][frame=1][step0] hidden_cos=... layer_cos=...
[compare][code_predictor][frame=1] greedy_match=True ...
```

**C++ 侧**：
- 在 `RunCpFrame` 中，j=0 结束后，dump `hidden_step`（post_norm 前）和 `logits_fp32`（lm_head 后）。
- 对比 Python `ax_debug["first_hidden_norm"]` 与 C++ `hidden_step`（post_norm 后）的 cosine。
- 对比 Python `ax_debug["first_logits"]` 与 C++ `logits_fp32` 的 argmax 和 top-5。

#### 阶段 D：CP 完整帧 greedy 对齐

- 对比单帧的 16 个 codes（1 个 primary + 15 个 residual）是否完全一致。
- 如果单帧一致，继续对比多帧（前 5 帧）。

### 5.2 可视化 Diff 工具

建议在 `tools/` 目录下新增一个 `compare_tts_intermediate.cpp` 或 Python 脚本，自动化以下对比：

1. **Talker hidden diff**：读取 Python dump 的 `prefill_embeds.bin` + C++ dump 的 `debug_talker_prefill_last_hidden_ax.bin`，计算 L2/cosine。
2. **CP logits diff**：逐 frame、逐 j 对比 lm_head 输出的 logits argmax。
3. **Codes diff**：逐 frame 对比 16 个 codes 的匹配率。

### 5.3 已知架构级差异（可接受的不对齐）

以下差异已知且属于可接受范围，无需在代码中消除：

1. **PyTorch RMSNorm vs C++ `rmsnorm_bf16` 的数值精度**：两者 eps=1e-6，但 bf16↔fp32 转换顺序可能导致 <1e-4 的误差。只要 cosine > 0.999 即认为对齐。
2. **采样逻辑的随机性**：greedy 模式下必须完全一致；sample 模式下因 random seed 机制不同可能不一致，属于正常。
3. **Talker decode indices 的多行结构**：当前 C++ 的 decode indices 为单元素 scalar `[[current_len]]`，与 Python `decode_one` 的 `[[current_len]]` 一致（decode 时无需 3 行 repeat）。

---

## 六、修复优先级总结

| 优先级 | 问题编号 | 问题描述 | 修复文件 | 预计影响 |
|--------|----------|----------|----------|----------|
| P0 | P0-2 | CP 输入混入 `primary_embed` | `LLM_cp_tts_insert.inc` | **极高** |
| P0 | P0-1 | CP 输入来源错误（使用 `all_prefill_hidden[txt_pos]`） | `LLM_cp_tts_insert.inc` | **极高** |
| P0 | P0-3 | Talker decode 后缺少 RMSNorm | `LLM_cp_tts_insert.inc` | **高** |
| P1 | P1-4 | CP decode KV cache 只拷贝单 token | `LLM_cp_tts_insert.inc` | **高** |
| P1 | P1-5 | CP decode mask history_len 错位 | `LLM_cp_tts_insert.inc` | **中** |
| P2 | P2-6 | CP 采样参数未透传 | `qwen3_tts_infer.cpp` | **低** |

---

## 七、关键代码位置速查

| 功能 | 文件 | 函数/行号 |
|------|------|----------|
| Talker prefill indices 多行填充 | `LLM_cp_tts_insert.inc` | `RunTtsWithCpCallback` ~569-577 |
| Talker prefill 后 RMSNorm | `LLM_cp_tts_insert.inc` | `RunTtsWithCpCallback` ~628-637 |
| Talker decode 后 embed | `LLM_cp_tts_insert.inc` | `RunTtsWithCpCallback` ~905 (`#else AX650`) |
| CP 输入构造（`cp_ctx`） | `LLM_cp_tts_insert.inc` | `RunCpFrame` ~311-315 |
| CP `seq_len` / `history_len` | `LLM_cp_tts_insert.inc` | `RunCpFrame` ~329-333 |
| CP KV cache 输入 | `LLM_cp_tts_insert.inc` | `RunCpFrame` ~383-402 |
| CP mask 构造 | `LLM_cp_tts_insert.inc` | `RunCpFrame` ~346-355 |
| CP 采样 | `LLM_cp_tts_insert.inc` | `RunCpFrame` ~462-466 |
| Python CP prefill | `infer.py` | `StaticCodePredictorRunner.generate_from_inputs_embeds` ~1149-1265 |
| Python CP decode | `infer.py` | `StaticCodePredictorRunner._decode_one` ~1113-1147 |
| Python Talker norm | `infer.py` | `_AxEngineQwen3TTSTalkerModel._to_hidden_tensor` ~795-797 |

---

*分析日期：2026-05-13*
*输入数据：debug_bin (S=197, hidden_size=1024, trailing_start=7, streaming=false)*
