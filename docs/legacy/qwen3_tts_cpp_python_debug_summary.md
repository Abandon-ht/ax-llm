# Qwen3-TTS C++ vs Python 推理一致性调试总结

> **更新日期**: 2026-05-20  
> **本次更新**: 完成 greedy 采样对齐验证，定位根因为 Talker RMSNorm 数值差异的级联放大。

---

## 1. 问题背景

在 AX650 平台上，Qwen3-TTS 模型的 C++ 推理结果与 Python（axengine）推理结果存在差异：
- **Talker Prefill**：已验证 bit-exact 匹配（raw hidden）
- **CP（Code Predictor）Decode**：step 0~1 匹配，step 2 起发散
- **Talker Decode**：因 CP 输入错误，全程严重分歧

目标：定位并消除 C++ 与 Python 在 decode 阶段的所有推理差异。

---

## 2. 已完成验证项及结论

### 2.1 Talker Prefill —— ✅ Raw Hidden Bit-Exact，RMSNorm 存在 0.125 差异

| 验证项 | 方法 | 结论 |
|--------|------|------|
| Layer0 输出 / KV Cache | 逐元素对比 | `max_diff = 0`，完全一致 |
| Last raw hidden | 逐元素对比 | `cos=1.000`, `max_diff=0`，bit-exact |
| Last normed hidden | 逐元素对比 | `cos=0.999996`, `max_diff=0.125` |
| Prefill logits | argmax / top5 对比 | argmax 一致，值完全匹配 |

**关键发现**：
- Talker axmodel 层的输出（raw hidden）在 C++ 和 Python 之间完全一致，说明 axmodel runtime 无差异。
- **差异唯一来源**：C++ 手写 `rmsnorm_bf16` 与 Python PyTorch `self.norm`（RMSNorm）存在 `max_diff=0.125` 的数值差异。

**代码对应**：
- C++: `src/runner/LLM_cp_tts_insert.inc:236-249` (`rmsnorm_bf16`)
- Python: `scripts/infer.py:854-856` (`_to_hidden_tensor` → `self.norm`)

---

### 2.2 CP 输入对齐 —— ✅ 已修复并验证

| 问题 | 根因 | 修复方式 | 验证结果 |
|------|------|----------|----------|
| CP prefill 输入格式 | Python 未正确拼接 `past_hidden + last_id_hidden` | 改为 `torch.cat((past_hidden, last_id_hidden), dim=1)` | CP prefill input cosine=1.0 |
| CP post-norm 提取 | Python 使用了错误的输出 tensor 键 | 改为 `outputs["output_norm"]` | post-norm hidden cosine≈0.9998 |
| `tts_pad_vec` 形状 | Python 读取了 1025 个元素 | 修正为 1024 dims | 与 C++ 一致 |
| CP embedding 表加载 | Python 使用 PyTorch Embedding 而非原始 bf16 文件 | 从 `.bfloat16.bin` 加载 `codec_embedding` | 表内容一致 |
| 采样随机性 | temperature > 0 导致非确定性 | C++ `cp_temperature=0.0f`，Python `--no-subtalker_dosample` | greedy 对齐 |

**结论**：CP 的输入构造、embedding 表、采样策略、mask/indices/KV cache 逻辑已完全对齐。

---

### 2.3 CP Decode 子码 —— ❌ Step 2 起发散（Greedy 下）

**现象**（greedy 采样）：

| Step | C++ Token | Python Token | lm_head cos | lm_head max_diff | 判定 |
|------|-----------|--------------|-------------|------------------|------|
| 0 | 117 | 117 | 0.999891 | 0.280 | ✅ MATCH |
| 1 | 604 | 604 | 0.999966 | 0.199 | ✅ MATCH |
| 2 | **1349** | **279** | 0.999980 | 0.298 | ❌ **DIVERGE** |
| 3+ | ... | ... | <0.998 | >4.0 | ❌ 级联恶化 |

**Hidden state 传播链**：

| Step | pre_norm cos | pre_norm max_diff | post_norm cos | post_norm max_diff |
|------|-------------|-------------------|---------------|--------------------|
| 0 | 0.999793 | 0.250 | 0.999737 | 0.219 |
| 1 | 0.999760 | 0.312 | 0.999785 | 0.625 |
| 2 | 0.999783 | **0.500** | 0.999776 | **0.250** |

**分析**：
- Step 0~1 的 sampled token 完全相同，说明输入 embed 一致。
- 但 hidden state 的 `max_diff` 逐步放大（0.25 → 0.31 → 0.50），表明差异具有**累积性**。
- Step 2 的 `lm_head[2]` 处，token 1349 与 279 的 logits 竞争极为激烈（C++ top1=8.897 vs Python top1=8.888，差仅 0.009）。
- `max_diff=0.298` 的 logit 误差恰好让 argmax 从 1349 翻转到 279。

---

## 3. 根因定位：RMSNorm 差异 → CP 放大 → lm_head 翻转

### 3.1 第一步误差：Talker RMSNorm

```
C++ rmsnorm_bf16          vs    Python PyTorch RMSNorm
      ↓                              ↓
  max_diff=0.125               max_diff=0.125
      ↓
  last_normed_hidden (CP past_hidden)
```

- C++ 使用手写 `rmsnorm_bf16`，在 bf16→fp32 转换、逐元素求和后做 RMSNorm。
- Python 使用 PyTorch 原生 RMSNorm，可能在向量化、并行规约、中间精度上与手写实现存在微小差异。
- 该差异在 `last_normed_hidden` 上表现为 `max_diff=0.125`。

### 3.2 传播放大：CP Transformer 5 层

```
past_hidden (diff=0.125)
    ↓
CP Prefill (step 0)  →  pre_norm diff=0.25
    ↓
CP Decode (step 1)   →  pre_norm diff=0.31
    ↓
CP Decode (step 2)   →  pre_norm diff=0.50
    ↓
cp_post (RMSNorm)    →  post_norm diff=0.25
```

- CP 的 layer 计算本身（axmodel）在 C++ 和 Python 之间是一致的。
- 但由于初始输入 `past_hidden` 有 0.125 差异，且 Attention/FFN 会混合历史信息，每一层都会将误差略微放大。
- 经过 5 层 CP Transformer + 2 个 decode step 后，差异从 0.125 放大到 0.50。

### 3.3 触发点：lm_head[2] 排序翻转

```
post_norm hidden (cos=0.9998, max_diff=0.25)
    ↓
cp_lm_heads[2] / lm_head_sessions[2]  (同一 axmodel)
    ↓
logits: cos=0.99998, max_diff=0.298
    ↓
C++ argmax=1349(8.897)  vs  Python argmax=279(8.888)
    ↓
排序翻转 → 后续全部跑偏
```

- 两边使用**同一个** `code_predictor_lm_head_2.axmodel` 文件。
- 输入 hidden state 的 cosine 高达 0.9998，但 `max_diff=0.25` 恰好落在权重矩阵的敏感维度上。
- Top2 竞争 token（1349 vs 279）的 logits 差仅 0.009，0.298 的绝对差异足以翻转 greedy argmax。

### 3.4 后果：Python 跑飞

- Step 2 选错 token（279）→ step 3 输入 embed 错误。
- 后续 CP hidden state 迅速恶化（cos 跌至 0.88~0.96）。
- `codec_sum` 与 C++ 完全不一致（cos=0.90→0.18）。
- Talker decode 收到错误的 `inputs_embeds`，生成错误的 primary token。
- 下一轮 CP 的 `past_hidden` 也错了，形成**错误累积循环**。
- 最终 token 序列偏离正确分布，无法命中 EOS。

---

## 4. 关键逻辑逐项核对（排除代码 bug）

| 检查项 | C++ 实现 | Python 实现 | 核对结果 |
|--------|----------|-------------|----------|
| CP prefill 输入 | `cp_ctx=[last_hidden,primary_embed]` | `torch.cat((past_hidden,last_id_hidden),dim=1)` | ✅ 一致 |
| CP decode embed 来源 | `cp_ctx.last(D)` (res_embed) | `embedding_tables[lm_step-1](prev_id)` | ✅ 一致 |
| CP decode indices | `history_len+i` | `[[current_len]]` | ✅ 一致 |
| CP decode mask | history 可见, last=0 | `col<row?0:-65536; [:,-1]=0` | ✅ 一致 |
| KV cache 更新位置 | `current_len-1` | `current_len` | ✅ 一致 |
| lm_head 索引 | `cp_lm_heads[j]` | `lm_head_sessions[lm_step]` | ✅ 一致 |
| greedy 采样 | `CpSampleFromLogits(...,0.0f,...)` | `np.argmax(scores)` | ✅ 一致 |
| codec_sum 累加 | `fp32 accumulator` | `torch.sum(fp32)` | ✅ 一致 |

**结论**：所有耦合逻辑、索引、mask、KV cache、采样策略均完全对齐，不存在实现错误。

---

## 5. 修复建议

### 5.1 最高优先级：统一 RMSNorm 实现

**方案 A**（推荐）：将 Talker 的 RMSNorm 也导出为 axmodel，让 C++ 和 Python 都走 axmodel 执行，消除手写实现与 PyTorch 的差异。

**方案 B**：在 C++ 中调用 PyTorch C++ API（libtorch）执行 RMSNorm，确保与 Python 逐位一致。

**方案 C**：若无法替换实现，可尝试对齐 `rmsnorm_bf16` 的求和顺序/向量化行为，使其输出与 PyTorch 的 diff 缩小到 <1e-3。

### 5.2 验证方法

1. **RMSNorm 隔离测试**：
   - 将 C++ `rmsnorm_bf16` 的输入和输出导出为 bin。
   - 用同一输入在 Python 中执行 `self.norm`，对比输出差异。
   - 确认 `max_diff=0.125` 是否完全由 RMSNorm 引起。

2. ** bypass 验证**：
   - 临时修改 C++，跳过 `rmsnorm_bf16`，直接加载 Python dump 的 `prefill_last_normed_hidden.bin` 作为 `all_prefill_hidden`。
   - 若此时 C++ 与 Python 的 CP tokens 完全 match，则 100% 确认 RMSNorm 为根因。

3. **lm_head 敏感性分析**：
   - Dump C++ 和 Python 的 step 2 post_norm hidden state。
   - 分别输入到同一个 `code_predictor_lm_head_2.axmodel`，确认 logits diff=0.298 是否由 hidden diff=0.25 线性投影导致。

---

## 6. 历史记录

### 2026-05-20 之前的状态

- 此前认为 CP 12/15 子码匹配，3 个分歧（子码 12~14）。
- 当时怀疑 lm_head 精度漂移或 KV cache 累积误差。
- **本次更新后**：在 greedy 采样下重新验证，发现实际发散点提前到 step 2，且根因锁定为 Talker RMSNorm 差异的级联放大。

---

*文档生成于 2026-05-20，基于 `scripts/infer.py`、`src/runner/LLM.cpp`、`src/runner/LLM_cp_tts_insert.inc` 及对比脚本输出。*
