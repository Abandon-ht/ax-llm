# Qwen3-TTS C++ vs Python 推理一致性调试总结

## 1. 问题背景

在 AX650 平台上，Qwen3-TTS 模型的 C++ 推理结果与 Python（axengine）推理结果存在差异：
- **Talker Prefill**：已验证 bit-exact 匹配
- **CP（Code Predictor）Prefill**：输入已对齐，但 decode 阶段后 3/15 个子码分歧
- **Talker Decode Step 1**：输入 `next_embed` cosine≈0.84，raw hidden 严重分歧（cosine≈0.06）

目标：定位并消除 C++ 与 Python 在 decode 阶段的所有推理差异。

---

## 2. 已完成验证项及结论

### 2.1 Talker Prefill —— ✅ Bit-Exact 匹配

| 验证项 | 方法 | 结论 |
|--------|------|------|
| Layer0 输出 | 逐元素对比 | `max_diff = 0`，完全一致 |
| KV Cache | 逐元素对比 | `max_diff = 0`，完全一致 |
| Last raw hidden / normed hidden | 逐元素对比 | 完全一致 |
| Prefill logits | argmax 对比 | 均为 1130，完全一致 |
| 跨运行时验证 | C++ embed 输入 Python axengine | logits 完全一致，排除 axmodel/runtime 差异 |

**结论**：Talker prefill 阶段 C++ 与 Python 完全等价，差异不来源于 talker 模型本身或 axengine 运行时。

---

### 2.2 CP 输入对齐 —— ✅ 已修复并验证

| 问题 | 根因 | 修复方式 | 验证结果 |
|------|------|----------|----------|
| CP prefill 输入格式错误 | Python 未正确拼接 `past_hidden + last_id_hidden` | 改为 `torch.cat((past_hidden, last_id_hidden), dim=1)` | CP prefill input cosine=1.0 |
| CP post-norm 提取错误 | Python 使用了错误的输出 tensor 键 | 改为 `outputs["output_norm"]` | post-norm hidden cosine≈0.9998 |
| `tts_pad_vec` 形状错误 | Python 读取了 1025 个元素（格式混淆） | 修正为 1024 dims | 与 C++ 一致 |
| CP embedding 表加载差异 | Python 使用 PyTorch Embedding 而非原始 bf16 文件 | 从 `.bfloat16.bin` 加载 `codec_embedding` | 表内容一致 |
| 采样随机性 | temperature > 0 导致非确定性 | C++ 硬编码 `cp_temperature=0.0f`，Python `do_sample=False` | 贪婪采样，排除随机性 |

**结论**：CP 的输入、embedding 表、采样策略已完全对齐。

---

### 2.3 CP Decode 子码 —— ⚠️ 12/15 匹配，3 个分歧

**现象**：
- 子码 0–11：C++ 与 Python **完全匹配**
- 子码 12–14：分歧
  - C++: `[..., 2027, 812, 803]`
  - Python: `[..., 1422, 85, 185]`

**分析**：
- 前 12 步匹配说明：KV cache 更新、mask 构建、indices 设置、embedding 查找、lm_head 执行在步骤 0–11 均正确。
- 分歧从第 12 步开始出现，说明差异具有**累积性**，或第 12 步的某个输入/模型执行存在微小差异被放大。

**可能原因**（待验证）：
1. **lm_head 精度漂移**：C++ `ax_runner_ax650` 与 Python `axengine.InferenceSession` 对同一 lm_head axmodel 可能产生微小差异，前 11 步 argmax 恰好相同，第 12 步跨越决策边界。
2. **KV cache 累积误差**：bf16/fp32 转换或 cache 写入位置的微小差异在 12 步后放大。
3. **Mask/Indices 配置**：decode 阶段 `history_len=13` 时的 mask 或 indices 存在边界条件差异。

---

### 2.4 Talker Decode Step 1 —— ❌ 严重分歧

**现象**：
- `next_embed`（codec_sum + tts_pad_vec）cosine≈0.84，max_diff≈0.70
- Talker decode raw hidden：C++ norm=18.16 vs Python norm=31.79，cosine≈0.065

**分析**：
- `next_embed` 的分歧是 CP 子码分歧的直接后果（不同 residual code → 不同 embedding → 不同的 codec_sum）。
- 但即使 `next_embed` 存在差异，talker decode raw hidden 的 **极度严重分歧**（cosine≈0.06）暗示 talker decode 本身可能存在独立问题：
  - **KV cache 状态不一致**：prefill 结束后 KV cache 内容或 shape 存在差异。
  - **position_ids / cache_position 不匹配**：decode 第一步的 position index 设置错误。
  - **Mask 配置错误**：decode mask 的可见范围或形状与 Python 不一致。

---

## 3. 当前状态

### 3.1 调试代码清理

此前为定位问题临时添加的大量对称 dump 代码（CP prefill/layer0/talker prefill 全量 KV cache 等）**已完成清理**。仅保留两处非 dump 的实质性修复：
- `scripts/infer.py`：`frame_idx` 自增顺序修复（先取值再递增）。
- `src/runner/LLM_cp_tts_insert.inc`：CP `lm_head` 输入格式修复（bf16 → fp32）。

### 3.2 重新设计的调试工具（最小化方案）

基于已验证结论，新方案遵循**最小侵入原则**：只 dump 分歧点，跳过已确认 bit-exact 的阶段。

**控制方式**：
- C++：通过 `SetDebugDumpDir()` 控制，仅当目录非空时触发。
- Python：通过 `--dump_debug_dir` 参数控制。
- 对称目录：`{dump_dir}/cpp/` 与 `{dump_dir}/python/`，便于脚本自动对比。

**CP Decode dump**（针对 12/15 子码分歧）：
| 文件名 | 内容 | 目的 |
|--------|------|------|
| `cp_decode_step{j:03d}_pre_norm.bin` | 进入 post-norm 前的 hidden state | 定位 hidden state 首次漂移的步骤 |
| `cp_decode_step{j:03d}_lm_head_logits.bin` | lm_head 输出的 fp32 logits | 判断是 hidden drift 还是 lm_head 运行时差异 |
| `cp_decode_codes.bin` | 最终 15 个子码 | 快速确认是否复现分歧 |

**Talker Decode Step 1 dump**（针对 raw hidden 严重分歧，仅 `step == 0`）：
| 文件名 | 内容 | 目的 |
|--------|------|------|
| `talker_decode_step1_k_cache_layer0.bin` | Layer 0 的 K_cache | 验证 KV cache 初始化是否与 Python 一致 |
| `talker_decode_step1_v_cache_layer0.bin` | Layer 0 的 V_cache | 同上 |
| `talker_decode_step1_indices.bin` | decode indices | 验证 position id 是否一致 |
| `talker_decode_step1_mask.bin` | decode mask | 验证 mask 形状/值是否一致 |
| `talker_decode_step1_input.bin` | `next_embed`（codec_sum + pad/txt） | 确认输入差异是否由 CP 分歧导致 |
| `talker_decode_step1_output_raw.bin` | Layer 最后一层输出的 raw hidden | 确认 talker decode 本身是否产生严重分歧 |

**对比脚本**：`scripts/compare_debug_dumps.py`
- 自动遍历 `{dump_dir}/cpp/` 与 `{dump_dir}/python/` 同名文件。
- 计算 cosine similarity 与 max diff。
- 对 CP decode 逐步骤报告首次出现 `cosine < 1.0` 或 `argmax` 分歧的位置。
- 对 Talker decode step1 直接输出各输入 tensor 的对比结果。

### 3.3 待执行的验证

1. **部署并运行最小化 dump**：在 AX650 上同时运行 C++ 和 Python，仅收集上述精简 dump 文件。
2. **执行 `compare_debug_dumps.py`**：
   - 若 CP `pre_norm` 在 step 12 之前已出现 `cosine < 1.0` → 问题在 CP layer 执行或 KV cache 累积误差。
   - 若 CP `pre_norm` 完全一致，但 `lm_head_logits` 在 step 12 分歧 → 问题在 lm_head 模型运行时差异（可进一步做 lm_head 隔离测试：将 C++ step 12 hidden state 输入 Python axengine 运行 lm_head_12）。
   - 若 Talker decode step 1 的 `K/V cache`、`indices`、`mask` 不完全一致 → 问题在 talker KV cache 初始化或 mask 构建逻辑。
   - 若 Talker decode step 1 的输入完全一致，但 `output_raw` 严重分歧 → 问题在 talker decode 模型运行时或层间数据搬运（d2d/d2h）。
3. **lm_head 隔离测试**（条件触发）：将 C++ 的 step 12 hidden state 输入 Python `axengine` 运行 `lm_head_12.axmodel`，对比 logits。
4. **Talker decode KV cache 隔离**（条件触发）：对比 prefill 结束后 layer 0 的 K_cache 内容（C++ vs Python）。

---

## 4. 关键假设与风险

| 假设 | 风险 |
|------|------|
| CP 前 12 步完全匹配意味着 KV cache 完全一致 | 可能存在微小差异（<1e-3）未被 argmax 放大，在第 12 步才显现 |
| C++ `get_output(gid, "K_cache_out")` 能正确访问 group 1 | 已验证 axmodel 输出名为 `K_cache_out_1`，但 AX650 引擎可能在内部做了名称归一化；若实际访问的是 group 0 输出，则 KV cache 拷贝将完全错误（但目前看前 12 步匹配，此风险较低） |
| `cp_embed_tables[j]` 与 Python `embedding_tables[lm_step-1]` 一一对应 | 对于第一帧 `start_lm_step=0`，对应关系为 `cp_embed_tables[j-1]` ↔ `embedding_tables[j-1]`，已验证正确 |

---

## 5. 下一步行动

1. 实现最小化 dump 代码（C++ & Python），按 3.2 节方案在精确节点添加对称 dump。
2. 部署最新 C++ binary 到 AX650。
3. 执行 C++ 和 Python 推理，收集 dump 到同一目录（仅 3.2 节列出的文件）。
4. 运行 `python scripts/compare_debug_dumps.py <dump_dir>`，观察：
   - CP 在哪一步首次出现 `cosine < 1.0` 或 `argmax` 分歧。
   - Talker decode step 1 的 K/V cache、indices、mask 是否完全一致。
5. 根据对比结果，针对性修复：
   - 若是 CP lm_head 问题 → 执行 lm_head 隔离测试，对比 C++ vs Python 对同一 lm_head axmodel 的输出。
   - 若是 talker KV/mask 问题 → 修复 talker decode 的 cache/mask 构建逻辑。
   - 若是 talker decode 输入一致但 output_raw 分歧 → 检查 decode 阶段的层间数据搬运（d2d/d2h）或模型运行时差异。
