# Qwen3-TTS Talker + CP 深度对齐验证策略

> **版本**: v1.0  
> **目的**: 基于 `qwen3_tts_talker_cp_coupling_spec.md` 和 `debug.md`，设计一套从输入到输出的分层、全链路、可自动化的验证体系，用于持续对齐 Python（Golden）与 C++（AXEngine）推理实现。  
> **适用范围**: Talker Prefill、CP Prefill/Decode、Talker Decode、Talker↔CP 耦合数据流、端到端输出。

---

## 一、验证目标与核心原则

### 1.1 目标

| 层级 | 目标 | 判定标准 |
|------|------|----------|
| **L0 输入对齐** | Python 与 C++ 的 Talker/CP 输入张量完全一致 | `max_diff < 1e-6`，cosine > 0.999999 |
| **L1 Talker Prefill** | Talker prefill 输出的 hidden / logits / KV cache 与 Python 一致 | logits argmax 一致；hidden cosine > 0.9999 |
| **L2 CP 独立** | CP 在固定输入下，每步 sub-code 的 hidden / logits / sample 与 Python 一致 | 15 步 lm_head argmax 全部一致 |
| **L3 耦合数据流** | Talker 每 decode 步的输入构造（codec_sum + trailing_text）与 Python 一致 | decode input cosine > 0.9999 |
| **L4 端到端** | 最终 output_codes 的帧级、码本级分布与 Python 一致 | frame match rate > 95%（受采样影响） |

### 1.2 核心原则

1. **单变量原则**：每次只对比一个环节，确保差异可被唯一归因。
2. **从静到动**：先对齐 Prefill（静态、无历史依赖），再对齐 Decode（动态、有累积误差）。
3. **先值后采样**：先对齐 logits / hidden 的数值，再接受采样导致的 token 差异。
4. **FP32 对比**：C++ 侧的 BF16 数据在 dump 时统一转 FP32，避免格式差异干扰。

---

## 二、分层验证架构（L0 ~ L4）

### 2.0 L0: 输入数据对齐验证

**为什么重要**：C++ 侧的 `prefill_embeds.bin`、`trailing_text_hiddens.bin`、`tts_pad_vec.bin` 由 Python 脚本 `infer.py` dump 生成。如果这些输入本身就与 Python 内部使用的张量不一致，后续所有对比都是无意义的。

**检查点**：

| # | 检查项 | Python 来源 | C++ 输入文件 | 对比脚本 |
|---|--------|-------------|--------------|----------|
| L0-1 | Prefill embeds | `infer.py` 构造的 `active_embeds` | `prefill_embeds.bin` / `.bf16.bin` | `scripts/compare_prefill_checkpoints.py` (input 部分) |
| L0-2 | Trailing text hiddens | `infer.py` 的 `trailing_text_hidden` | `trailing_text_hiddens.bin` / `.bf16.bin` | 手动 `np.fromfile` 对比 |
| L0-3 | TTS pad embed | `infer.py` 的 `tts_pad_embed` | `tts_pad_vec.bin` / `.bf16.bin` | 手动 `np.fromfile` 对比 |
| L0-4 | Meta 信息 | `infer.py` 的 `S`, `hidden_size`, `vocab_size` 等 | `meta.json` | 目视检查 |

**通过标准**：L0-1 ~ L0-3 的 `max_diff < 1e-6`。

**实施步骤**：
```bash
# 1. Python 侧生成输入 dump
conda activate qwen3-tts
./scripts/infer.sh  # 确保 dump_cpp_input_dir 参数已配置

# 2. 直接对比 Python 内存值与 dump 文件值（可在 infer.py 中加断言）
```

---

### 2.1 L1: Talker Prefill 验证

**目标**：验证 Talker Transformer 的 prefill 前向传播是否与 Python 一致。

**已有能力**：
- C++ 侧可 dump `debug_talker_kvcache_ax/`、`debug_talker_prefill_last_hidden_ax.bin`、`debug_talker_prefill_logits_ax.bin`
- Python 侧可 dump `python_kvcache/`、`python_prefill_last_raw_hidden.bin`、`python_prefill_logits.bin`
- 对比脚本：`compare_prefill_checkpoints.py`、`compare_talker_prefill.py`、`compare_talker_kvcache.py`

**检查点**：

| # | 检查项 | 说明 | 已有脚本覆盖 |
|---|--------|------|--------------|
| L1-1 | Prefill input | `[S, H]` embeds | ✅ `compare_prefill_checkpoints.py` |
| L1-2 | Layer-0 output | 验证第一层即对齐，排除后续层累积误差 | ✅ `compare_prefill_checkpoints.py` |
| L1-3 | KV cache (all layers) | 每层的 K/V `[S, kv_dim]` | ✅ `compare_talker_kvcache.py` |
| L1-4 | Last raw hidden | pre-norm 的末帧 hidden | ✅ `compare_prefill_checkpoints.py` |
| L1-5 | Last normed hidden | post-RMSNorm 的末帧 hidden，即 `past_hidden` | ✅ `compare_prefill_checkpoints.py` |
| L1-6 | Prefill logits | `codec_head` 输出 | ✅ `compare_prefill_checkpoints.py` |
| L1-7 | **All-prefill hidden** | 全部 token 的 normed hidden（C++ 的 `all_prefill_hidden`） | ❌ **待补充** |

**待补充**：
- Python 侧需要 dump `python_prefill_all_hidden.bin`（`[S, H]` 的 normed hidden），与 C++ 的 `all_prefill_hidden` 逐 token 对比。
- 这是因为 CP 的 `past_hidden` 在第一步取的是末帧，但后续若需调试中间帧，需要全序列 hidden。

---

### 2.2 L2: CP 独立验证

**目标**：在**固定输入**（`past_hidden` + `primary_code`）下，验证 CP 自回归生成的 15 个 sub-codes 是否与 Python 一致。

**耦合 spec 关键点**：
- CP prefill 输入：`[past_hidden, primary_embed]`，`len=2`
- `generation_steps` 从 0 开始，每步 +1
- `lm_head[generation_steps]` 选择对应 head
- `codec_embedding[generation_steps - 1]` 选择对应 embedding table（decode 时）

**已有能力**：
- `compare_cp_dumps.py` 可对比 CP lm_head logits 和 hidden states，但它期望的文件 `cpp_cp_lm_head_xxx.bin`、`cpp_cp_hidden_xxx.bin` **目前在 C++ 侧没有生成逻辑**。

**C++ 侧需补充的 dump（在 `RunCpFrame` 中）**：

| # | dump 文件名 | 内容 | 插入位置 |
|---|-------------|------|----------|
| L2-1 | `cpp_cp_input_embeds.bin` | CP prefill 的输入 `[2, D]`（past_hidden + primary_embed） | `RunCpFrame` 开头，primary_embed 查表后 |
| L2-2 | `cpp_cp_frame_{step:03d}_hidden_pre_norm_{j:03d}.bin` | CP Transformer 输出的末帧 hidden（pre-RMSNorm） | 5 层 CP 跑完后，`hidden_step` 提取后 |
| L2-3 | `cpp_cp_frame_{step:03d}_hidden_post_norm_{j:03d}.bin` | CP post norm 后的 hidden | `cp_post.inference()` 后 |
| L2-4 | `cpp_cp_frame_{step:03d}_lm_head_{j:03d}_logits.bin` | `lm_head[j]` 输出的 logits | `lmh.inference()` 后 |
| L2-5 | `cpp_cp_frame_{step:03d}_sampled_token_{j:03d}.bin` | 采样后的 token id（int32） | `CpSampleFromLogits` 后 |
| L2-6 | `cpp_cp_frame_{step:03d}_codec_sum.bin` | 当前帧的 `codec_sum_bf16` | 一帧 CP 结束后 |

> 注：`step` 是 Talker decode 步数（0-based），`j` 是 CP sub-code 步数（0~14）。

**Python 侧需补充的 dump**：
- 在原始 PyTorch `modeling_qwen3_tts.py` 的 `Qwen3TTSTalkerCodePredictorModelForConditionalGeneration.forward()` 中，或 `infer.py` 的 CP 替换实现中，添加等效 dump。
- 由于 `infer.py` 使用的是 AXEngine，其数值已与 AX 模型绑定，因此**Golden 基准应使用原始 PyTorch 模型**（`~/Qwen3-TTS/examples/test_model_12hz_base_single_batch.py`）来 dump。

**通过标准**：
- L2-2 pre_norm hidden: cosine > 0.9999, max_diff < 1e-3
- L2-4 logits: argmax 完全一致（greedy 场景）
- L2-5 sampled token: 在 greedy、相同 temperature/top_p/top_k 下必须一致
- L2-6 codec_sum: cosine > 0.9999（这是 Talker 下一步的输入基础）

---

### 2.3 L3: Talker-CP 耦合数据流验证

**目标**：验证 Talker 每 decode 步的**输入构造逻辑**是否与 Python 一致。

**耦合 spec 关键点**：
1. Talker 每 decode 一步调用一次 CP → 得 `frame_codes`
2. `codec_hiddens = [primary_embed] + [cp_embed[i](frame_codes[i+1]) for i in 0..14]`
3. `inputs_embeds = codec_hiddens.sum(dim=1, keepdim=True)`
4. `inputs_embeds += trailing_text_hidden[:, step]` 或 `tts_pad_embed`
5. Talker model decode → 输出 next hidden / logits

**C++ 侧需补充的 dump（在 `RunTts` decode loop 中）**：

| # | dump 文件名 | 内容 | 插入位置 |
|---|-------------|------|----------|
| L3-1 | `cpp_talker_decode_step{step:03d}_codec_sum.bin` | `codec_sum_bf16`（来自 CP） | `RunCpFrame` 返回后 |
| L3-2 | `cpp_talker_decode_step{step:03d}_trailing_text.bin` | 当前 step 叠加的 trailing_text 或 tts_pad | 叠加前 |
| L3-3 | `cpp_talker_decode_step{step:03d}_inputs_embeds.bin` | 最终输入 Talker 的 `[1, H]` | `next_embed` 构造完成后 |
| L3-4 | `cpp_talker_decode_step{step:03d}_raw_hidden.bin` | Talker decode 输出的 raw hidden | 末层 output 后 |
| L3-5 | `cpp_talker_decode_step{step:03d}_logits.bin` | `post_process` 前的 logits | `llama_post.inference()` 后 |
| L3-6 | `cpp_talker_decode_step{step:03d}_next_token.bin` | 采样后的 primary token id | `post_process` 后 |
| L3-7 | `cpp_talker_decode_step{step:03d}_kv_cache_update.bin` | 可选：dump 某一层的 KV cache 增量 | decode 结束后 |

**Python 侧需补充的 dump**：
- 在 `modeling_qwen3_tts.py:1681-1692` 附近（sub-code embed 拼接 + trailing text 叠加）和 `1713-1727`（Talker decode）之间插入 dump 逻辑。
- 或使用 `infer.py` 中的 `_AxEngineQwen3TTSTalkerModel` 在 forward 时 dump（但注意这是 AXEngine 替换实现，数值已与 AX 绑定，**不能作为 Golden**）。
- **Golden 仍需原始 PyTorch**。

**通过标准**：
- L3-3 inputs_embeds: cosine > 0.9999（这是最关键的耦合点）
- L3-4 raw_hidden: cosine > 0.9999
- L3-5 logits: argmax 一致（greedy 下）
- L3-6 next_token: greedy 下必须一致

---

### 2.4 L4: 端到端输出验证

**目标**：验证完整推理链路输出的 `output_codes` 质量。

**已有能力**：
- `scripts/qwen3_tts_ablation_analysis.py` 可对比 4 种模式（AX/ONNX Talker × AX/ONNX CP）的 output_codes，并生成音频和报告。

**检查点**：

| # | 检查项 | 方法 | 工具 |
|---|--------|------|------|
| L4-1 | 输出 codes 二进制对比 | `diff output_codes_cpp.bin output_codes_py.bin` | `cmp` / `diff` |
| L4-2 | 帧级精确匹配率 | 统计完全一致帧的比例 | `qwen3_tts_ablation_analysis.py` |
| L4-3 | 码本级准确率 | 每 codebook 的 token 一致率 | `qwen3_tts_ablation_analysis.py` |
| L4-4 | 音频听感对比 | 人工试听（spec 要求不使用 ASR） | 播放器 |
| L4-5 | 长度一致性 | 输出帧数差异 | `output_meta.json` |

**通过标准**：
- greedy 解码下：L4-2 frame match rate = 100%，L4-5 帧数完全一致。
- sampling 下：L4-2 > 95%，L4-3 primary codebook > 98%。

---

## 三、Dump 点补充实施指南（C++ 侧）

### 3.1 CP 内部 dump（`RunCpFrame`）

在 `src/runner/LLM_cp_tts_insert.inc` 的 `RunCpFrame` 函数中，使用已有的 `debug_dump_dir_` 机制：

```cpp
// 在 RunCpFrame 开头，若 debug_dump_dir_ 非空，创建子目录
cp_dump_enabled = !debug_dump_dir_.empty();
std::string cp_dir;
if (cp_dump_enabled) {
    cp_dir = debug_dump_dir_.back() == '/' ? debug_dump_dir_ : debug_dump_dir_ + "/";
    cp_dir += "cpp_cp_dump/";
    std::filesystem::create_directories(cp_dir);
}

// 在 CP prefill 输入构造后（j=0, embed_tmp 已填充）
if (cp_dump_enabled && j == 0) {
    // dump cp_input_embeds [seq_len=2, D]
    FILE *fp = fopen((cp_dir + "cpp_cp_input_embeds.bin").c_str(), "wb");
    if (fp) { fwrite(embed_tmp.data(), sizeof(unsigned short), seq_len * D, fp); fclose(fp); }
}

// 在 5 层 CP 跑完后，dump hidden_pre_norm
if (cp_dump_enabled) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/cpp_cp_frame_%03d_hidden_pre_norm_%03d.bin",
             cp_dir.c_str(), talker_step, j);
    FILE *fp = fopen(path, "wb");
    if (fp) {
        std::vector<float> buf(D);
        for (int d = 0; d < D; ++d) buf[d] = bfloat16(hidden_step[d]).fp32();
        fwrite(buf.data(), sizeof(float), D, fp);
        fclose(fp);
    }
}

// 在 cp_post 后，dump hidden_post_norm
if (cp_dump_enabled) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/cpp_cp_frame_%03d_hidden_post_norm_%03d.bin",
             cp_dir.c_str(), talker_step, j);
    FILE *fp = fopen(path, "wb");
    if (fp) {
        std::vector<float> buf(D);
        for (int d = 0; d < D; ++d) buf[d] = bfloat16(hidden_step[d]).fp32();
        fwrite(buf.data(), sizeof(float), D, fp);
        fclose(fp);
    }
}

// 在 lm_head 后，dump logits
if (cp_dump_enabled) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/cpp_cp_frame_%03d_lm_head_%03d_logits.bin",
             cp_dir.c_str(), talker_step, j);
    FILE *fp = fopen(path, "wb");
    if (fp) {
        fwrite(logits_fp32.data(), sizeof(float), logits_n, fp);
        fclose(fp);
    }
}

// 在采样后，dump sampled token
if (cp_dump_enabled) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/cpp_cp_frame_%03d_sampled_token_%03d.bin",
             cp_dir.c_str(), talker_step, j);
    FILE *fp = fopen(path, "wb");
    if (fp) {
        int32_t tok = out_frame_codes[j + 1];
        fwrite(&tok, sizeof(int32_t), 1, fp);
        fclose(fp);
    }
}

// 在 RunCpFrame 结束前，dump codec_sum
if (cp_dump_enabled) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/cpp_cp_frame_%03d_codec_sum.bin", cp_dir.c_str(), talker_step);
    FILE *fp = fopen(path, "wb");
    if (fp) {
        std::vector<float> buf(D);
        for (int d = 0; d < D; ++d) buf[d] = bfloat16(out_codec_sum_bf16[d]).fp32();
        fwrite(buf.data(), sizeof(float), D, fp);
        fclose(fp);
    }
}
```

> **注意**：`RunCpFrame` 目前签名中没有 `talker_step`，需要增加一个 `int frame_idx` 参数用于文件命名。

### 3.2 Talker Decode dump（`RunTts` decode loop）

在 `RunTts` 的 decode loop 中：

```cpp
// CP 返回后
diag_bf16_stats("codec_sum    ", step, codec_sum_bf16.data(), D);

// 构造 next_embed 后（含 trailing_text 叠加）
if (!debug_dump_dir_.empty()) {
    char path[1024];
    snprintf(path, sizeof(path), "%scpp_talker_decode_step%03d_inputs_embeds.bin",
             dir.c_str(), step);
    FILE *fp = fopen(path, "wb");
    if (fp) {
        std::vector<float> buf(D);
        for (int d = 0; d < D; ++d) buf[d] = bfloat16(next_embed[d]).fp32();
        fwrite(buf.data(), sizeof(float), D, fp);
        fclose(fp);
    }
}

// Talker decode 结束后（embed 已更新）
if (!debug_dump_dir_.empty()) {
    char path[1024];
    snprintf(path, sizeof(path), "%scpp_talker_decode_step%03d_raw_hidden.bin",
             dir.c_str(), step);
    FILE *fp = fopen(path, "wb");
    if (fp) {
        std::vector<float> buf(D);
        for (int d = 0; d < D; ++d) buf[d] = bfloat16(embed[d]).fp32();
        fwrite(buf.data(), sizeof(float), D, fp);
        fclose(fp);
    }
}
```

---

## 四、对比脚本矩阵与自动化

### 4.1 脚本矩阵

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `compare_prefill_checkpoints.py` | L1 Talker Prefill 全面对比 | `cpp_dir/`, `py_dir/` | 终端报告 |
| `compare_talker_kvcache.py` | L1 KV cache 逐层对比 | `npy_dir/` | 终端报告 + 偏差层列表 |
| `compare_talker_prefill.py` | L1 AX vs ONNX prefill | `npy_dir/` | 终端报告 + 诊断建议 |
| `compare_cp_dumps.py` | L2 CP 独立对比（需增强） | `cpp_dir/`, `py_dir/` | 终端报告 |
| `compare_talker_decode_full.py` | **L3 Talker Decode 多步对比（新建）** | `cpp_dir/`, `py_dir/` | 逐 step 报告 + 首个发散点 |
| `compare_coupling_flow.py` | **L3 耦合数据流专项验证（新建）** | `cpp_dir/`, `py_dir/` | codec_sum / trailing_text / inputs_embeds 报告 |
| `validate_tts_pipeline.py` | **L0~L4 全链路自动化（新建）** | `cpp_dump/`, `py_dump/` | 综合报告 `validation_report.md` |
| `qwen3_tts_ablation_analysis.py` | L4 端到端消融分析 | `codes_dir/`, `tokenizer_decode` | 音频 + `ablation_report.md` |

### 4.2 主控脚本 `validate_tts_pipeline.py` 设计

```bash
python scripts/validate_tts_pipeline.py \
    --cpp-dir ./debug_bin/cpp_dump \
    --py-dir ./debug_bin/py_dump \
    --mode all  # or: l0_input, l1_prefill, l2_cp, l3_coupling, l4_e2e
```

**内部流程**：
1. 扫描目录，确认各层所需文件是否存在，缺失则报错提示。
2. 按 L0 → L1 → L2 → L3 → L4 顺序执行，**任一环节失败即停止**（单变量原则）。
3. 生成 `validation_report.md`，包含：
   - 每层通过/失败状态
   - 关键指标的数值（cosine, max_diff, argmax_match）
   - 首个发散点的精确定位（文件、step、index）
   - 下一步诊断建议

---

## 五、问题诊断决策树

```
开始验证
│
├─ L0 输入对齐失败？
│  ├─ 检查 infer.py dump 逻辑（bf16 vs fp32 转换、shape）
│  └─ 检查 meta.json 中的 S / hidden_size 是否与 C++ 期望一致
│
├─ L1 Talker Prefill 失败？
│  ├─ Layer-0 即发散 → 输入 embed 或 weights 加载错误
│  ├─ 某中间层 KV cache 发散 → 该层 attention mask / indices 有误
│  ├─ 仅 last hidden/logits 发散 → RMSNorm gamma 或 post head 问题
│  └─ 所有层 KV 都对但 logits 不对 → post_process / temperature / top_p 差异
│
├─ L2 CP 独立失败？
│  ├─ j=0 (prefill) 即发散 → CP prefill input [past_hidden+primary_embed] 不对
│  ├─ j=1 发散但 j=0 对 → CP decode embed lookup（table index）错误
│  ├─ hidden_pre_norm 对但 logits 错 → lm_head[j] 选择错误或 weights 错误
│  └─ logits 对但 sampled token 错 → sampling 参数（temperature/top_k/top_p）不一致
│
├─ L3 耦合失败？
│  ├─ codec_sum 发散 → CP 返回的 embedding 叠加逻辑错误（应用 fp32 累加）
│  ├─ trailing_text 发散 → step 索引越界或 tts_pad_embed 未正确填充
│  ├─ inputs_embeds 对但 raw_hidden 错 → Talker decode KV cache / indices / mask 错误
│  └─ raw_hidden 对但 logits 错 → post head / codec_head 问题
│
└─ L4 端到端失败？
   ├─ 长度不一致 → 提前 hit EOS 或 max_new_tokens 截断差异
   ├─ 帧级完全不匹配但单步都对 → 采样随机种子不一致（sampling 模式下正常）
   └─ primary codebook 对但 sub-codes 错 → CP 耦合问题，回到 L2/L3
```

---

## 六、实施路线图（建议执行顺序）

### Phase 1: 补齐 C++ Dump（1~2 天）
- [ ] 修改 `RunCpFrame` 签名，增加 `int frame_idx`
- [ ] 在 `RunCpFrame` 内添加 L2 所需的 6 类 dump
- [ ] 在 `RunTts` decode loop 内添加 L3 所需的 7 类 dump
- [ ] 编译验证：`./build_ax650.sh`
- [ ] 上传 pyramid 运行，确认 dump 文件生成正确

### Phase 2: 补齐 Python Golden Dump（1 天）
- [ ] 在 `~/Qwen3-TTS/examples/test_model_12hz_base_single_batch.py` 或原始 `modeling_qwen3_tts.py` 中添加等效 dump
- [ ] 确保 Python dump 的文件名与 C++ 侧对齐
- [ ] 运行生成 `py_dump/`

### Phase 3: 编写/增强对比脚本（1~2 天）
- [ ] 新建 `compare_talker_decode_full.py`
- [ ] 新建 `compare_coupling_flow.py`
- [ ] 增强 `compare_cp_dumps.py` 支持 `talker_step` 维度
- [ ] 新建 `validate_tts_pipeline.py`

### Phase 4: 全链路验证与问题修复（持续）
- [ ] 运行 `validate_tts_pipeline.py --mode all`
- [ ] 根据报告定位首个发散点
- [ ] 修复 C++ 或 Python 实现
- [ ] 重新验证，直到 L0~L3 全部通过
- [ ] 最终 L4 人工听感确认

---

## 七、关键注意事项

1. **BF16 → FP32 转换一致性**：C++ 侧使用 `bfloat16(x).fp32()`，Python 侧需使用相同的位运算转换（`(uint16 << 16).view(float32)`），不能用 `torch.bfloat16` 的隐式转换（可能有舍入差异）。
2. **文件命名约定**：C++ 和 Python 的 dump 文件名必须严格对齐（大小写、下划线、零填充位数），否则对比脚本会报 missing。
3. **采样参数一致性**：如果 Python 使用 greedy（`do_sample=False`），C++ 必须也使用 greedy（`temperature=0` 或等价逻辑），否则 token 差异是预期内的。
4. **KV cache 格式**：C++ 侧 dump 的 KV cache 是 `[seq_len, kv_dim]` FP32，Python 侧需确保相同 layout。
5. **Step 索引**：Talker `generation_step` 从 0 开始（prefill 后为 0），CP `generation_steps` 也从 0 开始。任何 off-by-one 都会在 L3  inputs_embeds 中暴露。

---

## 八、修订记录

| 日期 | 版本 | 说明 |
|------|------|------|
| 2026-05-20 | v1.0 | 初始版本，基于 coupling spec 和 debug 文档，设计 L0~L4 分层验证体系、dump 点补充方案、自动化脚本矩阵和问题诊断决策树。 |
