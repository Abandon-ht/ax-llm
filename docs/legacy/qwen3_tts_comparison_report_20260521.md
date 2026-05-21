# Qwen3-TTS C++ vs Python 推理一致性对比报告

> **分析日期**: 2026-05-21
> **数据来源**: `debug_bin/cpp_dump/` (C++) vs `debug_bin/py_dump/` (Python)
> **参考文档**: `docs/scripts_comparison_guide.md`, `docs/qwen3_tts_cpp_python_debug_summary.md`

---

## 1. 执行脚本总览

| 脚本 | 状态 | 关键结论 |
|------|------|----------|
| `compare_prefill_checkpoints.py` | ✅ 已执行 | Raw hidden bit-exact; Normed hidden 有 0.125 差异 |
| `compare_bf16_bytes.py` | ✅ 已执行 | BF16 字节级 EXACT MATCH |
| `compare_cp_dumps.py` | ✅ 已执行 | CP step 2 首次发散 |
| `compare_coupling_flow.py` | ✅ 已执行 | Step 0 codec_sum 即发散; residual(trail) 高度一致 |
| `compare_talker_decode_full.py` | ✅ 已执行 | Step 0 全面发散 |

---

## 2. L0/L1 — Talker Prefill 阶段

### 2.1 Last Raw Hidden (pre-RMSNorm)

```
cosine=1.00000000  max_diff=0.00000000  mean_diff=0.00000000
argmax: cpp=635, py=635 ✓
```

**结论**: C++ 与 Python 的 Talker axmodel 输出（raw hidden）**逐位精确匹配**。

### 2.2 BF16 字节级验证

```
fp32 max_diff=0.00000000
bf16 bytes: EXACT MATCH
```

**结论**: C++ `debug_talker_prefill_last_hidden_ax.bin` 与 Python `prefill_last_raw_hidden.bin` 在 bf16 字节级别完全一致，确认 axmodel runtime 无差异。

### 2.3 Last Normed Hidden (post-RMSNorm)

```
cosine=0.99999642  max_diff=0.12500000  mean_diff=0.00927627
argmax: cpp=635, py=635 ✓
```

**结论**: RMSNorm 后存在 `max_diff=0.125` 的系统性数值差异。根因锁定为 **C++ 手写 `rmsnorm_bf16` 与 Python PyTorch RMSNorm 实现不一致**。

---

## 3. L2 — CP (Code Predictor) 独立一致性

### 3.1 Hidden State 传播链

| Step | pre_norm cos | pre_norm max_diff | post_norm cos | post_norm max_diff |
|------|-------------|-------------------|---------------|--------------------|
| 0 | 0.999793 | **0.250** | 0.999737 | 0.219 |
| 1 | 0.999760 | **0.312** | 0.999785 | 0.625 |
| 2 | 0.999783 | **0.500** | 0.999776 | 0.250 |
| 3 | 0.969188 | **2.625** | 0.970545 | 3.307 |
| ... | ... | ... | ... | ... |

### 3.2 Sampled Tokens (Greedy)

| Step | C++ Token | Python Token | lm_head max_diff | 判定 |
|------|-----------|--------------|------------------|------|
| 0 | 117 | 117 | 0.280 | ✅ MATCH |
| 1 | 604 | 604 | 0.199 | ✅ MATCH |
| 2 | **1349** | **279** | 0.298 | ❌ **DIVERGE** |
| 3 | 879 | 2047 | 4.440 | ❌ DIVERGE |
| ... | ... | ... | ... | ❌ 级联恶化 |

**关键发现**:
- Step 0~1 token 完全相同，说明 CP prefill 输入 embed 一致。
- Step 2 的 C++ top1=1349(8.898) vs Python top1=279(8.889)，差仅 **0.009**。
- `max_diff=0.298` 的 logit 误差恰好让 argmax 翻转。
- Step 3+ hidden state 迅速恶化（cos 跌至 0.88~0.97）。

---

## 4. L3 — Talker+CP 耦合流

### 4.1 逐步对比 (Step 0~19)

| Step | codec_sum cos | inputs_embeds cos | residual(trail) cos | residual max_diff |
|------|---------------|-------------------|---------------------|-------------------|
| 0 | 0.898850 | 0.932558 | **0.999984** | 0.007324 |
| 1 | 0.257889 | 0.522276 | **0.999985** | 0.006836 |
| 2 | 0.180358 | 0.454599 | **0.999982** | 0.001953 |
| ... | ... | ... | **~0.99998** | <0.007 |

**结论**:
- `codec_sum` 从 step 0 即严重发散，是 Talker decode 输入错误的主因。
- `residual(trail)` = `inputs_embeds - codec_sum` 高度一致（cos≈0.99998），说明 **trailing_text_hidden / tts_pad_embed 的注入逻辑完全正确**。
- 差异唯一来源是 `codec_sum`（CP 输出的 sub-code embedding 累加）。

---

## 5. L3 — Talker Decode 全量

### 5.1 Step 0 详细数据

```
[Talker Decode Step 0]
  [codec_sum]       cos=0.898850  max_diff=0.111328  DIVERGE
  [inputs_embeds]   cos=0.932558  max_diff=0.111084  DIVERGE
  [raw_hidden]      cos=0.032728  max_diff=100.527   DIVERGE
  [logits]          cos=0.167188  max_diff=36.937    DIVERGE
  [next_token]      cpp=1737  py=1174              DIVERGE
```

### 5.2 发散演进

| Step | codec_sum cos | raw_hidden cos | next_token 判定 |
|------|---------------|----------------|-----------------|
| 0 | 0.898850 | 0.032728 | ❌ DIVERGE |
| 1 | 0.257889 | 0.014224 | ❌ DIVERGE |
| 2 | 0.180358 | -0.093616 | ❌ DIVERGE |
| ... | 持续低相关 | ~0.01~-0.09 | ❌ 全程发散 |

**结论**: Talker decode 从第 0 步起即全面跑偏，raw_hidden 与 logits 几乎不相关。

---

## 6. 根因链路总结

```
Talker Prefill
    │
    ▼
C++ rmsnorm_bf16  vs  Python PyTorch RMSNorm
    │                           │
    └── max_diff = 0.125 ───────┘
    │
    ▼
last_normed_hidden (CP past_hidden) 有 0.125 差异
    │
    ▼
CP Prefill (step 0)  →  pre_norm diff=0.25
    │
CP Decode (step 1)   →  pre_norm diff=0.31
    │
CP Decode (step 2)   →  pre_norm diff=0.50
    │
lm_head[2] logits: max_diff=0.298
    │
    ▼
排序翻转: C++ argmax=1349(8.898) vs Python argmax=279(8.889) 差仅 0.009
    │
    ▼
Step 2 token 错误 → codec_sum 错误 → Talker decode inputs_embeds 错误
    │
    ▼
Talker decode step 0: raw_hidden cos=0.033, next_token 1737 vs 1174
    │
    ▼
错误累积循环 → 全程发散
```

---

## 7. 判定与建议

| 检查项 | 状态 | 说明 |
|--------|------|------|
| Talker axmodel 精度 | ✅ 无差异 | raw hidden bit-exact |
| RMSNorm 实现 | ⚠️ 有差异 | max_diff=0.125，为级联放大起点 |
| CP axmodel 精度 | ✅ 无差异 | 同输入下同输出 |
| CP 输入构造 | ✅ 一致 | past_hidden + last_id_hidden 正确 |
| trailing_text 注入 | ✅ 一致 | residual 高度匹配 |
| greedy 采样逻辑 | ✅ 一致 | step 0~1 token match |

**根因**: **C++ 手写 `rmsnorm_bf16` 与 PyTorch RMSNorm 的数值差异**，经 CP Transformer 5 层放大后，在 lm_head[2] 的敏感竞争区间触发排序翻转，导致后续全部跑偏。

**修复建议**（与文档一致）:
1. **方案 A（推荐）**: 将 Talker RMSNorm 导出为 axmodel，统一走 axmodel 执行。
2. **方案 B**: C++ 调用 PyTorch C++ API (libtorch) 执行 RMSNorm。
3. **方案 C**: 对齐 `rmsnorm_bf16` 的求和顺序/向量化行为，使 diff < 1e-3。

---

*报告基于 `scripts/compare_*.py` 输出自动生成。*
