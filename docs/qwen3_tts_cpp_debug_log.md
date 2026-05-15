# Qwen3-TTS C++ 推理调试纪要

> 记录 C++ 推理与 Python 推理输出不一致 / 生成音频胡言乱语的逐步调试过程。
> 时间：2026-05-15。涉及文件：`src/runner/LLM_cp_tts_insert.inc`、`tools/qwen3_tts_infer.cpp`、参考 `scripts/infer.py`、`/home/m5stack/Workspace/Qwen3-TTS/qwen_tts/core/models/modeling_qwen3_tts.py`。

---

## 一、现象

- **同模型同输入**：Python `scripts/infer.py` 走 axengine 路径生成的音频正常；C++ `qwen3_tts_infer` 生成的音频是连续胡言乱语 / 杂噪。
- **采样参数（temperature、top_k、top_p、seed）只影响总帧数**，不影响音频内容（永远是乱码）。
- **关闭采样（greedy）**：Python 直接陷入确定性死循环跑满 `max_new_tokens`，无法做为基准。
- 第一个 `primary_code`（codes[0]）会随采样参数变，第二个 code（codes[1] from CP lm_head[0]）在多次运行中很稳定（含义是 CP prefill 路径正常），但从 codes[2] 开始就是噪声分布。

---

## 二、模型架构关键事实

| 项 | 值 |
|---|---|
| Talker 层数 | 28（`axmodel_num: 28`） |
| Talker `tokens_embed_num` | 3072（codec primary vocab） |
| Talker hidden size | 1024 |
| Talker decode `kv_cache_num` | 2048 |
| Talker prefill_token_num | 128 |
| Talker prefill groups | gid 1..5，`kv_cap` 分别 0 / 128 / 256 / 384 / 512 |
| Talker decode indices | `[1, 1]`，单值（MRoPE 3 维同值，axmodel 内部广播） |
| Talker decode mask | `[1, 1, 2049]`（2048 cache + 1 self 槽） |
| Talker `full_attention_interval` | 0（无 sliding-window 层） |
| Talker `is_linear_layer` 数 | 0（28 层全是 full-attention） |
| CP 层数 | 5 |
| CP `kv_cache_num` | 128（`mask: 1x1x129`） |
| CP `cp_prefill_token_num` | 64 |
| CP lm_head 输出 vocab | 2048 |
| CP residual codec embed 数 | 15 张（codes[1..15]），每张 `2048 × 1024` bf16 |
| 非流式 `next_embed` 公式 | `codec_sum + tts_pad_vec`（HF: line 1707） |
| Python 端 `past_hidden` | `hidden_states[:, -1:, :]`（post-norm last hidden） |

---

## 三、确认排除的假设

### #1 Talker decode indices 多行未填满（候选 #1）

**结论：不成立**。日志 `talker decode indices nSize=4 (elems=1)`，单元素，C++ 写 4 字节正确。

### #2 Decode mask buffer 与 axmodel 张量尺寸不符（候选 #2）

**结论：不成立**。日志 `mask nSize=4098 (elems=2049, host mask elems=2049)`，C++ 主机 buffer 与 axmodel 张量精确匹配。

### #3 `is_linear_layer` 分支擦掉 KV 历史（候选 #3）

**结论：不成立**。日志 `is_linear_count=0, full_attention_interval=0`，没有任何 talker 层走 linear 分支。

### #4 非流式 `next_embed` 公式错误

**结论：不成立**。从 `modeling_qwen3_tts.py:1697-1707` 确认：
```python
inputs_embeds = codec_hiddens.sum(1, keepdim=True)
if generation_step < trailing_text_hidden.shape[1]:
    inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step].unsqueeze(1)
else:
    inputs_embeds = inputs_embeds + tts_pad_embed
```
并且非流式分支（`modeling_qwen3_tts.py:2244`）把 `trailing_text_hidden = tts_pad_embed`，所以每一步都等价于 `codec_sum + tts_pad_vec`，C++ 一致。

### #5 采样器算法 / 随机数源（之前误判过的候选）

**结论：不是主因**。即使两端用不同 RNG（numpy MT19937 vs `std::mt19937`），分布层面音频质量应当相近；现象是"质量整体差到无意义"，说明不是采样问题。

### #6 RMSNorm 精度差异

**结论：影响微小，不是主因**。Python `nn.RMSNorm` vs C++ `rmsnorm_bf16`，eps 都是 `1e-6`，gamma 来源同一个 `talker.model.norm.weight.bfloat16.bin`。精度差异 cosine 应 >0.999，不足以解释完全的胡言乱语。

---

## 四、已实施的修复

### 修复 1：CP decode mask（已修，未解决问题）

**位置**：`src/runner/LLM_cp_tts_insert.inc`，`RunCpFrame` 内 CP decode 路径。

**原 bug**：
```cpp
for (int i = 0; i < mask_fill; ++i) {
    mask_tmp[i] = (i <= history_len) ? bf16(0) : bf16(-65536);  // off-by-one
}
// mask_tmp[mask_cap - 1] 未单独处理 → self 槽被屏蔽
```

**对照 Python（`infer.py:423-428`）**：
```python
mask_2d = np.where(col_ids < row_ids, 0.0, -65536.0)  # 严格 <
mask_2d[:, -1] = 0.0                                   # 末位 self 永远可见
```

**修复后**：
```cpp
for (int i = 0; i < mask_fill; ++i) {
    mask_tmp[i] = (i < history_len) ? bf16(0) : bf16(-65536);
}
if (mask_fill > 0) mask_tmp[mask_fill - 1] = bf16(0);
```

**验证结果**：codes[2..15] 数值确实变了（说明 mask 修改有效），但音频整体仍为乱码。这个 bug 真实存在，但**不是唯一的根因**。

### 修复 2：Decode indices / CP indices 多行写入兼容（已修，本模型为 no-op）

**位置**：Talker decode 的 AXCL / AX650 两个路径，以及 CP 的 indices 写入。

**思路**：检测 `t_idx.nSize / sizeof(uint)`，若 > 1 则把 `indices` 值写入所有行，避免未来某些 MRoPE 多行模型的兼容问题。

**实测**：本模型 `decode indices elems=1`，分支退化为单值写入，与原行为等价。**保留作为防御性代码**。

### 修复 3（最关键）：Prefill chunk 间 KV history 没有传递

**位置**：`RunTtsWithCpCallback` 内 prefill chunk 循环（`LLM_cp_tts_insert.inc:555` 附近）。

**Bug 描述**：

C++ 多 chunk prefill 时，每个 chunk 使用一个**不同的 `prefill_grpid`**（gid=1 / gid=2 / ...，按 history_len 选择能容纳的最小 KV cap）。但 axengine 的 K_cache 输入是按 grpid 分配的独立 buffer：

```
gid=1: pre_k.nSize=2048    (1 槽)    - 给 chunk 0
gid=2: pre_k.nSize=262144  (128 槽)  - 给 chunk 1
gid=3: pre_k.nSize=524288  (256 槽)  - 给 chunk 2
gid=4: pre_k.nSize=786432  (384 槽)  - 给 chunk 3
gid=5: pre_k.nSize=1048576 (512 槽)  - 给 chunk 4
```

原 C++ 代码在每个 chunk 推理完后做两件事：
```cpp
// WRITE 1: out_k → pre_k of CURRENT grpid（同 grpid，无意义，且会越界）
llm_d2d(WADDR(pre_k) + kv_off, RADDR(out_k), kv_sz, devid);
// WRITE 2: out_k → dec_k of decode_grpid（这是正确的）
llm_d2d(WADDR(dec_k) + kv_off, RADDR(out_k), kv_sz, devid);
```

WRITE 1 写入的是**当前 grpid 的 pre_k**，但下一 chunk 使用**别的 grpid**，所以 WRITE 1 写入的数据再也不会被读到。**而下一 chunk 的 `pre_k` 是未初始化的（多半全 0）**，意味着 chunk 2..N 的 prefill 是在"看不到历史 K"的前提下做的，attention 输出从位置 128 开始就错了。

对照 Python `StaticTalkerLayerRunner.prefill()`：
```python
for chunk_idx in range(chunk_count):
    if chunk_idx == 0:
        k_input = np.zeros(...)
    else:
        k_input = k_caches[layer_idx][:, :start, :]  # ← 显式取上一 chunk 的输出
    outputs = session.run({"K_cache": k_input, ...}, ...)
    k_caches[layer_idx][:, start:cache_end, :] = k_prefill[...]   # 累积
```

Python 端使用 host 侧 `k_caches` 列表显式累积并显式喂回 axengine，**C++ 完全缺失这一步**。

**修复方案（已应用）**：

```cpp
// 1) 进入 chunk p>0 的每一层前，从 decode_grpid 的 dec_k 复制前 history_len 槽到当前 prefill_grpid 的 pre_k：
if (p > 0) {
    auto &dec_k_src = lyr.layer.get_input(decode_grpid, "K_cache");
    auto &dec_v_src = lyr.layer.get_input(decode_grpid, "V_cache");
    auto &pre_k_dst = lyr.layer.get_input(prefill_grpid, "K_cache");
    auto &pre_v_dst = lyr.layer.get_input(prefill_grpid, "V_cache");
    size_t history_bytes = (size_t)history_len * _attr.kv_cache_size * sizeof(unsigned short);
    history_bytes = std::min(history_bytes, (size_t)pre_k_dst.nSize);
    if (history_bytes > 0) {
        llm_d2d(LLM_WADDR(pre_k_dst), LLM_RADDR(dec_k_src), history_bytes, devid);
        llm_d2d(LLM_WADDR(pre_v_dst), LLM_RADDR(dec_v_src), history_bytes, devid);
    }
}

// 2) 推理完后的 WRITE 1（写回当前 grpid pre_k）加越界保护：
{
    size_t pre_k_max = (size_t)pre_k.nSize;
    size_t pre_off_b = (size_t)kv_off * sizeof(unsigned short);
    if (pre_off_b + kv_sz <= pre_k_max) {
        llm_d2d((unsigned short *)LLM_WADDR(pre_k) + kv_off, LLM_RADDR(out_k), kv_sz, devid);
        llm_d2d((unsigned short *)LLM_WADDR(pre_v) + kv_off, LLM_RADDR(out_v), kv_sz, devid);
    }
}
```

为什么用 `dec_k` 作为 history 源：WRITE 2 已经把每个 chunk 的输出 K_cache_out 按 `kv_off = history_len * kv_cache_size` 偏移写入 `dec_k`。所以 chunk p 开始时，`dec_k[0..p*128 - 1]` 已经是前 p 个 chunk 累积的正确 K 历史。直接从 `dec_k` d2d 到 `pre_k` 一次拷贝即可。

---

## 五、还未验证的潜在 bug 列表（供下次参考）

按可能性排序：

### A. 修复 3 后若仍胡言乱语，下一可疑点：

#### A1. CP 第一帧 `txt_hidden` 双重 norm？

C++ `step == 0` 时使用 `all_prefill_hidden[input_embed_num - 1]`，但 `all_prefill_hidden` 已经在 prefill 后被 in-place RMSNorm 过（`LLM_cp_tts_insert.inc:626-634`）。Python 端的 `_to_hidden_tensor` 每次 forward 调用都返回 normed，所以 CP 第一帧的输入也是 normed。这部分对齐。但要注意 `embed`（最后一个 token 的 raw_hidden，用作 talker_post 输入）不能被 normed —— 看代码确实没动 `embed`，只动了 `all_prefill_hidden`。✓

#### A2. Talker post head 的 dtype 路径

C++ 把 talker post 的输出当 bf16 raw 读，然后由 `post_process` 转 fp32 + 采样。Python axengine 直接 `outputs[out_key].astype(np.float32)`。如果 axmodel 实际输出已经是 fp32，C++ 当 bf16 读会全错。**核对 `talker_post.axmodel` 的 output tensor 真实 dtype**（可以打印 `t_out.nSize` 跟 `tokens_embed_num` 关系，bf16 时 size = vocab × 2，fp32 时 size = vocab × 4）。

诊断方法：
```cpp
auto &t_out = llama_post.get_output("output");
ALOGI("[diag] talker_post output nSize=%d, vocab=%d, dtype-bf16-bytes=%d, dtype-fp32-bytes=%d",
      t_out.nSize, _attr.tokens_embed_num,
      _attr.tokens_embed_num * 2, _attr.tokens_embed_num * 4);
```

#### A3. `embed_selector.getByIndex` 的步长

如果 `tokens_embed_size` 在 init 时与实际 bin 文件 stride 不符（罕见），`primary_embed` 全错。已经核对过 config.json：`tokens_embed_size: 1024` 与文件 `3072 × 1024 × 2 = 6MB` 一致。✓

#### A4. RMSNorm 的 eps

C++ 硬编码 `1e-6`。Qwen3 系列的 `rms_norm_eps` 通常也是 `1e-6`，但**实际值取决于 HF model config**。要核对 `~/Qwen/Qwen3-TTS-12Hz-0.6B-Base/config.json` 的 `talker_config.rms_norm_eps` 字段。

#### A5. CP 第一帧后续 res_code 的 embedding 表索引

C++：`cp_embed_tables[j]` 对应 codes[j+1]（j ∈ [0, 14]）。Python：`embedding_tables[lm_step - 1]` 对应 `generated_ids[-1]`。两边索引顺序一致，已 verify。✓

### B. 其它需要更深 dump 才能验证的：

- 用 C++ dump `all_prefill_hidden`（fp32 [197, 1024]），与 Python 的 `trailing_text_hiddens.bin` 做 cosine 相似度比较，定位 prefill 之前/之后第一个分叉点。
- 在 Python 端额外 dump frame 0 的 `codec_sum` 与 frame 1 的 `next_embed`，与 C++ 的同名 dump 做 cosine 比较。
- 单独跑 talker 第一步 decode，固定 next_embed 输入（人造或 dump 自 Python），看 C++ 的 talker decode 输出是否与 Python 一致。

---

## 六、调试过程的关键诊断指令

### C++ 端启用日志（已加在代码里，无 CLI flag）

```bash
./build/install/bin/qwen3_tts_infer rsp/Qwen3-TTS-12Hz-0.6B-Base-AX650/talker/ debug_bin/ --max_new_tokens 2048 2>&1 | grep -E '\[diag\]'
```

输出涵盖：
- 每个 prefill_grpid 的 `pre_k.nSize` / `out_k.nSize` / kv_cap
- 每层 `is_linear_layer` 状态、`full_attention_interval`
- talker decode indices / mask 张量尺寸 vs host buffer
- CP prefill / decode indices 尺寸
- 前 2 帧的 `txt_hidden_in / codec_sum / tts_pad_vec / next_embed / decode_raw_h` 量级统计

### Python 端 dump 中间数据

```bash
cd ~/Workspace/AXERA/ax-llm
bash scripts/infer.sh
# 产物：
#   debug_bin/prefill_embeds.bin       (S × H bf16 raw)
#   debug_bin/prefill_embeds_bf16.bin
#   debug_bin/trailing_text_hiddens.bin    (S × H fp32，normed prefill hidden)
#   debug_bin/trailing_text_hiddens_bf16.bin
#   debug_bin/tts_pad_vec.bin              (int32 header + H fp32)
#   debug_bin/tts_pad_vec_bf16.bin
#   debug_bin/meta.json
#   scripts/output_codes.bin               (N × 16 int32，Python 实际生成的 codes)
#   scripts/output_meta.json
```

### 关键张量量级参考（Python 正常时应当看到的范围）

| 张量 | 形状 | 预期 L2 范围 | 备注 |
|---|---|---|---|
| `txt_hidden_in` | [1024] bf16 | 30 ~ 200 | 取决于 gamma；本模型实测 frame 0 = 172，看起来在范围内 |
| `codec_sum` | [1024] bf16 | 1 ~ 5 | 16 个 codec embed 求和，单 embed 量级 ~0.1 |
| `tts_pad_vec` | [1024] bf16 | 1 ~ 3 | 固定值，frame 间不变 |
| `next_embed` | [1024] bf16 | 与 codec_sum 相近 | codec_sum + tts_pad_vec |
| `decode_raw_h` | [1024] bf16 | 10 ~ 100 | last layer raw output |

**红旗信号**：
- 某个张量 L2 = 0 或 NaN（除以零 / 未初始化）
- 跨帧量级跳变 10× 以上（hidden state 漂走）
- `codec_sum` 量级在 30+（codec embedding 索引错误）
- `decode_raw_h` 量级 > 1000（attention 爆炸，KV cache 错乱的典型征兆）

---

## 七、文件改动清单

`src/runner/LLM_cp_tts_insert.inc`：
1. `RunCpFrame` CP 每层 indices 多行写入兼容（防御性）
2. `RunCpFrame` CP decode mask 修正（`<=` → `<`，新增末位 self 槽）
3. `RunTtsWithCpCallback` 加入诊断日志（is_linear_layer 普查、prefill gid 的 pre_k 尺寸、前 2 帧张量量级统计）
4. `RunTtsWithCpCallback` Talker decode AX650 + AXCL 路径 indices 多行写入兼容（防御性）
5. `RunTtsWithCpCallback` Prefill chunk 间 KV history 从 dec_k 复制到下一 chunk 的 pre_k（**核心修复**）
6. `RunTtsWithCpCallback` Prefill 内 WRITE 1（out_k → 当前 grpid pre_k）加越界保护

无 `git add`、无 `git commit`、无 `git stash`。所有改动留在工作树。

---

## 八、给"下次的我"的建议

1. **不要再从采样切入**：Qwen3-TTS 在 codec_sum 求和融合架构下，greedy 模式会确定性死锁（见 `qwen3_tts_single_sampling_coupling_analysis.md`）。Python 自身 greedy 不能跑，没有 baseline。
2. **优先看张量量级**：在前 2 帧的关键张量上加 L1/L2/min/max 打印，比逐元素 dump 更高效定位"哪一层炸了"。
3. **多 chunk prefill 必须显式传递 KV history**：axengine 的多 grpid 各自有独立 K_cache 输入 buffer，跨 grpid 不会自动继承数据。这是 C++ 实现里最容易漏掉的设计陷阱。
4. **`is_linear_layer` 路径是潜在地雷**：现行模型不触发，但只要换一个 sliding-window 架构就立即坏。这块代码逻辑跟 Python 的差异需要单独评估。
5. **如果 frame 0 的 codes[0] / codes[1] 正常但音频还乱**：基本能锁定在"prefill chunk 1+ 的 KV"或"talker decode KV 写入" —— 这两条是从 frame 1 开始才级联放大的环节。
6. **Talker post 输出 dtype 一定要核对**：bf16 当 fp32 读（或反之）会导致整个采样分布完全偏移，是 silent corruption 的常见来源。
