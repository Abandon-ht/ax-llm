# Qwen3-TTS CP 调试记录

## 调试命令

### 1. 构建（x86 交叉编译）
```bash
cd /home/m5stack/Workspace/AXERA/ax-llm
./build_ax650.sh
```
输出二进制：`build/install/bin/qwen3_tts_debug`

### 2. 部署到 AX650 板子
```bash
scp build/install/bin/qwen3_tts_debug pyramid:/root/
```
板子地址：`root@192.168.20.54`（Host `pyramid`）

### 3. 在板子上运行
```bash
ssh pyramid "cd /root && ./qwen3_tts_debug /root/rsp/Qwen3-TTS-12Hz-0.6B-Base-AX650/talker/ tts_embeds/ [max_new_tokens]"
```

**参数说明：**
| 参数 | 说明 | 示例 |
|------|------|------|
| `model_dir` | Talker 模型目录（CP 目录自动推导为同级 `code-predictor/`） | `/root/rsp/Qwen3-TTS-12Hz-0.6B-Base-AX650/talker/` |
| `npy_dir` | 预计算预填充数据目录 | `tts_embeds/` |
| `max_new_tokens` | 最大生成帧数（可选，默认 128） | `128` |

---

## 输入数据

### 预填充数据（`npy_dir` 目录）

| 文件 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `prefill_embeds.bin` | `[85, 1024]` | BF16 | 预计算的 85 个 token 嵌入 |
| `meta.json` | - | JSON | 包含 `S`, `S0`, `hidden_size`, `audio_token_id`, `audio_slots` 等元数据 |

**Prefill 布局（85 tokens）：**
- `trailing_start = 7`，即 `text[0]` 在位置 7
- 解码时 `txt_pos = trailing_start + step`，`step` 从 0 开始

### 模型文件结构

**Talker:**
- 28 层 transformer (`qwen3_tts_talker_p128_l*.axmodel`)
- Post (`qwen3_tts_talker_post.axmodel`)
- Codec embed table (`model.embed_tokens`，vocab=3072）

**Code Predictor (CP):**
- 5 层 transformer (`qwen3_tts_talker_code_predictor_p64_l*.axmodel`)
- Post (`qwen3_tts_talker_code_predictor_post.axmodel`，输出名 `output_norm`)
- 15 个 lm_head (`code_predictor_lm_head_*.axmodel`，输出 FP32 [1,1,2048]）
- 15 个 embed table (`talker.code_predictor.model.codec_embedding.*.weight.bfloat16.bin`，2048×1024)

---

## 输出数据

### 标准输出
示例：
```
max_new_tokens : 128
[INFO] Running TTS decode (S=85 tokens, max_new_tokens=128)...
frame=0 primary=922 res_0=240 res_1=18 res_2=20 ... res_14=13
frame=1 primary=1995 res_0=1764 res_1=37 ...
...
frame=127 primary=72 res_0=1975 ... res_14=181
[TIME]   28464.47 ms
[RESULT] frames=128
```

### 输出文件（`npy_dir` 目录）

| 文件 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `output_codes.bin` | `[num_frames, 16]` | int32 | 音频码本，每帧 16 个 codebook |
| `output_meta.json` | - | JSON | 元数据：shape、num_codebooks、codec_eos_token_id=2150 |

**码本格式：**
- `codes[0]`：primary code（Talker 输出，vocab=3072）
- `codes[1..15]`：residual codes（CP 输出，vocab=2048）

---

## 关键调试日志与含义

### CP 初始化日志
```
CP layer0 groups=2           # CP 只有 2 个 group（decode + prefill）
Talker layer0 groups=6       # Talker 有 6 个 group
  group 0 inputs:            # decode 组：indices=1, input=1x1x1024
  group 1 inputs:            # prefill 组：indices=64, input=1x64x1024
CP post in[0]: name=input shape=1x1x1024
CP post out[0]: name=output_norm shape=1x1x1024   # 注意不是 "output"
CP lm_head_0 in[0]: name=input shape=1x1x1024 nSize=4096   # 输入是 BF16
CP lm_head_0 out[0]: name=output shape=1x1x2048 nSize=8192 # 输出是 FP32！
CP prefill_gid=1 token_num=64 kv_cache_num=0
```

### CP 推理日志（精简后）
```
RunCpFrame lm_head j=0 logits_n=2048 max_logit=0.584712 max_idx=240
```
- `logits_n=2048`：CP vocab 大小
- `max_logit`：最大 logit 值（正常应非零）
- `max_idx`：greedy decode 的 token id

**异常指标：**
| 现象 | 含义 | 排查方向 |
|------|------|----------|
| `max_logit=0.000000` | CP 输出全零 | 检查 `inference` 返回值是否为 -1；检查 K/V cache 内存是否分配 |
| `max_idx=0` | 始终选 token 0 | 同上，或检查 embed 输入是否全零 |
| `inference done ret=-1` | 模型推理失败 | K/V cache `phyAddr=0`，见下方根因 |

---

## 已修复的 Bug

### Bug 1：CP `inference` 返回 -1，输出全 0

**根因：** `ax_runner_ax650::sub_init()` 中内存共享逻辑：
```cpp
if (io_count > 2) {
    // 将 last group 的 K_cache/V_cache 指向 group 0 的共享内存
}
```
Talker 有 6 个 group（`io_count=6`），条件成立。
CP 只有 2 个 group（`io_count=2`），条件不成立 → CP group 1 的 `K_cache`/`V_cache` `phyAddr=0` → `AX_ENGINE_RunGroupIOSync` 返回 -1。

**修复：** `src/runner/ax_model_runner/ax_model_runner_ax650.cpp`
```cpp
// if (io_count > 2)   // 旧代码
if (io_count >= 2)      // 新代码
```

**验证：** `inference done ret=0`，`max_logit` 从 0.0 变为 0.58+。

### Bug 2：CP post 输出张量名错误

**根因：** CP post 模型的输出名是 `output_norm`，不是 `output`。

**修复：** `RunCpFrame` 中：
```cpp
auto &t_out = cp_post.get_output("output_norm");  // 不是 "output"
```

### Bug 3：CP lm_head 输出类型错误

**根因：** CP lm_head 输出 `nSize=8192` 对应 2048 个 **FP32**（`2048*4=8192`），不是 BF16。

**修复：** `RunCpFrame` 中读取 lm_head 输出时：
```cpp
std::vector<float> logits_fp32(logits_n);
llm_d2h(logits_fp32.data(), LLM_RADDR(t_out), logits_n * sizeof(float), ...);
```

### Bug 4：Talker decode 后 `embed` 未更新（USE_AXCL 路径）

**根因：** `USE_AXCL` 路径的 Talker decode 循环中没有将最后一层输出复制到 `embed` 变量。

**修复：** 在最后一层输出复制到 post input 后，追加：
```cpp
llm_d2h(embed.data(), LLM_RADDR(cur_out), embed.size() * sizeof(unsigned short), devid);
```

---

## 性能数据

| 指标 | 数值 |
|------|------|
| 生成 128 帧耗时 | ~28.3 秒 |
| 速度 | ~4.55 token/s |
| 每帧推理次数 | 21 次（5 CP layers + 1 post + 15 lm_heads） |

**瓶颈：** CP 每帧对每个残差 step 都执行 full prefill（seq_len 2→17），且重置 KV cache。

**日志级别影响：**
- 全量 debug 日志（~2000 条/128帧）：~30.2 秒
- 精简日志（仅 frame 输出）：~28.3 秒
- 日志开销约 2 秒，提升约 7%
