# Scripts 对比脚本使用指南

> **版本**: v1.0  
> **日期**: 2026-05-20  
> **用途**: 说明 `scripts/compare_*.py` 各脚本的作用、参数及使用方法。

---

## 0. 脚本总览

| 脚本 | 对比层级 | 作用 | 典型调用时机 |
|------|----------|------|--------------|
| `compare_prefill_checkpoints.py` | L0-L1 | Prefill 全量检查点（输入、layer0、KV cache、last hidden、logits） | prefill 后立即 |
| `compare_bf16_bytes.py` | L1 | Prefill last hidden 的 **bf16 字节级**精确对比 | 排查 bf16 转换差异 |
| `compare_talker_prefill.py` | L1 | AX Talker vs ONNX Talker prefill 输出 | ablation 模式 |
| `compare_cp_dumps.py` | L2 | CP (Code Predictor) 独立一致性 | CP 单步验证 |
| `compare_coupling_flow.py` | L3 | Talker-CP 耦合流（codec_sum / inputs_embeds / residual） | decode 发散排查 |
| `compare_talker_decode_full.py` | L3 | Talker decode 每步全量（codec_sum / inputs / hidden / logits / token） | end-to-end 验证 |
| `compare_talker_decode_inputs.py` | L3 | Talker decode **输入状态**（KV cache / indices / mask） | decode raw_hidden 发散根因排查 |
| `compare_talker_kvcache.py` | - | AX vs ONNX KV cache 层间对比 | KV cache 精度排查 |

---

## 1. `compare_prefill_checkpoints.py`

**作用**：对比 C++ 与 Python 在 Talker **Prefill 阶段**的全量检查点，覆盖 L0（输入）到 L1（logits）。

**对比项**：
1. prefill input (`talker_prefill_input.bin` vs `prefill_embeds.bin`)
2. layer0 output (`talker_layer0_prefill_output.bin` vs `python_layer0_prefill_output.bin`)
3. KV cache 全部层 (`debug_talker_kvcache_ax/` vs `python_kvcache/`)
4. last raw hidden (pre-RMSNorm): `debug_talker_prefill_last_hidden_ax.bin` vs `prefill_last_raw_hidden.bin`
5. last normed hidden (post-RMSNorm): `prefill_last_normed_hidden_ax.bin` vs `prefill_last_normed_hidden.bin`
6. prefill logits (`debug_talker_prefill_logits_ax.bin` vs `python_prefill_logits.bin`)

**参数**：

```bash
python3 scripts/compare_prefill_checkpoints.py \
    --cpp_dir ./debug_bin/cpp_dump \
    --py_dir ./debug_bin/py_dump \
    [--hidden_size 1024] \
    [--vocab_size 3072] \
    [--S 197] \
    [--num_layers 28] \
    [--kv_dim 256]
```

- `--cpp_dir`：C++ dump 目录（含 `meta.json`、`.bin` 文件）
- `--py_dir`：Python dump 目录（含 `meta.json`、`.bin` 文件）
- `--S`：prefill 序列长度；若省略，自动从 `meta.json` 读取
- `--hidden_size`, `--vocab_size`, `--num_layers`, `--kv_dim`：模型参数；若省略，自动从 `meta.json` 读取

**期望输入文件结构**：

```
cpp_dir/
  meta.json
  talker_prefill_input.bin
  talker_layer0_prefill_output.bin
  debug_talker_kvcache_ax/
    layer_00_k.bin ... layer_27_v.bin
  debug_talker_prefill_last_hidden_ax.bin   # raw hidden (pre-RMSNorm)
  prefill_last_normed_hidden_ax.bin         # normed hidden (post-RMSNorm) ⭐新增
  debug_talker_prefill_logits_ax.bin

py_dir/
  meta.json
  prefill_embeds.bin
  python_layer0_prefill_output.bin
  python_kvcache/
    layer_00_k.bin ... layer_27_v.bin
  prefill_last_raw_hidden.bin               # raw hidden (pre-RMSNorm) ⭐新增
  prefill_last_normed_hidden.bin            # normed hidden (post-RMSNorm) ⭐新增
  python_prefill_logits.bin
```

**输出示例**：

```
=== Prefill Checkpoint Comparison ===
S=197 hidden_size=1024 vocab_size=3072

[1_prefill_input] shape=(197, 1024)
  cosine=1.00000000  max_diff=0.00000000  mean_diff=0.00000000

[4_last_raw_hidden] shape=(1024,)
  cosine=0.99999999  max_diff=0.00006104  mean_diff=0.00000345

[4_last_normed_hidden] shape=(1024,)
  cosine=0.99999583  max_diff=0.50000000  mean_diff=0.00860000
```

---

## 2. `compare_bf16_bytes.py`

**作用**：在 **bf16 字节级别**精确对比 C++ `embed` 与 Python `raw_hidden` 的最后一个 token。用于排查 FP32→BF16 转换、字节序或量化误差。

**参数**：无命令行参数，路径硬编码。

**期望输入文件**（相对于工作目录）：

```
./debug_bin/cpp_dump/debug_talker_prefill_last_hidden_ax.bin   # C++ raw hidden
./debug_bin/prefill_last_raw_hidden.bin                        # Python raw hidden ⭐新路径
```

> 旧路径 `./debug_bin/python_prefill_last_raw_hidden.bin` 仍兼容，优先尝试新路径。

**输出示例**：

```
C++ embed fp32 shape=(1024,) first5=[ 0.123 -0.456 ...]
Python raw fp32 shape=(1024,) first5=[ 0.123 -0.456 ...]

fp32 max_diff=0.00006104
bf16 bytes: EXACT MATCH
```

---

## 3. `compare_talker_prefill.py`

**作用**：对比 **AX Talker** 与 **ONNX Talker** 的 prefill 输出（last hidden + logits）。用于 ablation 分析，确认 axmodel 转换是否引入误差。

**参数**：

```bash
python3 scripts/compare_talker_prefill.py <npy_dir>
```

- `<npy_dir>`：包含 AX 与 ONNX dump 的目录（通常由 `qwen3_tts_ablation` 生成）

**期望输入文件**：

```
npy_dir/
  debug_talker_prefill_last_hidden_ax.bin    # AX raw hidden
  debug_talker_prefill_logits_ax.bin         # AX logits
  debug_talker_prefill_last_hidden_onnx.bin  # ONNX hidden
  debug_talker_prefill_logits_onnx.bin       # ONNX logits
```

**输出示例**：

```
[last_hidden] 对比结果
  Cosine Sim    : 0.99999999
  Max Abs Diff  : 0.00006104
  判定          : ✅ 高度一致
```

---

## 4. `compare_cp_dumps.py`

**作用**：**L2 独立验证** —— 对比 C++ 与 Python 的 CP (Code Predictor) 在单帧内的全部中间结果。

**对比项**：
1. CP prefill input embeds
2. CP hidden states（pre-norm / post-norm）
3. CP lm_head logits（每 sub-code 步）
4. CP sampled tokens
5. CP codec_sum
6. Talker decode step 1 输入（K/V cache、indices、mask）
7. Talker prefill last hidden（raw + normed）⭐已更新

**参数**：

```bash
python3 scripts/compare_cp_dumps.py \
    [--cpp-dir ./debug_bin/cpp_dump] \
    [--py-dir ./debug_bin/py_dump] \
    [--frame 0] \
    [-v]
```

- `--cpp-dir`：C++ dump 目录（或含 `cpp_cp_dump/` 子目录）
- `--py-dir`：Python dump 目录（或含 `python_cp_dump/` 子目录）
- `--frame`：要对比的 talker decode 帧索引（默认 0）
- `-v`：打印首个差异索引

**期望输入文件结构**：

```
cpp_dir/
  cpp_cp_dump/
    cpp_cp_frame_000_input_embeds.bin
    cpp_cp_frame_000_hidden_pre_norm_000.bin
    cpp_cp_frame_000_hidden_post_norm_000.bin
    cpp_cp_frame_000_lm_head_000_logits.bin
    cpp_cp_frame_000_sampled_token_000.bin
    cpp_cp_frame_000_codec_sum.bin
  debug_talker_prefill_last_hidden_ax.bin      # raw hidden
  prefill_last_normed_hidden_ax.bin            # normed hidden ⭐新增

py_dir/
  python_cp_dump/
    python_cp_frame_000_input_embeds.bin
    python_cp_frame_000_hidden_pre_norm_000.bin
    python_cp_frame_000_hidden_post_norm_000.bin
    python_cp_frame_000_lm_head_000_logits.bin
    python_cp_frame_000_sampled_token_000.bin
    python_cp_frame_000_codec_sum.bin
  prefill_last_raw_hidden.bin                  # raw hidden ⭐新增
  prefill_last_normed_hidden.bin               # normed hidden ⭐新增
```

---

## 5. `compare_coupling_flow.py`

**作用**：**L3 耦合流验证** —— 检查 Talker 与 CP 的耦合数据构造是否正确。核心公式：

```
inputs_embeds = codec_sum + trailing_text_hidden[step]
```

脚本计算 `residual = inputs_embeds - codec_sum`，对比 C++ 与 Python 的 residual 是否一致，从而定位 trailing text / tts_pad 注入问题。

**参数**：

```bash
python3 scripts/compare_coupling_flow.py \
    --cpp-dir ./debug_bin/cpp_dump \
    --py-dir ./debug_bin/py_dump \
    [--max-steps 20] \
    [--hidden-size 1024] \
    [-v]
```

- `--cpp-dir` / `--py-dir`：C++ / Python dump 目录
- `--max-steps`：最大对比步数（默认 20）
- `--hidden-size`：hidden size（默认 1024）
- `-v`：verbose

**期望输入文件**（每步）：

```
cpp_talker_decode_step{NNN}_codec_sum.bin
python_talker_decode_step{NNN}_codec_sum.bin
cpp_talker_decode_step{NNN}_inputs_embeds.bin
python_talker_decode_step{NNN}_inputs_embeds.bin
```

**输出示例**：

```
[Step 0]
  [codec_sum] cos=0.977500 max_diff=0.060500 DIVERGE
  [inputs_embeds] cos=0.811600 max_diff=0.746100 DIVERGE
  [cpp_residual] mean=0.123456 max=0.060500
  [py_residual]  mean=0.123456 max=0.060500
  [residual(trail)] cos=0.999995 max_diff=0.000500 MATCH
```

---

## 6. `compare_talker_decode_full.py`

**作用**：**L3 end-to-end 验证** —— 逐帧对比 Talker decode 的全部输出：codec_sum、inputs_embeds、raw_hidden、logits、next_token。

---

## 7. `compare_talker_decode_inputs.py` ⭐新增

**作用**：**L3 输入状态验证** —— 在 `compare_talker_decode_full.py` 发现 `raw_hidden` 发散后，进一步 dump 并对比 Talker decode step 0 的四个内部输入，定位根因。

**对比项**：
1. `k_cache_l00` —— prefill 后 layer 0 的 K cache
2. `v_cache_l00` —— prefill 后 layer 0 的 V cache
3. `indices` —— decode 时传入的 position index
4. `mask` —— decode 时传入的 attention mask

**参数**：

```bash
python3 scripts/compare_talker_decode_inputs.py \
    --cpp-dir ./debug_bin/cpp_dump \
    --py-dir ./debug_bin/py_dump \
    [--step 0] \
    [-v]
```

- `--cpp-dir` / `--py-dir`：C++ / Python dump 目录
- `--step`：要对比的 decode step（默认 0，即第一个 decode step）
- `-v`：verbose，打印首个差异索引

**期望输入文件结构**：

```
cpp_dir/
  debug_talker_kvcache_ax/
    layer_00_k.bin          # prefill 后 layer 0 K cache
    layer_00_v.bin          # prefill 后 layer 0 V cache
  cpp_talker_decode_step000_indices.bin
  cpp_talker_decode_step000_mask.bin

py_dir/
  python_talker_decode/
    python_talker_decode_step000_k_cache_l00.bin
    python_talker_decode_step000_v_cache_l00.bin
    python_talker_decode_step000_indices.bin
    python_talker_decode_step000_mask.bin
```

**输出示例**：

```
=== Talker Decode Inputs Comparison (step 0) ===
  [k_cache_l00] cos=1.000000 max_diff=0.000000 mean_diff=0.000000 MATCH
  [v_cache_l00] cos=1.000000 max_diff=0.000000 mean_diff=0.000000 MATCH
  [indices]     cos=1.000000 max_diff=0.000000 mean_diff=0.000000 MATCH
  [mask]        cos=1.000000 max_diff=0.000000 mean_diff=0.000000 MATCH

============================================================
Summary
============================================================
✅ All decode inputs MATCH

→ 问题在 axmodel 内部（层间 buffer、device 状态等）
→ 需 dump 逐层中间输出（layer0 output, layer1 input...）
```

**判定树**：

```
对比 decode step 0 输入
    ├── K_cache / V_cache 不一致
    │       → 检查 prefill 后 state 的保存逻辑
    │       → 检查 C++ prefill KV cache 更新位置 vs Python
    ├── indices 不一致
    │       → 检查 _last_static_position_id() 返回值 vs C++ decode_start
    │       → 检查 position_ids / cache_position 传递链条
    ├── mask 不一致
    │       → 检查 _build_decode_mask_cache() 的 fp32 数值 vs C++ mask vector
    │       → 特别关注最后一个元素是否为 0
    └── 四个输入全部一致
            → 说明问题在 axmodel 内部（层间 buffer、device 状态等）
            → 需 dump 逐层中间输出（layer0 output, layer1 input...）
```

**参数**：

```bash
python3 scripts/compare_talker_decode_full.py \
    --cpp-dir ./debug_bin/cpp_dump \
    --py-dir ./debug_bin/py_dump \
    [--max-steps 20] \
    [--hidden-size 1024] \
    [--vocab-size 3072] \
    [-v]
```

- `--cpp-dir` / `--py-dir`：C++ / Python dump 目录
- `--max-steps`：最大对比步数（默认 20）
- `--hidden-size`：默认 1024
- `--vocab-size`：默认 3072
- `-v`：打印首个差异索引

**期望输入文件**（每步）：

```
cpp_talker_decode_step{NNN}_codec_sum.bin
python_talker_decode_step{NNN}_codec_sum.bin
cpp_talker_decode_step{NNN}_inputs_embeds.bin
python_talker_decode_step{NNN}_inputs_embeds.bin
cpp_talker_decode_step{NNN}_raw_hidden.bin
python_talker_decode_step{NNN}_raw_hidden.bin
cpp_talker_decode_step{NNN}_logits.bin
python_talker_decode_step{NNN}_logits.bin
cpp_talker_decode_step{NNN}_next_token.bin
python_talker_decode_step{NNN}_next_token.bin
```

> **注意**：Python `next_token` 文件是 float32 写入的（如 `1174.0`），脚本内部会先用 `np.float32` 读取再截断为 `int`。

**输出示例**：

```
[Talker Decode Step 0]
  [codec_sum] cos=0.977500 max_diff=0.060500 mean_diff=0.008600 DIVERGE
  [inputs_embeds] cos=0.811600 max_diff=0.746100 mean_diff=0.123400 DIVERGE
  [raw_hidden] cos=0.012100 max_diff=102.000000 mean_diff=45.000000 DIVERGE
  [logits] cos=0.120400 max_diff=32.400000 mean_diff=12.300000 DIVERGE
  [next_token] cpp=1546 py=1174 DIVERGE

Summary
❌ FIRST DIVERGENCE at step 0, field=raw_hidden
  → Talker decode KV cache / indices / mask differs. Check L1 KV cache consistency.
```

---

## 8. `compare_talker_kvcache.py`

**作用**：层间对比 AX Talker 与 ONNX Talker 的 **KV cache**（每层的 K 和 V）。用于排查特定层的 KV cache 偏差。

**参数**：

```bash
python3 scripts/compare_talker_kvcache.py <npy_dir>
```

- `<npy_dir>`：包含 `debug_talker_kvcache_ax/` 与 `debug_talker_kvcache_onnx/` 的目录

**期望输入文件结构**：

```
npy_dir/
  debug_talker_kvcache_ax/
    layer_00_k.bin ... layer_27_v.bin
  debug_talker_kvcache_onnx/
    layer_00_k.bin ... layer_27_v.bin
```

**输出示例**：

```
Layer 00 K: cos=1.000000 max_diff=0.000000 ✅
Layer 00 V: cos=1.000000 max_diff=0.000000 ✅
...
Layer 15 K: cos=0.999800 max_diff=0.001200 ⚠️
```

---

## 9. 推荐验证流程

```
Step 1: L0 Prefill 输入
  → compare_prefill_checkpoints.py --cpp_dir ... --py_dir ...
  → 检查 [1_prefill_input] cosine 是否为 1.0

Step 2: L1 Talker Prefill 输出
  → 同 Step 1，检查 [4_last_raw_hidden] / [4_last_normed_hidden] / [5_prefill_logits]
  → 若 raw_hidden 有差但 logits 一致，说明 RMSNorm 在 post axmodel 内完成，属预期行为

Step 3: L2 CP 独立一致性
  → compare_cp_dumps.py --cpp-dir ... --py-dir ... --frame 0 -v
  → 检查 hidden_pre_norm / post_norm / lm_head_logits / sampled_tokens

Step 4: L3 Talker+CP 耦合
  → compare_coupling_flow.py --cpp-dir ... --py-dir ...
  → 确认 residual(trail) 是否一致

Step 5: L3 Decode 全量
  → compare_talker_decode_full.py --cpp-dir ... --py-dir ... -v
  → 定位 first divergence step 和 field

Step 6: L3 Decode 输入状态（当 raw_hidden 在 step 0 发散时）
  → compare_talker_decode_inputs.py --cpp-dir ... --py-dir ... --step 0 -v
  → 判定根因：KV cache / indices / mask / axmodel 内部
```

---

## 10. 文件名映射速查（C++ ↔ Python）

| 语义 | C++ 文件名 | Python 文件名 |
|------|-----------|--------------|
| prefill 输入 | `talker_prefill_input.bin` | `prefill_embeds.bin` |
| prefill 末帧 **raw** hidden | `debug_talker_prefill_last_hidden_ax.bin` | `prefill_last_raw_hidden.bin` |
| prefill 末帧 **normed** hidden | `prefill_last_normed_hidden_ax.bin` | `prefill_last_normed_hidden.bin` |
| prefill logits | `debug_talker_prefill_logits_ax.bin` | `python_prefill_logits.bin` |
| trailing text 全量 normed | —（未 dump 全量） | `trailing_text_hiddens.bin` |
| tts_pad_vec | —（由 `SetTtsPadVec` 传入） | `tts_pad_vec.bin` |
| decode codec_sum | `cpp_talker_decode_step{NNN}_codec_sum.bin` | `python_talker_decode_step{NNN}_codec_sum.bin` |
| decode inputs_embeds | `cpp_talker_decode_step{NNN}_inputs_embeds.bin` | `python_talker_decode_step{NNN}_inputs_embeds.bin` |
| decode raw_hidden | `cpp_talker_decode_step{NNN}_raw_hidden.bin` | `python_talker_decode_step{NNN}_raw_hidden.bin` |
| decode logits | `cpp_talker_decode_step{NNN}_logits.bin` | `python_talker_decode_step{NNN}_logits.bin` |
| decode next_token | `cpp_talker_decode_step{NNN}_next_token.bin` | `python_talker_decode_step{NNN}_next_token.bin` |

---

*文档生成于 2026-05-20，对应代码变更：同时 dump prefill last raw hidden 与 last normed hidden。*
