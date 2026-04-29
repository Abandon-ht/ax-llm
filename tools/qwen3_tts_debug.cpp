/**
 * qwen3_tts_debug.cpp
 * ────────────────────────────────────────────────────────────────────────────
 * Qwen3-TTS AX650 解码器调试工具
 *
 * 功能：
 *   1. 加载 LLM 模型（配置与 axllm 主程序一致）
 *   2. 从 npy_dir 读取预计算的 prefill_embeds.bin 和 meta.json
 *   3. 调用 LLM::Run(embed) 执行 prefill + greedy decode
 *   4. 打印识别结果
 *
 * 用法：
 *   qwen3_tts_debug <model_dir> <npy_dir> [max_new_tokens]
 *
 * 其中 npy_dir 由 dump_tts_debug_npy.py 生成，包含：
 *   prefill_embeds.bin   bfloat16 raw，形状 [S, hidden_size]
 *   meta.json                 包含 S, hidden_size, audio_token_id 等
 *
 * 编译：
 *   在 ax-llm 工程根目录执行 ./build_ax650.sh
 *   生成的二进制在 build_aarch64/install/bin/qwen3_tts_debug
 * ────────────────────────────────────────────────────────────────────────────
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>

#include "runner/LLM.hpp"
#include "runner/utils/sample_log.h"
#include "utils/json.hpp"   // nlohmann/json（ax-llm 已包含）

#ifdef USE_AXCL
#include <axcl.h>
#else
#include <ax_sys_api.h>
#include <ax_engine_api.h>
#endif

// ─────────────────────────────────────────────
// 工具函数：路径处理
// ─────────────────────────────────────────────

static std::string resolve_path(const std::string &base, const std::string &p)
{
    if (p.empty()) return p;
    if (p.rfind("http://", 0) == 0 || p.rfind("https://", 0) == 0) return p;
    namespace fs = std::filesystem;
    if (fs::path(p).is_absolute()) return p;
    return (fs::path(base) / p).lexically_normal().string();
}

// ─────────────────────────────────────────────
// 工具函数：读取 raw binary 文件到 vector
// ─────────────────────────────────────────────

static bool read_binary_file(const std::string &path, std::vector<uint8_t> &out)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open())
    {
        ALOGE("Cannot open file: %s", path.c_str());
        return false;
    }
    std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);
    out.resize((size_t)sz);
    if (!f.read(reinterpret_cast<char *>(out.data()), sz))
    {
        ALOGE("Read failed: %s", path.c_str());
        return false;
    }
    return true;
}

// ─────────────────────────────────────────────
// 简单 .npy 加载器（仅支持 float32 / int64 / uint16）
// 足够读取 dump_asr_debug_npy.py 生成的文件
// ─────────────────────────────────────────────

struct NpyHeader
{
    std::string dtype;
    bool fortran_order = false;
    std::vector<size_t> shape;
    size_t data_offset = 0; // 文件中数据起始字节偏移
};

static bool parse_npy_header(const std::string &path, NpyHeader &hdr)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) { ALOGE("Cannot open: %s", path.c_str()); return false; }

    // magic "\x93NUMPY"
    char magic[6];
    f.read(magic, 6);
    if (memcmp(magic, "\x93NUMPY", 6) != 0) { ALOGE("Not a npy file: %s", path.c_str()); return false; }

    uint8_t major, minor;
    f.read(reinterpret_cast<char *>(&major), 1);
    f.read(reinterpret_cast<char *>(&minor), 1);
    (void)minor;

    uint32_t header_len;
    if (major == 1)
    {
        uint16_t h16;
        f.read(reinterpret_cast<char *>(&h16), 2);
        header_len = h16;
    }
    else
    {
        f.read(reinterpret_cast<char *>(&header_len), 4);
    }

    std::string hdr_str(header_len, '\0');
    f.read(&hdr_str[0], header_len);
    hdr.data_offset = (size_t)f.tellg();

    // 解析 dtype
    auto find_val = [&](const std::string &key) -> std::string {
        auto pos = hdr_str.find(key);
        if (pos == std::string::npos) return "";
        pos = hdr_str.find('\'', pos + key.size());
        if (pos == std::string::npos)
        {
            // 可能是 False/True（fortran_order）
            pos = hdr_str.find(':', pos);
            if (pos == std::string::npos) return "";
        }
        size_t s = pos + 1;
        size_t e = hdr_str.find('\'', s);
        if (e == std::string::npos) return "";
        return hdr_str.substr(s, e - s);
    };

    hdr.dtype = find_val("'descr'");
    // strip endian prefix if any
    if (!hdr.dtype.empty() && (hdr.dtype[0] == '<' || hdr.dtype[0] == '>' || hdr.dtype[0] == '='))
        hdr.dtype = hdr.dtype.substr(1);

    // fortran_order
    auto fo_pos = hdr_str.find("'fortran_order'");
    if (fo_pos != std::string::npos)
    {
        auto colon = hdr_str.find(':', fo_pos);
        if (colon != std::string::npos)
        {
            auto t = hdr_str.find("True", colon);
            auto comma = hdr_str.find(',', colon);
            hdr.fortran_order = (t != std::string::npos && (comma == std::string::npos || t < comma));
        }
    }

    // shape
    auto shape_pos = hdr_str.find("'shape'");
    if (shape_pos != std::string::npos)
    {
        auto lp = hdr_str.find('(', shape_pos);
        auto rp = hdr_str.find(')', shape_pos);
        if (lp != std::string::npos && rp != std::string::npos)
        {
            std::string shape_str = hdr_str.substr(lp + 1, rp - lp - 1);
            size_t pos2 = 0;
            while (pos2 < shape_str.size())
            {
                while (pos2 < shape_str.size() && (shape_str[pos2] == ' ' || shape_str[pos2] == ','))
                    ++pos2;
                if (pos2 >= shape_str.size()) break;
                size_t end2 = pos2;
                while (end2 < shape_str.size() && shape_str[end2] >= '0' && shape_str[end2] <= '9')
                    ++end2;
                if (end2 > pos2)
                    hdr.shape.push_back(std::stoull(shape_str.substr(pos2, end2 - pos2)));
                pos2 = end2;
            }
        }
    }
    return true;
}

static bool load_npy_float32(const std::string &path, std::vector<float> &out, std::vector<size_t> &shape)
{
    NpyHeader hdr;
    if (!parse_npy_header(path, hdr)) return false;
    if (hdr.dtype != "f4" && hdr.dtype != "f8" && hdr.dtype != "float32")
    {
        ALOGE("Expected float32 in %s, got %s", path.c_str(), hdr.dtype.c_str());
        return false;
    }
    size_t n = 1;
    for (auto d : hdr.shape) n *= d;
    shape = hdr.shape;

    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f.seekg((std::streamoff)hdr.data_offset);
    out.resize(n);
    f.read(reinterpret_cast<char *>(out.data()), (std::streamsize)(n * sizeof(float)));
    return f.good() || f.eof();
}

static bool load_npy_int64(const std::string &path, std::vector<int64_t> &out, std::vector<size_t> &shape)
{
    NpyHeader hdr;
    if (!parse_npy_header(path, hdr)) return false;
    if (hdr.dtype != "i8" && hdr.dtype != "int64")
    {
        ALOGE("Expected int64 in %s, got dtype='%s'", path.c_str(), hdr.dtype.c_str());
        return false;
    }
    size_t n = 1;
    for (auto d : hdr.shape) n *= d;
    shape = hdr.shape;

    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    f.seekg((std::streamoff)hdr.data_offset);
    out.resize(n);
    f.read(reinterpret_cast<char *>(out.data()), (std::streamsize)(n * sizeof(int64_t)));
    return f.good() || f.eof();
}

// ─────────────────────────────────────────────
// LLM 配置加载（与 main.cpp / llm_smoke.cpp 一致）
// ─────────────────────────────────────────────

static bool load_llm_config(const std::string &model_dir, LLMAttrType &attr)
{
    const std::string cfg_path = model_dir + "/config.json";
    if (!std::filesystem::exists(cfg_path))
    {
        ALOGE("config.json not found in %s", model_dir.c_str());
        return false;
    }
    try
    {
        std::ifstream f(cfg_path);
        nlohmann::json j;
        f >> j;

        attr.template_filename_axmodel = resolve_path(model_dir, j["template_filename_axmodel"].get<std::string>());
        attr.filename_post_axmodel     = resolve_path(model_dir, j["filename_post_axmodel"].get<std::string>());
        attr.url_tokenizer_model       = resolve_path(model_dir, j["url_tokenizer_model"].get<std::string>());
        attr.tokenizer_type            = j.value("tokenizer_type", std::string("Qwen3"));
        attr.filename_tokens_embed     = resolve_path(model_dir, j["filename_tokens_embed"].get<std::string>());
        attr.post_config_path          = resolve_path(model_dir, j.value("post_config_path", std::string("post_config.json")));
        attr.axmodel_num               = j["axmodel_num"].get<int>();
        attr.tokens_embed_num          = j["tokens_embed_num"].get<int>();
        attr.tokens_embed_size         = j["tokens_embed_size"].get<int>();

        if (j.contains("b_use_mmap_load_embed"))
            attr.b_use_mmap_load_embed = j["b_use_mmap_load_embed"].get<bool>();
        else if (j.contains("use_mmap_load_embed"))
            attr.b_use_mmap_load_embed = j["use_mmap_load_embed"].get<bool>();

        if (j.contains("full_attention_interval"))
            attr.full_attention_interval = j["full_attention_interval"].get<int>();

#ifdef USE_AXCL
        attr.dev_ids = j.value("devices", std::vector<int>{0});
#endif
        return true;
    }
    catch (const std::exception &e)
    {
        ALOGE("Failed to parse config: %s", e.what());
        return false;
    }
}

// ─────────────────────────────────────────────
// main
// ─────────────────────────────────────────────

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        fprintf(stderr,
            "Usage: qwen3_tts_debug <model_dir> <npy_dir> [max_new_tokens]\n"
            "\n"
            "  model_dir      Qwen3-TTS-0.6B axmodel 目录（含 config.json）\n"
            "  npy_dir        由 dump_tts_debug_npy.py 生成的调试数据目录\n"
            "  max_new_tokens 最多生成 token 数（默认 128）\n"
            "\n"
            "npy_dir 需包含：\n"
            "  prefill_embeds.bin   bfloat16 embedding，[S, hidden_size]\n"
            "  meta.json                 形状元信息\n"
        );
        return 1;
    }

    const std::string model_dir    = argv[1];
    const std::string npy_dir      = argv[2];
    const int max_new_tokens       = (argc >= 4) ? std::atoi(argv[3]) : 128;

    printf("model_dir      : %s\n", model_dir.c_str());
    printf("npy_dir        : %s\n", npy_dir.c_str());
    printf("max_new_tokens : %d\n", max_new_tokens);

    // ── 0. 读取 meta.json ──────────────────────────────────────────────────
    const std::string meta_path = npy_dir + "/meta.json";
    if (!std::filesystem::exists(meta_path))
    {
        ALOGE("meta.json not found in %s", npy_dir.c_str());
        return 1;
    }

    nlohmann::json meta;
    {
        std::ifstream f(meta_path);
        f >> meta;
    }
    const int S              = meta.value("S", 8);
    const int S0             = meta.value("S0", S);
    const int hidden_size    = meta.value("hidden_size", 1024);
    const int audio_token_id = meta.value("audio_token_id", 151644);
    const int audio_slots    = meta.value("audio_slots_filled", 0);

    printf("S=%d  S0=%d  hidden_size=%d  audio_token_id=%d  audio_slots=%d\n",
           S, S0, hidden_size, audio_token_id, audio_slots);

    // ── 1. 读取 prefill_embeds.bin ────────────────────────────────────
    const std::string embed_path = npy_dir + "/prefill_embeds.bin";
    std::vector<uint8_t> embed_raw;
    if (!read_binary_file(embed_path, embed_raw))
    {
        ALOGE("Failed to load %s", embed_path.c_str());
        return 1;
    }
    const size_t expected_bytes = (size_t)S * (size_t)hidden_size * sizeof(uint16_t);
    if (embed_raw.size() != expected_bytes)
    {
        ALOGE("embed size mismatch: got %zu bytes, expected %zu (S=%d * H=%d * 2)",
              embed_raw.size(), expected_bytes, S, hidden_size);
        return 1;
    }
    // 转为 unsigned short vector（bfloat16）
    std::vector<unsigned short> combined_embed(
        reinterpret_cast<uint16_t *>(embed_raw.data()),
        reinterpret_cast<uint16_t *>(embed_raw.data()) + (size_t)S * (size_t)hidden_size
    );
    printf("combined_embed loaded: %zu elements = [%d, %d] bfloat16\n",
           combined_embed.size(), S, hidden_size);

    // ── 2. 初始化 AX650 系统 ────────────────────────────────────────────────
#ifdef USE_AXCL
    // AXCL 初始化在 LLM::Init 内完成
#else
    AX_ENGINE_NPU_ATTR_T npu_attr;
    memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    AX_SYS_Init();
    int ret = AX_ENGINE_Init(&npu_attr);
    if (ret != 0)
    {
        ALOGE("AX_ENGINE_Init failed: 0x%x", ret);
        AX_SYS_Deinit();
        return 1;
    }
#endif

    // ── 3. 初始化 LLM ──────────────────────────────────────────────────────
    LLMAttrType attr;
    if (!load_llm_config(model_dir, attr))
    {
        ALOGE("load_llm_config failed");
        return 1;
    }

    // 设置 callback（打印解码过程中的 token）
    attr.runing_callback = [](std::string str, float /*tps*/, void * /*r*/) {
        fprintf(stdout, "%s", str.c_str());
        fflush(stdout);
    };

    LLM llm;
    printf("\n[INFO] Initializing LLM...\n");
    if (!llm.Init(attr))
    {
        ALOGE("LLM::Init failed");
        return 1;
    }
    printf("[INFO] LLM initialized OK\n\n");

    // ── 4. 重置 KV cache，设置初始 context ────────────────────────────────
    llm.ResetKVCache();

    // LLM::Run(embed) 内部将：
    //   - 按 prefill_token_num 分块做 prefill（通常 64 tokens/block）
    //   - Prefill 结束后调用 post.axmodel 得到第一个 token 的 logits
    //   - 循环 decode，每步调用 decode 组（单 token）
    //   - 遇到 EOS 停止，返回 tokenizer.decode(generated_ids)

    printf("[INFO] Running ASR decode (S=%d tokens, max_new_tokens=%d)...\n\n",
           S, max_new_tokens);

    const auto t0 = std::chrono::steady_clock::now();
    std::string result = llm.Run(combined_embed, max_new_tokens);
    const auto t1 = std::chrono::steady_clock::now();

    const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("\n\n[RESULT] %s\n", result.c_str());
    printf("[TIME]   %.2f ms\n", elapsed_ms);

    // ── 5. 可选：从 input_ids.npy 验证 ────────────────────────────────────
    const std::string ids_path = npy_dir + "/input_ids.npy";
    if (std::filesystem::exists(ids_path))
    {
        std::vector<int64_t> input_ids;
        std::vector<size_t> ids_shape;
        if (load_npy_int64(ids_path, input_ids, ids_shape))
        {
            int audio_slots_in_ids = 0;
            for (auto v : input_ids)
                if (v == audio_token_id) ++audio_slots_in_ids;
            printf("[DEBUG] input_ids shape=[");
            for (size_t i = 0; i < ids_shape.size(); ++i)
                printf("%s%zu", i ? "," : "", ids_shape[i]);
            printf("], audio_pad count=%d\n", audio_slots_in_ids);
        }
    }

    // ── 6. 反初始化 ────────────────────────────────────────────────────────
    llm.Deinit();

#ifndef USE_AXCL
    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
#endif

    return 0;
}
