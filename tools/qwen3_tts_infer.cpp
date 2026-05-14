/**
 * qwen3_tts_infer.cpp
 * ────────────────────────────────────────────────────────────────────────────
 * Qwen3-TTS AX650 推理工具（纯 AXModel）
 *
 * 主要参考 infer.py 的推理逻辑：
 *   - 支持完整的采样参数（temperature / top_k / top_p / repetition_penalty）
 *   - 支持流式 / 非流式模式
 *   - 支持设置随机种子
 *
 * 用法：
 *   qwen3_tts_infer <model_dir> <npy_dir> [options]
 *
 * npy_dir 需包含：
 *   prefill_embeds.bin   bfloat16 raw，形状 [S, hidden_size]
 *   meta.json            包含 S, hidden_size, audio_token_id, trailing_start 等
 *   tts_pad_vec.bin      非流式模式必需 (int32 hidden_size + fp32 data)
 *
 * Options:
 *   --max_new_tokens <N>       最大生成帧数（默认 128）
 *   --temperature <float>      采样温度（默认 0.9）
 *   --top_k <int>              top-k 采样（默认 50）
 *   --top_p <float>            top-p 采样（默认 1.0）
 *   --repetition_penalty <f>   重复惩罚（默认 1.05）
 *   --streaming                启用流式模式
 *   --codec_eos_token_id <id>  EOS token ID（默认 2150）
 *   --output <prefix>          输出文件前缀（默认 <npy_dir>/output）
 *   --seed <int>               随机种子（默认不设置）
 *
 * 编译：
 *   ./build_ax650.sh
 *   输出：build/install/bin/qwen3_tts_infer
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
#include <algorithm>

#include "runner/LLM.hpp"
#include "runner/LLMPostprocess.hpp"
#include "runner/utils/sample_log.h"
#include "utils/json.hpp"

#ifdef USE_AXCL
#include <axcl.h>
#else
#include <ax_sys_api.h>
#include <ax_engine_api.h>
#endif

// ─────────────────────────────────────────────
// 工具函数
// ─────────────────────────────────────────────

static std::string resolve_path(const std::string &base, const std::string &p)
{
    if (p.empty()) return p;
    if (p.rfind("http://", 0) == 0 || p.rfind("https://", 0) == 0) return p;
    namespace fs = std::filesystem;
    if (fs::path(p).is_absolute()) return p;
    return (fs::path(base) / p).lexically_normal().string();
}

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
// LLM 配置加载
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
// 命令行参数
// ─────────────────────────────────────────────

struct Args
{
    std::string model_dir;
    std::string npy_dir;
    int max_new_tokens = 128;
    float temperature = 0.9f;
    int top_k = 50;
    float top_p = 1.0f;
    float repetition_penalty = 1.05f;
    bool streaming = false;
    int codec_eos_token_id = 2150;
    std::string output_prefix;
    int seed = -1;
};

static void print_usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s <model_dir> <npy_dir> [options]\n"
        "\n"
        "  model_dir      AX650 talker model directory (with config.json)\n"
        "  npy_dir        Directory containing prefill_embeds.bin and meta.json\n"
        "\n"
        "Options:\n"
        "  --max_new_tokens <N>       Max frames to generate (default: 128)\n"
        "  --temperature <float>      Sampling temperature (default: 0.9)\n"
        "  --top_k <int>              Top-k sampling (default: 50)\n"
        "  --top_p <float>            Top-p sampling (default: 1.0)\n"
        "  --repetition_penalty <f>   Repetition penalty (default: 1.05)\n"
        "  --streaming                Enable streaming mode\n"
        "  --codec_eos_token_id <id>  Codec EOS token id (default: 2150)\n"
        "  --output <prefix>          Output file prefix (default: <npy_dir>/output)\n"
        "  --seed <int>               Random seed (default: not set)\n"
        "\n"
        "npy_dir must contain:\n"
        "  prefill_embeds.bin   bfloat16 [S, hidden_size]\n"
        "  meta.json            metadata (S, hidden_size, trailing_start...)\n"
        "  tts_pad_vec.bin      Required for non-streaming mode\n"
        , prog);
}

static bool parse_args(int argc, char **argv, Args &args)
{
    if (argc < 3)
    {
        print_usage(argv[0]);
        return false;
    }
    args.model_dir = argv[1];
    args.npy_dir = argv[2];

    for (int i = 3; i < argc; ++i)
    {
        if (strcmp(argv[i], "--max_new_tokens") == 0 && i + 1 < argc)
        {
            args.max_new_tokens = std::atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc)
        {
            args.temperature = std::atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--top_k") == 0 && i + 1 < argc)
        {
            args.top_k = std::atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--top_p") == 0 && i + 1 < argc)
        {
            args.top_p = std::atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--repetition_penalty") == 0 && i + 1 < argc)
        {
            args.repetition_penalty = std::atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--streaming") == 0)
        {
            args.streaming = true;
        }
        else if (strcmp(argv[i], "--codec_eos_token_id") == 0 && i + 1 < argc)
        {
            args.codec_eos_token_id = std::atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
        {
            args.output_prefix = argv[++i];
        }
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
        {
            args.seed = std::atoi(argv[++i]);
        }
        else
        {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return false;
        }
    }

    if (args.output_prefix.empty())
    {
        args.output_prefix = args.npy_dir.back() == '/' ? args.npy_dir + "output"
                                                          : args.npy_dir + "/output";
    }

    return true;
}

// ─────────────────────────────────────────────
// main
// ─────────────────────────────────────────────

int main(int argc, char **argv)
{
    Args args;
    if (!parse_args(argc, argv, args))
        return 1;

    printf("model_dir            : %s\n", args.model_dir.c_str());
    printf("npy_dir              : %s\n", args.npy_dir.c_str());
    printf("max_new_tokens       : %d\n", args.max_new_tokens);
    printf("temperature          : %.2f\n", args.temperature);
    printf("top_k                : %d\n", args.top_k);
    printf("top_p                : %.2f\n", args.top_p);
    printf("repetition_penalty   : %.2f\n", args.repetition_penalty);
    printf("streaming            : %s\n", args.streaming ? "true" : "false");
    printf("codec_eos_token_id   : %d\n", args.codec_eos_token_id);
    printf("output_prefix        : %s\n", args.output_prefix.c_str());
    if (args.seed >= 0)
        printf("seed                 : %d\n", args.seed);

    // ── 0. 读取 meta.json ──────────────────────────────────────────────────
    const std::string meta_path = args.npy_dir + "/meta.json";
    if (!std::filesystem::exists(meta_path))
    {
        ALOGE("meta.json not found in %s", args.npy_dir.c_str());
        return 1;
    }

    nlohmann::json meta;
    {
        std::ifstream f(meta_path);
        f >> meta;
    }
    const int S              = meta.value("S", 8);
    const int hidden_size    = meta.value("hidden_size", 1024);
    const int audio_token_id = meta.value("audio_token_id", 151644);
    const int trailing_start = meta.value("trailing_start", 7);

    printf("S=%d hidden_size=%d trailing_start=%d\n", S, hidden_size, trailing_start);

    // ── 1. 读取 prefill_embeds.bin ────────────────────────────────────
    const std::string embed_path = args.npy_dir + "/prefill_embeds.bin";
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
    std::vector<unsigned short> prefill_embeds(
        reinterpret_cast<uint16_t *>(embed_raw.data()),
        reinterpret_cast<uint16_t *>(embed_raw.data()) + (size_t)S * (size_t)hidden_size
    );
    printf("prefill_embeds loaded: %zu elements = [%d, %d] bfloat16\n",
           prefill_embeds.size(), S, hidden_size);

    // ── 1.5 读取 tts_pad_vec.bin（非流式模式）───────────────────────────
    std::vector<unsigned short> tts_pad_vec_bf16;
    if (!args.streaming)
    {
        const std::string pad_vec_path = args.npy_dir.back() == '/' ? args.npy_dir + "tts_pad_vec.bin"
                                                                    : args.npy_dir + "/tts_pad_vec.bin";
        std::vector<uint8_t> pad_raw;
        if (!read_binary_file(pad_vec_path, pad_raw))
        {
            ALOGE("Non-streaming mode requires tts_pad_vec.bin, but failed to load it.");
            return 1;
        }
        if (pad_raw.size() < sizeof(int32_t))
        {
            ALOGE("tts_pad_vec too small");
            return 1;
        }
        int32_t pad_hidden = *reinterpret_cast<int32_t *>(pad_raw.data());
        if (pad_hidden != hidden_size)
        {
            ALOGE("tts_pad_vec hidden_size mismatch: file=%d, expected=%d", pad_hidden, hidden_size);
            return 1;
        }
        const size_t pad_expected = sizeof(int32_t) + (size_t)hidden_size * sizeof(float);
        if (pad_raw.size() != pad_expected)
        {
            ALOGE("tts_pad_vec size mismatch: %zu vs expected %zu", pad_raw.size(), pad_expected);
            return 1;
        }
        std::vector<float> tts_pad_vec(
            reinterpret_cast<float *>(pad_raw.data() + sizeof(int32_t)),
            reinterpret_cast<float *>(pad_raw.data() + sizeof(int32_t)) + hidden_size
        );
        // convert fp32 -> bf16
        tts_pad_vec_bf16.resize(hidden_size);
        for (int d = 0; d < hidden_size; ++d)
        {
            union { float f; unsigned int u; } tmp;
            tmp.f = tts_pad_vec[d];
            tts_pad_vec_bf16[d] = (unsigned short)(tmp.u >> 16);
        }
        printf("tts_pad_vec loaded: %d floats -> bf16\n", hidden_size);
    }

    // ── 2. 初始化 AX650 系统 ────────────────────────────────────────────────
#ifndef USE_AXCL
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
    if (!load_llm_config(args.model_dir, attr))
    {
        ALOGE("load_llm_config failed");
        return 1;
    }

    // 推断 code-predictor 目录
    std::filesystem::path model_path(args.model_dir);
    std::string cp_model_dir = (model_path / ".." / "code-predictor").lexically_normal().string();
    if (std::filesystem::exists(cp_model_dir))
    {
        attr.cp_model_dir = cp_model_dir;
        printf("[INFO] CP model dir: %s\n", cp_model_dir.c_str());
    }
    else
    {
        ALOGW("CP model dir not found: %s", cp_model_dir.c_str());
    }

    // 设置 callback：TTS token 不是文本，decode 出来是乱码，因此不打印字符串。
    attr.runing_callback = [](std::string /*str*/, float /*tps*/, void * /*r*/) {
        // 不输出乱码字符串
    };

    LLM llm;
    printf("\n[INFO] Initializing LLM...\n");
    if (!llm.Init(attr))
    {
        ALOGE("LLM::Init failed");
        return 1;
    }
    printf("[INFO] LLM initialized OK\n\n");

    // ── 4. 设置采样参数（仅在 CLI 显式指定或值与“关闭状态”不同时覆盖 post_config.json）─
    LLMPostprocess *postprocess = llm.getPostprocess();
    if (postprocess)
    {
        postprocess->set_temperature(args.temperature != 1.0f, args.temperature);
        postprocess->set_top_k_sampling(args.top_k > 0, args.top_k);
        postprocess->set_top_p_sampling(args.top_p > 0.0f && args.top_p < 1.0f, args.top_p);
        postprocess->set_repetition_penalty(args.repetition_penalty != 1.0f, args.repetition_penalty);
        if (args.seed >= 0)
        {
            postprocess->set_seed(args.seed);
            printf("[INFO] Set random seed to %d\n", args.seed);
        }
        printf("[INFO] Postprocess config set according to CLI / generation_config defaults\n");
    }
    else
    {
        ALOGW("LLM::getPostprocess() returned nullptr; using post_config.json defaults");
    }

    // ── 5. 设置 tts_pad_vec（非流式）──────────────────────────────────────
    llm.ResetKVCache();
    if (!args.streaming && !tts_pad_vec_bf16.empty())
    {
        llm.SetTtsPadVec(tts_pad_vec_bf16);
        printf("[INFO] SetTtsPadVec called for non-streaming mode\n");
    }

    // ── 6. 运行 TTS decode ─────────────────────────────────────────────────
    printf("[INFO] Running TTS decode (S=%d tokens, max_new_tokens=%d, streaming=%s)...\n\n",
           S, args.max_new_tokens, args.streaming ? "true" : "false");

    LLM::TtsDecodeResult tts_result;
    const auto t0 = std::chrono::steady_clock::now();
    bool ok = llm.RunTts(prefill_embeds, args.max_new_tokens, args.codec_eos_token_id, tts_result, args.streaming);
    const auto t1 = std::chrono::steady_clock::now();

    const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("\n[TIME]   %.2f ms\n", elapsed_ms);
    printf("[RESULT] frames=%zu\n", tts_result.frames.size());

    // ── 7. 保存结果 ────────────────────────────────────────────────────────
    if (ok && !tts_result.frames.empty())
    {
        // output_codes.bin
        std::string out_bin = args.output_prefix + "_codes.bin";
        FILE *fp = fopen(out_bin.c_str(), "wb");
        if (fp)
        {
            for (const auto &f : tts_result.frames)
            {
                int32_t buf[16];
                for (int i = 0; i < 16; ++i) buf[i] = f.codes[i];
                fwrite(buf, sizeof(int32_t), 16, fp);
            }
            fclose(fp);
            printf("[SAVE]   %s  (%zu frames)\n", out_bin.c_str(), tts_result.frames.size());
        }
        else
        {
            ALOGE("Failed to write %s", out_bin.c_str());
        }

        // output_meta.json
        std::string out_meta = args.output_prefix + "_meta.json";
        {
            nlohmann::json j;
            j["num_frames"] = (int)tts_result.frames.size();
            j["num_codebooks"] = 16;
            j["dtype"] = "int32";
            j["shape"] = { (int)tts_result.frames.size(), 16 };
            j["codec_eos_token_id"] = args.codec_eos_token_id;
            j["streaming"] = args.streaming;
            j["max_new_tokens"] = args.max_new_tokens;
            j["temperature"] = args.temperature;
            j["top_k"] = args.top_k;
            j["top_p"] = args.top_p;
            j["repetition_penalty"] = args.repetition_penalty;
            j["seed"] = args.seed;
            std::ofstream ofs(out_meta);
            ofs << j.dump(2);
            printf("[SAVE]   %s\n", out_meta.c_str());
        }
    }
    else
    {
        ALOGE("RunTts failed or no frames generated");
    }

    // ── 8. 反初始化 ────────────────────────────────────────────────────────
    llm.Deinit();

#ifndef USE_AXCL
    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
#endif

    return ok ? 0 : 1;
}
