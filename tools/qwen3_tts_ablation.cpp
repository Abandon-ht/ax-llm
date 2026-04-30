/**
 * qwen3_tts_ablation.cpp
 * ────────────────────────────────────────────────────────────────────────────
 * Qwen3-TTS 消融实验工具：支持 4 种 Talker / CP 组合
 *
 * 用法:
 *   qwen3_tts_ablation <model_dir> <onnx_dir> <npy_dir> --mode=<0|1|2|3> [max_new_tokens]
 *
 * 模式:
 *   0 = AXModel Talker + AXModel CP    (基准)
 *   1 = ONNX Talker    + AXModel CP    (验证 Talker)
 *   2 = AXModel Talker + ONNX CP       (验证 CP)
 *   3 = ONNX Talker    + ONNX CP       (Golden Reference)
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
#include <memory>
#include <algorithm>
#include <numeric>

#include "runner/LLM.hpp"
#include "runner/utils/sample_log.h"
#include "utils/json.hpp"

#ifdef USE_AXCL
#include <axcl.h>
#else
#include <ax_sys_api.h>
#include <ax_engine_api.h>
#endif

// ── ONNX Runtime ───────────────────────────────────────────────────────────
#include <onnxruntime_cxx_api.h>

// ── BF16 helper (same as LLM_cp_tts_insert.inc) ────────────────────────────
static inline void bf16_vec_to_fp32(const unsigned short *src, float *dst, int n)
{
    for (int i = 0; i < n; ++i) {
        unsigned int u = ((unsigned int)src[i]) << 16;
        dst[i] = *reinterpret_cast<float *>(&u);
    }
}

static inline void fp32_vec_to_bf16(const float *src, unsigned short *dst, int n)
{
    for (int i = 0; i < n; ++i) {
        union { float f; unsigned int u; } tmp;
        tmp.f = src[i];
        dst[i] = (unsigned short)(tmp.u >> 16);
    }
}

// ── 路径处理 ───────────────────────────────────────────────────────────────
static std::string resolve_path(const std::string &base, const std::string &p)
{
    if (p.empty()) return p;
    if (p.rfind("http://", 0) == 0 || p.rfind("https://", 0) == 0) return p;
    namespace fs = std::filesystem;
    if (fs::path(p).is_absolute()) return p;
    return (fs::path(base) / p).lexically_normal().string();
}

// ── 读取 raw binary ────────────────────────────────────────────────────────
static bool read_binary_file(const std::string &path, std::vector<uint8_t> &out)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) { ALOGE("Cannot open file: %s", path.c_str()); return false; }
    std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);
    out.resize((size_t)sz);
    if (!f.read(reinterpret_cast<char *>(out.data()), sz)) { ALOGE("Read failed: %s", path.c_str()); return false; }
    return true;
}

// ── LLM 配置加载 ───────────────────────────────────────────────────────────
static bool load_llm_config(const std::string &model_dir, LLMAttrType &attr)
{
    const std::string cfg_path = model_dir + "/config.json";
    if (!std::filesystem::exists(cfg_path)) { ALOGE("config.json not found in %s", model_dir.c_str()); return false; }
    try {
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
        if (j.contains("b_use_mmap_load_embed")) attr.b_use_mmap_load_embed = j["b_use_mmap_load_embed"].get<bool>();
        else if (j.contains("use_mmap_load_embed")) attr.b_use_mmap_load_embed = j["use_mmap_load_embed"].get<bool>();
        if (j.contains("full_attention_interval")) attr.full_attention_interval = j["full_attention_interval"].get<int>();
#ifdef USE_AXCL
        attr.dev_ids = j.value("devices", std::vector<int>{0});
#endif
        return true;
    } catch (const std::exception &e) { ALOGE("Failed to parse config: %s", e.what()); return false; }
}

// ═══════════════════════════════════════════════════════════════════════════
// ONNX 封装
// ═══════════════════════════════════════════════════════════════════════════

class OnnxTalker {
public:
    struct State {
        std::vector<Ort::Value> kv_cache;
    };
    struct Result {
        std::vector<float> logits;       // last position only [vocab_size]
        std::vector<float> last_hidden;  // last position only [hidden_size]
        State state;
    };

    OnnxTalker(const std::string &prefill_path, const std::string &decode_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "qwen3_tts_ablation"),
          allocator_()
    {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(8);
        opts.SetInterOpNumThreads(8);
        prefill_sess_ = std::make_unique<Ort::Session>(env_, prefill_path.c_str(), opts);
        decode_sess_  = std::make_unique<Ort::Session>(env_, decode_path.c_str(), opts);
        // capture names
        GetIONames(prefill_sess_.get(), prefill_in_names_str_, prefill_in_names_, true);
        GetIONames(prefill_sess_.get(), prefill_out_names_str_, prefill_out_names_, false);
        GetIONames(decode_sess_.get(), decode_in_names_str_, decode_in_names_, true);
        GetIONames(decode_sess_.get(), decode_out_names_str_, decode_out_names_, false);
    }

    Result Prefill(const float *embeds_fp32, int prefill_len, int hidden_size)
    {
        const int64_t batch = 1;
        std::array<int64_t, 3> emb_shape = {batch, prefill_len, hidden_size};
        Ort::Value emb = Ort::Value::CreateTensor<float>(allocator_, emb_shape.data(), emb_shape.size());
        std::copy(embeds_fp32, embeds_fp32 + prefill_len * hidden_size, emb.GetTensorMutableData<float>());

        std::array<int64_t, 2> mask_shape = {batch, prefill_len};
        Ort::Value mask = Ort::Value::CreateTensor<int64_t>(allocator_, mask_shape.data(), mask_shape.size());
        std::fill(mask.GetTensorMutableData<int64_t>(), mask.GetTensorMutableData<int64_t>() + prefill_len, 1LL);

        std::array<Ort::Value, 2> inputs = {std::move(emb), std::move(mask)};
        const char *in_ptrs[] = {"inputs_embeds", "attention_mask"};
        auto outputs = prefill_sess_->Run({}, in_ptrs, inputs.data(), inputs.size(),
                                           prefill_out_names_.data(), prefill_out_names_.size());
        return PackResult(outputs, hidden_size, /*is_prefill=*/true);
    }

    Result Decode(const float *next_embed_fp32, int total_seq_len, int hidden_size, State &state)
    {
        const int64_t batch = 1;
        std::array<int64_t, 3> emb_shape = {batch, 1, hidden_size};
        Ort::Value emb = Ort::Value::CreateTensor<float>(allocator_, emb_shape.data(), emb_shape.size());
        std::copy(next_embed_fp32, next_embed_fp32 + hidden_size, emb.GetTensorMutableData<float>());

        std::array<int64_t, 2> mask_shape = {batch, total_seq_len};
        Ort::Value mask = Ort::Value::CreateTensor<int64_t>(allocator_, mask_shape.data(), mask_shape.size());
        std::fill(mask.GetTensorMutableData<int64_t>(), mask.GetTensorMutableData<int64_t>() + total_seq_len, 1LL);

        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(emb));
        inputs.push_back(std::move(mask));
        for (auto &kv : state.kv_cache) inputs.push_back(std::move(kv));
        auto outputs = decode_sess_->Run({}, decode_in_names_.data(), inputs.data(), inputs.size(),
                                           decode_out_names_.data(), decode_out_names_.size());
        return PackResult(outputs, hidden_size, /*is_prefill=*/false);
    }

private:
    Ort::Env env_;
    Ort::SessionOptions sess_opts_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::unique_ptr<Ort::Session> prefill_sess_;
    std::unique_ptr<Ort::Session> decode_sess_;
    std::vector<std::string> prefill_in_names_str_, prefill_out_names_str_;
    std::vector<std::string> decode_in_names_str_, decode_out_names_str_;
    std::vector<const char*> prefill_in_names_, prefill_out_names_;
    std::vector<const char*> decode_in_names_, decode_out_names_;

    void GetIONames(Ort::Session *sess,
                    std::vector<std::string> &names,
                    std::vector<const char *> &ptrs,
                    bool is_input)
    {
        Ort::AllocatorWithDefaultOptions alloc;
        size_t n = is_input ? sess->GetInputCount() : sess->GetOutputCount();
        names.reserve(n);
        ptrs.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            auto name_ptr = is_input ? sess->GetInputNameAllocated(i, alloc)
                                     : sess->GetOutputNameAllocated(i, alloc);
            names.emplace_back(name_ptr.get());
            ptrs.push_back(names.back().c_str());
        }
    }

    Result PackResult(std::vector<Ort::Value> &outs, int hidden_size, bool is_prefill)
    {
        Result r;
        // outs[0] = logits, outs[1] = last_hidden, outs[2..] = kv_cache
        auto &logits_tensor = outs[0];
        auto &hidden_tensor = outs[1];
        auto logits_shape = logits_tensor.GetTensorTypeAndShapeInfo().GetShape();
        auto hidden_shape = hidden_tensor.GetTensorTypeAndShapeInfo().GetShape();
        int vocab_size = static_cast<int>(logits_shape.back());
        const float *logits_data = logits_tensor.GetTensorData<float>();
        const float *hidden_data = hidden_tensor.GetTensorData<float>();
        // take last position
        int64_t logits_total = 1;
        for (auto s : logits_shape) logits_total *= s;
        int64_t hidden_total = 1;
        for (auto s : hidden_shape) hidden_total *= s;
        r.logits.assign(logits_data + (logits_total - vocab_size),
                        logits_data + logits_total);
        r.last_hidden.assign(hidden_data + (hidden_total - hidden_size),
                             hidden_data + hidden_total);
        for (size_t i = 2; i < outs.size(); ++i) {
            r.state.kv_cache.push_back(std::move(outs[i]));
        }
        return r;
    }
};

// ── ONNX Code Predictor ────────────────────────────────────────────────────
class OnnxCp {
public:
    struct Result {
        std::vector<int> frame_codes;      // [16]
        std::vector<float> codec_sum_fp32; // [hidden_size]
    };

    OnnxCp(const std::string &cp_path, const std::string &cp_embed_path,
           const std::string &codec_embed_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "qwen3_tts_ablation"),
          allocator_()
    {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(8);
        cp_sess_ = std::make_unique<Ort::Session>(env_, cp_path.c_str(), opts);
        cp_embed_sess_ = std::make_unique<Ort::Session>(env_, cp_embed_path.c_str(), opts);
        codec_embed_sess_ = std::make_unique<Ort::Session>(env_, codec_embed_path.c_str(), opts);
    }

    Result RunFrame(const float *last_hidden_fp32, int primary_code, int hidden_size)
    {
        const int D = hidden_size;
        Result res;
        res.frame_codes.resize(16);
        res.frame_codes[0] = primary_code;
        res.codec_sum_fp32.resize(D);

        // primary embed via codec_embed.onnx
        std::array<int64_t, 2> ids_shape = {1, 1};
        Ort::Value primary_ids = Ort::Value::CreateTensor<int64_t>(allocator_, ids_shape.data(), ids_shape.size());
        primary_ids.GetTensorMutableData<int64_t>()[0] = primary_code;
        auto primary_emb = RunSessionSingleOutput(codec_embed_sess_.get(), "input_ids", std::move(primary_ids));
        const float *primary_emb_data = primary_emb.GetTensorData<float>();
        std::copy(primary_emb_data, primary_emb_data + D, res.codec_sum_fp32.begin());

        // cp_ctx = [last_hidden, primary_embed]
        std::vector<float> cp_ctx;
        cp_ctx.reserve(17 * D);
        cp_ctx.insert(cp_ctx.end(), last_hidden_fp32, last_hidden_fp32 + D);
        cp_ctx.insert(cp_ctx.end(), primary_emb_data, primary_emb_data + D);

        for (int j = 0; j < 15; ++j) {
            int cp_len = static_cast<int>(cp_ctx.size()) / D;
            std::array<int64_t, 3> cp_shape = {1, cp_len, D};
            Ort::Value cp_emb = Ort::Value::CreateTensor<float>(allocator_, cp_shape.data(), cp_shape.size());
            std::copy(cp_ctx.begin(), cp_ctx.end(), cp_emb.GetTensorMutableData<float>());

            std::array<int64_t, 1> gs_shape = {1};
            Ort::Value gen_step = Ort::Value::CreateTensor<int64_t>(allocator_, gs_shape.data(), gs_shape.size());
            gen_step.GetTensorMutableData<int64_t>()[0] = j;

            auto cp_logits = RunCpSession(std::move(cp_emb), std::move(gen_step));
            const float *logits_data = cp_logits.GetTensorData<float>();
            int vocab_size = static_cast<int>(cp_logits.GetTensorTypeAndShapeInfo().GetShape()[1]);
            int res_code = static_cast<int>(std::max_element(logits_data, logits_data + vocab_size) - logits_data);
            res.frame_codes[j + 1] = res_code;

            // residual embed via code_predictor_embed.onnx
            std::array<int64_t, 2> rid_shape = {1, 1};
            Ort::Value rid = Ort::Value::CreateTensor<int64_t>(allocator_, rid_shape.data(), rid_shape.size());
            rid.GetTensorMutableData<int64_t>()[0] = res_code;
            Ort::Value gs2 = Ort::Value::CreateTensor<int64_t>(allocator_, gs_shape.data(), gs_shape.size());
            gs2.GetTensorMutableData<int64_t>()[0] = j;
            auto res_emb = RunCpEmbedSession(std::move(rid), std::move(gs2));
            const float *rd = res_emb.GetTensorData<float>();
            cp_ctx.insert(cp_ctx.end(), rd, rd + D);
            for (int d = 0; d < D; ++d) res.codec_sum_fp32[d] += rd[d];
        }
        return res;
    }

private:
    Ort::Env env_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::unique_ptr<Ort::Session> cp_sess_;
    std::unique_ptr<Ort::Session> cp_embed_sess_;
    std::unique_ptr<Ort::Session> codec_embed_sess_;

    Ort::Value RunSessionSingleOutput(Ort::Session *sess, const char *in_name, Ort::Value in_val)
    {
        const char *in_names[] = {in_name};
        Ort::Value in_vals[] = {std::move(in_val)};
        // single output assumed: look up the first output name dynamically if needed,
        // but all our single-output models use a consistent name pattern.
        // For codec_embed: output name is "embeds"
        const char *out_names[] = {"embeds"};
        auto out = sess->Run({}, in_names, in_vals, 1, out_names, 1);
        return std::move(out[0]);
    }

    Ort::Value RunCpSession(Ort::Value inputs_embeds, Ort::Value generation_step)
    {
        const char *in_names[] = {"inputs_embeds", "generation_step"};
        Ort::Value in_vals[] = {std::move(inputs_embeds), std::move(generation_step)};
        const char *out_names[] = {"logits"};
        auto out = cp_sess_->Run({}, in_names, in_vals, 2, out_names, 1);
        return std::move(out[0]);
    }

    Ort::Value RunCpEmbedSession(Ort::Value input_ids, Ort::Value generation_step)
    {
        const char *in_names[] = {"input_ids", "generation_step"};
        Ort::Value in_vals[] = {std::move(input_ids), std::move(generation_step)};
        const char *out_names[] = {"embeds"};
        auto out = cp_embed_sess_->Run({}, in_names, in_vals, 2, out_names, 1);
        return std::move(out[0]);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

enum class AblationMode { AX_AX = 0, ONNX_AX = 1, AX_ONNX = 2, ONNX_ONNX = 3 };

static AblationMode parse_mode(const char *s)
{
    if (strcmp(s, "0") == 0 || strcmp(s, "ax_ax") == 0) return AblationMode::AX_AX;
    if (strcmp(s, "1") == 0 || strcmp(s, "onnx_ax") == 0) return AblationMode::ONNX_AX;
    if (strcmp(s, "2") == 0 || strcmp(s, "ax_onnx") == 0) return AblationMode::AX_ONNX;
    if (strcmp(s, "3") == 0 || strcmp(s, "onnx_onnx") == 0) return AblationMode::ONNX_ONNX;
    return AblationMode::AX_AX;
}

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr,
            "Usage: qwen3_tts_ablation <model_dir> <onnx_dir> <npy_dir> --mode=<0|1|2|3> [max_new_tokens]\n"
            "\n"
            "  model_dir      AX650 talker model directory (with config.json)\n"
            "  onnx_dir       ONNX model directory (talker_prefill/decode/code_predictor etc.)\n"
            "  npy_dir        Directory containing prefill_embeds.bin and meta.json\n"
            "  --mode=0       AX Talker + AX CP     (baseline)\n"
            "  --mode=1       ONNX Talker + AX CP   (ablate talker)\n"
            "  --mode=2       AX Talker + ONNX CP   (ablate CP)\n"
            "  --mode=3       ONNX Talker + ONNX CP (golden reference)\n"
        );
        return 1;
    }

    const std::string model_dir = argv[1];
    const std::string onnx_dir  = argv[2];
    const std::string npy_dir   = argv[3];
    AblationMode mode = AblationMode::AX_AX;
    int max_new_tokens = 128;

    for (int i = 4; i < argc; ++i) {
        if (strncmp(argv[i], "--mode=", 7) == 0) {
            mode = parse_mode(argv[i] + 7);
        } else {
            max_new_tokens = std::atoi(argv[i]);
        }
    }

    printf("model_dir      : %s\n", model_dir.c_str());
    printf("onnx_dir       : %s\n", onnx_dir.c_str());
    printf("npy_dir        : %s\n", npy_dir.c_str());
    printf("mode           : %d\n", static_cast<int>(mode));
    printf("max_new_tokens : %d\n", max_new_tokens);

    // ── 0. 读取 meta.json ──────────────────────────────────────────────────
    const std::string meta_path = npy_dir + "/meta.json";
    if (!std::filesystem::exists(meta_path)) { ALOGE("meta.json not found"); return 1; }
    nlohmann::json meta;
    { std::ifstream f(meta_path); f >> meta; }
    const int S           = meta.value("S", 85);
    const int hidden_size = meta.value("hidden_size", 1024);
    const int audio_token_id = meta.value("audio_token_id", 151644);
    printf("S=%d hidden_size=%d\n", S, hidden_size);

    // ── 1. 读取 prefill_embeds.bin ──────────────────────────────────────────
    const std::string embed_path = npy_dir + "/prefill_embeds.bin";
    std::vector<uint8_t> embed_raw;
    if (!read_binary_file(embed_path, embed_raw)) return 1;
    const size_t expected_bytes = (size_t)S * hidden_size * sizeof(uint16_t);
    if (embed_raw.size() != expected_bytes) { ALOGE("embed size mismatch"); return 1; }
    std::vector<unsigned short> prefill_embeds_bf16(
        reinterpret_cast<uint16_t *>(embed_raw.data()),
        reinterpret_cast<uint16_t *>(embed_raw.data()) + (size_t)S * hidden_size
    );

    // fp32 copy for ONNX talker
    std::vector<float> prefill_embeds_fp32((size_t)S * hidden_size);
    bf16_vec_to_fp32(prefill_embeds_bf16.data(), prefill_embeds_fp32.data(), (int)prefill_embeds_bf16.size());

    // ── 2. 初始化 AX650 系统 ────────────────────────────────────────────────
#ifndef USE_AXCL
    AX_ENGINE_NPU_ATTR_T npu_attr; memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    AX_SYS_Init();
    int ret = AX_ENGINE_Init(&npu_attr);
    if (ret != 0) { ALOGE("AX_ENGINE_Init failed"); AX_SYS_Deinit(); return 1; }
#endif

    // ── 3. 初始化 LLM (AXModel) ────────────────────────────────────────────
    LLMAttrType attr;
    if (!load_llm_config(model_dir, attr)) return 1;
    std::filesystem::path model_path(model_dir);
    std::string cp_model_dir = (model_path / ".." / "code-predictor").lexically_normal().string();
    if (std::filesystem::exists(cp_model_dir)) attr.cp_model_dir = cp_model_dir;
    attr.runing_callback = [](std::string, float, void *) {};
    LLM llm;
    if (!llm.Init(attr)) { ALOGE("LLM::Init failed"); return 1; }
    llm.ResetKVCache();

    // ── 4. 初始化 ONNX 模型（如需要）────────────────────────────────────────
    std::unique_ptr<OnnxTalker> onnx_talker;
    std::unique_ptr<OnnxCp> onnx_cp;
    if (mode == AblationMode::ONNX_AX || mode == AblationMode::ONNX_ONNX) {
        onnx_talker = std::make_unique<OnnxTalker>(
            onnx_dir + "/talker_prefill.onnx",
            onnx_dir + "/talker_decode.onnx"
        );
    }
    if (mode == AblationMode::AX_ONNX || mode == AblationMode::ONNX_ONNX) {
        onnx_cp = std::make_unique<OnnxCp>(
            onnx_dir + "/code_predictor.onnx",
            onnx_dir + "/code_predictor_embed.onnx",
            onnx_dir + "/codec_embed.onnx"
        );
    }

    // CP callback wrapper for Mode 2 (AX Talker + ONNX CP)
    struct OnnxCpCallback : public LLM::TtsCpCallback {
        OnnxCp *cp = nullptr;
        int hidden_size = 1024;
        bool OnCpFrame(const std::vector<unsigned short> &last_hidden_bf16,
                       int primary_code,
                       std::vector<int> &out_frame_codes,
                       std::vector<unsigned short> &out_codec_sum_bf16) override
        {
            if (!cp) return false;
            std::vector<float> last_hidden_fp32(hidden_size);
            bf16_vec_to_fp32(last_hidden_bf16.data(), last_hidden_fp32.data(), hidden_size);
            auto res = cp->RunFrame(last_hidden_fp32.data(), primary_code, hidden_size);
            out_frame_codes = res.frame_codes;
            out_codec_sum_bf16.resize(hidden_size);
            fp32_vec_to_bf16(res.codec_sum_fp32.data(), out_codec_sum_bf16.data(), hidden_size);
            return true;
        }
    };
    OnnxCpCallback onnx_cp_callback;
    if (mode == AblationMode::AX_ONNX) {
        onnx_cp_callback.cp = onnx_cp.get();
        onnx_cp_callback.hidden_size = hidden_size;
    }

    const int codec_eos_token_id = 2150;
    const int trailing_start = 7; // text[0] position in 85-token prefill layout
    LLM::TtsDecodeResult tts_result;
    const auto t0 = std::chrono::steady_clock::now();

    // ═══════════════════════════════════════════════════════════════════════
    // Mode 0: AX Talker + AX CP  → 直接调用 LLM::RunTts
    // ═══════════════════════════════════════════════════════════════════════
    if (mode == AblationMode::AX_AX) {
        bool ok = llm.RunTts(prefill_embeds_bf16, max_new_tokens, codec_eos_token_id, tts_result);
        if (!ok) { ALOGE("RunTts failed"); }
    }
    // ═══════════════════════════════════════════════════════════════════════
    // Mode 1/2/3: 需要手动 loop
    // ═══════════════════════════════════════════════════════════════════════
    else {
        // For mixed modes we need to run the loop manually.
        // Save prefill hidden states from AX talker prefill (needed for txt_hidden in all modes).
        // We'll run a single-step prefill using AXModel to get all_prefill_hidden and first token.
        // Actually for Mode 1/3 (ONNX Talker) we don't need AX prefill, but we still need
        // all_prefill_hidden for trailing text. For simplicity, always prefill with AXModel
        // when mode!=0? No, for Mode 1/3 ONNX Talker does its own prefill.
        // But we still need the trailing text hidden states. These are positions 7..84 of prefill_embeds.
        // We can extract them directly from prefill_embeds_bf16!
        // So no need to run AXModel prefill for Mode 1/3.

        std::vector<unsigned short> all_prefill_hidden;
        std::vector<float> all_prefill_hidden_fp32;
        int first_primary_code = -1;

        if (mode == AblationMode::AX_ONNX) {
            // Use AXModel Talker + ONNX CP via callback
            bool ok = llm.RunTtsWithCpCallback(prefill_embeds_bf16, max_new_tokens, codec_eos_token_id, tts_result, &onnx_cp_callback);
            if (!ok) { ALOGE("RunTtsWithCpCallback failed"); }
        }

        // Mode 1 (ONNX Talker + AX CP) and Mode 3 (ONNX Talker + ONNX CP)
        // Extract trailing text hidden states from prefill_embeds directly
        // Positions 7..84 are the trailing text embeddings.
        all_prefill_hidden_fp32.resize((size_t)S * hidden_size);
        for (size_t i = 0; i < prefill_embeds_bf16.size(); ++i) {
            unsigned int u = ((unsigned int)prefill_embeds_bf16[i]) << 16;
            all_prefill_hidden_fp32[i] = *reinterpret_cast<float *>(&u);
        }

        // ONNX Talker prefill
        auto pr = onnx_talker->Prefill(prefill_embeds_fp32.data(), S, hidden_size);
        // Sample first primary code (greedy)
        first_primary_code = static_cast<int>(std::max_element(pr.logits.begin(), pr.logits.end()) - pr.logits.begin());
        printf("first_primary_code=%d\n", first_primary_code);

        if (first_primary_code == codec_eos_token_id) {
            printf("EOS at first step\n");
        } else {
            // AR loop
            std::vector<int> generated_primary = {first_primary_code};
            OnnxTalker::State talker_state = std::move(pr.state);
            std::vector<float> last_hidden_fp32 = pr.last_hidden;
            int next_token = first_primary_code;
            int total_seq_len = S;
            int step = 0;

            for (; step < max_new_tokens; ++step) {
                int primary_code = next_token;
                if (primary_code == codec_eos_token_id) break;

                // CP frame
                std::vector<int> frame_codes;
                std::vector<unsigned short> codec_sum_bf16;
                std::vector<float> codec_sum_fp32;

                if (mode == AblationMode::ONNX_AX) {
                    // AXModel CP: need bf16 last_hidden
                    std::vector<unsigned short> last_hidden_bf16(hidden_size);
                    fp32_vec_to_bf16(last_hidden_fp32.data(), last_hidden_bf16.data(), hidden_size);
                    if (!llm.RunCpFrame(last_hidden_bf16, primary_code, frame_codes, codec_sum_bf16)) {
                        ALOGE("RunCpFrame failed at step %d", step); break;
                    }
                    // convert codec_sum_bf16 -> fp32 for ONNX talker next input
                    codec_sum_fp32.resize(hidden_size);
                    bf16_vec_to_fp32(codec_sum_bf16.data(), codec_sum_fp32.data(), hidden_size);
                } else { // ONNX_ONNX
                    auto cp_res = onnx_cp->RunFrame(last_hidden_fp32.data(), primary_code, hidden_size);
                    frame_codes = cp_res.frame_codes;
                    codec_sum_fp32 = cp_res.codec_sum_fp32;
                }

                printf("frame=%d primary=%d", step, primary_code);
                for (int j = 0; j < 15; ++j) printf(" res_%d=%d", j, frame_codes[j + 1]);
                printf("\n"); fflush(stdout);

                // Build next talker input = codec_sum + txt_hidden[step]
                std::vector<float> next_in_fp32(hidden_size);
                int txt_pos = trailing_start + step;
                if (txt_pos < S) {
                    for (int d = 0; d < hidden_size; ++d) {
                        next_in_fp32[d] = codec_sum_fp32[d] + all_prefill_hidden_fp32[(size_t)txt_pos * hidden_size + d];
                    }
                } else {
                    next_in_fp32 = codec_sum_fp32;
                }

                // ONNX Talker decode
                total_seq_len++;
                auto dr = onnx_talker->Decode(next_in_fp32.data(), total_seq_len, hidden_size, talker_state);
                next_token = static_cast<int>(std::max_element(dr.logits.begin(), dr.logits.end()) - dr.logits.begin());
                last_hidden_fp32 = dr.last_hidden;
                talker_state = std::move(dr.state);
                generated_primary.push_back(next_token);

                // save frame
                LLM::TtsFrame frame;
                for (int i = 0; i < 16; ++i) frame.codes[i] = frame_codes[i];
                tts_result.frames.push_back(frame);
            }
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("[TIME]   %.2f ms\n", elapsed_ms);
    printf("[RESULT] frames=%zu\n", tts_result.frames.size());

    // ── 保存结果 ────────────────────────────────────────────────────────────
    if (!tts_result.frames.empty()) {
        char suffix[16]; snprintf(suffix, sizeof(suffix), "_%d", static_cast<int>(mode));
        std::string out_bin = npy_dir.back() == '/' ? npy_dir + std::string("output_codes") + suffix + ".bin"
                                                    : npy_dir + "/output_codes" + suffix + ".bin";
        FILE *fp = fopen(out_bin.c_str(), "wb");
        if (fp) {
            for (const auto &f : tts_result.frames) {
                int32_t buf[16];
                for (int i = 0; i < 16; ++i) buf[i] = f.codes[i];
                fwrite(buf, sizeof(int32_t), 16, fp);
            }
            fclose(fp);
            printf("[SAVE]   %s  (%zu frames)\n", out_bin.c_str(), tts_result.frames.size());
        }
        std::string out_meta = npy_dir.back() == '/' ? npy_dir + std::string("output_meta") + suffix + ".json"
                                                     : npy_dir + "/output_meta" + suffix + ".json";
        {
            nlohmann::json j;
            j["num_frames"] = (int)tts_result.frames.size();
            j["num_codebooks"] = 16;
            j["dtype"] = "int32";
            j["shape"] = { (int)tts_result.frames.size(), 16 };
            j["codec_eos_token_id"] = codec_eos_token_id;
            j["mode"] = static_cast<int>(mode);
            std::ofstream ofs(out_meta);
            ofs << j.dump(2);
            printf("[SAVE]   %s\n", out_meta.c_str());
        }
    }

    llm.Deinit();
#ifndef USE_AXCL
    AX_ENGINE_Deinit(); AX_SYS_Deinit();
#endif
    return 0;
}
