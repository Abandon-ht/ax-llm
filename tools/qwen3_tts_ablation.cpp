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
#include <random>

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

    void DumpPrefillKVCache(const std::string &dir, const Result &result, int prefill_len) const
    {
        namespace fs = std::filesystem;
        fs::create_directories(dir);
        nlohmann::json meta;
        meta["prefill_len"] = prefill_len;
        meta["num_kv_tensors"] = (int)result.state.kv_cache.size();
        nlohmann::json tensors = nlohmann::json::array();

        for (size_t i = 0; i < result.state.kv_cache.size(); ++i) {
            const auto &tensor = result.state.kv_cache[i];
            auto shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
            const float *data = tensor.GetTensorData<float>();
            int64_t total = 1;
            for (auto s : shape) total *= s;

            // Parse layer index and kv type from output name
            int layer_idx = -1;
            std::string kv_type;
            if (i + 2 < prefill_out_names_str_.size()) {
                const std::string &name = prefill_out_names_str_[i + 2];
                std::string lower = name;
                std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                if (lower.find("key") != std::string::npos || lower.find(".k") != std::string::npos) kv_type = "k";
                else if (lower.find("val") != std::string::npos || lower.find(".v") != std::string::npos) kv_type = "v";
                // Extract first number as layer index
                size_t pos = 0;
                while (pos < name.size() && !std::isdigit(name[pos])) ++pos;
                if (pos < name.size()) layer_idx = std::atoi(name.c_str() + pos);
            }

            std::vector<float> buf;
            if (shape.size() == 4 && shape[0] == 1) {
                // [1, num_heads, seq_len, head_dim] -> [seq_len, num_heads*head_dim]
                int num_heads = static_cast<int>(shape[1]);
                int seq_len   = static_cast<int>(shape[2]);
                int head_dim  = static_cast<int>(shape[3]);
                buf.resize((size_t)seq_len * num_heads * head_dim);
                for (int s = 0; s < seq_len; ++s) {
                    for (int h = 0; h < num_heads; ++h) {
                        for (int d = 0; d < head_dim; ++d) {
                            int src_idx = h * seq_len * head_dim + s * head_dim + d;
                            int dst_idx = s * num_heads * head_dim + h * head_dim + d;
                            buf[(size_t)dst_idx] = data[src_idx];
                        }
                    }
                }
            } else if (shape.size() == 3 && shape[0] == 1) {
                // [1, seq_len, hidden_size]
                int seq_len = static_cast<int>(shape[1]);
                int hidden  = static_cast<int>(shape[2]);
                buf.assign(data, data + (size_t)seq_len * hidden);
            } else {
                buf.assign(data, data + total);
            }

            char fname[256];
            if (layer_idx >= 0 && !kv_type.empty()) {
                snprintf(fname, sizeof(fname), "layer_%02d_%s.bin", layer_idx, kv_type.c_str());
            } else {
                snprintf(fname, sizeof(fname), "kv_%03zu.bin", i);
            }
            std::string path = (fs::path(dir) / fname).string();
            FILE *fp = fopen(path.c_str(), "wb");
            if (fp) {
                fwrite(buf.data(), sizeof(float), buf.size(), fp);
                fclose(fp);
            }

            nlohmann::json tinfo;
            tinfo["filename"] = fname;
            tinfo["name"] = (i + 2 < prefill_out_names_str_.size()) ? prefill_out_names_str_[i + 2] : "";
            tinfo["orig_shape"] = shape;
            tinfo["dump_shape"] = {(int)(buf.size() / prefill_len), prefill_len};
            tensors.push_back(tinfo);
        }
        meta["tensors"] = tensors;
        std::string meta_path = (fs::path(dir) / "meta.json").string();
        std::ofstream ofs(meta_path);
        ofs << meta.dump(2);
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

// ═══════════════════════════════════════════════════════════════════════════
// Sampling helper (参考 sherpa-onnx SampleFromLogits)
// ═══════════════════════════════════════════════════════════════════════════

static int SampleFromLogits(const float *logits_data, int32_t total, int32_t vocab_size,
                            float temperature, int32_t top_k, float top_p,
                            float repetition_penalty,
                            const std::vector<int> &generated_ids,
                            int32_t suppress_start, int32_t suppress_end,
                            int suppress_exception, bool suppress_eos)
{
    const int32_t V = total >= vocab_size ? vocab_size : total;
    const float *src = logits_data + (total - V);
    std::vector<float> buf(src, src + V);

    if (suppress_start >= 0 && suppress_end > suppress_start)
        for (int32_t i = suppress_start; i < std::min(suppress_end, V); ++i)
            if (i != suppress_exception) buf[i] = -1e9f;

    if (suppress_eos && suppress_exception >= 0 && suppress_exception < V)
        buf[suppress_exception] = -1e9f;

    if (repetition_penalty > 1.0f)
        for (auto id : generated_ids)
            if (id >= 0 && id < V)
                buf[id] = buf[id] > 0 ? buf[id] / repetition_penalty
                                      : buf[id] * repetition_penalty;

    if (temperature < 1e-6f)
        return static_cast<int>(std::max_element(buf.begin(), buf.end()) - buf.begin());

    for (auto &v : buf) v /= temperature;

    if (top_k > 0 && top_k < V) {
        std::vector<float> tmp(buf.begin(), buf.end());
        std::partial_sort(tmp.begin(), tmp.begin() + top_k, tmp.end(), std::greater<float>());
        const float thr = tmp[top_k - 1];
        for (auto &v : buf)
            if (v < thr) v = -1e9f;
    }

    const float max_v = *std::max_element(buf.begin(), buf.end());
    float sum = 0;
    for (auto &v : buf) {
        v = std::exp(v - max_v);
        sum += v;
    }
    for (auto &v : buf) v /= sum;

    if (top_p < 1.0f && top_p > 0.0f) {
        std::vector<std::pair<float, int32_t>> pi(V);
        for (int32_t i = 0; i < V; ++i) pi[i] = {buf[i], i};
        std::sort(pi.begin(), pi.end(),
                  [](const auto &a, const auto &b) { return a.first > b.first; });
        float cum = 0;
        int32_t cut = V;
        for (int32_t i = 0; i < V; ++i) {
            cum += pi[i].first;
            if (cum >= top_p) {
                cut = i + 1;
                break;
            }
        }
        for (int32_t i = cut; i < V; ++i) buf[pi[i].second] = 0.0f;
        float ns = 0;
        for (auto v : buf) ns += v;
        if (ns > 0)
            for (auto &v : buf) v /= ns;
    }

    thread_local std::mt19937 rng(std::random_device{}());
    return static_cast<int>(
        std::discrete_distribution<int32_t>(buf.begin(), buf.end())(rng));
}

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

    Result RunFrame(const float *last_hidden_fp32, int primary_code, int hidden_size,
                    float temperature, int top_k, float top_p, int vocab_size)
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
            int logits_total = static_cast<int>(cp_logits.GetTensorTypeAndShapeInfo().GetShape()[1]);
            int res_code = SampleFromLogits(
                logits_data, logits_total, vocab_size,
                temperature, top_k, top_p, /*repetition_penalty=*/1.0f,
                /*generated_ids=*/{}, /*suppress_start=*/-1, /*suppress_end=*/-1,
                /*suppress_exception=*/-1, /*suppress_eos=*/false);
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
            "Usage: qwen3_tts_ablation <model_dir> <onnx_dir> <npy_dir> --mode=<0|1|2|3> [options] [max_new_tokens]\n"
            "\n"
            "  model_dir      AX650 talker model directory (with config.json)\n"
            "  onnx_dir       ONNX model directory (talker_prefill/decode/code_predictor etc.)\n"
            "  npy_dir        Directory containing prefill_embeds.bin and meta.json\n"
            "  --mode=0       AX Talker + AX CP     (baseline)\n"
            "  --mode=1       ONNX Talker + AX CP   (ablate talker)\n"
            "  --mode=2       AX Talker + ONNX CP   (ablate CP)\n"
            "  --mode=3       ONNX Talker + ONNX CP (golden reference)\n"
            "  --streaming    Enable streaming text input (default: non-streaming)\n"
            "                 In streaming mode, decode uses trailing_text_hiddens\n"
            "                 instead of tts_pad_vec for each AR step.\n"
            "                 Trailing text is read from trailing_text_hiddens.bin\n"
            "                 or extracted from prefill_embeds[trailing_start:].\n"
        );
        return 1;
    }

    const std::string model_dir = argv[1];
    const std::string onnx_dir  = argv[2];
    const std::string npy_dir   = argv[3];
    AblationMode mode = AblationMode::AX_AX;
    int max_new_tokens = 128;
    bool streaming = false;

    for (int i = 4; i < argc; ++i) {
        if (strncmp(argv[i], "--mode=", 7) == 0) {
            mode = parse_mode(argv[i] + 7);
        } else if (strcmp(argv[i], "--streaming") == 0) {
            streaming = true;
        } else {
            max_new_tokens = std::atoi(argv[i]);
        }
    }

    printf("model_dir      : %s\n", model_dir.c_str());
    printf("onnx_dir       : %s\n", onnx_dir.c_str());
    printf("npy_dir        : %s\n", npy_dir.c_str());
    printf("mode           : %d\n", static_cast<int>(mode));
    printf("streaming      : %s\n", streaming ? "true" : "false");
    printf("max_new_tokens : %d\n", max_new_tokens);

    // ── 0. 读取 meta.json ──────────────────────────────────────────────────
    const std::string meta_path = npy_dir + "/meta.json";
    if (!std::filesystem::exists(meta_path)) { ALOGE("meta.json not found"); return 1; }
    nlohmann::json meta;
    { std::ifstream f(meta_path); f >> meta; }
    const int S           = meta.value("S", 85);
    const int hidden_size = meta.value("hidden_size", 1024);
    const int audio_token_id = meta.value("audio_token_id", 151644);
    const int trailing_start = meta.value("trailing_start", 7);
    printf("S=%d hidden_size=%d trailing_start=%d\n", S, hidden_size, trailing_start);

    // ── 1. 读取 prefill_embeds.bin (float32) ────────────────────────────────
    const std::string embed_path = npy_dir + "/prefill_embeds.bin";
    std::vector<uint8_t> embed_raw;
    if (!read_binary_file(embed_path, embed_raw)) return 1;
    const size_t expected_bytes = (size_t)S * hidden_size * sizeof(float);
    if (embed_raw.size() != expected_bytes) { ALOGE("embed size mismatch"); return 1; }
    std::vector<float> prefill_embeds_fp32(
        reinterpret_cast<float *>(embed_raw.data()),
        reinterpret_cast<float *>(embed_raw.data()) + (size_t)S * hidden_size
    );

    // ── 1.5 读取/构造 trailing_text_hiddens（流式输入用）────────────────────
    std::vector<std::vector<float>> trailing_text_hiddens;
    if (streaming) {
        const std::string trail_path = npy_dir.back() == '/' ? npy_dir + "trailing_text_hiddens.bin"
                                                             : npy_dir + "/trailing_text_hiddens.bin";
        if (std::filesystem::exists(trail_path)) {
            std::vector<uint8_t> trail_raw;
            if (read_binary_file(trail_path, trail_raw)) {
                int T = static_cast<int>(trail_raw.size() / (hidden_size * sizeof(float)));
                const float *trail_data = reinterpret_cast<const float *>(trail_raw.data());
                trailing_text_hiddens.resize(T);
                for (int t = 0; t < T; ++t) {
                    trailing_text_hiddens[t].assign(trail_data + t * hidden_size,
                                                    trail_data + (t + 1) * hidden_size);
                }
                printf("Loaded trailing_text_hiddens.bin: T=%d\n", T);
            }
        } else if (S > trailing_start + 2) {
            // Only extract from prefill_embeds if it looks like a mixed layout
            // (e.g. S=85/93 where trailing text is appended after prefill core).
            // For dedicated streaming prefill (S≈8), trailing must be provided
            // separately via trailing_text_hiddens.bin.
            int T = S - trailing_start;
            trailing_text_hiddens.resize(T);
            for (int t = 0; t < T; ++t) {
                const float *src = prefill_embeds_fp32.data() + (trailing_start + t) * hidden_size;
                trailing_text_hiddens[t].assign(src, src + hidden_size);
            }
            printf("Extracted trailing_text from prefill_embeds: trailing_start=%d T=%d\n",
                   trailing_start, T);
            printf("[WARNING] For true streaming, provide trailing_text_hiddens.bin "
                   "or ensure prefill_embeds is mixed layout (S >> trailing_start).\n");
        } else {
            printf("[ERROR] streaming mode requires trailing_text_hiddens.bin, "
                   "but file not found and prefill_embeds (S=%d) is too short to extract.\n", S);
            printf("        Please generate streaming data or run without --streaming.\n");
            return 1;
        }
    }

    // bf16 copy for AXModel talker
    std::vector<unsigned short> prefill_embeds_bf16((size_t)S * hidden_size);
    fp32_vec_to_bf16(prefill_embeds_fp32.data(), prefill_embeds_bf16.data(), (int)prefill_embeds_fp32.size());

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
    llm.SetDebugDumpDir(npy_dir);

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
        float temperature = 0.9f;
        int top_k = 50;
        float top_p = 1.0f;
        int vocab_size = 2048;
        bool OnCpFrame(const std::vector<unsigned short> &last_hidden_bf16,
                       int primary_code,
                       std::vector<int> &out_frame_codes,
                       std::vector<unsigned short> &out_codec_sum_bf16) override
        {
            if (!cp) return false;
            std::vector<float> last_hidden_fp32(hidden_size);
            bf16_vec_to_fp32(last_hidden_bf16.data(), last_hidden_fp32.data(), hidden_size);
            auto res = cp->RunFrame(last_hidden_fp32.data(), primary_code, hidden_size,
                                    temperature, top_k, top_p, vocab_size);
            out_frame_codes = res.frame_codes;
            out_codec_sum_bf16.resize(hidden_size);
            fp32_vec_to_bf16(res.codec_sum_fp32.data(), out_codec_sum_bf16.data(), hidden_size);
            return true;
        }
    };
    const int codec_eos_token_id = 2150;
    const int talker_vocab_size = 3072;
    const int code_predictor_vocab_size = 2048;
    const float temperature = 0.9f;
    const int top_k = 50;
    const float top_p = 1.0f;
    const float repetition_penalty = 1.05f;
    const float sub_temperature = 0.9f;
    const int sub_top_k = 50;
    const float sub_top_p = 1.0f;
    const int suppress_start = talker_vocab_size - 1024;
    const int suppress_end = talker_vocab_size;
    constexpr int kMinNewTokens = 2;

    OnnxCpCallback onnx_cp_callback;
    if (mode == AblationMode::AX_ONNX) {
        onnx_cp_callback.cp = onnx_cp.get();
        onnx_cp_callback.hidden_size = hidden_size;
        onnx_cp_callback.temperature = sub_temperature;
        onnx_cp_callback.top_k = sub_top_k;
        onnx_cp_callback.top_p = sub_top_p;
        onnx_cp_callback.vocab_size = code_predictor_vocab_size;
    }

    // ── 读取 tts_pad_vec.bin（所有模式共用，非流式 decode 必需）─────────────
    const std::string pad_vec_path = npy_dir.back() == '/' ? npy_dir + "tts_pad_vec.bin"
                                                           : npy_dir + "/tts_pad_vec.bin";
    std::vector<uint8_t> pad_raw;
    if (!read_binary_file(pad_vec_path, pad_raw)) return 1;
    if (pad_raw.size() < sizeof(int32_t)) { ALOGE("tts_pad_vec too small"); return 1; }
    int32_t pad_hidden = *reinterpret_cast<int32_t *>(pad_raw.data());
    if (pad_hidden != hidden_size) {
        ALOGE("tts_pad_vec hidden_size mismatch: file=%d, expected=%d", pad_hidden, hidden_size);
        return 1;
    }
    const size_t pad_expected = sizeof(int32_t) + (size_t)hidden_size * sizeof(float);
    if (pad_raw.size() != pad_expected) {
        ALOGE("tts_pad_vec size mismatch: %zu vs expected %zu", pad_raw.size(), pad_expected);
        return 1;
    }
    std::vector<float> tts_pad_vec(
        reinterpret_cast<float *>(pad_raw.data() + sizeof(int32_t)),
        reinterpret_cast<float *>(pad_raw.data() + sizeof(int32_t)) + hidden_size
    );
    // bf16 copy for AXModel non-streaming decode
    std::vector<unsigned short> tts_pad_vec_bf16(hidden_size);
    fp32_vec_to_bf16(tts_pad_vec.data(), tts_pad_vec_bf16.data(), hidden_size);
    llm.SetTtsPadVec(tts_pad_vec_bf16);

    LLM::TtsDecodeResult tts_result;
    const auto t0 = std::chrono::steady_clock::now();

    // ═══════════════════════════════════════════════════════════════════════
    // Mode 0: AX Talker + AX CP  → 直接调用 LLM::RunTts
    // ═══════════════════════════════════════════════════════════════════════
    if (mode == AblationMode::AX_AX) {
        bool ok = llm.RunTts(prefill_embeds_bf16, max_new_tokens, codec_eos_token_id, tts_result, streaming);
        if (!ok) { ALOGE("RunTts failed"); }
    }
    // ═══════════════════════════════════════════════════════════════════════
    // Mode 2: AX Talker + ONNX CP  → 通过 callback 使用 ONNX CP
    // ═══════════════════════════════════════════════════════════════════════
    else if (mode == AblationMode::AX_ONNX) {
        bool ok = llm.RunTtsWithCpCallback(prefill_embeds_bf16, max_new_tokens, codec_eos_token_id, tts_result, &onnx_cp_callback, streaming);
        if (!ok) { ALOGE("RunTtsWithCpCallback failed"); }
    }
    // ═══════════════════════════════════════════════════════════════════════
    // Mode 1/3: ONNX Talker + (AX CP or ONNX CP)  → 手动 loop
    // ═══════════════════════════════════════════════════════════════════════
    else {

        // ONNX Talker prefill
        auto pr = onnx_talker->Prefill(prefill_embeds_fp32.data(), S, hidden_size);

        // ---- Debug dump ONNX prefill outputs ----
        {
            std::string dir = npy_dir.back() == '/' ? npy_dir : npy_dir + "/";
            std::string lh_path = dir + "debug_talker_prefill_last_hidden_onnx.bin";
            FILE *fp = fopen(lh_path.c_str(), "wb");
            if (fp) {
                fwrite(pr.last_hidden.data(), sizeof(float), pr.last_hidden.size(), fp);
                fclose(fp);
                printf("[DEBUG] Saved ONNX prefill last_hidden -> %s\n", lh_path.c_str());
            }
            std::string lg_path = dir + "debug_talker_prefill_logits_onnx.bin";
            fp = fopen(lg_path.c_str(), "wb");
            if (fp) {
                fwrite(pr.logits.data(), sizeof(float), pr.logits.size(), fp);
                fclose(fp);
                printf("[DEBUG] Saved ONNX prefill logits -> %s\n", lg_path.c_str());
            }
        }

        // ---- Debug dump ONNX KV cache ----
        {
            std::string dir = npy_dir.back() == '/' ? npy_dir : npy_dir + "/";
            std::string kvcache_dir = dir + "debug_talker_kvcache_onnx";
            onnx_talker->DumpPrefillKVCache(kvcache_dir, pr, S);
            printf("[DEBUG] Saved ONNX KV cache -> %s\n", kvcache_dir.c_str());
        }

        int first_primary_code = SampleFromLogits(
            pr.logits.data(), static_cast<int32_t>(pr.logits.size()), talker_vocab_size,
            temperature, top_k, top_p, repetition_penalty,
            /*generated_ids=*/{}, suppress_start, suppress_end, codec_eos_token_id,
            /*suppress_eos=*/true);
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
                    auto cp_res = onnx_cp->RunFrame(last_hidden_fp32.data(), primary_code, hidden_size,
                                                    sub_temperature, sub_top_k, sub_top_p,
                                                    code_predictor_vocab_size);
                    frame_codes = cp_res.frame_codes;
                    codec_sum_fp32 = cp_res.codec_sum_fp32;
                }

                printf("frame=%d %d", step, primary_code);
                for (int j = 0; j < 15; ++j) printf(" %d", frame_codes[j + 1]);
                printf("\n"); fflush(stdout);

                // Build next talker input
                // streaming:  codec_sum + trailing_text[step]
                // non-streaming: codec_sum + tts_pad_vec
                std::vector<float> next_in_fp32(hidden_size);
                if (streaming && step < static_cast<int>(trailing_text_hiddens.size())) {
                    const auto &txt_hidden = trailing_text_hiddens[step];
                    for (int d = 0; d < hidden_size; ++d) {
                        next_in_fp32[d] = codec_sum_fp32[d] + txt_hidden[d];
                    }
                } else {
                    for (int d = 0; d < hidden_size; ++d) {
                        next_in_fp32[d] = codec_sum_fp32[d] + tts_pad_vec[d];
                    }
                }

                // ONNX Talker decode
                total_seq_len++;
                auto dr = onnx_talker->Decode(next_in_fp32.data(), total_seq_len, hidden_size, talker_state);
                next_token = SampleFromLogits(
                    dr.logits.data(), static_cast<int32_t>(dr.logits.size()), talker_vocab_size,
                    temperature, top_k, top_p, repetition_penalty,
                    generated_primary, suppress_start, suppress_end, codec_eos_token_id,
                    /*suppress_eos=*/(step + 1) < kMinNewTokens);
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
            j["streaming"] = streaming;
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
