/**
 * qwen3_tts_onnx.cpp
 * ────────────────────────────────────────────────────────────────────────────
 * Qwen3-TTS ONNX-only inference tool (Mode 3: ONNX Talker + ONNX CP).
 *
 * Runs entirely on CPU via ONNX Runtime.  No AXera SDK / NPU required.
 *
 * Usage:
 *   qwen3_tts_onnx <onnx_dir> <embed_dir> [max_new_tokens]
 *
 *   onnx_dir    Directory containing talker_prefill.onnx, talker_decode.onnx,
 *               code_predictor.onnx, code_predictor_embed.onnx, codec_embed.onnx
 *   embed_dir   Directory containing prefill_embeds.bin and meta.json
 *   max_new_tokens  Default 128
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

#include "utils/json.hpp"

#include <onnxruntime_cxx_api.h>

// ── BF16 helper ────────────────────────────────────────────────────────────
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

// ── 读取 raw binary ────────────────────────────────────────────────────────
static bool read_binary_file(const std::string &path, std::vector<uint8_t> &out)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) { fprintf(stderr, "[ERROR] Cannot open file: %s\n", path.c_str()); return false; }
    std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);
    out.resize((size_t)sz);
    if (!f.read(reinterpret_cast<char *>(out.data()), sz)) { fprintf(stderr, "[ERROR] Read failed: %s\n", path.c_str()); return false; }
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// ONNX Talker
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
        : env_(ORT_LOGGING_LEVEL_WARNING, "qwen3_tts_onnx"),
          allocator_()
    {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(8);
        opts.SetInterOpNumThreads(8);
        prefill_sess_ = std::make_unique<Ort::Session>(env_, prefill_path.c_str(), opts);
        decode_sess_  = std::make_unique<Ort::Session>(env_, decode_path.c_str(), opts);
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
        auto &logits_tensor = outs[0];
        auto &hidden_tensor = outs[1];
        auto logits_shape = logits_tensor.GetTensorTypeAndShapeInfo().GetShape();
        auto hidden_shape = hidden_tensor.GetTensorTypeAndShapeInfo().GetShape();
        int vocab_size = static_cast<int>(logits_shape.back());
        const float *logits_data = logits_tensor.GetTensorData<float>();
        const float *hidden_data = hidden_tensor.GetTensorData<float>();
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
// Sampling helper forward declaration
// ═══════════════════════════════════════════════════════════════════════════
static int SampleFromLogits(const float *logits_data, int32_t total, int32_t vocab_size,
                            float temperature, int32_t top_k, float top_p,
                            float repetition_penalty,
                            const std::vector<int> &generated_ids,
                            int32_t suppress_start, int32_t suppress_end,
                            int suppress_exception, bool suppress_eos);

// ═══════════════════════════════════════════════════════════════════════════
// ONNX Code Predictor
// ═══════════════════════════════════════════════════════════════════════════
class OnnxCp {
public:
    struct Result {
        std::vector<int> frame_codes;      // [16]
        std::vector<float> codec_sum_fp32; // [hidden_size]
    };

    OnnxCp(const std::string &cp_path, const std::string &cp_embed_path,
           const std::string &codec_embed_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "qwen3_tts_onnx"),
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

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

struct TtsFrame {
    int codes[16];
};

struct TtsDecodeResult {
    std::vector<TtsFrame> frames;
};

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
            "Usage: qwen3_tts_onnx <onnx_dir> <embed_dir> [max_new_tokens]\n"
            "\n"
            "  onnx_dir       ONNX model directory\n"
            "  embed_dir      Directory with prefill_embeds.bin and meta.json\n"
            "  max_new_tokens Default 128\n"
        );
        return 1;
    }

    const std::string onnx_dir  = argv[1];
    const std::string embed_dir = argv[2];
    int max_new_tokens = (argc >= 4) ? std::atoi(argv[3]) : 128;

    printf("onnx_dir       : %s\n", onnx_dir.c_str());
    printf("embed_dir      : %s\n", embed_dir.c_str());
    printf("max_new_tokens : %d\n", max_new_tokens);

    // ── 0. 读取 meta.json ──────────────────────────────────────────────────
    const std::string meta_path = embed_dir + "/meta.json";
    if (!std::filesystem::exists(meta_path)) { fprintf(stderr, "[ERROR] meta.json not found\n"); return 1; }
    nlohmann::json meta;
    { std::ifstream f(meta_path); f >> meta; }
    const int S           = meta.value("S", 85);
    const int hidden_size = meta.value("hidden_size", 1024);
    const int audio_token_id = meta.value("audio_token_id", 151644);
    printf("S=%d hidden_size=%d\n", S, hidden_size);

    // ── 1. 读取 prefill_embeds.bin (float32) ────────────────────────────────
    const std::string embed_path = embed_dir + "/prefill_embeds.bin";
    std::vector<uint8_t> embed_raw;
    if (!read_binary_file(embed_path, embed_raw)) return 1;
    const size_t expected_bytes = (size_t)S * hidden_size * sizeof(float);
    if (embed_raw.size() != expected_bytes) { fprintf(stderr, "[ERROR] embed size mismatch\n"); return 1; }
    std::vector<float> prefill_embeds_fp32(
        reinterpret_cast<float *>(embed_raw.data()),
        reinterpret_cast<float *>(embed_raw.data()) + (size_t)S * hidden_size
    );

    // ── 1.5 读取 tts_pad_vec.bin (sherpa-onnx non-streaming decode 用) ───────
    const std::string pad_vec_path = embed_dir + "/tts_pad_vec.bin";
    std::vector<uint8_t> pad_raw;
    if (!read_binary_file(pad_vec_path, pad_raw)) return 1;
    if (pad_raw.size() < sizeof(int32_t)) { fprintf(stderr, "[ERROR] tts_pad_vec too small\n"); return 1; }
    int32_t pad_hidden = *reinterpret_cast<int32_t *>(pad_raw.data());
    if (pad_hidden != hidden_size) {
        fprintf(stderr, "[ERROR] tts_pad_vec hidden_size mismatch: file=%d, expected=%d\n", pad_hidden, hidden_size);
        return 1;
    }
    const size_t pad_expected = sizeof(int32_t) + (size_t)hidden_size * sizeof(float);
    if (pad_raw.size() != pad_expected) {
        fprintf(stderr, "[ERROR] tts_pad_vec size mismatch: %zu vs expected %zu\n", pad_raw.size(), pad_expected);
        return 1;
    }
    std::vector<float> tts_pad_vec(
        reinterpret_cast<float *>(pad_raw.data() + sizeof(int32_t)),
        reinterpret_cast<float *>(pad_raw.data() + sizeof(int32_t)) + hidden_size
    );

    // ── 2. 初始化 ONNX 模型 ────────────────────────────────────────────────
    OnnxTalker onnx_talker(onnx_dir + "/talker_prefill.onnx",
                           onnx_dir + "/talker_decode.onnx");
    OnnxCp onnx_cp(onnx_dir + "/code_predictor.onnx",
                   onnx_dir + "/code_predictor_embed.onnx",
                   onnx_dir + "/codec_embed.onnx");

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

    TtsDecodeResult tts_result;
    const auto t0 = std::chrono::steady_clock::now();

    // ONNX Talker prefill
    auto pr = onnx_talker.Prefill(prefill_embeds_fp32.data(), S, hidden_size);
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
            auto cp_res = onnx_cp.RunFrame(last_hidden_fp32.data(), primary_code, hidden_size,
                                            sub_temperature, sub_top_k, sub_top_p,
                                            code_predictor_vocab_size);

            printf("frame=%d %d", step, primary_code);
            for (int j = 0; j < 15; ++j) printf(" %d", cp_res.frame_codes[j + 1]);
            printf("\n"); fflush(stdout);

            // Build next talker input = codec_sum + tts_pad_vec
            // (与 sherpa-onnx non-streaming 模式一致：decode 阶段始终用 tts_pad_vec)
            std::vector<float> next_in_fp32(hidden_size);
            for (int d = 0; d < hidden_size; ++d) {
                next_in_fp32[d] = cp_res.codec_sum_fp32[d] + tts_pad_vec[d];
            }

            // ONNX Talker decode
            total_seq_len++;
            auto dr = onnx_talker.Decode(next_in_fp32.data(), total_seq_len, hidden_size, talker_state);
            next_token = SampleFromLogits(
                dr.logits.data(), static_cast<int32_t>(dr.logits.size()), talker_vocab_size,
                temperature, top_k, top_p, repetition_penalty,
                generated_primary, suppress_start, suppress_end, codec_eos_token_id,
                /*suppress_eos=*/(step + 1) < kMinNewTokens);
            last_hidden_fp32 = dr.last_hidden;
            talker_state = std::move(dr.state);
            generated_primary.push_back(next_token);

            // save frame
            TtsFrame frame;
            for (int i = 0; i < 16; ++i) frame.codes[i] = cp_res.frame_codes[i];
            tts_result.frames.push_back(frame);
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("[TIME]   %.2f ms\n", elapsed_ms);
    printf("[RESULT] frames=%zu\n", tts_result.frames.size());

    // ── 保存结果 ────────────────────────────────────────────────────────────
    if (!tts_result.frames.empty()) {
        std::string out_bin = embed_dir.back() == '/' ? embed_dir + "output_codes_3.bin"
                                                      : embed_dir + "/output_codes_3.bin";
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
        std::string out_meta = embed_dir.back() == '/' ? embed_dir + "output_meta_3.json"
                                                       : embed_dir + "/output_meta_3.json";
        {
            nlohmann::json j;
            j["num_frames"] = (int)tts_result.frames.size();
            j["num_codebooks"] = 16;
            j["dtype"] = "int32";
            j["shape"] = { (int)tts_result.frames.size(), 16 };
            j["codec_eos_token_id"] = codec_eos_token_id;
            j["mode"] = 3;
            std::ofstream ofs(out_meta);
            ofs << j.dump(2);
            printf("[SAVE]   %s\n", out_meta.c_str());
        }
    }

    return 0;
}
