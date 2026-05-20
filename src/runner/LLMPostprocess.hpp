#pragma once
#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>

#include "utils/json.hpp"
#include "utils/sample_log.h"

class LLMPostprocess
{
private:
    // 	控制随机性
    void apply_temperature(std::vector<float> &logits, float temperature)
    {
        for (float &logit : logits)
        {
            logit /= temperature;
        }
    }

    

    // 增强多样性
    void apply_diversity_penalty(std::vector<float> &logits, const std::vector<int> &common_phrases, float penalty)
    {
        for (int token : common_phrases)
        {
            if (token < logits.size())
            {
                logits[token] *= penalty;
            }
        }
    }

    // Softmax function
    std::vector<float> softmax(const std::vector<float> &logits)
    {
        std::vector<float> probs(logits.size());
        float max_logit = *std::max_element(logits.begin(), logits.end());
        float sum = 0.0f;

        for (size_t i = 0; i < logits.size(); ++i)
        {
            probs[i] = std::exp(logits[i] - max_logit);
            sum += probs[i];
        }

        for (float &p : probs)
        {
            p /= sum;
        }

        return probs;
    }

    // 	动态裁剪低概率 token
    int faster_top_p_sampling(const std::vector<float> &logits, float top_p)
    {
        // 计算softmax
        std::vector<float> probs = softmax(logits);

        // 构建最大堆（概率和索引的配对）
        std::vector<std::pair<float, size_t>> prob_index;
        prob_index.reserve(logits.size());
        for (size_t i = 0; i < logits.size(); ++i)
        {
            prob_index.emplace_back(probs[i], i);
        }
        auto cmp = [](const auto &a, const auto &b)
        { return a.first < b.first; };
        std::make_heap(prob_index.begin(), prob_index.end(), cmp);

        // 提取top-p元素
        std::vector<size_t> filtered_indices;
        std::vector<float> filtered_probs;
        float cumulative_prob = 0.0f;

        while (!prob_index.empty() && cumulative_prob < top_p)
        {
            std::pop_heap(prob_index.begin(), prob_index.end(), cmp);
            auto [prob, index] = prob_index.back();
            prob_index.pop_back();

            cumulative_prob += prob;
            filtered_indices.push_back(index);
            filtered_probs.push_back(prob);

            if (cumulative_prob >= top_p)
                break;
        }

        // 处理边缘情况（概率全零时返回第一个元素）
        if (filtered_indices.empty())
            return 0;

        std::discrete_distribution<int> dist(filtered_probs.begin(), filtered_probs.end());
        return filtered_indices[dist(rng_)];
    }
    int top_p_sampling(const std::vector<float> &logits, float top_p)
    {
        std::vector<float> probs = softmax(logits);

        // Sort indices by probability in descending order
        std::vector<size_t> indices(logits.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j)
                  { return probs[i] > probs[j]; });

        // Compute cumulative probabilities
        float cumulative_prob = 0.0f;
        size_t cut_off = 0;
        for (; cut_off < indices.size(); ++cut_off)
        {
            cumulative_prob += probs[indices[cut_off]];
            if (cumulative_prob >= top_p)
                break;
        }

        // Keep only the top-p probabilities
        std::vector<size_t> filtered_indices(indices.begin(), indices.begin() + cut_off + 1);
        std::vector<float> filtered_probs(filtered_indices.size());
        for (size_t i = 0; i < filtered_indices.size(); ++i)
        {
            filtered_probs[i] = probs[filtered_indices[i]];
        }

        // Normalize the probabilities
        float filtered_sum = std::accumulate(filtered_probs.begin(), filtered_probs.end(), 0.0f);
        for (float &p : filtered_probs)
        {
            p /= filtered_sum;
        }

        std::discrete_distribution<int> dist(filtered_probs.begin(), filtered_probs.end());
        return filtered_indices[dist(rng_)];
    }

    // 限制候选 token 数
    int top_k_sampling(const std::vector<float> &logits, int k)
    {
        // std::vector<float> probs = softmax(logits);

        // 获取 top-k 索引
        std::vector<size_t> indices(logits.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + k, indices.end(), [&](size_t i, size_t j)
                          { return logits[i] > logits[j]; });

        // 仅保留 top-k 概率
        std::vector<size_t> filtered_indices(indices.begin(), indices.begin() + k);
        std::vector<float> filtered_probs(k);
        for (size_t i = 0; i < k; ++i)
        {
            filtered_probs[i] = logits[filtered_indices[i]];
        }
        filtered_probs = softmax(filtered_probs);

        // 归一化
        float sum = std::accumulate(filtered_probs.begin(), filtered_probs.end(), 0.0f);
        for (float &p : filtered_probs)
        {
            p /= sum;
        }

        std::discrete_distribution<int> dist(filtered_probs.begin(), filtered_probs.end());
        return filtered_indices[dist(rng_)];
    }

    bool enable_temperature = false;
    float temperature = 1.0f;

    bool enable_repetition_penalty = false;
    float repetition_penalty = 1.0f;

    bool enable_diversity_penalty = false;
    std::vector<int> common_phrases;
    float diversity_penalty = 1.0f;

    bool enable_top_p_sampling = false;
    float top_p = 1.0f;

    bool enable_top_k_sampling = false;
    int top_k = 1;

    bool enable_suppress_tokens_ = false;
    int suppress_range_begin_ = -1;
    int suppress_range_end_ = -1;
    int suppress_except_token_ = -1;

    std::mt19937 rng_;

public:
    LLMPostprocess() : rng_(std::random_device{}()) {}

    void set_temperature(bool enable, float temperature)
    {
        enable_temperature = enable;
        this->temperature = temperature;
    }

    void set_repetition_penalty(bool enable, float penalty)
    {
        enable_repetition_penalty = enable;
        this->repetition_penalty = penalty;
    }

    void set_diversity_penalty(bool enable, const std::vector<int> &common_phrases, float penalty)
    {
        enable_diversity_penalty = enable;
        this->common_phrases = common_phrases;
        this->diversity_penalty = penalty;
    }

    void set_top_p_sampling(bool enable, float top_p)
    {
        enable_top_p_sampling = enable;
        this->top_p = top_p;
    }

    void set_top_k_sampling(bool enable, int top_k)
    {
        enable_top_k_sampling = enable;
        this->top_k = top_k;
    }

    void set_suppress_range(int begin, int end, int except_token = -1)
    {
        enable_suppress_tokens_ = true;
        suppress_range_begin_ = begin;
        suppress_range_end_ = end;
        suppress_except_token_ = except_token;
    }

    void set_seed(int seed)
    {
        rng_.seed(static_cast<unsigned int>(seed));
    }

    bool load_config(std::string config_path)
    {
        std::ifstream config_file(config_path);
        if (!config_file.is_open())
        {
            ALOGE("config file(%s) open failed", config_path.c_str());
            return false;
        }
        nlohmann::json config = nlohmann::json::parse(config_file);
        ALOGI("load config: \n%s\n", config.dump(4).c_str());

        enable_temperature = config["enable_temperature"];
        temperature = config["temperature"];
        if (temperature <= 0.0f) temperature = 1.0f;

        enable_repetition_penalty = config["enable_repetition_penalty"];
        repetition_penalty = config["repetition_penalty"];
        if (repetition_penalty < 0.0f) repetition_penalty = 1.0f;

        enable_top_p_sampling = config["enable_top_p_sampling"];
        top_p = config["top_p"];
        if (top_p <= 0.0f) top_p = 0.9f; // reasonable default
        if (top_p > 1.0f) top_p = 1.0f;

        enable_top_k_sampling = config["enable_top_k_sampling"];
        top_k = config["top_k"];
        if (top_k < 1) top_k = 1;

        // top_k 与 top_p 可同时开启，组合策略与 Sherpa-ONNX 一致
        return true;
    }

    // Aligned with Python _select_next_code_from_logits (float64 precision)
    int sample_from_logits(std::vector<float> &buf,
                           float temperature,
                           int top_k,
                           float top_p,
                           float repetition_penalty,
                           const std::vector<int> &generated_ids)
    {
        const int V = static_cast<int>(buf.size());

        if (repetition_penalty > 1.0f) {
            for (int id : generated_ids) {
                if (id >= 0 && id < V) {
                    buf[id] = buf[id] >= 0 ? buf[id] / repetition_penalty
                                          : buf[id] * repetition_penalty;
                }
            }
        }

        if (temperature < 1e-6f) {
            return static_cast<int>(std::max_element(buf.begin(), buf.end()) - buf.begin());
        }

        // Convert to double for precision alignment with Python float64
        std::vector<double> scores(V);
        for (int i = 0; i < V; ++i) scores[i] = static_cast<double>(buf[i]) / temperature;

        if (top_k > 0 && top_k < V) {
            std::vector<double> tmp(scores.begin(), scores.end());
            std::partial_sort(tmp.begin(), tmp.begin() + top_k, tmp.end(), std::greater<double>());
            const double thr = tmp[top_k - 1];
            for (auto &v : scores)
                if (v < thr) v = -1e30;
        }

        const double max_v = *std::max_element(scores.begin(), scores.end());
        double sum = 0;
        for (auto &v : scores) {
            v = std::exp(v - max_v);
            sum += v;
        }
        sum = std::max(sum, 1e-12);
        for (auto &v : scores) v /= sum;

        if (top_p < 1.0f && top_p > 0.0f) {
            std::vector<std::pair<double, int>> pi(V);
            for (int i = 0; i < V; ++i) pi[i] = {scores[i], i};
            std::sort(pi.begin(), pi.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
            double cum = 0;
            int cut = V;
            for (int i = 0; i < V; ++i) {
                cum += pi[i].first;
                if (cum > top_p) { cut = i + 1; break; }
            }
            if (cut == 0) cut = 1;
            for (int i = cut; i < V; ++i) scores[pi[i].second] = 0.0;
            double ns = 0;
            for (auto v : scores) ns += v;
            ns = std::max(ns, 1e-12);
            for (auto &v : scores) v /= ns;
        }

        std::discrete_distribution<int> dist(scores.begin(), scores.end());
        return dist(rng_);
    }

    int apply(std::vector<float> &logits, const std::vector<int> &history)
    {
        if (enable_diversity_penalty)
            apply_diversity_penalty(logits, common_phrases, diversity_penalty);

        if (enable_suppress_tokens_ && suppress_range_begin_ >= 0)
        {
            const int end = std::min(suppress_range_end_, static_cast<int>(logits.size()));
            const float neg_inf = -1e30f;
            for (int i = suppress_range_begin_; i < end; ++i)
            {
                if (i != suppress_except_token_)
                    logits[i] = neg_inf;
            }
        }

        float temp = enable_temperature ? temperature : 0.0f;
        int k = enable_top_k_sampling ? top_k : 0;
        float p = enable_top_p_sampling ? top_p : 1.0f;
        float rep_pen = enable_repetition_penalty ? repetition_penalty : 1.0f;
        return sample_from_logits(logits, temp, k, p, rep_pen, history);
    }
};
