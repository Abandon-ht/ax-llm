
    // ========================================================================
    // Qwen3-TTS Code Predictor & TTS Decode
    // ========================================================================

    bool InitCp(const std::string &cp_dir)
    {
        char path[1024];
        cp_layers.resize(5);
        auto shape_to_str2 = [](const std::vector<unsigned int> &shape) -> std::string {
            std::string s;
            for (size_t i2 = 0; i2 < shape.size(); ++i2) { if (i2) s.push_back('x'); s += std::to_string(shape[i2]); }
            if (s.empty()) s = "(none)";
            return s;
        };
        for (int i = 0; i < 5; i++)
        {
            std::snprintf(path, sizeof(path), "%s/qwen3_tts_talker_code_predictor_p64_l%d_together.axmodel", cp_dir.c_str(), i);
            cp_layers[i].filename = path;
            int ret = cp_layers[i].layer.init(static_cast<const char*>(path), (int)-1);
            if (ret != 0) { ALOGE("init cp layer(%s) failed", path); return false; }
            if (i == 0) {
                ALOGI("CP layer0 groups=%d", cp_layers[0].layer.get_num_input_groups());
                for (int gid = 0; gid < cp_layers[0].layer.get_num_input_groups(); ++gid) {
                    ALOGI("  group %d inputs:", gid);
                    for (int k = 0; k < cp_layers[0].layer.get_num_inputs(); ++k) {
                        auto &t = cp_layers[0].layer.get_input(gid, k);
                        ALOGI("    in[%d]: name=%s shape=%s nSize=%d", k, t.sName.c_str(), shape_to_str2(t.vShape).c_str(), t.nSize);
                    }
                    ALOGI("  group %d outputs:", gid);
                    for (int k = 0; k < cp_layers[0].layer.get_num_outputs(); ++k) {
                        auto &t = cp_layers[0].layer.get_output(gid, k);
                        ALOGI("    out[%d]: name=%s shape=%s nSize=%d", k, t.sName.c_str(), shape_to_str2(t.vShape).c_str(), t.nSize);
                    }
                }
            }
        }
        {
            std::snprintf(path, sizeof(path), "%s/qwen3_tts_talker_code_predictor_post.axmodel", cp_dir.c_str());
            int ret = cp_post.init(static_cast<const char*>(path), (int)-1);
            if (ret != 0) { ALOGE("init cp post(%s) failed", path); return false; }
            auto shape_to_str2 = [](const std::vector<unsigned int> &shape) -> std::string {
                std::string s;
                for (size_t i = 0; i < shape.size(); ++i) { if (i) s.push_back('x'); s += std::to_string(shape[i]); }
                if (s.empty()) s = "(none)";
                return s;
            };
            ALOGI("CP post inputs=%d outputs=%d", cp_post.get_num_inputs(), cp_post.get_num_outputs());
            for (int i = 0; i < cp_post.get_num_inputs(); ++i) {
                auto &t = cp_post.get_input(i);
                ALOGI("  CP post in[%d]: name=%s shape=%s nSize=%d", i, t.sName.c_str(), shape_to_str2(t.vShape).c_str(), t.nSize);
            }
            for (int i = 0; i < cp_post.get_num_outputs(); ++i) {
                auto &t = cp_post.get_output(i);
                ALOGI("  CP post out[%d]: name=%s shape=%s nSize=%d", i, t.sName.c_str(), shape_to_str2(t.vShape).c_str(), t.nSize);
            }
        }
        cp_lm_heads.resize(15);
        for (int i = 0; i < 15; i++)
        {
            std::snprintf(path, sizeof(path), "%s/code_predictor_lm_head_%d.axmodel", cp_dir.c_str(), i);
            int ret = cp_lm_heads[i].init(static_cast<const char*>(path), (int)-1);
            if (ret != 0) { ALOGE("init cp lm_head(%s) failed", path); return false; }
            if (i == 0) {
                auto shape_to_str2 = [](const std::vector<unsigned int> &shape) -> std::string {
                    std::string s;
                    for (size_t i2 = 0; i2 < shape.size(); ++i2) { if (i2) s.push_back('x'); s += std::to_string(shape[i2]); }
                    if (s.empty()) s = "(none)";
                    return s;
                };
                ALOGI("CP lm_head_0 inputs=%d outputs=%d", cp_lm_heads[0].get_num_inputs(), cp_lm_heads[0].get_num_outputs());
                for (int k = 0; k < cp_lm_heads[0].get_num_inputs(); ++k) {
                    auto &t = cp_lm_heads[0].get_input(k);
                    ALOGI("  CP lm_head_0 in[%d]: name=%s shape=%s nSize=%d", k, t.sName.c_str(), shape_to_str2(t.vShape).c_str(), t.nSize);
                }
                for (int k = 0; k < cp_lm_heads[0].get_num_outputs(); ++k) {
                    auto &t = cp_lm_heads[0].get_output(k);
                    ALOGI("  CP lm_head_0 out[%d]: name=%s shape=%s nSize=%d", k, t.sName.c_str(), shape_to_str2(t.vShape).c_str(), t.nSize);
                }
            }
        }
        // load 15 embedding tables: 2048 x 1024 x 2 = 4MB each
        cp_embed_tables.resize(15);
        for (int i = 0; i < 15; i++)
        {
            std::snprintf(path, sizeof(path), "%s/talker.code_predictor.model.codec_embedding.%d.weight.bfloat16.bin", cp_dir.c_str(), i);
            std::ifstream fin(path, std::ios::binary | std::ios::ate);
            if (!fin.is_open()) { ALOGE("cp embed(%s) open failed", path); return false; }
            std::streamsize sz = fin.tellg();
            fin.seekg(0, std::ios::beg);
            const size_t expect = (size_t)2048 * 1024 * 2;
            if ((size_t)sz != expect) { ALOGE("cp embed(%s) size mismatch %zu vs %zu", path, (size_t)sz, expect); return false; }
            cp_embed_tables[i].resize((size_t)2048 * 1024);
            fin.read((char *)cp_embed_tables[i].data(), sz);
            if (!fin) { ALOGE("cp embed(%s) read failed", path); return false; }
        }

        // detect prefill/decode gid from first cp layer
        {
            auto &ref = cp_layers[0].layer;
            int best_gid = 0;
            size_t best_elems = 0;
            int ng = ref.get_num_input_groups();
            for (int gid = 0; gid < ng; ++gid)
            {
                auto &t_idx = ref.get_input(gid, "indices");
                size_t elems = t_idx.nSize / sizeof(unsigned int);
                if (elems > best_elems) { best_elems = elems; best_gid = gid; }
            }
            cp_prefill_gid = best_gid;

            int decode_gid = -1;
            for (int gid = 0; gid < ng; ++gid)
            {
                auto &t_idx = ref.get_input(gid, "indices");
                size_t elems = t_idx.nSize / sizeof(unsigned int);
                if (elems == 1) { decode_gid = gid; break; }
            }
            cp_decode_gid = decode_gid;

            {
                auto &t_idx = ref.get_input(cp_prefill_gid, "indices");
                if (!t_idx.vShape.empty()) cp_prefill_token_num = (int)t_idx.vShape.back();
                else cp_prefill_token_num = (int)(t_idx.nSize / sizeof(unsigned int));
            }
            if (cp_prefill_token_num <= 1) cp_prefill_token_num = 64;

            cp_kv_cache_num = 0;
            for (int gid = 0; gid < ng; ++gid) {
                auto &t_k = ref.get_input(gid, "K_cache");
                int detected = 0;
                if (t_k.vShape.size() >= 2) {
                    detected = (int)t_k.vShape[1];
                } else if (cp_kv_dim > 0) {
                    detected = (int)(t_k.nSize / sizeof(unsigned short)) / cp_kv_dim;
                }
                ALOGI("CP K_cache group %d shape=%s detected_kv_cache_num=%d",
                      gid, shape_to_str2(t_k.vShape).c_str(), detected);
                if (detected > cp_kv_cache_num) cp_kv_cache_num = detected;
            }
            if (cp_kv_cache_num <= 0) {
                auto &t_mask = ref.get_input(cp_prefill_gid, "mask");
                int mask_elems = (int)(t_mask.nSize / sizeof(unsigned short));
                if (mask_elems > 0 && (mask_elems % cp_prefill_token_num) == 0)
                    cp_kv_cache_num = mask_elems / cp_prefill_token_num - cp_prefill_token_num;
                else
                    cp_kv_cache_num = cp_prefill_token_num;
            }
            ALOGI("CP kv_cache_num=%d (max from all groups)", cp_kv_cache_num);
            ALOGI("CP prefill_gid=%d decode_gid=%d token_num=%d kv_cache_num=%d", cp_prefill_gid, cp_decode_gid, cp_prefill_token_num, cp_kv_cache_num);
        }

        // Detect cp_kv_dim from K_cache shape
        {
            auto &t_k = cp_layers[0].layer.get_input(cp_prefill_gid, "K_cache");
            if (t_k.vShape.size() >= 3) {
                cp_kv_dim = (int)t_k.vShape[2];
            } else {
                cp_kv_dim = (t_k.nSize / sizeof(unsigned short)) / cp_kv_cache_num;
            }
            ALOGI("CP kv_dim=%d (from K_cache shape)", cp_kv_dim);
        }

        // Pre-allocate CP KV caches
        cp_k_cache.resize(5);
        cp_v_cache.resize(5);
        for (int m = 0; m < 5; ++m) {
            cp_k_cache[m].assign((size_t)cp_kv_cache_num * cp_kv_dim, 0);
            cp_v_cache[m].assign((size_t)cp_kv_cache_num * cp_kv_dim, 0);
        }

        // Load talker norm gamma for CP input
        {
            std::string talker_dir = (std::filesystem::path(cp_dir) / ".." / "talker").lexically_normal().string();
            std::string norm_path = talker_dir + "/talker.model.norm.weight.bfloat16.bin";
            std::ifstream fin(norm_path, std::ios::binary | std::ios::ate);
            if (fin.is_open()) {
                std::streamsize sz = fin.tellg();
                fin.seekg(0, std::ios::beg);
                int n = sz / sizeof(unsigned short);
                cp_norm_gamma.resize(n);
                std::vector<unsigned short> tmp(n);
                fin.read((char*)tmp.data(), sz);
                for (int i = 0; i < n; ++i) cp_norm_gamma[i] = bfloat16(tmp[i]).fp32();
                ALOGI("Loaded talker norm gamma from %s, elems=%d", norm_path.c_str(), n);
            } else {
                ALOGW("Cannot load talker norm gamma from %s, CP input will be unnormalized!", norm_path.c_str());
            }
        }

        cp_loaded = true;
        return true;
    }

    void DeinitCp()
    {
        for (auto &lyr : cp_layers) lyr.layer.deinit();
        cp_post.deinit();
        for (auto &m : cp_lm_heads) m.deinit();
        cp_lm_heads.clear();
        cp_embed_tables.clear();
        cp_k_cache.clear();
        cp_v_cache.clear();
        cp_loaded = false;
    }

    static inline void rmsnorm_bf16(unsigned short *dst, const unsigned short *src,
                                     const float *gamma, int n, float eps=1e-6)
    {
        float sum_sq = 0;
        for (int i = 0; i < n; ++i) {
            float v = bfloat16(src[i]).fp32();
            sum_sq += v * v;
        }
        float rms = std::sqrt(sum_sq / n + eps);
        for (int i = 0; i < n; ++i) {
            float v = bfloat16(src[i]).fp32();
            dst[i] = bfloat16((v / rms) * gamma[i]).data;
        }
    }

    static inline void bf16_vec_to_fp32(const unsigned short *src, float *dst, int n)
    {
        for (int i = 0; i < n; ++i)
        {
            unsigned int u = ((unsigned int)src[i]) << 16;
            dst[i] = *reinterpret_cast<float *>(&u);
        }
    }

    static inline void fp32_vec_to_bf16(const float *src, unsigned short *dst, int n)
    {
        for (int i = 0; i < n; ++i) dst[i] = bfloat16(src[i]).data;
    }

    // Sampling for CP logits (aligned with Python float64 precision)
    int CpSampleFromLogits(const std::vector<float> &logits,
                            float temperature,
                            int top_k,
                            float top_p)
    {
        if (logits.empty()) return 0;
        const int V = static_cast<int>(logits.size());

        if (temperature < 1e-6f) {
            return static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
        }

        // Convert to double for precision alignment with Python float64
        std::vector<double> scores(V);
        for (int i = 0; i < V; ++i) scores[i] = static_cast<double>(logits[i]) / temperature;

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

        return static_cast<int>(std::discrete_distribution<int>(scores.begin(), scores.end())(cp_rng_));
    }

    bool RunCpFrame(const std::vector<unsigned short> &last_hidden_bf16,
                    int primary_code,
                    std::vector<int> &out_frame_codes,
                    std::vector<unsigned short> &out_codec_sum_bf16)
    {
        if (!cp_loaded) return false;
        const int D = _attr.tokens_embed_size; // 1024

        // Reset CP KV cache at the start of each frame
        for (int m = 0; m < 5; ++m) {
            std::fill(cp_k_cache[m].begin(), cp_k_cache[m].end(), (unsigned short)0);
            std::fill(cp_v_cache[m].begin(), cp_v_cache[m].end(), (unsigned short)0);
        }

        // 1. primary embed via talker codec embedding table (first 3072 entries)
        std::vector<unsigned short> primary_embed(D);
        embed_selector.getByIndex((unsigned int)primary_code, primary_embed.data());

        out_frame_codes.resize(16);
        out_frame_codes[0] = primary_code;
        out_codec_sum_bf16 = primary_embed;

        // fp32 accumulation buffer for codec_sum (align with Python torch.sum fp32)
        std::vector<float> codec_sum_fp32(D);
        for (int d = 0; d < D; ++d) codec_sum_fp32[d] = bfloat16(primary_embed[d]).fp32();

        // cp_ctx starts with [last_hidden, primary_embed] to align with Python:
        // Python CP prefill input is torch.cat((past_hidden, last_id_hidden), dim=1)
        std::vector<unsigned short> cp_ctx = last_hidden_bf16;
        cp_ctx.insert(cp_ctx.end(), primary_embed.begin(), primary_embed.end());
        cp_ctx.reserve(17 * D);


        std::vector<unsigned short> mask_tmp((size_t)cp_prefill_token_num * (cp_kv_cache_num + cp_prefill_token_num));
        std::vector<unsigned int> idx_tmp((size_t)cp_prefill_token_num, 0);
        std::vector<unsigned short> embed_tmp((size_t)cp_prefill_token_num * D, 0);

        const float cp_temperature = _attr.cp_temperature;
        const int cp_top_k = _attr.cp_top_k;
        const float cp_top_p = _attr.cp_top_p;

        for (int j = 0; j < 15; ++j)
        {
            const bool is_prefill = (j == 0);
            const int gid = is_prefill ? cp_prefill_gid : (cp_decode_gid >= 0 ? cp_decode_gid : cp_prefill_gid);
            const int seq_len = is_prefill ? 2 : 1;
            const int history_len = is_prefill ? 0 : (j + 1);
            const int current_len = history_len + seq_len;
            if (history_len + seq_len > cp_prefill_token_num)
            {
                ALOGE("CP history_len+seq_len(%d) > cp_prefill_token_num(%d)", history_len + seq_len, cp_prefill_token_num);
                return false;
            }

            // zero buffers
            std::fill(mask_tmp.begin(), mask_tmp.end(), bfloat16(-65536.f).data);
            std::fill(embed_tmp.begin(), embed_tmp.end(), (unsigned short)0);
            std::fill(idx_tmp.begin(), idx_tmp.end(), 0U);

            // build mask
            if (is_prefill || cp_decode_gid < 0) {
                // CP prefill mask is square (token_num x token_num), no KV cache dimension in mask shape
                build_prefill_mask(mask_tmp, 0, cp_prefill_token_num, history_len, seq_len);
            } else {
                // CP decode mask layout matches Python decode_mask_cache row=current_len:
                //   - cols [0 .. history_len-1]      : history KV slots, visible (strict <, not <=)
                //   - cols [history_len .. mask_cap-2]: empty / future slots, masked
                //   - col  [mask_cap - 1]            : current token's own "self" K computed in
                //                                      this forward; always visible (Python's
                //                                      `mask_2d[:, -1] = 0.0` override)
                const int mask_cap = cp_kv_cache_num + 1;
                const int mask_fill = std::min((int)mask_tmp.size(), mask_cap);
                for (int i = 0; i < mask_fill; ++i) {
                    mask_tmp[i] = (i < history_len) ? bfloat16(0.f).data : bfloat16(-65536.f).data;
                }
                if (mask_fill > 0) mask_tmp[mask_fill - 1] = bfloat16(0.f).data;
            }

            // indices
            for (int i = 0; i < seq_len; ++i) idx_tmp[i] = (unsigned int)(history_len + i);

            // embed: cp_ctx[0..D-1] is last_hidden, [D..2D-1] is primary_embed; appended res_embeds follow
            if (is_prefill) {
                memcpy(embed_tmp.data(), cp_ctx.data(), (size_t)seq_len * D * sizeof(unsigned short));
            } else {
                memcpy(embed_tmp.data(), cp_ctx.data() + (cp_ctx.size() - D), D * sizeof(unsigned short));
            }

            // run CP 5 layers
            for (int m = 0; m < 5; ++m)
            {
                auto &lyr = cp_layers[m];
                int devid = LLM_DEVID(lyr);
                auto &t_in = lyr.layer.get_input(gid, "input");
                size_t in_copy = std::min((size_t)t_in.nSize, (size_t)seq_len * D * sizeof(unsigned short));
                llm_h2d(LLM_WADDR(t_in), embed_tmp.data(), in_copy, devid);
                auto &t_mask = lyr.layer.get_input(gid, "mask");
                size_t mask_copy = std::min((size_t)t_mask.nSize, mask_tmp.size() * sizeof(unsigned short));
                llm_h2d(LLM_WADDR(t_mask), mask_tmp.data(), mask_copy, devid);
                auto &t_idx = lyr.layer.get_input(gid, "indices");
                {
                    const int idx_total_elems = (int)(t_idx.nSize / sizeof(unsigned int));
                    const int idx_per_row = (gid == cp_prefill_gid) ? cp_prefill_token_num : 1;
                    const int n_rows = std::max(1, idx_total_elems / std::max(1, idx_per_row));
                    std::vector<unsigned int> full_idx((size_t)idx_total_elems, 0U);
                    for (int r = 0; r < n_rows; ++r) {
                        for (int i = 0; i < seq_len && i < idx_per_row; ++i) {
                            full_idx[(size_t)r * idx_per_row + i] = (unsigned int)(history_len + i);
                        }
                    }
                    llm_h2d(LLM_WADDR(t_idx), full_idx.data(), t_idx.nSize, devid);
                }

                // K/V cache input: always pass full buffer (model uses indices/mask to locate)
                {
                    auto &tk = lyr.layer.get_input(gid, "K_cache");
                    llm_h2d(LLM_WADDR(tk), cp_k_cache[m].data(),
                            std::min((size_t)tk.nSize, cp_k_cache[m].size() * sizeof(unsigned short)), devid);
                }
                {
                    auto &tv = lyr.layer.get_input(gid, "V_cache");
                    llm_h2d(LLM_WADDR(tv), cp_v_cache[m].data(),
                            std::min((size_t)tv.nSize, cp_v_cache[m].size() * sizeof(unsigned short)), devid);
                }

                int inf_ret = lyr.layer.inference(gid);
                if (inf_ret != 0) ALOGE("RunCpFrame layer m=%d inference failed ret=%d", m, inf_ret);

                auto &t_out = lyr.layer.get_output(gid, "output");
                llm_d2h(embed_tmp.data(), LLM_RADDR(t_out),
                        std::min((size_t)t_out.nSize, (size_t)seq_len * D * sizeof(unsigned short)), devid);

                // Update KV cache
                if (is_prefill) {
                    auto &tk_out = lyr.layer.get_output(gid, "K_cache_out");
                    llm_d2h(cp_k_cache[m].data(), LLM_RADDR(tk_out),
                            std::min((size_t)tk_out.nSize, (size_t)seq_len * cp_kv_dim * sizeof(unsigned short)), devid);
                    auto &tv_out = lyr.layer.get_output(gid, "V_cache_out");
                    llm_d2h(cp_v_cache[m].data(), LLM_RADDR(tv_out),
                            std::min((size_t)tv_out.nSize, (size_t)seq_len * cp_kv_dim * sizeof(unsigned short)), devid);
                } else {
                    auto &tk_out = lyr.layer.get_output(gid, "K_cache_out");
                    llm_d2h(cp_k_cache[m].data() + (current_len - 1) * cp_kv_dim, LLM_RADDR(tk_out),
                            std::min((size_t)tk_out.nSize, (size_t)cp_kv_dim * sizeof(unsigned short)), devid);
                    auto &tv_out = lyr.layer.get_output(gid, "V_cache_out");
                    llm_d2h(cp_v_cache[m].data() + (current_len - 1) * cp_kv_dim, LLM_RADDR(tv_out),
                            std::min((size_t)tv_out.nSize, (size_t)cp_kv_dim * sizeof(unsigned short)), devid);
                }
            }

            // take last position hidden state
            std::vector<unsigned short> hidden_step(D);
            memcpy(hidden_step.data(), embed_tmp.data() + (seq_len - 1) * D, D * sizeof(unsigned short));

            // run CP post
            {
                auto &t_in = cp_post.get_input("input");
                size_t in_copy = std::min((size_t)t_in.nSize, hidden_step.size() * sizeof(unsigned short));
                llm_h2d(LLM_WADDR(t_in), hidden_step.data(), in_copy, cp_post.get_devid());
                cp_post.inference();
                auto &t_out = cp_post.get_output("output_norm");
                size_t out_words = (size_t)t_out.nSize / sizeof(unsigned short);
                if (out_words > 0) {
                    std::vector<unsigned short> post_buf(out_words);
                    llm_d2h(post_buf.data(), LLM_RADDR(t_out), t_out.nSize, cp_post.get_devid());
                    size_t copy_words = std::min(out_words, (size_t)D);
                    memcpy(hidden_step.data(), post_buf.data(), copy_words * sizeof(unsigned short));
                }
            }

            // run lm_head_j -> logits [2048]
            auto &lmh = cp_lm_heads[j];
            {
                auto &t_in = lmh.get_input("input");
                size_t in_copy = std::min((size_t)t_in.nSize, hidden_step.size() * sizeof(unsigned short));
                llm_h2d(LLM_WADDR(t_in), hidden_step.data(), in_copy, lmh.get_devid());
                lmh.inference();
                auto &t_out = lmh.get_output("output");
                int logits_n = (int)(t_out.nSize / sizeof(float));
                if (logits_n == 0) { ALOGE("lm_head %d output size=0", j); return false; }
                if (logits_n > 2048) logits_n = 2048;
                std::vector<float> logits_fp32(logits_n);
                llm_d2h(logits_fp32.data(), LLM_RADDR(t_out), logits_n * sizeof(float), lmh.get_devid());
                float max_logit = -1e30f; int max_idx = 0;
                for (int i = 0; i < logits_n; ++i) {
                    if (logits_fp32[i] > max_logit) { max_logit = logits_fp32[i]; max_idx = i; }
                }
                int res_code = CpSampleFromLogits(logits_fp32, cp_temperature, cp_top_k, cp_top_p);
                out_frame_codes[j + 1] = res_code;

                // lookup residual embed
                std::vector<unsigned short> res_embed(D);
                if ((size_t)res_code < 2048)
                {
                    memcpy(res_embed.data(), cp_embed_tables[j].data() + (size_t)res_code * D, D * sizeof(unsigned short));
                }
                // accumulate codec_sum in fp32 (align with Python torch.sum fp32 accumulation)
                for (int d = 0; d < D; ++d)
                {
                    codec_sum_fp32[d] += bfloat16(res_embed[d]).fp32();
                }
                // append to cp_ctx for next step
                cp_ctx.insert(cp_ctx.end(), res_embed.begin(), res_embed.end());
            }
        }

        // Convert fp32 codec_sum back to bf16 (single truncation, matching Python torch.sum)
        for (int d = 0; d < D; ++d) out_codec_sum_bf16[d] = bfloat16(codec_sum_fp32[d]).data;

        return true;
    }

    bool RunTts(std::vector<unsigned short> &prefill_embeds,
                int max_new_tokens,
                int codec_eos_token_id,
                TtsDecodeResult &result,
                bool streaming = true)
    {
        return RunTtsWithCpCallback(prefill_embeds, max_new_tokens, codec_eos_token_id, result, nullptr, streaming);
    }

    bool RunTtsWithCpCallback(std::vector<unsigned short> &prefill_embeds,
                              int max_new_tokens,
                              int codec_eos_token_id,
                              TtsDecodeResult &result,
                              LLM::TtsCpCallback *cp_callback,
                              bool streaming = true)
    {
        if (!cp_loaded) { ALOGE("CP not loaded, cannot RunTts"); return false; }
        b_stop.store(false, std::memory_order_relaxed);
        bfloat16 bf16 = -65536.f;
        bfloat16 bf16_one = 1.0f;
        int max_decode_cap = _attr.max_token_len;
        if (!decode_max_token_len_grp_.empty()) max_decode_cap = std::max(max_decode_cap, decode_max_token_len_grp_.back());
        if (max_decode_cap <= 0) max_decode_cap = _attr.kv_cache_num;
        std::vector<unsigned short> mask((size_t)max_decode_cap + 1, bf16.data);
        if (!mask.empty()) mask.back() = 0;
        for (const int cap : decode_max_token_len_grp_)
        {
            if (cap >= 0 && cap < (int)mask.size()) mask[(size_t)cap] = 0;
        }

        std::vector<unsigned short> embed(_attr.tokens_embed_size, 0);
        std::vector<int> token_ids;
        int input_embed_num = (int)(prefill_embeds.size() / _attr.tokens_embed_size);
        for (int i = 0; i < input_embed_num && i < (int)mask.size(); i++) mask[(size_t)i] = 0;
        int prefill_split_num = (int)ceil((double)input_embed_num / _attr.prefill_token_num);
        ALOGI("RunTts input_embed_num=%d prefill_split_num=%d", input_embed_num, prefill_split_num);

        // prefill group selection
        std::vector<int> prefill_grp_list;
        const int default_prefill_gid = prefill_grpids_.empty() ? 1 : prefill_grpids_.front();
        prefill_grp_list.resize(prefill_split_num, default_prefill_gid);
        int max_prefill_gid = default_prefill_gid;
        for (int p = 0; p < prefill_split_num; ++p)
        {
            const int history_len = p * _attr.prefill_token_num;
            int g = choose_prefill_gid(history_len);
            prefill_grp_list[p] = g;
            if (g > max_prefill_gid) max_prefill_gid = g;
        }

        // ---- Prefill (save all hidden states) ----
        std::vector<unsigned short> all_prefill_hidden((size_t)input_embed_num * (size_t)_attr.tokens_embed_size, 0);
        for (int p = 0; p < prefill_split_num; p++)
        {
            if (b_stop.load(std::memory_order_relaxed)) break;
            int input_num_token = (p == prefill_split_num - 1) ? input_embed_num - p * _attr.prefill_token_num : _attr.prefill_token_num;
            const int history_len = p * _attr.prefill_token_num;
            const int prefill_grpid = prefill_grp_list[p];
            int kv_cache_num_p = prefill_capacity_by_gid(prefill_grpid);
            {
                const auto &mask_t = llama_layers[(size_t)cache_ref_full_layer_idx].layer.get_input(prefill_grpid, "mask");
                const int mask_elems = (int)(mask_t.nSize / (int)sizeof(unsigned short));
                if (_attr.prefill_token_num > 0 && mask_elems > 0 && (mask_elems % _attr.prefill_token_num) == 0) {
                    const int cols = mask_elems / _attr.prefill_token_num;
                    const int kv_from_mask = cols - _attr.prefill_token_num;
                    if (kv_from_mask >= 0) kv_cache_num_p = kv_from_mask;
                }
            }

            std::vector<unsigned short> embed_tmp(_attr.prefill_token_num * _attr.tokens_embed_size, 0);
            std::vector<unsigned short> mask_tmp(_attr.prefill_token_num * (kv_cache_num_p + _attr.prefill_token_num), bf16.data);
            build_prefill_mask(mask_tmp, kv_cache_num_p, _attr.prefill_token_num, history_len, input_num_token);
            size_t copy_tokens = (p == prefill_split_num - 1) ? (size_t)(input_embed_num - p * _attr.prefill_token_num) : (size_t)_attr.prefill_token_num;
            memcpy(embed_tmp.data(), prefill_embeds.data() + p * _attr.prefill_token_num * _attr.tokens_embed_size, copy_tokens * _attr.tokens_embed_size * sizeof(unsigned short));

            for (int m = 0; m < _attr.axmodel_num; m++)
            {
                if (b_stop.load(std::memory_order_relaxed)) break;
                auto &lyr = llama_layers[m]; int devid = LLM_DEVID(lyr);

                // ★ FIX: Propagate previous chunks' K/V history into the CURRENT chunk's
                // prefill K_cache / V_cache input. Each chunk uses a different prefill_grpid,
                // so the per-grpid pre_k buffer is NOT carried over by axengine. Python's
                // StaticTalkerLayerRunner.prefill() (infer.py: k_input = k_caches[:, :start, :])
                // does this explicitly. We mirror that by copying from the decode group's
                // KV buffer, which has been populated by previous chunks via the d2d below.
                if (p > 0) {
                    auto &dec_k_src = lyr.layer.get_input(decode_grpid, "K_cache");
                    auto &dec_v_src = lyr.layer.get_input(decode_grpid, "V_cache");
                    auto &pre_k_dst = lyr.layer.get_input(prefill_grpid, "K_cache");
                    auto &pre_v_dst = lyr.layer.get_input(prefill_grpid, "V_cache");
                    size_t history_bytes = (size_t)history_len * _attr.kv_cache_size * sizeof(unsigned short);
                    history_bytes = std::min(history_bytes, (size_t)pre_k_dst.nSize);
                    history_bytes = std::min(history_bytes, (size_t)pre_v_dst.nSize);
                    history_bytes = std::min(history_bytes, (size_t)dec_k_src.nSize);
                    history_bytes = std::min(history_bytes, (size_t)dec_v_src.nSize);
                    if (history_bytes > 0) {
                        llm_d2d(LLM_WADDR(pre_k_dst), LLM_RADDR(dec_k_src), history_bytes, devid);
                        llm_d2d(LLM_WADDR(pre_v_dst), LLM_RADDR(dec_v_src), history_bytes, devid);
                    }
                }

                auto &t_idx = lyr.layer.get_input(prefill_grpid, "indices");
                unsigned int *idx_ptr = (unsigned int *)t_idx.pVirAddr; memset(idx_ptr, 0, t_idx.nSize);
                const int idx_elems = (int)(t_idx.nSize / sizeof(unsigned int));
                int idx_rows = (idx_elems > 0 && _attr.prefill_token_num > 0) ? (idx_elems / _attr.prefill_token_num) : 1;
                if (idx_rows <= 0) idx_rows = 1;
                for (int r = 0; r < idx_rows; ++r) {
                    for (int i = 0; i < input_num_token; ++i) {
                        idx_ptr[r * _attr.prefill_token_num + i] = (unsigned int)(history_len + i);
                    }
                }
                llm_h2d(LLM_WADDR(t_idx), idx_ptr, t_idx.nSize, devid);
                auto &t_mask = lyr.layer.get_input(prefill_grpid, "mask");
                if (is_linear_layer(m))
                {
                    const size_t elems = (size_t)t_mask.nSize / sizeof(unsigned short);
                    std::vector<unsigned short> linear_mask_tmp(elems, 0);
                    const int n = std::min((int)elems, input_num_token);
                    for (int i = 0; i < n; ++i) linear_mask_tmp[(size_t)i] = bf16_one.data;
                    llm_h2d(LLM_WADDR(t_mask), linear_mask_tmp.data(), linear_mask_tmp.size() * sizeof(unsigned short), devid);
                }
                else
                {
                    llm_h2d(LLM_WADDR(t_mask), mask_tmp.data(), std::min((size_t)t_mask.nSize, mask_tmp.size() * sizeof(unsigned short)), devid);
                }
                auto &t_in = lyr.layer.get_input(prefill_grpid, "input");
                llm_h2d(LLM_WADDR(t_in), embed_tmp.data(), std::min((size_t)t_in.nSize, embed_tmp.size() * sizeof(unsigned short)), devid);
                lyr.layer.inference(prefill_grpid);
                auto &out_k = lyr.layer.get_output(prefill_grpid, "K_cache_out");
                auto &out_v = lyr.layer.get_output(prefill_grpid, "V_cache_out");
                auto &pre_k = lyr.layer.get_input(prefill_grpid, "K_cache");
                auto &pre_v = lyr.layer.get_input(prefill_grpid, "V_cache");
                int kv_off = history_len * _attr.kv_cache_size;
                size_t kv_sz = (size_t)input_num_token * _attr.kv_cache_size * sizeof(unsigned short);
                // ★ FIX: Skip writing back to current grpid's pre_k when it would overflow
                // (e.g. grpid=1 has pre_k.nSize=2048 but we'd want to write 256KB). This write
                // is also useless since the next chunk uses a different grpid; history is
                // propagated via the dec_k path above.
                {
                    size_t pre_k_max = (size_t)pre_k.nSize;
                    size_t pre_off_b  = (size_t)kv_off * sizeof(unsigned short);
                    if (pre_off_b + kv_sz <= pre_k_max) {
                        llm_d2d((unsigned short *)LLM_WADDR(pre_k) + kv_off, LLM_RADDR(out_k), kv_sz, devid);
                        llm_d2d((unsigned short *)LLM_WADDR(pre_v) + kv_off, LLM_RADDR(out_v), kv_sz, devid);
                    }
                }
                // sync to decode group
                auto &dec_k = lyr.layer.get_input(decode_grpid, "K_cache");
                auto &dec_v = lyr.layer.get_input(decode_grpid, "V_cache");
                if (is_linear_layer(m))
                {
                    llm_d2d(LLM_WADDR(dec_k), LLM_RADDR(out_k), std::min((size_t)dec_k.nSize, (size_t)out_k.nSize), devid);
                    llm_d2d(LLM_WADDR(dec_v), LLM_RADDR(out_v), std::min((size_t)dec_v.nSize, (size_t)out_v.nSize), devid);
                }
                else
                {
                    llm_d2d((unsigned short *)LLM_WADDR(dec_k) + kv_off, LLM_RADDR(out_k), kv_sz, devid);
                    llm_d2d((unsigned short *)LLM_WADDR(dec_v) + kv_off, LLM_RADDR(out_v), kv_sz, devid);
                }
                auto &t_out = lyr.layer.get_output(prefill_grpid, "output");
                llm_d2h(embed_tmp.data(), LLM_RADDR(t_out), embed_tmp.size() * sizeof(unsigned short), devid);
            }

            // save hidden states for this chunk
            memcpy(all_prefill_hidden.data() + (size_t)history_len * _attr.tokens_embed_size,
                   embed_tmp.data(), copy_tokens * _attr.tokens_embed_size * sizeof(unsigned short));

            if (p == prefill_split_num - 1)
                memcpy(embed.data(), embed_tmp.data() + (input_num_token - 1) * _attr.tokens_embed_size, _attr.tokens_embed_size * sizeof(unsigned short));
        }

        // ---- Apply RMSNorm to all_prefill_hidden (align with Python normed hidden) ----
        if (!cp_norm_gamma.empty()) {
            for (int t = 0; t < input_embed_num; ++t) {
                rmsnorm_bf16(
                    all_prefill_hidden.data() + (size_t)t * _attr.tokens_embed_size,
                    all_prefill_hidden.data() + (size_t)t * _attr.tokens_embed_size,
                    cp_norm_gamma.data(), _attr.tokens_embed_size);
            }
            ALOGI("RunTts applied RMSNorm to all_prefill_hidden (%d tokens)", input_embed_num);
        }

        // ---- Debug dump: KV cache (after prefill, from decode group) ----
        if (!debug_dump_dir_.empty())
        {
            std::string dir = debug_dump_dir_.back() == '/' ? debug_dump_dir_ : debug_dump_dir_ + "/";
            std::string kvcache_dir = dir + "debug_talker_kvcache_ax";
            std::filesystem::create_directories(kvcache_dir);
            for (int m = 0; m < _attr.axmodel_num; m++)
            {
                auto &lyr = llama_layers[m];
                int devid = LLM_DEVID(lyr);
                auto &t_k = lyr.layer.get_input(decode_grpid, "K_cache");
                auto &t_v = lyr.layer.get_input(decode_grpid, "V_cache");
                size_t valid_kv_bytes = (size_t)input_embed_num * _attr.kv_cache_size * sizeof(unsigned short);

                std::vector<unsigned short> k_bf16((size_t)input_embed_num * _attr.kv_cache_size);
                std::vector<unsigned short> v_bf16((size_t)input_embed_num * _attr.kv_cache_size);
                llm_d2h(k_bf16.data(), LLM_RADDR(t_k), valid_kv_bytes, devid);
                llm_d2h(v_bf16.data(), LLM_RADDR(t_v), valid_kv_bytes, devid);

                std::vector<float> k_fp32(k_bf16.size());
                std::vector<float> v_fp32(v_bf16.size());
                for (size_t i = 0; i < k_bf16.size(); ++i) k_fp32[i] = bfloat16(k_bf16[i]).fp32();
                for (size_t i = 0; i < v_bf16.size(); ++i) v_fp32[i] = bfloat16(v_bf16[i]).fp32();

                char path[1024];
                snprintf(path, sizeof(path), "%s/layer_%02d_k.bin", kvcache_dir.c_str(), m);
                FILE *fp = fopen(path, "wb");
                if (fp) { fwrite(k_fp32.data(), sizeof(float), k_fp32.size(), fp); fclose(fp); }
                snprintf(path, sizeof(path), "%s/layer_%02d_v.bin", kvcache_dir.c_str(), m);
                fp = fopen(path, "wb");
                if (fp) { fwrite(v_fp32.data(), sizeof(float), v_fp32.size(), fp); fclose(fp); }
            }
            ALOGI("[DEBUG] Saved AX KV cache -> %s", kvcache_dir.c_str());
        }

        // ---- First post-process ----
        int next_token = -1;
        {
            auto &t_in = llama_post.get_input("input");
            llm_h2d(LLM_WADDR(t_in), embed.data(), embed.size() * sizeof(unsigned short), LLM_DEVID(llama_layers.back()));
            llama_post.inference();
            auto &t_out = llama_post.get_output("output");
            llm_d2h(t_out.pVirAddr, LLM_RADDR(t_out), t_out.nSize, llama_post.get_devid());
            unsigned short *post_out = (unsigned short *)t_out.pVirAddr;
            next_token = post_process(postprocess, post_out, _attr.tokens_embed_num, token_ids, nullptr);
            token_ids.push_back(next_token);
            ALOGI("RunTts first primary_code=%d", next_token);
        }

        // ---- Debug dump: prefill last_hidden & logits ----
        if (!debug_dump_dir_.empty())
        {
            std::string dir = debug_dump_dir_.back() == '/' ? debug_dump_dir_ : debug_dump_dir_ + "/";
            // Save last hidden state (last token of prefill) as FP32
            {
                std::string path = dir + "debug_talker_prefill_last_hidden_ax.bin";
                FILE *fp = fopen(path.c_str(), "wb");
                if (fp) {
                    std::vector<float> buf(_attr.tokens_embed_size);
                    for (int d = 0; d < _attr.tokens_embed_size; ++d) buf[d] = bfloat16(embed[d]).fp32();
                    fwrite(buf.data(), sizeof(float), buf.size(), fp);
                    fclose(fp);
                    ALOGI("[DEBUG] Saved AX prefill last_hidden -> %s", path.c_str());
                }
            }
            // Save prefill logits (post_out of last token) as FP32
            {
                std::string path = dir + "debug_talker_prefill_logits_ax.bin";
                FILE *fp = fopen(path.c_str(), "wb");
                if (fp) {
                    auto &t_out = llama_post.get_output("output");
                    unsigned short *post_out = (unsigned short *)t_out.pVirAddr;
                    std::vector<float> buf(_attr.tokens_embed_num);
                    for (int d = 0; d < _attr.tokens_embed_num; ++d) buf[d] = bfloat16(post_out[d]).fp32();
                    fwrite(buf.data(), sizeof(float), buf.size(), fp);
                    fclose(fp);
                    ALOGI("[DEBUG] Saved AX prefill logits -> %s", path.c_str());
                }
            }
        }

        // ---- Decode loop with CP ----
        int trailing_start = 7; // text[0] position in 85-token prefill layout
        int step = 0;
        timer t_cost; t_cost.start();
        unsigned int decode_start = (unsigned int)input_embed_num;
        // Diagnostic: log decode indices / mask tensor shape + is_linear_layer status
        {
            auto &t_idx0 = llama_layers[0].layer.get_input(decode_grpid, "indices");
            auto &t_mask0 = llama_layers[0].layer.get_input(decode_grpid, "mask");
            ALOGI("[diag] talker decode indices nSize=%d (elems=%d)  mask nSize=%d (elems=%d, host mask elems=%zu)",
                  t_idx0.nSize, (int)(t_idx0.nSize / sizeof(unsigned int)),
                  t_mask0.nSize, (int)(t_mask0.nSize / sizeof(unsigned short)), mask.size());
            // Per-layer K_cache / V_cache shape + is_linear_layer status
            int linear_count = 0;
            for (int m = 0; m < _attr.axmodel_num; ++m) {
                auto &tk_dec = llama_layers[m].layer.get_input(decode_grpid, "K_cache");
                bool islin = is_linear_layer(m);
                if (islin) ++linear_count;
                if (m < 4 || m == _attr.axmodel_num - 1 || islin) {
                    ALOGI("[diag] talker layer %02d: decode K_cache nSize=%d, is_linear=%d", m, tk_dec.nSize, (int)islin);
                }
            }
            ALOGI("[diag] talker layers: total=%d, is_linear_count=%d, full_attention_interval=%d",
                  _attr.axmodel_num, linear_count, _attr.full_attention_interval);
            // Per-prefill-group pre_k sizes (to spot d2d overrun)
            for (size_t gi = 0; gi < prefill_grpids_.size(); ++gi) {
                int gid = prefill_grpids_[(size_t)gi];
                auto &pre_k_g = llama_layers[0].layer.get_input(gid, "K_cache");
                auto &out_k_g = llama_layers[0].layer.get_output(gid, "K_cache_out");
                ALOGI("[diag] talker prefill gid=%d  pre_k.nSize=%d  out_k.nSize=%d  (kv_cap=%d)",
                      gid, pre_k_g.nSize, out_k_g.nSize, prefill_capacity_by_gid(gid));
            }
            if (cp_loaded) {
                auto &cp_idx0 = cp_layers[0].layer.get_input(cp_prefill_gid, "indices");
                ALOGI("[diag] cp prefill_gid indices nSize=%d (elems=%d)",
                      cp_idx0.nSize, (int)(cp_idx0.nSize / sizeof(unsigned int)));
                if (cp_decode_gid >= 0) {
                    auto &cp_idx1 = cp_layers[0].layer.get_input(cp_decode_gid, "indices");
                    ALOGI("[diag] cp decode_gid indices nSize=%d (elems=%d)",
                          cp_idx1.nSize, (int)(cp_idx1.nSize / sizeof(unsigned int)));
                }
            }
        }
        // Helper lambda: print fp32 stats of a bf16 vector (for codec_sum / next_embed / embed)
        auto diag_bf16_stats = [&](const char *tag, int frame_idx, const unsigned short *p, int n) {
            double sum = 0, sumsq = 0, sumabs = 0;
            float mn = 1e30f, mx = -1e30f;
            for (int i = 0; i < n; ++i) {
                float v = bfloat16(p[i]).fp32();
                sum += v; sumsq += (double)v * v; sumabs += std::abs(v);
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            double mean = sum / n;
            double l2 = std::sqrt(sumsq);
            ALOGI("[diag] %s frame=%d  mean=%.5f  L2=%.4f  L1=%.4f  min=%.4f  max=%.4f  first5=[%.4f %.4f %.4f %.4f %.4f]",
                  tag, frame_idx, mean, l2, sumabs, (double)mn, (double)mx,
                  (double)bfloat16(p[0]).fp32(), (double)bfloat16(p[1]).fp32(),
                  (double)bfloat16(p[2]).fp32(), (double)bfloat16(p[3]).fp32(),
                  (double)bfloat16(p[4]).fp32());
        };
        for (unsigned int indices = decode_start; indices < (unsigned int)_attr.max_token_len; indices++)
        {
            if (b_stop.load(std::memory_order_relaxed)) break;

            int primary_code = next_token;
            if (primary_code == codec_eos_token_id)
            {
                ALOGI("RunTts hit codec_eos at step %d", step);
                break;
            }

            // 1. CP frame
            std::vector<int> frame_codes;
            std::vector<unsigned short> codec_sum_bf16;

            // CP input: normed hidden from current talker step (NOT prefill history token)
            std::vector<unsigned short> txt_hidden_bf16(_attr.tokens_embed_size);
            if (step == 0) {
                // First frame: use last token of talker prefill (already normed)
                memcpy(txt_hidden_bf16.data(),
                       all_prefill_hidden.data() + (size_t)(input_embed_num - 1) * _attr.tokens_embed_size,
                       _attr.tokens_embed_size * sizeof(unsigned short));
            } else {
                // Subsequent frames: use normed talker decode output
                if (!cp_norm_gamma.empty()) {
                    rmsnorm_bf16(txt_hidden_bf16.data(), embed.data(), cp_norm_gamma.data(), _attr.tokens_embed_size);
                } else {
                    memcpy(txt_hidden_bf16.data(), embed.data(), _attr.tokens_embed_size * sizeof(unsigned short));
                }
            }
            bool cp_ok = false;
            if (cp_callback) {
                cp_ok = cp_callback->OnCpFrame(txt_hidden_bf16, primary_code, frame_codes, codec_sum_bf16);
            } else {
                cp_ok = RunCpFrame(txt_hidden_bf16, primary_code, frame_codes, codec_sum_bf16);
            }
            if (!cp_ok)
            {
                ALOGE("RunCpFrame failed at step %d", step);
                break;
            }
            if (step < 2) {
                diag_bf16_stats("txt_hidden_in", step, txt_hidden_bf16.data(), _attr.tokens_embed_size);
                diag_bf16_stats("codec_sum    ", step, codec_sum_bf16.data(), _attr.tokens_embed_size);
            }
            // print frame
            printf("frame=%d %d", step, primary_code);
            for (int j = 0; j < 15; ++j) printf(" %d", frame_codes[j + 1]);
            printf("\n"); fflush(stdout);

            // 2. construct next_embed = codec_sum + txt_hidden[step]
            std::vector<unsigned short> next_embed(_attr.tokens_embed_size, 0);            if (streaming)
            {
                int txt_pos = trailing_start + step;
                if (txt_pos < input_embed_num)
                {
                    for (int d = 0; d < _attr.tokens_embed_size; ++d)
                    {
                        float a = bfloat16(codec_sum_bf16[d]).fp32();
                        float b = bfloat16(all_prefill_hidden[(size_t)txt_pos * _attr.tokens_embed_size + d]).fp32();
                        next_embed[d] = bfloat16(a + b).data;
                    }
                }
                else
                {
                    // no more txt_hidden, use codec_sum only (or pad)
                    next_embed = codec_sum_bf16;
                }
            }
            else
            {
                // non-streaming: next_embed = codec_sum + tts_pad_vec
                if (tts_pad_vec_bf16.empty())
                {
                    ALOGE("RunTts non-streaming mode requires tts_pad_vec. Call SetTtsPadVec() before RunTts.");
                    break;
                }
                for (int d = 0; d < _attr.tokens_embed_size; ++d)
                {
                    float a = bfloat16(codec_sum_bf16[d]).fp32();
                    float b = bfloat16(tts_pad_vec_bf16[d]).fp32();
                    next_embed[d] = bfloat16(a + b).data;
                }
            }
            if (step < 2) {
                diag_bf16_stats("tts_pad_vec  ", step, tts_pad_vec_bf16.data(), _attr.tokens_embed_size);
                diag_bf16_stats("next_embed   ", step, next_embed.data(), _attr.tokens_embed_size);
            }

            // 3. Talker decode step
            {
                const int want_gid = choose_decode_gid((int)indices + 1);
                if (want_gid != decode_grpid) decode_grpid = want_gid;
            }
#ifdef USE_AXCL
            {
                auto &l0_in = llama_layers[0].layer.get_input(decode_grpid, "input");
                llm_h2d(LLM_WADDR(l0_in), next_embed.data(), l0_in.nSize, llama_layers[0].layer.get_devid());
            }
            for (int m = 0; m < _attr.axmodel_num; m++)
            {
                if (b_stop.load(std::memory_order_relaxed)) break;
                auto &lyr = llama_layers[m]; int devid = lyr.layer.get_devid();
                auto &t_idx = lyr.layer.get_input(decode_grpid, "indices");
                {
                    const int decode_idx_elems = (int)(t_idx.nSize / sizeof(unsigned int));
                    if (decode_idx_elems > 1) {
                        std::vector<unsigned int> decode_idx_buf((size_t)decode_idx_elems, indices);
                        llm_h2d(LLM_WADDR(t_idx), decode_idx_buf.data(), t_idx.nSize, devid);
                    } else {
                        llm_h2d(LLM_WADDR(t_idx), &indices, sizeof(indices), devid);
                    }
                }
                auto &t_mask = lyr.layer.get_input(decode_grpid, "mask");
                if (is_linear_layer(m))
                {
                    const size_t elems = (size_t)t_mask.nSize / sizeof(unsigned short);
                    std::vector<unsigned short> linear_decode_mask(elems, bf16_one.data);
                    if (linear_decode_mask.empty()) linear_decode_mask.push_back(bf16_one.data);
                    llm_h2d(LLM_WADDR(t_mask), linear_decode_mask.data(), linear_decode_mask.size() * sizeof(unsigned short), devid);
                }
                else
                {
                    llm_h2d(LLM_WADDR(t_mask), mask.data(), std::min((size_t)t_mask.nSize, mask.size() * sizeof(unsigned short)), devid);
                }
                lyr.layer.inference(decode_grpid);
                auto &out_k = lyr.layer.get_output(decode_grpid, "K_cache_out");
                auto &out_v = lyr.layer.get_output(decode_grpid, "V_cache_out");
                auto &in_k  = lyr.layer.get_input(decode_grpid, "K_cache");
                auto &in_v  = lyr.layer.get_input(decode_grpid, "V_cache");
                if (is_linear_layer(m))
                {
                    llm_d2d(LLM_WADDR(in_k), LLM_RADDR(out_k), std::min((size_t)in_k.nSize, (size_t)out_k.nSize), devid);
                    llm_d2d(LLM_WADDR(in_v), LLM_RADDR(out_v), std::min((size_t)in_v.nSize, (size_t)out_v.nSize), devid);
                }
                else
                {
                    llm_d2d((unsigned short *)LLM_WADDR(in_k) + indices * _attr.kv_cache_size, LLM_RADDR(out_k), out_k.nSize, devid);
                    llm_d2d((unsigned short *)LLM_WADDR(in_v) + indices * _attr.kv_cache_size, LLM_RADDR(out_v), out_v.nSize, devid);
                }
                auto &cur_out = lyr.layer.get_output(decode_grpid, "output");
                if (m == _attr.axmodel_num - 1)
                {
                    auto &post_in = llama_post.get_input("input");
                    if (llama_post.get_devid() == devid) { llm_d2d(LLM_WADDR(post_in), LLM_RADDR(cur_out), post_in.nSize, devid); }
                    else { llm_d2h(cur_out.pVirAddr, LLM_RADDR(cur_out), cur_out.nSize, devid); llm_h2d(LLM_WADDR(post_in), cur_out.pVirAddr, post_in.nSize, llama_post.get_devid()); }
                    // update embed for next CP frame
                    llm_d2h(embed.data(), LLM_RADDR(cur_out), embed.size() * sizeof(unsigned short), devid);
                }
                else
                {
                    auto &next_in = llama_layers[m + 1].layer.get_input(decode_grpid, "input"); int next_devid = llama_layers[m + 1].layer.get_devid();
                    if (next_devid == devid) { llm_d2d(LLM_WADDR(next_in), LLM_RADDR(cur_out), next_in.nSize, devid); }
                    else { llm_d2h(cur_out.pVirAddr, LLM_RADDR(cur_out), cur_out.nSize, devid); llm_h2d(LLM_WADDR(next_in), cur_out.pVirAddr, next_in.nSize, next_devid); }
                }
            }
            llama_post.inference();
            {
                auto &t_out = llama_post.get_output("output"); llm_d2h(t_out.pVirAddr, LLM_RADDR(t_out), t_out.nSize, llama_post.get_devid());
                unsigned short *post_out = (unsigned short *)t_out.pVirAddr; next_token = post_process(postprocess, post_out, _attr.tokens_embed_num, token_ids, nullptr);
            }
#else // AX650
            for (int m = 0; m < _attr.axmodel_num; m++)
            {
                if (b_stop.load(std::memory_order_relaxed)) break;
                auto &lyr = llama_layers[m];
                auto &in_k = lyr.layer.get_input(decode_grpid, "K_cache"); auto *in_k_ptr = (unsigned short *)in_k.pVirAddr;
                auto &in_v = lyr.layer.get_input(decode_grpid, "V_cache"); auto *in_v_ptr = (unsigned short *)in_v.pVirAddr;
                auto &t_idx = lyr.layer.get_input(decode_grpid, "indices");
                {
                    const int decode_idx_elems = (int)(t_idx.nSize / sizeof(unsigned int));
                    unsigned int *idx_ptr = (unsigned int *)t_idx.pVirAddr;
                    for (int r = 0; r < decode_idx_elems; ++r) idx_ptr[r] = indices;
                }
                auto &t_mask = lyr.layer.get_input(decode_grpid, "mask");
                if (is_linear_layer(m))
                {
                    const size_t elems = (size_t)t_mask.nSize / sizeof(unsigned short);
                    std::vector<unsigned short> linear_decode_mask(elems, bf16_one.data);
                    if (linear_decode_mask.empty()) linear_decode_mask.push_back(bf16_one.data);
                    memcpy(t_mask.pVirAddr, linear_decode_mask.data(), std::min((size_t)t_mask.nSize, linear_decode_mask.size() * sizeof(unsigned short)));
                }
                else
                {
                    memcpy(t_mask.pVirAddr, mask.data(), std::min((size_t)t_mask.nSize, mask.size() * sizeof(unsigned short)));
                }
                auto &t_in = lyr.layer.get_input(decode_grpid, "input"); memcpy(t_in.pVirAddr, next_embed.data(), next_embed.size() * sizeof(unsigned short));
                lyr.layer.inference(decode_grpid);
                auto &out_k = lyr.layer.get_output(decode_grpid, "K_cache_out");
                auto &out_v = lyr.layer.get_output(decode_grpid, "V_cache_out");
                if (is_linear_layer(m))
                {
                    memcpy(in_k.pVirAddr, out_k.pVirAddr, std::min((size_t)in_k.nSize, (size_t)out_k.nSize));
                    memcpy(in_v.pVirAddr, out_v.pVirAddr, std::min((size_t)in_v.nSize, (size_t)out_v.nSize));
                }
                else
                {
                    memcpy(in_k_ptr + indices * _attr.kv_cache_size, out_k.pVirAddr, sizeof(unsigned short) * _attr.kv_cache_size);
                    memcpy(in_v_ptr + indices * _attr.kv_cache_size, out_v.pVirAddr, sizeof(unsigned short) * _attr.kv_cache_size);
                }
                auto &t_out = lyr.layer.get_output(decode_grpid, "output"); memcpy(embed.data(), t_out.pVirAddr, embed.size() * sizeof(unsigned short));
            }
            auto &t_in = llama_post.get_input("input"); memcpy(t_in.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));
            llama_post.inference(); auto &t_out = llama_post.get_output("output");
            unsigned short *post_out = (unsigned short *)t_out.pVirAddr; next_token = post_process(postprocess, post_out, _attr.tokens_embed_num, token_ids, nullptr);
#endif

            if (indices < mask.size()) mask[indices] = 0;
            token_ids.push_back(next_token);

            if (step < 2) {
                diag_bf16_stats("decode_raw_h ", step, embed.data(), _attr.tokens_embed_size);
                ALOGI("[diag] step=%d  next_primary=%d", step, next_token);
            }

            // save frame
            TtsFrame frame;
            for (int i = 0; i < 16; ++i) frame.codes[i] = frame_codes[i];
            result.frames.push_back(frame);

            step++;
            if (max_new_tokens > 0 && step >= max_new_tokens)
            {
                ALOGI("RunTts reached max_new_tokens %d", max_new_tokens);
                break;
            }
        }

        float t_ms = t_cost.cost();
        ALOGN("RunTts finished frames=%zu avg %.2f token/s", result.frames.size(), (t_ms > 0) ? (result.frames.size() / (t_ms / 1000.0f)) : 0.0f);
        return true;
    }
