
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
                        try {
                            auto &t = cp_layers[0].layer.get_input(gid, k);
                            ALOGI("    in[%d]: name=%s shape=%s nSize=%d", k, t.sName.c_str(), shape_to_str2(t.vShape).c_str(), t.nSize);
                        } catch (...) {}
                    }
                    ALOGI("  group %d outputs:", gid);
                    for (int k = 0; k < cp_layers[0].layer.get_num_outputs(); ++k) {
                        try {
                            auto &t = cp_layers[0].layer.get_output(gid, k);
                            ALOGI("    out[%d]: name=%s shape=%s nSize=%d", k, t.sName.c_str(), shape_to_str2(t.vShape).c_str(), t.nSize);
                        } catch (...) {}
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

        // detect prefill gid from first cp layer
        {
            auto &ref = cp_layers[0].layer;
            int best_gid = 0;
            size_t best_elems = 0;
            int ng = ref.get_num_input_groups();
            for (int gid = 0; gid < ng; ++gid)
            {
                try {
                    auto &t_idx = ref.get_input(gid, "indices");
                    size_t elems = t_idx.nSize / sizeof(unsigned int);
                    if (elems > best_elems) { best_elems = elems; best_gid = gid; }
                } catch (...) {}
            }
            cp_prefill_gid = best_gid;
            try {
                auto &t_idx = ref.get_input(cp_prefill_gid, "indices");
                if (!t_idx.vShape.empty()) cp_prefill_token_num = (int)t_idx.vShape.back();
                else cp_prefill_token_num = (int)(t_idx.nSize / sizeof(unsigned int));
            } catch (...) { cp_prefill_token_num = 64; }
            if (cp_prefill_token_num <= 1) cp_prefill_token_num = 64;
            try {
                auto &t_mask = ref.get_input(cp_prefill_gid, "mask");
                int mask_elems = (int)(t_mask.nSize / sizeof(unsigned short));
                if (mask_elems > 0 && (mask_elems % cp_prefill_token_num) == 0)
                    cp_kv_cache_num = mask_elems / cp_prefill_token_num - cp_prefill_token_num;
                else
                    cp_kv_cache_num = cp_prefill_token_num;
            } catch (...) { cp_kv_cache_num = cp_prefill_token_num; }
            ALOGI("CP prefill_gid=%d token_num=%d kv_cache_num=%d", cp_prefill_gid, cp_prefill_token_num, cp_kv_cache_num);
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

    bool RunCpFrame(const std::vector<unsigned short> &last_hidden_bf16,
                    int primary_code,
                    std::vector<int> &out_frame_codes,
                    std::vector<unsigned short> &out_codec_sum_bf16)
    {
        if (!cp_loaded) return false;
        const int D = _attr.tokens_embed_size; // 1024

        // Reset CP KV cache at the start of each frame (new primary token from talker)
        cp_k_cache.clear();
        cp_v_cache.clear();
        cp_k_cache.resize(5);
        cp_v_cache.resize(5);

        // 1. primary embed via talker codec embedding table (first 3072 entries)
        std::vector<unsigned short> primary_embed(D);
        embed_selector.getByIndex((unsigned int)primary_code, primary_embed.data());

        out_frame_codes.resize(16);
        out_frame_codes[0] = primary_code;
        out_codec_sum_bf16 = primary_embed;

        // cp_ctx starts with [last_hidden, primary_embed]
        std::vector<unsigned short> cp_ctx;
        cp_ctx.reserve(17 * D);
        cp_ctx.insert(cp_ctx.end(), last_hidden_bf16.begin(), last_hidden_bf16.end());
        cp_ctx.insert(cp_ctx.end(), primary_embed.begin(), primary_embed.end());

        std::vector<unsigned short> mask_tmp((size_t)cp_prefill_token_num * (cp_kv_cache_num + cp_prefill_token_num));
        std::vector<unsigned int> idx_tmp((size_t)cp_prefill_token_num, 0);
        std::vector<unsigned short> embed_tmp((size_t)cp_prefill_token_num * D, 0);
        std::vector<unsigned short> layer_out((size_t)cp_prefill_token_num * D, 0);

        for (int j = 0; j < 15; ++j)
        {
            int seq_len = (j == 0) ? 2 : 1;
            int history_len = (j == 0) ? 0 : (j + 1);
            if (history_len + seq_len > cp_prefill_token_num)
            {
                ALOGE("CP history_len+seq_len(%d) > cp_prefill_token_num(%d)", history_len + seq_len, cp_prefill_token_num);
                return false;
            }

            // zero buffers
            std::fill(mask_tmp.begin(), mask_tmp.end(), bfloat16(-65536.f).data);
            std::fill(embed_tmp.begin(), embed_tmp.end(), (unsigned short)0);
            std::fill(idx_tmp.begin(), idx_tmp.end(), 0U);

            // build mask with proper history_len for KV-cache attention
            build_prefill_mask(mask_tmp, cp_kv_cache_num, cp_prefill_token_num, history_len, seq_len);

            // indices
            for (int i = 0; i < seq_len; ++i) idx_tmp[i] = (unsigned int)(history_len + i);

            // embed
            if (j == 0) {
                memcpy(embed_tmp.data(), cp_ctx.data(), seq_len * D * sizeof(unsigned short));
            } else {
                // only the latest residual embedding is the new input
                memcpy(embed_tmp.data(), cp_ctx.data() + (cp_ctx.size() - D), D * sizeof(unsigned short));
            }

            // run CP 5 layers
            for (int m = 0; m < 5; ++m)
            {
                auto &lyr = cp_layers[m];
                int devid = LLM_DEVID(lyr);
                int gid = cp_prefill_gid;
                auto &t_in = lyr.layer.get_input(gid, "input");
                size_t in_copy = std::min((size_t)t_in.nSize, embed_tmp.size() * sizeof(unsigned short));
                llm_h2d(LLM_WADDR(t_in), embed_tmp.data(), in_copy, devid);
                auto &t_mask = lyr.layer.get_input(gid, "mask");
                size_t mask_copy = std::min((size_t)t_mask.nSize, mask_tmp.size() * sizeof(unsigned short));
                llm_h2d(LLM_WADDR(t_mask), mask_tmp.data(), mask_copy, devid);
                auto &t_idx = lyr.layer.get_input(gid, "indices");
                size_t idx_copy = std::min((size_t)t_idx.nSize, idx_tmp.size() * sizeof(unsigned int));
                llm_h2d(LLM_WADDR(t_idx), idx_tmp.data(), idx_copy, devid);

                // K/V cache: zero on first step (frame start), reuse previous output otherwise
                try {
                    auto &tk = lyr.layer.get_input(gid, "K_cache");
                    if (tk.pVirAddr) {
                        if (j == 0) {
                            llm_memset(LLM_WADDR(tk), 0, tk.nSize, devid);
                        } else if (!cp_k_cache[m].empty()) {
                            llm_h2d(LLM_WADDR(tk), cp_k_cache[m].data(), std::min((size_t)tk.nSize, cp_k_cache[m].size() * sizeof(unsigned short)), devid);
                        } else {
                            llm_memset(LLM_WADDR(tk), 0, tk.nSize, devid);
                        }
                    }
                } catch (...) {}
                try {
                    auto &tv = lyr.layer.get_input(gid, "V_cache");
                    if (tv.pVirAddr) {
                        if (j == 0) {
                            llm_memset(LLM_WADDR(tv), 0, tv.nSize, devid);
                        } else if (!cp_v_cache[m].empty()) {
                            llm_h2d(LLM_WADDR(tv), cp_v_cache[m].data(), std::min((size_t)tv.nSize, cp_v_cache[m].size() * sizeof(unsigned short)), devid);
                        } else {
                            llm_memset(LLM_WADDR(tv), 0, tv.nSize, devid);
                        }
                    }
                } catch (...) {}

                int inf_ret = lyr.layer.inference(gid);
                if (inf_ret != 0) ALOGE("RunCpFrame layer m=%d inference failed ret=%d", m, inf_ret);

                auto &t_out = lyr.layer.get_output(gid, "output");
                llm_d2h(layer_out.data(), LLM_RADDR(t_out), std::min((size_t)t_out.nSize, layer_out.size() * sizeof(unsigned short)), devid);
                memcpy(embed_tmp.data(), layer_out.data(), seq_len * D * sizeof(unsigned short));

                // Save K/V cache output for next step within this frame
                try {
                    auto &tk_out = lyr.layer.get_output(gid, "K_cache_out");
                    size_t kv_words = tk_out.nSize / sizeof(unsigned short);
                    if (cp_k_cache[m].size() < kv_words) cp_k_cache[m].resize(kv_words);
                    llm_d2h(cp_k_cache[m].data(), LLM_RADDR(tk_out), tk_out.nSize, devid);
                } catch (...) {}
                try {
                    auto &tv_out = lyr.layer.get_output(gid, "V_cache_out");
                    size_t kv_words = tv_out.nSize / sizeof(unsigned short);
                    if (cp_v_cache[m].size() < kv_words) cp_v_cache[m].resize(kv_words);
                    llm_d2h(cp_v_cache[m].data(), LLM_RADDR(tv_out), tv_out.nSize, devid);
                } catch (...) {}
            }

            // take last position hidden state
            std::vector<unsigned short> hidden_step(D);
            memcpy(hidden_step.data(), embed_tmp.data() + (seq_len - 1) * D, D * sizeof(unsigned short));

            // run CP post (if it has input named "input")
            bool has_cp_post = true;
            try {
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
            } catch (const std::exception &e) {
                has_cp_post = false;
                ALOGI("RunCpFrame post skipped: %s", e.what());
            }

            // run lm_head_j -> logits [2048]
            auto &lmh = cp_lm_heads[j];
            try {
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
                int res_code = postprocess.apply(logits_fp32, {});
                out_frame_codes[j + 1] = res_code;

                // lookup residual embed
                std::vector<unsigned short> res_embed(D);
                if ((size_t)res_code < 2048)
                {
                    memcpy(res_embed.data(), cp_embed_tables[j].data() + (size_t)res_code * D, D * sizeof(unsigned short));
                }
                // accumulate codec_sum (element-wise fp32 add, then convert back)
                for (int d = 0; d < D; ++d)
                {
                    float a = bfloat16(out_codec_sum_bf16[d]).fp32();
                    float b = bfloat16(res_embed[d]).fp32();
                    out_codec_sum_bf16[d] = bfloat16(a + b).data;
                }
                // append to cp_ctx for next step
                cp_ctx.insert(cp_ctx.end(), res_embed.begin(), res_embed.end());
            } catch (const std::exception &e) {
                ALOGE("CP lm_head %d failed: %s", j, e.what());
                return false;
            }
        }
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
                auto &t_idx = lyr.layer.get_input(prefill_grpid, "indices");
                unsigned int *idx_ptr = (unsigned int *)t_idx.pVirAddr; memset(idx_ptr, 0, t_idx.nSize);
                int idx_i = 0; for (int i = 0; i < input_num_token; ++i) idx_ptr[idx_i++] = (unsigned int)(history_len + i);
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
                llm_d2d((unsigned short *)LLM_WADDR(pre_k) + kv_off, LLM_RADDR(out_k), kv_sz, devid);
                llm_d2d((unsigned short *)LLM_WADDR(pre_v) + kv_off, LLM_RADDR(out_v), kv_sz, devid);
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
            std::vector<unsigned short> last_hidden_bf16 = embed; // [D]
            bool cp_ok = false;
            if (cp_callback) {
                cp_ok = cp_callback->OnCpFrame(last_hidden_bf16, primary_code, frame_codes, codec_sum_bf16);
            } else {
                cp_ok = RunCpFrame(last_hidden_bf16, primary_code, frame_codes, codec_sum_bf16);
            }
            if (!cp_ok)
            {
                ALOGE("RunCpFrame failed at step %d", step);
                break;
            }

            // print frame
            printf("frame=%d %d", step, primary_code);
            for (int j = 0; j < 15; ++j) printf(" %d", frame_codes[j + 1]);
            printf("\n"); fflush(stdout);

            // 2. construct next_embed = codec_sum + txt_hidden[step]
            std::vector<unsigned short> next_embed(_attr.tokens_embed_size, 0);
            if (streaming)
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
                llm_h2d(LLM_WADDR(t_idx), &indices, sizeof(indices), devid);
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
                auto &t_idx = lyr.layer.get_input(decode_grpid, "indices"); memcpy(t_idx.pVirAddr, &indices, sizeof(indices));
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
