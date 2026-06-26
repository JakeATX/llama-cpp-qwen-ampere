#include "llama-graph.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-batch.h"
#include "llama-context.h"
#include "llama-cparams.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-kvarn.h"
#include "llama-kv-cache-kvarn-iswa.h"
#include "llama-kv-cache-iswa.h"
#include "llama-kv-cache-dsa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-kvarn.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

// dedup helpers

static bool kvarn_graph_parse_env_flag(const char * name) {
    const char * env = std::getenv(name);
    if (env == nullptr) {
        return false;
    }

    char * end = nullptr;
    errno = 0;
    const long value = std::strtol(env, &end, 10);
    if (env[0] == '\0' || end == nullptr || *end != '\0' || errno == ERANGE ||
            (value != 0 && value != 1)) {
        throw std::runtime_error(std::string("invalid KVarN environment flag ") + name +
                "=" + env + "; expected integer 0 or 1");
    }
    return value != 0;
}

static bool kvarn_graph_paper_mixed_frame_enabled() {
    if (!kvarn_graph_parse_env_flag("LLAMA_KVARN_PAPER_MIXED_FRAME")) {
        return false;
    }
    if (!kvarn_graph_parse_env_flag("LLAMA_KVARN_UNSAFE_ALLOW_PAPER_MIXED_FRAME")) {
        throw std::runtime_error(
                "LLAMA_KVARN_PAPER_MIXED_FRAME is diagnostic-only because it can silently "
                "disagree with graph/output frame handling. Set "
                "LLAMA_KVARN_UNSAFE_ALLOW_PAPER_MIXED_FRAME=1 only for targeted frame A/B tests.");
    }
    return true;
}

static bool kvarn_graph_prefill_direct_store_disabled() {
    return kvarn_graph_parse_env_flag("LLAMA_KVARN_DISABLE_PREFILL_DIRECT_STORE");
}

static bool kvarn_graph_prefill_direct_attn_disabled() {
    return kvarn_graph_parse_env_flag("LLAMA_KVARN_DISABLE_PREFILL_DIRECT_ATTN");
}

static bool kvarn_graph_iswa_sinktail_mha_disabled() {
    return kvarn_graph_parse_env_flag("LLAMA_KVARN_DISABLE_ISWA_SINKTAIL_MHA");
}

static bool kvarn_graph_iswa_prefill_direct_attn_allowed() {
    return kvarn_graph_parse_env_flag("LLAMA_KVARN_UNSAFE_ENABLE_ISWA_PREFILL_DIRECT_ATTN");
}

static bool kvarn_graph_iswa_force_full_normal_attn() {
    return kvarn_graph_parse_env_flag("LLAMA_KVARN_ISWA_DEBUG_FULL_NORMAL_ATTN");
}

static bool kvarn_graph_iswa_materialize_mha_enabled() {
    return kvarn_graph_parse_env_flag("LLAMA_KVARN_ISWA_DEBUG_MATERIALIZE_MHA");
}

static bool kvarn_graph_iswa_dual_mha_compare_enabled() {
    return kvarn_graph_parse_env_flag("LLAMA_KVARN_ISWA_DEBUG_DUAL_MHA_COMPARE");
}

static bool kvarn_graph_iswa_raw_mha_compare_enabled() {
    return kvarn_graph_parse_env_flag("LLAMA_KVARN_ISWA_DEBUG_RAW_MHA_COMPARE");
}

static bool kvarn_graph_iswa_turbo_v_frame_enabled() {
    return kvarn_graph_parse_env_flag("LLAMA_KVARN_ISWA_DEBUG_TURBO_V_FRAME");
}

static int kvarn_graph_parse_env_int(const char * name, int default_value) {
    const char * env = std::getenv(name);
    if (env == nullptr) {
        return default_value;
    }

    char * end = nullptr;
    errno = 0;
    const long value = std::strtol(env, &end, 10);
    if (env[0] == '\0' || end == nullptr || *end != '\0' || errno == ERANGE ||
            value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string("invalid KVarN environment integer ") + name +
                "=" + env);
    }
    return int(value);
}

static bool kvarn_graph_attn_trace_enabled() {
    return kvarn_graph_parse_env_flag("LLAMA_KVARN_ATTN_TRACE");
}

static bool kvarn_graph_attn_trace_claim() {
    static std::atomic<int> n_trace{0};

    const int limit = kvarn_graph_parse_env_int("LLAMA_KVARN_ATTN_TRACE_LIMIT", 64);
    if (limit <= 0) {
        return true;
    }

    return n_trace.fetch_add(1, std::memory_order_relaxed) < limit;
}

static void kvarn_graph_attn_trace_tensor(const char * name, const ggml_tensor * t) {
    if (t == nullptr) {
        std::fprintf(stderr, "  %-12s: null\n", name);
        return;
    }

    std::fprintf(stderr,
            "  %-12s: type=%s ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] nb=[%zu,%zu,%zu,%zu]\n",
            name, ggml_type_name(t->type),
            t->ne[0], t->ne[1], t->ne[2], t->ne[3],
            t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
}

static bool kvarn_graph_reuse_trace_enabled() {
    return kvarn_graph_parse_env_flag("LLAMA_KVARN_GRAPH_REUSE_TRACE");
}

static bool kvarn_graph_prefill_direct_trace_enabled() {
    return kvarn_graph_parse_env_flag("LLAMA_KVARN_PREFILL_DIRECT_TRACE");
}

static bool kvarn_graph_prefill_direct_trace_claim() {
    static std::atomic<int> n_trace{0};

    const int limit = kvarn_graph_parse_env_int("LLAMA_KVARN_PREFILL_DIRECT_TRACE_LIMIT", 64);
    if (limit <= 0) {
        return true;
    }

    return n_trace.fetch_add(1, std::memory_order_relaxed) < limit;
}

static void kvarn_graph_reuse_trace_miss(const char * where, const char * check) {
    // llama-bench installs a null log callback; mirror kvarn cache instrumentation.
    std::fprintf(stderr, "%s: KVarN graph reuse miss: %s\n", where, check);
}

static ggml_tensor * build_attn_inp_kq_mask(
        ggml_context * ctx,
        uint32_t n_kv,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams) {
    const auto n_tokens = ubatch.n_tokens;
    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    // flash attention requires an f16 mask
    const auto type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    ggml_tensor * res = ggml_new_tensor_4d(ctx, type, n_kv, n_tokens/n_stream, 1, n_stream);
    ggml_set_input(res);
    ggml_set_name(res, "attn_inp_kq_mask");

    return res;
}

static ggml_tensor * build_attn_inp_kq_mask(
        ggml_context * ctx,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams) {
    return build_attn_inp_kq_mask(ctx, mctx->get_n_kv(), ubatch, cparams);
}

static bool can_reuse_kq_mask(
        ggml_tensor * kq_mask,
        uint32_t n_kv,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams) {
    const auto n_tokens = ubatch.n_tokens;
    const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;

    bool res = true;

    res &= (kq_mask->ne[0] == n_kv);
    res &= (kq_mask->ne[1] == n_tokens/n_stream);
    res &= (kq_mask->ne[2] == 1);
    res &= (kq_mask->ne[3] == n_stream);

    return res;
}

static bool can_reuse_kq_mask(
        ggml_tensor * kq_mask,
        const llama_kv_cache_context * mctx,
        const llama_ubatch & ubatch,
        const llama_cparams & cparams) {
    return can_reuse_kq_mask(kq_mask, mctx->get_n_kv(), ubatch, cparams);
}

// impl

static ggml_tensor * ggml_mul_mat_aux(
        ggml_context * ctx,
        ggml_tensor * cur,
        ggml_tensor * rot,
        bool hadamard_hint = true,
        ggml_backend_sched_t sched = nullptr,
        ggml_backend_t backend = nullptr) {
    const auto n = rot->ne[0];

    ggml_tensor * res;

    if (!ggml_is_contiguous(cur)) {
        res = ggml_cont_2d   (ctx, cur, n, ggml_nelements(cur)/n);
    } else {
        res = ggml_reshape_2d(ctx, cur, n, ggml_nelements(cur)/n);
    }
    res = ggml_mul_mat   (ctx, rot, res);
    if (sched != nullptr && backend != nullptr) {
        ggml_backend_sched_set_tensor_backend(sched, res, backend);
    }
    if (hadamard_hint) {
        ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD);
    }
    res = ggml_reshape_4d(ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);

    return res;
}

void llm_graph_input_embd::set_input(const llama_ubatch * ubatch) {
    if (ubatch->token) {
        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
    }

    if (ubatch->embd) {
        GGML_ASSERT(n_embd == embd->ne[0]);

        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(embd));
    }
}

bool llm_graph_input_embd::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!params.ubatch.token) || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (embd   &&   embd->ne[1] == params.ubatch.n_tokens);

    return res;
}

void llm_graph_input_embd_h::set_input(const llama_ubatch * ubatch) {
    const int64_t n_tokens = ubatch->n_tokens;

    if (ubatch->token) {
        ggml_backend_tensor_set(tokens, ubatch->token, 0, n_tokens*ggml_element_size(tokens));
    } else {
        // note: mtmd embedding input goes through here
        GGML_ASSERT(ubatch->embd);
        GGML_ASSERT(n_embd == embd->ne[0]);

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(h));
    }

    // TODO: extend llama_ubatch to differentiate between token embeddings and hidden states
    //       for now, we assume that the hidden state is always provided as an embedding
    //       ref: https://github.com/ggml-org/llama.cpp/pull/23643
    if (ubatch->embd) {
        GGML_ASSERT(n_embd == h->ne[0]);

        ggml_backend_tensor_set(h, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(h));
    }
}

bool llm_graph_input_embd_h::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!params.ubatch.token) || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (embd   && embd->ne[1]   == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (h      && h->ne[1]      == params.ubatch.n_tokens);

    return res;
}

void llm_graph_input_pos::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && pos) {
        const int64_t n_tokens = ubatch->n_tokens;

        if (ubatch->token && n_pos_per_embd == 4) {
            // in case we're using M-RoPE with text tokens, convert the 1D positions to 4D
            // the 3 first dims are the same, and 4th dim is all 0
            std::vector<llama_pos> pos_data(n_tokens*n_pos_per_embd);
            // copy the first dimension
            for (int i = 0; i < n_tokens; ++i) {
                pos_data[               i] = ubatch->pos[i];
                pos_data[    n_tokens + i] = ubatch->pos[i];
                pos_data[2 * n_tokens + i] = ubatch->pos[i];
                pos_data[3 * n_tokens + i] = 0; // 4th dim is 0
            }
            ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size()*ggml_element_size(pos));
        } else {
            ggml_backend_tensor_set(pos, ubatch->pos, 0, n_tokens*n_pos_per_embd*ggml_element_size(pos));
        }
    }
}

bool llm_graph_input_pos::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= pos->ne[0] == params.ubatch.n_tokens*n_pos_per_embd;

    return res;
}

void llm_graph_input_attn_temp::set_input(const llama_ubatch * ubatch) {
    if (ubatch->pos && attn_scale) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(f_attn_temp_scale != 0.0f);
        GGML_ASSERT(n_attn_temp_floor_scale != 0);

        std::vector<float> attn_scale_data(n_tokens, 0.0f);
        for (int i = 0; i < n_tokens; ++i) {
            const float pos = ubatch->pos[i];
            attn_scale_data[i] = std::log(
                std::floor((pos + f_attn_temp_offset) / n_attn_temp_floor_scale) + 1.0
            ) * f_attn_temp_scale + 1.0;
        }

        ggml_backend_tensor_set(attn_scale, attn_scale_data.data(), 0, n_tokens*ggml_element_size(attn_scale));
    }
}

void llm_graph_input_pos_bucket::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        const int64_t n_tokens = ubatch->n_tokens;

        GGML_ASSERT(ggml_backend_buffer_is_host(pos_bucket->buffer));
        GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

        int32_t * data = (int32_t *) pos_bucket->data;

        for (int j = 0; j < n_tokens; ++j) {
            for (int i = 0; i < n_tokens; ++i) {
                data[j*n_tokens + i] = llama_relative_position_bucket(ubatch->pos[i], ubatch->pos[j], hparams.n_rel_attn_bkts, true);
            }
        }
    }
}

void llm_graph_input_pos_bucket_kv::set_input(const llama_ubatch * ubatch) {
    if (pos_bucket) {
        mctx->set_input_pos_bucket(pos_bucket, ubatch);
    }
}

void llm_graph_input_out_ids::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(out_ids);

    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(out_ids->buffer));
    int32_t * data = (int32_t *) out_ids->data;

    if (n_outputs == n_tokens) {
        for (int i = 0; i < n_tokens; ++i) {
            data[i] = i;
        }

        return;
    }

    GGML_ASSERT(ubatch->output);

    int n_outputs = 0;

    for (int i = 0; i < n_tokens; ++i) {
        if (ubatch->output[i]) {
            data[n_outputs++] = i;
        }
    }
}

bool llm_graph_input_out_ids::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= n_outputs == params.n_outputs;

    return res;
}

void llm_graph_input_mean::set_input(const llama_ubatch * ubatch) {
    if (cparams.embeddings   &&
       (cparams.pooling_type == LLAMA_POOLING_TYPE_MEAN ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK )) {

        const int64_t n_tokens     = ubatch->n_tokens;
        const int64_t n_seq_tokens = ubatch->n_seq_tokens;
        const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

        GGML_ASSERT(mean);
        GGML_ASSERT(ggml_backend_buffer_is_host(mean->buffer));

        float * data = (float *) mean->data;
        memset(mean->data, 0, n_tokens*n_seqs_unq*ggml_element_size(mean));

        std::vector<uint64_t> sums(n_seqs_unq, 0);
        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                sums[seq_idx] += ubatch->n_seq_tokens;
            }
        }

        std::vector<float> div(n_seqs_unq, 0.0f);
        for (int s = 0; s < n_seqs_unq; ++s) {
            const uint64_t sum = sums[s];
            if (sum > 0) {
                div[s] = 1.0f/float(sum);
            }
        }

        for (int i = 0; i < n_tokens; i += n_seq_tokens) {
            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                for (int j = 0; j < n_seq_tokens; ++j) {
                    data[seq_idx*n_tokens + i + j] = div[seq_idx];
                }
            }
        }
    }
}

void llm_graph_input_cls::set_input(const llama_ubatch * ubatch) {
    const int64_t n_tokens     = ubatch->n_tokens;
    const int64_t n_seqs_unq   = ubatch->n_seqs_unq;

    if (cparams.embeddings && (
        cparams.pooling_type == LLAMA_POOLING_TYPE_CLS  ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_RANK ||
        cparams.pooling_type == LLAMA_POOLING_TYPE_LAST
    )) {
        GGML_ASSERT(cls);
        GGML_ASSERT(ggml_backend_buffer_is_host(cls->buffer));

        uint32_t * data = (uint32_t *) cls->data;
        memset(cls->data, 0, n_seqs_unq*ggml_element_size(cls));

        std::vector<int> target_pos(n_seqs_unq, -1);
        std::vector<int> target_row(n_seqs_unq, -1);

        const bool last = (
             cparams.pooling_type == LLAMA_POOLING_TYPE_LAST ||
            (cparams.pooling_type == LLAMA_POOLING_TYPE_RANK && (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_QWEN3VL)) // qwen3 reranking & embedding models use last token
        );

        for (int i = 0; i < n_tokens; ++i) {
            const llama_pos pos = ubatch->pos[i];

            for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id  = ubatch->seq_id[i][s];
                const int32_t      seq_idx = ubatch->seq_idx[seq_id];

                if (
                    (target_pos[seq_idx] == -1) ||
                    ( last && pos >= target_pos[seq_idx]) ||
                    (!last && pos <  target_pos[seq_idx])
                ) {
                    target_pos[seq_idx] = pos;
                    target_row[seq_idx] = i;
                }
            }
        }

        for (int s = 0; s < n_seqs_unq; ++s) {
            if (target_row[s] >= 0) {
                data[s] = target_row[s];
            }
        }
    }
}

void llm_graph_input_rs::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    const int64_t n_rs = mctx->get_n_rs();

    if (s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(s_copy->buffer));
        int32_t * data = (int32_t *) s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->s_copy(i);
        }
    }
}

bool llm_graph_input_rs::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_recurrent_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= s_copy->ne[0] == mctx->get_n_rs();

    res &= s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= s_copy_extra->ne[0] == mctx->get_n_rs() - params.ubatch.n_seqs;

    res &= head == mctx->get_head();
    res &= rs_z == mctx->get_rs_z();

    return res;
}

void llm_graph_input_cross_embd::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (cross_embd && !cross->v_embd.empty()) {
        assert(cross_embd->type == GGML_TYPE_F32);

        ggml_backend_tensor_set(cross_embd, cross->v_embd.data(), 0, ggml_nbytes(cross_embd));
    }
}

template <typename T>
static void print_mask(const T * data, int64_t n_tokens, int64_t n_kv, int64_t n_swa, llama_swa_type swa_type) {
    LLAMA_LOG_DEBUG("%s: === Attention mask ===\n", __func__);
    const char * swa_type_str = "unknown";

    switch (swa_type) {
        case LLAMA_SWA_TYPE_NONE:      swa_type_str = "LLAMA_SWA_TYPE_NONE"; break;
        case LLAMA_SWA_TYPE_STANDARD:  swa_type_str = "LLAMA_SWA_TYPE_STANDARD"; break;
        case LLAMA_SWA_TYPE_CHUNKED:   swa_type_str = "LLAMA_SWA_TYPE_CHUNKED"; break;
        case LLAMA_SWA_TYPE_SYMMETRIC: swa_type_str = "LLAMA_SWA_TYPE_SYMMETRIC"; break;
    };

    LLAMA_LOG_DEBUG("%s: n_swa : %d, n_kv: %d, swa_type: %s\n", __func__, (int)n_swa, (int)n_kv, swa_type_str);
    LLAMA_LOG_DEBUG("%s: '0' = can attend, '∞' = masked\n", __func__);
    LLAMA_LOG_DEBUG("%s: Rows = query tokens, Columns = key/value tokens\n\n", __func__);

    LLAMA_LOG_DEBUG("    ");
    for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
        LLAMA_LOG_DEBUG("%2d", j);
    }
    LLAMA_LOG_DEBUG("\n");

    for (int i = 0; i < std::min((int64_t)20, n_tokens); ++i) {
        LLAMA_LOG_DEBUG(" %2d ", i);
        for (int j = 0; j < std::min((int64_t)20, n_kv); ++j) {
            float val = llama_cast<float>(data[i * n_kv + j]);
            if (val == -INFINITY) {
                LLAMA_LOG_DEBUG(" ∞");
            } else {
                LLAMA_LOG_DEBUG(" 0");
            }
        }
        LLAMA_LOG_DEBUG("\n");
    }
}

void llm_graph_input_attn_no_cache::set_input(const llama_ubatch * ubatch) {
    const int64_t n_kv     = ubatch->n_tokens;
    const int64_t n_tokens = ubatch->n_tokens;

    const auto fill_mask = [&](auto * data, int64_t ne, int n_swa, llama_swa_type swa_type) {
        using T = std::remove_reference_t<decltype(*data)>;
        std::fill(data, data + ne, llama_cast<T>(-INFINITY));

        for (int i1 = 0; i1 < n_tokens; ++i1) {
            const llama_seq_id s1 = ubatch->seq_id[i1][0];
            const llama_pos    p1 = ubatch->pos[i1];

            const uint64_t idst = i1*n_kv;

            for (int i0 = 0; i0 < n_tokens; ++i0) {
                const llama_seq_id s0 = ubatch->seq_id[i0][0];
                const llama_pos p0    = ubatch->pos[i0];

                // mask different sequences
                if (s0 != s1) {
                    continue;
                }

                // mask future tokens
                if (cparams.causal_attn && p0 > p1) {
                    continue;
                }

                // apply SWA if any
                if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                    continue;
                }

                data[idst + i0] = llama_cast<T>(hparams.use_alibi ? -std::abs(p0 - p1) : 0.0f);
            }
        }

        if (debug) {
            print_mask(data, n_tokens, n_kv, n_swa, swa_type);
        }
    };

    GGML_ASSERT(self_kq_mask);
    GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask->buffer));
    if (self_kq_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) self_kq_mask->data, ggml_nelements(self_kq_mask), 0, LLAMA_SWA_TYPE_NONE);
    } else {
        fill_mask((float       *) self_kq_mask->data, ggml_nelements(self_kq_mask), 0, LLAMA_SWA_TYPE_NONE);
    }

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        GGML_ASSERT(self_kq_mask_swa);
        GGML_ASSERT(ggml_backend_buffer_is_host(self_kq_mask_swa->buffer));
        if (self_kq_mask_swa->type == GGML_TYPE_F16) {
            fill_mask((ggml_fp16_t *) self_kq_mask_swa->data, ggml_nelements(self_kq_mask_swa), hparams.n_swa, hparams.swa_type);
        } else {
            fill_mask((float       *) self_kq_mask_swa->data, ggml_nelements(self_kq_mask_swa), hparams.n_swa, hparams.swa_type);
        }
    }
}

void llm_graph_input_attn_kv::set_input(const llama_ubatch * ubatch) {
    mctx->set_input_k_idxs(self_k_idxs, ubatch);
    mctx->set_input_v_idxs(self_v_idxs, ubatch);

    mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);

    if (self_k_rot) {
        mctx->set_input_k_rot(self_k_rot);
    }

    if (self_v_rot) {
        mctx->set_input_v_rot(self_v_rot);
    }
}

bool llm_graph_input_attn_kv::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= can_reuse_kq_mask(self_kq_mask, mctx, params.ubatch, params.cparams);

    return res;
}

struct kvarn_active_window {
    int32_t n_sink = 0;
    int32_t n_records = 0;
    int32_t n_pending = 0;
    int32_t n_tail = 0;
    int32_t tail_start = 0;
    int64_t n_kv = 0;
    bool valid = false;
};

static llama_kvarn_params kvarn_graph_effective_params(llama_kvarn_params params, uint32_t kv_size) {
    if (kv_size != 0 && uint64_t(params.sink_tokens) + uint64_t(params.tail_tokens) > kv_size) {
        if (params.sink_tokens >= kv_size) {
            params.tail_tokens = 0;
        } else {
            params.tail_tokens = kv_size - params.sink_tokens;
        }
    }
    return params;
}

static uint32_t kvarn_graph_count_tail_evictions(const llama_kvarn_params & params, const llama_ubatch & ubatch, uint32_t kv_size) {
    const llama_kvarn_params p = kvarn_graph_effective_params(params, kv_size);
    uint32_t n = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_pos pos = ubatch.pos ? ubatch.pos[i] : llama_pos(i);
        if (pos >= 0 && uint32_t(pos) >= p.sink_tokens + p.tail_tokens) {
            ++n;
        }
    }
    return n;
}

static std::vector<uint32_t> kvarn_graph_seal_records(const llama_kvarn_params & params, const llama_ubatch & ubatch, uint32_t kv_size) {
    const llama_kvarn_params p = kvarn_graph_effective_params(params, kv_size);
    std::vector<uint32_t> records;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_pos pos = ubatch.pos ? ubatch.pos[i] : llama_pos(i);
        if (pos < 0 || uint32_t(pos) < p.sink_tokens + p.tail_tokens) {
            continue;
        }

        const uint32_t evicted_pos = uint32_t(pos) - p.tail_tokens;
        const uint32_t body_pos = evicted_pos - p.sink_tokens;
        const uint32_t offset = body_pos%p.group_size;
        if (offset + 1 != p.group_size) {
            continue;
        }

        const uint32_t record = body_pos/p.group_size;
        if (std::find(records.begin(), records.end(), record) == records.end()) {
            records.push_back(record);
        }
    }
    return records;
}

struct kvarn_tail_evict_slice {
    int64_t start = 0;
    int64_t count = 0;
    bool contiguous = true;
};

static kvarn_tail_evict_slice kvarn_graph_tail_evict_slice_for_record(
        const llama_kvarn_params & params,
        const llama_ubatch & ubatch,
        uint32_t kv_size,
        uint32_t target_record) {
    const llama_kvarn_params p = kvarn_graph_effective_params(params, kv_size);
    kvarn_tail_evict_slice result;
    int64_t j = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_pos pos = ubatch.pos ? ubatch.pos[i] : llama_pos(i);
        if (pos < 0 || uint32_t(pos) < p.sink_tokens + p.tail_tokens) {
            continue;
        }

        const uint32_t evicted_pos = uint32_t(pos) - p.tail_tokens;
        const uint32_t body_pos = evicted_pos - p.sink_tokens;
        const uint32_t record = body_pos/p.group_size;
        if (record == target_record) {
            if (result.count > 0 && result.start + result.count != j) {
                result.contiguous = false;
            }
            if (result.count == 0) {
                result.start = j;
            }
            ++result.count;
        }
        ++j;
    }
    return result;
}

static kvarn_active_window kvarn_graph_active_window(const llama_kvarn_params & params, const llama_ubatch & ubatch, uint32_t kv_size) {
    const llama_kvarn_params p = kvarn_graph_effective_params(params, kv_size);
    kvarn_active_window result;
    if (ubatch.n_tokens == 0) {
        return result;
    }

    const uint32_t i_last = ubatch.n_tokens - 1;
    const llama_pos pos = ubatch.pos ? ubatch.pos[i_last] : llama_pos(i_last);
    if (pos < 0) {
        return result;
    }

    const uint32_t n_seen = uint32_t(pos) + 1;
    if (kv_size != 0 && n_seen > kv_size) {
        return result;
    }

    const uint32_t n_sink = std::min<uint32_t>(n_seen, p.sink_tokens);
    const uint32_t n_after_sink = n_seen - n_sink;
    const uint32_t n_tail = std::min<uint32_t>(n_after_sink, p.tail_tokens);
    const uint32_t n_body_pending = n_after_sink - n_tail;
    const uint32_t n_records = n_body_pending/p.group_size;
    const uint32_t n_pending = n_body_pending%p.group_size;
    const uint32_t tail_start = n_tail == 0 ? 0 : (n_body_pending%p.tail_tokens);

    result.n_sink = int32_t(n_sink);
    result.n_records = int32_t(n_records);
    result.n_pending = int32_t(n_pending);
    result.n_tail = int32_t(n_tail);
    result.tail_start = int32_t(tail_start);
    result.n_kv = int64_t(n_sink) + int64_t(n_records)*p.group_size + n_pending + n_tail;
    result.valid = true;
    return result;
}

static bool kvarn_graph_decode_stable_topology(const llama_ubatch & ubatch) {
    return ubatch.n_tokens == 1;
}

static kvarn_active_window kvarn_graph_worst_active_window(
        const llama_kvarn_params & params,
        uint32_t kv_size) {
    llama_pos pos = llama_pos(kv_size > 0 ? int(kv_size) - 1 : 0);
    llama_ubatch ubatch{};
    ubatch.n_tokens = 1;
    ubatch.pos = &pos;
    return kvarn_graph_active_window(params, ubatch, kv_size);
}

static uint32_t kvarn_graph_mask_n_kv(
        const kvarn_active_window & window,
        uint32_t kv_size,
        const llama_ubatch & ubatch) {
    if (kvarn_graph_decode_stable_topology(ubatch) && window.valid) {
        return kv_size;
    }
    return window.valid ? uint32_t(window.n_kv) : kv_size;
}

static kvarn_active_window kvarn_graph_build_scratch_window(
        const kvarn_active_window & window,
        const llama_kvarn_params & params,
        uint32_t kv_size,
        const llama_ubatch & ubatch) {
    if (kvarn_graph_decode_stable_topology(ubatch) && window.valid && kv_size > 0) {
        return kvarn_graph_worst_active_window(params, kv_size);
    }
    return window;
}

static uint32_t kvarn_graph_reuse_mask_n_kv(
        const kvarn_active_window & window,
        uint32_t kv_size,
        const llama_ubatch & ubatch) {
    if (kvarn_graph_decode_stable_topology(ubatch)) {
        return kv_size;
    }
    return uint32_t(window.n_kv);
}

static void kvarn_graph_update_mixed_attn_params(ggml_tensor * node, const kvarn_active_window & window) {
    GGML_ASSERT(node->op == GGML_OP_KVARN_ATTN_MIXED);

    struct {
        int32_t n_sink;
        int32_t n_records;
        int32_t n_pending;
        int32_t n_tail;
        int32_t tail_start;
        int32_t head_dim;
        int32_t group_size;
        int32_t key_bits;
        int32_t value_bits;
        float   scale;
        float   logit_softcap;
        int32_t frame_flags;
        int32_t v_layout;
    } params;

    std::memcpy(&params, node->op_params, sizeof(params));
    params.n_sink    = window.n_sink;
    params.n_records = window.n_records;
    params.n_pending = window.n_pending;
    params.n_tail    = window.n_tail;
    params.tail_start = window.tail_start;

    std::memcpy(node->op_params, &params, sizeof(params));

    if (kvarn_graph_attn_trace_enabled() && kvarn_graph_attn_trace_claim()) {
        const ggml_tensor * q           = node->src[0];
        const ggml_tensor * sink_tail_k = node->src[1];
        const ggml_tensor * sink_tail_v = node->src[2];
        const ggml_tensor * body_k      = node->src[3];
        const ggml_tensor * body_v      = node->src[4];
        const ggml_tensor * scales_k    = node->src[5];
        const ggml_tensor * scales_v    = node->src[6];
        const ggml_tensor * pending_k   = node->src[7];
        const ggml_tensor * pending_v   = node->src[8];
        const ggml_tensor * scratch     = node->src[9];
        const ggml_tensor * kq_mask     = node->src[10];

        std::fprintf(stderr,
                "KVarN graph mixed-attn trace: n_queries=%" PRId64 " n_head=%" PRId64
                " n_head_kv=%" PRId64 " n_sink=%d n_records=%d n_pending=%d n_tail=%d"
                " tail_start=%d n_kv=%" PRId64 " head_dim=%d group_size=%d k_bits=%d v_bits=%d"
                " scale=%.9g logit_softcap=%.9g frame_flags=%d scratch_elems=%" PRId64 "\n",
                node->ne[2], node->ne[1], sink_tail_k ? sink_tail_k->ne[1] : 0,
                params.n_sink, params.n_records, params.n_pending, params.n_tail,
                params.tail_start, window.n_kv, params.head_dim, params.group_size,
                params.key_bits, params.value_bits, double(params.scale), double(params.logit_softcap), params.frame_flags,
                scratch ? ggml_nelements(scratch) : 0);
        kvarn_graph_attn_trace_tensor("q", q);
        kvarn_graph_attn_trace_tensor("sink_tail_k", sink_tail_k);
        kvarn_graph_attn_trace_tensor("sink_tail_v", sink_tail_v);
        kvarn_graph_attn_trace_tensor("body_k", body_k);
        kvarn_graph_attn_trace_tensor("body_v", body_v);
        kvarn_graph_attn_trace_tensor("scales_k", scales_k);
        kvarn_graph_attn_trace_tensor("scales_v", scales_v);
        kvarn_graph_attn_trace_tensor("pending_k", pending_k);
        kvarn_graph_attn_trace_tensor("pending_v", pending_v);
        kvarn_graph_attn_trace_tensor("scratch", scratch);
        kvarn_graph_attn_trace_tensor("kq_mask", kq_mask);
        kvarn_graph_attn_trace_tensor("out", node);
    }
}

static bool kvarn_graph_use_attn_scratch_ref() {
    return kvarn_graph_parse_env_flag("LLAMA_KVARN_ATTN_REF_SCRATCH");
}

static bool kvarn_graph_use_attn_f16_body_mirror(int64_t head_dim) {
    if (head_dim < 256) {
        return false;
    }
    return kvarn_graph_parse_env_flag("LLAMA_KVARN_ATTN_REF_SCRATCH") ||
        kvarn_graph_parse_env_flag("LLAMA_KVARN_ENABLE_F32_DEQUANT_CACHE") ||
        kvarn_graph_parse_env_flag("LLAMA_KVARN_ATTN_ENABLE_BODY_F32_MIRROR");
}

static bool kvarn_graph_reuse_unsafe_forced_512_fused(
        const kvarn_active_window & window,
        const ggml_tensor * node) {
    GGML_UNUSED(window);

    if (node == nullptr || node->op != GGML_OP_KVARN_ATTN_MIXED) {
        return false;
    }

    const int32_t head_dim = node->op_params[5];
    if (head_dim < 512) {
        return false;
    }

    return kvarn_graph_parse_env_flag("LLAMA_KVARN_ATTN_FUSED_BATCH") &&
           kvarn_graph_parse_env_flag("LLAMA_KVARN_UNSAFE_ALLOW_FUSED_BATCH");
}

static int64_t kvarn_graph_attn_scratch_floats(
        const kvarn_active_window & window,
        int64_t n_head_kv,
        int64_t n_records_scratch,
        int64_t head_dim,
        int64_t group_size) {
    int64_t result = window.n_kv;
    if (n_records_scratch > 0 &&
            (kvarn_graph_use_attn_scratch_ref() || kvarn_graph_use_attn_f16_body_mirror(head_dim))) {
        result += 2*n_head_kv*n_records_scratch*head_dim*group_size;
    }
    return result;
}

static ggml_tensor * kvarn_graph_build_hadamard_input(
        ggml_context * ctx,
        uint32_t head_dim,
        ggml_tensor * & tensor,
        std::vector<float> & host,
        bool & filled,
        uint32_t & dim,
        const char * name) {
    if (tensor == nullptr || dim != head_dim) {
        tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim, head_dim);
        ggml_set_input(tensor);
        ggml_set_name(tensor, name);
        host = llama_kvarn_hadamard_matrix(head_dim);
        dim = head_dim;
        filled = false;
    }
    return tensor;
}

static std::vector<float> kvarn_graph_turbo_wht_matrix(uint32_t head_dim, bool inverse) {
    static constexpr int8_t signs1[128] = {
        -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
        -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
        -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
        -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1,
    };
    static constexpr int8_t signs2[128] = {
         1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
         1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
         1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
         1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1,
    };

    if (head_dim == 0 || head_dim % 128 != 0) {
        throw std::invalid_argument("Turbo V frame requires head_dim_v to be a non-zero multiple of 128");
    }

    const int8_t * first = inverse ? signs2 : signs1;
    const int8_t * last  = inverse ? signs1 : signs2;
    const float norm = 1.0f/std::sqrt(128.0f);

    std::vector<float> M(size_t(head_dim)*head_dim, 0.0f);
    std::vector<float> col(128, 0.0f);
    for (uint32_t block = 0; block < head_dim; block += 128) {
        for (uint32_t j = 0; j < 128; ++j) {
            std::fill(col.begin(), col.end(), 0.0f);
            col[j] = float(first[j]);
            for (uint32_t step = 1; step < 128; step <<= 1) {
                for (uint32_t base = 0; base < 128; base += 2*step) {
                    for (uint32_t i = 0; i < step; ++i) {
                        const uint32_t i0 = base + i;
                        const uint32_t i1 = i0 + step;
                        const float a = col[i0];
                        const float b = col[i1];
                        col[i0] = a + b;
                        col[i1] = a - b;
                    }
                }
            }
            for (uint32_t r = 0; r < 128; ++r) {
                M[size_t(block + r)*head_dim + (block + j)] = col[r]*norm*float(last[r]);
            }
        }
    }
    return M;
}

static std::vector<float> kvarn_graph_identity_matrix(uint32_t dim) {
    std::vector<float> M(size_t(dim)*dim, 0.0f);
    for (uint32_t i = 0; i < dim; ++i) {
        M[size_t(i)*dim + i] = 1.0f;
    }
    return M;
}

static std::vector<float> kvarn_graph_turbo_sign_vector(uint32_t head_dim, int which) {
    static constexpr int8_t signs1[128] = {
        -1, 1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1, 1,-1, 1, 1,-1,-1,-1,
        -1, 1, 1,-1, 1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1, 1, 1, 1,-1,-1,-1,-1,-1, 1,-1, 1, 1, 1, 1,-1, 1,
        -1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1, 1,-1,-1, 1, 1, 1,-1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
        -1, 1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1,-1, 1, 1,-1,-1,-1,-1,-1, 1, 1,-1, 1, 1,-1, 1,
    };
    static constexpr int8_t signs2[128] = {
         1, 1, 1, 1,-1, 1, 1,-1, 1,-1,-1,-1, 1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1,
         1, 1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1, 1,-1, 1,-1, 1, 1, 1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
         1,-1, 1,-1,-1,-1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1,-1, 1,-1,-1,-1,-1, 1,-1,-1, 1,-1,
         1,-1, 1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,-1,
    };
    if (head_dim == 0 || head_dim % 128 != 0) {
        throw std::invalid_argument("Turbo V sign vector requires head_dim_v to be a non-zero multiple of 128");
    }
    const int8_t * signs = which == 1 ? signs1 : signs2;
    std::vector<float> out(head_dim);
    for (uint32_t i = 0; i < head_dim; ++i) {
        out[i] = float(signs[i % 128]);
    }
    return out;
}

static ggml_tensor * kvarn_graph_build_turbo_sign_input(
        ggml_context * ctx,
        uint32_t head_dim,
        int which,
        ggml_backend_sched_t sched,
        ggml_backend_t backend,
        ggml_tensor * & tensor,
        std::vector<float> & host,
        bool & filled,
        uint32_t & dim,
        const char * name) {
    if (tensor != nullptr && dim != 0 && dim != head_dim) {
        throw std::runtime_error("LLAMA_KVARN_ISWA_DEBUG_TURBO_V_FRAME does not support mixed head_dim_v in one graph input");
    }
    if (tensor == nullptr) {
        tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
        ggml_set_input(tensor);
        ggml_set_name(tensor, name);
        host = kvarn_graph_turbo_sign_vector(head_dim, which);
        dim = head_dim;
        filled = false;
    }
    if (sched != nullptr && backend != nullptr) {
        ggml_backend_sched_set_tensor_backend(sched, tensor, backend);
    }
    return tensor;
}

static ggml_tensor * kvarn_graph_build_turbo_wht_input(
        ggml_context * ctx,
        uint32_t head_dim,
        bool inverse,
        bool identity,
        ggml_backend_sched_t sched,
        ggml_backend_t backend,
        ggml_tensor * & tensor,
        std::vector<float> & host,
        bool & filled,
        uint32_t & dim,
        const char * name) {
    if (tensor != nullptr && dim != 0 && dim != head_dim) {
        throw std::runtime_error("LLAMA_KVARN_ISWA_DEBUG_TURBO_V_FRAME does not support mixed head_dim_v in one graph input");
    }
    if (tensor == nullptr) {
        tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_dim, head_dim);
        ggml_set_input(tensor);
        ggml_set_name(tensor, name);
        host = identity ? kvarn_graph_identity_matrix(head_dim) : kvarn_graph_turbo_wht_matrix(head_dim, inverse);
        dim = head_dim;
        filled = false;
    }
    if (sched != nullptr && backend != nullptr) {
        ggml_backend_sched_set_tensor_backend(sched, tensor, backend);
    }
    return tensor;
}

static ggml_tensor * kvarn_graph_apply_turbo_wht(
        ggml_context * ctx,
        ggml_tensor * cur,
        ggml_tensor * H,
        ggml_tensor * s1,
        ggml_tensor * s2,
        bool inverse,
        ggml_backend_sched_t sched,
        ggml_backend_t backend) {
    ggml_tensor * first = inverse ? s2 : s1;
    ggml_tensor * last  = inverse ? s1 : s2;
    cur = ggml_mul(ctx, cur, first);
    if (sched != nullptr && backend != nullptr) {
        ggml_backend_sched_set_tensor_backend(sched, cur, backend);
    }
    cur = ggml_mul_mat_aux(ctx, cur, H, true, sched, backend);
    cur = ggml_mul(ctx, cur, last);
    if (sched != nullptr && backend != nullptr) {
        ggml_backend_sched_set_tensor_backend(sched, cur, backend);
    }
    return cur;
}

static ggml_tensor * kvarn_graph_apply_hadamard(
        ggml_context * ctx,
        ggml_tensor * H,
        ggml_tensor * x) {
    if (!ggml_is_contiguous(x)) {
        x = ggml_cont(ctx, x);
    }
    ggml_tensor * res = ggml_mul_mat(ctx, H, x);
    ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD);
    return res;
}

void llm_graph_input_attn_kvarn::set_input(const llama_ubatch * ubatch) {
    mctx_kvarn->set_input_sink_tail_idxs(sink_tail_idxs, ubatch);
    mctx_kvarn->set_input_body_plan(body_plan, ubatch);
    mctx_kvarn->set_input_body_offsets(body_offsets, ubatch);
    mctx_kvarn->set_input_tail_evict_idxs(tail_evict_idxs, ubatch);
    mctx_kvarn->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    if (kvarn_hadamard != nullptr && kvarn_hadamard->buffer != nullptr && !kvarn_hadamard_filled) {
        ggml_backend_tensor_set(
                kvarn_hadamard,
                kvarn_hadamard_host.data(),
                0,
                kvarn_hadamard_host.size()*sizeof(float));
        kvarn_hadamard_filled = true;
    }

    const kvarn_active_window window = kvarn_graph_active_window(cparams.kvarn, *ubatch, mctx_kvarn->get_size());
    GGML_ASSERT(window.valid);
    if (window_indirect && kvarn_window != nullptr && kvarn_window->buffer != nullptr) {
        GGML_ASSERT(window.n_records == 0 && window.n_pending == 0);
        const int32_t win[8] = {
            int32_t(window.n_sink), int32_t(window.n_records), int32_t(window.n_pending),
            int32_t(window.n_tail), int32_t(window.tail_start), 0, 0, 0,
        };
        ggml_backend_tensor_set(kvarn_window, win, 0, sizeof(win));
    } else {
        for (ggml_tensor * node : mixed_attn_nodes) {
            kvarn_graph_update_mixed_attn_params(node, window);
        }
    }
}

bool llm_graph_input_attn_kvarn::can_reuse(const llm_graph_params & params) {
    const auto * mctx = dynamic_cast<const llama_kv_cache_kvarn_context *>(params.mctx);
    if (mctx == nullptr) {
        return false;
    }

    this->mctx_kvarn = mctx;

    const kvarn_active_window window = kvarn_graph_active_window(params.cparams.kvarn, params.ubatch, mctx->get_size());
    if (!window.valid) {
        return false;
    }

    const std::vector<uint32_t> cur_seal_records = kvarn_graph_seal_records(params.cparams.kvarn, params.ubatch, mctx->get_size());
    if (has_body_store_ops) {
        if (cur_seal_records != baked_seal_records) {
            return false;
        }
    } else if (!cur_seal_records.empty()) {
        return false;
    }

    const uint32_t n_tail_evict = kvarn_graph_count_tail_evictions(params.cparams.kvarn, params.ubatch, mctx->get_size());

    bool res = true;
    res &= sink_tail_idxs->ne[0] == params.ubatch.n_tokens;
    res &= body_plan->ne[1] == n_tail_evict;
    res &= body_offsets->ne[0] == n_tail_evict;
    res &= tail_evict_idxs->ne[0] == n_tail_evict;
    res &= can_reuse_kq_mask(
            self_kq_mask,
            kvarn_graph_reuse_mask_n_kv(window, mctx->get_size(), params.ubatch),
            params.ubatch,
            params.cparams);
    res &= !mixed_attn_nodes.empty();
    if (window_indirect) {
        res &= kvarn_graph_decode_stable_topology(params.ubatch) &&
               window.n_records == 0 && window.n_pending == 0;
    }

    for (const ggml_tensor * node : mixed_attn_nodes) {
        res &= node->op == GGML_OP_KVARN_ATTN_MIXED;
        if (kvarn_graph_reuse_unsafe_forced_512_fused(window, node)) {
            return false;
        }
        const int64_t n_head_kv = node->src[1] ? node->src[1]->ne[1] : 0;
        const kvarn_active_window scratch_window = kvarn_graph_build_scratch_window(
                window, params.cparams.kvarn, mctx->get_size(), params.ubatch);
        const int64_t required_scratch = kvarn_graph_attn_scratch_floats(
                scratch_window, n_head_kv, scratch_window.n_records, node->op_params[5], node->op_params[6]);
        res &= node->src[9] != nullptr && ggml_nelements(node->src[9]) >= required_scratch;
    }

    return res;
}

void llm_graph_input_attn_kvarn_filter::set_input(const llama_ubatch * ubatch) {
    inp_normal->set_input(ubatch);
    inp_kvarn->set_input(ubatch);
}

bool llm_graph_input_attn_kvarn_filter::can_reuse(const llm_graph_params & params) {
    GGML_UNUSED(params);
    // Diagnostic layer-filter graphs combine two attention memories. Rebuilding
    // prevents stale mixed KVarN topology or normal-KV masks from being reused
    // across layer-bisection cells.
    return false;
}

llm_graph_input_attn_kv * llm_graph_input_attn_kvarn_filter::input_for_layer(int32_t il) const {
    const auto * kvarn = dynamic_cast<const llm_graph_input_attn_kvarn *>(inp_kvarn.get());
    GGML_ASSERT(kvarn != nullptr);

    if (kvarn->mctx_kvarn->has_layer(il)) {
        return inp_kvarn.get();
    }

    GGML_ASSERT(inp_normal->mctx != nullptr);
    if (!inp_normal->mctx->has_layer(il)) {
        throw std::runtime_error("KVarN layer-filter graph has neither KVarN nor normal KV storage for layer");
    }

    return inp_normal.get();
}

void llm_graph_input_attn_k::set_input(const llama_ubatch * ubatch) {
    mctx->set_input_k_idxs(self_k_idxs, ubatch);

    mctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
}

bool llm_graph_input_attn_k::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(self_kq_mask, mctx, params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_k_dsa::set_input(const llama_ubatch * ubatch) {
    mctx->get_mla()->set_input_k_idxs(self_k_idxs_mla, ubatch);

    mctx->get_mla()->set_input_kq_mask(self_kq_mask_mla, ubatch, cparams.causal_attn);

    mctx->get_lid()->set_input_k_idxs(self_k_idxs_lid, ubatch);

    mctx->get_lid()->set_input_kq_mask(self_kq_mask_lid, ubatch, cparams.causal_attn);

    mctx->get_lid()->set_input_k_rot(self_k_rot_lid);
}

bool llm_graph_input_attn_k_dsa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_kv_cache_dsa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= self_k_idxs_mla->ne[0] == params.ubatch.n_tokens;
    res &= self_k_idxs_lid->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(self_kq_mask_mla, mctx->get_mla(), params.ubatch, params.cparams);
    res &= can_reuse_kq_mask(self_kq_mask_lid, mctx->get_lid(), params.ubatch, params.cparams);

    return res;
}

void llm_graph_input_attn_kv_iswa::set_input(const llama_ubatch * ubatch) {
    if (mctx_kvarn_iswa != nullptr) {
        const auto * base_ctx = mctx_kvarn_iswa->get_base();
        const bool have_base_kvarn_inputs =
            base_sink_tail_idxs  != nullptr && base_sink_tail_idxs->buffer  != nullptr &&
            base_body_plan       != nullptr && base_body_plan->buffer       != nullptr &&
            base_body_offsets    != nullptr && base_body_offsets->buffer    != nullptr &&
            base_tail_evict_idxs != nullptr && base_tail_evict_idxs->buffer != nullptr &&
            base_kvarn_kq_mask   != nullptr && base_kvarn_kq_mask->buffer   != nullptr;
        if (have_base_kvarn_inputs) {
            base_ctx->set_input_sink_tail_idxs(base_sink_tail_idxs, ubatch);
            base_ctx->set_input_body_plan(base_body_plan, ubatch);
            base_ctx->set_input_body_offsets(base_body_offsets, ubatch);
            base_ctx->set_input_tail_evict_idxs(base_tail_evict_idxs, ubatch);
            base_ctx->set_input_kq_mask(base_kvarn_kq_mask, ubatch, cparams.causal_attn);
        } else if (!base_mixed_attn_nodes.empty()) {
            throw std::runtime_error("KVarN+ISWA graph has mixed-attention nodes but missing base KVarN inputs");
        }
        if (base_kvarn_hadamard != nullptr && base_kvarn_hadamard->buffer != nullptr && !base_kvarn_hadamard_filled) {
            ggml_backend_tensor_set(
                    base_kvarn_hadamard,
                    base_kvarn_hadamard_host.data(),
                    0,
                    base_kvarn_hadamard_host.size()*sizeof(float));
            base_kvarn_hadamard_filled = true;
        }
        if (base_kvarn_turbo_v_fwd != nullptr && base_kvarn_turbo_v_fwd->buffer != nullptr) {
            ggml_backend_tensor_set(
                    base_kvarn_turbo_v_fwd,
                    base_kvarn_turbo_v_fwd_host.data(),
                    0,
                    base_kvarn_turbo_v_fwd_host.size()*sizeof(float));
            base_kvarn_turbo_v_fwd_filled = true;
        }
        if (base_kvarn_turbo_v_inv != nullptr && base_kvarn_turbo_v_inv->buffer != nullptr) {
            ggml_backend_tensor_set(
                    base_kvarn_turbo_v_inv,
                    base_kvarn_turbo_v_inv_host.data(),
                    0,
                    base_kvarn_turbo_v_inv_host.size()*sizeof(float));
            base_kvarn_turbo_v_inv_filled = true;
        }
        if (base_kvarn_turbo_v_s1 != nullptr && base_kvarn_turbo_v_s1->buffer != nullptr) {
            ggml_backend_tensor_set(
                    base_kvarn_turbo_v_s1,
                    base_kvarn_turbo_v_s1_host.data(),
                    0,
                    base_kvarn_turbo_v_s1_host.size()*sizeof(float));
            base_kvarn_turbo_v_s1_filled = true;
        }
        if (base_kvarn_turbo_v_s2 != nullptr && base_kvarn_turbo_v_s2->buffer != nullptr) {
            ggml_backend_tensor_set(
                    base_kvarn_turbo_v_s2,
                    base_kvarn_turbo_v_s2_host.data(),
                    0,
                    base_kvarn_turbo_v_s2_host.size()*sizeof(float));
            base_kvarn_turbo_v_s2_filled = true;
        }

        const kvarn_active_window window = kvarn_graph_active_window(cparams.kvarn, *ubatch, base_ctx->get_size());
        GGML_ASSERT(window.valid);
        if (have_base_kvarn_inputs && base_window_indirect && base_kvarn_window != nullptr && base_kvarn_window->buffer != nullptr) {
            // Frozen op_params + device-side live window: never rewrite the
            // node params here, or CUDA graph node properties change and the
            // captured decode graph re-captures every token.
            GGML_ASSERT(window.n_records == 0 && window.n_pending == 0);
            const int32_t win[8] = {
                int32_t(window.n_sink), int32_t(window.n_records), int32_t(window.n_pending),
                int32_t(window.n_tail), int32_t(window.tail_start), 0, 0, 0,
            };
            ggml_backend_tensor_set(base_kvarn_window, win, 0, sizeof(win));
        } else if (have_base_kvarn_inputs) {
            for (ggml_tensor * node : base_mixed_attn_nodes) {
                kvarn_graph_update_mixed_attn_params(node, window);
            }
        }

        const auto * swa_ctx = mctx_kvarn_iswa->get_swa();
        if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
            swa_ctx->set_input_k_idxs(self_k_idxs_swa, ubatch);
            swa_ctx->set_input_v_idxs(self_v_idxs_swa, ubatch);
            swa_ctx->set_input_kq_mask(self_kq_mask_swa, ubatch, cparams.causal_attn);
        }
        if (self_k_rot_swa) {
            swa_ctx->set_input_k_rot(self_k_rot_swa);
        }
        if (self_v_rot_swa) {
            swa_ctx->set_input_v_rot(self_v_rot_swa);
        }
        if (const auto * full_ctx = mctx_kvarn_iswa->get_full_normal()) {
            if (self_k_idxs && self_k_idxs->buffer) {
                full_ctx->set_input_k_idxs(self_k_idxs, ubatch);
                full_ctx->set_input_v_idxs(self_v_idxs, ubatch);
            }
            if (self_kq_mask && self_kq_mask->buffer) {
                full_ctx->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
            }
            if (self_k_rot) {
                full_ctx->set_input_k_rot(self_k_rot);
            }
            if (self_v_rot) {
                full_ctx->set_input_v_rot(self_v_rot);
            }
        }
        return;
    }

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        mctx->get_base()->set_input_k_idxs(self_k_idxs, ubatch);
        mctx->get_base()->set_input_v_idxs(self_v_idxs, ubatch);
    }

    // the kq mask guards on its own buffer: shared cells leave idxs unbacked while the mask stays live
    if (self_kq_mask && self_kq_mask->buffer) {
        mctx->get_base()->set_input_kq_mask(self_kq_mask, ubatch, cparams.causal_attn);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        mctx->get_swa()->set_input_k_idxs(self_k_idxs_swa, ubatch);
        mctx->get_swa()->set_input_v_idxs(self_v_idxs_swa, ubatch);
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        mctx->get_swa()->set_input_kq_mask(self_kq_mask_swa, ubatch, cparams.causal_attn);
    }

    if (self_k_rot) {
        mctx->get_base()->set_input_k_rot(self_k_rot);
    }

    if (self_v_rot) {
        mctx->get_base()->set_input_v_rot(self_v_rot);
    }

    if (self_k_rot_swa) {
        mctx->get_swa()->set_input_k_rot(self_k_rot_swa);
    }

    if (self_v_rot_swa) {
        mctx->get_swa()->set_input_v_rot(self_v_rot_swa);
    }
}

void llm_graph_input_attn_kvarn::rewire_kvarn_mixed_attn_inputs() {
    const size_t n = mixed_attn_nodes.size();
    for (size_t i = 0; i < n; ++i) {
        ggml_tensor * node = mixed_attn_nodes[i];
        if (node == nullptr || node->op != GGML_OP_KVARN_ATTN_MIXED) {
            continue;
        }
        if (i < mixed_attn_scores.size() && mixed_attn_scores[i] != nullptr) {
            node->src[9] = mixed_attn_scores[i];
        }
        if (self_kq_mask_cnv != nullptr) {
            node->src[10] = self_kq_mask_cnv;
        }
    }
}

void llm_graph_input_attn_kv_iswa::rewire_kvarn_mixed_attn_inputs() {
    const size_t n = base_mixed_attn_nodes.size();
    for (size_t i = 0; i < n; ++i) {
        ggml_tensor * node = base_mixed_attn_nodes[i];
        if (node == nullptr || node->op != GGML_OP_KVARN_ATTN_MIXED) {
            continue;
        }
        if (i < base_mixed_attn_scores.size() && base_mixed_attn_scores[i] != nullptr) {
            node->src[9] = base_mixed_attn_scores[i];
        }
        if (base_kvarn_kq_mask_cnv != nullptr) {
            node->src[10] = base_kvarn_kq_mask_cnv;
        }
    }
}

void llm_graph_input_attn_kv_iswa::refresh_kvarn_params(const llama_ubatch & ubatch) {
    if (mctx_kvarn_iswa == nullptr || base_mixed_attn_nodes.empty()) {
        return;
    }
    // Window-indirect graphs keep frozen op_params by design (CUDA graph
    // replay stability) — never rewrite them here.
    if (base_window_indirect) {
        return;
    }
    const kvarn_active_window window = kvarn_graph_active_window(
            cparams.kvarn, ubatch, mctx_kvarn_iswa->get_base()->get_size());
    if (!window.valid) {
        return;
    }
    for (ggml_tensor * node : base_mixed_attn_nodes) {
        kvarn_graph_update_mixed_attn_params(node, window);
    }
}

bool llm_graph_input_attn_kv_iswa::can_reuse(const llm_graph_params & params) {
    if (const auto * mctx = dynamic_cast<const llama_kv_cache_kvarn_iswa_context *>(params.mctx)) {
        this->mctx_kvarn_iswa = mctx;
        this->mctx = nullptr;
        const bool reuse_trace = kvarn_graph_reuse_trace_enabled();
        if (reuse_trace) {
            const char * reason = mctx->get_full_normal() != nullptr ?
                "diagnostic KVarN+ISWA normal fallback disables graph reuse" :
                "KVarN+ISWA graph reuse disabled pending independent correctness proof";
            kvarn_graph_reuse_trace_miss(__func__, reason);
        }
        return false;

        const auto * base_ctx = mctx->get_base();
        const kvarn_active_window window = kvarn_graph_active_window(params.cparams.kvarn, params.ubatch, base_ctx->get_size());
        if (!window.valid) {
            if (reuse_trace) {
                kvarn_graph_reuse_trace_miss(__func__, "kvarn-active-window-invalid");
            }
            return false;
        }

        const std::vector<uint32_t> cur_seal_records = kvarn_graph_seal_records(params.cparams.kvarn, params.ubatch, base_ctx->get_size());
        if (base_has_body_store_ops) {
            if (cur_seal_records != base_baked_seal_records) {
                if (reuse_trace) {
                    std::fprintf(stderr,
                            "%s: KVarN graph reuse miss: seal-records mismatch baked=%zu current=%zu\n",
                            __func__, base_baked_seal_records.size(), cur_seal_records.size());
                }
                return false;
            }
        } else if (!cur_seal_records.empty()) {
            if (reuse_trace) {
                std::fprintf(stderr,
                        "%s: KVarN graph reuse miss: graph has no baked body-store ops but current ubatch seals %zu record(s)\n",
                        __func__, cur_seal_records.size());
            }
            return false;
        }

        const uint32_t n_tail_evict = kvarn_graph_count_tail_evictions(params.cparams.kvarn, params.ubatch, base_ctx->get_size());
        bool res = true;
        auto check = [&](bool ok, const char * name) {
            if (!ok && reuse_trace) {
                kvarn_graph_reuse_trace_miss(__func__, name);
            }
            res &= ok;
        };

        // A window-indirect graph carries frozen op_params and dispatches the
        // pure sink/tail branch unconditionally; it must never be reused for
        // a ubatch whose live window has body records or pending tokens.
        if (base_window_indirect) {
            check(kvarn_graph_decode_stable_topology(params.ubatch) &&
                    window.n_records == 0 && window.n_pending == 0,
                    "window-indirect graph vs non-sink/tail regime");
        }

        check(base_sink_tail_idxs->ne[0] == params.ubatch.n_tokens,
                "base_sink_tail_idxs.ne[0] != ubatch.n_tokens");
        check(base_body_plan->ne[1] == n_tail_evict,
                "base_body_plan.ne[1] != n_tail_evict");
        check(base_body_offsets->ne[0] == n_tail_evict,
                "base_body_offsets.ne[0] != n_tail_evict");
        check(base_tail_evict_idxs->ne[0] == n_tail_evict,
                "base_tail_evict_idxs.ne[0] != n_tail_evict");
        check(can_reuse_kq_mask(
                base_kvarn_kq_mask,
                kvarn_graph_reuse_mask_n_kv(window, base_ctx->get_size(), params.ubatch),
                params.ubatch,
                params.cparams),
                "base KVarN kq mask shape mismatch");
        check(!base_mixed_attn_nodes.empty(),
                "base_mixed_attn_nodes empty");

        for (const ggml_tensor * node : base_mixed_attn_nodes) {
            check(node->op == GGML_OP_KVARN_ATTN_MIXED,
                    "base mixed-attn node op mismatch");
            if (kvarn_graph_reuse_unsafe_forced_512_fused(window, node)) {
                if (reuse_trace) {
                    kvarn_graph_reuse_trace_miss(__func__, "unsafe forced 512 fused disables reuse");
                }
                return false;
            }
            const int64_t n_head_kv = node->src[1] ? node->src[1]->ne[1] : 0;
            const kvarn_active_window scratch_window = kvarn_graph_build_scratch_window(
                    window, params.cparams.kvarn, base_ctx->get_size(), params.ubatch);
            const int64_t required_scratch = kvarn_graph_attn_scratch_floats(
                    scratch_window, n_head_kv, scratch_window.n_records, node->op_params[5], node->op_params[6]);
            check(node->src[9] != nullptr && ggml_nelements(node->src[9]) >= required_scratch,
                    "base mixed-attn scratch too small/null");
        }

        const auto * swa_ctx = mctx->get_swa();
        if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
            check(self_k_idxs_swa->ne[0] == params.ubatch.n_tokens,
                    "self_k_idxs_swa.ne[0] != ubatch.n_tokens");
            check(can_reuse_kq_mask(self_kq_mask_swa, swa_ctx, params.ubatch, params.cparams),
                    "SWA self_kq_mask_swa shape mismatch");
        }

        return res;
    }

    const auto * mctx = static_cast<const llama_kv_cache_iswa_context *>(params.mctx);

    this->mctx = mctx;
    this->mctx_kvarn_iswa = nullptr;

    bool res = true;

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (self_k_idxs && self_k_idxs->buffer) {
        res &= self_k_idxs->ne[0] == params.ubatch.n_tokens;
      //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    if (self_kq_mask && self_kq_mask->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask, mctx->get_base(), params.ubatch, params.cparams);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (self_k_idxs_swa && self_k_idxs_swa->buffer) {
        res &= self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
      //res &= self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    if (self_kq_mask_swa && self_kq_mask_swa->buffer) {
        res &= can_reuse_kq_mask(self_kq_mask_swa, mctx->get_swa(), params.ubatch, params.cparams);
    }

    return res;
}

void llm_graph_input_attn_cross::set_input(const llama_ubatch * ubatch) {
    GGML_ASSERT(cross_kq_mask);

    const int64_t n_enc    = cross_kq_mask->ne[0];
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(cross_kq_mask->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    const auto fill_mask = [&](auto * data) {
        using T = std::remove_reference_t<decltype(*data)>;
        for (int i = 0; i < n_tokens; ++i) {
            GGML_ASSERT(!cross->seq_ids_enc.empty() && "llama_encode must be called first");
            for (int j = 0; j < n_enc; ++j) {
                float f = -INFINITY;

                for (int s = 0; s < ubatch->n_seq_id[i]; ++s) {
                    const llama_seq_id seq_id = ubatch->seq_id[i][s];

                    if (cross->seq_ids_enc[j].find(seq_id) != cross->seq_ids_enc[j].end()) {
                        f = 0.0f;
                    }
                }

                data[i*n_enc + j] = llama_cast<T>(f);
            }
        }
    };

    if (cross_kq_mask->type == GGML_TYPE_F16) {
        fill_mask((ggml_fp16_t *) cross_kq_mask->data);
    } else {
        fill_mask((float *) cross_kq_mask->data);
    }
}

void llm_graph_input_mem_hybrid::set_input(const llama_ubatch * ubatch) {
    mctx->get_attn()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);
    mctx->get_attn()->set_input_v_idxs(inp_attn->self_v_idxs, ubatch);

    mctx->get_attn()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);

    if (inp_attn->self_k_rot) {
        mctx->get_attn()->set_input_k_rot(inp_attn->self_k_rot);
    }

    if (inp_attn->self_v_rot) {
        mctx->get_attn()->set_input_v_rot(inp_attn->self_v_rot);
    }

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
  //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, mctx->get_attn(), params.ubatch, params.cparams);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

// TODO: Hybrid input classes are a bit redundant.
// Instead of creating a hybrid input, the graph can simply create 2 separate inputs.
// Refactoring is required in the future.
void llm_graph_input_mem_hybrid_k::set_input(const llama_ubatch * ubatch) {
    mctx->get_attn()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);

    mctx->get_attn()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid_k::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, mctx->get_attn(), params.ubatch, params.cparams);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

void llm_graph_input_mem_hybrid_kvarn::set_input(const llama_ubatch * ubatch) {
    inp_attn->set_input(ubatch);

    const int64_t n_rs = mctx_kvarn->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx_kvarn->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid_kvarn::can_reuse(const llm_graph_params & params) {
    const auto * mctx = dynamic_cast<const llama_memory_hybrid_kvarn_context *>(params.mctx);
    if (mctx == nullptr) {
        return false;
    }

    this->mctx_kvarn = mctx;

    llm_graph_params params_attn = params;
    params_attn.mctx = mctx->get_attn();

    bool res = true;

    res &= inp_attn->can_reuse(params_attn);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

void llm_graph_input_mem_hybrid_iswa::set_input(const llama_ubatch * ubatch) {
    const auto * attn_ctx = mctx->get_attn();

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        attn_ctx->get_base()->set_input_k_idxs(inp_attn->self_k_idxs, ubatch);
        attn_ctx->get_base()->set_input_v_idxs(inp_attn->self_v_idxs, ubatch);
    }

    if (inp_attn->self_kq_mask && inp_attn->self_kq_mask->buffer) {
        attn_ctx->get_base()->set_input_kq_mask(inp_attn->self_kq_mask, ubatch, cparams.causal_attn);
    }

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        attn_ctx->get_swa()->set_input_k_idxs(inp_attn->self_k_idxs_swa, ubatch);
        attn_ctx->get_swa()->set_input_v_idxs(inp_attn->self_v_idxs_swa, ubatch);
    }

    if (inp_attn->self_kq_mask_swa && inp_attn->self_kq_mask_swa->buffer) {
        attn_ctx->get_swa()->set_input_kq_mask(inp_attn->self_kq_mask_swa, ubatch, cparams.causal_attn);
    }

    if (inp_attn->self_k_rot) {
        attn_ctx->get_base()->set_input_k_rot(inp_attn->self_k_rot);
    }

    if (inp_attn->self_v_rot) {
        attn_ctx->get_base()->set_input_v_rot(inp_attn->self_v_rot);
    }

    if (inp_attn->self_k_rot_swa) {
        attn_ctx->get_swa()->set_input_k_rot(inp_attn->self_k_rot_swa);
    }

    if (inp_attn->self_v_rot_swa) {
        attn_ctx->get_swa()->set_input_v_rot(inp_attn->self_v_rot_swa);
    }

    const int64_t n_rs = mctx->get_recr()->get_n_rs();

    if (inp_rs->s_copy) {
        GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer));
        int32_t * data = (int32_t *) inp_rs->s_copy->data;

        // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
        for (uint32_t i = 0; i < n_rs; ++i) {
            data[i] = mctx->get_recr()->s_copy(i);
        }
    }
}

bool llm_graph_input_mem_hybrid_iswa::can_reuse(const llm_graph_params & params) {
    const auto * mctx = static_cast<const llama_memory_hybrid_iswa_context *>(params.mctx);

    this->mctx = mctx;

    bool res = true;

    const auto * attn_ctx = mctx->get_attn();

    // base tensors may not be allocated if there are no non-SWA attention layers
    if (inp_attn->self_k_idxs && inp_attn->self_k_idxs->buffer) {
        res &= inp_attn->self_k_idxs->ne[0] == params.ubatch.n_tokens;
      //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask, attn_ctx->get_base(), params.ubatch, params.cparams);

    // swa tensors may not be allocated if there are no SWA attention layers
    if (inp_attn->self_k_idxs_swa && inp_attn->self_k_idxs_swa->buffer) {
        res &= inp_attn->self_k_idxs_swa->ne[0] == params.ubatch.n_tokens;
      //res &= inp_attn->self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
    }

    res &= can_reuse_kq_mask(inp_attn->self_kq_mask_swa, attn_ctx->get_swa(), params.ubatch, params.cparams);

    res &= inp_rs->s_copy->ne[0] == mctx->get_recr()->get_n_rs();

    res &= inp_rs->s_copy_main->ne[0]  == params.ubatch.n_seqs;
    res &= inp_rs->s_copy_extra->ne[0] == mctx->get_recr()->get_n_rs() - params.ubatch.n_seqs;

    res &= inp_rs->head == mctx->get_recr()->get_head();
    res &= inp_rs->rs_z == mctx->get_recr()->get_rs_z();

    return res;
}

void llm_graph_input_sampling::set_input(const llama_ubatch * ubatch) {
    // set the inputs only for the active samplers in the current ubatch
    std::unordered_set<llama_seq_id> active_samplers;
    for (uint32_t i = 0; i < ubatch->n_tokens; i++) {
        if (ubatch->output[i]) {
            llama_seq_id seq_id = ubatch->seq_id[i][0];
            active_samplers.insert(seq_id);
        }
    }

    for (auto seq_id : active_samplers) {
        if (samplers.find(seq_id) == samplers.end()) {
            continue;
        }

        auto & sampler = samplers[seq_id];

        if (sampler->iface->backend_set_input) {
            sampler->iface->backend_set_input(sampler);
        }
    }
}

bool llm_graph_input_sampling::can_reuse(const llm_graph_params & params) {
    if (samplers.size() != params.samplers.size()) {
        return false;
    }

    for (const auto & [seq_id, sampler] : params.samplers) {
        if (samplers[seq_id] != sampler) {
            return false;
        }
    }

    return true;
}

//
// llm_graph_result
//

llm_graph_result::llm_graph_result(int64_t max_nodes) : max_nodes(max_nodes) {
    reset();

    const char * LLAMA_GRAPH_RESULT_DEBUG = getenv("LLAMA_GRAPH_RESULT_DEBUG");
    debug = LLAMA_GRAPH_RESULT_DEBUG ? atoi(LLAMA_GRAPH_RESULT_DEBUG) : 0;
}

void llm_graph_result::prepare_rebind() {
    ggml_context * ctx = ctx_compute.get();
    if (ctx == nullptr) {
        return;
    }
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        t->buffer = nullptr;
        t->data   = nullptr;
    }
}

void llm_graph_result::refresh_kvarn_params(const llama_ubatch & ubatch) {
    for (auto & input : inputs) {
        if (auto * iswa = dynamic_cast<llm_graph_input_attn_kv_iswa *>(input.get())) {
            iswa->refresh_kvarn_params(ubatch);
        }
    }
}

void llm_graph_result::rewire_kvarn_mixed_attn_inputs() {
    for (auto & input : inputs) {
        if (auto * kvarn = dynamic_cast<llm_graph_input_attn_kvarn *>(input.get())) {
            kvarn->rewire_kvarn_mixed_attn_inputs();
        } else if (auto * iswa = dynamic_cast<llm_graph_input_attn_kv_iswa *>(input.get())) {
            iswa->rewire_kvarn_mixed_attn_inputs();
        }
    }
}

int64_t llm_graph_result::get_max_nodes() const {
    return max_nodes;
}

void llm_graph_result::reset() {
    t_inp_tokens  = nullptr;
    t_inp_embd    = nullptr;
    t_logits      = nullptr;
    t_embd        = nullptr;
    t_embd_pooled = nullptr;
    t_sampled.clear();
    t_sampled_probs.clear();
    t_sampled_logits.clear();
    t_candidates.clear();

    params = {};

    inputs.clear();

    buf_compute_meta.resize(ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false));

    ggml_init_params params = {
        /*.mem_size   =*/ buf_compute_meta.size(),
        /*.mem_buffer =*/ buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    ctx_compute.reset(ggml_init(params));

    gf = ggml_new_graph_custom(ctx_compute.get(), max_nodes, false);
}

void llm_graph_result::set_inputs(const llama_ubatch * ubatch) {
    for (auto & input : inputs) {
        input->set_input(ubatch);
    }
}

void llm_graph_result::set_outputs() {
    if (t_logits != nullptr) {
        ggml_set_output(t_logits);
    }
    if (t_embd != nullptr) {
        ggml_set_output(t_embd);
    }
    if (t_embd_pooled != nullptr) {
        ggml_set_output(t_embd_pooled);
    }
    if (t_h_nextn != nullptr) {
        ggml_set_output(t_h_nextn);
    }
    for (auto & [seq_id, t] : t_sampled) {
        if (t != nullptr) {
            ggml_set_output(t);
        }
    }
    for (auto & [seq_id, t] : t_sampled_probs) {
        if (t != nullptr) {
            ggml_set_output(t);
        }
    }
    for (auto & [seq_id, t] : t_sampled_logits) {
        if (t != nullptr) {
            ggml_set_output(t);
        }
    }
    for (auto & [seq_id, t] : t_candidates) {
        if (t != nullptr) {
            ggml_set_output(t);
        }
    }
}

bool llm_graph_result::can_reuse(const llm_graph_params & params) {
    if (!this->params.allow_reuse(params)) {
        if (debug > 1) {
            LLAMA_LOG_DEBUG("%s: cannot reuse graph due to incompatible graph parameters\n", __func__);
        }

        return false;
    }

    if (debug > 1) {
        LLAMA_LOG_DEBUG("%s: checking compatibility of %d inputs:\n", __func__, (int) inputs.size());
    }

    bool res = true;

    for (auto & input : inputs) {
        const bool cur = input->can_reuse(params);

        if (debug > 1) {
            LLAMA_LOG_DEBUG("%s: can_reuse = %d\n", "placeholder", cur);
        }

        res = res && cur;
    }

    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s: can reuse graph = %d\n", __func__, res);
    }

    return res;
}

llm_graph_input_i * llm_graph_result::add_input(llm_graph_input_ptr input) {
    inputs.emplace_back(std::move(input));
    return inputs.back().get();
}

void llm_graph_result::set_params(const llm_graph_params & params) {
    this->params = params;
}

//
// llm_graph_context
//

llm_graph_context::llm_graph_context(const llm_graph_params & params) :
    arch             (params.arch),
    hparams          (params.hparams),
    cparams          (params.cparams),
    ubatch           (params.ubatch),
    n_embd           (hparams.n_embd),
    n_layer          (hparams.n_layer()),
    n_layer_nextn    (hparams.n_layer_nextn),
    n_rot            (hparams.n_rot()),
    n_ctx            (cparams.n_ctx),
    n_head           (hparams.n_head()),
    n_head_kv        (hparams.n_head_kv()),
    n_embd_head_k    (hparams.n_embd_head_k()),
    n_embd_k_gqa     (hparams.n_embd_k_gqa()),
    n_embd_head_v    (hparams.n_embd_head_v()),
    n_embd_v_gqa     (hparams.n_embd_v_gqa()),
    n_expert         (hparams.n_expert),
    n_expert_used    (cparams.warmup ? hparams.n_expert : hparams.n_expert_used),
    freq_base        (cparams.rope_freq_base),
    freq_scale       (cparams.rope_freq_scale),
    ext_factor       (cparams.yarn_ext_factor),
    attn_factor      (cparams.yarn_attn_factor),
    beta_fast        (cparams.yarn_beta_fast),
    beta_slow        (cparams.yarn_beta_slow),
    norm_eps         (hparams.f_norm_eps),
    norm_rms_eps     (hparams.f_norm_rms_eps),
    n_tokens         (ubatch.n_tokens),
    n_outputs        (params.n_outputs),
    n_ctx_orig       (cparams.n_ctx_orig_yarn),
    pooling_type     (cparams.pooling_type),
    rope_type        (hparams.rope_type),
    sched            (params.sched),
    backend_cpu      (params.backend_cpu),
    cvec             (params.cvec),
    loras            (params.loras),
    mctx             (params.mctx),
    cross            (params.cross),
    samplers         (params.samplers),
    cb_func          (params.cb),
    res              (params.res),
    ctx0             (res->get_ctx()),
    gf               (res->get_gf()) {
        res->set_params(params);
    }

void llm_graph_context::cb(ggml_tensor * cur, const char * name, int il) const {
    if (cb_func) {
        cb_func(ubatch, cur, name, il);
    }
}

ggml_tensor * llm_graph_context::build_cvec(
         ggml_tensor * cur,
                 int   il) const {
    return cvec->apply_to(ctx0, cur, il);
}

ggml_tensor * llm_graph_context::build_lora_mm(
          ggml_tensor * w,
          ggml_tensor * cur,
          ggml_tensor * w_s) const {
    ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);

    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float adapter_scale = lora.second;
        const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

        ggml_tensor * ab_cur = ggml_mul_mat(
                ctx0, lw->b,
                ggml_mul_mat(ctx0, lw->a, cur)
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    if (w_s) {
        res = ggml_mul(ctx0, res, w_s);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_lora_mm_id(
          ggml_tensor * w,   // ggml_tensor * as
          ggml_tensor * cur, // ggml_tensor * b
          ggml_tensor * ids) const {
    ggml_tensor * res = ggml_mul_mat_id(ctx0, w, cur, ids);
    for (const auto & lora : *loras) {
        llama_adapter_lora_weight * lw = lora.first->get_weight(w);
        if (lw == nullptr) {
            continue;
        }

        const float alpha = lora.first->alpha;
        const float rank  = (float) lw->b->ne[0];
        const float scale = alpha ? lora.second * alpha / rank : lora.second;

        ggml_tensor * ab_cur = ggml_mul_mat_id(
                ctx0, lw->b,
                ggml_mul_mat_id(ctx0, lw->a, cur, ids),
                ids
                );

        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }

    return res;
}

ggml_tensor * llm_graph_context::build_norm(
         ggml_tensor * cur,
         ggml_tensor * mw,
         ggml_tensor * mb,
       llm_norm_type   type,
                 int   il) const {
    switch (type) {
        case LLM_NORM:       cur = ggml_norm    (ctx0, cur, hparams.f_norm_eps);     break;
        case LLM_NORM_RMS:   cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps); break;
        case LLM_NORM_GROUP:
            {
                cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);
                cur = ggml_group_norm(ctx0, cur, hparams.n_norm_groups, hparams.f_norm_group_eps);
                cur = ggml_reshape_2d(ctx0, cur, cur->ne[0],    cur->ne[2]);
            } break;
    }

    if (mw || mb) {
        cb(cur, "norm", il);
    }

    if (mw) {
        cur = ggml_mul(ctx0, cur, mw);
        if (mb) {
            cb(cur, "norm_w", il);
        }
    }

    if (mb) {
        cur = ggml_add(ctx0, cur, mb);
    }

    return cur;
}


llm_graph_qkv llm_graph_context::build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head,
                  int64_t   n_head,
                  int64_t   n_head_kv,
                      int   il) const {
    const int64_t n_embd_q  = n_embd_head * n_head;
    const int64_t n_embd_kv = n_embd_head * n_head_kv;

    ggml_tensor * Qcur, * Kcur, * Vcur;

    if (layer.wqkv) {
        // fused QKV path
        ggml_tensor * qkv = build_lora_mm(layer.wqkv, cur, layer.wqkv_s);
        cb(qkv, "wqkv", il);
        if (layer.wqkv_b) {
            qkv = ggml_add(ctx0, qkv, layer.wqkv_b);
            cb(qkv, "wqkv_b", il);
        }
        if (hparams.f_clamp_kqv > 0.0f) {
            qkv = ggml_clamp(ctx0, qkv, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(qkv, "wqkv_clamped", il);
        }
        Qcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head,    n_tokens,
            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1], 0);
        Kcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens,
            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1],
            ggml_row_size(qkv->type, n_embd_q));
        Vcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head_kv, n_tokens,
            ggml_row_size(qkv->type, n_embd_head), qkv->nb[1],
            ggml_row_size(qkv->type, n_embd_q + n_embd_kv));
    } else {
        // separate Q/K/V path
        Qcur = build_lora_mm(layer.wq, cur, layer.wq_s);
        cb(Qcur, "Qcur", il);
        if (layer.wq_b) {
            Qcur = ggml_add(ctx0, Qcur, layer.wq_b);
            cb(Qcur, "Qcur", il);
        }
        if (hparams.f_clamp_kqv > 0.0f) {
            Qcur = ggml_clamp(ctx0, Qcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Qcur, "Qcur_clamped", il);
        }
        Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
        cb(Kcur, "Kcur", il);
        if (layer.wk_b) {
            Kcur = ggml_add(ctx0, Kcur, layer.wk_b);
            cb(Kcur, "Kcur", il);
        }
        if (hparams.f_clamp_kqv > 0.0f) {
            Kcur = ggml_clamp(ctx0, Kcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Kcur, "Kcur_clamped", il);
        }
        Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
        cb(Vcur, "Vcur", il);
        if (layer.wv_b) {
            Vcur = ggml_add(ctx0, Vcur, layer.wv_b);
            cb(Vcur, "Vcur", il);
        }
        if (hparams.f_clamp_kqv > 0.0f) {
            Vcur = ggml_clamp(ctx0, Vcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
            cb(Vcur, "Vcur_clamped", il);
        }
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    }

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    return { Qcur, Kcur, Vcur };
}


ggml_tensor * llm_graph_context::build_ffn(
         ggml_tensor * cur,
         ggml_tensor * up,
         ggml_tensor * up_b,
         ggml_tensor * up_s,
         ggml_tensor * gate,
         ggml_tensor * gate_b,
         ggml_tensor * gate_s,
         ggml_tensor * down,
         ggml_tensor * down_b,
         ggml_tensor * down_s,
         ggml_tensor * act_scales,
     llm_ffn_op_type   type_op,
   llm_ffn_gate_type   type_gate,
                 int   il) const {
    ggml_tensor * tmp = up ? build_lora_mm(up, cur) : cur;
    cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx0, tmp, up_b);
        cb(tmp, "ffn_up_b", il);
    }

    if (up_s) {
        tmp = ggml_mul(ctx0, tmp, up_s);
        cb(tmp, "ffn_up_s", il);
    }

    if (gate) {
        switch (type_gate) {
            case LLM_FFN_SEQ:
                {
                    cur = build_lora_mm(gate, tmp);
                    cb(cur, "ffn_gate", il);
                } break;
            case LLM_FFN_PAR:
                {
                    cur = build_lora_mm(gate, cur);
                    cb(cur, "ffn_gate", il);
                } break;
        }

        if (gate_b) {
            cur = ggml_add(ctx0, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }

        if (gate_s) {
            cur = ggml_mul(ctx0, cur, gate_s);
            cb(cur, "ffn_gate_s", il);
        }

    } else {
        cur = tmp;
    }

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate && type_gate == LLM_FFN_PAR) {
                // Step35: HF clamps gate (after SiLU) and up before multiplication
                if (arch == LLM_ARCH_STEP35 && il >= 0) {
                    const float limit = hparams.swiglu_clamp_shexp[il];
                    constexpr float eps = 1e-6f;
                    if (limit > eps) {
                        ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                        cb(gate_act, "ffn_silu", il);
                        gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                        cb(gate_act, "ffn_silu_clamped", il);

                        tmp = ggml_clamp(ctx0, tmp, -limit, limit);
                        cb(tmp, "ffn_up_clamped", il);

                        cur = ggml_mul(ctx0, gate_act, tmp);
                        cb(cur, "ffn_swiglu_limited", il);
                        type_gate = LLM_FFN_SEQ;
                        break;
                    }
                }

                cur = ggml_swiglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_swiglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_geglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_geglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_gelu", il);
                if (act_scales != NULL) {
                    cur = ggml_div(ctx0, cur, act_scales);
                    cb(cur, "ffn_act", il);
                }
            } break;
        case LLM_FFN_RELU:
            if (gate && type_gate == LLM_FFN_PAR) {
                cur = ggml_reglu_split(ctx0, cur, tmp);
                cb(cur, "ffn_reglu", il);
                type_gate = LLM_FFN_SEQ;
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_relu", il);

                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_sqr(relu)", il);
            } break;
        case LLM_FFN_SWIGLU:
            {
                cur = ggml_swiglu(ctx0, cur);
                cb(cur, "ffn_swiglu", il);
            } break;
        case LLM_FFN_GEGLU:
            {
                cur = ggml_geglu(ctx0, cur);
                cb(cur, "ffn_geglu", il);
            } break;
        case LLM_FFN_REGLU:
            {
                cur = ggml_reglu(ctx0, cur);
                cb(cur, "ffn_reglu", il);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    if (gate && type_gate == LLM_FFN_PAR) {
        cur = ggml_mul(ctx0, cur, tmp);
        cb(cur, "ffn_gate_par", il);
    }

    if (down) {
        cur = build_lora_mm(down, cur);
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
            // GLM4, GLM4_MOE, and JAIS2 seem to have numerical issues with half-precision accumulators
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
        }
    }

    if (down_b) {
        cb(cur, "ffn_down", il);
    }

    if (down_b) {
        cur = ggml_add(ctx0, cur, down_b);
    }

    if (down_s) {
        cur = ggml_mul(ctx0, cur, down_s);
        cb(cur, "ffn_down_s", il);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * up_exps,
         ggml_tensor * gate_exps,
         ggml_tensor * down_exps,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
         llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s) const {
    return build_moe_ffn(
        cur,
        gate_inp,  /* gate_inp_b  */ nullptr,
        up_exps,   /* up_exps_b   */ nullptr,
        gate_exps, /* gate_exps_b */ nullptr,
        down_exps, /* down_exps_b */ nullptr,
        exp_probs_b,
        n_expert,
        n_expert_used,
        type_op,
        norm_w,
        w_scale,
        gating_op,
        il,
        probs_in,
        gate_up_exps,
        /* gate_up_exps_b */ nullptr,
        up_exps_s,
        gate_exps_s,
        down_exps_s
    );
}

ggml_tensor * llm_graph_context::build_moe_ffn(
         ggml_tensor * cur,
         ggml_tensor * gate_inp,
         ggml_tensor * gate_inp_b,
         ggml_tensor * up_exps,
         ggml_tensor * up_exps_b,
         ggml_tensor * gate_exps,
         ggml_tensor * gate_exps_b,
         ggml_tensor * down_exps,
         ggml_tensor * down_exps_b,
         ggml_tensor * exp_probs_b,
             int64_t   n_expert,
             int64_t   n_expert_used,
     llm_ffn_op_type   type_op,
                bool   norm_w,
               float   w_scale,
        llama_expert_gating_func_type gating_op,
                 int   il,
         ggml_tensor * probs_in,
         ggml_tensor * gate_up_exps,
         ggml_tensor * gate_up_exps_b,
         ggml_tensor * up_exps_s,
         ggml_tensor * gate_exps_s,
         ggml_tensor * down_exps_s) const {
    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const bool weight_before_ffn = arch == LLM_ARCH_LLAMA4; // for llama4, we apply the sigmoid-ed weights before the FFN

    ggml_tensor * logits = nullptr;

    if (probs_in == nullptr) {
        logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]
        cb(logits, "ffn_moe_logits", il);
    } else {
        logits = probs_in;
    }

    if (gate_inp_b) {
        logits = ggml_add(ctx0, logits, gate_inp_b);
        cb(logits, "ffn_moe_logits_biased", il);
    }

    ggml_tensor * probs = nullptr;
    switch (gating_op) {
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
            {
                probs = ggml_soft_max(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
            {
                probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
            } break;
        case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT:
            {
                probs = logits; // [n_expert, n_tokens]
            } break;
        default:
            GGML_ABORT("fatal error");
    }
    cb(probs, "ffn_moe_probs", il);

    // add experts selection bias - introduced in DeepSeek V3
    // leave probs unbiased as it's later used to get expert weights
    ggml_tensor * selection_probs = probs;
    if (exp_probs_b != nullptr) {
        selection_probs = ggml_add(ctx0, probs, exp_probs_b);
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // llama4 doesn't have exp_probs_b, and sigmoid is only used after top_k
    // see: https://github.com/meta-llama/llama-models/blob/699a02993512fb36936b1b0741e13c06790bcf98/models/llama4/moe.py#L183-L198
    if (arch == LLM_ARCH_LLAMA4) {
        selection_probs = logits;
    }

    if (arch == LLM_ARCH_GROVEMOE) {
        selection_probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_biased", il);
    }

    // select top n_group_used expert groups
    // https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/modeling_deepseek.py#L440-L457
    if (hparams.n_expert_groups > 1 && n_tokens > 0) {
        const int64_t n_exp_per_group = n_expert / hparams.n_expert_groups;

        // organize experts into n_expert_groups
        ggml_tensor * selection_groups = ggml_reshape_3d(ctx0, selection_probs, n_exp_per_group, hparams.n_expert_groups, n_tokens); // [n_exp_per_group, n_expert_groups, n_tokens]

        ggml_tensor * group_scores = ggml_argsort_top_k(ctx0, selection_groups, 2); // [2, n_expert_groups, n_tokens]
        group_scores = ggml_get_rows(ctx0, ggml_reshape_4d(ctx0, selection_groups, 1, selection_groups->ne[0], selection_groups->ne[1], selection_groups->ne[2]), group_scores); // [1, 2, n_expert_groups, n_tokens]

        // get top n_group_used expert groups
        group_scores = ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2], group_scores->ne[3])); // [1, n_expert_groups, n_tokens]
        group_scores = ggml_reshape_2d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2]); // [n_expert_groups, n_tokens]

        ggml_tensor * expert_groups = ggml_argsort_top_k(ctx0, group_scores, hparams.n_group_used); // [n_group_used, n_tokens]
        cb(expert_groups, "ffn_moe_group_topk", il);

        // mask out the other groups
        selection_probs = ggml_get_rows(ctx0, selection_groups, expert_groups); // [n_exp_per_group, n_group_used, n_tokens]
        selection_probs = ggml_set_rows(ctx0, ggml_fill(ctx0, selection_groups, -INFINITY), selection_probs, expert_groups); // [n_exp_per_group, n_expert_groups, n_tokens]
        selection_probs = ggml_reshape_2d(ctx0, selection_probs, n_expert, n_tokens); // [n_expert, n_tokens]
        cb(selection_probs, "ffn_moe_probs_masked", il);
    }

    // select experts
    ggml_tensor * selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
    cb(selected_experts->src[0], "ffn_moe_argsort", il);
    cb(selected_experts, "ffn_moe_topk", il);

    if (arch == LLM_ARCH_GROVEMOE && n_expert != hparams.n_expert) {
        // TODO: Use scalar div instead when/if implemented
        ggml_tensor * f_sel = ggml_cast(ctx0, selected_experts, GGML_TYPE_F32);
        selected_experts = ggml_cast(ctx0, ggml_scale(ctx0, f_sel, 1.0f / float(hparams.n_group_experts)), GGML_TYPE_I32);
        probs = ggml_reshape_3d(ctx0, probs, 1, hparams.n_expert, n_tokens);
    } else {
        probs = ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens);
    }

    ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected_experts); // [1, n_expert_used, n_tokens]
    cb(weights, "ffn_moe_weights", il);


    if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
        weights = ggml_soft_max(ctx0, weights); // [n_expert_used, n_tokens]
        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
        cb(weights, "ffn_moe_weights_softmax", il);
    }

    if (norm_w) {
        weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);

        ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);

        // Avoid division by zero, clamp to smallest number representable by F16
        weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
        cb(weights_sum, "ffn_moe_weights_sum_clamped", il);

        weights = ggml_div(ctx0, weights, weights_sum); // [n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
    }
    if (w_scale != 0.0f && w_scale != 1.0f) {
        weights = ggml_scale(ctx0, weights, w_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }

    //call early so that topk-moe can be used
    ggml_build_forward_expand(gf, weights);

    cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);

    if (weight_before_ffn) {
        // repeat cur to [n_embd, n_expert_used, n_tokens]
        ggml_tensor * repeated = ggml_repeat_4d(ctx0, cur, n_embd, n_expert_used, n_tokens, 1);
        cur = ggml_mul(ctx0, repeated, weights);
        cb(cur, "ffn_moe_weighted", il);
    }

    ggml_tensor * up = nullptr;
    ggml_tensor * experts = nullptr;

    if (gate_up_exps) {
        // merged gate_up path: one mul_mat_id, then split into gate and up views
        ggml_tensor * gate_up = build_lora_mm_id(gate_up_exps, cur, selected_experts); // [n_ff*2, n_expert_used, n_tokens]
        cb(gate_up, "ffn_moe_gate_up", il);

        if (gate_up_exps_b) {
            gate_up = ggml_add_id(ctx0, gate_up, gate_up_exps_b, selected_experts);
            cb(gate_up, "ffn_moe_gate_up_biased", il);
        }

        // apply per-expert scale2 to merged gate_up (use up_exps_s since gate and up are fused)
        if (up_exps_s) {
            ggml_tensor * s = ggml_reshape_3d(ctx0, up_exps_s, 1, n_expert, 1);
            s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
            s = ggml_get_rows(ctx0, s, selected_experts); // [1, n_expert_used, n_tokens]
            gate_up = ggml_mul(ctx0, gate_up, s);
            cb(gate_up, "ffn_moe_gate_up_scaled", il);
        }

        const int64_t n_ff = gate_up->ne[0] / 2;
        cur = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
        cb(cur, "ffn_moe_gate", il);
        up  = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], n_ff * gate_up->nb[0]);
        cb(up, "ffn_moe_up", il);
    } else {
        // separate gate and up path
        up = build_lora_mm_id(up_exps, cur, selected_experts); // [n_ff, n_expert_used, n_tokens]
        cb(up, "ffn_moe_up", il);

        if (up_exps_b) {
            up = ggml_add_id(ctx0, up, up_exps_b, selected_experts);
            cb(up, "ffn_moe_up_biased", il);
        }

        // apply per-expert scale2 to up
        if (up_exps_s) {
            ggml_tensor * s = ggml_reshape_3d(ctx0, up_exps_s, 1, n_expert, 1);
            s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
            s = ggml_get_rows(ctx0, s, selected_experts); // [1, n_expert_used, n_tokens]
            up = ggml_mul(ctx0, up, s);
            cb(up, "ffn_moe_up_scaled", il);
        }

        if (gate_exps) {
            cur = build_lora_mm_id(gate_exps, cur, selected_experts); // [n_ff, n_expert_used, n_tokens]
            cb(cur, "ffn_moe_gate", il);
        } else {
            cur = up;
        }

        if (gate_exps_b) {
            cur = ggml_add_id(ctx0, cur, gate_exps_b, selected_experts);
            cb(cur, "ffn_moe_gate_biased", il);
        }

        // apply per-expert scale2 to gate
        if (gate_exps_s) {
            ggml_tensor * s = ggml_reshape_3d(ctx0, gate_exps_s, 1, n_expert, 1);
            s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
            s = ggml_get_rows(ctx0, s, selected_experts); // [1, n_expert_used, n_tokens]
            cur = ggml_mul(ctx0, cur, s);
            cb(cur, "ffn_moe_gate_scaled", il);
        }
    }

    const bool has_gate = gate_exps || gate_up_exps;

    switch (type_op) {
        case LLM_FFN_SILU:
            if (gate_exps) {
                // Step35: per-layer clamp for routed experts
                if (arch == LLM_ARCH_STEP35 && il >= 0) {
                    const float limit = hparams.swiglu_clamp_exp[il];
                    constexpr float eps = 1e-6f;
                    if (limit > eps) {
                        ggml_tensor * gate_act = ggml_silu(ctx0, cur);
                        cb(gate_act, "ffn_moe_silu", il);
                        gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
                        cb(gate_act, "ffn_moe_silu_clamped", il);

                        up = ggml_clamp(ctx0, up, -limit, limit);
                        cb(up, "ffn_moe_up_clamped", il);

                        cur = ggml_mul(ctx0, gate_act, up);
                        cb(cur, "ffn_moe_swiglu_limited", il);
                        break;
                    }
                }
            }

            if (has_gate) {
                cur = ggml_swiglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_swiglu", il);
            } else {
                cur = ggml_silu(ctx0, cur);
                cb(cur, "ffn_moe_silu", il);
            } break;
        case LLM_FFN_GELU:
            if (has_gate) {
                cur = ggml_geglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_geglu", il);
            } else {
                cur = ggml_gelu(ctx0, cur);
                cb(cur, "ffn_moe_gelu", il);
            } break;
        case LLM_FFN_SWIGLU_OAI_MOE:
            {
                // TODO: move to hparams?
                constexpr float alpha = 1.702f;
                constexpr float limit = 7.0f;
                cur = ggml_swiglu_oai(ctx0, cur, up, alpha, limit);
                cb(cur, "ffn_moe_swiglu_oai", il);
            } break;
        case LLM_FFN_RELU:
            if (has_gate) {
                cur = ggml_reglu_split(ctx0, cur, up);
                cb(cur, "ffn_moe_reglu", il);
            } else {
                cur = ggml_relu(ctx0, cur);
                cb(cur, "ffn_moe_relu", il);
            } break;
        case LLM_FFN_RELU_SQR:
            if (has_gate) {
                // TODO: add support for gated squared relu
                GGML_ABORT("fatal error: gated squared relu not implemented");
            } else {
                cur = ggml_relu(ctx0, cur);
                cur = ggml_sqr(ctx0, cur);
                cb(cur, "ffn_moe_relu_sqr", il);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    experts = build_lora_mm_id(down_exps, cur, selected_experts); // [n_embd, n_expert_used, n_tokens]
    cb(experts, "ffn_moe_down", il);

    if (down_exps_b) {
        experts = ggml_add_id(ctx0, experts, down_exps_b, selected_experts);
        cb(experts, "ffn_moe_down_biased", il);
    }

    // apply per-expert scale2 to down
    if (down_exps_s) {
        ggml_tensor * s = ggml_reshape_3d(ctx0, down_exps_s, 1, n_expert, 1);
        s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
        s = ggml_get_rows(ctx0, s, selected_experts); // [1, n_expert_used, n_tokens]
        experts = ggml_mul(ctx0, experts, s);
        cb(experts, "ffn_moe_down_scaled", il);
    }

    if (!weight_before_ffn) {
        experts = ggml_mul(ctx0, experts, weights);
        cb(experts, "ffn_moe_weighted", il);
    }

    ggml_build_forward_expand(gf, experts);

    ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };

    assert(n_expert_used > 0);

    // order the views before the adds
    for (uint32_t i = 0; i < hparams.n_expert_used; ++i) {
        cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);

        ggml_build_forward_expand(gf, cur_experts[i]);
    }

    // aggregate experts
    // note: here we explicitly use hparams.n_expert_used instead of n_expert_used
    //       to avoid potentially a large number of add nodes during warmup
    //       ref: https://github.com/ggml-org/llama.cpp/pull/14753
    ggml_tensor * moe_out = cur_experts[0];

    for (uint32_t i = 1; i < hparams.n_expert_used; ++i) {
        moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);

        ggml_build_forward_expand(gf, moe_out);
    }

    if (hparams.n_expert_used == 1) {
        // avoid returning a non-contiguous tensor
        moe_out = ggml_cont(ctx0, moe_out);
    }

    cb(moe_out, "ffn_moe_out", il);

    return moe_out;
}

// input embeddings with optional lora
ggml_tensor * llm_graph_context::build_inp_embd(ggml_tensor * tok_embd) const {
    const int64_t n_embd_inp = hparams.n_embd_inp();
    const int64_t n_embd     = hparams.n_embd;

    assert(n_embd_inp >= n_embd);

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
    cb(inp->tokens, "inp_tokens", -1);
    ggml_set_input(inp->tokens);
    res->t_inp_tokens = inp->tokens;

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, ubatch.n_tokens);
    cb(inp->embd, "inp_embd", -1);
    ggml_set_input(inp->embd);

    // select one of the 2 inputs, based on the batch contents
    // ref: https://github.com/ggml-org/llama.cpp/pull/18550
    std::array<ggml_tensor *, 2> inps;

    // token embeddings path (ubatch.token != nullptr)
    {
        auto & cur = inps[0];

        cur = ggml_get_rows(ctx0, tok_embd, inp->tokens);

        // apply lora for embedding tokens if needed
        for (const auto & lora : *loras) {
            llama_adapter_lora_weight * lw = lora.first->get_weight(tok_embd);
            if (lw == nullptr) {
                continue;
            }

            const float adapter_scale = lora.second;
            const float scale = lw->get_scale(lora.first->alpha, adapter_scale);

            ggml_tensor * inpL_delta = ggml_scale(ctx0, ggml_mul_mat(
                        ctx0, lw->b, // non-transposed lora_b
                        ggml_get_rows(ctx0, lw->a, inp->tokens)
                        ), scale);

            cur = ggml_add(ctx0, cur, inpL_delta);
        }

        if (n_embd_inp != n_embd) {
            cur = ggml_pad(ctx0, cur, hparams.n_embd_inp() - n_embd, 0, 0, 0);
        }
    }

    // vector embeddings path (ubatch.embd != nullptr)
    {
        auto & cur = inps[1];

        cur = inp->embd;
    }

    assert(ggml_are_same_shape (inps[0], inps[1]));
    assert(ggml_are_same_stride(inps[0], inps[1]));

    ggml_tensor * cur = ggml_build_forward_select(gf, inps.data(), inps.size(), ubatch.token ? 0 : 1);

    if (n_embd_inp != n_embd) {
        cur = ggml_view_2d(ctx0, cur, n_embd, n_tokens, cur->nb[1], 0);
    }

    res->t_inp_embd = cur;

    // For Granite architecture
    // NOTE: Only apply scale to token inputs. Raw embeddings are assumed to be
    //  multimodal inputs that should not be scaled.
    if (ubatch.token && hparams.f_embedding_scale != 0.0f) {
        if (!ggml_is_contiguous(cur)) {
            cur = ggml_cont(ctx0, cur);
        }
        cur = ggml_scale(ctx0, cur, hparams.f_embedding_scale);
    }

    cb(cur, "embd", -1);

    res->add_input(std::move(inp));

    // make sure the produced embeddings are immediately materialized in the ggml graph
    // ref: https://github.com/ggml-org/llama.cpp/pull/18599
    ggml_build_forward_expand(gf, cur);

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos() const {
    auto inp = std::make_unique<llm_graph_input_pos>(hparams.n_pos_per_embd());

    auto & cur = inp->pos;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t)n_tokens*hparams.n_pos_per_embd());
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_attn_scale() const {
    auto inp = std::make_unique<llm_graph_input_attn_temp>(hparams.n_attn_temp_floor_scale, hparams.f_attn_temp_scale, hparams.f_attn_temp_offset);

    auto & cur = inp->attn_scale;

    // this need to be 1x1xN for broadcasting
    cur = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, n_tokens);
    ggml_set_input(cur);
    ggml_set_name(cur, "attn_scale");

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_out_ids() const {
    // note: when all tokens are output, we could skip this optimization to spare the ggml_get_rows() calls,
    //       but this would make the graph topology depend on the number of output tokens, which can interfere with
    //       features that require constant topology such as pipeline parallelism
    //       ref: https://github.com/ggml-org/llama.cpp/pull/14275#issuecomment-2987424471
    //if (n_outputs < n_tokens) {
    //    return nullptr;
    //}

    auto inp = std::make_unique<llm_graph_input_out_ids>(hparams, cparams, n_outputs);

    auto & cur = inp->out_ids;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_outputs);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_mean() const {
    auto inp = std::make_unique<llm_graph_input_mean>(cparams);

    auto & cur = inp->mean;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, ubatch.n_seqs_unq);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cls() const {
    auto inp = std::make_unique<llm_graph_input_cls>(cparams, arch);

    auto & cur = inp->cls;

    cur = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_seqs_unq);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_cross_embd() const {
    auto inp = std::make_unique<llm_graph_input_cross_embd>(cross);

    auto & cur = inp->cross_embd;

    // if we have the output embeddings from the encoder, use them directly
    // TODO: needs more work to be correct, for now just use the tensor shape
    //if (cross->t_embd) {
    //    cur = ggml_view_tensor(ctx0, cross->t_embd);

    //    return cur;
    //}

    const auto n_embd = !cross->v_embd.empty() ? cross->n_embd : hparams.n_embd_inp();
    const auto n_enc  = !cross->v_embd.empty() ? cross->n_enc  : hparams.n_ctx_train;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_enc);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_enc() const {
    auto inp = std::make_unique<llm_graph_input_pos_bucket>(hparams);

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_tokens);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_inp_pos_bucket_dec() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_pos_bucket_kv>(hparams, mctx_cur);

    const auto n_kv = mctx_cur->get_n_kv();

    auto & cur = inp->pos_bucket;

    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_tokens);
    ggml_set_input(cur);

    res->add_input(std::move(inp));

    return cur;
}

ggml_tensor * llm_graph_context::build_pos_bias(ggml_tensor * pos_bucket, ggml_tensor * attn_rel_b) const {
    ggml_tensor * pos_bucket_1d = ggml_reshape_1d(ctx0, pos_bucket, pos_bucket->ne[0] * pos_bucket->ne[1]);
    cb(pos_bucket_1d, "pos_bucket_1d", -1);

    ggml_tensor * pos_bias = ggml_get_rows(ctx0, attn_rel_b, pos_bucket_1d);

    pos_bias = ggml_reshape_3d(ctx0, pos_bias, pos_bias->ne[0], pos_bucket->ne[0], pos_bucket->ne[1]);
    pos_bias = ggml_permute   (ctx0, pos_bias, 2, 0, 1, 3);
    pos_bias = ggml_cont      (ctx0, pos_bias);

    cb(pos_bias, "pos_bias", -1);

    return pos_bias;
}

ggml_tensor * llm_graph_context::build_attn_mha(
         ggml_tensor * q,
         ggml_tensor * k,
         ggml_tensor * v,
         ggml_tensor * kq_b,
         ggml_tensor * kq_mask,
         ggml_tensor * sinks,
         ggml_tensor * v_mla,
               float   kq_scale,
                 int   il) const {
    const bool v_trans = v->nb[1] > v->nb[2];

    // split the batch into streams if needed
    const auto n_stream = k->ne[3];

    q = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream, q->nb[1], q->nb[2], q->nb[3]/n_stream, 0);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);

    // in the flash-attn path. The VEC kernel bug (wrong Q/K stride in

    ggml_tensor * cur;

    const bool use_flash_attn = cparams.flash_attn && kq_b == nullptr;
    if (use_flash_attn) {
        GGML_ASSERT(kq_b == nullptr && "Flash attention does not support KQ bias yet");

        if (v_trans) {
            v = ggml_transpose(ctx0, v);
        }

        // this can happen when KV cache is not used (e.g. an embedding model with non-causal attn)
        if (k->type == GGML_TYPE_F32) {
            k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        }

        if (v->type == GGML_TYPE_F32) {
            v = ggml_cast(ctx0, v, GGML_TYPE_F16);
        }

        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias,
                                  hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        cb(cur, LLAMA_TENSOR_NAME_FATTN, il);

        ggml_flash_attn_ext_add_sinks(cur, sinks);
        ggml_flash_attn_ext_set_prec (cur, GGML_PREC_F32);

        // For MLA, V is a view of K with different ne[0] (e.g. V=512, K=576).
        // Group size must come from K (which determines the WHT rotation), not V.

        if (v_mla) {
#if 0
            // v_mla can be applied as a matrix-vector multiplication with broadcasting across dimension 3 == n_tokens.
            // However, the code is optimized for dimensions 0 and 1 being large, so this is inefficient.
            cur = ggml_reshape_4d(ctx0, cur, v_mla->ne[0], 1, n_head, n_tokens);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
#else
            // It's preferable to do the calculation as a matrix-matrix multiplication with n_tokens in dimension 1.
            // The permutations are noops and only change how the tensor data is interpreted.
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
            cb(cur, "fattn_mla", il);
            cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
            cur = ggml_cont(ctx0, cur); // Needed because ggml_reshape_2d expects contiguous inputs.
#endif
        }

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        cb(kq, "kq", il);

        // note: this op tends to require high floating point range
        //       while for some models F16 is enough, for others it is not, so we default to F32 here
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

        if (arch == LLM_ARCH_GROK) {
            // need to do the following:
            // multiply by attn_output_multiplier
            // and then :
            // kq = 30 * tanh(kq / 30)
            // before the softmax below

            kq = ggml_tanh(ctx0, ggml_scale(ctx0, kq, hparams.f_attn_out_scale / hparams.f_attn_logit_softcapping));
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled", il);
        }

        if (hparams.attn_soft_cap) {
            kq = ggml_scale(ctx0, kq, 1.0f / hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_1", il);
            kq = ggml_tanh (ctx0, kq);
            cb(kq, "kq_tanh", il);
            kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
            cb(kq, "kq_scaled_2", il);
        }

        if (kq_b) {
            kq = ggml_add(ctx0, kq, kq_b);
            cb(kq, "kq_plus_kq_b", il);
        }

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        ggml_soft_max_add_sinks(kq, sinks);
        cb(kq, "kq_soft_max", il);

        if (!v_trans) {
            // note: avoid this branch
            v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
            cb(v, "v_cont", il);
        }

        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        cb(kqv, "kqv", il);


        // for MLA with the absorption optimization, we need to "decompress" from MQA back to MHA
        if (v_mla) {
            kqv = ggml_mul_mat(ctx0, v_mla, kqv);
            cb(kqv, "kqv_mla", il);
        }

        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);

        // recombine streams
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);

        if (!cparams.offload_kqv) {
            // all nodes between the KV store and the attention output are run on the CPU
            ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
        }
    }


    ggml_build_forward_expand(gf, cur);

    return cur;
}

llm_graph_input_attn_no_cache * llm_graph_context::build_attn_inp_no_cache() const {
    auto inp = std::make_unique<llm_graph_input_attn_no_cache>(hparams, cparams);

    // flash attention requires an f16 mask
    const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    // note: there is no KV cache, so the number of KV values is equal to the number of tokens in the batch
    inp->self_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_tokens, n_tokens, 1, 1);
    ggml_set_input(inp->self_kq_mask);

    inp->self_kq_mask_cnv = inp->self_kq_mask;

    if (hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        inp->self_kq_mask_swa = ggml_new_tensor_4d(ctx0, type_mask, n_tokens, n_tokens, 1, 1);
        ggml_set_input(inp->self_kq_mask_swa);

        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
    } else {
        inp->self_kq_mask_swa     = nullptr;
        inp->self_kq_mask_swa_cnv = nullptr;
    }

    return (llm_graph_input_attn_no_cache *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_no_cache * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    GGML_UNUSED(n_tokens);

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const bool is_swa = hparams.is_swa(il);

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    // [TAG_NO_CACHE_PAD]
    // TODO: if ubatch.equal_seqs() == true, we can split the three tensors below into ubatch.n_seqs_unq streams
    //       but it might not be worth it: https://github.com/ggml-org/llama.cpp/pull/15636
    //assert(!ubatch.equal_seqs() || (k_cur->ne[3] == 1 && k_cur->ne[3] == ubatch.n_seqs_unq));

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
        cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
        cb(cur, "kqv_wo_b", il);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_kv> build_attn_inp_kv_impl(
    ggml_context * ctx0,
    const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_kv>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur, ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    inp->self_k_rot = mctx_cur->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_cur->build_input_v_rot(ctx0);

    return inp;
}

static std::unique_ptr<llm_graph_input_attn_kv> build_attn_inp_kvarn_impl(
    ggml_context * ctx0,
    const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_kvarn_context * mctx_kvarn) {
    auto inp = std::make_unique<llm_graph_input_attn_kvarn>(hparams, cparams, mctx_kvarn);
    inp->sink_tail_idxs = mctx_kvarn->build_input_sink_tail_idxs(ctx0, ubatch);
    inp->body_plan = mctx_kvarn->build_input_body_plan(ctx0, ubatch);
    inp->body_offsets = mctx_kvarn->build_input_body_offsets(ctx0, ubatch);
    inp->tail_evict_idxs = mctx_kvarn->build_input_tail_evict_idxs(ctx0, ubatch);
    const kvarn_active_window mask_window = kvarn_graph_active_window(cparams.kvarn, ubatch, mctx_kvarn->get_size());
    const uint32_t mask_n_kv = kvarn_graph_mask_n_kv(mask_window, mctx_kvarn->get_size(), ubatch);
    inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mask_n_kv, ubatch, cparams);
    inp->self_kq_mask_cnv = inp->self_kq_mask;
    return inp;
}

static std::unique_ptr<llm_graph_input_attn_kv> build_attn_inp_kvarn_filter_impl(
    ggml_context * ctx0,
    const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_normal,
    const llama_kv_cache_kvarn_context * mctx_kvarn) {
    auto inp_normal = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_normal);
    auto inp_kvarn  = build_attn_inp_kvarn_impl(ctx0, ubatch, hparams, cparams, mctx_kvarn);
    return std::make_unique<llm_graph_input_attn_kvarn_filter>(
            hparams, cparams, std::move(inp_normal), std::move(inp_kvarn));
}

llm_graph_input_attn_kv * llm_graph_context::build_attn_inp_kv() const {
    if (const auto * mctx_kvarn = dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx)) {
        auto inp = build_attn_inp_kvarn_impl(ctx0, ubatch, hparams, cparams, mctx_kvarn);
        return (llm_graph_input_attn_kv *) res->add_input(std::move(inp));
    }

    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_kv *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla, // TODO: remove
            float     kq_scale,
            int       il) const {
    GGML_ASSERT(v_mla == nullptr);

    if (auto * inp_filter = dynamic_cast<llm_graph_input_attn_kvarn_filter *>(inp)) {
        return build_attn(inp_filter->input_for_layer(il),
                wo, wo_b, wo_s, q_cur, k_cur, v_cur, kq_b, sinks, v_mla, kq_scale, il);
    }

    if (auto * inp_kvarn = dynamic_cast<llm_graph_input_attn_kvarn *>(inp)) {
        ggml_build_forward_expand(gf, q_cur);
        if ((k_cur == nullptr) != (v_cur == nullptr)) {
            throw std::runtime_error("KVarN graph backend requires K and V cache writes to be paired");
        }

        const bool stores_kv = k_cur != nullptr;
        if (stores_kv) {
            ggml_build_forward_expand(gf, v_cur);
            ggml_build_forward_expand(gf, k_cur);
        }

        // Runtime KVarN memory splits work into one-token ubatches. Context
        // initialization can still reserve a larger worst-case graph; building
        // that graph is safe because it is not executed for prompt chunks.
        if (inp->self_k_rot || inp->self_v_rot || kq_b || sinks) {
            throw std::runtime_error(
                    "KVarN graph backend does not yet support attention rotations, KQ bias, or attention sinks");
        }

        const auto & idxs = inp_kvarn->get_sink_tail_idxs();
        const llama_kvarn_layer_view layer = inp_kvarn->mctx_kvarn->get_layer_view(il);
        const bool kvarn_paper_frame = kvarn_graph_parse_env_flag("LLAMA_KVARN_ENABLE_PAPER_FRAME");
        if (kvarn_paper_frame) {
            (void) kvarn_graph_paper_mixed_frame_enabled();
        }
        ggml_tensor * kvarn_H = nullptr;
        if (kvarn_paper_frame) {
            if (layer.head_dim_k != layer.head_dim_v) {
                throw std::runtime_error("KVarN paper-frame path requires equal K/V head dimensions");
            }
            kvarn_H = kvarn_graph_build_hadamard_input(
                    ctx0, layer.head_dim_k,
                    inp_kvarn->kvarn_hadamard,
                    inp_kvarn->kvarn_hadamard_host,
                    inp_kvarn->kvarn_hadamard_filled,
                    inp_kvarn->kvarn_hadamard_dim,
                    "kvarn_hadamard");
        }
        GGML_ASSERT(layer.body_k != nullptr);
        GGML_ASSERT(layer.body_v != nullptr);
        GGML_ASSERT(layer.scales_k != nullptr);
        GGML_ASSERT(layer.scales_v != nullptr);
        GGML_ASSERT(layer.layout_k.key_bits >= 2 && layer.layout_k.key_bits <= 8);
        GGML_ASSERT(layer.layout_v.value_bits >= 2 && layer.layout_v.value_bits <= 8);
        GGML_ASSERT(inp_kvarn->mctx_kvarn->body_store_scratch_floats(il) > 0);
        ggml_tensor * sink_tail_k_src = layer.sink_tail_k;
        ggml_tensor * sink_tail_v_src = layer.sink_tail_v;
        ggml_tensor * pending_k_src   = layer.pending_k;
        ggml_tensor * pending_v_src   = layer.pending_v;
        bool contiguous_prefill_chunk = stores_kv &&
            !kvarn_graph_prefill_direct_store_disabled() &&
            ubatch.pos != nullptr &&
            ubatch.n_tokens > 1 &&
            q_cur != nullptr && k_cur != nullptr && v_cur != nullptr &&
            q_cur->ne[2] == int64_t(ubatch.n_tokens) &&
            k_cur->ne[2] == int64_t(ubatch.n_tokens) &&
            v_cur->ne[2] == int64_t(ubatch.n_tokens);
        if (contiguous_prefill_chunk) {
            const llama_pos chunk_start = ubatch.pos[0];
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                if (ubatch.pos[i] != chunk_start + llama_pos(i)) {
                    contiguous_prefill_chunk = false;
                    break;
                }
            }
        }
        if (stores_kv) {
            const std::vector<uint32_t> seal_records = kvarn_graph_seal_records(cparams.kvarn, ubatch, inp_kvarn->mctx_kvarn->get_size());
            bool direct_prefill_store = contiguous_prefill_chunk && !seal_records.empty();
            if (direct_prefill_store) {
                const kvarn_active_window final_window =
                    kvarn_graph_active_window(cparams.kvarn, ubatch, inp_kvarn->mctx_kvarn->get_size());
                if (!final_window.valid || final_window.n_pending != 0) {
                    direct_prefill_store = false;
                }
            }
            if (direct_prefill_store) {
                const llama_pos chunk_start = ubatch.pos[0];
                const llama_pos chunk_end = chunk_start + llama_pos(ubatch.n_tokens);
                for (const uint32_t record : seal_records) {
                    const llama_pos record_start =
                        llama_pos(cparams.kvarn.sink_tokens) + llama_pos(record)*llama_pos(cparams.kvarn.group_size);
                    const llama_pos record_end = record_start + llama_pos(cparams.kvarn.group_size);
                    if (record_start < chunk_start || record_end > chunk_end) {
                        direct_prefill_store = false;
                        break;
                    }
                }
            }
            if (direct_prefill_store) {
                const llama_pos chunk_start = ubatch.pos[0];
                const llama_pos chunk_end = chunk_start + llama_pos(ubatch.n_tokens);
                for (const uint32_t record : seal_records) {
                    const llama_pos record_start = llama_pos(cparams.kvarn.sink_tokens + record*cparams.kvarn.group_size);
                    const llama_pos record_end = record_start + llama_pos(cparams.kvarn.group_size);
                    if (record >= layer.n_records || record_start < chunk_start || record_end > chunk_end) {
                        direct_prefill_store = false;
                        break;
                    }
                }
            }
            ggml_build_forward_expand(gf, inp_kvarn->get_body_plan());
            ggml_build_forward_expand(gf, inp_kvarn->get_body_offsets());
            ggml_build_forward_expand(gf, inp_kvarn->get_tail_evict_idxs());
            ggml_tensor * pending_k_copy = nullptr;
            ggml_tensor * pending_v_copy = nullptr;
            if (!direct_prefill_store && seal_records.empty() && inp_kvarn->get_body_offsets()->ne[0] > 0) {
                pending_k_copy = inp_kvarn->mctx_kvarn->cpy_tail_evict_pending_k(
                            ctx0, inp_kvarn->get_tail_evict_idxs(), inp_kvarn->get_body_offsets(), il);
                pending_v_copy = inp_kvarn->mctx_kvarn->cpy_tail_evict_pending_v(
                            ctx0, inp_kvarn->get_tail_evict_idxs(), inp_kvarn->get_body_offsets(), il);
                ggml_build_forward_expand(gf, pending_k_copy);
                ggml_build_forward_expand(gf, pending_v_copy);
                pending_k_src = ggml_reshape_3d(
                        ctx0, pending_k_copy, layer.head_dim_k, layer.n_head_kv, cparams.kvarn.group_size);
                pending_v_src = ggml_reshape_3d(
                        ctx0, pending_v_copy, layer.head_dim_v, layer.n_head_kv, cparams.kvarn.group_size);
            }

            inp_kvarn->has_body_store_ops = !seal_records.empty();
            inp_kvarn->baked_seal_records = seal_records;

            ggml_tensor * body_store_scratch = nullptr;
            if (!seal_records.empty() || direct_prefill_store) {
                body_store_scratch = inp_kvarn->mctx_kvarn->build_body_store_scratch(ctx0, il);
            }

            ggml_tensor * last_body_store = nullptr;
            const auto build_body_store = [&](ggml_tensor * store) {
                if (last_body_store != nullptr) {
                    ggml_kvarn_store_kv_body_add_dep(store, last_body_store);
                }
                ggml_build_forward_expand(gf, store);
                last_body_store = store;
            };
            const auto copy_pending_slice_for_record = [&](uint32_t record) {
                const kvarn_tail_evict_slice slice = kvarn_graph_tail_evict_slice_for_record(
                        cparams.kvarn, ubatch, inp_kvarn->mctx_kvarn->get_size(), record);
                if (slice.count <= 0) {
                    throw std::runtime_error("KVarN graph backend could not find evicted pending rows for sealed body record");
                }
                if (!slice.contiguous) {
                    throw std::runtime_error("KVarN graph backend cannot seal non-contiguous pending rows for a body record");
                }

                ggml_tensor * tail_idxs = ggml_view_1d(
                        ctx0, inp_kvarn->get_tail_evict_idxs(), slice.count,
                        size_t(slice.start)*inp_kvarn->get_tail_evict_idxs()->nb[0]);
                ggml_tensor * offsets = ggml_view_1d(
                        ctx0, inp_kvarn->get_body_offsets(), slice.count,
                        size_t(slice.start)*inp_kvarn->get_body_offsets()->nb[0]);
                pending_k_copy = inp_kvarn->mctx_kvarn->cpy_tail_evict_pending_k(ctx0, tail_idxs, offsets, il);
                pending_v_copy = inp_kvarn->mctx_kvarn->cpy_tail_evict_pending_v(ctx0, tail_idxs, offsets, il);
                if (last_body_store != nullptr) {
                    ggml_set_rows_add_dep(pending_k_copy, last_body_store);
                    ggml_set_rows_add_dep(pending_v_copy, last_body_store);
                }
                ggml_build_forward_expand(gf, pending_k_copy);
                ggml_build_forward_expand(gf, pending_v_copy);
                pending_k_src = ggml_reshape_3d(
                        ctx0, pending_k_copy, layer.head_dim_k, layer.n_head_kv, cparams.kvarn.group_size);
                pending_v_src = ggml_reshape_3d(
                        ctx0, pending_v_copy, layer.head_dim_v, layer.n_head_kv, cparams.kvarn.group_size);
            };

            if (direct_prefill_store) {
                constexpr uint32_t direct_record_batch_max = 8;
                const bool use_direct_record_batch =
                    kvarn_paper_frame || kvarn_graph_parse_env_flag("LLAMA_KVARN_ENABLE_DIRECT_RECORD_BATCH");
                if (kvarn_graph_prefill_direct_trace_enabled() && kvarn_graph_prefill_direct_trace_claim()) {
                    std::fprintf(stderr,
                            "KVarN graph prefill-direct-store trace: layer=%d head_dim=%u heads=%u direct_records=%u batch_enabled=%d batch_max=%u chunks=%u\n",
                            il, layer.head_dim_k, layer.n_head_kv, uint32_t(seal_records.size()),
                            use_direct_record_batch ? 1 : 0, direct_record_batch_max,
                            (uint32_t(seal_records.size()) + direct_record_batch_max - 1)/direct_record_batch_max);
                }
                if (use_direct_record_batch) {
                    for (uint32_t i_record = 0; i_record < seal_records.size();) {
                        const uint32_t record0 = seal_records[i_record];
                        uint32_t n_batch = 1;
                        while (i_record + n_batch < seal_records.size() &&
                                n_batch < direct_record_batch_max &&
                                seal_records[i_record + n_batch] == record0 + n_batch) {
                            ++n_batch;
                        }
                        const uint32_t pos0 = cparams.kvarn.sink_tokens + record0*cparams.kvarn.group_size - uint32_t(ubatch.pos[0]);
                        ggml_tensor * k_tiles = ggml_view_4d(
                                ctx0, k_cur,
                                layer.head_dim_k, layer.n_head_kv, cparams.kvarn.group_size, n_batch,
                                k_cur->nb[1], k_cur->nb[2], size_t(cparams.kvarn.group_size)*k_cur->nb[2],
                                size_t(pos0)*k_cur->nb[2]);
                        ggml_tensor * v_tiles = ggml_view_4d(
                                ctx0, v_cur,
                                layer.head_dim_v, layer.n_head_kv, cparams.kvarn.group_size, n_batch,
                                v_cur->nb[1], v_cur->nb[2], size_t(cparams.kvarn.group_size)*v_cur->nb[2],
                                size_t(pos0)*v_cur->nb[2]);
                        build_body_store(inp_kvarn->mctx_kvarn->store_kv_body_records_all_heads(
                                    ctx0, k_tiles, v_tiles, body_store_scratch, il, record0, n_batch));
                        i_record += n_batch;
                    }
                } else {
                    for (const uint32_t record : seal_records) {
                        const uint32_t pos0 = cparams.kvarn.sink_tokens + record*cparams.kvarn.group_size - uint32_t(ubatch.pos[0]);
                        ggml_tensor * k_tile = ggml_view_3d(
                                ctx0, k_cur,
                                layer.head_dim_k, layer.n_head_kv, cparams.kvarn.group_size,
                                k_cur->nb[1], k_cur->nb[2], size_t(pos0)*k_cur->nb[2]);
                        ggml_tensor * v_tile = ggml_view_3d(
                                ctx0, v_cur,
                                layer.head_dim_v, layer.n_head_kv, cparams.kvarn.group_size,
                                v_cur->nb[1], v_cur->nb[2], size_t(pos0)*v_cur->nb[2]);
                        build_body_store(inp_kvarn->mctx_kvarn->store_kv_body_all_heads(
                                    ctx0, k_tile, v_tile, body_store_scratch, il, record));
                    }
                }
            } else {
                for (const uint32_t seal_record : seal_records) {
                    if (seal_record >= layer.n_records) {
                        throw std::runtime_error("KVarN graph backend attempted to seal a body record outside cache capacity");
                    }

                    copy_pending_slice_for_record(seal_record);
                    if (layer.head_dim_k >= 256 && layer.n_head_kv > 1) {
                        build_body_store(inp_kvarn->mctx_kvarn->store_kv_body_all_heads_from_pending(
                                    ctx0, body_store_scratch, il, seal_record, pending_k_src, pending_v_src));
                    } else {
                        for (uint32_t ih = 0; ih < layer.n_head_kv; ++ih) {
                            build_body_store(inp_kvarn->mctx_kvarn->store_kv_body_record_from_pending(
                                        ctx0, body_store_scratch, il, ih, seal_record, pending_k_src, pending_v_src));
                        }
                    }
                }
                const kvarn_active_window final_window =
                    kvarn_graph_active_window(cparams.kvarn, ubatch, inp_kvarn->mctx_kvarn->get_size());
                if (final_window.valid && final_window.n_pending > 0) {
                    copy_pending_slice_for_record(uint32_t(final_window.n_records));
                }
            }

            ggml_tensor * k_st_src = k_cur;
            ggml_tensor * v_st_src = v_cur;
            if (kvarn_paper_frame) {
                k_st_src = kvarn_graph_apply_hadamard(ctx0, kvarn_H, k_st_src);
                v_st_src = kvarn_graph_apply_hadamard(ctx0, kvarn_H, v_st_src);
            }
            ggml_tensor * sink_k_copy = inp_kvarn->mctx_kvarn->cpy_sink_tail_k(ctx0, k_st_src, idxs, il);
            ggml_tensor * sink_v_copy = inp_kvarn->mctx_kvarn->cpy_sink_tail_v(ctx0, v_st_src, idxs, il);
            if (pending_k_copy != nullptr) {
                ggml_set_rows_add_dep(sink_k_copy, pending_k_copy);
            }
            if (pending_v_copy != nullptr) {
                ggml_set_rows_add_dep(sink_v_copy, pending_v_copy);
            }
            if (last_body_store != nullptr) {
                ggml_set_rows_add_dep(sink_k_copy, last_body_store);
                ggml_set_rows_add_dep(sink_v_copy, last_body_store);
            }
            ggml_build_forward_expand(gf, sink_k_copy);
            ggml_build_forward_expand(gf, sink_v_copy);
            sink_tail_k_src = ggml_reshape_3d(
                    ctx0, sink_k_copy, layer.head_dim_k, layer.n_head_kv, layer.sink_tail_k->ne[2]);
            sink_tail_v_src = ggml_reshape_3d(
                    ctx0, sink_v_copy, layer.head_dim_v, layer.n_head_kv, layer.sink_tail_v->ne[2]);
        }

        const ggml_tensor * kq_mask = inp->get_kq_mask();
        const bool prefill_direct_attn =
            stores_kv &&
            !kvarn_graph_prefill_direct_attn_disabled() &&
            q_cur != nullptr && k_cur != nullptr && v_cur != nullptr && kq_mask != nullptr &&
            q_cur->ne[2] > 1 &&
            k_cur->ne[2] == q_cur->ne[2] &&
            v_cur->ne[2] == q_cur->ne[2] &&
            kq_mask->ne[0] == k_cur->ne[2] &&
            kq_mask->ne[1] == q_cur->ne[2];
        if (kvarn_graph_prefill_direct_trace_enabled() && kvarn_graph_prefill_direct_trace_claim()) {
            std::fprintf(stderr,
                    "KVarN graph prefill-direct trace: use=%d stores_kv=%d q_tokens=%" PRId64
                    " k_tokens=%" PRId64 " v_tokens=%" PRId64 " mask=[%" PRId64 ",%" PRId64 "]\n",
                    prefill_direct_attn ? 1 : 0, stores_kv ? 1 : 0,
                    q_cur ? q_cur->ne[2] : int64_t(-1),
                    k_cur ? k_cur->ne[2] : int64_t(-1),
                    v_cur ? v_cur->ne[2] : int64_t(-1),
                    kq_mask ? kq_mask->ne[0] : int64_t(-1),
                    kq_mask ? kq_mask->ne[1] : int64_t(-1));
        }
        if (prefill_direct_attn) {
            ggml_tensor * cur = build_attn_mha(q_cur, k_cur, v_cur, kq_b, inp->get_kq_mask(), sinks, v_mla, kq_scale, il);
            cb(cur, "kvarn_prefill_direct_kqv_out", il);

            if (wo) {
                if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
                    cur = build_lora_mm(wo, cur);
                    ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
                    if (wo_s) {
                        cur = ggml_mul(ctx0, cur, wo_s);
                    }
                } else {
                    cur = build_lora_mm(wo, cur, wo_s);
                }
                cb(cur, "kvarn_prefill_direct_kqv_wo", il);
            }

            if (wo_b) {
                cur = ggml_add(ctx0, cur, wo_b);
                cb(cur, "kvarn_prefill_direct_kqv_wo_b", il);
            }

            return cur;
        }

        const llama_pos pos = ubatch.pos ? ubatch.pos[0] : llama_pos(0);
        if (pos < 0) {
            throw std::runtime_error("KVarN graph backend cannot attend to a negative-position decode token");
        }

        const kvarn_active_window window = kvarn_graph_active_window(cparams.kvarn, ubatch, inp_kvarn->mctx_kvarn->get_size());
        if (!window.valid) {
            throw std::runtime_error("KVarN graph backend cannot build active decode attention window");
        }

        if (uint32_t(window.n_records) > layer.n_records) {
            throw std::runtime_error("KVarN graph backend active body record count exceeds allocated cache capacity");
        }

        const kvarn_active_window scratch_window = kvarn_graph_build_scratch_window(
                window, cparams.kvarn, inp_kvarn->mctx_kvarn->get_size(), ubatch);
        const int64_t scores_floats = kvarn_graph_attn_scratch_floats(
                scratch_window, layer.n_head_kv, scratch_window.n_records, layer.head_dim_k, cparams.kvarn.group_size);
        ggml_tensor * scores = inp_kvarn->mctx_kvarn->build_attn_mixed_scratch(ctx0, il, scores_floats);
        inp_kvarn->mixed_attn_scores.push_back(scores);

        const bool window_indirect = kvarn_graph_decode_stable_topology(ubatch) &&
                window.n_records == 0 && window.n_pending == 0 &&
                !inp_kvarn->has_body_store_ops;

        if (window_indirect && inp_kvarn->kvarn_window == nullptr) {
            inp_kvarn->kvarn_window = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 8);
            ggml_set_input(inp_kvarn->kvarn_window);
            ggml_set_name(inp_kvarn->kvarn_window, "kvarn_window");
            inp_kvarn->window_indirect = true;
        }

        const llama_kvarn_params effective_kvarn =
            kvarn_graph_effective_params(cparams.kvarn, inp_kvarn->mctx_kvarn->get_size());
        const int32_t op_n_sink     = window_indirect ? int32_t(effective_kvarn.sink_tokens) : int32_t(window.n_sink);
        const int32_t op_n_records  = window_indirect ? 0 : int32_t(window.n_records);
        const int32_t op_n_pending  = window_indirect ? 0 : int32_t(window.n_pending);
        const int32_t op_n_tail     = window_indirect ? int32_t(effective_kvarn.tail_tokens) : int32_t(window.n_tail);
        const int32_t op_tail_start = window_indirect ? 0 : int32_t(window.tail_start);

        const bool kvarn_fused_fwht =
            kvarn_paper_frame &&
            !kvarn_graph_parse_env_flag("LLAMA_KVARN_DISABLE_FUSED_FWHT") &&
            (q_cur->ne[2] == 1 || layer.head_dim_k >= 512) &&
            op_n_records == 0 &&
            op_n_pending == 0;

        ggml_tensor * q_in = q_cur;
        if (kvarn_paper_frame && !kvarn_fused_fwht) {
            q_in = kvarn_graph_apply_hadamard(ctx0, kvarn_H, q_in);
        }
        ggml_tensor * cur = ggml_kvarn_attn_mixed(
                ctx0, q_in, sink_tail_k_src, sink_tail_v_src, layer.body_k, layer.body_v,
                layer.scales_k, layer.scales_v, pending_k_src, pending_v_src, scores, inp->get_kq_mask(),
                op_n_sink, op_n_records, op_n_pending, op_n_tail, op_tail_start,
                int32_t(layer.head_dim_k), int32_t(cparams.kvarn.group_size),
                int32_t(layer.layout_k.key_bits), int32_t(layer.layout_v.value_bits), kq_scale,
                hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        ggml_kvarn_attn_mixed_set_v_layout(cur, int32_t(layer.layout_v.v_layout));
        if (kvarn_fused_fwht) {
            ggml_kvarn_attn_mixed_set_frame_flags(cur, GGML_KVARN_ATTN_FRAME_FUSED_PAPER_FULL);
        }
        if (window_indirect) {
            ggml_kvarn_attn_mixed_set_window(cur, inp_kvarn->kvarn_window);
        }
        inp_kvarn->mixed_attn_nodes.push_back(cur);
        cb(cur, "kvarn_kqv_out", il);
        if (kvarn_paper_frame && !kvarn_fused_fwht) {
            cur = kvarn_graph_apply_hadamard(ctx0, kvarn_H, cur);
            cb(cur, "kvarn_kqv_out_unrot", il);
        }
        cur = ggml_reshape_2d(ctx0, cur, q_cur->ne[0]*q_cur->ne[1], q_cur->ne[2]);
        cb(cur, "kvarn_kqv_out_2d", il);

        if (wo) {
            if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
                cur = build_lora_mm(wo, cur);
                ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
                if (wo_s) {
                    cur = ggml_mul(ctx0, cur, wo_s);
                }
            } else {
                cur = build_lora_mm(wo, cur, wo_s);
            }
            cb(cur, "kvarn_kqv_wo", il);
        }

        if (wo_b) {
            cur = ggml_add(ctx0, cur, wo_b);
            cb(cur, "kvarn_kqv_wo_b", il);
        }

        return cur;
    }

    if (inp->self_k_rot) {
        q_cur = ggml_mul_mat_aux(ctx0, q_cur, inp->self_k_rot);
        k_cur = ggml_mul_mat_aux(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = ggml_mul_mat_aux(ctx0, v_cur, inp->self_v_rot);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    const auto & kq_mask = inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);


    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);


    if (inp->self_v_rot) {
        cur = ggml_mul_mat_aux(ctx0, cur, inp->self_v_rot);
    }

    if (wo) {
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
            // GLM4, GLM4_MOE, and JAIS2 seem to have numerical issues with half-precision accumulators
            cur = build_lora_mm(wo, cur);
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
            if (wo_s) {
                cur = ggml_mul(ctx0, cur, wo_s);
            }
        } else {
            cur = build_lora_mm(wo, cur, wo_s);
        }
        cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
        cb(cur, "kqv_wo_b", il);
    }

    return cur;
}

static std::unique_ptr<llm_graph_input_attn_k> build_attn_inp_k_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_hparams & hparams,
    const llama_cparams & cparams,
    const llama_kv_cache_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_attn_k>(hparams, cparams, mctx_cur);

    {
        GGML_ASSERT(hparams.swa_type == LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache_iswa for SWA");

        inp->self_k_idxs = mctx_cur->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur, ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    return inp;
}

llm_graph_input_attn_k * llm_graph_context::build_attn_inp_k() const {
    if (dynamic_cast<const llama_kv_cache_kvarn_context *>(mctx) != nullptr) {
        throw std::runtime_error(
                "KVarN graph backend is not wired yet: K-only attention graph needs KVarN sink/tail/body/scale inputs "
                "instead of llama_kv_cache_context tensors");
    }

    const auto * mctx_cur = static_cast<const llama_kv_cache_context *>(mctx);

    auto inp = build_attn_inp_k_impl(ctx0, ubatch, hparams, cparams, mctx_cur);

    return (llm_graph_input_attn_k *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);


    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);


    if (wo) {
        if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE) {
            // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
            cur = build_lora_mm(wo, cur);
            ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
            if (wo_s) {
                cur = ggml_mul(ctx0, cur, wo_s);
            }
        } else {
            cur = build_lora_mm(wo, cur, wo_s);
        }
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_k_dsa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
        ggml_tensor * top_k,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx->get_mla();

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs_mla();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    const auto & kq_mask = inp->get_kq_mask_mla();

    // prepare new kq mask - starts filled with -INFINITY
    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);

    // reshape KQ mask into tensor with rows of size 1:
    // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    // prepare zero-filled tensor with rows of size 1: [1, n_top_k, n_batch, n_stream]
    // this will be our source of zero values for unmasking top k mask elements
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    // modify KQ mask by unmasking elements that are in top_k indices
    // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

    // reshape to restore the original shape of KQ mask:
    // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3], kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    // combine with the original kq mask
    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask_top_k, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }

    return cur;
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_kv_iswa * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    const bool is_swa = hparams.is_swa(il);

    auto * k_rot = is_swa ? inp->self_k_rot_swa : inp->self_k_rot;
    auto * v_rot = is_swa ? inp->self_v_rot_swa : inp->self_v_rot;

    if (k_rot) {
        q_cur = ggml_mul_mat_aux(ctx0, q_cur, k_rot);
        if (k_cur) {
            k_cur = ggml_mul_mat_aux(ctx0, k_cur, k_rot);
        }
    }
    if (v_rot) {
        if (v_cur) {
            v_cur = ggml_mul_mat_aux(ctx0, v_cur, v_rot);
        }
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);

    if (k_cur) {
        ggml_build_forward_expand(gf, k_cur);
    }

    if (v_cur) {
        ggml_build_forward_expand(gf, v_cur);
    }

    if (!is_swa && inp->mctx_kvarn_iswa != nullptr && inp->mctx_kvarn_iswa->get_base()->has_layer(il) &&
            !kvarn_graph_iswa_force_full_normal_attn()) {
        const auto * mctx_kvarn = inp->mctx_kvarn_iswa->get_base();

        if (inp->self_k_rot || inp->self_v_rot || kq_b || sinks) {
            throw std::runtime_error(
                    "KVarN+ISWA graph backend does not yet support attention rotations, KQ bias, or attention sinks on non-SWA layers");
        }
        if ((k_cur == nullptr) != (v_cur == nullptr)) {
            throw std::runtime_error("KVarN+ISWA graph backend requires K and V cache writes to be paired");
        }

        const bool stores_kv = k_cur != nullptr;
        const auto & idxs = inp->base_sink_tail_idxs;
        const llama_kvarn_layer_view layer = mctx_kvarn->get_layer_view(il);
        const bool kvarn_paper_frame = kvarn_graph_parse_env_flag("LLAMA_KVARN_ENABLE_PAPER_FRAME");
        const bool kvarn_paper_mixed_frame =
            kvarn_paper_frame && kvarn_graph_paper_mixed_frame_enabled();
        ggml_tensor * kvarn_H = nullptr;
        ggml_tensor * turbo_v_fwd = nullptr;
        ggml_tensor * turbo_v_inv = nullptr;
        ggml_tensor * turbo_v_s1 = nullptr;
        ggml_tensor * turbo_v_s2 = nullptr;
        if (kvarn_paper_frame || kvarn_graph_iswa_turbo_v_frame_enabled()) {
            if (layer.head_dim_k != layer.head_dim_v) {
                throw std::runtime_error("KVarN+ISWA paper/Turbo-frame path requires equal K/V head dimensions");
            }
            kvarn_H = kvarn_graph_build_hadamard_input(
                    ctx0, layer.head_dim_k,
                    inp->base_kvarn_hadamard,
                    inp->base_kvarn_hadamard_host,
                    inp->base_kvarn_hadamard_filled,
                    inp->base_kvarn_hadamard_dim,
                    "kvarn_iswa_hadamard");
        }
        const bool turbo_v_frame = kvarn_graph_iswa_turbo_v_frame_enabled();
        ggml_backend_t turbo_v_backend = nullptr;
        const bool turbo_v_frame_identity =
            turbo_v_frame && kvarn_graph_parse_env_flag("LLAMA_KVARN_ISWA_DEBUG_TURBO_V_FRAME_IDENTITY");
        const bool turbo_v_frame_reverse =
            turbo_v_frame && kvarn_graph_parse_env_flag("LLAMA_KVARN_ISWA_DEBUG_TURBO_V_FRAME_REVERSE");
        const bool turbo_v_frame_keep_f32 =
            turbo_v_frame && kvarn_graph_parse_env_flag("LLAMA_KVARN_ISWA_DEBUG_TURBO_V_FRAME_KEEP_F32");
        const bool turbo_v_frame_dense =
            turbo_v_frame && kvarn_graph_parse_env_flag("LLAMA_KVARN_ISWA_DEBUG_TURBO_V_FRAME_DENSE");
        if (turbo_v_frame) {
            if (!kvarn_graph_iswa_materialize_mha_enabled()) {
                throw std::runtime_error("LLAMA_KVARN_ISWA_DEBUG_TURBO_V_FRAME requires LLAMA_KVARN_ISWA_DEBUG_MATERIALIZE_MHA=1");
            }
            if (layer.head_dim_v % 128 != 0) {
                throw std::runtime_error("LLAMA_KVARN_ISWA_DEBUG_TURBO_V_FRAME requires head_dim_v to be a multiple of 128");
            }
            if (v_mla != nullptr) {
                throw std::runtime_error("LLAMA_KVARN_ISWA_DEBUG_TURBO_V_FRAME does not support MLA attention outputs");
            }
            turbo_v_backend = ggml_backend_sched_get_backend(sched, 0);
            if (!turbo_v_frame_identity && turbo_v_frame_dense) {
                turbo_v_fwd = kvarn_graph_build_turbo_wht_input(
                        ctx0, layer.head_dim_v, turbo_v_frame_reverse, false, sched, turbo_v_backend,
                        inp->base_kvarn_turbo_v_fwd,
                        inp->base_kvarn_turbo_v_fwd_host,
                        inp->base_kvarn_turbo_v_fwd_filled,
                        inp->base_kvarn_turbo_v_fwd_dim,
                        "kvarn_iswa_turbo_v_fwd");
                turbo_v_inv = kvarn_graph_build_turbo_wht_input(
                        ctx0, layer.head_dim_v, !turbo_v_frame_reverse, false, sched, turbo_v_backend,
                        inp->base_kvarn_turbo_v_inv,
                        inp->base_kvarn_turbo_v_inv_host,
                        inp->base_kvarn_turbo_v_inv_filled,
                        inp->base_kvarn_turbo_v_inv_dim,
                        "kvarn_iswa_turbo_v_inv");
            } else if (!turbo_v_frame_identity) {
                turbo_v_s1 = kvarn_graph_build_turbo_sign_input(
                        ctx0, layer.head_dim_v, 1, sched, turbo_v_backend,
                        inp->base_kvarn_turbo_v_s1,
                        inp->base_kvarn_turbo_v_s1_host,
                        inp->base_kvarn_turbo_v_s1_filled,
                        inp->base_kvarn_turbo_v_s1_dim,
                        "kvarn_iswa_turbo_v_s1");
                turbo_v_s2 = kvarn_graph_build_turbo_sign_input(
                        ctx0, layer.head_dim_v, 2, sched, turbo_v_backend,
                        inp->base_kvarn_turbo_v_s2,
                        inp->base_kvarn_turbo_v_s2_host,
                        inp->base_kvarn_turbo_v_s2_filled,
                        inp->base_kvarn_turbo_v_s2_dim,
                        "kvarn_iswa_turbo_v_s2");
            }
        }
        GGML_ASSERT(layer.body_k != nullptr);
        GGML_ASSERT(layer.body_v != nullptr);
        GGML_ASSERT(layer.scales_k != nullptr);
        GGML_ASSERT(layer.scales_v != nullptr);
        GGML_ASSERT(layer.layout_k.key_bits >= 2 && layer.layout_k.key_bits <= 8);
        GGML_ASSERT(layer.layout_v.value_bits >= 2 && layer.layout_v.value_bits <= 8);
        GGML_ASSERT(mctx_kvarn->body_store_scratch_floats(il) > 0);
        ggml_tensor * sink_tail_k_src = layer.sink_tail_k;
        ggml_tensor * sink_tail_v_src = layer.sink_tail_v;
        ggml_tensor * pending_k_src   = layer.pending_k;
        ggml_tensor * pending_v_src   = layer.pending_v;

        bool contiguous_prefill_chunk = stores_kv &&
            !kvarn_graph_prefill_direct_store_disabled() &&
            ubatch.pos != nullptr &&
            ubatch.n_tokens > 1 &&
            q_cur != nullptr && k_cur != nullptr && v_cur != nullptr &&
            q_cur->ne[2] == int64_t(ubatch.n_tokens) &&
            k_cur->ne[2] == int64_t(ubatch.n_tokens) &&
            v_cur->ne[2] == int64_t(ubatch.n_tokens);
        if (contiguous_prefill_chunk) {
            const llama_pos chunk_start = ubatch.pos[0];
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                if (ubatch.pos[i] != chunk_start + llama_pos(i)) {
                    contiguous_prefill_chunk = false;
                    break;
                }
            }
        }

        if (stores_kv) {
            const std::vector<uint32_t> seal_records = kvarn_graph_seal_records(cparams.kvarn, ubatch, mctx_kvarn->get_size());
            bool direct_prefill_store = contiguous_prefill_chunk && !seal_records.empty();
            if (kvarn_graph_prefill_direct_trace_enabled() && kvarn_graph_prefill_direct_trace_claim()) {
                const llama_pos pos0_dbg = ubatch.pos != nullptr && ubatch.n_tokens > 0 ? ubatch.pos[0] : llama_pos(-1);
                std::fprintf(stderr,
                        "KVarN graph iswa store trace: layer=%d kv_size=%u ubatch_tokens=%u pos0=%d contiguous=%d "
                        "seal_records=%u first_record=%d body_offsets=%" PRId64 "\n",
                        il, mctx_kvarn->get_size(), ubatch.n_tokens, int(pos0_dbg),
                        contiguous_prefill_chunk ? 1 : 0, uint32_t(seal_records.size()),
                        seal_records.empty() ? -1 : int(seal_records.front()),
                        inp->base_body_offsets ? inp->base_body_offsets->ne[0] : int64_t(-1));
            }
            if (direct_prefill_store) {
                const kvarn_active_window final_window =
                    kvarn_graph_active_window(cparams.kvarn, ubatch, mctx_kvarn->get_size());
                if (!final_window.valid || final_window.n_pending != 0) {
                    direct_prefill_store = false;
                }
            }
            if (direct_prefill_store) {
                const llama_pos chunk_start = ubatch.pos[0];
                const llama_pos chunk_end = chunk_start + llama_pos(ubatch.n_tokens);
                for (const uint32_t record : seal_records) {
                    const llama_pos record_start =
                        llama_pos(cparams.kvarn.sink_tokens) + llama_pos(record)*llama_pos(cparams.kvarn.group_size);
                    const llama_pos record_end = record_start + llama_pos(cparams.kvarn.group_size);
                    if (record_start < chunk_start || record_end > chunk_end) {
                        direct_prefill_store = false;
                        break;
                    }
                }
            }
            if (direct_prefill_store) {
                const llama_pos chunk_start = ubatch.pos[0];
                const llama_pos chunk_end = chunk_start + llama_pos(ubatch.n_tokens);
                for (const uint32_t record : seal_records) {
                    const llama_pos record_start = llama_pos(cparams.kvarn.sink_tokens + record*cparams.kvarn.group_size);
                    const llama_pos record_end = record_start + llama_pos(cparams.kvarn.group_size);
                    if (record >= layer.n_records || record_start < chunk_start || record_end > chunk_end) {
                        direct_prefill_store = false;
                        break;
                    }
                }
            }
            ggml_build_forward_expand(gf, inp->base_body_plan);
            ggml_build_forward_expand(gf, inp->base_body_offsets);
            ggml_build_forward_expand(gf, inp->base_tail_evict_idxs);
            ggml_tensor * pending_k_copy = nullptr;
            ggml_tensor * pending_v_copy = nullptr;
            if (!direct_prefill_store && seal_records.empty() && inp->base_body_offsets->ne[0] > 0) {
                pending_k_copy = mctx_kvarn->cpy_tail_evict_pending_k(
                            ctx0, inp->base_tail_evict_idxs, inp->base_body_offsets, il);
                pending_v_copy = mctx_kvarn->cpy_tail_evict_pending_v(
                            ctx0, inp->base_tail_evict_idxs, inp->base_body_offsets, il);
                ggml_build_forward_expand(gf, pending_k_copy);
                ggml_build_forward_expand(gf, pending_v_copy);
                pending_k_src = ggml_reshape_3d(
                        ctx0, pending_k_copy, layer.head_dim_k, layer.n_head_kv, cparams.kvarn.group_size);
                pending_v_src = ggml_reshape_3d(
                        ctx0, pending_v_copy, layer.head_dim_v, layer.n_head_kv, cparams.kvarn.group_size);
            }

            inp->base_has_body_store_ops = !seal_records.empty();
            inp->base_baked_seal_records = seal_records;
            if (kvarn_graph_prefill_direct_trace_enabled() && kvarn_graph_prefill_direct_trace_claim()) {
                std::fprintf(stderr,
                        "KVarN graph iswa store branch trace: layer=%d direct_prefill_store=%d pending_copy=%d "
                        "seal_records=%u\n",
                        il, direct_prefill_store ? 1 : 0,
                        (!direct_prefill_store && inp->base_body_offsets->ne[0] > 0) ? 1 : 0,
                        uint32_t(seal_records.size()));
            }

            ggml_tensor * body_store_scratch = nullptr;
            if (!seal_records.empty() || direct_prefill_store) {
                body_store_scratch = mctx_kvarn->build_body_store_scratch(ctx0, il);
            }

            ggml_tensor * last_body_store = nullptr;
            const auto build_body_store = [&](ggml_tensor * store) {
                if (last_body_store != nullptr) {
                    ggml_kvarn_store_kv_body_add_dep(store, last_body_store);
                }
                ggml_build_forward_expand(gf, store);
                last_body_store = store;
            };
            const auto copy_pending_slice_for_record = [&](uint32_t record) {
                const kvarn_tail_evict_slice slice = kvarn_graph_tail_evict_slice_for_record(
                        cparams.kvarn, ubatch, mctx_kvarn->get_size(), record);
                if (slice.count <= 0) {
                    throw std::runtime_error("KVarN+ISWA graph backend could not find evicted pending rows for sealed body record");
                }
                if (!slice.contiguous) {
                    throw std::runtime_error("KVarN+ISWA graph backend cannot seal non-contiguous pending rows for a body record");
                }

                ggml_tensor * tail_idxs = ggml_view_1d(
                        ctx0, inp->base_tail_evict_idxs, slice.count,
                        size_t(slice.start)*inp->base_tail_evict_idxs->nb[0]);
                ggml_tensor * offsets = ggml_view_1d(
                        ctx0, inp->base_body_offsets, slice.count,
                        size_t(slice.start)*inp->base_body_offsets->nb[0]);
                pending_k_copy = mctx_kvarn->cpy_tail_evict_pending_k(ctx0, tail_idxs, offsets, il);
                pending_v_copy = mctx_kvarn->cpy_tail_evict_pending_v(ctx0, tail_idxs, offsets, il);
                if (last_body_store != nullptr) {
                    ggml_set_rows_add_dep(pending_k_copy, last_body_store);
                    ggml_set_rows_add_dep(pending_v_copy, last_body_store);
                }
                ggml_build_forward_expand(gf, pending_k_copy);
                ggml_build_forward_expand(gf, pending_v_copy);
                pending_k_src = ggml_reshape_3d(
                        ctx0, pending_k_copy, layer.head_dim_k, layer.n_head_kv, cparams.kvarn.group_size);
                pending_v_src = ggml_reshape_3d(
                        ctx0, pending_v_copy, layer.head_dim_v, layer.n_head_kv, cparams.kvarn.group_size);
            };

            if (direct_prefill_store) {
                constexpr uint32_t direct_record_batch_max = 8;
                const bool use_direct_record_batch =
                    kvarn_paper_frame || kvarn_graph_parse_env_flag("LLAMA_KVARN_ENABLE_DIRECT_RECORD_BATCH");
                if (kvarn_graph_prefill_direct_trace_enabled() && kvarn_graph_prefill_direct_trace_claim()) {
                    std::fprintf(stderr,
                            "KVarN graph iswa prefill-direct-store trace: layer=%d head_dim=%u heads=%u direct_records=%u batch_enabled=%d batch_max=%u chunks=%u\n",
                            il, layer.head_dim_k, layer.n_head_kv, uint32_t(seal_records.size()),
                            use_direct_record_batch ? 1 : 0, direct_record_batch_max,
                            (uint32_t(seal_records.size()) + direct_record_batch_max - 1)/direct_record_batch_max);
                }
                if (use_direct_record_batch) {
                    for (uint32_t i_record = 0; i_record < seal_records.size();) {
                        const uint32_t record0 = seal_records[i_record];
                        uint32_t n_batch = 1;
                        while (i_record + n_batch < seal_records.size() &&
                                n_batch < direct_record_batch_max &&
                                seal_records[i_record + n_batch] == record0 + n_batch) {
                            ++n_batch;
                        }
                        const uint32_t pos0 = cparams.kvarn.sink_tokens + record0*cparams.kvarn.group_size - uint32_t(ubatch.pos[0]);
                        ggml_tensor * k_tiles = ggml_view_4d(
                                ctx0, k_cur,
                                layer.head_dim_k, layer.n_head_kv, cparams.kvarn.group_size, n_batch,
                                k_cur->nb[1], k_cur->nb[2], size_t(cparams.kvarn.group_size)*k_cur->nb[2],
                                size_t(pos0)*k_cur->nb[2]);
                        ggml_tensor * v_tiles = ggml_view_4d(
                                ctx0, v_cur,
                                layer.head_dim_v, layer.n_head_kv, cparams.kvarn.group_size, n_batch,
                                v_cur->nb[1], v_cur->nb[2], size_t(cparams.kvarn.group_size)*v_cur->nb[2],
                                size_t(pos0)*v_cur->nb[2]);
                        build_body_store(mctx_kvarn->store_kv_body_records_all_heads(
                                    ctx0, k_tiles, v_tiles, body_store_scratch, il, record0, n_batch));
                        i_record += n_batch;
                    }
                } else {
                    for (const uint32_t record : seal_records) {
                        const uint32_t pos0 = cparams.kvarn.sink_tokens + record*cparams.kvarn.group_size - uint32_t(ubatch.pos[0]);
                        ggml_tensor * k_tile = ggml_view_3d(
                                ctx0, k_cur,
                                layer.head_dim_k, layer.n_head_kv, cparams.kvarn.group_size,
                                k_cur->nb[1], k_cur->nb[2], size_t(pos0)*k_cur->nb[2]);
                        ggml_tensor * v_tile = ggml_view_3d(
                                ctx0, v_cur,
                                layer.head_dim_v, layer.n_head_kv, cparams.kvarn.group_size,
                                v_cur->nb[1], v_cur->nb[2], size_t(pos0)*v_cur->nb[2]);
                        build_body_store(mctx_kvarn->store_kv_body_all_heads(
                                    ctx0, k_tile, v_tile, body_store_scratch, il, record));
                    }
                }
            } else {
                for (const uint32_t seal_record : seal_records) {
                    if (seal_record >= layer.n_records) {
                        throw std::runtime_error("KVarN+ISWA graph backend attempted to seal a body record outside cache capacity");
                    }

                    copy_pending_slice_for_record(seal_record);
                    if (layer.head_dim_k >= 256 && layer.n_head_kv > 1) {
                        build_body_store(mctx_kvarn->store_kv_body_all_heads_from_pending(
                                    ctx0, body_store_scratch, il, seal_record, pending_k_src, pending_v_src));
                    } else {
                        for (uint32_t ih = 0; ih < layer.n_head_kv; ++ih) {
                            build_body_store(mctx_kvarn->store_kv_body_record_from_pending(
                                        ctx0, body_store_scratch, il, ih, seal_record, pending_k_src, pending_v_src));
                        }
                    }
                }
                const kvarn_active_window final_window =
                    kvarn_graph_active_window(cparams.kvarn, ubatch, mctx_kvarn->get_size());
                if (final_window.valid && final_window.n_pending > 0) {
                    copy_pending_slice_for_record(uint32_t(final_window.n_records));
                }
            }

            ggml_tensor * k_st_src = k_cur;
            ggml_tensor * v_st_src = v_cur;
            if (kvarn_paper_frame && !kvarn_paper_mixed_frame) {
                k_st_src = kvarn_graph_apply_hadamard(ctx0, kvarn_H, k_st_src);
                v_st_src = kvarn_graph_apply_hadamard(ctx0, kvarn_H, v_st_src);
            }
            ggml_tensor * sink_k_copy = mctx_kvarn->cpy_sink_tail_k(ctx0, k_st_src, idxs, il);
            ggml_tensor * sink_v_copy = mctx_kvarn->cpy_sink_tail_v(ctx0, v_st_src, idxs, il);
            if (pending_k_copy != nullptr) {
                ggml_set_rows_add_dep(sink_k_copy, pending_k_copy);
            }
            if (pending_v_copy != nullptr) {
                ggml_set_rows_add_dep(sink_v_copy, pending_v_copy);
            }
            if (last_body_store != nullptr) {
                ggml_set_rows_add_dep(sink_k_copy, last_body_store);
                ggml_set_rows_add_dep(sink_v_copy, last_body_store);
            }
            ggml_build_forward_expand(gf, sink_k_copy);
            ggml_build_forward_expand(gf, sink_v_copy);
            sink_tail_k_src = ggml_reshape_3d(
                    ctx0, sink_k_copy, layer.head_dim_k, layer.n_head_kv, layer.sink_tail_k->ne[2]);
            sink_tail_v_src = ggml_reshape_3d(
                    ctx0, sink_v_copy, layer.head_dim_v, layer.n_head_kv, layer.sink_tail_v->ne[2]);
        }

        const ggml_tensor * kq_mask = inp->get_kq_mask_kvarn();
        const bool prefill_direct_attn =
            stores_kv &&
            kvarn_graph_iswa_prefill_direct_attn_allowed() &&
            !kvarn_graph_prefill_direct_attn_disabled() &&
            q_cur != nullptr && k_cur != nullptr && v_cur != nullptr && kq_mask != nullptr &&
            q_cur->ne[2] > 1 &&
            k_cur->ne[2] == q_cur->ne[2] &&
            v_cur->ne[2] == q_cur->ne[2] &&
            kq_mask->ne[0] == k_cur->ne[2] &&
            kq_mask->ne[1] == q_cur->ne[2];
        if (kvarn_graph_prefill_direct_trace_enabled() && kvarn_graph_prefill_direct_trace_claim()) {
            std::fprintf(stderr,
                    "KVarN graph iswa prefill-direct trace: use=%d stores_kv=%d q_tokens=%" PRId64
                    " k_tokens=%" PRId64 " v_tokens=%" PRId64 " mask=[%" PRId64 ",%" PRId64 "]\n",
                    prefill_direct_attn ? 1 : 0, stores_kv ? 1 : 0,
                    q_cur ? q_cur->ne[2] : int64_t(-1),
                    k_cur ? k_cur->ne[2] : int64_t(-1),
                    v_cur ? v_cur->ne[2] : int64_t(-1),
                    kq_mask ? kq_mask->ne[0] : int64_t(-1),
                    kq_mask ? kq_mask->ne[1] : int64_t(-1));
        }
        if (prefill_direct_attn) {
            ggml_tensor * k_direct = k_cur;
            ggml_tensor * v_direct = v_cur;
            if (k_direct->type == GGML_TYPE_F32) {
                k_direct = ggml_cast(ctx0, k_direct, GGML_TYPE_F16);
            }
            if (v_direct->type == GGML_TYPE_F32) {
                v_direct = ggml_cast(ctx0, v_direct, GGML_TYPE_F16);
            }
            ggml_tensor * cur = build_attn_mha(q_cur, k_direct, v_direct, kq_b, inp->get_kq_mask_kvarn(), sinks, v_mla, kq_scale, il);
            cb(cur, "kvarn_iswa_prefill_direct_kqv_out", il);
            cur = ggml_reshape_2d(ctx0, cur, q_cur->ne[0]*q_cur->ne[1], q_cur->ne[2]);

            if (wo) {
                cur = build_lora_mm(wo, cur, wo_s);
                cb(cur, "kvarn_iswa_prefill_direct_kqv_wo", il);
            }
            if (wo_b) {
                cur = ggml_add(ctx0, cur, wo_b);
                cb(cur, "kvarn_iswa_prefill_direct_kqv_wo_b", il);
            }

            return cur;
        }

        const llama_pos pos = ubatch.pos ? ubatch.pos[0] : llama_pos(0);
        if (pos < 0) {
            throw std::runtime_error("KVarN+ISWA graph backend cannot attend to a negative-position decode token");
        }

        const kvarn_active_window window = kvarn_graph_active_window(cparams.kvarn, ubatch, mctx_kvarn->get_size());
        if (!window.valid) {
            throw std::runtime_error("KVarN+ISWA graph backend cannot build active decode attention window");
        }
        if (uint32_t(window.n_records) > layer.n_records) {
            throw std::runtime_error("KVarN+ISWA graph backend active body record count exceeds allocated cache capacity");
        }

        if (!kvarn_graph_iswa_sinktail_mha_disabled() &&
                window.n_records == 0 && window.n_pending == 0 && window.tail_start == 0 &&
                inp->get_kq_mask_kvarn()->ne[0] == window.n_kv) {
            ggml_tensor * k_cache = ggml_view_3d(
                    ctx0, sink_tail_k_src,
                    layer.sink_tail_k->ne[0], layer.sink_tail_k->ne[1], window.n_kv,
                    layer.sink_tail_k->nb[1], layer.sink_tail_k->nb[2], 0);
            ggml_tensor * v_cache = ggml_view_3d(
                    ctx0, sink_tail_v_src,
                    layer.sink_tail_v->ne[0], layer.sink_tail_v->ne[1], window.n_kv,
                    layer.sink_tail_v->nb[1], layer.sink_tail_v->nb[2], 0);
            ggml_tensor * q_cache = q_cur;
            if (kvarn_paper_frame && !kvarn_paper_mixed_frame) {
                q_cache = kvarn_graph_apply_hadamard(ctx0, kvarn_H, q_cache);
            }
            ggml_tensor * cur = build_attn_mha(q_cache, k_cache, v_cache, kq_b, inp->get_kq_mask_kvarn(), sinks, v_mla, kq_scale, il);
            cb(cur, "kvarn_iswa_sinktail_mha_kqv_out", il);
            if (kvarn_paper_frame && !kvarn_paper_mixed_frame) {
                cur = ggml_reshape_3d(ctx0, cur, layer.head_dim_v, q_cur->ne[1], q_cur->ne[2]);
                cur = kvarn_graph_apply_hadamard(ctx0, kvarn_H, cur);
                cb(cur, "kvarn_iswa_sinktail_mha_kqv_out_unrot", il);
            }
            cur = ggml_reshape_2d(ctx0, cur, q_cur->ne[0]*q_cur->ne[1], q_cur->ne[2]);

            if (wo) {
                cur = build_lora_mm(wo, cur, wo_s);
                cb(cur, "kvarn_iswa_sinktail_mha_kqv_wo", il);
            }
            if (wo_b) {
                cur = ggml_add(ctx0, cur, wo_b);
                cb(cur, "kvarn_iswa_sinktail_mha_kqv_wo_b", il);
            }

            return cur;
        }

        const kvarn_active_window scratch_window = kvarn_graph_build_scratch_window(
                window, cparams.kvarn, mctx_kvarn->get_size(), ubatch);
        const int64_t scores_floats = kvarn_graph_attn_scratch_floats(
                scratch_window, layer.n_head_kv, scratch_window.n_records, layer.head_dim_k, cparams.kvarn.group_size);
        ggml_tensor * scores = mctx_kvarn->build_attn_mixed_scratch(ctx0, il, scores_floats);
        inp->base_mixed_attn_scores.push_back(scores);

        // CUDA-graph-replay-safe decode: in the pure sink/tail regime
        // (n_tokens==1, no body records, no pending) the op is built with
        // frozen worst-case op_params and the live window is streamed through
        // a shared I32 input tensor (src[11]). Node properties then stay
        // bit-stable across decode tokens, so the CUDA backend can capture
        // the decode graph once and replay it instead of paying per-kernel
        // launch overhead on all ~48 layers every token.
        const bool window_indirect = kvarn_graph_decode_stable_topology(ubatch) &&
                window.n_records == 0 && window.n_pending == 0 &&
                !inp->base_has_body_store_ops;

        if (window_indirect && inp->base_kvarn_window == nullptr) {
            inp->base_kvarn_window = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 8);
            ggml_set_input(inp->base_kvarn_window);
            ggml_set_name(inp->base_kvarn_window, "kvarn_iswa_window");
            inp->base_window_indirect = true;
        }

        const llama_kvarn_params effective_kvarn =
            kvarn_graph_effective_params(cparams.kvarn, mctx_kvarn->get_size());
        const int32_t op_n_sink     = window_indirect ? int32_t(effective_kvarn.sink_tokens) : int32_t(window.n_sink);
        const int32_t op_n_records  = window_indirect ? 0 : int32_t(window.n_records);
        const int32_t op_n_pending  = window_indirect ? 0 : int32_t(window.n_pending);
        const int32_t op_n_tail     = window_indirect ? int32_t(effective_kvarn.tail_tokens) : int32_t(window.n_tail);
        const int32_t op_tail_start = window_indirect ? 0 : int32_t(window.tail_start);

        const bool kvarn_fused_fwht =
            kvarn_paper_frame &&
            !kvarn_paper_mixed_frame &&
            !kvarn_graph_parse_env_flag("LLAMA_KVARN_DISABLE_FUSED_FWHT") &&
            (q_cur->ne[2] == 1 || layer.head_dim_k >= 512) &&
            op_n_records == 0 &&
            op_n_pending == 0;

        ggml_tensor * q_in = q_cur;
        ggml_tensor * q_body = nullptr;
        if (kvarn_paper_frame && !kvarn_fused_fwht) {
            q_body = kvarn_graph_apply_hadamard(ctx0, kvarn_H, q_cur);
        }
        if (kvarn_paper_frame && !kvarn_paper_mixed_frame) {
            q_in = kvarn_fused_fwht ? q_cur : q_body;
        }
        const bool debug_materialize_mha = kvarn_graph_iswa_materialize_mha_enabled();
        const bool debug_dual_mha_compare = kvarn_graph_iswa_dual_mha_compare_enabled();
        const bool debug_raw_mha_compare = kvarn_graph_iswa_raw_mha_compare_enabled();
        if ((debug_materialize_mha || debug_dual_mha_compare || debug_raw_mha_compare) && !window_indirect) {
            if (kvarn_paper_frame || kvarn_paper_mixed_frame || q_body != nullptr || kvarn_fused_fwht) {
                throw std::runtime_error("LLAMA_KVARN_ISWA_DEBUG_MATERIALIZE_MHA only supports the normal KVarN frame");
            }
            if ((debug_dual_mha_compare || debug_raw_mha_compare) && turbo_v_frame) {
                throw std::runtime_error("LLAMA_KVARN_ISWA_DEBUG_DUAL_MHA_COMPARE and LLAMA_KVARN_ISWA_DEBUG_RAW_MHA_COMPARE only support the normal KVarN frame");
            }
            ggml_tensor * compact_mask = inp->get_kq_mask_kvarn();
            if (compact_mask == nullptr ||
                    compact_mask->ne[0] != scratch_window.n_kv ||
                    compact_mask->ne[1] != q_cur->ne[2]) {
                throw std::runtime_error("LLAMA_KVARN_ISWA_DEBUG_MATERIALIZE_MHA requires an exact-width compact KVarN mask");
            }
            if (turbo_v_frame && op_n_records > 0 &&
                    (!kvarn_graph_parse_env_flag("LLAMA_KVARN_DEBUG_RAW_BODY_K") ||
                     !kvarn_graph_parse_env_flag("LLAMA_KVARN_DEBUG_RAW_BODY_V"))) {
                throw std::runtime_error(
                        "LLAMA_KVARN_ISWA_DEBUG_TURBO_V_FRAME is a frame-only diagnostic and requires "
                        "LLAMA_KVARN_DEBUG_RAW_BODY_K=1 and LLAMA_KVARN_DEBUG_RAW_BODY_V=1 when body records are active");
            }
            if (debug_raw_mha_compare && op_n_records > 0 &&
                    !kvarn_graph_parse_env_flag("LLAMA_KVARN_DEBUG_CAPTURE_RAW_BODY_MIRROR") &&
                    (!kvarn_graph_parse_env_flag("LLAMA_KVARN_DEBUG_RAW_BODY_K") ||
                     !kvarn_graph_parse_env_flag("LLAMA_KVARN_DEBUG_RAW_BODY_V"))) {
                throw std::runtime_error(
                        "LLAMA_KVARN_ISWA_DEBUG_RAW_MHA_COMPARE requires "
                        "LLAMA_KVARN_DEBUG_CAPTURE_RAW_BODY_MIRROR=1, or both "
                        "LLAMA_KVARN_DEBUG_RAW_BODY_K=1 and LLAMA_KVARN_DEBUG_RAW_BODY_V=1, so the raw body mirror is captured");
            }

            ggml_tensor * k_mat = ggml_kvarn_materialize_kv(
                    ctx0, sink_tail_k_src, layer.body_k, layer.scales_k, pending_k_src,
                    0, op_n_sink, op_n_records, op_n_pending, op_n_tail, op_tail_start,
                    int32_t(layer.head_dim_k), int32_t(cparams.kvarn.group_size), int32_t(layer.layout_k.key_bits));
            if (!debug_raw_mha_compare && kvarn_graph_parse_env_flag("LLAMA_KVARN_DEBUG_RAW_BODY_K")) {
                ggml_kvarn_materialize_kv_set_debug_raw_body(k_mat, 1);
            }
            cb(k_mat, "kvarn_iswa_materialized_k", il);

            ggml_tensor * v_mat = ggml_kvarn_materialize_kv(
                    ctx0, sink_tail_v_src, layer.body_v, layer.scales_v, pending_v_src,
                    1, op_n_sink, op_n_records, op_n_pending, op_n_tail, op_tail_start,
                    int32_t(layer.head_dim_v), int32_t(cparams.kvarn.group_size), int32_t(layer.layout_v.value_bits));
            ggml_kvarn_materialize_kv_set_v_layout(v_mat, int32_t(layer.layout_v.v_layout));
            if (!debug_raw_mha_compare && kvarn_graph_parse_env_flag("LLAMA_KVARN_DEBUG_RAW_BODY_V")) {
                ggml_kvarn_materialize_kv_set_debug_raw_body(v_mat, 1);
            }
            cb(v_mat, "kvarn_iswa_materialized_v", il);
            if (turbo_v_frame) {
                if (!turbo_v_frame_identity) {
                    if (turbo_v_frame_dense) {
                        v_mat = ggml_mul_mat_aux(ctx0, v_mat, turbo_v_fwd, false, sched, turbo_v_backend);
                    } else {
                        v_mat = kvarn_graph_apply_turbo_wht(
                                ctx0, v_mat, kvarn_H, turbo_v_s1, turbo_v_s2,
                                turbo_v_frame_reverse, sched, turbo_v_backend);
                    }
                    cb(v_mat, "kvarn_iswa_materialized_v_turbo_rot", il);
                    if (!turbo_v_frame_keep_f32) {
                        v_mat = ggml_cast(ctx0, v_mat, GGML_TYPE_F16);
                        cb(v_mat, "kvarn_iswa_materialized_v_turbo_rot_f16", il);
                    }
                }
            }

            if (kvarn_graph_attn_trace_enabled() && kvarn_graph_attn_trace_claim()) {
                std::fprintf(stderr,
                        "KVarN graph materialized-MHA trace: layer=%d q_tokens=%" PRId64
                        " n_sink=%d n_records=%d n_pending=%d n_tail=%d tail_start=%d n_kv=%" PRId64
                        " mask=[%" PRId64 ",%" PRId64 "] bits=k%d/v%d turbo_v_frame_only=%d turbo_v_identity=%d turbo_v_reverse=%d turbo_v_keep_f32=%d turbo_v_dense=%d turbo_v_compression=0\n",
                        il, q_cur->ne[2], op_n_sink, op_n_records, op_n_pending, op_n_tail, op_tail_start,
                        scratch_window.n_kv, compact_mask->ne[0], compact_mask->ne[1],
                        int32_t(layer.layout_k.key_bits), int32_t(layer.layout_v.value_bits),
                        turbo_v_frame ? 1 : 0, turbo_v_frame_identity ? 1 : 0,
                        turbo_v_frame_reverse ? 1 : 0, turbo_v_frame_keep_f32 ? 1 : 0,
                        turbo_v_frame_dense ? 1 : 0);
                kvarn_graph_attn_trace_tensor("q_in", q_in);
                kvarn_graph_attn_trace_tensor("k_mat", k_mat);
                kvarn_graph_attn_trace_tensor("v_mat", v_mat);
                kvarn_graph_attn_trace_tensor("mask", compact_mask);
            }

            ggml_tensor * cur = build_attn_mha(q_in, k_mat, v_mat, kq_b, compact_mask, sinks, v_mla, kq_scale, il);
            cb(cur, "kvarn_iswa_materialized_mha_kqv_out", il);
            if (turbo_v_frame) {
                if (ggml_nelements(cur) % layer.head_dim_v != 0) {
                    throw std::runtime_error("LLAMA_KVARN_ISWA_DEBUG_TURBO_V_FRAME reached an attention output not divisible by head_dim_v");
                }
                if (!turbo_v_frame_identity) {
                    if (turbo_v_frame_dense) {
                        cur = ggml_mul_mat_aux(ctx0, cur, turbo_v_inv, false, sched, turbo_v_backend);
                    } else {
                        cur = kvarn_graph_apply_turbo_wht(
                                ctx0, cur, kvarn_H, turbo_v_s1, turbo_v_s2,
                                !turbo_v_frame_reverse, sched, turbo_v_backend);
                    }
                    cb(cur, "kvarn_iswa_materialized_mha_kqv_out_turbo_unrot", il);
                }
            }
            cur = ggml_reshape_2d(ctx0, cur, q_cur->ne[0]*q_cur->ne[1], q_cur->ne[2]);
            cb(cur, "kvarn_iswa_materialized_mha_kqv_out_2d", il);

            if (wo) {
                cur = build_lora_mm(wo, cur, wo_s);
                cb(cur, "kvarn_iswa_materialized_mha_kqv_wo", il);
            }
            if (wo_b) {
                cur = ggml_add(ctx0, cur, wo_b);
                cb(cur, "kvarn_iswa_materialized_mha_kqv_wo_b", il);
            }

            if (debug_raw_mha_compare) {
                ggml_tensor * k_raw = ggml_kvarn_materialize_kv(
                        ctx0, sink_tail_k_src, layer.body_k, layer.scales_k, pending_k_src,
                        0, op_n_sink, op_n_records, op_n_pending, op_n_tail, op_tail_start,
                        int32_t(layer.head_dim_k), int32_t(cparams.kvarn.group_size), int32_t(layer.layout_k.key_bits));
                ggml_kvarn_materialize_kv_set_debug_raw_body(k_raw, 1);
                cb(k_raw, "kvarn_iswa_materialized_raw_k", il);

                ggml_tensor * v_raw = ggml_kvarn_materialize_kv(
                        ctx0, sink_tail_v_src, layer.body_v, layer.scales_v, pending_v_src,
                        1, op_n_sink, op_n_records, op_n_pending, op_n_tail, op_tail_start,
                        int32_t(layer.head_dim_v), int32_t(cparams.kvarn.group_size), int32_t(layer.layout_v.value_bits));
                ggml_kvarn_materialize_kv_set_v_layout(v_raw, int32_t(layer.layout_v.v_layout));
                ggml_kvarn_materialize_kv_set_debug_raw_body(v_raw, 1);
                cb(v_raw, "kvarn_iswa_materialized_raw_v", il);

                ggml_tensor * raw_cur = build_attn_mha(q_in, k_raw, v_raw, kq_b, compact_mask, sinks, v_mla, kq_scale, il);
                cb(raw_cur, "kvarn_iswa_materialized_raw_mha_kqv_out", il);
                raw_cur = ggml_reshape_2d(ctx0, raw_cur, q_cur->ne[0]*q_cur->ne[1], q_cur->ne[2]);
                cb(raw_cur, "kvarn_iswa_materialized_raw_mha_kqv_out_2d", il);
                if (wo) {
                    raw_cur = build_lora_mm(wo, raw_cur, wo_s);
                    cb(raw_cur, "kvarn_iswa_materialized_raw_mha_kqv_wo", il);
                }
                if (wo_b) {
                    raw_cur = ggml_add(ctx0, raw_cur, wo_b);
                    cb(raw_cur, "kvarn_iswa_materialized_raw_mha_kqv_wo_b", il);
                }
                ggml_build_forward_expand(gf, raw_cur);
            }

            if (debug_materialize_mha) {
                return cur;
            }

            ggml_build_forward_expand(gf, cur);
        }
        ggml_tensor * cur = ggml_kvarn_attn_mixed(
                ctx0, q_in, sink_tail_k_src, sink_tail_v_src, layer.body_k, layer.body_v,
                layer.scales_k, layer.scales_v, pending_k_src, pending_v_src, scores, inp->get_kq_mask_kvarn(),
                op_n_sink, op_n_records, op_n_pending, op_n_tail, op_tail_start,
                int32_t(layer.head_dim_k), int32_t(cparams.kvarn.group_size),
                int32_t(layer.layout_k.key_bits), int32_t(layer.layout_v.value_bits), kq_scale,
                hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        ggml_kvarn_attn_mixed_set_v_layout(cur, int32_t(layer.layout_v.v_layout));
        if (kvarn_fused_fwht) {
            ggml_kvarn_attn_mixed_set_frame_flags(cur, GGML_KVARN_ATTN_FRAME_FUSED_PAPER_FULL);
        }
        if (kvarn_paper_mixed_frame && q_body != nullptr) {
            ggml_kvarn_attn_mixed_set_q_body(cur, q_body);
        }
        if (window_indirect) {
            ggml_kvarn_attn_mixed_set_window(cur, inp->base_kvarn_window);
        }
        inp->base_mixed_attn_nodes.push_back(cur);
        cb(cur, "kvarn_iswa_kqv_out", il);
        if (kvarn_paper_frame && !kvarn_paper_mixed_frame && !kvarn_fused_fwht) {
            cur = kvarn_graph_apply_hadamard(ctx0, kvarn_H, cur);
            cb(cur, "kvarn_iswa_kqv_out_unrot", il);
        }
        cur = ggml_reshape_2d(ctx0, cur, q_cur->ne[0]*q_cur->ne[1], q_cur->ne[2]);
        cb(cur, "kvarn_iswa_kqv_out_2d", il);

        if (wo) {
            cur = build_lora_mm(wo, cur, wo_s);
            cb(cur, "kvarn_iswa_kqv_wo", il);
        }
        if (wo_b) {
            cur = ggml_add(ctx0, cur, wo_b);
            cb(cur, "kvarn_iswa_kqv_wo_b", il);
        }

        return cur;
    }

    const llama_kv_cache_context * mctx_cur = nullptr;
    if (inp->mctx_kvarn_iswa != nullptr) {
        if (!is_swa) {
            mctx_cur = inp->mctx_kvarn_iswa->get_full_normal();
            if (mctx_cur == nullptr || !mctx_cur->has_layer(il)) {
                throw std::runtime_error("KVarN+ISWA graph backend reached normal KV path for a non-SWA layer without a diagnostic normal fallback");
            }
        } else {
            mctx_cur = inp->mctx_kvarn_iswa->get_swa();
        }
    } else {
        const auto * mctx_iswa = inp->mctx;
        mctx_cur = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();
    }

    // optionally store to KV cache
    if (k_cur) {
        const auto & k_idxs = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
    }

    if (v_cur) {
        const auto & v_idxs = is_swa ? inp->get_v_idxs_swa() : inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);


    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);


    if (v_rot) {
        cur = ggml_mul_mat_aux(ctx0, cur, v_rot);
    }

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
        cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
        cb(cur, "kqv_wo_b", il);
    }

    return cur;
}

llm_graph_input_attn_cross * llm_graph_context::build_attn_inp_cross() const {
    auto inp = std::make_unique<llm_graph_input_attn_cross>(cross);

    const int32_t n_enc = !cross->v_embd.empty() ? cross->n_enc : hparams.n_ctx_train;

    // flash attention requires an f16 mask
    const auto type_mask = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;

    inp->cross_kq_mask = ggml_new_tensor_4d(ctx0, type_mask, n_enc, n_tokens, 1, 1);
    ggml_set_input(inp->cross_kq_mask);

    inp->cross_kq_mask_cnv = inp->cross_kq_mask;

    return (llm_graph_input_attn_cross *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_attn(
        llm_graph_input_attn_cross * inp,
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * wo_s,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_b,
        ggml_tensor * sinks,
        ggml_tensor * v_mla,
            float     kq_scale,
            int       il) const {
    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, k_cur);
    ggml_build_forward_expand(gf, v_cur);

    const auto & kq_mask = inp->get_kq_mask_cross();

    ggml_tensor * q = q_cur;
    ggml_tensor * k = k_cur;
    ggml_tensor * v = v_cur;

    ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_lora_mm(wo, cur, wo_s);
        cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
        cb(cur, "kqv_wo_b", il);
    }

    return cur;
}

llm_graph_input_attn_k_dsa * llm_graph_context::build_attn_inp_k_dsa() const {
    const auto * mctx_cur = static_cast<const llama_kv_cache_dsa_context *>(mctx);

    auto inp = std::make_unique<llm_graph_input_attn_k_dsa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs_mla = mctx_cur->get_mla()->build_input_k_idxs(ctx0, ubatch);

        inp->self_kq_mask_mla = build_attn_inp_kq_mask(ctx0, mctx_cur->get_mla(), ubatch, cparams);
        inp->self_kq_mask_mla_cnv = inp->self_kq_mask_mla;
    }

    {
        inp->self_k_idxs_lid = mctx_cur->get_lid()->build_input_k_idxs(ctx0, ubatch);

        // ensure F32 mask
        auto cparams_copy = cparams;
        cparams_copy.flash_attn = false;

        inp->self_kq_mask_lid = build_attn_inp_kq_mask(ctx0, mctx_cur->get_lid(), ubatch, cparams_copy);
        inp->self_kq_mask_lid_cnv = inp->self_kq_mask_lid;

        inp->self_k_rot_lid = mctx_cur->get_lid()->build_input_k_rot(ctx0);
    }

    return (llm_graph_input_attn_k_dsa *) res->add_input(std::move(inp));
}

// TODO: maybe separate the inner implementation into a separate function
//       like with the non-sliding window equivalent
//       once sliding-window hybrid caches are a thing.
llm_graph_input_attn_kv_iswa * llm_graph_context::build_attn_inp_kv_iswa() const {
    if (const auto * mctx_kvarn_iswa = dynamic_cast<const llama_kv_cache_kvarn_iswa_context *>(mctx)) {
        auto inp = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, mctx_kvarn_iswa);

        const auto * base_ctx = mctx_kvarn_iswa->get_base();
        inp->base_sink_tail_idxs = base_ctx->build_input_sink_tail_idxs(ctx0, ubatch);
        inp->base_body_plan = base_ctx->build_input_body_plan(ctx0, ubatch);
        inp->base_body_offsets = base_ctx->build_input_body_offsets(ctx0, ubatch);
        inp->base_tail_evict_idxs = base_ctx->build_input_tail_evict_idxs(ctx0, ubatch);
        const kvarn_active_window mask_window = kvarn_graph_active_window(cparams.kvarn, ubatch, base_ctx->get_size());
        const uint32_t mask_n_kv = kvarn_graph_mask_n_kv(mask_window, base_ctx->get_size(), ubatch);
        inp->base_kvarn_kq_mask = build_attn_inp_kq_mask(ctx0, mask_n_kv, ubatch, cparams);
        inp->base_kvarn_kq_mask_cnv = inp->base_kvarn_kq_mask;

        if (const auto * full_ctx = mctx_kvarn_iswa->get_full_normal()) {
            inp->self_k_idxs = full_ctx->build_input_k_idxs(ctx0, ubatch);
            inp->self_v_idxs = full_ctx->build_input_v_idxs(ctx0, ubatch);
            inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, full_ctx, ubatch, cparams);
            inp->self_kq_mask_cnv = inp->self_kq_mask;
            inp->self_k_rot = full_ctx->build_input_k_rot(ctx0);
            inp->self_v_rot = full_ctx->build_input_v_rot(ctx0);
        } else {
            inp->self_kq_mask = inp->base_kvarn_kq_mask;
            inp->self_kq_mask_cnv = inp->base_kvarn_kq_mask_cnv;
        }

        const auto * swa_ctx = mctx_kvarn_iswa->get_swa();
        inp->self_k_idxs_swa = swa_ctx->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs_swa = swa_ctx->build_input_v_idxs(ctx0, ubatch);
        inp->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, swa_ctx, ubatch, cparams);
        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
        inp->self_k_rot_swa = swa_ctx->build_input_k_rot(ctx0);
        inp->self_v_rot_swa = swa_ctx->build_input_v_rot(ctx0);

        return (llm_graph_input_attn_kv_iswa *) res->add_input(std::move(inp));
    }

    const auto * mctx_cur = dynamic_cast<const llama_kv_cache_iswa_context *>(mctx);
    if (!mctx_cur) {
        throw std::runtime_error("ISWA attention graph requested a non-ISWA memory context");
    }

    auto inp = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, mctx_cur);

    {
        inp->self_k_idxs = mctx_cur->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs = mctx_cur->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask = build_attn_inp_kq_mask(ctx0, mctx_cur->get_base(), ubatch, cparams);
        inp->self_kq_mask_cnv = inp->self_kq_mask;
    }

    {
        GGML_ASSERT(hparams.swa_type != LLAMA_SWA_TYPE_NONE && "Use llama_kv_cache for non-SWA");

        inp->self_k_idxs_swa = mctx_cur->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp->self_v_idxs_swa = mctx_cur->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, mctx_cur->get_swa(), ubatch, cparams);
        inp->self_kq_mask_swa_cnv = inp->self_kq_mask_swa;
    }

    inp->self_k_rot = mctx_cur->get_base()->build_input_k_rot(ctx0);
    inp->self_v_rot = mctx_cur->get_base()->build_input_v_rot(ctx0);

    inp->self_k_rot_swa = mctx_cur->get_swa()->build_input_k_rot(ctx0);
    inp->self_v_rot_swa = mctx_cur->get_swa()->build_input_v_rot(ctx0);

    return (llm_graph_input_attn_kv_iswa *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        ggml_tensor * s,
        ggml_tensor * state_copy_main,
        ggml_tensor * state_copy_extra,
            int32_t   state_size,
            int32_t   n_seqs,
           uint32_t   n_rs,
           uint32_t   rs_head,
           uint32_t   rs_size,
            int32_t   rs_zero,
        const llm_graph_get_rows_fn & get_state_rows) const {

    GGML_UNUSED(rs_size);
    ggml_tensor * states = ggml_reshape_2d(ctx0, s, state_size, s->ne[1]);

    // Clear a single state which will then be copied to the other cleared states.
    // Note that this is a no-op when the view is zero-sized.
    ggml_tensor * state_zero = ggml_view_1d(ctx0, states, state_size*(rs_zero >= 0), rs_zero*states->nb[1]*(rs_zero >= 0));
    ggml_build_forward_expand(gf, ggml_scale_inplace(ctx0, state_zero, 0));

    // copy states
    // NOTE: assuming the copy destinations are ALL contained between rs_head and rs_head + n_rs
    // {state_size, rs_size} -> {state_size, n_seqs}
    ggml_tensor * output_states = get_state_rows(ctx0, states, state_copy_main);
    ggml_build_forward_expand(gf, output_states);

    // copy extra states which won't be changed further (between n_seqs and n_rs)
    ggml_tensor * states_extra = ggml_get_rows(ctx0, states, state_copy_extra);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0,
            states_extra,
            ggml_view_2d(ctx0, s, state_size, (n_rs - n_seqs), s->nb[1], (rs_head + n_seqs)*s->nb[1])));

    return output_states;
}

static std::unique_ptr<llm_graph_input_rs> build_rs_inp_impl(
           ggml_context * ctx0,
     const llama_ubatch & ubatch,
    const llama_memory_recurrent_context * mctx_cur) {

    auto inp = std::make_unique<llm_graph_input_rs>(mctx_cur);

    const int64_t n_rs   = mctx_cur->get_n_rs();
    const int64_t n_seqs = ubatch.n_seqs;

    inp->s_copy = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rs);
    ggml_set_input(inp->s_copy);

    inp->s_copy_main  = ggml_view_1d(ctx0, inp->s_copy, n_seqs, 0);
    inp->s_copy_extra = ggml_view_1d(ctx0, inp->s_copy, n_rs - n_seqs, n_seqs * inp->s_copy->nb[0]);

    inp->head = mctx_cur->get_head();
    inp->rs_z = mctx_cur->get_rs_z();

    return inp;
}

llm_graph_input_rs * llm_graph_context::build_rs_inp() const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    auto inp = build_rs_inp_impl(ctx0, ubatch, mctx_cur);

    return (llm_graph_input_rs *) res->add_input(std::move(inp));
}

ggml_tensor * llm_graph_context::build_rs(
        llm_graph_input_rs * inp,
        ggml_tensor * s,
            int32_t   state_size,
            int32_t   n_seqs,
        const llm_graph_get_rows_fn & get_state_rows) const {
    const auto * kv_state = inp->mctx;

    return build_rs(s, inp->s_copy_main, inp->s_copy_extra, state_size, n_seqs,
                    kv_state->get_n_rs(), kv_state->get_head(), kv_state->get_size(), kv_state->get_rs_z(),
                    get_state_rows);
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_load(
    llm_graph_input_rs * inp,
    const llama_ubatch & ubatch,
                   int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;

    const int64_t n_seqs  = ubatch.n_seqs;

    ggml_tensor * token_shift_all = mctx_cur->get_r_l(il);

    ggml_tensor * token_shift = build_rs(
            inp, token_shift_all,
            hparams.n_embd_r(), n_seqs);

    token_shift = ggml_reshape_3d(ctx0, token_shift, hparams.n_embd, token_shift_count, n_seqs);

    return token_shift;
}

ggml_tensor * llm_graph_context::build_rwkv_token_shift_store(
         ggml_tensor * token_shift,
  const llama_ubatch & ubatch,
                 int   il) const {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const auto token_shift_count = hparams.token_shift_count;
    const auto n_embd = hparams.n_embd;

    const int64_t n_seqs = ubatch.n_seqs;

    const auto kv_head = mctx_cur->get_head();

    return ggml_cpy(
        ctx0,
        ggml_view_1d(ctx0, token_shift, n_embd * n_seqs * token_shift_count, 0),
        ggml_view_1d(ctx0, mctx_cur->get_r_l(il), hparams.n_embd_r()*n_seqs, hparams.n_embd_r()*kv_head*ggml_element_size(mctx_cur->get_r_l(il)))
    );
}

llm_graph_input_mem_hybrid * llm_graph_context::build_inp_mem_hybrid() const {
    if (const auto * mctx_kvarn = dynamic_cast<const llama_memory_hybrid_kvarn_context *>(mctx)) {
        auto inp_rs   = build_rs_inp_impl(ctx0, ubatch, mctx_kvarn->get_recr());
        std::unique_ptr<llm_graph_input_attn_kv> inp_attn;
        if (mctx_kvarn->has_attn_normal()) {
            inp_attn = build_attn_inp_kvarn_filter_impl(
                    ctx0, ubatch, hparams, cparams,
                    mctx_kvarn->get_attn_normal(), mctx_kvarn->get_attn());
        } else {
            inp_attn = build_attn_inp_kvarn_impl(ctx0, ubatch, hparams, cparams, mctx_kvarn->get_attn());
        }

        auto inp = std::make_unique<llm_graph_input_mem_hybrid_kvarn>(
                cparams, std::move(inp_attn), std::move(inp_rs), mctx_kvarn);

        return (llm_graph_input_mem_hybrid *) res->add_input(std::move(inp));
    }

    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_kv_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_k * llm_graph_context::build_inp_mem_hybrid_k() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_context *>(mctx);

    auto inp_rs   = build_rs_inp_impl     (ctx0, ubatch, mctx_cur->get_recr());
    auto inp_attn = build_attn_inp_k_impl(ctx0, ubatch, hparams, cparams, mctx_cur->get_attn());

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_k>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid_k *) res->add_input(std::move(inp));
}

llm_graph_input_mem_hybrid_iswa * llm_graph_context::build_inp_mem_hybrid_iswa() const {
    const auto * mctx_cur = static_cast<const llama_memory_hybrid_iswa_context *>(mctx);

    auto inp_rs = build_rs_inp_impl(ctx0, ubatch, mctx_cur->get_recr());

    // build iswa attention input
    const auto * attn_ctx = mctx_cur->get_attn();

    auto inp_attn = std::make_unique<llm_graph_input_attn_kv_iswa>(hparams, cparams, attn_ctx);

    {
        inp_attn->self_k_idxs = attn_ctx->get_base()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs = attn_ctx->get_base()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_kq_mask = build_attn_inp_kq_mask(ctx0, attn_ctx->get_base(), ubatch, cparams);
        inp_attn->self_kq_mask_cnv = inp_attn->self_kq_mask;
    }

    {
        inp_attn->self_k_idxs_swa = attn_ctx->get_swa()->build_input_k_idxs(ctx0, ubatch);
        inp_attn->self_v_idxs_swa = attn_ctx->get_swa()->build_input_v_idxs(ctx0, ubatch);

        inp_attn->self_kq_mask_swa = build_attn_inp_kq_mask(ctx0, attn_ctx->get_swa(), ubatch, cparams);
        inp_attn->self_kq_mask_swa_cnv = inp_attn->self_kq_mask_swa;
    }

    auto inp = std::make_unique<llm_graph_input_mem_hybrid_iswa>(cparams, std::move(inp_attn), std::move(inp_rs), mctx_cur);

    return (llm_graph_input_mem_hybrid_iswa *) res->add_input(std::move(inp));
}

void llm_graph_context::build_dense_out(
    ggml_tensor * dense_2,
    ggml_tensor * dense_2_b,
    ggml_tensor * dense_3) const {
    if (!cparams.embeddings || !(dense_2 || dense_2_b || dense_3)) {
        return;
    }
    ggml_tensor * cur = res->t_embd_pooled != nullptr ? res->t_embd_pooled : res->t_embd;
    GGML_ASSERT(cur != nullptr && "missing t_embd_pooled/t_embd");

    if (dense_2) {
        cur = ggml_mul_mat(ctx0, dense_2, cur);
    }
    if (dense_2_b) {
        cur = ggml_add(ctx0, cur, dense_2_b);
    }
    if (dense_3) {
        cur = ggml_mul_mat(ctx0, dense_3, cur);
    }
    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;
    ggml_build_forward_expand(gf, cur);
}


void llm_graph_context::build_pooling(
        ggml_tensor * cls,
        ggml_tensor * cls_b,
        ggml_tensor * cls_out,
        ggml_tensor * cls_out_b,
        ggml_tensor * cls_norm) const {
    if (!cparams.embeddings) {
        return;
    }

    ggml_tensor * inp = res->t_embd;

    //// find result_norm tensor for input
    //for (int i = ggml_graph_n_nodes(gf) - 1; i >= 0; --i) {
    //    inp = ggml_graph_node(gf, i);
    //    if (strcmp(inp->name, "result_norm") == 0 || strcmp(inp->name, "result_embd") == 0) {
    //        break;
    //    }

    //    inp = nullptr;
    //}

    GGML_ASSERT(inp != nullptr && "missing result_norm/result_embd tensor");

    ggml_tensor * cur;

    switch (pooling_type) {
        case LLAMA_POOLING_TYPE_NONE:
            {
                cur = inp;
            } break;
        case LLAMA_POOLING_TYPE_MEAN:
            {
                ggml_tensor * inp_mean = build_inp_mean();
                cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
            } break;
        case LLAMA_POOLING_TYPE_CLS:
        case LLAMA_POOLING_TYPE_LAST:
            {
                ggml_tensor * inp_cls = build_inp_cls();
                cur = ggml_get_rows(ctx0, inp, inp_cls);
            } break;
        case LLAMA_POOLING_TYPE_RANK:
            {
                if (arch == LLM_ARCH_MODERN_BERT) {
                    // modern bert gte reranker builds mean first then applies prediction head and classifier
                    // https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modular_modernbert.py#L1404-1411
                    ggml_tensor * inp_mean = build_inp_mean();
                    cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
                } else {
                    ggml_tensor * inp_cls = build_inp_cls();
                    cur = ggml_get_rows(ctx0, inp, inp_cls);
                }

                // classification head
                // https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/roberta/modeling_roberta.py#L1566
                if (cls) {
                    cur = ggml_mul_mat(ctx0, cls, cur);
                    if (cls_b) {
                        cur = ggml_add(ctx0, cur, cls_b);
                    }
                    if (arch == LLM_ARCH_MODERN_BERT) {
                        cur = ggml_gelu(ctx0, cur);
                    } else {
                        cur = ggml_tanh(ctx0, cur);
                    }
                    if (cls_norm) {
                        // head norm
                        cur = build_norm(cur, cls_norm, NULL, LLM_NORM, -1);
                    }
                }

                // some models don't have `cls_out`, for example: https://huggingface.co/jinaai/jina-reranker-v1-tiny-en
                // https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/blob/cb5347e43979c3084a890e3f99491952603ae1b7/modeling_bert.py#L884-L896
                // Single layer classification head (direct projection)
                // https://github.com/huggingface/transformers/blob/f4fc42216cd56ab6b68270bf80d811614d8d59e4/src/transformers/models/bert/modeling_bert.py#L1476
                if (cls_out) {
                    cur = ggml_mul_mat(ctx0, cls_out, cur);
                    if (cls_out_b) {
                        cur = ggml_add(ctx0, cur, cls_out_b);
                    }
                }

                // softmax for qwen3 reranker
                if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_QWEN3VL) {
                    cur = ggml_soft_max(ctx0, cur);
                }
            } break;
        default:
            {
                GGML_ABORT("unknown pooling type");
            }
    }

    cb(cur, "result_embd_pooled", -1);
    res->t_embd_pooled = cur;

    ggml_build_forward_expand(gf, cur);
}

void llm_graph_context::build_sampling() const {
    if (samplers.empty() || !res->t_logits) {
        return;
    }

    std::array<ggml_tensor *, 2> outs;
    outs[0] = res->t_logits;

    auto inp_sampling = std::make_unique<llm_graph_input_sampling>(samplers);
    res->add_input(std::move(inp_sampling));

    std::map<llama_seq_id, int32_t> seq_to_logit_row;
    int32_t logit_row_idx = 0;

    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (ubatch.output[i]) {
            llama_seq_id seq_id = ubatch.seq_id[i][0];
            seq_to_logit_row[seq_id] = logit_row_idx;
            logit_row_idx++;
        }
    }

    // res->t_logits will contain logits for all tokens that want the logits calculated (logits=1 or output=1)
    GGML_ASSERT(res->t_logits != nullptr && "missing t_logits tensor");

    // add a dummy row of logits
    // this trick makes the graph static, regardless of which samplers are activated
    // this is important in order to minimize graph reallocations
    ggml_tensor * logits_t = ggml_pad(ctx0, res->t_logits, 0, 1, 0, 0);

    for (const auto & [seq_id, sampler] : samplers) {
        const auto it = seq_to_logit_row.find(seq_id);

        // inactive samplers always work on the first row
        const auto row_idx = it != seq_to_logit_row.end() ? it->second : 0;
        const int i_out    = it != seq_to_logit_row.end() ? 1          : 0;

        ggml_tensor * logits_seq = ggml_view_1d(ctx0, logits_t, logits_t->ne[0], row_idx * logits_t->nb[1]);
        ggml_format_name(logits_seq, "logits_seq_%d", seq_id);

        struct llama_sampler_data data = {
            /*.logits      =*/ logits_seq,
            /*.probs       =*/ nullptr,
            /*.sampled     =*/ nullptr,
            /*.candidates  =*/ nullptr,
        };

        assert(sampler->iface->backend_apply);
        sampler->iface->backend_apply(sampler, ctx0, gf, &data);

        if (data.sampled != nullptr) {
            res->t_sampled[seq_id] = data.sampled;
            outs[1] = data.sampled;
            ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
        }

        if (data.probs != nullptr) {
            res->t_sampled_probs[seq_id] = data.probs;
            outs[1] = data.probs;
            ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
        }

        if (data.logits != nullptr) {
            res->t_sampled_logits[seq_id] = data.logits;
            outs[1] = data.logits;
            ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
        }

        if (data.candidates != nullptr) {
            res->t_candidates[seq_id] = data.candidates;
            outs[1] = data.candidates;
            ggml_build_forward_select(gf, outs.data(), outs.size(), i_out);
        }
    }

    // TODO: Call llama_sampler_accept_ggml after all samplers have been applied.
    /*
    for (const auto & [seq_id, sampler] : samplers) {
        if (auto it = res->t_sampled.find(seq_id); it != res->t_sampled.end()) {
            ggml_tensor * selected_token = it->second;
            if (selected_token != nullptr) {
                llama_sampler_accept_ggml(sampler, ctx0, gf, selected_token);
            }
        }
    }
    */
}

int32_t llama_relative_position_bucket(llama_pos x, llama_pos y, uint64_t n_buckets, bool bidirectional) {
    // TODO move to hparams if a T5 variant appears that uses a different value
    const int64_t max_distance = 128;

    if (bidirectional) {
        n_buckets >>= 1;
    }

    const int64_t max_exact = n_buckets >> 1;

    int32_t relative_position = x - y;
    int32_t relative_bucket = 0;

    if (bidirectional) {
        relative_bucket += (relative_position > 0) * n_buckets;
        relative_position = std::abs(relative_position);
    } else {
        relative_position = -std::min<int32_t>(relative_position, 0);
    }

    int32_t relative_position_if_large = floorf(max_exact + logf(1.0 * relative_position / max_exact) * (n_buckets - max_exact) / log(1.0 * max_distance / max_exact));
    relative_position_if_large = std::min<int32_t>(relative_position_if_large, n_buckets - 1);
    relative_bucket += (relative_position < max_exact ? relative_position : relative_position_if_large);

    return relative_bucket;
}
