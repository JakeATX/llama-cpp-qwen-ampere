#include "arg.h"
#include "common.h"
#include "llama.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <vector>

static llama_context * make_ctx(const common_params & params, llama_model * model) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = 8;
    cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) (cparams.n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) (cparams.n_rs_seq + 1));
    return llama_init_from_model(model, cparams);
}

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, uint32_t count) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t pos = 0; pos < count; ++pos) {
        common_batch_add(batch, tokens[pos], pos, { 0 }, false);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, tok, pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static llama_memory_recurrent * get_recurrent_memory(llama_memory_t mem) {
    if (auto * recurrent = dynamic_cast<llama_memory_recurrent *>(mem)) {
        return recurrent;
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) {
        return hybrid->get_mem_recr();
    }
    if (auto * hybrid_iswa = dynamic_cast<llama_memory_hybrid_iswa *>(mem)) {
        return hybrid_iswa->get_mem_recr();
    }
    return nullptr;
}

static size_t recurrent_breakdown_bytes(const llama_memory_recurrent & mem) {
    size_t result = 0;
    for (const auto & entry : mem.memory_breakdown()) {
        result += entry.second;
    }
    return result;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s : skipping for non-recurrent model\n", __func__);
        return 0;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    llama_context * ctx_src = make_ctx(params, model);
    llama_context * ctx_dst = make_ctx(params, model);
    if (ctx_src == nullptr || ctx_dst == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        return 1;
    }

    llama_memory_recurrent * recurrent_real = get_recurrent_memory(llama_get_memory(ctx_src));
    if (recurrent_real == nullptr) {
        fprintf(stderr, "%s : unsupported recurrent memory composition\n", __func__);
        return 1;
    }
    if (recurrent_real->backend_buffer_bytes() == 0) {
        fprintf(stderr, "%s : real recurrent memory has no backing allocation\n", __func__);
        return 1;
    }

    {
        llama_model_params mparams_no_alloc = common_model_params_to_llama(params);
        mparams_no_alloc.no_alloc = true;
        mparams_no_alloc.use_mmap = false;
        llama_model_ptr model_no_alloc { llama_model_load_from_file(params.model.path.c_str(), mparams_no_alloc) };
        if (!model_no_alloc) {
            fprintf(stderr, "%s : failed to load no-alloc model\n", __func__);
            return 1;
        }

        llama_context_ptr ctx_no_alloc { make_ctx(params, model_no_alloc.get()) };
        if (!ctx_no_alloc) {
            fprintf(stderr, "%s : failed to create no-alloc context\n", __func__);
            return 1;
        }

        llama_memory_recurrent * recurrent_no_alloc = get_recurrent_memory(llama_get_memory(ctx_no_alloc.get()));
        if (recurrent_no_alloc == nullptr) {
            fprintf(stderr, "%s : unsupported no-alloc recurrent memory composition\n", __func__);
            return 1;
        }
        const size_t bytes_real     = recurrent_breakdown_bytes(*recurrent_real);
        const size_t bytes_no_alloc = recurrent_breakdown_bytes(*recurrent_no_alloc);
        if (recurrent_no_alloc->backend_buffer_bytes() != 0 ||
            bytes_no_alloc == 0 || bytes_no_alloc != bytes_real) {
            fprintf(stderr, "%s : recurrent no-alloc sizing mismatch (%zu physical, %zu projected, %zu real)\n",
                    __func__, recurrent_no_alloc->backend_buffer_bytes(), bytes_no_alloc, bytes_real);
            return 1;
        }
    }

    if (llama_n_rs_seq(ctx_src) == 0) {
        fprintf(stderr, "%s : skipping because n_rs_seq is disabled\n", __func__);
        llama_free(ctx_src);
        llama_free(ctx_dst);
        return 0;
    }

    std::vector<llama_token> tokens = common_tokenize(ctx_src, "The quick brown fox jumps", true);
    const uint32_t n_rs_seq = llama_n_rs_seq(ctx_src);
    if (tokens.size() > n_rs_seq + 1) {
        tokens.resize(n_rs_seq + 1);
    }
    if (tokens.size() < 2) {
        fprintf(stderr, "%s : not enough prompt tokens\n", __func__);
        return 1;
    }
    const uint32_t    n_tokens = tokens.size();
    const llama_token last_tok = tokens.back();
    const llama_pos   last_pos = (llama_pos) n_tokens - 2;

    // Decode the full prompt on the source, then roll back the last position.
    // Rollback leaves the recurrent memory in a snapshot state (rs_idx != 0).
    if (!decode_tokens(ctx_src, tokens, n_tokens)) {
        fprintf(stderr, "%s : failed to decode prompt\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, last_pos, -1)) {
        fprintf(stderr, "%s : rollback failed\n", __func__);
        return 1;
    }

    // Save the rolled-back state and restore it into a fresh context.
    common_prompt_checkpoint ckpt;
    ckpt.update_tgt(ctx_src, 0, 0);
    ckpt.load_tgt(ctx_dst, 0, 0);

    // Replay the rolled-back token on both contexts and compare logits.
    if (!decode_one(ctx_src, last_tok, last_pos) ||
        !decode_one(ctx_dst, last_tok, last_pos)) {
        fprintf(stderr, "%s : replay failed\n", __func__);
        return 1;
    }

    const float * logits_src = llama_get_logits_ith(ctx_src, 0);
    const float * logits_dst = llama_get_logits_ith(ctx_dst, 0);
    if (logits_src == nullptr || logits_dst == nullptr) {
        fprintf(stderr, "%s : missing logits\n", __func__);
        return 1;
    }

    constexpr float eps = 1e-5f;
    for (int i = 0; i < n_vocab; ++i) {
        if (std::fabs(logits_src[i] - logits_dst[i]) > eps) {
            fprintf(stderr, "%s : logits mismatch at token %d (%g != %g)\n",
                    __func__, i, (double) logits_src[i], (double) logits_dst[i]);
            return 1;
        }
    }

    // Repeat the load into a context that already has its own rollback state:
    // groups 1..n_rs_seq hold a *different* prompt's history, and rs_idx[0] is
    // non-zero at load time. The restore must wipe that state and still match.
    llama_context * ctx_dirty = make_ctx(params, model);
    if (ctx_dirty == nullptr) {
        fprintf(stderr, "%s : failed to init dirty ctx\n", __func__);
        return 1;
    }

    std::vector<llama_token> noise = tokens;
    for (auto & t : noise) {
        t = (t + 1) % n_vocab;
        if (t < 0) {
            t = 0;
        }
    }
    if (!decode_tokens(ctx_dirty, noise, n_tokens)) {
        fprintf(stderr, "%s : dirty prompt decode failed\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_dirty), 0, last_pos, -1)) {
        fprintf(stderr, "%s : dirty rollback failed\n", __func__);
        return 1;
    }

    ckpt.load_tgt(ctx_dirty, 0, 0);

    if (!decode_one(ctx_dirty, last_tok, last_pos)) {
        fprintf(stderr, "%s : dirty replay failed\n", __func__);
        return 1;
    }

    const float * logits_dirty = llama_get_logits_ith(ctx_dirty, 0);
    if (logits_dirty == nullptr) {
        fprintf(stderr, "%s : missing dirty logits\n", __func__);
        return 1;
    }

    for (int i = 0; i < n_vocab; ++i) {
        if (std::fabs(logits_src[i] - logits_dirty[i]) > eps) {
            fprintf(stderr, "%s : dirty-ctx logits mismatch at token %d (%g != %g)\n",
                    __func__, i, (double) logits_src[i], (double) logits_dirty[i]);
            return 1;
        }
    }

    llama_memory_t mem_src = llama_get_memory(ctx_src);
    if (last_pos > 0) {
        if (llama_memory_seq_rm(mem_src, -1, 0, last_pos) ||
            llama_memory_seq_pos_max(mem_src, 0) != last_pos) {
            fprintf(stderr, "%s : finite partial prefix was not rejected atomically\n", __func__);
            return 1;
        }
    }
    if (llama_memory_seq_rm(mem_src, -1, last_pos, last_pos + 1) ||
        llama_memory_seq_pos_max(mem_src, 0) != last_pos) {
        fprintf(stderr, "%s : finite tail-only removal was not rejected atomically\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(mem_src, -2, 0, last_pos + 1) ||
        llama_memory_seq_pos_max(mem_src, 0) != -1) {
        fprintf(stderr, "%s : finite whole-cache removal failed\n", __func__);
        return 1;
    }
    if (!decode_one(ctx_src, tokens[0], 0)) {
        fprintf(stderr, "%s : decode after finite whole-cache removal failed\n", __func__);
        return 1;
    }

    fprintf(stderr, "%s : recurrent rollback checkpoint restored successfully\n", __func__);
    llama_free(ctx_src);
    llama_free(ctx_dst);
    llama_free(ctx_dirty);
    return 0;
}
