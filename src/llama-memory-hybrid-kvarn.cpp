#include "llama-memory-hybrid-kvarn.h"

#include "llama-context.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-kvarn-ubatch.h"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <limits>

static uint32_t kvarn_ubatch_limit(uint32_t default_limit, bool & invalid_debug_override) {
    invalid_debug_override = false;
    const char * env = std::getenv("LLAMA_KVARN_DEBUG_UBATCH");
    if (env != nullptr) {
        char * end = nullptr;
        errno = 0;
        const long long value = std::strtoll(env, &end, 10);
        if (env[0] == '\0' || end == nullptr || *end != '\0' || errno == ERANGE ||
                value <= 0 || value > (long long) std::numeric_limits<uint32_t>::max()) {
            invalid_debug_override = true;
            return 0;
        }
        return uint32_t(value);
    }

    return std::max<uint32_t>(1, default_limit);
}

llama_memory_hybrid_kvarn::llama_memory_hybrid_kvarn(
        const llama_model & model,
        llama_kvarn_params params,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_attn_normal) :
    hparams(model.hparams),
    mem_attn(new llama_kv_cache_kvarn(
        &model,
        model.hparams,
        params,
        offload,
        kv_size,
        n_seq_max,
        n_pad,
        filter_attn == nullptr ?
            [&](int32_t il) { return !hparams.is_recr(il); }
            : filter_attn
    )),
    mem_attn_normal(filter_attn_normal ?
        new llama_kv_cache(
            model,
            model.hparams,
            type_k,
            type_v,
            v_trans,
            offload,
            unified,
            kv_size,
            n_seq_max,
            n_pad,
            0,
            LLAMA_SWA_TYPE_NONE,
            nullptr,
            filter_attn_normal,
            nullptr,
            nullptr)
        : nullptr
    ),
    mem_recr(new llama_memory_recurrent(
        model,
        type_r,
        type_s,
        offload,
        rs_size,
        n_seq_max,
        n_rs_seq,
        filter_recr == nullptr ?
            [&](int32_t il) { return hparams.is_recr(il); }
            : filter_recr
    )) {}

llama_memory_context_ptr llama_memory_hybrid_kvarn::init_batch(
        llama_batch_allocr & balloc,
        uint32_t n_ubatch,
        bool embd_all) {
    GGML_UNUSED(embd_all);

    balloc.split_reset();

    std::vector<llama_ubatch> ubatches;
    bool invalid_debug_ubatch = false;
    const uint32_t n_sink_tokens = mem_attn->get_sink_tokens();
    const uint32_t n_tail_tokens = mem_attn->get_tail_tokens();
    const uint32_t default_kvarn_ubatch = n_ubatch;
    const uint32_t max_kvarn_ubatch = kvarn_ubatch_limit(default_kvarn_ubatch, invalid_debug_ubatch);
    if (invalid_debug_ubatch) {
        LLAMA_LOG_ERROR("%s: KVarN debug ubatch override must be a positive integer: LLAMA_KVARN_DEBUG_UBATCH=%s\n",
                __func__, std::getenv("LLAMA_KVARN_DEBUG_UBATCH"));
        return std::make_unique<llama_memory_hybrid_kvarn_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }
    if (n_tail_tokens == 0) {
        LLAMA_LOG_ERROR("%s: KVarN requires a non-empty tail ring\n", __func__);
        return std::make_unique<llama_memory_hybrid_kvarn_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    while (true) {
        const uint32_t n_kvarn_ubatch =
            kvarn_tail_safe_ubatch_limit(balloc, max_kvarn_ubatch, n_sink_tokens, n_tail_tokens);
        // Recurrent slots require equal_seqs ubatches; split_simple is only safe
        // for attention-only KVarN paths (see llama-memory-recurrent.cpp).
        const bool need_equal_seqs = mem_recr != nullptr;
        auto ubatch = (!need_equal_seqs && kvarn_batch_is_single_seq_contiguous(balloc))
            ? balloc.split_simple(n_kvarn_ubatch)
            : balloc.split_equal(n_kvarn_ubatch, true);
        if (ubatch.n_tokens == 0) {
            break;
        }

        ubatches.push_back(std::move(ubatch));
    }

    if (balloc.get_n_used() < balloc.get_n_tokens()) {
        return std::make_unique<llama_memory_hybrid_kvarn_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    if (!mem_recr->prepare(ubatches)) {
        LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
        return std::make_unique<llama_memory_hybrid_kvarn_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    const auto sinfos_attn = mem_attn->prepare(ubatches);
    if (sinfos_attn.empty()) {
        LLAMA_LOG_ERROR("%s: failed to prepare KVarN attention ubatches\n", __func__);
        return std::make_unique<llama_memory_hybrid_kvarn_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    llama_kv_cache::slot_info_vec_t sinfos_attn_normal;
    if (mem_attn_normal) {
        sinfos_attn_normal = mem_attn_normal->prepare(ubatches);
        if (sinfos_attn_normal.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare normal attention ubatches for KVarN layer-filter fallback\n", __func__);
            return std::make_unique<llama_memory_hybrid_kvarn_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }
    }

    return std::make_unique<llama_memory_hybrid_kvarn_context>(
            this, std::move(sinfos_attn), std::move(sinfos_attn_normal), std::move(ubatches));
}

llama_memory_context_ptr llama_memory_hybrid_kvarn::init_full() {
    return std::make_unique<llama_memory_hybrid_kvarn_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_kvarn::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_kvarn_context>(this, lctx, optimize);
}

bool llama_memory_hybrid_kvarn::get_can_shift() const {
    return mem_attn->get_can_shift() && (!mem_attn_normal || mem_attn_normal->get_can_shift());
}

void llama_memory_hybrid_kvarn::clear(bool data) {
    mem_attn->clear(data);
    if (mem_attn_normal) {
        mem_attn_normal->clear(data);
    }
    mem_recr->clear(data);
}

bool llama_memory_hybrid_kvarn::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (!mem_attn->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    if (!mem_recr->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    if (mem_attn_normal && !mem_attn_normal->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    return true;
}

void llama_memory_hybrid_kvarn::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    mem_attn->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    if (mem_attn_normal) {
        mem_attn_normal->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }
    mem_recr->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_hybrid_kvarn::seq_keep(llama_seq_id seq_id) {
    mem_attn->seq_keep(seq_id);
    if (mem_attn_normal) {
        mem_attn_normal->seq_keep(seq_id);
    }
    mem_recr->seq_keep(seq_id);
}

void llama_memory_hybrid_kvarn::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    mem_attn->seq_add(seq_id, p0, p1, shift);
    if (mem_attn_normal) {
        mem_attn_normal->seq_add(seq_id, p0, p1, shift);
    }
    mem_recr->seq_add(seq_id, p0, p1, shift);
}

void llama_memory_hybrid_kvarn::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    mem_attn->seq_div(seq_id, p0, p1, d);
    if (mem_attn_normal) {
        mem_attn_normal->seq_div(seq_id, p0, p1, d);
    }
    mem_recr->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_hybrid_kvarn::seq_pos_min(llama_seq_id seq_id) const {
    llama_pos res = std::max(mem_attn->seq_pos_min(seq_id), mem_recr->seq_pos_min(seq_id));
    if (mem_attn_normal) {
        res = std::max(res, mem_attn_normal->seq_pos_min(seq_id));
    }
    return res;
}

llama_pos llama_memory_hybrid_kvarn::seq_pos_max(llama_seq_id seq_id) const {
    llama_pos res = std::min(mem_attn->seq_pos_max(seq_id), mem_recr->seq_pos_max(seq_id));
    if (mem_attn_normal) {
        res = std::min(res, mem_attn_normal->seq_pos_max(seq_id));
    }
    return res;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_kvarn::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = mem_attn->memory_breakdown();
    if (mem_attn_normal) {
        for (const auto & buft_size : mem_attn_normal->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }
    for (const auto & buft_size : mem_recr->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

void llama_memory_hybrid_kvarn::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_write(io, seq_id, flags);
        if (mem_attn_normal) {
            mem_attn_normal->state_write(io, seq_id, flags);
        }
    }
    mem_recr->state_write(io, seq_id, flags);
}

void llama_memory_hybrid_kvarn::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (seq_id != -1 || flags != 0) {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            mem_attn->state_read(io, seq_id, flags);
            if (mem_attn_normal) {
                mem_attn_normal->state_read(io, seq_id, flags);
            }
        }
        mem_recr->state_read(io, seq_id, flags);
        return;
    }

    try {
        mem_attn->state_read(io, seq_id, flags);
        if (mem_attn_normal) {
            mem_attn_normal->state_read(io, seq_id, flags);
        }
        mem_recr->state_read(io, seq_id, flags);
    } catch (...) {
        clear(true);
        throw;
    }
}

llama_kv_cache_kvarn * llama_memory_hybrid_kvarn::get_mem_attn() const {
    return mem_attn.get();
}

llama_kv_cache * llama_memory_hybrid_kvarn::get_mem_attn_normal() const {
    return mem_attn_normal.get();
}

bool llama_memory_hybrid_kvarn::has_mem_attn_normal() const {
    return mem_attn_normal != nullptr;
}

llama_memory_recurrent * llama_memory_hybrid_kvarn::get_mem_recr() const {
    return mem_recr.get();
}

llama_memory_hybrid_kvarn_context::llama_memory_hybrid_kvarn_context(llama_memory_status status) :
    status(status) {
}

llama_memory_hybrid_kvarn_context::llama_memory_hybrid_kvarn_context(llama_memory_hybrid_kvarn * mem) :
    ctx_attn(mem->get_mem_attn()->init_full()),
    ctx_attn_normal(mem->has_mem_attn_normal() ? mem->get_mem_attn_normal()->init_full() : nullptr),
    ctx_recr(mem->get_mem_recr()->init_full()),
    status(llama_memory_status_combine(
            llama_memory_status_combine(ctx_attn->get_status(),
                    ctx_attn_normal ? ctx_attn_normal->get_status() : LLAMA_MEMORY_STATUS_NO_UPDATE),
            ctx_recr->get_status())) {
}

llama_memory_hybrid_kvarn_context::llama_memory_hybrid_kvarn_context(
        llama_memory_hybrid_kvarn * mem,
              llama_context       * lctx,
                       bool         optimize) :
    ctx_attn(mem->get_mem_attn()->init_update(lctx, optimize)),
    ctx_attn_normal(mem->has_mem_attn_normal() ? mem->get_mem_attn_normal()->init_update(lctx, optimize) : nullptr),
    ctx_recr(mem->get_mem_recr()->init_update(lctx, optimize)),
    status(llama_memory_status_combine(
            llama_memory_status_combine(ctx_attn->get_status(),
                    ctx_attn_normal ? ctx_attn_normal->get_status() : LLAMA_MEMORY_STATUS_NO_UPDATE),
            ctx_recr->get_status())) {
}

llama_memory_hybrid_kvarn_context::llama_memory_hybrid_kvarn_context(
              llama_memory_hybrid_kvarn * mem,
                  slot_info_vec_t         sinfos_attn,
       llama_kv_cache::slot_info_vec_t     sinfos_attn_normal,
        std::vector<llama_ubatch>         ubatches) :
    ubatches(std::move(ubatches)),
    ctx_attn(new llama_kv_cache_kvarn_context(mem->get_mem_attn(), std::move(sinfos_attn), this->ubatches)),
    ctx_attn_normal(mem->has_mem_attn_normal()
            ? llama_memory_context_ptr(new llama_kv_cache_context(
                    mem->get_mem_attn_normal(), std::move(sinfos_attn_normal), this->ubatches))
            : nullptr),
    ctx_recr(new llama_memory_recurrent_context(mem->get_mem_recr(), this->ubatches)),
    status(llama_memory_status_combine(
            llama_memory_status_combine(ctx_attn->get_status(),
                    ctx_attn_normal ? ctx_attn_normal->get_status() : LLAMA_MEMORY_STATUS_NO_UPDATE),
            ctx_recr->get_status())) {
}

bool llama_memory_hybrid_kvarn_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_attn->next();
    if (ctx_attn_normal) {
        ctx_attn_normal->next();
    }
    ctx_recr->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_hybrid_kvarn_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_attn->apply();
    if (ctx_attn_normal) {
        res = res & ctx_attn_normal->apply();
    }
    res = res & ctx_recr->apply();

    return res;
}

llama_memory_status llama_memory_hybrid_kvarn_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_hybrid_kvarn_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

const llama_kv_cache_kvarn_context * llama_memory_hybrid_kvarn_context::get_attn() const {
    return static_cast<const llama_kv_cache_kvarn_context *>(ctx_attn.get());
}

const llama_kv_cache_context * llama_memory_hybrid_kvarn_context::get_attn_normal() const {
    return static_cast<const llama_kv_cache_context *>(ctx_attn_normal.get());
}

bool llama_memory_hybrid_kvarn_context::has_attn_normal() const {
    return ctx_attn_normal != nullptr;
}

const llama_memory_recurrent_context * llama_memory_hybrid_kvarn_context::get_recr() const {
    return static_cast<const llama_memory_recurrent_context *>(ctx_recr.get());
}
