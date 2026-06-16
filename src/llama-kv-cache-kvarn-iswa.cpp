#include "llama-kv-cache-kvarn-iswa.h"

#include "llama-batch.h"
#include "llama-hparams.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-kvarn-ubatch.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstring>
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

static bool kvarn_iswa_prepare_trace_enabled() {
    const char * env = std::getenv("LLAMA_KVARN_ISWA_PREPARE_TRACE");
    return env != nullptr && std::strcmp(env, "0") != 0;
}

llama_kv_cache_kvarn_iswa::llama_kv_cache_kvarn_iswa(
        const llama_model & model,
        llama_kvarn_params kvarn_params,
                ggml_type   swa_type_k,
                ggml_type   swa_type_v,
                     bool   swa_v_trans,
                     bool   offload,
                     bool   swa_full,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_ubatch,
                 uint32_t   n_pad,
    const layer_filter_cb & filter,
    const layer_filter_cb & filter_full_normal_arg,
    const  layer_reuse_cb & reuse) :
    hparams(model.hparams),
    unified(unified) {

    const layer_filter_cb filter_base = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return !hparams.is_swa(il);
    };

    const layer_filter_cb filter_swa = [&](int32_t il) {
        return hparams.is_swa(il);
    };

    const bool has_diagnostic_full_normal = bool(filter_full_normal_arg);

    const uint32_t size_base = kv_size;
    uint32_t size_swa = GGML_PAD(std::min(size_base, hparams.n_swa*(unified ? n_seq_max : 1) + n_ubatch), 256);

    if (swa_full) {
        LLAMA_LOG_WARN("%s: using full-size SWA cache (ref: %s)\n",
                __func__, "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
        size_swa = size_base;
    }

    LLAMA_LOG_INFO("%s: creating non-SWA KVarN KV cache, size = %u cells\n", __func__, size_base);

    kv_base = std::make_unique<llama_kv_cache_kvarn>(
            &model,
            hparams,
            kvarn_params,
            offload,
            size_base,
            n_seq_max,
            n_pad,
            filter_base,
            reuse);

    LLAMA_LOG_INFO("%s: creating     SWA KV cache, size = %u cells\n", __func__, size_swa);

    kv_swa = std::make_unique<llama_kv_cache>(
            model,
            hparams,
            swa_type_k,
            swa_type_v,
            swa_v_trans,
            offload,
            unified,
            size_swa,
            n_seq_max,
            n_pad,
            hparams.n_swa,
            hparams.swa_type,
            nullptr,
            filter_swa,
            reuse,
            nullptr);

    if (has_diagnostic_full_normal) {
        LLAMA_LOG_WARN(
                "%s: diagnostic KVarN+ISWA layer filter enabled; creating normal full-attention fallback KV cache, size = %u cells\n",
                __func__, size_base);

        kv_full_normal = std::make_unique<llama_kv_cache>(
                model,
                hparams,
                swa_type_k,
                swa_type_v,
                swa_v_trans,
                offload,
                unified,
                size_base,
                n_seq_max,
                n_pad,
                0,
                LLAMA_SWA_TYPE_NONE,
                nullptr,
                filter_full_normal_arg,
                reuse,
                nullptr);
    }
}

llama_memory_context_ptr llama_kv_cache_kvarn_iswa::init_batch(
        llama_batch_allocr & balloc,
        uint32_t n_ubatch,
        bool embd_all) {
    GGML_UNUSED(embd_all);

    balloc.split_reset();

    std::vector<llama_ubatch> ubatches;
    bool invalid_debug_ubatch = false;
    const uint32_t n_sink_tokens = kv_base->get_sink_tokens();
    const uint32_t n_tail_tokens = kv_base->get_tail_tokens();
    const uint32_t max_kvarn_ubatch = kvarn_ubatch_limit(n_ubatch, invalid_debug_ubatch);
    if (invalid_debug_ubatch) {
        LLAMA_LOG_ERROR("%s: KVarN debug ubatch override must be a positive integer: LLAMA_KVARN_DEBUG_UBATCH=%s\n",
                __func__, std::getenv("LLAMA_KVARN_DEBUG_UBATCH"));
        return std::make_unique<llama_kv_cache_kvarn_iswa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }
    if (n_tail_tokens == 0) {
        LLAMA_LOG_ERROR("%s: KVarN requires a non-empty tail ring\n", __func__);
        return std::make_unique<llama_kv_cache_kvarn_iswa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    while (true) {
        const uint32_t n_kvarn_ubatch =
            kvarn_tail_safe_ubatch_limit(balloc, max_kvarn_ubatch, n_sink_tokens, n_tail_tokens);
        auto ubatch = kvarn_batch_is_single_seq_contiguous(balloc)
            ? balloc.split_simple(n_kvarn_ubatch)
            : balloc.split_equal(n_kvarn_ubatch, !unified);
        if (ubatch.n_tokens == 0) {
            break;
        }

        ubatches.push_back(std::move(ubatch));
    }

    if (balloc.get_n_used() < balloc.get_n_tokens()) {
        return std::make_unique<llama_kv_cache_kvarn_iswa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    const auto t_base0 = std::chrono::steady_clock::now();
    auto sinfos_base = kv_base->prepare(ubatches);
    const auto t_base1 = std::chrono::steady_clock::now();
    if (sinfos_base.empty()) {
        LLAMA_LOG_ERROR("%s: failed to prepare KVarN base ubatches\n", __func__);
        return std::make_unique<llama_kv_cache_kvarn_iswa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    const auto t_swa0 = std::chrono::steady_clock::now();
    auto sinfos_swa = kv_swa->prepare(ubatches);
    const auto t_swa1 = std::chrono::steady_clock::now();
    if (sinfos_swa.empty()) {
        LLAMA_LOG_ERROR("%s: failed to prepare SWA ubatches\n", __func__);
        return std::make_unique<llama_kv_cache_kvarn_iswa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    llama_kv_cache::slot_info_vec_t sinfos_full_normal;
    if (kv_full_normal) {
        sinfos_full_normal = kv_full_normal->prepare(ubatches);
        if (sinfos_full_normal.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare normal full-attention fallback ubatches\n", __func__);
            return std::make_unique<llama_kv_cache_kvarn_iswa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }
    }

    if (kvarn_iswa_prepare_trace_enabled()) {
        const auto base_us = std::chrono::duration_cast<std::chrono::microseconds>(t_base1 - t_base0).count();
        const auto swa_us  = std::chrono::duration_cast<std::chrono::microseconds>(t_swa1  - t_swa0 ).count();
        uint32_t n_tokens_total = 0;
        for (const llama_ubatch & ub : ubatches) {
            n_tokens_total += ub.n_tokens;
        }
        LLAMA_LOG_INFO("%s: KVarN+ISWA prepare trace: ubatches=%zu tokens=%u base_us=%lld swa_us=%lld\n",
                __func__, ubatches.size(), n_tokens_total, (long long) base_us, (long long) swa_us);
    }

    return std::make_unique<llama_kv_cache_kvarn_iswa_context>(
            this, std::move(sinfos_base), std::move(sinfos_swa), std::move(sinfos_full_normal), std::move(ubatches));
}

llama_memory_context_ptr llama_kv_cache_kvarn_iswa::init_full() {
    return std::make_unique<llama_kv_cache_kvarn_iswa_context>(this);
}

llama_memory_context_ptr llama_kv_cache_kvarn_iswa::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_kv_cache_kvarn_iswa_context>(this, lctx, optimize);
}

bool llama_kv_cache_kvarn_iswa::get_can_shift() const {
    return kv_base->get_can_shift() &&
           kv_swa->get_can_shift() &&
           (!kv_full_normal || kv_full_normal->get_can_shift()) &&
           kv_base->get_size() == kv_swa->get_size();
}

void llama_kv_cache_kvarn_iswa::clear(bool data) {
    kv_base->clear(data);
    kv_swa ->clear(data);
    if (kv_full_normal) {
        kv_full_normal->clear(data);
    }
}

bool llama_kv_cache_kvarn_iswa::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    bool res = true;
    res = res & kv_base->seq_rm(seq_id, p0, p1);
    res = res & kv_swa ->seq_rm(seq_id, p0, p1);
    if (kv_full_normal) {
        res = res & kv_full_normal->seq_rm(seq_id, p0, p1);
    }
    return res;
}

void llama_kv_cache_kvarn_iswa::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    kv_base->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    kv_swa ->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    if (kv_full_normal) {
        kv_full_normal->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }
}

void llama_kv_cache_kvarn_iswa::seq_keep(llama_seq_id seq_id) {
    kv_base->seq_keep(seq_id);
    kv_swa ->seq_keep(seq_id);
    if (kv_full_normal) {
        kv_full_normal->seq_keep(seq_id);
    }
}

void llama_kv_cache_kvarn_iswa::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    kv_base->seq_add(seq_id, p0, p1, shift);
    kv_swa ->seq_add(seq_id, p0, p1, shift);
    if (kv_full_normal) {
        kv_full_normal->seq_add(seq_id, p0, p1, shift);
    }
}

void llama_kv_cache_kvarn_iswa::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    kv_base->seq_div(seq_id, p0, p1, d);
    kv_swa ->seq_div(seq_id, p0, p1, d);
    if (kv_full_normal) {
        kv_full_normal->seq_div(seq_id, p0, p1, d);
    }
}

llama_pos llama_kv_cache_kvarn_iswa::seq_pos_min(llama_seq_id seq_id) const {
    return kv_swa->seq_pos_min(seq_id);
}

llama_pos llama_kv_cache_kvarn_iswa::seq_pos_max(llama_seq_id seq_id) const {
    return kv_swa->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_kvarn_iswa::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = kv_base->memory_breakdown();
    for (const auto & buft_size : kv_swa->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    if (kv_full_normal) {
        for (const auto & buft_size : kv_full_normal->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }
    return mb;
}

void llama_kv_cache_kvarn_iswa::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        kv_base->state_write(io, seq_id, flags);
        if (kv_full_normal) {
            kv_full_normal->state_write(io, seq_id, flags);
        }
    }

    kv_swa->state_write(io, seq_id, flags);
}

void llama_kv_cache_kvarn_iswa::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        kv_base->state_read(io, seq_id, flags);
        if (kv_full_normal) {
            kv_full_normal->state_read(io, seq_id, flags);
        }
    }

    kv_swa->state_read(io, seq_id, flags);
}

llama_kv_cache_kvarn * llama_kv_cache_kvarn_iswa::get_base() const {
    return kv_base.get();
}

llama_kv_cache * llama_kv_cache_kvarn_iswa::get_swa() const {
    return kv_swa.get();
}

llama_kv_cache * llama_kv_cache_kvarn_iswa::get_full_normal() const {
    return kv_full_normal.get();
}

bool llama_kv_cache_kvarn_iswa::has_full_normal() const {
    return kv_full_normal != nullptr;
}

llama_kv_cache_kvarn_iswa_context::llama_kv_cache_kvarn_iswa_context(llama_memory_status status) :
    status(status) {
}

llama_kv_cache_kvarn_iswa_context::llama_kv_cache_kvarn_iswa_context(llama_kv_cache_kvarn_iswa * kv) :
    ctx_base(kv->get_base()->init_full()),
    ctx_swa (kv->get_swa ()->init_full()),
    ctx_full_normal(kv->get_full_normal() ? kv->get_full_normal()->init_full() : nullptr),
    status(llama_memory_status_combine(
            llama_memory_status_combine(ctx_base->get_status(), ctx_swa->get_status()),
            ctx_full_normal ? ctx_full_normal->get_status() : LLAMA_MEMORY_STATUS_SUCCESS)) {
}

llama_kv_cache_kvarn_iswa_context::llama_kv_cache_kvarn_iswa_context(
        llama_kv_cache_kvarn_iswa * kv,
        llama_context             * lctx,
        bool                        optimize) :
    ctx_base(kv->get_base()->init_update(lctx, optimize)),
    ctx_swa (kv->get_swa ()->init_update(lctx, optimize)),
    ctx_full_normal(kv->get_full_normal() ? kv->get_full_normal()->init_update(lctx, optimize) : nullptr),
    status(llama_memory_status_combine(
            llama_memory_status_combine(ctx_base->get_status(), ctx_swa->get_status()),
            ctx_full_normal ? ctx_full_normal->get_status() : LLAMA_MEMORY_STATUS_SUCCESS)) {
}

llama_kv_cache_kvarn_iswa_context::llama_kv_cache_kvarn_iswa_context(
        llama_kv_cache_kvarn_iswa * kv,
        slot_info_vec_t             sinfos_base,
        slot_info_swa_vec_t         sinfos_swa,
        slot_info_swa_vec_t         sinfos_full_normal,
        std::vector<llama_ubatch>   ubatches) :
    ubatches(std::move(ubatches)),
    ctx_base(new llama_kv_cache_kvarn_context(kv->get_base(), std::move(sinfos_base), this->ubatches)),
    ctx_swa (new llama_kv_cache_context      (kv->get_swa (), std::move(sinfos_swa),  this->ubatches)),
    ctx_full_normal(kv->get_full_normal() ?
            llama_memory_context_ptr(new llama_kv_cache_context(
                    kv->get_full_normal(), std::move(sinfos_full_normal), this->ubatches)) :
            nullptr),
    status(llama_memory_status_combine(
            llama_memory_status_combine(ctx_base->get_status(), ctx_swa->get_status()),
            ctx_full_normal ? ctx_full_normal->get_status() : LLAMA_MEMORY_STATUS_SUCCESS)) {
}

bool llama_kv_cache_kvarn_iswa_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_base->next();
    ctx_swa ->next();
    if (ctx_full_normal) {
        ctx_full_normal->next();
    }

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_kvarn_iswa_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;
    res = res & ctx_base->apply();
    res = res & ctx_swa ->apply();
    if (ctx_full_normal) {
        res = res & ctx_full_normal->apply();
    }
    return res;
}

llama_memory_status llama_kv_cache_kvarn_iswa_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_kvarn_iswa_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

const llama_kv_cache_kvarn_context * llama_kv_cache_kvarn_iswa_context::get_base() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return static_cast<const llama_kv_cache_kvarn_context *>(ctx_base.get());
}

const llama_kv_cache_context * llama_kv_cache_kvarn_iswa_context::get_swa() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return static_cast<const llama_kv_cache_context *>(ctx_swa.get());
}

const llama_kv_cache_context * llama_kv_cache_kvarn_iswa_context::get_full_normal() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ctx_full_normal ? static_cast<const llama_kv_cache_context *>(ctx_full_normal.get()) : nullptr;
}
