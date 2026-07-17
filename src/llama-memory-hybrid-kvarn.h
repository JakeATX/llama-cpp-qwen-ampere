#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-kvarn.h"
#include "llama-memory.h"
#include "llama-memory-recurrent.h"

#include <memory>
#include <vector>

// Hybrid memory for architectures that mix recurrent layers with full
// attention layers, using KVarN for the attention side.

enum class llama_kvarn_qwen_hybrid_policy {
    normal_hybrid,
    experimental_kvarn,
};

constexpr llama_kvarn_qwen_hybrid_policy llama_kvarn_choose_qwen_hybrid_policy(bool force_experimental) {
    return force_experimental ? llama_kvarn_qwen_hybrid_policy::experimental_kvarn
                              : llama_kvarn_qwen_hybrid_policy::normal_hybrid;
}

class llama_memory_hybrid_kvarn : public llama_memory_i {
public:
    llama_memory_hybrid_kvarn(
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
    const layer_filter_cb & filter_attn = nullptr,
    const layer_filter_cb & filter_recr = nullptr,
    const layer_filter_cb & filter_attn_normal = nullptr);

    ~llama_memory_hybrid_kvarn() = default;

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    llama_kv_cache_kvarn * get_mem_attn() const;
    llama_kv_cache * get_mem_attn_normal() const;
    bool has_mem_attn_normal() const;
    llama_memory_recurrent * get_mem_recr() const;

private:
    const llama_hparams & hparams;

    const std::unique_ptr<llama_kv_cache_kvarn> mem_attn;
    const std::unique_ptr<llama_kv_cache> mem_attn_normal;
    const std::unique_ptr<llama_memory_recurrent> mem_recr;
};

class llama_memory_hybrid_kvarn_context : public llama_memory_context_i {
public:
    using slot_info_vec_t = llama_kv_cache_kvarn::slot_info_vec_t;

    explicit llama_memory_hybrid_kvarn_context(llama_memory_status status);
    explicit llama_memory_hybrid_kvarn_context(llama_memory_hybrid_kvarn * mem);
    llama_memory_hybrid_kvarn_context(
        llama_memory_hybrid_kvarn * mem,
              llama_context       * lctx,
                       bool         optimize);
    llama_memory_hybrid_kvarn_context(
              llama_memory_hybrid_kvarn * mem,
                  slot_info_vec_t         sinfos_attn,
       llama_kv_cache::slot_info_vec_t     sinfos_attn_normal,
        std::vector<llama_ubatch>         ubatches);

    ~llama_memory_hybrid_kvarn_context() = default;

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    const llama_kv_cache_kvarn_context * get_attn() const;
    const llama_kv_cache_context * get_attn_normal() const;
    bool has_attn_normal() const;
    const llama_memory_recurrent_context * get_recr() const;

private:
    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    const llama_memory_context_ptr ctx_attn;
    const llama_memory_context_ptr ctx_attn_normal;
    const llama_memory_context_ptr ctx_recr;

    const llama_memory_status status;
};
