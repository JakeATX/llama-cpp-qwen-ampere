#pragma once

#include "llama.h"
#include "llama-batch.h"
#include "llama-kv-cells.h"
#include "llama-memory.h"
#include "ggml.h"
#include "ggml-cpp.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <unordered_map>
#include <vector>

struct llama_hparams;
class llama_model;
class llama_batch_allocr;
class llama_io_read_i;
class llama_io_write_i;

struct llama_kvarn_layout {
    uint32_t head_dim;
    uint32_t group_size;
    uint32_t key_bits;
    uint32_t value_bits;

    size_t k_body_bytes;
    size_t v_body_bytes;
    size_t k_scale_floats;
    size_t v_scale_floats;
    size_t total_record_bytes;
};

struct llama_kvarn_body_record {
    llama_kvarn_layout layout;
    std::vector<uint8_t> k_body;
    std::vector<uint8_t> v_body;
    std::vector<float>   k_scales;
    std::vector<float>   v_scales;
};

llama_kvarn_layout llama_kvarn_make_layout(const llama_kvarn_params & params, uint32_t head_dim);

void llama_kvarn_hadamard_channels(
        const std::vector<float> & src,
        std::vector<float> & dst,
        uint32_t rows,
        uint32_t cols,
        bool channels_are_rows);

std::vector<float> llama_kvarn_hadamard_matrix(uint32_t d);

void llama_kvarn_pack_bits(const std::vector<uint8_t> & src, uint32_t bits, std::vector<uint8_t> & dst);
void llama_kvarn_unpack_bits(const std::vector<uint8_t> & src, uint32_t bits, size_t n_values, std::vector<uint8_t> & dst);

llama_kvarn_body_record llama_kvarn_store_reference(
        const llama_kvarn_params & params,
        uint32_t head_dim,
        const std::vector<float> & k_tile,
        const std::vector<float> & v_tile);

void llama_kvarn_dequant_reference(
        const llama_kvarn_body_record & record,
        std::vector<float> & k_tile,
        std::vector<float> & v_tile);

struct llama_kvarn_reference_cache_stats {
    uint32_t n_sink;
    uint32_t n_tail;
    uint32_t n_pending_body;
    uint32_t n_body_records;
    uint32_t n_tokens;
    size_t sink_tail_fp16_values;
    size_t body_packed_bytes;
    size_t scale_values;
};

struct llama_kvarn_memory_estimate {
    size_t fp16_sink_tail_bytes;
    size_t body_packed_bytes;
    size_t scale_bytes;
    size_t total_bytes;
};

llama_kvarn_memory_estimate llama_kvarn_estimate_memory(
        const llama_kvarn_params & params,
        const llama_hparams & hparams,
        uint32_t kv_size,
        const llama_memory_i::layer_filter_cb & filter = nullptr);

class llama_kvarn_reference_cache {
public:
    llama_kvarn_reference_cache(llama_kvarn_params params, uint32_t head_dim);

    void append_token(const std::vector<float> & k_token, const std::vector<float> & v_token);
    void clear();

    llama_kvarn_reference_cache_stats stats() const;

    const std::vector<llama_kvarn_body_record> & body_records() const;
    void materialize_tokens(std::vector<float> & k_tokens, std::vector<float> & v_tokens) const;

private:
    void seal_pending_group();

    llama_kvarn_params params;
    uint32_t head_dim;
    uint32_t n_tokens = 0;

    std::vector<ggml_fp16_t> sink_k;
    std::vector<ggml_fp16_t> sink_v;
    std::vector<ggml_fp16_t> tail_k;
    std::vector<ggml_fp16_t> tail_v;
    std::vector<float> pending_k;
    std::vector<float> pending_v;
    std::vector<llama_kvarn_body_record> records;
};

struct llama_kvarn_runtime_storage_stats {
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_tokens;
    uint32_t n_body_records;
    uint32_t n_pending_body;
    uint32_t n_sink;
    uint32_t n_tail;

    size_t fp16_sink_tail_values;
    size_t body_packed_bytes;
    size_t scale_values;
};

struct llama_kvarn_layer_view {
    uint32_t il;
    uint32_t n_head_kv;
    uint32_t n_records;
    uint32_t head_dim_k;
    uint32_t head_dim_v;
    llama_kvarn_layout layout_k;
    llama_kvarn_layout layout_v;

    ggml_tensor * sink_tail_k;
    ggml_tensor * sink_tail_v;
    ggml_tensor * body_k;
    ggml_tensor * body_v;
    ggml_tensor * scales_k;
    ggml_tensor * scales_v;
    ggml_tensor * pending_k;
    ggml_tensor * pending_v;
    ggml_tensor * attn_mixed_scratch; // F32 [worst-case mixed-attn scratch]
};

class llama_kv_cache_kvarn;

class llama_kv_cache_kvarn_context : public llama_memory_context_i {
public:
    struct slot_info {
        std::vector<uint32_t> idxs;

        bool empty() const {
            return idxs.empty();
        }

        size_t size() const {
            return idxs.size();
        }
    };

    using slot_info_vec_t = std::vector<slot_info>;

    llama_kv_cache_kvarn_context(llama_memory_status status);
    llama_kv_cache_kvarn_context(llama_kv_cache_kvarn * kv, slot_info_vec_t sinfos, std::vector<llama_ubatch> ubatches);

    bool next() override;
    bool apply() override;

    llama_memory_status get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    ggml_tensor * build_input_sink_tail_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    void set_input_sink_tail_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    ggml_tensor * build_input_body_plan(ggml_context * ctx, const llama_ubatch & ubatch) const;
    void set_input_body_plan(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    ggml_tensor * build_input_body_offsets(ggml_context * ctx, const llama_ubatch & ubatch) const;
    void set_input_body_offsets(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    ggml_tensor * build_input_tail_evict_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    void set_input_tail_evict_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;

    uint32_t get_size() const;

    ggml_tensor * cpy_sink_tail_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * idxs, int32_t il) const;
    ggml_tensor * cpy_sink_tail_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * idxs, int32_t il) const;
    ggml_tensor * cpy_tail_evict_pending_k(ggml_context * ctx, ggml_tensor * tail_idxs, ggml_tensor * offsets, int32_t il) const;
    ggml_tensor * cpy_tail_evict_pending_v(ggml_context * ctx, ggml_tensor * tail_idxs, ggml_tensor * offsets, int32_t il) const;

    llama_kvarn_layer_view get_layer_view(int32_t il) const;
    size_t body_store_scratch_floats(int32_t il) const;
    ggml_tensor * build_body_store_scratch(ggml_context * ctx, int32_t il) const;
    int64_t attn_mixed_scratch_floats_worst(int32_t il) const;
    ggml_tensor * build_attn_mixed_scratch(ggml_context * ctx, int32_t il, int64_t n_floats) const;
    ggml_tensor * view_k_body_record    (ggml_context * ctx, int32_t il, uint32_t ih, uint32_t record) const;
    ggml_tensor * view_v_body_record    (ggml_context * ctx, int32_t il, uint32_t ih, uint32_t record) const;
    ggml_tensor * view_k_scales_record  (ggml_context * ctx, int32_t il, uint32_t ih, uint32_t record) const;
    ggml_tensor * view_v_scales_record  (ggml_context * ctx, int32_t il, uint32_t ih, uint32_t record) const;
    ggml_tensor * view_k_body_record_span_heads(ggml_context * ctx, int32_t il, uint32_t record0, uint32_t n_records) const;
    ggml_tensor * view_v_body_record_span_heads(ggml_context * ctx, int32_t il, uint32_t record0, uint32_t n_records) const;
    ggml_tensor * view_k_scales_record_span_heads(ggml_context * ctx, int32_t il, uint32_t record0, uint32_t n_records) const;
    ggml_tensor * view_v_scales_record_span_heads(ggml_context * ctx, int32_t il, uint32_t record0, uint32_t n_records) const;
    ggml_tensor * store_k_body_record(
            ggml_context * ctx, ggml_tensor * k_tile, ggml_tensor * scratch, int32_t il, uint32_t ih, uint32_t record) const;
    ggml_tensor * store_v_body_record(
            ggml_context * ctx, ggml_tensor * v_tile, ggml_tensor * scratch, int32_t il, uint32_t ih, uint32_t record) const;
    ggml_tensor * store_kv_body_record(
            ggml_context * ctx, ggml_tensor * k_tile, ggml_tensor * v_tile, ggml_tensor * scratch, int32_t il, uint32_t ih, uint32_t record) const;
    ggml_tensor * store_kv_body_all_heads(
            ggml_context * ctx, ggml_tensor * k_tile, ggml_tensor * v_tile, ggml_tensor * scratch, int32_t il, uint32_t record) const;
    ggml_tensor * store_kv_body_records_all_heads(
            ggml_context * ctx, ggml_tensor * k_tiles, ggml_tensor * v_tiles, ggml_tensor * scratch,
            int32_t il, uint32_t record0, uint32_t n_records) const;
    ggml_tensor * store_k_body_record_from_pending(
            ggml_context * ctx, ggml_tensor * scratch, int32_t il, uint32_t ih, uint32_t record) const;
    ggml_tensor * store_v_body_record_from_pending(
            ggml_context * ctx, ggml_tensor * scratch, int32_t il, uint32_t ih, uint32_t record) const;
    ggml_tensor * store_kv_body_record_from_pending(
            ggml_context * ctx, ggml_tensor * scratch, int32_t il, uint32_t ih, uint32_t record) const;
    ggml_tensor * store_kv_body_all_heads_from_pending(
            ggml_context * ctx, ggml_tensor * scratch, int32_t il, uint32_t record) const;
    ggml_tensor * store_kv_body_records_from_pending(
            ggml_context * ctx, ggml_tensor * scratch, int32_t il, const std::vector<uint32_t> & records) const;

private:
    llama_memory_status status;
    llama_kv_cache_kvarn * kv = nullptr;

    uint32_t i_cur = 0;
    slot_info_vec_t sinfos;
    std::vector<llama_ubatch> ubatches;
};

class llama_kv_cache_kvarn : public llama_memory_i {
public:
    using slot_info = llama_kv_cache_kvarn_context::slot_info;
    using slot_info_vec_t = llama_kv_cache_kvarn_context::slot_info_vec_t;

    llama_kv_cache_kvarn(
            const llama_model * model,
            const llama_hparams & hparams,
            llama_kvarn_params params,
            bool offload,
            uint32_t kv_size,
            uint32_t n_seq_max,
            uint32_t n_pad,
            const layer_filter_cb & filter,
            const layer_reuse_cb & reuse = nullptr);

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
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    uint32_t get_size() const;
    uint32_t get_n_layer() const;
    uint32_t get_sink_tokens() const { return params.sink_tokens; }
    uint32_t get_tail_tokens() const { return params.tail_tokens; }
    uint32_t get_group_size() const { return params.group_size; }
    llama_kvarn_memory_estimate estimate() const;

    llama_kvarn_runtime_storage_stats storage_stats() const;
    size_t backend_tensor_bytes() const;

    void append_layer_tokens_reference(
            uint32_t il,
            const std::vector<float> & k_tokens,
            const std::vector<float> & v_tokens,
            uint32_t n_tokens);
    void materialize_layer_tokens_reference(
            uint32_t il,
            std::vector<float> & k_tokens,
            std::vector<float> & v_tokens) const;

    ggml_tensor * build_input_sink_tail_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    void set_input_sink_tail_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    ggml_tensor * build_input_body_plan(ggml_context * ctx, const llama_ubatch & ubatch) const;
    void set_input_body_plan(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    ggml_tensor * build_input_body_offsets(ggml_context * ctx, const llama_ubatch & ubatch) const;
    void set_input_body_offsets(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    ggml_tensor * build_input_tail_evict_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    void set_input_tail_evict_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;

    ggml_tensor * cpy_sink_tail_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * idxs, int32_t il) const;
    ggml_tensor * cpy_sink_tail_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * idxs, int32_t il) const;
    ggml_tensor * cpy_tail_evict_pending_k(ggml_context * ctx, ggml_tensor * tail_idxs, ggml_tensor * offsets, int32_t il) const;
    ggml_tensor * cpy_tail_evict_pending_v(ggml_context * ctx, ggml_tensor * tail_idxs, ggml_tensor * offsets, int32_t il) const;

    llama_kvarn_layer_view get_layer_view(int32_t il) const;
    size_t body_store_scratch_floats(int32_t il) const;
    ggml_tensor * build_body_store_scratch(ggml_context * ctx, int32_t il) const;
    int64_t attn_mixed_scratch_floats_worst(int32_t il) const;
    ggml_tensor * build_attn_mixed_scratch(ggml_context * ctx, int32_t il, int64_t n_floats) const;
    ggml_tensor * view_k_body_record    (ggml_context * ctx, int32_t il, uint32_t ih, uint32_t record) const;
    ggml_tensor * view_v_body_record    (ggml_context * ctx, int32_t il, uint32_t ih, uint32_t record) const;
    ggml_tensor * view_k_scales_record  (ggml_context * ctx, int32_t il, uint32_t ih, uint32_t record) const;
    ggml_tensor * view_v_scales_record  (ggml_context * ctx, int32_t il, uint32_t ih, uint32_t record) const;
    ggml_tensor * view_k_body_record_span_heads(ggml_context * ctx, int32_t il, uint32_t record0, uint32_t n_records) const;
    ggml_tensor * view_v_body_record_span_heads(ggml_context * ctx, int32_t il, uint32_t record0, uint32_t n_records) const;
    ggml_tensor * view_k_scales_record_span_heads(ggml_context * ctx, int32_t il, uint32_t record0, uint32_t n_records) const;
    ggml_tensor * view_v_scales_record_span_heads(ggml_context * ctx, int32_t il, uint32_t record0, uint32_t n_records) const;
    ggml_tensor * store_k_body_record(
            ggml_context * ctx, ggml_tensor * k_tile, ggml_tensor * scratch, int32_t il, uint32_t ih, uint32_t record) const;
    ggml_tensor * store_v_body_record(
            ggml_context * ctx, ggml_tensor * v_tile, ggml_tensor * scratch, int32_t il, uint32_t ih, uint32_t record) const;
    ggml_tensor * store_kv_body_record(
            ggml_context * ctx, ggml_tensor * k_tile, ggml_tensor * v_tile, ggml_tensor * scratch, int32_t il, uint32_t ih, uint32_t record) const;
    ggml_tensor * store_kv_body_all_heads(
            ggml_context * ctx, ggml_tensor * k_tile, ggml_tensor * v_tile, ggml_tensor * scratch, int32_t il, uint32_t record) const;
    ggml_tensor * store_kv_body_records_all_heads(
            ggml_context * ctx, ggml_tensor * k_tiles, ggml_tensor * v_tiles, ggml_tensor * scratch,
            int32_t il, uint32_t record0, uint32_t n_records) const;
    ggml_tensor * store_k_body_record_from_pending(
            ggml_context * ctx, ggml_tensor * scratch, int32_t il, uint32_t ih, uint32_t record) const;
    ggml_tensor * store_v_body_record_from_pending(
            ggml_context * ctx, ggml_tensor * scratch, int32_t il, uint32_t ih, uint32_t record) const;
    ggml_tensor * store_kv_body_record_from_pending(
            ggml_context * ctx, ggml_tensor * scratch, int32_t il, uint32_t ih, uint32_t record) const;
    ggml_tensor * store_kv_body_all_heads_from_pending(
            ggml_context * ctx, ggml_tensor * scratch, int32_t il, uint32_t record) const;
    ggml_tensor * store_kv_body_records_from_pending(
            ggml_context * ctx, ggml_tensor * scratch, int32_t il, const std::vector<uint32_t> & records) const;
    ggml_tensor * view_k_body_record_heads(ggml_context * ctx, int32_t il, uint32_t record) const;
    ggml_tensor * view_v_body_record_heads(ggml_context * ctx, int32_t il, uint32_t record) const;
    ggml_tensor * view_k_scales_record_heads(ggml_context * ctx, int32_t il, uint32_t record) const;
    ggml_tensor * view_v_scales_record_heads(ggml_context * ctx, int32_t il, uint32_t record) const;

    slot_info find_slot(const llama_ubatch & ubatch) const;
    slot_info_vec_t prepare(const std::vector<llama_ubatch> & ubatches) const;
    void apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch);

private:
    struct layer_storage {
        uint32_t il = 0;
        uint32_t n_head_kv = 0;
        uint32_t n_sink_tail = 0;
        uint32_t n_records = 0;

        ggml_tensor * sink_tail_k = nullptr; // F16 [head_dim, n_head_kv, n_sink_tail]
        ggml_tensor * sink_tail_v = nullptr; // F16 [head_dim, n_head_kv, n_sink_tail]
        ggml_tensor * body_k      = nullptr; // I8  [k_body_bytes, n_records, n_head_kv]
        ggml_tensor * body_v      = nullptr; // I8  [v_body_bytes, n_records, n_head_kv]
        ggml_tensor * scales_k    = nullptr; // F32 [k_scale_floats, n_records, n_head_kv]
        ggml_tensor * scales_v    = nullptr; // F32 [v_scale_floats, n_records, n_head_kv]
        ggml_tensor * pending_k   = nullptr; // F32 [head_dim, n_head_kv, group_size]
        ggml_tensor * pending_v   = nullptr; // F32 [head_dim, n_head_kv, group_size]
        ggml_tensor * attn_mixed_scratch = nullptr; // F32 [worst-case mixed-attn scratch]
    };

    const llama_model * model = nullptr;
    const llama_hparams & hparams;
    llama_kvarn_params params;
    bool offload;

    uint32_t kv_size;
    uint32_t n_seq_max;
    uint32_t n_pad;

    std::vector<uint32_t> layer_ids;
    std::unordered_map<int32_t, int32_t> map_layer_ids;
    std::vector<uint32_t> layer_heads;
    std::vector<layer_storage> layer_tensors;
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;
    std::vector<uint32_t> v_heads;
    std::vector<llama_kv_cells> v_cells;
    std::vector<uint32_t> seq_to_stream;
    std::vector<std::vector<llama_kvarn_reference_cache>> runtime_cache;

    llama_kvarn_memory_estimate mem_estimate;

    size_t layer_storage_index(uint32_t il) const;
};
