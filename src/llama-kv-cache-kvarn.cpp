#include "llama-kv-cache-kvarn.h"

#ifdef LLAMA_BUILD
#include "llama-batch.h"
#include "llama-model.h"
#endif
#include "llama-hparams.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

static bool is_power_of_2(uint32_t n) {
    return n != 0 && (n & (n - 1)) == 0;
}

#ifdef LLAMA_BUILD
static uint32_t kvarn_ubatch_limit(uint32_t n_ubatch, uint32_t tail_tokens) {
    const char * env = std::getenv("LLAMA_KVARN_DEBUG_UBATCH");
    if (env != nullptr) {
        const int value = std::atoi(env);
        return value > 0 ? uint32_t(value) : 1;
    }

    return std::max<uint32_t>(1, std::min<uint32_t>(n_ubatch, tail_tokens));
}
#endif

static size_t packed_nbytes(size_t n_values, uint32_t bits) {
    return (n_values*bits + 7)/8;
}

static bool kvarn_hparams_has_kv(const llama_hparams & hparams, uint32_t il) {
    if (hparams.kv_only_nextn) {
        return hparams.nextn_predict_layers > 0 && il >= (hparams.n_layer - hparams.nextn_predict_layers);
    }

    if (hparams.n_layer_kv_from_start >= 0) {
        return il < (uint32_t) hparams.n_layer_kv_from_start;
    }

    return true;
}

static uint32_t kvarn_hparams_n_embd_head_k(const llama_hparams & hparams, uint32_t il) {
    return hparams.is_swa_impl[il] ? hparams.n_embd_head_k_swa : hparams.n_embd_head_k_full;
}

static uint32_t kvarn_hparams_n_embd_head_v(const llama_hparams & hparams, uint32_t il) {
    return hparams.is_swa_impl[il] ? hparams.n_embd_head_v_swa : hparams.n_embd_head_v_full;
}

static ggml_backend_buffer_type_t kvarn_layer_buft(
        const llama_model * model,
        uint32_t il,
        bool offload,
        const char ** dev_name) {
    *dev_name = "CPU";
#ifdef LLAMA_BUILD
    if (model != nullptr && offload) {
        auto * dev = model->dev_layer(il);
        *dev_name = ggml_backend_dev_name(dev);
        return ggml_backend_dev_buffer_type(dev);
    }
#else
    GGML_UNUSED(model);
    GGML_UNUSED(il);
    GGML_UNUSED(offload);
#endif

    return ggml_backend_cpu_buffer_type();
}

struct ggml_backend_buft_comparator {
    bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
        return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
    }
};

static float clamp_quantile(float q) {
    return std::min(1.0f, std::max(0.000001f, q));
}

static void sinkhorn_variance_normalize(
        std::vector<float> & data,
        std::vector<float> & row_scale,
        std::vector<float> & col_scale,
        uint32_t rows,
        uint32_t cols,
        uint32_t iters) {
    row_scale.assign(rows, 1.0f);
    col_scale.assign(cols, 1.0f);

    constexpr float eps = 1.0e-6f;

    for (uint32_t iter = 0; iter < iters; ++iter) {
        for (uint32_t r = 0; r < rows; ++r) {
            double ss = 0.0;
            for (uint32_t c = 0; c < cols; ++c) {
                const float v = data[r*cols + c];
                ss += double(v)*double(v);
            }
            const float rms = std::sqrt(float(ss/cols) + eps);
            row_scale[r] *= rms;
            for (uint32_t c = 0; c < cols; ++c) {
                data[r*cols + c] /= rms;
            }
        }

        for (uint32_t c = 0; c < cols; ++c) {
            double ss = 0.0;
            for (uint32_t r = 0; r < rows; ++r) {
                const float v = data[r*cols + c];
                ss += double(v)*double(v);
            }
            const float rms = std::sqrt(float(ss/rows) + eps);
            col_scale[c] *= rms;
            for (uint32_t r = 0; r < rows; ++r) {
                data[r*cols + c] /= rms;
            }
        }
    }
}

static void quantize_asym_per_row(
        const std::vector<float> & src,
        std::vector<uint8_t> & q,
        std::vector<float> & scale,
        std::vector<float> & zp,
        uint32_t rows,
        uint32_t cols,
        uint32_t bits,
        float quantile) {
    const uint32_t qmax = (1u << bits) - 1u;
    q.resize(size_t(rows)*cols);
    scale.resize(rows);
    zp.resize(rows);

    std::vector<float> work(cols);
    const float qt = clamp_quantile(quantile);

    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            work[c] = src[r*cols + c];
        }
        std::sort(work.begin(), work.end());

        const size_t lo_i = size_t((1.0f - qt)*0.5f*(cols - 1));
        const size_t hi_i = size_t((1.0f - (1.0f - qt)*0.5f)*(cols - 1));
        const float mn = work[lo_i];
        const float mx = work[hi_i];
        const float s = (mx > mn) ? (mx - mn)/float(qmax) : 1.0f;

        scale[r] = s;
        zp[r] = mn;

        for (uint32_t c = 0; c < cols; ++c) {
            const float v = std::min(mx, std::max(mn, src[r*cols + c]));
            q[r*cols + c] = (uint8_t) std::min(qmax, (uint32_t) std::lround((v - mn)/s));
        }
    }
}

static void append_fp32_as_fp16(std::vector<ggml_fp16_t> & dst, const std::vector<float> & src) {
    const size_t off = dst.size();
    dst.resize(off + src.size());
    ggml_fp32_to_fp16_row(src.data(), dst.data() + off, src.size());
}

static void append_fp16_as_fp32(
        std::vector<float> & dst,
        const std::vector<ggml_fp16_t>::const_iterator begin,
        const std::vector<ggml_fp16_t>::const_iterator end) {
    const size_t n = std::distance(begin, end);
    if (n == 0) {
        return;
    }
    const size_t off = dst.size();
    dst.resize(off + n);
    ggml_fp16_to_fp32_row(&(*begin), dst.data() + off, n);
}

llama_kvarn_layout llama_kvarn_make_layout(const llama_kvarn_params & params, uint32_t head_dim) {
    if (head_dim == 0 || params.group_size == 0) {
        throw std::invalid_argument("KVarN layout requires non-zero head_dim and group_size");
    }
    if (params.key_bits == 0 || params.key_bits > 8 || params.value_bits == 0 || params.value_bits > 8) {
        throw std::invalid_argument("KVarN layout requires bit widths in [1, 8]");
    }

    llama_kvarn_layout layout = {
        /*.head_dim          =*/ head_dim,
        /*.group_size        =*/ params.group_size,
        /*.key_bits          =*/ params.key_bits,
        /*.value_bits        =*/ params.value_bits,
        /*.k_body_bytes      =*/ packed_nbytes(size_t(head_dim)*params.group_size, params.key_bits),
        /*.v_body_bytes      =*/ packed_nbytes(size_t(head_dim)*params.group_size, params.value_bits),
        /*.k_scale_floats    =*/ size_t(2)*head_dim + params.group_size,
        /*.v_scale_floats    =*/ head_dim + size_t(2)*params.group_size,
        /*.total_record_bytes=*/ 0,
    };

    layout.total_record_bytes =
        layout.k_body_bytes + layout.v_body_bytes +
        (layout.k_scale_floats + layout.v_scale_floats)*sizeof(float);

    return layout;
}

void llama_kvarn_hadamard_channels(
        const std::vector<float> & src,
        std::vector<float> & dst,
        uint32_t rows,
        uint32_t cols,
        bool channels_are_rows) {
    const uint32_t n_channels = channels_are_rows ? rows : cols;
    if (!is_power_of_2(n_channels)) {
        throw std::invalid_argument("KVarN Hadamard channel count must be a power of two");
    }
    if (src.size() != size_t(rows)*cols) {
        throw std::invalid_argument("KVarN Hadamard input size mismatch");
    }

    dst = src;
    const float norm = 1.0f/std::sqrt(float(n_channels));

    if (channels_are_rows) {
        for (uint32_t c = 0; c < cols; ++c) {
            for (uint32_t step = 1; step < rows; step <<= 1) {
                for (uint32_t base = 0; base < rows; base += 2*step) {
                    for (uint32_t i = 0; i < step; ++i) {
                        const uint32_t r0 = base + i;
                        const uint32_t r1 = r0 + step;
                        const float a = dst[r0*cols + c];
                        const float b = dst[r1*cols + c];
                        dst[r0*cols + c] = a + b;
                        dst[r1*cols + c] = a - b;
                    }
                }
            }
        }
    } else {
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t step = 1; step < cols; step <<= 1) {
                for (uint32_t base = 0; base < cols; base += 2*step) {
                    for (uint32_t i = 0; i < step; ++i) {
                        const uint32_t c0 = base + i;
                        const uint32_t c1 = c0 + step;
                        const float a = dst[r*cols + c0];
                        const float b = dst[r*cols + c1];
                        dst[r*cols + c0] = a + b;
                        dst[r*cols + c1] = a - b;
                    }
                }
            }
        }
    }

    for (float & v : dst) {
        v *= norm;
    }
}

void llama_kvarn_pack_bits(const std::vector<uint8_t> & src, uint32_t bits, std::vector<uint8_t> & dst) {
    if (bits == 0 || bits > 8) {
        throw std::invalid_argument("KVarN pack bit width must be in [1, 8]");
    }

    dst.assign(packed_nbytes(src.size(), bits), 0);
    const uint32_t mask = (1u << bits) - 1u;

    size_t bit_pos = 0;
    for (uint8_t v : src) {
        const uint32_t q = uint32_t(v) & mask;
        const size_t byte_pos = bit_pos >> 3;
        const uint32_t shift = uint32_t(bit_pos & 7);
        dst[byte_pos] |= uint8_t(q << shift);
        if (shift + bits > 8) {
            dst[byte_pos + 1] |= uint8_t(q >> (8 - shift));
        }
        bit_pos += bits;
    }
}

void llama_kvarn_unpack_bits(const std::vector<uint8_t> & src, uint32_t bits, size_t n_values, std::vector<uint8_t> & dst) {
    if (bits == 0 || bits > 8) {
        throw std::invalid_argument("KVarN unpack bit width must be in [1, 8]");
    }
    if (src.size() < packed_nbytes(n_values, bits)) {
        throw std::invalid_argument("KVarN packed input is too small");
    }

    dst.assign(n_values, 0);
    const uint32_t mask = (1u << bits) - 1u;

    size_t bit_pos = 0;
    for (size_t i = 0; i < n_values; ++i) {
        const size_t byte_pos = bit_pos >> 3;
        const uint32_t shift = uint32_t(bit_pos & 7);
        uint32_t q = uint32_t(src[byte_pos]) >> shift;
        if (shift + bits > 8) {
            q |= uint32_t(src[byte_pos + 1]) << (8 - shift);
        }
        dst[i] = uint8_t(q & mask);
        bit_pos += bits;
    }
}

llama_kvarn_body_record llama_kvarn_store_reference(
        const llama_kvarn_params & params,
        uint32_t head_dim,
        const std::vector<float> & k_tile,
        const std::vector<float> & v_tile) {
    const llama_kvarn_layout layout = llama_kvarn_make_layout(params, head_dim);
    const size_t n = size_t(head_dim)*params.group_size;
    if (k_tile.size() != n || v_tile.size() != n) {
        throw std::invalid_argument("KVarN tile size mismatch");
    }

    std::vector<float> k_rot;
    std::vector<float> v_rot;
    llama_kvarn_hadamard_channels(k_tile, k_rot, head_dim, params.group_size, true);
    llama_kvarn_hadamard_channels(v_tile, v_rot, params.group_size, head_dim, false);

    std::vector<float> k_row_scale, k_col_scale, v_row_scale, v_col_scale;
    sinkhorn_variance_normalize(k_rot, k_row_scale, k_col_scale, head_dim, params.group_size, params.sinkhorn_iters);
    sinkhorn_variance_normalize(v_rot, v_row_scale, v_col_scale, params.group_size, head_dim, params.sinkhorn_iters);

    std::vector<uint8_t> k_q, v_q;
    std::vector<float> k_rtn_scale, k_rtn_zp, v_rtn_scale, v_rtn_zp;
    quantize_asym_per_row(k_rot, k_q, k_rtn_scale, k_rtn_zp, head_dim, params.group_size, params.key_bits, params.rtn_quantile);
    quantize_asym_per_row(v_rot, v_q, v_rtn_scale, v_rtn_zp, params.group_size, head_dim, params.value_bits, params.rtn_quantile);

    llama_kvarn_body_record record;
    record.layout = layout;
    llama_kvarn_pack_bits(k_q, params.key_bits, record.k_body);
    llama_kvarn_pack_bits(v_q, params.value_bits, record.v_body);

    record.k_scales.reserve(layout.k_scale_floats);
    for (uint32_t d = 0; d < head_dim; ++d) {
        record.k_scales.push_back(k_row_scale[d]*k_rtn_scale[d]);
    }
    for (uint32_t d = 0; d < head_dim; ++d) {
        record.k_scales.push_back(k_row_scale[d]*k_rtn_zp[d]);
    }
    for (uint32_t g = 0; g < params.group_size; ++g) {
        record.k_scales.push_back(k_col_scale[g]);
    }

    record.v_scales.reserve(layout.v_scale_floats);
    for (uint32_t d = 0; d < head_dim; ++d) {
        record.v_scales.push_back(v_col_scale[d]);
    }
    for (uint32_t g = 0; g < params.group_size; ++g) {
        record.v_scales.push_back(v_row_scale[g]*v_rtn_scale[g]);
    }
    for (uint32_t g = 0; g < params.group_size; ++g) {
        record.v_scales.push_back(v_row_scale[g]*v_rtn_zp[g]);
    }

    return record;
}

void llama_kvarn_dequant_reference(
        const llama_kvarn_body_record & record,
        std::vector<float> & k_tile,
        std::vector<float> & v_tile) {
    const llama_kvarn_layout & layout = record.layout;
    const size_t n = size_t(layout.head_dim)*layout.group_size;

    std::vector<uint8_t> k_q, v_q;
    llama_kvarn_unpack_bits(record.k_body, layout.key_bits, n, k_q);
    llama_kvarn_unpack_bits(record.v_body, layout.value_bits, n, v_q);

    k_tile.assign(n, 0.0f);
    v_tile.assign(n, 0.0f);

    const float * k_s_col = record.k_scales.data();
    const float * k_zp    = record.k_scales.data() + layout.head_dim;
    const float * k_s_row = record.k_scales.data() + 2*layout.head_dim;

    for (uint32_t d = 0; d < layout.head_dim; ++d) {
        for (uint32_t g = 0; g < layout.group_size; ++g) {
            k_tile[d*layout.group_size + g] =
                (float(k_q[d*layout.group_size + g])*k_s_col[d] + k_zp[d])*k_s_row[g];
        }
    }

    const float * v_s_col = record.v_scales.data();
    const float * v_s_row = record.v_scales.data() + layout.head_dim;
    const float * v_zp    = record.v_scales.data() + layout.head_dim + layout.group_size;

    for (uint32_t g = 0; g < layout.group_size; ++g) {
        for (uint32_t d = 0; d < layout.head_dim; ++d) {
            v_tile[g*layout.head_dim + d] =
                (float(v_q[g*layout.head_dim + d])*v_s_row[g] + v_zp[g])*v_s_col[d];
        }
    }
}

llama_kvarn_reference_cache::llama_kvarn_reference_cache(llama_kvarn_params params, uint32_t head_dim) :
    params(params), head_dim(head_dim) {
    if (head_dim == 0 || params.group_size == 0 || params.sink_tokens == 0 || params.tail_tokens == 0) {
        throw std::invalid_argument("invalid KVarN reference cache geometry");
    }
}

void llama_kvarn_reference_cache::append_token(const std::vector<float> & k_token, const std::vector<float> & v_token) {
    if (k_token.size() != head_dim || v_token.size() != head_dim) {
        throw std::invalid_argument("KVarN token size mismatch");
    }

    if (n_tokens < params.sink_tokens) {
        append_fp32_as_fp16(sink_k, k_token);
        append_fp32_as_fp16(sink_v, v_token);
    } else {
        append_fp32_as_fp16(tail_k, k_token);
        append_fp32_as_fp16(tail_v, v_token);

        if (tail_k.size()/head_dim > params.tail_tokens) {
            append_fp16_as_fp32(pending_k, tail_k.begin(), tail_k.begin() + head_dim);
            append_fp16_as_fp32(pending_v, tail_v.begin(), tail_v.begin() + head_dim);
            tail_k.erase(tail_k.begin(), tail_k.begin() + head_dim);
            tail_v.erase(tail_v.begin(), tail_v.begin() + head_dim);
        }

        if (pending_k.size()/head_dim == params.group_size) {
            seal_pending_group();
        }
    }

    ++n_tokens;
}

void llama_kvarn_reference_cache::clear() {
    n_tokens = 0;
    sink_k.clear();
    sink_v.clear();
    tail_k.clear();
    tail_v.clear();
    pending_k.clear();
    pending_v.clear();
    records.clear();
}

llama_kvarn_reference_cache_stats llama_kvarn_reference_cache::stats() const {
    size_t body_packed_bytes = 0;
    size_t scale_values = 0;
    for (const llama_kvarn_body_record & record : records) {
        body_packed_bytes += record.k_body.size() + record.v_body.size();
        scale_values      += record.k_scales.size() + record.v_scales.size();
    }

    return {
        /*.n_sink         =*/ uint32_t(sink_k.size()/head_dim),
        /*.n_tail         =*/ uint32_t(tail_k.size()/head_dim),
        /*.n_pending_body =*/ uint32_t(pending_k.size()/head_dim),
        /*.n_body_records =*/ uint32_t(records.size()),
        /*.n_tokens       =*/ n_tokens,
        /*.sink_tail_fp16_values =*/ sink_k.size() + sink_v.size() + tail_k.size() + tail_v.size(),
        /*.body_packed_bytes     =*/ body_packed_bytes,
        /*.scale_values          =*/ scale_values,
    };
}

const std::vector<llama_kvarn_body_record> & llama_kvarn_reference_cache::body_records() const {
    return records;
}

void llama_kvarn_reference_cache::materialize_tokens(std::vector<float> & k_tokens, std::vector<float> & v_tokens) const {
    k_tokens.clear();
    v_tokens.clear();
    k_tokens.reserve(size_t(n_tokens)*head_dim);
    v_tokens.reserve(size_t(n_tokens)*head_dim);

    append_fp16_as_fp32(k_tokens, sink_k.begin(), sink_k.end());
    append_fp16_as_fp32(v_tokens, sink_v.begin(), sink_v.end());

    for (const llama_kvarn_body_record & record : records) {
        std::vector<float> k_tile;
        std::vector<float> v_tile;
        llama_kvarn_dequant_reference(record, k_tile, v_tile);

        if (record.layout.head_dim != head_dim) {
            throw std::runtime_error("KVarN record head_dim mismatch");
        }

        for (uint32_t g = 0; g < record.layout.group_size; ++g) {
            for (uint32_t d = 0; d < head_dim; ++d) {
                k_tokens.push_back(k_tile[d*record.layout.group_size + g]);
                v_tokens.push_back(v_tile[g*head_dim + d]);
            }
        }
    }

    k_tokens.insert(k_tokens.end(), pending_k.begin(), pending_k.end());
    v_tokens.insert(v_tokens.end(), pending_v.begin(), pending_v.end());
    append_fp16_as_fp32(k_tokens, tail_k.begin(), tail_k.end());
    append_fp16_as_fp32(v_tokens, tail_v.begin(), tail_v.end());
}

void llama_kvarn_reference_cache::seal_pending_group() {
    std::vector<float> k_tile(size_t(head_dim)*params.group_size);
    std::vector<float> v_tile(size_t(head_dim)*params.group_size);

    for (uint32_t g = 0; g < params.group_size; ++g) {
        for (uint32_t d = 0; d < head_dim; ++d) {
            k_tile[d*params.group_size + g] = pending_k[g*head_dim + d];
            v_tile[g*head_dim + d] = pending_v[g*head_dim + d];
        }
    }

    records.push_back(llama_kvarn_store_reference(params, head_dim, k_tile, v_tile));
    pending_k.clear();
    pending_v.clear();
}

llama_kvarn_memory_estimate llama_kvarn_estimate_memory(
        const llama_kvarn_params & params,
        const llama_hparams & hparams,
        uint32_t kv_size,
        const llama_memory_i::layer_filter_cb & filter) {
    if (params.group_size == 0) {
        throw std::invalid_argument("KVarN memory estimate requires non-zero group_size");
    }

    llama_kvarn_memory_estimate result = {};

    const uint32_t n_sink_tail = std::min<uint32_t>(kv_size, params.sink_tokens + params.tail_tokens);
    const uint32_t n_body      = kv_size > n_sink_tail ? kv_size - n_sink_tail : 0;
    const uint32_t n_records   = (n_body + params.group_size - 1)/params.group_size;

    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
        if (!kvarn_hparams_has_kv(hparams, il)) {
            continue;
        }
        if (filter && !filter(il)) {
            continue;
        }

        const uint32_t n_head_kv = hparams.n_head_kv_arr[il];
        const uint32_t head_k    = kvarn_hparams_n_embd_head_k(hparams, il);
        const uint32_t head_v    = kvarn_hparams_n_embd_head_v(hparams, il);

        result.fp16_sink_tail_bytes += size_t(n_head_kv)*n_sink_tail*(head_k + head_v)*sizeof(uint16_t);

        const llama_kvarn_layout layout_k = llama_kvarn_make_layout(params, head_k);
        const llama_kvarn_layout layout_v = llama_kvarn_make_layout(params, head_v);
        result.body_packed_bytes += size_t(n_head_kv)*n_records*(layout_k.k_body_bytes + layout_v.v_body_bytes);
        result.scale_bytes       += size_t(n_head_kv)*n_records*(layout_k.k_scale_floats + layout_v.v_scale_floats)*sizeof(float);
    }

    result.total_bytes = result.fp16_sink_tail_bytes + result.body_packed_bytes + result.scale_bytes;
    return result;
}

llama_kv_cache_kvarn_context::llama_kv_cache_kvarn_context(llama_memory_status status) : status(status) {
}

llama_kv_cache_kvarn_context::llama_kv_cache_kvarn_context(
        llama_kv_cache_kvarn * kv,
        slot_info_vec_t sinfos,
        std::vector<llama_ubatch> ubatches) :
    status(LLAMA_MEMORY_STATUS_SUCCESS),
    kv(kv),
    sinfos(std::move(sinfos)),
    ubatches(std::move(ubatches)) {
}

bool llama_kv_cache_kvarn_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_cur >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_kvarn_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    kv->apply_ubatch(sinfos[i_cur], ubatches[i_cur]);
    return true;
}

llama_memory_status llama_kv_cache_kvarn_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_kvarn_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_cur];
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_sink_tail_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return kv->build_input_sink_tail_idxs(ctx, ubatch);
}

void llama_kv_cache_kvarn_context::set_input_sink_tail_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    kv->set_input_sink_tail_idxs(dst, ubatch);
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_body_plan(
        ggml_context * ctx,
        const llama_ubatch & ubatch) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return kv->build_input_body_plan(ctx, ubatch);
}

void llama_kv_cache_kvarn_context::set_input_body_plan(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    kv->set_input_body_plan(dst, ubatch);
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_body_offsets(
        ggml_context * ctx,
        const llama_ubatch & ubatch) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return kv->build_input_body_offsets(ctx, ubatch);
}

void llama_kv_cache_kvarn_context::set_input_body_offsets(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    kv->set_input_body_offsets(dst, ubatch);
}

ggml_tensor * llama_kv_cache_kvarn_context::build_input_tail_evict_idxs(
        ggml_context * ctx,
        const llama_ubatch & ubatch) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return kv->build_input_tail_evict_idxs(ctx, ubatch);
}

void llama_kv_cache_kvarn_context::set_input_tail_evict_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    kv->set_input_tail_evict_idxs(dst, ubatch);
}

void llama_kv_cache_kvarn_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    kv->set_input_kq_mask(dst, ubatch, causal_attn);
}

uint32_t llama_kv_cache_kvarn_context::get_size() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return kv->get_size();
}

ggml_tensor * llama_kv_cache_kvarn_context::cpy_sink_tail_k(
        ggml_context * ctx,
        ggml_tensor * k_cur,
        ggml_tensor * idxs,
        int32_t il) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return kv->cpy_sink_tail_k(ctx, k_cur, idxs, il);
}

ggml_tensor * llama_kv_cache_kvarn_context::cpy_sink_tail_v(
        ggml_context * ctx,
        ggml_tensor * v_cur,
        ggml_tensor * idxs,
        int32_t il) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return kv->cpy_sink_tail_v(ctx, v_cur, idxs, il);
}

ggml_tensor * llama_kv_cache_kvarn_context::cpy_tail_evict_pending_k(
        ggml_context * ctx,
        ggml_tensor * tail_idxs,
        ggml_tensor * offsets,
        int32_t il) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return kv->cpy_tail_evict_pending_k(ctx, tail_idxs, offsets, il);
}

ggml_tensor * llama_kv_cache_kvarn_context::cpy_tail_evict_pending_v(
        ggml_context * ctx,
        ggml_tensor * tail_idxs,
        ggml_tensor * offsets,
        int32_t il) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return kv->cpy_tail_evict_pending_v(ctx, tail_idxs, offsets, il);
}

llama_kvarn_layer_view llama_kv_cache_kvarn_context::get_layer_view(int32_t il) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return kv->get_layer_view(il);
}

size_t llama_kv_cache_kvarn_context::body_store_scratch_floats(int32_t il) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return kv->body_store_scratch_floats(il);
}

ggml_tensor * llama_kv_cache_kvarn_context::build_body_store_scratch(ggml_context * ctx, int32_t il) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return kv->build_body_store_scratch(ctx, il);
}

ggml_tensor * llama_kv_cache_kvarn_context::view_k_body_record(
        ggml_context * ctx, int32_t il, uint32_t ih, uint32_t record) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return kv->view_k_body_record(ctx, il, ih, record);
}

ggml_tensor * llama_kv_cache_kvarn_context::view_v_body_record(
        ggml_context * ctx, int32_t il, uint32_t ih, uint32_t record) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return kv->view_v_body_record(ctx, il, ih, record);
}

ggml_tensor * llama_kv_cache_kvarn_context::view_k_scales_record(
        ggml_context * ctx, int32_t il, uint32_t ih, uint32_t record) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return kv->view_k_scales_record(ctx, il, ih, record);
}

ggml_tensor * llama_kv_cache_kvarn_context::view_v_scales_record(
        ggml_context * ctx, int32_t il, uint32_t ih, uint32_t record) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return kv->view_v_scales_record(ctx, il, ih, record);
}

ggml_tensor * llama_kv_cache_kvarn_context::store_k_body_record(
        ggml_context * ctx,
        ggml_tensor * k_tile,
        ggml_tensor * scratch,
        int32_t il,
        uint32_t ih,
        uint32_t record) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return kv->store_k_body_record(ctx, k_tile, scratch, il, ih, record);
}

ggml_tensor * llama_kv_cache_kvarn_context::store_v_body_record(
        ggml_context * ctx,
        ggml_tensor * v_tile,
        ggml_tensor * scratch,
        int32_t il,
        uint32_t ih,
        uint32_t record) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return kv->store_v_body_record(ctx, v_tile, scratch, il, ih, record);
}

ggml_tensor * llama_kv_cache_kvarn_context::store_k_body_record_from_pending(
        ggml_context * ctx,
        ggml_tensor * scratch,
        int32_t il,
        uint32_t ih,
        uint32_t record) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return kv->store_k_body_record_from_pending(ctx, scratch, il, ih, record);
}

ggml_tensor * llama_kv_cache_kvarn_context::store_v_body_record_from_pending(
        ggml_context * ctx,
        ggml_tensor * scratch,
        int32_t il,
        uint32_t ih,
        uint32_t record) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return kv->store_v_body_record_from_pending(ctx, scratch, il, ih, record);
}

llama_kv_cache_kvarn::llama_kv_cache_kvarn(
        const llama_model * model,
        const llama_hparams & hparams,
        llama_kvarn_params params,
        bool offload,
        uint32_t kv_size,
        uint32_t n_seq_max,
        uint32_t n_pad,
        const layer_filter_cb & filter) :
    model(model),
    hparams(hparams),
    params(params),
    offload(offload),
    kv_size(kv_size),
    n_seq_max(n_seq_max),
    n_pad(n_pad) {
    if (kv_size == 0 || n_seq_max == 0) {
        throw std::invalid_argument("KVarN cache requires non-zero kv_size and n_seq_max");
    }

    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params init_params = {
                /*.mem_size   =*/ size_t(8u*hparams.n_layer*ggml_tensor_overhead()),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(init_params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);
            return ctx;
        }

        return it->second.get();
    };

    const uint32_t n_sink_tail = std::min<uint32_t>(kv_size, params.sink_tokens + params.tail_tokens);
    const uint32_t n_body      = kv_size > n_sink_tail ? kv_size - n_sink_tail : 0;
    const uint32_t n_records   = (n_body + params.group_size - 1)/params.group_size;

    const uint32_t n_records_alloc = std::max<uint32_t>(1, n_records);
    const uint32_t n_sink_tail_alloc = std::max<uint32_t>(1, n_sink_tail);

    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
        if (!kvarn_hparams_has_kv(hparams, il)) {
            continue;
        }
        if (filter && !filter(il)) {
            continue;
        }

        const char * dev_name = nullptr;
        ggml_backend_buffer_type_t buft = kvarn_layer_buft(this->model, il, this->offload, &dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for KVarN KV cache");
        }

        layer_ids.push_back(il);
        const uint32_t n_head_kv = hparams.n_head_kv_arr[il];
        const uint32_t head_k = kvarn_hparams_n_embd_head_k(hparams, il);
        const uint32_t head_v = kvarn_hparams_n_embd_head_v(hparams, il);
        const llama_kvarn_layout layout_k = llama_kvarn_make_layout(params, head_k);
        const llama_kvarn_layout layout_v = llama_kvarn_make_layout(params, head_v);
        layer_heads.push_back(n_head_kv);

        layer_storage st = {};
        st.il = il;
        st.n_head_kv = n_head_kv;
        st.n_sink_tail = n_sink_tail;
        st.n_records = n_records;
        st.sink_tail_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, head_k, n_head_kv, n_sink_tail_alloc);
        st.sink_tail_v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, head_v, n_head_kv, n_sink_tail_alloc);
        st.body_k      = ggml_new_tensor_3d(ctx, GGML_TYPE_I8,  layout_k.k_body_bytes,   n_records_alloc, n_head_kv);
        st.body_v      = ggml_new_tensor_3d(ctx, GGML_TYPE_I8,  layout_v.v_body_bytes,   n_records_alloc, n_head_kv);
        st.scales_k    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, layout_k.k_scale_floats, n_records_alloc, n_head_kv);
        st.scales_v    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, layout_v.v_scale_floats, n_records_alloc, n_head_kv);
        st.pending_k   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_k, n_head_kv, params.group_size);
        st.pending_v   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_v, n_head_kv, params.group_size);

        ggml_format_name(st.sink_tail_k, "kvarn_sink_tail_k_l%d", il);
        ggml_format_name(st.sink_tail_v, "kvarn_sink_tail_v_l%d", il);
        ggml_format_name(st.body_k,      "kvarn_body_k_l%d", il);
        ggml_format_name(st.body_v,      "kvarn_body_v_l%d", il);
        ggml_format_name(st.scales_k,    "kvarn_scales_k_l%d", il);
        ggml_format_name(st.scales_v,    "kvarn_scales_v_l%d", il);
        ggml_format_name(st.pending_k,   "kvarn_pending_k_l%d", il);
        ggml_format_name(st.pending_v,   "kvarn_pending_v_l%d", il);

        layer_tensors.push_back(st);

        std::vector<llama_kvarn_reference_cache> caches;
        caches.reserve(n_head_kv);
        for (uint32_t ih = 0; ih < n_head_kv; ++ih) {
            GGML_UNUSED(ih);
            caches.emplace_back(params, kvarn_hparams_n_embd_head_k(hparams, il));
        }
        runtime_cache.push_back(std::move(caches));

        std::fprintf(stderr, "%s: KVarN layer %3u storage dev = %s, heads = %u, body records = %u\n",
                __func__, il, dev_name, n_head_kv, n_records);
    }

    if (layer_ids.empty()) {
        throw std::invalid_argument("KVarN cache requires at least one KV layer");
    }

    v_heads.resize(1, 0);
    v_cells.resize(1);
    v_cells[0].resize(kv_size);

    seq_to_stream.resize(n_seq_max, 0);

    mem_estimate = llama_kvarn_estimate_memory(params, hparams, kv_size, filter);

    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (hparams.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0);
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf;
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate KVarN KV cache buffer");
        }

        std::fprintf(stderr, "%s: %10s KVarN buffer size = %8.2f MiB\n",
                __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ggml_backend_buffer_clear(buf, 0);
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    std::fprintf(stderr, "%s: KVarN metadata cache = %u cells, %u layers, estimate %.2f MiB "
            "(sink/tail %.2f MiB, body %.2f MiB, scales %.2f MiB)\n",
            __func__,
            kv_size,
            (uint32_t) layer_ids.size(),
            double(mem_estimate.total_bytes)/(1024.0*1024.0),
            double(mem_estimate.fp16_sink_tail_bytes)/(1024.0*1024.0),
            double(mem_estimate.body_packed_bytes)/(1024.0*1024.0),
            double(mem_estimate.scale_bytes)/(1024.0*1024.0));
}

llama_memory_context_ptr llama_kv_cache_kvarn::init_batch(
        llama_batch_allocr & balloc,
        uint32_t n_ubatch,
        bool embd_all) {
    GGML_UNUSED(embd_all);

#ifdef LLAMA_BUILD
    balloc.split_reset();

    std::vector<llama_ubatch> ubatches;
    const uint32_t n_kvarn_ubatch = hparams.n_expert > 0 ? 1 : kvarn_ubatch_limit(n_ubatch, params.tail_tokens);
    while (true) {
        // Keep each graph within one tail-ring span so tokens written earlier
        // in the graph are not evicted before their pending-body copy runs.
        auto ubatch = balloc.split_simple(n_kvarn_ubatch);
        if (ubatch.n_tokens == 0) {
            break;
        }

        ubatches.push_back(std::move(ubatch));
    }

    if (balloc.get_n_used() < balloc.get_n_tokens()) {
        return std::make_unique<llama_kv_cache_kvarn_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    const auto sinfos = prepare(ubatches);
    if (sinfos.empty()) {
        return std::make_unique<llama_kv_cache_kvarn_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    return std::make_unique<llama_kv_cache_kvarn_context>(this, std::move(sinfos), std::move(ubatches));
#else
    GGML_UNUSED(balloc);

    std::fprintf(stderr, "%s: KVarN graph backend is not wired yet; refusing to run through the normal KV cache path\n", __func__);
    return std::make_unique<llama_kv_cache_kvarn_context>(LLAMA_MEMORY_STATUS_FAILED_COMPUTE);
#endif
}

llama_memory_context_ptr llama_kv_cache_kvarn::init_full() {
    llama_ubatch ubatch = {};
    slot_info sinfo;
    sinfo.idxs.resize(1, 0);

    return std::make_unique<llama_kv_cache_kvarn_context>(
            this,
            slot_info_vec_t{ std::move(sinfo) },
            std::vector<llama_ubatch>{ ubatch });
}

llama_memory_context_ptr llama_kv_cache_kvarn::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(lctx);
    GGML_UNUSED(optimize);

    if (v_cells[0].get_has_shift()) {
        std::fprintf(stderr, "%s: KVarN shift/update graph path is not wired yet\n", __func__);
        return std::make_unique<llama_kv_cache_kvarn_context>(LLAMA_MEMORY_STATUS_FAILED_COMPUTE);
    }

    return std::make_unique<llama_kv_cache_kvarn_context>(LLAMA_MEMORY_STATUS_NO_UPDATE);
}

bool llama_kv_cache_kvarn::get_can_shift() const {
    return hparams.rope_type != LLAMA_ROPE_TYPE_MROPE && hparams.rope_type != LLAMA_ROPE_TYPE_IMROPE;
}

void llama_kv_cache_kvarn::clear(bool data) {
    for (auto & cells : v_cells) {
        cells.reset();
    }
    std::fill(v_heads.begin(), v_heads.end(), 0);

    for (auto & layer : runtime_cache) {
        for (auto & head : layer) {
            head.clear();
        }
    }

    if (data) {
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }
}

bool llama_kv_cache_kvarn::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (p0 < 0) {
        p0 = 0;
    }
    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    auto & cells = v_cells[0];
    auto & head  = v_heads[0];
    uint32_t new_head = cells.size();

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (seq_id < 0) {
            cells.rm(i);
            if (new_head == cells.size()) {
                new_head = i;
            }
        } else if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }

    return true;
}

void llama_kv_cache_kvarn::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    GGML_ASSERT(seq_id_src >= 0 && (size_t) seq_id_src < seq_to_stream.size());
    GGML_ASSERT(seq_id_dst >= 0 && (size_t) seq_id_dst < seq_to_stream.size());

    if (seq_id_src == seq_id_dst) {
        return;
    }
    if (p0 < 0) {
        p0 = 0;
    }
    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    auto & cells = v_cells[0];
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.pos_in(i, p0, p1) && cells.seq_has(i, seq_id_src) && !cells.seq_has(i, seq_id_dst)) {
            cells.seq_add(i, seq_id_dst);
        }
    }
}

void llama_kv_cache_kvarn::seq_keep(llama_seq_id seq_id) {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    auto & cells = v_cells[0];
    auto & head  = v_heads[0];
    uint32_t new_head = cells.size();

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.seq_keep(i, seq_id)) {
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }
}

void llama_kv_cache_kvarn::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(get_can_shift() && "seq_add() is only supported for n_pos_per_embd() == 1");

    if (shift == 0) {
        return;
    }
    if (p0 < 0) {
        p0 = 0;
    }
    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }
    if (p0 == p1) {
        return;
    }

    auto & cells = v_cells[0];
    auto & head  = v_heads[0];
    uint32_t new_head = cells.size();

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.pos_in(i, p0, p1) && cells.seq_has(i, seq_id) && cells.pos_add(i, shift)) {
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    head = new_head != cells.size() ? new_head : 0;
}

void llama_kv_cache_kvarn::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(get_can_shift() && "seq_div() is only supported for n_pos_per_embd() == 1");

    if (d == 1) {
        return;
    }
    if (p0 < 0) {
        p0 = 0;
    }
    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }
    if (p0 == p1) {
        return;
    }

    auto & cells = v_cells[0];
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.pos_in(i, p0, p1) && cells.seq_has(i, seq_id)) {
            cells.pos_div(i, d);
        }
    }
}

llama_pos llama_kv_cache_kvarn::seq_pos_min(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    return v_cells[0].seq_pos_min(seq_id);
}

llama_pos llama_kv_cache_kvarn::seq_pos_max(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    return v_cells[0].seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_kvarn::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> result;
    for (const auto & [ctx, buf] : ctxs_bufs) {
        if (ggml_get_first_tensor(ctx.get()) == nullptr) {
            continue;
        }

        result[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
    }

    if (result.empty()) {
        result[ggml_backend_cpu_buffer_type()] = mem_estimate.total_bytes;
    }

    return result;
}

void llama_kv_cache_kvarn::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    GGML_UNUSED(io);
    GGML_UNUSED(seq_id);
    GGML_UNUSED(flags);

    throw std::runtime_error("KVarN state serialization is not implemented yet");
}

void llama_kv_cache_kvarn::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(io);
    GGML_UNUSED(seq_id);
    GGML_UNUSED(flags);

    throw std::runtime_error("KVarN state deserialization is not implemented yet");
}

uint32_t llama_kv_cache_kvarn::get_size() const {
    return kv_size;
}

uint32_t llama_kv_cache_kvarn::get_n_layer() const {
    return layer_ids.size();
}

llama_kvarn_memory_estimate llama_kv_cache_kvarn::estimate() const {
    return mem_estimate;
}

llama_kvarn_runtime_storage_stats llama_kv_cache_kvarn::storage_stats() const {
    llama_kvarn_runtime_storage_stats result = {};
    result.n_layers = uint32_t(runtime_cache.size());

    for (const auto & layer : runtime_cache) {
        result.n_heads += uint32_t(layer.size());
        for (const llama_kvarn_reference_cache & head : layer) {
            const llama_kvarn_reference_cache_stats stats = head.stats();
            result.n_tokens       = std::max(result.n_tokens, stats.n_tokens);
            result.n_body_records += stats.n_body_records;
            result.n_pending_body += stats.n_pending_body;
            result.n_sink         += stats.n_sink;
            result.n_tail         += stats.n_tail;
            result.fp16_sink_tail_values += stats.sink_tail_fp16_values;
            result.body_packed_bytes     += stats.body_packed_bytes;
            result.scale_values          += stats.scale_values;
        }
    }

    return result;
}

size_t llama_kv_cache_kvarn::backend_tensor_bytes() const {
    size_t result = 0;
    for (const layer_storage & layer : layer_tensors) {
        result += ggml_nbytes(layer.sink_tail_k);
        result += ggml_nbytes(layer.sink_tail_v);
        result += ggml_nbytes(layer.body_k);
        result += ggml_nbytes(layer.body_v);
        result += ggml_nbytes(layer.scales_k);
        result += ggml_nbytes(layer.scales_v);
        result += ggml_nbytes(layer.pending_k);
        result += ggml_nbytes(layer.pending_v);
    }

    return result;
}

size_t llama_kv_cache_kvarn::layer_storage_index(uint32_t il) const {
    const auto it = std::find(layer_ids.begin(), layer_ids.end(), il);
    if (it == layer_ids.end()) {
        throw std::invalid_argument("KVarN layer is not present in runtime storage");
    }

    return size_t(std::distance(layer_ids.begin(), it));
}

llama_kvarn_layer_view llama_kv_cache_kvarn::get_layer_view(int32_t il) const {
    if (il < 0) {
        throw std::invalid_argument("KVarN layer id must be non-negative");
    }

    const size_t li = layer_storage_index(uint32_t(il));
    const layer_storage & st = layer_tensors[li];

    const uint32_t head_k = kvarn_hparams_n_embd_head_k(hparams, uint32_t(il));
    const uint32_t head_v = kvarn_hparams_n_embd_head_v(hparams, uint32_t(il));

    return {
        /*.il          =*/ st.il,
        /*.n_head_kv   =*/ st.n_head_kv,
        /*.n_records   =*/ st.n_records,
        /*.head_dim_k  =*/ head_k,
        /*.head_dim_v  =*/ head_v,
        /*.layout_k    =*/ llama_kvarn_make_layout(params, head_k),
        /*.layout_v    =*/ llama_kvarn_make_layout(params, head_v),
        /*.sink_tail_k =*/ st.sink_tail_k,
        /*.sink_tail_v =*/ st.sink_tail_v,
        /*.body_k      =*/ st.body_k,
        /*.body_v      =*/ st.body_v,
        /*.scales_k    =*/ st.scales_k,
        /*.scales_v    =*/ st.scales_v,
        /*.pending_k   =*/ st.pending_k,
        /*.pending_v   =*/ st.pending_v,
    };
}

size_t llama_kv_cache_kvarn::body_store_scratch_floats(int32_t il) const {
    const llama_kvarn_layer_view view = get_layer_view(il);
    if (view.head_dim_k != view.head_dim_v) {
        throw std::invalid_argument("KVarN body store scratch requires equal K and V head dimensions");
    }

    return size_t(view.head_dim_k)*params.group_size + 2*std::max<uint32_t>(view.head_dim_k, params.group_size);
}

ggml_tensor * llama_kv_cache_kvarn::build_body_store_scratch(ggml_context * ctx, int32_t il) const {
    ggml_tensor * scratch = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, body_store_scratch_floats(il));
    ggml_format_name(scratch, "kvarn_body_store_scratch_l%d", il);
    return scratch;
}

static void assert_kvarn_record_view_bounds(const llama_kvarn_layer_view & view, uint32_t ih, uint32_t record) {
    if (ih >= view.n_head_kv) {
        throw std::out_of_range("KVarN body record head index is out of range");
    }
    if (record >= view.n_records) {
        throw std::out_of_range("KVarN body record index is out of range");
    }
}

ggml_tensor * llama_kv_cache_kvarn::view_k_body_record(
        ggml_context * ctx, int32_t il, uint32_t ih, uint32_t record) const {
    const llama_kvarn_layer_view view = get_layer_view(il);
    assert_kvarn_record_view_bounds(view, ih, record);
    const size_t offset = size_t(ih)*view.body_k->nb[2] + size_t(record)*view.body_k->nb[1];
    ggml_tensor * result = ggml_view_1d(ctx, view.body_k, view.layout_k.k_body_bytes, offset);
    ggml_format_name(result, "kvarn_k_body_l%d_h%u_r%u", il, ih, record);
    return result;
}

ggml_tensor * llama_kv_cache_kvarn::view_v_body_record(
        ggml_context * ctx, int32_t il, uint32_t ih, uint32_t record) const {
    const llama_kvarn_layer_view view = get_layer_view(il);
    assert_kvarn_record_view_bounds(view, ih, record);
    const size_t offset = size_t(ih)*view.body_v->nb[2] + size_t(record)*view.body_v->nb[1];
    ggml_tensor * result = ggml_view_1d(ctx, view.body_v, view.layout_v.v_body_bytes, offset);
    ggml_format_name(result, "kvarn_v_body_l%d_h%u_r%u", il, ih, record);
    return result;
}

ggml_tensor * llama_kv_cache_kvarn::view_k_scales_record(
        ggml_context * ctx, int32_t il, uint32_t ih, uint32_t record) const {
    const llama_kvarn_layer_view view = get_layer_view(il);
    assert_kvarn_record_view_bounds(view, ih, record);
    const size_t offset = size_t(ih)*view.scales_k->nb[2] + size_t(record)*view.scales_k->nb[1];
    ggml_tensor * result = ggml_view_1d(ctx, view.scales_k, view.layout_k.k_scale_floats, offset);
    ggml_format_name(result, "kvarn_k_scales_l%d_h%u_r%u", il, ih, record);
    return result;
}

ggml_tensor * llama_kv_cache_kvarn::view_v_scales_record(
        ggml_context * ctx, int32_t il, uint32_t ih, uint32_t record) const {
    const llama_kvarn_layer_view view = get_layer_view(il);
    assert_kvarn_record_view_bounds(view, ih, record);
    const size_t offset = size_t(ih)*view.scales_v->nb[2] + size_t(record)*view.scales_v->nb[1];
    ggml_tensor * result = ggml_view_1d(ctx, view.scales_v, view.layout_v.v_scale_floats, offset);
    ggml_format_name(result, "kvarn_v_scales_l%d_h%u_r%u", il, ih, record);
    return result;
}

ggml_tensor * llama_kv_cache_kvarn::store_k_body_record(
        ggml_context * ctx,
        ggml_tensor * k_tile,
        ggml_tensor * scratch,
        int32_t il,
        uint32_t ih,
        uint32_t record) const {
    const llama_kvarn_layer_view view = get_layer_view(il);
    ggml_tensor * body   = view_k_body_record(ctx, il, ih, record);
    ggml_tensor * scales = view_k_scales_record(ctx, il, ih, record);
    return ggml_kvarn_store_k_body(
            ctx, k_tile, body, scales, scratch,
            view.head_dim_k, params.group_size, params.key_bits, params.sinkhorn_iters, params.rtn_quantile);
}

ggml_tensor * llama_kv_cache_kvarn::store_v_body_record(
        ggml_context * ctx,
        ggml_tensor * v_tile,
        ggml_tensor * scratch,
        int32_t il,
        uint32_t ih,
        uint32_t record) const {
    const llama_kvarn_layer_view view = get_layer_view(il);
    ggml_tensor * body   = view_v_body_record(ctx, il, ih, record);
    ggml_tensor * scales = view_v_scales_record(ctx, il, ih, record);
    return ggml_kvarn_store_v_body(
            ctx, v_tile, body, scales, scratch,
            view.head_dim_v, params.group_size, params.value_bits, params.sinkhorn_iters, params.rtn_quantile);
}

ggml_tensor * llama_kv_cache_kvarn::store_k_body_record_from_pending(
        ggml_context * ctx,
        ggml_tensor * scratch,
        int32_t il,
        uint32_t ih,
        uint32_t record) const {
    const size_t li = layer_storage_index(il);
    const llama_kvarn_layer_view view = get_layer_view(il);
    assert_kvarn_record_view_bounds(view, ih, record);
    GGML_ASSERT(view.head_dim_k == uint32_t(layer_tensors[li].pending_k->ne[0]));

    const size_t offset = size_t(ih)*layer_tensors[li].pending_k->nb[1];
    ggml_tensor * pending = ggml_view_2d(
            ctx, layer_tensors[li].pending_k,
            view.head_dim_k, params.group_size,
            layer_tensors[li].pending_k->nb[2], offset);
    ggml_tensor * k_tile = ggml_cont(ctx, ggml_transpose(ctx, pending));
    return store_k_body_record(ctx, k_tile, scratch, il, ih, record);
}

ggml_tensor * llama_kv_cache_kvarn::store_v_body_record_from_pending(
        ggml_context * ctx,
        ggml_tensor * scratch,
        int32_t il,
        uint32_t ih,
        uint32_t record) const {
    const size_t li = layer_storage_index(il);
    const llama_kvarn_layer_view view = get_layer_view(il);
    assert_kvarn_record_view_bounds(view, ih, record);
    GGML_ASSERT(view.head_dim_v == uint32_t(layer_tensors[li].pending_v->ne[0]));

    const size_t offset = size_t(ih)*layer_tensors[li].pending_v->nb[1];
    ggml_tensor * pending = ggml_view_2d(
            ctx, layer_tensors[li].pending_v,
            view.head_dim_v, params.group_size,
            layer_tensors[li].pending_v->nb[2], offset);
    ggml_tensor * v_tile = ggml_cont(ctx, pending);
    return store_v_body_record(ctx, v_tile, scratch, il, ih, record);
}

void llama_kv_cache_kvarn::append_layer_tokens_reference(
        uint32_t il,
        const std::vector<float> & k_tokens,
        const std::vector<float> & v_tokens,
        uint32_t n_tokens) {
    const size_t li = layer_storage_index(il);
    const uint32_t n_head_kv = layer_heads[li];
    const uint32_t head_k = kvarn_hparams_n_embd_head_k(hparams, il);
    const uint32_t head_v = kvarn_hparams_n_embd_head_v(hparams, il);

    if (head_k != head_v) {
        throw std::invalid_argument("KVarN reference append requires equal K and V head dimensions");
    }
    if (k_tokens.size() != size_t(n_tokens)*n_head_kv*head_k ||
        v_tokens.size() != size_t(n_tokens)*n_head_kv*head_v) {
        throw std::invalid_argument("KVarN reference append token buffer size mismatch");
    }

    std::vector<float> k_token(head_k);
    std::vector<float> v_token(head_v);
    for (uint32_t t = 0; t < n_tokens; ++t) {
        for (uint32_t ih = 0; ih < n_head_kv; ++ih) {
            const size_t off_k = (size_t(t)*n_head_kv + ih)*head_k;
            const size_t off_v = (size_t(t)*n_head_kv + ih)*head_v;
            std::copy(k_tokens.begin() + off_k, k_tokens.begin() + off_k + head_k, k_token.begin());
            std::copy(v_tokens.begin() + off_v, v_tokens.begin() + off_v + head_v, v_token.begin());
            runtime_cache[li][ih].append_token(k_token, v_token);
        }
    }
}

void llama_kv_cache_kvarn::materialize_layer_tokens_reference(
        uint32_t il,
        std::vector<float> & k_tokens,
        std::vector<float> & v_tokens) const {
    const size_t li = layer_storage_index(il);
    const uint32_t n_head_kv = layer_heads[li];
    const uint32_t head_k = kvarn_hparams_n_embd_head_k(hparams, il);
    const uint32_t head_v = kvarn_hparams_n_embd_head_v(hparams, il);

    std::vector<std::vector<float>> k_by_head(n_head_kv);
    std::vector<std::vector<float>> v_by_head(n_head_kv);

    uint32_t n_tokens = 0;
    for (uint32_t ih = 0; ih < n_head_kv; ++ih) {
        runtime_cache[li][ih].materialize_tokens(k_by_head[ih], v_by_head[ih]);
        const uint32_t n_head_tokens = uint32_t(k_by_head[ih].size()/head_k);
        if (ih == 0) {
            n_tokens = n_head_tokens;
        } else if (n_tokens != n_head_tokens) {
            throw std::runtime_error("KVarN runtime storage has inconsistent head token counts");
        }
    }

    k_tokens.assign(size_t(n_tokens)*n_head_kv*head_k, 0.0f);
    v_tokens.assign(size_t(n_tokens)*n_head_kv*head_v, 0.0f);

    for (uint32_t t = 0; t < n_tokens; ++t) {
        for (uint32_t ih = 0; ih < n_head_kv; ++ih) {
            const size_t dst_k = (size_t(t)*n_head_kv + ih)*head_k;
            const size_t dst_v = (size_t(t)*n_head_kv + ih)*head_v;
            const size_t src_k = size_t(t)*head_k;
            const size_t src_v = size_t(t)*head_v;
            std::copy(k_by_head[ih].begin() + src_k, k_by_head[ih].begin() + src_k + head_k, k_tokens.begin() + dst_k);
            std::copy(v_by_head[ih].begin() + src_v, v_by_head[ih].begin() + src_v + head_v, v_tokens.begin() + dst_v);
        }
    }
}

ggml_tensor * llama_kv_cache_kvarn::build_input_sink_tail_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    ggml_tensor * idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, ubatch.n_tokens);
    ggml_set_input(idxs);
    ggml_set_name(idxs, "kvarn_sink_tail_idxs");
    return idxs;
}

static uint32_t kvarn_count_tail_evictions(const llama_kvarn_params & params, const llama_ubatch & ubatch) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_pos pos = ubatch.pos ? ubatch.pos[i] : llama_pos(i);
        if (pos >= 0 && uint32_t(pos) >= params.sink_tokens + params.tail_tokens) {
            ++n;
        }
    }
    return n;
}

static int64_t kvarn_tail_slot(const llama_kvarn_params & params, uint32_t pos) {
    GGML_ASSERT(pos >= params.sink_tokens);
    GGML_ASSERT(params.tail_tokens > 0);
    return int64_t(params.sink_tokens + ((pos - params.sink_tokens)%params.tail_tokens));
}

void llama_kv_cache_kvarn::set_input_sink_tail_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(dst->type == GGML_TYPE_I64);
    GGML_ASSERT(dst->ne[0] == ubatch->n_tokens);

    int64_t * data = (int64_t *) dst->data;
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_pos pos = ubatch->pos ? ubatch->pos[i] : llama_pos(i);
        if (pos < 0) {
            data[i] = 0;
        } else if (uint32_t(pos) < params.sink_tokens) {
            data[i] = int64_t(pos);
        } else {
            data[i] = kvarn_tail_slot(params, uint32_t(pos));
        }
    }
}

ggml_tensor * llama_kv_cache_kvarn::build_input_body_plan(ggml_context * ctx, const llama_ubatch & ubatch) const {
    ggml_tensor * plan = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, 3, kvarn_count_tail_evictions(params, ubatch));
    ggml_set_input(plan);
    ggml_set_name(plan, "kvarn_body_plan");
    return plan;
}

void llama_kv_cache_kvarn::set_input_body_plan(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(dst->type == GGML_TYPE_I64);
    GGML_ASSERT(dst->ne[0] == 3);
    GGML_ASSERT(dst->ne[1] == kvarn_count_tail_evictions(params, *ubatch));

    int64_t * data = (int64_t *) dst->data;
    uint32_t j = 0;
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_pos pos = ubatch->pos ? ubatch->pos[i] : llama_pos(i);

        if (pos < 0 || uint32_t(pos) < params.sink_tokens + params.tail_tokens) {
            continue;
        }

        const uint32_t evicted_pos = uint32_t(pos) - params.tail_tokens;
        const uint32_t body_pos = evicted_pos - params.sink_tokens;
        const uint32_t record   = body_pos/params.group_size;
        const uint32_t offset   = body_pos%params.group_size;
        const size_t off = size_t(j++)*3;

        data[off + 0] = int64_t(record);
        data[off + 1] = int64_t(offset);
        data[off + 2] = offset + 1 == params.group_size ? int64_t(record) : -1;
    }
}

ggml_tensor * llama_kv_cache_kvarn::build_input_body_offsets(ggml_context * ctx, const llama_ubatch & ubatch) const {
    ggml_tensor * offsets = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, kvarn_count_tail_evictions(params, ubatch));
    ggml_set_input(offsets);
    ggml_set_name(offsets, "kvarn_body_offsets");
    return offsets;
}

void llama_kv_cache_kvarn::set_input_body_offsets(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(dst->type == GGML_TYPE_I64);
    GGML_ASSERT(dst->ne[0] == kvarn_count_tail_evictions(params, *ubatch));

    int64_t * data = (int64_t *) dst->data;
    uint32_t j = 0;
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_pos pos = ubatch->pos ? ubatch->pos[i] : llama_pos(i);
        if (pos < 0 || uint32_t(pos) < params.sink_tokens + params.tail_tokens) {
            continue;
        }

        const uint32_t evicted_pos = uint32_t(pos) - params.tail_tokens;
        data[j++] = int64_t((evicted_pos - params.sink_tokens)%params.group_size);
    }
}

ggml_tensor * llama_kv_cache_kvarn::build_input_tail_evict_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    ggml_tensor * idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kvarn_count_tail_evictions(params, ubatch));
    ggml_set_input(idxs);
    ggml_set_name(idxs, "kvarn_tail_evict_idxs");
    return idxs;
}

void llama_kv_cache_kvarn::set_input_tail_evict_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->ne[0] == kvarn_count_tail_evictions(params, *ubatch));

    int32_t * data = (int32_t *) dst->data;
    uint32_t j = 0;
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_pos pos = ubatch->pos ? ubatch->pos[i] : llama_pos(i);
        if (pos < 0 || uint32_t(pos) < params.sink_tokens + params.tail_tokens) {
            continue;
        }

        const uint32_t evicted_pos = uint32_t(pos) - params.tail_tokens;
        data[j++] = int32_t(kvarn_tail_slot(params, evicted_pos));
    }
}

void llama_kv_cache_kvarn::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->ne[0] == kv_size);
    GGML_ASSERT(dst->ne[1] == ubatch->n_tokens);

    const auto write = [&](int64_t t, int64_t q, float v) {
        char * p = (char *) dst->data + size_t(q)*dst->nb[1] + size_t(t)*dst->nb[0];
        if (dst->type == GGML_TYPE_F16) {
            *(ggml_fp16_t *) p = ggml_fp32_to_fp16(v);
        } else {
            *(float *) p = v;
        }
    };

    for (int64_t q = 0; q < dst->ne[1]; ++q) {
        const llama_pos pos = ubatch->pos ? ubatch->pos[q] : llama_pos(q);
        const int64_t visible = causal_attn && pos >= 0 ? std::min<int64_t>(int64_t(kv_size), int64_t(pos) + 1) : int64_t(kv_size);
        for (int64_t t = 0; t < dst->ne[0]; ++t) {
            write(t, q, t < visible ? 0.0f : -INFINITY);
        }
    }
}

ggml_tensor * llama_kv_cache_kvarn::cpy_sink_tail_k(
        ggml_context * ctx,
        ggml_tensor * k_cur,
        ggml_tensor * idxs,
        int32_t il) const {
    const size_t li = layer_storage_index(il);
    ggml_tensor * k = layer_tensors[li].sink_tail_k;

    const int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head      = k_cur->ne[1];
    const int64_t n_tokens    = k_cur->ne[2];
    const int64_t n_embd_gqa  = n_embd_head*n_head;

    GGML_ASSERT(n_embd_head == k->ne[0]);
    GGML_ASSERT(n_head == k->ne[1]);
    GGML_ASSERT(ggml_row_size(k_cur->type, n_embd_head) == k_cur->nb[1]);

    k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);
    ggml_tensor * k_view = ggml_reshape_2d(ctx, k, n_embd_gqa, k->ne[2]);
    return ggml_set_rows(ctx, k_view, k_cur, idxs);
}

ggml_tensor * llama_kv_cache_kvarn::cpy_sink_tail_v(
        ggml_context * ctx,
        ggml_tensor * v_cur,
        ggml_tensor * idxs,
        int32_t il) const {
    const size_t li = layer_storage_index(il);
    ggml_tensor * v = layer_tensors[li].sink_tail_v;

    const int64_t n_embd_head = v_cur->ne[0];
    const int64_t n_head      = v_cur->ne[1];
    const int64_t n_tokens    = v_cur->ne[2];
    const int64_t n_embd_gqa  = n_embd_head*n_head;

    GGML_ASSERT(n_embd_head == v->ne[0]);
    GGML_ASSERT(n_head == v->ne[1]);
    GGML_ASSERT(ggml_row_size(v_cur->type, n_embd_head) == v_cur->nb[1]);

    v_cur = ggml_view_2d(ctx, v_cur, n_embd_gqa, n_tokens, v_cur->nb[2], 0);
    ggml_tensor * v_view = ggml_reshape_2d(ctx, v, n_embd_gqa, v->ne[2]);
    return ggml_set_rows(ctx, v_view, v_cur, idxs);
}

ggml_tensor * llama_kv_cache_kvarn::cpy_tail_evict_pending_k(
        ggml_context * ctx,
        ggml_tensor * tail_idxs,
        ggml_tensor * offsets,
        int32_t il) const {
    const size_t li = layer_storage_index(il);
    ggml_tensor * src = layer_tensors[li].sink_tail_k;
    ggml_tensor * pending = layer_tensors[li].pending_k;

    const int64_t n_embd_head = src->ne[0];
    const int64_t n_head      = src->ne[1];
    const int64_t n_evict     = offsets->ne[0];
    const int64_t n_embd_gqa  = n_embd_head*n_head;

    GGML_ASSERT(n_embd_head == pending->ne[0]);
    GGML_ASSERT(n_head == pending->ne[1]);
    GGML_ASSERT(tail_idxs->type == GGML_TYPE_I32);
    GGML_ASSERT(offsets->type == GGML_TYPE_I64);
    GGML_ASSERT(tail_idxs->ne[0] == n_evict);

    src = ggml_reshape_2d(ctx, src, n_embd_gqa, src->ne[2]);
    ggml_tensor * evicted = ggml_cast(ctx, ggml_get_rows(ctx, src, tail_idxs), GGML_TYPE_F32);
    ggml_tensor * pending_view = ggml_reshape_2d(ctx, pending, n_embd_gqa, params.group_size);
    return ggml_set_rows(ctx, pending_view, evicted, offsets);
}

ggml_tensor * llama_kv_cache_kvarn::cpy_tail_evict_pending_v(
        ggml_context * ctx,
        ggml_tensor * tail_idxs,
        ggml_tensor * offsets,
        int32_t il) const {
    const size_t li = layer_storage_index(il);
    ggml_tensor * src = layer_tensors[li].sink_tail_v;
    ggml_tensor * pending = layer_tensors[li].pending_v;

    const int64_t n_embd_head = src->ne[0];
    const int64_t n_head      = src->ne[1];
    const int64_t n_evict     = offsets->ne[0];
    const int64_t n_embd_gqa  = n_embd_head*n_head;

    GGML_ASSERT(n_embd_head == pending->ne[0]);
    GGML_ASSERT(n_head == pending->ne[1]);
    GGML_ASSERT(tail_idxs->type == GGML_TYPE_I32);
    GGML_ASSERT(offsets->type == GGML_TYPE_I64);
    GGML_ASSERT(tail_idxs->ne[0] == n_evict);

    src = ggml_reshape_2d(ctx, src, n_embd_gqa, src->ne[2]);
    ggml_tensor * evicted = ggml_cast(ctx, ggml_get_rows(ctx, src, tail_idxs), GGML_TYPE_F32);
    ggml_tensor * pending_view = ggml_reshape_2d(ctx, pending, n_embd_gqa, params.group_size);
    return ggml_set_rows(ctx, pending_view, evicted, offsets);
}

llama_kv_cache_kvarn::slot_info llama_kv_cache_kvarn::find_slot(const llama_ubatch & ubatch) const {
    slot_info result;

    const uint32_t n_tokens = ubatch.n_tokens;
    if (n_tokens == 0 || n_tokens > kv_size) {
        return result;
    }

    const auto & cells = v_cells[0];
    const uint32_t head = v_heads[0] % kv_size;

    result.idxs.reserve(n_tokens);
    for (uint32_t scanned = 0; scanned < kv_size && result.idxs.size() < n_tokens; ++scanned) {
        const uint32_t idx = (head + scanned) % kv_size;
        if (cells.is_empty(idx)) {
            result.idxs.push_back(idx);
        }
    }

    if (result.idxs.size() != n_tokens) {
        result.idxs.clear();
    }

    return result;
}

llama_kv_cache_kvarn::slot_info_vec_t llama_kv_cache_kvarn::prepare(const std::vector<llama_ubatch> & ubatches) const {
    llama_kv_cells cells = v_cells[0];
    uint32_t head = v_heads[0];

    slot_info_vec_t result;
    result.reserve(ubatches.size());

    for (const llama_ubatch & ubatch : ubatches) {
        slot_info sinfo;

        const uint32_t n_tokens = ubatch.n_tokens;
        if (n_tokens == 0 || n_tokens > kv_size) {
            return {};
        }

        sinfo.idxs.reserve(n_tokens);
        for (uint32_t scanned = 0; scanned < kv_size && sinfo.idxs.size() < n_tokens; ++scanned) {
            const uint32_t idx = (head + scanned) % kv_size;
            if (cells.is_empty(idx)) {
                sinfo.idxs.push_back(idx);
            }
        }
        if (sinfo.idxs.size() != n_tokens) {
            return {};
        }

        for (uint32_t i = 0; i < n_tokens; ++i) {
            const uint32_t idx = sinfo.idxs[i];
            cells.pos_set(idx, ubatch.pos[i]);
            if (ubatch.is_pos_2d()) {
                llama_kv_cell_ext ext {
                    /*.x =*/ ubatch.pos[i + ubatch.n_tokens*2],
                    /*.y =*/ ubatch.pos[i + ubatch.n_tokens],
                };
                cells.ext_set(idx, ext);
            }
            for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
                cells.seq_add(idx, ubatch.seq_id[i][s]);
            }
        }

        head = sinfo.idxs.back() + 1;
        result.push_back(std::move(sinfo));
    }

    return result;
}

void llama_kv_cache_kvarn::apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch) {
    GGML_ASSERT(ubatch.n_tokens == sinfo.size());

    auto & cells = v_cells[0];

    llama_seq_id seq_pos_max_rm[LLAMA_MAX_SEQ];
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        seq_pos_max_rm[s] = -1;
    }

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const uint32_t idx = sinfo.idxs[i];

        if (!cells.is_empty(idx)) {
            assert(cells.seq_count(idx) == 1);
            const llama_seq_id old_seq = cells.seq_get(idx);
            const llama_pos old_pos = cells.pos_get(idx);
            seq_pos_max_rm[old_seq] = std::max(seq_pos_max_rm[old_seq], old_pos);
            cells.rm(idx);
        }

        cells.pos_set(idx, ubatch.pos[i]);
        if (ubatch.is_pos_2d()) {
            llama_kv_cell_ext ext {
                /*.x =*/ ubatch.pos[i + ubatch.n_tokens*2],
                /*.y =*/ ubatch.pos[i + ubatch.n_tokens],
            };
            cells.ext_set(idx, ext);
        }
        for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
            cells.seq_add(idx, ubatch.seq_id[i][s]);
        }
    }

    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (seq_pos_max_rm[s] != -1 && cells.seq_pos_min(s) <= seq_pos_max_rm[s]) {
            seq_rm(s, cells.seq_pos_min(s), seq_pos_max_rm[s] + 1);
        }
    }

    v_heads[0] = sinfo.idxs.back() + 1;
}
