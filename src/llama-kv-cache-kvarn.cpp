#include "llama-kv-cache-kvarn.h"

#ifdef LLAMA_BUILD
#include "llama-batch.h"
#include "llama-model.h"
#endif
#include "llama-hparams.h"
#include "llama-io.h"
#include "llama-kvarn-ubatch.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

static bool is_power_of_2(uint32_t n) {
    return n != 0 && (n & (n - 1)) == 0;
}

static size_t packed_nbytes(size_t n_values, uint32_t bits);

namespace {

constexpr std::array<uint8_t, 8> KVAR_N_STATE_MAGIC = { 'K', 'V', 'A', 'R', 'N', 'K', 'V', 0 };
constexpr uint32_t KVAR_N_STATE_VERSION = 1;
constexpr uint32_t KVAR_N_STATE_TENSORS_PER_LAYER = 8;

template<typename T>
void kvarn_state_write_uint(llama_io_write_i & io, T value) {
    static_assert(std::is_unsigned_v<T>);
    std::array<uint8_t, sizeof(T)> bytes = {};
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = uint8_t(value >> (8*i));
    }
    io.write(bytes.data(), bytes.size());
}

template<typename T>
T kvarn_state_read_uint(llama_io_read_i & io) {
    static_assert(std::is_unsigned_v<T>);
    std::array<uint8_t, sizeof(T)> bytes = {};
    io.read(bytes.data(), bytes.size());
    T value = 0;
    for (size_t i = 0; i < bytes.size(); ++i) {
        value |= T(bytes[i]) << (8*i);
    }
    return value;
}

void kvarn_state_write_i64(llama_io_write_i & io, int64_t value) {
    kvarn_state_write_uint(io, uint64_t(value));
}

int64_t kvarn_state_read_i64(llama_io_read_i & io) {
    const uint64_t bits = kvarn_state_read_uint<uint64_t>(io);
    int64_t value;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint32_t kvarn_float_bits(float value) {
    uint32_t bits;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void kvarn_state_write_layout(llama_io_write_i & io, const llama_kvarn_layout & layout) {
    kvarn_state_write_uint(io, layout.head_dim);
    kvarn_state_write_uint(io, layout.group_size);
    kvarn_state_write_uint(io, layout.key_bits);
    kvarn_state_write_uint(io, layout.value_bits);
    kvarn_state_write_uint(io, layout.v_layout);
    kvarn_state_write_uint(io, uint64_t(layout.k_body_bytes));
    kvarn_state_write_uint(io, uint64_t(layout.v_body_bytes));
    kvarn_state_write_uint(io, uint64_t(layout.k_scale_floats));
    kvarn_state_write_uint(io, uint64_t(layout.v_scale_floats));
    kvarn_state_write_uint(io, uint64_t(layout.total_record_bytes));
}

bool kvarn_state_read_layout_matches(llama_io_read_i & io, const llama_kvarn_layout & expected) {
    return kvarn_state_read_uint<uint32_t>(io) == expected.head_dim &&
           kvarn_state_read_uint<uint32_t>(io) == expected.group_size &&
           kvarn_state_read_uint<uint32_t>(io) == expected.key_bits &&
           kvarn_state_read_uint<uint32_t>(io) == expected.value_bits &&
           kvarn_state_read_uint<uint32_t>(io) == expected.v_layout &&
           kvarn_state_read_uint<uint64_t>(io) == uint64_t(expected.k_body_bytes) &&
           kvarn_state_read_uint<uint64_t>(io) == uint64_t(expected.v_body_bytes) &&
           kvarn_state_read_uint<uint64_t>(io) == uint64_t(expected.k_scale_floats) &&
           kvarn_state_read_uint<uint64_t>(io) == uint64_t(expected.v_scale_floats) &&
           kvarn_state_read_uint<uint64_t>(io) == uint64_t(expected.total_record_bytes);
}

std::array<ggml_tensor *, KVAR_N_STATE_TENSORS_PER_LAYER> kvarn_state_tensors(
        const llama_kvarn_layer_view & layer) {
    return { layer.sink_tail_k, layer.sink_tail_v, layer.body_k, layer.body_v,
             layer.scales_k, layer.scales_v, layer.pending_k, layer.pending_v };
}

void kvarn_state_require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

} // namespace

static bool kvarn_env_flag_enabled(const char * name) {
    const char * env = std::getenv(name);
    return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

static bool kvarn_env_flag_01_enabled(const char * name) {
    const char * env = std::getenv(name);
    if (env == nullptr) {
        return false;
    }

    char * end = nullptr;
    errno = 0;
    const long value = std::strtol(env, &end, 10);
    if (env[0] == '\0' || end == nullptr || *end != '\0' || errno == ERANGE ||
            (value != 0 && value != 1)) {
        throw std::invalid_argument(std::string("invalid KVarN environment flag ") + name +
                "=" + env + "; expected integer 0 or 1");
    }
    return value != 0;
}

static bool kvarn_experimental_turbo_v_layout_enabled() {
    return kvarn_env_flag_01_enabled("LLAMA_KVARN_EXPERIMENTAL_TURBO_V_LAYOUT");
}

static size_t kvarn_turbo_v_block_bytes(uint32_t value_bits) {
    if (value_bits == 2) {
        return 2 + 128/4;
    }
    if (value_bits == 4) {
        return 2 + 2 + 128/2;
    }
    throw std::invalid_argument("Turbo V layout currently supports only value_bits 2 or 4");
}

static size_t kvarn_v_body_bytes(uint32_t head_dim, uint32_t group_size, uint32_t value_bits) {
    if (!kvarn_experimental_turbo_v_layout_enabled()) {
        return packed_nbytes(size_t(head_dim)*group_size, value_bits);
    }
    if (group_size != 128 || (head_dim % 128) != 0) {
        throw std::invalid_argument("LLAMA_KVARN_EXPERIMENTAL_TURBO_V_LAYOUT requires group_size=128 and head_dim multiple of 128");
    }
    return size_t(group_size)*(head_dim/128u)*kvarn_turbo_v_block_bytes(value_bits);
}

static uint32_t kvarn_v_layout_id() {
    return kvarn_experimental_turbo_v_layout_enabled() ?
        LLAMA_KVARN_V_LAYOUT_TURBO_CANONICAL : LLAMA_KVARN_V_LAYOUT_LEGACY;
}

static std::string kvarn_trim(std::string s) {
    const char * ws = " \t\r\n";
    const size_t first = s.find_first_not_of(ws);
    if (first == std::string::npos) {
        return "";
    }
    const size_t last = s.find_last_not_of(ws);
    return s.substr(first, last - first + 1);
}

static bool kvarn_parse_uint32_token(const std::string & text, uint32_t & out) {
    if (text.empty()) {
        return false;
    }
    char * end = nullptr;
    errno = 0;
    const unsigned long value = std::strtoul(text.c_str(), &end, 10);
    if (end == nullptr || *end != '\0' || errno == ERANGE || value > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    out = uint32_t(value);
    return true;
}

static bool kvarn_layer_spec_token_contains(const std::string & raw, uint32_t il) {
    const std::string token = kvarn_trim(raw);
    if (token.empty()) {
        return false;
    }

    const size_t dash = token.find('-');
    if (dash == std::string::npos) {
        uint32_t single = 0;
        if (!kvarn_parse_uint32_token(token, single)) {
            throw std::invalid_argument("invalid KVarN layer-bit override layer id: " + token);
        }
        return single == il;
    }

    const size_t colon = token.find(':', dash + 1);
    const std::string start_text = token.substr(0, dash);
    const std::string end_text = colon == std::string::npos ? token.substr(dash + 1) : token.substr(dash + 1, colon - dash - 1);
    const std::string step_text = colon == std::string::npos ? "1" : token.substr(colon + 1);

    uint32_t start = 0;
    uint32_t end = 0;
    uint32_t step = 0;
    if (!kvarn_parse_uint32_token(kvarn_trim(start_text), start) ||
            !kvarn_parse_uint32_token(kvarn_trim(end_text), end) ||
            !kvarn_parse_uint32_token(kvarn_trim(step_text), step) ||
            step == 0 || end < start) {
        throw std::invalid_argument("invalid KVarN layer-bit override range: " + token);
    }

    return il >= start && il <= end && ((il - start)%step) == 0;
}

static bool kvarn_layer_spec_contains(const std::string & spec, uint32_t il) {
    size_t pos = 0;
    while (pos <= spec.size()) {
        const size_t comma = spec.find(',', pos);
        const std::string token = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (kvarn_layer_spec_token_contains(token, il)) {
            return true;
        }
        if (comma == std::string::npos) {
            break;
        }
        pos = comma + 1;
    }
    return false;
}

static bool kvarn_apply_layer_bit_env(
        const char * env_name,
        uint32_t il,
        uint32_t & bits) {
    const char * env = std::getenv(env_name);
    if (env == nullptr || env[0] == '\0') {
        return false;
    }

    bool changed = false;
    const std::string rules(env);
    size_t pos = 0;
    while (pos <= rules.size()) {
        const size_t semi = rules.find(';', pos);
        const std::string rule = kvarn_trim(rules.substr(pos, semi == std::string::npos ? std::string::npos : semi - pos));
        if (!rule.empty()) {
            const size_t eq = rule.find('=');
            if (eq == std::string::npos) {
                throw std::invalid_argument(std::string(env_name) + " rule must be '<layers>=<bits>': " + rule);
            }

            uint32_t parsed_bits = 0;
            if (!kvarn_parse_uint32_token(kvarn_trim(rule.substr(eq + 1)), parsed_bits) || parsed_bits < 2 || parsed_bits > 8) {
                throw std::invalid_argument(std::string(env_name) + " bits must be in [2,8]: " + rule);
            }

            if (kvarn_layer_spec_contains(rule.substr(0, eq), il)) {
                bits = parsed_bits;
                changed = true;
            }
        }
        if (semi == std::string::npos) {
            break;
        }
        pos = semi + 1;
    }
    return changed;
}

static llama_kvarn_params kvarn_params_for_layer(llama_kvarn_params params, uint32_t il, bool log_changes = false) {
    const uint32_t old_k = params.key_bits;
    const uint32_t old_v = params.value_bits;
    const bool changed_k = kvarn_apply_layer_bit_env("LLAMA_KVARN_LAYER_KEY_BITS", il, params.key_bits);
    const bool changed_v = kvarn_apply_layer_bit_env("LLAMA_KVARN_LAYER_VALUE_BITS", il, params.value_bits);
    if (log_changes && (changed_k || changed_v) && (old_k != params.key_bits || old_v != params.value_bits)) {
        std::fprintf(stderr,
                "llama_kv_cache_kvarn: layer %u KVarN bit override k%u/v%u -> k%u/v%u\n",
                il, old_k, old_v, params.key_bits, params.value_bits);
    }
    return params;
}

static llama_kvarn_params kvarn_apply_high_gqa_policy(
        llama_kvarn_params params,
        uint32_t il,
        uint32_t n_head,
        uint32_t n_head_kv,
        bool log_changes = false) {
    if (n_head_kv == 0 || params.key_bits >= 8 ||
            kvarn_env_flag_01_enabled("LLAMA_KVARN_DISABLE_HIGH_GQA_K8")) {
        return params;
    }

    const uint32_t gqa_ratio = n_head/n_head_kv;
    if (n_head < 6*n_head_kv) {
        return params;
    }

    const uint32_t old_k = params.key_bits;
    params.key_bits = 8;
    if (log_changes) {
        std::fprintf(stderr,
                "llama_kv_cache_kvarn: layer %u high-GQA ratio %u:%u promotes KVarN key bits k%u -> k8 "
                "(set LLAMA_KVARN_DISABLE_HIGH_GQA_K8=1 only for ablations)\n",
                il, n_head, n_head_kv, old_k);
    }
    GGML_UNUSED(gqa_ratio);
    return params;
}

static llama_kvarn_params kvarn_params_for_layer(
        llama_kvarn_params params,
        const llama_hparams & hparams,
        uint32_t il,
        bool log_changes = false) {
    params = kvarn_params_for_layer(params, il, log_changes);
    return kvarn_apply_high_gqa_policy(params, il, hparams.n_head(il), hparams.n_head_kv(il), log_changes);
}

#ifdef LLAMA_BUILD
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

#endif

static size_t packed_nbytes(size_t n_values, uint32_t bits) {
    return (n_values*bits + 7)/8;
}

static bool kvarn_hparams_has_kv(const llama_hparams & hparams, uint32_t il) {
    if (hparams.kv_only_nextn) {
        return hparams.n_layer_nextn > 0 && il >= hparams.n_layer();
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

static bool log_std_sinkhorn_enabled() {
    if (kvarn_env_flag_enabled("LLAMA_KVARN_DISABLE_LOG_STD_SINKHORN")) {
        return false;
    }
    const char * env = std::getenv("LLAMA_KVARN_ENABLE_LOG_STD_SINKHORN");
    if (env != nullptr) {
        return env[0] != '\0' && std::strcmp(env, "0") != 0;
    }
    return true;
}

static float log_std_scale_update(float prev, float stdv) {
    stdv = std::min(1.0e3f, std::max(1.0e-3f, stdv));
    const float log_prev = std::log(std::max(prev, 1.0e-20f));
    const float log_next = std::min(10.0f, std::max(-0.3f, log_prev + std::log(stdv)));
    return std::exp(log_next);
}

// The reference log-std recipe clamps accumulated log scales to [-0.3, 10].
// Raw K/V tiles with global RMS below ~0.74 pin every row/column scale at the
// clamp floor, so no variance is equalized at all. Normalizing the tile by its
// global RMS first keeps the clamps as degenerate-row guards while letting the
// dual scaling act at any tile magnitude; the global factor is folded back into
// the stored row scales so the packed format and dequant are unchanged.
static bool kvarn_global_norm_enabled() {
    return !kvarn_env_flag_enabled("LLAMA_KVARN_DISABLE_GLOBAL_NORM");
}

// MSE-optimal clip half-width in std units for uniform asymmetric RTN on
// variance-normalized (~Gaussian) rows. Full range is already near-optimal
// for bits >= 4; at 2 bits clipping recovers ~3x error energy.
static float kvarn_rtn_clip_sigma(uint32_t bits) {
    if (kvarn_env_flag_enabled("LLAMA_KVARN_DISABLE_RTN_CLIP")) {
        return 0.0f;
    }
    switch (bits) {
        case 1: return 1.0f;
        case 2: return 1.5f;
        case 3: return 2.05f;
        default: return 0.0f;
    }
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

    if (log_std_sinkhorn_enabled()) {
        float global_scale = 1.0f;
        if (iters > 0 && kvarn_global_norm_enabled()) {
            double ss_all = 0.0;
            for (const float v : data) {
                ss_all += double(v)*double(v);
            }
            const float rms = float(std::sqrt(ss_all/(double(rows)*double(cols))));
            if (rms > 1.0e-20f) {
                global_scale = rms;
                const float inv = 1.0f/rms;
                for (float & v : data) {
                    v *= inv;
                }
            }
        }

        auto imbalance = [&](const std::vector<float> & tile) {
            float col_min = std::numeric_limits<float>::max();
            float col_max = 0.0f;
            for (uint32_t c = 0; c < cols; ++c) {
                double sum = 0.0;
                double ss = 0.0;
                for (uint32_t r = 0; r < rows; ++r) {
                    const float v = tile[r*cols + c];
                    sum += double(v);
                    ss += double(v)*double(v);
                }
                double var = 0.0;
                if (rows > 1) {
                    var = (ss - (sum*sum)/double(rows))/double(rows - 1);
                }
                const float stdv = std::sqrt(float(std::max(0.0, var)));
                col_min = std::min(col_min, stdv);
                col_max = std::max(col_max, stdv);
            }

            float row_min = std::numeric_limits<float>::max();
            float row_max = 0.0f;
            for (uint32_t r = 0; r < rows; ++r) {
                double sum = 0.0;
                double ss = 0.0;
                for (uint32_t c = 0; c < cols; ++c) {
                    const float v = tile[r*cols + c];
                    sum += double(v);
                    ss += double(v)*double(v);
                }
                double var = 0.0;
                if (cols > 1) {
                    var = (ss - (sum*sum)/double(cols))/double(cols - 1);
                }
                const float stdv = std::sqrt(float(std::max(0.0, var)));
                row_min = std::min(row_min, stdv);
                row_max = std::max(row_max, stdv);
            }

            return col_max/std::max(col_min, 1.0e-8f) + row_max/std::max(row_min, 1.0e-8f);
        };

        float best_imb = imbalance(data);
        std::vector<float> best_data = data;
        std::vector<float> best_row_scale = row_scale;
        std::vector<float> best_col_scale = col_scale;

        for (uint32_t iter = 0; iter < iters; ++iter) {
            for (uint32_t c = 0; c < cols; ++c) {
                double sum = 0.0;
                double ss = 0.0;
                for (uint32_t r = 0; r < rows; ++r) {
                    const float v = data[r*cols + c];
                    sum += double(v);
                    ss += double(v)*double(v);
                }
                double var = 0.0;
                if (rows > 1) {
                    var = (ss - (sum*sum)/double(rows))/double(rows - 1);
                }
                const float prev = col_scale[c];
                const float next = log_std_scale_update(prev, std::sqrt(float(std::max(0.0, var))));
                col_scale[c] = next;
                const float factor = prev/next;
                for (uint32_t r = 0; r < rows; ++r) {
                    data[r*cols + c] *= factor;
                }
            }

            for (uint32_t r = 0; r < rows; ++r) {
                double sum = 0.0;
                double ss = 0.0;
                for (uint32_t c = 0; c < cols; ++c) {
                    const float v = data[r*cols + c];
                    sum += double(v);
                    ss += double(v)*double(v);
                }
                double var = 0.0;
                if (cols > 1) {
                    var = (ss - (sum*sum)/double(cols))/double(cols - 1);
                }
                const float prev = row_scale[r];
                const float next = log_std_scale_update(prev, std::sqrt(float(std::max(0.0, var))));
                row_scale[r] = next;
                const float factor = prev/next;
                for (uint32_t c = 0; c < cols; ++c) {
                    data[r*cols + c] *= factor;
                }
            }

            const float imb = imbalance(data);
            if (imb <= best_imb) {
                best_imb = imb;
                best_data = data;
                best_row_scale = row_scale;
                best_col_scale = col_scale;
            }
        }
        data = std::move(best_data);
        row_scale = std::move(best_row_scale);
        col_scale = std::move(best_col_scale);
        if (global_scale != 1.0f) {
            for (uint32_t r = 0; r < rows; ++r) {
                row_scale[r] *= global_scale;
            }
        }
        return;
    }

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
    const float clip_sigma = qt >= 1.0f ? kvarn_rtn_clip_sigma(bits) : 0.0f;

    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            work[c] = src[r*cols + c];
        }
        std::sort(work.begin(), work.end());

        const size_t lo_i = size_t((1.0f - qt)*0.5f*(cols - 1));
        const size_t hi_i = size_t((1.0f - (1.0f - qt)*0.5f)*(cols - 1));
        float mn = work[lo_i];
        float mx = work[hi_i];
        if (clip_sigma > 0.0f && cols > 1) {
            double sum = 0.0;
            double ss = 0.0;
            for (uint32_t c = 0; c < cols; ++c) {
                const float v = src[r*cols + c];
                sum += double(v);
                ss += double(v)*double(v);
            }
            const double n = double(cols);
            const float mu = float(sum/n);
            const float sd = float(std::sqrt(std::max(0.0, ss/n - (sum/n)*(sum/n))));
            const float lo = std::max(mn, mu - clip_sigma*sd);
            const float hi = std::min(mx, mu + clip_sigma*sd);
            if (hi > lo) {
                mn = lo;
                mx = hi;
            }
        }
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
        /*.v_layout          =*/ kvarn_v_layout_id(),
        /*.k_body_bytes      =*/ packed_nbytes(size_t(head_dim)*params.group_size, params.key_bits),
        /*.v_body_bytes      =*/ kvarn_v_body_bytes(head_dim, params.group_size, params.value_bits),
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

std::vector<float> llama_kvarn_hadamard_matrix(uint32_t d) {
    if (!is_power_of_2(d)) {
        throw std::invalid_argument("KVarN Hadamard matrix dim must be a power of two");
    }

    std::vector<float> H(size_t(d)*d, 0.0f);
    std::vector<float> col(d, 0.0f);
    const float norm = 1.0f/std::sqrt(float(d));
    for (uint32_t j = 0; j < d; ++j) {
        std::fill(col.begin(), col.end(), 0.0f);
        col[j] = 1.0f;
        for (uint32_t step = 1; step < d; step <<= 1) {
            for (uint32_t base = 0; base < d; base += 2*step) {
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
        for (uint32_t r = 0; r < d; ++r) {
            H[size_t(r)*d + j] = col[r]*norm;
        }
    }
    return H;
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
    if (layout.v_layout == LLAMA_KVARN_V_LAYOUT_TURBO_CANONICAL) {
        throw std::runtime_error("KVarN CPU reference store does not implement canonical Turbo V layout");
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
    if (layout.v_layout == LLAMA_KVARN_V_LAYOUT_TURBO_CANONICAL) {
        throw std::runtime_error("KVarN CPU reference dequant does not implement canonical Turbo V layout");
    }
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

    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
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

        const llama_kvarn_params layer_params = kvarn_params_for_layer(params, hparams, il);
        const llama_kvarn_layout layout_k = llama_kvarn_make_layout(layer_params, head_k);
        const llama_kvarn_layout layout_v = llama_kvarn_make_layout(layer_params, head_v);
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

bool llama_kv_cache_kvarn_context::has_layer(int32_t il) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return kv->has_layer(il);
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

int64_t llama_kv_cache_kvarn_context::attn_mixed_scratch_floats_worst(int32_t il) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return kv->attn_mixed_scratch_floats_worst(il);
}

ggml_tensor * llama_kv_cache_kvarn_context::build_attn_mixed_scratch(
        ggml_context * ctx, int32_t il, int64_t n_floats) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return kv->build_attn_mixed_scratch(ctx, il, n_floats);
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

ggml_tensor * llama_kv_cache_kvarn_context::store_kv_body_record(
        ggml_context * ctx,
        ggml_tensor * k_tile,
        ggml_tensor * v_tile,
        ggml_tensor * scratch,
        int32_t il,
        uint32_t ih,
        uint32_t record) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return kv->store_kv_body_record(ctx, k_tile, v_tile, scratch, il, ih, record);
}

ggml_tensor * llama_kv_cache_kvarn_context::store_kv_body_all_heads(
        ggml_context * ctx,
        ggml_tensor * k_tile,
        ggml_tensor * v_tile,
        ggml_tensor * scratch,
        int32_t il,
        uint32_t record) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return kv->store_kv_body_all_heads(ctx, k_tile, v_tile, scratch, il, record);
}

ggml_tensor * llama_kv_cache_kvarn_context::store_kv_body_records_all_heads(
        ggml_context * ctx,
        ggml_tensor * k_tiles,
        ggml_tensor * v_tiles,
        ggml_tensor * scratch,
        int32_t il,
        uint32_t record0,
        uint32_t n_records) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return kv->store_kv_body_records_all_heads(ctx, k_tiles, v_tiles, scratch, il, record0, n_records);
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

ggml_tensor * llama_kv_cache_kvarn_context::store_kv_body_record_from_pending(
        ggml_context * ctx,
        ggml_tensor * scratch,
        int32_t il,
        uint32_t ih,
        uint32_t record,
        ggml_tensor * pending_k,
        ggml_tensor * pending_v) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return kv->store_kv_body_record_from_pending(ctx, scratch, il, ih, record, pending_k, pending_v);
}

ggml_tensor * llama_kv_cache_kvarn_context::store_kv_body_all_heads_from_pending(
        ggml_context * ctx,
        ggml_tensor * scratch,
        int32_t il,
        uint32_t record,
        ggml_tensor * pending_k,
        ggml_tensor * pending_v) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return kv->store_kv_body_all_heads_from_pending(ctx, scratch, il, record, pending_k, pending_v);
}

ggml_tensor * llama_kv_cache_kvarn_context::store_kv_body_records_from_pending(
        ggml_context * ctx,
        ggml_tensor * scratch,
        int32_t il,
        const std::vector<uint32_t> & records,
        ggml_tensor * pending_k,
        ggml_tensor * pending_v) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return kv->store_kv_body_records_from_pending(ctx, scratch, il, records, pending_k, pending_v);
}

llama_kv_cache_kvarn::llama_kv_cache_kvarn(
        const llama_model * model,
        const llama_hparams & hparams,
        llama_kvarn_params params,
        bool offload,
        uint32_t kv_size,
        uint32_t n_seq_max,
        uint32_t n_pad,
        const layer_filter_cb & filter,
        const layer_reuse_cb & reuse) :
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
    if (uint64_t(params.sink_tokens) + uint64_t(params.tail_tokens) > kv_size) {
        if (params.sink_tokens >= kv_size) {
            throw std::invalid_argument("KVarN sink tokens must be smaller than the KV cache size");
        }
        const uint32_t clamped_tail = kv_size - params.sink_tokens;
        std::fprintf(stderr,
                "%s: clamping KVarN tail tokens from %u to %u because sink+tail exceeds kv_size=%u\n",
                __func__, params.tail_tokens, clamped_tail, kv_size);
        params.tail_tokens = clamped_tail;
        this->params.tail_tokens = clamped_tail;
    }

    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params init_params = {
                /*.mem_size   =*/ size_t(9u*hparams.n_layer_all*ggml_tensor_overhead()),
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

    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
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
        map_layer_ids[il] = int32_t(layer_tensors.size());
        const uint32_t n_head_kv = hparams.n_head_kv_arr[il];
        const uint32_t head_k = kvarn_hparams_n_embd_head_k(hparams, il);
        const uint32_t head_v = kvarn_hparams_n_embd_head_v(hparams, il);
        if (head_k != head_v) {
            throw std::invalid_argument("KVarN cache requires equal K and V head dimensions");
        }
        const llama_kvarn_params layer_params = kvarn_params_for_layer(params, hparams, il, true);
        const llama_kvarn_layout layout_k = llama_kvarn_make_layout(layer_params, head_k);
        const llama_kvarn_layout layout_v = llama_kvarn_make_layout(layer_params, head_v);
        layer_heads.push_back(n_head_kv);

        layer_storage st = {};
        st.il = il;
        st.n_head_kv = n_head_kv;
        st.n_sink_tail = n_sink_tail;
        st.n_records = n_records;
        st.layout_k = layout_k;
        st.layout_v = layout_v;
        st.sink_tail_k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, head_k, n_head_kv, n_sink_tail_alloc);
        st.sink_tail_v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, head_v, n_head_kv, n_sink_tail_alloc);
        st.body_k      = ggml_new_tensor_3d(ctx, GGML_TYPE_I8,  layout_k.k_body_bytes,   n_records_alloc, n_head_kv);
        st.body_v      = ggml_new_tensor_3d(ctx, GGML_TYPE_I8,  layout_v.v_body_bytes,   n_records_alloc, n_head_kv);
        st.scales_k    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, layout_k.k_scale_floats, n_records_alloc, n_head_kv);
        st.scales_v    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, layout_v.v_scale_floats, n_records_alloc, n_head_kv);
        st.pending_k   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_k, n_head_kv, params.group_size);
        st.pending_v   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_v, n_head_kv, params.group_size);

        {
            const uint32_t n_seen = kv_size;
            const uint32_t n_sink = std::min(n_seen, params.sink_tokens);
            const uint32_t n_after_sink = n_seen - n_sink;
            const uint32_t n_tail = std::min(n_after_sink, params.tail_tokens);
            const uint32_t n_body = n_after_sink - n_tail;
            const uint32_t n_records_w = std::min(n_body/params.group_size, n_records_alloc);
            const uint32_t n_pending = n_body%params.group_size;
            int64_t scratch_floats = int64_t(n_sink) + int64_t(n_records_w)*params.group_size + n_pending + n_tail;
            const bool mirror_scratch =
                kvarn_env_flag_enabled("LLAMA_KVARN_ATTN_REF_SCRATCH") ||
                kvarn_env_flag_enabled("LLAMA_KVARN_ENABLE_F32_DEQUANT_CACHE") ||
                kvarn_env_flag_enabled("LLAMA_KVARN_ATTN_ENABLE_BODY_F32_MIRROR");
            if (n_records_alloc > 0 && mirror_scratch) {
                scratch_floats += 2*int64_t(n_head_kv)*int64_t(n_records_alloc)*int64_t(head_k)*int64_t(params.group_size);
            }
            st.attn_mixed_scratch = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, scratch_floats);
            ggml_format_name(st.attn_mixed_scratch, "kvarn_attn_mixed_scratch_l%d", il);
        }

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
            caches.emplace_back(layer_params, kvarn_hparams_n_embd_head_k(hparams, il));
        }
        runtime_cache.push_back(std::move(caches));

        std::fprintf(stderr,
                "%s: KVarN layer %3u storage dev = %s, heads = %u, body records = %u, "
                "requested k%u/v%u effective k%u/v%u\n",
                __func__, il, dev_name, n_head_kv, n_records,
                params.key_bits, params.value_bits,
                layout_k.key_bits, layout_v.value_bits);
    }

    if (reuse) {
        for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
            const int32_t il_reuse = reuse(il);
            if (il_reuse < 0) {
                continue;
            }

            if (filter && !filter(il)) {
                continue;
            }

            const auto it = map_layer_ids.find(il_reuse);
            if (it == map_layer_ids.end()) {
                throw std::runtime_error("KVarN cache layer cannot reuse a missing physical layer");
            }

            map_layer_ids[int32_t(il)] = it->second;
        }
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
    bool invalid_debug_ubatch = false;
    const uint32_t default_kvarn_ubatch = n_ubatch;
    const uint32_t n_kvarn_ubatch = kvarn_ubatch_limit(default_kvarn_ubatch, invalid_debug_ubatch);
    if (invalid_debug_ubatch) {
        std::fprintf(stderr,
                "%s: KVarN debug ubatch override must be a positive integer: LLAMA_KVARN_DEBUG_UBATCH=%s\n",
                __func__, std::getenv("LLAMA_KVARN_DEBUG_UBATCH"));
        return std::make_unique<llama_kv_cache_kvarn_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }
    if (params.tail_tokens == 0) {
        std::fprintf(stderr,
                "%s: KVarN requires a non-empty tail ring\n",
                __func__);
        return std::make_unique<llama_kv_cache_kvarn_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    while (true) {
        const uint32_t n_safe_ubatch =
            kvarn_tail_safe_ubatch_limit(balloc, n_kvarn_ubatch, params.sink_tokens, params.tail_tokens);
        auto ubatch = balloc.split_simple(n_safe_ubatch);
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
    // K is stored post-RoPE (rotated f16 sink/tail + quantized body records)
    // and there is no KVarN shift graph: init_update() fails the decode when
    // cells carry a pending shift. Advertising shift support invites
    // llama-server context-shift, which would either brick the session or
    // silently attend with mis-rotated K. Refuse so callers fall back to
    // reprocessing.
    return false;
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
    GGML_ASSERT(seq_id < 0 || (size_t) seq_id < seq_to_stream.size());

    if (p0 < 0) {
        p0 = 0;
    }
    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }
    if (p0 >= p1) {
        return true;
    }

    // Physical storage is position-derived. Refuse every overlapping partial
    // removal so the advertised capability cannot change after the tail ring
    // starts reusing slots. Full removal and true no-ops remain exact.
    {
        const auto & cells_ro = v_cells[0];
        llama_pos cur_min = std::numeric_limits<llama_pos>::max();
        llama_pos cur_max = -1;
        for (uint32_t i = 0; i < cells_ro.size(); ++i) {
            if (cells_ro.is_empty(i)) {
                continue;
            }
            if (seq_id >= 0 && !cells_ro.seq_has(i, seq_id)) {
                continue;
            }
            const llama_pos pos = cells_ro.pos_get(i);
            cur_min = std::min(cur_min, pos);
            cur_max = std::max(cur_max, pos);
        }

        if (cur_max >= 0 && p0 <= cur_max && p1 > cur_min) { // removal overlaps stored tokens
            const bool full_removal = p0 <= cur_min && p1 > cur_max;
            if (!full_removal) {
                return false;
            }
        }
    }

    return seq_rm_cells(seq_id, p0, p1);
}

bool llama_kv_cache_kvarn::seq_rm_cells(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
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

    if (shift == 0) {
        return;
    }
    GGML_ASSERT(get_can_shift() && "seq_add() is only supported for n_pos_per_embd() == 1");
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

        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());
        if (hparams.no_alloc) {
            GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) == nullptr);
            result[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            result[buft] += ggml_backend_buffer_get_size(buf.get());
        }
    }

    if (result.empty()) {
        result[ggml_backend_cpu_buffer_type()] = mem_estimate.total_bytes;
    }

    return result;
}

void llama_kv_cache_kvarn::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if (seq_id != -1 || flags != 0) {
        throw std::runtime_error("KVarN state serialization supports only full-cache state with no flags");
    }
    if (v_cells.size() != 1 || v_heads.size() != 1) {
        throw std::runtime_error("KVarN state serialization requires exactly one metadata stream");
    }
    if (v_cells[0].get_has_shift()) {
        throw std::runtime_error("KVarN state serialization does not support shifted cells");
    }

    io.write(KVAR_N_STATE_MAGIC.data(), KVAR_N_STATE_MAGIC.size());
    kvarn_state_write_uint(io, KVAR_N_STATE_VERSION);
    kvarn_state_write_uint(io, kv_size);
    kvarn_state_write_uint(io, n_seq_max);
    kvarn_state_write_uint(io, n_pad);
    kvarn_state_write_uint(io, params.group_size);
    kvarn_state_write_uint(io, params.key_bits);
    kvarn_state_write_uint(io, params.value_bits);
    kvarn_state_write_uint(io, params.sink_tokens);
    kvarn_state_write_uint(io, params.tail_tokens);
    kvarn_state_write_uint(io, params.sinkhorn_iters);
    kvarn_state_write_uint(io, kvarn_float_bits(params.rtn_quantile));
    kvarn_state_write_uint(io, uint32_t(layer_tensors.size()));

    std::set<uint32_t> unique_layers;
    for (const layer_storage & layer : layer_tensors) {
        if (!unique_layers.insert(layer.il).second) {
            throw std::runtime_error("KVarN state serialization found duplicate physical layer id");
        }
        kvarn_state_write_uint(io, layer.il);
        kvarn_state_write_uint(io, layer.n_head_kv);
        kvarn_state_write_uint(io, layer.n_sink_tail);
        kvarn_state_write_uint(io, layer.n_records);
        kvarn_state_write_layout(io, layer.layout_k);
        kvarn_state_write_layout(io, layer.layout_v);
    }

    kvarn_state_write_uint(io, v_heads[0]);
    kvarn_state_write_uint(io, uint32_t(v_cells[0].size()));
    for (uint32_t i = 0; i < v_cells[0].size(); ++i) {
        const bool occupied = !v_cells[0].is_empty(i);
        kvarn_state_write_uint<uint8_t>(io, occupied ? 1 : 0);
        kvarn_state_write_i64(io, occupied ? int64_t(v_cells[0].pos_get(i)) : -1);
        const llama_kv_cell_ext ext = occupied ? v_cells[0].ext_get(i) : llama_kv_cell_ext{};
        kvarn_state_write_i64(io, int64_t(ext.x));
        kvarn_state_write_i64(io, int64_t(ext.y));
        std::vector<uint32_t> seq_ids;
        if (occupied) {
            for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (v_cells[0].seq_has(i, llama_seq_id(s))) {
                    if (s >= n_seq_max) {
                        throw std::runtime_error("KVarN state cell contains sequence id outside n_seq_max");
                    }
                    seq_ids.push_back(s);
                }
            }
            if (seq_ids.size() != size_t(v_cells[0].seq_count(i))) {
                throw std::runtime_error("KVarN state cell sequence metadata is inconsistent");
            }
        }
        kvarn_state_write_uint(io, uint32_t(seq_ids.size()));
        for (uint32_t s : seq_ids) {
            kvarn_state_write_uint(io, s);
        }
    }

    // Descriptors intentionally precede all payloads, allowing readers to
    // reject every incompatibility before mutating backend storage.
    for (const layer_storage & storage : layer_tensors) {
        const auto tensors = kvarn_state_tensors(get_layer_view(storage.il));
        for (uint32_t it = 0; it < tensors.size(); ++it) {
            const ggml_tensor * tensor = tensors[it];
            kvarn_state_write_uint(io, it);
            kvarn_state_write_uint(io, uint32_t(tensor->type));
            for (uint32_t d = 0; d < GGML_MAX_DIMS; ++d) {
                kvarn_state_write_uint(io, uint64_t(tensor->ne[d]));
            }
            kvarn_state_write_uint(io, uint64_t(ggml_nbytes(tensor)));
        }
    }
    for (const layer_storage & storage : layer_tensors) {
        const auto tensors = kvarn_state_tensors(get_layer_view(storage.il));
        for (ggml_tensor * tensor : tensors) {
            io.write_tensor(tensor, 0, ggml_nbytes(tensor));
        }
    }
}

void llama_kv_cache_kvarn::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (seq_id != -1 || flags != 0) {
        throw std::runtime_error("KVarN state deserialization supports only full-cache state with no flags");
    }

    try {
        std::array<uint8_t, KVAR_N_STATE_MAGIC.size()> magic = {};
        io.read(magic.data(), magic.size());
        kvarn_state_require(magic == KVAR_N_STATE_MAGIC, "invalid KVarN state magic");
        kvarn_state_require(kvarn_state_read_uint<uint32_t>(io) == KVAR_N_STATE_VERSION,
                "unsupported KVarN state version");
        kvarn_state_require(kvarn_state_read_uint<uint32_t>(io) == kv_size, "KVarN state kv_size mismatch");
        kvarn_state_require(kvarn_state_read_uint<uint32_t>(io) == n_seq_max, "KVarN state n_seq_max mismatch");
        kvarn_state_require(kvarn_state_read_uint<uint32_t>(io) == n_pad, "KVarN state n_pad mismatch");
        kvarn_state_require(kvarn_state_read_uint<uint32_t>(io) == params.group_size, "KVarN state group_size mismatch");
        kvarn_state_require(kvarn_state_read_uint<uint32_t>(io) == params.key_bits, "KVarN state key_bits mismatch");
        kvarn_state_require(kvarn_state_read_uint<uint32_t>(io) == params.value_bits, "KVarN state value_bits mismatch");
        kvarn_state_require(kvarn_state_read_uint<uint32_t>(io) == params.sink_tokens, "KVarN state sink_tokens mismatch");
        kvarn_state_require(kvarn_state_read_uint<uint32_t>(io) == params.tail_tokens, "KVarN state tail_tokens mismatch");
        kvarn_state_require(kvarn_state_read_uint<uint32_t>(io) == params.sinkhorn_iters, "KVarN state sinkhorn_iters mismatch");
        kvarn_state_require(kvarn_state_read_uint<uint32_t>(io) == kvarn_float_bits(params.rtn_quantile),
                "KVarN state rtn_quantile mismatch");

        const uint32_t n_layers = kvarn_state_read_uint<uint32_t>(io);
        kvarn_state_require(n_layers == layer_tensors.size(), "KVarN state physical layer count mismatch");
        std::set<uint32_t> unique_layers;
        for (uint32_t i = 0; i < n_layers; ++i) {
            const layer_storage & layer = layer_tensors[i];
            const uint32_t il = kvarn_state_read_uint<uint32_t>(io);
            kvarn_state_require(unique_layers.insert(il).second, "KVarN state contains duplicate physical layer id");
            kvarn_state_require(il == layer.il, "KVarN state physical layer order mismatch");
            kvarn_state_require(kvarn_state_read_uint<uint32_t>(io) == layer.n_head_kv, "KVarN state layer head geometry mismatch");
            kvarn_state_require(kvarn_state_read_uint<uint32_t>(io) == layer.n_sink_tail, "KVarN state sink/tail geometry mismatch");
            kvarn_state_require(kvarn_state_read_uint<uint32_t>(io) == layer.n_records, "KVarN state record geometry mismatch");
            kvarn_state_require(kvarn_state_read_layout_matches(io, layer.layout_k), "KVarN state K layout mismatch");
            kvarn_state_require(kvarn_state_read_layout_matches(io, layer.layout_v), "KVarN state V layout mismatch");
        }

        const uint32_t staged_head = kvarn_state_read_uint<uint32_t>(io);
        kvarn_state_require(staged_head <= kv_size, "KVarN state head is out of range");
        const uint32_t n_cells = kvarn_state_read_uint<uint32_t>(io);
        kvarn_state_require(n_cells == kv_size, "KVarN state cell count mismatch");
        llama_kv_cells staged_cells;
        staged_cells.resize(kv_size);
        for (uint32_t i = 0; i < n_cells; ++i) {
            const uint8_t occupied = kvarn_state_read_uint<uint8_t>(io);
            kvarn_state_require(occupied <= 1, "KVarN state has invalid occupied marker");
            const int64_t pos = kvarn_state_read_i64(io);
            const int64_t ext_x = kvarn_state_read_i64(io);
            const int64_t ext_y = kvarn_state_read_i64(io);
            const uint32_t n_seq = kvarn_state_read_uint<uint32_t>(io);
            kvarn_state_require(n_seq <= n_seq_max && n_seq <= LLAMA_MAX_SEQ,
                    "KVarN state cell sequence count is out of range");
            kvarn_state_require((occupied && pos >= 0) || (!occupied && pos == -1),
                    "KVarN state cell position is inconsistent");
            kvarn_state_require(occupied || (ext_x == 0 && ext_y == 0 && n_seq == 0),
                    "KVarN empty cell contains metadata");
            std::set<uint32_t> unique_seq;
            for (uint32_t j = 0; j < n_seq; ++j) {
                const uint32_t s = kvarn_state_read_uint<uint32_t>(io);
                kvarn_state_require(s < n_seq_max && s < LLAMA_MAX_SEQ,
                        "KVarN state sequence id is out of range");
                kvarn_state_require(unique_seq.insert(s).second, "KVarN state cell contains duplicate sequence id");
            }
            if (occupied) {
                kvarn_state_require(pos <= std::numeric_limits<llama_pos>::max(), "KVarN state position overflows llama_pos");
                kvarn_state_require(ext_x >= std::numeric_limits<llama_pos>::min() && ext_x <= std::numeric_limits<llama_pos>::max() &&
                                    ext_y >= std::numeric_limits<llama_pos>::min() && ext_y <= std::numeric_limits<llama_pos>::max(),
                        "KVarN state cell extension overflows llama_pos");
                staged_cells.pos_set(i, llama_pos(pos));
                staged_cells.ext_set(i, { llama_pos(ext_x), llama_pos(ext_y) });
                for (uint32_t s : unique_seq) {
                    staged_cells.seq_add(i, llama_seq_id(s));
                }
            }
        }

        if (n_seq_max == 1) {
            std::vector<bool> seen_positions(kv_size, false);
            llama_pos max_pos = -1;
            for (uint32_t i = 0; i < n_cells; ++i) {
                if (staged_cells.is_empty(i)) {
                    continue;
                }

                kvarn_state_require(staged_cells.seq_count(i) == 1 && staged_cells.seq_has(i, 0),
                        "KVarN single-stream state cell must belong only to sequence 0");
                const llama_pos pos = staged_cells.pos_get(i);
                kvarn_state_require(pos >= 0 && uint64_t(pos) < uint64_t(kv_size),
                        "KVarN single-stream state position is out of range");
                kvarn_state_require(!seen_positions[size_t(pos)],
                        "KVarN single-stream state contains a duplicate position");
                seen_positions[size_t(pos)] = true;
                max_pos = std::max(max_pos, pos);
            }
            if (max_pos >= 0) {
                for (uint32_t pos = 0; pos <= uint32_t(max_pos); ++pos) {
                    kvarn_state_require(seen_positions[size_t(pos)],
                            "KVarN single-stream state positions are not dense from zero");
                }
            }
        }

        struct tensor_restore {
            ggml_tensor * tensor;
            size_t nbytes;
        };
        using invalidate_restored_body_fn = void (*)(ggml_backend_buffer_t owner, const void * key);
        struct body_invalidation {
            invalidate_restored_body_fn fn;
            ggml_backend_buffer_t owner;
            const void * key;
        };
        std::vector<tensor_restore> restores;
        restores.reserve(size_t(n_layers)*KVAR_N_STATE_TENSORS_PER_LAYER);
        uint64_t max_nbytes = 0;
        for (const layer_storage & storage : layer_tensors) {
            const auto tensors = kvarn_state_tensors(get_layer_view(storage.il));
            for (uint32_t it = 0; it < tensors.size(); ++it) {
                ggml_tensor * tensor = tensors[it];
                kvarn_state_require(kvarn_state_read_uint<uint32_t>(io) == it, "KVarN state tensor order mismatch");
                kvarn_state_require(kvarn_state_read_uint<uint32_t>(io) == uint32_t(tensor->type), "KVarN state tensor type mismatch");
                for (uint32_t d = 0; d < GGML_MAX_DIMS; ++d) {
                    kvarn_state_require(kvarn_state_read_uint<uint64_t>(io) == uint64_t(tensor->ne[d]),
                            "KVarN state tensor shape mismatch");
                }
                const uint64_t nbytes = kvarn_state_read_uint<uint64_t>(io);
                kvarn_state_require(nbytes == uint64_t(ggml_nbytes(tensor)), "KVarN state tensor byte size mismatch");
                kvarn_state_require(nbytes <= uint64_t(std::numeric_limits<size_t>::max()), "KVarN state tensor is too large");
                max_nbytes = std::max(max_nbytes, nbytes);
                restores.push_back({ tensor, size_t(nbytes) });
            }
        }

        // Resolve every device hook before reading the first payload byte so a
        // GPU backend lacking restore invalidation cannot leave a partial state.
        std::vector<body_invalidation> body_invalidations;
        body_invalidations.reserve(layer_tensors.size());
        for (const layer_storage & storage : layer_tensors) {
            ggml_tensor * root = storage.body_k;
            while (root->view_src != nullptr) {
                root = root->view_src;
            }
            kvarn_state_require(root->buffer != nullptr, "KVarN state body root has no owner buffer");
            ggml_backend_buffer_t owner = root->buffer;
            ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(owner);
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
            if (dev == nullptr || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                continue;
            }
            kvarn_state_require(root->data != nullptr, "KVarN GPU state body root is not allocated");
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            kvarn_state_require(reg != nullptr, "KVarN GPU backend registry is unavailable during restore");
            auto fn = reinterpret_cast<invalidate_restored_body_fn>(
                    ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_kvarn_invalidate_restored_body"));
            kvarn_state_require(fn != nullptr, "KVarN GPU backend lacks restored-body invalidation");
            body_invalidations.push_back({ fn, owner, root->data });
        }

        std::vector<uint8_t> buffer(static_cast<size_t>(max_nbytes), uint8_t{});
        for (const tensor_restore & restore : restores) {
            io.read(buffer.data(), restore.nbytes);
            ggml_backend_tensor_set(restore.tensor, buffer.data(), 0, restore.nbytes);
        }

        for (const body_invalidation & invalidation : body_invalidations) {
            invalidation.fn(invalidation.owner, invalidation.key);
        }

        v_cells[0].set(0, staged_cells);
        v_heads[0] = staged_head;
        for (auto & layer : runtime_cache) {
            for (auto & head : layer) {
                head.clear();
            }
        }
    } catch (...) {
        clear(true);
        throw;
    }
}

uint32_t llama_kv_cache_kvarn::get_size() const {
    return kv_size;
}

uint32_t llama_kv_cache_kvarn::get_n_layer() const {
    return layer_ids.size();
}

bool llama_kv_cache_kvarn::has_layer(int32_t il) const {
    return map_layer_ids.find(il) != map_layer_ids.end();
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
        result += ggml_nbytes(layer.attn_mixed_scratch);
    }

    return result;
}

size_t llama_kv_cache_kvarn::layer_storage_index(uint32_t il) const {
    const auto it = map_layer_ids.find(int32_t(il));
    if (it == map_layer_ids.end()) {
        throw std::invalid_argument("KVarN layer is not present in runtime storage");
    }

    return size_t(it->second);
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
        /*.layout_k    =*/ st.layout_k,
        /*.layout_v    =*/ st.layout_v,
        /*.sink_tail_k =*/ st.sink_tail_k,
        /*.sink_tail_v =*/ st.sink_tail_v,
        /*.body_k      =*/ st.body_k,
        /*.body_v      =*/ st.body_v,
        /*.scales_k    =*/ st.scales_k,
        /*.scales_v    =*/ st.scales_v,
        /*.pending_k   =*/ st.pending_k,
        /*.pending_v   =*/ st.pending_v,
        /*.attn_mixed_scratch =*/ st.attn_mixed_scratch,
    };
}

int64_t llama_kv_cache_kvarn::attn_mixed_scratch_floats_worst(int32_t il) const {
    const llama_kvarn_layer_view view = get_layer_view(il);
    return view.attn_mixed_scratch ? ggml_nelements(view.attn_mixed_scratch) : 0;
}

ggml_tensor * llama_kv_cache_kvarn::build_attn_mixed_scratch(
        ggml_context * ctx, int32_t il, int64_t n_floats) const {
    const llama_kvarn_layer_view view = get_layer_view(il);
    if (view.attn_mixed_scratch == nullptr) {
        throw std::runtime_error("KVarN mixed-attn scratch is not allocated for this layer");
    }
    if (ggml_nelements(view.attn_mixed_scratch) < n_floats) {
        throw std::runtime_error("KVarN mixed-attn scratch is smaller than the active attention window requires");
    }
    ggml_tensor * scratch = view.attn_mixed_scratch;
    if (ggml_nelements(scratch) > n_floats) {
        scratch = ggml_view_1d(ctx, scratch, n_floats, 0);
        ggml_format_name(scratch, "kvarn_attn_mixed_scratch_view_l%d", il);
    }
    return scratch;
}

size_t llama_kv_cache_kvarn::body_store_scratch_floats(int32_t il) const {
    const llama_kvarn_layer_view view = get_layer_view(il);
    if (view.head_dim_k != view.head_dim_v) {
        throw std::invalid_argument("KVarN body store scratch requires equal K and V head dimensions");
    }

    const size_t tile_floats = size_t(view.head_dim_k)*params.group_size;
    // Best-so-far scratch per tile: row scales + col scales + imbalance + global RMS.
    const size_t per_pipeline =
        tile_floats + 2*std::max<uint32_t>(view.head_dim_k, params.group_size) +
        view.head_dim_k + params.group_size + 2;
    const size_t pipeline_scratch = view.head_dim_k >= 256 ? 2*per_pipeline : per_pipeline;
    // Multi-head stores gather one K and one V tile per head before the fused store kernel.
    // Reserve those transpose tiles for every multi-head layer, including 128-dim Qwen paths.
    const bool needs_pending_head_tiles = view.n_head_kv > 1 || view.head_dim_k >= 512;
    size_t result = needs_pending_head_tiles ? 2*tile_floats + pipeline_scratch : pipeline_scratch;

    // Direct prefill batches at most eight contiguous records per graph op.
    // Production K2/K4/K8 with V2/V4/V8 batches record x head phases by default,
    // including exact log-std best-so-far state for both K and V.
    if ((view.layout_k.key_bits == 2 || view.layout_k.key_bits == 4 || view.layout_k.key_bits == 8) &&
            (view.layout_v.value_bits == 2 || view.layout_v.value_bits == 4 || view.layout_v.value_bits == 8)) {
        constexpr uint32_t direct_record_batch_max = 8;
        const size_t n_tiles = size_t(view.n_head_kv)*direct_record_batch_max;
        const size_t data_floats = n_tiles*tile_floats;
        const size_t best_floats =
            n_tiles*(size_t(view.head_dim_k) + params.group_size + 2);
        const size_t batched_phase_scratch = 2*data_floats + 2*best_floats;
        result = std::max(result, batched_phase_scratch);
    }
    return result;
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
            view.head_dim_k, params.group_size, view.layout_k.key_bits, params.sinkhorn_iters, params.rtn_quantile);
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
    ggml_tensor * store = ggml_kvarn_store_v_body(
            ctx, v_tile, body, scales, scratch,
            view.head_dim_v, params.group_size, view.layout_v.value_bits, params.sinkhorn_iters, params.rtn_quantile);
    ggml_kvarn_store_body_set_v_layout(store, int32_t(view.layout_v.v_layout));
    return store;
}

ggml_tensor * llama_kv_cache_kvarn::store_kv_body_record(
        ggml_context * ctx,
        ggml_tensor * k_tile,
        ggml_tensor * v_tile,
        ggml_tensor * scratch,
        int32_t il,
        uint32_t ih,
        uint32_t record) const {
    const llama_kvarn_layer_view view = get_layer_view(il);
    if (view.head_dim_k != view.head_dim_v) {
        throw std::runtime_error("KVarN fused K/V body store requires equal K and V head dimensions");
    }
    ggml_tensor * k_body   = view_k_body_record(ctx, il, ih, record);
    ggml_tensor * v_body   = view_v_body_record(ctx, il, ih, record);
    ggml_tensor * k_scales = view_k_scales_record(ctx, il, ih, record);
    ggml_tensor * v_scales = view_v_scales_record(ctx, il, ih, record);
    ggml_tensor * store = ggml_kvarn_store_kv_body(
            ctx, k_tile, v_tile, k_body, v_body, k_scales, v_scales, scratch,
            view.head_dim_k, params.group_size, view.layout_k.key_bits, view.layout_v.value_bits, params.sinkhorn_iters, params.rtn_quantile);
    ggml_kvarn_store_kv_body_set_v_layout(store, int32_t(view.layout_v.v_layout));
    return store;
}

ggml_tensor * llama_kv_cache_kvarn::store_kv_body_all_heads(
        ggml_context * ctx,
        ggml_tensor * k_tile,
        ggml_tensor * v_tile,
        ggml_tensor * scratch,
        int32_t il,
        uint32_t record) const {
    const llama_kvarn_layer_view view = get_layer_view(il);
    if (record >= view.n_records) {
        throw std::out_of_range("KVarN body record index is out of range");
    }
    if (view.head_dim_k != view.head_dim_v) {
        throw std::runtime_error("KVarN fused K/V body store requires equal K and V head dimensions");
    }
    if (view.n_head_kv <= 1) {
        return store_kv_body_record(ctx, k_tile, v_tile, scratch, il, 0, record);
    }

    ggml_tensor * k_body   = view_k_body_record_heads(ctx, il, record);
    ggml_tensor * v_body   = view_v_body_record_heads(ctx, il, record);
    ggml_tensor * k_scales = view_k_scales_record_heads(ctx, il, record);
    ggml_tensor * v_scales = view_v_scales_record_heads(ctx, il, record);
    ggml_tensor * store = ggml_kvarn_store_kv_body_pending_heads(
            ctx, k_tile, v_tile, k_body, v_body, k_scales, v_scales, scratch,
            int32_t(view.n_head_kv), int32_t(record),
            int32_t(view.head_dim_k), int32_t(params.group_size),
            int32_t(view.layout_k.key_bits), int32_t(view.layout_v.value_bits),
            int32_t(params.sinkhorn_iters), params.rtn_quantile);
    ggml_kvarn_store_kv_body_set_v_layout(store, int32_t(view.layout_v.v_layout));
    return store;
}

ggml_tensor * llama_kv_cache_kvarn::store_kv_body_records_all_heads(
        ggml_context * ctx,
        ggml_tensor * k_tiles,
        ggml_tensor * v_tiles,
        ggml_tensor * scratch,
        int32_t il,
        uint32_t record0,
        uint32_t n_records) const {
    const llama_kvarn_layer_view view = get_layer_view(il);
    if (n_records == 0) {
        throw std::invalid_argument("KVarN direct record batch requires at least one record");
    }
    if (n_records > 16) {
        throw std::invalid_argument("KVarN direct record batch supports at most sixteen records per op");
    }
    if (record0 + n_records > view.n_records) {
        throw std::out_of_range("KVarN direct body record span is out of range");
    }
    if (view.head_dim_k != view.head_dim_v) {
        throw std::runtime_error("KVarN fused K/V body store requires equal K and V head dimensions");
    }

    ggml_tensor * k_body   = view_k_body_record_span_heads(ctx, il, record0, n_records);
    ggml_tensor * v_body   = view_v_body_record_span_heads(ctx, il, record0, n_records);
    ggml_tensor * k_scales = view_k_scales_record_span_heads(ctx, il, record0, n_records);
    ggml_tensor * v_scales = view_v_scales_record_span_heads(ctx, il, record0, n_records);
    ggml_tensor * store = ggml_kvarn_store_kv_body_direct_records(
            ctx, k_tiles, v_tiles, k_body, v_body, k_scales, v_scales, scratch,
            int32_t(view.n_head_kv), int32_t(record0), int32_t(n_records),
            int32_t(view.head_dim_k), int32_t(params.group_size),
            int32_t(view.layout_k.key_bits), int32_t(view.layout_v.value_bits),
            int32_t(params.sinkhorn_iters), params.rtn_quantile);
    ggml_kvarn_store_kv_body_set_v_layout(store, int32_t(view.layout_v.v_layout));
    return store;
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
    ggml_tensor * store = store_k_body_record(ctx, k_tile, scratch, il, ih, record);
    ggml_kvarn_store_body_set_src_pending(store);
    return store;
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
    ggml_tensor * store = store_v_body_record(ctx, v_tile, scratch, il, ih, record);
    ggml_kvarn_store_body_set_src_pending(store);
    return store;
}

ggml_tensor * llama_kv_cache_kvarn::view_k_body_record_heads(
        ggml_context * ctx, int32_t il, uint32_t record) const {
    const llama_kvarn_layer_view view = get_layer_view(il);
    if (record >= view.n_records) {
        throw std::out_of_range("KVarN body record index is out of range");
    }

    const size_t offset = size_t(record)*view.body_k->nb[1];
    ggml_tensor * result = ggml_view_2d(
            ctx, view.body_k, view.layout_k.k_body_bytes, view.n_head_kv,
            view.body_k->nb[2], offset);
    ggml_format_name(result, "kvarn_k_body_l%d_r%u_heads", il, record);
    return result;
}

ggml_tensor * llama_kv_cache_kvarn::view_k_body_record_span_heads(
        ggml_context * ctx, int32_t il, uint32_t record0, uint32_t n_records) const {
    const llama_kvarn_layer_view view = get_layer_view(il);
    if (n_records == 0 || record0 + n_records > view.n_records) {
        throw std::out_of_range("KVarN body record span is out of range");
    }

    const size_t offset = size_t(record0)*view.body_k->nb[1];
    ggml_tensor * result = ggml_view_3d(
            ctx, view.body_k, view.layout_k.k_body_bytes, n_records, view.n_head_kv,
            view.body_k->nb[1], view.body_k->nb[2], offset);
    ggml_format_name(result, "kvarn_k_body_l%d_r%u_n%u_heads", il, record0, n_records);
    return result;
}

ggml_tensor * llama_kv_cache_kvarn::view_v_body_record_heads(
        ggml_context * ctx, int32_t il, uint32_t record) const {
    const llama_kvarn_layer_view view = get_layer_view(il);
    if (record >= view.n_records) {
        throw std::out_of_range("KVarN body record index is out of range");
    }

    const size_t offset = size_t(record)*view.body_v->nb[1];
    ggml_tensor * result = ggml_view_2d(
            ctx, view.body_v, view.layout_v.v_body_bytes, view.n_head_kv,
            view.body_v->nb[2], offset);
    ggml_format_name(result, "kvarn_v_body_l%d_r%u_heads", il, record);
    return result;
}

ggml_tensor * llama_kv_cache_kvarn::view_v_body_record_span_heads(
        ggml_context * ctx, int32_t il, uint32_t record0, uint32_t n_records) const {
    const llama_kvarn_layer_view view = get_layer_view(il);
    if (n_records == 0 || record0 + n_records > view.n_records) {
        throw std::out_of_range("KVarN body record span is out of range");
    }

    const size_t offset = size_t(record0)*view.body_v->nb[1];
    ggml_tensor * result = ggml_view_3d(
            ctx, view.body_v, view.layout_v.v_body_bytes, n_records, view.n_head_kv,
            view.body_v->nb[1], view.body_v->nb[2], offset);
    ggml_format_name(result, "kvarn_v_body_l%d_r%u_n%u_heads", il, record0, n_records);
    return result;
}

ggml_tensor * llama_kv_cache_kvarn::view_k_scales_record_heads(
        ggml_context * ctx, int32_t il, uint32_t record) const {
    const llama_kvarn_layer_view view = get_layer_view(il);
    if (record >= view.n_records) {
        throw std::out_of_range("KVarN body record index is out of range");
    }

    const size_t offset = size_t(record)*view.scales_k->nb[1];
    ggml_tensor * result = ggml_view_2d(
            ctx, view.scales_k, view.layout_k.k_scale_floats, view.n_head_kv,
            view.scales_k->nb[2], offset);
    ggml_format_name(result, "kvarn_k_scales_l%d_r%u_heads", il, record);
    return result;
}

ggml_tensor * llama_kv_cache_kvarn::view_k_scales_record_span_heads(
        ggml_context * ctx, int32_t il, uint32_t record0, uint32_t n_records) const {
    const llama_kvarn_layer_view view = get_layer_view(il);
    if (n_records == 0 || record0 + n_records > view.n_records) {
        throw std::out_of_range("KVarN scale record span is out of range");
    }

    const size_t offset = size_t(record0)*view.scales_k->nb[1];
    ggml_tensor * result = ggml_view_3d(
            ctx, view.scales_k, view.layout_k.k_scale_floats, n_records, view.n_head_kv,
            view.scales_k->nb[1], view.scales_k->nb[2], offset);
    ggml_format_name(result, "kvarn_k_scales_l%d_r%u_n%u_heads", il, record0, n_records);
    return result;
}

ggml_tensor * llama_kv_cache_kvarn::view_v_scales_record_heads(
        ggml_context * ctx, int32_t il, uint32_t record) const {
    const llama_kvarn_layer_view view = get_layer_view(il);
    if (record >= view.n_records) {
        throw std::out_of_range("KVarN body record index is out of range");
    }

    const size_t offset = size_t(record)*view.scales_v->nb[1];
    ggml_tensor * result = ggml_view_2d(
            ctx, view.scales_v, view.layout_v.v_scale_floats, view.n_head_kv,
            view.scales_v->nb[2], offset);
    ggml_format_name(result, "kvarn_v_scales_l%d_r%u_heads", il, record);
    return result;
}

ggml_tensor * llama_kv_cache_kvarn::view_v_scales_record_span_heads(
        ggml_context * ctx, int32_t il, uint32_t record0, uint32_t n_records) const {
    const llama_kvarn_layer_view view = get_layer_view(il);
    if (n_records == 0 || record0 + n_records > view.n_records) {
        throw std::out_of_range("KVarN scale record span is out of range");
    }

    const size_t offset = size_t(record0)*view.scales_v->nb[1];
    ggml_tensor * result = ggml_view_3d(
            ctx, view.scales_v, view.layout_v.v_scale_floats, n_records, view.n_head_kv,
            view.scales_v->nb[1], view.scales_v->nb[2], offset);
    ggml_format_name(result, "kvarn_v_scales_l%d_r%u_n%u_heads", il, record0, n_records);
    return result;
}

ggml_tensor * llama_kv_cache_kvarn::store_kv_body_records_from_pending(
        ggml_context * ctx,
        ggml_tensor * scratch,
        int32_t il,
        const std::vector<uint32_t> & records,
        ggml_tensor * pending_k,
        ggml_tensor * pending_v) const {
    const size_t li = layer_storage_index(il);
    const llama_kvarn_layer_view view = get_layer_view(il);
    if (records.empty()) {
        throw std::invalid_argument("KVarN record batch seal requires at least one record");
    }
    if (records.size() > 1) {
        throw std::invalid_argument(
                "KVarN pending record batch seal is disabled: pending storage holds only one body record; "
                "use direct record store or seal records one at a time");
    }
    for (const uint32_t record : records) {
        if (record >= view.n_records) {
            throw std::out_of_range("KVarN body record index is out of range");
        }
    }
    if (view.head_dim_k != view.head_dim_v) {
        throw std::runtime_error("KVarN fused pending K/V body store requires equal K and V head dimensions");
    }

    std::vector<int32_t> record_ids(records.begin(), records.end());
    ggml_tensor * pending_k_src = pending_k != nullptr ? pending_k : layer_tensors[li].pending_k;
    ggml_tensor * pending_v_src = pending_v != nullptr ? pending_v : layer_tensors[li].pending_v;
    ggml_tensor * store = ggml_kvarn_store_kv_body_pending_records(
            ctx, pending_k_src, pending_v_src,
            layer_tensors[li].body_k, layer_tensors[li].body_v,
            layer_tensors[li].scales_k, layer_tensors[li].scales_v,
            scratch,
            record_ids.data(), int32_t(record_ids.size()),
            int32_t(view.head_dim_k), int32_t(params.group_size),
            int32_t(view.layout_k.key_bits), int32_t(view.layout_v.value_bits),
            int32_t(params.sinkhorn_iters), params.rtn_quantile);
    ggml_kvarn_store_kv_body_set_v_layout(store, int32_t(view.layout_v.v_layout));
    ggml_kvarn_store_kv_body_set_src_pending(store);
    return store;
}

ggml_tensor * llama_kv_cache_kvarn::store_kv_body_all_heads_from_pending(
        ggml_context * ctx,
        ggml_tensor * scratch,
        int32_t il,
        uint32_t record,
        ggml_tensor * pending_k,
        ggml_tensor * pending_v) const {
    const size_t li = layer_storage_index(il);
    const llama_kvarn_layer_view view = get_layer_view(il);
    if (record >= view.n_records) {
        throw std::out_of_range("KVarN body record index is out of range");
    }
    if (view.head_dim_k != view.head_dim_v) {
        throw std::runtime_error("KVarN fused pending K/V body store requires equal K and V head dimensions");
    }
    if (view.n_head_kv <= 1) {
        return store_kv_body_record_from_pending(ctx, scratch, il, 0, record, pending_k, pending_v);
    }

    ggml_tensor * pending_k_src = pending_k != nullptr ? pending_k : layer_tensors[li].pending_k;
    ggml_tensor * pending_v_src = pending_v != nullptr ? pending_v : layer_tensors[li].pending_v;
    ggml_tensor * k_body   = view_k_body_record_heads(ctx, il, record);
    ggml_tensor * v_body   = view_v_body_record_heads(ctx, il, record);
    ggml_tensor * k_scales = view_k_scales_record_heads(ctx, il, record);
    ggml_tensor * v_scales = view_v_scales_record_heads(ctx, il, record);
    ggml_tensor * store = ggml_kvarn_store_kv_body_pending_heads(
            ctx, pending_k_src, pending_v_src,
            k_body, v_body, k_scales, v_scales, scratch,
            int32_t(view.n_head_kv), int32_t(record),
            int32_t(view.head_dim_k), int32_t(params.group_size),
            int32_t(view.layout_k.key_bits), int32_t(view.layout_v.value_bits),
            int32_t(params.sinkhorn_iters), params.rtn_quantile);
    ggml_kvarn_store_kv_body_set_v_layout(store, int32_t(view.layout_v.v_layout));
    ggml_kvarn_store_kv_body_set_src_pending(store);
    return store;
}

ggml_tensor * llama_kv_cache_kvarn::store_kv_body_record_from_pending(
        ggml_context * ctx,
        ggml_tensor * scratch,
        int32_t il,
        uint32_t ih,
        uint32_t record,
        ggml_tensor * pending_k,
        ggml_tensor * pending_v) const {
    const size_t li = layer_storage_index(il);
    const llama_kvarn_layer_view view = get_layer_view(il);
    assert_kvarn_record_view_bounds(view, ih, record);
    if (view.head_dim_k != view.head_dim_v) {
        throw std::runtime_error("KVarN fused pending K/V body store requires equal K and V head dimensions");
    }
    ggml_tensor * pending_k_src = pending_k != nullptr ? pending_k : layer_tensors[li].pending_k;
    ggml_tensor * pending_v_src = pending_v != nullptr ? pending_v : layer_tensors[li].pending_v;
    GGML_ASSERT(view.head_dim_k == uint32_t(pending_k_src->ne[0]));
    GGML_ASSERT(view.head_dim_v == uint32_t(pending_v_src->ne[0]));

    const size_t k_offset = size_t(ih)*pending_k_src->nb[1];
    ggml_tensor * k_pending = ggml_view_2d(
            ctx, pending_k_src,
            view.head_dim_k, params.group_size,
            pending_k_src->nb[2], k_offset);
    ggml_tensor * k_tile = ggml_cont(ctx, ggml_transpose(ctx, k_pending));

    const size_t v_offset = size_t(ih)*pending_v_src->nb[1];
    ggml_tensor * v_pending = ggml_view_2d(
            ctx, pending_v_src,
            view.head_dim_v, params.group_size,
            pending_v_src->nb[2], v_offset);
    ggml_tensor * v_tile = ggml_cont(ctx, v_pending);

    ggml_tensor * store = store_kv_body_record(ctx, k_tile, v_tile, scratch, il, ih, record);
    ggml_kvarn_store_kv_body_set_src_pending(store);
    return store;
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

// Must match kvarn_graph_effective_params() in llama-graph.cpp: with
// kv_size < sink+tail the graph-side window/seal math clamps the tail, so
// the eviction plans and slot indices built here have to use the same
// geometry or the two sides disagree about when tokens leave the ring.
static llama_kvarn_params kvarn_effective_params(llama_kvarn_params params, uint32_t kv_size) {
    if (kv_size != 0 && uint64_t(params.sink_tokens) + uint64_t(params.tail_tokens) > kv_size) {
        if (params.sink_tokens >= kv_size) {
            params.tail_tokens = 0;
        } else {
            params.tail_tokens = kv_size - params.sink_tokens;
        }
    }
    return params;
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

    const llama_kvarn_params p = kvarn_effective_params(params, kv_size);
    int64_t * data = (int64_t *) dst->data;
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_pos pos = ubatch->pos ? ubatch->pos[i] : llama_pos(i);
        if (pos < 0) {
            data[i] = 0;
        } else if (uint32_t(pos) < p.sink_tokens) {
            data[i] = int64_t(pos);
        } else {
            data[i] = kvarn_tail_slot(p, uint32_t(pos));
        }
    }
}

ggml_tensor * llama_kv_cache_kvarn::build_input_body_plan(ggml_context * ctx, const llama_ubatch & ubatch) const {
    ggml_tensor * plan = ggml_new_tensor_2d(ctx, GGML_TYPE_I64, 3,
            kvarn_count_tail_evictions(kvarn_effective_params(params, kv_size), ubatch));
    ggml_set_input(plan);
    ggml_set_name(plan, "kvarn_body_plan");
    return plan;
}

void llama_kv_cache_kvarn::set_input_body_plan(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(dst->type == GGML_TYPE_I64);
    GGML_ASSERT(dst->ne[0] == 3);
    const llama_kvarn_params p = kvarn_effective_params(params, kv_size);
    GGML_ASSERT(dst->ne[1] == kvarn_count_tail_evictions(p, *ubatch));

    int64_t * data = (int64_t *) dst->data;
    uint32_t j = 0;
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_pos pos = ubatch->pos ? ubatch->pos[i] : llama_pos(i);

        if (pos < 0 || uint32_t(pos) < p.sink_tokens + p.tail_tokens) {
            continue;
        }

        const uint32_t evicted_pos = uint32_t(pos) - p.tail_tokens;
        const uint32_t body_pos = evicted_pos - p.sink_tokens;
        const uint32_t record   = body_pos/p.group_size;
        const uint32_t offset   = body_pos%p.group_size;
        const size_t off = size_t(j++)*3;

        data[off + 0] = int64_t(record);
        data[off + 1] = int64_t(offset);
        data[off + 2] = offset + 1 == p.group_size ? int64_t(record) : -1;
    }
}

ggml_tensor * llama_kv_cache_kvarn::build_input_body_offsets(ggml_context * ctx, const llama_ubatch & ubatch) const {
    ggml_tensor * offsets = ggml_new_tensor_1d(ctx, GGML_TYPE_I64,
            kvarn_count_tail_evictions(kvarn_effective_params(params, kv_size), ubatch));
    ggml_set_input(offsets);
    ggml_set_name(offsets, "kvarn_body_offsets");
    return offsets;
}

void llama_kv_cache_kvarn::set_input_body_offsets(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(dst->type == GGML_TYPE_I64);
    const llama_kvarn_params p = kvarn_effective_params(params, kv_size);
    GGML_ASSERT(dst->ne[0] == kvarn_count_tail_evictions(p, *ubatch));

    int64_t * data = (int64_t *) dst->data;
    uint32_t j = 0;
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_pos pos = ubatch->pos ? ubatch->pos[i] : llama_pos(i);
        if (pos < 0 || uint32_t(pos) < p.sink_tokens + p.tail_tokens) {
            continue;
        }

        const uint32_t evicted_pos = uint32_t(pos) - p.tail_tokens;
        data[j++] = int64_t((evicted_pos - p.sink_tokens)%p.group_size);
    }
}

ggml_tensor * llama_kv_cache_kvarn::build_input_tail_evict_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    ggml_tensor * idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I32,
            kvarn_count_tail_evictions(kvarn_effective_params(params, kv_size), ubatch));
    ggml_set_input(idxs);
    ggml_set_name(idxs, "kvarn_tail_evict_idxs");
    return idxs;
}

void llama_kv_cache_kvarn::set_input_tail_evict_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    const llama_kvarn_params p = kvarn_effective_params(params, kv_size);
    GGML_ASSERT(dst->ne[0] == kvarn_count_tail_evictions(p, *ubatch));

    int32_t * data = (int32_t *) dst->data;
    uint32_t j = 0;
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_pos pos = ubatch->pos ? ubatch->pos[i] : llama_pos(i);
        if (pos < 0 || uint32_t(pos) < p.sink_tokens + p.tail_tokens) {
            continue;
        }

        const uint32_t evicted_pos = uint32_t(pos) - p.tail_tokens;
        data[j++] = int32_t(kvarn_tail_slot(p, evicted_pos));
    }
}

void llama_kv_cache_kvarn::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->ne[0] <= kv_size);
    GGML_ASSERT(dst->ne[1] == ubatch->n_tokens);

    const int64_t n_kv = dst->ne[0];
    const int64_t n_tokens = ubatch->n_tokens;

    llama_pos last_pos = llama_pos(n_tokens > 0 ? n_tokens - 1 : 0);
    if (ubatch->pos != nullptr && n_tokens > 0) {
        last_pos = ubatch->pos[n_tokens - 1];
    }

    const llama_pos q_base = causal_attn && n_tokens > 0 && last_pos >= llama_pos(n_tokens - 1) ?
        last_pos - llama_pos(n_tokens - 1) :
        llama_pos(0);
    bool use_contiguous_query_positions = causal_attn;
    if (causal_attn && ubatch->pos != nullptr) {
        use_contiguous_query_positions = last_pos >= llama_pos(n_tokens - 1);
        for (int64_t q = 0; q < n_tokens && use_contiguous_query_positions; ++q) {
            use_contiguous_query_positions = ubatch->pos[q] == q_base + llama_pos(q);
        }
    }

    const auto write = [&](int64_t t, int64_t q, float v) {
        char * p = (char *) dst->data + size_t(q)*dst->nb[1] + size_t(t)*dst->nb[0];
        if (dst->type == GGML_TYPE_F16) {
            *(ggml_fp16_t *) p = ggml_fp32_to_fp16(v);
        } else {
            *(float *) p = v;
        }
    };

    const llama_kvarn_params effective_params = kvarn_effective_params(params, kv_size);

    const int64_t invalid_key_pos = std::numeric_limits<int64_t>::max();
    std::vector<int64_t> key_positions(size_t(n_kv), invalid_key_pos);
    if (last_pos >= 0) {
        const uint32_t n_seen = uint32_t(last_pos) + 1;
        const uint32_t n_sink = std::min<uint32_t>(n_seen, effective_params.sink_tokens);
        const uint32_t n_after_sink = n_seen - n_sink;
        const uint32_t n_tail = std::min<uint32_t>(n_after_sink, effective_params.tail_tokens);
        const uint32_t n_body_pending = n_after_sink - n_tail;
        const uint32_t n_records = n_body_pending/effective_params.group_size;
        const uint32_t n_pending = n_body_pending%effective_params.group_size;
        const uint32_t n_body = n_records*effective_params.group_size;

        uint32_t t = 0;
        for (uint32_t i = 0; i < n_sink && t < uint32_t(n_kv); ++i, ++t) {
            key_positions[t] = int64_t(i);
        }
        for (uint32_t i = 0; i < n_body && t < uint32_t(n_kv); ++i, ++t) {
            key_positions[t] = int64_t(effective_params.sink_tokens) + i;
        }
        for (uint32_t i = 0; i < n_pending && t < uint32_t(n_kv); ++i, ++t) {
            key_positions[t] = int64_t(effective_params.sink_tokens) + n_body + i;
        }
        for (uint32_t i = 0; i < n_tail && t < uint32_t(n_kv); ++i, ++t) {
            key_positions[t] = int64_t(effective_params.sink_tokens) + n_body_pending + i;
        }
    } else if (!causal_attn) {
        for (int64_t t = 0; t < n_kv; ++t) {
            key_positions[size_t(t)] = t;
        }
    }

    for (int64_t q = 0; q < dst->ne[1]; ++q) {
        const llama_pos pos = causal_attn ?
            (use_contiguous_query_positions ? q_base + llama_pos(q) : ubatch->pos[q]) :
            (ubatch->pos ? ubatch->pos[q] : llama_pos(q));
        for (int64_t t = 0; t < n_kv; ++t) {
            const bool visible = !causal_attn || (key_positions[size_t(t)] != invalid_key_pos &&
                pos >= 0 && key_positions[size_t(t)] <= int64_t(pos));
            write(t, q, visible ? 0.0f : -INFINITY);
        }
    }

    if (kvarn_env_flag_enabled("LLAMA_KVARN_MASK_TRACE")) {
        static uint64_t trace_count = 0;
        const char * limit_env = std::getenv("LLAMA_KVARN_MASK_TRACE_LIMIT");
        uint64_t limit = 64;
        if (limit_env != nullptr && limit_env[0] != '\0') {
            char * end = nullptr;
            errno = 0;
            const unsigned long long parsed = std::strtoull(limit_env, &end, 10);
            if (end != limit_env && end != nullptr && *end == '\0' && errno != ERANGE) {
                limit = uint64_t(parsed);
            }
        }

        const uint64_t trace_index = trace_count++;
        if (trace_index < limit) {
            const auto read = [&](int64_t t, int64_t q) -> float {
                const char * p = (const char *) dst->data + size_t(q)*dst->nb[1] + size_t(t)*dst->nb[0];
                return dst->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(*(const ggml_fp16_t *) p) : *(const float *) p;
            };
            const auto count_visible = [&](int64_t q) -> int64_t {
                int64_t n = 0;
                for (int64_t t = 0; t < n_kv; ++t) {
                    if (read(t, q) > -1.0e20f) {
                        ++n;
                    }
                }
                return n;
            };
            const int64_t q0 = 0;
            const int64_t qm = dst->ne[1] > 0 ? dst->ne[1]/2 : 0;
            const int64_t ql = dst->ne[1] > 0 ? dst->ne[1] - 1 : 0;
            std::fprintf(stderr,
                    "KVarN mask trace: call=%llu n_kv=%lld n_q=%lld causal=%d last_pos=%d q_base=%d"
                    " visible[q0=%lld]=%lld visible[qm=%lld]=%lld visible[ql=%lld]=%lld\n",
                    (unsigned long long) trace_index,
                    (long long) n_kv, (long long) dst->ne[1], causal_attn ? 1 : 0,
                    int(last_pos), int(q_base),
                    (long long) q0, (long long) count_visible(q0),
                    (long long) qm, (long long) count_visible(qm),
                    (long long) ql, (long long) count_visible(ql));
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
    if (n_seq_max == 1) {
        int64_t expected_pos = int64_t(seq_pos_max(0)) + 1;
        const char * invalid_reason = nullptr;

        for (const llama_ubatch & ubatch : ubatches) {
            if (ubatch.pos == nullptr || ubatch.n_pos < 1) {
                invalid_reason = "position data is missing";
                break;
            }
            if (ubatch.n_seq_id == nullptr || ubatch.seq_id == nullptr) {
                invalid_reason = "sequence metadata is missing";
                break;
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                if (ubatch.n_seq_id[i] != 1 || ubatch.seq_id[i] == nullptr || ubatch.seq_id[i][0] != 0) {
                    invalid_reason = "each token must belong only to sequence 0";
                    break;
                }
                if (int64_t(ubatch.pos[i]) != expected_pos) {
                    invalid_reason = "positions must append contiguously";
                    break;
                }
                ++expected_pos;
            }
            if (invalid_reason != nullptr) {
                break;
            }
        }

        if (invalid_reason != nullptr) {
            std::fprintf(stderr, "%s: invalid single-stream KVarN ubatch: %s\n", __func__, invalid_reason);
            return {};
        }
    }

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
            seq_rm_cells(s, cells.seq_pos_min(s), seq_pos_max_rm[s] + 1);
        }
    }

    v_heads[0] = sinfo.idxs.back() + 1;
}
