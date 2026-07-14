#include "llama-kv-cache-kvarn.h"
#include "llama-kv-cache-kvarn-iswa.h"
#include "llama-hparams.h"
#include "llama-io.h"
#include "llama.h"
#include "ggml-backend.h"

#include <cmath>
#include <array>
#include <random>
#include <utility>
#include <exception>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * msg) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
}

static bool phase_trace_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("LLAMA_KVARN_TEST_PHASE_TRACE");
        return env != nullptr && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static void trace_phase(const char * name, const char * state) {
    if (!phase_trace_enabled()) {
        return;
    }
    std::fprintf(stderr, "KVarN test phase: %s %s\n", name, state);
    std::fflush(stderr);
}

static void set_test_env(const char * name, const char * value) {
#if defined(_WIN32)
    if (value != nullptr) {
        _putenv_s(name, value);
    } else {
        _putenv_s(name, "");
    }
#else
    if (value != nullptr) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

static void run_phase(const char * name, void (*fn)()) {
    trace_phase(name, "start");
    try {
        fn();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s threw std::exception: %s\n", name, e.what());
        std::exit(1);
    } catch (...) {
        std::fprintf(stderr, "FAIL: %s threw unknown exception\n", name);
        std::exit(1);
    }
    trace_phase(name, "pass");
}

static void test_layout() {
    llama_kvarn_params params = llama_kvarn_default_params();
    llama_kvarn_layout layout = llama_kvarn_make_layout(params, 128);

    require(layout.k_body_bytes == 8192, "K body bytes");
    require(layout.v_body_bytes == 4096, "V body bytes");
    require(layout.k_scale_floats == 384, "K scale float count");
    require(layout.v_scale_floats == 384, "V scale float count");
    require(layout.total_record_bytes == 15360, "total record bytes");

    llama_kvarn_layout layout256 = llama_kvarn_make_layout(params, 256);
    require(layout256.k_body_bytes == 16384, "256-dim K body bytes");
    require(layout256.v_body_bytes == 8192, "256-dim V body bytes");
    require(layout256.k_scale_floats == 640, "256-dim K scale float count");
    require(layout256.v_scale_floats == 512, "256-dim V scale float count");
    require(layout256.total_record_bytes == 29184, "256-dim total record bytes");

    llama_kvarn_layout layout512 = llama_kvarn_make_layout(params, 512);
    require(layout512.k_body_bytes == 32768, "512-dim K body bytes");
    require(layout512.v_body_bytes == 16384, "512-dim V body bytes");
    require(layout512.k_scale_floats == 1152, "512-dim K scale float count");
    require(layout512.v_scale_floats == 768, "512-dim V scale float count");
    require(layout512.total_record_bytes == 56832, "512-dim total record bytes");
}

class kvarn_test_writer final : public llama_io_write_i {
public:
    void write(const void * src, size_t size) override {
        const uint8_t * bytes = static_cast<const uint8_t *>(src);
        data.insert(data.end(), bytes, bytes + size);
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        const size_t begin = data.size();
        data.resize(begin + size);
        ggml_backend_tensor_get(tensor, data.data() + begin, offset, size);
    }

    size_t n_bytes() override { return data.size(); }

    std::vector<uint8_t> data;
};

class kvarn_test_sizer final : public llama_io_write_i {
public:
    void write(const void *, size_t size) override { count += size; }
    void write_tensor(ggml_tensor *, size_t, size_t size) override { count += size; }
    size_t n_bytes() override { return count; }

private:
    size_t count = 0;
};

class kvarn_test_reader final : public llama_io_read_i {
public:
    explicit kvarn_test_reader(const std::vector<uint8_t> & data) : data(data) {}

    void read(void * dst, size_t size) override {
        if (offset > data.size() || size > data.size() - offset) {
            throw std::runtime_error("truncated KVarN test state");
        }
        std::memcpy(dst, data.data() + offset, size);
        offset += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t tensor_offset, size_t size) override {
        std::vector<uint8_t> bytes(size);
        read(bytes.data(), size);
        ggml_backend_tensor_set(tensor, bytes.data(), tensor_offset, size);
    }

    size_t n_bytes() override { return offset; }

private:
    const std::vector<uint8_t> & data;
    size_t offset = 0;
};

static void test_iswa_full_normal_policy() {
    using policy = llama_kvarn_iswa_full_normal_policy;

    require(llama_kvarn_iswa_choose_full_normal_policy(false, false) == policy::none,
            "all-KVarN route does not allocate a normal full-KV cache");
    require(llama_kvarn_iswa_choose_full_normal_policy(false, true) == policy::route_fallback,
            "mixed route allocates only its normal fallback layers");
    require(llama_kvarn_iswa_choose_full_normal_policy(true, false) == policy::diagnostic_all,
            "diagnostic route allocates all normal full-KV layers");
    require(llama_kvarn_iswa_choose_full_normal_policy(true, true) == policy::diagnostic_all,
            "diagnostic full-normal route takes precedence over mixed fallback");
}

static void test_pack_roundtrip() {
    for (uint32_t bits : { 2u, 4u }) {
        std::vector<uint8_t> src(257);
        const uint32_t qmax = (1u << bits) - 1u;
        for (size_t i = 0; i < src.size(); ++i) {
            src[i] = uint8_t((i*7 + 3) & qmax);
        }

        std::vector<uint8_t> packed;
        std::vector<uint8_t> unpacked;
        llama_kvarn_pack_bits(src, bits, packed);
        llama_kvarn_unpack_bits(packed, bits, src.size(), unpacked);

        require(src == unpacked, "bit pack roundtrip");
    }
}

static void test_hadamard_inverse() {
    const uint32_t rows = 128;
    const uint32_t cols = 5;
    std::vector<float> src(size_t(rows)*cols);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = std::sin(float(i)*0.013f) + std::cos(float(i)*0.021f);
    }

    std::vector<float> tmp;
    std::vector<float> dst;
    llama_kvarn_hadamard_channels(src, tmp, rows, cols, true);
    llama_kvarn_hadamard_channels(tmp, dst, rows, cols, true);

    float max_err = 0.0f;
    for (size_t i = 0; i < src.size(); ++i) {
        max_err = std::max(max_err, std::fabs(src[i] - dst[i]));
    }
    require(max_err < 1.0e-5f, "Hadamard inverse");
}

static void test_reference_store_dequant() {
    llama_kvarn_params params = llama_kvarn_default_params();
    params.sinkhorn_iters = 4;

    for (const uint32_t head_dim : { 128u, 256u, 512u }) {
    const uint32_t group = params.group_size;
    const llama_kvarn_layout layout = llama_kvarn_make_layout(params, head_dim);
    std::vector<float> k_tile(size_t(head_dim)*group);
    std::vector<float> v_tile(size_t(head_dim)*group);

    for (size_t i = 0; i < k_tile.size(); ++i) {
        k_tile[i] = 0.05f*std::sin(float(i)*0.017f) + 0.02f*std::cos(float(i)*0.003f);
        v_tile[i] = 0.04f*std::cos(float(i)*0.011f) - 0.03f*std::sin(float(i)*0.007f);
    }

    llama_kvarn_body_record record = llama_kvarn_store_reference(params, head_dim, k_tile, v_tile);
    require(record.k_body.size() == layout.k_body_bytes, "stored K body size");
    require(record.v_body.size() == layout.v_body_bytes, "stored V body size");
    require(record.k_scales.size() == layout.k_scale_floats, "stored K scale count");
    require(record.v_scales.size() == layout.v_scale_floats, "stored V scale count");

    std::vector<float> k_deq;
    std::vector<float> v_deq;
    llama_kvarn_dequant_reference(record, k_deq, v_deq);

    require(k_deq.size() == k_tile.size(), "dequant K size");
    require(v_deq.size() == v_tile.size(), "dequant V size");

    float k_abs = 0.0f;
    float v_abs = 0.0f;
    for (size_t i = 0; i < k_deq.size(); ++i) {
        require(std::isfinite(k_deq[i]), "finite K dequant");
        require(std::isfinite(v_deq[i]), "finite V dequant");
        k_abs = std::max(k_abs, std::fabs(k_deq[i]));
        v_abs = std::max(v_abs, std::fabs(v_deq[i]));
    }
    require(k_abs > 0.0f, "nonzero K dequant");
    require(v_abs > 0.0f, "nonzero V dequant");
    }
}

// The global-RMS Sinkhorn pre-normalization and the bit-aware RTN clip must
// strictly improve reconstruction NMSE on realistic small-magnitude tiles with
// token-scale outliers (the paper's target failure mode). Uses env toggles to
// A/B the same tiles through both quantizer configurations.
static void test_reference_quantizer_fidelity() {
    llama_kvarn_params params = llama_kvarn_default_params();
    params.sinkhorn_iters = 8;

    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (const uint32_t head_dim : { 128u, 512u }) {
        const uint32_t group = params.group_size;
        std::vector<float> k_tile(size_t(head_dim)*group);
        std::vector<float> v_tile(size_t(head_dim)*group);

        // Small-magnitude tiles (raw RMS ~0.05) with 8x token outliers: the
        // clamp-pinned reference recipe leaves these unnormalized.
        for (uint32_t g = 0; g < group; ++g) {
            const float token_scale = (g % 16 == 0) ? 0.4f : 0.05f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                k_tile[size_t(d)*group + g] = token_scale*dist(rng);
                v_tile[size_t(g)*head_dim + d] = token_scale*dist(rng);
            }
        }

        // The reference dequant returns tiles in the rotated (paper) frame;
        // compare against the same rotation of the source tiles.
        std::vector<float> k_rot;
        std::vector<float> v_rot;
        llama_kvarn_hadamard_channels(k_tile, k_rot, head_dim, group, true);
        llama_kvarn_hadamard_channels(v_tile, v_rot, group, head_dim, false);

        const auto nmse = [&](const llama_kvarn_params & p) {
            llama_kvarn_body_record record = llama_kvarn_store_reference(p, head_dim, k_tile, v_tile);
            std::vector<float> k_deq;
            std::vector<float> v_deq;
            llama_kvarn_dequant_reference(record, k_deq, v_deq);
            double k_err = 0.0, k_ref = 0.0, v_err = 0.0, v_ref = 0.0;
            for (size_t i = 0; i < k_rot.size(); ++i) {
                const double dk = double(k_deq[i]) - k_rot[i];
                const double dv = double(v_deq[i]) - v_rot[i];
                k_err += dk*dk; k_ref += double(k_rot[i])*k_rot[i];
                v_err += dv*dv; v_ref += double(v_rot[i])*v_rot[i];
            }
            return std::pair<double, double>(k_err/k_ref, v_err/v_ref);
        };

        const auto with_env = [&](const char * name, const auto & fn) {
#ifdef _WIN32
            _putenv_s(name, "1");
            auto result = fn();
            _putenv_s(name, "");
#else
            setenv(name, "1", 1);
            auto result = fn();
            unsetenv(name);
#endif
            return result;
        };

        // K4/V2 default (global-norm + clip on)
        const auto now = nmse(params);
        // legacy: global-norm off
        const auto no_norm = with_env("LLAMA_KVARN_DISABLE_GLOBAL_NORM", [&]() { return nmse(params); });
        // legacy: clip off
        const auto no_clip = with_env("LLAMA_KVARN_DISABLE_RTN_CLIP", [&]() { return nmse(params); });

        require(std::isfinite(now.first) && std::isfinite(now.second), "fidelity NMSE finite");
        // Global-norm must improve K NMSE substantially on this tile shape.
        require(now.first < 0.75*no_norm.first, "global-norm improves K reconstruction");
        // Clip must improve V2 NMSE substantially.
        require(now.second < 0.6*no_clip.second, "RTN clip improves V2 reconstruction");
        // Sanity ceilings for the default K4/V2 preset on Gaussianized tiles.
        require(now.first < 0.03, "K4 reconstruction NMSE ceiling");
        require(now.second < 0.2, "V2 reconstruction NMSE ceiling");

        // K2/V2 must roundtrip and stay bounded as well.
        llama_kvarn_params p22 = params;
        p22.key_bits = 2;
        const auto k2 = nmse(p22);
        require(std::isfinite(k2.first) && k2.first < 0.25, "K2 reconstruction NMSE ceiling");
    }
}

static void test_reference_cache_sealing() {
    llama_kvarn_params params = llama_kvarn_default_params();
    params.group_size = 4;
    params.sink_tokens = 2;
    params.tail_tokens = 2;
    params.sinkhorn_iters = 2;

    llama_kvarn_reference_cache cache(params, 8);

    for (uint32_t t = 0; t < 9; ++t) {
        std::vector<float> k(8);
        std::vector<float> v(8);
        for (uint32_t d = 0; d < 8; ++d) {
            k[d] = float(t)*0.1f + float(d)*0.01f;
            v[d] = float(t)*0.2f - float(d)*0.02f;
        }
        cache.append_token(k, v);
    }

    const llama_kvarn_reference_cache_stats s = cache.stats();
    require(s.n_tokens == 9, "reference cache token count");
    require(s.n_sink == 2, "reference cache sink count");
    require(s.n_tail == 2, "reference cache tail count");
    require(s.n_pending_body == 1, "reference cache pending count");
    require(s.n_body_records == 1, "reference cache body record count");
    require(cache.body_records()[0].k_body.size() == 16, "reference cache sealed K bytes");
    require(cache.body_records()[0].v_body.size() == 8, "reference cache sealed V bytes");

    std::vector<float> k_mat;
    std::vector<float> v_mat;
    cache.materialize_tokens(k_mat, v_mat);
    require(k_mat.size() == size_t(s.n_tokens)*8, "materialized K token count");
    require(v_mat.size() == size_t(s.n_tokens)*8, "materialized V token count");

    for (uint32_t d = 0; d < 8; ++d) {
        require(std::fabs(k_mat[d] - (float(d)*0.01f)) < 1.0e-4f, "materialized sink K order");
        require(std::fabs(v_mat[d] - (-float(d)*0.02f)) < 1.0e-4f, "materialized sink V order");
    }

    bool body_nonzero = false;
    for (uint32_t i = params.sink_tokens*8; i < (params.sink_tokens + params.group_size)*8; ++i) {
        body_nonzero = body_nonzero || std::fabs(k_mat[i]) > 0.0f || std::fabs(v_mat[i]) > 0.0f;
    }
    require(body_nonzero, "materialized sealed body contributes data");

    cache.clear();
    const llama_kvarn_reference_cache_stats empty = cache.stats();
    require(empty.n_tokens == 0, "reference cache clear token count");
    require(empty.n_body_records == 0, "reference cache clear records");
}

static llama_hparams make_test_hparams(uint32_t head_dim = 128) {
    llama_hparams hparams = {};
    hparams.n_layer_all = 2;
    hparams.n_embd_head_k_full = head_dim;
    hparams.n_embd_head_v_full = head_dim;
    hparams.n_embd_head_k_swa = head_dim;
    hparams.n_embd_head_v_swa = head_dim;
    hparams.n_head_kv_arr[0] = 4;
    hparams.n_head_kv_arr[1] = 4;
    hparams.n_head_arr[0] = 4;
    hparams.n_head_arr[1] = 4;
    return hparams;
}

static llama_hparams make_small_storage_hparams() {
    llama_hparams hparams = {};
    hparams.n_layer_all = 2;
    hparams.n_embd_head_k_full = 8;
    hparams.n_embd_head_v_full = 8;
    hparams.n_embd_head_k_swa = 8;
    hparams.n_embd_head_v_swa = 8;
    hparams.n_head_kv_arr[0] = 2;
    hparams.n_head_kv_arr[1] = 2;
    hparams.n_head_arr[0] = 2;
    hparams.n_head_arr[1] = 2;
    return hparams;
}

static llama_hparams make_shared_kv_hparams(uint32_t n_layer, int32_t n_layer_kv_from_start) {
    llama_hparams hparams = {};
    hparams.n_layer_all = n_layer;
    hparams.n_layer_kv_from_start = n_layer_kv_from_start;
    return hparams;
}

static size_t expected_body_store_scratch_floats(
        const llama_kvarn_layer_view & view,
        const llama_kvarn_params & params) {
    const size_t tile = size_t(view.head_dim_k)*params.group_size;
    // Best-so-far scratch per tile: row scales + col scales + imbalance + global RMS.
    const size_t per_pipeline = tile + 2*std::max<uint32_t>(view.head_dim_k, params.group_size) +
        view.head_dim_k + params.group_size + 2;
    const size_t pipeline_scratch = view.head_dim_k >= 256 ? 2*per_pipeline : per_pipeline;
    const bool needs_pending_head_tiles = view.n_head_kv > 1 || view.head_dim_k >= 512;
    size_t expected = needs_pending_head_tiles ? 2*tile + pipeline_scratch : pipeline_scratch;

    if ((view.layout_k.key_bits == 2 || view.layout_k.key_bits == 4 || view.layout_k.key_bits == 8) &&
            (view.layout_v.value_bits == 2 || view.layout_v.value_bits == 4 || view.layout_v.value_bits == 8)) {
        constexpr uint32_t direct_record_batch_max = 8;
        const size_t n_tiles = size_t(view.n_head_kv)*direct_record_batch_max;
        const size_t data_floats = n_tiles*tile;
        const size_t best_floats = n_tiles*(size_t(view.head_dim_k) + params.group_size + 2);
        expected = std::max(expected, 2*data_floats + 2*best_floats);
    }
    return expected;
}

static int64_t store_pipeline_scratch_floats(int32_t head_dim, int32_t group_size) {
    return int64_t(head_dim)*group_size + 2*std::max(head_dim, group_size) + head_dim + group_size + 2;
}

static int64_t fused_store_scratch_floats(int32_t head_dim, int32_t group_size) {
    const int64_t per_pipeline = store_pipeline_scratch_floats(head_dim, group_size);
    return head_dim >= 256 ? 2*per_pipeline : per_pipeline;
}

static int64_t batched_store_scratch_floats(int32_t head_dim, int32_t group_size) {
    return 2*int64_t(head_dim)*group_size + fused_store_scratch_floats(head_dim, group_size);
}

static ggml_backend_dev_t find_cuda_device() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (reg != nullptr && std::strcmp(ggml_backend_reg_name(reg), "CUDA") == 0) {
            return dev;
        }
    }
    return nullptr;
}

static void require_cuda_scratch_boundary(
        ggml_backend_dev_t cuda_dev,
        ggml_tensor * op,
        ggml_tensor * scratch,
        const char * exact_msg,
        const char * short_msg) {
    if (cuda_dev == nullptr) {
        return;
    }

    require(ggml_backend_dev_supports_op(cuda_dev, op), exact_msg);
    require(scratch->ne[0] > 1, "KVarN scratch boundary has a decrementable extent");
    --scratch->ne[0];
    require(!ggml_backend_dev_supports_op(cuda_dev, op), short_msg);
    ++scratch->ne[0];
    require(ggml_backend_dev_supports_op(cuda_dev, op), exact_msg);
}


static void test_shared_kv_reuse_layer_matching_attention_type() {
    {
        llama_hparams hparams = make_shared_kv_hparams(6, -1);
        for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
            require(hparams.kv_reuse_layer_matching_attention_type((int32_t) il) == -1,
                    "no shared KV returns no donor");
        }
    }

    {
        llama_hparams hparams = make_shared_kv_hparams(6, 6);
        for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
            require(hparams.kv_reuse_layer_matching_attention_type((int32_t) il) == -1,
                    "zero shared layers returns no donor");
        }
    }

    {
        llama_hparams hparams = make_shared_kv_hparams(12, 6);
        hparams.set_swa_pattern(6, false);

        require(hparams.kv_reuse_layer_matching_attention_type(0) == -1,
                "physical SWA layer has no donor");
        require(hparams.kv_reuse_layer_matching_attention_type(5) == -1,
                "physical full-attention layer has no donor");
        require(hparams.kv_reuse_layer_matching_attention_type(6) == 4,
                "shared SWA layer reuses last physical SWA donor");
        require(hparams.kv_reuse_layer_matching_attention_type(10) == 4,
                "later shared SWA layer reuses last physical SWA donor");
        require(hparams.kv_reuse_layer_matching_attention_type(11) == 5,
                "shared full-attention layer reuses last physical full-attention donor");
    }

    {
        llama_hparams hparams = make_shared_kv_hparams(8, 4);
        hparams.is_swa_impl[0] = true;
        hparams.is_swa_impl[1] = false;
        hparams.is_swa_impl[2] = true;
        hparams.is_swa_impl[3] = false;
        hparams.is_swa_impl[4] = false;
        hparams.is_swa_impl[5] = true;
        hparams.is_swa_impl[6] = false;
        hparams.is_swa_impl[7] = true;

        require(hparams.kv_reuse_layer_matching_attention_type(4) == 3,
                "nonstandard full-attention shared layer reuses matching donor");
        require(hparams.kv_reuse_layer_matching_attention_type(5) == 2,
                "nonstandard SWA shared layer reuses matching donor");
        require(hparams.kv_reuse_layer_matching_attention_type(6) == 3,
                "nonstandard later full-attention shared layer reuses same matching donor");
        require(hparams.kv_reuse_layer_matching_attention_type(7) == 2,
                "nonstandard later SWA shared layer reuses same matching donor");
    }
}

static llama_ubatch make_test_ubatch(uint32_t n_tokens, llama_seq_id seq_id) {
    llama_ubatch ubatch = {};
    ubatch.b_equal_seqs = false;
    ubatch.n_tokens = n_tokens;
    ubatch.n_seq_tokens = 1;
    ubatch.n_seqs = n_tokens;
    ubatch.n_seqs_unq = 1;
    ubatch.n_pos = 1;
    ubatch.data = std::make_shared<llama_ubatch::data_t>();

    auto & data = *ubatch.data;
    data.pos.resize(n_tokens);
    data.n_seq_id.resize(n_tokens, 1);
    data.seq_id.resize(n_tokens);
    data.seq_id_unq = { seq_id };
    data.seq_idx.resize(LLAMA_MAX_SEQ, -1);
    data.seq_idx[seq_id] = 0;
    data.output.resize(n_tokens, 0);
    data.seq_id_data.resize(n_tokens, seq_id);

    for (uint32_t i = 0; i < n_tokens; ++i) {
        data.pos[i] = i;
        data.seq_id[i] = &data.seq_id_data[i];
    }

    ubatch.pos = data.pos.data();
    ubatch.n_seq_id = data.n_seq_id.data();
    ubatch.seq_id = data.seq_id.data();
    ubatch.seq_id_unq = data.seq_id_unq.data();
    ubatch.seq_idx = data.seq_idx.data();
    ubatch.output = data.output.data();
    return ubatch;
}

// ---------------------------------------------------------------------------
// Stream-consistency simulation.
//
// Drives a full contiguous token stream through the cache's input-building
// machinery (sink/tail slot indices, tail-eviction plans, pending offsets) and
// tracks which logical position each physical slot holds. After every ubatch
// it recomputes the active window and the causal mask and verifies that the
// order the mixed-attention loader would read (sink | body records | pending |
// tail ring at tail_start) yields exactly positions 0..last.
//
// The window and seal rules below deliberately mirror
// kvarn_graph_active_window() / kvarn_graph_seal_records() in llama-graph.cpp;
// if either side changes, this test fails.
// ---------------------------------------------------------------------------

struct kvarn_sim_window {
    uint32_t n_sink = 0;
    uint32_t n_records = 0;
    uint32_t n_pending = 0;
    uint32_t n_tail = 0;
    uint32_t tail_start = 0;
    uint32_t n_kv = 0;
};

static llama_kvarn_params kvarn_sim_effective(llama_kvarn_params p, uint32_t kv_size) {
    if (kv_size != 0 && uint64_t(p.sink_tokens) + uint64_t(p.tail_tokens) > kv_size) {
        p.tail_tokens = p.sink_tokens >= kv_size ? 0 : kv_size - p.sink_tokens;
    }
    return p;
}

static kvarn_sim_window kvarn_sim_active_window(const llama_kvarn_params & params, uint32_t kv_size, uint32_t last_pos) {
    const llama_kvarn_params p = kvarn_sim_effective(params, kv_size);
    kvarn_sim_window w;
    const uint32_t n_seen = last_pos + 1;
    w.n_sink = std::min<uint32_t>(n_seen, p.sink_tokens);
    const uint32_t after_sink = n_seen - w.n_sink;
    w.n_tail = std::min<uint32_t>(after_sink, p.tail_tokens);
    const uint32_t body_pending = after_sink - w.n_tail;
    w.n_records = body_pending/p.group_size;
    w.n_pending = body_pending%p.group_size;
    w.tail_start = w.n_tail == 0 ? 0 : body_pending%p.tail_tokens;
    w.n_kv = w.n_sink + w.n_records*p.group_size + w.n_pending + w.n_tail;
    return w;
}

static void kvarn_sim_run_stream(
        uint32_t sink,
        uint32_t tail,
        uint32_t group,
        uint32_t kv_size,
        const std::vector<uint32_t> & ubatch_sizes_cycle,
        uint32_t n_stream,
        const char * label) {
    llama_kvarn_params params = llama_kvarn_default_params();
    params.sink_tokens = sink;
    params.tail_tokens = tail;
    params.group_size = group;

    const llama_kvarn_params eff = kvarn_sim_effective(params, kv_size);

    llama_hparams hparams = make_test_hparams();
    llama_kv_cache_kvarn cache(nullptr, hparams, params, false, kv_size, 4, 1, nullptr);

    const uint32_t n_sink_tail = std::min<uint32_t>(kv_size, sink + tail);

    constexpr int64_t kvarn_sim_unset = -1;
    std::vector<int64_t> ring_model(n_sink_tail, kvarn_sim_unset);      // slot -> pos
    std::vector<int64_t> pending_model(group, kvarn_sim_unset);         // offset -> pos
    std::vector<std::vector<int64_t>> record_model;                     // record -> offsets -> pos

    uint32_t next_pos = 0;
    size_t cycle_i = 0;
    while (next_pos < n_stream) {
        const uint32_t n_tokens = std::min<uint32_t>(
                ubatch_sizes_cycle[cycle_i++ % ubatch_sizes_cycle.size()], n_stream - next_pos);
        llama_ubatch ubatch = make_test_ubatch(n_tokens, 0);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            ubatch.data->pos[i] = llama_pos(next_pos + i);
        }
        ubatch.pos = ubatch.data->pos.data();

        ggml_init_params init_params = { 256*1024, nullptr, true };
        ggml_context_ptr ctx { ggml_init(init_params) };

        ggml_tensor * idxs      = cache.build_input_sink_tail_idxs(ctx.get(), ubatch);
        ggml_tensor * plan      = cache.build_input_body_plan(ctx.get(), ubatch);
        ggml_tensor * offsets   = cache.build_input_body_offsets(ctx.get(), ubatch);
        ggml_tensor * tail_idxs = cache.build_input_tail_evict_idxs(ctx.get(), ubatch);
        const kvarn_sim_window w = kvarn_sim_active_window(params, kv_size, next_pos + n_tokens - 1);
        ggml_tensor * mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, w.n_kv, n_tokens);
        ggml_backend_buffer_ptr buf {
            ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type())
        };
        require(buf != nullptr, "kvarn sim buffer");

        cache.set_input_sink_tail_idxs(idxs, &ubatch);
        cache.set_input_body_plan(plan, &ubatch);
        cache.set_input_body_offsets(offsets, &ubatch);
        cache.set_input_tail_evict_idxs(tail_idxs, &ubatch);
        cache.set_input_kq_mask(mask, &ubatch, true);

        // 1+2. Evictions and seals interleave exactly like the graph: the
        //    pending buffer holds one group, and each seal's slice copy runs
        //    before the next record's evictions overwrite the ring of pending
        //    offsets (enforced in the graph with explicit set-rows deps).
        //    Evictions read the sink/tail ring BEFORE this ubatch's writes.
        const int64_t n_evict = offsets->ne[0];
        const int64_t * offset_data = (const int64_t *) offsets->data;
        const int32_t * tail_idx_data = (const int32_t *) tail_idxs->data;
        {
            int64_t j = 0;
            for (uint32_t i = 0; i < n_tokens; ++i) {
                const uint32_t pos = next_pos + i;
                if (pos < eff.sink_tokens + eff.tail_tokens) {
                    continue;
                }
                require(j < n_evict, "kvarn sim eviction count");
                const int32_t slot = tail_idx_data[j];
                require(slot >= int32_t(eff.sink_tokens) && uint32_t(slot) < n_sink_tail,
                        "kvarn sim eviction slot in tail range");
                const int64_t evicted = ring_model[slot];
                require(evicted == int64_t(pos) - int64_t(eff.tail_tokens),
                        "kvarn sim eviction reads the displaced token");
                const int64_t off = offset_data[j];
                require(off >= 0 && off < int64_t(group), "kvarn sim pending offset range");
                pending_model[size_t(off)] = evicted;
                ++j;

                // Seal fires when this eviction completes a record (mirror of
                // kvarn_graph_seal_records), consuming the pending group
                // before any later-record eviction can overwrite it.
                const uint32_t body_pos = uint32_t(evicted) - eff.sink_tokens;
                if (body_pos%eff.group_size + 1 != eff.group_size) {
                    continue;
                }
                const uint32_t record = body_pos/eff.group_size;
                require(record == record_model.size(), "kvarn sim records seal in order");
                for (uint32_t g = 0; g < group; ++g) {
                    require(pending_model[g] == int64_t(eff.sink_tokens) + int64_t(record)*group + g,
                            "kvarn sim sealed record holds its group positions");
                }
                record_model.push_back(pending_model);
            }
            require(j == n_evict, "kvarn sim eviction plan length");
        }

        // 3. This ubatch's tokens land in their sink/tail slots.
        const int64_t * idx_data = (const int64_t *) idxs->data;
        for (uint32_t i = 0; i < n_tokens; ++i) {
            const uint32_t pos = next_pos + i;
            const int64_t slot = idx_data[i];
            require(slot >= 0 && uint32_t(slot) < n_sink_tail, "kvarn sim sink/tail slot bounds");
            if (pos < eff.sink_tokens) {
                require(slot == int64_t(pos), "kvarn sim sink slot identity");
            }
            ring_model[size_t(slot)] = int64_t(pos);
        }

        // 4. Seal timing must agree with the active-window record count.
        require(w.n_records == record_model.size(), "kvarn sim window records match seal timing");

        // 5. The loader order (sink | records | pending | tail ring) must
        //    enumerate exactly positions 0..last.
        std::vector<int64_t> loader_pos(w.n_kv, kvarn_sim_unset);
        for (uint32_t t = 0; t < w.n_kv; ++t) {
            int64_t pos = kvarn_sim_unset;
            if (t < w.n_sink) {
                pos = ring_model[t];
            } else if (t < w.n_sink + w.n_records*group) {
                const uint32_t body_t = t - w.n_sink;
                pos = record_model[body_t/group][body_t%group];
            } else if (t < w.n_sink + w.n_records*group + w.n_pending) {
                pos = pending_model[t - w.n_sink - w.n_records*group];
            } else {
                const uint32_t i = t - w.n_sink - w.n_records*group - w.n_pending;
                const uint32_t slot = eff.sink_tokens + (w.tail_start + i)%w.n_tail;
                pos = ring_model[slot];
            }
            require(pos == int64_t(t < w.n_sink ? t :
                    t < w.n_kv - w.n_tail ? eff.sink_tokens + (t - w.n_sink) :
                    (next_pos + n_tokens) - w.n_tail + (t - (w.n_kv - w.n_tail))),
                    label);
            loader_pos[t] = pos;
        }

        // 6. The causal mask must expose exactly the keys at loader positions
        //    <= each query position.
        const float * mask_data = (const float *) mask->data;
        for (uint32_t q = 0; q < n_tokens; ++q) {
            const int64_t q_pos = int64_t(next_pos + q);
            for (uint32_t t = 0; t < w.n_kv; ++t) {
                const bool visible = mask_data[size_t(q)*w.n_kv + t] > -1.0e20f;
                require(visible == (loader_pos[t] <= q_pos), "kvarn sim mask matches loader positions");
            }
        }

        next_pos += n_tokens;
    }
}

static void test_runtime_stream_consistency() {
    // Ring wraps many times, records seal across ubatch boundaries.
    kvarn_sim_run_stream(2, 4, 4, 64, { 3, 1, 4, 2 }, 40, "kvarn sim loader positions (2/4/4)");
    // Non-power-of-two tail ring stresses the modulo/tail_start math.
    kvarn_sim_run_stream(3, 5, 4, 64, { 5, 2, 1, 3 }, 45, "kvarn sim loader positions (3/5/4)");
    // Minimal decode-like geometry.
    kvarn_sim_run_stream(1, 2, 2, 32, { 1, 2 }, 20, "kvarn sim loader positions (1/2/2)");
    // kv_size smaller than sink+tail: effective-tail clamping must keep the
    // plans, slots, window, and mask on the same geometry.
    kvarn_sim_run_stream(4, 6, 4, 8, { 2, 1 }, 8, "kvarn sim loader positions (clamped tail)");
}

static void test_runtime_state_safety() {
    llama_kvarn_params params = llama_kvarn_default_params();
    params.group_size = 4;
    params.sink_tokens = 2;
    params.tail_tokens = 4;

    llama_hparams hparams = make_test_hparams();
    llama_kv_cache_kvarn cache(nullptr, hparams, params, false, 32, 4, 1, nullptr);
    llama_kv_cache_kvarn single_stream_cache(nullptr, hparams, params, false, 32, 1, 1, nullptr);

    const auto prepare_positions = [&](std::initializer_list<llama_pos> positions) {
        llama_ubatch ubatch = make_test_ubatch(uint32_t(positions.size()), 0);
        std::copy(positions.begin(), positions.end(), ubatch.data->pos.begin());
        ubatch.pos = ubatch.data->pos.data();
        return single_stream_cache.prepare({ ubatch });
    };

    require(prepare_positions({ 1 }).empty(), "KVarN initial offset position rejected");
    require(prepare_positions({ 0, 2 }).empty(), "KVarN initial position gap rejected");
    require(prepare_positions({ 0, 0 }).empty(), "KVarN duplicate position rejected");
    require(prepare_positions({ -1 }).empty(), "KVarN negative position rejected");

    llama_ubatch append0 = make_test_ubatch(3, 0);
    llama_ubatch append1 = make_test_ubatch(2, 0);
    append1.data->pos = { 3, 4 };
    append1.pos = append1.data->pos.data();
    const auto append_sinfos = single_stream_cache.prepare({ append0, append1 });
    require(append_sinfos.size() == 2, "KVarN contiguous positions append across ubatches");
    single_stream_cache.apply_ubatch(append_sinfos[0], append0);
    single_stream_cache.apply_ubatch(append_sinfos[1], append1);
    require(single_stream_cache.seq_pos_max(0) == 4, "KVarN contiguous multi-ubatch append applied");
    require(prepare_positions({ 4 }).empty(), "KVarN rewind position rejected");
    require(prepare_positions({ 6 }).empty(), "KVarN skipped append position rejected");

    // No shift graph exists; advertising shift support would let context-shift
    // silently mis-rotate stored K.
    require(!cache.get_can_shift(), "KVarN refuses K-shift support");

    // Fill positions 0..11: the tail ring (slots for pos 2..5) has wrapped
    // (pos 6..11 reused the slots of pos 2..7).
    for (uint32_t pos = 0; pos < 12; ++pos) {
        llama_ubatch ubatch = make_test_ubatch(1, 0);
        ubatch.data->pos[0] = llama_pos(pos);
        ubatch.pos = ubatch.data->pos.data();
        const auto sinfo = cache.find_slot(ubatch);
        require(!sinfo.empty(), "KVarN state safety slot");
        cache.apply_ubatch(sinfo, ubatch);
    }

    cache.seq_add(0, 0, -1, 0);
    require(cache.seq_pos_min(0) == 0 && cache.seq_pos_max(0) == 11,
            "KVarN zero-shift seq_add is a no-op");

    // Rolling back into a reused ring region would leave the mask pointing at
    // slots holding future-token values; must be refused.
    require(!cache.seq_rm(0, 8, -1), "KVarN post-wrap suffix seq_rm refused");
    require(cache.seq_pos_max(0) == 11, "KVarN refused seq_rm left state unchanged");

    // Mid-range removal breaks position contiguity; must be refused.
    require(!cache.seq_rm(0, 3, 7), "KVarN mid-range seq_rm refused");

    require(cache.seq_rm(0, 20, 30), "KVarN non-overlapping seq_rm is a no-op");
    require(cache.seq_rm(0, 5, 5), "KVarN empty-range seq_rm is a no-op");
    require(cache.seq_pos_max(0) == 11, "KVarN no-op removals leave state unchanged");

    // Removing everything is always exact.
    require(cache.seq_rm(0, 0, -1), "KVarN full-range seq_rm accepted");
    require(cache.seq_pos_max(0) == -1, "KVarN full-range seq_rm cleared");

    // Partial rollback is refused even before slot reuse so the capability
    // probe cannot advertise support that disappears later in the session.
    for (uint32_t pos = 0; pos < 5; ++pos) {
        llama_ubatch ubatch = make_test_ubatch(1, 0);
        ubatch.data->pos[0] = llama_pos(pos);
        ubatch.pos = ubatch.data->pos.data();
        const auto sinfo = cache.find_slot(ubatch);
        require(!sinfo.empty(), "KVarN state safety refill slot");
        cache.apply_ubatch(sinfo, ubatch);
    }
    require(!cache.seq_rm(0, 3, -1), "KVarN pre-wrap suffix seq_rm refused consistently");
    require(cache.seq_pos_max(0) == 4, "KVarN refused pre-wrap seq_rm left state unchanged");

    cache.seq_cp(0, 1, 0, -1);
    require(cache.seq_rm(-1, 0, -1), "KVarN all-sequence full removal accepted");
    require(cache.seq_pos_max(0) == -1 && cache.seq_pos_max(1) == -1,
            "KVarN all-sequence full removal cleared every sequence");
}

static std::array<ggml_tensor *, 8> kvarn_test_layer_tensors(const llama_kvarn_layer_view & view) {
    return { view.sink_tail_k, view.sink_tail_v, view.body_k, view.body_v,
             view.scales_k, view.scales_v, view.pending_k, view.pending_v };
}

static void kvarn_test_seed_state(llama_kv_cache_kvarn & cache) {
    llama_ubatch ubatch = make_test_ubatch(3, 0);
    ubatch.n_pos = 4;
    ubatch.data->pos.resize(12);
    for (uint32_t i = 0; i < 3; ++i) {
        ubatch.data->pos[i] = llama_pos(10 + i);
        ubatch.data->pos[i + 3] = llama_pos(20 + i);
        ubatch.data->pos[i + 6] = llama_pos(30 + i);
        ubatch.data->pos[i + 9] = llama_pos(40 + i);
    }
    ubatch.pos = ubatch.data->pos.data();
    llama_kv_cache_kvarn::slot_info sinfo;
    sinfo.idxs = { 13, 14, 15 }; // exercises the valid one-past-end head
    cache.apply_ubatch(sinfo, ubatch);
    cache.seq_cp(0, 1, 10, 12);

    for (uint32_t il = 0; il < cache.get_n_layer(); ++il) {
        const auto tensors = kvarn_test_layer_tensors(cache.get_layer_view(int32_t(il)));
        for (uint32_t it = 0; it < tensors.size(); ++it) {
            ggml_tensor * tensor = tensors[it];
            std::vector<uint8_t> bytes(ggml_nbytes(tensor));
            for (size_t i = 0; i < bytes.size(); ++i) {
                bytes[i] = uint8_t((il*67 + it*29 + i*7 + 3) & 0xff);
            }
            ggml_backend_tensor_set(tensor, bytes.data(), 0, bytes.size());
        }
    }
}

static void test_runtime_state_roundtrip() {
    llama_kvarn_params params = llama_kvarn_default_params();
    params.group_size = 4;
    params.sink_tokens = 2;
    params.tail_tokens = 2;
    llama_hparams hparams = make_test_hparams();
    llama_kv_cache_kvarn source(nullptr, hparams, params, false, 16, 4, 1, nullptr);
    llama_kv_cache_kvarn restored(nullptr, hparams, params, false, 16, 4, 1, nullptr);
    kvarn_test_seed_state(source);

    kvarn_test_writer writer;
    source.state_write(writer);
    require(writer.n_bytes() == writer.data.size(), "KVarN state writer exact byte count");
    kvarn_test_sizer sizer;
    source.state_write(sizer);
    require(sizer.n_bytes() == writer.n_bytes(), "KVarN state dummy and writer sizes match exactly");
    kvarn_test_reader reader(writer.data);
    restored.state_read(reader);
    require(reader.n_bytes() == writer.data.size(), "KVarN state reader consumed exact byte count");

    kvarn_test_writer rewritten;
    restored.state_write(rewritten);
    require(rewritten.data == writer.data, "KVarN full state roundtrip is byte exact");

    bool unsupported = false;
    try {
        source.state_write(rewritten, 0, 0);
    } catch (const std::runtime_error &) {
        unsupported = true;
    }
    require(unsupported, "KVarN per-sequence state write rejected");

    kvarn_test_reader unsupported_reader(writer.data);
    unsupported = false;
    try {
        restored.state_read(unsupported_reader, 0, 0);
    } catch (const std::runtime_error &) {
        unsupported = true;
    }
    require(unsupported, "KVarN per-sequence state read rejected");
    require(restored.seq_pos_min(0) == 10 && restored.seq_pos_max(0) == 12,
            "unsupported KVarN per-sequence state read is non-mutating");
}

static void test_runtime_state_rejects_corruption() {
    llama_kvarn_params params = llama_kvarn_default_params();
    params.group_size = 4;
    params.sink_tokens = 2;
    params.tail_tokens = 2;
    llama_hparams hparams = make_test_hparams();
    llama_kv_cache_kvarn source(nullptr, hparams, params, false, 16, 4, 1, nullptr);
    kvarn_test_seed_state(source);
    kvarn_test_writer writer;
    source.state_write(writer);

    std::vector<std::vector<uint8_t>> corruptions;
    corruptions.push_back(writer.data);
    corruptions.back()[0] ^= 0xff; // magic
    corruptions.push_back(writer.data);
    corruptions.back()[8] ^= 0xff; // version
    corruptions.push_back(writer.data);
    corruptions.back()[12] ^= 0x01; // kv_size geometry
    corruptions.push_back(writer.data);
    corruptions.back().pop_back(); // payload truncation

    for (const std::vector<uint8_t> & bytes : corruptions) {
        llama_kv_cache_kvarn destination(nullptr, hparams, params, false, 16, 4, 1, nullptr);
        kvarn_test_seed_state(destination);
        kvarn_test_reader reader(bytes);
        bool rejected = false;
        try {
            destination.state_read(reader);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        require(rejected, "corrupt KVarN state rejected");
        require(destination.seq_pos_min(0) == -1 && destination.seq_pos_min(1) == -1,
                "failed KVarN restore clears cell metadata");
        const auto tensors = kvarn_test_layer_tensors(destination.get_layer_view(0));
        std::vector<uint8_t> tensor_bytes(ggml_nbytes(tensors[0]));
        ggml_backend_tensor_get(tensors[0], tensor_bytes.data(), 0, tensor_bytes.size());
        require(std::all_of(tensor_bytes.begin(), tensor_bytes.end(), [](uint8_t b) { return b == 0; }),
                "failed KVarN restore clears backend tensors");
    }

    // The writer remains permissive, but a matching single-stream reader must
    // reject sparse metadata produced through the low-level direct-apply API.
    llama_kv_cache_kvarn sparse(nullptr, hparams, params, false, 16, 1, 1, nullptr);
    llama_ubatch sparse_ubatch = make_test_ubatch(2, 0);
    sparse_ubatch.data->pos = { 0, 2 };
    sparse_ubatch.pos = sparse_ubatch.data->pos.data();
    llama_kv_cache_kvarn::slot_info sparse_sinfo;
    sparse_sinfo.idxs = { 0, 1 };
    sparse.apply_ubatch(sparse_sinfo, sparse_ubatch);
    kvarn_test_writer sparse_writer;
    sparse.state_write(sparse_writer);

    llama_kv_cache_kvarn sparse_destination(nullptr, hparams, params, false, 16, 1, 1, nullptr);
    llama_ubatch existing = make_test_ubatch(1, 0);
    llama_kv_cache_kvarn::slot_info existing_sinfo;
    existing_sinfo.idxs = { 0 };
    sparse_destination.apply_ubatch(existing_sinfo, existing);
    const auto destination_tensors = kvarn_test_layer_tensors(sparse_destination.get_layer_view(0));
    std::vector<uint8_t> nonzero(ggml_nbytes(destination_tensors[0]), uint8_t{ 0x5a });
    ggml_backend_tensor_set(destination_tensors[0], nonzero.data(), 0, nonzero.size());

    kvarn_test_reader sparse_reader(sparse_writer.data);
    bool sparse_rejected = false;
    try {
        sparse_destination.state_read(sparse_reader);
    } catch (const std::runtime_error & e) {
        sparse_rejected = std::string(e.what()).find("not dense from zero") != std::string::npos;
    }
    require(sparse_rejected, "sparse single-stream KVarN state rejected by density validation");
    require(sparse_destination.seq_pos_min(0) == -1,
            "failed sparse KVarN restore clears cell metadata");
    std::vector<uint8_t> cleared(nonzero.size());
    ggml_backend_tensor_get(destination_tensors[0], cleared.data(), 0, cleared.size());
    require(std::all_of(cleared.begin(), cleared.end(), [](uint8_t b) { return b == 0; }),
            "failed sparse KVarN restore clears backend tensors");
}

static void test_reference_store_scale_invariance() {
    llama_kvarn_params params = llama_kvarn_default_params();
    params.sinkhorn_iters = 8;

    const uint32_t head_dim = 128;
    const uint32_t group = params.group_size;
    std::mt19937 rng(77);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> k_base(size_t(head_dim)*group);
    std::vector<float> v_base(size_t(head_dim)*group);
    for (size_t i = 0; i < k_base.size(); ++i) {
        k_base[i] = dist(rng);
        v_base[i] = dist(rng);
    }

    const auto nmse_at_scale = [&](float scale) {
        std::vector<float> k_tile(k_base.size());
        std::vector<float> v_tile(v_base.size());
        for (size_t i = 0; i < k_base.size(); ++i) {
            k_tile[i] = scale*k_base[i];
            v_tile[i] = scale*v_base[i];
        }
        std::vector<float> k_rot;
        std::vector<float> v_rot;
        llama_kvarn_hadamard_channels(k_tile, k_rot, head_dim, group, true);
        llama_kvarn_hadamard_channels(v_tile, v_rot, group, head_dim, false);
        llama_kvarn_body_record record = llama_kvarn_store_reference(params, head_dim, k_tile, v_tile);
        std::vector<float> k_deq;
        std::vector<float> v_deq;
        llama_kvarn_dequant_reference(record, k_deq, v_deq);
        double err = 0.0, ref = 0.0;
        for (size_t i = 0; i < k_rot.size(); ++i) {
            const double dk = double(k_deq[i]) - k_rot[i];
            const double dv = double(v_deq[i]) - v_rot[i];
            err += dk*dk + dv*dv;
            ref += double(k_rot[i])*k_rot[i] + double(v_rot[i])*v_rot[i];
        }
        return err/ref;
    };

    // With the global-RMS pre-normalization the quantizer must be invariant to
    // the raw tile magnitude; the clamp-pinned recipe degraded badly at small
    // scales.
    const double e_small = nmse_at_scale(0.02f);
    const double e_unit  = nmse_at_scale(1.0f);
    const double e_large = nmse_at_scale(30.0f);
    require(std::isfinite(e_small) && std::isfinite(e_unit) && std::isfinite(e_large),
            "scale invariance NMSE finite");
    require(e_small < 1.3*e_unit && e_unit < 1.3*e_small, "store NMSE invariant to small tile scale");
    require(e_large < 1.3*e_unit && e_unit < 1.3*e_large, "store NMSE invariant to large tile scale");
}

static void test_memory_estimate() {
    llama_kvarn_params params = llama_kvarn_default_params();
    llama_hparams hparams = make_test_hparams();

    const llama_kvarn_memory_estimate est = llama_kvarn_estimate_memory(params, hparams, 512);
    require(est.fp16_sink_tail_bytes == 1048576, "KVarN sink/tail estimate");
    require(est.body_packed_bytes == 196608, "KVarN body estimate");
    require(est.scale_bytes == 49152, "KVarN scale estimate");
    require(est.total_bytes == 1294336, "KVarN total estimate");

    llama_hparams hparams256 = make_test_hparams(256);
    const llama_kvarn_memory_estimate est256 = llama_kvarn_estimate_memory(params, hparams256, 512);
    require(est256.fp16_sink_tail_bytes == 2097152, "256-dim KVarN sink/tail estimate");
    require(est256.body_packed_bytes == 393216, "256-dim KVarN body estimate");
    require(est256.scale_bytes == 73728, "256-dim KVarN scale estimate");
    require(est256.total_bytes == 2564096, "256-dim KVarN total estimate");

    llama_hparams hparams512 = make_test_hparams(512);
    const llama_kvarn_memory_estimate est512 = llama_kvarn_estimate_memory(params, hparams512, 512);
    require(est512.fp16_sink_tail_bytes == 4194304, "512-dim KVarN sink/tail estimate");
    require(est512.body_packed_bytes == 786432, "512-dim KVarN body estimate");
    require(est512.scale_bytes == 122880, "512-dim KVarN scale estimate");
    require(est512.total_bytes == 5103616, "512-dim KVarN total estimate");

    llama_hparams hparams_high_gqa = make_test_hparams(128);
    hparams_high_gqa.n_head_arr[0] = 16;
    hparams_high_gqa.n_head_kv_arr[0] = 1;
    const llama_kvarn_memory_estimate est_high_gqa = llama_kvarn_estimate_memory(params, hparams_high_gqa, 512);
    require(est_high_gqa.fp16_sink_tail_bytes == 655360, "high-GQA KVarN sink/tail estimate");
    require(est_high_gqa.body_packed_bytes == 139264, "high-GQA KVarN promoted K8 body estimate");
    require(est_high_gqa.scale_bytes == 30720, "high-GQA KVarN scale estimate");
    require(est_high_gqa.total_bytes == 825344, "high-GQA KVarN total estimate");

}

static void test_runtime_metadata() {
    llama_kvarn_params params = llama_kvarn_default_params();
    params.sink_tokens = 8;
    params.tail_tokens = 8;
    llama_hparams hparams = make_test_hparams();
    llama_kv_cache_kvarn cache(nullptr, hparams, params, false, 16, 4, 1, nullptr);

    require(cache.get_size() == 16, "KVarN runtime size");
    require(cache.get_n_layer() == 2, "KVarN runtime layer count");
    require(cache.backend_tensor_bytes() > 0, "KVarN backend tensors allocated");
    const llama_kvarn_layer_view view = cache.get_layer_view(0);
    require(view.il == 0, "KVarN layer view id");
    require(view.n_head_kv == 4, "KVarN layer view head count");
    require(view.n_records == 0, "KVarN layer view logical body records");
    require(view.head_dim_k == 128, "KVarN layer view K head dim");
    require(view.head_dim_v == 128, "KVarN layer view V head dim");
    require(view.layout_k.k_body_bytes == 8192, "KVarN layer view K body bytes");
    require(view.layout_v.v_body_bytes == 4096, "KVarN layer view V body bytes");
    require(view.sink_tail_k->type == GGML_TYPE_F16, "KVarN layer view sink K type");
    require(view.sink_tail_v->type == GGML_TYPE_F16, "KVarN layer view sink V type");
    require(view.body_k->type == GGML_TYPE_I8, "KVarN layer view body K type");
    require(view.body_v->type == GGML_TYPE_I8, "KVarN layer view body V type");
    require(view.scales_k->type == GGML_TYPE_F32, "KVarN layer view scale K type");
    require(view.scales_v->type == GGML_TYPE_F32, "KVarN layer view scale V type");
    require(view.sink_tail_k->ne[0] == 128 && view.sink_tail_k->ne[1] == 4, "KVarN layer view sink K shape");
    require(view.body_k->ne[0] == (int64_t) view.layout_k.k_body_bytes, "KVarN layer view body K shape");
    require(view.body_v->ne[0] == (int64_t) view.layout_v.v_body_bytes, "KVarN layer view body V shape");
    require(view.scales_k->ne[0] == (int64_t) view.layout_k.k_scale_floats, "KVarN layer view scale K shape");
    require(view.scales_v->ne[0] == (int64_t) view.layout_v.v_scale_floats, "KVarN layer view scale V shape");
    require(cache.body_store_scratch_floats(0) == expected_body_store_scratch_floats(view, params),
            "128-dim multi-head KVarN body store scratch floats");
    size_t breakdown_bytes = 0;
    for (const auto & entry : cache.memory_breakdown()) {
        breakdown_bytes += entry.second;
    }
    require(breakdown_bytes >= cache.backend_tensor_bytes(), "KVarN memory breakdown covers backend tensors");
    require(cache.seq_pos_min(0) == -1, "empty sequence min");

    llama_ubatch ubatch = make_test_ubatch(3, 0);
    llama_kv_cache_kvarn::slot_info sinfo;
    sinfo.idxs = { 0, 1, 2 };
    cache.apply_ubatch(sinfo, ubatch);

    require(cache.seq_pos_min(0) == 0, "KVarN sequence min after apply");
    require(cache.seq_pos_max(0) == 2, "KVarN sequence max after apply");

    cache.seq_cp(0, 1, 1, 3);
    require(cache.seq_pos_min(1) == 1, "KVarN copied sequence min");
    require(cache.seq_pos_max(1) == 2, "KVarN copied sequence max");

    // Prefix removals cannot be honored by the position-derived layout (the
    // mask labels slots by position math, not by cell metadata), so seq_rm
    // must refuse and leave the state unchanged for the caller to reprocess.
    require(!cache.seq_rm(0, 0, 2), "KVarN prefix seq_rm refused");
    require(cache.seq_pos_min(0) == 0, "KVarN prefix seq_rm leaves state unchanged");

    // Refuse pre-wrap suffix removal as well, otherwise the short capability
    // probe advertises partial removal that fails after the tail ring wraps.
    require(!cache.seq_rm(0, 2, -1), "KVarN pre-wrap suffix seq_rm refused");
    require(cache.seq_pos_max(0) == 2, "KVarN refused suffix seq_rm leaves max unchanged");
    require(cache.seq_rm(0, 0, -1), "KVarN full seq_rm accepted");
    require(cache.seq_pos_min(0) == -1, "KVarN full seq_rm cleared sequence");

    cache.seq_keep(1);
    require(cache.seq_pos_min(0) == -1, "KVarN sequence keep removed old sequence");
    require(cache.seq_pos_max(1) == 2, "KVarN sequence keep preserved target");

    cache.clear(true);
    require(cache.seq_pos_min(1) == -1, "KVarN clear metadata");

    llama_memory_context_ptr full_ctx = cache.init_full();
    require(full_ctx != nullptr, "KVarN init_full returns context");
    require(full_ctx->get_status() == LLAMA_MEMORY_STATUS_SUCCESS, "KVarN init_full success status");
    require(dynamic_cast<llama_kv_cache_kvarn_context *>(full_ctx.get()) != nullptr, "KVarN init_full context type");

    llama_hparams hparams256 = make_test_hparams(256);
    llama_kv_cache_kvarn cache256(nullptr, hparams256, params, false, 16, 4, 1, nullptr);
    const llama_kvarn_layer_view view256 = cache256.get_layer_view(0);
    require(view256.head_dim_k == 256, "256-dim KVarN layer view K head dim");
    require(view256.head_dim_v == 256, "256-dim KVarN layer view V head dim");
    require(view256.layout_k.k_body_bytes == 16384, "256-dim KVarN layer view K body bytes");
    require(view256.layout_v.v_body_bytes == 8192, "256-dim KVarN layer view V body bytes");
    require(view256.scales_k->ne[0] == 640, "256-dim KVarN layer view scale K shape");
    require(view256.scales_v->ne[0] == 512, "256-dim KVarN layer view scale V shape");
    require(cache256.body_store_scratch_floats(0) == expected_body_store_scratch_floats(view256, params),
            "256-dim KVarN body store scratch floats");

    llama_hparams hparams512 = make_test_hparams(512);
    llama_kv_cache_kvarn cache512(nullptr, hparams512, params, false, 16, 4, 1, nullptr);
    const llama_kvarn_layer_view view512 = cache512.get_layer_view(0);
    require(view512.head_dim_k == 512, "512-dim KVarN layer view K head dim");
    require(view512.head_dim_v == 512, "512-dim KVarN layer view V head dim");
    require(view512.layout_k.k_body_bytes == 32768, "512-dim KVarN layer view K body bytes");
    require(view512.layout_v.v_body_bytes == 16384, "512-dim KVarN layer view V body bytes");
    require(view512.scales_k->ne[0] == 1152, "512-dim KVarN layer view scale K shape");
    require(view512.scales_v->ne[0] == 768, "512-dim KVarN layer view scale V shape");
    require(cache512.body_store_scratch_floats(0) == expected_body_store_scratch_floats(view512, params),
            "512-dim KVarN body store scratch floats");

    llama_hparams hparams_reuse = make_test_hparams(512);
    hparams_reuse.n_layer_kv_from_start = 1;
    llama_kv_cache_kvarn cache_reuse(
            nullptr, hparams_reuse, params, false, 16, 4, 1, nullptr,
            [](int32_t il) { return il == 1 ? 0 : -1; });
    require(cache_reuse.get_n_layer() == 1, "KVarN reuse cache allocates one physical KV layer");
    const llama_kvarn_layer_view view_reuse0 = cache_reuse.get_layer_view(0);
    const llama_kvarn_layer_view view_reuse1 = cache_reuse.get_layer_view(1);
    require(view_reuse1.il == 0, "KVarN reuse layer view maps to physical source layer");
    require(view_reuse1.sink_tail_k == view_reuse0.sink_tail_k, "KVarN reuse layer shares sink K storage");
    require(view_reuse1.body_k == view_reuse0.body_k, "KVarN reuse layer shares body K storage");
    require(cache_reuse.body_store_scratch_floats(1) == expected_body_store_scratch_floats(view_reuse1, params),
            "KVarN reuse body scratch uses logical layer head dim");

    llama_hparams hparams_asym = make_test_hparams(256);
    hparams_asym.n_embd_head_v_full = 128;
    hparams_asym.n_embd_head_v_swa = 128;
    bool rejected_asym = false;
    try {
        llama_kv_cache_kvarn cache_asym(nullptr, hparams_asym, params, false, 16, 4, 1, nullptr);
    } catch (const std::invalid_argument & e) {
        rejected_asym = std::strstr(e.what(), "equal K and V head dimensions") != nullptr;
    }
    require(rejected_asym, "KVarN rejects asymmetric K/V head dimensions");

    llama_hparams hparams_high_gqa = make_test_hparams(128);
    hparams_high_gqa.n_head_arr[0] = 16;
    hparams_high_gqa.n_head_kv_arr[0] = 1;
    llama_kv_cache_kvarn cache_high_gqa(nullptr, hparams_high_gqa, params, false, 512, 4, 1, nullptr);
    const llama_kvarn_layer_view view_high_gqa = cache_high_gqa.get_layer_view(0);
    const llama_kvarn_layer_view view_non_high_gqa = cache_high_gqa.get_layer_view(1);
    require(view_high_gqa.layout_k.key_bits == 8, "high-GQA KVarN promotes low-bit K to K8");
    require(view_high_gqa.layout_v.value_bits == params.value_bits, "high-GQA KVarN preserves requested V bits");
    require(view_high_gqa.layout_k.k_body_bytes == 16384, "high-GQA KVarN K8 body bytes");
    require(view_non_high_gqa.layout_k.key_bits == params.key_bits, "non-high-GQA KVarN layer preserves requested K bits");

    set_test_env("LLAMA_KVARN_DISABLE_HIGH_GQA_K8", "0");
    llama_kv_cache_kvarn cache_high_gqa_zero(nullptr, hparams_high_gqa, params, false, 512, 4, 1, nullptr);
    set_test_env("LLAMA_KVARN_DISABLE_HIGH_GQA_K8", nullptr);
    const llama_kvarn_layer_view view_high_gqa_zero = cache_high_gqa_zero.get_layer_view(0);
    require(view_high_gqa_zero.layout_k.key_bits == 8, "high-GQA K8 policy treats opt-out flag 0 as disabled");

    set_test_env("LLAMA_KVARN_DISABLE_HIGH_GQA_K8", "1");
    llama_kv_cache_kvarn cache_high_gqa_optout(nullptr, hparams_high_gqa, params, false, 512, 4, 1, nullptr);
    set_test_env("LLAMA_KVARN_DISABLE_HIGH_GQA_K8", nullptr);
    const llama_kvarn_layer_view view_high_gqa_optout = cache_high_gqa_optout.get_layer_view(0);
    require(view_high_gqa_optout.layout_k.key_bits == params.key_bits, "high-GQA K8 policy opt-out preserves requested K bits");

    bool rejected_bad_high_gqa_env = false;
    set_test_env("LLAMA_KVARN_DISABLE_HIGH_GQA_K8", "false");
    try {
        llama_kv_cache_kvarn cache_high_gqa_bad_env(nullptr, hparams_high_gqa, params, false, 512, 4, 1, nullptr);
    } catch (const std::invalid_argument & e) {
        rejected_bad_high_gqa_env = std::strstr(e.what(), "LLAMA_KVARN_DISABLE_HIGH_GQA_K8=false") != nullptr;
    }
    set_test_env("LLAMA_KVARN_DISABLE_HIGH_GQA_K8", nullptr);
    require(rejected_bad_high_gqa_env, "high-GQA K8 opt-out rejects non-0/1 env values");

    set_test_env("LLAMA_KVARN_LAYER_KEY_BITS", "0=4");
    llama_kv_cache_kvarn cache_high_gqa_key_override(nullptr, hparams_high_gqa, params, false, 512, 4, 1, nullptr);
    set_test_env("LLAMA_KVARN_LAYER_KEY_BITS", nullptr);
    require(cache_high_gqa_key_override.get_layer_view(0).layout_k.key_bits == 8,
            "high-GQA K8 policy still protects K after a low-bit layer override");

    set_test_env("LLAMA_KVARN_LAYER_VALUE_BITS", "0=4");
    llama_kv_cache_kvarn cache_high_gqa_v_override(nullptr, hparams_high_gqa, params, false, 512, 4, 1, nullptr);
    set_test_env("LLAMA_KVARN_LAYER_VALUE_BITS", nullptr);
    const llama_kvarn_layer_view view_high_gqa_v_override = cache_high_gqa_v_override.get_layer_view(0);
    require(view_high_gqa_v_override.layout_k.key_bits == 8, "high-GQA K8 policy preserves K protection with V override");
    require(view_high_gqa_v_override.layout_v.value_bits == 4, "high-GQA K8 policy preserves V override");

}

static void test_runtime_storage_sealing() {
    llama_kvarn_params params = llama_kvarn_default_params();
    params.group_size = 4;
    params.sink_tokens = 2;
    params.tail_tokens = 2;
    params.sinkhorn_iters = 2;

    llama_hparams hparams = make_small_storage_hparams();
    llama_kv_cache_kvarn cache(nullptr, hparams, params, false, 32, 4, 1, nullptr);

    const uint32_t n_tokens = 9;
    const uint32_t n_heads = 2;
    const uint32_t head_dim = 8;
    std::vector<float> k_tokens(size_t(n_tokens)*n_heads*head_dim);
    std::vector<float> v_tokens(k_tokens.size());

    for (uint32_t t = 0; t < n_tokens; ++t) {
        for (uint32_t h = 0; h < n_heads; ++h) {
            for (uint32_t d = 0; d < head_dim; ++d) {
                const size_t off = (size_t(t)*n_heads + h)*head_dim + d;
                k_tokens[off] = 0.01f*float(t + 1) + 0.10f*float(h) + 0.001f*float(d);
                v_tokens[off] = 0.02f*float(t + 1) - 0.07f*float(h) - 0.002f*float(d);
            }
        }
    }

    cache.append_layer_tokens_reference(0, k_tokens, v_tokens, n_tokens);

    const llama_kvarn_runtime_storage_stats stats = cache.storage_stats();
    require(stats.n_layers == 2, "runtime storage layer count");
    require(stats.n_heads == 4, "runtime storage head count");
    require(stats.n_tokens == n_tokens, "runtime storage token count");
    require(stats.n_body_records == 2, "runtime storage sealed body records");
    require(stats.n_pending_body == 2, "runtime storage pending body count");
    require(stats.n_sink == 4, "runtime storage sink tokens across heads");
    require(stats.n_tail == 4, "runtime storage tail tokens across heads");
    require(stats.fp16_sink_tail_values == 128, "runtime storage fp16 sink/tail values");
    require(stats.body_packed_bytes == 48, "runtime storage packed body bytes");
    require(stats.scale_values == 72, "runtime storage scale values");
    require(cache.backend_tensor_bytes() >= stats.fp16_sink_tail_values*sizeof(uint16_t) + stats.body_packed_bytes,
            "runtime backend tensor capacity covers materialized storage");

    std::vector<float> k_mat;
    std::vector<float> v_mat;
    cache.materialize_layer_tokens_reference(0, k_mat, v_mat);
    require(k_mat.size() == k_tokens.size(), "runtime materialized K size");
    require(v_mat.size() == v_tokens.size(), "runtime materialized V size");

    require(std::fabs(k_mat[0] - k_tokens[0]) < 1.0e-4f, "runtime materialized first sink K");
    require(std::fabs(v_mat[0] - v_tokens[0]) < 1.0e-4f, "runtime materialized first sink V");

    const size_t tail_off = (size_t(n_tokens - 1)*n_heads + 1)*head_dim + 7;
    require(std::fabs(k_mat[tail_off] - k_tokens[tail_off]) < 1.0e-4f, "runtime materialized tail K");
    require(std::fabs(v_mat[tail_off] - v_tokens[tail_off]) < 1.0e-4f, "runtime materialized tail V");

    bool body_nonzero = false;
    const size_t body_begin = size_t(params.sink_tokens)*n_heads*head_dim;
    const size_t body_end = size_t(params.sink_tokens + params.group_size)*n_heads*head_dim;
    for (size_t i = body_begin; i < body_end; ++i) {
        body_nonzero = body_nonzero || std::fabs(k_mat[i]) > 0.0f || std::fabs(v_mat[i]) > 0.0f;
    }
    require(body_nonzero, "runtime materialized quantized body data");

    cache.clear(true);
    const llama_kvarn_runtime_storage_stats empty = cache.storage_stats();
    require(empty.n_tokens == 0, "runtime storage clear tokens");
    require(empty.n_body_records == 0, "runtime storage clear bodies");
}

static void test_runtime_sink_tail_graph_api() {
    llama_kvarn_params params = llama_kvarn_default_params();
    params.group_size = 4;
    params.sink_tokens = 2;
    params.tail_tokens = 4;

    llama_hparams hparams = make_test_hparams();
    hparams.n_head_kv_arr[0] = 2;
    hparams.n_head_arr[0] = 2;
    llama_kv_cache_kvarn cache(nullptr, hparams, params, false, 8, 4, 1, nullptr);

    llama_ubatch ubatch = make_test_ubatch(5, 0);
    ubatch.data->pos = { 0, 1, 2, 3, 6 };
    ubatch.pos = ubatch.data->pos.data();

    ggml_init_params init_params = {
        /*.mem_size   =*/ 16*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx { ggml_init(init_params) };

    ggml_tensor * idxs = cache.build_input_sink_tail_idxs(ctx.get(), ubatch);
    ggml_tensor * k_cur = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 128, 2, ubatch.n_tokens);
    ggml_tensor * v_cur = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 128, 2, ubatch.n_tokens);

    ggml_tensor * k_store = cache.cpy_sink_tail_k(ctx.get(), k_cur, idxs, 0);
    ggml_tensor * v_store = cache.cpy_sink_tail_v(ctx.get(), v_cur, idxs, 0);
    ggml_set_rows_add_dep(v_store, k_store);
    require(k_store->op == GGML_OP_SET_ROWS, "KVarN sink/tail K store op");
    require(v_store->op == GGML_OP_SET_ROWS, "KVarN sink/tail V store op");
    require(v_store->src[3] == k_store, "KVarN sink/tail SET_ROWS dependency source");
    require(k_store->ne[0] == 256, "KVarN sink/tail K store row width");
    require(v_store->ne[0] == 256, "KVarN sink/tail V store row width");
    require(k_store->ne[1] == 6, "KVarN sink/tail K slot count");
    require(v_store->ne[1] == 6, "KVarN sink/tail V slot count");

    ggml_backend_buffer_ptr buf {
        ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type())
    };
    require(buf != nullptr, "KVarN sink/tail graph test buffer");

    cache.set_input_sink_tail_idxs(idxs, &ubatch);
    const int64_t * idx_data = (const int64_t *) idxs->data;
    require(idx_data[0] == 0, "KVarN sink slot index 0");
    require(idx_data[1] == 1, "KVarN sink slot index 1");
    require(idx_data[2] == 2, "KVarN tail slot index 2");
    require(idx_data[3] == 3, "KVarN tail slot index 3");
    require(idx_data[4] == 2, "KVarN tail slot wraps");
}

static void test_runtime_body_plan_graph_api() {
    llama_kvarn_params params = llama_kvarn_default_params();
    params.group_size = 4;
    params.sink_tokens = 2;
    params.tail_tokens = 4;

    llama_hparams hparams = make_test_hparams();
    llama_kv_cache_kvarn cache(nullptr, hparams, params, false, 16, 4, 1, nullptr);

    llama_ubatch ubatch = make_test_ubatch(9, 0);
    ubatch.data->pos = { 0, 1, 2, 3, 5, 6, 7, 8, 9 };
    ubatch.pos = ubatch.data->pos.data();

    ggml_init_params init_params = {
        /*.mem_size   =*/ 8*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx { ggml_init(init_params) };

    ggml_tensor * plan = cache.build_input_body_plan(ctx.get(), ubatch);
    ggml_tensor * offsets = cache.build_input_body_offsets(ctx.get(), ubatch);
    ggml_tensor * tail_idxs = cache.build_input_tail_evict_idxs(ctx.get(), ubatch);
    require(plan->type == GGML_TYPE_I64, "KVarN body plan type");
    require(plan->ne[0] == 3, "KVarN body plan fields");
    require(plan->ne[1] == 4, "KVarN body plan evictions");
    require(offsets->type == GGML_TYPE_I64, "KVarN body offsets type");
    require(offsets->ne[0] == 4, "KVarN body offsets evictions");
    require(tail_idxs->type == GGML_TYPE_I32, "KVarN tail eviction index type");
    require(tail_idxs->ne[0] == 4, "KVarN tail eviction index count");

    ggml_backend_buffer_ptr buf {
        ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type())
    };
    require(buf != nullptr, "KVarN body plan graph test buffer");

    cache.set_input_body_plan(plan, &ubatch);
    cache.set_input_body_offsets(offsets, &ubatch);
    cache.set_input_tail_evict_idxs(tail_idxs, &ubatch);
    const int64_t * data = (const int64_t *) plan->data;
    const int64_t * offset_data = (const int64_t *) offsets->data;
    const int32_t * tail_idx_data = (const int32_t *) tail_idxs->data;

    const int64_t expected[][3] = {
        {  0,  0, -1 },
        {  0,  1, -1 },
        {  0,  2, -1 },
        {  0,  3,  0 },
    };
    const int32_t expected_tail_idxs[] = {
        2, 3, 4, 5,
    };

    for (uint32_t i = 0; i < 4; ++i) {
        const size_t off = size_t(i)*3;
        require(data[off + 0] == expected[i][0], "KVarN body plan record index");
        require(data[off + 1] == expected[i][1], "KVarN body plan token offset");
        require(data[off + 2] == expected[i][2], "KVarN body plan seal record");
        require(offset_data[i] == expected[i][1], "KVarN body offset value");
        require(tail_idx_data[i] == expected_tail_idxs[i], "KVarN tail eviction slot index");
    }
}

static void test_runtime_body_plan_multi_record_seals() {
    llama_kvarn_params params = llama_kvarn_default_params();
    params.group_size = 4;
    params.sink_tokens = 2;
    params.tail_tokens = 2;

    llama_hparams hparams = make_small_storage_hparams();
    llama_kv_cache_kvarn cache(nullptr, hparams, params, false, 32, 4, 1, nullptr);

    llama_ubatch ubatch = make_test_ubatch(7, 0);
    ubatch.data->pos = { 5, 6, 7, 8, 9, 10, 11 };
    ubatch.pos = ubatch.data->pos.data();

    ggml_init_params init_params = {
        /*.mem_size   =*/ 8*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx { ggml_init(init_params) };

    ggml_tensor * plan = cache.build_input_body_plan(ctx.get(), ubatch);
    ggml_tensor * offsets = cache.build_input_body_offsets(ctx.get(), ubatch);
    ggml_tensor * tail_idxs = cache.build_input_tail_evict_idxs(ctx.get(), ubatch);
    require(plan->ne[0] == 3 && plan->ne[1] == 7, "KVarN multi-record body plan shape");
    require(offsets->ne[0] == 7, "KVarN multi-record body offsets shape");
    require(tail_idxs->ne[0] == 7, "KVarN multi-record tail idx shape");

    ggml_backend_buffer_ptr buf {
        ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type())
    };
    require(buf != nullptr, "KVarN multi-record body plan buffer");

    cache.set_input_body_plan(plan, &ubatch);
    cache.set_input_body_offsets(offsets, &ubatch);
    cache.set_input_tail_evict_idxs(tail_idxs, &ubatch);

    const int64_t * data = (const int64_t *) plan->data;
    const int64_t * offset_data = (const int64_t *) offsets->data;
    const int32_t * tail_idx_data = (const int32_t *) tail_idxs->data;

    const int64_t expected[][3] = {
        { 0, 1, -1 },
        { 0, 2, -1 },
        { 0, 3,  0 },
        { 1, 0, -1 },
        { 1, 1, -1 },
        { 1, 2, -1 },
        { 1, 3,  1 },
    };
    const int64_t expected_offsets[] = {
        1, 2, 3, 0, 1, 2, 3,
    };
    const int32_t expected_tail_idxs[] = {
        3, 2, 3, 2, 3, 2, 3,
    };

    for (uint32_t i = 0; i < 7; ++i) {
        const size_t off = size_t(i)*3;
        require(data[off + 0] == expected[i][0], "KVarN multi-record body plan record");
        require(data[off + 1] == expected[i][1], "KVarN multi-record body plan offset");
        require(data[off + 2] == expected[i][2], "KVarN multi-record body plan seal");
        require(offset_data[i] == expected_offsets[i], "KVarN multi-record body offset");
        require(tail_idx_data[i] == expected_tail_idxs[i], "KVarN multi-record tail idx");
    }
}

static void test_runtime_body_plan_active_boundary_512() {
    llama_kvarn_params params = llama_kvarn_default_params();
    params.group_size = 128;
    params.sink_tokens = 128;
    params.tail_tokens = 896;

    llama_hparams hparams = make_small_storage_hparams();
    llama_kv_cache_kvarn cache(nullptr, hparams, params, false, 2048, 4, 1, nullptr);

    llama_ubatch ubatch = make_test_ubatch(512, 0);
    ubatch.data->pos.resize(512);
    for (uint32_t i = 0; i < 512; ++i) {
        ubatch.data->pos[i] = llama_pos(1024 + i);
    }
    ubatch.pos = ubatch.data->pos.data();

    ggml_init_params init_params = {
        /*.mem_size   =*/ 64*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx { ggml_init(init_params) };

    ggml_tensor * plan = cache.build_input_body_plan(ctx.get(), ubatch);
    ggml_tensor * offsets = cache.build_input_body_offsets(ctx.get(), ubatch);
    ggml_tensor * tail_idxs = cache.build_input_tail_evict_idxs(ctx.get(), ubatch);
    require(plan->ne[0] == 3 && plan->ne[1] == 512, "KVarN active-boundary body plan shape");
    require(offsets->ne[0] == 512, "KVarN active-boundary body offsets shape");
    require(tail_idxs->ne[0] == 512, "KVarN active-boundary tail idx shape");

    ggml_backend_buffer_ptr buf {
        ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type())
    };
    require(buf != nullptr, "KVarN active-boundary body plan buffer");

    cache.set_input_body_plan(plan, &ubatch);
    cache.set_input_body_offsets(offsets, &ubatch);
    cache.set_input_tail_evict_idxs(tail_idxs, &ubatch);

    const int64_t * data = (const int64_t *) plan->data;
    const int64_t * offset_data = (const int64_t *) offsets->data;
    const int32_t * tail_idx_data = (const int32_t *) tail_idxs->data;

    uint32_t record_starts[4] = {};
    uint32_t record_counts[4] = {};
    bool record_seen[4] = {};
    bool record_contiguous[4] = { true, true, true, true };

    for (uint32_t i = 0; i < 512; ++i) {
        const uint32_t expected_record = i/128;
        const uint32_t expected_offset = i%128;
        const size_t off = size_t(i)*3;
        require(data[off + 0] == int64_t(expected_record), "KVarN active-boundary body plan record");
        require(data[off + 1] == int64_t(expected_offset), "KVarN active-boundary body plan offset");
        require(data[off + 2] == (expected_offset == 127 ? int64_t(expected_record) : -1),
                "KVarN active-boundary body plan seal");
        require(offset_data[i] == int64_t(expected_offset), "KVarN active-boundary body offset");
        require(tail_idx_data[i] == int32_t(128 + i), "KVarN active-boundary tail idx");

        if (!record_seen[expected_record]) {
            record_seen[expected_record] = true;
            record_starts[expected_record] = i;
        } else if (record_starts[expected_record] + record_counts[expected_record] != i) {
            record_contiguous[expected_record] = false;
        }
        ++record_counts[expected_record];
    }

    for (uint32_t record = 0; record < 4; ++record) {
        require(record_seen[record], "KVarN active-boundary record seen");
        require(record_starts[record] == record*128, "KVarN active-boundary record slice start");
        require(record_counts[record] == 128, "KVarN active-boundary record slice count");
        require(record_contiguous[record], "KVarN active-boundary record slice contiguous");
    }
}

static float read_kq_mask_value(const ggml_tensor * mask, int64_t t, int64_t q) {
    const char * p = (const char *) mask->data + size_t(q)*mask->nb[1] + size_t(t)*mask->nb[0];
    if (mask->type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(*(const ggml_fp16_t *) p);
    }
    return *(const float *) p;
}

static void require_mask_keep(float v, const char * msg) {
    require(std::fabs(v) < 1.0e-6f, msg);
}

static void require_mask_drop(float v, const char * msg) {
    require(std::isinf(v) && v < 0.0f, msg);
}

static void test_runtime_kq_mask_graph_api() {
    llama_kvarn_params params = llama_kvarn_default_params();
    params.group_size = 4;
    params.sink_tokens = 2;
    params.tail_tokens = 2;

    llama_hparams hparams = make_small_storage_hparams();
    llama_kv_cache_kvarn cache(nullptr, hparams, params, false, 12, 4, 1, nullptr);

    llama_ubatch ubatch = make_test_ubatch(5, 0);
    ubatch.data->pos = { 0, 1, 2, 5, 6 };
    ubatch.pos = ubatch.data->pos.data();

    ggml_init_params init_params = {
        /*.mem_size   =*/ 8*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx { ggml_init(init_params) };

    ggml_tensor * causal_mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cache.get_size(), ubatch.n_tokens);
    ggml_tensor * noncausal_mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cache.get_size(), ubatch.n_tokens);
    ggml_tensor * causal_mask_f16 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, cache.get_size(), ubatch.n_tokens);

    ggml_backend_buffer_ptr buf {
        ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type())
    };
    require(buf != nullptr, "KVarN KQ mask graph test buffer");

    cache.set_input_kq_mask(causal_mask, &ubatch, true);
    cache.set_input_kq_mask(noncausal_mask, &ubatch, false);
    cache.set_input_kq_mask(causal_mask_f16, &ubatch, true);

    const llama_pos expected_pos[] = { 0, 1, 2, 5, 6 };
    const int64_t invalid_key_pos = -1;
    const int64_t expected_key_pos[] = {
        0, 1, // sink
        2, 3, 4, // pending body
        5, 6, // tail
        invalid_key_pos, invalid_key_pos, invalid_key_pos, invalid_key_pos, invalid_key_pos,
    };
    for (uint32_t q = 0; q < ubatch.n_tokens; ++q) {
        for (int64_t t = 0; t < int64_t(cache.get_size()); ++t) {
            const bool keep = expected_key_pos[t] != invalid_key_pos && expected_key_pos[t] <= int64_t(expected_pos[q]);
            if (keep) {
                require_mask_keep(read_kq_mask_value(causal_mask, t, q), "KVarN causal KQ mask keeps visible slot");
                require_mask_keep(read_kq_mask_value(causal_mask_f16, t, q), "KVarN causal F16 KQ mask keeps visible slot");
            } else {
                require_mask_drop(read_kq_mask_value(causal_mask, t, q), "KVarN causal KQ mask drops future slot");
                require_mask_drop(read_kq_mask_value(causal_mask_f16, t, q), "KVarN causal F16 KQ mask drops future slot");
            }
            require_mask_keep(read_kq_mask_value(noncausal_mask, t, q), "KVarN non-causal KQ mask keeps all slots");
        }
    }
}

static void test_runtime_body_record_graph_api() {
    llama_kvarn_params params = llama_kvarn_default_params();
    params.group_size = 4;
    params.sink_tokens = 2;
    params.tail_tokens = 2;
    params.sinkhorn_iters = 2;

    llama_hparams hparams = make_small_storage_hparams();
    llama_kv_cache_kvarn cache(nullptr, hparams, params, false, 32, 4, 1, nullptr);
    const llama_kvarn_layer_view view = cache.get_layer_view(0);

    require(view.n_records == 7, "KVarN body record view count");
    require(view.layout_k.k_body_bytes == 16, "KVarN K body record bytes");
    require(view.layout_v.v_body_bytes == 8, "KVarN V body record bytes");
    require(view.layout_k.k_scale_floats == 20, "KVarN K scale record floats");
    require(view.layout_v.v_scale_floats == 16, "KVarN V scale record floats");
    require(cache.body_store_scratch_floats(0) == expected_body_store_scratch_floats(view, params),
            "KVarN small multi-head body store scratch floats");

    ggml_init_params init_params = {
        /*.mem_size   =*/ 64*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx { ggml_init(init_params) };

    ggml_tensor * k_tile = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, params.group_size, view.head_dim_k);
    ggml_tensor * v_tile = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, view.head_dim_v, params.group_size);
    ggml_tensor * body_offsets = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, params.group_size);
    ggml_tensor * scratch = cache.build_body_store_scratch(ctx.get(), 0);

    const uint32_t ih = 1;
    const uint32_t record = 2;
    ggml_tensor * k_body = cache.view_k_body_record(ctx.get(), 0, ih, record);
    ggml_tensor * v_body = cache.view_v_body_record(ctx.get(), 0, ih, record);
    ggml_tensor * k_scales = cache.view_k_scales_record(ctx.get(), 0, ih, record);
    ggml_tensor * v_scales = cache.view_v_scales_record(ctx.get(), 0, ih, record);

    require(k_body->type == GGML_TYPE_I8 && k_body->ne[0] == (int64_t) view.layout_k.k_body_bytes,
            "KVarN K body record view shape");
    require(v_body->type == GGML_TYPE_I8 && v_body->ne[0] == (int64_t) view.layout_v.v_body_bytes,
            "KVarN V body record view shape");
    require(k_scales->type == GGML_TYPE_F32 && k_scales->ne[0] == (int64_t) view.layout_k.k_scale_floats,
            "KVarN K scale record view shape");
    require(v_scales->type == GGML_TYPE_F32 && v_scales->ne[0] == (int64_t) view.layout_v.v_scale_floats,
            "KVarN V scale record view shape");
    require(k_body->view_src == view.body_k, "KVarN K body record view source");
    require(v_body->view_src == view.body_v, "KVarN V body record view source");
    require(k_scales->view_src == view.scales_k, "KVarN K scale record view source");
    require(v_scales->view_src == view.scales_v, "KVarN V scale record view source");
    require(k_body->view_offs == size_t(ih)*view.body_k->nb[2] + size_t(record)*view.body_k->nb[1],
            "KVarN K body record view offset");
    require(v_body->view_offs == size_t(ih)*view.body_v->nb[2] + size_t(record)*view.body_v->nb[1],
            "KVarN V body record view offset");
    require(k_scales->view_offs == size_t(ih)*view.scales_k->nb[2] + size_t(record)*view.scales_k->nb[1],
            "KVarN K scale record view offset");
    require(v_scales->view_offs == size_t(ih)*view.scales_v->nb[2] + size_t(record)*view.scales_v->nb[1],
            "KVarN V scale record view offset");

    ggml_tensor * k_store = cache.store_k_body_record(ctx.get(), k_tile, scratch, 0, ih, record);
    ggml_tensor * v_store = cache.store_v_body_record(ctx.get(), v_tile, scratch, 0, ih, record);
    ggml_tensor * kv_store = cache.store_kv_body_record(ctx.get(), k_tile, v_tile, scratch, 0, ih, record);
    ggml_tensor * k_all_heads = ggml_new_tensor_3d(
            ctx.get(), GGML_TYPE_F32, view.head_dim_k, view.n_head_kv, params.group_size);
    ggml_tensor * v_all_heads = ggml_new_tensor_3d(
            ctx.get(), GGML_TYPE_F32, view.head_dim_v, view.n_head_kv, params.group_size);
    ggml_tensor * kv_all_heads = cache.store_kv_body_all_heads(ctx.get(), k_all_heads, v_all_heads, scratch, 0, record);

    require(k_store->op == GGML_OP_KVARN_STORE_BODY, "KVarN K body record store op");
    require(v_store->op == GGML_OP_KVARN_STORE_BODY, "KVarN V body record store op");
    require(kv_store->op == GGML_OP_KVARN_STORE_KV_BODY, "KVarN fused KV body record store op");
    require(kv_all_heads->op == GGML_OP_KVARN_STORE_KV_BODY, "KVarN fused KV all-head body record store op");
    require(k_store->src[0] == k_tile && k_store->src[1]->view_src == view.scales_k &&
            k_store->src[2] == scratch && k_store->src[3]->view_src == view.body_k,
            "KVarN K body record store sources");
    require(v_store->src[0] == v_tile && v_store->src[1]->view_src == view.scales_v &&
            v_store->src[2] == scratch && v_store->src[3]->view_src == view.body_v,
            "KVarN V body record store sources");
    require(kv_store->src[0] == k_tile && kv_store->src[1] == v_tile &&
            kv_store->src[2]->view_src == view.scales_k && kv_store->src[3]->view_src == view.scales_v &&
            kv_store->src[4] == scratch && kv_store->src[5]->view_src == view.body_k &&
            kv_store->src[6]->view_src == view.body_v,
            "KVarN fused KV body record store sources");
    require(k_store->op_params[1] == (int32_t) view.head_dim_k &&
            k_store->op_params[2] == (int32_t) params.group_size &&
            k_store->op_params[3] == (int32_t) params.key_bits,
            "KVarN K body record store params");
    require(v_store->op_params[1] == (int32_t) view.head_dim_v &&
            v_store->op_params[3] == (int32_t) params.value_bits &&
            v_store->op_params[4] == (int32_t) params.sinkhorn_iters,
            "KVarN V body record store params");
    require(kv_store->op_params[0] == (int32_t) view.head_dim_k &&
            kv_store->op_params[1] == (int32_t) params.group_size &&
            kv_store->op_params[2] == (int32_t) params.key_bits &&
            kv_store->op_params[3] == (int32_t) params.value_bits &&
            kv_store->op_params[4] == (int32_t) params.sinkhorn_iters,
            "KVarN fused KV body record store params");

    ggml_tensor * tail_idxs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, params.group_size);
    ggml_tensor * k_pending = cache.cpy_tail_evict_pending_k(ctx.get(), tail_idxs, body_offsets, 0);
    ggml_tensor * v_pending = cache.cpy_tail_evict_pending_v(ctx.get(), tail_idxs, body_offsets, 0);
    ggml_tensor * k_from_pending = cache.store_k_body_record_from_pending(ctx.get(), scratch, 0, ih, record);
    ggml_tensor * v_from_pending = cache.store_v_body_record_from_pending(ctx.get(), scratch, 0, ih, record);
    ggml_tensor * kv_from_pending = cache.store_kv_body_record_from_pending(ctx.get(), scratch, 0, ih, record);
    ggml_tensor * k_pending_src = ggml_reshape_3d(ctx.get(), k_pending, view.head_dim_k, view.n_head_kv, params.group_size);
    ggml_tensor * v_pending_src = ggml_reshape_3d(ctx.get(), v_pending, view.head_dim_v, view.n_head_kv, params.group_size);
    ggml_tensor * kv_from_pending_dep = cache.store_kv_body_all_heads_from_pending(
            ctx.get(), scratch, 0, record, k_pending_src, v_pending_src);
    ggml_kvarn_store_kv_body_add_dep(kv_from_pending_dep, kv_from_pending);

    require(k_pending->op == GGML_OP_SET_ROWS, "KVarN pending K store op");
    require(v_pending->op == GGML_OP_SET_ROWS, "KVarN pending V store op");
    require(k_from_pending->op == GGML_OP_KVARN_STORE_BODY, "KVarN pending K body store op");
    require(v_from_pending->op == GGML_OP_KVARN_STORE_BODY, "KVarN pending V body store op");
    require(kv_from_pending->op == GGML_OP_KVARN_STORE_KV_BODY, "KVarN pending fused KV body store op");
    require(k_from_pending->src[0]->type == GGML_TYPE_F32 && k_from_pending->src[0]->ne[0] == (int64_t) params.group_size,
            "KVarN pending K body tile shape");
    require(v_from_pending->src[0]->type == GGML_TYPE_F32 && v_from_pending->src[0]->ne[0] == (int64_t) view.head_dim_v,
            "KVarN pending V body tile shape");
    require(kv_from_pending->src[0]->type == GGML_TYPE_F32 && kv_from_pending->src[0]->ne[0] == (int64_t) params.group_size &&
            kv_from_pending->src[1]->type == GGML_TYPE_F32 && kv_from_pending->src[1]->ne[0] == (int64_t) view.head_dim_v,
            "KVarN pending fused KV body tile shapes");
    require(kv_from_pending_dep->src[0] == k_pending_src && kv_from_pending_dep->src[1] == v_pending_src,
            "KVarN pending fused KV body store accepts producer pending tensors");
    require(kv_from_pending_dep->src[7] == kv_from_pending,
            "KVarN fused KV body store dependency source");

    bool rejected_multi_record_pending = false;
    try {
        std::vector<uint32_t> records = { 0, 1 };
        (void) cache.store_kv_body_records_from_pending(ctx.get(), scratch, 0, records);
    } catch (const std::invalid_argument &) {
        rejected_multi_record_pending = true;
    }
    require(rejected_multi_record_pending,
            "KVarN pending fused KV body store rejects unsafe multi-record batches");
}

static void test_kvarn_store_body_ggml_ops() {
    llama_kvarn_params params = llama_kvarn_default_params();
    params.rtn_quantile = 0.75f;

    ggml_init_params init_params = {
        /*.mem_size   =*/ 512*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx { ggml_init(init_params) };

    for (const int32_t head_dim : { 128, 256, 512 }) {
        llama_kvarn_layout layout = llama_kvarn_make_layout(params, head_dim);

        ggml_tensor * k_tile = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, params.group_size, head_dim);
        ggml_tensor * v_tile = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, head_dim, params.group_size);
        ggml_tensor * k_body = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, layout.k_body_bytes);
        ggml_tensor * v_body = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, layout.v_body_bytes);
        ggml_tensor * k_scales = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, layout.k_scale_floats);
        ggml_tensor * v_scales = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, layout.v_scale_floats);
        ggml_tensor * scratch = ggml_new_tensor_1d(
                ctx.get(), GGML_TYPE_F32, store_pipeline_scratch_floats(head_dim, params.group_size));

        ggml_tensor * k_store = ggml_kvarn_store_k_body(
                ctx.get(), k_tile, k_body, k_scales, scratch,
                head_dim, params.group_size, params.key_bits, params.sinkhorn_iters, params.rtn_quantile);
        ggml_tensor * v_store = ggml_kvarn_store_v_body(
                ctx.get(), v_tile, v_body, v_scales, scratch,
                head_dim, params.group_size, params.value_bits, params.sinkhorn_iters, params.rtn_quantile);

        require(k_store->op == GGML_OP_KVARN_STORE_BODY, "KVarN K body store op");
        require(v_store->op == GGML_OP_KVARN_STORE_BODY, "KVarN V body store op");
        require(k_store->type == GGML_TYPE_I8, "KVarN K body store result type");
        require(v_store->type == GGML_TYPE_I8, "KVarN V body store result type");
        require(k_store->view_src == k_body, "KVarN K body store returns body view");
        require(v_store->view_src == v_body, "KVarN V body store returns body view");
        require(k_store->src[0] == k_tile && k_store->src[1] == k_scales && k_store->src[2] == scratch && k_store->src[3] == k_body,
                "KVarN K body store sources");
        require(v_store->src[0] == v_tile && v_store->src[1] == v_scales && v_store->src[2] == scratch && v_store->src[3] == v_body,
                "KVarN V body store sources");
        require(k_store->op_params[0] == 0, "KVarN K body store mode");
        require(v_store->op_params[0] == 1, "KVarN V body store mode");
        require(k_store->op_params[1] == head_dim && k_store->op_params[2] == (int32_t) params.group_size,
                "KVarN K body store geometry");
        require(v_store->op_params[3] == (int32_t) params.value_bits && v_store->op_params[4] == (int32_t) params.sinkhorn_iters,
                "KVarN V body store params");
        float k_quantile = 0.0f;
        float v_quantile = 0.0f;
        std::memcpy(&k_quantile, &k_store->op_params[5], sizeof(float));
        std::memcpy(&v_quantile, &v_store->op_params[5], sizeof(float));
        require(std::fabs(k_quantile - params.rtn_quantile) < 1.0e-6f &&
                std::fabs(v_quantile - params.rtn_quantile) < 1.0e-6f,
                "KVarN body store quantile params");
    }
}

static void test_kvarn_store_body_scratch_contracts() {
    const llama_kvarn_params params = llama_kvarn_default_params();
    ggml_backend_dev_t cuda_dev = find_cuda_device();

    for (const int32_t head_dim : { 128, 256, 512 }) {
        const int32_t group_size = params.group_size;
        const int32_t n_heads = 2;
        const int32_t n_records = 2;
        const llama_kvarn_layout layout = llama_kvarn_make_layout(params, head_dim);

        ggml_init_params init_params = {
            /*.mem_size   =*/ 2*1024*1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr ctx { ggml_init(init_params) };

        // Single K/V stores use one pipeline at every supported head dimension.
        ggml_tensor * k_tile = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, group_size, head_dim);
        ggml_tensor * v_tile = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, head_dim, group_size);
        ggml_tensor * k_body = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, layout.k_body_bytes);
        ggml_tensor * v_body = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, layout.v_body_bytes);
        ggml_tensor * k_scales = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, layout.k_scale_floats);
        ggml_tensor * v_scales = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, layout.v_scale_floats);
        ggml_tensor * single_scratch = ggml_new_tensor_1d(
                ctx.get(), GGML_TYPE_F32, store_pipeline_scratch_floats(head_dim, group_size));
        ggml_tensor * k_store = ggml_kvarn_store_k_body(
                ctx.get(), k_tile, k_body, k_scales, single_scratch,
                head_dim, group_size, params.key_bits, params.sinkhorn_iters, params.rtn_quantile);
        ggml_tensor * v_store = ggml_kvarn_store_v_body(
                ctx.get(), v_tile, v_body, v_scales, single_scratch,
                head_dim, group_size, params.value_bits, params.sinkhorn_iters, params.rtn_quantile);
        require_cuda_scratch_boundary(cuda_dev, k_store, single_scratch,
                "CUDA supports exact-minimum KVarN single-K scratch",
                "CUDA rejects one-short KVarN single-K scratch");
        require_cuda_scratch_boundary(cuda_dev, v_store, single_scratch,
                "CUDA supports exact-minimum KVarN single-V scratch",
                "CUDA rejects one-short KVarN single-V scratch");

        // A simple fused store uses two independent pipelines starting at 256d.
        ggml_tensor * fused_scratch = ggml_new_tensor_1d(
                ctx.get(), GGML_TYPE_F32, fused_store_scratch_floats(head_dim, group_size));
        ggml_tensor * fused_store = ggml_kvarn_store_kv_body(
                ctx.get(), k_tile, v_tile, k_body, v_body, k_scales, v_scales, fused_scratch,
                head_dim, group_size, params.key_bits, params.value_bits,
                params.sinkhorn_iters, params.rtn_quantile);
        require_cuda_scratch_boundary(cuda_dev, fused_store, fused_scratch,
                "CUDA supports exact-minimum KVarN simple fused scratch",
                "CUDA rejects one-short KVarN simple fused scratch");

        const int64_t batched_scratch_floats = batched_store_scratch_floats(head_dim, group_size);

        // Pending-head sources gather one record across multiple heads.
        ggml_tensor * pending_heads_k = ggml_new_tensor_3d(
                ctx.get(), GGML_TYPE_F32, head_dim, n_heads, group_size);
        ggml_tensor * pending_heads_v = ggml_new_tensor_3d(
                ctx.get(), GGML_TYPE_F32, head_dim, n_heads, group_size);
        ggml_tensor * heads_k_body = ggml_new_tensor_2d(
                ctx.get(), GGML_TYPE_I8, layout.k_body_bytes, n_heads);
        ggml_tensor * heads_v_body = ggml_new_tensor_2d(
                ctx.get(), GGML_TYPE_I8, layout.v_body_bytes, n_heads);
        ggml_tensor * heads_k_scales = ggml_new_tensor_2d(
                ctx.get(), GGML_TYPE_F32, layout.k_scale_floats, n_heads);
        ggml_tensor * heads_v_scales = ggml_new_tensor_2d(
                ctx.get(), GGML_TYPE_F32, layout.v_scale_floats, n_heads);
        ggml_tensor * pending_heads_scratch = ggml_new_tensor_1d(
                ctx.get(), GGML_TYPE_F32, batched_scratch_floats);
        ggml_tensor * pending_heads_store = ggml_kvarn_store_kv_body_pending_heads(
                ctx.get(), pending_heads_k, pending_heads_v, heads_k_body, heads_v_body,
                heads_k_scales, heads_v_scales, pending_heads_scratch,
                n_heads, 0, head_dim, group_size, params.key_bits, params.value_bits,
                params.sinkhorn_iters, params.rtn_quantile);
        require_cuda_scratch_boundary(cuda_dev, pending_heads_store, pending_heads_scratch,
                "CUDA supports exact-minimum KVarN pending-head scratch",
                "CUDA rejects one-short KVarN pending-head scratch");

        // Pending-record sources gather one selected record for one head.
        ggml_tensor * pending_records_k = ggml_new_tensor_3d(
                ctx.get(), GGML_TYPE_F32, head_dim, 1, group_size);
        ggml_tensor * pending_records_v = ggml_new_tensor_3d(
                ctx.get(), GGML_TYPE_F32, head_dim, 1, group_size);
        ggml_tensor * records_k_body = ggml_new_tensor_3d(
                ctx.get(), GGML_TYPE_I8, layout.k_body_bytes, n_records, 1);
        ggml_tensor * records_v_body = ggml_new_tensor_3d(
                ctx.get(), GGML_TYPE_I8, layout.v_body_bytes, n_records, 1);
        ggml_tensor * records_k_scales = ggml_new_tensor_3d(
                ctx.get(), GGML_TYPE_F32, layout.k_scale_floats, n_records, 1);
        ggml_tensor * records_v_scales = ggml_new_tensor_3d(
                ctx.get(), GGML_TYPE_F32, layout.v_scale_floats, n_records, 1);
        ggml_tensor * pending_records_scratch = ggml_new_tensor_1d(
                ctx.get(), GGML_TYPE_F32, batched_scratch_floats);
        const int32_t record = 0;
        ggml_tensor * pending_records_store = ggml_kvarn_store_kv_body_pending_records(
                ctx.get(), pending_records_k, pending_records_v, records_k_body, records_v_body,
                records_k_scales, records_v_scales, pending_records_scratch, &record, 1,
                head_dim, group_size, params.key_bits, params.value_bits,
                params.sinkhorn_iters, params.rtn_quantile);
        require_cuda_scratch_boundary(cuda_dev, pending_records_store, pending_records_scratch,
                "CUDA supports exact-minimum KVarN pending-record scratch",
                "CUDA rejects one-short KVarN pending-record scratch");

        // Direct-record sources batch multiple records and heads in a 4-D tile tensor.
        ggml_tensor * direct_k = ggml_new_tensor_4d(
                ctx.get(), GGML_TYPE_F32, head_dim, n_heads, group_size, n_records);
        ggml_tensor * direct_v = ggml_new_tensor_4d(
                ctx.get(), GGML_TYPE_F32, head_dim, n_heads, group_size, n_records);
        ggml_tensor * direct_k_body = ggml_new_tensor_3d(
                ctx.get(), GGML_TYPE_I8, layout.k_body_bytes, n_records, n_heads);
        ggml_tensor * direct_v_body = ggml_new_tensor_3d(
                ctx.get(), GGML_TYPE_I8, layout.v_body_bytes, n_records, n_heads);
        ggml_tensor * direct_k_scales = ggml_new_tensor_3d(
                ctx.get(), GGML_TYPE_F32, layout.k_scale_floats, n_records, n_heads);
        ggml_tensor * direct_v_scales = ggml_new_tensor_3d(
                ctx.get(), GGML_TYPE_F32, layout.v_scale_floats, n_records, n_heads);
        ggml_tensor * direct_scratch = ggml_new_tensor_1d(
                ctx.get(), GGML_TYPE_F32, batched_scratch_floats);
        ggml_tensor * direct_store = ggml_kvarn_store_kv_body_direct_records(
                ctx.get(), direct_k, direct_v, direct_k_body, direct_v_body,
                direct_k_scales, direct_v_scales, direct_scratch,
                n_heads, 0, n_records, head_dim, group_size, params.key_bits, params.value_bits,
                params.sinkhorn_iters, params.rtn_quantile);
        require_cuda_scratch_boundary(cuda_dev, direct_store, direct_scratch,
                "CUDA supports exact-minimum KVarN direct-record scratch",
                "CUDA rejects one-short KVarN direct-record scratch");
    }
}

static void test_kvarn_mixed_attention_ggml_op() {
    llama_kvarn_params params = llama_kvarn_default_params();
    params.group_size = 4;
    params.sink_tokens = 2;
    params.tail_tokens = 2;
    const int32_t head_dim = 8;
    const int32_t n_head = 4;
    const int32_t n_head_kv = 2;
    const int32_t n_records = 3;
    const int32_t n_pending = 2;
    const int32_t n_tokens = 5;
    const llama_kvarn_layout layout = llama_kvarn_make_layout(params, head_dim);

    ggml_init_params init_params = {
        /*.mem_size   =*/ 128*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx { ggml_init(init_params) };

    ggml_tensor * q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, head_dim, n_head, n_tokens);
    ggml_tensor * sink_tail_k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, head_dim, n_head_kv, params.sink_tokens + params.tail_tokens);
    ggml_tensor * sink_tail_v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, head_dim, n_head_kv, params.sink_tokens + params.tail_tokens);
    ggml_tensor * body_k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I8, layout.k_body_bytes, n_records, n_head_kv);
    ggml_tensor * body_v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I8, layout.v_body_bytes, n_records, n_head_kv);
    ggml_tensor * scales_k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, layout.k_scale_floats, n_records, n_head_kv);
    ggml_tensor * scales_v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, layout.v_scale_floats, n_records, n_head_kv);
    ggml_tensor * pending_k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, head_dim, n_head_kv, params.group_size);
    ggml_tensor * pending_v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, head_dim, n_head_kv, params.group_size);
    ggml_tensor * scratch = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, params.sink_tokens + n_records*params.group_size + n_pending + params.tail_tokens);
    ggml_tensor * kq_mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32,
            params.sink_tokens + n_records*params.group_size + n_pending + params.tail_tokens, n_tokens);
    ggml_tensor * window = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 5);
    ggml_tensor * q_body = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, head_dim, n_head, n_tokens);

    ggml_tensor * out = ggml_kvarn_attn_mixed(
            ctx.get(), q, sink_tail_k, sink_tail_v, body_k, body_v, scales_k, scales_v, pending_k, pending_v, scratch,
            kq_mask,
            params.sink_tokens, n_records, n_pending, params.tail_tokens, 1, head_dim, params.group_size,
            params.key_bits, params.value_bits, 1.0f/std::sqrt(float(head_dim)), 0.0f);

    require(out->op == GGML_OP_KVARN_ATTN_MIXED, "KVarN mixed attention op");
    require(out->type == GGML_TYPE_F32, "KVarN mixed attention result type");
    require(out->ne[0] == head_dim && out->ne[1] == n_head && out->ne[2] == n_tokens,
            "KVarN mixed attention result shape");
    require(out->src[0] == q && out->src[1] == sink_tail_k && out->src[2] == sink_tail_v,
            "KVarN mixed attention primary sources");
    require(out->src[3] == body_k && out->src[4] == body_v && out->src[5] == scales_k &&
            out->src[6] == scales_v && out->src[7] == pending_k && out->src[8] == pending_v &&
            out->src[9] == scratch,
            "KVarN mixed attention cache sources");
    require(out->src[10] == kq_mask, "KVarN mixed attention mask source");
    ggml_kvarn_attn_mixed_set_window(out, window);
    ggml_kvarn_attn_mixed_set_q_body(out, q_body);
    require(out->src[11] == window, "KVarN mixed attention window source");
    require(out->src[12] == q_body, "KVarN mixed attention body-query source");
    require(out->op_params[0] == (int32_t) params.sink_tokens &&
            out->op_params[1] == n_records &&
            out->op_params[2] == n_pending &&
            out->op_params[3] == (int32_t) params.tail_tokens &&
            out->op_params[4] == 1,
            "KVarN mixed attention token params");
    require(out->op_params[5] == head_dim &&
            out->op_params[6] == (int32_t) params.group_size &&
            out->op_params[7] == (int32_t) params.key_bits &&
            out->op_params[8] == (int32_t) params.value_bits,
            "KVarN mixed attention layout params");
    require(out->op_params[11] == GGML_KVARN_ATTN_FRAME_NONE,
            "KVarN mixed attention default frame flags");
    ggml_kvarn_attn_mixed_set_frame_flags(out, GGML_KVARN_ATTN_FRAME_FUSED_PAPER_FULL);
    require(out->op_params[11] == GGML_KVARN_ATTN_FRAME_FUSED_PAPER_FULL,
            "KVarN mixed attention frame flags setter");
}

static void test_cpu_backend_rejects_kvarn_ops() {
    llama_kvarn_params params = llama_kvarn_default_params();
    llama_kvarn_layout layout = llama_kvarn_make_layout(params, 128);

    ggml_init_params init_params = {
        /*.mem_size   =*/ 256*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx { ggml_init(init_params) };

    ggml_tensor * k_tile = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, params.group_size, 128);
    ggml_tensor * k_body = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, layout.k_body_bytes);
    ggml_tensor * k_scales = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, layout.k_scale_floats);
    ggml_tensor * scratch = ggml_new_tensor_1d(
            ctx.get(), GGML_TYPE_F32, store_pipeline_scratch_floats(128, params.group_size));

    ggml_tensor * k_store = ggml_kvarn_store_k_body(
            ctx.get(), k_tile, k_body, k_scales, scratch,
            128, params.group_size, params.key_bits, params.sinkhorn_iters, params.rtn_quantile);

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    require(cpu_dev != nullptr, "CPU backend device is registered");
    ggml_backend_reg_t cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
    require(cpu_reg != nullptr, "CPU backend registry is registered");
    require(std::strcmp(ggml_backend_reg_name(cpu_reg), "CUDA") != 0,
            "CPU backend registry is not CUDA");
    require(!ggml_backend_dev_supports_op(cpu_dev, k_store),
            "CPU backend rejects KVarN body-store op");

    ggml_tensor * q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 128, 1, 1);
    ggml_tensor * sink_tail_k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, 128, 1, params.sink_tokens + params.tail_tokens);
    ggml_tensor * sink_tail_v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, 128, 1, params.sink_tokens + params.tail_tokens);
    ggml_tensor * body_k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I8, layout.k_body_bytes, 1, 1);
    ggml_tensor * body_v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I8, layout.v_body_bytes, 1, 1);
    ggml_tensor * scales_k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, layout.k_scale_floats, 1, 1);
    ggml_tensor * scales_v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, layout.v_scale_floats, 1, 1);
    ggml_tensor * pending_k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 128, 1, params.group_size);
    ggml_tensor * pending_v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 128, 1, params.group_size);
    ggml_tensor * logits = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, params.sink_tokens + params.group_size + params.tail_tokens);
    ggml_tensor * mixed = ggml_kvarn_attn_mixed(
            ctx.get(), q, sink_tail_k, sink_tail_v, body_k, body_v, scales_k, scales_v, pending_k, pending_v, logits,
            nullptr,
            params.sink_tokens, 1, 0, params.tail_tokens, 1, 128, params.group_size,
            params.key_bits, params.value_bits, 1.0f/std::sqrt(128.0f), 0.0f);

    require(!ggml_backend_dev_supports_op(cpu_dev, mixed),
            "CPU backend rejects KVarN mixed-attention op");
}

int main() {
    trace_phase("llama_backend_init", "start");
    llama_backend_init();
    trace_phase("llama_backend_init", "pass");

    run_phase("test_layout", test_layout);
    run_phase("test_iswa_full_normal_policy", test_iswa_full_normal_policy);
    run_phase("test_pack_roundtrip", test_pack_roundtrip);
    run_phase("test_hadamard_inverse", test_hadamard_inverse);
    run_phase("test_reference_store_dequant", test_reference_store_dequant);
    run_phase("test_reference_quantizer_fidelity", test_reference_quantizer_fidelity);
    run_phase("test_reference_store_scale_invariance", test_reference_store_scale_invariance);
    run_phase("test_reference_cache_sealing", test_reference_cache_sealing);
    run_phase("test_shared_kv_reuse_layer_matching_attention_type", test_shared_kv_reuse_layer_matching_attention_type);
    run_phase("test_memory_estimate", test_memory_estimate);
    run_phase("test_runtime_metadata", test_runtime_metadata);
    run_phase("test_runtime_storage_sealing", test_runtime_storage_sealing);
    run_phase("test_runtime_sink_tail_graph_api", test_runtime_sink_tail_graph_api);
    run_phase("test_runtime_body_plan_graph_api", test_runtime_body_plan_graph_api);
    run_phase("test_runtime_body_plan_multi_record_seals", test_runtime_body_plan_multi_record_seals);
    run_phase("test_runtime_body_plan_active_boundary_512", test_runtime_body_plan_active_boundary_512);
    run_phase("test_runtime_stream_consistency", test_runtime_stream_consistency);
    run_phase("test_runtime_state_safety", test_runtime_state_safety);
    run_phase("test_runtime_state_roundtrip", test_runtime_state_roundtrip);
    run_phase("test_runtime_state_rejects_corruption", test_runtime_state_rejects_corruption);
    run_phase("test_runtime_kq_mask_graph_api", test_runtime_kq_mask_graph_api);
    run_phase("test_runtime_body_record_graph_api", test_runtime_body_record_graph_api);
    run_phase("test_kvarn_store_body_ggml_ops", test_kvarn_store_body_ggml_ops);
    run_phase("test_kvarn_store_body_scratch_contracts", test_kvarn_store_body_scratch_contracts);
    run_phase("test_kvarn_mixed_attention_ggml_op", test_kvarn_mixed_attention_ggml_op);
    run_phase("test_cpu_backend_rejects_kvarn_ops", test_cpu_backend_rejects_kvarn_ops);

    trace_phase("llama_backend_free", "start");
    llama_backend_free();
    trace_phase("llama_backend_free", "pass");
    return 0;
}
