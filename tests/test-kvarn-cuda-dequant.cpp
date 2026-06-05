#include "kvarn.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <vector>

static void set_env_var(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    if (value[0] == '\0') {
        unsetenv(name);
    } else {
        setenv(name, value, 1);
    }
#endif
}

static void require(bool ok, const char * msg) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
}

static void require_cuda(cudaError_t err, const char * msg) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "FAIL: %s: %s\n", msg, cudaGetErrorString(err));
        std::exit(1);
    }
}

template <typename T>
static T * cuda_upload(const std::vector<T> & src) {
    T * ptr = nullptr;
    require_cuda(cudaMalloc(&ptr, src.size()*sizeof(T)), "cudaMalloc upload");
    require_cuda(cudaMemcpy(ptr, src.data(), src.size()*sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy upload");
    return ptr;
}

template <typename T>
static void append_vec(std::vector<T> & dst, const std::vector<T> & src) {
    dst.insert(dst.end(), src.begin(), src.end());
}

static uint16_t f32_to_f16_bits(float v) {
    const __half h = __float2half(v);
    return reinterpret_cast<const uint16_t *>(&h)[0];
}

static float f16_bits_to_f32(uint16_t v) {
    __half h;
    reinterpret_cast<uint16_t *>(&h)[0] = v;
    return __half2float(h);
}

struct llama_kvarn_params {
    uint32_t group_size = 128;
    uint32_t key_bits = 4;
    uint32_t value_bits = 2;
    uint32_t sinkhorn_iters = 4;
    float rtn_quantile = 1.0f;
};

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
    std::vector<float> k_scales;
    std::vector<float> v_scales;
};

static llama_kvarn_params llama_kvarn_default_params() {
    return {};
}

static bool is_power_of_2(uint32_t n) {
    return n != 0 && (n & (n - 1)) == 0;
}

static size_t packed_nbytes(size_t n_values, uint32_t bits) {
    return (n_values*bits + 7)/8;
}

static float clamp_quantile(float q) {
    return std::min(1.0f, std::max(0.000001f, q));
}

static llama_kvarn_layout llama_kvarn_make_layout(const llama_kvarn_params & params, uint32_t head_dim) {
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

static void llama_kvarn_hadamard_channels(
        const std::vector<float> & src,
        std::vector<float> & dst,
        uint32_t rows,
        uint32_t cols,
        bool channels_are_rows) {
    const uint32_t n_channels = channels_are_rows ? rows : cols;
    if (!is_power_of_2(n_channels)) {
        throw std::invalid_argument("KVarN Hadamard channel count must be a power of two");
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
            q[r*cols + c] = uint8_t(std::min(qmax, uint32_t(std::lround((v - mn)/s))));
        }
    }
}

static void llama_kvarn_pack_bits(const std::vector<uint8_t> & src, uint32_t bits, std::vector<uint8_t> & dst) {
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

static void llama_kvarn_unpack_bits(
        const std::vector<uint8_t> & src,
        uint32_t bits,
        size_t n_values,
        std::vector<uint8_t> & dst) {
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

static llama_kvarn_body_record llama_kvarn_store_reference(
        const llama_kvarn_params & params,
        uint32_t head_dim,
        const std::vector<float> & k_tile,
        const std::vector<float> & v_tile) {
    const llama_kvarn_layout layout = llama_kvarn_make_layout(params, head_dim);

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

static void llama_kvarn_dequant_reference(
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

static void run_case(uint32_t head_dim) {
    llama_kvarn_params params = llama_kvarn_default_params();
    params.sinkhorn_iters = 4;
    params.rtn_quantile = 0.95f;

    const uint32_t group = params.group_size;
    const size_t n = size_t(head_dim)*group;

    std::vector<float> k_tile(n);
    std::vector<float> v_tile(n);
    for (size_t i = 0; i < n; ++i) {
        k_tile[i] = 0.03f*std::sin(float(i)*0.017f) + 0.02f*std::cos(float(i)*0.005f);
        v_tile[i] = 0.04f*std::cos(float(i)*0.011f) - 0.01f*std::sin(float(i)*0.019f);
    }

    llama_kvarn_body_record record = llama_kvarn_store_reference(params, head_dim, k_tile, v_tile);

    std::vector<float> k_ref;
    std::vector<float> v_ref;
    llama_kvarn_dequant_reference(record, k_ref, v_ref);

    float * k_tile_d = cuda_upload(k_tile);
    float * v_tile_d = cuda_upload(v_tile);

    uint8_t * k_body_store_d = nullptr;
    uint8_t * v_body_store_d = nullptr;
    float * k_scales_store_d = nullptr;
    float * v_scales_store_d = nullptr;
    float * store_scratch_d = nullptr;
    require_cuda(cudaMalloc(&k_body_store_d, record.k_body.size()), "cudaMalloc store K body");
    require_cuda(cudaMalloc(&v_body_store_d, record.v_body.size()), "cudaMalloc store V body");
    require_cuda(cudaMalloc(&k_scales_store_d, record.k_scales.size()*sizeof(float)), "cudaMalloc store K scales");
    require_cuda(cudaMalloc(&v_scales_store_d, record.v_scales.size()*sizeof(float)), "cudaMalloc store V scales");
    require_cuda(cudaMalloc(&store_scratch_d, (n + 2*std::max(head_dim, group))*sizeof(float)), "cudaMalloc store scratch");

    ggml_cuda_kvarn_store_body_reference_minmax(
            k_tile_d, v_tile_d,
            k_body_store_d, v_body_store_d,
            k_scales_store_d, v_scales_store_d,
            store_scratch_d,
            head_dim, group, params.key_bits, params.value_bits,
            params.sinkhorn_iters, params.rtn_quantile,
            nullptr);
    require_cuda(cudaGetLastError(), "KVarN CUDA store-body launch");
    require_cuda(cudaDeviceSynchronize(), "KVarN CUDA store-body sync");

    std::vector<uint8_t> k_body_store(record.k_body.size());
    std::vector<uint8_t> v_body_store(record.v_body.size());
    std::vector<float> k_scales_store(record.k_scales.size());
    std::vector<float> v_scales_store(record.v_scales.size());
    require_cuda(cudaMemcpy(k_body_store.data(), k_body_store_d, k_body_store.size(), cudaMemcpyDeviceToHost), "copy stored K body");
    require_cuda(cudaMemcpy(v_body_store.data(), v_body_store_d, v_body_store.size(), cudaMemcpyDeviceToHost), "copy stored V body");
    require_cuda(cudaMemcpy(k_scales_store.data(), k_scales_store_d, k_scales_store.size()*sizeof(float), cudaMemcpyDeviceToHost), "copy stored K scales");
    require_cuda(cudaMemcpy(v_scales_store.data(), v_scales_store_d, v_scales_store.size()*sizeof(float), cudaMemcpyDeviceToHost), "copy stored V scales");

    size_t packed_diff = 0;
    for (size_t i = 0; i < record.k_body.size(); ++i) {
        packed_diff += record.k_body[i] != k_body_store[i];
    }
    for (size_t i = 0; i < record.v_body.size(); ++i) {
        packed_diff += record.v_body[i] != v_body_store[i];
    }
    require(packed_diff == 0, "CUDA store-body packed bytes match CPU reference");

    float scale_max_err = 0.0f;
    for (size_t i = 0; i < record.k_scales.size(); ++i) {
        scale_max_err = std::max(scale_max_err, std::fabs(record.k_scales[i] - k_scales_store[i]));
    }
    for (size_t i = 0; i < record.v_scales.size(); ++i) {
        scale_max_err = std::max(scale_max_err, std::fabs(record.v_scales[i] - v_scales_store[i]));
    }
    require(scale_max_err < 1.0e-5f, "CUDA store-body scales match CPU reference");

    uint8_t * k_body_d = cuda_upload(record.k_body);
    uint8_t * v_body_d = cuda_upload(record.v_body);
    float * k_scales_d = cuda_upload(record.k_scales);
    float * v_scales_d = cuda_upload(record.v_scales);

    float * k_out_d = nullptr;
    float * v_out_d = nullptr;
    float * scores_d = nullptr;
    float * av_out_d = nullptr;
    float * attn_out_d = nullptr;
    float * attn_probs_d = nullptr;
    require_cuda(cudaMalloc(&k_out_d, n*sizeof(float)), "cudaMalloc K output");
    require_cuda(cudaMalloc(&v_out_d, n*sizeof(float)), "cudaMalloc V output");
    require_cuda(cudaMalloc(&scores_d, group*sizeof(float)), "cudaMalloc scores output");
    require_cuda(cudaMalloc(&av_out_d, head_dim*sizeof(float)), "cudaMalloc AV output");
    require_cuda(cudaMalloc(&attn_out_d, head_dim*sizeof(float)), "cudaMalloc attention output");
    require_cuda(cudaMalloc(&attn_probs_d, group*sizeof(float)), "cudaMalloc attention probs output");

    ggml_cuda_kvarn_dequant_body(
            k_body_d, v_body_d,
            k_scales_d, v_scales_d,
            k_out_d, v_out_d,
            head_dim, group, params.key_bits, params.value_bits,
            nullptr);
    require_cuda(cudaGetLastError(), "KVarN CUDA dequant launch");
    require_cuda(cudaDeviceSynchronize(), "KVarN CUDA dequant sync");

    std::vector<float> k_gpu(n);
    std::vector<float> v_gpu(n);
    require_cuda(cudaMemcpy(k_gpu.data(), k_out_d, n*sizeof(float), cudaMemcpyDeviceToHost), "copy K output");
    require_cuda(cudaMemcpy(v_gpu.data(), v_out_d, n*sizeof(float), cudaMemcpyDeviceToHost), "copy V output");

    float max_err = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        max_err = std::max(max_err, std::fabs(k_ref[i] - k_gpu[i]));
        max_err = std::max(max_err, std::fabs(v_ref[i] - v_gpu[i]));
    }
    require(max_err < 1.0e-6f, "CUDA dequant matches CPU reference");

    std::vector<float> q(head_dim);
    for (uint32_t d = 0; d < head_dim; ++d) {
        q[d] = 0.02f*std::sin(float(d)*0.13f) - 0.03f*std::cos(float(d)*0.07f);
    }

    const float scale = 1.0f/std::sqrt(float(head_dim));
    float * q_d = cuda_upload(q);

    ggml_cuda_kvarn_qk_body(
            q_d, k_body_d, k_scales_d, scores_d,
            head_dim, group, params.key_bits, scale,
            nullptr);
    require_cuda(cudaGetLastError(), "KVarN CUDA QK launch");
    require_cuda(cudaDeviceSynchronize(), "KVarN CUDA QK sync");

    std::vector<float> scores_gpu(group);
    require_cuda(cudaMemcpy(scores_gpu.data(), scores_d, group*sizeof(float), cudaMemcpyDeviceToHost), "copy QK output");

    std::vector<float> scores_ref(group, 0.0f);
    for (uint32_t g = 0; g < group; ++g) {
        for (uint32_t d = 0; d < head_dim; ++d) {
            scores_ref[g] += q[d]*k_ref[d*group + g];
        }
        scores_ref[g] *= scale;
    }

    float qk_max_err = 0.0f;
    for (uint32_t g = 0; g < group; ++g) {
        qk_max_err = std::max(qk_max_err, std::fabs(scores_ref[g] - scores_gpu[g]));
    }
    require(qk_max_err < 1.0e-6f, "CUDA packed-K QK matches CPU reference");

    float max_score = scores_ref[0];
    for (uint32_t g = 1; g < group; ++g) {
        max_score = std::max(max_score, scores_ref[g]);
    }
    std::vector<float> probs(group);
    float denom = 0.0f;
    for (uint32_t g = 0; g < group; ++g) {
        probs[g] = std::exp(scores_ref[g] - max_score);
        denom += probs[g];
    }
    for (float & p : probs) {
        p /= denom;
    }

    float * probs_d = cuda_upload(probs);

    ggml_cuda_kvarn_av_body(
            probs_d, v_body_d, v_scales_d, av_out_d,
            head_dim, group, params.value_bits,
            nullptr);
    require_cuda(cudaGetLastError(), "KVarN CUDA AV launch");
    require_cuda(cudaDeviceSynchronize(), "KVarN CUDA AV sync");

    std::vector<float> av_gpu(head_dim);
    require_cuda(cudaMemcpy(av_gpu.data(), av_out_d, head_dim*sizeof(float), cudaMemcpyDeviceToHost), "copy AV output");

    std::vector<float> av_ref(head_dim, 0.0f);
    for (uint32_t d = 0; d < head_dim; ++d) {
        for (uint32_t g = 0; g < group; ++g) {
            av_ref[d] += probs[g]*v_ref[g*head_dim + d];
        }
    }

    float av_max_err = 0.0f;
    for (uint32_t d = 0; d < head_dim; ++d) {
        av_max_err = std::max(av_max_err, std::fabs(av_ref[d] - av_gpu[d]));
    }
    require(av_max_err < 1.0e-6f, "CUDA packed-V AV matches CPU reference");

    ggml_cuda_kvarn_attn_body(
            q_d, k_body_d, v_body_d,
            k_scales_d, v_scales_d,
            attn_out_d, attn_probs_d,
            head_dim, group, params.key_bits, params.value_bits, scale,
            nullptr);
    require_cuda(cudaGetLastError(), "KVarN CUDA attention body launch");
    require_cuda(cudaDeviceSynchronize(), "KVarN CUDA attention body sync");

    std::vector<float> attn_gpu(head_dim);
    std::vector<float> attn_probs_gpu(group);
    require_cuda(cudaMemcpy(attn_gpu.data(), attn_out_d, head_dim*sizeof(float), cudaMemcpyDeviceToHost), "copy attention output");
    require_cuda(cudaMemcpy(attn_probs_gpu.data(), attn_probs_d, group*sizeof(float), cudaMemcpyDeviceToHost), "copy attention probs");

    float attn_max_err = 0.0f;
    for (uint32_t d = 0; d < head_dim; ++d) {
        attn_max_err = std::max(attn_max_err, std::fabs(av_ref[d] - attn_gpu[d]));
    }
    require(attn_max_err < 1.0e-6f, "CUDA packed-body attention matches CPU reference");

    float probs_max_err = 0.0f;
    for (uint32_t g = 0; g < group; ++g) {
        probs_max_err = std::max(probs_max_err, std::fabs(probs[g] - attn_probs_gpu[g]));
    }
    require(probs_max_err < 1.0e-6f, "CUDA packed-body attention probabilities match CPU reference");

    constexpr uint32_t n_records = 3;
    std::vector<llama_kvarn_body_record> records;
    std::vector<float> multi_k_ref;
    std::vector<float> multi_v_ref;
    std::vector<uint8_t> multi_k_body;
    std::vector<uint8_t> multi_v_body;
    std::vector<float> multi_k_scales;
    std::vector<float> multi_v_scales;

    for (uint32_t r = 0; r < n_records; ++r) {
        std::vector<float> k_rec(n);
        std::vector<float> v_rec(n);
        for (size_t i = 0; i < n; ++i) {
            k_rec[i] = 0.02f*std::sin(float(i + r*17)*0.015f) + 0.03f*std::cos(float(i + r*5)*0.007f);
            v_rec[i] = 0.01f*std::cos(float(i + r*11)*0.019f) - 0.04f*std::sin(float(i + r*3)*0.009f);
        }

        llama_kvarn_body_record rec = llama_kvarn_store_reference(params, head_dim, k_rec, v_rec);
        std::vector<float> k_deq;
        std::vector<float> v_deq;
        llama_kvarn_dequant_reference(rec, k_deq, v_deq);

        append_vec(multi_k_ref, k_deq);
        append_vec(multi_v_ref, v_deq);
        append_vec(multi_k_body, rec.k_body);
        append_vec(multi_v_body, rec.v_body);
        append_vec(multi_k_scales, rec.k_scales);
        append_vec(multi_v_scales, rec.v_scales);
        records.push_back(std::move(rec));
    }

    uint8_t * multi_k_body_d = cuda_upload(multi_k_body);
    uint8_t * multi_v_body_d = cuda_upload(multi_v_body);
    float * multi_k_scales_d = cuda_upload(multi_k_scales);
    float * multi_v_scales_d = cuda_upload(multi_v_scales);

    float * multi_out_d = nullptr;
    float * multi_probs_d = nullptr;
    require_cuda(cudaMalloc(&multi_out_d, head_dim*sizeof(float)), "cudaMalloc multi attention output");
    require_cuda(cudaMalloc(&multi_probs_d, n_records*group*sizeof(float)), "cudaMalloc multi attention probs");

    ggml_cuda_kvarn_attn_body_n(
            q_d,
            multi_k_body_d, multi_v_body_d,
            multi_k_scales_d, multi_v_scales_d,
            multi_out_d, multi_probs_d,
            n_records, head_dim, group,
            params.key_bits, params.value_bits,
            records[0].k_body.size(), records[0].v_body.size(),
            records[0].k_scales.size(), records[0].v_scales.size(),
            scale,
            nullptr);
    require_cuda(cudaGetLastError(), "KVarN CUDA multi-record attention launch");
    require_cuda(cudaDeviceSynchronize(), "KVarN CUDA multi-record attention sync");

    std::vector<float> multi_scores_ref(n_records*group, 0.0f);
    for (uint32_t r = 0; r < n_records; ++r) {
        const size_t rec_off = size_t(r)*n;
        for (uint32_t g = 0; g < group; ++g) {
            float s = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                s += q[d]*multi_k_ref[rec_off + d*group + g];
            }
            multi_scores_ref[size_t(r)*group + g] = s*scale;
        }
    }

    float multi_max_score = multi_scores_ref[0];
    for (float s : multi_scores_ref) {
        multi_max_score = std::max(multi_max_score, s);
    }
    std::vector<float> multi_probs_ref(n_records*group);
    float multi_denom = 0.0f;
    for (size_t i = 0; i < multi_probs_ref.size(); ++i) {
        multi_probs_ref[i] = std::exp(multi_scores_ref[i] - multi_max_score);
        multi_denom += multi_probs_ref[i];
    }
    for (float & p : multi_probs_ref) {
        p /= multi_denom;
    }

    std::vector<float> multi_out_ref(head_dim, 0.0f);
    for (uint32_t r = 0; r < n_records; ++r) {
        const size_t rec_off = size_t(r)*n;
        for (uint32_t d = 0; d < head_dim; ++d) {
            for (uint32_t g = 0; g < group; ++g) {
                multi_out_ref[d] += multi_probs_ref[size_t(r)*group + g]*multi_v_ref[rec_off + g*head_dim + d];
            }
        }
    }

    std::vector<float> multi_out_gpu(head_dim);
    std::vector<float> multi_probs_gpu(n_records*group);
    require_cuda(cudaMemcpy(multi_out_gpu.data(), multi_out_d, head_dim*sizeof(float), cudaMemcpyDeviceToHost), "copy multi attention output");
    require_cuda(cudaMemcpy(multi_probs_gpu.data(), multi_probs_d, n_records*group*sizeof(float), cudaMemcpyDeviceToHost), "copy multi attention probs");

    float multi_out_max_err = 0.0f;
    for (uint32_t d = 0; d < head_dim; ++d) {
        multi_out_max_err = std::max(multi_out_max_err, std::fabs(multi_out_ref[d] - multi_out_gpu[d]));
    }
    require(multi_out_max_err < 1.0e-6f, "CUDA multi-record packed-body attention matches CPU reference");

    float multi_probs_max_err = 0.0f;
    for (size_t i = 0; i < multi_probs_ref.size(); ++i) {
        multi_probs_max_err = std::max(multi_probs_max_err, std::fabs(multi_probs_ref[i] - multi_probs_gpu[i]));
    }
    require(multi_probs_max_err < 1.0e-6f, "CUDA multi-record packed-body probabilities match CPU reference");

    const uint32_t n_queries = head_dim == 256 ? 128 : 3;
    std::vector<float> q_batch(size_t(n_queries)*head_dim);
    for (uint32_t iq = 0; iq < n_queries; ++iq) {
        for (uint32_t d = 0; d < head_dim; ++d) {
            q_batch[size_t(iq)*head_dim + d] =
                0.015f*std::sin(float(d + iq*13)*0.11f) - 0.025f*std::cos(float(d + iq*7)*0.05f);
        }
    }

    float * q_batch_d = cuda_upload(q_batch);
    float * batch_out_d = nullptr;
    float * batch_probs_d = nullptr;
    require_cuda(cudaMalloc(&batch_out_d, size_t(n_queries)*head_dim*sizeof(float)), "cudaMalloc batch attention output");
    require_cuda(cudaMalloc(&batch_probs_d, size_t(n_queries)*n_records*group*sizeof(float)), "cudaMalloc batch attention probs");

    ggml_cuda_kvarn_attn_body_n_batch(
            q_batch_d,
            multi_k_body_d, multi_v_body_d,
            multi_k_scales_d, multi_v_scales_d,
            batch_out_d, batch_probs_d,
            n_queries, n_records, head_dim, group,
            params.key_bits, params.value_bits,
            head_dim, head_dim, n_records*group,
            records[0].k_body.size(), records[0].v_body.size(),
            records[0].k_scales.size(), records[0].v_scales.size(),
            scale,
            nullptr);
    require_cuda(cudaGetLastError(), "KVarN CUDA batched multi-record attention launch");
    require_cuda(cudaDeviceSynchronize(), "KVarN CUDA batched multi-record attention sync");

    std::vector<float> batch_out_gpu(size_t(n_queries)*head_dim);
    std::vector<float> batch_probs_gpu(size_t(n_queries)*n_records*group);
    require_cuda(cudaMemcpy(batch_out_gpu.data(), batch_out_d, batch_out_gpu.size()*sizeof(float), cudaMemcpyDeviceToHost), "copy batched attention output");
    require_cuda(cudaMemcpy(batch_probs_gpu.data(), batch_probs_d, batch_probs_gpu.size()*sizeof(float), cudaMemcpyDeviceToHost), "copy batched attention probs");

    std::vector<float> batch_out_ref(size_t(n_queries)*head_dim, 0.0f);
    std::vector<float> batch_probs_ref(size_t(n_queries)*n_records*group);

    for (uint32_t iq = 0; iq < n_queries; ++iq) {
        std::vector<float> row_scores(n_records*group, 0.0f);
        for (uint32_t r = 0; r < n_records; ++r) {
            const size_t rec_off = size_t(r)*n;
            for (uint32_t g = 0; g < group; ++g) {
                float s = 0.0f;
                for (uint32_t d = 0; d < head_dim; ++d) {
                    s += q_batch[size_t(iq)*head_dim + d]*multi_k_ref[rec_off + d*group + g];
                }
                row_scores[size_t(r)*group + g] = s*scale;
            }
        }

        float row_max = row_scores[0];
        for (float s : row_scores) {
            row_max = std::max(row_max, s);
        }
        float row_denom = 0.0f;
        for (size_t i = 0; i < row_scores.size(); ++i) {
            const float p = std::exp(row_scores[i] - row_max);
            batch_probs_ref[size_t(iq)*n_records*group + i] = p;
            row_denom += p;
        }
        for (size_t i = 0; i < row_scores.size(); ++i) {
            batch_probs_ref[size_t(iq)*n_records*group + i] /= row_denom;
        }

        for (uint32_t r = 0; r < n_records; ++r) {
            const size_t rec_off = size_t(r)*n;
            for (uint32_t d = 0; d < head_dim; ++d) {
                for (uint32_t g = 0; g < group; ++g) {
                    batch_out_ref[size_t(iq)*head_dim + d] +=
                        batch_probs_ref[size_t(iq)*n_records*group + size_t(r)*group + g]*multi_v_ref[rec_off + g*head_dim + d];
                }
            }
        }
    }

    float batch_out_max_err = 0.0f;
    for (size_t i = 0; i < batch_out_ref.size(); ++i) {
        batch_out_max_err = std::max(batch_out_max_err, std::fabs(batch_out_ref[i] - batch_out_gpu[i]));
    }
    require(batch_out_max_err < 1.0e-6f, "CUDA batched multi-record packed-body attention matches CPU reference");

    float batch_probs_max_err = 0.0f;
    for (size_t i = 0; i < batch_probs_ref.size(); ++i) {
        batch_probs_max_err = std::max(batch_probs_max_err, std::fabs(batch_probs_ref[i] - batch_probs_gpu[i]));
    }
    require(batch_probs_max_err < 1.0e-6f, "CUDA batched multi-record packed-body probabilities match CPU reference");

    const uint32_t n_sink = 5;
    const uint32_t n_tail = 7;
    std::vector<float> sink_k(size_t(n_sink)*head_dim);
    std::vector<float> sink_v(size_t(n_sink)*head_dim);
    std::vector<float> tail_k(size_t(n_tail)*head_dim);
    std::vector<float> tail_v(size_t(n_tail)*head_dim);
    for (size_t i = 0; i < sink_k.size(); ++i) {
        sink_k[i] = 0.025f*std::sin(float(i)*0.031f) + 0.01f*std::cos(float(i)*0.013f);
        sink_v[i] = 0.017f*std::cos(float(i)*0.027f) - 0.02f*std::sin(float(i)*0.021f);
    }
    for (size_t i = 0; i < tail_k.size(); ++i) {
        tail_k[i] = 0.022f*std::sin(float(i + 19)*0.029f) - 0.012f*std::cos(float(i + 7)*0.017f);
        tail_v[i] = 0.019f*std::cos(float(i + 11)*0.023f) + 0.015f*std::sin(float(i + 5)*0.019f);
    }

    float * sink_k_d = cuda_upload(sink_k);
    float * sink_v_d = cuda_upload(sink_v);
    float * tail_k_d = cuda_upload(tail_k);
    float * tail_v_d = cuda_upload(tail_v);
    float * mixed_out_d = nullptr;
    float * mixed_probs_d = nullptr;
    const uint32_t n_mixed_tokens = n_sink + n_records*group + n_tail;
    require_cuda(cudaMalloc(&mixed_out_d, head_dim*sizeof(float)), "cudaMalloc mixed attention output");
    require_cuda(cudaMalloc(&mixed_probs_d, n_mixed_tokens*sizeof(float)), "cudaMalloc mixed attention probs");

    ggml_cuda_kvarn_attn_mixed(
            q_d,
            sink_k_d, sink_v_d,
            multi_k_body_d, multi_v_body_d,
            multi_k_scales_d, multi_v_scales_d,
            tail_k_d, tail_v_d,
            mixed_out_d, mixed_probs_d,
            n_sink, n_records, n_tail, head_dim, group,
            params.key_bits, params.value_bits,
            records[0].k_body.size(), records[0].v_body.size(),
            records[0].k_scales.size(), records[0].v_scales.size(),
            scale,
            nullptr);
    require_cuda(cudaGetLastError(), "KVarN CUDA mixed attention launch");
    require_cuda(cudaDeviceSynchronize(), "KVarN CUDA mixed attention sync");

    float * body_k_scratch_d = nullptr;
    float * body_v_scratch_d = nullptr;
    float * mixed_scratch_out_d = nullptr;
    float * mixed_scratch_probs_d = nullptr;
    require_cuda(cudaMalloc(&body_k_scratch_d, size_t(n_records)*n*sizeof(float)), "cudaMalloc scratch K body");
    require_cuda(cudaMalloc(&body_v_scratch_d, size_t(n_records)*n*sizeof(float)), "cudaMalloc scratch V body");
    require_cuda(cudaMalloc(&mixed_scratch_out_d, head_dim*sizeof(float)), "cudaMalloc scratch mixed attention output");
    require_cuda(cudaMalloc(&mixed_scratch_probs_d, n_mixed_tokens*sizeof(float)), "cudaMalloc scratch mixed attention probs");

    ggml_cuda_kvarn_dequant_body_n(
            multi_k_body_d, multi_v_body_d,
            multi_k_scales_d, multi_v_scales_d,
            body_k_scratch_d, body_v_scratch_d,
            n_records, head_dim, group,
            params.key_bits, params.value_bits,
            records[0].k_body.size(), records[0].v_body.size(),
            records[0].k_scales.size(), records[0].v_scales.size(),
            n, n,
            nullptr);
    require_cuda(cudaGetLastError(), "KVarN CUDA multi-record scratch dequant launch");
    require_cuda(cudaDeviceSynchronize(), "KVarN CUDA multi-record scratch dequant sync");

    std::vector<float> body_k_scratch(size_t(n_records)*n);
    std::vector<float> body_v_scratch(size_t(n_records)*n);
    require_cuda(cudaMemcpy(body_k_scratch.data(), body_k_scratch_d, body_k_scratch.size()*sizeof(float), cudaMemcpyDeviceToHost),
            "copy scratch K body");
    require_cuda(cudaMemcpy(body_v_scratch.data(), body_v_scratch_d, body_v_scratch.size()*sizeof(float), cudaMemcpyDeviceToHost),
            "copy scratch V body");

    float scratch_dequant_max_err = 0.0f;
    for (size_t i = 0; i < body_k_scratch.size(); ++i) {
        scratch_dequant_max_err = std::max(scratch_dequant_max_err, std::fabs(multi_k_ref[i] - body_k_scratch[i]));
        scratch_dequant_max_err = std::max(scratch_dequant_max_err, std::fabs(multi_v_ref[i] - body_v_scratch[i]));
    }
    require(scratch_dequant_max_err < 1.0e-6f, "CUDA multi-record scratch dequant matches CPU reference");

    ggml_cuda_kvarn_attn_mixed_f32_scratch(
            q_d,
            sink_k_d, sink_v_d,
            body_k_scratch_d, body_v_scratch_d,
            tail_k_d, tail_v_d,
            mixed_scratch_out_d, mixed_scratch_probs_d,
            n_sink, n_records, n_tail,
            head_dim, group,
            n, n,
            scale,
            nullptr);
    require_cuda(cudaGetLastError(), "KVarN CUDA scratch mixed attention launch");
    require_cuda(cudaDeviceSynchronize(), "KVarN CUDA scratch mixed attention sync");

    std::vector<float> mixed_scores_ref(n_mixed_tokens, 0.0f);
    for (uint32_t t = 0; t < n_sink; ++t) {
        for (uint32_t d = 0; d < head_dim; ++d) {
            mixed_scores_ref[t] += q[d]*sink_k[size_t(t)*head_dim + d];
        }
        mixed_scores_ref[t] *= scale;
    }
    for (uint32_t r = 0; r < n_records; ++r) {
        const size_t rec_off = size_t(r)*n;
        for (uint32_t g = 0; g < group; ++g) {
            float s = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                s += q[d]*multi_k_ref[rec_off + d*group + g];
            }
            mixed_scores_ref[n_sink + size_t(r)*group + g] = s*scale;
        }
    }
    for (uint32_t t = 0; t < n_tail; ++t) {
        const size_t score_i = size_t(n_sink) + n_records*group + t;
        for (uint32_t d = 0; d < head_dim; ++d) {
            mixed_scores_ref[score_i] += q[d]*tail_k[size_t(t)*head_dim + d];
        }
        mixed_scores_ref[score_i] *= scale;
    }

    float mixed_max_score = mixed_scores_ref[0];
    for (float s : mixed_scores_ref) {
        mixed_max_score = std::max(mixed_max_score, s);
    }
    std::vector<float> mixed_probs_ref(n_mixed_tokens);
    float mixed_denom = 0.0f;
    for (uint32_t i = 0; i < n_mixed_tokens; ++i) {
        mixed_probs_ref[i] = std::exp(mixed_scores_ref[i] - mixed_max_score);
        mixed_denom += mixed_probs_ref[i];
    }
    for (float & p : mixed_probs_ref) {
        p /= mixed_denom;
    }

    std::vector<float> mixed_out_ref(head_dim, 0.0f);
    for (uint32_t t = 0; t < n_sink; ++t) {
        for (uint32_t d = 0; d < head_dim; ++d) {
            mixed_out_ref[d] += mixed_probs_ref[t]*sink_v[size_t(t)*head_dim + d];
        }
    }
    for (uint32_t r = 0; r < n_records; ++r) {
        const size_t rec_off = size_t(r)*n;
        for (uint32_t d = 0; d < head_dim; ++d) {
            for (uint32_t g = 0; g < group; ++g) {
                mixed_out_ref[d] += mixed_probs_ref[size_t(n_sink) + size_t(r)*group + g]*multi_v_ref[rec_off + g*head_dim + d];
            }
        }
    }
    for (uint32_t t = 0; t < n_tail; ++t) {
        const size_t prob_i = size_t(n_sink) + n_records*group + t;
        for (uint32_t d = 0; d < head_dim; ++d) {
            mixed_out_ref[d] += mixed_probs_ref[prob_i]*tail_v[size_t(t)*head_dim + d];
        }
    }

    std::vector<float> mixed_out_gpu(head_dim);
    std::vector<float> mixed_probs_gpu(n_mixed_tokens);
    std::vector<float> mixed_scratch_out_gpu(head_dim);
    std::vector<float> mixed_scratch_probs_gpu(n_mixed_tokens);
    require_cuda(cudaMemcpy(mixed_out_gpu.data(), mixed_out_d, head_dim*sizeof(float), cudaMemcpyDeviceToHost), "copy mixed attention output");
    require_cuda(cudaMemcpy(mixed_probs_gpu.data(), mixed_probs_d, n_mixed_tokens*sizeof(float), cudaMemcpyDeviceToHost), "copy mixed attention probs");
    require_cuda(cudaMemcpy(mixed_scratch_out_gpu.data(), mixed_scratch_out_d, head_dim*sizeof(float), cudaMemcpyDeviceToHost),
            "copy scratch mixed attention output");
    require_cuda(cudaMemcpy(mixed_scratch_probs_gpu.data(), mixed_scratch_probs_d, n_mixed_tokens*sizeof(float), cudaMemcpyDeviceToHost),
            "copy scratch mixed attention probs");

    float mixed_out_max_err = 0.0f;
    for (uint32_t d = 0; d < head_dim; ++d) {
        mixed_out_max_err = std::max(mixed_out_max_err, std::fabs(mixed_out_ref[d] - mixed_out_gpu[d]));
    }
    require(mixed_out_max_err < 1.0e-6f, "CUDA mixed sink/body/tail attention matches CPU reference");

    float mixed_scratch_out_max_err = 0.0f;
    for (uint32_t d = 0; d < head_dim; ++d) {
        mixed_scratch_out_max_err = std::max(mixed_scratch_out_max_err, std::fabs(mixed_scratch_out_gpu[d] - mixed_out_gpu[d]));
    }
    require(mixed_scratch_out_max_err < 1.0e-6f, "CUDA packed mixed attention matches scratch-dequant attention");

    float mixed_probs_max_err = 0.0f;
    float mixed_scratch_probs_max_err = 0.0f;
    for (uint32_t i = 0; i < n_mixed_tokens; ++i) {
        mixed_probs_max_err = std::max(mixed_probs_max_err, std::fabs(mixed_probs_ref[i] - mixed_probs_gpu[i]));
        mixed_scratch_probs_max_err = std::max(mixed_scratch_probs_max_err, std::fabs(mixed_scratch_probs_gpu[i] - mixed_probs_gpu[i]));
    }
    require(mixed_probs_max_err < 1.0e-6f, "CUDA mixed sink/body/tail probabilities match CPU reference");
    require(mixed_scratch_probs_max_err < 1.0e-6f, "CUDA packed mixed probabilities match scratch-dequant probabilities");

    const uint32_t n_head_kv = 2;
    const uint32_t n_head = head_dim == 256 ? 16 : 4;
    const uint32_t n_pending_mha = 3;
    std::vector<float> q_mha(size_t(n_queries)*n_head*head_dim);
    for (uint32_t iq = 0; iq < n_queries; ++iq) {
        for (uint32_t ih = 0; ih < n_head; ++ih) {
            for (uint32_t d = 0; d < head_dim; ++d) {
                q_mha[(size_t(iq)*n_head + ih)*head_dim + d] =
                    0.011f*std::sin(float(d + 5*ih + 17*iq)*0.073f) -
                    0.021f*std::cos(float(d + 3*ih + 11*iq)*0.041f);
            }
        }
    }

    const uint32_t n_sink_tail = n_sink + n_tail;
    std::vector<uint16_t> sink_tail_k_f16(size_t(n_sink_tail)*n_head_kv*head_dim);
    std::vector<uint16_t> sink_tail_v_f16(size_t(n_sink_tail)*n_head_kv*head_dim);
    std::vector<float> sink_tail_k_ref(size_t(n_sink_tail)*n_head_kv*head_dim);
    std::vector<float> sink_tail_v_ref(size_t(n_sink_tail)*n_head_kv*head_dim);
    for (uint32_t t = 0; t < n_sink_tail; ++t) {
        for (uint32_t ikh = 0; ikh < n_head_kv; ++ikh) {
            for (uint32_t d = 0; d < head_dim; ++d) {
                const size_t off = (size_t(t)*n_head_kv + ikh)*head_dim + d;
                const float k = 0.018f*std::sin(float(d + 13*t + 7*ikh)*0.037f) +
                                0.009f*std::cos(float(d + 5*t + 3*ikh)*0.019f);
                const float v = 0.014f*std::cos(float(d + 11*t + 2*ikh)*0.031f) -
                                0.016f*std::sin(float(d + 7*t + 5*ikh)*0.023f);
                sink_tail_k_f16[off] = f32_to_f16_bits(k);
                sink_tail_v_f16[off] = f32_to_f16_bits(v);
                sink_tail_k_ref[off] = f16_bits_to_f32(sink_tail_k_f16[off]);
                sink_tail_v_ref[off] = f16_bits_to_f32(sink_tail_v_f16[off]);
            }
        }
    }

    std::vector<float> pending_k_mha(size_t(n_pending_mha)*n_head_kv*head_dim);
    std::vector<float> pending_v_mha(size_t(n_pending_mha)*n_head_kv*head_dim);
    for (uint32_t t = 0; t < n_pending_mha; ++t) {
        for (uint32_t ikh = 0; ikh < n_head_kv; ++ikh) {
            for (uint32_t d = 0; d < head_dim; ++d) {
                const size_t off = (size_t(t)*n_head_kv + ikh)*head_dim + d;
                pending_k_mha[off] = 0.013f*std::sin(float(d + 17*t + 11*ikh)*0.047f) -
                                      0.008f*std::cos(float(d + 3*t + 13*ikh)*0.029f);
                pending_v_mha[off] = 0.015f*std::cos(float(d + 19*t + 7*ikh)*0.043f) +
                                      0.010f*std::sin(float(d + 5*t + 3*ikh)*0.031f);
            }
        }
    }

    std::vector<uint8_t> mha_k_body;
    std::vector<uint8_t> mha_v_body;
    std::vector<float> mha_k_scales;
    std::vector<float> mha_v_scales;
    for (uint32_t ikh = 0; ikh < n_head_kv; ++ikh) {
        (void) ikh;
        append_vec(mha_k_body, multi_k_body);
        append_vec(mha_v_body, multi_v_body);
        append_vec(mha_k_scales, multi_k_scales);
        append_vec(mha_v_scales, multi_v_scales);
    }

    float * q_mha_d = cuda_upload(q_mha);
    uint16_t * sink_tail_k_f16_d = cuda_upload(sink_tail_k_f16);
    uint16_t * sink_tail_v_f16_d = cuda_upload(sink_tail_v_f16);
    uint8_t * mha_k_body_d = cuda_upload(mha_k_body);
    uint8_t * mha_v_body_d = cuda_upload(mha_v_body);
    float * mha_k_scales_d = cuda_upload(mha_k_scales);
    float * mha_v_scales_d = cuda_upload(mha_v_scales);
    float * pending_k_mha_d = cuda_upload(pending_k_mha);
    float * pending_v_mha_d = cuda_upload(pending_v_mha);
    float * mha_mixed_out_d = nullptr;
    float * mha_mixed_fused_out_d = nullptr;
    float * mha_mixed_scores_d = nullptr;
    float * mha_mixed_fused_scores_d = nullptr;
    const uint32_t n_mha_mixed_tokens = n_sink + n_records*group + n_pending_mha + n_tail;
    const uint32_t tail_start = 1;
    require_cuda(cudaMalloc(&mha_mixed_out_d, size_t(n_queries)*n_head*head_dim*sizeof(float)), "cudaMalloc MHA mixed output");
    require_cuda(cudaMalloc(&mha_mixed_fused_out_d, size_t(n_queries)*n_head*head_dim*sizeof(float)), "cudaMalloc MHA fused mixed output");
    require_cuda(cudaMalloc(&mha_mixed_scores_d, n_mha_mixed_tokens*sizeof(float)), "cudaMalloc MHA mixed scores");
    require_cuda(cudaMalloc(&mha_mixed_fused_scores_d, n_mha_mixed_tokens*sizeof(float)), "cudaMalloc MHA fused mixed scores");

    const uint32_t mha_mask_stride_tokens = head_dim == 256 ? 1024 : n_mha_mixed_tokens;
    std::vector<uint16_t> mha_mask_f16(size_t(n_queries)*mha_mask_stride_tokens, f32_to_f16_bits(-1.0e30f));
    std::vector<float> mha_mask_ref(size_t(n_queries)*mha_mask_stride_tokens, -1.0e30f);
    for (uint32_t iq = 0; iq < n_queries; ++iq) {
        for (uint32_t t = 0; t < mha_mask_stride_tokens; ++t) {
            const float bias = t > n_sink + n_records*group + iq ? -25.0f : 0.0f;
            const size_t off = size_t(iq)*mha_mask_stride_tokens + t;
            mha_mask_f16[off] = f32_to_f16_bits(bias);
            mha_mask_ref[off] = f16_bits_to_f32(mha_mask_f16[off]);
        }
    }
    uint16_t * mha_mask_f16_d = cuda_upload(mha_mask_f16);

    ggml_cuda_kvarn_attn_mixed_f16_batch(
            q_mha_d,
            sink_tail_k_f16_d, sink_tail_v_f16_d,
            mha_k_body_d, mha_v_body_d,
            mha_k_scales_d, mha_v_scales_d,
            pending_k_mha_d, pending_v_mha_d,
            mha_mask_f16_d,
            mha_mixed_out_d, mha_mixed_scores_d,
            n_queries, n_head, n_head_kv,
            n_sink, n_records, n_pending_mha, n_tail, tail_start, head_dim, group,
            params.key_bits, params.value_bits,
            head_dim, size_t(n_head)*head_dim,
            head_dim, size_t(n_head)*head_dim,
            head_dim, size_t(n_head_kv)*head_dim,
            head_dim, size_t(n_head_kv)*head_dim,
            records[0].k_body.size(), records[0].v_body.size(),
            size_t(n_records)*records[0].k_body.size(), size_t(n_records)*records[0].v_body.size(),
            records[0].k_scales.size(), records[0].v_scales.size(),
            size_t(n_records)*records[0].k_scales.size(), size_t(n_records)*records[0].v_scales.size(),
            size_t(mha_mask_stride_tokens)*sizeof(uint16_t), sizeof(uint16_t), 2,
            scale,
            nullptr);
    require_cuda(cudaGetLastError(), "KVarN CUDA batched F16 mixed attention launch");
    require_cuda(cudaDeviceSynchronize(), "KVarN CUDA batched F16 mixed attention sync");

    set_env_var("LLAMA_KVARN_ATTN_FUSED_BATCH", "1");
    ggml_cuda_kvarn_attn_mixed_f16_batch(
            q_mha_d,
            sink_tail_k_f16_d, sink_tail_v_f16_d,
            mha_k_body_d, mha_v_body_d,
            mha_k_scales_d, mha_v_scales_d,
            pending_k_mha_d, pending_v_mha_d,
            mha_mask_f16_d,
            mha_mixed_fused_out_d, mha_mixed_fused_scores_d,
            n_queries, n_head, n_head_kv,
            n_sink, n_records, n_pending_mha, n_tail, tail_start, head_dim, group,
            params.key_bits, params.value_bits,
            head_dim, size_t(n_head)*head_dim,
            head_dim, size_t(n_head)*head_dim,
            head_dim, size_t(n_head_kv)*head_dim,
            head_dim, size_t(n_head_kv)*head_dim,
            records[0].k_body.size(), records[0].v_body.size(),
            size_t(n_records)*records[0].k_body.size(), size_t(n_records)*records[0].v_body.size(),
            records[0].k_scales.size(), records[0].v_scales.size(),
            size_t(n_records)*records[0].k_scales.size(), size_t(n_records)*records[0].v_scales.size(),
            size_t(mha_mask_stride_tokens)*sizeof(uint16_t), sizeof(uint16_t), 2,
            scale,
            nullptr);
    set_env_var("LLAMA_KVARN_ATTN_FUSED_BATCH", "");
    require_cuda(cudaGetLastError(), "KVarN CUDA forced fused batched F16 mixed attention launch");
    require_cuda(cudaDeviceSynchronize(), "KVarN CUDA forced fused batched F16 mixed attention sync");

    std::vector<float> mha_mixed_out_gpu(size_t(n_queries)*n_head*head_dim);
    std::vector<float> mha_mixed_fused_out_gpu(size_t(n_queries)*n_head*head_dim);
    require_cuda(cudaMemcpy(mha_mixed_out_gpu.data(), mha_mixed_out_d, mha_mixed_out_gpu.size()*sizeof(float), cudaMemcpyDeviceToHost),
            "copy MHA mixed attention output");
    require_cuda(cudaMemcpy(mha_mixed_fused_out_gpu.data(), mha_mixed_fused_out_d, mha_mixed_fused_out_gpu.size()*sizeof(float), cudaMemcpyDeviceToHost),
            "copy forced fused MHA mixed attention output");

    std::vector<float> mha_mixed_out_ref(size_t(n_queries)*n_head*head_dim, 0.0f);
    for (uint32_t iq = 0; iq < n_queries; ++iq) {
        for (uint32_t ih = 0; ih < n_head; ++ih) {
            const uint32_t ikh = ih/(n_head/n_head_kv);
            std::vector<float> row_scores(n_mha_mixed_tokens, 0.0f);
            const float * q_row = q_mha.data() + (size_t(iq)*n_head + ih)*head_dim;

            for (uint32_t t = 0; t < n_sink; ++t) {
                for (uint32_t d = 0; d < head_dim; ++d) {
                    row_scores[t] += q_row[d]*sink_tail_k_ref[(size_t(t)*n_head_kv + ikh)*head_dim + d];
                }
                row_scores[t] *= scale;
            }
            for (uint32_t r = 0; r < n_records; ++r) {
                const size_t rec_off = size_t(r)*n;
                for (uint32_t g = 0; g < group; ++g) {
                    float s = 0.0f;
                    for (uint32_t d = 0; d < head_dim; ++d) {
                        s += q_row[d]*multi_k_ref[rec_off + d*group + g];
                    }
                    row_scores[size_t(n_sink) + size_t(r)*group + g] = s*scale;
                }
            }
            for (uint32_t t = 0; t < n_pending_mha; ++t) {
                const size_t score_i = size_t(n_sink) + n_records*group + t;
                for (uint32_t d = 0; d < head_dim; ++d) {
                    row_scores[score_i] += q_row[d]*pending_k_mha[(size_t(t)*n_head_kv + ikh)*head_dim + d];
                }
                row_scores[score_i] *= scale;
            }
            for (uint32_t t = 0; t < n_tail; ++t) {
                const size_t score_i = size_t(n_sink) + n_records*group + n_pending_mha + t;
                const uint32_t slot = n_sink + (tail_start + t)%n_tail;
                for (uint32_t d = 0; d < head_dim; ++d) {
                    row_scores[score_i] += q_row[d]*sink_tail_k_ref[(size_t(slot)*n_head_kv + ikh)*head_dim + d];
                }
                row_scores[score_i] *= scale;
            }
            for (uint32_t t = 0; t < n_mha_mixed_tokens; ++t) {
                row_scores[t] += mha_mask_ref[size_t(iq)*mha_mask_stride_tokens + t];
            }

            float row_max = row_scores[0];
            for (float s : row_scores) {
                row_max = std::max(row_max, s);
            }
            std::vector<float> row_probs(n_mha_mixed_tokens);
            float row_denom = 0.0f;
            for (uint32_t i = 0; i < n_mha_mixed_tokens; ++i) {
                row_probs[i] = std::exp(row_scores[i] - row_max);
                row_denom += row_probs[i];
            }
            for (float & p : row_probs) {
                p /= row_denom;
            }

            float * out_row = mha_mixed_out_ref.data() + (size_t(iq)*n_head + ih)*head_dim;
            for (uint32_t t = 0; t < n_sink; ++t) {
                for (uint32_t d = 0; d < head_dim; ++d) {
                    out_row[d] += row_probs[t]*sink_tail_v_ref[(size_t(t)*n_head_kv + ikh)*head_dim + d];
                }
            }
            for (uint32_t r = 0; r < n_records; ++r) {
                const size_t rec_off = size_t(r)*n;
                for (uint32_t d = 0; d < head_dim; ++d) {
                    for (uint32_t g = 0; g < group; ++g) {
                        out_row[d] += row_probs[size_t(n_sink) + size_t(r)*group + g]*multi_v_ref[rec_off + g*head_dim + d];
                    }
                }
            }
            for (uint32_t t = 0; t < n_pending_mha; ++t) {
                const size_t prob_i = size_t(n_sink) + n_records*group + t;
                for (uint32_t d = 0; d < head_dim; ++d) {
                    out_row[d] += row_probs[prob_i]*pending_v_mha[(size_t(t)*n_head_kv + ikh)*head_dim + d];
                }
            }
            for (uint32_t t = 0; t < n_tail; ++t) {
                const size_t prob_i = size_t(n_sink) + n_records*group + n_pending_mha + t;
                const uint32_t slot = n_sink + (tail_start + t)%n_tail;
                for (uint32_t d = 0; d < head_dim; ++d) {
                    out_row[d] += row_probs[prob_i]*sink_tail_v_ref[(size_t(slot)*n_head_kv + ikh)*head_dim + d];
                }
            }
        }
    }

    float mha_mixed_max_err = 0.0f;
    for (size_t i = 0; i < mha_mixed_out_ref.size(); ++i) {
        mha_mixed_max_err = std::max(mha_mixed_max_err, std::fabs(mha_mixed_out_ref[i] - mha_mixed_out_gpu[i]));
    }
    require(mha_mixed_max_err < 1.0e-5f, "CUDA batched F16 sink/body/tail mixed attention matches CPU reference");

    float mha_mixed_fused_max_err = 0.0f;
    for (size_t i = 0; i < mha_mixed_out_ref.size(); ++i) {
        mha_mixed_fused_max_err = std::max(mha_mixed_fused_max_err, std::fabs(mha_mixed_out_ref[i] - mha_mixed_fused_out_gpu[i]));
    }
    require(mha_mixed_fused_max_err < 1.0e-5f, "CUDA forced fused batched F16 sink/body/tail mixed attention matches CPU reference");

    if (head_dim == 256) {
        const uint32_t n_sink_only_queries = 49;
        const uint32_t n_sink_only = 49;
        const uint32_t sink_only_mask_stride_tokens = 512;
        std::vector<float> q_sink_only(size_t(n_sink_only_queries)*n_head*head_dim);
        std::vector<float> k_sink_only(size_t(n_sink_only)*n_head_kv*head_dim);
        std::vector<float> v_sink_only(size_t(n_sink_only)*n_head_kv*head_dim);
        for (uint32_t iq = 0; iq < n_sink_only_queries; ++iq) {
            for (uint32_t ih = 0; ih < n_head; ++ih) {
                for (uint32_t d = 0; d < head_dim; ++d) {
                    q_sink_only[(size_t(iq)*n_head + ih)*head_dim + d] =
                        0.180f*std::sin(float(d + 3*ih + 7*iq)*0.019f) -
                        0.110f*std::cos(float(5*d + ih + 11*iq)*0.013f);
                }
            }
        }
        for (uint32_t t = 0; t < n_sink_only; ++t) {
            for (uint32_t ikh = 0; ikh < n_head_kv; ++ikh) {
                for (uint32_t d = 0; d < head_dim; ++d) {
                    const size_t off = (size_t(t)*n_head_kv + ikh)*head_dim + d;
                    k_sink_only[off] = 0.120f*std::sin(float(d + 13*t + 5*ikh)*0.021f);
                    v_sink_only[off] = 0.140f*std::cos(float(3*d + 17*t + 7*ikh)*0.017f);
                }
            }
        }

        std::vector<uint16_t> k_sink_only_f16(k_sink_only.size());
        std::vector<uint16_t> v_sink_only_f16(v_sink_only.size());
        for (size_t i = 0; i < k_sink_only.size(); ++i) {
            k_sink_only_f16[i] = f32_to_f16_bits(k_sink_only[i]);
            v_sink_only_f16[i] = f32_to_f16_bits(v_sink_only[i]);
            k_sink_only[i] = f16_bits_to_f32(k_sink_only_f16[i]);
            v_sink_only[i] = f16_bits_to_f32(v_sink_only_f16[i]);
        }

        std::vector<uint16_t> sink_only_mask(size_t(n_sink_only_queries)*sink_only_mask_stride_tokens, f32_to_f16_bits(-1.0e30f));
        std::vector<float> sink_only_mask_ref(size_t(n_sink_only_queries)*sink_only_mask_stride_tokens);
        for (uint32_t iq = 0; iq < n_sink_only_queries; ++iq) {
            for (uint32_t t = 0; t < sink_only_mask_stride_tokens; ++t) {
                const float bias = t <= iq ? 0.0f : -1.0e30f;
                const size_t off = size_t(iq)*sink_only_mask_stride_tokens + t;
                sink_only_mask[off] = f32_to_f16_bits(bias);
                sink_only_mask_ref[off] = f16_bits_to_f32(sink_only_mask[off]);
            }
        }

        float * q_sink_only_d = cuda_upload(q_sink_only);
        uint16_t * k_sink_only_d = cuda_upload(k_sink_only_f16);
        uint16_t * v_sink_only_d = cuda_upload(v_sink_only_f16);
        uint16_t * sink_only_mask_d = cuda_upload(sink_only_mask);
        float * sink_only_split_d = nullptr;
        float * sink_only_fused_d = nullptr;
        float * sink_only_scores_d = nullptr;
        float * sink_only_fused_scores_d = nullptr;
        require_cuda(cudaMalloc(&sink_only_split_d, q_sink_only.size()*sizeof(float)), "cudaMalloc sink-only split output");
        require_cuda(cudaMalloc(&sink_only_fused_d, q_sink_only.size()*sizeof(float)), "cudaMalloc sink-only fused output");
        require_cuda(cudaMalloc(&sink_only_scores_d, n_sink_only*sizeof(float)), "cudaMalloc sink-only scores");
        require_cuda(cudaMalloc(&sink_only_fused_scores_d, n_sink_only*sizeof(float)), "cudaMalloc sink-only fused scores");

        ggml_cuda_kvarn_attn_mixed_f16_batch(
                q_sink_only_d, k_sink_only_d, v_sink_only_d,
                mha_k_body_d, mha_v_body_d, mha_k_scales_d, mha_v_scales_d, pending_k_mha_d, pending_v_mha_d,
                sink_only_mask_d,
                sink_only_split_d, sink_only_scores_d,
                n_sink_only_queries, n_head, n_head_kv,
                n_sink_only, 0, 0, 0, 0, head_dim, group,
                params.key_bits, params.value_bits,
                head_dim, size_t(n_head)*head_dim,
                head_dim, size_t(n_head)*head_dim,
                head_dim, size_t(n_head_kv)*head_dim,
                head_dim, size_t(n_head_kv)*head_dim,
                records[0].k_body.size(), records[0].v_body.size(),
                size_t(n_records)*records[0].k_body.size(), size_t(n_records)*records[0].v_body.size(),
                records[0].k_scales.size(), records[0].v_scales.size(),
                size_t(n_records)*records[0].k_scales.size(), size_t(n_records)*records[0].v_scales.size(),
                size_t(sink_only_mask_stride_tokens)*sizeof(uint16_t), sizeof(uint16_t), 2,
                scale,
                nullptr);
        require_cuda(cudaGetLastError(), "KVarN CUDA split sink-only launch");
        require_cuda(cudaDeviceSynchronize(), "KVarN CUDA split sink-only sync");

        set_env_var("LLAMA_KVARN_ATTN_FUSED_BATCH", "1");
        ggml_cuda_kvarn_attn_mixed_f16_batch(
                q_sink_only_d, k_sink_only_d, v_sink_only_d,
                mha_k_body_d, mha_v_body_d, mha_k_scales_d, mha_v_scales_d, pending_k_mha_d, pending_v_mha_d,
                sink_only_mask_d,
                sink_only_fused_d, sink_only_fused_scores_d,
                n_sink_only_queries, n_head, n_head_kv,
                n_sink_only, 0, 0, 0, 0, head_dim, group,
                params.key_bits, params.value_bits,
                head_dim, size_t(n_head)*head_dim,
                head_dim, size_t(n_head)*head_dim,
                head_dim, size_t(n_head_kv)*head_dim,
                head_dim, size_t(n_head_kv)*head_dim,
                records[0].k_body.size(), records[0].v_body.size(),
                size_t(n_records)*records[0].k_body.size(), size_t(n_records)*records[0].v_body.size(),
                records[0].k_scales.size(), records[0].v_scales.size(),
                size_t(n_records)*records[0].k_scales.size(), size_t(n_records)*records[0].v_scales.size(),
                size_t(sink_only_mask_stride_tokens)*sizeof(uint16_t), sizeof(uint16_t), 2,
                scale,
                nullptr);
        set_env_var("LLAMA_KVARN_ATTN_FUSED_BATCH", "");
        require_cuda(cudaGetLastError(), "KVarN CUDA fused sink-only launch");
        require_cuda(cudaDeviceSynchronize(), "KVarN CUDA fused sink-only sync");

        std::vector<float> sink_only_split(q_sink_only.size());
        std::vector<float> sink_only_fused(q_sink_only.size());
        require_cuda(cudaMemcpy(sink_only_split.data(), sink_only_split_d, sink_only_split.size()*sizeof(float), cudaMemcpyDeviceToHost),
                "copy split sink-only output");
        require_cuda(cudaMemcpy(sink_only_fused.data(), sink_only_fused_d, sink_only_fused.size()*sizeof(float), cudaMemcpyDeviceToHost),
                "copy fused sink-only output");

        std::vector<float> sink_only_ref(q_sink_only.size(), 0.0f);
        for (uint32_t iq = 0; iq < n_sink_only_queries; ++iq) {
            for (uint32_t ih = 0; ih < n_head; ++ih) {
                const uint32_t ikh = ih/(n_head/n_head_kv);
                const float * q_row = q_sink_only.data() + (size_t(iq)*n_head + ih)*head_dim;
                std::vector<float> row_scores(n_sink_only, 0.0f);
                for (uint32_t t = 0; t < n_sink_only; ++t) {
                    for (uint32_t d = 0; d < head_dim; ++d) {
                        row_scores[t] += q_row[d]*k_sink_only[(size_t(t)*n_head_kv + ikh)*head_dim + d];
                    }
                    row_scores[t] = row_scores[t]*scale + sink_only_mask_ref[size_t(iq)*sink_only_mask_stride_tokens + t];
                }
                float row_max = row_scores[0];
                for (float s : row_scores) {
                    row_max = std::max(row_max, s);
                }
                std::vector<float> row_probs(n_sink_only);
                float row_denom = 0.0f;
                for (uint32_t t = 0; t < n_sink_only; ++t) {
                    row_probs[t] = std::exp(row_scores[t] - row_max);
                    row_denom += row_probs[t];
                }
                float * out_row = sink_only_ref.data() + (size_t(iq)*n_head + ih)*head_dim;
                for (uint32_t t = 0; t < n_sink_only; ++t) {
                    const float p = row_probs[t]/row_denom;
                    for (uint32_t d = 0; d < head_dim; ++d) {
                        out_row[d] += p*v_sink_only[(size_t(t)*n_head_kv + ikh)*head_dim + d];
                    }
                }
            }
        }

        float sink_only_split_err = 0.0f;
        float sink_only_fused_err = 0.0f;
        for (size_t i = 0; i < sink_only_ref.size(); ++i) {
            sink_only_split_err = std::max(sink_only_split_err, std::fabs(sink_only_ref[i] - sink_only_split[i]));
            sink_only_fused_err = std::max(sink_only_fused_err, std::fabs(sink_only_ref[i] - sink_only_fused[i]));
        }
        require(sink_only_split_err < 1.0e-5f, "CUDA split sink-only causal attention matches CPU reference with F16 mask");
        require(sink_only_fused_err < 1.0e-5f, "CUDA fused sink-only causal attention matches CPU reference with F16 mask");

        std::vector<float> sink_only_mask_f32(size_t(n_sink_only_queries)*sink_only_mask_stride_tokens);
        for (size_t i = 0; i < sink_only_mask_f32.size(); ++i) {
            sink_only_mask_f32[i] = sink_only_mask_ref[i];
        }
        float * sink_only_mask_f32_d = cuda_upload(sink_only_mask_f32);

        ggml_cuda_kvarn_attn_mixed_f16_batch(
                q_sink_only_d, k_sink_only_d, v_sink_only_d,
                mha_k_body_d, mha_v_body_d, mha_k_scales_d, mha_v_scales_d, pending_k_mha_d, pending_v_mha_d,
                sink_only_mask_f32_d,
                sink_only_split_d, sink_only_scores_d,
                n_sink_only_queries, n_head, n_head_kv,
                n_sink_only, 0, 0, 0, 0, head_dim, group,
                params.key_bits, params.value_bits,
                head_dim, size_t(n_head)*head_dim,
                head_dim, size_t(n_head)*head_dim,
                head_dim, size_t(n_head_kv)*head_dim,
                head_dim, size_t(n_head_kv)*head_dim,
                records[0].k_body.size(), records[0].v_body.size(),
                size_t(n_records)*records[0].k_body.size(), size_t(n_records)*records[0].v_body.size(),
                records[0].k_scales.size(), records[0].v_scales.size(),
                size_t(n_records)*records[0].k_scales.size(), size_t(n_records)*records[0].v_scales.size(),
                size_t(sink_only_mask_stride_tokens)*sizeof(float), sizeof(float), 1,
                scale,
                nullptr);
        require_cuda(cudaGetLastError(), "KVarN CUDA split sink-only F32 mask launch");
        require_cuda(cudaDeviceSynchronize(), "KVarN CUDA split sink-only F32 mask sync");

        set_env_var("LLAMA_KVARN_ATTN_FUSED_BATCH", "1");
        ggml_cuda_kvarn_attn_mixed_f16_batch(
                q_sink_only_d, k_sink_only_d, v_sink_only_d,
                mha_k_body_d, mha_v_body_d, mha_k_scales_d, mha_v_scales_d, pending_k_mha_d, pending_v_mha_d,
                sink_only_mask_f32_d,
                sink_only_fused_d, sink_only_fused_scores_d,
                n_sink_only_queries, n_head, n_head_kv,
                n_sink_only, 0, 0, 0, 0, head_dim, group,
                params.key_bits, params.value_bits,
                head_dim, size_t(n_head)*head_dim,
                head_dim, size_t(n_head)*head_dim,
                head_dim, size_t(n_head_kv)*head_dim,
                head_dim, size_t(n_head_kv)*head_dim,
                records[0].k_body.size(), records[0].v_body.size(),
                size_t(n_records)*records[0].k_body.size(), size_t(n_records)*records[0].v_body.size(),
                records[0].k_scales.size(), records[0].v_scales.size(),
                size_t(n_records)*records[0].k_scales.size(), size_t(n_records)*records[0].v_scales.size(),
                size_t(sink_only_mask_stride_tokens)*sizeof(float), sizeof(float), 1,
                scale,
                nullptr);
        set_env_var("LLAMA_KVARN_ATTN_FUSED_BATCH", "");
        require_cuda(cudaGetLastError(), "KVarN CUDA fused sink-only F32 mask launch");
        require_cuda(cudaDeviceSynchronize(), "KVarN CUDA fused sink-only F32 mask sync");

        require_cuda(cudaMemcpy(sink_only_split.data(), sink_only_split_d, sink_only_split.size()*sizeof(float), cudaMemcpyDeviceToHost),
                "copy split sink-only F32 mask output");
        require_cuda(cudaMemcpy(sink_only_fused.data(), sink_only_fused_d, sink_only_fused.size()*sizeof(float), cudaMemcpyDeviceToHost),
                "copy fused sink-only F32 mask output");

        float sink_only_f32_mask_split_err = 0.0f;
        float sink_only_f32_mask_fused_err = 0.0f;
        for (size_t i = 0; i < sink_only_ref.size(); ++i) {
            sink_only_f32_mask_split_err = std::max(sink_only_f32_mask_split_err, std::fabs(sink_only_ref[i] - sink_only_split[i]));
            sink_only_f32_mask_fused_err = std::max(sink_only_f32_mask_fused_err, std::fabs(sink_only_ref[i] - sink_only_fused[i]));
        }
        require(sink_only_f32_mask_split_err < 1.0e-5f, "CUDA split sink-only causal attention matches CPU reference with F32 mask");
        require(sink_only_f32_mask_fused_err < 1.0e-5f, "CUDA fused sink-only causal attention matches CPU reference with F32 mask");

        cudaFree(q_sink_only_d);
        cudaFree(k_sink_only_d);
        cudaFree(v_sink_only_d);
        cudaFree(sink_only_mask_d);
        cudaFree(sink_only_mask_f32_d);
        cudaFree(sink_only_split_d);
        cudaFree(sink_only_fused_d);
        cudaFree(sink_only_scores_d);
        cudaFree(sink_only_fused_scores_d);
    }

    cudaFree(k_body_d);
    cudaFree(v_body_d);
    cudaFree(k_scales_d);
    cudaFree(v_scales_d);
    cudaFree(k_out_d);
    cudaFree(v_out_d);
    cudaFree(scores_d);
    cudaFree(av_out_d);
    cudaFree(attn_out_d);
    cudaFree(attn_probs_d);
    cudaFree(q_d);
    cudaFree(probs_d);
    cudaFree(multi_k_body_d);
    cudaFree(multi_v_body_d);
    cudaFree(multi_k_scales_d);
    cudaFree(multi_v_scales_d);
    cudaFree(multi_out_d);
    cudaFree(multi_probs_d);
    cudaFree(q_batch_d);
    cudaFree(batch_out_d);
    cudaFree(batch_probs_d);
    cudaFree(sink_k_d);
    cudaFree(sink_v_d);
    cudaFree(tail_k_d);
    cudaFree(tail_v_d);
    cudaFree(mixed_out_d);
    cudaFree(mixed_probs_d);
    cudaFree(body_k_scratch_d);
    cudaFree(body_v_scratch_d);
    cudaFree(mixed_scratch_out_d);
    cudaFree(mixed_scratch_probs_d);
    cudaFree(q_mha_d);
    cudaFree(sink_tail_k_f16_d);
    cudaFree(sink_tail_v_f16_d);
    cudaFree(mha_k_body_d);
    cudaFree(mha_v_body_d);
    cudaFree(mha_k_scales_d);
    cudaFree(mha_v_scales_d);
    cudaFree(pending_k_mha_d);
    cudaFree(pending_v_mha_d);
    cudaFree(mha_mixed_out_d);
    cudaFree(mha_mixed_fused_out_d);
    cudaFree(mha_mixed_scores_d);
    cudaFree(mha_mixed_fused_scores_d);
    cudaFree(mha_mask_f16_d);

    cudaFree(k_tile_d);
    cudaFree(v_tile_d);
    cudaFree(k_body_store_d);
    cudaFree(v_body_store_d);
    cudaFree(k_scales_store_d);
    cudaFree(v_scales_store_d);
    cudaFree(store_scratch_d);
}

int main() {
    run_case(128);
    run_case(256);
    return 0;
}
