#include "kvarn.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

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

static uint16_t f32_to_f16_bits(float v) {
    const __half h = __float2half(v);
    return reinterpret_cast<const uint16_t *>(&h)[0];
}

static float f16_bits_to_f32(uint16_t v) {
    __half h;
    *reinterpret_cast<uint16_t *>(&h) = v;
    return __half2float(h);
}

static std::vector<uint8_t> pack_bits(const std::vector<uint8_t> & src, uint32_t bits) {
    std::vector<uint8_t> dst((src.size()*bits + 7)/8, 0);
    const uint32_t mask = (1u << bits) - 1u;
    size_t bit_pos = 0;
    for (uint8_t v : src) {
        const uint32_t q = uint32_t(v)&mask;
        const size_t byte_pos = bit_pos >> 3;
        const uint32_t shift = uint32_t(bit_pos&7);
        dst[byte_pos] |= uint8_t(q << shift);
        if (shift + bits > 8) {
            dst[byte_pos + 1] |= uint8_t(q >> (8 - shift));
        }
        bit_pos += bits;
    }
    return dst;
}

int main() {
    constexpr uint32_t n_queries = 1;
    constexpr uint32_t n_head = 1;
    constexpr uint32_t n_head_kv = 1;
    constexpr uint32_t n_sink = 2;
    constexpr uint32_t n_records = 0;
    constexpr uint32_t n_pending = 0;
    constexpr uint32_t n_tail = 3;
    constexpr uint32_t tail_start = 1;
    constexpr uint32_t head_dim = 4;
    constexpr uint32_t group_size = 128;
    constexpr uint32_t key_bits = 4;
    constexpr uint32_t value_bits = 2;
    constexpr float scale = 0.5f;

    const uint32_t n_tokens = n_sink + n_tail;
    std::vector<float> q = {
        0.25f, -0.50f, 0.75f, 1.00f,
    };

    // Physical slots are sink0, sink1, tail0, tail1, tail2. With tail_start=1,
    // logical chronological tail order is tail1, tail2, tail0.
    std::vector<float> sink_tail_k_ref = {
         0.10f,  0.20f, -0.10f,  0.30f,
        -0.25f,  0.40f,  0.15f, -0.35f,
         0.75f, -0.10f,  0.30f,  0.20f,
        -0.45f,  0.60f,  0.25f,  0.10f,
         0.15f, -0.55f,  0.70f, -0.20f,
    };
    std::vector<float> sink_tail_v_ref = {
         0.30f, -0.20f,  0.50f,  0.10f,
        -0.10f,  0.25f, -0.35f,  0.45f,
         0.80f,  0.10f, -0.20f,  0.05f,
        -0.35f,  0.55f,  0.40f, -0.15f,
         0.20f, -0.45f,  0.15f,  0.60f,
    };

    std::vector<uint16_t> sink_tail_k_f16(sink_tail_k_ref.size());
    std::vector<uint16_t> sink_tail_v_f16(sink_tail_v_ref.size());
    for (size_t i = 0; i < sink_tail_k_ref.size(); ++i) {
        sink_tail_k_f16[i] = f32_to_f16_bits(sink_tail_k_ref[i]);
        sink_tail_v_f16[i] = f32_to_f16_bits(sink_tail_v_ref[i]);
        sink_tail_k_ref[i] = f16_bits_to_f32(sink_tail_k_f16[i]);
        sink_tail_v_ref[i] = f16_bits_to_f32(sink_tail_v_f16[i]);
    }

    float * q_d = cuda_upload(q);
    uint16_t * sink_tail_k_d = cuda_upload(sink_tail_k_f16);
    uint16_t * sink_tail_v_d = cuda_upload(sink_tail_v_f16);

    uint8_t * k_body_d = nullptr;
    uint8_t * v_body_d = nullptr;
    float * k_scales_d = nullptr;
    float * v_scales_d = nullptr;
    float * pending_k_d = nullptr;
    float * pending_v_d = nullptr;
    float * out_d = nullptr;
    float * scores_d = nullptr;

    require_cuda(cudaMalloc(&out_d, head_dim*sizeof(float)), "cudaMalloc output");
    require_cuda(cudaMalloc(&scores_d, n_tokens*sizeof(float)), "cudaMalloc scores");

    ggml_cuda_kvarn_attn_mixed_f16_batch(
            q_d, sink_tail_k_d, sink_tail_v_d,
            k_body_d, v_body_d, k_scales_d, v_scales_d, pending_k_d, pending_v_d,
            nullptr,
            out_d, scores_d,
            n_queries, n_head, n_head_kv,
            n_sink, n_records, n_pending, n_tail, tail_start, head_dim, group_size,
            key_bits, value_bits,
            head_dim, n_head*head_dim,
            head_dim, n_head*head_dim,
            head_dim, n_head_kv*head_dim,
            head_dim, n_head_kv*head_dim,
            1, 1, 1, 1, 1, 1, 1, 1,
            0, 0, 0,
            scale,
            nullptr);
    require_cuda(cudaGetLastError(), "KVarN CUDA wrapped-tail mixed attention launch");
    require_cuda(cudaDeviceSynchronize(), "KVarN CUDA wrapped-tail mixed attention sync");

    std::vector<float> out_gpu(head_dim);
    require_cuda(cudaMemcpy(out_gpu.data(), out_d, out_gpu.size()*sizeof(float), cudaMemcpyDeviceToHost),
            "copy wrapped-tail mixed attention output");

    std::vector<float> scores(n_tokens, 0.0f);
    for (uint32_t t = 0; t < n_sink; ++t) {
        for (uint32_t d = 0; d < head_dim; ++d) {
            scores[t] += q[d]*sink_tail_k_ref[size_t(t)*head_dim + d];
        }
        scores[t] *= scale;
    }
    for (uint32_t t = 0; t < n_tail; ++t) {
        const uint32_t slot = n_sink + (tail_start + t)%n_tail;
        const size_t score_i = n_sink + t;
        for (uint32_t d = 0; d < head_dim; ++d) {
            scores[score_i] += q[d]*sink_tail_k_ref[size_t(slot)*head_dim + d];
        }
        scores[score_i] *= scale;
    }

    const float max_score = *std::max_element(scores.begin(), scores.end());
    std::vector<float> probs(n_tokens, 0.0f);
    float denom = 0.0f;
    for (uint32_t i = 0; i < n_tokens; ++i) {
        probs[i] = std::exp(scores[i] - max_score);
        denom += probs[i];
    }
    for (float & p : probs) {
        p /= denom;
    }

    std::vector<float> out_ref(head_dim, 0.0f);
    for (uint32_t t = 0; t < n_sink; ++t) {
        for (uint32_t d = 0; d < head_dim; ++d) {
            out_ref[d] += probs[t]*sink_tail_v_ref[size_t(t)*head_dim + d];
        }
    }
    for (uint32_t t = 0; t < n_tail; ++t) {
        const uint32_t slot = n_sink + (tail_start + t)%n_tail;
        const size_t prob_i = n_sink + t;
        for (uint32_t d = 0; d < head_dim; ++d) {
            out_ref[d] += probs[prob_i]*sink_tail_v_ref[size_t(slot)*head_dim + d];
        }
    }

    float max_err = 0.0f;
    for (uint32_t d = 0; d < head_dim; ++d) {
        max_err = std::max(max_err, std::fabs(out_ref[d] - out_gpu[d]));
    }
    if (max_err >= 1.0e-6f) {
        std::fprintf(stderr, "max_err = %.9g\n", max_err);
        for (uint32_t d = 0; d < head_dim; ++d) {
            std::fprintf(stderr, "d%u ref=%.9g gpu=%.9g\n", d, out_ref[d], out_gpu[d]);
        }
    }
    require(max_err < 1.0e-6f, "KVarN CUDA mixed F16 wrapper reads wrapped tail in chronological order");

    cudaFree(q_d);
    cudaFree(sink_tail_k_d);
    cudaFree(sink_tail_v_d);
    cudaFree(out_d);
    cudaFree(scores_d);

    {
        constexpr uint32_t n_sink2 = 1;
        constexpr uint32_t n_records2 = 1;
        constexpr uint32_t n_pending2 = 2;
        constexpr uint32_t n_tail2 = 2;
        constexpr uint32_t tail_start2 = 1;
        constexpr uint32_t head_dim2 = 4;
        constexpr uint32_t group_size2 = 4;
        constexpr uint32_t key_bits2 = 4;
        constexpr uint32_t value_bits2 = 2;
        constexpr uint32_t n_tokens2 = n_sink2 + n_records2*group_size2 + n_pending2 + n_tail2;
        constexpr float scale2 = 0.25f;

        std::vector<float> q2 = {
            -0.20f, 0.35f, 0.60f, -0.45f,
        };
        std::vector<float> sink_tail_k2 = {
            0.30f, -0.10f, 0.20f, 0.45f,
            0.10f,  0.60f, 0.30f, 0.20f,
            0.70f, -0.30f, 0.15f, 0.05f,
        };
        std::vector<float> sink_tail_v2 = {
            -0.20f, 0.50f, 0.10f, 0.25f,
             0.40f, 0.05f, 0.30f, 0.10f,
            -0.10f, 0.35f, 0.55f, 0.20f,
        };
        std::vector<uint16_t> sink_tail_k2_f16(sink_tail_k2.size());
        std::vector<uint16_t> sink_tail_v2_f16(sink_tail_v2.size());
        for (size_t i = 0; i < sink_tail_k2.size(); ++i) {
            sink_tail_k2_f16[i] = f32_to_f16_bits(sink_tail_k2[i]);
            sink_tail_v2_f16[i] = f32_to_f16_bits(sink_tail_v2[i]);
            sink_tail_k2[i] = f16_bits_to_f32(sink_tail_k2_f16[i]);
            sink_tail_v2[i] = f16_bits_to_f32(sink_tail_v2_f16[i]);
        }

        const std::vector<uint8_t> k_q = {
            1, 2, 3, 4,
            2, 0, 5, 1,
            0, 3, 1, 2,
            4, 1, 0, 3,
        };
        const std::vector<uint8_t> v_q = {
            0, 1, 2, 3,
            3, 2, 1, 0,
            1, 0, 3, 2,
            2, 3, 0, 1,
        };
        const std::vector<uint8_t> k_body = pack_bits(k_q, key_bits2);
        const std::vector<uint8_t> v_body = pack_bits(v_q, value_bits2);
        std::vector<float> k_scales(2*head_dim2 + group_size2, 1.0f);
        std::vector<float> v_scales(head_dim2 + 2*group_size2, 1.0f);
        for (uint32_t d = 0; d < head_dim2; ++d) {
            k_scales[head_dim2 + d] = 0.0f;
        }
        for (uint32_t g = 0; g < group_size2; ++g) {
            v_scales[head_dim2 + group_size2 + g] = 0.0f;
        }

        std::vector<float> pending_k = {
            0.25f, 0.10f, -0.20f, 0.40f,
           -0.15f, 0.30f,  0.45f, 0.05f,
        };
        std::vector<float> pending_v = {
             0.20f, -0.25f, 0.15f, 0.50f,
             0.45f,  0.10f, 0.35f, -0.05f,
        };

        float * q2_d = cuda_upload(q2);
        uint16_t * sink_tail_k2_d = cuda_upload(sink_tail_k2_f16);
        uint16_t * sink_tail_v2_d = cuda_upload(sink_tail_v2_f16);
        uint8_t * k_body2_d = cuda_upload(k_body);
        uint8_t * v_body2_d = cuda_upload(v_body);
        float * k_scales2_d = cuda_upload(k_scales);
        float * v_scales2_d = cuda_upload(v_scales);
        float * pending_k2_d = cuda_upload(pending_k);
        float * pending_v2_d = cuda_upload(pending_v);
        float * out2_d = nullptr;
        float * scores2_d = nullptr;
        require_cuda(cudaMalloc(&out2_d, head_dim2*sizeof(float)), "cudaMalloc body/pending output");
        require_cuda(cudaMalloc(&scores2_d, n_tokens2*sizeof(float)), "cudaMalloc body/pending scores");

        ggml_cuda_kvarn_attn_mixed_f16_batch(
                q2_d, sink_tail_k2_d, sink_tail_v2_d,
                k_body2_d, v_body2_d, k_scales2_d, v_scales2_d, pending_k2_d, pending_v2_d,
                nullptr,
                out2_d, scores2_d,
                n_queries, n_head, n_head_kv,
                n_sink2, n_records2, n_pending2, n_tail2, tail_start2, head_dim2, group_size2,
                key_bits2, value_bits2,
                head_dim2, n_head*head_dim2,
                head_dim2, n_head*head_dim2,
                head_dim2, n_head_kv*head_dim2,
                head_dim2, n_head_kv*head_dim2,
                k_body.size(), v_body.size(), k_body.size(), v_body.size(),
                k_scales.size(), v_scales.size(), k_scales.size(), v_scales.size(),
                0, 0, 0,
                scale2,
                nullptr);
        require_cuda(cudaGetLastError(), "KVarN CUDA body/pending/tail mixed attention launch");
        require_cuda(cudaDeviceSynchronize(), "KVarN CUDA body/pending/tail mixed attention sync");

        std::vector<float> out2_gpu(head_dim2);
        require_cuda(cudaMemcpy(out2_gpu.data(), out2_d, out2_gpu.size()*sizeof(float), cudaMemcpyDeviceToHost),
                "copy body/pending/tail mixed output");

        std::vector<float> scores2(n_tokens2, 0.0f);
        for (uint32_t d = 0; d < head_dim2; ++d) {
            scores2[0] += q2[d]*sink_tail_k2[d];
        }
        scores2[0] *= scale2;

        for (uint32_t g = 0; g < group_size2; ++g) {
            const size_t score_i = n_sink2 + g;
            for (uint32_t d = 0; d < head_dim2; ++d) {
                scores2[score_i] += q2[d]*float(k_q[d*group_size2 + g]);
            }
            scores2[score_i] *= scale2;
        }

        for (uint32_t t = 0; t < n_pending2; ++t) {
            const size_t score_i = n_sink2 + group_size2 + t;
            for (uint32_t d = 0; d < head_dim2; ++d) {
                scores2[score_i] += q2[d]*pending_k[size_t(t)*head_dim2 + d];
            }
            scores2[score_i] *= scale2;
        }

        for (uint32_t t = 0; t < n_tail2; ++t) {
            const uint32_t slot = n_sink2 + (tail_start2 + t)%n_tail2;
            const size_t score_i = n_sink2 + group_size2 + n_pending2 + t;
            for (uint32_t d = 0; d < head_dim2; ++d) {
                scores2[score_i] += q2[d]*sink_tail_k2[size_t(slot)*head_dim2 + d];
            }
            scores2[score_i] *= scale2;
        }

        const float max_score2 = *std::max_element(scores2.begin(), scores2.end());
        std::vector<float> probs2(n_tokens2, 0.0f);
        float denom2 = 0.0f;
        for (uint32_t i = 0; i < n_tokens2; ++i) {
            probs2[i] = std::exp(scores2[i] - max_score2);
            denom2 += probs2[i];
        }
        for (float & p : probs2) {
            p /= denom2;
        }

        std::vector<float> out2_ref(head_dim2, 0.0f);
        for (uint32_t d = 0; d < head_dim2; ++d) {
            out2_ref[d] += probs2[0]*sink_tail_v2[d];
        }
        for (uint32_t g = 0; g < group_size2; ++g) {
            const size_t prob_i = n_sink2 + g;
            for (uint32_t d = 0; d < head_dim2; ++d) {
                out2_ref[d] += probs2[prob_i]*float(v_q[g*head_dim2 + d]);
            }
        }
        for (uint32_t t = 0; t < n_pending2; ++t) {
            const size_t prob_i = n_sink2 + group_size2 + t;
            for (uint32_t d = 0; d < head_dim2; ++d) {
                out2_ref[d] += probs2[prob_i]*pending_v[size_t(t)*head_dim2 + d];
            }
        }
        for (uint32_t t = 0; t < n_tail2; ++t) {
            const uint32_t slot = n_sink2 + (tail_start2 + t)%n_tail2;
            const size_t prob_i = n_sink2 + group_size2 + n_pending2 + t;
            for (uint32_t d = 0; d < head_dim2; ++d) {
                out2_ref[d] += probs2[prob_i]*sink_tail_v2[size_t(slot)*head_dim2 + d];
            }
        }

        float max_err2 = 0.0f;
        for (uint32_t d = 0; d < head_dim2; ++d) {
            max_err2 = std::max(max_err2, std::fabs(out2_ref[d] - out2_gpu[d]));
        }
        if (max_err2 >= 1.0e-6f) {
            std::fprintf(stderr, "body/pending max_err = %.9g\n", max_err2);
            for (uint32_t d = 0; d < head_dim2; ++d) {
                std::fprintf(stderr, "d%u ref=%.9g gpu=%.9g\n", d, out2_ref[d], out2_gpu[d]);
            }
        }
        require(max_err2 < 1.0e-6f, "KVarN CUDA mixed F16 wrapper matches body/pending/wrapped-tail CPU reference");

        cudaFree(q2_d);
        cudaFree(sink_tail_k2_d);
        cudaFree(sink_tail_v2_d);
        cudaFree(k_body2_d);
        cudaFree(v_body2_d);
        cudaFree(k_scales2_d);
        cudaFree(v_scales2_d);
        cudaFree(pending_k2_d);
        cudaFree(pending_v2_d);
        cudaFree(out2_d);
        cudaFree(scores2_d);
    }

    return 0;
}
