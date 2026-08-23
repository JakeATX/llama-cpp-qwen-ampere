#include "mmv-cr.cuh"

#include "dequantize.cuh"

#include <cstdint>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

static __device__ __forceinline__ float convrot_inverse_group(
        float buf[2][QK8_CR], const int tid) {
    __syncthreads();
    float v = buf[0][tid];
    const int lane = tid & 31;

#pragma unroll
    for (int len = 4; len <= 16; len *= 4) {
        const int half = len/4;
        const int base = lane & ~(len - 1);
        const int r = (lane/half) & 3;
        const int col = lane & (half - 1);
        const float a = __shfl_sync(0xffffffff, v, base + col);
        const float b = __shfl_sync(0xffffffff, v, base + half + col);
        const float c = __shfl_sync(0xffffffff, v, base + 2*half + col);
        const float d = __shfl_sync(0xffffffff, v, base + 3*half + col);
        v = r == 0 ?  a + b + c - d :
            r == 1 ?  a + b - c + d :
            r == 2 ?  a - b + c + d :
                      -a + b + c + d;
    }

    buf[0][tid] = v;
    __syncthreads();

    constexpr int half64 = 16;
    const int base64 = tid & ~63;
    const int r64 = (tid/half64) & 3;
    const int col64 = tid & (half64 - 1);
    const float a64 = buf[0][base64 + col64];
    const float b64 = buf[0][base64 + half64 + col64];
    const float c64 = buf[0][base64 + 2*half64 + col64];
    const float d64 = buf[0][base64 + 3*half64 + col64];
    buf[1][tid] = r64 == 0 ?  a64 + b64 + c64 - d64 :
                  r64 == 1 ?  a64 + b64 - c64 + d64 :
                  r64 == 2 ?  a64 - b64 + c64 + d64 :
                              -a64 + b64 + c64 + d64;
    __syncthreads();

    constexpr int half256 = 64;
    const int r256 = tid/half256;
    const int col256 = tid & (half256 - 1);
    const float a256 = buf[1][col256];
    const float b256 = buf[1][half256 + col256];
    const float c256 = buf[1][2*half256 + col256];
    const float d256 = buf[1][3*half256 + col256];
    return (r256 == 0 ?  a256 + b256 + c256 - d256 :
            r256 == 1 ?  a256 + b256 - c256 + d256 :
            r256 == 2 ?  a256 - b256 + c256 + d256 :
                        -a256 + b256 + c256 + d256) * (1.0f/16.0f);
}

template<int qk, int qr, dequantize_kernel_t dequantize_kernel>
static __global__ void mul_mat_vec_cr_reference(
        const void * __restrict__ vx,
        const float * __restrict__ x,
        float * __restrict__ y,
        const int64_t k) {
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;
    const int64_t blocks_per_row = k/QK8_CR;
    float acc = 0.0f;

    __shared__ float buf[2][QK8_CR];
    for (int64_t block = 0; block < blocks_per_row; ++block) {
        if (tid < QK8_CR/2) {
            const int i00 = 2*tid;
            const int64_t ib = (row*blocks_per_row + block)*(QK8_CR/qk) + i00/qk;
            const int iqs = (i00%qk)/qr;
            const int iybs = i00 - i00%qk;
            const int y_offset = qr == 1 ? 1 : qk/2;
            float2 values;
            dequantize_kernel(vx, ib, iqs, values);
            buf[0][iybs + iqs] = values.x;
            buf[0][iybs + iqs + y_offset] = values.y;
        }
        const float weight = convrot_inverse_group(buf, tid);
        acc += weight * x[block*QK8_CR + tid];
        __syncthreads();
    }

    buf[0][tid] = acc;
    __syncthreads();
    for (int stride = QK8_CR/2; stride > 0; stride /= 2) {
        if (tid < stride) {
            buf[0][tid] += buf[0][tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        y[row] = buf[0][0];
    }
}

static __global__ void mul_mat_vec_q6_cr_reference(
        const void * __restrict__ vx,
        const float * __restrict__ x,
        float * __restrict__ y,
        const int64_t k) {
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;
    const int64_t blocks_per_row = k/QK8_CR;
    float acc = 0.0f;

    __shared__ float buf[2][QK8_CR];
    for (int64_t block = 0; block < blocks_per_row; ++block) {
        if (tid < 64) {
            dequantize_q6_K(vx, row*blocks_per_row + block, buf[0], tid);
        }
        const float weight = convrot_inverse_group(buf, tid);
        acc += weight * x[block*QK8_CR + tid];
        __syncthreads();
    }

    buf[0][tid] = acc;
    __syncthreads();
    for (int stride = QK8_CR/2; stride > 0; stride /= 2) {
        if (tid < stride) {
            buf[0][tid] += buf[0][tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        y[row] = buf[0][0];
    }
}

static void launch_reference(
        ggml_type type,
        const void * weights,
        const float * x,
        float * y,
        int64_t m,
        int64_t k,
        cudaStream_t stream) {
    switch (type) {
        case GGML_TYPE_Q5_CR:
            mul_mat_vec_cr_reference<QK5_0, QR5_0, dequantize_q5_0>
                <<<m, QK8_CR, 0, stream>>>(weights, x, y, k);
            break;
        case GGML_TYPE_Q6_CR:
            mul_mat_vec_q6_cr_reference<<<m, QK8_CR, 0, stream>>>(weights, x, y, k);
            break;
        case GGML_TYPE_Q8_CR:
            mul_mat_vec_cr_reference<QK8_0, QR8_0, dequantize_q8_0>
                <<<m, QK8_CR, 0, stream>>>(weights, x, y, k);
            break;
        default:
            GGML_ABORT("unsupported CR type");
    }
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_mul_mat_vec_cr(
        ggml_type type,
        const void * weights,
        const float * x,
        float * y,
        int64_t m,
        int64_t n,
        int64_t k,
        int cc,
        cudaStream_t stream) {
    GGML_UNUSED(cc);
    if (n != 1 || k % QK8_CR != 0) {
        return false;
    }
    if (m == 0 || k == 0) {
        return true;
    }
    launch_reference(type, weights, x, y, m, k, stream);
    return true;
}

#endif
