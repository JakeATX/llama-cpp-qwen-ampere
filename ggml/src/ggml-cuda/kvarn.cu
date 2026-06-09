#include "kvarn.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>

// NVCC can vectorize shared-memory reduction loads at the end of the scratch
// region. Keep a small pad so sanitizer-visible overreads stay inside the
// dynamic shared allocation.
static constexpr size_t KVARN_ATTN_SHMEM_PAD_FLOATS = 8;

static bool kvarn_env_flag(const char * name) {
    const char * env = std::getenv(name);
    if (env == nullptr) {
        return false;
    }

    char * end = nullptr;
    errno = 0;
    const long value = std::strtol(env, &end, 10);
    if (env[0] == '\0' || end == nullptr || *end != '\0' || errno == ERANGE ||
            (value != 0 && value != 1)) {
        std::fprintf(stderr, "invalid KVarN CUDA environment flag %s=%s; expected integer 0 or 1\n", name, env);
        std::abort();
    }
    return value != 0;
}

struct kvarn_aux_cuda_state {
    cudaStream_t stream     = nullptr;
    cudaEvent_t  main_ready = nullptr;
    cudaEvent_t  aux_done   = nullptr;
    int          device     = -1;
};

static kvarn_aux_cuda_state & kvarn_aux_cuda_state_get() {
    thread_local kvarn_aux_cuda_state st;

    int dev = 0;
    cudaGetDevice(&dev);
    if (st.stream == nullptr || st.device != dev) {
        if (st.stream != nullptr) {
            cudaStreamDestroy(st.stream);
            cudaEventDestroy(st.main_ready);
            cudaEventDestroy(st.aux_done);
        }
        st.device = dev;
        cudaStreamCreateWithFlags(&st.stream, cudaStreamNonBlocking);
        cudaEventCreateWithFlags(&st.main_ready, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&st.aux_done,   cudaEventDisableTiming);
    }
    return st;
}

static bool kvarn_cuda_dynamic_shmem_fits(size_t nbytes) {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return true;
    }

    cudaDeviceProp prop = {};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        return true;
    }

    return nbytes <= prop.sharedMemPerBlock;
}

static __host__ __device__ size_t kvarn_packed_nbytes(size_t n_values, uint32_t bits) {
    return (n_values*bits + 7)/8;
}

static __device__ uint32_t kvarn_unpack_one(const uint8_t * src, uint32_t bits, size_t i) {
    const uint32_t mask = (1u << bits) - 1u;
    const size_t bit_pos = i*bits;
    const size_t byte_pos = bit_pos >> 3;
    const uint32_t shift = uint32_t(bit_pos & 7);

    uint32_t q = uint32_t(src[byte_pos]) >> shift;
    if (shift + bits > 8) {
        q |= uint32_t(src[byte_pos + 1]) << (8 - shift);
    }

    return q & mask;
}

static __device__ void kvarn_pack_one(uint8_t * dst, uint32_t bits, size_t i, uint32_t q) {
    const uint32_t mask = (1u << bits) - 1u;
    const size_t bit_pos = i*bits;
    const size_t byte_pos = bit_pos >> 3;
    const uint32_t shift = uint32_t(bit_pos & 7);
    q &= mask;

    dst[byte_pos] |= uint8_t(q << shift);
    if (shift + bits > 8) {
        dst[byte_pos + 1] |= uint8_t(q >> (8 - shift));
    }
}

static __device__ __forceinline__ float kvarn_kq_mask_bias(
        const void * __restrict__ kq_mask,
        uint32_t kq_mask_type,
        size_t kq_mask_stride_token_bytes,
        uint32_t t) {
    if (kq_mask == nullptr || kq_mask_type == 0) {
        return 0.0f;
    }

    const char * p = (const char *) kq_mask + size_t(t)*kq_mask_stride_token_bytes;
    if (kq_mask_type == 1) {
        return *(const float *) p;
    }
    if (kq_mask_type == 2) {
        return __half2float(*(const __half *) p);
    }
    return 0.0f;
}

static __device__ __forceinline__ float kvarn_warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

[[maybe_unused]] static __device__ __forceinline__ float kvarn_warp_reduce_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return v;
}

static __device__ float kvarn_select_kth_row_value(const float * src, uint32_t cols, uint32_t kth) {
    for (uint32_t i = 0; i < cols; ++i) {
        const float v = src[i];
        uint32_t n_less = 0;
        uint32_t n_less_equal = 0;
        for (uint32_t j = 0; j < cols; ++j) {
            const float x = src[j];
            n_less       += x <  v;
            n_less_equal += x <= v;
        }
        if (n_less <= kth && kth < n_less_equal) {
            return v;
        }
    }

    return src[cols - 1];
}

static __global__ void kvarn_fill_f32_kernel(float * dst, uint32_t n, float value) {
    const uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = value;
    }
}

static __global__ void kvarn_hadamard_rows_kernel(
        const float * __restrict__ src,
        float * __restrict__ dst,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t r = blockIdx.x;
    if (r >= rows || threadIdx.x != 0) {
        return;
    }

    float * row = dst + size_t(r)*cols;
    const float * src_row = src + size_t(r)*cols;
    for (uint32_t c = 0; c < cols; ++c) {
        row[c] = src_row[c];
    }

    const float norm = rsqrtf(float(cols));
    for (uint32_t step = 1; step < cols; step <<= 1) {
        for (uint32_t base = 0; base < cols; base += 2*step) {
            for (uint32_t i = 0; i < step; ++i) {
                const uint32_t c0 = base + i;
                const uint32_t c1 = c0 + step;
                const float a = row[c0];
                const float b = row[c1];
                row[c0] = a + b;
                row[c1] = a - b;
            }
        }
    }

    for (uint32_t c = 0; c < cols; ++c) {
        row[c] *= norm;
    }
}

static __global__ void kvarn_hadamard_cols_kernel(
        const float * __restrict__ src,
        float * __restrict__ dst,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t c = blockIdx.x;
    if (c >= cols || threadIdx.x != 0) {
        return;
    }

    for (uint32_t r = 0; r < rows; ++r) {
        dst[size_t(r)*cols + c] = src[size_t(r)*cols + c];
    }

    const float norm = rsqrtf(float(rows));
    for (uint32_t step = 1; step < rows; step <<= 1) {
        for (uint32_t base = 0; base < rows; base += 2*step) {
            for (uint32_t i = 0; i < step; ++i) {
                const uint32_t r0 = base + i;
                const uint32_t r1 = r0 + step;
                const size_t i0 = size_t(r0)*cols + c;
                const size_t i1 = size_t(r1)*cols + c;
                const float a = dst[i0];
                const float b = dst[i1];
                dst[i0] = a + b;
                dst[i1] = a - b;
            }
        }
    }

    for (uint32_t r = 0; r < rows; ++r) {
        dst[size_t(r)*cols + c] *= norm;
    }
}

static __global__ void kvarn_hadamard_rows_parallel_kernel(
        const float * __restrict__ src,
        float * __restrict__ dst,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t r = blockIdx.x;
    if (r >= rows) {
        return;
    }

    extern __shared__ float shmem[];
    const uint32_t c = threadIdx.x;
    const float norm = rsqrtf(float(cols));

    if (c < cols) {
        shmem[c] = src[size_t(r)*cols + c];
    }
    __syncthreads();

    for (uint32_t step = 1; step < cols; step <<= 1) {
        if (c < cols && (c & step) == 0) {
            const uint32_t c1 = c + step;
            const float a = shmem[c];
            const float b = shmem[c1];
            shmem[c]  = a + b;
            shmem[c1] = a - b;
        }
        __syncthreads();
    }

    if (c < cols) {
        dst[size_t(r)*cols + c] = shmem[c]*norm;
    }
}

static __global__ void kvarn_hadamard_cols_parallel_kernel(
        const float * __restrict__ src,
        float * __restrict__ dst,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t c_col = blockIdx.x;
    if (c_col >= cols) {
        return;
    }

    extern __shared__ float shmem[];
    const uint32_t r = threadIdx.x;
    const float norm = rsqrtf(float(rows));

    if (r < rows) {
        shmem[r] = src[size_t(r)*cols + c_col];
    }
    __syncthreads();

    for (uint32_t step = 1; step < rows; step <<= 1) {
        if (r < rows && (r & step) == 0) {
            const uint32_t r1 = r + step;
            const float a = shmem[r];
            const float b = shmem[r1];
            shmem[r]  = a + b;
            shmem[r1] = a - b;
        }
        __syncthreads();
    }

    if (r < rows) {
        dst[size_t(r)*cols + c_col] = shmem[r]*norm;
    }
}

static int kvarn_pow2_block(uint32_t n) {
    int block = 1;
    while (block < int(n)) {
        block <<= 1;
    }
    return block;
}

static __global__ void kvarn_sinkhorn_rows_kernel(
        float * __restrict__ data,
        float * __restrict__ row_scale,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t r = blockIdx.x;
    if (r >= rows || threadIdx.x != 0) {
        return;
    }

    constexpr float eps = 1.0e-6f;
    double ss = 0.0;
    for (uint32_t c = 0; c < cols; ++c) {
        const float v = data[size_t(r)*cols + c];
        ss += double(v)*double(v);
    }

    const float rms = sqrtf(float(ss/cols) + eps);
    row_scale[r] *= rms;
    for (uint32_t c = 0; c < cols; ++c) {
        data[size_t(r)*cols + c] /= rms;
    }
}

static __global__ void kvarn_sinkhorn_cols_kernel(
        float * __restrict__ data,
        float * __restrict__ col_scale,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t c = blockIdx.x;
    if (c >= cols || threadIdx.x != 0) {
        return;
    }

    constexpr float eps = 1.0e-6f;
    double ss = 0.0;
    for (uint32_t r = 0; r < rows; ++r) {
        const float v = data[size_t(r)*cols + c];
        ss += double(v)*double(v);
    }

    const float rms = sqrtf(float(ss/rows) + eps);
    col_scale[c] *= rms;
    for (uint32_t r = 0; r < rows; ++r) {
        data[size_t(r)*cols + c] /= rms;
    }
}

static __global__ void kvarn_quantize_asym_minmax_pack_rows_kernel(
        const float * __restrict__ src,
        uint8_t * __restrict__ body,
        float * __restrict__ row_scale,
        float * __restrict__ row_zp,
        uint32_t rows,
        uint32_t cols,
        uint32_t bits,
        float quantile) {
    const uint32_t r = blockIdx.x;
    if (r >= rows || threadIdx.x != 0) {
        return;
    }

    const uint32_t qmax = (1u << bits) - 1u;
    const float qt = fminf(1.0f, fmaxf(0.000001f, quantile));
    const uint32_t lo_i = uint32_t((1.0f - qt)*0.5f*float(cols - 1));
    const uint32_t hi_i = uint32_t((1.0f - (1.0f - qt)*0.5f)*float(cols - 1));
    const float * row = src + size_t(r)*cols;
    const float mn = kvarn_select_kth_row_value(row, cols, lo_i);
    const float mx = kvarn_select_kth_row_value(row, cols, hi_i);
    const float s = (mx > mn) ? (mx - mn)/float(qmax) : 1.0f;

    row_scale[r] = s;
    row_zp[r] = mn;

    for (uint32_t c = 0; c < cols; ++c) {
        const float v = fminf(mx, fmaxf(mn, src[size_t(r)*cols + c]));
        const uint32_t q = min(qmax, uint32_t(llroundf((v - mn)/s)));
        kvarn_pack_one(body, bits, size_t(r)*cols + c, q);
    }
}

static __global__ void kvarn_quantize_asym_fullrange_pack_rows_kernel(
        const float * __restrict__ src,
        uint8_t * __restrict__ body,
        float * __restrict__ row_scale,
        float * __restrict__ row_zp,
        uint32_t rows,
        uint32_t cols,
        uint32_t bits) {
    const uint32_t r = blockIdx.x;
    if (r >= rows || threadIdx.x != 0) {
        return;
    }

    const uint32_t qmax = (1u << bits) - 1u;
    const float * row = src + size_t(r)*cols;
    float mn = row[0];
    float mx = row[0];
    for (uint32_t c = 1; c < cols; ++c) {
        const float v = row[c];
        mn = fminf(mn, v);
        mx = fmaxf(mx, v);
    }

    const float s = (mx > mn) ? (mx - mn)/float(qmax) : 1.0f;
    row_scale[r] = s;
    row_zp[r] = mn;

    for (uint32_t c = 0; c < cols; ++c) {
        const float v = fminf(mx, fmaxf(mn, row[c]));
        const uint32_t q = min(qmax, uint32_t(llroundf((v - mn)/s)));
        kvarn_pack_one(body, bits, size_t(r)*cols + c, q);
    }
}

static __global__ void kvarn_quantize_asym_fullrange_pack_rows_parallel_kernel(
        const float * __restrict__ src,
        uint8_t * __restrict__ body,
        float * __restrict__ row_scale,
        float * __restrict__ row_zp,
        uint32_t rows,
        uint32_t cols,
        uint32_t bits) {
    const uint32_t r = blockIdx.x;
    if (r >= rows) {
        return;
    }

    const uint32_t qmax = (1u << bits) - 1u;
    const float * row = src + size_t(r)*cols;

    float local_mn = 3.4028234663852886e38f;
    float local_mx = -3.4028234663852886e38f;
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = row[c];
        local_mn = fminf(local_mn, v);
        local_mx = fmaxf(local_mx, v);
    }

    extern __shared__ float shmem[];
    shmem[threadIdx.x*2]     = local_mn;
    shmem[threadIdx.x*2 + 1] = local_mx;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shmem[threadIdx.x*2]     = fminf(shmem[threadIdx.x*2],     shmem[(threadIdx.x + stride)*2]);
            shmem[threadIdx.x*2 + 1] = fmaxf(shmem[threadIdx.x*2 + 1], shmem[(threadIdx.x + stride)*2 + 1]);
        }
        __syncthreads();
    }

    const float mn = shmem[0];
    const float mx = shmem[1];
    const float s  = (mx > mn) ? (mx - mn)/float(qmax) : 1.0f;

    if (threadIdx.x == 0) {
        row_scale[r] = s;
        row_zp[r]    = mn;
    }
    __syncthreads();

    // Write one packed output byte from one CUDA thread. Calling kvarn_pack_one()
    // per value lets multiple threads OR into the same byte for k4/v2 rows.
    const size_t row_bit0 = size_t(r)*cols*bits;
    const size_t row_nbits = size_t(cols)*bits;
    const bool row_byte_aligned = ((row_bit0 | row_nbits) & 7u) == 0u;
    if (!row_byte_aligned) {
        if (threadIdx.x == 0) {
            for (uint32_t c = 0; c < cols; ++c) {
                const float v = fminf(mx, fmaxf(mn, row[c]));
                const uint32_t q = min(qmax, uint32_t(llroundf((v - mn)/s)));
                kvarn_pack_one(body, bits, size_t(r)*cols + c, q);
            }
        }
        return;
    }

    const size_t row_byte0 = row_bit0 >> 3;
    const size_t row_nbytes = row_nbits >> 3;

    if (bits == 4) {
        for (size_t b = threadIdx.x; b < row_nbytes; b += blockDim.x) {
            const uint32_t c0 = uint32_t(2*b);
            const uint32_t c1 = c0 + 1;

            const float v0 = fminf(mx, fmaxf(mn, row[c0]));
            const uint32_t q0 = min(qmax, uint32_t(llroundf((v0 - mn)/s)));

            uint32_t q1 = 0;
            if (c1 < cols) {
                const float v1 = fminf(mx, fmaxf(mn, row[c1]));
                q1 = min(qmax, uint32_t(llroundf((v1 - mn)/s)));
            }

            body[row_byte0 + b] = uint8_t(q0 | (q1 << 4));
        }
    } else if (bits == 2) {
        for (size_t b = threadIdx.x; b < row_nbytes; b += blockDim.x) {
            uint32_t packed = 0;
            const uint32_t c_base = uint32_t(4*b);
            for (uint32_t j = 0; j < 4; ++j) {
                const uint32_t c = c_base + j;
                if (c < cols) {
                    const float v = fminf(mx, fmaxf(mn, row[c]));
                    const uint32_t q = min(qmax, uint32_t(llroundf((v - mn)/s)));
                    packed |= q << (2*j);
                }
            }
            body[row_byte0 + b] = uint8_t(packed);
        }
    } else if (threadIdx.x == 0) {
        for (uint32_t c = 0; c < cols; ++c) {
            const float v = fminf(mx, fmaxf(mn, row[c]));
            const uint32_t q = min(qmax, uint32_t(llroundf((v - mn)/s)));
            kvarn_pack_one(body, bits, size_t(r)*cols + c, q);
        }
    }
}

static __global__ void kvarn_store_k_finalize_scales_kernel(
        float * __restrict__ k_scales,
        const float * __restrict__ rtn_scale,
        const float * __restrict__ rtn_zp,
        uint32_t head_dim) {
    const uint32_t d = blockIdx.x*blockDim.x + threadIdx.x;
    if (d >= head_dim) {
        return;
    }

    const float row = k_scales[d];
    k_scales[d] = row*rtn_scale[d];
    k_scales[head_dim + d] = row*rtn_zp[d];
}

static __global__ void kvarn_store_v_finalize_scales_kernel(
        float * __restrict__ v_scales,
        const float * __restrict__ rtn_scale,
        const float * __restrict__ rtn_zp,
        uint32_t head_dim,
        uint32_t group_size) {
    const uint32_t g = blockIdx.x*blockDim.x + threadIdx.x;
    if (g >= group_size) {
        return;
    }

    float * v_row_scale = v_scales + head_dim;
    float * v_zp        = v_scales + head_dim + group_size;
    const float row = v_row_scale[g];
    v_row_scale[g] = row*rtn_scale[g];
    v_zp[g]        = row*rtn_zp[g];
}

static __global__ void kvarn_sinkhorn_rows_parallel_kernel(
        float * __restrict__ data,
        float * __restrict__ row_scale,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t r = blockIdx.x;
    if (r >= rows) {
        return;
    }

    constexpr float eps = 1.0e-6f;
    float local_ss = 0.0f;
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = data[size_t(r)*cols + c];
        local_ss += v*v;
    }

    extern __shared__ float shmem[];
    shmem[threadIdx.x] = local_ss;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shmem[threadIdx.x] += shmem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float rms = sqrtf(shmem[0]/float(cols) + eps);
        row_scale[r] *= rms;
        shmem[0] = rms;
    }
    __syncthreads();

    const float inv_rms = 1.0f/shmem[0];
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        data[size_t(r)*cols + c] *= inv_rms;
    }
}

static __global__ void kvarn_sinkhorn_cols_parallel_kernel(
        float * __restrict__ data,
        float * __restrict__ col_scale,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t c = blockIdx.x;
    if (c >= cols) {
        return;
    }

    constexpr float eps = 1.0e-6f;
    float local_ss = 0.0f;
    for (uint32_t r = threadIdx.x; r < rows; r += blockDim.x) {
        const float v = data[size_t(r)*cols + c];
        local_ss += v*v;
    }

    extern __shared__ float shmem[];
    shmem[threadIdx.x] = local_ss;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shmem[threadIdx.x] += shmem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float rms = sqrtf(shmem[0]/float(rows) + eps);
        col_scale[c] *= rms;
        shmem[0] = rms;
    }
    __syncthreads();

    const float inv_rms = 1.0f/shmem[0];
    for (uint32_t r = threadIdx.x; r < rows; r += blockDim.x) {
        data[size_t(r)*cols + c] *= inv_rms;
    }
}

static void kvarn_sinkhorn_variance_normalize_parallel(
        float * data,
        float * row_scale,
        float * col_scale,
        uint32_t rows,
        uint32_t cols,
        uint32_t iters,
        cudaStream_t stream) {
    const int block = 128;
    const size_t shmem = size_t(block)*sizeof(float);
    kvarn_fill_f32_kernel<<<int((rows + block - 1)/block), block, 0, stream>>>(row_scale, rows, 1.0f);
    kvarn_fill_f32_kernel<<<int((cols + block - 1)/block), block, 0, stream>>>(col_scale, cols, 1.0f);

    for (uint32_t iter = 0; iter < iters; ++iter) {
        kvarn_sinkhorn_rows_parallel_kernel<<<int(rows), block, shmem, stream>>>(data, row_scale, rows, cols);
        kvarn_sinkhorn_cols_parallel_kernel<<<int(cols), block, shmem, stream>>>(data, col_scale, rows, cols);
    }
}

static bool kvarn_fullrange_packer_overwrites_body(uint32_t cols, uint32_t bits) {
    if (bits != 2 && bits != 4) {
        return false;
    }

    return (size_t(cols)*bits) % 8 == 0;
}

void ggml_cuda_kvarn_store_k_body_reference_minmax(
        const float * k_tile,
        uint8_t * k_body,
        float * k_scales,
        float * scratch,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t key_bits,
        uint32_t sinkhorn_iters,
        float rtn_quantile,
        void * stream) {
    const size_t n = size_t(head_dim)*group_size;
    const uint32_t tmp_rows = head_dim > group_size ? head_dim : group_size;
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    float * data      = scratch;
    float * rtn_scale = scratch + n;
    float * rtn_zp    = scratch + n + tmp_rows;

    if (rtn_quantile < 1.0f || !kvarn_fullrange_packer_overwrites_body(group_size, key_bits)) {
        cudaMemsetAsync(k_body, 0, kvarn_packed_nbytes(n, key_bits), cuda_stream);
    }
    const int k_hadamard_block = kvarn_pow2_block(head_dim);
    if (k_hadamard_block <= 1024) {
        kvarn_hadamard_cols_parallel_kernel<<<int(group_size), k_hadamard_block, size_t(k_hadamard_block)*sizeof(float), cuda_stream>>>(
                k_tile, data, head_dim, group_size);
    } else {
        kvarn_hadamard_cols_kernel<<<int(group_size), 1, 0, cuda_stream>>>(k_tile, data, head_dim, group_size);
    }
    float * k_row_scale = k_scales;
    float * k_col_scale = k_scales + 2*head_dim;
    kvarn_sinkhorn_variance_normalize_parallel(
            data, k_row_scale, k_col_scale, head_dim, group_size, sinkhorn_iters, cuda_stream);
    if (rtn_quantile >= 1.0f) {
        const int k_quant_block = kvarn_pow2_block(group_size);
        if (k_quant_block <= 1024) {
            kvarn_quantize_asym_fullrange_pack_rows_parallel_kernel<<<int(head_dim), k_quant_block, size_t(k_quant_block)*2*sizeof(float), cuda_stream>>>(
                    data, k_body, rtn_scale, rtn_zp, head_dim, group_size, key_bits);
        } else {
            kvarn_quantize_asym_fullrange_pack_rows_kernel<<<int(head_dim), 1, 0, cuda_stream>>>(
                    data, k_body, rtn_scale, rtn_zp, head_dim, group_size, key_bits);
        }
    } else {
        kvarn_quantize_asym_minmax_pack_rows_kernel<<<int(head_dim), 1, 0, cuda_stream>>>(
                data, k_body, rtn_scale, rtn_zp, head_dim, group_size, key_bits, rtn_quantile);
    }
    const int block = 128;
    kvarn_store_k_finalize_scales_kernel<<<int((head_dim + block - 1)/block), block, 0, cuda_stream>>>(
            k_scales, rtn_scale, rtn_zp, head_dim);
}

void ggml_cuda_kvarn_store_v_body_reference_minmax(
        const float * v_tile,
        uint8_t * v_body,
        float * v_scales,
        float * scratch,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t value_bits,
        uint32_t sinkhorn_iters,
        float rtn_quantile,
        void * stream) {
    const size_t n = size_t(head_dim)*group_size;
    const uint32_t tmp_rows = head_dim > group_size ? head_dim : group_size;
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    float * data      = scratch;
    float * rtn_scale = scratch + n;
    float * rtn_zp    = scratch + n + tmp_rows;

    if (rtn_quantile < 1.0f || !kvarn_fullrange_packer_overwrites_body(head_dim, value_bits)) {
        cudaMemsetAsync(v_body, 0, kvarn_packed_nbytes(n, value_bits), cuda_stream);
    }
    const int v_hadamard_block = kvarn_pow2_block(head_dim);
    if (v_hadamard_block <= 1024) {
        kvarn_hadamard_rows_parallel_kernel<<<int(group_size), v_hadamard_block, size_t(v_hadamard_block)*sizeof(float), cuda_stream>>>(
                v_tile, data, group_size, head_dim);
    } else {
        kvarn_hadamard_rows_kernel<<<int(group_size), 1, 0, cuda_stream>>>(v_tile, data, group_size, head_dim);
    }
    float * v_col_scale = v_scales;
    float * v_row_scale = v_scales + head_dim;
    kvarn_sinkhorn_variance_normalize_parallel(
            data, v_row_scale, v_col_scale, group_size, head_dim, sinkhorn_iters, cuda_stream);
    if (rtn_quantile >= 1.0f) {
        const int v_quant_block = kvarn_pow2_block(head_dim);
        if (v_quant_block <= 1024) {
            kvarn_quantize_asym_fullrange_pack_rows_parallel_kernel<<<int(group_size), v_quant_block, size_t(v_quant_block)*2*sizeof(float), cuda_stream>>>(
                    data, v_body, rtn_scale, rtn_zp, group_size, head_dim, value_bits);
        } else {
            kvarn_quantize_asym_fullrange_pack_rows_kernel<<<int(group_size), 1, 0, cuda_stream>>>(
                    data, v_body, rtn_scale, rtn_zp, group_size, head_dim, value_bits);
        }
    } else {
        kvarn_quantize_asym_minmax_pack_rows_kernel<<<int(group_size), 1, 0, cuda_stream>>>(
                data, v_body, rtn_scale, rtn_zp, group_size, head_dim, value_bits, rtn_quantile);
    }
    const int block = 128;
    kvarn_store_v_finalize_scales_kernel<<<int((group_size + block - 1)/block), block, 0, cuda_stream>>>(
            v_scales, rtn_scale, rtn_zp, head_dim, group_size);
}

static size_t kvarn_store_scratch_floats_one(uint32_t head_dim, uint32_t group_size) {
    const uint32_t tmp_rows = head_dim > group_size ? head_dim : group_size;
    return size_t(head_dim)*group_size + 2*tmp_rows;
}

static void ggml_cuda_kvarn_store_kv_body_512_pipelined(
        const float * k_tile,
        const float * v_tile,
        uint8_t * k_body,
        uint8_t * v_body,
        float * k_scales,
        float * v_scales,
        float * scratch,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t key_bits,
        uint32_t value_bits,
        uint32_t sinkhorn_iters,
        float rtn_quantile,
        void * stream) {
    const size_t n = size_t(head_dim)*group_size;
    const uint32_t tmp_rows = head_dim > group_size ? head_dim : group_size;
    const size_t per_pipeline = kvarn_store_scratch_floats_one(head_dim, group_size);
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    float * k_data      = scratch;
    float * k_rtn_scale = scratch + n;
    float * k_rtn_zp    = scratch + n + tmp_rows;

    float * v_data      = scratch + per_pipeline;
    float * v_rtn_scale = v_data + n;
    float * v_rtn_zp    = v_data + n + tmp_rows;

    if (rtn_quantile < 1.0f || !kvarn_fullrange_packer_overwrites_body(group_size, key_bits)) {
        cudaMemsetAsync(k_body, 0, kvarn_packed_nbytes(n, key_bits), cuda_stream);
    }
    if (rtn_quantile < 1.0f || !kvarn_fullrange_packer_overwrites_body(head_dim, value_bits)) {
        cudaMemsetAsync(v_body, 0, kvarn_packed_nbytes(n, value_bits), cuda_stream);
    }

    kvarn_aux_cuda_state & aux = kvarn_aux_cuda_state_get();
    cudaStream_t aux_stream = aux.stream;

    // k_tile/v_tile are produced by prior kernels on cuda_stream. The aux stream
    // must not read them until those writes are visible.
    cudaEventRecord(aux.main_ready, cuda_stream);
    cudaStreamWaitEvent(aux_stream, aux.main_ready, 0);

    const int k_hadamard_block = kvarn_pow2_block(head_dim);
    if (k_hadamard_block <= 1024) {
        kvarn_hadamard_cols_parallel_kernel<<<int(group_size), k_hadamard_block, size_t(k_hadamard_block)*sizeof(float), cuda_stream>>>(
                k_tile, k_data, head_dim, group_size);
    } else {
        kvarn_hadamard_cols_kernel<<<int(group_size), 1, 0, cuda_stream>>>(k_tile, k_data, head_dim, group_size);
    }

    const int v_hadamard_block = kvarn_pow2_block(head_dim);
    if (v_hadamard_block <= 1024) {
        kvarn_hadamard_rows_parallel_kernel<<<int(group_size), v_hadamard_block, size_t(v_hadamard_block)*sizeof(float), aux_stream>>>(
                v_tile, v_data, group_size, head_dim);
    } else {
        kvarn_hadamard_rows_kernel<<<int(group_size), 1, 0, aux_stream>>>(v_tile, v_data, group_size, head_dim);
    }

    float * k_row_scale = k_scales;
    float * k_col_scale = k_scales + 2*head_dim;
    float * v_col_scale = v_scales;
    float * v_row_scale = v_scales + head_dim;

    kvarn_sinkhorn_variance_normalize_parallel(
            k_data, k_row_scale, k_col_scale, head_dim, group_size, sinkhorn_iters, cuda_stream);
    kvarn_sinkhorn_variance_normalize_parallel(
            v_data, v_row_scale, v_col_scale, group_size, head_dim, sinkhorn_iters, aux_stream);

    if (rtn_quantile >= 1.0f) {
        const int k_quant_block = kvarn_pow2_block(group_size);
        if (k_quant_block <= 1024) {
            kvarn_quantize_asym_fullrange_pack_rows_parallel_kernel<<<int(head_dim), k_quant_block, size_t(k_quant_block)*2*sizeof(float), cuda_stream>>>(
                    k_data, k_body, k_rtn_scale, k_rtn_zp, head_dim, group_size, key_bits);
        } else {
            kvarn_quantize_asym_fullrange_pack_rows_kernel<<<int(head_dim), 1, 0, cuda_stream>>>(
                    k_data, k_body, k_rtn_scale, k_rtn_zp, head_dim, group_size, key_bits);
        }

        const int v_quant_block = kvarn_pow2_block(head_dim);
        if (v_quant_block <= 1024) {
            kvarn_quantize_asym_fullrange_pack_rows_parallel_kernel<<<int(group_size), v_quant_block, size_t(v_quant_block)*2*sizeof(float), aux_stream>>>(
                    v_data, v_body, v_rtn_scale, v_rtn_zp, group_size, head_dim, value_bits);
        } else {
            kvarn_quantize_asym_fullrange_pack_rows_kernel<<<int(group_size), 1, 0, aux_stream>>>(
                    v_data, v_body, v_rtn_scale, v_rtn_zp, group_size, head_dim, value_bits);
        }
    } else {
        kvarn_quantize_asym_minmax_pack_rows_kernel<<<int(head_dim), 1, 0, cuda_stream>>>(
                k_data, k_body, k_rtn_scale, k_rtn_zp, head_dim, group_size, key_bits, rtn_quantile);
        kvarn_quantize_asym_minmax_pack_rows_kernel<<<int(group_size), 1, 0, aux_stream>>>(
                v_data, v_body, v_rtn_scale, v_rtn_zp, group_size, head_dim, value_bits, rtn_quantile);
    }

    const int block = 128;
    kvarn_store_k_finalize_scales_kernel<<<int((head_dim + block - 1)/block), block, 0, cuda_stream>>>(
            k_scales, k_rtn_scale, k_rtn_zp, head_dim);
    kvarn_store_v_finalize_scales_kernel<<<int((group_size + block - 1)/block), block, 0, aux_stream>>>(
            v_scales, v_rtn_scale, v_rtn_zp, head_dim, group_size);

    cudaEventRecord(aux.aux_done, aux_stream);
    cudaStreamWaitEvent(cuda_stream, aux.aux_done, 0);
}

static __global__ void kvarn_transpose_pending_k_head_kernel(
        const float * __restrict__ pending,
        float * __restrict__ k_tile,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t pending_head_stride) {
    const size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const size_t n = size_t(head_dim)*group_size;
    if (i >= n) {
        return;
    }

    const uint32_t g = uint32_t(i/head_dim);
    const uint32_t d = uint32_t(i%head_dim);
    k_tile[i] = pending[size_t(d) + size_t(g)*pending_head_stride];
}

static __global__ void kvarn_gather_pending_v_head_kernel(
        const float * __restrict__ pending,
        float * __restrict__ v_tile,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t pending_head_stride) {
    const size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const size_t n = size_t(head_dim)*group_size;
    if (i >= n) {
        return;
    }

    const uint32_t g = uint32_t(i/head_dim);
    const uint32_t d = uint32_t(i%head_dim);
    v_tile[i] = pending[size_t(d) + size_t(g)*pending_head_stride];
}

void ggml_cuda_kvarn_store_body_pending_records_minmax(
        const float * pending_k,
        const float * pending_v,
        uint8_t * k_body,
        uint8_t * v_body,
        float * k_scales,
        float * v_scales,
        float * scratch,
        const int32_t * records,
        uint32_t n_record_batch,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t key_bits,
        uint32_t value_bits,
        uint32_t sinkhorn_iters,
        float rtn_quantile,
        size_t k_body_record_stride_bytes,
        size_t v_body_record_stride_bytes,
        size_t k_body_head_stride_bytes,
        size_t v_body_head_stride_bytes,
        size_t k_scale_record_stride_floats,
        size_t v_scale_record_stride_floats,
        size_t k_scale_head_stride_floats,
        size_t v_scale_head_stride_floats,
        size_t pending_k_head_stride_floats,
        size_t pending_v_head_stride_floats,
        void * stream) {
    const size_t tile_floats = size_t(head_dim)*group_size;
    float * k_tile = scratch;
    float * v_tile = scratch + tile_floats;
    float * pipeline = scratch + 2*tile_floats;
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    const int block = 256;
    const int grid = int((tile_floats + block - 1)/block);

    const float * k_pending = pending_k;
    const float * v_pending = pending_v;
    kvarn_transpose_pending_k_head_kernel<<<grid, block, 0, cuda_stream>>>(
            k_pending, k_tile, head_dim, group_size, uint32_t(pending_k_head_stride_floats));
    kvarn_gather_pending_v_head_kernel<<<grid, block, 0, cuda_stream>>>(
            v_pending, v_tile, head_dim, group_size, uint32_t(pending_v_head_stride_floats));

    for (uint32_t bi = 0; bi < n_record_batch; ++bi) {
        const uint32_t record = uint32_t(records[bi]);
        if (head_dim >= 512) {
            ggml_cuda_kvarn_store_kv_body_512_pipelined(
                    k_tile, v_tile,
                    k_body + size_t(record)*k_body_record_stride_bytes,
                    v_body + size_t(record)*v_body_record_stride_bytes,
                    k_scales + size_t(record)*k_scale_record_stride_floats,
                    v_scales + size_t(record)*v_scale_record_stride_floats,
                    pipeline,
                    head_dim, group_size, key_bits, value_bits, sinkhorn_iters, rtn_quantile, stream);
        } else {
            ggml_cuda_kvarn_store_k_body_reference_minmax(
                    k_tile,
                    k_body + size_t(record)*k_body_record_stride_bytes,
                    k_scales + size_t(record)*k_scale_record_stride_floats,
                    pipeline,
                    head_dim, group_size, key_bits, sinkhorn_iters, rtn_quantile, stream);
            ggml_cuda_kvarn_store_v_body_reference_minmax(
                    v_tile,
                    v_body + size_t(record)*v_body_record_stride_bytes,
                    v_scales + size_t(record)*v_scale_record_stride_floats,
                    pipeline,
                    head_dim, group_size, value_bits, sinkhorn_iters, rtn_quantile, stream);
        }
    }
}

void ggml_cuda_kvarn_store_body_pending_heads_minmax(
        const float * pending_k,
        const float * pending_v,
        uint8_t * k_body,
        uint8_t * v_body,
        float * k_scales,
        float * v_scales,
        float * scratch,
        uint32_t n_heads,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t key_bits,
        uint32_t value_bits,
        uint32_t sinkhorn_iters,
        float rtn_quantile,
        size_t k_body_stride_bytes,
        size_t v_body_stride_bytes,
        size_t k_scale_stride_floats,
        size_t v_scale_stride_floats,
        size_t pending_k_head_stride_floats,
        size_t pending_v_head_stride_floats,
        void * stream) {
    const size_t tile_floats = size_t(head_dim)*group_size;
    const size_t per_pipeline = kvarn_store_scratch_floats_one(head_dim, group_size);
    float * k_tile = scratch;
    float * v_tile = scratch + tile_floats;
    float * pipeline = scratch + 2*tile_floats;
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    const int block = 256;
    const int grid = int((tile_floats + block - 1)/block);

    for (uint32_t ih = 0; ih < n_heads; ++ih) {
        const float * k_pending = pending_k + size_t(ih)*pending_k_head_stride_floats;
        const float * v_pending = pending_v + size_t(ih)*pending_v_head_stride_floats;
        kvarn_transpose_pending_k_head_kernel<<<grid, block, 0, cuda_stream>>>(
                k_pending, k_tile, head_dim, group_size, uint32_t(pending_k_head_stride_floats));
        kvarn_gather_pending_v_head_kernel<<<grid, block, 0, cuda_stream>>>(
                v_pending, v_tile, head_dim, group_size, uint32_t(pending_v_head_stride_floats));

        if (head_dim >= 512) {
            ggml_cuda_kvarn_store_kv_body_512_pipelined(
                    k_tile, v_tile,
                    k_body + size_t(ih)*k_body_stride_bytes,
                    v_body + size_t(ih)*v_body_stride_bytes,
                    k_scales + size_t(ih)*k_scale_stride_floats,
                    v_scales + size_t(ih)*v_scale_stride_floats,
                    pipeline,
                    head_dim, group_size, key_bits, value_bits, sinkhorn_iters, rtn_quantile, stream);
        } else {
            ggml_cuda_kvarn_store_k_body_reference_minmax(
                    k_tile,
                    k_body + size_t(ih)*k_body_stride_bytes,
                    k_scales + size_t(ih)*k_scale_stride_floats,
                    pipeline,
                    head_dim, group_size, key_bits, sinkhorn_iters, rtn_quantile, stream);
            ggml_cuda_kvarn_store_v_body_reference_minmax(
                    v_tile,
                    v_body + size_t(ih)*v_body_stride_bytes,
                    v_scales + size_t(ih)*v_scale_stride_floats,
                    pipeline,
                    head_dim, group_size, value_bits, sinkhorn_iters, rtn_quantile, stream);
        }
    }
}

void ggml_cuda_kvarn_store_body_reference_minmax(
        const float * k_tile,
        const float * v_tile,
        uint8_t * k_body,
        uint8_t * v_body,
        float * k_scales,
        float * v_scales,
        float * scratch,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t key_bits,
        uint32_t value_bits,
        uint32_t sinkhorn_iters,
        float rtn_quantile,
        void * stream) {
    if (head_dim >= 512) {
        ggml_cuda_kvarn_store_kv_body_512_pipelined(
                k_tile, v_tile, k_body, v_body, k_scales, v_scales, scratch,
                head_dim, group_size, key_bits, value_bits, sinkhorn_iters, rtn_quantile, stream);
        return;
    }

    ggml_cuda_kvarn_store_k_body_reference_minmax(
            k_tile, k_body, k_scales, scratch,
            head_dim, group_size, key_bits, sinkhorn_iters, rtn_quantile, stream);
    ggml_cuda_kvarn_store_v_body_reference_minmax(
            v_tile, v_body, v_scales, scratch,
            head_dim, group_size, value_bits, sinkhorn_iters, rtn_quantile, stream);
}

static __global__ void kvarn_dequant_kernel(
        const uint8_t * __restrict__ k_body,
        const uint8_t * __restrict__ v_body,
        const float * __restrict__ k_scales,
        const float * __restrict__ v_scales,
        float * __restrict__ k_out,
        float * __restrict__ v_out,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t key_bits,
        uint32_t value_bits,
        size_t n) {
    const size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const uint32_t d_k = i / group_size;
    const uint32_t g_k = i - size_t(d_k)*group_size;

    const float * k_s_col = k_scales;
    const float * k_zp    = k_scales + head_dim;
    const float * k_s_row = k_scales + 2*head_dim;

    const uint32_t kq = kvarn_unpack_one(k_body, key_bits, i);
    k_out[i] = (float(kq)*k_s_col[d_k] + k_zp[d_k])*k_s_row[g_k];

    const uint32_t g_v = i / head_dim;
    const uint32_t d_v = i - size_t(g_v)*head_dim;

    const float * v_s_col = v_scales;
    const float * v_s_row = v_scales + head_dim;
    const float * v_zp    = v_scales + head_dim + group_size;

    const uint32_t vq = kvarn_unpack_one(v_body, value_bits, i);
    v_out[i] = (float(vq)*v_s_row[g_v] + v_zp[g_v])*v_s_col[d_v];
}

void ggml_cuda_kvarn_dequant_body(
        const uint8_t * k_body,
        const uint8_t * v_body,
        const float * k_scales,
        const float * v_scales,
        float * k_out,
        float * v_out,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t key_bits,
        uint32_t value_bits,
        void * stream) {
    const size_t n = size_t(head_dim)*group_size;
    const int block = 256;
    const int grid = int((n + block - 1)/block);

    kvarn_dequant_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
            k_body, v_body, k_scales, v_scales, k_out, v_out,
            head_dim, group_size, key_bits, value_bits, n);
}

static __global__ void kvarn_dequant_n_kernel(
        const uint8_t * __restrict__ k_body,
        const uint8_t * __restrict__ v_body,
        const float * __restrict__ k_scales,
        const float * __restrict__ v_scales,
        float * __restrict__ k_out,
        float * __restrict__ v_out,
        uint32_t n_records,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t key_bits,
        uint32_t value_bits,
        size_t k_body_stride_bytes,
        size_t v_body_stride_bytes,
        size_t k_scale_stride_floats,
        size_t v_scale_stride_floats,
        size_t k_out_stride_floats,
        size_t v_out_stride_floats,
        size_t n_per_record,
        uint32_t k_token_major) {
    const size_t i_all = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const size_t n_total = size_t(n_records)*n_per_record;
    if (i_all >= n_total) {
        return;
    }

    const uint32_t r = uint32_t(i_all / n_per_record);
    const size_t i = i_all - size_t(r)*n_per_record;

    const uint8_t * k_record = k_body + size_t(r)*k_body_stride_bytes;
    const uint8_t * v_record = v_body + size_t(r)*v_body_stride_bytes;
    const float * k_record_scales = k_scales + size_t(r)*k_scale_stride_floats;
    const float * v_record_scales = v_scales + size_t(r)*v_scale_stride_floats;
    float * k_record_out = k_out + size_t(r)*k_out_stride_floats;
    float * v_record_out = v_out + size_t(r)*v_out_stride_floats;

    // The persistent packed K layout is always record/dim-major
    // (i_packed = d*group_size + g). When k_token_major is set, the f32
    // output index i is interpreted as token-major (i = g*head_dim + d) so
    // attention kernels get coalesced per-lane K reads; the packed source
    // index is recomputed accordingly. When clear, output matches the
    // packed layout (i_packed == i), preserving legacy behavior.
    uint32_t d_k, g_k;
    if (k_token_major != 0) {
        g_k = uint32_t(i / head_dim);
        d_k = uint32_t(i - size_t(g_k)*head_dim);
    } else {
        d_k = uint32_t(i / group_size);
        g_k = uint32_t(i - size_t(d_k)*group_size);
    }
    const size_t i_packed_k = size_t(d_k)*group_size + g_k;

    const float * k_s_col = k_record_scales;
    const float * k_zp    = k_record_scales + head_dim;
    const float * k_s_row = k_record_scales + 2*head_dim;

    const uint32_t kq = kvarn_unpack_one(k_record, key_bits, i_packed_k);
    k_record_out[i] = (float(kq)*k_s_col[d_k] + k_zp[d_k])*k_s_row[g_k];

    const uint32_t g_v = i / head_dim;
    const uint32_t d_v = i - size_t(g_v)*head_dim;

    const float * v_s_col = v_record_scales;
    const float * v_s_row = v_record_scales + head_dim;
    const float * v_zp    = v_record_scales + head_dim + group_size;

    const uint32_t vq = kvarn_unpack_one(v_record, value_bits, i);
    v_record_out[i] = (float(vq)*v_s_row[g_v] + v_zp[g_v])*v_s_col[d_v];
}

void ggml_cuda_kvarn_dequant_body_n(
        const uint8_t * k_body,
        const uint8_t * v_body,
        const float * k_scales,
        const float * v_scales,
        float * k_out,
        float * v_out,
        uint32_t n_records,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t key_bits,
        uint32_t value_bits,
        size_t k_body_stride_bytes,
        size_t v_body_stride_bytes,
        size_t k_scale_stride_floats,
        size_t v_scale_stride_floats,
        size_t k_out_stride_floats,
        size_t v_out_stride_floats,
        void * stream) {
    const size_t n_per_record = size_t(head_dim)*group_size;
    const size_t n_total = size_t(n_records)*n_per_record;
    const int block = 256;
    const int grid = int((n_total + block - 1)/block);

    kvarn_dequant_n_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
            k_body, v_body, k_scales, v_scales, k_out, v_out,
            n_records, head_dim, group_size, key_bits, value_bits,
            k_body_stride_bytes, v_body_stride_bytes,
            k_scale_stride_floats, v_scale_stride_floats,
            k_out_stride_floats, v_out_stride_floats, n_per_record, 0u);
}

// Same as ggml_cuda_kvarn_dequant_body_n but writes the K f32 output in
// token-major layout (g*head_dim + d), matching the V layout, so that
// attention kernels can issue coalesced per-lane K loads. The packed
// persistent K layout is unchanged. V output layout is identical in both
// variants.
void ggml_cuda_kvarn_dequant_body_n_k_token_major(
        const uint8_t * k_body,
        const uint8_t * v_body,
        const float * k_scales,
        const float * v_scales,
        float * k_out,
        float * v_out,
        uint32_t n_records,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t key_bits,
        uint32_t value_bits,
        size_t k_body_stride_bytes,
        size_t v_body_stride_bytes,
        size_t k_scale_stride_floats,
        size_t v_scale_stride_floats,
        size_t k_out_stride_floats,
        size_t v_out_stride_floats,
        void * stream) {
    const size_t n_per_record = size_t(head_dim)*group_size;
    const size_t n_total = size_t(n_records)*n_per_record;
    const int block = 256;
    const int grid = int((n_total + block - 1)/block);

    kvarn_dequant_n_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
            k_body, v_body, k_scales, v_scales, k_out, v_out,
            n_records, head_dim, group_size, key_bits, value_bits,
            k_body_stride_bytes, v_body_stride_bytes,
            k_scale_stride_floats, v_scale_stride_floats,
            k_out_stride_floats, v_out_stride_floats, n_per_record, 1u);
}

static __global__ void kvarn_qk_body_kernel(
        const float * __restrict__ q,
        const uint8_t * __restrict__ k_body,
        const float * __restrict__ k_scales,
        float * __restrict__ scores,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t key_bits,
        float scale) {
    const uint32_t g = blockIdx.x;
    if (g >= group_size) {
        return;
    }

    extern __shared__ float tmp[];

    const float * k_s_col = k_scales;
    const float * k_zp    = k_scales + head_dim;
    const float * k_s_row = k_scales + 2*head_dim;

    float sum = 0.0f;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const size_t i = size_t(d)*group_size + g;
        const uint32_t kq = kvarn_unpack_one(k_body, key_bits, i);
        const float k = (float(kq)*k_s_col[d] + k_zp[d])*k_s_row[g];
        sum += q[d]*k;
    }

    tmp[threadIdx.x] = sum;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            tmp[threadIdx.x] += tmp[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        scores[g] = tmp[0]*scale;
    }
}

void ggml_cuda_kvarn_qk_body(
        const float * q,
        const uint8_t * k_body,
        const float * k_scales,
        float * scores,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t key_bits,
        float scale,
        void * stream) {
    int block = 1;
    while (block < int(head_dim)) {
        block <<= 1;
    }
    block = block > 256 ? 256 : block;

    kvarn_qk_body_kernel<<<group_size, block, block*sizeof(float), static_cast<cudaStream_t>(stream)>>>(
            q, k_body, k_scales, scores, head_dim, group_size, key_bits, scale);
}

static __global__ void kvarn_av_body_kernel(
        const float * __restrict__ probs,
        const uint8_t * __restrict__ v_body,
        const float * __restrict__ v_scales,
        float * __restrict__ out,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t value_bits) {
    const uint32_t d = blockIdx.x*blockDim.x + threadIdx.x;
    if (d >= head_dim) {
        return;
    }

    const float * v_s_col = v_scales;
    const float * v_s_row = v_scales + head_dim;
    const float * v_zp    = v_scales + head_dim + group_size;

    float sum = 0.0f;
    for (uint32_t g = 0; g < group_size; ++g) {
        const size_t i = size_t(g)*head_dim + d;
        const uint32_t vq = kvarn_unpack_one(v_body, value_bits, i);
        const float v = (float(vq)*v_s_row[g] + v_zp[g])*v_s_col[d];
        sum += probs[g]*v;
    }

    out[d] = sum;
}

void ggml_cuda_kvarn_av_body(
        const float * probs,
        const uint8_t * v_body,
        const float * v_scales,
        float * out,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t value_bits,
        void * stream) {
    const int block = 128;
    const int grid = int((head_dim + block - 1)/block);

    kvarn_av_body_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
            probs, v_body, v_scales, out, head_dim, group_size, value_bits);
}

static __global__ void kvarn_attn_scores_softmax_kernel(
        const float * __restrict__ q,
        const uint8_t * __restrict__ k_body,
        const float * __restrict__ k_scales,
        float * __restrict__ probs,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t key_bits,
        float scale) {
    extern __shared__ float tmp[];
    float * scores = tmp;
    float * reduce = tmp + group_size;

    const float * k_s_col = k_scales;
    const float * k_zp    = k_scales + head_dim;
    const float * k_s_row = k_scales + 2*head_dim;

    for (uint32_t g = threadIdx.x; g < group_size; g += blockDim.x) {
        float sum = 0.0f;
        for (uint32_t d = 0; d < head_dim; ++d) {
            const size_t i = size_t(d)*group_size + g;
            const uint32_t kq = kvarn_unpack_one(k_body, key_bits, i);
            const float k = (float(kq)*k_s_col[d] + k_zp[d])*k_s_row[g];
            sum += q[d]*k;
        }
        scores[g] = sum*scale;
    }
    __syncthreads();

    float local_max = -3.4028234663852886e38f;
    for (uint32_t g = threadIdx.x; g < group_size; g += blockDim.x) {
        local_max = fmaxf(local_max, scores[g]);
    }
    reduce[threadIdx.x] = local_max;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float max_score = reduce[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (uint32_t g = threadIdx.x; g < group_size; g += blockDim.x) {
        const float p = expf(scores[g] - max_score);
        probs[g] = p;
        local_sum += p;
    }
    reduce[threadIdx.x] = local_sum;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_denom = 1.0f/reduce[0];

    for (uint32_t g = threadIdx.x; g < group_size; g += blockDim.x) {
        probs[g] *= inv_denom;
    }
}

void ggml_cuda_kvarn_attn_body(
        const float * q,
        const uint8_t * k_body,
        const uint8_t * v_body,
        const float * k_scales,
        const float * v_scales,
        float * out,
        float * scores,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t key_bits,
        uint32_t value_bits,
        float scale,
        void * stream) {
    int block = 1;
    while (block < int(group_size)) {
        block <<= 1;
    }
    block = block > 256 ? 256 : block;

    const size_t shmem = (size_t(group_size) + size_t(block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
    auto cuda_stream = static_cast<cudaStream_t>(stream);
    kvarn_attn_scores_softmax_kernel<<<1, block, shmem, cuda_stream>>>(
            q, k_body, k_scales, scores, head_dim, group_size, key_bits, scale);

    const int av_block = 128;
    const int av_grid = int((head_dim + av_block - 1)/av_block);
    kvarn_av_body_kernel<<<av_grid, av_block, 0, cuda_stream>>>(
            scores, v_body, v_scales, out, head_dim, group_size, value_bits);
}

static __global__ void kvarn_attn_scores_softmax_n_kernel(
        const float * __restrict__ q,
        const uint8_t * __restrict__ k_body,
        const float * __restrict__ k_scales,
        float * __restrict__ probs,
        uint32_t n_records,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t key_bits,
        size_t k_body_stride_bytes,
        size_t k_scale_stride_floats,
        float scale) {
    extern __shared__ float tmp[];
    float * scores = tmp;
    float * reduce = tmp + n_records*group_size;

    const uint32_t n_tokens = n_records*group_size;

    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        const uint32_t r = t / group_size;
        const uint32_t g = t - r*group_size;

        const uint8_t * k_record = k_body + size_t(r)*k_body_stride_bytes;
        const float * k_record_scales = k_scales + size_t(r)*k_scale_stride_floats;
        const float * k_s_col = k_record_scales;
        const float * k_zp    = k_record_scales + head_dim;
        const float * k_s_row = k_record_scales + 2*head_dim;

        float sum = 0.0f;
        for (uint32_t d = 0; d < head_dim; ++d) {
            const size_t i = size_t(d)*group_size + g;
            const uint32_t kq = kvarn_unpack_one(k_record, key_bits, i);
            const float k = (float(kq)*k_s_col[d] + k_zp[d])*k_s_row[g];
            sum += q[d]*k;
        }
        scores[t] = sum*scale;
    }
    __syncthreads();

    float local_max = -3.4028234663852886e38f;
    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        local_max = fmaxf(local_max, scores[t]);
    }
    reduce[threadIdx.x] = local_max;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float max_score = reduce[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        const float p = expf(scores[t] - max_score);
        probs[t] = p;
        local_sum += p;
    }
    reduce[threadIdx.x] = local_sum;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_denom = 1.0f/reduce[0];

    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        probs[t] *= inv_denom;
    }
}

static __global__ void kvarn_av_body_n_kernel(
        const float * __restrict__ probs,
        const uint8_t * __restrict__ v_body,
        const float * __restrict__ v_scales,
        float * __restrict__ out,
        uint32_t n_records,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t value_bits,
        size_t v_body_stride_bytes,
        size_t v_scale_stride_floats) {
    const uint32_t d = blockIdx.x*blockDim.x + threadIdx.x;
    if (d >= head_dim) {
        return;
    }

    float sum = 0.0f;
    for (uint32_t r = 0; r < n_records; ++r) {
        const uint8_t * v_record = v_body + size_t(r)*v_body_stride_bytes;
        const float * v_record_scales = v_scales + size_t(r)*v_scale_stride_floats;
        const float * v_s_col = v_record_scales;
        const float * v_s_row = v_record_scales + head_dim;
        const float * v_zp    = v_record_scales + head_dim + group_size;

        for (uint32_t g = 0; g < group_size; ++g) {
            const size_t i = size_t(g)*head_dim + d;
            const uint32_t vq = kvarn_unpack_one(v_record, value_bits, i);
            const float v = (float(vq)*v_s_row[g] + v_zp[g])*v_s_col[d];
            sum += probs[size_t(r)*group_size + g]*v;
        }
    }

    out[d] = sum;
}

void ggml_cuda_kvarn_attn_body_n(
        const float * q,
        const uint8_t * k_body,
        const uint8_t * v_body,
        const float * k_scales,
        const float * v_scales,
        float * out,
        float * scores,
        uint32_t n_records,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t key_bits,
        uint32_t value_bits,
        size_t k_body_stride_bytes,
        size_t v_body_stride_bytes,
        size_t k_scale_stride_floats,
        size_t v_scale_stride_floats,
        float scale,
        void * stream) {
    const uint32_t n_tokens = n_records*group_size;

    int block = 1;
    while (block < int(n_tokens)) {
        block <<= 1;
    }
    block = block > 256 ? 256 : block;

    const size_t shmem = (size_t(n_tokens) + size_t(block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
    auto cuda_stream = static_cast<cudaStream_t>(stream);
    kvarn_attn_scores_softmax_n_kernel<<<1, block, shmem, cuda_stream>>>(
            q, k_body, k_scales, scores, n_records, head_dim, group_size, key_bits,
            k_body_stride_bytes, k_scale_stride_floats, scale);

    const int av_block = 128;
    const int av_grid = int((head_dim + av_block - 1)/av_block);
    kvarn_av_body_n_kernel<<<av_grid, av_block, 0, cuda_stream>>>(
            scores, v_body, v_scales, out, n_records, head_dim, group_size, value_bits,
            v_body_stride_bytes, v_scale_stride_floats);
}

static __global__ void kvarn_attn_scores_softmax_n_batch_kernel(
        const float * __restrict__ q,
        const uint8_t * __restrict__ k_body,
        const float * __restrict__ k_scales,
        float * __restrict__ probs,
        uint32_t n_records,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t key_bits,
        size_t q_stride_floats,
        size_t score_stride_floats,
        size_t k_body_stride_bytes,
        size_t k_scale_stride_floats,
        float scale) {
    const uint32_t iq = blockIdx.x;
    const float * q_cur = q + size_t(iq)*q_stride_floats;
    float * probs_cur = probs + size_t(iq)*score_stride_floats;

    extern __shared__ float tmp[];
    float * scores = tmp;
    float * reduce = tmp + n_records*group_size;

    const uint32_t n_tokens = n_records*group_size;

    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        const uint32_t r = t / group_size;
        const uint32_t g = t - r*group_size;

        const uint8_t * k_record = k_body + size_t(r)*k_body_stride_bytes;
        const float * k_record_scales = k_scales + size_t(r)*k_scale_stride_floats;
        const float * k_s_col = k_record_scales;
        const float * k_zp    = k_record_scales + head_dim;
        const float * k_s_row = k_record_scales + 2*head_dim;

        float sum = 0.0f;
        for (uint32_t d = 0; d < head_dim; ++d) {
            const size_t i = size_t(d)*group_size + g;
            const uint32_t kq = kvarn_unpack_one(k_record, key_bits, i);
            const float k = (float(kq)*k_s_col[d] + k_zp[d])*k_s_row[g];
            sum += q_cur[d]*k;
        }
        scores[t] = sum*scale;
    }
    __syncthreads();

    float local_max = -3.4028234663852886e38f;
    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        local_max = fmaxf(local_max, scores[t]);
    }
    reduce[threadIdx.x] = local_max;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float max_score = reduce[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        const float p = expf(scores[t] - max_score);
        probs_cur[t] = p;
        local_sum += p;
    }
    reduce[threadIdx.x] = local_sum;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_denom = 1.0f/reduce[0];

    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        probs_cur[t] *= inv_denom;
    }
}

static __global__ void kvarn_av_body_n_batch_kernel(
        const float * __restrict__ probs,
        const uint8_t * __restrict__ v_body,
        const float * __restrict__ v_scales,
        float * __restrict__ out,
        uint32_t n_records,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t value_bits,
        size_t out_stride_floats,
        size_t score_stride_floats,
        size_t v_body_stride_bytes,
        size_t v_scale_stride_floats) {
    const uint32_t d = blockIdx.x*blockDim.x + threadIdx.x;
    const uint32_t iq = blockIdx.y;
    if (d >= head_dim) {
        return;
    }

    const float * probs_cur = probs + size_t(iq)*score_stride_floats;

    float sum = 0.0f;
    for (uint32_t r = 0; r < n_records; ++r) {
        const uint8_t * v_record = v_body + size_t(r)*v_body_stride_bytes;
        const float * v_record_scales = v_scales + size_t(r)*v_scale_stride_floats;
        const float * v_s_col = v_record_scales;
        const float * v_s_row = v_record_scales + head_dim;
        const float * v_zp    = v_record_scales + head_dim + group_size;

        for (uint32_t g = 0; g < group_size; ++g) {
            const size_t i = size_t(g)*head_dim + d;
            const uint32_t vq = kvarn_unpack_one(v_record, value_bits, i);
            const float v = (float(vq)*v_s_row[g] + v_zp[g])*v_s_col[d];
            sum += probs_cur[size_t(r)*group_size + g]*v;
        }
    }

    out[size_t(iq)*out_stride_floats + d] = sum;
}

void ggml_cuda_kvarn_attn_body_n_batch(
        const float * q,
        const uint8_t * k_body,
        const uint8_t * v_body,
        const float * k_scales,
        const float * v_scales,
        float * out,
        float * scores,
        uint32_t n_queries,
        uint32_t n_records,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t key_bits,
        uint32_t value_bits,
        size_t q_stride_floats,
        size_t out_stride_floats,
        size_t score_stride_floats,
        size_t k_body_stride_bytes,
        size_t v_body_stride_bytes,
        size_t k_scale_stride_floats,
        size_t v_scale_stride_floats,
        float scale,
        void * stream) {
    const uint32_t n_tokens = n_records*group_size;

    int block = 1;
    while (block < int(n_tokens)) {
        block <<= 1;
    }
    block = block > 256 ? 256 : block;

    const size_t shmem = (size_t(n_tokens) + size_t(block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
    auto cuda_stream = static_cast<cudaStream_t>(stream);
    kvarn_attn_scores_softmax_n_batch_kernel<<<n_queries, block, shmem, cuda_stream>>>(
            q, k_body, k_scales, scores, n_records, head_dim, group_size, key_bits,
            q_stride_floats, score_stride_floats, k_body_stride_bytes, k_scale_stride_floats, scale);

    const int av_block = 128;
    const dim3 av_grid(int((head_dim + av_block - 1)/av_block), n_queries, 1);
    kvarn_av_body_n_batch_kernel<<<av_grid, av_block, 0, cuda_stream>>>(
            scores, v_body, v_scales, out, n_records, head_dim, group_size, value_bits,
            out_stride_floats, score_stride_floats, v_body_stride_bytes, v_scale_stride_floats);
}

static __global__ void kvarn_attn_mixed_scores_softmax_kernel(
        const float * __restrict__ q,
        const float * __restrict__ sink_k,
        const uint8_t * __restrict__ k_body,
        const float * __restrict__ k_scales,
        const float * __restrict__ tail_k,
        float * __restrict__ probs,
        uint32_t n_sink,
        uint32_t n_records,
        uint32_t n_tail,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t key_bits,
        size_t k_body_stride_bytes,
        size_t k_scale_stride_floats,
        float scale) {
    extern __shared__ float tmp[];
    const uint32_t n_body_tokens = n_records*group_size;
    const uint32_t n_tokens = n_sink + n_body_tokens + n_tail;
    float * scores = tmp;
    float * reduce = tmp + n_tokens;

    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        float sum = 0.0f;

        if (t < n_sink) {
            const float * k = sink_k + size_t(t)*head_dim;
            for (uint32_t d = 0; d < head_dim; ++d) {
                sum += q[d]*k[d];
            }
        } else if (t < n_sink + n_body_tokens) {
            const uint32_t body_t = t - n_sink;
            const uint32_t r = body_t / group_size;
            const uint32_t g = body_t - r*group_size;

            const uint8_t * k_record = k_body + size_t(r)*k_body_stride_bytes;
            const float * k_record_scales = k_scales + size_t(r)*k_scale_stride_floats;
            const float * k_s_col = k_record_scales;
            const float * k_zp    = k_record_scales + head_dim;
            const float * k_s_row = k_record_scales + 2*head_dim;

            for (uint32_t d = 0; d < head_dim; ++d) {
                const size_t i = size_t(d)*group_size + g;
                const uint32_t kq = kvarn_unpack_one(k_record, key_bits, i);
                const float k = (float(kq)*k_s_col[d] + k_zp[d])*k_s_row[g];
                sum += q[d]*k;
            }
        } else {
            const uint32_t tail_t = t - n_sink - n_body_tokens;
            const float * k = tail_k + size_t(tail_t)*head_dim;
            for (uint32_t d = 0; d < head_dim; ++d) {
                sum += q[d]*k[d];
            }
        }

        scores[t] = sum*scale;
    }
    __syncthreads();

    float local_max = -3.4028234663852886e38f;
    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        local_max = fmaxf(local_max, scores[t]);
    }
    reduce[threadIdx.x] = local_max;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float max_score = reduce[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        const float p = expf(scores[t] - max_score);
        probs[t] = p;
        local_sum += p;
    }
    reduce[threadIdx.x] = local_sum;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_denom = 1.0f/reduce[0];

    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        probs[t] *= inv_denom;
    }
}

static __global__ void kvarn_attn_mixed_av_kernel(
        const float * __restrict__ probs,
        const float * __restrict__ sink_v,
        const uint8_t * __restrict__ v_body,
        const float * __restrict__ v_scales,
        const float * __restrict__ tail_v,
        float * __restrict__ out,
        uint32_t n_sink,
        uint32_t n_records,
        uint32_t n_tail,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t value_bits,
        size_t v_body_stride_bytes,
        size_t v_scale_stride_floats) {
    const uint32_t d = blockIdx.x*blockDim.x + threadIdx.x;
    if (d >= head_dim) {
        return;
    }

    const uint32_t n_body_tokens = n_records*group_size;
    float sum = 0.0f;

    for (uint32_t t = 0; t < n_sink; ++t) {
        sum += probs[t]*sink_v[size_t(t)*head_dim + d];
    }

    for (uint32_t r = 0; r < n_records; ++r) {
        const uint8_t * v_record = v_body + size_t(r)*v_body_stride_bytes;
        const float * v_record_scales = v_scales + size_t(r)*v_scale_stride_floats;
        const float * v_s_col = v_record_scales;
        const float * v_s_row = v_record_scales + head_dim;
        const float * v_zp    = v_record_scales + head_dim + group_size;

        for (uint32_t g = 0; g < group_size; ++g) {
            const size_t i = size_t(g)*head_dim + d;
            const uint32_t vq = kvarn_unpack_one(v_record, value_bits, i);
            const float v = (float(vq)*v_s_row[g] + v_zp[g])*v_s_col[d];
            sum += probs[size_t(n_sink) + size_t(r)*group_size + g]*v;
        }
    }

    for (uint32_t t = 0; t < n_tail; ++t) {
        sum += probs[size_t(n_sink) + n_body_tokens + t]*tail_v[size_t(t)*head_dim + d];
    }

    out[d] = sum;
}

void ggml_cuda_kvarn_attn_mixed(
        const float * q,
        const float * sink_k,
        const float * sink_v,
        const uint8_t * k_body,
        const uint8_t * v_body,
        const float * k_scales,
        const float * v_scales,
        const float * tail_k,
        const float * tail_v,
        float * out,
        float * scores,
        uint32_t n_sink,
        uint32_t n_records,
        uint32_t n_tail,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t key_bits,
        uint32_t value_bits,
        size_t k_body_stride_bytes,
        size_t v_body_stride_bytes,
        size_t k_scale_stride_floats,
        size_t v_scale_stride_floats,
        float scale,
        void * stream) {
    const uint32_t n_tokens = n_sink + n_records*group_size + n_tail;

    int block = 1;
    while (block < int(n_tokens)) {
        block <<= 1;
    }
    block = block > 256 ? 256 : block;

    const size_t shmem = (size_t(n_tokens) + size_t(block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
    auto cuda_stream = static_cast<cudaStream_t>(stream);
    kvarn_attn_mixed_scores_softmax_kernel<<<1, block, shmem, cuda_stream>>>(
            q, sink_k, k_body, k_scales, tail_k, scores,
            n_sink, n_records, n_tail, head_dim, group_size, key_bits,
            k_body_stride_bytes, k_scale_stride_floats, scale);

    const int av_block = 128;
    const int av_grid = int((head_dim + av_block - 1)/av_block);
    kvarn_attn_mixed_av_kernel<<<av_grid, av_block, 0, cuda_stream>>>(
            scores, sink_v, v_body, v_scales, tail_v, out,
            n_sink, n_records, n_tail, head_dim, group_size, value_bits,
            v_body_stride_bytes, v_scale_stride_floats);
}

static __global__ void kvarn_attn_mixed_scratch_scores_softmax_kernel(
        const float * __restrict__ q,
        const float * __restrict__ sink_k,
        const float * __restrict__ body_k,
        const float * __restrict__ tail_k,
        float * __restrict__ probs,
        uint32_t n_sink,
        uint32_t n_records,
        uint32_t n_tail,
        uint32_t head_dim,
        uint32_t group_size,
        size_t k_body_stride_floats,
        float scale) {
    extern __shared__ float tmp[];
    const uint32_t n_body_tokens = n_records*group_size;
    const uint32_t n_tokens = n_sink + n_body_tokens + n_tail;
    float * scores = tmp;
    float * reduce = tmp + n_tokens;

    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        float sum = 0.0f;

        if (t < n_sink) {
            const float * k = sink_k + size_t(t)*head_dim;
            for (uint32_t d = 0; d < head_dim; ++d) {
                sum += q[d]*k[d];
            }
        } else if (t < n_sink + n_body_tokens) {
            const uint32_t body_t = t - n_sink;
            const uint32_t r = body_t / group_size;
            const uint32_t g = body_t - r*group_size;
            const float * k_record = body_k + size_t(r)*k_body_stride_floats;
            for (uint32_t d = 0; d < head_dim; ++d) {
                sum += q[d]*k_record[size_t(d)*group_size + g];
            }
        } else {
            const uint32_t tail_t = t - n_sink - n_body_tokens;
            const float * k = tail_k + size_t(tail_t)*head_dim;
            for (uint32_t d = 0; d < head_dim; ++d) {
                sum += q[d]*k[d];
            }
        }

        scores[t] = sum*scale;
    }
    __syncthreads();

    float local_max = -3.4028234663852886e38f;
    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        local_max = fmaxf(local_max, scores[t]);
    }
    reduce[threadIdx.x] = local_max;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float max_score = reduce[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        const float p = expf(scores[t] - max_score);
        probs[t] = p;
        local_sum += p;
    }
    reduce[threadIdx.x] = local_sum;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_denom = 1.0f/reduce[0];

    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        probs[t] *= inv_denom;
    }
}

static __global__ void kvarn_attn_mixed_scratch_av_kernel(
        const float * __restrict__ probs,
        const float * __restrict__ sink_v,
        const float * __restrict__ body_v,
        const float * __restrict__ tail_v,
        float * __restrict__ out,
        uint32_t n_sink,
        uint32_t n_records,
        uint32_t n_tail,
        uint32_t head_dim,
        uint32_t group_size,
        size_t v_body_stride_floats) {
    const uint32_t d = blockIdx.x*blockDim.x + threadIdx.x;
    if (d >= head_dim) {
        return;
    }

    const uint32_t n_body_tokens = n_records*group_size;
    float sum = 0.0f;

    for (uint32_t t = 0; t < n_sink; ++t) {
        sum += probs[t]*sink_v[size_t(t)*head_dim + d];
    }

    for (uint32_t r = 0; r < n_records; ++r) {
        const float * v_record = body_v + size_t(r)*v_body_stride_floats;
        for (uint32_t g = 0; g < group_size; ++g) {
            sum += probs[size_t(n_sink) + size_t(r)*group_size + g]*v_record[size_t(g)*head_dim + d];
        }
    }

    for (uint32_t t = 0; t < n_tail; ++t) {
        sum += probs[size_t(n_sink) + n_body_tokens + t]*tail_v[size_t(t)*head_dim + d];
    }

    out[d] = sum;
}

void ggml_cuda_kvarn_attn_mixed_f32_scratch(
        const float * q,
        const float * sink_k,
        const float * sink_v,
        const float * body_k,
        const float * body_v,
        const float * tail_k,
        const float * tail_v,
        float * out,
        float * scores,
        uint32_t n_sink,
        uint32_t n_records,
        uint32_t n_tail,
        uint32_t head_dim,
        uint32_t group_size,
        size_t k_body_stride_floats,
        size_t v_body_stride_floats,
        float scale,
        void * stream) {
    const uint32_t n_tokens = n_sink + n_records*group_size + n_tail;

    int block = 1;
    while (block < int(n_tokens)) {
        block <<= 1;
    }
    block = block > 256 ? 256 : block;

    const size_t shmem = (size_t(n_tokens) + size_t(block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
    auto cuda_stream = static_cast<cudaStream_t>(stream);
    kvarn_attn_mixed_scratch_scores_softmax_kernel<<<1, block, shmem, cuda_stream>>>(
            q, sink_k, body_k, tail_k, scores,
            n_sink, n_records, n_tail, head_dim, group_size, k_body_stride_floats, scale);

    const int av_block = 128;
    const int av_grid = int((head_dim + av_block - 1)/av_block);
    kvarn_attn_mixed_scratch_av_kernel<<<av_grid, av_block, 0, cuda_stream>>>(
            scores, sink_v, body_v, tail_v, out,
            n_sink, n_records, n_tail, head_dim, group_size, v_body_stride_floats);
}

static __global__ void kvarn_attn_mixed_f16_scratch_scores_softmax_kernel(
        const float * __restrict__ q,
        const uint16_t * __restrict__ sink_tail_k,
        const float * __restrict__ body_k,
        const float * __restrict__ pending_k,
        const void * __restrict__ kq_mask,
        float * __restrict__ scores,
        uint32_t n_sink,
        uint32_t n_records,
        uint32_t n_pending,
        uint32_t n_tail,
        uint32_t tail_start,
        uint32_t head_dim,
        uint32_t group_size,
        size_t sink_tail_stride_token_f16,
        size_t pending_stride_token_floats,
        size_t k_body_stride_record_floats,
        size_t kq_mask_stride_token_bytes,
        uint32_t kq_mask_type,
        float scale) {
    extern __shared__ float shared[];
    float * probs = shared;
    float * reduce = shared + n_sink + n_records*group_size + n_pending + n_tail;

    const uint32_t n_body_tokens = n_records*group_size;
    const uint32_t n_tokens = n_sink + n_body_tokens + n_pending + n_tail;

    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        float sum = 0.0f;

        if (t < n_sink) {
            const uint16_t * k = sink_tail_k + size_t(t)*sink_tail_stride_token_f16;
            for (uint32_t d = 0; d < head_dim; ++d) {
                sum += q[d]*__half2float(reinterpret_cast<const __half *>(k)[d]);
            }
        } else if (t < n_sink + n_body_tokens) {
            const uint32_t body_t = t - n_sink;
            const uint32_t r = body_t/group_size;
            const uint32_t g = body_t - r*group_size;
            const float * k_record = body_k + size_t(r)*k_body_stride_record_floats;
            for (uint32_t d = 0; d < head_dim; ++d) {
                sum += q[d]*k_record[size_t(d)*group_size + g];
            }
        } else if (t < n_sink + n_body_tokens + n_pending) {
            const uint32_t pending_t = t - n_sink - n_body_tokens;
            const float * k = pending_k + size_t(pending_t)*pending_stride_token_floats;
            for (uint32_t d = 0; d < head_dim; ++d) {
                sum += q[d]*k[d];
            }
        } else {
            const uint32_t tail_t = t - n_sink - n_body_tokens - n_pending;
            const uint32_t tail_slot = n_tail == 0 ? 0 : (tail_start + tail_t)%n_tail;
            const uint16_t * k = sink_tail_k + size_t(n_sink + tail_slot)*sink_tail_stride_token_f16;
            for (uint32_t d = 0; d < head_dim; ++d) {
                sum += q[d]*__half2float(reinterpret_cast<const __half *>(k)[d]);
            }
        }

        scores[t] = sum*scale + kvarn_kq_mask_bias(kq_mask, kq_mask_type, kq_mask_stride_token_bytes, t);
    }
    __syncthreads();

    float local_max = -3.4028234663852886e38f;
    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        local_max = fmaxf(local_max, scores[t]);
    }
    reduce[threadIdx.x] = local_max;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float max_score = reduce[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        const float p = expf(scores[t] - max_score);
        probs[t] = p;
        local_sum += p;
    }
    reduce[threadIdx.x] = local_sum;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_denom = 1.0f/reduce[0];

    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        scores[t] = probs[t]*inv_denom;
    }
}

static __global__ void kvarn_attn_mixed_f16_scratch_av_kernel(
        const float * __restrict__ probs,
        const uint16_t * __restrict__ sink_tail_v,
        const float * __restrict__ body_v,
        const float * __restrict__ pending_v,
        float * __restrict__ out,
        uint32_t n_sink,
        uint32_t n_records,
        uint32_t n_pending,
        uint32_t n_tail,
        uint32_t tail_start,
        uint32_t head_dim,
        uint32_t group_size,
        size_t sink_tail_stride_token_f16,
        size_t pending_stride_token_floats,
        size_t v_body_stride_record_floats) {
    const uint32_t d = blockIdx.x*blockDim.x + threadIdx.x;
    if (d >= head_dim) {
        return;
    }

    const uint32_t n_body_tokens = n_records*group_size;
    float sum = 0.0f;

    for (uint32_t t = 0; t < n_sink; ++t) {
        const uint16_t * v = sink_tail_v + size_t(t)*sink_tail_stride_token_f16;
        sum += probs[t]*__half2float(reinterpret_cast<const __half *>(v)[d]);
    }

    for (uint32_t r = 0; r < n_records; ++r) {
        const float * v_record = body_v + size_t(r)*v_body_stride_record_floats;
        for (uint32_t g = 0; g < group_size; ++g) {
            sum += probs[size_t(n_sink) + size_t(r)*group_size + g]*v_record[size_t(g)*head_dim + d];
        }
    }

    for (uint32_t t = 0; t < n_pending; ++t) {
        const float * v = pending_v + size_t(t)*pending_stride_token_floats;
        sum += probs[size_t(n_sink) + n_body_tokens + t]*v[d];
    }

    for (uint32_t t = 0; t < n_tail; ++t) {
        const uint32_t tail_slot = (tail_start + t)%n_tail;
        const uint16_t * v = sink_tail_v + size_t(n_sink + tail_slot)*sink_tail_stride_token_f16;
        sum += probs[size_t(n_sink) + n_body_tokens + n_pending + t]*__half2float(reinterpret_cast<const __half *>(v)[d]);
    }

    out[d] = sum;
}

void ggml_cuda_kvarn_attn_mixed_f16_batch_scratch(
        const float * q,
        const uint16_t * sink_tail_k,
        const uint16_t * sink_tail_v,
        const float * body_k,
        const float * body_v,
        const float * pending_k,
        const float * pending_v,
        const void * kq_mask,
        float * out,
        float * scores,
        uint32_t n_queries,
        uint32_t n_head,
        uint32_t n_head_kv,
        uint32_t n_sink,
        uint32_t n_records,
        uint32_t n_pending,
        uint32_t n_tail,
        uint32_t tail_start,
        uint32_t head_dim,
        uint32_t group_size,
        size_t q_stride_head_floats,
        size_t q_stride_query_floats,
        size_t out_stride_head_floats,
        size_t out_stride_query_floats,
        size_t sink_tail_stride_head_f16,
        size_t sink_tail_stride_token_f16,
        size_t pending_stride_head_floats,
        size_t pending_stride_token_floats,
        size_t k_body_stride_record_floats,
        size_t v_body_stride_record_floats,
        size_t k_body_stride_head_floats,
        size_t v_body_stride_head_floats,
        size_t kq_mask_stride_query_bytes,
        size_t kq_mask_stride_token_bytes,
        uint32_t kq_mask_type,
        float scale,
        void * stream) {
    const uint32_t n_tokens = n_sink + n_records*group_size + n_pending + n_tail;
    const uint32_t n_gqa = n_head/n_head_kv;

    int block = 1;
    while (block < int(n_tokens)) {
        block <<= 1;
    }
    block = block > 256 ? 256 : block;

    const size_t shmem = (size_t(n_tokens) + size_t(block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
    const int av_block = 128;
    const int av_grid = int((head_dim + av_block - 1)/av_block);
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    for (uint32_t iq = 0; iq < n_queries; ++iq) {
        for (uint32_t ih = 0; ih < n_head; ++ih) {
            const uint32_t ikh = ih/n_gqa;
            const float * q_ptr = q + size_t(iq)*q_stride_query_floats + size_t(ih)*q_stride_head_floats;
            float * out_ptr = out + size_t(iq)*out_stride_query_floats + size_t(ih)*out_stride_head_floats;
            const uint16_t * k_st_ptr = sink_tail_k + size_t(ikh)*sink_tail_stride_head_f16;
            const uint16_t * v_st_ptr = sink_tail_v + size_t(ikh)*sink_tail_stride_head_f16;
            const float * k_body_ptr = body_k + size_t(ikh)*k_body_stride_head_floats;
            const float * v_body_ptr = body_v + size_t(ikh)*v_body_stride_head_floats;
            const float * pending_k_ptr = pending_k + size_t(ikh)*pending_stride_head_floats;
            const float * pending_v_ptr = pending_v + size_t(ikh)*pending_stride_head_floats;
            const void * kq_mask_ptr = kq_mask == nullptr ? nullptr :
                (const char *) kq_mask + size_t(iq)*kq_mask_stride_query_bytes;

            kvarn_attn_mixed_f16_scratch_scores_softmax_kernel<<<1, block, shmem, cuda_stream>>>(
                    q_ptr, k_st_ptr, k_body_ptr, pending_k_ptr, kq_mask_ptr, scores,
                    n_sink, n_records, n_pending, n_tail, tail_start, head_dim, group_size,
                    sink_tail_stride_token_f16, pending_stride_token_floats,
                    k_body_stride_record_floats, kq_mask_stride_token_bytes, kq_mask_type, scale);
            kvarn_attn_mixed_f16_scratch_av_kernel<<<av_grid, av_block, 0, cuda_stream>>>(
                    scores, v_st_ptr, v_body_ptr, pending_v_ptr, out_ptr,
                    n_sink, n_records, n_pending, n_tail, tail_start, head_dim, group_size,
                    sink_tail_stride_token_f16, pending_stride_token_floats,
                    v_body_stride_record_floats);
        }
    }
}

static __global__ void kvarn_attn_mixed_f16_scores_softmax_kernel(
        const float * __restrict__ q,
        const uint16_t * __restrict__ sink_tail_k,
        const uint8_t * __restrict__ k_body,
        const float * __restrict__ k_scales,
        const float * __restrict__ pending_k,
        const void * __restrict__ kq_mask,
        float * __restrict__ scores,
        uint32_t n_sink,
        uint32_t n_records,
        uint32_t n_pending,
        uint32_t n_tail,
        uint32_t tail_start,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t key_bits,
        size_t sink_tail_stride_token_f16,
        size_t pending_stride_token_floats,
        size_t k_body_stride_bytes,
        size_t k_scale_stride_floats,
        size_t kq_mask_stride_token_bytes,
        uint32_t kq_mask_type,
        float scale) {
    extern __shared__ float shared[];
    float * probs = shared;
    float * reduce = shared + n_sink + n_records*group_size + n_pending + n_tail;

    const uint32_t n_body_tokens = n_records*group_size;
    const uint32_t n_tokens = n_sink + n_body_tokens + n_pending + n_tail;

    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        float sum = 0.0f;

        if (t < n_sink) {
            const uint16_t * k = sink_tail_k + size_t(t)*sink_tail_stride_token_f16;
            for (uint32_t d = 0; d < head_dim; ++d) {
                sum += q[d]*__half2float(reinterpret_cast<const __half *>(k)[d]);
            }
        } else if (t < n_sink + n_body_tokens) {
            const uint32_t body_t = t - n_sink;
            const uint32_t r = body_t/group_size;
            const uint32_t g = body_t - r*group_size;

            const uint8_t * k_record = k_body + size_t(r)*k_body_stride_bytes;
            const float * k_record_scales = k_scales + size_t(r)*k_scale_stride_floats;
            const float * k_s_col = k_record_scales;
            const float * k_zp    = k_record_scales + head_dim;
            const float * k_s_row = k_record_scales + 2*head_dim;

            for (uint32_t d = 0; d < head_dim; ++d) {
                const size_t i = size_t(d)*group_size + g;
                const uint32_t kq = kvarn_unpack_one(k_record, key_bits, i);
                const float k = (float(kq)*k_s_col[d] + k_zp[d])*k_s_row[g];
                sum += q[d]*k;
            }
        } else if (t < n_sink + n_body_tokens + n_pending) {
            const uint32_t pending_t = t - n_sink - n_body_tokens;
            const float * k = pending_k + size_t(pending_t)*pending_stride_token_floats;
            for (uint32_t d = 0; d < head_dim; ++d) {
                sum += q[d]*k[d];
            }
        } else {
            const uint32_t tail_t = t - n_sink - n_body_tokens - n_pending;
            const uint32_t tail_slot = n_tail == 0 ? 0 : (tail_start + tail_t)%n_tail;
            const uint16_t * k = sink_tail_k + size_t(n_sink + tail_slot)*sink_tail_stride_token_f16;
            for (uint32_t d = 0; d < head_dim; ++d) {
                sum += q[d]*__half2float(reinterpret_cast<const __half *>(k)[d]);
            }
        }

        scores[t] = sum*scale + kvarn_kq_mask_bias(kq_mask, kq_mask_type, kq_mask_stride_token_bytes, t);
    }
    __syncthreads();

    float local_max = -3.4028234663852886e38f;
    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        local_max = fmaxf(local_max, scores[t]);
    }
    reduce[threadIdx.x] = local_max;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float max_score = reduce[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        const float p = expf(scores[t] - max_score);
        probs[t] = p;
        local_sum += p;
    }
    reduce[threadIdx.x] = local_sum;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_denom = 1.0f/reduce[0];

    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        scores[t] = probs[t]*inv_denom;
    }
}

static __global__ void kvarn_attn_mixed_f16_av_kernel(
        const float * __restrict__ probs,
        const uint16_t * __restrict__ sink_tail_v,
        const uint8_t * __restrict__ v_body,
        const float * __restrict__ v_scales,
        const float * __restrict__ pending_v,
        float * __restrict__ out,
        uint32_t n_sink,
        uint32_t n_records,
        uint32_t n_pending,
        uint32_t n_tail,
        uint32_t tail_start,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t value_bits,
        size_t sink_tail_stride_token_f16,
        size_t pending_stride_token_floats,
        size_t v_body_stride_bytes,
        size_t v_scale_stride_floats) {
    const uint32_t d = blockIdx.x*blockDim.x + threadIdx.x;
    if (d >= head_dim) {
        return;
    }

    const uint32_t n_body_tokens = n_records*group_size;
    float sum = 0.0f;

    for (uint32_t t = 0; t < n_sink; ++t) {
        const uint16_t * v = sink_tail_v + size_t(t)*sink_tail_stride_token_f16;
        sum += probs[t]*__half2float(reinterpret_cast<const __half *>(v)[d]);
    }

    for (uint32_t r = 0; r < n_records; ++r) {
        const uint8_t * v_record = v_body + size_t(r)*v_body_stride_bytes;
        const float * v_record_scales = v_scales + size_t(r)*v_scale_stride_floats;
        const float * v_s_col = v_record_scales;
        const float * v_s_row = v_record_scales + head_dim;
        const float * v_zp    = v_record_scales + head_dim + group_size;

        for (uint32_t g = 0; g < group_size; ++g) {
            const size_t i = size_t(g)*head_dim + d;
            const uint32_t vq = kvarn_unpack_one(v_record, value_bits, i);
            const float v = (float(vq)*v_s_row[g] + v_zp[g])*v_s_col[d];
            sum += probs[size_t(n_sink) + size_t(r)*group_size + g]*v;
        }
    }

    for (uint32_t t = 0; t < n_pending; ++t) {
        const float * v = pending_v + size_t(t)*pending_stride_token_floats;
        sum += probs[size_t(n_sink) + n_body_tokens + t]*v[d];
    }

    for (uint32_t t = 0; t < n_tail; ++t) {
        const uint32_t tail_slot = (tail_start + t)%n_tail;
        const uint16_t * v = sink_tail_v + size_t(n_sink + tail_slot)*sink_tail_stride_token_f16;
        sum += probs[size_t(n_sink) + n_body_tokens + n_pending + t]*__half2float(reinterpret_cast<const __half *>(v)[d]);
    }

    out[d] = sum;
}

static __global__ void kvarn_attn_mixed_f16_fused_kernel(
        const float * __restrict__ q,
        const uint16_t * __restrict__ sink_tail_k,
        const uint16_t * __restrict__ sink_tail_v,
        const uint8_t * __restrict__ k_body,
        const uint8_t * __restrict__ v_body,
        const float * __restrict__ k_scales,
        const float * __restrict__ v_scales,
        const float * __restrict__ pending_k,
        const float * __restrict__ pending_v,
        const void * __restrict__ kq_mask,
        float * __restrict__ out,
        float * __restrict__ scores,
        uint32_t n_sink,
        uint32_t n_records,
        uint32_t n_pending,
        uint32_t n_tail,
        uint32_t tail_start,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t key_bits,
        uint32_t value_bits,
        size_t sink_tail_stride_token_f16,
        size_t pending_stride_token_floats,
        size_t k_body_stride_bytes,
        size_t v_body_stride_bytes,
        size_t k_scale_stride_floats,
        size_t v_scale_stride_floats,
        size_t kq_mask_stride_token_bytes,
        uint32_t kq_mask_type,
        float scale) {
    extern __shared__ float shared[];
    float * probs = shared;
    float * reduce = shared + n_sink + n_records*group_size + n_pending + n_tail;

    const uint32_t n_body_tokens = n_records*group_size;
    const uint32_t n_tokens = n_sink + n_body_tokens + n_pending + n_tail;

    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        float sum = 0.0f;

        if (t < n_sink) {
            const uint16_t * k = sink_tail_k + size_t(t)*sink_tail_stride_token_f16;
            for (uint32_t d = 0; d < head_dim; ++d) {
                sum += q[d]*__half2float(reinterpret_cast<const __half *>(k)[d]);
            }
        } else if (t < n_sink + n_body_tokens) {
            const uint32_t body_t = t - n_sink;
            const uint32_t r = body_t/group_size;
            const uint32_t g = body_t - r*group_size;

            const uint8_t * k_record = k_body + size_t(r)*k_body_stride_bytes;
            const float * k_record_scales = k_scales + size_t(r)*k_scale_stride_floats;
            const float * k_s_col = k_record_scales;
            const float * k_zp    = k_record_scales + head_dim;
            const float * k_s_row = k_record_scales + 2*head_dim;

            for (uint32_t d = 0; d < head_dim; ++d) {
                const size_t i = size_t(d)*group_size + g;
                const uint32_t kq = kvarn_unpack_one(k_record, key_bits, i);
                const float k = (float(kq)*k_s_col[d] + k_zp[d])*k_s_row[g];
                sum += q[d]*k;
            }
        } else if (t < n_sink + n_body_tokens + n_pending) {
            const uint32_t pending_t = t - n_sink - n_body_tokens;
            const float * k = pending_k + size_t(pending_t)*pending_stride_token_floats;
            for (uint32_t d = 0; d < head_dim; ++d) {
                sum += q[d]*k[d];
            }
        } else {
            const uint32_t tail_t = t - n_sink - n_body_tokens - n_pending;
            const uint32_t tail_slot = n_tail == 0 ? 0 : (tail_start + tail_t)%n_tail;
            const uint16_t * k = sink_tail_k + size_t(n_sink + tail_slot)*sink_tail_stride_token_f16;
            for (uint32_t d = 0; d < head_dim; ++d) {
                sum += q[d]*__half2float(reinterpret_cast<const __half *>(k)[d]);
            }
        }

        scores[t] = sum*scale + kvarn_kq_mask_bias(kq_mask, kq_mask_type, kq_mask_stride_token_bytes, t);
    }
    __syncthreads();

    float local_max = -3.4028234663852886e38f;
    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        local_max = fmaxf(local_max, scores[t]);
    }
    reduce[threadIdx.x] = local_max;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float max_score = reduce[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        const float p = expf(scores[t] - max_score);
        probs[t] = p;
        local_sum += p;
    }
    reduce[threadIdx.x] = local_sum;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_denom = 1.0f/reduce[0];

    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        probs[t] *= inv_denom;
        scores[t] = probs[t];
    }
    __syncthreads();

    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;

        for (uint32_t t = 0; t < n_sink; ++t) {
            const uint16_t * v = sink_tail_v + size_t(t)*sink_tail_stride_token_f16;
            sum += probs[t]*__half2float(reinterpret_cast<const __half *>(v)[d]);
        }

        for (uint32_t r = 0; r < n_records; ++r) {
            const uint8_t * v_record = v_body + size_t(r)*v_body_stride_bytes;
            const float * v_record_scales = v_scales + size_t(r)*v_scale_stride_floats;
            const float * v_s_col = v_record_scales;
            const float * v_s_row = v_record_scales + head_dim;
            const float * v_zp    = v_record_scales + head_dim + group_size;

            for (uint32_t g = 0; g < group_size; ++g) {
                const size_t i = size_t(g)*head_dim + d;
                const uint32_t vq = kvarn_unpack_one(v_record, value_bits, i);
                const float v = (float(vq)*v_s_row[g] + v_zp[g])*v_s_col[d];
                sum += probs[size_t(n_sink) + size_t(r)*group_size + g]*v;
            }
        }

        for (uint32_t t = 0; t < n_pending; ++t) {
            const float * v = pending_v + size_t(t)*pending_stride_token_floats;
            sum += probs[size_t(n_sink) + n_body_tokens + t]*v[d];
        }

        for (uint32_t t = 0; t < n_tail; ++t) {
            const uint32_t tail_slot = (tail_start + t)%n_tail;
            const uint16_t * v = sink_tail_v + size_t(n_sink + tail_slot)*sink_tail_stride_token_f16;
            sum += probs[size_t(n_sink) + n_body_tokens + n_pending + t]*__half2float(reinterpret_cast<const __half *>(v)[d]);
        }

        out[d] = sum;
    }
}

static __device__ __forceinline__ float kvarn_mixed_f16_load_k(
        const uint16_t * __restrict__ sink_tail_k,
        const uint8_t * __restrict__ k_body,
        const float * __restrict__ k_scales,
        const float * __restrict__ pending_k,
        uint32_t t,
        uint32_t d,
        uint32_t n_sink,
        uint32_t n_records,
        uint32_t n_pending,
        uint32_t n_tail,
        uint32_t tail_start,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t key_bits,
        size_t sink_tail_stride_token_f16,
        size_t pending_stride_token_floats,
        size_t k_body_stride_record_bytes,
        size_t k_scale_stride_record_floats) {
    const uint32_t n_body_tokens = n_records*group_size;

    if (t < n_sink) {
        const uint16_t * k = sink_tail_k + size_t(t)*sink_tail_stride_token_f16;
        return __half2float(reinterpret_cast<const __half *>(k)[d]);
    }

    if (t < n_sink + n_body_tokens) {
        const uint32_t body_t = t - n_sink;
        const uint32_t r = body_t/group_size;
        const uint32_t g = body_t - r*group_size;

        const uint8_t * k_record = k_body + size_t(r)*k_body_stride_record_bytes;
        const float * k_record_scales = k_scales + size_t(r)*k_scale_stride_record_floats;
        const float * k_s_col = k_record_scales;
        const float * k_zp    = k_record_scales + head_dim;
        const float * k_s_row = k_record_scales + 2*head_dim;

        const size_t i = size_t(d)*group_size + g;
        const uint32_t kq = kvarn_unpack_one(k_record, key_bits, i);
        return (float(kq)*k_s_col[d] + k_zp[d])*k_s_row[g];
    }

    if (t < n_sink + n_body_tokens + n_pending) {
        const uint32_t pending_t = t - n_sink - n_body_tokens;
        const float * k = pending_k + size_t(pending_t)*pending_stride_token_floats;
        return k[d];
    }

    const uint32_t tail_t = t - n_sink - n_body_tokens - n_pending;
    const uint32_t tail_slot = (tail_start + tail_t)%n_tail;
    const uint16_t * k = sink_tail_k + size_t(n_sink + tail_slot)*sink_tail_stride_token_f16;
    return __half2float(reinterpret_cast<const __half *>(k)[d]);
}

static __device__ __forceinline__ float kvarn_mixed_f16_load_v(
        const uint16_t * __restrict__ sink_tail_v,
        const uint8_t * __restrict__ v_body,
        const float * __restrict__ v_scales,
        const float * __restrict__ pending_v,
        uint32_t t,
        uint32_t d,
        uint32_t n_sink,
        uint32_t n_records,
        uint32_t n_pending,
        uint32_t n_tail,
        uint32_t tail_start,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t value_bits,
        size_t sink_tail_stride_token_f16,
        size_t pending_stride_token_floats,
        size_t v_body_stride_record_bytes,
        size_t v_scale_stride_record_floats) {
    const uint32_t n_body_tokens = n_records*group_size;

    if (t < n_sink) {
        const uint16_t * v = sink_tail_v + size_t(t)*sink_tail_stride_token_f16;
        return __half2float(reinterpret_cast<const __half *>(v)[d]);
    }

    if (t < n_sink + n_body_tokens) {
        const uint32_t body_t = t - n_sink;
        const uint32_t r = body_t/group_size;
        const uint32_t g = body_t - r*group_size;

        const uint8_t * v_record = v_body + size_t(r)*v_body_stride_record_bytes;
        const float * v_record_scales = v_scales + size_t(r)*v_scale_stride_record_floats;
        const float * v_s_col = v_record_scales;
        const float * v_s_row = v_record_scales + head_dim;
        const float * v_zp    = v_record_scales + head_dim + group_size;

        const size_t i = size_t(g)*head_dim + d;
        const uint32_t vq = kvarn_unpack_one(v_record, value_bits, i);
        return (float(vq)*v_s_row[g] + v_zp[g])*v_s_col[d];
    }

    if (t < n_sink + n_body_tokens + n_pending) {
        const uint32_t pending_t = t - n_sink - n_body_tokens;
        const float * v = pending_v + size_t(pending_t)*pending_stride_token_floats;
        return v[d];
    }

    const uint32_t tail_t = t - n_sink - n_body_tokens - n_pending;
    const uint32_t tail_slot = (tail_start + tail_t)%n_tail;
    const uint16_t * v = sink_tail_v + size_t(n_sink + tail_slot)*sink_tail_stride_token_f16;
    return __half2float(reinterpret_cast<const __half *>(v)[d]);
}

static __global__ void kvarn_attn_mixed_f16_sinktail_batch_kernel(
        const float * __restrict__ q,
        const uint16_t * __restrict__ sink_tail_k,
        const uint16_t * __restrict__ sink_tail_v,
        const void * __restrict__ kq_mask,
        float * __restrict__ out,
        uint32_t n_queries,
        uint32_t n_head,
        uint32_t n_head_kv,
        uint32_t n_sink,
        uint32_t n_tail,
        uint32_t tail_start,
        uint32_t head_dim,
        size_t q_stride_head_floats,
        size_t q_stride_query_floats,
        size_t out_stride_head_floats,
        size_t out_stride_query_floats,
        size_t sink_tail_stride_head_f16,
        size_t sink_tail_stride_token_f16,
        size_t kq_mask_stride_query_bytes,
        size_t kq_mask_stride_token_bytes,
        uint32_t kq_mask_type,
        float scale) {
    const uint32_t row = blockIdx.x;
    if (row >= n_queries*n_head) {
        return;
    }

    const uint32_t iq = row/n_head;
    const uint32_t ih = row - iq*n_head;
    const uint32_t n_gqa = n_head/n_head_kv;
    const uint32_t ikh = ih/n_gqa;

    const float * q_row = q + size_t(iq)*q_stride_query_floats + size_t(ih)*q_stride_head_floats;
    float * out_row = out + size_t(iq)*out_stride_query_floats + size_t(ih)*out_stride_head_floats;
    const uint16_t * k_st = sink_tail_k + size_t(ikh)*sink_tail_stride_head_f16;
    const uint16_t * v_st = sink_tail_v + size_t(ikh)*sink_tail_stride_head_f16;
    const void * kq_mask_row = kq_mask == nullptr ? nullptr :
        (const char *) kq_mask + size_t(iq)*kq_mask_stride_query_bytes;

    extern __shared__ float shared[];
    float * probs  = shared;
    float * reduce = shared + n_sink + n_tail;

    const uint32_t n_tokens = n_sink + n_tail;
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t n_warps = blockDim.x >> 5;

    for (uint32_t t = warp; t < n_tokens; t += n_warps) {
        const uint32_t slot = t < n_sink ? t : n_sink + ((tail_start + (t - n_sink))%n_tail);
        const uint16_t * k = k_st + size_t(slot)*sink_tail_stride_token_f16;

        float sum = 0.0f;
        for (uint32_t d = lane; d < head_dim; d += 32) {
            sum += q_row[d]*__half2float(reinterpret_cast<const __half *>(k)[d]);
        }
        sum = kvarn_warp_reduce_sum(sum);
        if (lane == 0) {
            probs[t] = sum*scale + kvarn_kq_mask_bias(kq_mask_row, kq_mask_type, kq_mask_stride_token_bytes, t);
        }
    }
    __syncthreads();

    float local_max = -3.4028234663852886e38f;
    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        local_max = fmaxf(local_max, probs[t]);
    }
    reduce[threadIdx.x] = local_max;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float max_score = reduce[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        const float p = expf(probs[t] - max_score);
        probs[t] = p;
        local_sum += p;
    }
    reduce[threadIdx.x] = local_sum;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_denom = 1.0f/reduce[0];

    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        probs[t] *= inv_denom;
    }
    __syncthreads();

    // Common Gemma tg64 and short-prompt path: no body records and no pending
    // body tokens. This avoids the general mixed-attention loader branches and
    // packed-body scale/unpack math while preserving the same sink/tail ring
    // order and KQ-mask semantics as the full kernel.
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (uint32_t t = 0; t < n_tokens; ++t) {
            const uint32_t slot = t < n_sink ? t : n_sink + ((tail_start + (t - n_sink))%n_tail);
            const uint16_t * v = v_st + size_t(slot)*sink_tail_stride_token_f16;
            sum += probs[t]*__half2float(reinterpret_cast<const __half *>(v)[d]);
        }
        out_row[d] = sum;
    }
}

// Decode path (n_queries=1): one CTA covers all query heads to avoid 16x kernel
// launch overhead per token on Gemma 4 512d sink/tail attention.
static __global__ void kvarn_attn_mixed_f16_sinktail_decode_kernel(
        const float * __restrict__ q,
        const uint16_t * __restrict__ sink_tail_k,
        const uint16_t * __restrict__ sink_tail_v,
        const void * __restrict__ kq_mask,
        float * __restrict__ out,
        uint32_t n_head,
        uint32_t n_head_kv,
        uint32_t n_sink,
        uint32_t n_tail,
        uint32_t tail_start,
        uint32_t head_dim,
        size_t q_stride_head_floats,
        size_t out_stride_head_floats,
        size_t sink_tail_stride_head_f16,
        size_t sink_tail_stride_token_f16,
        size_t kq_mask_stride_token_bytes,
        uint32_t kq_mask_type,
        float scale) {
    // One CTA per query head. The previous version launched a single CTA
    // with one warp per head, leaving all but one SM idle during decode —
    // the primary tg64 throughput ceiling for Gemma 512d sink/tail decode.
    const uint32_t ih = blockIdx.x;
    if (ih >= n_head) {
        return;
    }

    const uint32_t n_gqa = n_head/n_head_kv;
    const uint32_t ikh = ih/n_gqa;
    const float * q_row = q + size_t(ih)*q_stride_head_floats;
    float * out_row = out + size_t(ih)*out_stride_head_floats;
    const uint16_t * k_st = sink_tail_k + size_t(ikh)*sink_tail_stride_head_f16;
    const uint16_t * v_st = sink_tail_v + size_t(ikh)*sink_tail_stride_head_f16;
    const void * kq_mask_row = kq_mask;

    extern __shared__ float shared[];
    const uint32_t n_tokens = n_sink + n_tail;
    // Layout: q_sh[head_dim] | probs[n_tokens] | reduce[blockDim.x]
    float * q_sh = shared;
    float * probs = shared + head_dim;
    float * reduce = probs + n_tokens;

    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        q_sh[d] = q_row[d];
    }
    __syncthreads();

    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t n_warps = blockDim.x >> 5;

    for (uint32_t t = warp; t < n_tokens; t += n_warps) {
        const uint32_t slot = t < n_sink ? t : n_sink + ((tail_start + (t - n_sink))%n_tail);
        const uint16_t * k = k_st + size_t(slot)*sink_tail_stride_token_f16;

        float sum = 0.0f;
        for (uint32_t d = lane; d < head_dim; d += 32) {
            sum += q_sh[d]*__half2float(reinterpret_cast<const __half *>(k)[d]);
        }
        sum = kvarn_warp_reduce_sum(sum);
        if (lane == 0) {
            probs[t] = sum*scale + kvarn_kq_mask_bias(kq_mask_row, kq_mask_type, kq_mask_stride_token_bytes, t);
        }
    }
    __syncthreads();

    float local_max = -3.4028234663852886e38f;
    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        local_max = fmaxf(local_max, probs[t]);
    }
    reduce[threadIdx.x] = local_max;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float max_score = reduce[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        const float p = expf(probs[t] - max_score);
        probs[t] = p;
        local_sum += p;
    }
    reduce[threadIdx.x] = local_sum;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_denom = 1.0f/reduce[0];
    __syncthreads();

    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        probs[t] *= inv_denom;
    }
    __syncthreads();

    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (uint32_t t = 0; t < n_tokens; ++t) {
            const uint32_t slot = t < n_sink ? t : n_sink + ((tail_start + (t - n_sink))%n_tail);
            const uint16_t * v = v_st + size_t(slot)*sink_tail_stride_token_f16;
            sum += probs[t]*__half2float(reinterpret_cast<const __half *>(v)[d]);
        }
        out_row[d] = sum;
    }
}

static __global__ void kvarn_attn_mixed_f16_fused_batch_warpqk_kernel(
        const float * __restrict__ q,
        const uint16_t * __restrict__ sink_tail_k,
        const uint16_t * __restrict__ sink_tail_v,
        const uint8_t * __restrict__ k_body,
        const uint8_t * __restrict__ v_body,
        const float * __restrict__ k_scales,
        const float * __restrict__ v_scales,
        const float * __restrict__ pending_k,
        const float * __restrict__ pending_v,
        const void * __restrict__ kq_mask,
        float * __restrict__ out,
        float * __restrict__ scores,
        uint32_t n_queries,
        uint32_t n_head,
        uint32_t n_head_kv,
        uint32_t n_sink,
        uint32_t n_records,
        uint32_t n_pending,
        uint32_t n_tail,
        uint32_t tail_start,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t key_bits,
        uint32_t value_bits,
        size_t q_stride_head_floats,
        size_t q_stride_query_floats,
        size_t out_stride_head_floats,
        size_t out_stride_query_floats,
        size_t sink_tail_stride_head_f16,
        size_t sink_tail_stride_token_f16,
        size_t pending_stride_head_floats,
        size_t pending_stride_token_floats,
        size_t k_body_stride_record_bytes,
        size_t v_body_stride_record_bytes,
        size_t k_body_stride_head_bytes,
        size_t v_body_stride_head_bytes,
        size_t k_scale_stride_record_floats,
        size_t v_scale_stride_record_floats,
        size_t k_scale_stride_head_floats,
        size_t v_scale_stride_head_floats,
        size_t kq_mask_stride_query_bytes,
        size_t kq_mask_stride_token_bytes,
        uint32_t kq_mask_type,
        float scale,
        const float * __restrict__ body_k_f32,
        const float * __restrict__ body_v_f32,
        size_t body_f32_stride_head_floats) {
    (void) scores;
    const uint32_t row = blockIdx.x;
    if (row >= n_queries*n_head) {
        return;
    }

    const uint32_t iq = row/n_head;
    const uint32_t ih = row - iq*n_head;
    const uint32_t n_gqa = n_head/n_head_kv;
    const uint32_t ikh = ih/n_gqa;

    const float * q_row = q + size_t(iq)*q_stride_query_floats + size_t(ih)*q_stride_head_floats;
    float * out_row = out + size_t(iq)*out_stride_query_floats + size_t(ih)*out_stride_head_floats;
    const uint16_t * k_st = sink_tail_k + size_t(ikh)*sink_tail_stride_head_f16;
    const uint16_t * v_st = sink_tail_v + size_t(ikh)*sink_tail_stride_head_f16;
    const uint8_t * k_body_head = k_body + size_t(ikh)*k_body_stride_head_bytes;
    const uint8_t * v_body_head = v_body + size_t(ikh)*v_body_stride_head_bytes;
    const float * k_scales_head = k_scales + size_t(ikh)*k_scale_stride_head_floats;
    const float * v_scales_head = v_scales + size_t(ikh)*v_scale_stride_head_floats;
    const float * pending_k_head = pending_k + size_t(ikh)*pending_stride_head_floats;
    const float * pending_v_head = pending_v + size_t(ikh)*pending_stride_head_floats;
    const float * k_body_f32_head = body_k_f32 == nullptr ? nullptr :
        body_k_f32 + size_t(ikh)*body_f32_stride_head_floats;
    const float * v_body_f32_head = body_v_f32 == nullptr ? nullptr :
        body_v_f32 + size_t(ikh)*body_f32_stride_head_floats;
    const void * kq_mask_row = kq_mask == nullptr ? nullptr :
        (const char *) kq_mask + size_t(iq)*kq_mask_stride_query_bytes;

    extern __shared__ float shared[];
    // Layout: q_sh[head_dim] | probs[n_tokens] | reduce[blockDim.x].
    // q is reused for every token score; staging it in shared memory avoids
    // n_tokens redundant global reads of the 512-wide query row per CTA.
    float * q_sh = shared;
    float * probs = shared + head_dim;
    float * reduce = probs + n_sink + n_records*group_size + n_pending + n_tail;

    const uint32_t n_body_tokens = n_records*group_size;
    const uint32_t n_tokens = n_sink + n_body_tokens + n_pending + n_tail;

    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        q_sh[d] = q_row[d];
    }
    __syncthreads();

    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t n_warps = blockDim.x >> 5;

    for (uint32_t t = warp; t < n_tokens; t += n_warps) {
        float sum = 0.0f;
        for (uint32_t d = lane; d < head_dim; d += 32) {
            float k = 0.0f;
            if (k_body_f32_head != nullptr && t >= n_sink && t < n_sink + n_body_tokens) {
                const uint32_t body_t = t - n_sink;
                // Dequant scratch K is token-major (g*head_dim + d): adjacent
                // lanes read adjacent floats, giving fully coalesced loads.
                k = k_body_f32_head[size_t(body_t)*head_dim + d];
            } else {
                k = kvarn_mixed_f16_load_k(
                        k_st, k_body_head, k_scales_head, pending_k_head,
                        t, d, n_sink, n_records, n_pending, n_tail, tail_start,
                        head_dim, group_size, key_bits,
                        sink_tail_stride_token_f16, pending_stride_token_floats,
                        k_body_stride_record_bytes, k_scale_stride_record_floats);
            }
            sum += q_sh[d]*k;
        }
        sum = kvarn_warp_reduce_sum(sum);
        if (lane == 0) {
            probs[t] = sum*scale + kvarn_kq_mask_bias(kq_mask_row, kq_mask_type, kq_mask_stride_token_bytes, t);
        }
    }
    __syncthreads();

    float local_max = -3.4028234663852886e38f;
    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        local_max = fmaxf(local_max, probs[t]);
    }
    reduce[threadIdx.x] = local_max;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float max_score = reduce[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        const float p = expf(probs[t] - max_score);
        probs[t] = p;
        local_sum += p;
    }
    reduce[threadIdx.x] = local_sum;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_denom = 1.0f/reduce[0];

    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        probs[t] *= inv_denom;
    }
    __syncthreads();

    // Keep warp-cooperative q*k above, but make the value stage dimension
    // parallel. The first version reduced over tokens for every d, causing
    // head_dim (=512 for Gemma 4) block-wide reductions and synchronizations.
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (uint32_t t = 0; t < n_tokens; ++t) {
            float v = 0.0f;
            if (v_body_f32_head != nullptr && t >= n_sink && t < n_sink + n_body_tokens) {
                const uint32_t body_t = t - n_sink;
                // V scratch is token-major (g*head_dim + d), contiguous in t.
                v = v_body_f32_head[size_t(body_t)*head_dim + d];
            } else {
                v = kvarn_mixed_f16_load_v(
                        v_st, v_body_head, v_scales_head, pending_v_head,
                        t, d, n_sink, n_records, n_pending, n_tail, tail_start,
                        head_dim, group_size, value_bits,
                        sink_tail_stride_token_f16, pending_stride_token_floats,
                        v_body_stride_record_bytes, v_scale_stride_record_floats);
            }
            sum += probs[t]*v;
        }
        out_row[d] = sum;
    }
}

static __global__ void kvarn_attn_mixed_f16_fused_batch_kernel(
        const float * __restrict__ q,
        const uint16_t * __restrict__ sink_tail_k,
        const uint16_t * __restrict__ sink_tail_v,
        const uint8_t * __restrict__ k_body,
        const uint8_t * __restrict__ v_body,
        const float * __restrict__ k_scales,
        const float * __restrict__ v_scales,
        const float * __restrict__ pending_k,
        const float * __restrict__ pending_v,
        const void * __restrict__ kq_mask,
        float * __restrict__ out,
        float * __restrict__ scores,
        uint32_t n_queries,
        uint32_t n_head,
        uint32_t n_head_kv,
        uint32_t n_sink,
        uint32_t n_records,
        uint32_t n_pending,
        uint32_t n_tail,
        uint32_t tail_start,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t key_bits,
        uint32_t value_bits,
        size_t q_stride_head_floats,
        size_t q_stride_query_floats,
        size_t out_stride_head_floats,
        size_t out_stride_query_floats,
        size_t sink_tail_stride_head_f16,
        size_t sink_tail_stride_token_f16,
        size_t pending_stride_head_floats,
        size_t pending_stride_token_floats,
        size_t k_body_stride_record_bytes,
        size_t v_body_stride_record_bytes,
        size_t k_body_stride_head_bytes,
        size_t v_body_stride_head_bytes,
        size_t k_scale_stride_record_floats,
        size_t v_scale_stride_record_floats,
        size_t k_scale_stride_head_floats,
        size_t v_scale_stride_head_floats,
        size_t kq_mask_stride_query_bytes,
        size_t kq_mask_stride_token_bytes,
        uint32_t kq_mask_type,
        float scale) {
    const uint32_t row = blockIdx.x;
    if (row >= n_queries*n_head) {
        return;
    }

    const uint32_t iq = row/n_head;
    const uint32_t ih = row - iq*n_head;
    const uint32_t n_gqa = n_head/n_head_kv;
    const uint32_t ikh = ih/n_gqa;

    const float * q_row = q + size_t(iq)*q_stride_query_floats + size_t(ih)*q_stride_head_floats;
    float * out_row = out + size_t(iq)*out_stride_query_floats + size_t(ih)*out_stride_head_floats;
    const uint16_t * k_st = sink_tail_k + size_t(ikh)*sink_tail_stride_head_f16;
    const uint16_t * v_st = sink_tail_v + size_t(ikh)*sink_tail_stride_head_f16;
    const uint8_t * k_body_head = k_body + size_t(ikh)*k_body_stride_head_bytes;
    const uint8_t * v_body_head = v_body + size_t(ikh)*v_body_stride_head_bytes;
    const float * k_scales_head = k_scales + size_t(ikh)*k_scale_stride_head_floats;
    const float * v_scales_head = v_scales + size_t(ikh)*v_scale_stride_head_floats;
    const float * pending_k_head = pending_k + size_t(ikh)*pending_stride_head_floats;
    const float * pending_v_head = pending_v + size_t(ikh)*pending_stride_head_floats;
    const void * kq_mask_row = kq_mask == nullptr ? nullptr :
        (const char *) kq_mask + size_t(iq)*kq_mask_stride_query_bytes;

    extern __shared__ float shared[];
    float * probs = shared;
    float * reduce = shared + n_sink + n_records*group_size + n_pending + n_tail;

    const uint32_t n_body_tokens = n_records*group_size;
    const uint32_t n_tokens = n_sink + n_body_tokens + n_pending + n_tail;

    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        float sum = 0.0f;

        if (t < n_sink) {
            const uint16_t * k = k_st + size_t(t)*sink_tail_stride_token_f16;
            for (uint32_t d = 0; d < head_dim; ++d) {
                sum += q_row[d]*__half2float(reinterpret_cast<const __half *>(k)[d]);
            }
        } else if (t < n_sink + n_body_tokens) {
            const uint32_t body_t = t - n_sink;
            const uint32_t r = body_t/group_size;
            const uint32_t g = body_t - r*group_size;

            const uint8_t * k_record = k_body_head + size_t(r)*k_body_stride_record_bytes;
            const float * k_record_scales = k_scales_head + size_t(r)*k_scale_stride_record_floats;
            const float * k_s_col = k_record_scales;
            const float * k_zp    = k_record_scales + head_dim;
            const float * k_s_row = k_record_scales + 2*head_dim;

            for (uint32_t d = 0; d < head_dim; ++d) {
                const size_t i = size_t(d)*group_size + g;
                const uint32_t kq = kvarn_unpack_one(k_record, key_bits, i);
                const float k = (float(kq)*k_s_col[d] + k_zp[d])*k_s_row[g];
                sum += q_row[d]*k;
            }
        } else if (t < n_sink + n_body_tokens + n_pending) {
            const uint32_t pending_t = t - n_sink - n_body_tokens;
            const float * k = pending_k_head + size_t(pending_t)*pending_stride_token_floats;
            for (uint32_t d = 0; d < head_dim; ++d) {
                sum += q_row[d]*k[d];
            }
        } else {
            const uint32_t tail_t = t - n_sink - n_body_tokens - n_pending;
            const uint32_t tail_slot = n_tail == 0 ? 0 : (tail_start + tail_t)%n_tail;
            const uint16_t * k = k_st + size_t(n_sink + tail_slot)*sink_tail_stride_token_f16;
            for (uint32_t d = 0; d < head_dim; ++d) {
                sum += q_row[d]*__half2float(reinterpret_cast<const __half *>(k)[d]);
            }
        }

        probs[t] = sum*scale + kvarn_kq_mask_bias(kq_mask_row, kq_mask_type, kq_mask_stride_token_bytes, t);
    }
    __syncthreads();

    float local_max = -3.4028234663852886e38f;
    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        local_max = fmaxf(local_max, probs[t]);
    }
    reduce[threadIdx.x] = local_max;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float max_score = reduce[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        const float p = expf(probs[t] - max_score);
        probs[t] = p;
        local_sum += p;
    }
    reduce[threadIdx.x] = local_sum;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_denom = 1.0f/reduce[0];

    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        probs[t] *= inv_denom;
    }
    __syncthreads();

    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;

        for (uint32_t t = 0; t < n_sink; ++t) {
            const uint16_t * v = v_st + size_t(t)*sink_tail_stride_token_f16;
            sum += probs[t]*__half2float(reinterpret_cast<const __half *>(v)[d]);
        }

        for (uint32_t r = 0; r < n_records; ++r) {
            const uint8_t * v_record = v_body_head + size_t(r)*v_body_stride_record_bytes;
            const float * v_record_scales = v_scales_head + size_t(r)*v_scale_stride_record_floats;
            const float * v_s_col = v_record_scales;
            const float * v_s_row = v_record_scales + head_dim;
            const float * v_zp    = v_record_scales + head_dim + group_size;

            for (uint32_t g = 0; g < group_size; ++g) {
                const size_t i = size_t(g)*head_dim + d;
                const uint32_t vq = kvarn_unpack_one(v_record, value_bits, i);
                const float v = (float(vq)*v_s_row[g] + v_zp[g])*v_s_col[d];
                sum += probs[size_t(n_sink) + size_t(r)*group_size + g]*v;
            }
        }

        for (uint32_t t = 0; t < n_pending; ++t) {
            const float * v = pending_v_head + size_t(t)*pending_stride_token_floats;
            sum += probs[size_t(n_sink) + n_body_tokens + t]*v[d];
        }

        for (uint32_t t = 0; t < n_tail; ++t) {
            const uint32_t tail_slot = (tail_start + t)%n_tail;
            const uint16_t * v = v_st + size_t(n_sink + tail_slot)*sink_tail_stride_token_f16;
            sum += probs[size_t(n_sink) + n_body_tokens + n_pending + t]*__half2float(reinterpret_cast<const __half *>(v)[d]);
        }

        out_row[d] = sum;
    }
}

void ggml_cuda_kvarn_attn_mixed_f16_batch(
        const float * q,
        const uint16_t * sink_tail_k,
        const uint16_t * sink_tail_v,
        const uint8_t * k_body,
        const uint8_t * v_body,
        const float * k_scales,
        const float * v_scales,
        const float * pending_k,
        const float * pending_v,
        const void * kq_mask,
        float * out,
        float * scores,
        uint32_t n_queries,
        uint32_t n_head,
        uint32_t n_head_kv,
        uint32_t n_sink,
        uint32_t n_records,
        uint32_t n_pending,
        uint32_t n_tail,
        uint32_t tail_start,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t key_bits,
        uint32_t value_bits,
        size_t q_stride_head_floats,
        size_t q_stride_query_floats,
        size_t out_stride_head_floats,
        size_t out_stride_query_floats,
        size_t sink_tail_stride_head_f16,
        size_t sink_tail_stride_token_f16,
        size_t pending_stride_head_floats,
        size_t pending_stride_token_floats,
        size_t k_body_stride_record_bytes,
        size_t v_body_stride_record_bytes,
        size_t k_body_stride_head_bytes,
        size_t v_body_stride_head_bytes,
        size_t k_scale_stride_record_floats,
        size_t v_scale_stride_record_floats,
        size_t k_scale_stride_head_floats,
        size_t v_scale_stride_head_floats,
        size_t kq_mask_stride_query_bytes,
        size_t kq_mask_stride_token_bytes,
        uint32_t kq_mask_type,
        float scale,
        void * stream) {
    const uint32_t n_tokens = n_sink + n_records*group_size + n_pending + n_tail;
    const uint32_t n_gqa = n_head/n_head_kv;
    (void) kvarn_env_flag("LLAMA_KVARN_ATTN_FUSED_BATCH");
    const bool force_serial_fused = kvarn_env_flag("LLAMA_KVARN_ATTN_SERIAL_FUSED");
    const bool use_split_kernels = kvarn_env_flag("LLAMA_KVARN_ATTN_SPLIT_KERNELS");
    const bool use_serial_fused = force_serial_fused && !use_split_kernels;

    int block = 1;
    while (block < int(n_tokens)) {
        block <<= 1;
    }
    block = block > 256 ? 256 : block;

    const size_t shmem = (size_t(n_tokens) + size_t(block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
    const int av_block = 128;
    const int av_grid = int((head_dim + av_block - 1)/av_block);
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    // Per-head serial launches add n_head kernel launch overheads and regressed
    // Gemma tg64 (~56% vs 71% pre-refinement). Opt-in via env for A/B only.
    const bool use_decode_per_head = kvarn_env_flag("LLAMA_KVARN_ATTN_DECODE_PER_HEAD");

    if (!use_split_kernels && !use_serial_fused) {
        if (n_queries == 1 && use_decode_per_head) {
            const uint32_t n_gqa = n_head/n_head_kv;
            for (uint32_t ih = 0; ih < n_head; ++ih) {
                const uint32_t ikh = ih/n_gqa;
                const float * q_ptr = q + size_t(ih)*q_stride_head_floats;
                float * out_ptr = out + size_t(ih)*out_stride_head_floats;
                const uint16_t * k_st_ptr = sink_tail_k + size_t(ikh)*sink_tail_stride_head_f16;
                const uint16_t * v_st_ptr = sink_tail_v + size_t(ikh)*sink_tail_stride_head_f16;
                const uint8_t * k_body_ptr = k_body + size_t(ikh)*k_body_stride_head_bytes;
                const uint8_t * v_body_ptr = v_body + size_t(ikh)*v_body_stride_head_bytes;
                const float * k_scales_ptr = k_scales + size_t(ikh)*k_scale_stride_head_floats;
                const float * v_scales_ptr = v_scales + size_t(ikh)*v_scale_stride_head_floats;
                const float * pending_k_ptr = pending_k + size_t(ikh)*pending_stride_head_floats;
                const float * pending_v_ptr = pending_v + size_t(ikh)*pending_stride_head_floats;
                const void * kq_mask_ptr = kq_mask;

                kvarn_attn_mixed_f16_fused_kernel<<<1, block, shmem, cuda_stream>>>(
                        q_ptr, k_st_ptr, v_st_ptr, k_body_ptr, v_body_ptr, k_scales_ptr, v_scales_ptr,
                        pending_k_ptr, pending_v_ptr, kq_mask_ptr, out_ptr, scores,
                        n_sink, n_records, n_pending, n_tail, tail_start, head_dim, group_size, key_bits, value_bits,
                        sink_tail_stride_token_f16, pending_stride_token_floats,
                        k_body_stride_record_bytes, v_body_stride_record_bytes,
                        k_scale_stride_record_floats, v_scale_stride_record_floats,
                        kq_mask_stride_token_bytes, kq_mask_type, scale);
            }
            return;
        }

        // Gemma decode and short contexts often have no quantized body records
        // yet (tg64: n_records=0, n_pending=0). The generic 512d warpqk path is
        // built for mixed sink/body/tail windows and pays branch + packed-body
        // overhead even when all K/V are plain f16 sink/tail entries. Use a
        // dedicated sink/tail f16 path in that case. It keeps the fused-batch
        // launch shape (one CTA per query-head) and does not affect the passing
        // 128/256d Qwen path.
        if (head_dim >= 512 && n_records == 0 && n_pending == 0) {
            if (n_queries == 1) {
                // One CTA per head spreads decode attention across SMs.
                // shared = q_sh[head_dim] + probs[n_tokens] + reduce[block]
                const int decode_block = 128;
                const size_t decode_shmem = (size_t(head_dim) + size_t(n_tokens) + size_t(decode_block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
                if (kvarn_cuda_dynamic_shmem_fits(decode_shmem)) {
                    kvarn_attn_mixed_f16_sinktail_decode_kernel<<<int(n_head), decode_block, decode_shmem, cuda_stream>>>(
                            q, sink_tail_k, sink_tail_v, kq_mask, out,
                            n_head, n_head_kv,
                            n_sink, n_tail, tail_start, head_dim,
                            q_stride_head_floats, out_stride_head_floats,
                            sink_tail_stride_head_f16, sink_tail_stride_token_f16,
                            kq_mask_stride_token_bytes, kq_mask_type, scale);
                    return;
                }
            }

            const int sinktail_block = n_tokens <= 128 ? 256 : 512;
            const size_t sinktail_shmem = (size_t(n_tokens) + size_t(sinktail_block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
            if (kvarn_cuda_dynamic_shmem_fits(sinktail_shmem)) {
                kvarn_attn_mixed_f16_sinktail_batch_kernel<<<int(n_queries*n_head), sinktail_block, sinktail_shmem, cuda_stream>>>(
                        q, sink_tail_k, sink_tail_v, kq_mask, out,
                        n_queries, n_head, n_head_kv,
                        n_sink, n_tail, tail_start, head_dim,
                        q_stride_head_floats, q_stride_query_floats,
                        out_stride_head_floats, out_stride_query_floats,
                        sink_tail_stride_head_f16, sink_tail_stride_token_f16,
                        kq_mask_stride_query_bytes, kq_mask_stride_token_bytes,
                        kq_mask_type, scale);
                return;
            }
        }

        // Gemma 4 KVarN+ISWA has 512-dimensional K/V heads. The generic fused
        // kernel maps one token score to one thread, which makes f16 sink/tail
        // loads strided and serializes each 512-wide dot product within a lane.
        // Use a warp-per-score variant for 512d heads: lanes cooperate across
        // the head dimension and issue coalesced f16 sink/tail loads. Keep the
        // 128/256d path unchanged because it is already the passing Qwen path.
        if (head_dim >= 512) {
            const int warpqk_block = 512;
            // shared = q_sh[head_dim] + probs[n_tokens] + reduce[block]
            const size_t warpqk_shmem = (size_t(head_dim) + size_t(n_tokens) + size_t(warpqk_block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
            if (kvarn_cuda_dynamic_shmem_fits(warpqk_shmem)) {
                const float * body_k_f32 = nullptr;
                const float * body_v_f32 = nullptr;
                size_t body_f32_stride_head_floats = 0;

                if (n_records > 0 && scores != nullptr) {
                    const size_t n_per_record = size_t(group_size)*head_dim;
                    const size_t n_per_head = size_t(n_records)*n_per_record;
                    const size_t dequant_base = size_t(n_tokens);
                    float * k_dequant = scores + dequant_base;
                    float * v_dequant = k_dequant + size_t(n_head_kv)*n_per_head;

                    for (uint32_t ikh = 0; ikh < n_head_kv; ++ikh) {
                        // K is dequantized into token-major scratch so the
                        // warpqk QK loop issues coalesced per-lane loads. The
                        // persistent packed layout is unchanged.
                        ggml_cuda_kvarn_dequant_body_n_k_token_major(
                                k_body + size_t(ikh)*k_body_stride_head_bytes,
                                v_body + size_t(ikh)*v_body_stride_head_bytes,
                                k_scales + size_t(ikh)*k_scale_stride_head_floats,
                                v_scales + size_t(ikh)*v_scale_stride_head_floats,
                                k_dequant + size_t(ikh)*n_per_head,
                                v_dequant + size_t(ikh)*n_per_head,
                                n_records, head_dim, group_size, key_bits, value_bits,
                                k_body_stride_record_bytes, v_body_stride_record_bytes,
                                k_scale_stride_record_floats, v_scale_stride_record_floats,
                                n_per_record, n_per_record,
                                cuda_stream);
                    }

                    body_k_f32 = k_dequant;
                    body_v_f32 = v_dequant;
                    body_f32_stride_head_floats = n_per_head;
                }

                kvarn_attn_mixed_f16_fused_batch_warpqk_kernel<<<int(n_queries*n_head), warpqk_block, warpqk_shmem, cuda_stream>>>(
                        q, sink_tail_k, sink_tail_v, k_body, v_body, k_scales, v_scales, pending_k, pending_v,
                        kq_mask,
                        out, scores, n_queries, n_head, n_head_kv,
                        n_sink, n_records, n_pending, n_tail, tail_start, head_dim, group_size, key_bits, value_bits,
                        q_stride_head_floats, q_stride_query_floats,
                        out_stride_head_floats, out_stride_query_floats,
                        sink_tail_stride_head_f16, sink_tail_stride_token_f16,
                        pending_stride_head_floats, pending_stride_token_floats,
                        k_body_stride_record_bytes, v_body_stride_record_bytes,
                        k_body_stride_head_bytes, v_body_stride_head_bytes,
                        k_scale_stride_record_floats, v_scale_stride_record_floats,
                        k_scale_stride_head_floats, v_scale_stride_head_floats,
                        kq_mask_stride_query_bytes, kq_mask_stride_token_bytes, kq_mask_type, scale,
                        body_k_f32, body_v_f32, body_f32_stride_head_floats);
                return;
            }
        }

        kvarn_attn_mixed_f16_fused_batch_kernel<<<int(n_queries*n_head), block, shmem, cuda_stream>>>(
                q, sink_tail_k, sink_tail_v, k_body, v_body, k_scales, v_scales, pending_k, pending_v,
                kq_mask,
                out, scores, n_queries, n_head, n_head_kv,
                n_sink, n_records, n_pending, n_tail, tail_start, head_dim, group_size, key_bits, value_bits,
                q_stride_head_floats, q_stride_query_floats,
                out_stride_head_floats, out_stride_query_floats,
                sink_tail_stride_head_f16, sink_tail_stride_token_f16,
                pending_stride_head_floats, pending_stride_token_floats,
                k_body_stride_record_bytes, v_body_stride_record_bytes,
                k_body_stride_head_bytes, v_body_stride_head_bytes,
                k_scale_stride_record_floats, v_scale_stride_record_floats,
                k_scale_stride_head_floats, v_scale_stride_head_floats,
                kq_mask_stride_query_bytes, kq_mask_stride_token_bytes, kq_mask_type, scale);
        return;
    }

    for (uint32_t iq = 0; iq < n_queries; ++iq) {
        for (uint32_t ih = 0; ih < n_head; ++ih) {
            const uint32_t ikh = ih/n_gqa;
            const float * q_ptr = q + size_t(iq)*q_stride_query_floats + size_t(ih)*q_stride_head_floats;
            float * out_ptr = out + size_t(iq)*out_stride_query_floats + size_t(ih)*out_stride_head_floats;
            const uint16_t * k_st_ptr = sink_tail_k + size_t(ikh)*sink_tail_stride_head_f16;
            const uint16_t * v_st_ptr = sink_tail_v + size_t(ikh)*sink_tail_stride_head_f16;
            const uint8_t * k_body_ptr = k_body + size_t(ikh)*k_body_stride_head_bytes;
            const uint8_t * v_body_ptr = v_body + size_t(ikh)*v_body_stride_head_bytes;
            const float * k_scales_ptr = k_scales + size_t(ikh)*k_scale_stride_head_floats;
            const float * v_scales_ptr = v_scales + size_t(ikh)*v_scale_stride_head_floats;
            const float * pending_k_ptr = pending_k + size_t(ikh)*pending_stride_head_floats;
            const float * pending_v_ptr = pending_v + size_t(ikh)*pending_stride_head_floats;
            const void * kq_mask_ptr = kq_mask == nullptr ? nullptr :
                (const char *) kq_mask + size_t(iq)*kq_mask_stride_query_bytes;

            if (use_split_kernels) {
                kvarn_attn_mixed_f16_scores_softmax_kernel<<<1, block, shmem, cuda_stream>>>(
                        q_ptr, k_st_ptr, k_body_ptr, k_scales_ptr, pending_k_ptr, kq_mask_ptr, scores,
                        n_sink, n_records, n_pending, n_tail, tail_start, head_dim, group_size, key_bits,
                        sink_tail_stride_token_f16, pending_stride_token_floats,
                        k_body_stride_record_bytes, k_scale_stride_record_floats,
                        kq_mask_stride_token_bytes, kq_mask_type, scale);
                kvarn_attn_mixed_f16_av_kernel<<<av_grid, av_block, 0, cuda_stream>>>(
                        scores, v_st_ptr, v_body_ptr, v_scales_ptr, pending_v_ptr, out_ptr,
                        n_sink, n_records, n_pending, n_tail, tail_start, head_dim, group_size, value_bits,
                        sink_tail_stride_token_f16, pending_stride_token_floats,
                        v_body_stride_record_bytes, v_scale_stride_record_floats);
            } else {
                kvarn_attn_mixed_f16_fused_kernel<<<1, block, shmem, cuda_stream>>>(
                        q_ptr, k_st_ptr, v_st_ptr, k_body_ptr, v_body_ptr, k_scales_ptr, v_scales_ptr,
                        pending_k_ptr, pending_v_ptr, kq_mask_ptr, out_ptr, scores,
                        n_sink, n_records, n_pending, n_tail, tail_start, head_dim, group_size, key_bits, value_bits,
                        sink_tail_stride_token_f16, pending_stride_token_floats,
                        k_body_stride_record_bytes, v_body_stride_record_bytes,
                        k_scale_stride_record_floats, v_scale_stride_record_floats,
                        kq_mask_stride_token_bytes, kq_mask_type, scale);
            }
        }
    }
}
