#pragma once

// SM86 Qwen3.8 singleton attention prototype.
//
// One CTA owns three query heads that share one KV head. Warp 0 stages packed
// Q8_0 K and Turbo3 V rows. Warps 1..6 form three two-warp consumer pairs;
// each pair owns one query head and splits KV positions. This avoids making any
// lane retain two head accumulators, the source of the spills in the archived
// GQA2 prototype, while restoring intra-head sequence parallelism.

template <int TILE>
__launch_bounds__(224, 1)
static __global__ void flash_attn_ext_vec_q8_t3_gqa3_sm86(
        const char * Q_ptr,
        const char * K_ptr,
        const char * V_ptr,
        const char * mask_ptr,
        const char * sinks_ptr,
        const int  * KV_max_ptr,
        float      * dst_ptr,
        float2     * dst_meta_ptr,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#if defined(FLASH_ATTN_AVAILABLE) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 860
    constexpr int D = 256;
    constexpr int GQA_GROUP = 3;
    constexpr int HEAD_WARPS = 2;
    constexpr int NCONSUMERS = GQA_GROUP * HEAD_WARPS;
    constexpr int NQ_I32 = D / int(sizeof(int));
    constexpr int NQ_DS  = D / QK8_1;
    constexpr int K_BYTES = (D / QK8_0) * int(sizeof(block_q8_0));
    constexpr int V_BYTES = (D / QK_TURBO3) * int(sizeof(block_turbo3_0));

    static_assert(K_BYTES % int(sizeof(int)) == 0, "Q8 row must permit word copies");
    static_assert(V_BYTES % int(sizeof(int)) == 0, "Turbo3 row must permit word copies");
    static_assert(D / WARP_SIZE == 8, "prototype ownership assumes eight values per lane");

    __shared__ __align__(16) unsigned char K_stage[TILE][K_BYTES];
    __shared__ __align__(16) unsigned char V_stage[TILE][V_BYTES];
    __shared__ int    Q_i32_shared[GQA_GROUP][NQ_I32];
    __shared__ float2 Q_ds_shared [GQA_GROUP][NQ_DS];
    __shared__ float  partial_out [NCONSUMERS][D];
    __shared__ float  partial_max [NCONSUMERS];
    __shared__ float  partial_sum [NCONSUMERS];

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int consumer = warp - 1;
    const int consumer_head = consumer / HEAD_WARPS;
    const int consumer_split = consumer - consumer_head * HEAD_WARPS;

    const int groups_per_kv = 2; // Qwen3.8 GQA ratio 6, three consumers per CTA.
    const int groups_per_sequence = ne12 * groups_per_kv;
    const int sequence = blockIdx.z / groups_per_sequence;
    const int group_in_sequence = blockIdx.z - sequence * groups_per_sequence;
    const int kv_head = group_in_sequence / groups_per_kv;
    const int subgroup = group_in_sequence - kv_head * groups_per_kv;
    const int head0 = kv_head * 6 + subgroup * GQA_GROUP;

    const char * K_base = K_ptr + nb13 * sequence + nb12 * kv_head;
    const char * V_base = V_ptr + nb23 * sequence + nb22 * kv_head;
    const half * maskh = mask_ptr ?
        (const half *) (mask_ptr + nb33 * (sequence % ne33)) : nullptr;
    const int k_max = KV_max_ptr ? KV_max_ptr[sequence * gridDim.x + blockIdx.x] : ne11;

    int q_i32[2] = {0, 0};
    float2 q_ds[2] = {make_float2(0.0f, 0.0f), make_float2(0.0f, 0.0f)};
    float out[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float kq_max = -FLT_MAX / 2.0f;
    float kq_sum = 0.0f;
    int head = head0;
    float slope = 1.0f;

    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();

    if (warp > 0) {
        head = head0 + consumer_head;
        const float * Q = (const float *) (Q_ptr + nb03 * sequence + nb02 * head);
        if (consumer_split == 0) {
#pragma unroll
            for (int i0 = 0; i0 < NQ_I32; i0 += WARP_SIZE) {
                quantize_q8_1_to_shared<float2, WARP_SIZE>(
                    Q + i0 * int(sizeof(int)), scale,
                    Q_i32_shared[consumer_head] + i0,
                    Q_ds_shared[consumer_head] + i0 / QI8_1);
            }
        }
        slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);
    }
    __syncthreads();
    if (warp > 0) {
        q_i32[0] = Q_i32_shared[consumer_head][lane];
        q_i32[1] = Q_i32_shared[consumer_head][WARP_SIZE + lane];
        q_ds[0] = Q_ds_shared[consumer_head][lane / QI8_1];
        q_ds[1] = Q_ds_shared[consumer_head][(WARP_SIZE + lane) / QI8_1];
    }

    for (int tile0 = blockIdx.y * TILE; tile0 < k_max; tile0 += gridDim.y * TILE) {
        if (warp == 0) {
#pragma unroll
            for (int p = 0; p < TILE; ++p) {
                const int pos = tile0 + p;
                if (pos >= k_max) {
                    continue;
                }
                const int * K_src = (const int *) (K_base + int64_t(pos) * nb11);
                int * K_dst = (int *) K_stage[p];
                for (int i = lane; i < K_BYTES / int(sizeof(int)); i += WARP_SIZE) {
                    K_dst[i] = K_src[i];
                }
                const int * V_src = (const int *) (V_base + int64_t(pos) * nb21);
                int * V_dst = (int *) V_stage[p];
                for (int i = lane; i < V_BYTES / int(sizeof(int)); i += WARP_SIZE) {
                    V_dst[i] = V_src[i];
                }
            }
        }
        __syncthreads();

        if (warp > 0) {
            float logits[TILE];
            float tile_max = -FLT_MAX / 2.0f;
#pragma unroll
            for (int p = consumer_split; p < TILE; p += HEAD_WARPS) {
                const int pos = tile0 + p;
                float logit = -FLT_MAX / 2.0f;
                if (pos < k_max) {
                    logit = vec_dot_fattn_vec_KQ_q8_0<D, WARP_SIZE>(
                        (const char *) K_stage[p], nullptr, q_i32, q_ds);
                    logit = warp_reduce_sum<WARP_SIZE>(logit);
                    if (logit_softcap != 0.0f) {
                        logit = logit_softcap * tanhf(logit);
                    }
                    if (maskh) {
                        logit += slope * __half2float(maskh[pos]);
                    }
                    tile_max = fmaxf(tile_max, logit + FATTN_KQ_MAX_OFFSET);
                }
                logits[p] = logit;
            }

            const float new_max = fmaxf(kq_max, tile_max);
            const float old_scale = __expf(kq_max - new_max);
            kq_sum *= old_scale;
#pragma unroll
            for (int d = 0; d < 8; ++d) {
                out[d] *= old_scale;
            }
            kq_max = new_max;

#pragma unroll
            for (int p = consumer_split; p < TILE; p += HEAD_WARPS) {
                if (tile0 + p >= k_max) {
                    continue;
                }
                const float weight = __expf(logits[p] - kq_max);
                kq_sum += weight;
                float values0[4];
                float values1[4];
                dequantize_V_turbo3_0<float, 4>(V_stage[p], values0, lane * 8 + 0);
                dequantize_V_turbo3_0<float, 4>(V_stage[p], values1, lane * 8 + 4);
#pragma unroll
                for (int d = 0; d < 4; ++d) {
                    out[d + 0] += weight * values0[d];
                    out[d + 4] += weight * values1[d];
                }
            }
        }
        __syncthreads();
    }

    if (warp > 0) {
#pragma unroll
        for (int d = 0; d < 8; ++d) {
            partial_out[consumer][lane * 8 + d] = out[d];
        }
        if (lane == 0) {
            partial_max[consumer] = kq_max;
            partial_sum[consumer] = kq_sum;
        }
    }
    __syncthreads();

    if (warp > 0 && consumer_split == 0) {
        const float max0 = partial_max[consumer + 0];
        const float max1 = partial_max[consumer + 1];
        kq_max = fmaxf(max0, max1);
        const float scale0 = __expf(max0 - kq_max);
        const float scale1 = __expf(max1 - kq_max);
        kq_sum = scale0 * partial_sum[consumer + 0] + scale1 * partial_sum[consumer + 1];
#pragma unroll
        for (int d = 0; d < 8; ++d) {
            const int index = lane * 8 + d;
            out[d] = scale0 * partial_out[consumer + 0][index] +
                     scale1 * partial_out[consumer + 1][index];
        }

        if (sinks_ptr && blockIdx.y == 0) {
            const float sink = ((const float *) sinks_ptr)[head];
            const float new_max = fmaxf(sink, kq_max);
            const float old_scale = __expf(kq_max - new_max);
            kq_sum = kq_sum * old_scale + __expf(sink - new_max);
#pragma unroll
            for (int d = 0; d < 8; ++d) {
                out[d] *= old_scale;
            }
            kq_max = new_max;
        }

        float * dst = dst_ptr +
            (((sequence * int(ne01.z) + 0) * ne02 + head) * gridDim.y + blockIdx.y) * D;
#pragma unroll
        for (int d = 0; d < 8; ++d) {
            dst[lane * 8 + d] = gridDim.y == 1 ? out[d] / kq_sum : out[d];
        }
        if (gridDim.y != 1 && lane == 0) {
            dst_meta_ptr[((sequence * int(ne01.z) + 0) * ne02 + head) * gridDim.y + blockIdx.y] =
                make_float2(kq_max, kq_sum);
        }
    }

    GGML_UNUSED_VARS(ne00, ne03, nb01, ne10, ne13, ne31, ne32, nb31, nb32);
#else
    GGML_UNUSED_VARS(Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_meta_ptr,
        scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03, nb01, nb02, nb03,
        ne10, ne11, ne12, ne13, nb11, nb12, nb13,
        nb21, nb22, nb23, ne31, ne32, ne33, nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif
}

static void ggml_cuda_flash_attn_ext_vec_q8_t3_gqa3_sm86(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int TILE = 8;
    constexpr int nwarps = 7;
    constexpr size_t nbytes_shared = 0;
    fattn_kernel_t kernel = flash_attn_ext_vec_q8_t3_gqa3_sm86<TILE>;
    launch_fattn<256, 1, 3>(ctx, dst, kernel, nwarps, nbytes_shared, TILE, false, false, false);
}
