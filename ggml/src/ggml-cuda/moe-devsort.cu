#include "moe-devsort.cuh"

// One thread per expert-count slot: zero an int32 array.
static __global__ void k_moe_zero_i32(int32_t * p, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0;
}

// One thread per (token, used) slot: atomically count tokens per expert.
static __global__ void k_moe_hist(const char * __restrict__ ids, int n_slots, int n_used,
                                  size_t nb0, size_t nb1, int32_t * __restrict__ tpe) {
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= n_slots) return;
    const int i12 = slot / n_used;
    const int iex = slot % n_used;
    const int e = *(const int32_t *)(ids + (size_t)i12 * nb1 + (size_t)iex * nb0);
    atomicAdd(&tpe[e], 1);
}

// Single-thread exclusive prefix sum over ne02 experts (ne02 is small, ~<=256).
static __global__ void k_moe_excl_scan(const int32_t * __restrict__ in, int32_t * __restrict__ out, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int32_t acc = 0;
        for (int i = 0; i < n; i++) { out[i] = acc; acc += in[i]; }
    }
}

// One thread per (token, used) slot: scatter each slot to its expert's contiguous
// range at offsets[e] + running fill counter. Within-expert order is arbitrary
// (the ids_from_sorted inverse restores each token's position after the GEMM).
static __global__ void k_moe_scatter(const char * __restrict__ ids, int n_slots, int n_used, int ne11,
                                     size_t nb0, size_t nb1,
                                     const int32_t * __restrict__ offsets, int32_t * __restrict__ fill,
                                     int32_t * __restrict__ to_sorted, int32_t * __restrict__ from_sorted) {
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= n_slots) return;
    const int i12 = slot / n_used;
    const int iex = slot % n_used;
    const int e = *(const int32_t *)(ids + (size_t)i12 * nb1 + (size_t)iex * nb0);
    const int pos = offsets[e] + atomicAdd(&fill[e], 1);
    to_sorted[pos]    = i12 * ne11 + (iex % ne11);  // matches host: i12*ne11 + iex%ne11
    from_sorted[slot] = pos;
}

void ggml_cuda_moe_build_sorted_ids(
        const void * ids, int ne12, int n_expert_used, int ne02, int ne11,
        size_t ids_nb0, size_t ids_nb1,
        int32_t * ids_to_sorted, int32_t * ids_from_sorted,
        int32_t * tokens_per_expert_dev,
        int32_t * scratch_offsets, int32_t * scratch_fill,
        cudaStream_t stream) {
    const int n_slots = ne12 * n_expert_used;
    const int TPB = 256;
    const dim3 g_experts((ne02 + TPB - 1) / TPB);
    const dim3 g_slots((n_slots + TPB - 1) / TPB);

    k_moe_zero_i32<<<g_experts, TPB, 0, stream>>>(tokens_per_expert_dev, ne02);
    k_moe_zero_i32<<<g_experts, TPB, 0, stream>>>(scratch_fill, ne02);

    k_moe_hist<<<g_slots, TPB, 0, stream>>>(
        (const char *)ids, n_slots, n_expert_used, ids_nb0, ids_nb1, tokens_per_expert_dev);

    k_moe_excl_scan<<<1, 1, 0, stream>>>(tokens_per_expert_dev, scratch_offsets, ne02);

    k_moe_scatter<<<g_slots, TPB, 0, stream>>>(
        (const char *)ids, n_slots, n_expert_used, ne11, ids_nb0, ids_nb1,
        scratch_offsets, scratch_fill, ids_to_sorted, ids_from_sorted);

    CUDA_CHECK(cudaGetLastError());
}
