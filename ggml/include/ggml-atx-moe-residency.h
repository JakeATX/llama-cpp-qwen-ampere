#pragma once

#include "ggml-backend.h"

#include <stdint.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

struct ggml_atx_moe_direct_cache {
    const void * packed_src;
    const void * hot_data;
    const int32_t * expert_map;
    int type;
    int layer;
    int tensor_kind;
    int64_t n_expert;
    int64_t n_hot;
    size_t expert_stride_bytes;
    size_t hot_stride_bytes;
    size_t hot_stride_channel;
};

enum ggml_atx_moe_direct_kernel {
    GGML_ATX_MOE_DIRECT_KERNEL_MMVQ = 1,
    GGML_ATX_MOE_DIRECT_KERNEL_MMQ  = 2,
    GGML_ATX_MOE_DIRECT_KERNEL_MMF  = 3,
};

// ATX: explicit flush for MoE residency telemetry. Long-lived servers call this
// during graceful shutdown so residency stats are written before process exit.
GGML_API void ggml_backend_atx_moe_residency_flush_stats(void);

// ATX: lookup compact hot-expert storage for a staged packed-MoE tensor.
// CUDA uses this in experimental direct mode to consume resident experts
// without copying them into the per-token input_cpy staging tensor.
GGML_API bool ggml_backend_atx_moe_residency_get_direct_cache(
        const struct ggml_tensor * staged_src,
        struct ggml_atx_moe_direct_cache * out);

GGML_API bool ggml_backend_atx_moe_residency_direct_enabled(void);
GGML_API bool ggml_backend_atx_moe_residency_direct_require(void);
GGML_API bool ggml_backend_atx_moe_residency_strict_hot_no_stage(void);
GGML_API void ggml_backend_atx_moe_residency_note_direct_dispatch(int kernel);
GGML_API void ggml_backend_atx_moe_residency_note_direct_fallback(const char * reason);

#ifdef  __cplusplus
}
#endif
