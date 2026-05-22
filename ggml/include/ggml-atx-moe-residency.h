#pragma once

#include "ggml-backend.h"

#include <stdint.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

struct ggml_atx_moe_direct_cache {
    const void * hot_data;
    const int32_t * expert_map;
    int64_t n_expert;
    size_t hot_stride_channel;
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

#ifdef  __cplusplus
}
#endif
