#pragma once

#include "llama-batch.h"

#include <algorithm>
#include <cstdint>

// Bounds a KVarN ubatch chunk by tail-ring evictions per graph (<= tail_tokens),
// not raw token count. Only widens single-sequence contiguous batches.
inline static uint32_t kvarn_tail_safe_ubatch_limit(
        const llama_batch_allocr & balloc,
        uint32_t default_limit,
        uint32_t n_sink_tokens,
        uint32_t n_tail_tokens) {
    const llama_batch & batch = balloc.get_batch();
    const uint32_t i0 = balloc.get_n_used();
    if (default_limit == 0 || i0 >= uint32_t(batch.n_tokens)) {
        return 0;
    }

    const auto fallback = [&]() {
        return std::max<uint32_t>(1, std::min<uint32_t>(default_limit, n_tail_tokens));
    };

    if (batch.pos == nullptr || batch.seq_id == nullptr || batch.n_seq_id == nullptr || n_tail_tokens == 0) {
        return fallback();
    }
    if (batch.n_seq_id[i0] != 1) {
        return fallback();
    }
    const llama_seq_id seq0 = batch.seq_id[i0][0];
    for (uint32_t i = i0; i < uint32_t(batch.n_tokens); ++i) {
        if (batch.n_seq_id[i] != 1 || batch.seq_id[i][0] != seq0) {
            return fallback();
        }
    }

    const uint32_t n_sink_tail_tokens = n_sink_tokens + n_tail_tokens;
    const llama_pos first_pos = batch.pos[i0];
    if (first_pos < 0) {
        return 1;
    }

    uint32_t n_safe = 0;
    uint32_t n_tail_evict = 0;
    llama_pos expected_pos = first_pos;

    for (uint32_t i = i0; i < uint32_t(batch.n_tokens) && n_safe < default_limit; ++i) {
        const llama_pos pos = batch.pos[i];
        if (pos < 0 || pos != expected_pos) {
            break;
        }

        if (uint32_t(pos) >= n_sink_tail_tokens) {
            if (n_tail_evict >= n_tail_tokens) {
                break;
            }
            ++n_tail_evict;
        }

        ++n_safe;
        ++expected_pos;
    }

    return std::max<uint32_t>(1, n_safe);
}

// True when the remaining batch is a single contiguous sequence (split_simple safe).
inline static bool kvarn_batch_is_single_seq_contiguous(const llama_batch_allocr & balloc) {
    const llama_batch & batch = balloc.get_batch();
    const uint32_t i0 = balloc.get_n_used();
    if (i0 >= uint32_t(batch.n_tokens)) {
        return false;
    }
    if (batch.pos == nullptr || batch.seq_id == nullptr || batch.n_seq_id == nullptr) {
        return false;
    }
    if (batch.n_seq_id[i0] != 1) {
        return false;
    }
    const llama_seq_id seq0 = batch.seq_id[i0][0];
    const llama_pos first_pos = batch.pos[i0];
    if (first_pos < 0) {
        return false;
    }
    llama_pos expected_pos = first_pos;
    for (uint32_t i = i0; i < uint32_t(batch.n_tokens); ++i) {
        if (batch.n_seq_id[i] != 1 || batch.seq_id[i][0] != seq0) {
            return false;
        }
        if (batch.pos[i] != expected_pos) {
            return false;
        }
        ++expected_pos;
    }
    return true;
}
