#include "kvarn.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cinttypes>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

// NVCC can vectorize shared-memory reduction loads at the end of the scratch
// region. Keep a small pad so sanitizer-visible overreads stay inside the
// dynamic shared allocation.
static constexpr size_t KVARN_ATTN_SHMEM_PAD_FLOATS = 8;
static constexpr uint32_t KVARN_ATTN_FRAME_FUSED_PAPER_FULL = 1;

// Monotonic epoch allocator for packed-body store invalidation. The active
// epoch is tracked per packed K-body pointer so one layer sealing a record does
// not invalidate every other layer's persistent dequant mirror.
#include <atomic>
#include <unordered_map>
static std::atomic<uint64_t> g_kvarn_body_store_epoch_next{1};
static std::atomic<uint64_t> g_kvarn_dequant_cache_trace_count{0};
static std::atomic<uint64_t> g_kvarn_attn_trace_count{0};
static std::atomic<uint64_t> g_kvarn_store_phase_trace_count{0};
static std::atomic<uint64_t> g_kvarn_body_record_dump_count{0};

struct kvarn_debug_store_context {
    int32_t  layer = -1;
    uint32_t record0 = 0;
    uint32_t n_records = 0;
    uint32_t n_heads = 0;
    uint32_t src_layout = 0;
    uint32_t records_cap = 0;
    const void * raw_mirror_key = nullptr;
};

static thread_local kvarn_debug_store_context g_kvarn_debug_store_context;

void ggml_cuda_kvarn_debug_set_store_context(
        int32_t layer,
        uint32_t record0,
        uint32_t n_records,
        uint32_t n_heads,
        uint32_t src_layout,
        uint32_t records_cap,
        const void * raw_mirror_key) {
    g_kvarn_debug_store_context = {
        layer,
        record0,
        n_records,
        n_heads,
        src_layout,
        records_cap,
        raw_mirror_key,
    };
}

struct kvarn_dequant_cache_entry {
    const void * k_body    = nullptr;
    uint32_t     n_records = 0;
    uint64_t     epoch     = 0;
    uint32_t     head_dim  = 0;
    uint32_t     group_size = 0;
    uint32_t     n_head_kv = 0;
    uint8_t      format    = 0;
};
struct kvarn_body_store_state {
    uint64_t epoch      = 0;
    uint64_t prev_epoch = 0;
    uint32_t dirty_from = 0;
};
// Dispatch is single-threaded per context, but these registries are shared by
// every CUDA context and by buffer teardown.
static std::unordered_map<const void *, kvarn_body_store_state> g_kvarn_body_store_states;
static std::unordered_map<const void *, kvarn_dequant_cache_entry> g_kvarn_dequant_cache;
static std::mutex g_kvarn_registry_mutex;

struct kvarn_raw_body_mirror_entry {
    float * k = nullptr;
    float * v = nullptr;
    uint32_t n_records_cap = 0;
    uint32_t n_heads = 0;
    uint32_t head_dim = 0;
    uint32_t group_size = 0;
};

static std::unordered_map<const void *, kvarn_raw_body_mirror_entry> g_kvarn_raw_body_mirrors;

static bool kvarn_pointer_in_range(const void * pointer, const void * base, size_t size) {
    if (pointer == nullptr || base == nullptr || size == 0) {
        return false;
    }
    const uintptr_t p = reinterpret_cast<uintptr_t>(pointer);
    const uintptr_t b = reinterpret_cast<uintptr_t>(base);
    return p >= b && p - b < size;
}

static void kvarn_cuda_free_checked(void * pointer, const char * label) {
    if (pointer == nullptr) {
        return;
    }
    const cudaError_t err = cudaFree(pointer);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "KVarN CUDA registry: cudaFree(%s) failed: %s\n", label, cudaGetErrorString(err));
        std::abort();
    }
}

void ggml_cuda_kvarn_release_buffer_range(const void * base, size_t size) {
    if (base == nullptr || size == 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_kvarn_registry_mutex);

    for (auto it = g_kvarn_raw_body_mirrors.begin(); it != g_kvarn_raw_body_mirrors.end();) {
        if (!kvarn_pointer_in_range(it->first, base, size)) {
            ++it;
            continue;
        }
        kvarn_cuda_free_checked(it->second.k, "raw mirror K");
        kvarn_cuda_free_checked(it->second.v, "raw mirror V");
        it = g_kvarn_raw_body_mirrors.erase(it);
    }

    for (auto it = g_kvarn_body_store_states.begin(); it != g_kvarn_body_store_states.end();) {
        if (kvarn_pointer_in_range(it->first, base, size)) {
            it = g_kvarn_body_store_states.erase(it);
        } else {
            ++it;
        }
    }

    for (auto it = g_kvarn_dequant_cache.begin(); it != g_kvarn_dequant_cache.end();) {
        if (kvarn_pointer_in_range(it->first, base, size) ||
                kvarn_pointer_in_range(it->second.k_body, base, size)) {
            it = g_kvarn_dequant_cache.erase(it);
        } else {
            ++it;
        }
    }

    if (kvarn_pointer_in_range(g_kvarn_debug_store_context.raw_mirror_key, base, size)) {
        g_kvarn_debug_store_context.raw_mirror_key = nullptr;
    }
}

void ggml_cuda_kvarn_debug_get_raw_mirror_stats(size_t * count, size_t * allocated_bytes) {
    std::lock_guard<std::mutex> lock(g_kvarn_registry_mutex);
    size_t bytes = 0;
    for (const auto & item : g_kvarn_raw_body_mirrors) {
        const kvarn_raw_body_mirror_entry & entry = item.second;
        const size_t elements = size_t(entry.n_records_cap)*entry.n_heads*entry.head_dim*entry.group_size;
        bytes += (entry.k != nullptr ? elements*sizeof(float) : 0) +
                 (entry.v != nullptr ? elements*sizeof(float) : 0);
    }
    if (count != nullptr) {
        *count = g_kvarn_raw_body_mirrors.size();
    }
    if (allocated_bytes != nullptr) {
        *allocated_bytes = bytes;
    }
}

static bool kvarn_dequant_cache_trace_enabled();
static int  kvarn_dequant_cache_trace_limit();

void ggml_cuda_kvarn_mark_body_store_records(const void * k_body, uint32_t first_record, uint32_t n_records) {
    if (k_body == nullptr) {
        return;
    }
    const uint64_t epoch = g_kvarn_body_store_epoch_next.fetch_add(1, std::memory_order_relaxed) + 1;
    std::lock_guard<std::mutex> lock(g_kvarn_registry_mutex);
    kvarn_body_store_state & state = g_kvarn_body_store_states[k_body];
    state.prev_epoch = state.epoch;
    state.epoch = epoch;
    state.dirty_from = n_records == 0 ? 0 : first_record;
}

void ggml_cuda_kvarn_mark_body_store(const void * k_body) {
    ggml_cuda_kvarn_mark_body_store_records(k_body, 0, 0);
}

void ggml_cuda_kvarn_invalidate_restored_body(const void * k_body) {
    if (k_body == nullptr) {
        return;
    }

    const uint64_t epoch = g_kvarn_body_store_epoch_next.fetch_add(1, std::memory_order_relaxed) + 1;
    std::lock_guard<std::mutex> lock(g_kvarn_registry_mutex);

    kvarn_body_store_state & state = g_kvarn_body_store_states[k_body];
    state.prev_epoch = state.epoch;
    state.epoch = epoch;
    state.dirty_from = 0;

    for (auto it = g_kvarn_dequant_cache.begin(); it != g_kvarn_dequant_cache.end();) {
        if (it->second.k_body == k_body) {
            it = g_kvarn_dequant_cache.erase(it);
        } else {
            ++it;
        }
    }

    const auto raw = g_kvarn_raw_body_mirrors.find(k_body);
    if (raw != g_kvarn_raw_body_mirrors.end()) {
        kvarn_cuda_free_checked(raw->second.k, "restored raw mirror K");
        kvarn_cuda_free_checked(raw->second.v, "restored raw mirror V");
        g_kvarn_raw_body_mirrors.erase(raw);
    }

    if (g_kvarn_debug_store_context.raw_mirror_key == k_body) {
        g_kvarn_debug_store_context.raw_mirror_key = nullptr;
    }
}

uint64_t ggml_cuda_kvarn_debug_get_body_epoch(const void * k_body) {
    std::lock_guard<std::mutex> lock(g_kvarn_registry_mutex);
    const auto it = g_kvarn_body_store_states.find(k_body);
    return it != g_kvarn_body_store_states.end() ? it->second.epoch : 0;
}

static kvarn_body_store_state kvarn_body_store_state_for_unlocked(const void * k_body) {
    auto it = g_kvarn_body_store_states.find(k_body);
    if (it != g_kvarn_body_store_states.end()) {
        return it->second;
    }
    return {};
}

static uint32_t kvarn_dequant_cache_refill_from(
        const void * scratch_key,
        const void * k_body,
        uint32_t n_records,
        uint32_t n_head_kv,
        uint32_t head_dim,
        uint32_t group_size,
        uint8_t format) {
    if (scratch_key == nullptr || k_body == nullptr || n_records == 0) {
        return 0;
    }

    std::lock_guard<std::mutex> lock(g_kvarn_registry_mutex);
    const kvarn_body_store_state store_state = kvarn_body_store_state_for_unlocked(k_body);
    const uint64_t epoch = store_state.epoch;
    kvarn_dequant_cache_entry & e = g_kvarn_dequant_cache[scratch_key];
    uint32_t dequant_from = 0;
    bool compatible = false;
    if (e.k_body == k_body && e.head_dim == head_dim && e.group_size == group_size &&
            e.n_head_kv == n_head_kv && e.format == format && e.n_records <= n_records) {
        compatible = true;
        if (e.epoch == epoch) {
            dequant_from = e.n_records;
        } else if (e.epoch == store_state.prev_epoch) {
            dequant_from = store_state.dirty_from < e.n_records ? store_state.dirty_from : e.n_records;
        }
    }

    if (kvarn_dequant_cache_trace_enabled()) {
        const uint64_t trace_i = g_kvarn_dequant_cache_trace_count.fetch_add(1, std::memory_order_relaxed);
        if (trace_i < uint64_t(kvarn_dequant_cache_trace_limit())) {
            std::fprintf(stderr,
                    "KVarN CUDA dequant-cache trace: %s format=%s scratch=%p k_body=%p epoch=%" PRIu64
                    " prev_epoch=%" PRIu64 " dirty_from=%u cached_epoch=%" PRIu64
                    " cached_records=%u active_records=%u refill_from=%u head_dim=%u n_head_kv=%u\n",
                    compatible && dequant_from == n_records ? "hit" : (dequant_from == 0 ? "miss" : "partial"),
                    format == 1 ? "f32" : "f16",
                    scratch_key, k_body, epoch, store_state.prev_epoch, store_state.dirty_from, e.epoch,
                    e.n_records, n_records, dequant_from, head_dim, n_head_kv);
        }
    }

    if (dequant_from < n_records) {
        e.k_body = k_body;
        e.n_records = n_records;
        e.epoch = epoch;
        e.head_dim = head_dim;
        e.group_size = group_size;
        e.n_head_kv = n_head_kv;
        e.format = format;
    }

    return dequant_from;
}

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

static int kvarn_env_int(const char * name, int default_value) {
    const char * env = std::getenv(name);
    if (env == nullptr) {
        return default_value;
    }

    char * end = nullptr;
    errno = 0;
    const long value = std::strtol(env, &end, 10);
    if (env[0] == '\0' || end == nullptr || *end != '\0' || errno == ERANGE ||
            value <= 0 || value > 1000000) {
        std::fprintf(stderr, "invalid KVarN CUDA environment integer %s=%s; expected integer in [1,1000000]\n", name, env);
        std::abort();
    }
    return int(value);
}

static int kvarn_env_qt_override() {
    const char * env = std::getenv("LLAMA_KVARN_ATTN_WARPQK_FORCE_QT");
    if (env == nullptr) {
        return 0;
    }

    char * end = nullptr;
    errno = 0;
    const long value = std::strtol(env, &end, 10);
    if (env[0] == '\0' || end == nullptr || *end != '\0' || errno == ERANGE ||
            (value != 1 && value != 4 && value != 8 && value != 16)) {
        std::fprintf(stderr,
                "invalid KVarN CUDA environment integer LLAMA_KVARN_ATTN_WARPQK_FORCE_QT=%s; expected one of 1,4,8,16\n",
                env);
        std::abort();
    }
    return int(value);
}

static bool kvarn_enable_256d_warpqk() {
    return kvarn_env_flag("LLAMA_KVARN_ATTN_ENABLE_256D_WARPQK");
}

static bool kvarn_enable_512d_warpqk() {
    return kvarn_env_flag("LLAMA_KVARN_ATTN_ENABLE_512D_WARPQK");
}

static bool kvarn_disable_warpqk() {
    return kvarn_env_flag("LLAMA_KVARN_ATTN_DISABLE_WARPQK");
}

static bool kvarn_disable_body_mirror() {
    return kvarn_env_flag("LLAMA_KVARN_ATTN_DISABLE_BODY_MIRROR");
}

static bool kvarn_debug_raw_body_k_enabled() {
    return kvarn_env_flag("LLAMA_KVARN_DEBUG_RAW_BODY_K");
}

static bool kvarn_debug_raw_body_v_enabled() {
    return kvarn_env_flag("LLAMA_KVARN_DEBUG_RAW_BODY_V");
}

static bool kvarn_debug_raw_body_enabled() {
    return kvarn_debug_raw_body_k_enabled() || kvarn_debug_raw_body_v_enabled();
}

static bool kvarn_debug_raw_body_capture_enabled() {
    return kvarn_debug_raw_body_enabled() || kvarn_env_flag("LLAMA_KVARN_DEBUG_CAPTURE_RAW_BODY_MIRROR");
}

static bool kvarn_debug_raw_body_scalar_qt_enabled() {
    return kvarn_env_flag("LLAMA_KVARN_DEBUG_RAW_BODY_SCALAR_QT");
}

static bool kvarn_paper_frame_enabled() {
    if (kvarn_env_flag("LLAMA_KVARN_DISABLE_PAPER_FRAME")) {
        return false;
    }

    // Must match kvarn_graph_paper_frame_enabled() in llama-graph.cpp: the
    // graph decides which frame sink/tail/pending/q live in, and every store
    // path here must agree with it. The historical opt-in default diverged
    // from the graph default and silently changed the body frame; the
    // catastrophic paper-frame results came from pending-sourced tiles being
    // rotated a second time (H*H = identity), not from default-on itself.
    // Pending-sourced ops now carry an explicit src marker instead.
    (void) kvarn_env_flag("LLAMA_KVARN_ENABLE_PAPER_FRAME");
    return true;
}

static bool kvarn_global_norm_enabled() {
    return !kvarn_env_flag("LLAMA_KVARN_DISABLE_GLOBAL_NORM");
}

// MSE-optimal clip half-width in std units for uniform asymmetric RTN on
// variance-normalized (~Gaussian) rows; 0 disables clipping. Full range is
// already near-optimal for bits >= 4. Must match kvarn_rtn_clip_sigma() in
// src/llama-kv-cache-kvarn.cpp.
static float kvarn_rtn_clip_sigma_host(uint32_t bits) {
    if (kvarn_env_flag("LLAMA_KVARN_DISABLE_RTN_CLIP")) {
        return 0.0f;
    }
    switch (bits) {
        case 1: return 1.0f;
        case 2: return 1.5f;
        case 3: return 2.05f;
        default: return 0.0f;
    }
}

static bool kvarn_paper_mixed_frame_enabled() {
    if (!kvarn_env_flag("LLAMA_KVARN_PAPER_MIXED_FRAME")) {
        return false;
    }
    if (!kvarn_env_flag("LLAMA_KVARN_UNSAFE_ALLOW_PAPER_MIXED_FRAME")) {
        std::fprintf(stderr,
                "LLAMA_KVARN_PAPER_MIXED_FRAME is diagnostic-only because it can silently "
                "change K/V store-attention frame contracts; also set "
                "LLAMA_KVARN_UNSAFE_ALLOW_PAPER_MIXED_FRAME=1 only for targeted frame A/B tests.\n");
        std::abort();
    }
    return kvarn_paper_frame_enabled();
}

static bool kvarn_experimental_turbo_v_enabled() {
    return kvarn_env_flag("LLAMA_KVARN_EXPERIMENTAL_TURBO_V");
}

static bool kvarn_experimental_turbo_v_layout_enabled() {
    return kvarn_env_flag("LLAMA_KVARN_EXPERIMENTAL_TURBO_V_LAYOUT");
}

static bool kvarn_experimental_turbo_v_supported(uint32_t head_dim, uint32_t group_size, uint32_t value_bits) {
    return group_size == 128 && (head_dim % 128) == 0 && (value_bits == 2 || value_bits == 4);
}

static bool kvarn_experimental_turbo_v_active(uint32_t head_dim, uint32_t group_size, uint32_t value_bits) {
    if (!kvarn_experimental_turbo_v_enabled() && !kvarn_experimental_turbo_v_layout_enabled()) {
        return false;
    }
    if (!kvarn_experimental_turbo_v_supported(head_dim, group_size, value_bits)) {
        std::fprintf(stderr,
                "experimental Turbo V requires group_size=128, head_dim multiple of 128, and value_bits 2 or 4; got head_dim=%u group_size=%u value_bits=%u\n",
                head_dim, group_size, value_bits);
        std::abort();
    }
    return true;
}

static __host__ __device__ __forceinline__ uint32_t kvarn_turbo_v_block_bytes(uint32_t value_bits) {
    return value_bits == 2 ? 34u : 68u;
}

static __host__ __device__ __forceinline__ size_t kvarn_turbo_v_block_offset(
        uint32_t head_dim,
        uint32_t value_bits,
        uint32_t g,
        uint32_t block128) {
    const uint32_t blocks_per_row = head_dim/128u;
    return (size_t(g)*blocks_per_row + block128)*kvarn_turbo_v_block_bytes(value_bits);
}

static constexpr uint32_t KVARN_KQ_MASK_TYPE_CAUSAL = 3u;

static __host__ __device__ __forceinline__ int32_t kvarn_causal_mask_limit(
        uint32_t n_tokens,
        uint32_t n_queries,
        uint32_t iq) {
    if (n_queries == 0) {
        return -1;
    }
    if (n_tokens >= n_queries) {
        return int32_t(n_tokens - n_queries + iq);
    }
    return int32_t(iq);
}

static bool kvarn_log_std_sinkhorn_enabled() {
    if (kvarn_env_flag("LLAMA_KVARN_DISABLE_LOG_STD_SINKHORN")) {
        return false;
    }
    const char * env = std::getenv("LLAMA_KVARN_ENABLE_LOG_STD_SINKHORN");
    if (env != nullptr) {
        return kvarn_env_flag("LLAMA_KVARN_ENABLE_LOG_STD_SINKHORN");
    }
    return true;
}

static bool kvarn_dequant_cache_trace_enabled() {
    return kvarn_env_flag("LLAMA_KVARN_DEQUANT_CACHE_TRACE");
}

static bool kvarn_attn_trace_enabled() {
    return kvarn_env_flag("LLAMA_KVARN_ATTN_TRACE");
}

static bool kvarn_attn_trace_claim() {
    if (!kvarn_attn_trace_enabled()) {
        return false;
    }
    const int limit = kvarn_env_int("LLAMA_KVARN_ATTN_TRACE_LIMIT", 64);
    return g_kvarn_attn_trace_count.fetch_add(1, std::memory_order_relaxed) < uint64_t(limit);
}

static int kvarn_dequant_cache_trace_limit() {
    return kvarn_env_int("LLAMA_KVARN_DEQUANT_CACHE_TRACE_LIMIT", 256);
}

static bool kvarn_store_phase_trace_claim() {
    if (!kvarn_env_flag("LLAMA_KVARN_STORE_TRACE")) {
        return false;
    }
    const int limit = kvarn_env_int("LLAMA_KVARN_STORE_TRACE_LIMIT", 64);
    return g_kvarn_store_phase_trace_count.fetch_add(1, std::memory_order_relaxed) < uint64_t(limit);
}

struct kvarn_body_record_dump_state {
    bool active = false;
    uint64_t call_index = 0;
    int32_t layer = -1;
    uint32_t record = 0;
    uint32_t head = 0;
    uint32_t record0 = 0;
    uint32_t n_records = 0;
    uint32_t n_heads = 0;
    uint32_t src_layout = 0;
    std::string dir;
};

static int kvarn_env_optional_nonneg_int(const char * name) {
    const char * env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return -1;
    }
    char * end = nullptr;
    errno = 0;
    const long value = std::strtol(env, &end, 10);
    if (end == nullptr || *end != '\0' || errno == ERANGE || value < 0 || value > 1000000) {
        std::fprintf(stderr, "invalid KVarN CUDA environment integer %s=%s; expected integer in [0,1000000]\n", name, env);
        std::abort();
    }
    return int(value);
}

static std::string kvarn_join_path(const std::string & dir, const char * leaf) {
    if (dir.empty()) {
        return std::string(leaf);
    }
    const char last = dir[dir.size() - 1];
    if (last == '/' || last == '\\') {
        return dir + leaf;
    }
    return dir + "/" + leaf;
}

static void kvarn_mkdir_one(const std::string & path) {
    if (path.empty()) {
        return;
    }
#ifdef _WIN32
    (void) _mkdir(path.c_str());
#else
    (void) mkdir(path.c_str(), 0777);
#endif
}

static void kvarn_create_directories(const std::string & path) {
    std::string cur;
    cur.reserve(path.size());
    for (size_t i = 0; i < path.size(); ++i) {
        const char c = path[i];
        cur.push_back(c);
        if ((c == '/' || c == '\\') && cur.size() > 1 && cur[cur.size() - 2] != ':') {
            kvarn_mkdir_one(cur);
        }
    }
    kvarn_mkdir_one(path);
}

static void kvarn_write_binary_file(const std::string & path, const void * data, size_t nbytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "KVarN body-record dump: failed to open %s\n", path.c_str());
        return;
    }
    f.write(reinterpret_cast<const char *>(data), std::streamsize(nbytes));
    if (!f) {
        std::fprintf(stderr, "KVarN body-record dump: failed to write %s\n", path.c_str());
    }
}

static void kvarn_dump_device_buffer(const std::string & path, const void * dev, size_t nbytes) {
    std::vector<uint8_t> host(nbytes, uint8_t(0));
    cudaError_t err = cudaMemcpy(host.data(), dev, nbytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "KVarN body-record dump: cudaMemcpy failed for %s: %s\n",
                path.c_str(), cudaGetErrorString(err));
        return;
    }
    kvarn_write_binary_file(path, host.data(), host.size());
}

static kvarn_body_record_dump_state kvarn_body_record_dump_claim(uint32_t record, uint32_t head) {
    kvarn_body_record_dump_state dump;
    dump.layer = g_kvarn_debug_store_context.layer;
    dump.record = record;
    dump.head = head;
    dump.record0 = g_kvarn_debug_store_context.record0;
    dump.n_records = g_kvarn_debug_store_context.n_records;
    dump.n_heads = g_kvarn_debug_store_context.n_heads;
    dump.src_layout = g_kvarn_debug_store_context.src_layout;
    if (record == UINT32_MAX || head == UINT32_MAX) {
        return dump;
    }
    if (!kvarn_env_flag("LLAMA_KVARN_DEBUG_BODY_RECORD_DUMP")) {
        return dump;
    }

    const int layer_filter = kvarn_env_optional_nonneg_int("LLAMA_KVARN_DEBUG_BODY_RECORD_LAYER");
    if (layer_filter >= 0 && dump.layer != layer_filter) {
        return dump;
    }
    const int src_layout_filter = kvarn_env_optional_nonneg_int("LLAMA_KVARN_DEBUG_BODY_SRC_LAYOUT");
    if (src_layout_filter >= 0 && dump.src_layout != uint32_t(src_layout_filter)) {
        return dump;
    }
    const int record_filter = kvarn_env_optional_nonneg_int("LLAMA_KVARN_DEBUG_BODY_RECORD");
    if (record_filter >= 0 && uint32_t(record_filter) != record) {
        return dump;
    }
    const int head_filter = kvarn_env_optional_nonneg_int("LLAMA_KVARN_DEBUG_BODY_HEAD");
    if (head_filter >= 0 && uint32_t(head_filter) != head) {
        return dump;
    }

    const uint64_t call_index = g_kvarn_body_record_dump_count.fetch_add(1, std::memory_order_relaxed);
    const int call_filter = kvarn_env_optional_nonneg_int("LLAMA_KVARN_DEBUG_BODY_RECORD_CALL");
    if (call_filter >= 0 && call_index != uint64_t(call_filter)) {
        return dump;
    }
    const int call_start_filter = kvarn_env_optional_nonneg_int("LLAMA_KVARN_DEBUG_BODY_RECORD_CALL_START");
    if (call_filter < 0 && call_start_filter >= 0 && call_index < uint64_t(call_start_filter)) {
        return dump;
    }
    const int call_end_filter = kvarn_env_optional_nonneg_int("LLAMA_KVARN_DEBUG_BODY_RECORD_CALL_END");
    if (call_filter < 0 && call_end_filter >= 0 && call_index > uint64_t(call_end_filter)) {
        return dump;
    }
    const int limit = kvarn_env_int("LLAMA_KVARN_DEBUG_BODY_RECORD_LIMIT", 1);
    const uint64_t limit_index_base = call_start_filter >= 0 ? uint64_t(call_start_filter) : uint64_t(0);
    if (call_filter < 0 && limit >= 0 && call_index >= limit_index_base + uint64_t(limit)) {
        return dump;
    }

    const char * dir_env = std::getenv("LLAMA_KVARN_DEBUG_BODY_RECORD_DIR");
    const std::string root = dir_env != nullptr && dir_env[0] != '\0' ?
        std::string(dir_env) : std::string("artifacts/kvarn-body-record/default");
    std::ostringstream stem;
    stem << "store_" << std::setw(6) << std::setfill('0') << call_index
         << "_h" << head << "_r" << record;
    dump.dir = kvarn_join_path(root, stem.str().c_str());
    kvarn_create_directories(dump.dir);
    dump.active = true;
    dump.call_index = call_index;
    return dump;
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

static bool kvarn_cuda_dynamic_shmem_optin_fits(size_t nbytes) {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return true;
    }

    cudaDeviceProp prop = {};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        return true;
    }

    const size_t optin = prop.sharedMemPerBlockOptin > 0 ? size_t(prop.sharedMemPerBlockOptin) : size_t(prop.sharedMemPerBlock);
    return nbytes <= optin;
}

template <typename Kernel>
static bool kvarn_cuda_prepare_dynamic_shmem(Kernel kernel, size_t nbytes) {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return true;
    }

    cudaDeviceProp prop = {};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        return true;
    }
    if (nbytes <= size_t(prop.sharedMemPerBlock)) {
        return true;
    }
    if (prop.sharedMemPerBlockOptin <= 0 || nbytes > size_t(prop.sharedMemPerBlockOptin)) {
        return false;
    }

    return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(nbytes)) == cudaSuccess;
}

static void kvarn_cuda_trace_launch_error(
        const char * label,
        uint32_t n_queries,
        uint32_t n_head,
        uint32_t n_head_kv,
        uint32_t n_sink,
        uint32_t n_records,
        uint32_t n_pending,
        uint32_t n_tail,
        uint32_t head_dim,
        uint32_t n_tokens,
        int grid,
        int block,
        size_t shmem) {
    const cudaError_t err = cudaPeekAtLastError();
    if (err == cudaSuccess) {
        return;
    }
    std::fprintf(stderr,
            "KVarN CUDA mixed-attn launch error: branch=%s err=%s n_queries=%u n_head=%u n_head_kv=%u"
            " sink=%u records=%u pending=%u tail=%u tokens=%u head_dim=%u grid=%d block=%d shmem=%zu\n",
            label, cudaGetErrorString(err), n_queries, n_head, n_head_kv,
            n_sink, n_records, n_pending, n_tail, n_tokens, head_dim, grid, block, shmem);
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
    if (bits == 8) {
        dst[i] = uint8_t(q);
        return;
    }

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

static __constant__ float KVARN_TURBO_WHT_SIGNS1_128[128] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
     1.0f,-1.0f, 1.0f, -1.0f,  1.0f,-1.0f, -1.0f, 1.0f,  1.0f, 1.0f,-1.0f, 1.0f, 1.0f,-1.0f,-1.0f,-1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f,  1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f,-1.0f,-1.0f, 1.0f,-1.0f, 1.0f,
     1.0f, 1.0f, 1.0f, -1.0f, -1.0f,-1.0f, -1.0f,-1.0f,  1.0f,-1.0f, 1.0f, 1.0f, 1.0f, 1.0f,-1.0f, 1.0f,
    -1.0f,-1.0f, 1.0f, -1.0f, -1.0f,-1.0f,  1.0f,-1.0f, -1.0f,-1.0f, 1.0f,-1.0f,-1.0f,-1.0f, 1.0f, 1.0f,
     1.0f,-1.0f,-1.0f,  1.0f,  1.0f, 1.0f, -1.0f,-1.0f,  1.0f, 1.0f,-1.0f, 1.0f, 1.0f,-1.0f, 1.0f,-1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f,  1.0f,-1.0f,  1.0f,-1.0f,  1.0f, 1.0f, 1.0f, 1.0f,-1.0f, 1.0f,-1.0f, 1.0f,
     1.0f,-1.0f, 1.0f,  1.0f, -1.0f,-1.0f, -1.0f,-1.0f, -1.0f, 1.0f, 1.0f,-1.0f, 1.0f, 1.0f,-1.0f, 1.0f,
};

static __constant__ float KVARN_TURBO_WHT_SIGNS2_128[128] = {
     1.0f, 1.0f, 1.0f,  1.0f, -1.0f, 1.0f,  1.0f,-1.0f,  1.0f,-1.0f,-1.0f,-1.0f, 1.0f,-1.0f,-1.0f,-1.0f,
     1.0f, 1.0f,-1.0f, -1.0f,  1.0f,-1.0f,  1.0f,-1.0f,  1.0f,-1.0f,-1.0f, 1.0f,-1.0f, 1.0f, 1.0f, 1.0f,
     1.0f, 1.0f,-1.0f, -1.0f, -1.0f, 1.0f, -1.0f,-1.0f, -1.0f,-1.0f,-1.0f,-1.0f, 1.0f, 1.0f, 1.0f,-1.0f,
     1.0f,-1.0f, 1.0f,  1.0f,  1.0f,-1.0f, -1.0f, 1.0f, -1.0f,-1.0f,-1.0f,-1.0f,-1.0f,-1.0f, 1.0f, 1.0f,
     1.0f,-1.0f, 1.0f, -1.0f, -1.0f,-1.0f, -1.0f, 1.0f, -1.0f, 1.0f,-1.0f, 1.0f,-1.0f,-1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f,-1.0f,  1.0f,  1.0f,-1.0f,  1.0f,-1.0f, -1.0f,-1.0f,-1.0f, 1.0f,-1.0f,-1.0f, 1.0f,-1.0f,
     1.0f,-1.0f, 1.0f,  1.0f,  1.0f,-1.0f, -1.0f, 1.0f, -1.0f, 1.0f,-1.0f, 1.0f, 1.0f,-1.0f,-1.0f, 1.0f,
    -1.0f, 1.0f,-1.0f,  1.0f,  1.0f,-1.0f,  1.0f,-1.0f,  1.0f,-1.0f,-1.0f,-1.0f,-1.0f,-1.0f, 1.0f,-1.0f,
};

static __constant__ float KVARN_TURBO_CENTROIDS_2BIT[4] = {
    -0.133462f, -0.039994f, 0.039994f, 0.133462f,
};

static __constant__ float KVARN_TURBO_CENTROIDS_4BIT[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f,
};

static __device__ __forceinline__ float kvarn_turbo_centroid(uint32_t bits, uint32_t idx) {
    return bits == 2 ? KVARN_TURBO_CENTROIDS_2BIT[idx & 3u] : KVARN_TURBO_CENTROIDS_4BIT[idx & 15u];
}

static __device__ __forceinline__ uint32_t kvarn_turbo_nearest_centroid(float v, uint32_t bits) {
    if (bits == 2) {
        if (v < -0.086728f) {
            return 0;
        }
        if (v < 0.0f) {
            return 1;
        }
        if (v < 0.086728f) {
            return 2;
        }
        return 3;
    }

    uint32_t best = 0;
    float best_dist = fabsf(v - KVARN_TURBO_CENTROIDS_4BIT[0]);
    for (uint32_t i = 1; i < 16; ++i) {
        const float dist = fabsf(v - KVARN_TURBO_CENTROIDS_4BIT[i]);
        if (dist < best_dist) {
            best_dist = dist;
            best = i;
        }
    }
    return best;
}

static __device__ __forceinline__ float kvarn_turbo_v_dequant_rotated(
        const uint8_t * __restrict__ v_body,
        const float * __restrict__ v_scales,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t value_bits,
        uint32_t g,
        uint32_t d,
        uint32_t turbo_v_mode) {
    if (turbo_v_mode == 0u || turbo_v_mode == 3u || turbo_v_mode == 4u || turbo_v_mode == 5u) {
        const float * v_s_col = v_scales;
        const float * v_s_row = v_scales + head_dim;
        const float * v_zp    = v_scales + head_dim + group_size;
        const uint32_t vq = kvarn_unpack_one(v_body, value_bits, size_t(g)*head_dim + d);
        float value = (float(vq)*v_s_row[g] + v_zp[g])*v_s_col[d];
        if (turbo_v_mode == 3u || turbo_v_mode == 4u || turbo_v_mode == 5u) {
            const size_t prefix = (size_t(group_size)*head_dim*value_bits + 7u)/8u;
            const __half * factors = reinterpret_cast<const __half *>(v_body + prefix);
            const size_t factor_f16s = size_t(group_size) + head_dim;
            value += __half2float(factors[g])*__half2float(factors[group_size + d]);
            if (turbo_v_mode == 4u) {
                const __half * component2 = factors + factor_f16s;
                value += __half2float(component2[g])*__half2float(component2[group_size + d]);
            } else if (turbo_v_mode == 5u) {
                const __half * component2 = factors + factor_f16s;
                const __half * component3 = component2 + factor_f16s;
                value += __half2float(component2[g])*__half2float(component2[group_size + d]);
                value += __half2float(component3[g])*__half2float(component3[group_size + d]);
            }
        }
        return value;
    }

    const uint32_t block128 = d / 128u;
    uint32_t q = 0;
    float norm = 0.0f;
    if (turbo_v_mode == 2u) {
        const uint32_t j = d & 127u;
        const size_t block_off = kvarn_turbo_v_block_offset(head_dim, value_bits, g, block128);
        norm = __half2float(*reinterpret_cast<const __half *>(v_body + block_off));
        if (value_bits == 2) {
            const uint8_t * qs = v_body + block_off + 2u;
            q = (uint32_t(qs[j >> 2]) >> ((j & 3u)*2u)) & 0x3u;
        } else {
            const uint8_t * qs = v_body + block_off + 4u;
            q = (uint32_t(qs[j >> 1]) >> ((j & 1u)*4u)) & 0x0fu;
        }
    } else {
        q = kvarn_unpack_one(v_body, value_bits, size_t(g)*head_dim + d);
        norm = v_scales[size_t(g)*(head_dim/128u) + block128];
    }
    return kvarn_turbo_centroid(value_bits, q)*norm;
}

static __device__ __forceinline__ float kvarn_sparse_d512_v_base(
        const uint8_t * __restrict__ v_body,
        const float * __restrict__ v_scales,
        uint32_t g,
        uint32_t d) {
    const uint32_t q = kvarn_unpack_one(v_body, 2u, size_t(g)*512u + d);
    return (float(q)*v_scales[512u + g] + v_scales[640u + g])*v_scales[d];
}

static __device__ __forceinline__ float kvarn_sparse_d512_v_dequant_rotated(
        const uint8_t * __restrict__ v_body,
        const float * __restrict__ v_scales,
        uint32_t g,
        uint32_t d) {
    float value = kvarn_sparse_d512_v_base(v_body, v_scales, g, d);
    const uint16_t * col_start = reinterpret_cast<const uint16_t *>(v_body + 16384u);
    const uint8_t * rows = v_body + 17408u;
    const __half * values = reinterpret_cast<const __half *>(v_body + 18774u);
    const uint32_t begin = col_start[d];
    const uint32_t end = d + 1u < 512u ? col_start[d + 1u] : 1365u;
    if (begin <= end && end <= 1365u) {
        for (uint32_t i = begin; i < end; ++i) {
            if (rows[i] == g) {
                value += __half2float(values[i]);
                break;
            }
        }
    }
    return value;
}

static __device__ __forceinline__ bool kvarn_sparse_d512_column_bounds(
        const uint8_t * __restrict__ v_body,
        uint32_t d,
        uint32_t & begin,
        uint32_t & end) {
    const uint16_t * col_start = reinterpret_cast<const uint16_t *>(v_body + 16384u);
    begin = col_start[d];
    end = d + 1u < 512u ? col_start[d + 1u] : 1365u;
    return begin <= end && end <= 1365u;
}

static __device__ __forceinline__ float kvarn_sparse_d512_weighted_column(
        const uint8_t * __restrict__ v_body,
        const float * __restrict__ probs,
        uint32_t d) {
    uint32_t begin, end;
    if (!kvarn_sparse_d512_column_bounds(v_body, d, begin, end)) {
        return 0.0f;
    }
    const uint8_t * rows = v_body + 17408u;
    const __half * values = reinterpret_cast<const __half *>(v_body + 18774u);
    float sum = 0.0f;
    for (uint32_t i = begin; i < end; ++i) {
        if (rows[i] < 128u) {
            sum += probs[rows[i]]*__half2float(values[i]);
        }
    }
    return sum;
}

static __device__ __forceinline__ float kvarn_kq_mask_bias(
        const void * __restrict__ kq_mask,
        uint32_t kq_mask_type,
        size_t kq_mask_stride_token_bytes,
        uint32_t t,
        int32_t causal_limit) {
    if (kq_mask_type == KVARN_KQ_MASK_TYPE_CAUSAL) {
        return int32_t(t) <= causal_limit ? 0.0f : -INFINITY;
    }
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

static __device__ __forceinline__ float kvarn_softcap_attn_score(
        float sum,
        float scale,
        float logit_softcap) {
    float score = sum*scale;
    if (logit_softcap != 0.0f) {
        score = logit_softcap*tanhf(score/logit_softcap);
    }
    return score;
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

static __device__ void kvarn_fwht_shared_normalized(float * shmem, uint32_t n) {
    for (uint32_t step = 1; step < n; step <<= 1) {
        for (uint32_t c = threadIdx.x; c < n; c += blockDim.x) {
            if ((c & step) == 0) {
                const uint32_t c1 = c + step;
                const float a = shmem[c];
                const float b = shmem[c1];
                shmem[c]  = a + b;
                shmem[c1] = a - b;
            }
        }
        __syncthreads();
    }

    const float norm = rsqrtf(float(n));
    for (uint32_t c = threadIdx.x; c < n; c += blockDim.x) {
        shmem[c] *= norm;
    }
    __syncthreads();
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

// Shared clip-bound refinement: shrink the RTN range to mean +/- clip_sigma*std
// (never wider than the observed min/max). Must match the CPU reference in
// src/llama-kv-cache-kvarn.cpp quantize_asym_per_row().
static __device__ __forceinline__ void kvarn_rtn_apply_clip(
        float & mn, float & mx, double sum, double ss, uint32_t n, float clip_sigma) {
    if (clip_sigma <= 0.0f || n <= 1) {
        return;
    }
    const double dn = double(n);
    const float mu = float(sum/dn);
    const float sd = float(sqrt(fmax(0.0, ss/dn - (sum/dn)*(sum/dn))));
    const float lo = fmaxf(mn, mu - clip_sigma*sd);
    const float hi = fminf(mx, mu + clip_sigma*sd);
    if (hi > lo) {
        mn = lo;
        mx = hi;
    }
}

static void kvarn_validate_v_mode(uint32_t head_dim, uint32_t group_size, uint32_t value_bits, uint32_t mode) {
    const bool valid = mode == 0u ||
        ((mode == 1u || mode == 2u) && group_size == 128u && head_dim%128u == 0u &&
            (value_bits == 2u || value_bits == 4u)) ||
        ((mode == 3u || mode == 4u || mode == 5u) && group_size == 128u &&
            (head_dim == 256u || head_dim == 512u) && value_bits == 2u) ||
        (mode == 6u && head_dim == 512u && group_size == 128u && value_bits == 2u);
    if (!valid) {
        std::fprintf(stderr,
                "KVarN CUDA rejected V layout mode=%u head_dim=%u group_size=%u value_bits=%u\n",
                mode, head_dim, group_size, value_bits);
        std::abort();
    }
}

static __global__ void kvarn_v_r1_residual_kernel(
        const float * __restrict__ v_raw,
        const uint8_t * __restrict__ v_body,
        const float * __restrict__ v_scales,
        uint32_t n_tiles,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t value_bits,
        uint32_t n_heads,
        size_t raw_tile_stride_floats,
        size_t body_record_stride_bytes,
        size_t body_head_stride_bytes,
        size_t scale_record_stride_floats,
        size_t scale_head_stride_floats,
        uint32_t component) {
    const uint32_t tile = blockIdx.x;
    if (tile >= n_tiles || group_size != 128u || (head_dim != 256u && head_dim != 512u) || value_bits != 2u) {
        return;
    }

    extern __shared__ float smem[];
    float * u = smem;
    float * w = u + group_size;
    float * candidate = w + head_dim;
    __shared__ uint32_t best_d;
    __shared__ uint32_t degenerate;
    __shared__ float scalar;

    const uint32_t record_i = tile/n_heads;
    const uint32_t head_i = tile - record_i*n_heads;
    const float * raw = v_raw + size_t(tile)*raw_tile_stride_floats;
    uint8_t * record = const_cast<uint8_t *>(v_body + size_t(record_i)*body_record_stride_bytes + size_t(head_i)*body_head_stride_bytes);
    const float * scales = v_scales + size_t(record_i)*scale_record_stride_floats + size_t(head_i)*scale_head_stride_floats;
    const float * s_col = scales;
    const float * s_row = scales + head_dim;
    const float * zp = scales + head_dim + group_size;
    const size_t prefix = (size_t(group_size)*head_dim*value_bits + 7u)/8u;
    __half * factors = reinterpret_cast<__half *>(record + prefix);
    const size_t factor_f16s = size_t(group_size) + head_dim;
    __half * suffix = factors + size_t(component)*factor_f16s;

    auto residual_at = [&](uint32_t g, uint32_t d) {
        const uint32_t q = kvarn_unpack_one(record, value_bits, size_t(g)*head_dim + d);
        float r = raw[size_t(g)*head_dim + d] - (float(q)*s_row[g] + zp[g])*s_col[d];
        for (uint32_t c = 0; c < component; ++c) {
            const __half * prior = factors + size_t(c)*factor_f16s;
            r -= __half2float(prior[g])*__half2float(prior[group_size + d]);
        }
        return r;
    };

    float local_energy = -1.0f;
    uint32_t local_d = UINT32_MAX;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float energy = 0.0f;
        for (uint32_t g = 0; g < group_size; ++g) {
            const float r = residual_at(g, d);
            energy += r*r;
        }
        if (energy > local_energy || (energy == local_energy && d < local_d)) {
            local_energy = energy;
            local_d = d;
        }
    }
    candidate[threadIdx.x] = local_energy;
    w[threadIdx.x] = __uint_as_float(local_d);
    __syncthreads();
    if (threadIdx.x == 0) {
        float e = -1.0f;
        uint32_t d0 = UINT32_MAX;
        for (uint32_t t = 0; t < blockDim.x; ++t) {
            const uint32_t d = __float_as_uint(w[t]);
            if (candidate[t] > e || (candidate[t] == e && d < d0)) {
                e = candidate[t];
                d0 = d;
            }
        }
        best_d = d0;
        scalar = e;
        degenerate = 0u;
    }
    __syncthreads();
    if (!(scalar > 1.0e-20f)) {
        for (uint32_t i = threadIdx.x; i < group_size + head_dim; i += blockDim.x) suffix[i] = __float2half(0.0f);
        return;
    }
    if (threadIdx.x < group_size) {
        const uint32_t g = threadIdx.x;
        u[g] = residual_at(g, best_d);
    }
    __syncthreads();

    for (uint32_t iter = 0; iter < 4; ++iter) {
        if (threadIdx.x == 0) {
            float u2 = 0.0f;
            for (uint32_t g = 0; g < group_size; ++g) u2 += u[g]*u[g];
            scalar = u2;
            if (!(u2 > 1.0e-20f)) degenerate = 1u;
        }
        __syncthreads();
        if (degenerate) break;
        const float inv_u2 = 1.0f/scalar;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float sum = 0.0f;
            for (uint32_t g = 0; g < group_size; ++g) {
                sum += residual_at(g, d)*u[g];
            }
            w[d] = sum*inv_u2;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float w2 = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) w2 += w[d]*w[d];
            scalar = w2;
            if (!(w2 > 1.0e-20f)) degenerate = 1u;
        }
        __syncthreads();
        if (degenerate) break;
        const float inv_w2 = 1.0f/scalar;
        if (threadIdx.x < group_size) {
            const uint32_t g = threadIdx.x;
            float sum = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                sum += residual_at(g, d)*w[d];
            }
            u[g] = sum*inv_w2;
        }
        __syncthreads();
    }

    if (degenerate) {
        for (uint32_t i = threadIdx.x; i < group_size + head_dim; i += blockDim.x) {
            suffix[i] = __float2half(0.0f);
        }
        return;
    }

    if (threadIdx.x == 0) {
        float u2 = 0.0f;
        for (uint32_t g = 0; g < group_size; ++g) u2 += u[g]*u[g];
        scalar = u2;
        if (!(u2 > 1.0e-20f)) degenerate = 1u;
    }
    __syncthreads();
    if (!degenerate) {
        const float inv_u2 = 1.0f/scalar;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float sum = 0.0f;
            for (uint32_t g = 0; g < group_size; ++g) {
                sum += residual_at(g, d)*u[g];
            }
            w[d] = sum*inv_u2;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float un2 = 0.0f, wn2 = 0.0f, max_abs = -1.0f;
        uint32_t sign_d = 0;
        for (uint32_t g = 0; g < group_size; ++g) un2 += u[g]*u[g];
        for (uint32_t d = 0; d < head_dim; ++d) {
            wn2 += w[d]*w[d];
            const float a = fabsf(w[d]);
            if (a > max_abs) { max_abs = a; sign_d = d; }
        }
        if (!(un2 > 1.0e-20f) || !(wn2 > 1.0e-20f)) {
            degenerate = 1u;
            scalar = 0.0f;
        } else {
            const float un = sqrtf(un2);
            const float wn = sqrtf(wn2);
            const float a = sqrtf(wn/un);
            const float sign = w[sign_d] < 0.0f ? -1.0f : 1.0f;
            for (uint32_t g = 0; g < group_size; ++g) u[g] *= a*sign;
            for (uint32_t d = 0; d < head_dim; ++d) w[d] *= sign/a;
            scalar = 1.0f;
        }
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < group_size + head_dim; i += blockDim.x) {
        float x = degenerate ? 0.0f : (i < group_size ? u[i] : w[i - group_size]);
        suffix[i] = __float2half_rn(x);
    }
}

static __global__ void kvarn_v_sparse_d512_residual_kernel(
        const float * __restrict__ v_raw_rotated,
        uint8_t * __restrict__ v_body,
        const float * __restrict__ v_scales,
        uint32_t n_tiles,
        uint32_t n_heads,
        size_t raw_tile_stride_floats,
        size_t body_record_stride_bytes,
        size_t body_head_stride_bytes,
        size_t scale_record_stride_floats,
        size_t scale_head_stride_floats) {
    constexpr uint32_t head_dim = 512;
    constexpr uint32_t group_size = 128;
    constexpr uint32_t row_keep = 10;
    constexpr uint32_t extra_candidates_per_row = 1;
    constexpr uint32_t extra_keep = 85;
    constexpr uint32_t nnz = group_size*row_keep + extra_keep;
    constexpr size_t prefix_bytes = 16384;
    constexpr size_t col_start_bytes = head_dim*sizeof(uint16_t);
    constexpr size_t row_bytes = nnz;
    constexpr size_t value_offset = prefix_bytes + col_start_bytes + row_bytes + 1;

    const uint32_t tile = blockIdx.x;
    if (tile >= n_tiles || threadIdx.x >= group_size) {
        return;
    }
    const uint32_t record_i = tile/n_heads;
    const uint32_t head_i = tile - record_i*n_heads;
    const float * raw = v_raw_rotated + size_t(tile)*raw_tile_stride_floats;
    uint8_t * record = v_body + size_t(record_i)*body_record_stride_bytes + size_t(head_i)*body_head_stride_bytes;
    const float * scales = v_scales + size_t(record_i)*scale_record_stride_floats + size_t(head_i)*scale_head_stride_floats;
    const float * s_col = scales;
    const float * s_row = scales + head_dim;
    const float * zp = scales + head_dim + group_size;

    __shared__ uint16_t ranked[group_size][row_keep + extra_candidates_per_row];
    __shared__ float ranked_abs[group_size][row_keep + extra_candidates_per_row];
    __shared__ uint8_t keep_extra[group_size];
    __shared__ uint16_t counts[head_dim];
    __shared__ uint16_t cursors[head_dim];

    const uint32_t g = threadIdx.x;
    for (uint32_t k = 0; k < row_keep + extra_candidates_per_row; ++k) {
        ranked[g][k] = UINT16_MAX;
        ranked_abs[g][k] = -1.0f;
    }
    for (uint32_t d = 0; d < head_dim; ++d) {
        const uint32_t q = kvarn_unpack_one(record, 2u, size_t(g)*head_dim + d);
        const float base = (float(q)*s_row[g] + zp[g])*s_col[d];
        const float residual = raw[size_t(g)*head_dim + d] - base;
        const float magnitude = fabsf(residual);
        const float a = isnan(magnitude) ? __int_as_float(0x7f800000) : magnitude;
        uint32_t pos = row_keep + extra_candidates_per_row;
        for (uint32_t k = 0; k < row_keep + extra_candidates_per_row; ++k) {
            if (a > ranked_abs[g][k] || (a == ranked_abs[g][k] && d < ranked[g][k])) {
                pos = k;
                break;
            }
        }
        if (pos < row_keep + extra_candidates_per_row) {
            for (uint32_t k = row_keep + extra_candidates_per_row - 1; k > pos; --k) {
                ranked_abs[g][k] = ranked_abs[g][k - 1];
                ranked[g][k] = ranked[g][k - 1];
            }
            ranked_abs[g][pos] = a;
            ranked[g][pos] = uint16_t(d);
        }
    }
    keep_extra[g] = 0;
    __syncthreads();

    if (threadIdx.x == 0) {
        uint16_t best_g[extra_keep];
        float best_abs[extra_keep];
        for (uint32_t i = 0; i < extra_keep; ++i) {
            best_g[i] = UINT16_MAX;
            best_abs[i] = -1.0f;
        }
        for (uint32_t row = 0; row < group_size; ++row) {
            const float row_abs = ranked_abs[row][row_keep];
            const uint32_t flat = row*head_dim + ranked[row][row_keep];
            uint32_t pos = extra_keep;
            for (uint32_t i = 0; i < extra_keep; ++i) {
                const uint32_t other_flat = best_g[i] == UINT16_MAX ? UINT32_MAX :
                    uint32_t(best_g[i])*head_dim + ranked[best_g[i]][row_keep];
                if (row_abs > best_abs[i] || (row_abs == best_abs[i] && flat < other_flat)) {
                    pos = i;
                    break;
                }
            }
            if (pos < extra_keep) {
                for (uint32_t i = extra_keep - 1; i > pos; --i) {
                    best_abs[i] = best_abs[i - 1];
                    best_g[i] = best_g[i - 1];
                }
                best_abs[pos] = row_abs;
                best_g[pos] = uint16_t(row);
            }
        }
        for (uint32_t i = 0; i < extra_keep; ++i) {
            keep_extra[best_g[i]] = 1;
        }
        for (uint32_t d = 0; d < head_dim; ++d) {
            counts[d] = 0;
        }
        for (uint32_t row = 0; row < group_size; ++row) {
            const uint32_t n = row_keep + keep_extra[row];
            for (uint32_t k = 0; k < n; ++k) {
                ++counts[ranked[row][k]];
            }
        }
        uint16_t running = 0;
        uint16_t * col_start = reinterpret_cast<uint16_t *>(record + prefix_bytes);
        for (uint32_t d = 0; d < head_dim; ++d) {
            col_start[d] = running;
            cursors[d] = running;
            running = uint16_t(running + counts[d]);
        }
        uint8_t * rows = record + prefix_bytes + col_start_bytes;
        record[prefix_bytes + col_start_bytes + row_bytes] = 0;
        __half * values = reinterpret_cast<__half *>(record + value_offset);
        for (uint32_t row = 0; row < group_size; ++row) {
            const uint32_t n = row_keep + keep_extra[row];
            for (uint32_t k = 0; k < n; ++k) {
                const uint32_t d = ranked[row][k];
                const uint32_t at = cursors[d]++;
                rows[at] = uint8_t(row);
                const uint32_t q = kvarn_unpack_one(record, 2u, size_t(row)*head_dim + d);
                const float base = (float(q)*s_row[row] + zp[row])*s_col[d];
                values[at] = __float2half(raw[size_t(row)*head_dim + d] - base);
            }
        }
    }
}

static __global__ void kvarn_quantize_asym_fullrange_pack_rows_kernel(
        const float * __restrict__ src,
        uint8_t * __restrict__ body,
        float * __restrict__ row_scale,
        float * __restrict__ row_zp,
        uint32_t rows,
        uint32_t cols,
        uint32_t bits,
        float clip_sigma) {
    const uint32_t r = blockIdx.x;
    if (r >= rows || threadIdx.x != 0) {
        return;
    }

    const uint32_t qmax = (1u << bits) - 1u;
    const float * row = src + size_t(r)*cols;
    float mn = row[0];
    float mx = row[0];
    double sum = 0.0;
    double ss = 0.0;
    for (uint32_t c = 0; c < cols; ++c) {
        const float v = row[c];
        mn = fminf(mn, v);
        mx = fmaxf(mx, v);
        sum += double(v);
        ss += double(v)*double(v);
    }
    kvarn_rtn_apply_clip(mn, mx, sum, ss, cols, clip_sigma);

    const float s = (mx > mn) ? (mx - mn)/float(qmax) : 1.0f;
    row_scale[r] = s;
    row_zp[r] = mn;

    for (uint32_t c = 0; c < cols; ++c) {
        const float v = fminf(mx, fmaxf(mn, row[c]));
        const uint32_t q = min(qmax, uint32_t(llroundf((v - mn)/s)));
        kvarn_pack_one(body, bits, size_t(r)*cols + c, q);
    }
}

// Dynamic shared memory layout: 2*blockDim floats (min/max) followed by
// 2*blockDim doubles (sum/sumsq) plus 2 result floats; launch with
// kvarn_quantize_fullrange_parallel_shmem_bytes(block).
static __host__ __device__ __forceinline__ size_t kvarn_quantize_fullrange_parallel_shmem_bytes(int block) {
    return size_t(block)*2*sizeof(float) + size_t(block)*2*sizeof(double) + 2*sizeof(float);
}

static __global__ void kvarn_quantize_asym_fullrange_pack_rows_parallel_kernel(
        const float * __restrict__ src,
        uint8_t * __restrict__ body,
        float * __restrict__ row_scale,
        float * __restrict__ row_zp,
        uint32_t rows,
        uint32_t cols,
        uint32_t bits,
        float clip_sigma) {
    const uint32_t r = blockIdx.x;
    if (r >= rows) {
        return;
    }

    const uint32_t qmax = (1u << bits) - 1u;
    const float * row = src + size_t(r)*cols;

    float local_mn = 3.4028234663852886e38f;
    float local_mx = -3.4028234663852886e38f;
    double local_sum = 0.0;
    double local_ss  = 0.0;
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = row[c];
        local_mn = fminf(local_mn, v);
        local_mx = fmaxf(local_mx, v);
        local_sum += double(v);
        local_ss  += double(v)*double(v);
    }

    extern __shared__ float shmem[];
    double * sum_sh = (double *) (shmem + 2*blockDim.x);
    double * ss_sh  = sum_sh + blockDim.x;
    float * out_sh  = (float *) (ss_sh + blockDim.x);
    shmem[threadIdx.x*2]     = local_mn;
    shmem[threadIdx.x*2 + 1] = local_mx;
    sum_sh[threadIdx.x] = local_sum;
    ss_sh[threadIdx.x]  = local_ss;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shmem[threadIdx.x*2]     = fminf(shmem[threadIdx.x*2],     shmem[(threadIdx.x + stride)*2]);
            shmem[threadIdx.x*2 + 1] = fmaxf(shmem[threadIdx.x*2 + 1], shmem[(threadIdx.x + stride)*2 + 1]);
            sum_sh[threadIdx.x] += sum_sh[threadIdx.x + stride];
            ss_sh[threadIdx.x]  += ss_sh[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float mn0 = shmem[0];
        float mx0 = shmem[1];
        kvarn_rtn_apply_clip(mn0, mx0, sum_sh[0], ss_sh[0], cols, clip_sigma);
        out_sh[0] = mn0;
        out_sh[1] = mx0;
    }
    __syncthreads();

    const float mn = out_sh[0];
    const float mx = out_sh[1];
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
        uint32_t cols,
        uint32_t init_scale) {
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
        // First iteration initializes the multiplicative scale, replacing the
        // separate fill launch that previously set it to 1.0f.
        row_scale[r] = init_scale != 0 ? rms : row_scale[r]*rms;
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
        uint32_t cols,
        uint32_t init_scale) {
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
        col_scale[c] = init_scale != 0 ? rms : col_scale[c]*rms;
        shmem[0] = rms;
    }
    __syncthreads();

    const float inv_rms = 1.0f/shmem[0];
    for (uint32_t r = threadIdx.x; r < rows; r += blockDim.x) {
        data[size_t(r)*cols + c] *= inv_rms;
    }
}

static __device__ __forceinline__ float kvarn_logstd_update_scale(float prev, float stdv) {
    stdv = fminf(1.0e3f, fmaxf(1.0e-3f, stdv));
    const float log_prev = logf(fmaxf(prev, 1.0e-20f));
    const float log_next = fminf(10.0f, fmaxf(-0.3f, log_prev + logf(stdv)));
    return expf(log_next);
}

static __global__ void kvarn_sinkhorn_rows_logstd_parallel_kernel(
        float * __restrict__ data,
        float * __restrict__ row_scale,
        uint32_t rows,
        uint32_t cols,
        uint32_t init_scale) {
    const uint32_t r = blockIdx.x;
    if (r >= rows) {
        return;
    }

    float local_sum = 0.0f;
    float local_ss  = 0.0f;
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = data[size_t(r)*cols + c];
        local_sum += v;
        local_ss  += v*v;
    }

    extern __shared__ float shmem[];
    float * sum_sh = shmem;
    float * ss_sh  = shmem + blockDim.x;
    sum_sh[threadIdx.x] = local_sum;
    ss_sh[threadIdx.x]  = local_ss;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sum_sh[threadIdx.x] += sum_sh[threadIdx.x + stride];
            ss_sh[threadIdx.x]  += ss_sh[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float n = float(cols);
        float var = 0.0f;
        if (cols > 1) {
            var = (ss_sh[0] - (sum_sh[0]*sum_sh[0])/n)/float(cols - 1);
        }
        const float stdv = sqrtf(fmaxf(0.0f, var));
        const float prev = init_scale != 0 ? 1.0f : row_scale[r];
        const float next = kvarn_logstd_update_scale(prev, stdv);
        row_scale[r] = next;
        sum_sh[0] = prev/next;
    }
    __syncthreads();

    const float factor = sum_sh[0];
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        data[size_t(r)*cols + c] *= factor;
    }
}

static __global__ void kvarn_sinkhorn_cols_logstd_parallel_kernel(
        float * __restrict__ data,
        float * __restrict__ col_scale,
        uint32_t rows,
        uint32_t cols,
        uint32_t init_scale) {
    const uint32_t c = blockIdx.x;
    if (c >= cols) {
        return;
    }

    float local_sum = 0.0f;
    float local_ss  = 0.0f;
    for (uint32_t r = threadIdx.x; r < rows; r += blockDim.x) {
        const float v = data[size_t(r)*cols + c];
        local_sum += v;
        local_ss  += v*v;
    }

    extern __shared__ float shmem[];
    float * sum_sh = shmem;
    float * ss_sh  = shmem + blockDim.x;
    sum_sh[threadIdx.x] = local_sum;
    ss_sh[threadIdx.x]  = local_ss;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sum_sh[threadIdx.x] += sum_sh[threadIdx.x + stride];
            ss_sh[threadIdx.x]  += ss_sh[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float n = float(rows);
        float var = 0.0f;
        if (rows > 1) {
            var = (ss_sh[0] - (sum_sh[0]*sum_sh[0])/n)/float(rows - 1);
        }
        const float stdv = sqrtf(fmaxf(0.0f, var));
        const float prev = init_scale != 0 ? 1.0f : col_scale[c];
        const float next = kvarn_logstd_update_scale(prev, stdv);
        col_scale[c] = next;
        sum_sh[0] = prev/next;
    }
    __syncthreads();

    const float factor = sum_sh[0];
    for (uint32_t r = threadIdx.x; r < rows; r += blockDim.x) {
        data[size_t(r)*cols + c] *= factor;
    }
}

// Per-tile best-so-far scratch layout: [rows | cols | imbalance | global_rms].
static __host__ __device__ __forceinline__ size_t kvarn_sinkhorn_best_tile_floats(uint32_t rows, uint32_t cols) {
    return size_t(rows) + cols + 2;
}

// Block-parallel best-so-far update. Each thread computes whole column/row
// std values with the same sequential double accumulation as the CPU
// reference, so the selected iteration is unchanged; only the min/max
// reduction and the scale copy are parallel. Shared memory:
// 2*blockDim floats + 4 result floats + 1 flag int.
static __host__ __device__ __forceinline__ size_t kvarn_sinkhorn_best_update_shmem_bytes(int block) {
    return size_t(block)*2*sizeof(float) + 4*sizeof(float) + sizeof(int);
}

static __device__ void kvarn_sinkhorn_logstd_best_update_tile(
        const float * __restrict__ tile_data,
        const float * __restrict__ row_scale,
        const float * __restrict__ col_scale,
        float * __restrict__ best_row_scale,
        float * __restrict__ best_col_scale,
        float * __restrict__ best_imbalance,
        uint32_t rows,
        uint32_t cols) {
    extern __shared__ float shmem[];
    float * min_sh = shmem;
    float * max_sh = shmem + blockDim.x;
    float * stat_sh = shmem + 2*blockDim.x;
    int * flag_sh = (int *) (stat_sh + 4);

    float local_min = 3.4028234663852886e38f;
    float local_max = 0.0f;
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        double sum = 0.0;
        double ss = 0.0;
        for (uint32_t r = 0; r < rows; ++r) {
            const float v = tile_data[size_t(r)*cols + c];
            sum += double(v);
            ss += double(v)*double(v);
        }
        double var = 0.0;
        if (rows > 1) {
            var = (ss - (sum*sum)/double(rows))/double(rows - 1);
        }
        const float stdv = sqrtf(fmaxf(0.0f, float(var)));
        local_min = fminf(local_min, stdv);
        local_max = fmaxf(local_max, stdv);
    }
    min_sh[threadIdx.x] = local_min;
    max_sh[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            min_sh[threadIdx.x] = fminf(min_sh[threadIdx.x], min_sh[threadIdx.x + stride]);
            max_sh[threadIdx.x] = fmaxf(max_sh[threadIdx.x], max_sh[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        stat_sh[0] = min_sh[0];
        stat_sh[1] = max_sh[0];
    }
    __syncthreads();

    local_min = 3.4028234663852886e38f;
    local_max = 0.0f;
    for (uint32_t r = threadIdx.x; r < rows; r += blockDim.x) {
        double sum = 0.0;
        double ss = 0.0;
        for (uint32_t c = 0; c < cols; ++c) {
            const float v = tile_data[size_t(r)*cols + c];
            sum += double(v);
            ss += double(v)*double(v);
        }
        double var = 0.0;
        if (cols > 1) {
            var = (ss - (sum*sum)/double(cols))/double(cols - 1);
        }
        const float stdv = sqrtf(fmaxf(0.0f, float(var)));
        local_min = fminf(local_min, stdv);
        local_max = fmaxf(local_max, stdv);
    }
    min_sh[threadIdx.x] = local_min;
    max_sh[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            min_sh[threadIdx.x] = fminf(min_sh[threadIdx.x], min_sh[threadIdx.x + stride]);
            max_sh[threadIdx.x] = fmaxf(max_sh[threadIdx.x], max_sh[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float col_min = stat_sh[0];
        const float col_max = stat_sh[1];
        const float row_min = min_sh[0];
        const float row_max = max_sh[0];
        const float imbalance =
            col_max/fmaxf(col_min, 1.0e-8f) + row_max/fmaxf(row_min, 1.0e-8f);
        if (imbalance <= best_imbalance[0]) {
            best_imbalance[0] = imbalance;
            flag_sh[0] = 1;
        } else {
            flag_sh[0] = 0;
        }
    }
    __syncthreads();

    if (flag_sh[0] != 0) {
        for (uint32_t r = threadIdx.x; r < rows; r += blockDim.x) {
            best_row_scale[r] = row_scale[r];
        }
        for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
            best_col_scale[c] = col_scale[c];
        }
    }
}

static __global__ void kvarn_sinkhorn_logstd_best_update_kernel(
        const float * __restrict__ data,
        const float * __restrict__ row_scale,
        const float * __restrict__ col_scale,
        float * __restrict__ best_row_scale,
        float * __restrict__ best_col_scale,
        float * __restrict__ best_imbalance,
        uint32_t rows,
        uint32_t cols) {
    if (blockIdx.x != 0) {
        return;
    }
    kvarn_sinkhorn_logstd_best_update_tile(
            data, row_scale, col_scale, best_row_scale, best_col_scale, best_imbalance, rows, cols);
}

static __global__ void kvarn_sinkhorn_logstd_apply_best_kernel(
        float * __restrict__ data,
        const float * __restrict__ row_scale,
        const float * __restrict__ col_scale,
        const float * __restrict__ best_row_scale,
        const float * __restrict__ best_col_scale,
        uint32_t rows,
        uint32_t cols,
        size_t n) {
    const size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < n) {
        const uint32_t r = uint32_t(i / cols);
        const uint32_t c = uint32_t(i - size_t(r)*cols);
        const float numerator = row_scale[r]*col_scale[c];
        const float denominator = fmaxf(best_row_scale[r]*best_col_scale[c], 1.0e-20f);
        data[i] *= numerator/denominator;
    }
}

static __global__ void kvarn_sinkhorn_logstd_copy_best_scales_kernel(
        float * __restrict__ row_scale,
        float * __restrict__ col_scale,
        const float * __restrict__ best_row_scale,
        const float * __restrict__ best_col_scale,
        uint32_t rows,
        uint32_t cols) {
    const size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < rows) {
        row_scale[i] = best_row_scale[i];
    }
    if (i < cols) {
        col_scale[i] = best_col_scale[i];
    }
}

// Global-RMS pre-normalization for the log-std Sinkhorn. The reference recipe
// clamps accumulated log scales to [-0.3, 10]; raw tiles whose global RMS is
// far below 1 pin every row/column scale at the clamp floor, so no variance is
// equalized at all. Dividing the tile by its global RMS first keeps the clamps
// as degenerate-row guards while letting the dual scaling act at any tile
// magnitude. The factor is folded back into the row scales after
// best-iteration selection, so the packed format and dequant are unchanged.
static __global__ void kvarn_tile_rms_normalize_kernel(
        float * __restrict__ data,
        float * __restrict__ g_base,
        uint32_t n_tiles,
        size_t n_per_tile,
        size_t g_stride_floats) {
    const uint32_t tile = blockIdx.x;
    if (tile >= n_tiles) {
        return;
    }
    float * tile_data = data + size_t(tile)*n_per_tile;
    float * g_out = g_base + size_t(tile)*g_stride_floats;

    extern __shared__ float shmem[];
    double * ss_sh = (double *) shmem;

    double local_ss = 0.0;
    for (size_t i = threadIdx.x; i < n_per_tile; i += blockDim.x) {
        const float v = tile_data[i];
        local_ss += double(v)*double(v);
    }
    ss_sh[threadIdx.x] = local_ss;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            ss_sh[threadIdx.x] += ss_sh[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float rms = float(sqrt(ss_sh[0]/double(n_per_tile)));
        const float g = rms > 1.0e-20f ? rms : 1.0f;
        g_out[0] = g;
        ss_sh[0] = double(1.0f/g);
    }
    __syncthreads();

    const float inv = float(ss_sh[0]);
    if (inv != 1.0f) {
        for (size_t i = threadIdx.x; i < n_per_tile; i += blockDim.x) {
            tile_data[i] *= inv;
        }
    }
}

static __global__ void kvarn_sinkhorn_fold_global_scale_kernel(
        float * __restrict__ row_scale_base,
        const float * __restrict__ best_scratch,
        uint32_t n_tiles,
        uint32_t n_heads,
        uint32_t rows,
        uint32_t cols,
        size_t scale_record_stride_floats,
        size_t scale_head_stride_floats) {
    const size_t i_all = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i_all >= size_t(n_tiles)*rows) {
        return;
    }
    const uint32_t tile = uint32_t(i_all/rows);
    const uint32_t r = uint32_t(i_all - size_t(tile)*rows);
    const uint32_t record = tile/n_heads;
    const uint32_t ih = tile - record*n_heads;
    const float g = best_scratch[size_t(tile)*kvarn_sinkhorn_best_tile_floats(rows, cols) +
        size_t(rows) + cols + 1];
    float * row_scale = row_scale_base + size_t(ih)*scale_head_stride_floats +
        size_t(record)*scale_record_stride_floats;
    row_scale[r] *= g;
}

static void kvarn_sinkhorn_variance_normalize_parallel(
        float * data,
        float * row_scale,
        float * col_scale,
        uint32_t rows,
        uint32_t cols,
        uint32_t iters,
        cudaStream_t stream,
        float * best_scratch = nullptr,
        bool allow_global_norm = true) {
    const int block = 128;
    const size_t shmem = size_t(block)*sizeof(float);
    // Scale init (previously two fill launches of 1.0f) is folded into the
    // first row/col iteration via init_scale, trimming 4 launches per seal
    // (K + V pipelines). With iters==0 the fills are still required so the
    // finalize kernels see neutral scales.
    if (iters == 0) {
        kvarn_fill_f32_kernel<<<int((rows + block - 1)/block), block, 0, stream>>>(row_scale, rows, 1.0f);
        kvarn_fill_f32_kernel<<<int((cols + block - 1)/block), block, 0, stream>>>(col_scale, cols, 1.0f);
        return;
    }

    if (kvarn_log_std_sinkhorn_enabled()) {
        if (best_scratch == nullptr) {
            std::fprintf(stderr, "KVarN log-std Sinkhorn requires best-so-far scratch\n");
            std::abort();
        }
        float * best_row_scale = best_scratch;
        float * best_col_scale = best_row_scale + rows;
        float * best_imbalance = best_col_scale + cols;
        float * global_rms     = best_imbalance + 1;
        const bool global_norm = allow_global_norm && kvarn_global_norm_enabled();
        if (global_norm) {
            const int rms_block = 256;
            kvarn_tile_rms_normalize_kernel<<<1, rms_block, size_t(rms_block)*sizeof(double), stream>>>(
                    data, global_rms, 1, size_t(rows)*cols, 0);
        }

        kvarn_fill_f32_kernel<<<int((rows + block - 1)/block), block, 0, stream>>>(row_scale, rows, 1.0f);
        kvarn_fill_f32_kernel<<<int((cols + block - 1)/block), block, 0, stream>>>(col_scale, cols, 1.0f);
        kvarn_fill_f32_kernel<<<int((rows + block - 1)/block), block, 0, stream>>>(best_row_scale, rows, 1.0f);
        kvarn_fill_f32_kernel<<<int((cols + block - 1)/block), block, 0, stream>>>(best_col_scale, cols, 1.0f);
        kvarn_fill_f32_kernel<<<1, 1, 0, stream>>>(best_imbalance, 1, 3.4028234663852886e38f);

        const int best_block = 256;
        const size_t best_shmem = kvarn_sinkhorn_best_update_shmem_bytes(best_block);
        kvarn_sinkhorn_logstd_best_update_kernel<<<1, best_block, best_shmem, stream>>>(
                data, row_scale, col_scale, best_row_scale, best_col_scale, best_imbalance,
                rows, cols);

        const size_t logstd_shmem = size_t(2*block)*sizeof(float);
        for (uint32_t iter = 0; iter < iters; ++iter) {
            kvarn_sinkhorn_cols_logstd_parallel_kernel<<<int(cols), block, logstd_shmem, stream>>>(data, col_scale, rows, cols, 0);
            kvarn_sinkhorn_rows_logstd_parallel_kernel<<<int(rows), block, logstd_shmem, stream>>>(data, row_scale, rows, cols, 0);
            kvarn_sinkhorn_logstd_best_update_kernel<<<1, best_block, best_shmem, stream>>>(
                    data, row_scale, col_scale, best_row_scale, best_col_scale, best_imbalance,
                    rows, cols);
        }
        const size_t n = size_t(rows)*cols;
        kvarn_sinkhorn_logstd_apply_best_kernel<<<int((n + block - 1)/block), block, 0, stream>>>(
                data, row_scale, col_scale, best_row_scale, best_col_scale, rows, cols, n);
        const uint32_t scale_n = rows > cols ? rows : cols;
        kvarn_sinkhorn_logstd_copy_best_scales_kernel<<<int((scale_n + block - 1)/block), block, 0, stream>>>(
                row_scale, col_scale, best_row_scale, best_col_scale, rows, cols);
        if (global_norm) {
            kvarn_sinkhorn_fold_global_scale_kernel<<<int((rows + block - 1)/block), block, 0, stream>>>(
                    row_scale, best_scratch, 1, 1, rows, cols, 0, 0);
        }
        return;
    }

    for (uint32_t iter = 0; iter < iters; ++iter) {
        const uint32_t init = iter == 0 ? 1u : 0u;
        kvarn_sinkhorn_rows_parallel_kernel<<<int(rows), block, shmem, stream>>>(data, row_scale, rows, cols, init);
        kvarn_sinkhorn_cols_parallel_kernel<<<int(cols), block, shmem, stream>>>(data, col_scale, rows, cols, init);
    }
}

static __global__ void kvarn_direct_records_hadamard_k_batched_kernel(
        const float * __restrict__ k_tiles,
        float * __restrict__ k_data,
        uint32_t n_heads,
        uint32_t n_records,
        uint32_t head_dim,
        uint32_t group_size,
        size_t k_tile_head_stride_floats,
        size_t k_tile_group_stride_floats,
        size_t k_tile_record_stride_floats) {
    const uint32_t tile = blockIdx.x / group_size;
    const uint32_t g = blockIdx.x - tile*group_size;
    if (tile >= n_heads*n_records) {
        return;
    }
    const uint32_t record = tile / n_heads;
    const uint32_t ih = tile - record*n_heads;

    extern __shared__ float shmem[];
    float v = 0.0f;
    if (threadIdx.x < head_dim) {
        const float * src = k_tiles + size_t(record)*k_tile_record_stride_floats +
            size_t(ih)*k_tile_head_stride_floats + size_t(g)*k_tile_group_stride_floats;
        v = src[threadIdx.x];
    }
    shmem[threadIdx.x] = v;
    __syncthreads();

    for (uint32_t step = 1; step < head_dim; step <<= 1) {
        const uint32_t pair = threadIdx.x ^ step;
        const float other = shmem[pair];
        __syncthreads();
        if ((threadIdx.x & step) == 0) {
            shmem[threadIdx.x] += other;
        } else {
            shmem[threadIdx.x] = other - shmem[threadIdx.x];
        }
        __syncthreads();
    }

    if (threadIdx.x < head_dim) {
        const float norm = rsqrtf(float(head_dim));
        k_data[size_t(tile)*head_dim*group_size + size_t(threadIdx.x)*group_size + g] =
            shmem[threadIdx.x]*norm;
    }
}

static __global__ void kvarn_direct_records_hadamard_v_batched_kernel(
        const float * __restrict__ v_tiles,
        float * __restrict__ v_data,
        uint32_t n_heads,
        uint32_t n_records,
        uint32_t head_dim,
        uint32_t group_size,
        size_t v_tile_head_stride_floats,
        size_t v_tile_group_stride_floats,
        size_t v_tile_record_stride_floats) {
    const uint32_t tile = blockIdx.x / group_size;
    const uint32_t g = blockIdx.x - tile*group_size;
    if (tile >= n_heads*n_records) {
        return;
    }
    const uint32_t record = tile / n_heads;
    const uint32_t ih = tile - record*n_heads;

    extern __shared__ float shmem[];
    float v = 0.0f;
    if (threadIdx.x < head_dim) {
        const float * src = v_tiles + size_t(record)*v_tile_record_stride_floats +
            size_t(ih)*v_tile_head_stride_floats + size_t(g)*v_tile_group_stride_floats;
        v = src[threadIdx.x];
    }
    shmem[threadIdx.x] = v;
    __syncthreads();

    for (uint32_t step = 1; step < head_dim; step <<= 1) {
        const uint32_t pair = threadIdx.x ^ step;
        const float other = shmem[pair];
        __syncthreads();
        if ((threadIdx.x & step) == 0) {
            shmem[threadIdx.x] += other;
        } else {
            shmem[threadIdx.x] = other - shmem[threadIdx.x];
        }
        __syncthreads();
    }

    if (threadIdx.x < head_dim) {
        const float norm = rsqrtf(float(head_dim));
        v_data[size_t(tile)*head_dim*group_size + size_t(g)*head_dim + threadIdx.x] =
            shmem[threadIdx.x]*norm;
    }
}

static __global__ void kvarn_direct_records_gather_v_batched_kernel(
        const float * __restrict__ v_tiles,
        float * __restrict__ v_data,
        uint32_t n_heads,
        uint32_t n_records,
        uint32_t head_dim,
        uint32_t group_size,
        size_t v_tile_head_stride_floats,
        size_t v_tile_group_stride_floats,
        size_t v_tile_record_stride_floats) {
    const size_t i_all = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const size_t tile_floats = size_t(head_dim)*group_size;
    const size_t total = size_t(n_heads)*n_records*tile_floats;
    if (i_all >= total) {
        return;
    }
    const uint32_t tile = uint32_t(i_all/tile_floats);
    const uint32_t i = uint32_t(i_all - size_t(tile)*tile_floats);
    const uint32_t record = tile/n_heads;
    const uint32_t ih = tile - record*n_heads;
    const uint32_t g = i/head_dim;
    const uint32_t d = i - g*head_dim;
    const float * src = v_tiles + size_t(record)*v_tile_record_stride_floats +
        size_t(ih)*v_tile_head_stride_floats;
    v_data[size_t(tile)*head_dim*group_size + size_t(g)*head_dim + d] =
        src[size_t(g)*v_tile_group_stride_floats + d];
}

static __global__ void kvarn_sinkhorn_rows_batched_kernel(
        float * __restrict__ data,
        float * __restrict__ row_scale_base,
        uint32_t n_heads,
        uint32_t rows,
        uint32_t cols,
        size_t scale_record_stride_floats,
        size_t scale_head_stride_floats,
        uint32_t init_scale) {
    const uint32_t tile = blockIdx.x / rows;
    const uint32_t r = blockIdx.x - tile*rows;
    const uint32_t record = tile / n_heads;
    const uint32_t ih = tile - record*n_heads;
    float * tile_data = data + size_t(tile)*rows*cols;
    float * row_scale = row_scale_base + size_t(ih)*scale_head_stride_floats +
        size_t(record)*scale_record_stride_floats;

    constexpr float eps = 1.0e-6f;
    float local_ss = 0.0f;
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = tile_data[size_t(r)*cols + c];
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
        row_scale[r] = init_scale != 0 ? rms : row_scale[r]*rms;
        shmem[0] = rms;
    }
    __syncthreads();

    const float inv_rms = 1.0f/shmem[0];
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        tile_data[size_t(r)*cols + c] *= inv_rms;
    }
}

static __global__ void kvarn_sinkhorn_cols_batched_kernel(
        float * __restrict__ data,
        float * __restrict__ col_scale_base,
        uint32_t n_heads,
        uint32_t rows,
        uint32_t cols,
        size_t scale_record_stride_floats,
        size_t scale_head_stride_floats,
        uint32_t init_scale) {
    const uint32_t tile = blockIdx.x / cols;
    const uint32_t c = blockIdx.x - tile*cols;
    const uint32_t record = tile / n_heads;
    const uint32_t ih = tile - record*n_heads;
    float * tile_data = data + size_t(tile)*rows*cols;
    float * col_scale = col_scale_base + size_t(ih)*scale_head_stride_floats +
        size_t(record)*scale_record_stride_floats;

    constexpr float eps = 1.0e-6f;
    float local_ss = 0.0f;
    for (uint32_t r = threadIdx.x; r < rows; r += blockDim.x) {
        const float v = tile_data[size_t(r)*cols + c];
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
        col_scale[c] = init_scale != 0 ? rms : col_scale[c]*rms;
        shmem[0] = rms;
    }
    __syncthreads();

    const float inv_rms = 1.0f/shmem[0];
    for (uint32_t r = threadIdx.x; r < rows; r += blockDim.x) {
        tile_data[size_t(r)*cols + c] *= inv_rms;
    }
}

static __global__ void kvarn_sinkhorn_rows_logstd_batched_kernel(
        float * __restrict__ data,
        float * __restrict__ row_scale_base,
        uint32_t n_heads,
        uint32_t rows,
        uint32_t cols,
        size_t scale_record_stride_floats,
        size_t scale_head_stride_floats,
        uint32_t init_scale) {
    const uint32_t tile = blockIdx.x / rows;
    const uint32_t r = blockIdx.x - tile*rows;
    const uint32_t record = tile / n_heads;
    const uint32_t ih = tile - record*n_heads;
    float * tile_data = data + size_t(tile)*rows*cols;
    float * row_scale = row_scale_base + size_t(ih)*scale_head_stride_floats +
        size_t(record)*scale_record_stride_floats;

    float local_sum = 0.0f;
    float local_ss  = 0.0f;
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = tile_data[size_t(r)*cols + c];
        local_sum += v;
        local_ss  += v*v;
    }

    extern __shared__ float shmem[];
    float * sum_sh = shmem;
    float * ss_sh  = shmem + blockDim.x;
    sum_sh[threadIdx.x] = local_sum;
    ss_sh[threadIdx.x]  = local_ss;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sum_sh[threadIdx.x] += sum_sh[threadIdx.x + stride];
            ss_sh[threadIdx.x]  += ss_sh[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float n = float(cols);
        float var = 0.0f;
        if (cols > 1) {
            var = (ss_sh[0] - (sum_sh[0]*sum_sh[0])/n)/float(cols - 1);
        }
        const float stdv = sqrtf(fmaxf(0.0f, var));
        const float prev = init_scale != 0 ? 1.0f : row_scale[r];
        const float next = kvarn_logstd_update_scale(prev, stdv);
        row_scale[r] = next;
        sum_sh[0] = prev/next;
    }
    __syncthreads();

    const float factor = sum_sh[0];
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        tile_data[size_t(r)*cols + c] *= factor;
    }
}

static __global__ void kvarn_sinkhorn_cols_logstd_batched_kernel(
        float * __restrict__ data,
        float * __restrict__ col_scale_base,
        uint32_t n_heads,
        uint32_t rows,
        uint32_t cols,
        size_t scale_record_stride_floats,
        size_t scale_head_stride_floats,
        uint32_t init_scale) {
    const uint32_t tile = blockIdx.x / cols;
    const uint32_t c = blockIdx.x - tile*cols;
    const uint32_t record = tile / n_heads;
    const uint32_t ih = tile - record*n_heads;
    float * tile_data = data + size_t(tile)*rows*cols;
    float * col_scale = col_scale_base + size_t(ih)*scale_head_stride_floats +
        size_t(record)*scale_record_stride_floats;

    float local_sum = 0.0f;
    float local_ss  = 0.0f;
    for (uint32_t r = threadIdx.x; r < rows; r += blockDim.x) {
        const float v = tile_data[size_t(r)*cols + c];
        local_sum += v;
        local_ss  += v*v;
    }

    extern __shared__ float shmem[];
    float * sum_sh = shmem;
    float * ss_sh  = shmem + blockDim.x;
    sum_sh[threadIdx.x] = local_sum;
    ss_sh[threadIdx.x]  = local_ss;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sum_sh[threadIdx.x] += sum_sh[threadIdx.x + stride];
            ss_sh[threadIdx.x]  += ss_sh[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float n = float(rows);
        float var = 0.0f;
        if (rows > 1) {
            var = (ss_sh[0] - (sum_sh[0]*sum_sh[0])/n)/float(rows - 1);
        }
        const float stdv = sqrtf(fmaxf(0.0f, var));
        const float prev = init_scale != 0 ? 1.0f : col_scale[c];
        const float next = kvarn_logstd_update_scale(prev, stdv);
        col_scale[c] = next;
        sum_sh[0] = prev/next;
    }
    __syncthreads();

    const float factor = sum_sh[0];
    for (uint32_t r = threadIdx.x; r < rows; r += blockDim.x) {
        tile_data[size_t(r)*cols + c] *= factor;
    }
}

// Exact batched companion to kvarn_sinkhorn_logstd_best_update_kernel.
// One block owns one record/head tile; per-column/per-row double accumulation
// order matches the CPU reference so the selected best iteration is unchanged.
static __global__ void kvarn_sinkhorn_init_batched_kernel(
        float * __restrict__ row_scale_base,
        float * __restrict__ col_scale_base,
        float * __restrict__ best_scratch,
        uint32_t n_tiles,
        uint32_t n_heads,
        uint32_t rows,
        uint32_t cols,
        size_t scale_record_stride_floats,
        size_t scale_head_stride_floats) {
    const uint32_t tile = blockIdx.x;
    if (tile >= n_tiles) {
        return;
    }

    const uint32_t record = tile/n_heads;
    const uint32_t ih = tile - record*n_heads;
    float * row_scale = row_scale_base + size_t(ih)*scale_head_stride_floats +
        size_t(record)*scale_record_stride_floats;
    float * col_scale = col_scale_base + size_t(ih)*scale_head_stride_floats +
        size_t(record)*scale_record_stride_floats;
    float * best_row = best_scratch + size_t(tile)*kvarn_sinkhorn_best_tile_floats(rows, cols);
    float * best_col = best_row + rows;
    float * best_imbalance = best_col + cols;

    for (uint32_t r = threadIdx.x; r < rows; r += blockDim.x) {
        row_scale[r] = 1.0f;
        best_row[r] = 1.0f;
    }
    for (uint32_t c = threadIdx.x; c < cols; c += blockDim.x) {
        col_scale[c] = 1.0f;
        best_col[c] = 1.0f;
    }
    if (threadIdx.x == 0) {
        best_imbalance[0] = 3.4028234663852886e38f;
    }
}

static __global__ void kvarn_sinkhorn_logstd_best_update_batched_kernel(
        const float * __restrict__ data,
        const float * __restrict__ row_scale_base,
        const float * __restrict__ col_scale_base,
        float * __restrict__ best_scratch,
        uint32_t n_tiles,
        uint32_t n_heads,
        uint32_t rows,
        uint32_t cols,
        size_t scale_record_stride_floats,
        size_t scale_head_stride_floats) {
    const uint32_t tile = blockIdx.x;
    if (tile >= n_tiles) {
        return;
    }

    const uint32_t record = tile/n_heads;
    const uint32_t ih = tile - record*n_heads;
    const float * tile_data = data + size_t(tile)*rows*cols;
    const float * row_scale = row_scale_base + size_t(ih)*scale_head_stride_floats +
        size_t(record)*scale_record_stride_floats;
    const float * col_scale = col_scale_base + size_t(ih)*scale_head_stride_floats +
        size_t(record)*scale_record_stride_floats;
    float * best_row = best_scratch + size_t(tile)*kvarn_sinkhorn_best_tile_floats(rows, cols);
    float * best_col = best_row + rows;
    float * best_imbalance = best_col + cols;

    kvarn_sinkhorn_logstd_best_update_tile(
            tile_data, row_scale, col_scale, best_row, best_col, best_imbalance, rows, cols);
}

static __global__ void kvarn_sinkhorn_logstd_apply_best_batched_kernel(
        float * __restrict__ data,
        const float * __restrict__ row_scale_base,
        const float * __restrict__ col_scale_base,
        const float * __restrict__ best_scratch,
        uint32_t n_tiles,
        uint32_t n_heads,
        uint32_t rows,
        uint32_t cols,
        size_t scale_record_stride_floats,
        size_t scale_head_stride_floats) {
    const size_t n_per_tile = size_t(rows)*cols;
    const size_t i_all = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i_all >= size_t(n_tiles)*n_per_tile) {
        return;
    }

    const uint32_t tile = uint32_t(i_all/n_per_tile);
    const size_t i = i_all - size_t(tile)*n_per_tile;
    const uint32_t r = uint32_t(i/cols);
    const uint32_t c = uint32_t(i - size_t(r)*cols);
    const uint32_t record = tile/n_heads;
    const uint32_t ih = tile - record*n_heads;
    const float * row_scale = row_scale_base + size_t(ih)*scale_head_stride_floats +
        size_t(record)*scale_record_stride_floats;
    const float * col_scale = col_scale_base + size_t(ih)*scale_head_stride_floats +
        size_t(record)*scale_record_stride_floats;
    const float * best_row = best_scratch + size_t(tile)*kvarn_sinkhorn_best_tile_floats(rows, cols);
    const float * best_col = best_row + rows;
    const float numerator = row_scale[r]*col_scale[c];
    const float denominator = fmaxf(best_row[r]*best_col[c], 1.0e-20f);
    data[size_t(tile)*n_per_tile + i] *= numerator/denominator;
}

static __global__ void kvarn_sinkhorn_logstd_copy_best_batched_kernel(
        float * __restrict__ row_scale_base,
        float * __restrict__ col_scale_base,
        const float * __restrict__ best_scratch,
        uint32_t n_tiles,
        uint32_t n_heads,
        uint32_t rows,
        uint32_t cols,
        size_t scale_record_stride_floats,
        size_t scale_head_stride_floats) {
    const uint32_t width = rows > cols ? rows : cols;
    const size_t i_all = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i_all >= size_t(n_tiles)*width) {
        return;
    }

    const uint32_t tile = uint32_t(i_all/width);
    const uint32_t i = uint32_t(i_all - size_t(tile)*width);
    const uint32_t record = tile/n_heads;
    const uint32_t ih = tile - record*n_heads;
    float * row_scale = row_scale_base + size_t(ih)*scale_head_stride_floats +
        size_t(record)*scale_record_stride_floats;
    float * col_scale = col_scale_base + size_t(ih)*scale_head_stride_floats +
        size_t(record)*scale_record_stride_floats;
    const float * best_row = best_scratch + size_t(tile)*kvarn_sinkhorn_best_tile_floats(rows, cols);
    const float * best_col = best_row + rows;
    if (i < rows) {
        row_scale[i] = best_row[i];
    }
    if (i < cols) {
        col_scale[i] = best_col[i];
    }
}

static void kvarn_sinkhorn_variance_normalize_batched(
        float * data,
        float * row_scale_base,
        float * col_scale_base,
        uint32_t n_tiles,
        uint32_t n_heads,
        uint32_t rows,
        uint32_t cols,
        uint32_t iters,
        size_t scale_record_stride_floats,
        size_t scale_head_stride_floats,
        float * best_scratch,
        cudaStream_t stream,
        bool allow_global_norm = true) {
    const int block = 128;
    const size_t shmem = size_t(block)*sizeof(float);

    if (best_scratch == nullptr) {
        std::fprintf(stderr, "KVarN batched Sinkhorn requires scale-state scratch\n");
        std::abort();
    }
    kvarn_sinkhorn_init_batched_kernel<<<int(n_tiles), block, 0, stream>>>(
            row_scale_base, col_scale_base, best_scratch,
            n_tiles, n_heads, rows, cols,
            scale_record_stride_floats, scale_head_stride_floats);

    if (iters == 0) {
        return;
    }

    if (kvarn_log_std_sinkhorn_enabled()) {
        const bool global_norm = allow_global_norm && kvarn_global_norm_enabled();
        if (global_norm) {
            const int rms_block = 256;
            kvarn_tile_rms_normalize_kernel<<<int(n_tiles), rms_block, size_t(rms_block)*sizeof(double), stream>>>(
                    data, best_scratch + size_t(rows) + cols + 1, n_tiles, size_t(rows)*cols,
                    kvarn_sinkhorn_best_tile_floats(rows, cols));
        }

        const int best_block = 256;
        const size_t best_shmem = kvarn_sinkhorn_best_update_shmem_bytes(best_block);
        kvarn_sinkhorn_logstd_best_update_batched_kernel<<<int(n_tiles), best_block, best_shmem, stream>>>(
                data, row_scale_base, col_scale_base, best_scratch,
                n_tiles, n_heads, rows, cols,
                scale_record_stride_floats, scale_head_stride_floats);

        const size_t logstd_shmem = size_t(2*block)*sizeof(float);
        for (uint32_t iter = 0; iter < iters; ++iter) {
            kvarn_sinkhorn_cols_logstd_batched_kernel<<<int(n_tiles*cols), block, logstd_shmem, stream>>>(
                    data, col_scale_base, n_heads, rows, cols,
                    scale_record_stride_floats, scale_head_stride_floats, 0);
            kvarn_sinkhorn_rows_logstd_batched_kernel<<<int(n_tiles*rows), block, logstd_shmem, stream>>>(
                    data, row_scale_base, n_heads, rows, cols,
                    scale_record_stride_floats, scale_head_stride_floats, 0);
            kvarn_sinkhorn_logstd_best_update_batched_kernel<<<int(n_tiles), best_block, best_shmem, stream>>>(
                    data, row_scale_base, col_scale_base, best_scratch,
                    n_tiles, n_heads, rows, cols,
                    scale_record_stride_floats, scale_head_stride_floats);
        }

        const size_t n = size_t(n_tiles)*rows*cols;
        kvarn_sinkhorn_logstd_apply_best_batched_kernel<<<int((n + block - 1)/block), block, 0, stream>>>(
                data, row_scale_base, col_scale_base, best_scratch,
                n_tiles, n_heads, rows, cols,
                scale_record_stride_floats, scale_head_stride_floats);
        const size_t scale_n = size_t(n_tiles)*(rows > cols ? rows : cols);
        kvarn_sinkhorn_logstd_copy_best_batched_kernel<<<int((scale_n + block - 1)/block), block, 0, stream>>>(
                row_scale_base, col_scale_base, best_scratch,
                n_tiles, n_heads, rows, cols,
                scale_record_stride_floats, scale_head_stride_floats);
        if (global_norm) {
            const size_t fold_n = size_t(n_tiles)*rows;
            kvarn_sinkhorn_fold_global_scale_kernel<<<int((fold_n + block - 1)/block), block, 0, stream>>>(
                    row_scale_base, best_scratch,
                    n_tiles, n_heads, rows, cols,
                    scale_record_stride_floats, scale_head_stride_floats);
        }
        return;
    }

    for (uint32_t iter = 0; iter < iters; ++iter) {
        const uint32_t init = 0;
        kvarn_sinkhorn_rows_batched_kernel<<<int(n_tiles*rows), block, shmem, stream>>>(
                data, row_scale_base, n_heads, rows, cols,
                scale_record_stride_floats, scale_head_stride_floats, init);
        kvarn_sinkhorn_cols_batched_kernel<<<int(n_tiles*cols), block, shmem, stream>>>(
                data, col_scale_base, n_heads, rows, cols,
                scale_record_stride_floats, scale_head_stride_floats, init);
    }
}

static __global__ void kvarn_quantize_k_fullrange_batched_kernel(
        const float * __restrict__ data,
        uint8_t * __restrict__ k_body,
        float * __restrict__ k_scales,
        uint32_t n_heads,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t key_bits,
        float clip_sigma,
        size_t k_body_record_stride_bytes,
        size_t k_body_head_stride_bytes,
        size_t k_scale_record_stride_floats,
        size_t k_scale_head_stride_floats) {
    const uint32_t tile = blockIdx.x / head_dim;
    const uint32_t d = blockIdx.x - tile*head_dim;
    const uint32_t record = tile / n_heads;
    const uint32_t ih = tile - record*n_heads;
    const float * row = data + size_t(tile)*head_dim*group_size + size_t(d)*group_size;
    uint8_t * body = k_body + size_t(ih)*k_body_head_stride_bytes + size_t(record)*k_body_record_stride_bytes;
    float * scales = k_scales + size_t(ih)*k_scale_head_stride_floats + size_t(record)*k_scale_record_stride_floats;

    extern __shared__ float shmem[];
    float * mins = shmem;
    float * maxs = shmem + blockDim.x;
    double * sum_sh = (double *) (shmem + 2*blockDim.x);
    double * ss_sh  = sum_sh + blockDim.x;
    float * out_sh  = (float *) (ss_sh + blockDim.x);
    float mn = 3.4028234663852886e38f;
    float mx = -3.4028234663852886e38f;
    double sum = 0.0;
    double ss = 0.0;
    for (uint32_t c = threadIdx.x; c < group_size; c += blockDim.x) {
        const float v = row[c];
        mn = fminf(mn, v);
        mx = fmaxf(mx, v);
        sum += double(v);
        ss += double(v)*double(v);
    }
    mins[threadIdx.x] = mn;
    maxs[threadIdx.x] = mx;
    sum_sh[threadIdx.x] = sum;
    ss_sh[threadIdx.x] = ss;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            mins[threadIdx.x] = fminf(mins[threadIdx.x], mins[threadIdx.x + stride]);
            maxs[threadIdx.x] = fmaxf(maxs[threadIdx.x], maxs[threadIdx.x + stride]);
            sum_sh[threadIdx.x] += sum_sh[threadIdx.x + stride];
            ss_sh[threadIdx.x]  += ss_sh[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float mn0 = mins[0];
        float mx0 = maxs[0];
        kvarn_rtn_apply_clip(mn0, mx0, sum_sh[0], ss_sh[0], group_size, clip_sigma);
        out_sh[0] = mn0;
        out_sh[1] = mx0;
    }
    __syncthreads();

    const uint32_t qmax = (1u << key_bits) - 1u;
    const float row_mn = out_sh[0];
    const float row_mx = out_sh[1];
    const float s = (row_mx > row_mn) ? (row_mx - row_mn)/float(qmax) : 1.0f;
    const float varn_row = scales[d];
    if (threadIdx.x == 0) {
        scales[d] = varn_row*s;
        scales[head_dim + d] = varn_row*row_mn;
    }

    const size_t row_byte0 = (size_t(d)*group_size*key_bits) >> 3;
    if (key_bits == 2) {
        for (size_t b = threadIdx.x; b < (size_t(group_size)*2u >> 3); b += blockDim.x) {
            uint32_t packed = 0;
            const uint32_t c_base = uint32_t(4*b);
            for (uint32_t j = 0; j < 4; ++j) {
                const uint32_t c = c_base + j;
                if (c < group_size) {
                    const float v = fminf(row_mx, fmaxf(row_mn, row[c]));
                    const uint32_t q = min(qmax, uint32_t(llroundf((v - row_mn)/s)));
                    packed |= q << (2*j);
                }
            }
            body[row_byte0 + b] = uint8_t(packed);
        }
    } else if (key_bits == 4) {
        for (size_t b = threadIdx.x; b < (size_t(group_size)*4u >> 3); b += blockDim.x) {
            const uint32_t c0 = uint32_t(2*b);
            const uint32_t c1 = c0 + 1;
            const float v0 = fminf(row_mx, fmaxf(row_mn, row[c0]));
            const uint32_t q0 = min(qmax, uint32_t(llroundf((v0 - row_mn)/s)));
            uint32_t q1 = 0;
            if (c1 < group_size) {
                const float v1 = fminf(row_mx, fmaxf(row_mn, row[c1]));
                q1 = min(qmax, uint32_t(llroundf((v1 - row_mn)/s)));
            }
            body[row_byte0 + b] = uint8_t(q0 | (q1 << 4));
        }
    } else if (key_bits == 8) {
        for (uint32_t c = threadIdx.x; c < group_size; c += blockDim.x) {
            const float v = fminf(row_mx, fmaxf(row_mn, row[c]));
            const uint32_t q = min(qmax, uint32_t(llroundf((v - row_mn)/s)));
            body[row_byte0 + c] = uint8_t(q);
        }
    }
}

static __global__ void kvarn_quantize_v_fullrange_batched_kernel(
        const float * __restrict__ data,
        uint8_t * __restrict__ v_body,
        float * __restrict__ v_scales,
        uint32_t n_heads,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t value_bits,
        float clip_sigma,
        size_t v_body_record_stride_bytes,
        size_t v_body_head_stride_bytes,
        size_t v_scale_record_stride_floats,
        size_t v_scale_head_stride_floats) {
    const uint32_t tile = blockIdx.x / group_size;
    const uint32_t g = blockIdx.x - tile*group_size;
    const uint32_t record = tile / n_heads;
    const uint32_t ih = tile - record*n_heads;
    const float * row = data + size_t(tile)*head_dim*group_size + size_t(g)*head_dim;
    uint8_t * body = v_body + size_t(ih)*v_body_head_stride_bytes + size_t(record)*v_body_record_stride_bytes;
    float * scales = v_scales + size_t(ih)*v_scale_head_stride_floats + size_t(record)*v_scale_record_stride_floats;

    extern __shared__ float shmem[];
    float * mins = shmem;
    float * maxs = shmem + blockDim.x;
    double * sum_sh = (double *) (shmem + 2*blockDim.x);
    double * ss_sh  = sum_sh + blockDim.x;
    float * out_sh  = (float *) (ss_sh + blockDim.x);
    float mn = 3.4028234663852886e38f;
    float mx = -3.4028234663852886e38f;
    double sum = 0.0;
    double ss = 0.0;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float v = row[d];
        mn = fminf(mn, v);
        mx = fmaxf(mx, v);
        sum += double(v);
        ss += double(v)*double(v);
    }
    mins[threadIdx.x] = mn;
    maxs[threadIdx.x] = mx;
    sum_sh[threadIdx.x] = sum;
    ss_sh[threadIdx.x] = ss;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            mins[threadIdx.x] = fminf(mins[threadIdx.x], mins[threadIdx.x + stride]);
            maxs[threadIdx.x] = fmaxf(maxs[threadIdx.x], maxs[threadIdx.x + stride]);
            sum_sh[threadIdx.x] += sum_sh[threadIdx.x + stride];
            ss_sh[threadIdx.x]  += ss_sh[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float mn0 = mins[0];
        float mx0 = maxs[0];
        kvarn_rtn_apply_clip(mn0, mx0, sum_sh[0], ss_sh[0], head_dim, clip_sigma);
        out_sh[0] = mn0;
        out_sh[1] = mx0;
    }
    __syncthreads();

    const uint32_t qmax = (1u << value_bits) - 1u;
    const float row_mn = out_sh[0];
    const float row_mx = out_sh[1];
    const float s = (row_mx > row_mn) ? (row_mx - row_mn)/float(qmax) : 1.0f;
    float * v_row_scale = scales + head_dim;
    float * v_zp = scales + head_dim + group_size;
    const float varn_row = v_row_scale[g];
    if (threadIdx.x == 0) {
        v_row_scale[g] = varn_row*s;
        v_zp[g] = varn_row*row_mn;
    }

    const size_t row_byte0 = (size_t(g)*head_dim*value_bits) >> 3;
    if (value_bits == 2) {
        for (size_t b = threadIdx.x; b < (size_t(head_dim)*2u >> 3); b += blockDim.x) {
            uint32_t packed = 0;
            const uint32_t d_base = uint32_t(4*b);
            for (uint32_t j = 0; j < 4; ++j) {
                const uint32_t d = d_base + j;
                if (d < head_dim) {
                    const float v = fminf(row_mx, fmaxf(row_mn, row[d]));
                    const uint32_t q = min(qmax, uint32_t(llroundf((v - row_mn)/s)));
                    packed |= q << (2*j);
                }
            }
            body[row_byte0 + b] = uint8_t(packed);
        }
    } else if (value_bits == 4) {
        for (size_t b = threadIdx.x; b < (size_t(head_dim)*4u >> 3); b += blockDim.x) {
            const uint32_t d0 = uint32_t(2*b);
            const uint32_t d1 = d0 + 1;
            const float v0 = fminf(row_mx, fmaxf(row_mn, row[d0]));
            const uint32_t q0 = min(qmax, uint32_t(llroundf((v0 - row_mn)/s)));
            uint32_t q1 = 0;
            if (d1 < head_dim) {
                const float v1 = fminf(row_mx, fmaxf(row_mn, row[d1]));
                q1 = min(qmax, uint32_t(llroundf((v1 - row_mn)/s)));
            }
            body[row_byte0 + b] = uint8_t(q0 | (q1 << 4));
        }
    } else if (value_bits == 8) {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            const float v = fminf(row_mx, fmaxf(row_mn, row[d]));
            const uint32_t q = min(qmax, uint32_t(llroundf((v - row_mn)/s)));
            body[row_byte0 + d] = uint8_t(q);
        }
    }
}

static __global__ void kvarn_raw_body_store_k_token_major_kernel(
        const float * __restrict__ k_frame_channel_major,
        float * __restrict__ k_mirror_token_major,
        uint32_t head_dim,
        uint32_t group_size) {
    const size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const size_t n = size_t(head_dim)*group_size;
    if (i >= n) {
        return;
    }

    const uint32_t g = uint32_t(i/head_dim);
    const uint32_t d = uint32_t(i - size_t(g)*head_dim);
    k_mirror_token_major[i] = k_frame_channel_major[size_t(d)*group_size + g];
}

static void kvarn_raw_body_mirror_ensure_unlocked(
        const void * key,
        uint32_t n_records_cap,
        uint32_t n_heads,
        uint32_t head_dim,
        uint32_t group_size) {
    if (key == nullptr || n_records_cap == 0 || n_heads == 0 || head_dim == 0 || group_size == 0) {
        return;
    }

    kvarn_raw_body_mirror_entry & e = g_kvarn_raw_body_mirrors[key];
    if (e.k != nullptr && e.v != nullptr &&
            e.n_records_cap >= n_records_cap &&
            e.n_heads == n_heads &&
            e.head_dim == head_dim &&
            e.group_size == group_size) {
        return;
    }

    if (e.k != nullptr) {
        kvarn_cuda_free_checked(e.k, "resized raw mirror K");
    }
    if (e.v != nullptr) {
        kvarn_cuda_free_checked(e.v, "resized raw mirror V");
    }

    e = {};
    e.n_records_cap = n_records_cap;
    e.n_heads = n_heads;
    e.head_dim = head_dim;
    e.group_size = group_size;

    const size_t n = size_t(n_records_cap)*n_heads*head_dim*group_size;
    cudaError_t err = cudaMalloc(&e.k, n*sizeof(float));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "KVarN raw body mirror: cudaMalloc(K) failed: %s\n", cudaGetErrorString(err));
        std::abort();
    }
    err = cudaMalloc(&e.v, n*sizeof(float));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "KVarN raw body mirror: cudaMalloc(V) failed: %s\n", cudaGetErrorString(err));
        std::abort();
    }
    err = cudaMemset(e.k, 0, n*sizeof(float));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "KVarN raw body mirror: cudaMemset(K) failed: %s\n", cudaGetErrorString(err));
        std::abort();
    }
    err = cudaMemset(e.v, 0, n*sizeof(float));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "KVarN raw body mirror: cudaMemset(V) failed: %s\n", cudaGetErrorString(err));
        std::abort();
    }
}

static __global__ void kvarn_turbo_quantize_v_rows_kernel(
        const float * __restrict__ data,
        uint8_t * __restrict__ v_body,
        float * __restrict__ v_scales,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t value_bits,
        uint32_t canonical_layout) {
    const uint32_t g = blockIdx.x;
    const uint32_t block128 = blockIdx.y;
    const uint32_t j = threadIdx.x;
    if (g >= group_size || j >= 128) {
        return;
    }

    __shared__ float x[128];
    __shared__ uint8_t q[128];
    __shared__ float norm;
    __shared__ float recon_norm;

    const uint32_t d = block128*128u + j;
    const float v = data[size_t(g)*head_dim + d];
    x[j] = v;
    __syncthreads();

    if (j == 0) {
        double ss = 0.0;
        for (uint32_t i = 0; i < 128; ++i) {
            ss += double(x[i])*double(x[i]);
        }
        norm = sqrtf(float(ss));
    }
    __syncthreads();

    x[j] = norm > 1.0e-10f ? x[j]/norm : 0.0f;
    x[j] *= KVARN_TURBO_WHT_SIGNS1_128[j];
    __syncthreads();

    for (uint32_t step = 1; step < 128; step <<= 1) {
        if ((j & step) == 0) {
            const uint32_t j1 = j + step;
            const float a = x[j];
            const float b = x[j1];
            x[j]  = a + b;
            x[j1] = a - b;
        }
        __syncthreads();
    }

    x[j] *= 0.08838834764831845f*KVARN_TURBO_WHT_SIGNS2_128[j];
    q[j] = uint8_t(kvarn_turbo_nearest_centroid(x[j], value_bits));
    __syncthreads();

    if (value_bits == 2) {
        if ((j & 3u) == 0) {
            const uint8_t packed =
                uint8_t((q[j + 0] & 0x3u) |
                        ((q[j + 1] & 0x3u) << 2) |
                        ((q[j + 2] & 0x3u) << 4) |
                        ((q[j + 3] & 0x3u) << 6));
            const size_t byte_pos = canonical_layout != 0 ?
                kvarn_turbo_v_block_offset(head_dim, value_bits, g, block128) + 2u + (j >> 2) :
                (((size_t(g)*head_dim + d)*2u) >> 3);
            v_body[byte_pos] = packed;
        }
    } else {
        if ((j & 1u) == 0) {
            const uint8_t packed = uint8_t((q[j] & 0x0fu) | ((q[j + 1] & 0x0fu) << 4));
            const size_t byte_pos = canonical_layout != 0 ?
                kvarn_turbo_v_block_offset(head_dim, value_bits, g, block128) + 4u + (j >> 1) :
                (((size_t(g)*head_dim + d)*4u) >> 3);
            v_body[byte_pos] = packed;
        }
    }

    x[j] = kvarn_turbo_centroid(value_bits, q[j]);
    __syncthreads();

    if (j == 0) {
        double ss = 0.0;
        for (uint32_t i = 0; i < 128; ++i) {
            ss += double(x[i])*double(x[i]);
        }
        recon_norm = sqrtf(float(ss));
        const float corrected_norm = recon_norm > 1.0e-10f ? norm/recon_norm : norm;
        if (canonical_layout != 0) {
            const size_t block_off = kvarn_turbo_v_block_offset(head_dim, value_bits, g, block128);
            *reinterpret_cast<__half *>(v_body + block_off) = __float2half(corrected_norm);
            if (value_bits == 4) {
                *reinterpret_cast<__half *>(v_body + block_off + 2u) = __float2half(0.0f);
            }
        } else {
            const uint32_t blocks_per_row = head_dim/128u;
            v_scales[size_t(g)*blocks_per_row + block128] = corrected_norm;
        }
    }
}

static void kvarn_raw_body_mirror_store(
        const void * key,
        uint32_t record,
        uint32_t head,
        uint32_t n_records_cap,
        uint32_t n_heads,
        uint32_t head_dim,
        uint32_t group_size,
        const float * k_frame_channel_major,
        const float * v_frame_token_major,
        cudaStream_t stream) {
    if (!kvarn_debug_raw_body_capture_enabled() || key == nullptr ||
            record == UINT32_MAX || head == UINT32_MAX || record >= n_records_cap || head >= n_heads) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_kvarn_registry_mutex);
    kvarn_raw_body_mirror_ensure_unlocked(key, n_records_cap, n_heads, head_dim, group_size);
    kvarn_raw_body_mirror_entry & e = g_kvarn_raw_body_mirrors[key];
    if (e.k == nullptr || e.v == nullptr) {
        return;
    }

    const size_t tile_floats = size_t(head_dim)*group_size;
    const size_t tile = (size_t(head)*e.n_records_cap + record)*tile_floats;
    float * k_dst = e.k + tile;
    float * v_dst = e.v + tile;

    const int block = 256;
    const int grid = int((tile_floats + block - 1)/block);
    kvarn_raw_body_store_k_token_major_kernel<<<grid, block, 0, stream>>>(
            k_frame_channel_major, k_dst, head_dim, group_size);
    cudaMemcpyAsync(v_dst, v_frame_token_major, tile_floats*sizeof(float), cudaMemcpyDeviceToDevice, stream);
    if (kvarn_store_phase_trace_claim()) {
        std::fprintf(stderr,
                "KVarN raw body mirror store: key=%p record=%u head=%u records_cap=%u n_heads=%u head_dim=%u group_size=%u\n",
                key, record, head, n_records_cap, n_heads, head_dim, group_size);
    }
}

static bool kvarn_raw_body_mirror_find_unlocked(
        const void * key,
        uint32_t n_records,
        uint32_t n_heads,
        uint32_t head_dim,
        uint32_t group_size,
        kvarn_raw_body_mirror_entry & result) {
    auto it = g_kvarn_raw_body_mirrors.find(key);
    if (it == g_kvarn_raw_body_mirrors.end()) {
        return false;
    }
    const kvarn_raw_body_mirror_entry & e = it->second;
    if (e.k == nullptr || e.v == nullptr ||
            e.n_records_cap < n_records ||
            e.n_heads != n_heads ||
            e.head_dim != head_dim ||
            e.group_size != group_size) {
        return false;
    }
    result = e;
    return true;
}

static bool kvarn_raw_body_mirror_find(
        const void * key,
        uint32_t n_records,
        uint32_t n_heads,
        uint32_t head_dim,
        uint32_t group_size,
        kvarn_raw_body_mirror_entry & result) {
    std::lock_guard<std::mutex> lock(g_kvarn_registry_mutex);
    return kvarn_raw_body_mirror_find_unlocked(
            key, n_records, n_heads, head_dim, group_size, result);
}

static bool kvarn_raw_body_mirror_find_compatible(
        const void * raw_key,
        const void * k_body,
        uint32_t n_records,
        uint32_t n_heads,
        uint32_t head_dim,
        uint32_t group_size,
        kvarn_raw_body_mirror_entry & result) {
    std::lock_guard<std::mutex> lock(g_kvarn_registry_mutex);
    bool found = kvarn_raw_body_mirror_find_unlocked(
            raw_key, n_records, n_heads, head_dim, group_size, result);
    if (!found && raw_key != k_body) {
        found = kvarn_raw_body_mirror_find_unlocked(
                k_body, n_records, n_heads, head_dim, group_size, result);
    }
    return found;
}

static size_t kvarn_raw_body_mirror_count() {
    std::lock_guard<std::mutex> lock(g_kvarn_registry_mutex);
    return g_kvarn_raw_body_mirrors.size();
}

static bool ggml_cuda_kvarn_store_body_direct_records_batched_fullrange(
        const float * k_tiles,
        const float * v_tiles,
        uint8_t * k_body,
        uint8_t * v_body,
        float * k_scales,
        float * v_scales,
        float * scratch,
        uint32_t n_heads,
        uint32_t n_records,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t key_bits,
        uint32_t value_bits,
        uint32_t sinkhorn_iters,
        float rtn_quantile,
        size_t scratch_floats,
        size_t k_body_record_stride_bytes,
        size_t v_body_record_stride_bytes,
        size_t k_body_head_stride_bytes,
        size_t v_body_head_stride_bytes,
        size_t k_scale_record_stride_floats,
        size_t v_scale_record_stride_floats,
        size_t k_scale_head_stride_floats,
        size_t v_scale_head_stride_floats,
        size_t k_tile_head_stride_floats,
        size_t v_tile_head_stride_floats,
        size_t k_tile_group_stride_floats,
        size_t v_tile_group_stride_floats,
        size_t k_tile_record_stride_floats,
        size_t v_tile_record_stride_floats,
        uint32_t turbo_v_mode,
        cudaStream_t stream) {
    if (turbo_v_mode == 1u || turbo_v_mode == 2u) {
        return false;
    }

    const uint32_t total_tiles = n_heads*n_records;
    const size_t tile_floats = size_t(head_dim)*group_size;
    const size_t data_floats = size_t(total_tiles)*tile_floats;
    const size_t best_floats = size_t(total_tiles)*kvarn_sinkhorn_best_tile_floats(head_dim, group_size);
    const bool use_residual = turbo_v_mode == 3u || turbo_v_mode == 4u || turbo_v_mode == 5u || turbo_v_mode == 6u;
    const size_t required = (use_residual ? 3 : 2)*data_floats + 2*best_floats;
    if (total_tiles == 0 || scratch_floats < required || rtn_quantile < 1.0f ||
            (key_bits != 2 && key_bits != 4 && key_bits != 8) ||
            (value_bits != 2 && value_bits != 4 && value_bits != 8) ||
            (head_dim & (head_dim - 1)) != 0 || head_dim > 1024) {
        return false;
    }

    float * k_data = scratch;
    float * v_data = k_data + data_floats;
    float * v_raw = use_residual ? v_data + data_floats : nullptr;
    float * k_best = v_data + data_floats + (use_residual ? data_floats : 0);
    float * v_best = k_best + best_floats;
    const int hadamard_block = kvarn_pow2_block(head_dim);
    if (hadamard_block <= 0 || hadamard_block > 1024) {
        return false;
    }

    kvarn_aux_cuda_state & aux = kvarn_aux_cuda_state_get();
    cudaStream_t aux_stream = aux.stream;
    cudaEventRecord(aux.main_ready, stream);
    cudaStreamWaitEvent(aux_stream, aux.main_ready, 0);

    if (!kvarn_paper_frame_enabled()) {
        return false;
    }

    kvarn_direct_records_hadamard_k_batched_kernel<<<int(total_tiles*group_size), hadamard_block,
            size_t(hadamard_block)*sizeof(float), stream>>>(
            k_tiles, k_data, n_heads, n_records, head_dim, group_size,
            k_tile_head_stride_floats, k_tile_group_stride_floats, k_tile_record_stride_floats);
    if (kvarn_paper_mixed_frame_enabled()) {
        const int gather_block = 256;
        const size_t total_values = size_t(total_tiles)*tile_floats;
        kvarn_direct_records_gather_v_batched_kernel<<<int((total_values + gather_block - 1)/gather_block), gather_block,
                0, aux_stream>>>(
                v_tiles, v_data, n_heads, n_records, head_dim, group_size,
                v_tile_head_stride_floats, v_tile_group_stride_floats, v_tile_record_stride_floats);
    } else {
        kvarn_direct_records_hadamard_v_batched_kernel<<<int(total_tiles*group_size), hadamard_block,
                size_t(hadamard_block)*sizeof(float), aux_stream>>>(
                v_tiles, v_data, n_heads, n_records, head_dim, group_size,
                v_tile_head_stride_floats, v_tile_group_stride_floats, v_tile_record_stride_floats);
    }
    if (use_residual) {
        cudaMemcpyAsync(v_raw, v_data, data_floats*sizeof(float), cudaMemcpyDeviceToDevice, aux_stream);
    }
    kvarn_sinkhorn_variance_normalize_batched(
            k_data, k_scales, k_scales + 2*head_dim,
            total_tiles, n_heads, head_dim, group_size, sinkhorn_iters,
            k_scale_record_stride_floats, k_scale_head_stride_floats, k_best, stream,
            turbo_v_mode != 5u && turbo_v_mode != 6u);
    kvarn_sinkhorn_variance_normalize_batched(
            v_data, v_scales + head_dim, v_scales,
            total_tiles, n_heads, group_size, head_dim, sinkhorn_iters,
            v_scale_record_stride_floats, v_scale_head_stride_floats, v_best, aux_stream,
            turbo_v_mode != 5u && turbo_v_mode != 6u);

    const int k_quant_block = kvarn_pow2_block(group_size);
    const int v_quant_block = kvarn_pow2_block(head_dim);
    if (k_quant_block <= 0 || k_quant_block > 1024 || v_quant_block <= 0 || v_quant_block > 1024) {
        return false;
    }
    kvarn_quantize_k_fullrange_batched_kernel<<<int(total_tiles*head_dim), k_quant_block,
            kvarn_quantize_fullrange_parallel_shmem_bytes(k_quant_block), stream>>>(
            k_data, k_body, k_scales, n_heads, head_dim, group_size, key_bits,
            kvarn_rtn_clip_sigma_host(key_bits),
            k_body_record_stride_bytes, k_body_head_stride_bytes,
            k_scale_record_stride_floats, k_scale_head_stride_floats);
    kvarn_quantize_v_fullrange_batched_kernel<<<int(total_tiles*group_size), v_quant_block,
            kvarn_quantize_fullrange_parallel_shmem_bytes(v_quant_block), aux_stream>>>(
            v_data, v_body, v_scales, n_heads, head_dim, group_size, value_bits,
            kvarn_rtn_clip_sigma_host(value_bits),
            v_body_record_stride_bytes, v_body_head_stride_bytes,
            v_scale_record_stride_floats, v_scale_head_stride_floats);
    if (turbo_v_mode == 6u) {
        kvarn_v_sparse_d512_residual_kernel<<<int(total_tiles), 128, 0, aux_stream>>>(
                v_raw, v_body, v_scales, total_tiles, n_heads, tile_floats,
                v_body_record_stride_bytes, v_body_head_stride_bytes,
                v_scale_record_stride_floats, v_scale_head_stride_floats);
    } else if (use_residual) {
        const size_t r1_shmem = size_t(group_size + head_dim + 128u)*sizeof(float);
        kvarn_v_r1_residual_kernel<<<int(total_tiles), 128, r1_shmem, aux_stream>>>(
                v_raw, v_body, v_scales, total_tiles, head_dim, group_size, value_bits, n_heads,
                tile_floats, v_body_record_stride_bytes, v_body_head_stride_bytes,
                v_scale_record_stride_floats, v_scale_head_stride_floats, 0u);
        if (turbo_v_mode == 4u || turbo_v_mode == 5u) {
            kvarn_v_r1_residual_kernel<<<int(total_tiles), 128, r1_shmem, aux_stream>>>(
                    v_raw, v_body, v_scales, total_tiles, head_dim, group_size, value_bits, n_heads,
                    tile_floats, v_body_record_stride_bytes, v_body_head_stride_bytes,
                    v_scale_record_stride_floats, v_scale_head_stride_floats, 1u);
        }
        if (turbo_v_mode == 5u) {
            kvarn_v_r1_residual_kernel<<<int(total_tiles), 128, r1_shmem, aux_stream>>>(
                    v_raw, v_body, v_scales, total_tiles, head_dim, group_size, value_bits, n_heads,
                    tile_floats, v_body_record_stride_bytes, v_body_head_stride_bytes,
                    v_scale_record_stride_floats, v_scale_head_stride_floats, 2u);
        }
    }
    cudaEventRecord(aux.aux_done, aux_stream);
    cudaStreamWaitEvent(stream, aux.aux_done, 0);
    return true;
}

static bool kvarn_fullrange_packer_overwrites_body(uint32_t cols, uint32_t bits) {
    if (bits != 2 && bits != 4 && bits != 8) {
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
        bool input_already_rotated,
        void * stream) {
    const size_t n = size_t(head_dim)*group_size;
    const uint32_t tmp_rows = head_dim > group_size ? head_dim : group_size;
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    float * data      = scratch;
    float * rtn_scale = scratch + n;
    float * rtn_zp    = scratch + n + tmp_rows;
    float * best_scratch = scratch + n + 2*tmp_rows;

    if (rtn_quantile < 1.0f || !kvarn_fullrange_packer_overwrites_body(group_size, key_bits)) {
        cudaMemsetAsync(k_body, 0, kvarn_packed_nbytes(n, key_bits), cuda_stream);
    }
    if (!input_already_rotated && kvarn_paper_frame_enabled()) {
        const int k_hadamard_block = kvarn_pow2_block(head_dim);
        if (k_hadamard_block <= 1024) {
            kvarn_hadamard_cols_parallel_kernel<<<int(group_size), k_hadamard_block, size_t(k_hadamard_block)*sizeof(float), cuda_stream>>>(
                    k_tile, data, head_dim, group_size);
        } else {
            kvarn_hadamard_cols_kernel<<<int(group_size), 1, 0, cuda_stream>>>(k_tile, data, head_dim, group_size);
        }
    } else {
        cudaMemcpyAsync(data, k_tile, n*sizeof(float), cudaMemcpyDeviceToDevice, cuda_stream);
    }
    float * k_row_scale = k_scales;
    float * k_col_scale = k_scales + 2*head_dim;
    kvarn_sinkhorn_variance_normalize_parallel(
            data, k_row_scale, k_col_scale, head_dim, group_size, sinkhorn_iters, cuda_stream, best_scratch);
    if (rtn_quantile >= 1.0f) {
        const int k_quant_block = kvarn_pow2_block(group_size);
        if (k_quant_block <= 1024) {
            kvarn_quantize_asym_fullrange_pack_rows_parallel_kernel<<<int(head_dim), k_quant_block, kvarn_quantize_fullrange_parallel_shmem_bytes(k_quant_block), cuda_stream>>>(
                    data, k_body, rtn_scale, rtn_zp, head_dim, group_size, key_bits, kvarn_rtn_clip_sigma_host(key_bits));
        } else {
            kvarn_quantize_asym_fullrange_pack_rows_kernel<<<int(head_dim), 1, 0, cuda_stream>>>(
                    data, k_body, rtn_scale, rtn_zp, head_dim, group_size, key_bits, kvarn_rtn_clip_sigma_host(key_bits));
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
        uint32_t turbo_v_mode,
        bool input_already_rotated,
        void * stream) {
    kvarn_validate_v_mode(head_dim, group_size, value_bits, turbo_v_mode);
    const size_t n = size_t(head_dim)*group_size;
    const uint32_t tmp_rows = head_dim > group_size ? head_dim : group_size;
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    float * data      = scratch;
    float * rtn_scale = scratch + n;
    float * rtn_zp    = scratch + n + tmp_rows;
    float * best_scratch = scratch + n + 2*tmp_rows;
    float * v_raw = scratch + n + 2*tmp_rows + kvarn_sinkhorn_best_tile_floats(head_dim, group_size);

    if (rtn_quantile < 1.0f || !kvarn_fullrange_packer_overwrites_body(head_dim, value_bits)) {
        cudaMemsetAsync(v_body, 0, kvarn_packed_nbytes(n, value_bits), cuda_stream);
    }
    if (!input_already_rotated && kvarn_paper_frame_enabled() && !kvarn_paper_mixed_frame_enabled()) {
        const int v_hadamard_block = kvarn_pow2_block(head_dim);
        if (v_hadamard_block <= 1024) {
            kvarn_hadamard_rows_parallel_kernel<<<int(group_size), v_hadamard_block, size_t(v_hadamard_block)*sizeof(float), cuda_stream>>>(
                    v_tile, data, group_size, head_dim);
        } else {
            kvarn_hadamard_rows_kernel<<<int(group_size), 1, 0, cuda_stream>>>(v_tile, data, group_size, head_dim);
        }
    } else {
        cudaMemcpyAsync(data, v_tile, n*sizeof(float), cudaMemcpyDeviceToDevice, cuda_stream);
    }
    if (turbo_v_mode == 3u || turbo_v_mode == 4u || turbo_v_mode == 5u || turbo_v_mode == 6u) {
        cudaMemcpyAsync(v_raw, data, n*sizeof(float), cudaMemcpyDeviceToDevice, cuda_stream);
    }
    if (turbo_v_mode == 1u || turbo_v_mode == 2u) {
        cudaMemsetAsync(v_scales, 0, (size_t(head_dim) + 2u*group_size)*sizeof(float), cuda_stream);
        const dim3 grid(group_size, head_dim/128u);
        const uint32_t canonical_layout = turbo_v_mode == 2u ? 1u : 0u;
        kvarn_turbo_quantize_v_rows_kernel<<<grid, 128, 0, cuda_stream>>>(
                data, v_body, v_scales, head_dim, group_size, value_bits, canonical_layout);
        return;
    }

    float * v_col_scale = v_scales;
    float * v_row_scale = v_scales + head_dim;
    kvarn_sinkhorn_variance_normalize_parallel(
            data, v_row_scale, v_col_scale, group_size, head_dim, sinkhorn_iters, cuda_stream, best_scratch,
            turbo_v_mode != 5u && turbo_v_mode != 6u);
    if (rtn_quantile >= 1.0f) {
        const int v_quant_block = kvarn_pow2_block(head_dim);
        if (v_quant_block <= 1024) {
            kvarn_quantize_asym_fullrange_pack_rows_parallel_kernel<<<int(group_size), v_quant_block, kvarn_quantize_fullrange_parallel_shmem_bytes(v_quant_block), cuda_stream>>>(
                    data, v_body, rtn_scale, rtn_zp, group_size, head_dim, value_bits, kvarn_rtn_clip_sigma_host(value_bits));
        } else {
            kvarn_quantize_asym_fullrange_pack_rows_kernel<<<int(group_size), 1, 0, cuda_stream>>>(
                    data, v_body, rtn_scale, rtn_zp, group_size, head_dim, value_bits, kvarn_rtn_clip_sigma_host(value_bits));
        }
    } else {
        kvarn_quantize_asym_minmax_pack_rows_kernel<<<int(group_size), 1, 0, cuda_stream>>>(
                data, v_body, rtn_scale, rtn_zp, group_size, head_dim, value_bits, rtn_quantile);
    }
    const int block = 128;
    kvarn_store_v_finalize_scales_kernel<<<int((group_size + block - 1)/block), block, 0, cuda_stream>>>(
            v_scales, rtn_scale, rtn_zp, head_dim, group_size);
    if (turbo_v_mode == 6u) {
        kvarn_v_sparse_d512_residual_kernel<<<1, 128, 0, cuda_stream>>>(
                v_raw, v_body, v_scales, 1, 1, n,
                kvarn_packed_nbytes(n, value_bits), 0,
                size_t(head_dim + 2*group_size), 0);
    } else if (turbo_v_mode == 3u || turbo_v_mode == 4u || turbo_v_mode == 5u) {
        const size_t r1_shmem = size_t(group_size + head_dim + 128u)*sizeof(float);
        kvarn_v_r1_residual_kernel<<<1, 128, r1_shmem, cuda_stream>>>(
                v_raw, v_body, v_scales, 1, head_dim, group_size, value_bits, 1,
                n, kvarn_packed_nbytes(n, value_bits), 0,
                size_t(head_dim + 2*group_size), 0, 0u);
        if (turbo_v_mode == 4u || turbo_v_mode == 5u) {
            kvarn_v_r1_residual_kernel<<<1, 128, r1_shmem, cuda_stream>>>(
                    v_raw, v_body, v_scales, 1, head_dim, group_size, value_bits, 1,
                    n, kvarn_packed_nbytes(n, value_bits), 0,
                    size_t(head_dim + 2*group_size), 0, 1u);
        }
        if (turbo_v_mode == 5u) {
            kvarn_v_r1_residual_kernel<<<1, 128, r1_shmem, cuda_stream>>>(
                    v_raw, v_body, v_scales, 1, head_dim, group_size, value_bits, 1,
                    n, kvarn_packed_nbytes(n, value_bits), 0,
                    size_t(head_dim + 2*group_size), 0, 2u);
        }
    }
}

static size_t kvarn_store_scratch_floats_one(uint32_t head_dim, uint32_t group_size) {
    const uint32_t tmp_rows = head_dim > group_size ? head_dim : group_size;
    return size_t(head_dim)*group_size + 2*tmp_rows + kvarn_sinkhorn_best_tile_floats(head_dim, group_size);
}

static void ggml_cuda_kvarn_store_kv_body_pipelined(
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
        bool input_already_rotated,
        uint32_t debug_record,
        uint32_t debug_head,
        uint32_t turbo_v_mode,
        void * stream) {
    const size_t n = size_t(head_dim)*group_size;
    const uint32_t tmp_rows = head_dim > group_size ? head_dim : group_size;
    const size_t per_pipeline = kvarn_store_scratch_floats_one(head_dim, group_size);
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    float * k_data      = scratch;
    float * k_rtn_scale = scratch + n;
    float * k_rtn_zp    = scratch + n + tmp_rows;
    float * k_best      = scratch + n + 2*tmp_rows;

    float * v_data      = scratch + per_pipeline;
    float * v_rtn_scale = v_data + n;
    float * v_rtn_zp    = v_data + n + tmp_rows;
    float * v_best      = v_data + n + 2*tmp_rows;
    float * v_raw       = scratch + 2*per_pipeline;

    if (rtn_quantile < 1.0f || !kvarn_fullrange_packer_overwrites_body(group_size, key_bits)) {
        cudaMemsetAsync(k_body, 0, kvarn_packed_nbytes(n, key_bits), cuda_stream);
    }
    if ((turbo_v_mode == 0 || turbo_v_mode == 3 || turbo_v_mode == 4 || turbo_v_mode == 5 || turbo_v_mode == 6) &&
            (rtn_quantile < 1.0f || !kvarn_fullrange_packer_overwrites_body(head_dim, value_bits))) {
        cudaMemsetAsync(v_body, 0, kvarn_packed_nbytes(n, value_bits), cuda_stream);
    }

    kvarn_aux_cuda_state & aux = kvarn_aux_cuda_state_get();
    cudaStream_t aux_stream = aux.stream;

    // k_tile/v_tile are produced by prior kernels on cuda_stream. The aux stream
    // must not read them until those writes are visible.
    cudaEventRecord(aux.main_ready, cuda_stream);
    cudaStreamWaitEvent(aux_stream, aux.main_ready, 0);

    if (input_already_rotated || !kvarn_paper_frame_enabled()) {
        cudaMemcpyAsync(k_data, k_tile, n*sizeof(float), cudaMemcpyDeviceToDevice, cuda_stream);
        cudaMemcpyAsync(v_data, v_tile, n*sizeof(float), cudaMemcpyDeviceToDevice, aux_stream);
    } else {
        const int k_hadamard_block = kvarn_pow2_block(head_dim);
        if (k_hadamard_block <= 1024) {
            kvarn_hadamard_cols_parallel_kernel<<<int(group_size), k_hadamard_block, size_t(k_hadamard_block)*sizeof(float), cuda_stream>>>(
                    k_tile, k_data, head_dim, group_size);
        } else {
            kvarn_hadamard_cols_kernel<<<int(group_size), 1, 0, cuda_stream>>>(k_tile, k_data, head_dim, group_size);
        }

        if (kvarn_paper_mixed_frame_enabled()) {
            cudaMemcpyAsync(v_data, v_tile, n*sizeof(float), cudaMemcpyDeviceToDevice, aux_stream);
        } else {
            const int v_hadamard_block = kvarn_pow2_block(head_dim);
            if (v_hadamard_block <= 1024) {
                kvarn_hadamard_rows_parallel_kernel<<<int(group_size), v_hadamard_block, size_t(v_hadamard_block)*sizeof(float), aux_stream>>>(
                        v_tile, v_data, group_size, head_dim);
            } else {
                kvarn_hadamard_rows_kernel<<<int(group_size), 1, 0, aux_stream>>>(v_tile, v_data, group_size, head_dim);
            }
        }
    }

    if (turbo_v_mode == 3u || turbo_v_mode == 4u || turbo_v_mode == 5u || turbo_v_mode == 6u) {
        cudaMemcpyAsync(v_raw, v_data, n*sizeof(float), cudaMemcpyDeviceToDevice, aux_stream);
    }

    if (kvarn_debug_raw_body_capture_enabled()) {
        cudaEventRecord(aux.aux_done, aux_stream);
        cudaStreamWaitEvent(cuda_stream, aux.aux_done, 0);
        const uint32_t raw_record =
            g_kvarn_debug_store_context.src_layout == 1 && debug_record != UINT32_MAX ?
                g_kvarn_debug_store_context.record0 + debug_record : debug_record;
        const void * raw_key = g_kvarn_debug_store_context.raw_mirror_key;
        kvarn_raw_body_mirror_store(
                raw_key,
                raw_record,
                debug_head,
                g_kvarn_debug_store_context.records_cap,
                g_kvarn_debug_store_context.n_heads,
                head_dim, group_size, k_data, v_data, cuda_stream);
        cudaEventRecord(aux.main_ready, cuda_stream);
        cudaStreamWaitEvent(aux_stream, aux.main_ready, 0);
    }

    const kvarn_body_record_dump_state dump = kvarn_body_record_dump_claim(debug_record, debug_head);
    if (dump.active) {
        cudaError_t err = cudaStreamSynchronize(cuda_stream);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "KVarN body-record dump: cudaStreamSynchronize(main) failed: %s\n", cudaGetErrorString(err));
        }
        err = cudaStreamSynchronize(aux_stream);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "KVarN body-record dump: cudaStreamSynchronize(aux) failed: %s\n", cudaGetErrorString(err));
        }
        kvarn_dump_device_buffer(kvarn_join_path(dump.dir, "k_tile_input.bin"), k_tile, n*sizeof(float));
        kvarn_dump_device_buffer(kvarn_join_path(dump.dir, "v_tile_input.bin"), v_tile, n*sizeof(float));
        kvarn_dump_device_buffer(kvarn_join_path(dump.dir, "k_rot_or_copy.bin"), k_data, n*sizeof(float));
        kvarn_dump_device_buffer(kvarn_join_path(dump.dir, "v_rot_or_copy.bin"), v_data, n*sizeof(float));
    }

    float * k_row_scale = k_scales;
    float * k_col_scale = k_scales + 2*head_dim;
    float * v_col_scale = v_scales;
    float * v_row_scale = v_scales + head_dim;

    kvarn_sinkhorn_variance_normalize_parallel(
            k_data, k_row_scale, k_col_scale, head_dim, group_size, sinkhorn_iters, cuda_stream, k_best,
            turbo_v_mode != 5u && turbo_v_mode != 6u);

    if (rtn_quantile >= 1.0f) {
        const int k_quant_block = kvarn_pow2_block(group_size);
        if (k_quant_block <= 1024) {
            kvarn_quantize_asym_fullrange_pack_rows_parallel_kernel<<<int(head_dim), k_quant_block, kvarn_quantize_fullrange_parallel_shmem_bytes(k_quant_block), cuda_stream>>>(
                    k_data, k_body, k_rtn_scale, k_rtn_zp, head_dim, group_size, key_bits, kvarn_rtn_clip_sigma_host(key_bits));
        } else {
            kvarn_quantize_asym_fullrange_pack_rows_kernel<<<int(head_dim), 1, 0, cuda_stream>>>(
                    k_data, k_body, k_rtn_scale, k_rtn_zp, head_dim, group_size, key_bits, kvarn_rtn_clip_sigma_host(key_bits));
        }
    } else {
        kvarn_quantize_asym_minmax_pack_rows_kernel<<<int(head_dim), 1, 0, cuda_stream>>>(
                k_data, k_body, k_rtn_scale, k_rtn_zp, head_dim, group_size, key_bits, rtn_quantile);
    }

    const int block = 128;
    kvarn_store_k_finalize_scales_kernel<<<int((head_dim + block - 1)/block), block, 0, cuda_stream>>>(
            k_scales, k_rtn_scale, k_rtn_zp, head_dim);

    if (turbo_v_mode == 1u || turbo_v_mode == 2u) {
        cudaMemsetAsync(v_scales, 0, (size_t(head_dim) + 2u*group_size)*sizeof(float), aux_stream);
        const dim3 grid_turbo(group_size, head_dim/128u);
        const uint32_t canonical_layout = turbo_v_mode == 2u ? 1u : 0u;
        kvarn_turbo_quantize_v_rows_kernel<<<grid_turbo, 128, 0, aux_stream>>>(
                v_data, v_body, v_scales, head_dim, group_size, value_bits, canonical_layout);
    } else {
        kvarn_sinkhorn_variance_normalize_parallel(
                v_data, v_row_scale, v_col_scale, group_size, head_dim, sinkhorn_iters, aux_stream, v_best,
                turbo_v_mode != 5u && turbo_v_mode != 6u);
        if (rtn_quantile >= 1.0f) {
            const int v_quant_block = kvarn_pow2_block(head_dim);
            if (v_quant_block <= 1024) {
                kvarn_quantize_asym_fullrange_pack_rows_parallel_kernel<<<int(group_size), v_quant_block, kvarn_quantize_fullrange_parallel_shmem_bytes(v_quant_block), aux_stream>>>(
                        v_data, v_body, v_rtn_scale, v_rtn_zp, group_size, head_dim, value_bits, kvarn_rtn_clip_sigma_host(value_bits));
            } else {
                kvarn_quantize_asym_fullrange_pack_rows_kernel<<<int(group_size), 1, 0, aux_stream>>>(
                        v_data, v_body, v_rtn_scale, v_rtn_zp, group_size, head_dim, value_bits, kvarn_rtn_clip_sigma_host(value_bits));
            }
        } else {
            kvarn_quantize_asym_minmax_pack_rows_kernel<<<int(group_size), 1, 0, aux_stream>>>(
                    v_data, v_body, v_rtn_scale, v_rtn_zp, group_size, head_dim, value_bits, rtn_quantile);
        }
        kvarn_store_v_finalize_scales_kernel<<<int((group_size + block - 1)/block), block, 0, aux_stream>>>(
                v_scales, v_rtn_scale, v_rtn_zp, head_dim, group_size);
        if (turbo_v_mode == 6u) {
            kvarn_v_sparse_d512_residual_kernel<<<1, 128, 0, aux_stream>>>(
                    v_raw, v_body, v_scales, 1, 1, n,
                    kvarn_packed_nbytes(n, value_bits), 0,
                    size_t(head_dim + 2*group_size), 0);
        } else if (turbo_v_mode == 3u || turbo_v_mode == 4u || turbo_v_mode == 5u) {
            const size_t r1_shmem = size_t(group_size + head_dim + 128u)*sizeof(float);
            kvarn_v_r1_residual_kernel<<<1, 128, r1_shmem, aux_stream>>>(
                    v_raw, v_body, v_scales, 1, head_dim, group_size, value_bits, 1,
                    n, kvarn_packed_nbytes(n, value_bits), 0,
                    size_t(head_dim + 2*group_size), 0, 0u);
            if (turbo_v_mode == 4u || turbo_v_mode == 5u) {
                kvarn_v_r1_residual_kernel<<<1, 128, r1_shmem, aux_stream>>>(
                        v_raw, v_body, v_scales, 1, head_dim, group_size, value_bits, 1,
                        n, kvarn_packed_nbytes(n, value_bits), 0,
                        size_t(head_dim + 2*group_size), 0, 1u);
            }
            if (turbo_v_mode == 5u) {
                kvarn_v_r1_residual_kernel<<<1, 128, r1_shmem, aux_stream>>>(
                        v_raw, v_body, v_scales, 1, head_dim, group_size, value_bits, 1,
                        n, kvarn_packed_nbytes(n, value_bits), 0,
                        size_t(head_dim + 2*group_size), 0, 2u);
            }
        }
    }

    cudaEventRecord(aux.aux_done, aux_stream);
    cudaStreamWaitEvent(cuda_stream, aux.aux_done, 0);

    if (dump.active) {
        const cudaError_t err = cudaStreamSynchronize(cuda_stream);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "KVarN body-record dump: cudaStreamSynchronize(final) failed: %s\n", cudaGetErrorString(err));
        }
        kvarn_dump_device_buffer(kvarn_join_path(dump.dir, "k_normalized.bin"), k_data, n*sizeof(float));
        kvarn_dump_device_buffer(kvarn_join_path(dump.dir, "v_normalized.bin"), v_data, n*sizeof(float));
        kvarn_dump_device_buffer(kvarn_join_path(dump.dir, "k_body.bin"), k_body, kvarn_packed_nbytes(n, key_bits));
        kvarn_dump_device_buffer(kvarn_join_path(dump.dir, "v_body.bin"), v_body, kvarn_packed_nbytes(n, value_bits));
        kvarn_dump_device_buffer(kvarn_join_path(dump.dir, "scales_k.bin"), k_scales, size_t(2*head_dim + group_size)*sizeof(float));
        kvarn_dump_device_buffer(kvarn_join_path(dump.dir, "scales_v.bin"), v_scales, size_t(head_dim + 2*group_size)*sizeof(float));

        std::ofstream json(kvarn_join_path(dump.dir, "body_record.json"));
        json << "{\n";
        json << "  \"version\": 1,\n";
        json << "  \"mode\": \"kvarn-body-record-store\",\n";
        json << "  \"call_index\": " << dump.call_index << ",\n";
        json << "  \"layer\": " << dump.layer << ",\n";
        json << "  \"record\": " << dump.record << ",\n";
        json << "  \"head\": " << dump.head << ",\n";
        json << "  \"record0\": " << dump.record0 << ",\n";
        json << "  \"n_records_context\": " << dump.n_records << ",\n";
        json << "  \"n_heads_context\": " << dump.n_heads << ",\n";
        json << "  \"src_layout\": " << dump.src_layout << ",\n";
        json << "  \"head_dim\": " << head_dim << ",\n";
        json << "  \"group_size\": " << group_size << ",\n";
        json << "  \"key_bits\": " << key_bits << ",\n";
        json << "  \"value_bits\": " << value_bits << ",\n";
        json << "  \"sinkhorn_iters\": " << sinkhorn_iters << ",\n";
        json << "  \"rtn_quantile\": " << std::setprecision(9) << rtn_quantile << ",\n";
        json << "  \"input_already_rotated\": " << (input_already_rotated ? "true" : "false") << ",\n";
        json << "  \"paper_frame\": " << (kvarn_paper_frame_enabled() ? "true" : "false") << ",\n";
        json << "  \"paper_mixed_frame\": " << (kvarn_paper_mixed_frame_enabled() ? "true" : "false") << ",\n";
        json << "  \"log_std_sinkhorn\": " << (kvarn_log_std_sinkhorn_enabled() ? "true" : "false") << ",\n";
        json << "  \"k_layout\": \"channel-major [head_dim, group_size]\",\n";
        json << "  \"v_layout\": \"token-major [group_size, head_dim]\"\n";
        json << "}\n";
        json.close();

        std::fprintf(stderr,
                "KVarN body-record dump: call=%" PRIu64 " layer=%d head=%u record=%u path=%s\n",
                dump.call_index, dump.layer, dump.head, dump.record, dump.dir.c_str());
    }
}

static __global__ void kvarn_transpose_pending_k_head_kernel(
        const float * __restrict__ pending,
        float * __restrict__ k_tile,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t pending_head_stride) {
    // Pending K is stored token-major per head as pending[d + g*stride].
    // K body-store consumes channel-major tiles, k_tile[d*group_size + g].
    const size_t i = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const size_t n = size_t(head_dim)*group_size;
    if (i >= n) {
        return;
    }

    const uint32_t d = uint32_t(i/group_size);
    const uint32_t g = uint32_t(i - size_t(d)*group_size);
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
        uint32_t turbo_v_mode,
        bool src_in_frame,
        void * stream) {
    kvarn_validate_v_mode(head_dim, group_size, value_bits, turbo_v_mode);
    if (n_record_batch != 1) {
        throw std::invalid_argument(
                "KVarN pending record batch store requires exactly one body record; "
                "the pending buffer has no per-record source dimension");
    }

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

    // Pending tiles inherit the graph paper frame; raw tiles routed through
    // these ops (e.g. direct all-heads stores) must still be rotated here.
    const bool pending_already_rotated = src_in_frame && kvarn_paper_frame_enabled() && !kvarn_paper_mixed_frame_enabled();
    for (uint32_t bi = 0; bi < n_record_batch; ++bi) {
        const uint32_t record = uint32_t(records[bi]);
        if (head_dim >= 256) {
            ggml_cuda_kvarn_store_kv_body_pipelined(
                    k_tile, v_tile,
                    k_body + size_t(record)*k_body_record_stride_bytes,
                    v_body + size_t(record)*v_body_record_stride_bytes,
                    k_scales + size_t(record)*k_scale_record_stride_floats,
                    v_scales + size_t(record)*v_scale_record_stride_floats,
                    pipeline,
                    head_dim, group_size, key_bits, value_bits, sinkhorn_iters, rtn_quantile,
                    pending_already_rotated, record, 0, turbo_v_mode, stream);
        } else {
            ggml_cuda_kvarn_store_k_body_reference_minmax(
                    k_tile,
                    k_body + size_t(record)*k_body_record_stride_bytes,
                    k_scales + size_t(record)*k_scale_record_stride_floats,
                    pipeline,
                    head_dim, group_size, key_bits, sinkhorn_iters, rtn_quantile,
                    pending_already_rotated, stream);
            ggml_cuda_kvarn_store_v_body_reference_minmax(
                    v_tile,
                    v_body + size_t(record)*v_body_record_stride_bytes,
                    v_scales + size_t(record)*v_scale_record_stride_floats,
                    pipeline,
                    head_dim, group_size, value_bits, sinkhorn_iters, rtn_quantile, turbo_v_mode,
                    pending_already_rotated, stream);
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
        uint32_t turbo_v_mode,
        bool src_in_frame,
        void * stream) {
    kvarn_validate_v_mode(head_dim, group_size, value_bits, turbo_v_mode);
    const size_t tile_floats = size_t(head_dim)*group_size;
    const size_t per_pipeline = kvarn_store_scratch_floats_one(head_dim, group_size);
    float * k_tile = scratch;
    float * v_tile = scratch + tile_floats;
    float * pipeline = scratch + 2*tile_floats;
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    const int block = 256;
    const int grid = int((tile_floats + block - 1)/block);

    // Pending tiles inherit the graph paper frame; raw tiles routed through
    // these ops (e.g. direct all-heads stores) must still be rotated here.
    const bool pending_already_rotated = src_in_frame && kvarn_paper_frame_enabled() && !kvarn_paper_mixed_frame_enabled();
    for (uint32_t ih = 0; ih < n_heads; ++ih) {
        const float * k_pending = pending_k + size_t(ih)*head_dim;
        const float * v_pending = pending_v + size_t(ih)*head_dim;
        kvarn_transpose_pending_k_head_kernel<<<grid, block, 0, cuda_stream>>>(
                k_pending, k_tile, head_dim, group_size, uint32_t(pending_k_head_stride_floats));
        kvarn_gather_pending_v_head_kernel<<<grid, block, 0, cuda_stream>>>(
                v_pending, v_tile, head_dim, group_size, uint32_t(pending_v_head_stride_floats));

        if (head_dim >= 256) {
            ggml_cuda_kvarn_store_kv_body_pipelined(
                    k_tile, v_tile,
                    k_body + size_t(ih)*k_body_stride_bytes,
                    v_body + size_t(ih)*v_body_stride_bytes,
                    k_scales + size_t(ih)*k_scale_stride_floats,
                    v_scales + size_t(ih)*v_scale_stride_floats,
                    pipeline,
                    head_dim, group_size, key_bits, value_bits, sinkhorn_iters, rtn_quantile,
                    pending_already_rotated, g_kvarn_debug_store_context.record0, ih, turbo_v_mode, stream);
        } else {
            ggml_cuda_kvarn_store_k_body_reference_minmax(
                    k_tile,
                    k_body + size_t(ih)*k_body_stride_bytes,
                    k_scales + size_t(ih)*k_scale_stride_floats,
                    pipeline,
                    head_dim, group_size, key_bits, sinkhorn_iters, rtn_quantile,
                    pending_already_rotated, stream);
            ggml_cuda_kvarn_store_v_body_reference_minmax(
                    v_tile,
                    v_body + size_t(ih)*v_body_stride_bytes,
                    v_scales + size_t(ih)*v_scale_stride_floats,
                    pipeline,
                    head_dim, group_size, value_bits, sinkhorn_iters, rtn_quantile, turbo_v_mode,
                    pending_already_rotated, stream);
        }
    }
}

void ggml_cuda_kvarn_store_body_direct_records_minmax(
        const float * k_tiles,
        const float * v_tiles,
        uint8_t * k_body,
        uint8_t * v_body,
        float * k_scales,
        float * v_scales,
        float * scratch,
        uint32_t n_heads,
        uint32_t n_records,
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
        size_t k_tile_head_stride_floats,
        size_t v_tile_head_stride_floats,
        size_t k_tile_group_stride_floats,
        size_t v_tile_group_stride_floats,
        size_t k_tile_record_stride_floats,
        size_t v_tile_record_stride_floats,
        size_t scratch_floats,
        uint32_t turbo_v_mode,
        void * stream) {
    kvarn_validate_v_mode(head_dim, group_size, value_bits, turbo_v_mode);
    const size_t tile_floats = size_t(head_dim)*group_size;
    const size_t per_pipeline = kvarn_store_scratch_floats_one(head_dim, group_size);
    float * k_tile = scratch;
    float * v_tile = scratch + tile_floats;
    float * pipeline = scratch + 2*tile_floats;
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    const int block = 256;
    const int grid = int((tile_floats + block - 1)/block);

    const bool use_batched_phases =
        kvarn_paper_frame_enabled() &&
        !kvarn_env_flag("LLAMA_KVARN_DEBUG_BODY_RECORD_DUMP") &&
        !kvarn_debug_raw_body_capture_enabled() &&
        !kvarn_env_flag("LLAMA_KVARN_DISABLE_DIRECT_RECORD_BATCH_PHASES");
    if (use_batched_phases &&
            ggml_cuda_kvarn_store_body_direct_records_batched_fullrange(
                k_tiles, v_tiles, k_body, v_body, k_scales, v_scales, scratch,
                n_heads, n_records, head_dim, group_size, key_bits, value_bits,
                sinkhorn_iters, rtn_quantile, scratch_floats,
                k_body_record_stride_bytes, v_body_record_stride_bytes,
                k_body_head_stride_bytes, v_body_head_stride_bytes,
                k_scale_record_stride_floats, v_scale_record_stride_floats,
                k_scale_head_stride_floats, v_scale_head_stride_floats,
                k_tile_head_stride_floats, v_tile_head_stride_floats,
                k_tile_group_stride_floats, v_tile_group_stride_floats,
                k_tile_record_stride_floats, v_tile_record_stride_floats,
                turbo_v_mode,
                cuda_stream)) {
        if (kvarn_store_phase_trace_claim()) {
            std::fprintf(stderr,
                    "KVarN CUDA store-body batched-phases trace: used=1 head_dim=%u group_size=%u n_records=%u n_heads=%u scratch_floats=%zu\n",
                    head_dim, group_size, n_records, n_heads, scratch_floats);
        }
        return;
    }
    if (use_batched_phases && kvarn_store_phase_trace_claim()) {
        std::fprintf(stderr,
                "KVarN CUDA store-body batched-phases trace: used=0 head_dim=%u group_size=%u n_records=%u n_heads=%u scratch_floats=%zu\n",
                head_dim, group_size, n_records, n_heads, scratch_floats);
    }
    if (use_batched_phases && kvarn_env_flag("LLAMA_KVARN_REQUIRE_DIRECT_RECORD_BATCH_PHASES")) {
        std::fprintf(stderr,
                "KVarN CUDA store-body batched-phases required but unavailable: head_dim=%u group_size=%u "
                "key_bits=%u value_bits=%u n_records=%u n_heads=%u rtn_quantile=%g scratch_floats=%zu\n",
                head_dim, group_size, key_bits, value_bits, n_records, n_heads, double(rtn_quantile), scratch_floats);
        std::abort();
    }

    for (uint32_t record = 0; record < n_records; ++record) {
        for (uint32_t ih = 0; ih < n_heads; ++ih) {
            const float * k_src = k_tiles + size_t(record)*k_tile_record_stride_floats + size_t(ih)*k_tile_head_stride_floats;
            const float * v_src = v_tiles + size_t(record)*v_tile_record_stride_floats + size_t(ih)*v_tile_head_stride_floats;
            kvarn_transpose_pending_k_head_kernel<<<grid, block, 0, cuda_stream>>>(
                    k_src, k_tile, head_dim, group_size, uint32_t(k_tile_group_stride_floats));
            kvarn_gather_pending_v_head_kernel<<<grid, block, 0, cuda_stream>>>(
                    v_src, v_tile, head_dim, group_size, uint32_t(v_tile_group_stride_floats));

            if (head_dim >= 256) {
                ggml_cuda_kvarn_store_kv_body_pipelined(
                        k_tile, v_tile,
                        k_body + size_t(ih)*k_body_head_stride_bytes + size_t(record)*k_body_record_stride_bytes,
                        v_body + size_t(ih)*v_body_head_stride_bytes + size_t(record)*v_body_record_stride_bytes,
                        k_scales + size_t(ih)*k_scale_head_stride_floats + size_t(record)*k_scale_record_stride_floats,
                        v_scales + size_t(ih)*v_scale_head_stride_floats + size_t(record)*v_scale_record_stride_floats,
                        pipeline,
                        head_dim, group_size, key_bits, value_bits, sinkhorn_iters, rtn_quantile,
                        false, record, ih, turbo_v_mode, stream);
            } else {
                ggml_cuda_kvarn_store_k_body_reference_minmax(
                        k_tile,
                        k_body + size_t(ih)*k_body_head_stride_bytes + size_t(record)*k_body_record_stride_bytes,
                        k_scales + size_t(ih)*k_scale_head_stride_floats + size_t(record)*k_scale_record_stride_floats,
                        pipeline,
                        head_dim, group_size, key_bits, sinkhorn_iters, rtn_quantile,
                        false, stream);
                ggml_cuda_kvarn_store_v_body_reference_minmax(
                        v_tile,
                        v_body + size_t(ih)*v_body_head_stride_bytes + size_t(record)*v_body_record_stride_bytes,
                        v_scales + size_t(ih)*v_scale_head_stride_floats + size_t(record)*v_scale_record_stride_floats,
                        pipeline,
                        head_dim, group_size, value_bits, sinkhorn_iters, rtn_quantile, turbo_v_mode,
                        false, stream);
            }
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
        uint32_t turbo_v_mode,
        bool input_already_rotated,
        void * stream,
        uint32_t debug_record,
        uint32_t debug_head) {
    kvarn_validate_v_mode(head_dim, group_size, value_bits, turbo_v_mode);
    if (head_dim >= 256) {
        ggml_cuda_kvarn_store_kv_body_pipelined(
                k_tile, v_tile, k_body, v_body, k_scales, v_scales, scratch,
                head_dim, group_size, key_bits, value_bits, sinkhorn_iters, rtn_quantile,
                input_already_rotated, debug_record, debug_head, turbo_v_mode, stream);
        return;
    }

    ggml_cuda_kvarn_store_k_body_reference_minmax(
            k_tile, k_body, k_scales, scratch,
            head_dim, group_size, key_bits, sinkhorn_iters, rtn_quantile,
            input_already_rotated, stream);
    ggml_cuda_kvarn_store_v_body_reference_minmax(
            v_tile, v_body, v_scales, scratch,
            head_dim, group_size, value_bits, sinkhorn_iters, rtn_quantile, turbo_v_mode,
            input_already_rotated, stream);
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
        size_t n,
        uint32_t turbo_v) {
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

    if (turbo_v == 6u) {
        v_out[i] = kvarn_sparse_d512_v_dequant_rotated(v_body, v_scales, g_v, d_v);
    } else if (turbo_v != 0) {
        v_out[i] = kvarn_turbo_v_dequant_rotated(v_body, v_scales, head_dim, group_size, value_bits, g_v, d_v, turbo_v);
    } else {
        const uint32_t vq = kvarn_unpack_one(v_body, value_bits, i);
        v_out[i] = (float(vq)*v_s_row[g_v] + v_zp[g_v])*v_s_col[d_v];
    }
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
        uint32_t turbo_v_mode,
        void * stream) {
    kvarn_validate_v_mode(head_dim, group_size, value_bits, turbo_v_mode);
    const size_t n = size_t(head_dim)*group_size;
    const int block = 256;
    const int grid = int((n + block - 1)/block);

    kvarn_dequant_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
            k_body, v_body, k_scales, v_scales, k_out, v_out,
            head_dim, group_size, key_bits, value_bits, n, turbo_v_mode);
}

template<typename T>
static __device__ __forceinline__ T kvarn_dequant_store_cast(float v);
template<> __device__ __forceinline__ float kvarn_dequant_store_cast<float>(float v) { return v; }
template<> __device__ __forceinline__ __half kvarn_dequant_store_cast<__half>(float v) { return __float2half(v); }

template<typename T>
static __global__ void kvarn_dequant_n_kernel(
        const uint8_t * __restrict__ k_body,
        const uint8_t * __restrict__ v_body,
        const float * __restrict__ k_scales,
        const float * __restrict__ v_scales,
        T * __restrict__ k_out,
        T * __restrict__ v_out,
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
        uint32_t k_token_major,
        uint32_t turbo_v) {
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
    T * k_record_out = k_out + size_t(r)*k_out_stride_floats;
    T * v_record_out = v_out == nullptr ? nullptr : v_out + size_t(r)*v_out_stride_floats;

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
    k_record_out[i] = kvarn_dequant_store_cast<T>((float(kq)*k_s_col[d_k] + k_zp[d_k])*k_s_row[g_k]);

    const uint32_t g_v = i / head_dim;
    const uint32_t d_v = i - size_t(g_v)*head_dim;

    const float * v_s_col = v_record_scales;
    const float * v_s_row = v_record_scales + head_dim;
    const float * v_zp    = v_record_scales + head_dim + group_size;

    if (v_record_out != nullptr) {
        if (turbo_v == 6u) {
            v_record_out[i] = kvarn_dequant_store_cast<T>(
                    kvarn_sparse_d512_v_dequant_rotated(v_record, v_record_scales, g_v, d_v));
        } else if (turbo_v != 0) {
            v_record_out[i] = kvarn_dequant_store_cast<T>(
                    kvarn_turbo_v_dequant_rotated(v_record, v_record_scales, head_dim, group_size, value_bits, g_v, d_v, turbo_v));
        } else {
            const uint32_t vq = kvarn_unpack_one(v_record, value_bits, i);
            v_record_out[i] = kvarn_dequant_store_cast<T>((float(vq)*v_s_row[g_v] + v_zp[g_v])*v_s_col[d_v]);
        }
    }
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
        uint32_t turbo_v_mode,
        void * stream) {
    kvarn_validate_v_mode(head_dim, group_size, value_bits, turbo_v_mode);
    const size_t n_per_record = size_t(head_dim)*group_size;
    const size_t n_total = size_t(n_records)*n_per_record;
    const int block = 256;
    const int grid = int((n_total + block - 1)/block);
    kvarn_dequant_n_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
            k_body, v_body, k_scales, v_scales, k_out, v_out,
            n_records, head_dim, group_size, key_bits, value_bits,
            k_body_stride_bytes, v_body_stride_bytes,
            k_scale_stride_floats, v_scale_stride_floats,
            k_out_stride_floats, v_out_stride_floats, n_per_record, 0u, turbo_v_mode);
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
        uint32_t turbo_v_mode,
        void * stream) {
    kvarn_validate_v_mode(head_dim, group_size, value_bits, turbo_v_mode);
    const size_t n_per_record = size_t(head_dim)*group_size;
    const size_t n_total = size_t(n_records)*n_per_record;
    const int block = 256;
    const int grid = int((n_total + block - 1)/block);
    kvarn_dequant_n_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
            k_body, v_body, k_scales, v_scales, k_out, v_out,
            n_records, head_dim, group_size, key_bits, value_bits,
            k_body_stride_bytes, v_body_stride_bytes,
            k_scale_stride_floats, v_scale_stride_floats,
            k_out_stride_floats, v_out_stride_floats, n_per_record, 1u, turbo_v_mode);
}

// f16 variant of the token-major dequant: same indexing, __half outputs.
// Halves body-scratch read traffic in the 512d warpqk attention path. The
// f16 rounding (~1e-3 relative) is negligible against the 4-bit/2-bit body
// quantization error and is covered by Tier 2 logits gates.
void ggml_cuda_kvarn_dequant_body_n_k_token_major_f16(
        const uint8_t * k_body,
        const uint8_t * v_body,
        const float * k_scales,
        const float * v_scales,
        void * k_out,
        void * v_out,
        uint32_t n_records,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t key_bits,
        uint32_t value_bits,
        size_t k_body_stride_bytes,
        size_t v_body_stride_bytes,
        size_t k_scale_stride_floats,
        size_t v_scale_stride_floats,
        size_t k_out_stride_elems,
        size_t v_out_stride_elems,
        uint32_t turbo_v_mode,
        void * stream) {
    kvarn_validate_v_mode(head_dim, group_size, value_bits, turbo_v_mode);
    const size_t n_per_record = size_t(head_dim)*group_size;
    const size_t n_total = size_t(n_records)*n_per_record;
    const int block = 256;
    const int grid = int((n_total + block - 1)/block);
    kvarn_dequant_n_kernel<__half><<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
            k_body, v_body, k_scales, v_scales,
            static_cast<__half *>(k_out), static_cast<__half *>(v_out),
            n_records, head_dim, group_size, key_bits, value_bits,
            k_body_stride_bytes, v_body_stride_bytes,
            k_scale_stride_floats, v_scale_stride_floats,
            k_out_stride_elems, v_out_stride_elems, n_per_record, 1u, turbo_v_mode);
}

static __global__ void kvarn_materialize_kv_f16_kernel(
        const uint16_t * __restrict__ sink_tail,
        const uint8_t * __restrict__ body,
        const float * __restrict__ scales,
        const float * __restrict__ pending,
        const float * __restrict__ body_f32,
        uint16_t * __restrict__ out,
        uint32_t is_v,
        uint32_t n_sink,
        uint32_t n_records,
        uint32_t n_pending,
        uint32_t n_tail,
        uint32_t tail_start,
        uint32_t n_head_kv,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t bits,
        uint32_t turbo_v_mode,
        size_t sink_tail_stride_head_f16,
        size_t sink_tail_stride_token_f16,
        size_t body_stride_record_bytes,
        size_t body_stride_head_bytes,
        size_t scale_stride_record_floats,
        size_t scale_stride_head_floats,
        size_t pending_stride_head_floats,
        size_t pending_stride_token_floats,
        size_t body_f32_stride_head_floats,
        size_t out_stride_head_f16,
        size_t out_stride_token_f16,
        uint64_t total) {
    const uint64_t i0 = uint64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const uint64_t stride = uint64_t(blockDim.x)*gridDim.x;
    const uint32_t n_body = n_records*group_size;

    for (uint64_t i = i0; i < total; i += stride) {
        const uint32_t d = uint32_t(i%head_dim);
        const uint64_t q = i/head_dim;
        const uint32_t ih = uint32_t(q%n_head_kv);
        const uint32_t t = uint32_t(q/n_head_kv);

        float value = 0.0f;
        if (t < n_sink) {
            const uint16_t * src = sink_tail + size_t(ih)*sink_tail_stride_head_f16 + size_t(t)*sink_tail_stride_token_f16;
            value = __half2float(reinterpret_cast<const __half *>(src)[d]);
        } else if (t < n_sink + n_body) {
            const uint32_t body_t = t - n_sink;
            const uint32_t r = body_t/group_size;
            const uint32_t g = body_t - r*group_size;
            const uint8_t * record = body + size_t(ih)*body_stride_head_bytes + size_t(r)*body_stride_record_bytes;
            const float * record_scales = scales + size_t(ih)*scale_stride_head_floats + size_t(r)*scale_stride_record_floats;
            const float * body_f32_head = body_f32 == nullptr ? nullptr : body_f32 + size_t(ih)*body_f32_stride_head_floats;
            if (body_f32_head != nullptr) {
                value = body_f32_head[size_t(body_t)*head_dim + d];
            } else if (is_v) {
                if (turbo_v_mode == 6u) {
                    value = kvarn_sparse_d512_v_dequant_rotated(record, record_scales, g, d);
                } else if (turbo_v_mode != 0) {
                    value = kvarn_turbo_v_dequant_rotated(record, record_scales, head_dim, group_size, bits, g, d, turbo_v_mode);
                } else {
                    const float * v_s_col = record_scales;
                    const float * v_s_row = record_scales + head_dim;
                    const float * v_zp    = record_scales + head_dim + group_size;
                    const size_t qi = size_t(g)*head_dim + d;
                    const uint32_t vq = kvarn_unpack_one(record, bits, qi);
                    value = (float(vq)*v_s_row[g] + v_zp[g])*v_s_col[d];
                }
            } else {
                const float * k_s_col = record_scales;
                const float * k_zp    = record_scales + head_dim;
                const float * k_s_row = record_scales + 2*head_dim;
                const size_t qi = size_t(d)*group_size + g;
                const uint32_t kq = kvarn_unpack_one(record, bits, qi);
                value = (float(kq)*k_s_col[d] + k_zp[d])*k_s_row[g];
            }
        } else if (t < n_sink + n_body + n_pending) {
            const uint32_t pending_t = t - n_sink - n_body;
            const float * src = pending + size_t(ih)*pending_stride_head_floats + size_t(pending_t)*pending_stride_token_floats;
            value = src[d];
        } else {
            const uint32_t tail_t = t - n_sink - n_body - n_pending;
            const uint32_t tail_slot = n_tail == 0 ? 0 : (tail_start + tail_t)%n_tail;
            const uint16_t * src = sink_tail + size_t(ih)*sink_tail_stride_head_f16 + size_t(n_sink + tail_slot)*sink_tail_stride_token_f16;
            value = __half2float(reinterpret_cast<const __half *>(src)[d]);
        }

        uint16_t * dst = out + size_t(ih)*out_stride_head_f16 + size_t(t)*out_stride_token_f16;
        reinterpret_cast<__half *>(dst)[d] = __float2half(value);
    }
}

void ggml_cuda_kvarn_materialize_kv_f16(
        const uint16_t * sink_tail,
        const uint8_t * body,
        const float * scales,
        const float * pending,
        uint16_t * out,
        uint32_t is_v,
        uint32_t n_sink,
        uint32_t n_records,
        uint32_t n_pending,
        uint32_t n_tail,
        uint32_t tail_start,
        uint32_t n_head_kv,
        uint32_t head_dim,
        uint32_t group_size,
        uint32_t bits,
        uint32_t turbo_v_mode,
        uint32_t debug_raw_body,
        const void * raw_mirror_key,
        size_t sink_tail_stride_head_f16,
        size_t sink_tail_stride_token_f16,
        size_t body_stride_record_bytes,
        size_t body_stride_head_bytes,
        size_t scale_stride_record_floats,
        size_t scale_stride_head_floats,
        size_t pending_stride_head_floats,
        size_t pending_stride_token_floats,
        size_t out_stride_head_f16,
        size_t out_stride_token_f16,
        void * stream) {
    if (is_v != 0u) {
        kvarn_validate_v_mode(head_dim, group_size, bits, turbo_v_mode);
    } else if (turbo_v_mode != 0u) {
        std::fprintf(stderr, "KVarN CUDA rejected a V layout mode for K materialization\n");
        std::abort();
    }
    const uint64_t n_kv = uint64_t(n_sink) + uint64_t(n_records)*group_size + n_pending + n_tail;
    const uint64_t total = n_kv*n_head_kv*head_dim;
    if (total == 0) {
        return;
    }
    const bool use_raw_body = debug_raw_body && n_records > 0;
    const float * body_f32 = nullptr;
    size_t body_f32_stride_head_floats = 0;
    if (use_raw_body) {
        kvarn_raw_body_mirror_entry raw_body;
        const bool found_raw_body = raw_mirror_key != nullptr &&
            kvarn_raw_body_mirror_find(
                    raw_mirror_key, n_records, n_head_kv, head_dim, group_size, raw_body);
        if (!found_raw_body) {
            std::fprintf(stderr,
                    "KVarN materialize raw body requested without an exact captured mirror"
                    " (raw_key=%p body=%p mirror_count=%zu records=%u n_head_kv=%u head_dim=%u group_size=%u is_v=%u)\n",
                    raw_mirror_key, (const void *) body, kvarn_raw_body_mirror_count(),
                    n_records, n_head_kv, head_dim, group_size, is_v);
            std::abort();
        }
        body_f32 = is_v ? raw_body.v : raw_body.k;
        body_f32_stride_head_floats = size_t(raw_body.n_records_cap)*group_size*head_dim;
    }
    const int block = 256;
    const int grid = int(std::min<uint64_t>((total + block - 1)/block, 65535));
    kvarn_materialize_kv_f16_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
            sink_tail, body, scales, pending, body_f32, out,
            is_v, n_sink, n_records, n_pending, n_tail, tail_start,
            n_head_kv, head_dim, group_size, bits,
            turbo_v_mode,
            sink_tail_stride_head_f16, sink_tail_stride_token_f16,
            body_stride_record_bytes, body_stride_head_bytes,
            scale_stride_record_floats, scale_stride_head_floats,
            pending_stride_head_floats, pending_stride_token_floats,
            body_f32_stride_head_floats,
            out_stride_head_f16, out_stride_token_f16,
            total);
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
        const float * __restrict__ q_body,
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
        int32_t causal_limit,
        float scale,
        float logit_softcap) {
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
            const float * qk = q_body == nullptr ? q : q_body;
            for (uint32_t d = 0; d < head_dim; ++d) {
                sum += qk[d]*k_record[size_t(d)*group_size + g];
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

        scores[t] = kvarn_softcap_attn_score(sum, scale, logit_softcap) +
            kvarn_kq_mask_bias(kq_mask, kq_mask_type, kq_mask_stride_token_bytes, t, causal_limit);
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
        const float * q_body,
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
        size_t q_body_stride_head_floats,
        size_t q_body_stride_query_floats,
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
        void * stream,
        float logit_softcap) {
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
            const float * q_body_ptr = q_body == nullptr ? nullptr :
                q_body + size_t(iq)*q_body_stride_query_floats + size_t(ih)*q_body_stride_head_floats;
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
                    q_ptr, q_body_ptr, k_st_ptr, k_body_ptr, pending_k_ptr, kq_mask_ptr, scores,
                    n_sink, n_records, n_pending, n_tail, tail_start, head_dim, group_size,
                    sink_tail_stride_token_f16, pending_stride_token_floats,
                    k_body_stride_record_floats, kq_mask_stride_token_bytes, kq_mask_type,
                    kvarn_causal_mask_limit(n_tokens, n_queries, iq), scale, logit_softcap);
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
        const float * __restrict__ q_body,
        const uint16_t * __restrict__ sink_tail_k,
        const uint8_t * __restrict__ k_body,
        const float * __restrict__ k_scales,
        const float * __restrict__ body_k_f32,
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
        int32_t causal_limit,
        float scale,
        float logit_softcap) {
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
            const float * qk = q_body == nullptr ? q : q_body;
            const uint32_t body_t = t - n_sink;
            const uint32_t r = body_t/group_size;
            const uint32_t g = body_t - r*group_size;

            for (uint32_t d = 0; d < head_dim; ++d) {
                float k;
                if (body_k_f32 != nullptr) {
                    k = body_k_f32[size_t(body_t)*head_dim + d];
                } else {
                    const uint8_t * k_record = k_body + size_t(r)*k_body_stride_bytes;
                    const float * k_record_scales = k_scales + size_t(r)*k_scale_stride_floats;
                    const float * k_s_col = k_record_scales;
                    const float * k_zp    = k_record_scales + head_dim;
                    const float * k_s_row = k_record_scales + 2*head_dim;
                    const size_t i = size_t(d)*group_size + g;
                    const uint32_t kq = kvarn_unpack_one(k_record, key_bits, i);
                    k = (float(kq)*k_s_col[d] + k_zp[d])*k_s_row[g];
                }
                sum += qk[d]*k;
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

        scores[t] = kvarn_softcap_attn_score(sum, scale, logit_softcap) +
            kvarn_kq_mask_bias(kq_mask, kq_mask_type, kq_mask_stride_token_bytes, t, causal_limit);
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

static __global__ void kvarn_attn_mixed_f16_scores_softmax_global_kernel(
        const float * __restrict__ q,
        const float * __restrict__ q_body,
        const uint16_t * __restrict__ sink_tail_k,
        const uint8_t * __restrict__ k_body,
        const float * __restrict__ k_scales,
        const float * __restrict__ body_k_f32,
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
        int32_t causal_limit,
        float scale,
        float logit_softcap) {
    extern __shared__ float reduce[];

    const uint32_t n_body_tokens = n_records*group_size;
    const uint32_t n_tokens = n_sink + n_body_tokens + n_pending + n_tail;

    float local_max = -3.4028234663852886e38f;
    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        float sum = 0.0f;

        if (t < n_sink) {
            const uint16_t * k = sink_tail_k + size_t(t)*sink_tail_stride_token_f16;
            for (uint32_t d = 0; d < head_dim; ++d) {
                sum += q[d]*__half2float(reinterpret_cast<const __half *>(k)[d]);
            }
        } else if (t < n_sink + n_body_tokens) {
            const float * qk = q_body == nullptr ? q : q_body;
            const uint32_t body_t = t - n_sink;
            const uint32_t r = body_t/group_size;
            const uint32_t g = body_t - r*group_size;

            for (uint32_t d = 0; d < head_dim; ++d) {
                float k;
                if (body_k_f32 != nullptr) {
                    k = body_k_f32[size_t(body_t)*head_dim + d];
                } else {
                    const uint8_t * k_record = k_body + size_t(r)*k_body_stride_bytes;
                    const float * k_record_scales = k_scales + size_t(r)*k_scale_stride_floats;
                    const float * k_s_col = k_record_scales;
                    const float * k_zp    = k_record_scales + head_dim;
                    const float * k_s_row = k_record_scales + 2*head_dim;
                    const size_t i = size_t(d)*group_size + g;
                    const uint32_t kq = kvarn_unpack_one(k_record, key_bits, i);
                    k = (float(kq)*k_s_col[d] + k_zp[d])*k_s_row[g];
                }
                sum += qk[d]*k;
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

        const float score = kvarn_softcap_attn_score(sum, scale, logit_softcap) +
            kvarn_kq_mask_bias(kq_mask, kq_mask_type, kq_mask_stride_token_bytes, t, causal_limit);
        scores[t] = score;
        local_max = fmaxf(local_max, score);
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
        scores[t] = p;
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
        scores[t] *= inv_denom;
    }
}

static __global__ void kvarn_attn_mixed_f16_av_kernel(
        const float * __restrict__ probs,
        const uint16_t * __restrict__ sink_tail_v,
        const uint8_t * __restrict__ v_body,
        const float * __restrict__ v_scales,
        const float * __restrict__ body_v_f32,
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
        size_t v_scale_stride_floats,
        uint32_t turbo_v_mode) {
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
        for (uint32_t g = 0; g < group_size; ++g) {
            float v;
            if (body_v_f32 != nullptr) {
                v = body_v_f32[(size_t(r)*group_size + g)*head_dim + d];
            } else if (turbo_v_mode == 6u) {
                v = kvarn_sparse_d512_v_dequant_rotated(v_record, v_record_scales, g, d);
            } else {
                v = kvarn_turbo_v_dequant_rotated(v_record, v_record_scales, head_dim, group_size, value_bits, g, d, turbo_v_mode);
            }
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

// Canonical Turbo V stores each 128-channel body block in the
// S2*H_normalized*S1 basis.  Sink, pending, and tail values remain in the
// ordinary KVarN frame, so only the weighted body contribution may be
// transformed back before the components are added.
static __global__ void kvarn_attn_mixed_f16_turbo2_av_kernel(
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
    const uint32_t j = threadIdx.x;
    const uint32_t d = blockIdx.x*128u + j;
    if (j >= 128u || d >= head_dim) {
        return;
    }

    const uint32_t n_body_tokens = n_records*group_size;
    float exact_sum = 0.0f;
    float body_sum = 0.0f;

    for (uint32_t t = 0; t < n_sink; ++t) {
        const uint16_t * v = sink_tail_v + size_t(t)*sink_tail_stride_token_f16;
        exact_sum += probs[t]*__half2float(reinterpret_cast<const __half *>(v)[d]);
    }

    for (uint32_t r = 0; r < n_records; ++r) {
        const uint8_t * v_record = v_body + size_t(r)*v_body_stride_bytes;
        const float * v_record_scales = v_scales + size_t(r)*v_scale_stride_floats;
        for (uint32_t g = 0; g < group_size; ++g) {
            const float v = kvarn_turbo_v_dequant_rotated(
                    v_record, v_record_scales, head_dim, group_size, value_bits, g, d, 2u);
            body_sum += probs[size_t(n_sink) + size_t(r)*group_size + g]*v;
        }
    }

    for (uint32_t t = 0; t < n_pending; ++t) {
        const float * v = pending_v + size_t(t)*pending_stride_token_floats;
        exact_sum += probs[size_t(n_sink) + n_body_tokens + t]*v[d];
    }

    for (uint32_t t = 0; t < n_tail; ++t) {
        const uint32_t tail_slot = (tail_start + t)%n_tail;
        const uint16_t * v = sink_tail_v + size_t(n_sink + tail_slot)*sink_tail_stride_token_f16;
        exact_sum += probs[size_t(n_sink) + n_body_tokens + n_pending + t]*
                __half2float(reinterpret_cast<const __half *>(v)[d]);
    }

    __shared__ float body_block[128];
    body_block[j] = body_sum*KVARN_TURBO_WHT_SIGNS2_128[j];
    __syncthreads();

    for (uint32_t step = 1; step < 128; step <<= 1) {
        if ((j & step) == 0) {
            const uint32_t j1 = j + step;
            const float a = body_block[j];
            const float b = body_block[j1];
            body_block[j]  = a + b;
            body_block[j1] = a - b;
        }
        __syncthreads();
    }

    out[d] = exact_sum + body_block[j]*0.08838834764831845f*KVARN_TURBO_WHT_SIGNS1_128[j];
}

static __global__ void kvarn_attn_mixed_f16_fused_kernel(
        const float * __restrict__ q,
        const float * __restrict__ q_body,
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
        int32_t causal_limit,
        float scale,
        float logit_softcap,
        uint32_t turbo_v_mode) {
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
            const float * qk = q_body == nullptr ? q : q_body;
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
                sum += qk[d]*k;
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

        scores[t] = kvarn_softcap_attn_score(sum, scale, logit_softcap) +
            kvarn_kq_mask_bias(kq_mask, kq_mask_type, kq_mask_stride_token_bytes, t, causal_limit);
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

            for (uint32_t g = 0; g < group_size; ++g) {
                const float v = turbo_v_mode == 6u ?
                    kvarn_sparse_d512_v_dequant_rotated(v_record, v_record_scales, g, d) :
                    kvarn_turbo_v_dequant_rotated(v_record, v_record_scales, head_dim, group_size, value_bits, g, d, turbo_v_mode);
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
        size_t v_scale_stride_record_floats,
        uint32_t turbo_v_mode) {
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
        return turbo_v_mode == 6u ?
            kvarn_sparse_d512_v_dequant_rotated(v_record, v_record_scales, g, d) :
            kvarn_turbo_v_dequant_rotated(v_record, v_record_scales, head_dim, group_size, value_bits, g, d, turbo_v_mode);
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
        float scale,
        float logit_softcap,
        bool fused_paper_frame) {
    const uint32_t row = blockIdx.x;
    if (row >= n_queries*n_head) {
        return;
    }

    const uint32_t iq = row/n_head;
    const uint32_t ih = row - iq*n_head;
    const uint32_t n_gqa = n_head/n_head_kv;
    const uint32_t ikh = ih/n_gqa;
    const uint32_t n_tokens = n_sink + n_tail;
    const int32_t causal_limit = kvarn_causal_mask_limit(n_tokens, n_queries, iq);

    const float * q_row = q + size_t(iq)*q_stride_query_floats + size_t(ih)*q_stride_head_floats;
    float * out_row = out + size_t(iq)*out_stride_query_floats + size_t(ih)*out_stride_head_floats;
    const uint16_t * k_st = sink_tail_k + size_t(ikh)*sink_tail_stride_head_f16;
    const uint16_t * v_st = sink_tail_v + size_t(ikh)*sink_tail_stride_head_f16;
    const void * kq_mask_row = kq_mask == nullptr ? nullptr :
        (const char *) kq_mask + size_t(iq)*kq_mask_stride_query_bytes;

    extern __shared__ float shared[];
    float * q_sh   = fused_paper_frame ? shared : nullptr;
    float * probs  = fused_paper_frame ? shared + head_dim : shared;
    float * reduce = shared + n_sink + n_tail;
    if (fused_paper_frame) {
        reduce = probs + n_sink + n_tail;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            q_sh[d] = q_row[d];
        }
        __syncthreads();
        kvarn_fwht_shared_normalized(q_sh, head_dim);
    }

    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t n_warps = blockDim.x >> 5;

    for (uint32_t t = warp; t < n_tokens; t += n_warps) {
        const uint32_t slot = t < n_sink ? t : n_sink + ((tail_start + (t - n_sink))%n_tail);
        const uint16_t * k = k_st + size_t(slot)*sink_tail_stride_token_f16;

        float sum = 0.0f;
        for (uint32_t d = lane; d < head_dim; d += 32) {
            const float qv = fused_paper_frame ? q_sh[d] : q_row[d];
            sum += qv*__half2float(reinterpret_cast<const __half *>(k)[d]);
        }
        sum = kvarn_warp_reduce_sum(sum);
        if (lane == 0) {
            probs[t] = kvarn_softcap_attn_score(sum, scale, logit_softcap) +
                kvarn_kq_mask_bias(kq_mask_row, kq_mask_type, kq_mask_stride_token_bytes, t, causal_limit);
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
        if (fused_paper_frame) {
            q_sh[d] = sum;
        } else {
            out_row[d] = sum;
        }
    }
    if (fused_paper_frame) {
        __syncthreads();
        kvarn_fwht_shared_normalized(q_sh, head_dim);
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            out_row[d] = q_sh[d];
        }
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
        float scale,
        bool fused_paper_frame,
        const int32_t * __restrict__ window_dev,
        float logit_softcap) {
    // One CTA per query head. The previous version launched a single CTA
    // with one warp per head, leaving all but one SM idle during decode —
    // the primary tg64 throughput ceiling for Gemma 512d sink/tail decode.
    const uint32_t ih = blockIdx.x;
    if (ih >= n_head) {
        return;
    }

    uint32_t eff_sink = n_sink;
    uint32_t eff_tail = n_tail;
    uint32_t eff_tail_start = tail_start;
    if (window_dev != nullptr) {
        // Host args are frozen caps for grid/shmem; live window via src[11].
        eff_sink       = static_cast<uint32_t>(window_dev[0]);
        eff_tail       = static_cast<uint32_t>(window_dev[3]);
        eff_tail_start = static_cast<uint32_t>(window_dev[4]);
    }

    const uint32_t n_gqa = n_head/n_head_kv;
    const uint32_t ikh = ih/n_gqa;
    const float * q_row = q + size_t(ih)*q_stride_head_floats;
    float * out_row = out + size_t(ih)*out_stride_head_floats;
    const uint16_t * k_st = sink_tail_k + size_t(ikh)*sink_tail_stride_head_f16;
    const uint16_t * v_st = sink_tail_v + size_t(ikh)*sink_tail_stride_head_f16;
    const void * kq_mask_row = kq_mask;

    extern __shared__ float shared[];
    // Host n_sink/n_tail are frozen caps when window_dev is set; shmem is sized
    // for caps while loops iterate the live window read from device memory.
    const uint32_t cap_tokens = n_sink + n_tail;
    const uint32_t n_tokens   = eff_sink + eff_tail;
    const int32_t causal_limit = int32_t(n_tokens) - 1;
    // Layout: q_sh[head_dim] | probs[cap_tokens] | reduce[blockDim.x]
    float * q_sh = shared;
    float * probs = shared + head_dim;
    float * reduce = probs + cap_tokens;

    if (n_tokens == 1) {
        const uint16_t * v = v_st;
        if (fused_paper_frame) {
            for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
                q_sh[d] = __half2float(reinterpret_cast<const __half *>(v)[d]);
            }
            __syncthreads();
            kvarn_fwht_shared_normalized(q_sh, head_dim);
            for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
                out_row[d] = q_sh[d];
            }
        } else {
            for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
                out_row[d] = __half2float(reinterpret_cast<const __half *>(v)[d]);
            }
        }
        return;
    }

    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        q_sh[d] = q_row[d];
    }
    __syncthreads();
    if (fused_paper_frame) {
        kvarn_fwht_shared_normalized(q_sh, head_dim);
    }

    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t n_warps = blockDim.x >> 5;

    if (eff_tail == 0) {
        for (uint32_t t = warp; t < n_tokens; t += n_warps) {
            const uint16_t * k = k_st + size_t(t)*sink_tail_stride_token_f16;

            float sum = 0.0f;
            for (uint32_t d = lane; d < head_dim; d += 32) {
                sum += q_sh[d]*__half2float(reinterpret_cast<const __half *>(k)[d]);
            }
            sum = kvarn_warp_reduce_sum(sum);
            if (lane == 0) {
                probs[t] = kvarn_softcap_attn_score(sum, scale, logit_softcap) +
                    kvarn_kq_mask_bias(kq_mask_row, kq_mask_type, kq_mask_stride_token_bytes, t, causal_limit);
            }
        }
    } else {
        for (uint32_t t = warp; t < n_tokens; t += n_warps) {
            const uint32_t slot = t < eff_sink ? t : eff_sink + ((eff_tail_start + (t - eff_sink))%eff_tail);
            const uint16_t * k = k_st + size_t(slot)*sink_tail_stride_token_f16;

            float sum = 0.0f;
            for (uint32_t d = lane; d < head_dim; d += 32) {
                sum += q_sh[d]*__half2float(reinterpret_cast<const __half *>(k)[d]);
            }
            sum = kvarn_warp_reduce_sum(sum);
            if (lane == 0) {
                probs[t] = kvarn_softcap_attn_score(sum, scale, logit_softcap) +
                    kvarn_kq_mask_bias(kq_mask_row, kq_mask_type, kq_mask_stride_token_bytes, t, causal_limit);
            }
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

    if (eff_tail == 0) {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float sum = 0.0f;
            for (uint32_t t = 0; t < n_tokens; ++t) {
                const uint16_t * v = v_st + size_t(t)*sink_tail_stride_token_f16;
                sum += probs[t]*__half2float(reinterpret_cast<const __half *>(v)[d]);
            }
            if (fused_paper_frame) {
                q_sh[d] = sum;
            } else {
                out_row[d] = sum;
            }
        }
    } else {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float sum = 0.0f;
            for (uint32_t t = 0; t < n_tokens; ++t) {
                const uint32_t slot = t < eff_sink ? t : eff_sink + ((eff_tail_start + (t - eff_sink))%eff_tail);
                const uint16_t * v = v_st + size_t(slot)*sink_tail_stride_token_f16;
                sum += probs[t]*__half2float(reinterpret_cast<const __half *>(v)[d]);
            }
            if (fused_paper_frame) {
                q_sh[d] = sum;
            } else {
                out_row[d] = sum;
            }
        }
    }
    if (fused_paper_frame) {
        __syncthreads();
        kvarn_fwht_shared_normalized(q_sh, head_dim);
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            out_row[d] = q_sh[d];
        }
    }
}

// Q-tiled fused warpqk attention: each CTA computes QT query rows for one
// head, so every K/V element loaded from sink/tail/body/pending is reused
// across QT dot products. The scalar single-query version had zero
// cross-query reuse (the dominant remaining pp512 kernel gap vs the cuBLAS
// GEMM attention of the normal KV path). QT=1 instantiation preserves the
// previous behavior exactly for decode-shaped launches.
template<int QT>
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
        const __half * __restrict__ body_k_f16,
        const __half * __restrict__ body_v_f16,
        size_t body_f16_stride_head_elems,
        float logit_softcap,
        uint32_t turbo_v_mode) {
    (void) scores;
    const uint32_t n_tiles = (n_queries + QT - 1)/uint32_t(QT);
    if (blockIdx.x >= n_tiles*n_head) {
        return;
    }

    const uint32_t tile = blockIdx.x/n_head;
    const uint32_t ih   = blockIdx.x - tile*n_head;
    const uint32_t iq0  = tile*uint32_t(QT);
    const uint32_t qt_n = min(uint32_t(QT), n_queries - iq0);
    const uint32_t n_gqa = n_head/n_head_kv;
    const uint32_t ikh = ih/n_gqa;

    const float * q_tile = q + size_t(iq0)*q_stride_query_floats + size_t(ih)*q_stride_head_floats;
    float * out_tile = out + size_t(iq0)*out_stride_query_floats + size_t(ih)*out_stride_head_floats;
    const uint16_t * k_st = sink_tail_k + size_t(ikh)*sink_tail_stride_head_f16;
    const uint16_t * v_st = sink_tail_v + size_t(ikh)*sink_tail_stride_head_f16;
    const uint8_t * k_body_head = k_body + size_t(ikh)*k_body_stride_head_bytes;
    const uint8_t * v_body_head = v_body + size_t(ikh)*v_body_stride_head_bytes;
    const float * k_scales_head = k_scales + size_t(ikh)*k_scale_stride_head_floats;
    const float * v_scales_head = v_scales + size_t(ikh)*v_scale_stride_head_floats;
    const float * pending_k_head = pending_k + size_t(ikh)*pending_stride_head_floats;
    const float * pending_v_head = pending_v + size_t(ikh)*pending_stride_head_floats;
    const __half * k_body_f16_head = body_k_f16 == nullptr ? nullptr :
        body_k_f16 + size_t(ikh)*body_f16_stride_head_elems;
    const __half * v_body_f16_head = body_v_f16 == nullptr ? nullptr :
        body_v_f16 + size_t(ikh)*body_f16_stride_head_elems;
    const char * kq_mask_tile = kq_mask == nullptr ? nullptr :
        (const char *) kq_mask + size_t(iq0)*kq_mask_stride_query_bytes;

    const uint32_t n_body_tokens = n_records*group_size;
    const uint32_t n_tokens = n_sink + n_body_tokens + n_pending + n_tail;

    extern __shared__ float shared[];
    // Layout: q_sh[QT*head_dim] | probs[QT*n_tokens] | reduce[blockDim.x].
    float * q_sh = shared;
    float * probs = shared + size_t(QT)*head_dim;
    float * reduce = probs + size_t(QT)*n_tokens;

    // Stage the query tile; zero-fill rows past qt_n so the unrolled QK
    // accumulation needs no per-element bounds checks.
    for (uint32_t i = threadIdx.x; i < uint32_t(QT)*head_dim; i += blockDim.x) {
        const uint32_t j = i/head_dim;
        const uint32_t d = i - j*head_dim;
        q_sh[i] = j < qt_n ? q_tile[size_t(j)*q_stride_query_floats + d] : 0.0f;
    }
    __syncthreads();

    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t n_warps = blockDim.x >> 5;

    for (uint32_t t = warp; t < n_tokens; t += n_warps) {
        float sum[QT];
#pragma unroll
        for (int j = 0; j < QT; ++j) {
            sum[j] = 0.0f;
        }
        for (uint32_t d = lane; d < head_dim; d += 32) {
            float k = 0.0f;
            if (k_body_f16_head != nullptr && t >= n_sink && t < n_sink + n_body_tokens) {
                const uint32_t body_t = t - n_sink;
                // Dequant scratch K is token-major f16 (g*head_dim + d):
                // adjacent lanes read adjacent halves.
                k = __half2float(k_body_f16_head[size_t(body_t)*head_dim + d]);
            } else {
                k = kvarn_mixed_f16_load_k(
                        k_st, k_body_head, k_scales_head, pending_k_head,
                        t, d, n_sink, n_records, n_pending, n_tail, tail_start,
                        head_dim, group_size, key_bits,
                        sink_tail_stride_token_f16, pending_stride_token_floats,
                        k_body_stride_record_bytes, k_scale_stride_record_floats);
            }
#pragma unroll
            for (int j = 0; j < QT; ++j) {
                sum[j] += q_sh[size_t(j)*head_dim + d]*k;
            }
        }
#pragma unroll
        for (int j = 0; j < QT; ++j) {
            sum[j] = kvarn_warp_reduce_sum(sum[j]);
        }
        if (lane == 0) {
            for (uint32_t j = 0; j < qt_n; ++j) {
                const void * mask_row = kq_mask_tile == nullptr ? nullptr :
                    (const void *) (kq_mask_tile + size_t(j)*kq_mask_stride_query_bytes);
                const int32_t causal_limit = kvarn_causal_mask_limit(n_tokens, n_queries, iq0 + j);
                probs[size_t(j)*n_tokens + t] =
                    kvarn_softcap_attn_score(sum[j], scale, logit_softcap) +
                    kvarn_kq_mask_bias(mask_row, kq_mask_type, kq_mask_stride_token_bytes, t, causal_limit);
            }
        }
    }
    __syncthreads();

    // Per-row softmax: rows < qt_n get the standard max/exp/normalize pass;
    // rows >= qt_n are zeroed so the AV stage can run unguarded.
    for (uint32_t j = 0; j < uint32_t(QT); ++j) {
        float * row = probs + size_t(j)*n_tokens;
        if (j >= qt_n) {
            for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
                row[t] = 0.0f;
            }
            __syncthreads();
            continue;
        }

        float local_max = -3.4028234663852886e38f;
        for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
            local_max = fmaxf(local_max, row[t]);
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
            const float p = expf(row[t] - max_score);
            row[t] = p;
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
            row[t] *= inv_denom;
        }
        __syncthreads();
    }

    // Dimension-parallel AV with the V element loaded once and reused across
    // the QT probability rows.
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc[QT];
#pragma unroll
        for (int j = 0; j < QT; ++j) {
            acc[j] = 0.0f;
        }
        for (uint32_t t = 0; t < n_tokens; ++t) {
            float v = 0.0f;
            if (v_body_f16_head != nullptr && t >= n_sink && t < n_sink + n_body_tokens) {
                const uint32_t body_t = t - n_sink;
                // V scratch is token-major f16 (g*head_dim + d), contiguous in t.
                v = __half2float(v_body_f16_head[size_t(body_t)*head_dim + d]);
            } else {
                v = kvarn_mixed_f16_load_v(
                        v_st, v_body_head, v_scales_head, pending_v_head,
                        t, d, n_sink, n_records, n_pending, n_tail, tail_start,
                        head_dim, group_size, value_bits,
                        sink_tail_stride_token_f16, pending_stride_token_floats,
                        v_body_stride_record_bytes, v_scale_stride_record_floats, turbo_v_mode);
            }
#pragma unroll
            for (int j = 0; j < QT; ++j) {
                acc[j] += probs[size_t(j)*n_tokens + t]*v;
            }
        }
        for (uint32_t j = 0; j < qt_n; ++j) {
            out_tile[size_t(j)*out_stride_query_floats + d] = acc[j];
        }
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
        float scale,
        float logit_softcap,
        uint32_t turbo_v_mode) {
    const uint32_t row = blockIdx.x;
    if (row >= n_queries*n_head) {
        return;
    }

    const uint32_t iq = row/n_head;
    const uint32_t ih = row - iq*n_head;
    const uint32_t n_gqa = n_head/n_head_kv;
    const uint32_t ikh = ih/n_gqa;
    const uint32_t n_body_tokens = n_records*group_size;
    const uint32_t n_tokens = n_sink + n_body_tokens + n_pending + n_tail;
    const int32_t causal_limit = kvarn_causal_mask_limit(n_tokens, n_queries, iq);

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

        probs[t] = kvarn_softcap_attn_score(sum, scale, logit_softcap) +
            kvarn_kq_mask_bias(kq_mask_row, kq_mask_type, kq_mask_stride_token_bytes, t, causal_limit);
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

            for (uint32_t g = 0; g < group_size; ++g) {
                const float v = turbo_v_mode == 6u ?
                    kvarn_sparse_d512_v_dequant_rotated(v_record, v_record_scales, g, d) :
                    kvarn_turbo_v_dequant_rotated(v_record, v_record_scales, head_dim, group_size, value_bits, g, d, turbo_v_mode);
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

// Query-tiled scalar fused attention for active-body prompt batches.
// Unlike the warpqk kernel, every QK dot product still accumulates dimensions
// in the same serial order as kvarn_attn_mixed_f16_fused_batch_kernel. That
// keeps packed-vs-split logits exact while reusing each loaded/dequantized K/V
// value across QT query rows in the CTA.
template<int QT>
static __global__ void kvarn_attn_mixed_f16_fused_batch_scalar_qt_kernel(
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
        const float * __restrict__ body_k_f32,
        const float * __restrict__ body_v_f32,
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
        size_t body_f32_stride_head_elems,
        float logit_softcap,
        uint32_t turbo_v_mode) {
    (void) scores;
    const uint32_t n_tiles = (n_queries + QT - 1)/uint32_t(QT);
    if (blockIdx.x >= n_tiles*n_head) {
        return;
    }

    const uint32_t tile = blockIdx.x/n_head;
    const uint32_t ih   = blockIdx.x - tile*n_head;
    const uint32_t iq0  = tile*uint32_t(QT);
    const uint32_t qt_n = min(uint32_t(QT), n_queries - iq0);
    const uint32_t n_gqa = n_head/n_head_kv;
    const uint32_t ikh = ih/n_gqa;

    const float * q_tile = q + size_t(iq0)*q_stride_query_floats + size_t(ih)*q_stride_head_floats;
    float * out_tile = out + size_t(iq0)*out_stride_query_floats + size_t(ih)*out_stride_head_floats;
    const uint16_t * k_st = sink_tail_k + size_t(ikh)*sink_tail_stride_head_f16;
    const uint16_t * v_st = sink_tail_v + size_t(ikh)*sink_tail_stride_head_f16;
    const uint8_t * k_body_head = k_body + size_t(ikh)*k_body_stride_head_bytes;
    const uint8_t * v_body_head = v_body + size_t(ikh)*v_body_stride_head_bytes;
    const float * k_scales_head = k_scales + size_t(ikh)*k_scale_stride_head_floats;
    const float * v_scales_head = v_scales + size_t(ikh)*v_scale_stride_head_floats;
    const float * pending_k_head = pending_k + size_t(ikh)*pending_stride_head_floats;
    const float * pending_v_head = pending_v + size_t(ikh)*pending_stride_head_floats;
    const float * k_body_f32_head = body_k_f32 == nullptr ? nullptr : body_k_f32 + size_t(ikh)*body_f32_stride_head_elems;
    const float * v_body_f32_head = body_v_f32 == nullptr ? nullptr : body_v_f32 + size_t(ikh)*body_f32_stride_head_elems;
    const char * kq_mask_tile = kq_mask == nullptr ? nullptr :
        (const char *) kq_mask + size_t(iq0)*kq_mask_stride_query_bytes;

    const uint32_t n_body_tokens = n_records*group_size;
    const uint32_t n_tokens = n_sink + n_body_tokens + n_pending + n_tail;

    extern __shared__ float shared[];
    float * q_sh = shared;
    float * probs = shared + size_t(QT)*head_dim;
    float * reduce = probs + size_t(QT)*n_tokens;

    for (uint32_t i = threadIdx.x; i < qt_n*head_dim; i += blockDim.x) {
        const uint32_t j = i/head_dim;
        const uint32_t d = i - j*head_dim;
        q_sh[i] = q_tile[size_t(j)*q_stride_query_floats + d];
    }
    __syncthreads();

    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        float sum[QT];
#pragma unroll
        for (int j = 0; j < QT; ++j) {
            sum[j] = 0.0f;
        }

        if (t < n_sink) {
            const uint16_t * k = k_st + size_t(t)*sink_tail_stride_token_f16;
            for (uint32_t d = 0; d < head_dim; ++d) {
                const float kv = __half2float(reinterpret_cast<const __half *>(k)[d]);
#pragma unroll
                for (int j = 0; j < QT; ++j) {
                    if (uint32_t(j) < qt_n) {
                        sum[j] += q_sh[size_t(j)*head_dim + d]*kv;
                    }
                }
            }
        } else if (t < n_sink + n_body_tokens) {
            const uint32_t body_t = t - n_sink;
            const uint32_t r = body_t/group_size;
            const uint32_t g = body_t - r*group_size;
            for (uint32_t d = 0; d < head_dim; ++d) {
                float kv;
                if (k_body_f32_head != nullptr) {
                    kv = k_body_f32_head[size_t(body_t)*head_dim + d];
                } else {
                    const uint8_t * k_record = k_body_head + size_t(r)*k_body_stride_record_bytes;
                    const float * k_record_scales = k_scales_head + size_t(r)*k_scale_stride_record_floats;
                    const float * k_s_col = k_record_scales;
                    const float * k_zp    = k_record_scales + head_dim;
                    const float * k_s_row = k_record_scales + 2*head_dim;
                    const size_t i = size_t(d)*group_size + g;
                    const uint32_t kq = kvarn_unpack_one(k_record, key_bits, i);
                    kv = (float(kq)*k_s_col[d] + k_zp[d])*k_s_row[g];
                }
#pragma unroll
                for (int j = 0; j < QT; ++j) {
                    if (uint32_t(j) < qt_n) {
                        sum[j] += q_sh[size_t(j)*head_dim + d]*kv;
                    }
                }
            }
        } else if (t < n_sink + n_body_tokens + n_pending) {
            const uint32_t pending_t = t - n_sink - n_body_tokens;
            const float * k = pending_k_head + size_t(pending_t)*pending_stride_token_floats;
            for (uint32_t d = 0; d < head_dim; ++d) {
                const float kv = k[d];
#pragma unroll
                for (int j = 0; j < QT; ++j) {
                    if (uint32_t(j) < qt_n) {
                        sum[j] += q_sh[size_t(j)*head_dim + d]*kv;
                    }
                }
            }
        } else {
            const uint32_t tail_t = t - n_sink - n_body_tokens - n_pending;
            const uint32_t tail_slot = n_tail == 0 ? 0 : (tail_start + tail_t)%n_tail;
            const uint16_t * k = k_st + size_t(n_sink + tail_slot)*sink_tail_stride_token_f16;
            for (uint32_t d = 0; d < head_dim; ++d) {
                const float kv = __half2float(reinterpret_cast<const __half *>(k)[d]);
#pragma unroll
                for (int j = 0; j < QT; ++j) {
                    if (uint32_t(j) < qt_n) {
                        sum[j] += q_sh[size_t(j)*head_dim + d]*kv;
                    }
                }
            }
        }

        for (uint32_t j = 0; j < qt_n; ++j) {
            const void * mask_row = kq_mask_tile == nullptr ? nullptr :
                (const void *) (kq_mask_tile + size_t(j)*kq_mask_stride_query_bytes);
            const int32_t causal_limit = kvarn_causal_mask_limit(n_tokens, n_queries, iq0 + j);
            probs[size_t(j)*n_tokens + t] =
                kvarn_softcap_attn_score(sum[j], scale, logit_softcap) +
                kvarn_kq_mask_bias(mask_row, kq_mask_type, kq_mask_stride_token_bytes, t, causal_limit);
        }
    }
    __syncthreads();

    for (uint32_t j = 0; j < qt_n; ++j) {
        float * row = probs + size_t(j)*n_tokens;

        float local_max = -3.4028234663852886e38f;
        for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
            local_max = fmaxf(local_max, row[t]);
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
            const float p = expf(row[t] - max_score);
            row[t] = p;
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
            row[t] *= inv_denom;
        }
        __syncthreads();
    }
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc[QT];
#pragma unroll
        for (int j = 0; j < QT; ++j) {
            acc[j] = 0.0f;
        }

        for (uint32_t t = 0; t < n_sink; ++t) {
            const uint16_t * v = v_st + size_t(t)*sink_tail_stride_token_f16;
            const float vv = __half2float(reinterpret_cast<const __half *>(v)[d]);
#pragma unroll
            for (int j = 0; j < QT; ++j) {
                if (uint32_t(j) < qt_n) {
                    acc[j] += probs[size_t(j)*n_tokens + t]*vv;
                }
            }
        }

        for (uint32_t r = 0; r < n_records; ++r) {
            for (uint32_t g = 0; g < group_size; ++g) {
                float vv;
                if (v_body_f32_head != nullptr) {
                    vv = v_body_f32_head[(size_t(r)*group_size + g)*head_dim + d];
                } else {
                    const uint8_t * v_record = v_body_head + size_t(r)*v_body_stride_record_bytes;
                    const float * v_record_scales = v_scales_head + size_t(r)*v_scale_stride_record_floats;
                    vv = turbo_v_mode == 6u ?
                        kvarn_sparse_d512_v_dequant_rotated(v_record, v_record_scales, g, d) :
                        kvarn_turbo_v_dequant_rotated(v_record, v_record_scales, head_dim, group_size, value_bits, g, d, turbo_v_mode);
                }
                const uint32_t t = n_sink + r*group_size + g;
#pragma unroll
                for (int j = 0; j < QT; ++j) {
                    if (uint32_t(j) < qt_n) {
                        acc[j] += probs[size_t(j)*n_tokens + t]*vv;
                    }
                }
            }
        }

        for (uint32_t t = 0; t < n_pending; ++t) {
            const float * v = pending_v_head + size_t(t)*pending_stride_token_floats;
            const float vv = v[d];
            const uint32_t tok = n_sink + n_body_tokens + t;
#pragma unroll
            for (int j = 0; j < QT; ++j) {
                if (uint32_t(j) < qt_n) {
                    acc[j] += probs[size_t(j)*n_tokens + tok]*vv;
                }
            }
        }

        for (uint32_t t = 0; t < n_tail; ++t) {
            const uint32_t tail_slot = (tail_start + t)%n_tail;
            const uint16_t * v = v_st + size_t(n_sink + tail_slot)*sink_tail_stride_token_f16;
            const float vv = __half2float(reinterpret_cast<const __half *>(v)[d]);
            const uint32_t tok = n_sink + n_body_tokens + n_pending + t;
#pragma unroll
            for (int j = 0; j < QT; ++j) {
                if (uint32_t(j) < qt_n) {
                    acc[j] += probs[size_t(j)*n_tokens + tok]*vv;
                }
            }
        }

        for (uint32_t j = 0; j < qt_n; ++j) {
            out_tile[size_t(j)*out_stride_query_floats + d] = acc[j];
        }
    }
}

template<int QT, int HT>
static __global__ void kvarn_attn_mixed_f16_fused_batch_scalar_qt_gqa_kernel(
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
        const float * __restrict__ body_k_f32,
        const float * __restrict__ body_v_f32,
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
        size_t body_f32_stride_head_elems,
        float logit_softcap,
        uint32_t turbo_v_mode) {
    (void) scores;
    const uint32_t n_gqa = n_head/n_head_kv;
    const uint32_t h_tiles_per_kv = (n_gqa + HT - 1)/uint32_t(HT);
    const uint32_t n_q_tiles = (n_queries + QT - 1)/uint32_t(QT);
    const uint32_t n_head_tiles = n_head_kv*h_tiles_per_kv;
    if (blockIdx.x >= n_q_tiles*n_head_tiles) {
        return;
    }

    const uint32_t q_tile = blockIdx.x/n_head_tiles;
    const uint32_t head_tile = blockIdx.x - q_tile*n_head_tiles;
    const uint32_t ikh = head_tile/h_tiles_per_kv;
    const uint32_t h0_in_group = (head_tile - ikh*h_tiles_per_kv)*uint32_t(HT);
    const uint32_t iq0 = q_tile*uint32_t(QT);
    const uint32_t qt_n = min(uint32_t(QT), n_queries - iq0);
    const uint32_t ht_n = min(uint32_t(HT), n_gqa - h0_in_group);
    const uint32_t ih0 = ikh*n_gqa + h0_in_group;

    const uint16_t * k_st = sink_tail_k + size_t(ikh)*sink_tail_stride_head_f16;
    const uint16_t * v_st = sink_tail_v + size_t(ikh)*sink_tail_stride_head_f16;
    const uint8_t * k_body_head = k_body + size_t(ikh)*k_body_stride_head_bytes;
    const uint8_t * v_body_head = v_body + size_t(ikh)*v_body_stride_head_bytes;
    const float * k_scales_head = k_scales + size_t(ikh)*k_scale_stride_head_floats;
    const float * v_scales_head = v_scales + size_t(ikh)*v_scale_stride_head_floats;
    const float * pending_k_head = pending_k + size_t(ikh)*pending_stride_head_floats;
    const float * pending_v_head = pending_v + size_t(ikh)*pending_stride_head_floats;
    const float * k_body_f32_head = body_k_f32 == nullptr ? nullptr : body_k_f32 + size_t(ikh)*body_f32_stride_head_elems;
    const float * v_body_f32_head = body_v_f32 == nullptr ? nullptr : body_v_f32 + size_t(ikh)*body_f32_stride_head_elems;
    const char * kq_mask_tile = kq_mask == nullptr ? nullptr :
        (const char *) kq_mask + size_t(iq0)*kq_mask_stride_query_bytes;

    const uint32_t n_body_tokens = n_records*group_size;
    const uint32_t n_tokens = n_sink + n_body_tokens + n_pending + n_tail;

    extern __shared__ float shared[];
    float * q_sh = shared;
    float * probs = shared + size_t(HT)*size_t(QT)*head_dim;
    float * reduce = probs + size_t(HT)*size_t(QT)*n_tokens;

    for (uint32_t i = threadIdx.x; i < uint32_t(HT)*uint32_t(QT)*head_dim; i += blockDim.x) {
        const uint32_t h = i/(uint32_t(QT)*head_dim);
        const uint32_t rem = i - h*uint32_t(QT)*head_dim;
        const uint32_t j = rem/head_dim;
        const uint32_t d = rem - j*head_dim;
        if (h < ht_n && j < qt_n) {
            const uint32_t ih = ih0 + h;
            q_sh[i] = q[size_t(iq0 + j)*q_stride_query_floats + size_t(ih)*q_stride_head_floats + d];
        } else {
            q_sh[i] = 0.0f;
        }
    }
    __syncthreads();

    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        float sum[HT][QT];
#pragma unroll
        for (int h = 0; h < HT; ++h) {
#pragma unroll
            for (int j = 0; j < QT; ++j) {
                sum[h][j] = 0.0f;
            }
        }

        if (t < n_sink) {
            const uint16_t * k = k_st + size_t(t)*sink_tail_stride_token_f16;
            for (uint32_t d = 0; d < head_dim; ++d) {
                const float kv = __half2float(reinterpret_cast<const __half *>(k)[d]);
#pragma unroll
                for (int h = 0; h < HT; ++h) {
#pragma unroll
                    for (int j = 0; j < QT; ++j) {
                        if (uint32_t(h) < ht_n && uint32_t(j) < qt_n) {
                            sum[h][j] += q_sh[(size_t(h)*QT + j)*head_dim + d]*kv;
                        }
                    }
                }
            }
        } else if (t < n_sink + n_body_tokens) {
            const uint32_t body_t = t - n_sink;
            const uint32_t r = body_t/group_size;
            const uint32_t g = body_t - r*group_size;
            for (uint32_t d = 0; d < head_dim; ++d) {
                float kv;
                if (k_body_f32_head != nullptr) {
                    kv = k_body_f32_head[size_t(body_t)*head_dim + d];
                } else {
                    const uint8_t * k_record = k_body_head + size_t(r)*k_body_stride_record_bytes;
                    const float * k_record_scales = k_scales_head + size_t(r)*k_scale_stride_record_floats;
                    const float * k_s_col = k_record_scales;
                    const float * k_zp    = k_record_scales + head_dim;
                    const float * k_s_row = k_record_scales + 2*head_dim;
                    const size_t i = size_t(d)*group_size + g;
                    const uint32_t kq = kvarn_unpack_one(k_record, key_bits, i);
                    kv = (float(kq)*k_s_col[d] + k_zp[d])*k_s_row[g];
                }
#pragma unroll
                for (int h = 0; h < HT; ++h) {
#pragma unroll
                    for (int j = 0; j < QT; ++j) {
                        if (uint32_t(h) < ht_n && uint32_t(j) < qt_n) {
                            sum[h][j] += q_sh[(size_t(h)*QT + j)*head_dim + d]*kv;
                        }
                    }
                }
            }
        } else if (t < n_sink + n_body_tokens + n_pending) {
            const uint32_t pending_t = t - n_sink - n_body_tokens;
            const float * k = pending_k_head + size_t(pending_t)*pending_stride_token_floats;
            for (uint32_t d = 0; d < head_dim; ++d) {
                const float kv = k[d];
#pragma unroll
                for (int h = 0; h < HT; ++h) {
#pragma unroll
                    for (int j = 0; j < QT; ++j) {
                        if (uint32_t(h) < ht_n && uint32_t(j) < qt_n) {
                            sum[h][j] += q_sh[(size_t(h)*QT + j)*head_dim + d]*kv;
                        }
                    }
                }
            }
        } else {
            const uint32_t tail_t = t - n_sink - n_body_tokens - n_pending;
            const uint32_t tail_slot = n_tail == 0 ? 0 : (tail_start + tail_t)%n_tail;
            const uint16_t * k = k_st + size_t(n_sink + tail_slot)*sink_tail_stride_token_f16;
            for (uint32_t d = 0; d < head_dim; ++d) {
                const float kv = __half2float(reinterpret_cast<const __half *>(k)[d]);
#pragma unroll
                for (int h = 0; h < HT; ++h) {
#pragma unroll
                    for (int j = 0; j < QT; ++j) {
                        if (uint32_t(h) < ht_n && uint32_t(j) < qt_n) {
                            sum[h][j] += q_sh[(size_t(h)*QT + j)*head_dim + d]*kv;
                        }
                    }
                }
            }
        }

        for (uint32_t h = 0; h < ht_n; ++h) {
            for (uint32_t j = 0; j < qt_n; ++j) {
                const void * mask_row = kq_mask_tile == nullptr ? nullptr :
                    (const void *) (kq_mask_tile + size_t(j)*kq_mask_stride_query_bytes);
                const int32_t causal_limit = kvarn_causal_mask_limit(n_tokens, n_queries, iq0 + j);
                probs[(size_t(h)*QT + j)*n_tokens + t] =
                    kvarn_softcap_attn_score(sum[h][j], scale, logit_softcap) +
                    kvarn_kq_mask_bias(mask_row, kq_mask_type, kq_mask_stride_token_bytes, t, causal_limit);
            }
        }
    }
    __syncthreads();

    for (uint32_t h = 0; h < uint32_t(HT); ++h) {
        for (uint32_t j = 0; j < uint32_t(QT); ++j) {
            float * row = probs + (size_t(h)*QT + j)*n_tokens;
            if (h >= ht_n || j >= qt_n) {
                for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
                    row[t] = 0.0f;
                }
                __syncthreads();
                continue;
            }

            float local_max = -3.4028234663852886e38f;
            for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
                local_max = fmaxf(local_max, row[t]);
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
                const float p = expf(row[t] - max_score);
                row[t] = p;
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
                row[t] *= inv_denom;
            }
            __syncthreads();
        }
    }

    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc[HT][QT];
#pragma unroll
        for (int h = 0; h < HT; ++h) {
#pragma unroll
            for (int j = 0; j < QT; ++j) {
                acc[h][j] = 0.0f;
            }
        }
        bool used_packed_k8v2_av = false;
        if (key_bits == 8 && value_bits == 2 && group_size == 128 &&
                (turbo_v_mode == 0 || turbo_v_mode == 3 || turbo_v_mode == 4 || turbo_v_mode == 5) &&
                v_body_f32_head == nullptr) {
            for (uint32_t t = 0; t < n_sink; ++t) {
                const uint16_t * v = v_st + size_t(t)*sink_tail_stride_token_f16;
                const float vv = __half2float(reinterpret_cast<const __half *>(v)[d]);
#pragma unroll
                for (int h = 0; h < HT; ++h) {
#pragma unroll
                    for (int j = 0; j < QT; ++j) {
                        acc[h][j] += probs[(size_t(h)*QT + j)*n_tokens + t]*vv;
                    }
                }
            }

            for (uint32_t r = 0; r < n_records; ++r) {
                const uint8_t * v_record = v_body_head + size_t(r)*v_body_stride_record_bytes;
                const float * v_record_scales = v_scales_head + size_t(r)*v_scale_stride_record_floats;
                for (uint32_t g = 0; g < group_size; ++g) {
                    const float vv = kvarn_turbo_v_dequant_rotated(
                            v_record, v_record_scales, head_dim, group_size,
                            value_bits, g, d, turbo_v_mode);
                    const uint32_t t = n_sink + r*group_size + g;
#pragma unroll
                    for (int h = 0; h < HT; ++h) {
#pragma unroll
                        for (int j = 0; j < QT; ++j) {
                            acc[h][j] += probs[(size_t(h)*QT + j)*n_tokens + t]*vv;
                        }
                    }
                }
            }

            for (uint32_t p = 0; p < n_pending; ++p) {
                const float vv = pending_v_head[size_t(p)*pending_stride_token_floats + d];
                const uint32_t t = n_sink + n_body_tokens + p;
#pragma unroll
                for (int h = 0; h < HT; ++h) {
#pragma unroll
                    for (int j = 0; j < QT; ++j) {
                        acc[h][j] += probs[(size_t(h)*QT + j)*n_tokens + t]*vv;
                    }
                }
            }

            for (uint32_t tail_t = 0; tail_t < n_tail; ++tail_t) {
                const uint32_t tail_slot = (tail_start + tail_t)%n_tail;
                const uint16_t * v = v_st + size_t(n_sink + tail_slot)*sink_tail_stride_token_f16;
                const float vv = __half2float(reinterpret_cast<const __half *>(v)[d]);
                const uint32_t t = n_sink + n_body_tokens + n_pending + tail_t;
#pragma unroll
                for (int h = 0; h < HT; ++h) {
#pragma unroll
                    for (int j = 0; j < QT; ++j) {
                        acc[h][j] += probs[(size_t(h)*QT + j)*n_tokens + t]*vv;
                    }
                }
            }
            used_packed_k8v2_av = true;
        }

        if (!used_packed_k8v2_av) {
            for (uint32_t t = 0; t < n_tokens; ++t) {
                float v;
                if (v_body_f32_head != nullptr && t >= n_sink && t < n_sink + n_body_tokens) {
                    const uint32_t body_t = t - n_sink;
                    v = v_body_f32_head[size_t(body_t)*head_dim + d];
                } else {
                    v = kvarn_mixed_f16_load_v(
                            v_st, v_body_head, v_scales_head, pending_v_head,
                            t, d, n_sink, n_records, n_pending, n_tail, tail_start,
                            head_dim, group_size, value_bits,
                            sink_tail_stride_token_f16, pending_stride_token_floats,
                            v_body_stride_record_bytes, v_scale_stride_record_floats, turbo_v_mode);
                }
#pragma unroll
                for (int h = 0; h < HT; ++h) {
#pragma unroll
                    for (int j = 0; j < QT; ++j) {
                        acc[h][j] += probs[(size_t(h)*QT + j)*n_tokens + t]*v;
                    }
                }
            }
        }
        for (uint32_t h = 0; h < ht_n; ++h) {
            const uint32_t ih = ih0 + h;
            for (uint32_t j = 0; j < qt_n; ++j) {
                out[size_t(iq0 + j)*out_stride_query_floats + size_t(ih)*out_stride_head_floats + d] = acc[h][j];
            }
        }
    }
}

template<int QT, int HT>
static __global__ void kvarn_attn_mixed_f16_fused_batch_sparse_d512_scalar_qt_gqa_kernel(
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
        const float * __restrict__ body_k_f32,
        const float * __restrict__ body_v_f32,
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
        size_t body_f32_stride_head_elems,
        float logit_softcap,
        uint32_t turbo_v_mode) {
    (void) scores;
    if (head_dim != 512u || group_size != 128u || value_bits != 2u ||
            turbo_v_mode != 6u || body_v_f32 != nullptr) {
        return;
    }
    const uint32_t n_gqa = n_head/n_head_kv;
    const uint32_t h_tiles_per_kv = (n_gqa + HT - 1)/uint32_t(HT);
    const uint32_t n_q_tiles = (n_queries + QT - 1)/uint32_t(QT);
    const uint32_t n_head_tiles = n_head_kv*h_tiles_per_kv;
    if (blockIdx.x >= n_q_tiles*n_head_tiles) {
        return;
    }

    const uint32_t q_tile = blockIdx.x/n_head_tiles;
    const uint32_t head_tile = blockIdx.x - q_tile*n_head_tiles;
    const uint32_t ikh = head_tile/h_tiles_per_kv;
    const uint32_t h0_in_group = (head_tile - ikh*h_tiles_per_kv)*uint32_t(HT);
    const uint32_t iq0 = q_tile*uint32_t(QT);
    const uint32_t qt_n = min(uint32_t(QT), n_queries - iq0);
    const uint32_t ht_n = min(uint32_t(HT), n_gqa - h0_in_group);
    const uint32_t ih0 = ikh*n_gqa + h0_in_group;

    const uint16_t * k_st = sink_tail_k + size_t(ikh)*sink_tail_stride_head_f16;
    const uint16_t * v_st = sink_tail_v + size_t(ikh)*sink_tail_stride_head_f16;
    const uint8_t * k_body_head = k_body + size_t(ikh)*k_body_stride_head_bytes;
    const uint8_t * v_body_head = v_body + size_t(ikh)*v_body_stride_head_bytes;
    const float * k_scales_head = k_scales + size_t(ikh)*k_scale_stride_head_floats;
    const float * v_scales_head = v_scales + size_t(ikh)*v_scale_stride_head_floats;
    const float * pending_k_head = pending_k + size_t(ikh)*pending_stride_head_floats;
    const float * pending_v_head = pending_v + size_t(ikh)*pending_stride_head_floats;
    const float * k_body_f32_head = body_k_f32 == nullptr ? nullptr : body_k_f32 + size_t(ikh)*body_f32_stride_head_elems;
    const float * v_body_f32_head = body_v_f32 == nullptr ? nullptr : body_v_f32 + size_t(ikh)*body_f32_stride_head_elems;
    const char * kq_mask_tile = kq_mask == nullptr ? nullptr :
        (const char *) kq_mask + size_t(iq0)*kq_mask_stride_query_bytes;

    const uint32_t n_body_tokens = n_records*group_size;
    const uint32_t n_tokens = n_sink + n_body_tokens + n_pending + n_tail;

    extern __shared__ float shared[];
    float * q_sh = shared;
    float * probs = shared + size_t(HT)*size_t(QT)*head_dim;
    float * reduce = probs + size_t(HT)*size_t(QT)*n_tokens;

    for (uint32_t i = threadIdx.x; i < uint32_t(HT)*uint32_t(QT)*head_dim; i += blockDim.x) {
        const uint32_t h = i/(uint32_t(QT)*head_dim);
        const uint32_t rem = i - h*uint32_t(QT)*head_dim;
        const uint32_t j = rem/head_dim;
        const uint32_t d = rem - j*head_dim;
        if (h < ht_n && j < qt_n) {
            const uint32_t ih = ih0 + h;
            q_sh[i] = q[size_t(iq0 + j)*q_stride_query_floats + size_t(ih)*q_stride_head_floats + d];
        } else {
            q_sh[i] = 0.0f;
        }
    }
    __syncthreads();

    for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
        float sum[HT][QT];
#pragma unroll
        for (int h = 0; h < HT; ++h) {
#pragma unroll
            for (int j = 0; j < QT; ++j) {
                sum[h][j] = 0.0f;
            }
        }

        if (t < n_sink) {
            const uint16_t * k = k_st + size_t(t)*sink_tail_stride_token_f16;
            for (uint32_t d = 0; d < head_dim; ++d) {
                const float kv = __half2float(reinterpret_cast<const __half *>(k)[d]);
#pragma unroll
                for (int h = 0; h < HT; ++h) {
#pragma unroll
                    for (int j = 0; j < QT; ++j) {
                        if (uint32_t(h) < ht_n && uint32_t(j) < qt_n) {
                            sum[h][j] += q_sh[(size_t(h)*QT + j)*head_dim + d]*kv;
                        }
                    }
                }
            }
        } else if (t < n_sink + n_body_tokens) {
            const uint32_t body_t = t - n_sink;
            const uint32_t r = body_t/group_size;
            const uint32_t g = body_t - r*group_size;
            for (uint32_t d = 0; d < head_dim; ++d) {
                float kv;
                if (k_body_f32_head != nullptr) {
                    kv = k_body_f32_head[size_t(body_t)*head_dim + d];
                } else {
                    const uint8_t * k_record = k_body_head + size_t(r)*k_body_stride_record_bytes;
                    const float * k_record_scales = k_scales_head + size_t(r)*k_scale_stride_record_floats;
                    const float * k_s_col = k_record_scales;
                    const float * k_zp    = k_record_scales + head_dim;
                    const float * k_s_row = k_record_scales + 2*head_dim;
                    const size_t i = size_t(d)*group_size + g;
                    const uint32_t kq = kvarn_unpack_one(k_record, key_bits, i);
                    kv = (float(kq)*k_s_col[d] + k_zp[d])*k_s_row[g];
                }
#pragma unroll
                for (int h = 0; h < HT; ++h) {
#pragma unroll
                    for (int j = 0; j < QT; ++j) {
                        if (uint32_t(h) < ht_n && uint32_t(j) < qt_n) {
                            sum[h][j] += q_sh[(size_t(h)*QT + j)*head_dim + d]*kv;
                        }
                    }
                }
            }
        } else if (t < n_sink + n_body_tokens + n_pending) {
            const uint32_t pending_t = t - n_sink - n_body_tokens;
            const float * k = pending_k_head + size_t(pending_t)*pending_stride_token_floats;
            for (uint32_t d = 0; d < head_dim; ++d) {
                const float kv = k[d];
#pragma unroll
                for (int h = 0; h < HT; ++h) {
#pragma unroll
                    for (int j = 0; j < QT; ++j) {
                        if (uint32_t(h) < ht_n && uint32_t(j) < qt_n) {
                            sum[h][j] += q_sh[(size_t(h)*QT + j)*head_dim + d]*kv;
                        }
                    }
                }
            }
        } else {
            const uint32_t tail_t = t - n_sink - n_body_tokens - n_pending;
            const uint32_t tail_slot = n_tail == 0 ? 0 : (tail_start + tail_t)%n_tail;
            const uint16_t * k = k_st + size_t(n_sink + tail_slot)*sink_tail_stride_token_f16;
            for (uint32_t d = 0; d < head_dim; ++d) {
                const float kv = __half2float(reinterpret_cast<const __half *>(k)[d]);
#pragma unroll
                for (int h = 0; h < HT; ++h) {
#pragma unroll
                    for (int j = 0; j < QT; ++j) {
                        if (uint32_t(h) < ht_n && uint32_t(j) < qt_n) {
                            sum[h][j] += q_sh[(size_t(h)*QT + j)*head_dim + d]*kv;
                        }
                    }
                }
            }
        }

        for (uint32_t h = 0; h < ht_n; ++h) {
            for (uint32_t j = 0; j < qt_n; ++j) {
                const void * mask_row = kq_mask_tile == nullptr ? nullptr :
                    (const void *) (kq_mask_tile + size_t(j)*kq_mask_stride_query_bytes);
                const int32_t causal_limit = kvarn_causal_mask_limit(n_tokens, n_queries, iq0 + j);
                probs[(size_t(h)*QT + j)*n_tokens + t] =
                    kvarn_softcap_attn_score(sum[h][j], scale, logit_softcap) +
                    kvarn_kq_mask_bias(mask_row, kq_mask_type, kq_mask_stride_token_bytes, t, causal_limit);
            }
        }
    }
    __syncthreads();

    for (uint32_t h = 0; h < uint32_t(HT); ++h) {
        for (uint32_t j = 0; j < uint32_t(QT); ++j) {
            float * row = probs + (size_t(h)*QT + j)*n_tokens;
            if (h >= ht_n || j >= qt_n) {
                for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
                    row[t] = 0.0f;
                }
                __syncthreads();
                continue;
            }

            float local_max = -3.4028234663852886e38f;
            for (uint32_t t = threadIdx.x; t < n_tokens; t += blockDim.x) {
                local_max = fmaxf(local_max, row[t]);
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
                const float p = expf(row[t] - max_score);
                row[t] = p;
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
                row[t] *= inv_denom;
            }
            __syncthreads();
        }
    }

    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc[HT][QT];
#pragma unroll
        for (int h = 0; h < HT; ++h) {
#pragma unroll
            for (int j = 0; j < QT; ++j) {
                acc[h][j] = 0.0f;
            }
        }
        {
            for (uint32_t t = 0; t < n_sink; ++t) {
                const uint16_t * v = v_st + size_t(t)*sink_tail_stride_token_f16;
                const float vv = __half2float(reinterpret_cast<const __half *>(v)[d]);
#pragma unroll
                for (int h = 0; h < HT; ++h) {
#pragma unroll
                    for (int j = 0; j < QT; ++j) {
                        acc[h][j] += probs[(size_t(h)*QT + j)*n_tokens + t]*vv;
                    }
                }
            }

            for (uint32_t r = 0; r < n_records; ++r) {
                const uint8_t * v_record = v_body_head + size_t(r)*v_body_stride_record_bytes;
                const float * v_record_scales = v_scales_head + size_t(r)*v_scale_stride_record_floats;
                for (uint32_t g = 0; g < group_size; ++g) {
                    const float vv = kvarn_sparse_d512_v_base(v_record, v_record_scales, g, d);
                    const uint32_t t = n_sink + r*group_size + g;
#pragma unroll
                    for (int h = 0; h < HT; ++h) {
#pragma unroll
                        for (int j = 0; j < QT; ++j) {
                            acc[h][j] += probs[(size_t(h)*QT + j)*n_tokens + t]*vv;
                        }
                    }
                }

                // The sparse correction stays column-owned: each output lane
                // consumes its CSR column after the dense asymmetric V2 loop.
#pragma unroll
                for (int h = 0; h < HT; ++h) {
#pragma unroll
                    for (int j = 0; j < QT; ++j) {
                        acc[h][j] += kvarn_sparse_d512_weighted_column(
                                v_record,
                                probs + (size_t(h)*QT + j)*n_tokens + n_sink + size_t(r)*group_size,
                                d);
                    }
                }
            }

            for (uint32_t p = 0; p < n_pending; ++p) {
                const float vv = pending_v_head[size_t(p)*pending_stride_token_floats + d];
                const uint32_t t = n_sink + n_body_tokens + p;
#pragma unroll
                for (int h = 0; h < HT; ++h) {
#pragma unroll
                    for (int j = 0; j < QT; ++j) {
                        acc[h][j] += probs[(size_t(h)*QT + j)*n_tokens + t]*vv;
                    }
                }
            }

            for (uint32_t tail_t = 0; tail_t < n_tail; ++tail_t) {
                const uint32_t tail_slot = (tail_start + tail_t)%n_tail;
                const uint16_t * v = v_st + size_t(n_sink + tail_slot)*sink_tail_stride_token_f16;
                const float vv = __half2float(reinterpret_cast<const __half *>(v)[d]);
                const uint32_t t = n_sink + n_body_tokens + n_pending + tail_t;
#pragma unroll
                for (int h = 0; h < HT; ++h) {
#pragma unroll
                    for (int j = 0; j < QT; ++j) {
                        acc[h][j] += probs[(size_t(h)*QT + j)*n_tokens + t]*vv;
                    }
                }
            }
        }

        for (uint32_t h = 0; h < ht_n; ++h) {
            const uint32_t ih = ih0 + h;
            for (uint32_t j = 0; j < qt_n; ++j) {
                out[size_t(iq0 + j)*out_stride_query_floats + size_t(ih)*out_stride_head_floats + d] = acc[h][j];
            }
        }
    }
}

void ggml_cuda_kvarn_attn_mixed_f16_batch(
        const float * q,
        const float * q_body,
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
        size_t q_body_stride_head_floats,
        size_t q_body_stride_query_floats,
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
        void * stream,
        const int32_t * window_dev,
        int64_t scores_nelems,
        int64_t k_body_records_cap,
        const void * raw_body_store_key,
        uint32_t frame_flags,
        float logit_softcap,
        uint32_t turbo_v_mode) {
    kvarn_validate_v_mode(head_dim, group_size, value_bits, turbo_v_mode);
    const uint32_t n_tokens = n_sink + n_records*group_size + n_pending + n_tail;
    const uint32_t n_gqa = n_head/n_head_kv;
    (void) kvarn_env_flag("LLAMA_KVARN_ATTN_FUSED_BATCH");
    (void) kvarn_dequant_cache_trace_enabled();
    (void) kvarn_dequant_cache_trace_limit();
    const bool mixed_frame = q_body != nullptr;
    const bool force_serial_fused = kvarn_env_flag("LLAMA_KVARN_ATTN_SERIAL_FUSED");
    const bool use_raw_body_k = kvarn_debug_raw_body_k_enabled();
    const bool use_raw_body_v = kvarn_debug_raw_body_v_enabled();
    const bool use_raw_body_scalar_qt = kvarn_debug_raw_body_scalar_qt_enabled() && (use_raw_body_k || use_raw_body_v);
    const bool use_turbo2_body_inverse = turbo_v_mode == 2u && n_records != 0 && !use_raw_body_v;
    const bool use_split_kernels =
        mixed_frame ||
        use_turbo2_body_inverse ||
        ((use_raw_body_k || use_raw_body_v) && !use_raw_body_scalar_qt) ||
        kvarn_env_flag("LLAMA_KVARN_ATTN_SPLIT_KERNELS");
    const bool use_serial_fused = force_serial_fused && !use_split_kernels;
    const bool fused_paper_frame = (frame_flags & KVARN_ATTN_FRAME_FUSED_PAPER_FULL) != 0;
    if ((frame_flags & ~KVARN_ATTN_FRAME_FUSED_PAPER_FULL) != 0) {
        std::fprintf(stderr, "KVarN mixed attention received unsupported frame flags\n");
        std::abort();
    }
    if (turbo_v_mode == 6u) {
        if (mixed_frame || use_raw_body_k || use_raw_body_v) {
            std::fprintf(stderr,
                    "KVarN sparse-D512 attention does not support mixed-frame or raw-body diagnostics\n");
            std::abort();
        }
    }

    int block = 1;
    while (block < int(n_tokens)) {
        block <<= 1;
    }
    block = block > 256 ? 256 : block;

    const size_t shmem = (size_t(n_tokens) + size_t(block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
    const size_t reduce_shmem = size_t(block)*sizeof(float);
    const int av_block = 128;
    const int av_grid = int((head_dim + av_block - 1)/av_block);
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    // Per-head serial launches add n_head kernel launch overheads and regressed
    // Gemma tg64 (~56% vs 71% pre-refinement). Opt-in via env for A/B only.
    const bool use_decode_per_head = kvarn_env_flag("LLAMA_KVARN_ATTN_DECODE_PER_HEAD");

    if (fused_paper_frame && (use_split_kernels || use_serial_fused || n_records != 0 || n_pending != 0)) {
        std::fprintf(stderr, "KVarN fused paper-frame FWHT is only implemented for pure sink/tail attention\n");
        std::abort();
    }

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
                        q_ptr, nullptr, k_st_ptr, v_st_ptr, k_body_ptr, v_body_ptr, k_scales_ptr, v_scales_ptr,
                        pending_k_ptr, pending_v_ptr, kq_mask_ptr, out_ptr, scores,
                        n_sink, n_records, n_pending, n_tail, tail_start, head_dim, group_size, key_bits, value_bits,
                        sink_tail_stride_token_f16, pending_stride_token_floats,
                        k_body_stride_record_bytes, v_body_stride_record_bytes,
                        k_scale_stride_record_floats, v_scale_stride_record_floats,
                        kq_mask_stride_token_bytes, kq_mask_type,
                        int32_t(n_tokens) - 1, scale, logit_softcap, turbo_v_mode);
            }
            return;
        }

        // Short decode often has no quantized body records yet
        // (n_records=0, n_pending=0). The generic mixed path pays branch and
        // packed-body address overhead even when all K/V are plain f16
        // sink/tail entries. Use one CTA per query head for decode across all
        // supported head dimensions; keep the batch sink/tail specialization
        // scoped to the 512d path where it is logits-equivalent to split.
        if (n_records == 0 && n_pending == 0) {
            if (n_queries == 1) {
                // One CTA per head spreads decode attention across SMs.
                // shared = q_sh[head_dim] + probs[n_tokens] + reduce[block]
                const int decode_block = head_dim <= 256 ? 256 : 128;
                const size_t decode_shmem = (size_t(head_dim) + size_t(n_tokens) + size_t(decode_block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
                if (kvarn_cuda_dynamic_shmem_fits(decode_shmem)) {
                    kvarn_attn_mixed_f16_sinktail_decode_kernel<<<int(n_head), decode_block, decode_shmem, cuda_stream>>>(
                            q, sink_tail_k, sink_tail_v, kq_mask, out,
                            n_head, n_head_kv,
                            n_sink, n_tail, tail_start, head_dim,
                            q_stride_head_floats, out_stride_head_floats,
                            sink_tail_stride_head_f16, sink_tail_stride_token_f16,
                            kq_mask_stride_token_bytes, kq_mask_type, scale,
                            fused_paper_frame,
                            window_dev, logit_softcap);
                    return;
                }
            }

            if (head_dim >= 512) {
                const int sinktail_block = n_tokens <= 128 ? 256 : 512;
                const size_t sinktail_shmem =
                    (size_t(fused_paper_frame ? head_dim : 0) +
                     size_t(n_tokens) + size_t(sinktail_block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
                if (kvarn_cuda_dynamic_shmem_fits(sinktail_shmem)) {
                    kvarn_attn_mixed_f16_sinktail_batch_kernel<<<int(n_queries*n_head), sinktail_block, sinktail_shmem, cuda_stream>>>(
                            q, sink_tail_k, sink_tail_v, kq_mask, out,
                            n_queries, n_head, n_head_kv,
                            n_sink, n_tail, tail_start, head_dim,
                            q_stride_head_floats, q_stride_query_floats,
                            out_stride_head_floats, out_stride_query_floats,
                            sink_tail_stride_head_f16, sink_tail_stride_token_f16,
                            kq_mask_stride_query_bytes, kq_mask_stride_token_bytes,
                            kq_mask_type, scale, logit_softcap, fused_paper_frame);
                    return;
                }
            }

            if (fused_paper_frame) {
                std::fprintf(stderr, "KVarN fused paper-frame FWHT sink/tail shared-memory requirement was not met\n");
                std::abort();
            }
        }

        if (turbo_v_mode == 6u && n_records > 0 && n_queries == 1 && n_gqa >= 2) {
            constexpr int sparse_q1_ht = 2;
            constexpr int sparse_q1_block = 256;
            const size_t sparse_q1_shmem =
                (size_t(sparse_q1_ht)*head_dim + size_t(sparse_q1_ht)*n_tokens +
                 size_t(sparse_q1_block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
            if (!kvarn_cuda_dynamic_shmem_optin_fits(sparse_q1_shmem) ||
                    !kvarn_cuda_prepare_dynamic_shmem(
                        kvarn_attn_mixed_f16_fused_batch_sparse_d512_scalar_qt_gqa_kernel<1, sparse_q1_ht>,
                        sparse_q1_shmem)) {
                std::fprintf(stderr,
                        "KVarN sparse-D512 scalar-q1-GQA shared-memory requirement was not met\n");
                std::abort();
            }
            const int sparse_q1_grid = int(
                    n_head_kv*((n_gqa + uint32_t(sparse_q1_ht - 1))/uint32_t(sparse_q1_ht)));
            if (kvarn_attn_trace_claim()) {
                std::fprintf(stderr,
                        "KVarN CUDA mixed-attn inner trace: mode=sparse-d512-scalar-q1-gqa"
                        " head_dim=%u n_queries=%u n_head=%u n_head_kv=%u n_gqa=%u"
                        " sink=%u records=%u pending=%u tail=%u tokens=%u qt=1 ht=%d grid=%d shmem=%zu\n",
                        head_dim, n_queries, n_head, n_head_kv, n_gqa,
                        n_sink, n_records, n_pending, n_tail, n_tokens,
                        sparse_q1_ht, sparse_q1_grid, sparse_q1_shmem);
            }
            kvarn_attn_mixed_f16_fused_batch_sparse_d512_scalar_qt_gqa_kernel<1, sparse_q1_ht>
                <<<sparse_q1_grid, sparse_q1_block, sparse_q1_shmem, cuda_stream>>>(
                    q, sink_tail_k, sink_tail_v, k_body, v_body, k_scales, v_scales,
                    pending_k, pending_v, kq_mask,
                    nullptr, nullptr,
                    out, scores, n_queries, n_head, n_head_kv,
                    n_sink, n_records, n_pending, n_tail, tail_start,
                    head_dim, group_size, key_bits, value_bits,
                    q_stride_head_floats, q_stride_query_floats,
                    out_stride_head_floats, out_stride_query_floats,
                    sink_tail_stride_head_f16, sink_tail_stride_token_f16,
                    pending_stride_head_floats, pending_stride_token_floats,
                    k_body_stride_record_bytes, v_body_stride_record_bytes,
                    k_body_stride_head_bytes, v_body_stride_head_bytes,
                    k_scale_stride_record_floats, v_scale_stride_record_floats,
                    k_scale_stride_head_floats, v_scale_stride_head_floats,
                    kq_mask_stride_query_bytes, kq_mask_stride_token_bytes,
                    kq_mask_type, scale, 0, logit_softcap, turbo_v_mode);
            return;
        }

        // Body-active decode previously fell through to the generic
        // one-query-head-per-CTA kernel.  That repeats packed K/V unpack and
        // dual-scale recomposition for every GQA head.  Reuse the exact
        // scalar QT-GQA kernel with QT=1 so one packed KV load serves HT
        // query heads.  Passing null mirrors guarantees this remains the
        // real compressed-cache path.
        const bool q1_gqa_shape =
            (head_dim == 256 || head_dim == 512) &&
            n_records > 0 && n_queries == 1 && n_gqa >= 2;
        const bool q1_gqa_disabled = kvarn_env_flag("LLAMA_KVARN_ATTN_DISABLE_Q1_GQA_SCALAR");
        const bool q1_gqa_required = kvarn_env_flag("LLAMA_KVARN_ATTN_REQUIRE_Q1_GQA_SCALAR");
        if (q1_gqa_shape && !q1_gqa_disabled) {
            constexpr int q1_block = 256;

            if (head_dim == 256 && n_gqa >= 8) {
                constexpr int ht = 2;
                const size_t q1_shmem =
                    (size_t(ht)*head_dim + size_t(ht)*n_tokens +
                     size_t(q1_block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
                if (kvarn_cuda_dynamic_shmem_optin_fits(q1_shmem) &&
                        kvarn_cuda_prepare_dynamic_shmem(
                            kvarn_attn_mixed_f16_fused_batch_scalar_qt_gqa_kernel<1, ht>, q1_shmem)) {
                    const int grid = int(n_head_kv*((n_gqa + uint32_t(ht - 1))/uint32_t(ht)));
                    if (kvarn_attn_trace_claim()) {
                        std::fprintf(stderr,
                                "KVarN CUDA mixed-attn inner trace: mode=scalar-q1-gqa"
                                " head_dim=%u n_queries=%u n_head=%u n_head_kv=%u n_gqa=%u"
                                " sink=%u records=%u pending=%u tail=%u tail_start=%u"
                                " tokens=%u qt=1 ht=%d block=%d grid=%d shmem=%zu scores_nelems=%" PRId64
                                " body_records_cap=%" PRId64 " body_mirror_allowed=0 body_mirror_used=0"
                                " kq_mask_type=%u kq_mask_stride_query_bytes=%zu kq_mask_stride_token_bytes=%zu\n",
                                head_dim, n_queries, n_head, n_head_kv, n_gqa,
                                n_sink, n_records, n_pending, n_tail, tail_start, n_tokens,
                                ht, q1_block, grid, q1_shmem, scores_nelems, k_body_records_cap,
                                kq_mask_type, kq_mask_stride_query_bytes, kq_mask_stride_token_bytes);
                    }
                    kvarn_attn_mixed_f16_fused_batch_scalar_qt_gqa_kernel<1, ht>
                        <<<grid, q1_block, q1_shmem, cuda_stream>>>(
                            q, sink_tail_k, sink_tail_v, k_body, v_body, k_scales, v_scales,
                            pending_k, pending_v, kq_mask,
                            nullptr, nullptr,
                            out, scores, n_queries, n_head, n_head_kv,
                            n_sink, n_records, n_pending, n_tail, tail_start,
                            head_dim, group_size, key_bits, value_bits,
                            q_stride_head_floats, q_stride_query_floats,
                            out_stride_head_floats, out_stride_query_floats,
                            sink_tail_stride_head_f16, sink_tail_stride_token_f16,
                            pending_stride_head_floats, pending_stride_token_floats,
                            k_body_stride_record_bytes, v_body_stride_record_bytes,
                            k_body_stride_head_bytes, v_body_stride_head_bytes,
                            k_scale_stride_record_floats, v_scale_stride_record_floats,
                            k_scale_stride_head_floats, v_scale_stride_head_floats,
                            kq_mask_stride_query_bytes, kq_mask_stride_token_bytes,
                            kq_mask_type, scale, 0, logit_softcap, turbo_v_mode);
                    return;
                }
            }

            if (n_gqa >= 4) {
                constexpr int ht = 2;
                const size_t q1_shmem =
                    (size_t(ht)*head_dim + size_t(ht)*n_tokens +
                     size_t(q1_block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
                if (kvarn_cuda_dynamic_shmem_optin_fits(q1_shmem) &&
                        kvarn_cuda_prepare_dynamic_shmem(
                            kvarn_attn_mixed_f16_fused_batch_scalar_qt_gqa_kernel<1, ht>, q1_shmem)) {
                    const int grid = int(n_head_kv*((n_gqa + uint32_t(ht - 1))/uint32_t(ht)));
                    if (kvarn_attn_trace_claim()) {
                        std::fprintf(stderr,
                                "KVarN CUDA mixed-attn inner trace: mode=scalar-q1-gqa"
                                " head_dim=%u n_queries=%u n_head=%u n_head_kv=%u n_gqa=%u"
                                " sink=%u records=%u pending=%u tail=%u tail_start=%u"
                                " tokens=%u qt=1 ht=%d block=%d grid=%d shmem=%zu scores_nelems=%" PRId64
                                " body_records_cap=%" PRId64 " body_mirror_allowed=0 body_mirror_used=0"
                                " kq_mask_type=%u kq_mask_stride_query_bytes=%zu kq_mask_stride_token_bytes=%zu\n",
                                head_dim, n_queries, n_head, n_head_kv, n_gqa,
                                n_sink, n_records, n_pending, n_tail, tail_start, n_tokens,
                                ht, q1_block, grid, q1_shmem, scores_nelems, k_body_records_cap,
                                kq_mask_type, kq_mask_stride_query_bytes, kq_mask_stride_token_bytes);
                    }
                    kvarn_attn_mixed_f16_fused_batch_scalar_qt_gqa_kernel<1, ht>
                        <<<grid, q1_block, q1_shmem, cuda_stream>>>(
                            q, sink_tail_k, sink_tail_v, k_body, v_body, k_scales, v_scales,
                            pending_k, pending_v, kq_mask,
                            nullptr, nullptr,
                            out, scores, n_queries, n_head, n_head_kv,
                            n_sink, n_records, n_pending, n_tail, tail_start,
                            head_dim, group_size, key_bits, value_bits,
                            q_stride_head_floats, q_stride_query_floats,
                            out_stride_head_floats, out_stride_query_floats,
                            sink_tail_stride_head_f16, sink_tail_stride_token_f16,
                            pending_stride_head_floats, pending_stride_token_floats,
                            k_body_stride_record_bytes, v_body_stride_record_bytes,
                            k_body_stride_head_bytes, v_body_stride_head_bytes,
                            k_scale_stride_record_floats, v_scale_stride_record_floats,
                            k_scale_stride_head_floats, v_scale_stride_head_floats,
                            kq_mask_stride_query_bytes, kq_mask_stride_token_bytes,
                            kq_mask_type, scale, 0, logit_softcap, turbo_v_mode);
                    return;
                }
            }

            {
                constexpr int ht = 2;
                const size_t q1_shmem =
                    (size_t(ht)*head_dim + size_t(ht)*n_tokens +
                     size_t(q1_block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
                if (kvarn_cuda_dynamic_shmem_optin_fits(q1_shmem) &&
                        kvarn_cuda_prepare_dynamic_shmem(
                            kvarn_attn_mixed_f16_fused_batch_scalar_qt_gqa_kernel<1, ht>, q1_shmem)) {
                    const int grid = int(n_head_kv*((n_gqa + uint32_t(ht - 1))/uint32_t(ht)));
                    if (kvarn_attn_trace_claim()) {
                        std::fprintf(stderr,
                                "KVarN CUDA mixed-attn inner trace: mode=scalar-q1-gqa"
                                " head_dim=%u n_queries=%u n_head=%u n_head_kv=%u n_gqa=%u"
                                " sink=%u records=%u pending=%u tail=%u tail_start=%u"
                                " tokens=%u qt=1 ht=%d block=%d grid=%d shmem=%zu scores_nelems=%" PRId64
                                " body_records_cap=%" PRId64 " body_mirror_allowed=0 body_mirror_used=0"
                                " kq_mask_type=%u kq_mask_stride_query_bytes=%zu kq_mask_stride_token_bytes=%zu\n",
                                head_dim, n_queries, n_head, n_head_kv, n_gqa,
                                n_sink, n_records, n_pending, n_tail, tail_start, n_tokens,
                                ht, q1_block, grid, q1_shmem, scores_nelems, k_body_records_cap,
                                kq_mask_type, kq_mask_stride_query_bytes, kq_mask_stride_token_bytes);
                    }
                    kvarn_attn_mixed_f16_fused_batch_scalar_qt_gqa_kernel<1, ht>
                        <<<grid, q1_block, q1_shmem, cuda_stream>>>(
                            q, sink_tail_k, sink_tail_v, k_body, v_body, k_scales, v_scales,
                            pending_k, pending_v, kq_mask,
                            nullptr, nullptr,
                            out, scores, n_queries, n_head, n_head_kv,
                            n_sink, n_records, n_pending, n_tail, tail_start,
                            head_dim, group_size, key_bits, value_bits,
                            q_stride_head_floats, q_stride_query_floats,
                            out_stride_head_floats, out_stride_query_floats,
                            sink_tail_stride_head_f16, sink_tail_stride_token_f16,
                            pending_stride_head_floats, pending_stride_token_floats,
                            k_body_stride_record_bytes, v_body_stride_record_bytes,
                            k_body_stride_head_bytes, v_body_stride_head_bytes,
                            k_scale_stride_record_floats, v_scale_stride_record_floats,
                            k_scale_stride_head_floats, v_scale_stride_head_floats,
                            kq_mask_stride_query_bytes, kq_mask_stride_token_bytes,
                            kq_mask_type, scale, 0, logit_softcap, turbo_v_mode);
                    return;
                }
            }
        }
        if (q1_gqa_shape && q1_gqa_required) {
            std::fprintf(stderr,
                    "KVarN CUDA mixed-attn q1-GQA scalar path required but unavailable: "
                    "head_dim=%u n_queries=%u n_head=%u n_head_kv=%u n_gqa=%u "
                    "sink=%u records=%u pending=%u tail=%u tokens=%u disabled=%d\n",
                    head_dim, n_queries, n_head, n_head_kv, n_gqa,
                    n_sink, n_records, n_pending, n_tail, n_tokens, q1_gqa_disabled ? 1 : 0);
            std::abort();
        }

        const int sparse_mode6_qt_override = turbo_v_mode == 6u ? kvarn_env_qt_override() : 0;
        if (turbo_v_mode == 6u && n_records > 0 && n_queries > 1 && n_gqa >= 2) {
            constexpr int sparse_block = 256;
            const int sparse_qt_override = sparse_mode6_qt_override;
            const bool sparse_ht4_preferred = n_gqa >= 4 &&
                (sparse_qt_override == 0 || sparse_qt_override == 4) &&
                kvarn_env_flag("LLAMA_KVARN_ATTN_ENABLE_512D_GQA_SCALAR_QT_HT4");
            const bool sparse_q8_preferred = sparse_qt_override != 4 &&
                ((sparse_qt_override == 8 && n_queries > 4) ||
                 (sparse_qt_override == 0 && n_queries > 4));
            int sparse_qt = 0;
            int sparse_ht = 0;
            size_t sparse_shmem = 0;
            const auto sparse_shmem_for = [&](int qt, int ht) {
                return (size_t(qt)*size_t(ht)*head_dim +
                        size_t(qt)*size_t(ht)*n_tokens +
                        size_t(sparse_block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
            };
            if (sparse_ht4_preferred) {
                const size_t candidate_shmem = sparse_shmem_for(4, 4);
                if (kvarn_cuda_dynamic_shmem_optin_fits(candidate_shmem) &&
                        kvarn_cuda_prepare_dynamic_shmem(
                            kvarn_attn_mixed_f16_fused_batch_sparse_d512_scalar_qt_gqa_kernel<4, 4>,
                            candidate_shmem)) {
                    sparse_qt = 4;
                    sparse_ht = 4;
                    sparse_shmem = candidate_shmem;
                }
            }
            if (sparse_qt == 0 && sparse_q8_preferred) {
                const size_t candidate_shmem = sparse_shmem_for(8, 2);
                if (kvarn_cuda_dynamic_shmem_optin_fits(candidate_shmem) &&
                        kvarn_cuda_prepare_dynamic_shmem(
                            kvarn_attn_mixed_f16_fused_batch_sparse_d512_scalar_qt_gqa_kernel<8, 2>,
                            candidate_shmem)) {
                    sparse_qt = 8;
                    sparse_ht = 2;
                    sparse_shmem = candidate_shmem;
                }
            }
            if (sparse_qt == 0 && (sparse_qt_override != 8 || n_queries <= 4)) {
                const size_t candidate_shmem = sparse_shmem_for(4, 2);
                if (kvarn_cuda_dynamic_shmem_optin_fits(candidate_shmem) &&
                        kvarn_cuda_prepare_dynamic_shmem(
                            kvarn_attn_mixed_f16_fused_batch_sparse_d512_scalar_qt_gqa_kernel<4, 2>,
                            candidate_shmem)) {
                    sparse_qt = 4;
                    sparse_ht = 2;
                    sparse_shmem = candidate_shmem;
                }
            }
            if (sparse_qt == 0 && (sparse_qt_override != 8 || n_queries <= 4)) {
                const size_t candidate_shmem = sparse_shmem_for(4, 1);
                if (kvarn_cuda_dynamic_shmem_optin_fits(candidate_shmem) &&
                        kvarn_cuda_prepare_dynamic_shmem(
                            kvarn_attn_mixed_f16_fused_batch_sparse_d512_scalar_qt_gqa_kernel<4, 1>,
                            candidate_shmem)) {
                    sparse_qt = 4;
                    sparse_ht = 1;
                    sparse_shmem = candidate_shmem;
                }
            }
            if (sparse_qt == 0) {
                std::fprintf(stderr,
                        "KVarN sparse-D512 scalar-QT-GQA shared-memory requirement was not met"
                        " (override=%d tokens=%u)\n",
                        sparse_qt_override, n_tokens);
                std::abort();
            }
            const int sparse_grid = int(
                    ((n_queries + uint32_t(sparse_qt - 1))/uint32_t(sparse_qt))*n_head_kv*
                    ((n_gqa + uint32_t(sparse_ht - 1))/uint32_t(sparse_ht)));
            if (kvarn_attn_trace_claim()) {
                std::fprintf(stderr,
                        "KVarN CUDA mixed-attn inner trace: mode=sparse-d512-scalar-qt-gqa"
                        " head_dim=%u n_queries=%u n_head=%u n_head_kv=%u n_gqa=%u"
                        " sink=%u records=%u pending=%u tail=%u tokens=%u qt=%d ht=%d grid=%d shmem=%zu\n",
                        head_dim, n_queries, n_head, n_head_kv, n_gqa,
                        n_sink, n_records, n_pending, n_tail, n_tokens,
                        sparse_qt, sparse_ht, sparse_grid, sparse_shmem);
            }
#define KVARN_LAUNCH_SPARSE_D512_GQA(QT, HT) \
            kvarn_attn_mixed_f16_fused_batch_sparse_d512_scalar_qt_gqa_kernel<QT, HT> \
                <<<sparse_grid, sparse_block, sparse_shmem, cuda_stream>>>( \
                    q, sink_tail_k, sink_tail_v, k_body, v_body, k_scales, v_scales, \
                    pending_k, pending_v, kq_mask, nullptr, nullptr, \
                    out, scores, n_queries, n_head, n_head_kv, \
                    n_sink, n_records, n_pending, n_tail, tail_start, \
                    head_dim, group_size, key_bits, value_bits, \
                    q_stride_head_floats, q_stride_query_floats, \
                    out_stride_head_floats, out_stride_query_floats, \
                    sink_tail_stride_head_f16, sink_tail_stride_token_f16, \
                    pending_stride_head_floats, pending_stride_token_floats, \
                    k_body_stride_record_bytes, v_body_stride_record_bytes, \
                    k_body_stride_head_bytes, v_body_stride_head_bytes, \
                    k_scale_stride_record_floats, v_scale_stride_record_floats, \
                    k_scale_stride_head_floats, v_scale_stride_head_floats, \
                    kq_mask_stride_query_bytes, kq_mask_stride_token_bytes, \
                    kq_mask_type, scale, 0, logit_softcap, turbo_v_mode)
            if (sparse_ht == 4) {
                KVARN_LAUNCH_SPARSE_D512_GQA(4, 4);
            } else if (sparse_qt == 8) {
                KVARN_LAUNCH_SPARSE_D512_GQA(8, 2);
            } else if (sparse_ht == 2) {
                KVARN_LAUNCH_SPARSE_D512_GQA(4, 2);
            } else {
                KVARN_LAUNCH_SPARSE_D512_GQA(4, 1);
            }
#undef KVARN_LAUNCH_SPARSE_D512_GQA
            return;
        }

        if ((head_dim == 256 || head_dim == 512) && n_records > 0 && n_queries > 1 &&
                !kvarn_env_flag("LLAMA_KVARN_ATTN_DISABLE_256D_SCALAR_QT")) {
            const int scalar_qt_block = 256;
            const size_t scalar_qt_shmem_q4 =
                (4*size_t(head_dim) + 4*size_t(n_tokens) + size_t(scalar_qt_block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
            const size_t scalar_qt_shmem_q8 =
                (8*size_t(head_dim) + 8*size_t(n_tokens) + size_t(scalar_qt_block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
            const size_t scalar_qt_shmem_q16 =
                (16*size_t(head_dim) + 16*size_t(n_tokens) + size_t(scalar_qt_block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
            const int qt_override = kvarn_env_qt_override();
            const float * scalar_body_k_f32 = nullptr;
            const float * scalar_body_v_f32 = nullptr;
            size_t scalar_body_f32_stride_head_elems = 0;
            bool scalar_used_f32_body_mirror = false;
            const void * raw_key = raw_body_store_key != nullptr ? raw_body_store_key : (const void *) k_body;
            if (use_raw_body_scalar_qt) {
                kvarn_raw_body_mirror_entry raw_body;
                if (!kvarn_raw_body_mirror_find_compatible(
                            raw_key, (const void *) k_body, n_records, n_head_kv, head_dim, group_size, raw_body)) {
                    std::fprintf(stderr,
                            "KVarN raw body scalar-QT ablation requested but no compatible mirror was captured"
                            " (raw_key=%p k_body=%p mirror_count=%zu records=%u n_head_kv=%u head_dim=%u group_size=%u raw_k=%d raw_v=%d)\n",
                            raw_key, (const void *) k_body, kvarn_raw_body_mirror_count(),
                            n_records, n_head_kv, head_dim, group_size,
                            use_raw_body_k ? 1 : 0, use_raw_body_v ? 1 : 0);
                    std::abort();
                }
                scalar_body_f32_stride_head_elems = size_t(raw_body.n_records_cap)*group_size*head_dim;
                scalar_body_k_f32 = use_raw_body_k ? raw_body.k : nullptr;
                scalar_body_v_f32 = use_raw_body_v ? raw_body.v : nullptr;
                scalar_used_f32_body_mirror = true;
            }
            const bool scalar_allow_f32_body_mirror =
                !use_raw_body_scalar_qt &&
                (head_dim == 256 || head_dim >= 512) && scores != nullptr &&
                !kvarn_env_flag("LLAMA_KVARN_ATTN_DISABLE_BODY_F32_MIRROR") &&
                (kvarn_env_flag("LLAMA_KVARN_ATTN_REF_SCRATCH") ||
                 kvarn_env_flag("LLAMA_KVARN_ENABLE_F32_DEQUANT_CACHE") ||
                 kvarn_env_flag("LLAMA_KVARN_ATTN_ENABLE_BODY_F32_MIRROR"));
            if (scalar_allow_f32_body_mirror) {
                const size_t n_per_record = size_t(group_size)*head_dim;
                const size_t n_per_head = size_t(n_records)*n_per_record;
                const size_t cap_records = size_t(k_body_records_cap > 0 ? k_body_records_cap : n_records);
                const size_t cap_elems_per_head = cap_records*n_per_record;
                const size_t cap_floats = 2*size_t(n_head_kv)*cap_elems_per_head;
                const size_t active_floats = 2*size_t(n_head_kv)*n_per_head;
                const bool enable_f32_dequant_cache =
                    kvarn_env_flag("LLAMA_KVARN_ENABLE_F32_DEQUANT_CACHE");
                const bool anchored = enable_f32_dequant_cache && scores_nelems > 0 &&
                        size_t(scores_nelems) >= size_t(n_tokens) + cap_floats &&
                        cap_records >= size_t(n_records);
                const bool active_mirror_fits = scores_nelems > 0 &&
                        size_t(scores_nelems) >= size_t(n_tokens) + active_floats;
                if (anchored || active_mirror_fits) {
                    float * k_dequant = anchored ? scores + (size_t(scores_nelems) - cap_floats) : scores + size_t(n_tokens);
                    float * v_dequant = k_dequant + size_t(n_head_kv)*(anchored ? cap_elems_per_head : n_per_head);
                    const size_t head_stride_elems = anchored ? cap_elems_per_head : n_per_head;
                    uint32_t dequant_from = 0;
                    if (anchored) {
                        dequant_from = kvarn_dequant_cache_refill_from(
                                (const void *) scores, (const void *) k_body, n_records, n_head_kv,
                                head_dim, group_size, 1);
                    }
                    if (dequant_from < n_records) {
                        for (uint32_t ih = 0; ih < n_head_kv; ++ih) {
                            ggml_cuda_kvarn_dequant_body_n_k_token_major(
                                    k_body + size_t(ih)*k_body_stride_head_bytes + size_t(dequant_from)*k_body_stride_record_bytes,
                                    v_body + size_t(ih)*v_body_stride_head_bytes + size_t(dequant_from)*v_body_stride_record_bytes,
                                    k_scales + size_t(ih)*k_scale_stride_head_floats + size_t(dequant_from)*k_scale_stride_record_floats,
                                    v_scales + size_t(ih)*v_scale_stride_head_floats + size_t(dequant_from)*v_scale_stride_record_floats,
                                    k_dequant + size_t(ih)*head_stride_elems + size_t(dequant_from)*n_per_record,
                                    v_dequant + size_t(ih)*head_stride_elems + size_t(dequant_from)*n_per_record,
                                    n_records - dequant_from, head_dim, group_size, key_bits, value_bits,
                                    k_body_stride_record_bytes, v_body_stride_record_bytes,
                                    k_scale_stride_record_floats, v_scale_stride_record_floats,
                                    n_per_record, n_per_record, turbo_v_mode, stream);
                        }
                    }
                    scalar_body_k_f32 = k_dequant;
                    scalar_body_v_f32 = v_dequant;
                    scalar_body_f32_stride_head_elems = head_stride_elems;
                    scalar_used_f32_body_mirror = true;
                }
            }
            if (use_raw_body_scalar_qt && (scalar_body_k_f32 == nullptr && scalar_body_v_f32 == nullptr)) {
                std::fprintf(stderr, "KVarN raw body scalar-QT ablation requested but no raw K or V body was selected\n");
                std::abort();
            }
            if (n_gqa >= 2 && qt_override != 16 && !kvarn_env_flag("LLAMA_KVARN_ATTN_DISABLE_GQA_SCALAR_QT")) {
                if (head_dim == 256 && n_gqa >= 8 && !kvarn_env_flag("LLAMA_KVARN_ATTN_DISABLE_GQA_SCALAR_QT_HT8")) {
                    constexpr int scalar_gqa_ht8 = 8;
                    const size_t scalar_gqa_ht8_q4_shmem =
                        (size_t(scalar_gqa_ht8)*4*size_t(head_dim) + size_t(scalar_gqa_ht8)*4*size_t(n_tokens) +
                         size_t(scalar_qt_block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
                    const size_t scalar_gqa_ht8_q8_shmem =
                        (size_t(scalar_gqa_ht8)*8*size_t(head_dim) + size_t(scalar_gqa_ht8)*8*size_t(n_tokens) +
                         size_t(scalar_qt_block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
                    const bool scalar_gqa_ht8_q8 =
                        ((qt_override == 8 && n_queries > 4) || (qt_override == 0 && n_queries > 4)) &&
                        kvarn_cuda_dynamic_shmem_optin_fits(scalar_gqa_ht8_q8_shmem) &&
                        kvarn_cuda_prepare_dynamic_shmem(kvarn_attn_mixed_f16_fused_batch_scalar_qt_gqa_kernel<8, scalar_gqa_ht8>, scalar_gqa_ht8_q8_shmem);
                    const bool scalar_gqa_ht8_q4 =
                        !scalar_gqa_ht8_q8 &&
                        qt_override == 4 && n_queries > 2 &&
                        kvarn_cuda_dynamic_shmem_optin_fits(scalar_gqa_ht8_q4_shmem) &&
                        kvarn_cuda_prepare_dynamic_shmem(kvarn_attn_mixed_f16_fused_batch_scalar_qt_gqa_kernel<4, scalar_gqa_ht8>, scalar_gqa_ht8_q4_shmem);
                    if (scalar_gqa_ht8_q8 || scalar_gqa_ht8_q4) {
                        const int selected_qt = scalar_gqa_ht8_q8 ? 8 : 4;
                        const size_t scalar_gqa_shmem = scalar_gqa_ht8_q8 ? scalar_gqa_ht8_q8_shmem : scalar_gqa_ht8_q4_shmem;
                        const int scalar_gqa_grid = int(((n_queries + uint32_t(selected_qt - 1))/uint32_t(selected_qt))*
                                n_head_kv*((n_gqa + uint32_t(scalar_gqa_ht8 - 1))/uint32_t(scalar_gqa_ht8)));
                        if (kvarn_attn_trace_claim()) {
                            std::fprintf(stderr,
                                    "KVarN CUDA mixed-attn inner trace: mode=scalar-qt-gqa head_dim=%u n_queries=%u n_head=%u"
                                    " n_head_kv=%u n_gqa=%u sink=%u records=%u pending=%u tail=%u tail_start=%u"
                                    " tokens=%u qt=%d ht=%d block=%d grid=%d shmem=%zu scores_nelems=%" PRId64
                                    " body_records_cap=%" PRId64 " body_mirror_allowed=%d body_mirror_used=%d"
                                    " kq_mask_type=%u kq_mask_stride_query_bytes=%zu kq_mask_stride_token_bytes=%zu\n",
                                    head_dim, n_queries, n_head, n_head_kv, n_gqa,
                                    n_sink, n_records, n_pending, n_tail, tail_start, n_tokens,
                                    selected_qt, scalar_gqa_ht8, scalar_qt_block, scalar_gqa_grid, scalar_gqa_shmem, scores_nelems,
                                    k_body_records_cap, scalar_allow_f32_body_mirror ? 1 : 0, scalar_used_f32_body_mirror ? 1 : 0,
                                    kq_mask_type, kq_mask_stride_query_bytes, kq_mask_stride_token_bytes);
                        }
                        if (scalar_gqa_ht8_q8) {
                            kvarn_attn_mixed_f16_fused_batch_scalar_qt_gqa_kernel<8, scalar_gqa_ht8><<<scalar_gqa_grid, scalar_qt_block, scalar_gqa_shmem, cuda_stream>>>(
                                    q, sink_tail_k, sink_tail_v, k_body, v_body, k_scales, v_scales, pending_k, pending_v,
                                    kq_mask, scalar_body_k_f32, scalar_body_v_f32,
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
                                    scalar_body_f32_stride_head_elems, logit_softcap, turbo_v_mode);
                        } else {
                            kvarn_attn_mixed_f16_fused_batch_scalar_qt_gqa_kernel<4, scalar_gqa_ht8><<<scalar_gqa_grid, scalar_qt_block, scalar_gqa_shmem, cuda_stream>>>(
                                    q, sink_tail_k, sink_tail_v, k_body, v_body, k_scales, v_scales, pending_k, pending_v,
                                    kq_mask, scalar_body_k_f32, scalar_body_v_f32,
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
                                    scalar_body_f32_stride_head_elems, logit_softcap, turbo_v_mode);
                        }
                        return;
                    }
                }
                if (head_dim == 256 && n_gqa >= 4 && !kvarn_env_flag("LLAMA_KVARN_ATTN_DISABLE_GQA_SCALAR_QT_HT4")) {
                    constexpr int scalar_gqa_ht4 = 4;
                    const size_t scalar_gqa_ht4_q4_shmem =
                        (size_t(scalar_gqa_ht4)*4*size_t(head_dim) + size_t(scalar_gqa_ht4)*4*size_t(n_tokens) +
                         size_t(scalar_qt_block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
                    const size_t scalar_gqa_ht4_q8_shmem =
                        (size_t(scalar_gqa_ht4)*8*size_t(head_dim) + size_t(scalar_gqa_ht4)*8*size_t(n_tokens) +
                         size_t(scalar_qt_block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
                    const bool scalar_gqa_ht4_q8 =
                        ((qt_override == 8 && n_queries > 4) || (qt_override == 0 && n_queries > 4)) &&
                        kvarn_cuda_dynamic_shmem_optin_fits(scalar_gqa_ht4_q8_shmem) &&
                        kvarn_cuda_prepare_dynamic_shmem(kvarn_attn_mixed_f16_fused_batch_scalar_qt_gqa_kernel<8, scalar_gqa_ht4>, scalar_gqa_ht4_q8_shmem);
                    const bool scalar_gqa_ht4_q4 =
                        !scalar_gqa_ht4_q8 &&
                        ((qt_override == 4 && n_queries > 2) || qt_override == 0) &&
                        kvarn_cuda_dynamic_shmem_optin_fits(scalar_gqa_ht4_q4_shmem) &&
                        kvarn_cuda_prepare_dynamic_shmem(kvarn_attn_mixed_f16_fused_batch_scalar_qt_gqa_kernel<4, scalar_gqa_ht4>, scalar_gqa_ht4_q4_shmem);
                    if (scalar_gqa_ht4_q8 || scalar_gqa_ht4_q4) {
                        const int selected_qt = scalar_gqa_ht4_q8 ? 8 : 4;
                        const size_t scalar_gqa_shmem = scalar_gqa_ht4_q8 ? scalar_gqa_ht4_q8_shmem : scalar_gqa_ht4_q4_shmem;
                        const int scalar_gqa_grid = int(((n_queries + uint32_t(selected_qt - 1))/uint32_t(selected_qt))*
                                n_head_kv*((n_gqa + uint32_t(scalar_gqa_ht4 - 1))/uint32_t(scalar_gqa_ht4)));
                        if (kvarn_attn_trace_claim()) {
                            std::fprintf(stderr,
                                    "KVarN CUDA mixed-attn inner trace: mode=scalar-qt-gqa head_dim=%u n_queries=%u n_head=%u"
                                    " n_head_kv=%u n_gqa=%u sink=%u records=%u pending=%u tail=%u tail_start=%u"
                                    " tokens=%u qt=%d ht=%d block=%d grid=%d shmem=%zu scores_nelems=%" PRId64
                                    " body_records_cap=%" PRId64 " body_mirror_allowed=%d body_mirror_used=%d"
                                    " kq_mask_type=%u kq_mask_stride_query_bytes=%zu kq_mask_stride_token_bytes=%zu\n",
                                    head_dim, n_queries, n_head, n_head_kv, n_gqa,
                                    n_sink, n_records, n_pending, n_tail, tail_start, n_tokens,
                                    selected_qt, scalar_gqa_ht4, scalar_qt_block, scalar_gqa_grid, scalar_gqa_shmem, scores_nelems,
                                    k_body_records_cap, scalar_allow_f32_body_mirror ? 1 : 0, scalar_used_f32_body_mirror ? 1 : 0,
                                    kq_mask_type, kq_mask_stride_query_bytes, kq_mask_stride_token_bytes);
                        }
                        if (scalar_gqa_ht4_q8) {
                            kvarn_attn_mixed_f16_fused_batch_scalar_qt_gqa_kernel<8, scalar_gqa_ht4><<<scalar_gqa_grid, scalar_qt_block, scalar_gqa_shmem, cuda_stream>>>(
                                    q, sink_tail_k, sink_tail_v, k_body, v_body, k_scales, v_scales, pending_k, pending_v,
                                    kq_mask, scalar_body_k_f32, scalar_body_v_f32,
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
                                    scalar_body_f32_stride_head_elems, logit_softcap, turbo_v_mode);
                        } else {
                            kvarn_attn_mixed_f16_fused_batch_scalar_qt_gqa_kernel<4, scalar_gqa_ht4><<<scalar_gqa_grid, scalar_qt_block, scalar_gqa_shmem, cuda_stream>>>(
                                    q, sink_tail_k, sink_tail_v, k_body, v_body, k_scales, v_scales, pending_k, pending_v,
                                    kq_mask, scalar_body_k_f32, scalar_body_v_f32,
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
                                    scalar_body_f32_stride_head_elems, logit_softcap, turbo_v_mode);
                        }
                        return;
                    }
                }
                if (head_dim == 512 && n_gqa >= 4 && kvarn_env_flag("LLAMA_KVARN_ATTN_ENABLE_512D_GQA_SCALAR_QT_HT4")) {
                    constexpr int scalar_gqa_512_ht4 = 4;
                    const size_t scalar_gqa_512_ht4_q4_shmem =
                        (size_t(scalar_gqa_512_ht4)*4*size_t(head_dim) + size_t(scalar_gqa_512_ht4)*4*size_t(n_tokens) +
                         size_t(scalar_qt_block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
                    const bool scalar_gqa_512_ht4_q4 =
                        ((qt_override == 4 && n_queries > 2) || qt_override == 0) &&
                        kvarn_cuda_dynamic_shmem_optin_fits(scalar_gqa_512_ht4_q4_shmem) &&
                        kvarn_cuda_prepare_dynamic_shmem(kvarn_attn_mixed_f16_fused_batch_scalar_qt_gqa_kernel<4, scalar_gqa_512_ht4>, scalar_gqa_512_ht4_q4_shmem);
                    if (scalar_gqa_512_ht4_q4) {
                        const int selected_qt = 4;
                        const int scalar_gqa_grid = int(((n_queries + uint32_t(selected_qt - 1))/uint32_t(selected_qt))*
                                n_head_kv*((n_gqa + uint32_t(scalar_gqa_512_ht4 - 1))/uint32_t(scalar_gqa_512_ht4)));
                        if (kvarn_attn_trace_claim()) {
                            std::fprintf(stderr,
                                    "KVarN CUDA mixed-attn inner trace: mode=scalar-qt-gqa head_dim=%u n_queries=%u n_head=%u"
                                    " n_head_kv=%u n_gqa=%u sink=%u records=%u pending=%u tail=%u tail_start=%u"
                                    " tokens=%u qt=%d ht=%d block=%d grid=%d shmem=%zu scores_nelems=%" PRId64
                                    " body_records_cap=%" PRId64 " body_mirror_allowed=%d body_mirror_used=%d"
                                    " kq_mask_type=%u kq_mask_stride_query_bytes=%zu kq_mask_stride_token_bytes=%zu\n",
                                    head_dim, n_queries, n_head, n_head_kv, n_gqa,
                                    n_sink, n_records, n_pending, n_tail, tail_start, n_tokens,
                                    selected_qt, scalar_gqa_512_ht4, scalar_qt_block, scalar_gqa_grid, scalar_gqa_512_ht4_q4_shmem, scores_nelems,
                                    k_body_records_cap, scalar_allow_f32_body_mirror ? 1 : 0, scalar_used_f32_body_mirror ? 1 : 0,
                                    kq_mask_type, kq_mask_stride_query_bytes, kq_mask_stride_token_bytes);
                        }
                        kvarn_attn_mixed_f16_fused_batch_scalar_qt_gqa_kernel<4, scalar_gqa_512_ht4><<<scalar_gqa_grid, scalar_qt_block, scalar_gqa_512_ht4_q4_shmem, cuda_stream>>>(
                                q, sink_tail_k, sink_tail_v, k_body, v_body, k_scales, v_scales, pending_k, pending_v,
                                kq_mask, scalar_body_k_f32, scalar_body_v_f32,
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
                                scalar_body_f32_stride_head_elems, logit_softcap, turbo_v_mode);
                        return;
                    }
                }
                constexpr int scalar_gqa_ht = 2;
                const size_t scalar_gqa_q4_shmem =
                    (size_t(scalar_gqa_ht)*4*size_t(head_dim) + size_t(scalar_gqa_ht)*4*size_t(n_tokens) +
                     size_t(scalar_qt_block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
                const size_t scalar_gqa_q8_shmem =
                    (size_t(scalar_gqa_ht)*8*size_t(head_dim) + size_t(scalar_gqa_ht)*8*size_t(n_tokens) +
                     size_t(scalar_qt_block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
                const bool scalar_gqa_q8 =
                    ((qt_override == 8 && n_queries > 4) || (qt_override == 0 && n_queries > 4)) &&
                    kvarn_cuda_dynamic_shmem_optin_fits(scalar_gqa_q8_shmem) &&
                    kvarn_cuda_prepare_dynamic_shmem(kvarn_attn_mixed_f16_fused_batch_scalar_qt_gqa_kernel<8, scalar_gqa_ht>, scalar_gqa_q8_shmem);
                const bool scalar_gqa_q4 =
                    !scalar_gqa_q8 &&
                    ((qt_override == 4 && n_queries > 2) || qt_override == 0) &&
                    kvarn_cuda_dynamic_shmem_optin_fits(scalar_gqa_q4_shmem) &&
                    kvarn_cuda_prepare_dynamic_shmem(kvarn_attn_mixed_f16_fused_batch_scalar_qt_gqa_kernel<4, scalar_gqa_ht>, scalar_gqa_q4_shmem);
                if (scalar_gqa_q8 || scalar_gqa_q4) {
                    const int selected_qt = scalar_gqa_q8 ? 8 : 4;
                    const size_t scalar_gqa_shmem = scalar_gqa_q8 ? scalar_gqa_q8_shmem : scalar_gqa_q4_shmem;
                    const int scalar_gqa_grid = int(((n_queries + uint32_t(selected_qt - 1))/uint32_t(selected_qt))*
                            n_head_kv*((n_gqa + uint32_t(scalar_gqa_ht - 1))/uint32_t(scalar_gqa_ht)));
                    if (kvarn_attn_trace_claim()) {
                        std::fprintf(stderr,
                                "KVarN CUDA mixed-attn inner trace: mode=scalar-qt-gqa head_dim=%u n_queries=%u n_head=%u"
                                " n_head_kv=%u n_gqa=%u sink=%u records=%u pending=%u tail=%u tail_start=%u"
                                " tokens=%u qt=%d ht=%d block=%d grid=%d shmem=%zu scores_nelems=%" PRId64
                                " body_records_cap=%" PRId64 " body_mirror_allowed=%d body_mirror_used=%d"
                                " kq_mask_type=%u kq_mask_stride_query_bytes=%zu kq_mask_stride_token_bytes=%zu\n",
                                head_dim, n_queries, n_head, n_head_kv, n_gqa,
                                n_sink, n_records, n_pending, n_tail, tail_start, n_tokens,
                                selected_qt, scalar_gqa_ht, scalar_qt_block, scalar_gqa_grid, scalar_gqa_shmem, scores_nelems,
                                k_body_records_cap, scalar_allow_f32_body_mirror ? 1 : 0, scalar_used_f32_body_mirror ? 1 : 0,
                                kq_mask_type, kq_mask_stride_query_bytes, kq_mask_stride_token_bytes);
                    }
                    if (scalar_gqa_q8) {
                        kvarn_attn_mixed_f16_fused_batch_scalar_qt_gqa_kernel<8, scalar_gqa_ht><<<scalar_gqa_grid, scalar_qt_block, scalar_gqa_shmem, cuda_stream>>>(
                                q, sink_tail_k, sink_tail_v, k_body, v_body, k_scales, v_scales, pending_k, pending_v,
                                kq_mask, scalar_body_k_f32, scalar_body_v_f32,
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
                                scalar_body_f32_stride_head_elems, logit_softcap, turbo_v_mode);
                    } else {
                        kvarn_attn_mixed_f16_fused_batch_scalar_qt_gqa_kernel<4, scalar_gqa_ht><<<scalar_gqa_grid, scalar_qt_block, scalar_gqa_shmem, cuda_stream>>>(
                                q, sink_tail_k, sink_tail_v, k_body, v_body, k_scales, v_scales, pending_k, pending_v,
                                kq_mask, scalar_body_k_f32, scalar_body_v_f32,
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
                                scalar_body_f32_stride_head_elems, logit_softcap, turbo_v_mode);
                    }
                    return;
                }
            }
            const bool scalar_qt16 =
                qt_override == 16 && n_queries > 8 && kvarn_cuda_dynamic_shmem_fits(scalar_qt_shmem_q16);
            const bool scalar_qt8 =
                !scalar_qt16 &&
                (qt_override == 8 && n_queries > 4 && kvarn_cuda_dynamic_shmem_fits(scalar_qt_shmem_q8)) ||
                (!scalar_qt16 && qt_override == 0 && n_queries > 4 && kvarn_cuda_dynamic_shmem_fits(scalar_qt_shmem_q8));
            const bool scalar_qt4 =
                !scalar_qt16 && !scalar_qt8 &&
                ((qt_override == 4 && kvarn_cuda_dynamic_shmem_fits(scalar_qt_shmem_q4)) ||
                 (qt_override == 0 && kvarn_cuda_dynamic_shmem_fits(scalar_qt_shmem_q4)));
            if (scalar_qt16 || scalar_qt8 || scalar_qt4) {
                const int selected_qt = scalar_qt16 ? 16 : (scalar_qt8 ? 8 : 4);
                const size_t scalar_qt_shmem = scalar_qt16 ? scalar_qt_shmem_q16 : (scalar_qt8 ? scalar_qt_shmem_q8 : scalar_qt_shmem_q4);
                const int scalar_qt_grid = scalar_qt16 ? int(((n_queries + 15)/16)*n_head) :
                    (scalar_qt8 ? int(((n_queries + 7)/8)*n_head) : int(((n_queries + 3)/4)*n_head));
                if (kvarn_attn_trace_claim()) {
                    std::fprintf(stderr,
                            "KVarN CUDA mixed-attn inner trace: mode=scalar-qt head_dim=%u n_queries=%u n_head=%u"
                            " n_head_kv=%u n_gqa=%u sink=%u records=%u pending=%u tail=%u tail_start=%u"
                            " tokens=%u qt=%d block=%d grid=%d shmem=%zu scores_nelems=%" PRId64
                            " body_records_cap=%" PRId64 " body_mirror_allowed=0 body_mirror_used=0"
                            " kq_mask_type=%u kq_mask_stride_query_bytes=%zu kq_mask_stride_token_bytes=%zu\n",
                            head_dim, n_queries, n_head, n_head_kv, n_gqa,
                            n_sink, n_records, n_pending, n_tail, tail_start, n_tokens,
                            selected_qt, scalar_qt_block, scalar_qt_grid, scalar_qt_shmem, scores_nelems,
                            k_body_records_cap, kq_mask_type, kq_mask_stride_query_bytes, kq_mask_stride_token_bytes);
                }
                if (scalar_qt16) {
                    kvarn_attn_mixed_f16_fused_batch_scalar_qt_kernel<16><<<scalar_qt_grid, scalar_qt_block, scalar_qt_shmem, cuda_stream>>>(
                            q, sink_tail_k, sink_tail_v, k_body, v_body, k_scales, v_scales, pending_k, pending_v,
                            kq_mask, nullptr, nullptr,
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
                             kq_mask_stride_query_bytes, kq_mask_stride_token_bytes, kq_mask_type, scale, 0, logit_softcap, turbo_v_mode);
                } else if (scalar_qt8) {
                    kvarn_attn_mixed_f16_fused_batch_scalar_qt_kernel<8><<<scalar_qt_grid, scalar_qt_block, scalar_qt_shmem, cuda_stream>>>(
                            q, sink_tail_k, sink_tail_v, k_body, v_body, k_scales, v_scales, pending_k, pending_v,
                            kq_mask, nullptr, nullptr,
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
                             kq_mask_stride_query_bytes, kq_mask_stride_token_bytes, kq_mask_type, scale, 0, logit_softcap, turbo_v_mode);
                } else {
                    kvarn_attn_mixed_f16_fused_batch_scalar_qt_kernel<4><<<scalar_qt_grid, scalar_qt_block, scalar_qt_shmem, cuda_stream>>>(
                            q, sink_tail_k, sink_tail_v, k_body, v_body, k_scales, v_scales, pending_k, pending_v,
                            kq_mask, nullptr, nullptr,
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
                             kq_mask_stride_query_bytes, kq_mask_stride_token_bytes, kq_mask_type, scale, 0, logit_softcap, turbo_v_mode);
                }
                return;
            }

            constexpr int scalar_gqa_q1_ht = 2;
            const bool scalar_gqa_q1_shape =
                qt_override == 0 &&
                key_bits == 8 && value_bits == 2 && group_size == 128 &&
                (turbo_v_mode == 0 || turbo_v_mode == 3 || turbo_v_mode == 4 || turbo_v_mode == 5) &&
                q_body == nullptr && frame_flags == 0 &&
                n_head_kv != 0 && n_head % n_head_kv == 0 && n_gqa >= 2 && n_gqa % 2 == 0 &&
                !kvarn_env_flag("LLAMA_KVARN_ATTN_DISABLE_GQA_SCALAR_QT") &&
                !kvarn_env_flag("LLAMA_KVARN_ATTN_DISABLE_Q1_GQA_SCALAR") &&
                scalar_body_k_f32 == nullptr && scalar_body_v_f32 == nullptr;
            if (scalar_gqa_q1_shape) {
                const size_t scalar_gqa_q1_shmem =
                    (size_t(scalar_gqa_q1_ht)*head_dim + size_t(scalar_gqa_q1_ht)*n_tokens +
                     size_t(scalar_qt_block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
                if (kvarn_cuda_dynamic_shmem_fits(scalar_gqa_q1_shmem) &&
                        kvarn_cuda_prepare_dynamic_shmem(
                            kvarn_attn_mixed_f16_fused_batch_scalar_qt_gqa_kernel<1, scalar_gqa_q1_ht>,
                            scalar_gqa_q1_shmem)) {
                    const int scalar_gqa_q1_grid = int(
                            n_queries*n_head_kv*(n_gqa/uint32_t(scalar_gqa_q1_ht)));
                    if (kvarn_attn_trace_claim()) {
                        std::fprintf(stderr,
                                "KVarN CUDA mixed-attn inner trace: mode=scalar-qt-gqa head_dim=%u n_queries=%u n_head=%u"
                                " n_head_kv=%u n_gqa=%u sink=%u records=%u pending=%u tail=%u tail_start=%u"
                                " tokens=%u qt=1 ht=%d block=%d grid=%d shmem=%zu scores_nelems=%" PRId64
                                " body_records_cap=%" PRId64 " body_mirror_allowed=0 body_mirror_used=0"
                                " kq_mask_type=%u kq_mask_stride_query_bytes=%zu kq_mask_stride_token_bytes=%zu\n",
                                head_dim, n_queries, n_head, n_head_kv, n_gqa,
                                n_sink, n_records, n_pending, n_tail, tail_start, n_tokens,
                                scalar_gqa_q1_ht, scalar_qt_block, scalar_gqa_q1_grid, scalar_gqa_q1_shmem,
                                scores_nelems, k_body_records_cap,
                                kq_mask_type, kq_mask_stride_query_bytes, kq_mask_stride_token_bytes);
                    }
                    kvarn_attn_mixed_f16_fused_batch_scalar_qt_gqa_kernel<1, scalar_gqa_q1_ht>
                        <<<scalar_gqa_q1_grid, scalar_qt_block, scalar_gqa_q1_shmem, cuda_stream>>>(
                            q, sink_tail_k, sink_tail_v, k_body, v_body, k_scales, v_scales,
                            pending_k, pending_v, kq_mask,
                            nullptr, nullptr,
                            out, scores, n_queries, n_head, n_head_kv,
                            n_sink, n_records, n_pending, n_tail, tail_start,
                            head_dim, group_size, key_bits, value_bits,
                            q_stride_head_floats, q_stride_query_floats,
                            out_stride_head_floats, out_stride_query_floats,
                            sink_tail_stride_head_f16, sink_tail_stride_token_f16,
                            pending_stride_head_floats, pending_stride_token_floats,
                            k_body_stride_record_bytes, v_body_stride_record_bytes,
                            k_body_stride_head_bytes, v_body_stride_head_bytes,
                            k_scale_stride_record_floats, v_scale_stride_record_floats,
                            k_scale_stride_head_floats, v_scale_stride_head_floats,
                            kq_mask_stride_query_bytes, kq_mask_stride_token_bytes,
                            kq_mask_type, scale, 0, logit_softcap, turbo_v_mode);
                    return;
                }
            }
        }

        if (use_raw_body_scalar_qt) {
            std::fprintf(stderr,
                    "KVarN raw body scalar-QT ablation requested but scalar-QT path was unavailable"
                    "; falling back to raw-body split kernels"
                    " (head_dim=%u n_queries=%u n_records=%u disable_scalar_qt=%d qt_override=%d)\n",
                    head_dim, n_queries, n_records,
                    kvarn_env_flag("LLAMA_KVARN_ATTN_DISABLE_256D_SCALAR_QT") ? 1 : 0,
                    kvarn_env_qt_override());
        }

        // Warp-QK is diagnostic-only for active-body paths: the 256d threshold
        // patch failed Qwen3.6 packed-vs-split logits, and 512d warp-QK failed
        // Gemma true-KVarN packed-vs-split. Keep production on exact scalar
        // paths unless an explicit env opt-in is being isolated.
        const bool use_warpqk_body =
            !kvarn_disable_warpqk() &&
            ((head_dim >= 512 && kvarn_enable_512d_warpqk()) ||
             (head_dim == 256 && kvarn_enable_256d_warpqk()));
        if (use_warpqk_body && n_records > 0) {
            const int warpqk_block = head_dim >= 512 ? 512 : 256;
            // Q-tile selection: QT=4 amortizes every K/V load across 4 query
            // rows (4x fewer CTAs, 4x FMA reuse) whenever the tiled shared
            // layout fits; decode-shaped launches (n_queries==1) and
            // shmem-constrained windows fall back to the QT=1 instantiation,
            // which is behavior-identical to the previous scalar kernel.
            // shared = q_sh[QT*head_dim] + probs[QT*n_tokens] + reduce[block]
            const size_t warpqk_shmem_q1 = (size_t(head_dim) + size_t(n_tokens) + size_t(warpqk_block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
            const size_t warpqk_shmem_q4 = (4*size_t(head_dim) + 4*size_t(n_tokens) + size_t(warpqk_block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
            const size_t warpqk_shmem_q8 = (8*size_t(head_dim) + 8*size_t(n_tokens) + size_t(warpqk_block) + KVARN_ATTN_SHMEM_PAD_FLOATS)*sizeof(float);
            const int qt_override = kvarn_env_qt_override();
            const bool warpqk_q8 =
                (qt_override == 8 && n_queries > 4 && kvarn_cuda_dynamic_shmem_fits(warpqk_shmem_q8)) ||
                (qt_override == 0 && n_queries > 4 && kvarn_cuda_dynamic_shmem_fits(warpqk_shmem_q8));
            const bool warpqk_q4 =
                !warpqk_q8 &&
                ((qt_override == 4 && n_queries > 1 && kvarn_cuda_dynamic_shmem_fits(warpqk_shmem_q4)) ||
                 (qt_override == 0 && n_queries > 1 && kvarn_cuda_dynamic_shmem_fits(warpqk_shmem_q4)));
            const size_t warpqk_shmem = warpqk_q8 ? warpqk_shmem_q8 : (warpqk_q4 ? warpqk_shmem_q4 : warpqk_shmem_q1);
            if (kvarn_cuda_dynamic_shmem_fits(warpqk_shmem)) {
                const __half * body_k_f16 = nullptr;
                const __half * body_v_f16 = nullptr;
                size_t body_f16_stride_head_elems = 0;

                const bool allow_f16_body_mirror =
                    !kvarn_disable_body_mirror() &&
                    (head_dim >= 512 || kvarn_env_flag("LLAMA_KVARN_ATTN_ENABLE_256D_BODY_MIRROR"));
                bool used_f16_body_mirror = false;

                if (n_records > 0 && scores != nullptr && allow_f16_body_mirror) {
                    const size_t n_per_record = size_t(group_size)*head_dim;
                    const size_t n_per_head = size_t(n_records)*n_per_record;
                    // K/V f16 mirrors are END-ANCHORED in the persistent attn scratch
                    // (capacity-sized region whose base does not move as the window
                    // grows). A host-side cache keyed by (scratch ptr -> k_body ptr,
                    // n_records, store-epoch) refills incrementally on seal only.
                    // Packed V in-kernel is cheaper in theory but measured slower on
                    // 512d Gemma warpqk AV; keep the f16 V mirror.
                    const size_t cap_records = size_t(k_body_records_cap > 0 ? k_body_records_cap : n_records);
                    const size_t cap_elems_per_head = cap_records*n_per_record;
                    const size_t cap_halves_kv = 2*size_t(n_head_kv)*cap_elems_per_head;
                    const size_t cap_floats = (cap_halves_kv + 1)/2;
                    const bool anchored = scores_nelems > 0 &&
                            size_t(scores_nelems) >= size_t(n_tokens) + cap_floats &&
                            cap_records >= size_t(n_records);
                    const bool active_mirror_fits = scores_nelems > 0 &&
                            size_t(scores_nelems) >= size_t(n_tokens) + (2*size_t(n_head_kv)*n_per_head + 1)/2;
                    if (anchored || active_mirror_fits) {
                        __half * k_dequant = anchored ?
                            reinterpret_cast<__half *>(scores + (size_t(scores_nelems) - cap_floats)) :
                            reinterpret_cast<__half *>(scores + size_t(n_tokens));
                        __half * v_dequant = k_dequant + size_t(n_head_kv)*(anchored ? cap_elems_per_head : n_per_head);

                        uint32_t dequant_from = 0;
                        if (anchored) {
                            dequant_from = kvarn_dequant_cache_refill_from(
                                    (const void *) scores, (const void *) k_body, n_records, n_head_kv,
                                    head_dim, group_size, 0);
                        }

                        if (dequant_from < n_records) {
                            const size_t head_stride_elems = anchored ? cap_elems_per_head : n_per_head;
                            for (uint32_t ikh = 0; ikh < n_head_kv; ++ikh) {
                                ggml_cuda_kvarn_dequant_body_n_k_token_major_f16(
                                        k_body + size_t(ikh)*k_body_stride_head_bytes + size_t(dequant_from)*k_body_stride_record_bytes,
                                        v_body + size_t(ikh)*v_body_stride_head_bytes + size_t(dequant_from)*v_body_stride_record_bytes,
                                        k_scales + size_t(ikh)*k_scale_stride_head_floats + size_t(dequant_from)*k_scale_stride_record_floats,
                                        v_scales + size_t(ikh)*v_scale_stride_head_floats + size_t(dequant_from)*v_scale_stride_record_floats,
                                        k_dequant + size_t(ikh)*head_stride_elems + size_t(dequant_from)*n_per_record,
                                        v_dequant + size_t(ikh)*head_stride_elems + size_t(dequant_from)*n_per_record,
                                        n_records - dequant_from, head_dim, group_size, key_bits, value_bits,
                                        k_body_stride_record_bytes, v_body_stride_record_bytes,
                                        k_scale_stride_record_floats, v_scale_stride_record_floats,
                                        n_per_record, n_per_record,
                                        turbo_v_mode,
                                        cuda_stream);
                            }
                        }

                        body_k_f16 = k_dequant;
                        body_v_f16 = v_dequant;
                        body_f16_stride_head_elems = anchored ? cap_elems_per_head : n_per_head;
                        used_f16_body_mirror = true;
                    }
                }

                if (head_dim == 256 && n_records > 0 &&
                        kvarn_env_flag("LLAMA_KVARN_ATTN_ENABLE_256D_BODY_MIRROR") &&
                        scores_nelems > 0 &&
                        !used_f16_body_mirror) {
                    std::fprintf(stderr,
                            "KVarN CUDA mixed-attn diagnostic error: LLAMA_KVARN_ATTN_ENABLE_256D_BODY_MIRROR=1"
                            " but body_mirror_used=0 (scores_nelems=%" PRId64 ", required_body_f16_elems=%zu)\n",
                            scores_nelems,
                            2*size_t(n_head_kv)*size_t(n_records)*size_t(group_size)*size_t(head_dim));
                    std::abort();
                }

                const int warpqk_grid = warpqk_q8 ? int(((n_queries + 7)/8)*n_head) :
                    (warpqk_q4 ? int(((n_queries + 3)/4)*n_head) : int(n_queries*n_head));
                if (kvarn_attn_trace_claim()) {
                    const int selected_qt = warpqk_q8 ? 8 : (warpqk_q4 ? 4 : 1);
                    std::fprintf(stderr,
                            "KVarN CUDA mixed-attn inner trace: mode=warpqk-f16 head_dim=%u n_queries=%u n_head=%u"
                            " n_head_kv=%u n_gqa=%u sink=%u records=%u pending=%u tail=%u tail_start=%u"
                            " tokens=%u qt=%d block=%d grid=%d shmem=%zu scores_nelems=%" PRId64
                            " body_records_cap=%" PRId64 " body_mirror_allowed=%d body_mirror_used=%d"
                            " kq_mask_type=%u kq_mask_stride_query_bytes=%zu kq_mask_stride_token_bytes=%zu\n",
                            head_dim, n_queries, n_head, n_head_kv, n_gqa,
                            n_sink, n_records, n_pending, n_tail, tail_start, n_tokens,
                            selected_qt, warpqk_block, warpqk_grid, warpqk_shmem, scores_nelems,
                            k_body_records_cap, allow_f16_body_mirror ? 1 : 0, used_f16_body_mirror ? 1 : 0,
                            kq_mask_type, kq_mask_stride_query_bytes, kq_mask_stride_token_bytes);
                }
                if (warpqk_q8) {
                kvarn_attn_mixed_f16_fused_batch_warpqk_kernel<8><<<warpqk_grid, warpqk_block, warpqk_shmem, cuda_stream>>>(
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
                        body_k_f16, body_v_f16, body_f16_stride_head_elems, logit_softcap, turbo_v_mode);
                } else if (warpqk_q4) {
                kvarn_attn_mixed_f16_fused_batch_warpqk_kernel<4><<<warpqk_grid, warpqk_block, warpqk_shmem, cuda_stream>>>(
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
                        body_k_f16, body_v_f16, body_f16_stride_head_elems, logit_softcap, turbo_v_mode);
                } else {
                kvarn_attn_mixed_f16_fused_batch_warpqk_kernel<1><<<warpqk_grid, warpqk_block, warpqk_shmem, cuda_stream>>>(
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
                        body_k_f16, body_v_f16, body_f16_stride_head_elems, logit_softcap, turbo_v_mode);
                }
                return;
            }
        }

        if (kvarn_cuda_dynamic_shmem_optin_fits(shmem) &&
                kvarn_cuda_prepare_dynamic_shmem(kvarn_attn_mixed_f16_fused_batch_kernel, shmem)) {
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
                    kq_mask_stride_query_bytes, kq_mask_stride_token_bytes, kq_mask_type, scale, logit_softcap, turbo_v_mode);
            kvarn_cuda_trace_launch_error(
                    "fused-batch-generic", n_queries, n_head, n_head_kv,
                    n_sink, n_records, n_pending, n_tail, head_dim, n_tokens,
                    int(n_queries*n_head), block, shmem);
            return;
        }
    }

    kvarn_raw_body_mirror_entry raw_body;
    bool have_raw_body = false;
    if ((use_raw_body_k || use_raw_body_v) && n_records > 0) {
        const void * raw_key = raw_body_store_key != nullptr ? raw_body_store_key : (const void *) k_body;
        have_raw_body = kvarn_raw_body_mirror_find_compatible(
                raw_key, (const void *) k_body, n_records, n_head_kv, head_dim, group_size, raw_body);
        if (!have_raw_body) {
            std::fprintf(stderr,
                    "KVarN raw body ablation requested but no compatible mirror was captured"
                    " (raw_key=%p k_body=%p mirror_count=%zu records=%u n_head_kv=%u head_dim=%u group_size=%u raw_k=%d raw_v=%d)\n",
                    raw_key, (const void *) k_body, kvarn_raw_body_mirror_count(),
                    n_records, n_head_kv, head_dim, group_size,
                    use_raw_body_k ? 1 : 0, use_raw_body_v ? 1 : 0);
            std::abort();
        }
    }
    const float * raw_body_k_base = (use_raw_body_k && have_raw_body) ? raw_body.k : nullptr;
    const float * raw_body_v_base = (use_raw_body_v && have_raw_body) ? raw_body.v : nullptr;
    const size_t raw_body_stride_head_floats = have_raw_body ?
        size_t(raw_body.n_records_cap)*group_size*head_dim : 0;

    for (uint32_t iq = 0; iq < n_queries; ++iq) {
        for (uint32_t ih = 0; ih < n_head; ++ih) {
            const uint32_t ikh = ih/n_gqa;
            const float * q_ptr = q + size_t(iq)*q_stride_query_floats + size_t(ih)*q_stride_head_floats;
            const float * q_body_ptr = q_body == nullptr ? nullptr :
                q_body + size_t(iq)*q_body_stride_query_floats + size_t(ih)*q_body_stride_head_floats;
            float * out_ptr = out + size_t(iq)*out_stride_query_floats + size_t(ih)*out_stride_head_floats;
            const uint16_t * k_st_ptr = sink_tail_k + size_t(ikh)*sink_tail_stride_head_f16;
            const uint16_t * v_st_ptr = sink_tail_v + size_t(ikh)*sink_tail_stride_head_f16;
            const uint8_t * k_body_ptr = k_body + size_t(ikh)*k_body_stride_head_bytes;
            const uint8_t * v_body_ptr = v_body + size_t(ikh)*v_body_stride_head_bytes;
            const float * k_scales_ptr = k_scales + size_t(ikh)*k_scale_stride_head_floats;
            const float * v_scales_ptr = v_scales + size_t(ikh)*v_scale_stride_head_floats;
            const float * pending_k_ptr = pending_k + size_t(ikh)*pending_stride_head_floats;
            const float * pending_v_ptr = pending_v + size_t(ikh)*pending_stride_head_floats;
            const float * raw_body_k_ptr = raw_body_k_base == nullptr ? nullptr :
                raw_body_k_base + size_t(ikh)*raw_body_stride_head_floats;
            const float * raw_body_v_ptr = raw_body_v_base == nullptr ? nullptr :
                raw_body_v_base + size_t(ikh)*raw_body_stride_head_floats;
            const void * kq_mask_ptr = kq_mask == nullptr ? nullptr :
                (const char *) kq_mask + size_t(iq)*kq_mask_stride_query_bytes;

            if (use_split_kernels) {
                if (kvarn_cuda_dynamic_shmem_optin_fits(shmem) &&
                        kvarn_cuda_prepare_dynamic_shmem(kvarn_attn_mixed_f16_scores_softmax_kernel, shmem)) {
                    kvarn_attn_mixed_f16_scores_softmax_kernel<<<1, block, shmem, cuda_stream>>>(
                            q_ptr, q_body_ptr, k_st_ptr, k_body_ptr, k_scales_ptr, raw_body_k_ptr,
                            pending_k_ptr, kq_mask_ptr, scores,
                            n_sink, n_records, n_pending, n_tail, tail_start, head_dim, group_size, key_bits,
                            sink_tail_stride_token_f16, pending_stride_token_floats,
                            k_body_stride_record_bytes, k_scale_stride_record_floats,
                            kq_mask_stride_token_bytes, kq_mask_type,
                            kvarn_causal_mask_limit(n_tokens, n_queries, iq), scale, logit_softcap);
                    kvarn_cuda_trace_launch_error(
                            "split-softmax", n_queries, n_head, n_head_kv,
                            n_sink, n_records, n_pending, n_tail, head_dim, n_tokens,
                            1, block, shmem);
                } else {
                    kvarn_attn_mixed_f16_scores_softmax_global_kernel<<<1, block, reduce_shmem, cuda_stream>>>(
                            q_ptr, q_body_ptr, k_st_ptr, k_body_ptr, k_scales_ptr, raw_body_k_ptr,
                            pending_k_ptr, kq_mask_ptr, scores,
                            n_sink, n_records, n_pending, n_tail, tail_start, head_dim, group_size, key_bits,
                            sink_tail_stride_token_f16, pending_stride_token_floats,
                            k_body_stride_record_bytes, k_scale_stride_record_floats,
                            kq_mask_stride_token_bytes, kq_mask_type,
                            kvarn_causal_mask_limit(n_tokens, n_queries, iq), scale, logit_softcap);
                    kvarn_cuda_trace_launch_error(
                            "split-softmax-global", n_queries, n_head, n_head_kv,
                            n_sink, n_records, n_pending, n_tail, head_dim, n_tokens,
                            1, block, reduce_shmem);
                }
                if (use_turbo2_body_inverse) {
                    kvarn_attn_mixed_f16_turbo2_av_kernel<<<av_grid, av_block, 0, cuda_stream>>>(
                            scores, v_st_ptr, v_body_ptr, v_scales_ptr,
                            pending_v_ptr, out_ptr,
                            n_sink, n_records, n_pending, n_tail, tail_start, head_dim, group_size, value_bits,
                            sink_tail_stride_token_f16, pending_stride_token_floats,
                            v_body_stride_record_bytes, v_scale_stride_record_floats);
                } else {
                    kvarn_attn_mixed_f16_av_kernel<<<av_grid, av_block, 0, cuda_stream>>>(
                            scores, v_st_ptr, v_body_ptr, v_scales_ptr, raw_body_v_ptr,
                            pending_v_ptr, out_ptr,
                            n_sink, n_records, n_pending, n_tail, tail_start, head_dim, group_size, value_bits,
                            sink_tail_stride_token_f16, pending_stride_token_floats,
                            v_body_stride_record_bytes, v_scale_stride_record_floats, turbo_v_mode);
                }
                kvarn_cuda_trace_launch_error(
                        "split-av", n_queries, n_head, n_head_kv,
                        n_sink, n_records, n_pending, n_tail, head_dim, n_tokens,
                        av_grid, av_block, 0);
            } else {
                if (kvarn_cuda_dynamic_shmem_optin_fits(shmem) &&
                        kvarn_cuda_prepare_dynamic_shmem(kvarn_attn_mixed_f16_fused_kernel, shmem)) {
                    kvarn_attn_mixed_f16_fused_kernel<<<1, block, shmem, cuda_stream>>>(
                            q_ptr, q_body_ptr, k_st_ptr, v_st_ptr, k_body_ptr, v_body_ptr, k_scales_ptr, v_scales_ptr,
                            pending_k_ptr, pending_v_ptr, kq_mask_ptr, out_ptr, scores,
                            n_sink, n_records, n_pending, n_tail, tail_start, head_dim, group_size, key_bits, value_bits,
                            sink_tail_stride_token_f16, pending_stride_token_floats,
                            k_body_stride_record_bytes, v_body_stride_record_bytes,
                            k_scale_stride_record_floats, v_scale_stride_record_floats,
                            kq_mask_stride_token_bytes, kq_mask_type,
                            kvarn_causal_mask_limit(n_tokens, n_queries, iq), scale, logit_softcap, turbo_v_mode);
                    kvarn_cuda_trace_launch_error(
                            "serial-fused", n_queries, n_head, n_head_kv,
                            n_sink, n_records, n_pending, n_tail, head_dim, n_tokens,
                            1, block, shmem);
                } else {
                    kvarn_attn_mixed_f16_scores_softmax_global_kernel<<<1, block, reduce_shmem, cuda_stream>>>(
                            q_ptr, q_body_ptr, k_st_ptr, k_body_ptr, k_scales_ptr, nullptr,
                            pending_k_ptr, kq_mask_ptr, scores,
                            n_sink, n_records, n_pending, n_tail, tail_start, head_dim, group_size, key_bits,
                            sink_tail_stride_token_f16, pending_stride_token_floats,
                            k_body_stride_record_bytes, k_scale_stride_record_floats,
                            kq_mask_stride_token_bytes, kq_mask_type,
                            kvarn_causal_mask_limit(n_tokens, n_queries, iq), scale, logit_softcap);
                    kvarn_cuda_trace_launch_error(
                            "serial-softmax-global", n_queries, n_head, n_head_kv,
                            n_sink, n_records, n_pending, n_tail, head_dim, n_tokens,
                            1, block, reduce_shmem);
                    if (use_turbo2_body_inverse) {
                        kvarn_attn_mixed_f16_turbo2_av_kernel<<<av_grid, av_block, 0, cuda_stream>>>(
                                scores, v_st_ptr, v_body_ptr, v_scales_ptr,
                                pending_v_ptr, out_ptr,
                                n_sink, n_records, n_pending, n_tail, tail_start, head_dim, group_size, value_bits,
                                sink_tail_stride_token_f16, pending_stride_token_floats,
                                v_body_stride_record_bytes, v_scale_stride_record_floats);
                    } else {
                        kvarn_attn_mixed_f16_av_kernel<<<av_grid, av_block, 0, cuda_stream>>>(
                                scores, v_st_ptr, v_body_ptr, v_scales_ptr, nullptr,
                                pending_v_ptr, out_ptr,
                                n_sink, n_records, n_pending, n_tail, tail_start, head_dim, group_size, value_bits,
                                sink_tail_stride_token_f16, pending_stride_token_floats,
                                v_body_stride_record_bytes, v_scale_stride_record_floats, turbo_v_mode);
                    }
                    kvarn_cuda_trace_launch_error(
                            "serial-av-global", n_queries, n_head, n_head_kv,
                            n_sink, n_records, n_pending, n_tail, head_dim, n_tokens,
                            av_grid, av_block, 0);
                }
            }
        }
    }
}
