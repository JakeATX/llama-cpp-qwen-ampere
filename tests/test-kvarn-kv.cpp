#include "llama-kv-cache-kvarn.h"
#include "llama-kv-cache-kvarn-iswa.h"
#include "llama-hparams.h"
#include "llama.h"
#include "ggml-backend.h"

#include <cmath>
#include <exception>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char * msg) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
}

static bool phase_trace_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("LLAMA_KVARN_TEST_PHASE_TRACE");
        return env != nullptr && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static void trace_phase(const char * name, const char * state) {
    if (!phase_trace_enabled()) {
        return;
    }
    std::fprintf(stderr, "KVarN test phase: %s %s\n", name, state);
    std::fflush(stderr);
}

static void run_phase(const char * name, void (*fn)()) {
    trace_phase(name, "start");
    try {
        fn();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s threw std::exception: %s\n", name, e.what());
        std::exit(1);
    } catch (...) {
        std::fprintf(stderr, "FAIL: %s threw unknown exception\n", name);
        std::exit(1);
    }
    trace_phase(name, "pass");
}

static void test_layout() {
    llama_kvarn_params params = llama_kvarn_default_params();
    llama_kvarn_layout layout = llama_kvarn_make_layout(params, 128);

    require(layout.k_body_bytes == 8192, "K body bytes");
    require(layout.v_body_bytes == 4096, "V body bytes");
    require(layout.k_scale_floats == 384, "K scale float count");
    require(layout.v_scale_floats == 384, "V scale float count");
    require(layout.total_record_bytes == 15360, "total record bytes");

    llama_kvarn_layout layout256 = llama_kvarn_make_layout(params, 256);
    require(layout256.k_body_bytes == 16384, "256-dim K body bytes");
    require(layout256.v_body_bytes == 8192, "256-dim V body bytes");
    require(layout256.k_scale_floats == 640, "256-dim K scale float count");
    require(layout256.v_scale_floats == 512, "256-dim V scale float count");
    require(layout256.total_record_bytes == 29184, "256-dim total record bytes");

    llama_kvarn_layout layout512 = llama_kvarn_make_layout(params, 512);
    require(layout512.k_body_bytes == 32768, "512-dim K body bytes");
    require(layout512.v_body_bytes == 16384, "512-dim V body bytes");
    require(layout512.k_scale_floats == 1152, "512-dim K scale float count");
    require(layout512.v_scale_floats == 768, "512-dim V scale float count");
    require(layout512.total_record_bytes == 56832, "512-dim total record bytes");
}

static void test_pack_roundtrip() {
    for (uint32_t bits : { 2u, 4u }) {
        std::vector<uint8_t> src(257);
        const uint32_t qmax = (1u << bits) - 1u;
        for (size_t i = 0; i < src.size(); ++i) {
            src[i] = uint8_t((i*7 + 3) & qmax);
        }

        std::vector<uint8_t> packed;
        std::vector<uint8_t> unpacked;
        llama_kvarn_pack_bits(src, bits, packed);
        llama_kvarn_unpack_bits(packed, bits, src.size(), unpacked);

        require(src == unpacked, "bit pack roundtrip");
    }
}

static void test_hadamard_inverse() {
    const uint32_t rows = 128;
    const uint32_t cols = 5;
    std::vector<float> src(size_t(rows)*cols);
    for (size_t i = 0; i < src.size(); ++i) {
        src[i] = std::sin(float(i)*0.013f) + std::cos(float(i)*0.021f);
    }

    std::vector<float> tmp;
    std::vector<float> dst;
    llama_kvarn_hadamard_channels(src, tmp, rows, cols, true);
    llama_kvarn_hadamard_channels(tmp, dst, rows, cols, true);

    float max_err = 0.0f;
    for (size_t i = 0; i < src.size(); ++i) {
        max_err = std::max(max_err, std::fabs(src[i] - dst[i]));
    }
    require(max_err < 1.0e-5f, "Hadamard inverse");
}

static void test_reference_store_dequant() {
    llama_kvarn_params params = llama_kvarn_default_params();
    params.sinkhorn_iters = 4;

    for (const uint32_t head_dim : { 128u, 256u, 512u }) {
    const uint32_t group = params.group_size;
    const llama_kvarn_layout layout = llama_kvarn_make_layout(params, head_dim);
    std::vector<float> k_tile(size_t(head_dim)*group);
    std::vector<float> v_tile(size_t(head_dim)*group);

    for (size_t i = 0; i < k_tile.size(); ++i) {
        k_tile[i] = 0.05f*std::sin(float(i)*0.017f) + 0.02f*std::cos(float(i)*0.003f);
        v_tile[i] = 0.04f*std::cos(float(i)*0.011f) - 0.03f*std::sin(float(i)*0.007f);
    }

    llama_kvarn_body_record record = llama_kvarn_store_reference(params, head_dim, k_tile, v_tile);
    require(record.k_body.size() == layout.k_body_bytes, "stored K body size");
    require(record.v_body.size() == layout.v_body_bytes, "stored V body size");
    require(record.k_scales.size() == layout.k_scale_floats, "stored K scale count");
    require(record.v_scales.size() == layout.v_scale_floats, "stored V scale count");

    std::vector<float> k_deq;
    std::vector<float> v_deq;
    llama_kvarn_dequant_reference(record, k_deq, v_deq);

    require(k_deq.size() == k_tile.size(), "dequant K size");
    require(v_deq.size() == v_tile.size(), "dequant V size");

    float k_abs = 0.0f;
    float v_abs = 0.0f;
    for (size_t i = 0; i < k_deq.size(); ++i) {
        require(std::isfinite(k_deq[i]), "finite K dequant");
        require(std::isfinite(v_deq[i]), "finite V dequant");
        k_abs = std::max(k_abs, std::fabs(k_deq[i]));
        v_abs = std::max(v_abs, std::fabs(v_deq[i]));
    }
    require(k_abs > 0.0f, "nonzero K dequant");
    require(v_abs > 0.0f, "nonzero V dequant");
    }
}

static void test_reference_cache_sealing() {
    llama_kvarn_params params = llama_kvarn_default_params();
    params.group_size = 4;
    params.sink_tokens = 2;
    params.tail_tokens = 2;
    params.sinkhorn_iters = 2;

    llama_kvarn_reference_cache cache(params, 8);

    for (uint32_t t = 0; t < 9; ++t) {
        std::vector<float> k(8);
        std::vector<float> v(8);
        for (uint32_t d = 0; d < 8; ++d) {
            k[d] = float(t)*0.1f + float(d)*0.01f;
            v[d] = float(t)*0.2f - float(d)*0.02f;
        }
        cache.append_token(k, v);
    }

    const llama_kvarn_reference_cache_stats s = cache.stats();
    require(s.n_tokens == 9, "reference cache token count");
    require(s.n_sink == 2, "reference cache sink count");
    require(s.n_tail == 2, "reference cache tail count");
    require(s.n_pending_body == 1, "reference cache pending count");
    require(s.n_body_records == 1, "reference cache body record count");
    require(cache.body_records()[0].k_body.size() == 16, "reference cache sealed K bytes");
    require(cache.body_records()[0].v_body.size() == 8, "reference cache sealed V bytes");

    std::vector<float> k_mat;
    std::vector<float> v_mat;
    cache.materialize_tokens(k_mat, v_mat);
    require(k_mat.size() == size_t(s.n_tokens)*8, "materialized K token count");
    require(v_mat.size() == size_t(s.n_tokens)*8, "materialized V token count");

    for (uint32_t d = 0; d < 8; ++d) {
        require(std::fabs(k_mat[d] - (float(d)*0.01f)) < 1.0e-4f, "materialized sink K order");
        require(std::fabs(v_mat[d] - (-float(d)*0.02f)) < 1.0e-4f, "materialized sink V order");
    }

    bool body_nonzero = false;
    for (uint32_t i = params.sink_tokens*8; i < (params.sink_tokens + params.group_size)*8; ++i) {
        body_nonzero = body_nonzero || std::fabs(k_mat[i]) > 0.0f || std::fabs(v_mat[i]) > 0.0f;
    }
    require(body_nonzero, "materialized sealed body contributes data");

    cache.clear();
    const llama_kvarn_reference_cache_stats empty = cache.stats();
    require(empty.n_tokens == 0, "reference cache clear token count");
    require(empty.n_body_records == 0, "reference cache clear records");
}

static llama_hparams make_test_hparams(uint32_t head_dim = 128) {
    llama_hparams hparams = {};
    hparams.n_layer_all = 2;
    hparams.n_embd_head_k_full = head_dim;
    hparams.n_embd_head_v_full = head_dim;
    hparams.n_embd_head_k_swa = head_dim;
    hparams.n_embd_head_v_swa = head_dim;
    hparams.n_head_kv_arr[0] = 4;
    hparams.n_head_kv_arr[1] = 4;
    hparams.n_head_arr[0] = 4;
    hparams.n_head_arr[1] = 4;
    return hparams;
}

static llama_hparams make_small_storage_hparams() {
    llama_hparams hparams = {};
    hparams.n_layer_all = 2;
    hparams.n_embd_head_k_full = 8;
    hparams.n_embd_head_v_full = 8;
    hparams.n_embd_head_k_swa = 8;
    hparams.n_embd_head_v_swa = 8;
    hparams.n_head_kv_arr[0] = 2;
    hparams.n_head_kv_arr[1] = 2;
    hparams.n_head_arr[0] = 2;
    hparams.n_head_arr[1] = 2;
    return hparams;
}

static llama_ubatch make_test_ubatch(uint32_t n_tokens, llama_seq_id seq_id) {
    llama_ubatch ubatch = {};
    ubatch.b_equal_seqs = false;
    ubatch.n_tokens = n_tokens;
    ubatch.n_seq_tokens = 1;
    ubatch.n_seqs = n_tokens;
    ubatch.n_seqs_unq = 1;
    ubatch.n_pos = 1;
    ubatch.data = std::make_shared<llama_ubatch::data_t>();

    auto & data = *ubatch.data;
    data.pos.resize(n_tokens);
    data.n_seq_id.resize(n_tokens, 1);
    data.seq_id.resize(n_tokens);
    data.seq_id_unq = { seq_id };
    data.seq_idx.resize(LLAMA_MAX_SEQ, -1);
    data.seq_idx[seq_id] = 0;
    data.output.resize(n_tokens, 0);
    data.seq_id_data.resize(n_tokens, seq_id);

    for (uint32_t i = 0; i < n_tokens; ++i) {
        data.pos[i] = i;
        data.seq_id[i] = &data.seq_id_data[i];
    }

    ubatch.pos = data.pos.data();
    ubatch.n_seq_id = data.n_seq_id.data();
    ubatch.seq_id = data.seq_id.data();
    ubatch.seq_id_unq = data.seq_id_unq.data();
    ubatch.seq_idx = data.seq_idx.data();
    ubatch.output = data.output.data();
    return ubatch;
}

static void test_memory_estimate() {
    llama_kvarn_params params = llama_kvarn_default_params();
    llama_hparams hparams = make_test_hparams();

    const llama_kvarn_memory_estimate est = llama_kvarn_estimate_memory(params, hparams, 512);
    require(est.fp16_sink_tail_bytes == 1048576, "KVarN sink/tail estimate");
    require(est.body_packed_bytes == 196608, "KVarN body estimate");
    require(est.scale_bytes == 49152, "KVarN scale estimate");
    require(est.total_bytes == 1294336, "KVarN total estimate");

    llama_hparams hparams256 = make_test_hparams(256);
    const llama_kvarn_memory_estimate est256 = llama_kvarn_estimate_memory(params, hparams256, 512);
    require(est256.fp16_sink_tail_bytes == 2097152, "256-dim KVarN sink/tail estimate");
    require(est256.body_packed_bytes == 393216, "256-dim KVarN body estimate");
    require(est256.scale_bytes == 73728, "256-dim KVarN scale estimate");
    require(est256.total_bytes == 2564096, "256-dim KVarN total estimate");

    llama_hparams hparams512 = make_test_hparams(512);
    const llama_kvarn_memory_estimate est512 = llama_kvarn_estimate_memory(params, hparams512, 512);
    require(est512.fp16_sink_tail_bytes == 4194304, "512-dim KVarN sink/tail estimate");
    require(est512.body_packed_bytes == 786432, "512-dim KVarN body estimate");
    require(est512.scale_bytes == 122880, "512-dim KVarN scale estimate");
    require(est512.total_bytes == 5103616, "512-dim KVarN total estimate");

}

static void test_runtime_metadata() {
    llama_kvarn_params params = llama_kvarn_default_params();
    params.sink_tokens = 8;
    params.tail_tokens = 8;
    llama_hparams hparams = make_test_hparams();
    llama_kv_cache_kvarn cache(nullptr, hparams, params, false, 16, 4, 1, nullptr);

    require(cache.get_size() == 16, "KVarN runtime size");
    require(cache.get_n_layer() == 2, "KVarN runtime layer count");
    require(cache.backend_tensor_bytes() > 0, "KVarN backend tensors allocated");
    const llama_kvarn_layer_view view = cache.get_layer_view(0);
    require(view.il == 0, "KVarN layer view id");
    require(view.n_head_kv == 4, "KVarN layer view head count");
    require(view.n_records == 0, "KVarN layer view logical body records");
    require(view.head_dim_k == 128, "KVarN layer view K head dim");
    require(view.head_dim_v == 128, "KVarN layer view V head dim");
    require(view.layout_k.k_body_bytes == 8192, "KVarN layer view K body bytes");
    require(view.layout_v.v_body_bytes == 4096, "KVarN layer view V body bytes");
    require(view.sink_tail_k->type == GGML_TYPE_F16, "KVarN layer view sink K type");
    require(view.sink_tail_v->type == GGML_TYPE_F16, "KVarN layer view sink V type");
    require(view.body_k->type == GGML_TYPE_I8, "KVarN layer view body K type");
    require(view.body_v->type == GGML_TYPE_I8, "KVarN layer view body V type");
    require(view.scales_k->type == GGML_TYPE_F32, "KVarN layer view scale K type");
    require(view.scales_v->type == GGML_TYPE_F32, "KVarN layer view scale V type");
    require(view.sink_tail_k->ne[0] == 128 && view.sink_tail_k->ne[1] == 4, "KVarN layer view sink K shape");
    require(view.body_k->ne[0] == (int64_t) view.layout_k.k_body_bytes, "KVarN layer view body K shape");
    require(view.body_v->ne[0] == (int64_t) view.layout_v.v_body_bytes, "KVarN layer view body V shape");
    require(view.scales_k->ne[0] == (int64_t) view.layout_k.k_scale_floats, "KVarN layer view scale K shape");
    require(view.scales_v->ne[0] == (int64_t) view.layout_v.v_scale_floats, "KVarN layer view scale V shape");
    {
        const size_t tile = size_t(128)*128;
        const size_t per_pipeline = tile + 2*128 + 128 + 128 + 1;
        const size_t expected = 2*tile + per_pipeline;
        require(cache.body_store_scratch_floats(0) == expected, "128-dim multi-head KVarN body store scratch floats");
    }
    size_t breakdown_bytes = 0;
    for (const auto & entry : cache.memory_breakdown()) {
        breakdown_bytes += entry.second;
    }
    require(breakdown_bytes >= cache.backend_tensor_bytes(), "KVarN memory breakdown covers backend tensors");
    require(cache.seq_pos_min(0) == -1, "empty sequence min");

    llama_ubatch ubatch = make_test_ubatch(3, 0);
    llama_kv_cache_kvarn::slot_info sinfo;
    sinfo.idxs = { 0, 1, 2 };
    cache.apply_ubatch(sinfo, ubatch);

    require(cache.seq_pos_min(0) == 0, "KVarN sequence min after apply");
    require(cache.seq_pos_max(0) == 2, "KVarN sequence max after apply");

    cache.seq_cp(0, 1, 1, 3);
    require(cache.seq_pos_min(1) == 1, "KVarN copied sequence min");
    require(cache.seq_pos_max(1) == 2, "KVarN copied sequence max");

    cache.seq_rm(0, 0, 2);
    require(cache.seq_pos_min(0) == 2, "KVarN sequence remove");

    cache.seq_keep(1);
    require(cache.seq_pos_min(0) == -1, "KVarN sequence keep removed old sequence");
    require(cache.seq_pos_max(1) == 2, "KVarN sequence keep preserved target");

    cache.clear(true);
    require(cache.seq_pos_min(1) == -1, "KVarN clear metadata");

    llama_memory_context_ptr full_ctx = cache.init_full();
    require(full_ctx != nullptr, "KVarN init_full returns context");
    require(full_ctx->get_status() == LLAMA_MEMORY_STATUS_SUCCESS, "KVarN init_full success status");
    require(dynamic_cast<llama_kv_cache_kvarn_context *>(full_ctx.get()) != nullptr, "KVarN init_full context type");

    llama_hparams hparams256 = make_test_hparams(256);
    llama_kv_cache_kvarn cache256(nullptr, hparams256, params, false, 16, 4, 1, nullptr);
    const llama_kvarn_layer_view view256 = cache256.get_layer_view(0);
    require(view256.head_dim_k == 256, "256-dim KVarN layer view K head dim");
    require(view256.head_dim_v == 256, "256-dim KVarN layer view V head dim");
    require(view256.layout_k.k_body_bytes == 16384, "256-dim KVarN layer view K body bytes");
    require(view256.layout_v.v_body_bytes == 8192, "256-dim KVarN layer view V body bytes");
    require(view256.scales_k->ne[0] == 640, "256-dim KVarN layer view scale K shape");
    require(view256.scales_v->ne[0] == 512, "256-dim KVarN layer view scale V shape");
    {
        const size_t tile = size_t(256)*128;
        const size_t per_pipeline = tile + 2*256 + 256 + 128 + 1;
        const size_t expected = 2*tile + 2*per_pipeline;
        require(cache256.body_store_scratch_floats(0) == expected, "256-dim KVarN body store scratch floats");
    }

    llama_hparams hparams512 = make_test_hparams(512);
    llama_kv_cache_kvarn cache512(nullptr, hparams512, params, false, 16, 4, 1, nullptr);
    const llama_kvarn_layer_view view512 = cache512.get_layer_view(0);
    require(view512.head_dim_k == 512, "512-dim KVarN layer view K head dim");
    require(view512.head_dim_v == 512, "512-dim KVarN layer view V head dim");
    require(view512.layout_k.k_body_bytes == 32768, "512-dim KVarN layer view K body bytes");
    require(view512.layout_v.v_body_bytes == 16384, "512-dim KVarN layer view V body bytes");
    require(view512.scales_k->ne[0] == 1152, "512-dim KVarN layer view scale K shape");
    require(view512.scales_v->ne[0] == 768, "512-dim KVarN layer view scale V shape");
  {
        const size_t tile = size_t(512)*128;
        const size_t per_pipeline = tile + 2*512 + 512 + 128 + 1;
        const size_t expected = 2*tile + 2*per_pipeline;
        require(cache512.body_store_scratch_floats(0) == expected, "512-dim KVarN body store scratch floats");
    }

    llama_hparams hparams_reuse = make_test_hparams(512);
    hparams_reuse.n_layer_kv_from_start = 1;
    llama_kv_cache_kvarn cache_reuse(
            nullptr, hparams_reuse, params, false, 16, 4, 1, nullptr,
            [](int32_t il) { return il == 1 ? 0 : -1; });
    require(cache_reuse.get_n_layer() == 1, "KVarN reuse cache allocates one physical KV layer");
    const llama_kvarn_layer_view view_reuse0 = cache_reuse.get_layer_view(0);
    const llama_kvarn_layer_view view_reuse1 = cache_reuse.get_layer_view(1);
    require(view_reuse1.il == 0, "KVarN reuse layer view maps to physical source layer");
    require(view_reuse1.sink_tail_k == view_reuse0.sink_tail_k, "KVarN reuse layer shares sink K storage");
    require(view_reuse1.body_k == view_reuse0.body_k, "KVarN reuse layer shares body K storage");
  {
        const size_t tile = size_t(512)*128;
        const size_t per_pipeline = tile + 2*512 + 512 + 128 + 1;
        const size_t expected = 2*tile + 2*per_pipeline;
        require(cache_reuse.body_store_scratch_floats(1) == expected, "KVarN reuse body scratch uses logical layer head dim");
    }

    llama_hparams hparams_asym = make_test_hparams(256);
    hparams_asym.n_embd_head_v_full = 128;
    hparams_asym.n_embd_head_v_swa = 128;
    bool rejected_asym = false;
    try {
        llama_kv_cache_kvarn cache_asym(nullptr, hparams_asym, params, false, 16, 4, 1, nullptr);
    } catch (const std::invalid_argument & e) {
        rejected_asym = std::strstr(e.what(), "equal K and V head dimensions") != nullptr;
    }
    require(rejected_asym, "KVarN rejects asymmetric K/V head dimensions");

}

static void test_runtime_storage_sealing() {
    llama_kvarn_params params = llama_kvarn_default_params();
    params.group_size = 4;
    params.sink_tokens = 2;
    params.tail_tokens = 2;
    params.sinkhorn_iters = 2;

    llama_hparams hparams = make_small_storage_hparams();
    llama_kv_cache_kvarn cache(nullptr, hparams, params, false, 32, 4, 1, nullptr);

    const uint32_t n_tokens = 9;
    const uint32_t n_heads = 2;
    const uint32_t head_dim = 8;
    std::vector<float> k_tokens(size_t(n_tokens)*n_heads*head_dim);
    std::vector<float> v_tokens(k_tokens.size());

    for (uint32_t t = 0; t < n_tokens; ++t) {
        for (uint32_t h = 0; h < n_heads; ++h) {
            for (uint32_t d = 0; d < head_dim; ++d) {
                const size_t off = (size_t(t)*n_heads + h)*head_dim + d;
                k_tokens[off] = 0.01f*float(t + 1) + 0.10f*float(h) + 0.001f*float(d);
                v_tokens[off] = 0.02f*float(t + 1) - 0.07f*float(h) - 0.002f*float(d);
            }
        }
    }

    cache.append_layer_tokens_reference(0, k_tokens, v_tokens, n_tokens);

    const llama_kvarn_runtime_storage_stats stats = cache.storage_stats();
    require(stats.n_layers == 2, "runtime storage layer count");
    require(stats.n_heads == 4, "runtime storage head count");
    require(stats.n_tokens == n_tokens, "runtime storage token count");
    require(stats.n_body_records == 2, "runtime storage sealed body records");
    require(stats.n_pending_body == 2, "runtime storage pending body count");
    require(stats.n_sink == 4, "runtime storage sink tokens across heads");
    require(stats.n_tail == 4, "runtime storage tail tokens across heads");
    require(stats.fp16_sink_tail_values == 128, "runtime storage fp16 sink/tail values");
    require(stats.body_packed_bytes == 48, "runtime storage packed body bytes");
    require(stats.scale_values == 72, "runtime storage scale values");
    require(cache.backend_tensor_bytes() >= stats.fp16_sink_tail_values*sizeof(uint16_t) + stats.body_packed_bytes,
            "runtime backend tensor capacity covers materialized storage");

    std::vector<float> k_mat;
    std::vector<float> v_mat;
    cache.materialize_layer_tokens_reference(0, k_mat, v_mat);
    require(k_mat.size() == k_tokens.size(), "runtime materialized K size");
    require(v_mat.size() == v_tokens.size(), "runtime materialized V size");

    require(std::fabs(k_mat[0] - k_tokens[0]) < 1.0e-4f, "runtime materialized first sink K");
    require(std::fabs(v_mat[0] - v_tokens[0]) < 1.0e-4f, "runtime materialized first sink V");

    const size_t tail_off = (size_t(n_tokens - 1)*n_heads + 1)*head_dim + 7;
    require(std::fabs(k_mat[tail_off] - k_tokens[tail_off]) < 1.0e-4f, "runtime materialized tail K");
    require(std::fabs(v_mat[tail_off] - v_tokens[tail_off]) < 1.0e-4f, "runtime materialized tail V");

    bool body_nonzero = false;
    const size_t body_begin = size_t(params.sink_tokens)*n_heads*head_dim;
    const size_t body_end = size_t(params.sink_tokens + params.group_size)*n_heads*head_dim;
    for (size_t i = body_begin; i < body_end; ++i) {
        body_nonzero = body_nonzero || std::fabs(k_mat[i]) > 0.0f || std::fabs(v_mat[i]) > 0.0f;
    }
    require(body_nonzero, "runtime materialized quantized body data");

    cache.clear(true);
    const llama_kvarn_runtime_storage_stats empty = cache.storage_stats();
    require(empty.n_tokens == 0, "runtime storage clear tokens");
    require(empty.n_body_records == 0, "runtime storage clear bodies");
}

static void test_runtime_sink_tail_graph_api() {
    llama_kvarn_params params = llama_kvarn_default_params();
    params.group_size = 4;
    params.sink_tokens = 2;
    params.tail_tokens = 4;

    llama_hparams hparams = make_test_hparams();
    hparams.n_head_kv_arr[0] = 2;
    hparams.n_head_arr[0] = 2;
    llama_kv_cache_kvarn cache(nullptr, hparams, params, false, 8, 4, 1, nullptr);

    llama_ubatch ubatch = make_test_ubatch(5, 0);
    ubatch.data->pos = { 0, 1, 2, 3, 6 };
    ubatch.pos = ubatch.data->pos.data();

    ggml_init_params init_params = {
        /*.mem_size   =*/ 16*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx { ggml_init(init_params) };

    ggml_tensor * idxs = cache.build_input_sink_tail_idxs(ctx.get(), ubatch);
    ggml_tensor * k_cur = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 128, 2, ubatch.n_tokens);
    ggml_tensor * v_cur = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 128, 2, ubatch.n_tokens);

    ggml_tensor * k_store = cache.cpy_sink_tail_k(ctx.get(), k_cur, idxs, 0);
    ggml_tensor * v_store = cache.cpy_sink_tail_v(ctx.get(), v_cur, idxs, 0);
    require(k_store->op == GGML_OP_SET_ROWS, "KVarN sink/tail K store op");
    require(v_store->op == GGML_OP_SET_ROWS, "KVarN sink/tail V store op");
    require(k_store->ne[0] == 256, "KVarN sink/tail K store row width");
    require(v_store->ne[0] == 256, "KVarN sink/tail V store row width");
    require(k_store->ne[1] == 6, "KVarN sink/tail K slot count");
    require(v_store->ne[1] == 6, "KVarN sink/tail V slot count");

    ggml_backend_buffer_ptr buf {
        ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type())
    };
    require(buf != nullptr, "KVarN sink/tail graph test buffer");

    cache.set_input_sink_tail_idxs(idxs, &ubatch);
    const int64_t * idx_data = (const int64_t *) idxs->data;
    require(idx_data[0] == 0, "KVarN sink slot index 0");
    require(idx_data[1] == 1, "KVarN sink slot index 1");
    require(idx_data[2] == 2, "KVarN tail slot index 2");
    require(idx_data[3] == 3, "KVarN tail slot index 3");
    require(idx_data[4] == 2, "KVarN tail slot wraps");
}

static void test_runtime_body_plan_graph_api() {
    llama_kvarn_params params = llama_kvarn_default_params();
    params.group_size = 4;
    params.sink_tokens = 2;
    params.tail_tokens = 4;

    llama_hparams hparams = make_test_hparams();
    llama_kv_cache_kvarn cache(nullptr, hparams, params, false, 16, 4, 1, nullptr);

    llama_ubatch ubatch = make_test_ubatch(9, 0);
    ubatch.data->pos = { 0, 1, 2, 3, 5, 6, 7, 8, 9 };
    ubatch.pos = ubatch.data->pos.data();

    ggml_init_params init_params = {
        /*.mem_size   =*/ 8*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx { ggml_init(init_params) };

    ggml_tensor * plan = cache.build_input_body_plan(ctx.get(), ubatch);
    ggml_tensor * offsets = cache.build_input_body_offsets(ctx.get(), ubatch);
    ggml_tensor * tail_idxs = cache.build_input_tail_evict_idxs(ctx.get(), ubatch);
    require(plan->type == GGML_TYPE_I64, "KVarN body plan type");
    require(plan->ne[0] == 3, "KVarN body plan fields");
    require(plan->ne[1] == 4, "KVarN body plan evictions");
    require(offsets->type == GGML_TYPE_I64, "KVarN body offsets type");
    require(offsets->ne[0] == 4, "KVarN body offsets evictions");
    require(tail_idxs->type == GGML_TYPE_I32, "KVarN tail eviction index type");
    require(tail_idxs->ne[0] == 4, "KVarN tail eviction index count");

    ggml_backend_buffer_ptr buf {
        ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type())
    };
    require(buf != nullptr, "KVarN body plan graph test buffer");

    cache.set_input_body_plan(plan, &ubatch);
    cache.set_input_body_offsets(offsets, &ubatch);
    cache.set_input_tail_evict_idxs(tail_idxs, &ubatch);
    const int64_t * data = (const int64_t *) plan->data;
    const int64_t * offset_data = (const int64_t *) offsets->data;
    const int32_t * tail_idx_data = (const int32_t *) tail_idxs->data;

    const int64_t expected[][3] = {
        {  0,  0, -1 },
        {  0,  1, -1 },
        {  0,  2, -1 },
        {  0,  3,  0 },
    };
    const int32_t expected_tail_idxs[] = {
        2, 3, 4, 5,
    };

    for (uint32_t i = 0; i < 4; ++i) {
        const size_t off = size_t(i)*3;
        require(data[off + 0] == expected[i][0], "KVarN body plan record index");
        require(data[off + 1] == expected[i][1], "KVarN body plan token offset");
        require(data[off + 2] == expected[i][2], "KVarN body plan seal record");
        require(offset_data[i] == expected[i][1], "KVarN body offset value");
        require(tail_idx_data[i] == expected_tail_idxs[i], "KVarN tail eviction slot index");
    }
}

static void test_runtime_body_plan_multi_record_seals() {
    llama_kvarn_params params = llama_kvarn_default_params();
    params.group_size = 4;
    params.sink_tokens = 2;
    params.tail_tokens = 2;

    llama_hparams hparams = make_small_storage_hparams();
    llama_kv_cache_kvarn cache(nullptr, hparams, params, false, 32, 4, 1, nullptr);

    llama_ubatch ubatch = make_test_ubatch(7, 0);
    ubatch.data->pos = { 5, 6, 7, 8, 9, 10, 11 };
    ubatch.pos = ubatch.data->pos.data();

    ggml_init_params init_params = {
        /*.mem_size   =*/ 8*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx { ggml_init(init_params) };

    ggml_tensor * plan = cache.build_input_body_plan(ctx.get(), ubatch);
    ggml_tensor * offsets = cache.build_input_body_offsets(ctx.get(), ubatch);
    ggml_tensor * tail_idxs = cache.build_input_tail_evict_idxs(ctx.get(), ubatch);
    require(plan->ne[0] == 3 && plan->ne[1] == 7, "KVarN multi-record body plan shape");
    require(offsets->ne[0] == 7, "KVarN multi-record body offsets shape");
    require(tail_idxs->ne[0] == 7, "KVarN multi-record tail idx shape");

    ggml_backend_buffer_ptr buf {
        ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type())
    };
    require(buf != nullptr, "KVarN multi-record body plan buffer");

    cache.set_input_body_plan(plan, &ubatch);
    cache.set_input_body_offsets(offsets, &ubatch);
    cache.set_input_tail_evict_idxs(tail_idxs, &ubatch);

    const int64_t * data = (const int64_t *) plan->data;
    const int64_t * offset_data = (const int64_t *) offsets->data;
    const int32_t * tail_idx_data = (const int32_t *) tail_idxs->data;

    const int64_t expected[][3] = {
        { 0, 1, -1 },
        { 0, 2, -1 },
        { 0, 3,  0 },
        { 1, 0, -1 },
        { 1, 1, -1 },
        { 1, 2, -1 },
        { 1, 3,  1 },
    };
    const int64_t expected_offsets[] = {
        1, 2, 3, 0, 1, 2, 3,
    };
    const int32_t expected_tail_idxs[] = {
        3, 2, 3, 2, 3, 2, 3,
    };

    for (uint32_t i = 0; i < 7; ++i) {
        const size_t off = size_t(i)*3;
        require(data[off + 0] == expected[i][0], "KVarN multi-record body plan record");
        require(data[off + 1] == expected[i][1], "KVarN multi-record body plan offset");
        require(data[off + 2] == expected[i][2], "KVarN multi-record body plan seal");
        require(offset_data[i] == expected_offsets[i], "KVarN multi-record body offset");
        require(tail_idx_data[i] == expected_tail_idxs[i], "KVarN multi-record tail idx");
    }
}

static float read_kq_mask_value(const ggml_tensor * mask, int64_t t, int64_t q) {
    const char * p = (const char *) mask->data + size_t(q)*mask->nb[1] + size_t(t)*mask->nb[0];
    if (mask->type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(*(const ggml_fp16_t *) p);
    }
    return *(const float *) p;
}

static void require_mask_keep(float v, const char * msg) {
    require(std::fabs(v) < 1.0e-6f, msg);
}

static void require_mask_drop(float v, const char * msg) {
    require(std::isinf(v) && v < 0.0f, msg);
}

static void test_runtime_kq_mask_graph_api() {
    llama_kvarn_params params = llama_kvarn_default_params();
    params.group_size = 4;
    params.sink_tokens = 2;
    params.tail_tokens = 2;

    llama_hparams hparams = make_small_storage_hparams();
    llama_kv_cache_kvarn cache(nullptr, hparams, params, false, 12, 4, 1, nullptr);

    llama_ubatch ubatch = make_test_ubatch(5, 0);
    ubatch.data->pos = { 0, 1, 2, 5, 6 };
    ubatch.pos = ubatch.data->pos.data();

    ggml_init_params init_params = {
        /*.mem_size   =*/ 8*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx { ggml_init(init_params) };

    ggml_tensor * causal_mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cache.get_size(), ubatch.n_tokens);
    ggml_tensor * noncausal_mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cache.get_size(), ubatch.n_tokens);
    ggml_tensor * causal_mask_f16 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, cache.get_size(), ubatch.n_tokens);

    ggml_backend_buffer_ptr buf {
        ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type())
    };
    require(buf != nullptr, "KVarN KQ mask graph test buffer");

    cache.set_input_kq_mask(causal_mask, &ubatch, true);
    cache.set_input_kq_mask(noncausal_mask, &ubatch, false);
    cache.set_input_kq_mask(causal_mask_f16, &ubatch, true);

    const llama_pos expected_pos[] = { 0, 1, 2, 5, 6 };
    for (uint32_t q = 0; q < ubatch.n_tokens; ++q) {
        const int64_t visible = std::min<int64_t>(cache.get_size(), int64_t(expected_pos[q]) + 1);
        for (int64_t t = 0; t < int64_t(cache.get_size()); ++t) {
            const bool keep = t < visible;
            if (keep) {
                require_mask_keep(read_kq_mask_value(causal_mask, t, q), "KVarN causal KQ mask keeps visible slot");
                require_mask_keep(read_kq_mask_value(causal_mask_f16, t, q), "KVarN causal F16 KQ mask keeps visible slot");
            } else {
                require_mask_drop(read_kq_mask_value(causal_mask, t, q), "KVarN causal KQ mask drops future slot");
                require_mask_drop(read_kq_mask_value(causal_mask_f16, t, q), "KVarN causal F16 KQ mask drops future slot");
            }
            require_mask_keep(read_kq_mask_value(noncausal_mask, t, q), "KVarN non-causal KQ mask keeps all slots");
        }
    }
}

static void test_runtime_body_record_graph_api() {
    llama_kvarn_params params = llama_kvarn_default_params();
    params.group_size = 4;
    params.sink_tokens = 2;
    params.tail_tokens = 2;
    params.sinkhorn_iters = 2;

    llama_hparams hparams = make_small_storage_hparams();
    llama_kv_cache_kvarn cache(nullptr, hparams, params, false, 32, 4, 1, nullptr);
    const llama_kvarn_layer_view view = cache.get_layer_view(0);

    require(view.n_records == 7, "KVarN body record view count");
    require(view.layout_k.k_body_bytes == 16, "KVarN K body record bytes");
    require(view.layout_v.v_body_bytes == 8, "KVarN V body record bytes");
    require(view.layout_k.k_scale_floats == 20, "KVarN K scale record floats");
    require(view.layout_v.v_scale_floats == 16, "KVarN V scale record floats");
    {
        const size_t tile = size_t(view.head_dim_k)*params.group_size;
        const size_t per_pipeline = tile + 2*std::max<uint32_t>(view.head_dim_k, params.group_size) +
            view.head_dim_k + params.group_size + 1;
        const size_t expected = 2*tile + per_pipeline;
        require(cache.body_store_scratch_floats(0) == expected, "KVarN small multi-head body store scratch floats");
    }

    ggml_init_params init_params = {
        /*.mem_size   =*/ 64*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx { ggml_init(init_params) };

    ggml_tensor * k_tile = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, params.group_size, view.head_dim_k);
    ggml_tensor * v_tile = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, view.head_dim_v, params.group_size);
    ggml_tensor * body_offsets = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, params.group_size);
    ggml_tensor * scratch = cache.build_body_store_scratch(ctx.get(), 0);

    const uint32_t ih = 1;
    const uint32_t record = 2;
    ggml_tensor * k_body = cache.view_k_body_record(ctx.get(), 0, ih, record);
    ggml_tensor * v_body = cache.view_v_body_record(ctx.get(), 0, ih, record);
    ggml_tensor * k_scales = cache.view_k_scales_record(ctx.get(), 0, ih, record);
    ggml_tensor * v_scales = cache.view_v_scales_record(ctx.get(), 0, ih, record);

    require(k_body->type == GGML_TYPE_I8 && k_body->ne[0] == (int64_t) view.layout_k.k_body_bytes,
            "KVarN K body record view shape");
    require(v_body->type == GGML_TYPE_I8 && v_body->ne[0] == (int64_t) view.layout_v.v_body_bytes,
            "KVarN V body record view shape");
    require(k_scales->type == GGML_TYPE_F32 && k_scales->ne[0] == (int64_t) view.layout_k.k_scale_floats,
            "KVarN K scale record view shape");
    require(v_scales->type == GGML_TYPE_F32 && v_scales->ne[0] == (int64_t) view.layout_v.v_scale_floats,
            "KVarN V scale record view shape");
    require(k_body->view_src == view.body_k, "KVarN K body record view source");
    require(v_body->view_src == view.body_v, "KVarN V body record view source");
    require(k_scales->view_src == view.scales_k, "KVarN K scale record view source");
    require(v_scales->view_src == view.scales_v, "KVarN V scale record view source");
    require(k_body->view_offs == size_t(ih)*view.body_k->nb[2] + size_t(record)*view.body_k->nb[1],
            "KVarN K body record view offset");
    require(v_body->view_offs == size_t(ih)*view.body_v->nb[2] + size_t(record)*view.body_v->nb[1],
            "KVarN V body record view offset");
    require(k_scales->view_offs == size_t(ih)*view.scales_k->nb[2] + size_t(record)*view.scales_k->nb[1],
            "KVarN K scale record view offset");
    require(v_scales->view_offs == size_t(ih)*view.scales_v->nb[2] + size_t(record)*view.scales_v->nb[1],
            "KVarN V scale record view offset");

    ggml_tensor * k_store = cache.store_k_body_record(ctx.get(), k_tile, scratch, 0, ih, record);
    ggml_tensor * v_store = cache.store_v_body_record(ctx.get(), v_tile, scratch, 0, ih, record);
    ggml_tensor * kv_store = cache.store_kv_body_record(ctx.get(), k_tile, v_tile, scratch, 0, ih, record);
    ggml_tensor * k_all_heads = ggml_new_tensor_3d(
            ctx.get(), GGML_TYPE_F32, view.head_dim_k, view.n_head_kv, params.group_size);
    ggml_tensor * v_all_heads = ggml_new_tensor_3d(
            ctx.get(), GGML_TYPE_F32, view.head_dim_v, view.n_head_kv, params.group_size);
    ggml_tensor * kv_all_heads = cache.store_kv_body_all_heads(ctx.get(), k_all_heads, v_all_heads, scratch, 0, record);

    require(k_store->op == GGML_OP_KVARN_STORE_BODY, "KVarN K body record store op");
    require(v_store->op == GGML_OP_KVARN_STORE_BODY, "KVarN V body record store op");
    require(kv_store->op == GGML_OP_KVARN_STORE_KV_BODY, "KVarN fused KV body record store op");
    require(kv_all_heads->op == GGML_OP_KVARN_STORE_KV_BODY, "KVarN fused KV all-head body record store op");
    require(k_store->src[0] == k_tile && k_store->src[1]->view_src == view.scales_k &&
            k_store->src[2] == scratch && k_store->src[3]->view_src == view.body_k,
            "KVarN K body record store sources");
    require(v_store->src[0] == v_tile && v_store->src[1]->view_src == view.scales_v &&
            v_store->src[2] == scratch && v_store->src[3]->view_src == view.body_v,
            "KVarN V body record store sources");
    require(kv_store->src[0] == k_tile && kv_store->src[1] == v_tile &&
            kv_store->src[2]->view_src == view.scales_k && kv_store->src[3]->view_src == view.scales_v &&
            kv_store->src[4] == scratch && kv_store->src[5]->view_src == view.body_k &&
            kv_store->src[6]->view_src == view.body_v,
            "KVarN fused KV body record store sources");
    require(k_store->op_params[1] == (int32_t) view.head_dim_k &&
            k_store->op_params[2] == (int32_t) params.group_size &&
            k_store->op_params[3] == (int32_t) params.key_bits,
            "KVarN K body record store params");
    require(v_store->op_params[1] == (int32_t) view.head_dim_v &&
            v_store->op_params[3] == (int32_t) params.value_bits &&
            v_store->op_params[4] == (int32_t) params.sinkhorn_iters,
            "KVarN V body record store params");
    require(kv_store->op_params[0] == (int32_t) view.head_dim_k &&
            kv_store->op_params[1] == (int32_t) params.group_size &&
            kv_store->op_params[2] == (int32_t) params.key_bits &&
            kv_store->op_params[3] == (int32_t) params.value_bits &&
            kv_store->op_params[4] == (int32_t) params.sinkhorn_iters,
            "KVarN fused KV body record store params");

    ggml_tensor * tail_idxs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, params.group_size);
    ggml_tensor * k_pending = cache.cpy_tail_evict_pending_k(ctx.get(), tail_idxs, body_offsets, 0);
    ggml_tensor * v_pending = cache.cpy_tail_evict_pending_v(ctx.get(), tail_idxs, body_offsets, 0);
    ggml_tensor * k_from_pending = cache.store_k_body_record_from_pending(ctx.get(), scratch, 0, ih, record);
    ggml_tensor * v_from_pending = cache.store_v_body_record_from_pending(ctx.get(), scratch, 0, ih, record);
    ggml_tensor * kv_from_pending = cache.store_kv_body_record_from_pending(ctx.get(), scratch, 0, ih, record);

    require(k_pending->op == GGML_OP_SET_ROWS, "KVarN pending K store op");
    require(v_pending->op == GGML_OP_SET_ROWS, "KVarN pending V store op");
    require(k_from_pending->op == GGML_OP_KVARN_STORE_BODY, "KVarN pending K body store op");
    require(v_from_pending->op == GGML_OP_KVARN_STORE_BODY, "KVarN pending V body store op");
    require(kv_from_pending->op == GGML_OP_KVARN_STORE_KV_BODY, "KVarN pending fused KV body store op");
    require(k_from_pending->src[0]->type == GGML_TYPE_F32 && k_from_pending->src[0]->ne[0] == (int64_t) params.group_size,
            "KVarN pending K body tile shape");
    require(v_from_pending->src[0]->type == GGML_TYPE_F32 && v_from_pending->src[0]->ne[0] == (int64_t) view.head_dim_v,
            "KVarN pending V body tile shape");
    require(kv_from_pending->src[0]->type == GGML_TYPE_F32 && kv_from_pending->src[0]->ne[0] == (int64_t) params.group_size &&
            kv_from_pending->src[1]->type == GGML_TYPE_F32 && kv_from_pending->src[1]->ne[0] == (int64_t) view.head_dim_v,
            "KVarN pending fused KV body tile shapes");

    bool rejected_multi_record_pending = false;
    try {
        std::vector<uint32_t> records = { 0, 1 };
        (void) cache.store_kv_body_records_from_pending(ctx.get(), scratch, 0, records);
    } catch (const std::invalid_argument &) {
        rejected_multi_record_pending = true;
    }
    require(rejected_multi_record_pending,
            "KVarN pending fused KV body store rejects unsafe multi-record batches");
}

static void test_kvarn_store_body_ggml_ops() {
    llama_kvarn_params params = llama_kvarn_default_params();
    params.rtn_quantile = 0.75f;

    ggml_init_params init_params = {
        /*.mem_size   =*/ 512*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx { ggml_init(init_params) };

    for (const int32_t head_dim : { 128, 256, 512 }) {
        llama_kvarn_layout layout = llama_kvarn_make_layout(params, head_dim);

        ggml_tensor * k_tile = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, params.group_size, head_dim);
        ggml_tensor * v_tile = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, head_dim, params.group_size);
        ggml_tensor * k_body = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, layout.k_body_bytes);
        ggml_tensor * v_body = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, layout.v_body_bytes);
        ggml_tensor * k_scales = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, layout.k_scale_floats);
        ggml_tensor * v_scales = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, layout.v_scale_floats);
        ggml_tensor * scratch = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, head_dim*params.group_size + 2*std::max<int32_t>(head_dim, params.group_size));

        ggml_tensor * k_store = ggml_kvarn_store_k_body(
                ctx.get(), k_tile, k_body, k_scales, scratch,
                head_dim, params.group_size, params.key_bits, params.sinkhorn_iters, params.rtn_quantile);
        ggml_tensor * v_store = ggml_kvarn_store_v_body(
                ctx.get(), v_tile, v_body, v_scales, scratch,
                head_dim, params.group_size, params.value_bits, params.sinkhorn_iters, params.rtn_quantile);

        require(k_store->op == GGML_OP_KVARN_STORE_BODY, "KVarN K body store op");
        require(v_store->op == GGML_OP_KVARN_STORE_BODY, "KVarN V body store op");
        require(k_store->type == GGML_TYPE_I8, "KVarN K body store result type");
        require(v_store->type == GGML_TYPE_I8, "KVarN V body store result type");
        require(k_store->view_src == k_body, "KVarN K body store returns body view");
        require(v_store->view_src == v_body, "KVarN V body store returns body view");
        require(k_store->src[0] == k_tile && k_store->src[1] == k_scales && k_store->src[2] == scratch && k_store->src[3] == k_body,
                "KVarN K body store sources");
        require(v_store->src[0] == v_tile && v_store->src[1] == v_scales && v_store->src[2] == scratch && v_store->src[3] == v_body,
                "KVarN V body store sources");
        require(k_store->op_params[0] == 0, "KVarN K body store mode");
        require(v_store->op_params[0] == 1, "KVarN V body store mode");
        require(k_store->op_params[1] == head_dim && k_store->op_params[2] == (int32_t) params.group_size,
                "KVarN K body store geometry");
        require(v_store->op_params[3] == (int32_t) params.value_bits && v_store->op_params[4] == (int32_t) params.sinkhorn_iters,
                "KVarN V body store params");
        float k_quantile = 0.0f;
        float v_quantile = 0.0f;
        std::memcpy(&k_quantile, &k_store->op_params[5], sizeof(float));
        std::memcpy(&v_quantile, &v_store->op_params[5], sizeof(float));
        require(std::fabs(k_quantile - params.rtn_quantile) < 1.0e-6f &&
                std::fabs(v_quantile - params.rtn_quantile) < 1.0e-6f,
                "KVarN body store quantile params");
    }
}

static void test_kvarn_mixed_attention_ggml_op() {
    llama_kvarn_params params = llama_kvarn_default_params();
    params.group_size = 4;
    params.sink_tokens = 2;
    params.tail_tokens = 2;
    const int32_t head_dim = 8;
    const int32_t n_head = 4;
    const int32_t n_head_kv = 2;
    const int32_t n_records = 3;
    const int32_t n_pending = 2;
    const int32_t n_tokens = 5;
    const llama_kvarn_layout layout = llama_kvarn_make_layout(params, head_dim);

    ggml_init_params init_params = {
        /*.mem_size   =*/ 128*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx { ggml_init(init_params) };

    ggml_tensor * q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, head_dim, n_head, n_tokens);
    ggml_tensor * sink_tail_k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, head_dim, n_head_kv, params.sink_tokens + params.tail_tokens);
    ggml_tensor * sink_tail_v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, head_dim, n_head_kv, params.sink_tokens + params.tail_tokens);
    ggml_tensor * body_k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I8, layout.k_body_bytes, n_records, n_head_kv);
    ggml_tensor * body_v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I8, layout.v_body_bytes, n_records, n_head_kv);
    ggml_tensor * scales_k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, layout.k_scale_floats, n_records, n_head_kv);
    ggml_tensor * scales_v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, layout.v_scale_floats, n_records, n_head_kv);
    ggml_tensor * pending_k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, head_dim, n_head_kv, params.group_size);
    ggml_tensor * pending_v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, head_dim, n_head_kv, params.group_size);
    ggml_tensor * scratch = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, params.sink_tokens + n_records*params.group_size + n_pending + params.tail_tokens);
    ggml_tensor * kq_mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32,
            params.sink_tokens + n_records*params.group_size + n_pending + params.tail_tokens, n_tokens);

    ggml_tensor * out = ggml_kvarn_attn_mixed(
            ctx.get(), q, sink_tail_k, sink_tail_v, body_k, body_v, scales_k, scales_v, pending_k, pending_v, scratch,
            kq_mask,
            params.sink_tokens, n_records, n_pending, params.tail_tokens, 1, head_dim, params.group_size,
            params.key_bits, params.value_bits, 1.0f/std::sqrt(float(head_dim)));

    require(out->op == GGML_OP_KVARN_ATTN_MIXED, "KVarN mixed attention op");
    require(out->type == GGML_TYPE_F32, "KVarN mixed attention result type");
    require(out->ne[0] == head_dim && out->ne[1] == n_head && out->ne[2] == n_tokens,
            "KVarN mixed attention result shape");
    require(out->src[0] == q && out->src[1] == sink_tail_k && out->src[2] == sink_tail_v,
            "KVarN mixed attention primary sources");
    require(out->src[3] == body_k && out->src[4] == body_v && out->src[5] == scales_k &&
            out->src[6] == scales_v && out->src[7] == pending_k && out->src[8] == pending_v &&
            out->src[9] == scratch,
            "KVarN mixed attention cache sources");
    require(out->src[10] == kq_mask, "KVarN mixed attention mask source");
    require(out->op_params[0] == (int32_t) params.sink_tokens &&
            out->op_params[1] == n_records &&
            out->op_params[2] == n_pending &&
            out->op_params[3] == (int32_t) params.tail_tokens &&
            out->op_params[4] == 1,
            "KVarN mixed attention token params");
    require(out->op_params[5] == head_dim &&
            out->op_params[6] == (int32_t) params.group_size &&
            out->op_params[7] == (int32_t) params.key_bits &&
            out->op_params[8] == (int32_t) params.value_bits,
            "KVarN mixed attention layout params");
}

static void test_cpu_backend_rejects_kvarn_ops() {
    llama_kvarn_params params = llama_kvarn_default_params();
    llama_kvarn_layout layout = llama_kvarn_make_layout(params, 128);

    ggml_init_params init_params = {
        /*.mem_size   =*/ 256*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx { ggml_init(init_params) };

    ggml_tensor * k_tile = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, params.group_size, 128);
    ggml_tensor * k_body = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, layout.k_body_bytes);
    ggml_tensor * k_scales = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, layout.k_scale_floats);
    ggml_tensor * scratch = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 128*params.group_size + 2*128);

    ggml_tensor * k_store = ggml_kvarn_store_k_body(
            ctx.get(), k_tile, k_body, k_scales, scratch,
            128, params.group_size, params.key_bits, params.sinkhorn_iters, params.rtn_quantile);

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    require(cpu_dev != nullptr, "CPU backend device is registered");
    ggml_backend_reg_t cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
    require(cpu_reg != nullptr, "CPU backend registry is registered");
    require(std::strcmp(ggml_backend_reg_name(cpu_reg), "CUDA") != 0,
            "CPU backend registry is not CUDA");
    require(!ggml_backend_dev_supports_op(cpu_dev, k_store),
            "CPU backend rejects KVarN body-store op");

    ggml_tensor * q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 128, 1, 1);
    ggml_tensor * sink_tail_k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, 128, 1, params.sink_tokens + params.tail_tokens);
    ggml_tensor * sink_tail_v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, 128, 1, params.sink_tokens + params.tail_tokens);
    ggml_tensor * body_k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I8, layout.k_body_bytes, 1, 1);
    ggml_tensor * body_v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I8, layout.v_body_bytes, 1, 1);
    ggml_tensor * scales_k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, layout.k_scale_floats, 1, 1);
    ggml_tensor * scales_v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, layout.v_scale_floats, 1, 1);
    ggml_tensor * pending_k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 128, 1, params.group_size);
    ggml_tensor * pending_v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 128, 1, params.group_size);
    ggml_tensor * logits = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, params.sink_tokens + params.group_size + params.tail_tokens);
    ggml_tensor * mixed = ggml_kvarn_attn_mixed(
            ctx.get(), q, sink_tail_k, sink_tail_v, body_k, body_v, scales_k, scales_v, pending_k, pending_v, logits,
            nullptr,
            params.sink_tokens, 1, 0, params.tail_tokens, 1, 128, params.group_size,
            params.key_bits, params.value_bits, 1.0f/std::sqrt(128.0f));

    require(!ggml_backend_dev_supports_op(cpu_dev, mixed),
            "CPU backend rejects KVarN mixed-attention op");
}

int main() {
    trace_phase("llama_backend_init", "start");
    llama_backend_init();
    trace_phase("llama_backend_init", "pass");

    run_phase("test_layout", test_layout);
    run_phase("test_pack_roundtrip", test_pack_roundtrip);
    run_phase("test_hadamard_inverse", test_hadamard_inverse);
    run_phase("test_reference_store_dequant", test_reference_store_dequant);
    run_phase("test_reference_cache_sealing", test_reference_cache_sealing);
    run_phase("test_memory_estimate", test_memory_estimate);
    run_phase("test_runtime_metadata", test_runtime_metadata);
    run_phase("test_runtime_storage_sealing", test_runtime_storage_sealing);
    run_phase("test_runtime_sink_tail_graph_api", test_runtime_sink_tail_graph_api);
    run_phase("test_runtime_body_plan_graph_api", test_runtime_body_plan_graph_api);
    run_phase("test_runtime_body_plan_multi_record_seals", test_runtime_body_plan_multi_record_seals);
    run_phase("test_runtime_kq_mask_graph_api", test_runtime_kq_mask_graph_api);
    run_phase("test_runtime_body_record_graph_api", test_runtime_body_record_graph_api);
    run_phase("test_kvarn_store_body_ggml_ops", test_kvarn_store_body_ggml_ops);
    run_phase("test_kvarn_mixed_attention_ggml_op", test_kvarn_mixed_attention_ggml_op);
    run_phase("test_cpu_backend_rejects_kvarn_ops", test_cpu_backend_rejects_kvarn_ops);

    trace_phase("llama_backend_free", "start");
    llama_backend_free();
    trace_phase("llama_backend_free", "pass");
    return 0;
}
