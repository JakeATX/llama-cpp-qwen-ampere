#include "layer-profile.h"

#include "common.h"
#include "log.h"

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <regex>
#include <string>

using clock_type = std::chrono::steady_clock;

struct common_layer_profile_user_data::impl {
    FILE * out = nullptr;
    common_layer_profile_config config;
    clock_type::time_point start;
    uint64_t node_index = 0;
    uint64_t profiled_nodes = 0;
    uint64_t skipped_warmup = 0;
    bool failed = false;
};

static std::string json_escape(const char * s) {
    std::string out;
    if (!s) {
        return out;
    }
    for (const char * p = s; *p; ++p) {
        switch (*p) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if ((unsigned char) *p < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char) *p);
                    out += buf;
                } else {
                    out += *p;
                }
        }
    }
    return out;
}

static int parse_layer_id(const ggml_tensor * t) {
    static const std::regex layer_re("blk\\.([0-9]+)");
    const ggml_tensor * candidates[3] = { t, t ? t->src[0] : nullptr, t ? t->src[1] : nullptr };
    for (const ggml_tensor * cur : candidates) {
        if (!cur || !cur->name[0]) {
            continue;
        }
        std::cmatch m;
        if (std::regex_search(cur->name, m, layer_re)) {
            return std::atoi(m[1].str().c_str());
        }
    }
    return -1;
}

static const char * op_family(const ggml_tensor * t) {
    if (!t) {
        return "unknown";
    }
    const char * desc = ggml_op_desc(t);
    const char * name = t->name;
    if (std::strstr(name, "attn") || std::strstr(name, "q_") || std::strstr(name, "k_") || std::strstr(name, "v_") ||
        std::strstr(name, "wo") || std::strstr(name, "wv") || std::strstr(name, "wk") || std::strstr(name, "wq")) {
        return "attention";
    }
    if (std::strstr(name, "ffn") || std::strstr(name, "moe") || std::strstr(desc, "MUL_MAT_ID")) {
        return "moe_ffn";
    }
    if (std::strstr(desc, "RMS_NORM") || std::strstr(desc, "NORM")) {
        return "norm";
    }
    if (std::strstr(desc, "ADD") || std::strstr(desc, "MUL")) {
        return "residual_dense";
    }
    return "other";
}

static const char * phase_from_tensor(const ggml_tensor * t) {
    if (!t) {
        return "unknown";
    }
    // Decode graphs usually have a single token in the sequence/batch axis.
    // This is a best-effort label; the offline analyzer treats it as advisory.
    if (t->ne[1] <= 1 && t->ne[2] <= 1) {
        return "decode_like";
    }
    return "prefill_like";
}

static size_t tensor_bytes_safe(const ggml_tensor * t) {
    return t ? ggml_nbytes(t) : 0;
}

common_layer_profile_user_data::common_layer_profile_user_data() : pimpl(std::make_unique<impl>()) {}

common_layer_profile_user_data::~common_layer_profile_user_data() {
    if (pimpl && pimpl->out) {
        fprintf(pimpl->out, "{\"type\":\"profile_end\",\"profiled_nodes\":%" PRIu64 ",\"skipped_warmup\":%" PRIu64 "}\n",
                pimpl->profiled_nodes, pimpl->skipped_warmup);
        fclose(pimpl->out);
        pimpl->out = nullptr;
    }
}

common_layer_profile_user_data::common_layer_profile_user_data(common_params & params, const common_layer_profile_config & config)
    : pimpl(std::make_unique<impl>()) {
    pimpl->config = config;
    pimpl->out = fopen(config.path.c_str(), "a");
    if (!pimpl->out) {
        LOG_ERR("%s: failed to open layer profile path '%s'\n", __func__, config.path.c_str());
        pimpl->failed = true;
        return;
    }
    setvbuf(pimpl->out, nullptr, _IOLBF, 0);
    fprintf(pimpl->out,
            "{\"type\":\"profile_start\",\"detail\":\"%s\",\"sync\":\"%s\",\"warmup\":%d,\"max_tokens\":%d}\n",
            json_escape(config.detail.c_str()).c_str(), json_escape(config.sync.c_str()).c_str(), config.warmup, config.max_tokens);

    params.cb_eval = common_layer_profile_cb_eval;
    params.cb_eval_user_data = this;
}

bool common_layer_profile_user_data::ok() const {
    return pimpl && !pimpl->failed && pimpl->out;
}

bool common_layer_profile_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (common_layer_profile_user_data *) user_data;
    if (!cb_data || !cb_data->pimpl || !cb_data->pimpl->out) {
        return true;
    }
    auto * p = cb_data->pimpl.get();

    if (ask) {
        if (p->config.detail == "off" || p->config.sync == "none") {
            return false;
        }
        const int layer = parse_layer_id(t);
        const bool wants_layer = layer >= 0;
        const bool detail_ops = p->config.detail == "ops";
        const bool should_profile = wants_layer || detail_ops;
        if (should_profile) {
            p->start = clock_type::now();
            return true;
        }
        return false;
    }

    const auto end = clock_type::now();
    const double elapsed_us = std::chrono::duration<double, std::micro>(end - p->start).count();
    const uint64_t node_index = p->node_index++;
    const int layer = parse_layer_id(t);

    if (p->config.warmup > 0 && (int) node_index < p->config.warmup) {
        p->skipped_warmup++;
        return true;
    }
    if (p->config.max_tokens >= 0 && (int) p->profiled_nodes >= p->config.max_tokens) {
        return true;
    }

    const ggml_tensor * src0 = t ? t->src[0] : nullptr;
    const ggml_tensor * src1 = t ? t->src[1] : nullptr;
    const char * buffer_name = (t && t->buffer) ? ggml_backend_buffer_name(t->buffer) : "";
    const char * op_desc = t ? ggml_op_desc(t) : "";

    fprintf(p->out,
            "{\"type\":\"layer_profile_node\","
            "\"node_index\":%" PRIu64 ","
            "\"layer\":%d,"
            "\"phase\":\"%s\","
            "\"op_family\":\"%s\","
            "\"op\":\"%s\","
            "\"name\":\"%s\","
            "\"src0\":\"%s\","
            "\"src1\":\"%s\","
            "\"elapsed_us\":%.3f,"
            "\"tensor_bytes\":%zu,"
            "\"src0_bytes\":%zu,"
            "\"src1_bytes\":%zu,"
            "\"buffer\":\"%s\","
            "\"ne0\":%" PRId64 ",\"ne1\":%" PRId64 ",\"ne2\":%" PRId64 ",\"ne3\":%" PRId64 "}\n",
            node_index,
            layer,
            phase_from_tensor(t),
            op_family(t),
            json_escape(op_desc).c_str(),
            json_escape(t ? t->name : "").c_str(),
            json_escape(src0 ? src0->name : "").c_str(),
            json_escape(src1 ? src1->name : "").c_str(),
            elapsed_us,
            tensor_bytes_safe(t),
            tensor_bytes_safe(src0),
            tensor_bytes_safe(src1),
            json_escape(buffer_name).c_str(),
            t ? t->ne[0] : 0,
            t ? t->ne[1] : 0,
            t ? t->ne[2] : 0,
            t ? t->ne[3] : 0);
    p->profiled_nodes++;
    return true;
}
