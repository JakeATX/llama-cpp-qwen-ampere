#include "common.h"
#include "get-model.h"
#include "server-context.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

struct logger_restore {
    ggml_log_callback callback;
    void * user_data;

    ~logger_restore() {
        llama_log_set(callback, user_data);
    }
};

int main(int argc, char ** argv) {
    char * model_path = get_model_or_exit(argc, argv);

    common_init();
    ggml_log_callback original_logger = nullptr;
    void * original_logger_user_data = nullptr;
    llama_log_get(&original_logger, &original_logger_user_data);

    bool ok = true;
    {
        logger_restore restore { original_logger, original_logger_user_data };
        std::string log;
        llama_log_set([](ggml_log_level, const char * text, void * user_data) {
            static_cast<std::string *>(user_data)->append(text);
        }, &log);

        llama_backend_init();

        struct test_case {
            const char * name;
            llama_kvarn_params kvarn;
            const char * diagnostic;
        };

        const char * preset_diagnostic =
                "KVarN backend requires group size 128 and key/value bits in [2, 8]";
        const char * parameter_diagnostic =
                "KVarN backend requires non-zero sink/tail tokens and Sinkhorn iterations, "
                "with rtn_quantile in (0, 1]";

        const llama_kvarn_params valid = llama_kvarn_default_params();
        std::vector<test_case> cases;
        cases.push_back({ "group size", valid, preset_diagnostic });
        cases.back().kvarn.group_size = 64;
        cases.push_back({ "zero sink tokens", valid, parameter_diagnostic });
        cases.back().kvarn.sink_tokens = 0;
        cases.push_back({ "zero tail tokens", valid, parameter_diagnostic });
        cases.back().kvarn.tail_tokens = 0;
        cases.push_back({ "zero Sinkhorn iterations", valid, parameter_diagnostic });
        cases.back().kvarn.sinkhorn_iters = 0;
        cases.push_back({ "zero RTN quantile", valid, parameter_diagnostic });
        cases.back().kvarn.rtn_quantile = 0.0f;
        cases.push_back({ "RTN quantile above one", valid, parameter_diagnostic });
        cases.back().kvarn.rtn_quantile = std::nextafter(1.0f, std::numeric_limits<float>::infinity());
        cases.push_back({ "NaN RTN quantile", valid, parameter_diagnostic });
        cases.back().kvarn.rtn_quantile = std::numeric_limits<float>::quiet_NaN();

        for (const test_case & test : cases) {
            common_params params;
            params.model.path = model_path;
            params.n_ctx = 256;
            params.n_batch = 32;
            params.n_ubatch = 32;
            params.n_parallel = 1;
            params.n_gpu_layers = 0;
            params.fit_params = false;
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            params.kv_cache_quant_type = LLAMA_KV_CACHE_QUANT_TYPE_KVARN;
            params.kvarn = test.kvarn;

            log.clear();
            server_context ctx_server;
            const bool loaded = ctx_server.load_model(params);
            if (loaded) {
                std::fprintf(stderr, "expected KVarN server model load to fail for %s\n", test.name);
                ok = false;
            }
            if (log.find(test.diagnostic) == std::string::npos) {
                std::fprintf(stderr, "missing expected KVarN diagnostic for %s: %s\n",
                        test.name, test.diagnostic);
                ok = false;
            }
        }
    } // restore logger before backend shutdown

    llama_backend_free();
    return ok ? 0 : 1;
}
