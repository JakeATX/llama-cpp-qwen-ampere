#include "common.h"
#include "get-model.h"
#include "server-context.h"

#include <cstdio>

int main(int argc, char ** argv) {
    char * model_path = get_model_or_exit(argc, argv);

    common_init();
    llama_backend_init();

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
    params.kvarn = llama_kvarn_default_params();

    server_context ctx_server;
    const bool loaded = ctx_server.load_model(params);

    llama_backend_free();

    if (loaded) {
        std::fprintf(stderr, "expected KVarN server model load to fail for unsupported fixture model\n");
        return 1;
    }

    return 0;
}
