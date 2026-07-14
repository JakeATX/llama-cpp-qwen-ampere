#include "server-context.h"

#include <cstdio>
#include <cstdlib>

static void expect(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

int main() {
    const auto no   = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    const auto part = COMMON_CONTEXT_SEQ_RM_TYPE_PART;
    const auto full = COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    const auto rs   = COMMON_CONTEXT_SEQ_RM_TYPE_RS;

    expect(!server_context_has_rollback_strategy(no, no, false),
            "NO target must force full prompt reprocessing");
    expect(server_context_has_rollback_strategy(part, no, false),
            "absent draft must not inherit the draft sentinel");
    expect(server_context_has_rollback_strategy(full, no, false),
            "FULL target must retain its proven checkpoint path");
    expect(server_context_has_rollback_strategy(rs, full, true),
            "present FULL draft must retain its proven checkpoint path");
    expect(!server_context_has_rollback_strategy(part, no, true),
            "present NO draft must block speculation, prefix reuse, and checkpoints");
    expect(!server_context_has_rollback_strategy(no, part, true),
            "NO target must block rollback even with a capable draft");

    expect(server_context_can_reuse_shifted_cache(part, no, false, true, false),
            "direct shift-capable target may reuse chunks without a draft");
    expect(server_context_can_reuse_shifted_cache(rs, part, true, true, true),
            "direct shift-capable target and draft may reuse chunks");
    expect(!server_context_can_reuse_shifted_cache(full, part, true, true, true),
            "checkpoint-only target must not enter direct chunk shifting");
    expect(!server_context_can_reuse_shifted_cache(part, full, true, true, true),
            "checkpoint-only draft must not enter direct chunk shifting");
    expect(!server_context_can_reuse_shifted_cache(part, no, true, true, true),
            "NO draft must block chunk shifting");
    expect(!server_context_can_reuse_shifted_cache(part, part, true, false, true),
            "non-shift-capable target must block chunk shifting");
    expect(!server_context_can_reuse_shifted_cache(part, part, true, true, false),
            "non-shift-capable draft must block chunk shifting");

    std::puts("server context rollback policy tests: PASS");
    return 0;
}
