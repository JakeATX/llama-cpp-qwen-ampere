#include "server-task.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void expect(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

static server_prompt make_prompt(std::initializer_list<llama_token> tokens, bool with_checkpoint = false) {
    server_prompt prompt;
    prompt.tokens = server_tokens(llama_tokens(tokens), false);
    if (with_checkpoint) {
        prompt.checkpoints.push_back({ 2, 3, 4, { 7, 8 }, { 9 } });
    }
    return prompt;
}

static server_prompt_cache::state_writer exact_writer(
        uint8_t value, int & calls, std::vector<std::string> * order = nullptr, const char * name = nullptr) {
    return [value, &calls, order, name](uint8_t * data, size_t size) {
        ++calls;
        if (order != nullptr) {
            order->push_back(name);
        }
        std::fill(data, data + size, value);
        return size;
    };
}

int main() {
    using result = server_prompt_cache_save_result;

    {
        server_prompt_cache cache(-1, 0);
        auto prompt = make_prompt({ 1, 2 });
        int main_calls = 0;
        int draft_calls = 0;
        auto main_writer = exact_writer(1, main_calls);
        auto draft_writer = exact_writer(2, draft_calls);

        expect(cache.save(prompt, 0, 0, main_writer) == result::unavailable,
                "zero target state must be unavailable");
        expect(cache.save(prompt, 4, 0, main_writer, draft_writer) == result::unavailable,
                "present draft writer with zero state must be unavailable");
        expect(main_calls == 0 && draft_calls == 0, "unavailable states must not call writers");
        expect(cache.states.empty(), "unavailable states must not mutate the cache");

        expect(cache.save(prompt, 4, 0, {}) == result::invalid_writer,
                "target size without a writer must be rejected");
        expect(cache.save(prompt, 4, 3, main_writer) == result::invalid_writer,
                "draft size without a writer must be rejected");
        expect(main_calls == 0 && draft_calls == 0, "invalid writer configurations must not call writers");
        expect(cache.states.empty(), "invalid writer configurations must not mutate the cache");
    }

    {
        server_prompt_cache cache(-1, 0);
        auto old_prompt = make_prompt({ 10 });
        int old_calls = 0;
        expect(cache.save(old_prompt, 2, 0, exact_writer(3, old_calls)) == result::success,
                "initial cache entry must save");

        auto replacement = make_prompt({ 10, 11 }, true);
        int main_calls = 0;
        int draft_calls = 0;
        auto short_main = [&main_calls](uint8_t *, size_t size) {
            ++main_calls;
            return size - 1;
        };
        expect(cache.save(replacement, 4, 3, short_main, exact_writer(4, draft_calls)) == result::short_main,
                "short target write must fail");
        expect(main_calls == 1 && draft_calls == 0, "target failure must stop before draft write");
        expect(cache.states.size() == 1 && cache.states.front().tokens.size() == 1,
                "short target write must preserve the obsolete prior entry");

        std::vector<std::string> order;
        main_calls = 0;
        draft_calls = 0;
        auto oversized_draft = [&draft_calls, &order](uint8_t *, size_t size) {
            ++draft_calls;
            order.push_back("draft");
            return size + 1;
        };
        expect(cache.save(replacement, 4, 3,
                    exact_writer(5, main_calls, &order, "main"), oversized_draft) == result::short_draft,
                "any draft byte-count mismatch must fail");
        expect(main_calls == 1 && draft_calls == 1, "draft failure must follow one target write");
        expect(order == std::vector<std::string>({ "main", "draft" }),
                "writers must run target then draft");
        expect(cache.states.size() == 1 && cache.states.front().tokens.size() == 1,
                "short draft write must preserve the obsolete prior entry");

        order.clear();
        main_calls = 0;
        draft_calls = 0;
        expect(cache.save(replacement, 4, 3,
                    exact_writer(0x51, main_calls, &order, "main"),
                    exact_writer(0x62, draft_calls, &order, "draft")) == result::success,
                "exact writes must commit");
        expect(order == std::vector<std::string>({ "main", "draft" }),
                "successful writers must run target then draft");
        expect(cache.states.size() == 1, "successful replacement must remove obsolete entry");
        const auto & saved = cache.states.front();
        expect(saved.tokens.size() == 2 && saved.data.main == std::vector<uint8_t>(4, 0x51) &&
                    saved.data.drft == std::vector<uint8_t>(3, 0x62),
                "successful save must commit exact tokens and bytes");
        expect(saved.checkpoints.size() == 1 && saved.checkpoints.front().data_tgt == std::vector<uint8_t>({ 7, 8 }) &&
                    saved.checkpoints.front().data_dft == std::vector<uint8_t>({ 9 }),
                "successful save must preserve checkpoints");

        main_calls = 0;
        draft_calls = 0;
        expect(cache.save(replacement, 4, 3,
                    exact_writer(9, main_calls), exact_writer(9, draft_calls)) == result::already_present,
                "already-cached prompt must be a no-op");
        expect(main_calls == 0 && draft_calls == 0, "already-cached prompt must not call writers");
        expect(cache.states.size() == 1, "already-cached prompt must not mutate cache");
    }

    std::puts("server prompt cache transactional save tests: PASS");
    return 0;
}
