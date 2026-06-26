#include "llama-batch.h"
#include "llama-vocab.h"
#include "llama-kvarn-ubatch.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

void llama_log_internal(enum ggml_log_level, const char *, ...) {}

uint32_t llama_vocab::n_tokens() const {
    return 1000000;
}

const std::string & llama_vocab::token_to_piece(llama_token) const {
    static const std::string empty;
    return empty;
}

static const llama_vocab & fake_vocab() {
    alignas(llama_vocab) static const unsigned char storage[sizeof(llama_vocab)] = {};
    return *reinterpret_cast<const llama_vocab *>(storage);
}

static void require(bool cond, const char * msg) {
    if (!cond) {
        fprintf(stderr, "test-batch-split: %s\n", msg);
        std::exit(1);
    }
}

static void set_env_var(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    if (value[0] == '\0') {
        unsetenv(name);
    } else {
        setenv(name, value, 1);
    }
#endif
}

static llama_batch make_embedding_batch(
        std::vector<float> & embd,
        std::vector<llama_pos> & pos,
        std::vector<int32_t> & n_seq_id,
        std::vector<llama_seq_id *> & seq_id,
        std::vector<llama_seq_id> & seq_id_data,
        std::vector<int8_t> & logits,
        int32_t n_tokens) {
    embd.assign(n_tokens, 1.0f);
    pos.resize(n_tokens);
    n_seq_id.assign(n_tokens, 1);
    seq_id.resize(n_tokens);
    seq_id_data.resize(n_tokens);
    logits.assign(n_tokens, 1);

    for (int32_t i = 0; i < n_tokens; ++i) {
        pos[i] = 0;
        seq_id_data[i] = i;
        seq_id[i] = &seq_id_data[i];
    }

    llama_batch batch = {};
    batch.n_tokens = n_tokens;
    batch.embd = embd.data();
    batch.pos = pos.data();
    batch.n_seq_id = n_seq_id.data();
    batch.seq_id = seq_id.data();
    batch.logits = logits.data();
    return batch;
}

static void test_split_equal_respects_sequence_limit(uint32_t n_ubatch) {
    std::vector<float> embd;
    std::vector<llama_pos> pos;
    std::vector<int32_t> n_seq_id;
    std::vector<llama_seq_id *> seq_id;
    std::vector<llama_seq_id> seq_id_data;
    std::vector<int8_t> logits;

    llama_batch batch = make_embedding_batch(embd, pos, n_seq_id, seq_id, seq_id_data, logits, 4);

    llama_batch_allocr balloc(/*n_pos_per_embd =*/ 1);
    require(balloc.init(batch, fake_vocab(), nullptr, /*n_embd =*/ 1, /*n_seq_max =*/ 4, /*output_all =*/ true),
            "batch allocator init");

    balloc.split_reset();
    llama_ubatch ubatch = balloc.split_equal(n_ubatch, true);

    require(ubatch.n_tokens > 0, "split_equal produces a non-empty ubatch");
    require(ubatch.n_tokens <= n_ubatch, "split_equal does not exceed n_ubatch token limit");
    require(ubatch.n_seqs <= n_ubatch, "split_equal does not exceed n_ubatch sequence limit");
    require(ubatch.n_seq_tokens == 1, "split_equal keeps one token per sequence in this fixture");
}

static llama_batch make_contiguous_single_seq_batch(
        std::vector<float> & embd,
        std::vector<llama_pos> & pos,
        std::vector<int32_t> & n_seq_id,
        std::vector<llama_seq_id *> & seq_id,
        std::vector<llama_seq_id> & seq_id_data,
        std::vector<int8_t> & logits,
        int32_t n_tokens,
        llama_pos pos0) {
    embd.assign(n_tokens, 1.0f);
    pos.resize(n_tokens);
    n_seq_id.assign(n_tokens, 1);
    seq_id.resize(n_tokens);
    seq_id_data.assign(n_tokens, 0);
    logits.assign(n_tokens, 0);

    for (int32_t i = 0; i < n_tokens; ++i) {
        pos[i] = pos0 + i;
        seq_id[i] = &seq_id_data[i];
    }

    llama_batch batch = {};
    batch.n_tokens = n_tokens;
    batch.embd = embd.data();
    batch.pos = pos.data();
    batch.n_seq_id = n_seq_id.data();
    batch.seq_id = seq_id.data();
    batch.logits = logits.data();
    return batch;
}

static void test_tail_safe_ubatch_limit_pp512() {
    constexpr uint32_t sink = 128;
    constexpr uint32_t tail = 128;
    constexpr uint32_t default_limit = 512;

    std::vector<float> embd;
    std::vector<llama_pos> pos;
    std::vector<int32_t> n_seq_id;
    std::vector<llama_seq_id *> seq_id;
    std::vector<llama_seq_id> seq_id_data;
    std::vector<int8_t> logits;

    llama_batch batch = make_contiguous_single_seq_batch(
            embd, pos, n_seq_id, seq_id, seq_id_data, logits, 512, 0);

    llama_batch_allocr balloc(/*n_pos_per_embd =*/ 1);
    require(balloc.init(batch, fake_vocab(), nullptr, /*n_embd =*/ 1, /*n_seq_max =*/ 1, /*output_all =*/ false),
            "tail-safe pp512 batch init");

    balloc.split_reset();
    const uint32_t chunk0 = kvarn_tail_safe_ubatch_limit(balloc, default_limit, sink, tail);
    require(chunk0 == 256, "pp512 direct-prefill first chunk keeps sink/tail row writes unique");

    auto ubatch0 = balloc.split_simple(chunk0);
    require(ubatch0.n_tokens == 256, "pp512 direct-prefill first split consumes 256 tokens");

    const uint32_t chunk1 = kvarn_tail_safe_ubatch_limit(balloc, default_limit, sink, tail);
    require(chunk1 == 128, "pp512 direct-prefill second chunk is one tail span");

    auto ubatch1 = balloc.split_simple(chunk1);
    require(ubatch1.n_tokens == 128, "pp512 direct-prefill second split consumes 128 tokens");

    const uint32_t chunk2 = kvarn_tail_safe_ubatch_limit(balloc, default_limit, sink, tail);
    require(chunk2 == 128, "pp512 direct-prefill third chunk is one tail span");

    auto ubatch2 = balloc.split_simple(chunk2);
    require(ubatch2.n_tokens == 128, "pp512 direct-prefill third split consumes 128 tokens");
    require(balloc.get_n_used() == 512, "pp512 direct-prefill batch fully consumed");
}

static void test_tail_safe_ubatch_limit_pp512_without_direct_prefill() {
    constexpr uint32_t sink = 128;
    constexpr uint32_t tail = 128;
    constexpr uint32_t default_limit = 512;

    std::vector<float> embd;
    std::vector<llama_pos> pos;
    std::vector<int32_t> n_seq_id;
    std::vector<llama_seq_id *> seq_id;
    std::vector<llama_seq_id> seq_id_data;
    std::vector<int8_t> logits;

    llama_batch batch = make_contiguous_single_seq_batch(
            embd, pos, n_seq_id, seq_id, seq_id_data, logits, 512, 0);

    llama_batch_allocr balloc(/*n_pos_per_embd =*/ 1);
    require(balloc.init(batch, fake_vocab(), nullptr, /*n_embd =*/ 1, /*n_seq_max =*/ 1, /*output_all =*/ false),
            "tail-safe pp512 no-direct batch init");

    set_env_var("LLAMA_KVARN_DISABLE_PREFILL_DIRECT_STORE", "1");
    balloc.split_reset();
    const uint32_t chunk0 = kvarn_tail_safe_ubatch_limit(balloc, default_limit, sink, tail);
    set_env_var("LLAMA_KVARN_DISABLE_PREFILL_DIRECT_STORE", "");
    require(chunk0 == 256, "pp512 no-direct first chunk keeps sink/tail row writes unique");

    auto ubatch0 = balloc.split_simple(chunk0);
    require(ubatch0.n_tokens == 256, "pp512 no-direct first split consumes 256 tokens");

    const uint32_t chunk1 = kvarn_tail_safe_ubatch_limit(balloc, default_limit, sink, tail);
    require(chunk1 == 128, "pp512 no-direct second chunk is 128 tokens");

    auto ubatch1 = balloc.split_simple(chunk1);
    require(ubatch1.n_tokens == 128, "pp512 no-direct second split consumes 128 tokens");

    const uint32_t chunk2 = kvarn_tail_safe_ubatch_limit(balloc, default_limit, sink, tail);
    require(chunk2 == 128, "pp512 no-direct third chunk is 128 tokens");

    auto ubatch2 = balloc.split_simple(chunk2);
    require(ubatch2.n_tokens == 128, "pp512 no-direct third split consumes 128 tokens");
    require(balloc.get_n_used() == 512, "pp512 no-direct batch fully consumed");
}

static void test_tail_safe_ubatch_limit_fallback() {
    constexpr uint32_t default_limit = 512;
    constexpr uint32_t tail = 128;

    std::vector<float> embd(4, 1.0f);
    std::vector<llama_pos> pos(4, 0);
    std::vector<int32_t> n_seq_id(4, 1);
    std::vector<llama_seq_id *> seq_id(4);
    std::vector<llama_seq_id> seq_id_data = { 0, 1, 2, 3 };
    std::vector<int8_t> logits(4, 0);
    for (int i = 0; i < 4; ++i) {
        seq_id[i] = &seq_id_data[i];
    }

    llama_batch batch = {};
    batch.n_tokens = 4;
    batch.embd = embd.data();
    batch.pos = pos.data();
    batch.n_seq_id = n_seq_id.data();
    batch.seq_id = seq_id.data();
    batch.logits = logits.data();

    llama_batch_allocr balloc(/*n_pos_per_embd =*/ 1);
    require(balloc.init(batch, fake_vocab(), nullptr, /*n_embd =*/ 1, /*n_seq_max =*/ 4, /*output_all =*/ false),
            "tail-safe fallback batch init");

    balloc.split_reset();
    const uint32_t limit = kvarn_tail_safe_ubatch_limit(balloc, default_limit, 0, tail);
    require(limit == tail, "multi-seq fallback min(default, tail)");

    batch.pos = nullptr;
    require(balloc.init(batch, fake_vocab(), nullptr, /*n_embd =*/ 1, /*n_seq_max =*/ 4, /*output_all =*/ false),
            "tail-safe missing pos batch init");
    balloc.split_reset();
    const uint32_t limit_no_pos = kvarn_tail_safe_ubatch_limit(balloc, default_limit, 0, tail);
    require(limit_no_pos == tail, "missing pos fallback min(default, tail)");
}

static void test_tail_safe_ubatch_limit_negative_pos() {
    std::vector<float> embd(3, 1.0f);
    std::vector<llama_pos> pos = { -1, 0, 1 };
    std::vector<int32_t> n_seq_id(3, 1);
    std::vector<llama_seq_id *> seq_id(3);
    std::vector<llama_seq_id> seq_id_data(3, 0);
    std::vector<int8_t> logits(3, 0);
    for (int i = 0; i < 3; ++i) {
        seq_id[i] = &seq_id_data[i];
    }

    llama_batch batch = {};
    batch.n_tokens = 3;
    batch.embd = embd.data();
    batch.pos = pos.data();
    batch.n_seq_id = n_seq_id.data();
    batch.seq_id = seq_id.data();
    batch.logits = logits.data();

    llama_batch_allocr balloc(/*n_pos_per_embd =*/ 1);
    require(balloc.init(batch, fake_vocab(), nullptr, /*n_embd =*/ 1, /*n_seq_max =*/ 1, /*output_all =*/ false),
            "tail-safe negative pos batch init");

    balloc.split_reset();
    const uint32_t limit = kvarn_tail_safe_ubatch_limit(balloc, 512, 0, 128);
    require(limit == 1, "first_pos < 0 limits chunk to 1");
}

static void test_split_equal_zero_limit() {
    llama_batch_allocr balloc(/*n_pos_per_embd =*/ 1);
    llama_ubatch ubatch = balloc.split_equal(0, true);
    require(ubatch.n_tokens == 0, "split_equal zero limit returns empty ubatch");
}

int main() {
    test_tail_safe_ubatch_limit_pp512();
    test_tail_safe_ubatch_limit_pp512_without_direct_prefill();
    test_tail_safe_ubatch_limit_fallback();
    test_tail_safe_ubatch_limit_negative_pos();
    test_split_equal_zero_limit();
    test_split_equal_respects_sequence_limit(1);
    test_split_equal_respects_sequence_limit(2);

    return 0;
}
