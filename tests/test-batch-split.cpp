#include "llama-batch.h"
#include "llama-vocab.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

void llama_log_internal(enum ggml_log_level, const char *, ...) {}

uint32_t llama_vocab::n_tokens() const {
    return 0;
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

static void test_split_equal_zero_limit() {
    llama_batch_allocr balloc(/*n_pos_per_embd =*/ 1);
    llama_ubatch ubatch = balloc.split_equal(0, true);
    require(ubatch.n_tokens == 0, "split_equal zero limit returns empty ubatch");
}

int main() {
    test_split_equal_zero_limit();
    test_split_equal_respects_sequence_limit(1);
    test_split_equal_respects_sequence_limit(2);

    return 0;
}
