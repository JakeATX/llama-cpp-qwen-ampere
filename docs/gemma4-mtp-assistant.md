# Gemma4 standalone MTP assistant support

This branch is based on `am17an:mtp-clean` from PR #22673. That PR adds a
working MTP speculative path for Qwen3.6 GGUFs that carry their MTP/NextN head
inside the same GGUF as the trunk model.

Gemma4 is a different shape. The public assistant artifact is a standalone
draft model with `general.architecture = gemma4_mtp`, not an embedded NextN
head inside the target GGUF. Supporting it requires a new model architecture
and a separate proposer/runtime path; it is not solved by only passing
`--spec-type mtp` or increasing `--spec-draft-n-max`.

## What works today

- Qwen3.6 MTP GGUFs such as `am17an/Qwen3.6-35BA3B-MTP-GGUF` use the current
  PR path:

  ```bash
  llama-cli \
    -m Qwen3.6-35BA3B-MTP.gguf \
    -p "<prompt>" \
    -n 128 \
    -ngl 5 \
    -ncmoe 32 \
    -fa 1 \
    -ctk q8_0 \
    -ctv q8_0 \
    --spec-type mtp \
    --spec-draft-n-max 3
  ```

- In server initialization, PR #22673 maps a supported trunk architecture to a
  synthetic MTP architecture loaded from the same GGUF:

  - `qwen35` -> `qwen35_mtp`
  - `qwen35moe` -> `qwen35moe_mtp`

- The implemented Qwen MTP graph expects `nextn_predict_layers` and tensors such
  as `blk.*.nextn_eh_proj`, `nextn_enorm`, `nextn_hnorm`, and optional shared
  head tensors.

## Why Gemma4 does not work yet

The Gemma4 assistant GGUFs expose a standalone architecture:

```text
general.architecture = gemma4_mtp
gemma4_mtp.embedding_length = 1024
gemma4_mtp.backbone_embedding_length = 5376
gemma4_mtp.block_count = 4
gemma4_mtp.centroid_count = 2048
gemma4_mtp.centroid_top_k = 32
```

Representative tensors:

```text
token_embd.weight
blk.{0..3}.attn_q.weight
blk.{0..3}.attn_output.weight
blk.{0..3}.ffn_gate.weight
blk.{0..3}.ffn_up.weight
blk.{0..3}.ffn_down.weight
mtp_pre_proj.weight
mtp_post_proj.weight
```

That is not the Qwen embedded-NextN layout. In particular:

- The assistant has Q-only attention. K/V are read from the target model cache.
- The assistant operates in a smaller draft hidden size, then projects back to
  the target/backbone hidden size.
- The assistant must keep its own draft-dim LM head; sharing the target LM head
  is wrong.
- Draft steps reuse the target model's last position instead of advancing normal
  positions in the same way as an ordinary decoder.
- Gemma4 target models have sliding/full attention groups, so KV sharing needs
  per-group metadata and layer-type-aware target mapping.

## Prior art

vLLM PR #41745 (`[Spec Decode] Add Gemma4 MTP speculative decoding support`)
merged a working CUDA implementation with the required pieces:

- `Gemma4MTP` model.
- `Gemma4Proposer`.
- Q-only attention layers sharing target KV cache.
- `pre_projection(2 * backbone_dim -> draft_dim)` and
  `post_projection(draft_dim -> backbone_dim)`.
- Constant draft positions.
- Per-group block tables for Gemma4 sliding/full attention groups.
- Optional centroid-masked logits for assistants with ordered embeddings.

That implementation is the best reference for porting Gemma4 assistant MTP into
llama.cpp.

## llama.cpp implementation checklist

1. Add `LLM_ARCH_GEMMA4_MTP` and map `general.architecture = gemma4_mtp`.
2. Add hparams for:
   - `gemma4_mtp.embedding_length`
   - `gemma4_mtp.backbone_embedding_length`
   - `gemma4_mtp.block_count`
   - `gemma4_mtp.centroid_count`
   - `gemma4_mtp.centroid_top_k`
3. Add tensor mappings for:
   - `token_embd.weight`
   - `blk.{i}.attn_q.weight`
   - `blk.{i}.attn_output.weight`
   - `blk.{i}.attn_norm.weight`
   - `blk.{i}.ffn_*`
   - `blk.{i}.pre_ffw_norm.weight`
   - `blk.{i}.post_attention_norm.weight`
   - `blk.{i}.layer_scale`
   - `mtp_pre_proj.weight`
   - `mtp_post_proj.weight`
   - optional centroid tensors when present.
4. Implement `llama_model_gemma4_mtp`:
   - embed draft token in draft hidden size.
   - concatenate token embedding with target/backbone hidden state.
   - apply pre-projection.
   - run the assistant decoder blocks.
   - run Q-only attention where K/V come from the target Gemma4 KV cache.
   - project assistant hidden state back to backbone hidden size for recursive
     draft steps.
   - compute logits with the assistant LM head, not the target LM head.
5. Extend the MTP loader path:
   - support a separate `--spec-draft-model` with architecture `gemma4_mtp`, or
     add a new explicit `--spec-mtp-model` alias.
   - do not require the assistant to live in the target GGUF.
6. Extend `common_speculative_state_mtp`:
   - pass target hidden states into the assistant.
   - keep a feedback hidden-state tensor from `mtp_post_proj`.
   - reuse constant draft positions for Gemma4.
   - wire rollback/removal so rejected draft tokens do not poison assistant
     state.
7. Add KV-sharing metadata:
   - map each assistant layer to the last non-KV-shared target layer of the same
     attention type where possible.
   - handle Gemma4's sliding/full attention cache groups.
8. Add tests:
   - metadata-only load of a Gemma4 target plus `gemma4_mtp` assistant.
   - 1-token greedy smoke with `--spec-draft-n-max 1`.
   - acceptance/timing smoke at `--spec-draft-n-max 2,4,8`.
   - long-context guard tests with KV cache types `f16`, `q8_0`, and `q4_0`.

## Current expected failure

Until the architecture and Q-only KV-sharing path above exist, attempting to load
a Gemma4 assistant GGUF in llama.cpp should fail around architecture detection:

```text
unknown model architecture: 'gemma4_mtp'
```

That failure is expected and distinct from the Qwen3.6 MTP path, which is the
path demonstrated in AJ's thread and PR #22673.

## References

- llama.cpp PR #22673: https://github.com/ggml-org/llama.cpp/pull/22673
- Qwen3.6 MTP GGUF: https://huggingface.co/am17an/Qwen3.6-35BA3B-MTP-GGUF
- vLLM Gemma4 MTP PR #41745: https://github.com/vllm-project/vllm/pull/41745
- Gemma4 assistant GGUF: https://huggingface.co/Radamanthys11/Gemma-4-31B-it-assistant-GGUF
