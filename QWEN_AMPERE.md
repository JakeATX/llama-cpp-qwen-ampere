# llama-cpp-qwen-ampere

Qwen3.8-27B on one Ampere card (RTX 3090 / 3090 Ti, 24 GB): a fork of
[TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)
(TurboQuant+ KV cache, native MTP speculative decoding) carrying SM86-specific
kernel and memory work, plus the quantization recipe it was tuned with.

Measured on a 3090 Ti at 350 W with the ATX-4-XS quant:

| | |
|---|---|
| populated 200K context | 20.9 GiB ready VRAM, 694 tok/s prefill, 65 tok/s decode |
| largest measured fit | 245,760-token window with a 240K prompt, 22.1 GiB ready |
| agent session 100K -> 245K context | 120K generated tokens, 52 tok/s cumulative (60 at 110K, 47 at 245K), 76% draft acceptance |
| per speculative round vs Q3_K_XL / Q4_K_M | +9-10% / +23% |

Single-user configuration: `--parallel 1`, one request at a time.

## Model

`sjakek/Qwen3.8-27B-ATX-4-XS-GGUF` on Hugging Face: the GGUF, the per-tensor
type map, and the recipe. 14.5 GiB file, 13.9 GiB on the GPU. Bulk tensors
IQ4_XS (the fastest format on SM86 at speculative verification widths), Q5_0 on
the tensors Unsloth's tier ladder upgrades first, Q6_K on attention K/V, and
Q8_0 on the GDN alpha/beta vectors and the attention K/V tensors Q4_K_M keeps at
Q8_0.

## Branches

| branch | what it is |
|---|---|
| `main` | the product: upstream TurboQuant+ as of 2026-09-03 plus all accepted SM86 work. Build from here. |
| `perf/qwen38-sm86-decode-product`, `perf/qwen38-sm86-prefill` | the same commit as `main` at the 2026-09-03 release, kept for existing links |
| `research/qwen38-sm86-*` | experiment branches with opt-in knobs; see the handover |
| `archive/*` | rejected or superseded experiments |
| everything else | inherited from the upstream llama.cpp mirror, not part of this project |

The experiment log, with hypotheses, results, and what did not work, is
`QWEN38_SM86_FRONTIER_HANDOVER.md` in this tree.

## Build and run

```bash
git clone -b main https://github.com/JakeATX/llama-cpp-qwen-ampere.git
cd llama-cpp-qwen-ampere
cmake -S . -B build-sm86 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_FA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_NATIVE=ON
cmake --build build-sm86 -j8 --target llama-server

GGML_Q8_TURBO3_MMA_FUSED=1 ./build-sm86/bin/llama-server -m Qwen3.8-27B-ATX-4-XS.gguf \
  -c 245760 -b 4096 -ub 1024 -t 8 -tb 8 -ngl 99 -fa on -ctk q8_0 -ctv turbo3 \
  --parallel 1 --jinja --fit off \
  --cache-prompt --cache-ram 8192 --ctx-checkpoints 40 --checkpoint-min-step 2048 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.45 \
  --spec-draft-type-k q8_0 --spec-draft-type-v turbo3
```

The three cache flags are what make a long conversation usable rather than merely possible. `--cache-prompt` keeps the conversation's KV cache in the slot between turns, so a new turn on a 200K conversation pays only for the new tokens instead of a 100-second re-prefill. `--ctx-checkpoints` matters specifically for this model: 48 of its layers are recurrent, and a recurrent state cannot be rewound, so when you edit or regenerate a turn the server needs a saved state from before the edit point; it keeps up to 40 of them, at least 2,048 tokens apart (and one at every user turn regardless), in host RAM at roughly 150 MB each; that covers edits within the last 80K tokens with at most a couple of seconds of replay, and older edit points fall back to a full re-prefill. The snapshots are fixed-size, so the RAM cost does not grow with context, only with the count. `--cache-ram` is a separate host-RAM budget for parking a whole conversation's KV (with its checkpoints) when another conversation takes the slot; a populated 200K conversation is about 7.4 GB, so 8 GiB holds one, and it only does work when you switch between chats. None of this touches VRAM. Budget about 14 GB of host RAM for it (6 GB of snapshots plus the 8 GiB park space) on top of the model's own mapping; on a 32 GB machine keep the desktop light, or drop the count to 10 at 8,192 spacing for the same reach at 1.5 GB.

Everything the project adds is on by default. `LLAMA_SHARED_COMPUTE=0` disables
the shared compute arena; the research knobs (`GGML_Q8_TURBO3_MMA_MIN_Q`,
`GGML_CUDA_SM86_MMQ_POLICY`, `GGML_CUDA_SM86_MMVQ_WARP_ROWS`) stay off unless
you are reproducing a rejected experiment.

## Credits

Qwen team for Qwen3.8; TheTom for TurboQuant+; Unsloth for the BF16 GGUF, the
importance matrix, and the tier ladder the recipe follows; ggml-org for
llama.cpp.
