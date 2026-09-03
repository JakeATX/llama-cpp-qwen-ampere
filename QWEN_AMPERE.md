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
  --cache-prompt --cache-ram 8192 --ctx-checkpoints 24 --checkpoint-min-step 10240 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.45 \
  --spec-draft-type-k q8_0 --spec-draft-type-v turbo3
```

The three cache flags are what make a long conversation usable rather than merely possible. `--cache-prompt` keeps the conversation's KV cache in the slot between turns, so a new turn on a 200K conversation pays only for the new tokens instead of a 100-second re-prefill. `--ctx-checkpoints` matters specifically for this model: 48 of its layers are recurrent, and a recurrent state cannot be rewound, so when you edit or regenerate a turn the server needs a saved state from before the edit point; it keeps up to 24 of them, at least 10,240 tokens apart (and one at every user turn regardless), in host RAM. Measured on this model, a snapshot is 150 MiB plus 1.5 KiB per token of position, because the MTP drafter's own single-layer KV cache is saved with the recurrent state: about 165 MiB at 10K, 495 MiB at 240K. 24 at 10,240 spacing covers the whole 245,760 window for about 7.8 GiB with at most ten seconds of replay after an edit; denser spacing multiplies that RAM (60 at 4,096 is about 20 GiB at the deep end). `--cache-ram` is a separate host-RAM budget for parking a whole conversation's KV (with its checkpoints) when another conversation takes the slot; a populated 200K conversation is about 7.4 GB, so 8 GiB holds one, and it only does work when you switch between chats. None of this touches VRAM. Budget about 16 GB of host RAM for it at the deep end (up to 8 GiB of snapshots plus the 8 GiB park space) on top of the model's own mapping; on a 32 GB machine keep the desktop light, or drop the count to 12 at 20,480 spacing for half the snapshot RAM.

**Disk tier for the prompt cache (this fork).** `--cache-disk-path DIR [--cache-disk-limit MiB]`
adds a second tier under the RAM prompt cache: a conversation evicted from RAM, or one too large
for the RAM budget in the first place (a full 200K-245K session is 7-9 GB), is written to `DIR`
instead of being dropped, and a later request that matches it is restored from the file with only
the new tokens processed. The index survives a server restart. Restores run at roughly 1.5 GB/s,
so a 200K conversation comes back in a few seconds instead of a 100-second re-prefill, and the
restored state is exact (greedy outputs match an uninterrupted session). Multimodal prompts stay
RAM-only. Off unless the path is given.

```bash
--cache-prompt --cache-ram 8192 --cache-disk-path /fast-nvme/llama-cache --cache-disk-limit 65536 \
--ctx-checkpoints 24 --checkpoint-min-step 10240
```

Everything the project adds is on by default. `LLAMA_SHARED_COMPUTE=0` disables
the shared compute arena; the research knobs (`GGML_Q8_TURBO3_MMA_MIN_Q`,
`GGML_CUDA_SM86_MMQ_POLICY`, `GGML_CUDA_SM86_MMVQ_WARP_ROWS`) stay off unless
you are reproducing a rejected experiment.

## Credits

Qwen team for Qwen3.8; TheTom for TurboQuant+; Unsloth for the BF16 GGUF, the
importance matrix, and the tier ladder the recipe follows; ggml-org for
llama.cpp.
