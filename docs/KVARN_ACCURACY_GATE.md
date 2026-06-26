# KVarN accuracy gate and code-review findings

This note accompanies `scripts/kvarn/run_accuracy_gate.ps1`. It records why
the gate exists and the line-level findings from a full read of the KVarN
attention/store path on `kvarn-atx-integration`.

## TL;DR

1. **There is no end-to-end accuracy gate.** Every current gate
   (packed-vs-split NMSE, mainline throughput parity) compares two KVarN code
   paths to each other, so a systematic KVarN numerical error is invisible.
   `run_accuracy_gate.ps1` adds the missing check: KVarN vs the f16 model on
   the same build/model/dataset (perplexity delta, or `-UseKLDivergence` for a
   more sensitive per-token check).

2. **Gemma4 needed two separate fixes.** The first problem was the test
   harness: Gemma was being judged on fixtures with literal `<unk>` and
   mismatched chat/control-token surfaces, so the absolute baseline PPL was not
   a valid model-health signal. The second problem was the forced
   Gemma4+ISWA KVarN route and validation corpus. The old small sanity fixture
   implicated donor layers 11 and 47, but a Gemma-rendered 16k chat fixture now
   passes with all full-attention donors KVarN-routed. The all-full route is the
   default; `LLAMA_KVARN_GEMMA4_ROUTE_CONSERVATIVE=1` keeps the old hybrid route
   only for rollback/reproduction.

3. **Why it is slower than mainline** is architectural, not a tuning miss: the
   "exact" production paths dequantize the cache to an f32/f16 scratch mirror
   and then run dense attention, which is *more* memory traffic and compute
   than mainline f16 + flash-attention. The only path that could win (warp-QK
   reading packed/f16 directly) is disabled for 256d.

## Finding 1 - Gemma fixture and route correctness

The old Gemma absolute-PPL numbers were not trustworthy. The model and
perplexity scorer can produce sane absolute PPL on the deliberately
low-entropy sanity fixture:

- normal KV, Q4 GPU, `ctx=512`: `PPL = 1.1071 +/- 0.07188`
- KVarN `kvarn_k4v2_g128`, conservative route: `PPL = 1.0724 +/- 0.03914`
- KVarN `kvarn_k8v8_g128`, conservative route: `PPL = 1.0461 +/- 0.02530`
- KVarN `kvarn_k8v8_g128`, all full-attention donors: `PPL = 484746.4909`

Behavioral bisection on the old short fixture implicated full-attention physical
donor layers 11 and 47 for this Gemma4-12B layout. That was not enough evidence
to permanently bypass those layers: on the Gemma-rendered 16k chat fixture,
all-full routing now passes for `kvarn_k4v2_g128`, `kvarn_k4v4_g128`, and
`kvarn_k8v8_g128`, with sane baseline PPL:

- `kvarn_k4v2_g128`: f16 `3.7114`, KVarN `3.2218`, delta `-13.19%`
- `kvarn_k4v4_g128`: f16 `3.7114`, KVarN `3.6006`, delta `-2.99%`
- `kvarn_k8v8_g128`: f16 `3.7114`, KVarN `3.6308`, delta `-2.17%`

The default forced Gemma4+ISWA route in `src/llama-model.cpp` therefore routes
all full-attention donors through KVarN. Set
`LLAMA_KVARN_GEMMA4_ROUTE_CONSERVATIVE=1` only to reproduce the old hybrid
rollback route (`5,17,23,29,35,41` KVarN; `11,47` normal full KV).

The critic caveat remains important: the current 16k fixture is synthetic and
negative PPL deltas mean it is not a broad quality proof by itself. It is,
however, enough to reject the old "layers 11 and 47 must stay uncompressed"
default and move the default route back to the production-relevant all-full
path.

## Finding 1b - Hadamard rotation still needs audit

The earlier Hadamard concern remains a separate audit item, not the current
explanation for the Gemma baseline PPL issue. The KVarN store path rotates the
quantized body while sink/tail KV remain plain and the read path must preserve
the attention identity across all segments. Existing packed-vs-split tests can
miss a shared systematic transform error because both sides reconstruct the
same stored representation. Keep this on the correctness checklist, but do not
use it as the primary explanation for the fixed Gemma fixture results above.

## Finding 2 - the throughput gap is structural

Mainline = flash-attention over an f16 cache. KVarN "exact" paths add work on
top of that:

- **Qwen prefill (pp512, ~79.5%)** routes to the scalar-QT-GQA path
  (`kvarn.cu:4469-4842`), which first dequantizes the whole body to an **f32**
  scratch mirror per head (`4486-4511`) and then attends over 4-byte/elem f32 -
  more bandwidth than mainline f16. That f32 mirror has **no caching** (unlike
  the f16 mirror epoch cache at `4908-4928`), so it is rebuilt every op.
- **Qwen decode (tg64, ~81.6%)** falls to `kvarn_attn_mixed_f16_fused_batch_kernel`
  (`kvarn.cu:5037`, `3550-3743`): grid = `n_head` CTAs (SM under-utilization),
  per-dimension scalar `kvarn_unpack_one`, and packed reads that coalesce on
  neither axis (`i = d*group_size + g`).
- The fast `..._warpqk_kernel` (`3324-3548`) is disabled for 256d
  (`4848-4851`), so Qwen never gets it.
- `-fa off` is mandatory, so KVarN competes against tuned FA with a
  softmax-materializing kernel.

Direction: make the warp-parallel kernel read packed K/V directly (register
dequant) so the cache stays small and bandwidth drops below f16 - that is where
vLLM's "throughput above FP16" comes from. Stop materializing f32/f16 mirrors.

## Finding 3 - the 256d warp-QK gate is mis-specified

The 256d warp-QK path is blocked on matching the serial scalar path at NMSE
**exactly 0**. A warp-parallel QK reduction (`kvarn.cu:3431-3454`) reorders FP
additions and can never be bit-identical to a serial `d=0..head_dim` loop; the
synthetic op test passes within tolerance but full-model logits accumulate the
reorder to ~6e-3 over 40 MTP layers. Replace bitexactness with a tolerance +
this accuracy gate, then 256d warp-QK can ship and Qwen gets the fast path.
While there, audit the f16 dequant-cache key (`4908-4928`): under MTP
draft/verify the same scratch can match `(n_records, epoch)` with different live
content and serve a stale K/V mirror.

## Usage

Build the KVarN tree (CUDA), then, from the repo root:

```powershell
# Perplexity-delta gate (default): KVarN must stay within 5% of f16 PPL.
scripts/kvarn/run_accuracy_gate.ps1 `
    -Model  "C:\path\to\model.gguf" `
    -Dataset "C:\path\to\wikitext-2-raw\wiki.test.raw" `
    -BuildDir "build-kvarn-cuda-static-vs" `
    -MaxPplIncrease 0.05

# More sensitive: per-token KL divergence of KVarN logits vs the f16 base.
scripts/kvarn/run_accuracy_gate.ps1 `
    -Model "C:\path\to\model.gguf" `
    -Dataset "C:\path\to\wiki.test.raw" `
    -UseKLDivergence -MaxMeanKL 0.02
```

For the Qwen3.6 MTP model add its extra args, e.g.
`-ExtraArgs @('-ncmoe','34')` and `-ExpectedKvarnLayers '3-39:4'`.

Both runs use the same binary so a gap reflects the KV-cache backend, not build
differences. A passing run is the precondition for trusting any KVarN
throughput number; a large gap is correctness evidence - start at Finding 1.

The script forces `-np 1 -fit off` and sets `-b <= -c` for both f16 and KVarN
runs because `llama-perplexity` derives its internal sequence count from
`batch/context`; KVarN rejects multi-sequence execution, including hidden
retries from the auto-fit path.

## Baseline and fixture sanity

The gate now refuses to compare KVarN against an unhealthy reference:

- It runs a fixture preflight before perplexity. The preflight fails on
  literal `<unk>` contamination, on chat markers that do not appear in the
  model's GGUF chat template, and on actual tokenizer unknown-token ID rates
  above `-MaxFixtureTokenUnkRate`.
- In perplexity-delta mode, it runs the normal-KV baseline first and skips the
  KVarN run when baseline PPL exceeds `-MaxBaselinePpl` (default `100`). Disable
  this only with an explicit `-MaxBaselinePpl 0` when you have an external
  reason to trust the corpus/model pairing.
- In KL-divergence mode, the normal-KV logits are the reference distribution, so
  the gate records baseline PPL but does not enforce `-MaxBaselinePpl` unless
  you pass that parameter explicitly. This is the preferred correctness mode for
  instruction-tuned Gemma fixtures where absolute next-token PPL on a
  hand-written corpus is not a meaningful model-health score.
- For Gemma/Qwen/etc. fixtures that contain literal control tokens such as
  `<|turn>` or `<turn|>`, pass `-ParseSpecial`. `llama-perplexity` otherwise
  tokenizes those strings as ordinary text, which invalidates chat-template
  correctness measurements.

Do not report Gemma KVarN correctness from the old
`artifacts/kvarn-long-multiturn/multiturn_16k_4turns.txt` fixture. That file
contains Qwen-style markers and a high literal `<unk>` rate, so Gemma's normal
KV baseline is not a trustworthy reference on it.

For a fast Gemma PPL harness sanity check, use
`scripts/kvarn/fixtures/gemma4_predictable_sanity.txt`. It is deliberately
low-entropy text and is not a benchmark-quality language-model evaluation, but
it proves that the model, tokenizer, file-input path, and perplexity scorer can
produce sane absolute PPL before long KVarN runs start.

Also do not use
`external/terminal-bench/tasks/word2vec-from-scratch/wikitext-data/validation.txt`
as a Gemma health baseline. It is WikiText-style preprocessed text with thousands
of literal `<unk>` placeholders and artifacts such as `@-@`, `@,@`, and spaced
punctuation. In Gemma-family tokenizers, `<unk>` is a meaningful unknown-token
surface form, while other model families may split the same string as ordinary
text. That makes cross-model absolute PPL comparisons on this fixture
misleading. Use it only after fixture preflight passes and the Gemma normal-KV
baseline is independently sane.
