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

2. **Suspected correctness bug to verify first.** The Hadamard rotation is
   applied to the quantized **body** K/V but not to Q, not to the plain-f16
   **sink/tail** K/V, and the output is never un-rotated. Within one attention
   op, body tokens use rotated K while sink/tail tokens use un-rotated K, both
   dotted against the same un-rotated Q - the two cannot both be correct. Run
   the gate; if KVarN perplexity is far above f16, this is the cause.

3. **Why it is slower than mainline** is architectural, not a tuning miss: the
   "exact" production paths dequantize the cache to an f32/f16 scratch mirror
   and then run dense attention, which is *more* memory traffic and compute
   than mainline f16 + flash-attention. The only path that could win (warp-QK
   reading packed/f16 directly) is disabled for 256d.

## Finding 1 - Hadamard rotation is applied inconsistently

Paper / vLLM premise: an orthonormal Hadamard rotation `H` in the channel
dimension preserves attention because `(Hq)*(Hk) = q*k`. That identity only
holds if **both** Q and K are rotated (and the V output un-rotated).

Store side rotates the body:
- `src/llama-kv-cache-kvarn.cpp:347-348` - body K/V `llama_kvarn_hadamard_channels(...)`
- `ggml/src/ggml-cuda/kvarn.cu:849-855, 900-906` - same on the CUDA store path

Read side does **not** complete the transform:
- `src/llama-graph.cpp:3255` - `q_cur` enters `ggml_kvarn_attn_mixed` un-rotated.
  The `ggml_mul_mat_aux(q_cur, self_k_rot)` at `3287-3290` is in the *non-KVarN*
  fall-through path, after the KVarN block returns.
- `ggml/src/ggml-cuda/kvarn.cu:2964-2966` - body K is reconstructed as
  `(kq*k_s_col[d] + k_zp[d])*k_s_row[g]` = `H*k`, dotted against raw `q`; no
  inverse `H`.
- `src/llama-kv-cache-kvarn.cpp:439-443` - sink/tail K/V stored raw
  (`append_fp32_as_fp16`), i.e. **un-rotated**, while the body is rotated.

Because every existing test reconstructs the rotated body identically on both
sides, they all pass while potentially being wrong vs true attention. Confirm
with the gate before any further perf work.

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
