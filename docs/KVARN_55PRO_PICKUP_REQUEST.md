# KVarN 5.5 Pro Pickup Request

Repo: `https://github.com/JakeATX/llama.cpp`

Branch: `kvarn-atx-integration`

Base observed locally: `c0d5c9cebc4d5a166e73b3899d43a72c1664d14d`

## Objective

Make KVarN nearly lossless on quality and production speed for Gemma and Qwen, measured honestly against non-KVarN llama.cpp baselines.

Target gates:

- Quality: Gemma and Qwen should stay functionally lossless at 16k context against the matching non-KVarN baseline.
- Speed: KVarN K8/V8 and K8/V2 need to approach production usefulness, with the explicit goal of at least 95% of mainline 8-bit KV speed.
- Testing must run one GPU/model job at a time. Do not run concurrent correctness tests.
- Keep work scoped to KVarN/Cave Arm. Do not spend time on TurboQuant.

## Current Known State

The Gemma baseline harness problem that previously produced huge absolute PPL was fixed earlier: the usable Gemma baseline on the 16k fixture is now about `3.71`, not hundreds.

Current Gemma model:

`C:\Users\sjake\OneDrive\Documents\New project\models\gemma-4-12b-it-GGUF\gemma-4-12b-it-UD-Q4_K_XL.gguf`

Current fixture:

`artifacts\kvarn-gemma-fixtures\gemma4_chat_16k_512turns.txt`

Known baseline logits:

`artifacts\kvarn-accuracy\gemma4-q4-knownfixture16k-k8v8-current-kl-forceiswa-20260625\baseline-logits.base.bin`

Important reproduced failure:

Gemma donor5-only KVarN K8/V8 is enough to create severe KL tails:

- command shape: `--kv-cache-quant kvarn --kvarn-preset kvarn_k8v8_g128`
- env: `LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA=1`, `LLAMA_KVARN_LAYER_FILTER=5`
- latest artifact: `artifacts\kvarn-rootcause\gemma4-donor5-row83-kvarn-klmode-cumrow-q4-20260626`
- mean KLD: `0.186077`
- max KLD: `28.331320`
- PPL ratio: `1.033298`
- RMS delta-p: `12.402%`
- same-top: `95.874%`
- top same-exec spike: `logit_pos=11987`, `target_token=577`

Clean control:

Normal full-KV fallback is clean:

- artifact: `artifacts\kvarn-rootcause\gemma4-kl-normal-fallback-control-q4-20260625`
- mean KLD: `0`
- max KLD: about `0.000305`
- PPL ratio: `1.0`
- same-top: `100%`

This strongly suggests the Gemma issue is in the KVarN route/cache/decode/materialization path or compression sensitivity, not in the Gemma tokenizer/template/perplexity harness.

## Diagnostic Changes In This Branch

`tools/perplexity/perplexity.cpp` now has diagnostic tensor dumping support useful for row-bound root-cause work:

- `LLAMA_KVARN_TENSOR_DUMP_DIR`
- `LLAMA_KVARN_TENSOR_DUMP_FILTER`
- `LLAMA_KVARN_TENSOR_DUMP_LIMIT`
- `LLAMA_KVARN_TENSOR_DUMP_ROW`
- `LLAMA_KVARN_TENSOR_DUMP_SCORED_ROW_OFFSET`

Each dump JSON now includes:

- `name_occurrence`
- `name_row_base`
- `source_row`
- `inferred_full_row`
- `inferred_scored_row`
- source tensor shape/stride metadata

Reason this matters: naive occurrence mapping was wrong because KVarN intermediate tensors can have variable row heights. Similarity matching is also unsafe because the fixture has repeated patterns. Use `inferred_full_row` / cumulative row-base metadata instead of guessed occurrence arithmetic.

## Immediate Pickup Task

Do not start with speed. First finish the Gemma donor5 causal diagnosis.

1. Use the copied runnable binary workaround if Windows Smart App Control blocks the OneDrive build output:

   ```powershell
   $dstDir = Join-Path $env:LOCALAPPDATA 'kvarn-test-bin'
   New-Item -ItemType Directory -Force -Path $dstDir | Out-Null
   Copy-Item -LiteralPath 'build-kvarn-cuda-static-vs\bin\Release\llama-perplexity.exe' -Destination (Join-Path $dstDir 'llama-perplexity.exe') -Force
   ```

2. Run one GPU job at a time. Check `nvidia-smi` and process list before every model job.

3. Reproduce the donor5 K8/V8 KL-mode failure with row-bound dump metadata:

   ```powershell
   $env:LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA = '1'
   $env:LLAMA_KVARN_LAYER_FILTER = '5'
   $env:LLAMA_KVARN_KL_DUMP_CSV = '<out>\kl.csv'
   $env:LLAMA_KVARN_TENSOR_DUMP_DIR = '<out>'
   $env:LLAMA_KVARN_TENSOR_DUMP_FILTER = '^(attn_norm-5|Qcur-5|Qcur_normed-5|Qcur_pos-5|Kcur_normed-5|Kcur_pos-5|Vcur_normed-5|kvarn_iswa_kqv_out_2d-5|kvarn_iswa_kqv_wo-5|attn_post_norm-5|attn_out-5|l_out-5|result_output)$'
   $env:LLAMA_KVARN_TENSOR_DUMP_LIMIT = '2500'
   $env:LLAMA_KVARN_TENSOR_DUMP_ROW = '83'
   $env:LLAMA_KVARN_TENSOR_DUMP_SCORED_ROW_OFFSET = '8192'
   ```

4. Compare against a normal dump for the same absolute row. For the current top spike `logit_pos=11987`, use row `211` on normal 512-row batches and row `83` on 128-row scored batches, but bind by `inferred_full_row`, not by occurrence arithmetic.

5. Name the first-bad tensor class before patching KVarN math:

   - If `attn_norm-5` / `Qcur-5` are already different at the same `inferred_full_row`, the row binding or graph scheduling metadata is still wrong.
   - If `Qcur-5` is close but `kvarn_iswa_kqv_out_2d-5` diverges, focus on KVarN attention/decode/mask/materialized KV.
   - If attention output is close but downstream logits diverge, inspect residual wiring and result output indexing.

## Critic Requirements

Use a dedicated critic/devil's-advocate agent. Its standing objections should be treated as hard gates:

- No KVarN math patch until the first-bad tensor class is named from same-row metadata.
- No claims based on cosine/similarity matching alone.
- No concurrent model jobs.
- No `llama-bench.exe` until quality is stabilized.
- No TurboQuant side work.
- No speed victory claims until compared against non-KVarN mainline 8-bit KV.

## Known Windows Blocker

Windows Smart App Control blocked the freshly rebuilt OneDrive-path binary:

`did not meet the Enterprise signing level requirements`

The same binary ran from:

`%LOCALAPPDATA%\kvarn-test-bin\llama-perplexity.exe`

Use that copy path for local runs if the OneDrive build output is blocked.

## Validation Already Run

Serial build succeeded:

```powershell
cmake --build build-kvarn-cuda-static-vs --config Release --parallel 1 --target llama-perplexity
```

The copied binary reports:

```text
version: 3407 (c0d5c9ceb)
built with MSVC 19.44.35224.0 for x64
```

Do not treat the current branch as final. It contains working diagnostic and KVarN implementation state, but Gemma K8/V8 donor5 still fails the quality gate and needs root-cause closure before performance optimization resumes.
