# KVarN Gemma/Qwen Quality V4 Patch Handover

## Purpose

This patch is quality-first. It fixes two classes of bug that can make Gemma 4
KVarN donor/layer diagnostics misleading or outright wrong:

1. KVarN paper-frame is made the default graph path. The validated Qwen/Gemma
   quality runs depend on paper-frame; no-paper-frame is now an explicit
   diagnostic negative-control path via `LLAMA_KVARN_DISABLE_PAPER_FRAME=1`.
2. `llama-perplexity` tensor dump row binding is corrected for ggml 3D/4D row
   semantics and for scored-output tensors.

Do not resume speed work until the donor5 quality diagnosis has been rerun with
this patch.

## Files touched

- `src/llama-graph.cpp`
- `tools/perplexity/perplexity.cpp`
- `scripts/kvarn/compare_tensor_dump_rows.py`
- `docs/KVARN_GEMMA_QUALITY_V4_HANDOVER.md`

## Key behavior changes

### Paper-frame default

`LLAMA_KVARN_ENABLE_PAPER_FRAME=1` is no longer required for the quality path.
KVarN graph construction now uses paper-frame unless:

```powershell
$env:LLAMA_KVARN_DISABLE_PAPER_FRAME = '1'
```

The old enable flag is still parsed for compatibility, but `ENABLE=0` is no
longer a silent production opt-out. Use the disable flag only for negative
controls.

### Tensor dump target rows

The dump callback now supports:

```powershell
LLAMA_KVARN_TENSOR_DUMP_SOURCE_ROW
LLAMA_KVARN_TENSOR_DUMP_TARGET_ROW
LLAMA_KVARN_TENSOR_DUMP_FULL_ROW
LLAMA_KVARN_TENSOR_DUMP_SCORED_ROW
LLAMA_KVARN_TENSOR_DUMP_SCORED_ROW_OFFSET
```

For the current spike:

```text
full/logit row: 11987
scored row offset: 8192
scored row: 3795
```

Use:

```powershell
$env:LLAMA_KVARN_TENSOR_DUMP_TARGET_ROW = '11987'
$env:LLAMA_KVARN_TENSOR_DUMP_SCORED_ROW_OFFSET = '8192'
```

The callback first binds full-row tensors via `11987`, then also allows
scored-output tensors like `result_output` to bind via `11987 - 8192 = 3795`.

## Required validation

Build:

```powershell
cmake --build build-kvarn-cuda-static-vs --config Release --parallel 1 --target llama-perplexity
python -m py_compile scripts/kvarn/compare_tensor_dump_rows.py
```

If Windows Smart App Control blocks the OneDrive binary:

```powershell
$dstDir = Join-Path $env:LOCALAPPDATA 'kvarn-test-bin'
New-Item -ItemType Directory -Force -Path $dstDir | Out-Null
Copy-Item -LiteralPath 'build-kvarn-cuda-static-vs\bin\Release\llama-perplexity.exe' -Destination (Join-Path $dstDir 'llama-perplexity.exe') -Force
$perplexity = Join-Path $dstDir 'llama-perplexity.exe'
```

Run one GPU/model job at a time. Do not run concurrent correctness jobs.

## Donor5 rerun

```powershell
$out = 'artifacts\kvarn-rootcause\gemma4-donor5-targetrow11987-k8v8-v4'
$perplexity = "$env:LOCALAPPDATA\kvarn-test-bin\llama-perplexity.exe"

$env:LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA = '1'
$env:LLAMA_KVARN_LAYER_FILTER = '5'
$env:LLAMA_KVARN_KL_DUMP_CSV = "$out\kl.csv"
$env:LLAMA_KVARN_TENSOR_DUMP_DIR = $out
$env:LLAMA_KVARN_TENSOR_DUMP_FILTER = '^(attn_norm-5|Qcur-5|Qcur_normed-5|Qcur_pos-5|Kcur_normed-5|Kcur_pos-5|Vcur_normed-5|kvarn_iswa_kqv_out_2d-5|kvarn_iswa_kqv_wo-5|kvarn_iswa_kqv_wo_b-5|attn_post_norm-5|attn_out-5|l_out-5|result_output)$'
$env:LLAMA_KVARN_TENSOR_DUMP_LIMIT = '2500'
$env:LLAMA_KVARN_TENSOR_DUMP_TARGET_ROW = '11987'
$env:LLAMA_KVARN_TENSOR_DUMP_SCORED_ROW_OFFSET = '8192'

& $perplexity `
  -m 'C:\Users\sjake\OneDrive\Documents\New project\models\gemma-4-12b-it-GGUF\gemma-4-12b-it-UD-Q4_K_XL.gguf' `
  -f artifacts\kvarn-gemma-fixtures\gemma4_chat_16k_512turns.txt `
  -c 16384 -b 128 -fa off `
  --kl-divergence-base artifacts\kvarn-accuracy\gemma4-q4-knownfixture16k-k8v8-current-kl-forceiswa-20260625\baseline-logits.base.bin `
  --kv-cache-quant kvarn `
  --kvarn-preset kvarn_k8v8_g128 `
  --kvarn-iters 16
```

Run the matching normal control into:

```text
artifacts\kvarn-rootcause\gemma4-targetrow11987-normal-v4
```

Then compare:

```powershell
python scripts\kvarn\compare_tensor_dump_rows.py `
  --base-dir artifacts\kvarn-rootcause\gemma4-targetrow11987-normal-v4 `
  --kvarn-dir artifacts\kvarn-rootcause\gemma4-donor5-targetrow11987-k8v8-v4 `
  --row 11987 `
  --row-mode full `
  --map kvarn_iswa_kqv_out_2d-5=kqv_out-5 `
  --map kvarn_iswa_kqv_wo-5=kqv_wo-5 `
  --map kvarn_iswa_kqv_wo_b-5=kqv_wo_b-5 `
  --csv artifacts\kvarn-rootcause\gemma4-donor5-targetrow11987-rowcompare-v4.csv `
  --first-bad-nmse 1e-4
```

## Stop rules

- If donor5 K8V8 is clean after this patch, the prior failure was likely a
  no-paper-frame footgun plus broken row diagnostics. Re-run full Gemma K8V8 and
  then K8V2 quality before touching speed.
- If donor5 remains bad, patch only the first-bad tensor class named by
  `compare_tensor_dump_rows.py`. Do not patch KVarN math from KL tails alone.
