# Round 32 mixed-attention boundary replay handoff

## Why this patch exists

Round 31 established that the selected Qwen ctx4096 store-boundary frame is correct:

- `k_frame_nmse=0`
- `v_frame_nmse=0`

but the same ctx4096 run still fails quality at about `+11.62%` PPL. The existing boundary dump already captures the mixed-attention input tensors and output, but the replay tool only covered the body-record store boundary. The next required diagnostic is to reconstruct the exact mixed-attention row independently and compare scores/probabilities/output.

## What changed

`ggml/src/ggml-cuda/ggml-cuda.cu`

- Removed the hard-coded `head_dim == 256` restriction from `LLAMA_KVARN_ATTN_BOUNDARY_DUMP`.
- Added optional `LLAMA_KVARN_ATTN_BOUNDARY_DUMP_HEAD_DIM=<D>`.
- Renamed the dump mode from `qwen36-256d-boundary-input` to `kvarn-mixed-attn-boundary-input`.

This matters because Gemma true KVarN+ISWA needs the same mixed-attention diagnostics at 512d.

`scripts/kvarn/replay_mixed_attn_boundary.py`

- Reads an existing `boundary.json` dump and reconstructs the selected row:
  - K token order: sink | body records | pending | tail
  - body K layout: `[record, D, G]` to token rows `[G, D]`
  - body V layout: `[record, G, D]`
  - mask row: `mask.bin`
  - query: `q.bin`
  - output: `warpqk_out.bin` or `mixed_out.bin`
- Computes:
  - `scores = K @ q * scale + mask`
  - `probs = softmax(scores)`
  - `out = probs @ V`
- Compares replayed output against the CUDA mixed-attention output.
- If future CUDA dumping writes `scores.bin` and `probs.bin`, the same script compares those too.

## How to run Qwen

Use the same failing ctx4096/chunks2 setup and add:

```powershell
$env:LLAMA_KVARN_ENABLE_PAPER_FRAME="1"
$env:LLAMA_KVARN_ATTN_BOUNDARY_DUMP="1"
$env:LLAMA_KVARN_ATTN_BOUNDARY_DUMP_FIRST_256D="1"
$env:LLAMA_KVARN_ATTN_BOUNDARY_DUMP_MIN_TOKENS="4096"
$env:LLAMA_KVARN_ATTN_BOUNDARY_DUMP_DIR="artifacts/kvarn-boundary/round32-qwen36-ctx4096-k4v2"
$env:LLAMA_KVARN_ATTN_BOUNDARY_DUMP_LIMIT="1"
```

Then:

```powershell
python scripts/kvarn/replay_mixed_attn_boundary.py `
  --dump artifacts/kvarn-boundary/round32-qwen36-ctx4096-k4v2 `
  --write-replay
```

## How to run Gemma

Use true KVarN+ISWA and add the generic head-dim filter:

```powershell
$env:LLAMA_KVARN_ENABLE_PAPER_FRAME="1"
$env:LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA="1"
$env:LLAMA_KVARN_ATTN_BOUNDARY_DUMP="1"
$env:LLAMA_KVARN_ATTN_BOUNDARY_DUMP_HEAD_DIM="512"
$env:LLAMA_KVARN_ATTN_BOUNDARY_DUMP_MIN_TOKENS="4096"
$env:LLAMA_KVARN_ATTN_BOUNDARY_DUMP_DIR="artifacts/kvarn-boundary/round32-gemma-ctx4096-k4v2"
$env:LLAMA_KVARN_ATTN_BOUNDARY_DUMP_LIMIT="1"
```

Then run the same replay script.

## Interpretation

If replayed output matches CUDA mixed-attention output:

- The selected mixed-attention row is internally consistent with packed body/scales.
- The remaining quality failure is likely cumulative quantization quality, layer routing, ISWA topology, or a different row/layer than the sampled one.
- Next: sample multiple layers/heads/queries and compare KVarN output against f16 baseline, not just KVarN internal replay.

If replayed output does not match:

- If scores differ: K layout, dequant, mask, scale, or token-order bug.
- If scores match but probabilities differ: softmax/reduction bug.
- If probabilities match but output differs: V layout/dequant/accumulation bug.

## Stop conditions

Do not resume speed patches until:

- Qwen3.6 ctx4096 PPL increase is under the 5% gate.
- Gemma true KVarN+ISWA ctx4096 PPL increase is under the 5% gate or fallback is explicitly kept as production default.
- Mixed-attention boundary replay passes on selected Qwen and Gemma body-active rows.

## Local validation results

Qwen3.6 MTP ctx4096/chunks2:

- Run artifact: `artifacts/kvarn-accuracy/round32-qwen36-ctx4096-k4v2-boundary`
- Boundary artifact: `artifacts/kvarn-boundary/round32-qwen36-ctx4096-k4v2/call_000000`
- f16 PPL `4.5837`
- KVarN PPL `5.1249`
- PPL increase `11.81%`
- Mixed-attention replay:
  - layer `3`, query `0`, head `0`
  - tokens `4096`, records `30`
  - CUDA mode `fused-batch`
  - `out_nmse=3.981132e-13`
  - `out_max_abs=4.470348e-07`

Gemma 4 12B true KVarN+ISWA ctx4096/chunks2:

- Run artifact: `artifacts/kvarn-accuracy/round32-gemma-ctx4096-k4v2-boundary`
- Boundary artifact: `artifacts/kvarn-boundary/round32-gemma-ctx4096-k4v2/call_000000`
- f16 PPL `404.2271`
- KVarN PPL `9466.9647`
- PPL increase `2241.99%`
- Mixed-attention replay:
  - layer `5`, query `0`, head `0`
  - tokens `4096`, records `30`
  - CUDA mode `warpqk-f16-dequant`
  - `out_nmse=1.050904e-12`
  - `out_max_abs=3.337860e-06`

These results show that the sampled mixed-attention rows are internally consistent with the packed KVarN body/scales. The remaining failures need f16-vs-KVarN activation comparison across layers/heads/queries and, for Gemma, ISWA topology/window-state investigation.
