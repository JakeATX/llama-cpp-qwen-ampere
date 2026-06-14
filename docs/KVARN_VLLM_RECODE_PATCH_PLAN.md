# KVarN vLLM Reference Port — RECODE Patch Plan

## Reviewed sources

Read together:

- `docs/KVARN_VLLM_REFERENCE_PORT_SPEC.md`
- `docs/KVARN_PRODUCTION_PATCH_HANDOFF.md`
- `docs/KVARN_PAPER_FIDELITY_AUDIT.md`
- `docs/KVARN_NEXT_PATCHES_HANDOVER.md`
- Huawei/vLLM reference:
  - `vllm/model_executor/layers/quantization/kvarn/sinkhorn.py`
  - `vllm/v1/attention/ops/kvarn_store.py`
  - `vllm/model_executor/layers/quantization/kvarn/config.py`

## Current state to preserve

The pending-K layout fix is accepted and must not be reverted:

```cpp
k_tile[d * group_size + g] = pending[d + g * pending_head_stride]
```

The remaining failure is not just low-bit resolution:

| Model / mode | Baseline PPL | KVarN PPL | Increase |
| --- | ---: | ---: | ---: |
| Qwen3.6 MTP ctx4096 paper-frame k4v2 | 4.5843 | 5.1155 | 11.59% |
| Qwen3.6 MTP ctx4096 paper-frame k8v8 | 4.5843 | 5.1320 | 11.95% |
| Gemma 4 12B true KVarN+ISWA ctx4096 k4v2 | 418.2027 | 944.8251 | 125.93% |
| Gemma 4 12B true KVarN+ISWA ctx4096 k8v8 | 418.2027 | 520.7935 | 24.53% |

A failing k8/v8 diagnostic means we still have a reference-math/topology/path bug or a severe normalization mismatch. Do not work on long-context speed until the 4096 accuracy gates pass.

## Patch order

### Patch 0001 — independent vLLM-style oracle

Apply `0001-kvarn-add-vllm-oracle.patch`.

This adds:

```text
scripts/kvarn/kvarn_vllm_oracle.py
scripts/kvarn/run_vllm_oracle_selftest.ps1
```

The oracle is deliberately independent of production CUDA store/dequant/mixed-attn code. It implements:

1. normalized Sylvester Hadamard along head_dim;
2. K body layout `[D, G]`;
3. V body layout `[G, D]`;
4. log-domain std Sinkhorn with best-iteration selection;
5. RTN quant/dequant;
6. rotated-Q attention;
7. output unrotation.

Run:

```powershell
scripts/kvarn/run_vllm_oracle_selftest.ps1
```

Stop if this fails.

### Patch 0002 — wire oracle into real body-record dumps

Next patch should add:

```text
LLAMA_KVARN_DEBUG_BODY_RECORD=<layer>:<record>:<head>
LLAMA_KVARN_DEBUG_BODY_RECORD_DIR=<artifact-dir>
```

Dump all of these for a selected store:

```text
raw source K/V tile
rotated source K/V tile
balanced/normalized K/V tile
packed K/V bytes
K/V scale tensors
dequantized rotated K/V tile
logical token span for the record
direct-vs-pending source type
paper-frame flag state
ISWA/SWA/base layer routing
```

Then a Python command should compare the dump against the oracle.

### Patch 0003 — opt-in log/std Sinkhorn production path

Do not replace current RMS Sinkhorn by default. Add:

```text
LLAMA_KVARN_ENABLE_LOG_STD_SINKHORN=1
```

The path must match the vLLM reference:

```text
std, not RMS
ddof/correction=1 equivalent
std clamp [1e-3, 1e3]
log scale clamp [-0.3, 10.0]
best-iteration selection by imbalance
scale absorption:
  K: s_col_K = s_row_sinkhorn * rtn_scale
     zp_K    = s_row_sinkhorn * rtn_zp
     s_row_K = s_col_sinkhorn
  V: s_col_V = s_col_sinkhorn
     s_row_V = s_row_sinkhorn * rtn_scale
     zp_V    = s_row_sinkhorn * rtn_zp
```

Implement CPU/reference first and only then CUDA.

### Patch 0004 — ISWA topology audit

Gemma is far worse than Qwen after k8/v8, so add a Gemma-like ISWA test that simulates:

1. fill sink/tail;
2. evict tail to pending;
3. seal pending to body;
4. advance until record reuse/wrap;
5. compare dequantized body record to independent CPU oracle at the expected logical token positions.

Focus on:

```text
base vs SWA layer split
tail_start under wrap
record span mapping
mask order: sink | body | pending | tail
rotation state during tail-to-pending copy
record epoch invalidation
```

### Patch 0005 — speed only after correctness

Only after both long-context accuracy gates pass:

```text
Qwen3.6 ctx4096 PPL increase <= 5%
Gemma true KVarN+ISWA ctx4096 PPL increase <= 5%
```

then resume:

```text
decode parallel body-record attention
fused dual-scale dequant/attention
long pp4096/tg4096 parity
```

## Required acceptance gates

```powershell
ctest --test-dir build-kvarn-cuda-static-vs -C Release `
  -R "test-batch-split|test-kvarn-kv|test-kvarn-cuda|test-kvarn-server-load-failure" `
  --output-on-failure

python scripts/kvarn/kv_memory_estimate.py --self-test
scripts/kvarn/run_vllm_oracle_selftest.ps1
```

Then:

```powershell
$env:LLAMA_KVARN_ENABLE_PAPER_FRAME="1"
$env:LLAMA_KVARN_ENABLE_LOG_STD_SINKHORN="1"
```

Run Qwen3.6 and Gemma ctx4096/chunks2 accuracy gates from the spec.

## Non-negotiables

- Do not benchmark speed before accuracy.
- Do not revert the pending-K layout fix.
- Keep new recipes default-off until proven.
- Treat k8/v8 failure as evidence of topology/reference mismatch.
- Packed-vs-split NMSE `0` is necessary but not sufficient; the independent oracle is now the required correctness oracle.
