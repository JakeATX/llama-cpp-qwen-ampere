# KVarN Round 40 Graph Audit Handoff

## Status

K8V8 graph-correctness proof is now green for both primary targets:

- Qwen3.6 MTP, paper-frame, KVarN layers `3-39:4`, ctx4096/chunks2: `+0.15%` PPL.
- Gemma 4 12B true KVarN+ISWA, paper-frame, KVarN layers `5-47:6`, ctx4096/chunks1: `+0.02%` PPL.

This supersedes the earlier Gemma catastrophic result in this document. The root cause was not body-record quantization or body-source layout; it was corrupted transferred KQ mask consumption in the KVarN CUDA mixed-attention path under Gemma+ISWA body-active prefill. Host-side mask construction was correct, but the device boundary dump showed wrong visible spans. The production path now uses an internal causal mask mode (`mask_type = 3`) for KVarN mixed attention when a causal mask is present, and the replay scripts synthesize the same causal mask independently.

Do not resume speed optimization until K4V4/K4V2 quality is measured after this fix. K8V8 is a graph proof configuration, not the intended production compression setting.

## Superseding Fix: Internal Causal Mask

### Hypothesis

Gemma+ISWA long-context failure is caused by CUDA mixed attention reading a corrupted transferred KQ mask, even though host-side KQ mask fill is correct.

### Evidence Before Fix

- Gemma pair filter `5,11`, ctx4096/b128/chunks1, K8V8 paper-frame:
  - Artifact: `artifacts/kvarn-rootcause/loop74-gemma4-pair5-11-masktrace`
  - KVarN-only PPL: `2757.1963`
- Host trace from `llama_kv_cache_kvarn::set_input_kq_mask()` was correct:
  - `n_kv = 1280`, `n_q = 128`, `last_pos = 1279`, `q_base = 1152`
  - visible counts `q0=1153`, `q64=1217`, `q127=1280`
- CUDA boundary full-mask dump was not correct:
  - rows `0-101` had random finite entries;
  - row `102` was partially corrupted;
  - rows `103+` looked causal.
- Old boundary replay was self-referential because it replayed the same corrupted mask dump consumed by the CUDA op.

### Patch

- `src/llama-kv-cache-kvarn.cpp`
  - `set_input_kq_mask()` now derives query positions from the final ubatch position instead of trusting every `ubatch->pos[q]`.
  - Added `LLAMA_KVARN_MASK_TRACE` / `LLAMA_KVARN_MASK_TRACE_LIMIT` diagnostics.
- `ggml/src/ggml-cuda/ggml-cuda.cu`
  - KVarN mixed attention sets `kq_mask_type = 3` when a mask exists, unless `LLAMA_KVARN_DISABLE_INTERNAL_CAUSAL_MASK=1` or `LLAMA_KVARN_ATTN_DISABLE_MASK=1`.
  - Boundary dumps synthesize causal masks for `mask_type = 3` instead of copying the transferred device mask.
- `ggml/src/ggml-cuda/kvarn.cu`
  - Added causal-mask limit computation and threaded it through all KVarN mixed-attention CUDA modes, including scratch-ref, split, fused, sink/tail, warpqk, scalar-QT, and scalar-QT-GQA.
- `scripts/kvarn/replay_mixed_attn_boundary.py`
  - Synthesizes causal masks for `mask_type = 3`.
- `scripts/kvarn/replay_f16_truth_boundary.py`
  - Synthesizes causal masks for `mask_type = 3`.
- `src/llama-context.cpp` and `scripts/kvarn/run_unsupported_smoke.ps1`
  - Front-load CUDA KVarN env validation so unsupported-mode smoke checks cannot be masked by early capability probes.

### Evidence After Fix

Gemma pair `5,11` is no longer catastrophic:

- Artifact: `artifacts/kvarn-rootcause/loop75-gemma4-pair5-11-layer11-call4-internal-causal-boundary`
- Accuracy: f16 PPL `414.0522`, KVarN PPL `422.3349`, increase `+2.00%`
- Boundary `call_000004`:
  - `mask_type = 3`
  - `n_tokens = 1792`, `n_queries = 128`
  - visible rows `[q0,q64,q127] = [1665,1729,1792]`
  - self replay worst full-Q/O NMSE `3.579354e-12`
  - independent f16-truth full-Q/O worst NMSE `3.230363e-05`

Full Gemma true KVarN+ISWA is now green:

- Artifact: `artifacts/kvarn-rootcause/loop76-gemma4-full-k8v8-c4096-internal-causal`
- Model: `gemma-4-12b-it-UD-Q4_K_XL.gguf`
- Config: `LLAMA_KVARN_ENABLE_PAPER_FRAME=1`, `LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA=1`, `kvarn_k8v8_g128`, `--kvarn-iters 16`, `-fa off`, ctx4096, batch128, chunks1
- Expected/routed KVarN layers: `5-47:6`
- f16 PPL `414.0522`
- KVarN PPL `414.1195`
- Increase `+0.02%`

Full Qwen remains green:

- Artifact: `artifacts/kvarn-rootcause/loop77-qwen36-full-k8v8-c4096-chunks2-internal-causal`
- Model: `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
- Config: `LLAMA_KVARN_ENABLE_PAPER_FRAME=1`, `kvarn_k8v8_g128`, `--kvarn-iters 16`, `-fa off`, ctx4096, batch4096, chunks2, `-ncmoe 34`
- Expected/routed KVarN layers: `3-39:4`
- f16 PPL `4.5837`
- KVarN PPL `4.5904`
- Increase `+0.15%`

### Devil's Advocate Verdict

`PASS` for the K8V8 graph-correctness proof on Qwen and Gemma true KVarN+ISWA.

`WARN` for production readiness:

- K8V8 is not the production compression target.
- K4V4 and K4V2 must be re-measured after the causal-mask fix.
- Short speed parity has not been rerun after this correctness fix.
- Gemma pair-filter `5,11` remains `+2.00%` while full Gemma is `+0.02%`; treat pair-filter as a diagnostic shape, not production evidence.

## Code Changes In This Round

- Added CUDA graph replay guards for stateful KVarN row-copy/store paths in `ggml/src/ggml-cuda/ggml-cuda.cu`.
  - Store ops are never captured.
  - KVarN-lineage `GET_ROWS` / `SET_ROWS` are never captured.
- Fixed Qwen multi-head pending body-store source offset in `ggml/src/ggml-cuda/kvarn.cu`.
  - Old code used token stride as the per-head base offset.
  - New code offsets pending K/V by `ih * head_dim` and keeps the token stride as stride.
- Fixed CUDA log-std scale update to match the CPU/vLLM-style oracle clamp/update semantics.
- Added diagnostic body-record call range filtering:
  - `LLAMA_KVARN_DEBUG_BODY_RECORD_CALL_START`
  - `LLAMA_KVARN_DEBUG_BODY_RECORD_CALL_END`
- Hardened replay scripts:
  - `compare_body_records_to_kcur.py` now accepts model-specific K/V tensor names.
  - `replay_f16_truth_boundary.py` now canonicalizes direct-record batch IDs with `record0 + record`.
  - Added `compare_sinktail_to_kcur.py` to independently prove sink/tail cache source tokens.
- Added CUDA regression coverage for the multi-head pending store path.

## Tests Run

```powershell
cmake --build build-kvarn-cuda-static-vs --config Release --target test-kvarn-cuda-scratch-ref llama-perplexity --parallel 8
ctest --test-dir build-kvarn-cuda-static-vs -C Release -R "test-batch-split|test-kvarn-kv|test-kvarn-cuda|test-kvarn-server-load-failure|test-arg-parser" --output-on-failure
python scripts/kvarn/kv_memory_estimate.py --self-test
python scripts/kvarn/kvarn_vllm_oracle.py --self-test --head-dims 128,256,512 --presets k4v2,k4v4,k8v8 --iters 16
python -m py_compile scripts/kvarn/compare_sinktail_to_kcur.py scripts/kvarn/compare_body_records_to_kcur.py scripts/kvarn/replay_f16_truth_boundary.py
```

All passed.

## Qwen Result

Qwen3.6 MTP K8V8 paper-frame ctx4096/chunks2 is now green:

- Layer 39 only:
  - Artifact: `artifacts/kvarn-layer-bisect/loop17-qwen36-layer39-k8v8-body-source-fixed-b4096`
  - f16 PPL `4.5837`
  - KVarN PPL `4.5837`
  - Increase `0.00%`
- Full KVarN layers `3-39:4`:
  - Artifact: `artifacts/kvarn-layer-bisect/loop18-qwen36-all-k8v8-body-source-fixed-b4096`
  - f16 PPL `4.5837`
  - KVarN PPL `4.5860`
  - Increase `0.05%`

Body-source proof:

- Head 0 artifact: `artifacts/kvarn-rootcause/loop14-qwen36-layer39-head0-b4096-rowcopy-graphguard-first3`
- Head 1 artifact: `artifacts/kvarn-rootcause/loop16-qwen36-layer39-head1-b4096-pending-headfix-first3`
- Both pass against independent `Kcur/Vcur` dumps.
- Offset / no-paper-frame / swap-head negative controls fail as expected.

## Earlier Gemma Result Before Internal Causal Mask Fix

This section is retained as historical evidence for the bug hunt. It is superseded by the internal causal mask fix above.

Before the fix, Gemma true KVarN+ISWA K8V8 paper-frame ctx4096/chunks2 was red:

- Artifact: `artifacts/kvarn-layer-bisect/loop19-gemma4-12b-true-iswa-k8v8-b4096`
- f16 PPL `404.2271`
- KVarN PPL `4849.4759`
- Increase `1099.69%`

This is true KVarN+ISWA, not fallback. Logs show KVarN layers `5,11,17,23,29,35,41,47`.

## Gemma Claims Proven

### Body Source Is Not The Sampled Root Cause

Layer 47 source checks pass for both first and second chunks:

- First chunk artifact: `artifacts/kvarn-rootcause/loop21-gemma4-layer47-head0-true-iswa-kcurpos-vnorm-first3`
  - `Kcur_pos-47` / `Vcur_normed-47`
  - worst input/source NMSE `4.37e-08`
- Second chunk artifact: `artifacts/kvarn-rootcause/loop22-gemma4-layer47-head0-true-iswa-secondchunk-first3`
  - worst input/source NMSE `4.30e-08`
- Wrong-chunk / offset controls fail with NMSE around `1.2+`.

### Full-Window Mixed Attention Is Not The Sampled Root Cause

All Gemma KVarN layers full-window mixed-attention boundaries replay cleanly.

Artifact: `artifacts/kvarn-rootcause/loop26-gemma4-alllayers-fullwindow-boundaries-fullqo`

Worst full-row f16-truth NMSE by layer:

| Layer | Worst f16-truth full-row NMSE |
|---:|---:|
| 5 | `6.10e-05` |
| 11 | `5.63e-05` |
| 17 | `9.99e-05` |
| 23 | `1.26e-04` |
| 29 | `1.72e-04` |
| 35 | `1.56e-04` |
| 41 | `1.88e-04` |
| 47 | `8.52e-05` |

KVarN self-replay is near exact for all of these boundaries (`~1e-10` or better full-row NMSE).

### Zero-Body Exact Sink/Tail Still Fails

This disproves "quantized body records are the whole Gemma issue."

Artifact: `artifacts/kvarn-accuracy/loop33-gemma4-true-iswa-k8v8-tail3968-chunks1`

- Args include `--kvarn-tail-tokens 3968`
- Logs show `body records = 0` for all KVarN layers.
- f16 PPL `431.7440`
- KVarN PPL `11695.3973`
- Increase `2608.87%`

No-paper-frame zero-body was also catastrophic:

- Artifact: `artifacts/kvarn-accuracy/loop37-gemma4-true-iswa-k8v8-tail3968-nopaper-chunks1`
- f16 PPL `431.7440`
- KVarN PPL `44255.3756`

### Zero-Body Sink/Tail Source Is Correct For Layer 5

Artifact: `artifacts/kvarn-rootcause/loop38-gemma4-tail3968-layer5-sinktail-source`

Command:

```powershell
python scripts/kvarn/compare_sinktail_to_kcur.py --boundary artifacts/kvarn-rootcause/loop38-gemma4-tail3968-layer5-sinktail-source/boundary/call_000000 --tensor-dump artifacts/kvarn-rootcause/loop38-gemma4-tail3968-layer5-sinktail-source/kcur-vcur --layer 5 --k-tensor-name Kcur_pos-5 --v-tensor-name Vcur_normed-5 --context-size 4096 --chunk-index 0 --paper-frame --max-nmse 1e-6
```

Result:

- K NMSE `4.315e-08`
- V NMSE `4.319e-08`
- Offset negative control produces NMSE `>1`.

### Gemma Divergence First Explodes At Layer 11

Default K8V8 paper-frame, ctx4096/chunks1 ordered `attn_out` comparison:

Artifact: `artifacts/kvarn-rootcause/loop30-gemma4-layers5to11-attnout-compare`

| Layer | `attn_out` NMSE vs f16 |
|---:|---:|
| 5 | `9.77e-06` |
| 6 | `4.35e-04` |
| 7 | `1.26e-03` |
| 8 | `2.09e-03` |
| 9 | `1.44e-03` |
| 10 | `1.44e-03` |
| 11 | `2.317e+00` |

Layer 11 Q/K/V source comparison:

Artifact: `artifacts/kvarn-rootcause/loop32-gemma4-layer11-qkv-source-compare`

| Tensor | NMSE vs f16 |
|---|---:|
| `attn_norm-11` | `1.17e-03` |
| `Qcur_pos-11` | `1.09e-03` |
| `Kcur_pos-11` | `1.24e-03` |
| `Vcur_normed-11` | `1.24e-03` |
| `kqv_out-11` | `3.25e-01` |
| `attn_out-11` | `2.317e+00` |

Interpretation: layer 11 attention is highly sensitive to the existing activation drift, but the layer 11 KVarN mixed-attn implementation is correct for its own Q/K/V inputs. The next proof needs to determine whether that drift is acceptable model sensitivity or a hidden graph/cache lifecycle mismatch before layer 11.

## Earlier Devil's Advocate Verdict Before Internal Causal Mask Fix

This verdict is superseded by the current verdict above. At this point in the investigation the verdict was `BLOCK` for production correctness.

Reasons:

- Gemma true KVarN+ISWA K8V8 remains far above the required `<=1%` PPL/KL threshold.
- Layer-filter isolation does not currently work for KVarN+ISWA. `LLAMA_KVARN_LAYER_FILTER=47` still routes all Gemma KVarN layers.
- Existing oracles prove many boundaries but do not yet prove full transformer block outputs or single-layer Gemma KVarN behavior.

## Superseded Next Mandatory Loop

The following was the pre-fix next loop. It is no longer the immediate next step now that full Gemma K8V8 is green, but it remains useful if Gemma regresses again:

1. Implement diagnostic Gemma KVarN+ISWA layer filtering with normal-ISWA fallback for excluded non-SWA layers.
   - Current `llama_kv_cache_kvarn_iswa` has only KVarN base + normal SWA.
   - A useful diagnostic needs normal full-attention KV for excluded non-SWA layers too.
   - Graph reuse must be disabled for mixed normal/KVarN ISWA filter graphs.

2. Run single-layer Gemma exact-mode gates:
   - layer 5 only, K8V8, tail3968, paper-frame
   - layer 11 only, K8V8, tail3968, paper-frame
   - then `5,11`, then full `5-47:6`

3. Add full block-output dumps:
   - after layer residual output, not just `attn_out`
   - FFN/MoE router logits and selected expert path for layers 6-11
   - compare ordered f16 vs KVarN sequences

4. Only after single-layer exact-mode is near f16 should body K8V8/K4V4/K4V2 quality be re-evaluated.

## Current Next Mandatory Loop

1. Re-run quality gates for production candidate presets after the causal-mask fix:
   - Qwen3.6 MTP: K4V4, then K4V2, ctx4096/chunks2, expected layers `3-39:4`, `-ncmoe 34`.
   - Gemma 4 12B true KVarN+ISWA: K4V4, then K4V2, ctx4096, expected layers `5-47:6`, `LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA=1`.
2. If K4V4 is green, run short mainline parity:
   - Qwen `pp512,tg64`, `-r 3`, `-ncmoe 34`.
   - Gemma true KVarN+ISWA `pp512,tg64`, `-r 3`, no fallback.
3. If K4V4 quality or speed fails, capture one boundary and compare:
   - K8V8 vs K4V4 dequant error.
   - raw f16 attention vs K4V4 attention output.
   - per-layer contribution to PPL/KL.
4. Only after K4V4 is characterized should K4V2 be considered a production option.

## Do Not Do Yet

- Do not optimize speed before K4V4/K4V2 quality gates are rerun after the causal-mask fix.
- Do not argue K4V4/K4V2 production quality from pre-fix measurements.
- Do not count Gemma fallback as true KVarN data.
- Do not accept mixed-attn self-replay as sufficient proof; use independent source/window/block oracles.
