# KVarN paper-fidelity audit

Date: 2026-06-13

Branch: `kvarn-atx-integration`

Audited code anchor: `aafdb00f1` plus local Round 26 script/doc changes.

Reference material:

- Paper: <https://arxiv.org/html/2606.03458v1>
- Author implementation clone used for local comparison: `C:\Users\sjake\OneDrive\Documents\New project\KVarN-reference`, commit `a601d2a`
- Relevant reference files:
  - `vllm/model_executor/layers/quantization/kvarn/sinkhorn.py`
  - `kvarn_mla_tilepack.py`
  - `kvarn_mla_tile_validate.py`

## Executive conclusion

The branch has the conceptual KVarN pieces, but the implementation shape does not yet match the paper's optimized systems assumptions.

Conceptually present:

- Hadamard rotation of body tiles before quantization.
- Dual row/column variance scales.
- Per-channel/per-token scale storage for K and V.
- Dequant formulas that apply the extra KVarN scale.
- FP16 sink/tail plus quantized body region.

Not yet faithful to the paper/reference systems shape:

- Body-store batching is mostly graph-level batching; CUDA still loops over record/head tiles internally.
- The direct-record batch path still launches the Hadamard, Sinkhorn, quantize, and finalize phases per tile.
- Long decode and long prefill still often materialize F32/F16 body mirrors or dequant scratch rather than applying dual scales directly inside a high-occupancy attention/dequant path.
- Production-model f16-vs-KVarN accuracy at 4096 context is currently failing, so no lower-iteration or fused optimization should be promoted until correctness is explained.

## Paper constraints that matter

From Sec. 3.3, KVarN is Hadamard rotation in the channel dimension plus dual variance scaling across token and channel dimensions. The additional scale is intended to add only one multiply per token/channel during dequantization.

From Sec. 4.2 and Appendix I, the reported overhead assumes an optimized batched/fused implementation. Appendix I specifically evaluates the dual scale as fused into dequantization, not as an extra scratch materialization pass.

The practical implication is that a serial per-record/per-head Sinkhorn pipeline is not the paper's claimed runtime shape.

## Reference implementation shape

The local author-reference prototypes make the intended frame explicit:

- `kvarn_mla_tilepack.py:35-45`
  - rotate a `[GROUP, R]` tile into channel frame;
  - run `variance_normalize(rot)`;
  - quantize per channel;
  - absorb one variance scale into RTN scale/zp;
  - store the other scale per token.
- `kvarn_mla_tilepack.py:58-68`
  - dequant reconstructs the rotated tile as `(q * scale_abs + zp_abs) * sr`.
- `kvarn_mla_tile_validate.py:51-55`
  - K scores are computed as rotated query dot dequantized rotated tile;
  - V attention output is compared after unrotating the dequantized value tile.
- `sinkhorn.py:55` and `sinkhorn.py:102`
  - expose scalar and batched variance normalization APIs; the batched form is the shape our CUDA path needs to emulate for records * heads tiles.

## Current llama.cpp implementation audit

### 1. Scale layout is conceptually aligned

Code:

- `src/llama-kv-cache-kvarn.cpp:223-224`
  - K scale floats are `2*head_dim + group_size`.
  - V scale floats are `head_dim + 2*group_size`.
- `ggml/src/ggml-cuda/kvarn.cu:726-755`
  - K finalization stores RTN scale/zp plus token scale.
  - V finalization stores channel scale plus RTN token scale/zp.
- `ggml/src/ggml-cuda/kvarn.cu:1372-1410`
  - Dequant applies both standard RTN scale/zp and the extra KVarN scale.

Assessment:

- The stored metadata shape matches the K/V asymmetric layout used by KVarN/KIVI-style KV quantization.
- This is not the main blocker.

### 2. Hadamard placement is only partially proven

Code:

- Store path:
  - K body store applies Hadamard over columns/channels in `ggml/src/ggml-cuda/kvarn.cu:903-909`.
  - V body store applies Hadamard over rows/channels in `ggml/src/ggml-cuda/kvarn.cu:954-960`.
  - Pipelined K/V store does the same at `ggml/src/ggml-cuda/kvarn.cu:1032-1045`.
- Non-KVarN graph path applies model-provided rotations before attention and after V output:
  - K/Q rotation: `src/llama-graph.cpp:3328-3330`.
  - V rotation before store: `src/llama-graph.cpp:3333-3334`.
  - output V inverse/application: `src/llama-graph.cpp:3403-3404`.
- KVarN graph path enters `ggml_kvarn_attn_mixed` before the later non-KVarN rotation block:
  - `src/llama-graph.cpp:3296-3303`.

Assessment:

- For layers without model-provided `self_k_rot` / `self_v_rot`, the KVarN body Hadamard can be internally consistent if attention kernels dot a matching rotated query or otherwise keep K/V in the correct frame.
- For any path with non-null model rotations, the KVarN path currently has explicit guards or bypasses in some places, but the full invariant is not documented in code.
- Production f16-vs-KVarN accuracy failures at 4096 mean this must be treated as an active correctness hypothesis, not a closed issue.

Required next check:

- Add a low-level KVarN tile-frame unit: store one K/V body tile, dequant it, compare:
  - K score: `qH . deq_rot_K` vs `q . fp16_K`
  - V output: `softmax(qH . deq_rot_K) . unrotate(deq_rot_V)` vs fp16
- Run this with Qwen 256d and Gemma 512d shapes.

### 3. VarN/body-store is not paper-faithful in systems shape

Code:

- `ggml/src/ggml-cuda/kvarn.cu:845-868`
  - `kvarn_sinkhorn_variance_normalize_parallel` launches row and column kernels per iteration for one tile.
- `ggml/src/ggml-cuda/kvarn.cu:988-1087`
  - `ggml_cuda_kvarn_store_kv_body_pipelined` processes one K/V tile pair.
- `ggml/src/ggml-cuda/kvarn.cu:1264-1335`
  - `ggml_cuda_kvarn_store_body_direct_records_minmax` loops over `record` then `head`, and calls the one-tile pipeline for each tile.
- `src/llama-graph.cpp:3143-3171` and `src/llama-graph.cpp:3751-3779`
  - `LLAMA_KVARN_ENABLE_DIRECT_RECORD_BATCH` batches graph nodes into 4D views, but the CUDA backend still serializes record/head tiles.

Assessment:

- This is the clearest paper-fidelity miss.
- The paper's low runtime overhead should be interpreted as "VarN is cheap when batched/fused over tiles", not "a per-tile Sinkhorn launch sequence is cheap".
- Our current direct-record batch path should stay diagnostic-only until it becomes a true records * heads batched kernel.

Patch B target:

- Replace the internal `for record { for head { ... one tile pipeline ... } }` with phase kernels that process all `n_records * n_heads` tiles in a grid.
- Allocate per-tile scratch slices.
- Launch one grid per phase, not per tile.
- Keep output bit-compatible with the existing one-tile pipeline before enabling by default.

### 4. Dequant/attention is not paper-faithful in systems shape

Code:

- Generic dequant materializes full K/V body tiles:
  - `ggml/src/ggml-cuda/kvarn.cu:1372-1427`.
- Scalar/GQA attention path can allocate and fill F32 body mirrors:
  - `ggml/src/ggml-cuda/kvarn.cu:4500-4548` region in current file (look for `scalar_body_k_f32`, `scalar_body_v_f32`, and `LLAMA_KVARN_ENABLE_F32_DEQUANT_CACHE`).
- Mixed attention scalar kernels do support direct packed loads, but occupancy and scratch behavior are still not aligned with the paper's "fused dual-scale dequant is effectively free" claim:
  - `ggml/src/ggml-cuda/kvarn.cu:4169` scalar/GQA kernel entry.

Assessment:

- Appendix I's claim depends on applying both scales inside the dequant/attention kernel, with no extra HBM pass.
- The current long decode bottleneck is consistent with too much scratch traffic and too little body-record parallelism.

Patch C target:

- Prefer direct packed K/V loads with dual-scale application inside attention.
- Avoid F32/F16 body mirrors in production long decode except as diagnostic correctness or fallback paths.
- For Qwen 256d, fix the direct packed path correctness first; do not re-enable the rejected 256d warp-QK path blindly.

### 5. Long decode occupancy is below the intended shape

Observed code:

- The generic fused/mixed paths still tend to map work around heads and query tiles, while a long `tg4096` body has many body records.
- Current long decode failure likely needs body-record parallelism: split records across CTAs per head/query and reduce partial score/value accumulations.

Patch D target:

- Add a decode kernel shape that partitions body records across CTAs per head/query.
- Reduce score max/sum/value partials across record partitions.
- Keep split/scratch reference as correctness oracle.

## Production accuracy findings from Round 26

The production accuracy gates are now failing at 4096-context bounded checks. This blocks all performance promotion.

Qwen3.6 MTP, `-ncmoe 34`, context 4096, batch 4096, `--chunks 2`, expected layers `3-39:4`:

- Artifact: `artifacts/kvarn-iters-sweep/round26-qwen36-ctx4096-chunks2-fixed`.
- `iters=1`: f16 PPL `4.5843`, KVarN PPL `6.6577`, increase `45.23%`.
- `iters=2`: increase `47.21%`.
- `iters=3`: increase `45.28%`.
- `iters=4`: increase `46.71%`.
- No candidate passed `MaxPplIncrease=5%`.

Gemma 4 12B true KVarN+ISWA, context 4096, batch 4096, `--chunks 2`, expected layers `5-47:6`:

- Artifact: `artifacts/kvarn-accuracy/round26-gemma-ctx4096-chunks2-iters4`.
- f16 PPL `418.2027`.
- KVarN PPL `137079490.9999`.
- Increase `32778141.51%`.
- This is a correctness/fidelity blocker, not a performance tuning result.

Interpretation:

- Iteration count is not the primary fix.
- The long-context production path likely has a frame/scale/window/topology correctness issue, especially for Gemma true KVarN+ISWA.
- It is unsafe to use long-context throughput numbers as production evidence until bounded f16-vs-KVarN accuracy passes.

## Revised patch sequence

1. Patch A: this audit plus handover updates.
2. Patch B: add tile-frame correctness unit tests for K/V body records and fix any Hadamard/Q/V/output-frame mismatch.
3. Patch C: implement true batched VarN/body-store over records * heads with per-tile scratch and grid-strided phase kernels.
4. Patch D: implement fused dual-scale dequant/attention without F32/F16 body mirrors as the production long-decode path.
5. Patch E: improve long decode occupancy by splitting body records across CTAs per head/query.
6. Patch F: only after production accuracy and long-path kernels are sound, revisit `--kvarn-iters` as a possible tuning knob.

## Stop conditions

- Do not lower production `--kvarn-iters` unless production-model f16-vs-KVarN PPL/KL passes.
- Do not enable `LLAMA_KVARN_ENABLE_DIRECT_RECORD_BATCH` by default until it is a true batched CUDA implementation and improves both Gemma and Qwen long prefill.
- Do not enable F32/F16 body mirror paths as production "fixes" for long decode unless measurements prove they outperform direct fused dequant and do not regress accuracy.
- Every optimized path must pass:
  - production f16-vs-KVarN accuracy gate,
  - packed-repeat,
  - packed-vs-split,
  - expected KVarN layer routing,
  - focused KVarN CTests.

## Round 27 update

The first paper-shaped implementation slice has landed as a diagnostic path, not a production default.

- `tests/test-kvarn-cuda-dequant.cpp` now has executable paper-frame checks for K and V Hadamard consumption.
- `LLAMA_KVARN_ENABLE_DIRECT_RECORD_BATCH_PHASES=1` adds a true records * heads batched direct-record body-store path for default `k4/v2`, `rtn_quantile=1.0` cases.
- The path is logits-clean on small Qwen3.5, Qwen3.6 MTP, and Gemma true KVarN+ISWA at the packed-repeat/split/scratch gates.
- It does not yet improve production Qwen3.6 pp4096; keep it opt-in and continue with fused/parallel body attention work.
- Anchored dequant mirrors now have append-aware invalidation for one-epoch append-only body stores, reducing redundant body scratch materialization in layouts that store only newly sealed records without changing the packed format or attention arithmetic.
- Qwen3.6 pp4096 tracing still shows the current direct-record prefill graph dirtying from record 0, so the append-aware cache is a correctness-safe infrastructure fix, not the long-prefill production fix by itself.
