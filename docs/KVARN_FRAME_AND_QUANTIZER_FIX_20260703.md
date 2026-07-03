# KVarN frame-contract and quantizer fidelity fix - 2026-07-03

Branch: `claude/kvarn-gemma-qwen-optimization-9d3ube` (based on `kvarn-atx-integration`)

This patch addresses the two long-standing KVarN failures with root causes that
were identified by line-by-line review of the store/frame plumbing and by
numeric analysis of the quantizer recipe:

1. The stubborn bit-width-independent quality failures (Qwen ctx4096 stuck at
   about +11-12% PPL at both k4v4 and k8v8; catastrophic Gemma paper-frame
   results) trace to a **store-side frame-contract bug**: pending-sourced body
   tiles were re-rotated by the CUDA store under `LLAMA_KVARN_ENABLE_PAPER_FRAME=1`.
   Because the Hadamard is involutive (H*H = I), double rotation silently
   stores body records in the RAW frame while attention consumes a ROTATED
   query - close enough to garbage to cost double-digit PPL, but invisible to
   every packed-vs-packed gate.
2. The **quantizer fidelity floor** was far above what the format allows:
   the reference log-std Sinkhorn clamps pinned all scales at exp(-0.3) for
   realistic small-magnitude tiles (variance normalization silently inert),
   and full-range min/max RTN at 2 bits is ~3x worse than an optimally
   clipped uniform quantizer.

## 1. Frame-contract fix (correctness, Gemma + Qwen)

### The bug chain

- The graph (`llama-graph.cpp`) decides the frame: under paper frame (default
  since v4) it rotates q, sink/tail K/V, and the pending buffer inherits the
  rotated tail through `cpy_tail_evict_pending_*`.
- The CUDA store had its own independent `kvarn_paper_frame_enabled()` with a
  DIFFERENT default (opt-in), and derived "is my input already rotated?" from
  that env instead of from the op's actual source:
  - `store_kv_body_record_from_pending` (used by every Gemma n_head_kv=1 seal)
    routed to the generic wrapper with `input_already_rotated=false`; with
    `ENABLE_PAPER_FRAME=1` set (as in every documented paper-frame validation
    run), the already-rotated pending tile was rotated AGAIN -> body records
    un-rotated vs rotated q. This is the mechanism behind the catastrophic
    Gemma paper-frame numbers (26M PPL ratio at 16k, 137e9 PPL at Round 26).
  - `ggml_cuda_kvarn_store_k/v_body_reference_minmax` (all head_dim < 256
    paths, i.e. 128-dim Qwen) had NO already-rotated awareness at all - same
    double rotation under the env. This matches the bit-width-independent
    Qwen +11-12% signature exactly: error proportional to the values, not to
    the quantization step; grows with the body fraction (ctx); invisible to
    packed-repeat/split/scratch and boundary replays (all consume the same
    mis-framed bytes).
  - The v4 commit split the defaults (graph ON / CUDA OFF), which
    accidentally made the *default* env-free configuration frame-consistent
    (memcpy of already-rotated pending) - that is why post-v4 donor5 KL
    dropped to ~0.19 - but it left `ENABLE_PAPER_FRAME=1` corrupt, direct
    stores mis-framed if ever reached, and the batched-phases store dead in
    production.

### The fix

- `kvarn_paper_frame_enabled()` in `ggml/src/ggml-cuda/kvarn.cu` now matches
  the graph-side default (ON unless `LLAMA_KVARN_DISABLE_PAPER_FRAME=1`).
  All configurations (default / ENABLE / DISABLE) are now equivalent between
  graph and CUDA.
- Body-store ops now carry the source frame explicitly:
  - `GGML_OP_KVARN_STORE_KV_BODY` reuses `src_layout` with a new value 2 =
    "pending-sourced (graph frame)"; `GGML_OP_KVARN_STORE_BODY` gained a
    trailing `src_layout` field. New setters
    `ggml_kvarn_store_kv_body_set_src_pending()` /
    `ggml_kvarn_store_body_set_src_pending()` are called by every
    `*_from_pending` builder in `src/llama-kv-cache-kvarn.cpp`.
  - The CUDA dispatcher passes `src_layout == 2` down to all store entry
    points; `input_already_rotated` is now plumbed through
    `ggml_cuda_kvarn_store_{k,v}_body_reference_minmax`, the generic wrapper,
    and both pending drivers. Pending-sourced tiles are never rotated again;
    raw tiles (direct stores, including `store_kv_body_all_heads` which
    reuses the pending-heads op) are rotated exactly once when paper frame is
    on.

Net effect: body records are in the same frame as q/sink/tail/pending in every
configuration, on every path, for every head_dim.

## 2. Quantizer fidelity fixes (quality at low bits)

### Global-RMS pre-normalization (Sinkhorn was inert)

The reference recipe clamps accumulated log scales to [-0.3, 10]. Raw K/V
tiles whose global RMS is below ~0.74 (typical: V tiles have RMS ~0.05) pin
every row/column scale at the clamp floor, so NO variance is equalized - the
paper's central mechanism ("errors driven primarily by incorrect token
scales") never engaged, and per-layer fidelity depended on each layer's raw
K/V magnitude (this is why per-layer/per-model sensitivity looked erratic).

Fix: divide each tile by its global RMS before the log-std Sinkhorn and fold
the factor back into the stored row scales after best-iteration selection.
Packed format, scale layout, and dequant are unchanged. Applied identically in
the CPU reference, the per-tile CUDA pipeline, and the batched CUDA phases.
Kill switch: `LLAMA_KVARN_DISABLE_GLOBAL_NORM=1`.

Measured (test-kvarn-kv fidelity test, small-magnitude tiles with 8x token
outliers, and the upgraded vLLM oracle):

| Path | before | after |
| --- | ---: | ---: |
| K4 tile NMSE (128d/512d) | 0.042 | 0.0069-0.0075 (5.5-6x) |
| K8 tile NMSE at raw scale 0.05 | 1.4e-4 | 2.6e-5 (5.5x) |
| Sinkhorn imbalance in oracle self-test | ~6.3 | 2.0 |

### Bit-aware RTN clipping (2-3 bit)

Full-range min/max RTN over a ~Gaussian row is ~3x worse than an optimally
clipped uniform quantizer at 2 bits. The store now clips the RTN range to
mean +/- c(bits)*std with c(2)=1.5, c(3)=2.05 (no clip at >= 4 bits, where
full range is already near-optimal and clipping can hurt heavy-tailed rows).
Same dequant, same format. Kill switch: `LLAMA_KVARN_DISABLE_RTN_CLIP=1`.

| Path | before | after |
| --- | ---: | ---: |
| V2 tile NMSE | 0.25-0.38 | 0.117-0.120 (Lloyd-Max floor) |
| K2 tile NMSE | 0.46 | 0.125 |

The oracle self-test now enforces NMSE ceilings per preset (V2 < 0.17,
V4 < 0.02, V8 < 5e-4), so PASS means fidelity, not just shape checks.

## 3. Speed changes

- `kvarn_sinkhorn_logstd_best_update_kernel` (and its batched companion) were
  single-threaded over the whole tile (64K elements, double precision, twice,
  once per iteration) - the dominant serial cost of every seal. Rewritten
  block-parallel with identical per-column/per-row double accumulation order,
  so the selected best iteration is bit-identical.
- Default `sinkhorn_iters` 16 -> 8 (the reference config treats 8 log-std
  iterations as lossless-grade; with global-norm the first iteration does the
  bulk of the equalization and best-iteration selection keeps the rest safe).
- The batched direct-record store phases now run in production (they were
  gated on the CUDA-side paper flag that used to be off by default) and cover
  k2/k4/k8 x v2/v4/v8.

## 4. New capabilities

- `kvarn_k2v2_g128` is now a first-class preset (parser already accepted it;
  llama-bench/oracle/tests now cover it; batched store packs k2 and v8).
  With clipping, K2 per-channel NMSE is ~0.125 - usable for ablations; the
  high-GQA K8 promotion still applies by default on GQA >= 6 routes.

## 5. Validation done here (CPU-only container, no GPU)

- `test-kvarn-kv` (includes the new `test_reference_quantizer_fidelity`
  regression: global-norm >= 1.33x K improvement, clip >= 1.67x V2
  improvement, absolute NMSE ceilings, K2 roundtrip): PASS
- `test-batch-split`: PASS
- `test-arg-parser` kvarn assertions incl. k2v2 (the download-URL test fails
  in this sandbox for network policy reasons, unrelated): PASS
- `python scripts/kvarn/kvarn_vllm_oracle.py --self-test --head-dims
  128,256,512 --presets k2v2,k4v2,k8v2,k8v4,k8v8 --iters 8`: PASS with the
  new NMSE gates
- `python scripts/kvarn/kv_memory_estimate.py --self-test`: PASS
- CUDA code could not be compiled here (no nvcc); the CUDA changes mirror the
  CPU reference math line-for-line and the existing CUDA tests
  (`test-kvarn-cuda-dequant`, whose embedded CPU mirror was updated
  identically) are the first thing to run on the GPU rig.

## 6. GPU runbook (in order; stop at the first failure)

```powershell
# 1. Unit and parity tests
ctest --test-dir build-kvarn-cuda-static-vs -C Release `
  -R "test-batch-split|test-kvarn-kv|test-kvarn-cuda|test-arg-parser" --output-on-failure

# 2. Short-context logits parity (both models)
scripts/kvarn/compare_cuda_logits_ref.ps1 -Model <qwen> -Context 512 -Batch 512 `
  -ExpectedKvarnLayers "3-39:4" -ExtraArgs "-ncmoe","34" -CheckPackedRepeat -CheckPackedSplit
scripts/kvarn/compare_cuda_logits_ref.ps1 -Model <gemma> -Context 512 -Batch 512 `
  -ExpectedKvarnLayers "5-47:6" -CheckPackedRepeat -CheckPackedSplit

# 3. The decisive quality gates. No KVarN env vars needed anymore:
#    paper frame is the default and now frame-consistent end to end.
#    Expectation: k8v8 should now be close to lossless (it was the frame bug,
#    not bit width); k8v2 and k4v2 benefit further from clip + global-norm.
scripts/kvarn/run_accuracy_gate.ps1 -Model <qwen> ... -ContextSize 4096 -BatchSize 4096 `
  -Chunks 2 -KvarnPreset kvarn_k8v8_g128 -KvarnIters 8 -ExpectedKvarnLayers "3-39:4"
scripts/kvarn/run_accuracy_gate.ps1 ... -KvarnPreset kvarn_k8v2_g128 ...
scripts/kvarn/run_accuracy_gate.ps1 ... -KvarnPreset kvarn_k2v2_g128 ...
# Gemma 16k KL gate (same fixture/baseline as strict-gemma16k-* runs)
$env:LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA = '1'
# k8v8 first, then k8v2, then k2v2

# 4. A/B the two quantizer changes if any gate regresses:
#    LLAMA_KVARN_DISABLE_GLOBAL_NORM=1 / LLAMA_KVARN_DISABLE_RTN_CLIP=1

# 5. Speed only after 3 passes:
scripts/kvarn/run_bench_matrix.ps1 -Model <gemma> -CaseList "pp512:512:0,tg64:0:64,pp4096:4096:0,tg4096:0:4096"
scripts/kvarn/run_bench_matrix.ps1 -Model <qwen>  -CaseList "pp512:512:0,tg64:0:64,pp4096:4096:0,tg4096:0:4096"
```

## 7. If Qwen ctx4096 still fails after this patch

The frame fix predicts a large drop in the +11-12% plateau. If a residual gap
remains at k8v8, it is NOT quantization; run the missing control that isolates
the KVarN ubatch splitting + hybrid wrapper from compression:

```powershell
$env:LLAMA_KVARN_LAYER_FILTER = '99'   # no layer matches: all layers normal KV,
                                       # but KVarN ubatch limits still apply
```

If that control shows a PPL gap vs mainline, the residual is in the hybrid
memory/ubatch path, not in the KVarN math.

## 8. Files touched

- `ggml/src/ggml-cuda/kvarn.cu` / `kvarn.cuh` - frame plumbing, global-norm,
  clip, parallel best-update, k2/v8 batched packing
- `ggml/src/ggml-cuda/ggml-cuda.cu` - dispatcher passes src frame
- `ggml/src/ggml.c` / `ggml/include/ggml.h` - src-pending markers
- `src/llama-kv-cache-kvarn.cpp` - CPU reference (global-norm, clip), pending
  markers, scratch sizing
- `src/llama.cpp` - sinkhorn_iters default 16 -> 8
- `scripts/kvarn/kvarn_vllm_oracle.py` - models production math, NMSE gates,
  k2v2 preset
- `tests/test-kvarn-kv.cpp` - fidelity regression test, scratch formula
- `tests/test-kvarn-cuda-dequant.cpp` - CPU mirror updated identically,
  signature updates, scratch sizes
- `tests/test-arg-parser.cpp` - k2v2 preset assertion
