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

---

# Round 2 - runtime state-machine audit and topology test harness (same day)

A second audit pass targeted the paths real server usage exercises but
perplexity/bench never do. Three defects found and fixed, plus a topology
simulation test that pins the whole streaming contract.

## Bugs found and fixed

1. **Partial `seq_rm` accepted removals the layout cannot honor** (rollback
   corruption). The tail is a position-derived ring (slot(p) == slot(p+tail)),
   and the mask labels slots by position math, not cell metadata. After a
   partial rollback (llama-server prompt-reuse trimming, speculative rejects),
   the mask would attribute ring slots to positions whose f16 rows had already
   been overwritten by the removed newer tokens - attention silently reads
   future-token values until the ring refills. `seq_rm` now refuses (returns
   false, so callers reprocess) everything except: full-range removal, no-op
   ranges, and suffix rollbacks taken before any ring slot was ever reused
   (those are exact). The ring-overwrite bookkeeping inside `apply_ubatch`
   uses a private `seq_rm_cells()` that bypasses the guard.
2. **`get_can_shift()` claimed shift support with no shift graph.** K is
   stored post-RoPE (rotated f16 sink/tail + quantized records) and
   `init_update()` fails the decode when cells carry a pending shift, so a
   server context-shift would brick the session (or corrupt, had init_update
   not guarded). Now returns false; ISWA/hybrid composition propagates it.
3. **Cache-side input builders ignored the small-kv_size tail clamp.** The
   graph-side window/seal math clamps `tail_tokens` when
   `kv_size < sink+tail`; the eviction plans, pending offsets, and sink/tail
   slot indices built in `llama-kv-cache-kvarn.cpp` used the raw params. The
   disagreement is unreachable today only because positions are bounded by
   kv_size; all five input builders now share `kvarn_effective_params()`
   (and `set_input_kq_mask` reuses it instead of a local copy). The
   `llama_kvarn_device_supports_ops` probe scratch formula was also one float
   behind `kvarn_store_scratch_floats_one`.

## New tests (tests/test-kvarn-kv.cpp)

- `test_runtime_stream_consistency`: drives full contiguous streams through
  the cache's real input-building machinery (sink/tail slot indices, tail
  eviction plans, pending offsets) while tracking which logical position each
  physical slot holds. After every ubatch it recomputes the active window and
  causal mask (deliberate mirrors of `kvarn_graph_active_window` /
  `kvarn_graph_seal_records`) and verifies the mixed-attention loader order
  (sink | body records | pending | tail ring at tail_start) enumerates exactly
  positions 0..last, and that the mask exposes exactly the causal prefix.
  Runs four geometries: ring wrapped 10x with odd ubatch cycles, a
  non-power-of-two tail ring (tail=5), a minimal decode-like shape, and the
  clamped `kv_size < sink+tail` shape. This transitively pins the ring modulo
  math, eviction slot reads, seal timing, tail_start, per-record pending
  slice ordering, and mask labeling against each other.
- `test_runtime_state_safety`: get_can_shift refusal; post-wrap suffix seq_rm
  refused with state unchanged; mid-range refused; full-range and pre-wrap
  suffix accepted.
- `test_reference_store_scale_invariance`: store->dequant NMSE must be
  invariant to raw tile magnitude (0.02x / 1x / 30x within 30% of each other) -
  the direct regression for the Sinkhorn clamp-inertness fix.

Note: while building the simulation, the per-record pending slice ordering in
the graph (seal consumes its slice before the next record's evictions
overwrite the one-group pending ring) was verified to be handled correctly by
the existing set-rows dependency chain - initially it looked like an
intra-ubatch overwrite bug, but the graph's slice-by-slice copies are sound;
the simulation now models that ordering exactly.

## Explicitly audited and found sound in this pass

- Hybrid (Qwen MTP) wrapper: recurrent child already refuses partial seq_rm
  ahead of the attention children; ubatch splitting always uses
  `split_equal(n, true)` under recurrence, consistent with the limiter.
- ISWA wrapper composition of seq_* / can_shift after the base-cache fixes.
- Ring-overwrite metadata cascade in `apply_ubatch` (now guard-exempt by
  design).
- `kvarn_tail_slot` for `sink >= kv_size` degenerate geometry (unreachable
  tail branch; window collapses to sink-only).
- Batched hadamard butterfly kernels (read/sync/write pattern), the
  fold/rms kernel scratch offsets, and the effective-params degenerate
  tail=0 case.

## Known remaining limitations (documented, not silent)

- `state_write/state_read` (session save/restore) throw
  "not implemented yet"; llama-server slot save/restore and
  `llama_state_*` APIs are unavailable with `--kv-cache-quant kvarn`.
- `kvarn_tail_safe_ubatch_limit` has no direct unit test (constructing a
  `llama_batch_allocr` in tests is heavy); its downstream contract is pinned
  by the stream simulation.
- The mixed-attention CUDA kernels remain uncompilable in this container;
  the GPU runbook in section 6 is unchanged and should be run before any
  quality/speed claims.
