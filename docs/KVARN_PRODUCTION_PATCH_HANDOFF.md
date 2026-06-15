# KVarN production patch handoff for coding agent

## Latest handoff for external review - 2026-06-12 Round 12

Repo: `JakeATX/llama.cpp`
Branch: `kvarn-atx-integration`
Current branch target: <https://github.com/JakeATX/llama.cpp/tree/kvarn-atx-integration>
Latest pushed baseline before this round: [`e5596d6ea`](https://github.com/JakeATX/llama.cpp/commit/e5596d6ea) - `kvarn: add round11 diagnostics and fallback parity`

This is the current entry point for 5.5 Pro / external architecture review. Older sections below are retained for history. The current production target is still mainline parity against upstream `llama.cpp`, not fork normal-KV parity.

### Round 12 implementation status

Accepted for push:

- `tests/test-kvarn-kv.cpp`
  - Added `LLAMA_KVARN_TEST_PHASE_TRACE=1` phase tracing and exception capture so future Windows fast-fail reports can be mapped to the exact test phase.
  - Fixed the local failure: `test_runtime_metadata()` used default `sink_tokens=128` with a 16-token test KV cache. The fixture now sets `sink_tokens=8` and `tail_tokens=8`.
  - Result: the prior `0xc0000409` report is no longer treated as an unexplained Windows policy issue in this checkout.
- `scripts/kvarn/run_production_gate.ps1`
  - Added `-MainlineBuildDir`.
  - Enforced `Tier1MinRatio >= 0.90`; low-threshold diagnostics must call the lower-level matrix scripts directly.
  - Qwen Tier 1 now uses `run_mainline_parity_matrix.ps1`, not same-build `run_bench_matrix.ps1`.
  - Qwen Tier 1 now enforces `-ncmoe 34` and expected KVarN layers `3-39:4`.
  - Gemma Tier 1 now measures the production normal-ISWA fallback against upstream/mainline using `-AllowKvarnFallback`.
  - `-RunGemmaExperimental` is a true KVarN+ISWA diagnostic and scopes `LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA=1` only to that subprocess.
- `tests/test-kvarn-cuda-dequant.cpp`
  - Added `LLAMA_KVARN_TEST_256D_WARPQK_REPRO=1` so the synthetic Qwen3.6-shaped 256d warpqk test runs first in a fresh process before cached env decisions can hide the diagnostic path.
  - Added worst-difference reporting for the Qwen3.6-shaped split/fused comparison.
- `ggml/src/ggml-cuda/kvarn.cu`
  - Added `LLAMA_KVARN_ATTN_TRACE=1` with `LLAMA_KVARN_ATTN_TRACE_LIMIT` for bounded host-side dispatch traces.
  - Trace records selected mixed-attention mode, `head_dim`, query/head counts, sink/body/tail topology, q-tiling, launch dimensions, scratch size, body mirror usage, and mask strides.
- `scripts/kvarn/README.md`
  - Updated the CUDA production gate text to describe true KVarN/mainline parity, Qwen `-ncmoe 34`, Gemma production fallback, and Gemma experimental diagnostic behavior.

Round 12 validation:

| Gate | Result | Notes |
|---|---:|---|
| Build `llama-bench`, `llama-cli`, `llama-results`, `test-kvarn-kv`, `test-kvarn-cuda-scratch-ref`, `test-kvarn-cuda-mixed-tail`, `test-kvarn-server-load-failure` | PASS | CUDA warnings only |
| `ctest -R "test-kvarn-kv|test-kvarn-cuda|test-batch-split|test-arg-parser|test-kvarn-server-load-failure"` | PASS | 7/7 |
| `python scripts/kvarn/kv_memory_estimate.py --self-test` | PASS | |
| Qwen2.5 CUDA smoke `256 512` | PASS | expected layers `0-27` |
| Qwen2.5 logits repeat/split/scratch | PASS | NMSE `0` |
| Qwen3.6 MTP default logits | PASS | packed repeat NMSE `0`, packed-vs-split NMSE `0`, expected layers `3-39:4`, `-ncmoe 34` |
| Qwen3.6 MTP 256d diagnostic full model | FAIL as expected | `LLAMA_KVARN_ATTN_ENABLE_256D_WARPQK=1`, `QT=1`, no 256d body mirror, packed-vs-split NMSE `6.196e-03` |
| Synthetic 256d warpqk repro QT=1/4/8 | PASS | does not reproduce the full Qwen3.6 MTP logits failure |

Round 12 short mainline parity, no trace, warmup enabled, `r=3`:

| Model/mode | Case | Mainline t/s | KVarN t/s | Ratio | Artifact |
|---|---:|---:|---:|---:|---|
| Qwen3.6 MTP true KVarN, `-ncmoe 34` | `pp512` | 292.33 | 232.42 | 79.5% | `artifacts/kvarn-mainline-parity/round12-qwen-short-r3` |
| Qwen3.6 MTP true KVarN, `-ncmoe 34` | `tg64` | 36.24 | 29.58 | 81.6% | `artifacts/kvarn-mainline-parity/round12-qwen-short-r3` |
| Gemma 4 production normal-ISWA fallback | `pp512` | 2485.14 | 2614.54 | 105.2% | `artifacts/kvarn-mainline-parity/round12-gemma-fallback-r3` |
| Gemma 4 production normal-ISWA fallback | `tg64` | 60.11 | 61.42 | 102.2% | `artifacts/kvarn-mainline-parity/round12-gemma-fallback-r3` |

Interpretation:

- Gemma production fallback is currently over the short production gate. This is fallback parity, not true KVarN+ISWA parity.
- Qwen true KVarN remains below the short production gate on both prefill and decode. The next production blocker is Qwen, not Gemma fallback.
- The synthetic 256d low-level test passing while full Qwen3.6 MTP fails means the existing synthetic shape is still missing at least one real-model factor: graph topology, mask layout/content, token position ordering, recurrent/MTP interaction, or exact packed-vs-split boundary state.
- Do not enable 256d warpqk by default. Keep the 256d switches diagnostic-only until full Qwen3.6 MTP packed-vs-split NMSE is `0`.

Recent upstream risk review:

- `gh` CLI was not available in this environment, so the review used GitHub web pages and `git ls-remote`.
- Latest upstream release observed on 2026-06-12: [`b9608`](https://github.com/ggml-org/llama.cpp/releases/tag/b9608), vendor cpp-httplib update.
- Relevant recent upstream release: [`b9606`](https://github.com/ggml-org/llama.cpp/releases/tag/b9606), EAGLE3 speculative decoding support with Gemma-related changes.
- Relevant recent upstream release: [`b9605`](https://github.com/ggml-org/llama.cpp/releases/tag/b9605), CUDA scalar concat support.
- High-risk PR for this branch: [`ggml-org/llama.cpp#24086`](https://github.com/ggml-org/llama.cpp/pull/24086), "Remove padding and multiple D2D copies for MTP", merged 2026-06-10. This touches MTP/recurrent behavior adjacent to the Qwen3.6 MTP failure surface and should be reviewed before any upstream rebase.

Immediate next coding target:

1. Capture full-model Qwen3.6 packed-run attention boundary evidence before split output is discarded by `compare_cuda_logits_ref.ps1`.
2. Add an opt-in dump for one failing Qwen layer/head/query that records score/prob/output boundaries for split vs 256d warpqk.
3. Extend the synthetic 256d unit to replay that real boundary dump, then fix mask/order/scaling/value accumulation until synthetic and full-model packed-vs-split NMSE are both `0`.
4. Re-run Qwen short mainline parity only after correctness is restored.
5. Keep Gemma fallback default. Run true Gemma KVarN+ISWA only as an explicit diagnostic via `LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA=1` or `-RunGemmaExperimental`.

## Latest handoff for external review - 2026-06-11

Repo: `JakeATX/llama.cpp`
Branch: `kvarn-atx-integration`
Current branch target: <https://github.com/JakeATX/llama.cpp/tree/kvarn-atx-integration>
Latest pushed baseline before Round 11: [`fa6361d78`](https://github.com/JakeATX/llama.cpp/commit/fa6361d78) - `docs: update KVarN Round 10 handoff`
Prior pushed baseline for this round: [`2c61f9c67`](https://github.com/JakeATX/llama.cpp/commit/2c61f9c67) - `kvarn: add mainline parity harness and per-body dequant epochs`

This is the current entry point for a 5.5 Pro / external architecture review. Older sections below are retained for history, but the current production target is mainline parity against upstream `llama.cpp`, not fork normal-KV parity.

### Round 11 implementation status

Accepted for push:

- `scripts/kvarn/run_mainline_parity_matrix.ps1`
  - Added `-AllowKvarnFallback` so Gemma production fallback can be measured without requiring `llama_kv_cache_kvarn:` logs.
  - Added `-RunOrder mainline-first|kvarn-first` for same-session order-effect checks.
  - Default behavior remains strict: missing KVarN cache logs still fail unless `-AllowKvarnFallback` is explicit.
- `ggml/src/ggml-cuda/kvarn.cu`
  - Added diagnostic-only 256d mixed-attention switches:
    - `LLAMA_KVARN_ATTN_ENABLE_256D_WARPQK=1`
    - `LLAMA_KVARN_ATTN_ENABLE_256D_BODY_MIRROR=1`
    - `LLAMA_KVARN_ATTN_WARPQK_FORCE_QT=1|4|8`
  - Default behavior remains unchanged. 512d warpqk/f16 mirror stays active; 256d warpqk and 256d f16 body mirror remain disabled unless explicitly requested.
- `ggml/src/ggml-cuda/ggml-cuda.cu`
  - Trace mode classification now reports the experimental 256d warpqk path correctly when the diagnostic env flag is set.

Round 11 validation:

| Gate | Result | Notes |
|---|---:|---|
| Build `llama-bench`, `llama-results`, `test-kvarn-cuda-scratch-ref` | PASS | CUDA warnings only |
| `python scripts/kvarn/kv_memory_estimate.py --self-test` | PASS | |
| Qwen2.5 CUDA smoke `256 512` | PASS | local model `Qwen2.5-1.5B-Instruct-GGUF`, expected layers `0-27` |
| Qwen3.6 MTP default logits | PASS | packed repeat NMSE `0`, packed-vs-split NMSE `0`, expected layers `3-39:4` |
| Gemma fallback harness | PASS | `-AllowKvarnFallback`, artifact `artifacts/kvarn-mainline-parity/20260611-223614` |
| 256d diagnostic QT=1 without f16 mirror | FAIL as expected | Qwen3.6 MTP packed-vs-split NMSE `6.196e-03` |
| Focused CTest regex | FAIL | only `test-kvarn-kv` fails, exit `0xc0000409` after CUDA device discovery |

Round 11 Qwen no-trace order check, `r=5`, warmup enabled, `-ncmoe 34`:

| Run order | Case | Mainline t/s | KVarN t/s | Ratio | Artifact |
|---|---:|---:|---:|---:|---|
| mainline-first | `pp512` | 443.65 | 291.28 | 65.7% | `artifacts/kvarn-mainline-parity/round11-qwen-r5-mainline-first` |
| mainline-first | `tg64` | 44.27 | 38.21 | 86.3% | `artifacts/kvarn-mainline-parity/round11-qwen-r5-mainline-first` |
| kvarn-first | `pp512` | 423.22 | 290.43 | 68.6% | `artifacts/kvarn-mainline-parity/round11-qwen-r5-kvarn-first` |
| kvarn-first | `tg64` | 44.05 | 37.87 | 86.0% | `artifacts/kvarn-mainline-parity/round11-qwen-r5-kvarn-first` |

Interpretation:

- The earlier Qwen `pp512 44.4%` number was too pessimistic for the current clean run, but Qwen still fails the short production gate.
- Run order does not explain the remaining short-gate gap in this session.
- The 256d isolation result is decisive: `QT=1` with no f16 mirror still fails full Qwen3.6 MTP packed-vs-split logits, so the divergence is in the base 256d warpqk math/mask/order path, not q-tiling or the f16 body mirror.
- Do not enable 256d warpqk by default, and do not benchmark it for production until Qwen3.6 MTP packed-vs-split NMSE is `0`.
- Do not start 4k production diagnostics as the next gate; first fix Qwen short `pp512` and `tg64`, then run long-context diagnostics.

Immediate next coding target:

1. Add a low-level forced 256d warpqk-vs-split unit case that mirrors the full Qwen3.6 MTP failure shape closely enough to debug without loading the full model.
2. Compare split and 256d warpqk at the score/prob/output boundaries to isolate mask/order/scaling vs value accumulation.
3. Keep Gemma fallback default unless `LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA=1`.
4. Re-run `test-kvarn-kv` in a clean path or separate machine; do not classify `0xc0000409` as the old Windows Application Control process-start block unless stderr proves it.

### Round 10 implementation status

Accepted and pushed:

Commit [`8e35bce0c`](https://github.com/JakeATX/llama.cpp/commit/8e35bce0c):

- Restored production-safe Gemma 4 behavior:
  - default Gemma 4 + `--kv-cache-quant kvarn` + ISWA uses normal ISWA fallback.
  - `LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA=1` opts into true KVarN+ISWA for benchmarking/development.
  - `LLAMA_KVARN_FORCE_NORMAL_ISWA_FALLBACK=1` remains a hard safe-path override.
- Added a scratch-capacity guard around the existing 512d f16 body mirror path so low-level callers that pass only score-buffer scratch cannot overrun into undeclared mirror storage.

Rejected after test:

- `0002-round10-qwen256-warpqk-dequant-cache-prototype.patch` was applied manually, tested, and backed out.
- Initial form failed `test-kvarn-cuda-scratch-ref`:
  - `FAIL: CUDA Qwen3.6-shaped 49-query forced fused mixed attention matches split output`
- After adding a scratch-capacity guard, the low-level CUDA unit passed, but real Qwen3.6 MTP logits failed:
  - packed repeat: PASS, NMSE `0`
  - packed-vs-split: FAIL, NMSE `6.196e-03`
- Narrowing the prototype to 256d warpqk without the f16 mirror still failed the same Qwen3.6 MTP packed-vs-split logits check.
- The 256d dispatch was backed out. Qwen3.6 MTP logits then passed again:
  - packed repeat: PASS, NMSE `0`
  - packed-vs-split: PASS, NMSE `0`

Current architect fix target:

1. Do not re-submit the simple `head_dim >= 256` warpqk threshold patch as-is; it is not logits-equivalent to split for Qwen3.6 MTP.
2. Debug why the 256d warpqk path diverges from split before any performance benchmarking:
   - first reproduce with the low-level Qwen3.6-shaped 49-query test;
   - then reproduce with full Qwen3.6 MTP logits at `Context=512`, `Batch=512`, `-ncmoe 34`;
   - require packed-vs-split NMSE `0` before measuring throughput.
3. Keep Gemma fallback default in place until true Gemma KVarN+ISWA short parity passes.
4. Keep 4k/long-context diagnostics blocked until both short production cells pass.

Round 10 validation notes:

| Gate | Result | Notes |
|---|---:|---|
| Build of CUDA libs, `llama-bench`, `llama-results`, KVarN CUDA tests | PASS | Aggregate `llama-server` target hit unrelated UI asset generation failure |
| `python scripts/kvarn/kv_memory_estimate.py --self-test` | PASS | |
| Qwen2.5 CUDA smoke `256 512` | PASS | expected layers `0-27` |
| Qwen2.5 logits repeat/split/scratch | PASS | NMSE `0` |
| Qwen3.6 MTP logits after backing out 256d dispatch | PASS | repeat/split NMSE `0` |
| `test-kvarn-cuda-scratch-ref` after guard/backout | PASS | |
| `test-kvarn-kv` | LOCAL BLOCKER | exits `0xc0000409` in this environment |

Latest short parity status:

| Model | Case | Mainline t/s | KVarN/fallback t/s | Ratio | Gate |
|---|---:|---:|---:|---:|---:|
| Qwen3.6 MTP, `-ncmoe 34`, `r=5` | `pp512` | 284.47 | 126.23 | 44.4% | FAIL |
| Qwen3.6 MTP, `-ncmoe 34`, `r=5` | `tg64` | 40.37 | 34.06 | 84.4% | FAIL |
| Gemma 4 fallback, `r=5` | `pp512` | 2284.38 | 2410.95 | 105.5% | PASS |

Qwen `r=5` artifact:

- `artifacts/kvarn-mainline-parity/20260611-201128`

Gemma fallback note:

- The current parity harness expects KVarN cache logs and aborts on Gemma fallback because fallback intentionally emits no KVarN cache initialization lines.
- For Gemma fallback parity, either run manual `llama-bench` pairs or update the harness with an explicit fallback-allowed mode.

### Current production gate

Required models:

| Model | Path | KVarN layers | Extra args |
|---|---|---:|---|
| Gemma 4 12B | `C:\Users\sjake\OneDrive\Documents\New project\models\gemma-4-12b-it-GGUF\gemma-4-12b-it-UD-Q4_K_XL.gguf` | `5-47:6` | none |
| Qwen3.6 35B A3B MTP | `C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.6-35B-A3B-MTP-GGUF\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | `3-39:4` | `-ncmoe 34` |

First production threshold: KVarN throughput must be at least 90% of upstream/mainline for both:

- `pp512:512:0`
- `tg64:0:64`

Long-context cases remain diagnostic until short parity is stable.

Default KVarN settings for parity:

```powershell
--kv-cache-quant kvarn --kvarn-preset kvarn_k4v2_g128 --kvarn-iters 4 --kvarn-rtn-quantile 1.0 -fa off
```

### What just landed

Commit [`2c61f9c67`](https://github.com/JakeATX/llama.cpp/commit/2c61f9c67):

- Repaired `scripts/kvarn/run_mainline_parity_matrix.ps1` so Markdown `llama-bench` throughput parses correctly.
- Added summary artifacts: `summary.csv`, `summary.md`, command files, git SHAs, model/build paths, CUDA device info, and pass/fail per case.
- Added KVarN evidence checks for expected layer logs, expected layer IDs, and optional body-record minimum.
- Replaced global KVarN body-store dequant epoch invalidation with per-body/per-layer epochs.
- Hardened unsupported-mode smoke coverage.

Commit [`d8787b7a9`](https://github.com/JakeATX/llama.cpp/commit/d8787b7a919d5352d579cdcfd788f0efae6e3c7b):

- Added bounded trace support to `run_mainline_parity_matrix.ps1`:
  - `-TraceAttn`
  - `-TraceStore`
  - `-TraceDequantCache`
  - trace limit controls
- Summary CSV now includes mixed-attn modes/shapes, store trace kinds/shapes, dequant-cache hit/miss/partial counts, and max body records.
- Broadened no-body decode sink/tail CUDA fast path to 128/256d decode when `n_queries == 1 && n_records == 0 && n_pending == 0`.
- Added `LLAMA_KVARN_DEQUANT_CACHE_TRACE_LIMIT` validation.
- Added effective sink/tail policy clamping so oversized requested sink+tail cannot trip `sink_tail_k->ne[2] >= n_sink + n_tail` graph assertions.

### Validation from 2026-06-11

Build:

```powershell
cmake --build build-kvarn-cuda-static-vs --config Release --target llama-bench llama-cli llama-results llama-server test-kvarn-kv test-kvarn-cuda-scratch-ref test-kvarn-cuda-mixed-tail test-kvarn-server-load-failure -j 8
```

Result: PASS. Existing MSVC/CUDA warnings only.

Runnable correctness gates:

| Gate | Result | Notes |
|---|---:|---|
| `python scripts/kvarn/kv_memory_estimate.py --self-test` | PASS |  |
| Qwen2.5 CUDA smoke, contexts `256 512`, expected layers `0-27` | PASS |  |
| Qwen2.5 logits repeat/split/scratch | PASS | NMSE `0` |
| Gemma logits repeat/split/scratch | PASS | Script reports `NaN` NMSE for existing zero-reference comparisons |
| Qwen3.6 MTP logits repeat/split | PASS | NMSE `0`, body-record and layer checks pass |
| Unsupported smoke | PASS | Includes invalid dequant trace-limit env |

Blocked local test:

- `ctest` could not start `test-kvarn-kv.exe`.
- Direct execution and a copied temp executable both failed with: `An Application Control policy has blocked this file`.
- This was an OS policy block before process start, not a KVarN assertion or test failure.

### Mainline parity results

No-trace production measurements:

| Model | Case | Mainline t/s | KVarN t/s | Ratio | Gate |
|---|---:|---:|---:|---:|---:|
| Gemma 4 12B | `pp512` | 2105.17 | 1824.41 | 86.7% | FAIL |
| Gemma 4 12B | `tg64` | 61.36 | 61.09 | 99.6% | PASS |
| Qwen3.6 MTP | `pp512` | 235.80 | 203.14 | 86.1% | FAIL |
| Qwen3.6 MTP | `tg64` | 29.72 | 29.45 | 99.1% | PASS |

Traced diagnostic measurements:

| Model | Case | Ratio | Trace evidence |
|---|---:|---:|---|
| Gemma 4 12B | `pp512` | 87.3% | `warpqk-f16-dequant=48`, store `kv=48`, dequant `miss=24; partial=24` |
| Gemma 4 12B | `tg64` | 99.6% | `sinktail-decode=16` |
| Qwen3.6 MTP | `pp512` | 81.4% | `fused-batch=60`, store `kv=64` |
| Qwen3.6 MTP | `tg64` | 85.4% under trace, 99.1% no-trace | `sinktail-decode=64`; trace overhead/noise is material |

Artifacts:

| Run | Path |
|---|---|
| Gemma traced short | `artifacts/kvarn-mainline-parity/20260611-143201` |
| Qwen traced short | `artifacts/kvarn-mainline-parity/20260611-143248` |
| Qwen no-trace `tg64` | `artifacts/kvarn-mainline-parity/20260611-143405` |
| Qwen no-trace `pp512` | `artifacts/kvarn-mainline-parity/20260611-143442` |
| Gemma no-trace `pp512` | `artifacts/kvarn-mainline-parity/20260611-143523` |

### Review questions for 5.5 Pro

1. Prefill is now the blocker. Decode short parity is effectively passing without trace. Review should focus on `pp512` body-store, body attention, and dequant-cache behavior.
2. Gemma `pp512` trace shows dequant-cache `miss=24; partial=24` and no full hit reuse. Determine whether graph/scratch identity or active-record keying prevents reuse across repetitions.
3. Qwen `pp512` uses 256d `fused-batch` body attention with no dequant-cache trace summary. A simple 256d warpqk/f16-mirror threshold patch was tested and rejected because it failed Qwen3.6 MTP packed-vs-split logits with NMSE `6.196e-03`; fix correctness before measuring performance.
4. Store/seal work is still visible in both Gemma and Qwen prefill traces. Review `GGML_OP_KVARN_STORE_KV_BODY`, Sinkhorn iterations, seal batching, and redundant K/V staging.
5. The 128/256d no-body decode sink/tail specialization is correct under logits and passes no-trace Qwen `tg64`, but traced throughput is noisy. Keep it unless a cleaner microbenchmark proves a regression.
6. Sink/tail policy now clamps tail when requested sink+tail exceeds `kv_size`. Review whether this should be a warning-only clamp, a CLI validation error for explicit user policy, or both.

### Recommended next patch sequence

1. Add lower-noise microbench or trace aggregation for prefill body-store vs mixed-attn time without printing inside the hot path.
2. Fix Gemma dequant-cache full-hit reuse if the trace confirms cache identity/key churn.
3. Debug and fix Qwen 256d warpqk/mirror correctness for `pp512`; require low-level CUDA split equivalence and full Qwen3.6 MTP packed-vs-split NMSE `0` before benchmarking.
4. Optimize store/seal cost after separating Sinkhorn cost from K/V staging and metadata sealing.
5. Only after short `pp512` passes for both models, rerun diagnostics:
   - Gemma `pp4096,tg4096`, `-r 1`
   - Qwen `pp2048,tg2048`, `-r 1`, `-ncmoe 34`

---

Repo: `JakeATX/llama.cpp`
Branch: `kvarn-atx-integration`
Code anchor reviewed: `f5bdd5b6c` (`cuda: Gemma 512d sinktail, pipelined body-store, and batch seal path`)  
**Implemented in:** `95390d5b1` — see [`docs/AGENT_CODE_REVIEW_HANDOVER.md`](AGENT_CODE_REVIEW_HANDOFF.md) for next-agent tasking.

This handoff is for implementation by the coding agent. I reviewed the uploaded next-thread handover, `docs/KVARN_CUDA_HANDOVER.md`, `docs/GEMMA_KVARN_FAILURE_DIAGNOSTIC.md`, and the relevant CUDA/graph/model code via the GitHub connector. I did not build or benchmark in this sandbox.

## Current conclusion

The requested `head_dim >= 512 && n_records == 0 && n_pending == 0` sink/tail fast path is already present. Do not re-implement that first; validate it only as a regression check.

The next two changes should be treated as P0:

1. Fix the pre-dequantized K layout in the 512d warpqk body-active path.
2. Fix event ordering in the dual-stream 512d K/V body-store path and remove the host-side `cudaStreamSynchronize(aux_stream)`.

Then rerun Tier 2 logits and Gemma true KVarN pp512/tg64 before any larger body-store batching work.

---

## Patch 1 — Fix pre-dequantized K layout in `kvarn_attn_mixed_f16_fused_batch_warpqk_kernel`

File: `ggml/src/ggml-cuda/kvarn.cu`

Problem:

`ggml_cuda_kvarn_dequant_body_n()` writes K records in record-major, dimension-major layout:

```cpp
// per record
k_out[d * group_size + g]
```

but the 512d warpqk path currently reads the optional `body_k_f32` scratch as if K were token-major:

```cpp
k_body_f32_head[(t - n_sink) * head_dim + d]
```

That is the V layout, not the K layout. This is a correctness risk and likely invalidates the intended 512d body-active speed path.

Minimal patch:

```diff
diff --git a/ggml/src/ggml-cuda/kvarn.cu b/ggml/src/ggml-cuda/kvarn.cu
@@
-            if (k_body_f32_head != nullptr && t >= n_sink && t < n_sink + n_body_tokens) {
-                k = k_body_f32_head[size_t(t - n_sink)*head_dim + d];
+            if (k_body_f32_head != nullptr && t >= n_sink && t < n_sink + n_body_tokens) {
+                const uint32_t body_t = t - n_sink;
+                const uint32_t r = body_t / group_size;
+                const uint32_t g = body_t - r*group_size;
+                k = k_body_f32_head[size_t(r)*size_t(head_dim)*group_size + size_t(d)*group_size + g];
             } else {
                 k = kvarn_mixed_f16_load_k(
@@
-            if (v_body_f32_head != nullptr && t >= n_sink && t < n_sink + n_body_tokens) {
-                v = v_body_f32_head[size_t(t - n_sink)*head_dim + d];
+            if (v_body_f32_head != nullptr && t >= n_sink && t < n_sink + n_body_tokens) {
+                const uint32_t body_t = t - n_sink;
+                const uint32_t r = body_t / group_size;
+                const uint32_t g = body_t - r*group_size;
+                v = v_body_f32_head[size_t(r)*size_t(group_size)*head_dim + size_t(g)*head_dim + d];
             } else {
                 v = kvarn_mixed_f16_load_v(
```

The V change is logically equivalent to the current contiguous token-major expression, but makes the K/V contrast explicit and prevents future layout regressions.

Required tests after patch:

```powershell
ctest --test-dir build-kvarn-cuda-static-vs -C Release -R "test-kvarn-kv|test-kvarn-cuda|test-batch-split" --output-on-failure
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\kvarn\compare_cuda_logits_ref.ps1 `
  -Model "C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf" `
  -BuildDir build-kvarn-cuda-static-vs `
  -Context 512 `
  -Batch 512 `
  -Repeat 16 `
  -KvarnIters 4 `
  -CheckPackedRepeat `
  -CheckPackedSplit `
  -ScratchMaxNmse 1e-5 `
  -SplitMaxNmse 1e-5 `
  -RepeatMaxNmse 1e-12 `
  -FlashAttn off
```

---

## Patch 2 — Make 512d dual-stream K/V body-store event-ordered and non-blocking

File: `ggml/src/ggml-cuda/kvarn.cu`

Problem:

`ggml_cuda_kvarn_store_kv_body_512_pipelined()` launches K work on the main CUDA stream and V work on a global nonblocking aux stream, then calls:

```cpp
cudaStreamSynchronize(aux_stream);
```

Two issues:

1. It blocks the host during every seal, undermining graph/launch overlap.
2. The aux stream reads `v_tile` after `v_tile` was filled on the main stream, but there is no explicit `cudaStreamWaitEvent()` making aux wait for main-stream tile readiness.

Implementation target:

- Replace the naked aux stream helper with a per-device/thread aux state that owns one aux stream and two timing-disabled events.
- At function entry, record a main-stream `ready` event and make aux wait before launching V kernels.
- At function exit, record aux completion and make the main stream wait, not the host.
- Delete `cudaStreamSynchronize(aux_stream)`.

Suggested code:

```diff
diff --git a/ggml/src/ggml-cuda/kvarn.cu b/ggml/src/ggml-cuda/kvarn.cu
@@
-static cudaStream_t kvarn_aux_cuda_stream() {
-    static cudaStream_t stream = nullptr;
-    if (stream == nullptr) {
-        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
-    }
-    return stream;
-}
+struct kvarn_aux_cuda_state {
+    cudaStream_t stream     = nullptr;
+    cudaEvent_t  main_ready = nullptr;
+    cudaEvent_t  aux_done   = nullptr;
+    int          device     = -1;
+};
+
+static kvarn_aux_cuda_state & kvarn_aux_cuda_state_get() {
+    thread_local kvarn_aux_cuda_state st;
+
+    int dev = 0;
+    cudaGetDevice(&dev);
+    if (st.stream == nullptr || st.device != dev) {
+        st.device = dev;
+        cudaStreamCreateWithFlags(&st.stream, cudaStreamNonBlocking);
+        cudaEventCreateWithFlags(&st.main_ready, cudaEventDisableTiming);
+        cudaEventCreateWithFlags(&st.aux_done,   cudaEventDisableTiming);
+    }
+    return st;
+}
@@
-    cudaStream_t aux_stream = kvarn_aux_cuda_stream();
+    kvarn_aux_cuda_state & aux = kvarn_aux_cuda_state_get();
+    cudaStream_t aux_stream = aux.stream;
+
+    // k_tile/v_tile are produced by prior kernels on cuda_stream. The aux stream
+    // must not read them until those writes are visible.
+    cudaEventRecord(aux.main_ready, cuda_stream);
+    cudaStreamWaitEvent(aux_stream, aux.main_ready, 0);
@@
-    cudaStreamSynchronize(aux_stream);
+    // Keep downstream consumers on cuda_stream ordered after the V-side aux work
+    // without blocking the host thread.
+    cudaEventRecord(aux.aux_done, aux_stream);
+    cudaStreamWaitEvent(cuda_stream, aux.aux_done, 0);
 }
```

Notes:

- If this code is ever captured inside a CUDA graph and event APIs are not allowed in that capture mode, guard the pipelined path behind a capture check or fall back to the single-stream reference path during capture.
- Also remove the currently unused `pipeline_scratch_floats` local in `ggml_cuda_kvarn_store_body_pending_heads_minmax()` if warnings are promoted.

---

## Patch 3 — Add `KVarN+ISWA prepare()` timing trace

File: `src/llama-kv-cache-kvarn-iswa.cpp`

Purpose:

The current composite path does `kv_base->prepare(ubatches)` and `kv_swa->prepare(ubatches)` for every batch. That is a plausible part of the remaining decode/prefill gap. Add a tiny opt-in trace before changing behavior.

Suggested patch:

```diff
diff --git a/src/llama-kv-cache-kvarn-iswa.cpp b/src/llama-kv-cache-kvarn-iswa.cpp
@@
 #include <limits>
+#include <chrono>
@@
 static uint32_t kvarn_ubatch_limit(uint32_t default_limit, bool & invalid_debug_override) {
@@
 }
+
+static bool kvarn_iswa_prepare_trace_enabled() {
+    const char * env = std::getenv("LLAMA_KVARN_ISWA_PREPARE_TRACE");
+    return env != nullptr && std::strcmp(env, "0") != 0;
+}
@@
-    auto sinfos_base = kv_base->prepare(ubatches);
+    const auto t_base0 = std::chrono::steady_clock::now();
+    auto sinfos_base = kv_base->prepare(ubatches);
+    const auto t_base1 = std::chrono::steady_clock::now();
@@
-    auto sinfos_swa = kv_swa->prepare(ubatches);
+    const auto t_swa0 = std::chrono::steady_clock::now();
+    auto sinfos_swa = kv_swa->prepare(ubatches);
+    const auto t_swa1 = std::chrono::steady_clock::now();
@@
     if (sinfos_swa.empty()) {
@@
     }
+
+    if (kvarn_iswa_prepare_trace_enabled()) {
+        const auto base_us = std::chrono::duration_cast<std::chrono::microseconds>(t_base1 - t_base0).count();
+        const auto swa_us  = std::chrono::duration_cast<std::chrono::microseconds>(t_swa1  - t_swa0 ).count();
+        uint32_t n_tokens_total = 0;
+        for (const llama_ubatch & ub : ubatches) {
+            n_tokens_total += ub.n_tokens;
+        }
+        LLAMA_LOG_INFO("%s: KVarN+ISWA prepare trace: ubatches=%zu tokens=%u base_us=%lld swa_us=%lld\n",
+                __func__, ubatches.size(), n_tokens_total, (long long) base_us, (long long) swa_us);
+    }
```

Add `#include <cstring>` if this translation unit does not already get it transitively.

Run Gemma true KVarN with:

```powershell
$env:LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA = "1"
$env:LLAMA_KVARN_ISWA_PREPARE_TRACE = "1"
# run tg64 and pp512 diagnostic matrix
```

---

## Patch 4 — If pp512 remains below gate: batch multiple seal records for Gemma `n_head_kv == 1`

Current code batches heads only when `layer.head_dim_k >= 512 && layer.n_head_kv > 1`, so it does not help Gemma because Gemma KVarN layers log `n_head_kv=1`. For Gemma pp512, the active shape is typically two body records on each of eight KVarN layers.

Implementation target:

- Add a new ggml op that seals multiple records for one KV head in one graph node.
- The op can internally loop records first; the immediate win is fewer graph nodes and fewer repeated op dispatches. It does not need a giant fused Sinkhorn kernel on first implementation.
- Prefer a tensor input of record IDs or, if all current pp512 record IDs are contiguous, a compact `{first_record, n_records}` op param.

Graph change location:

`src/llama-graph.cpp`, both standalone KVarN and KVarN+ISWA paths where `seal_records` is currently looped:

```cpp
for (const uint32_t seal_record : seal_records) {
    ... store_kv_body_record_from_pending(..., seal_record)
}
```

New policy:

```cpp
if (layer.head_dim_k >= 512 && layer.n_head_kv == 1 && seal_records.size() > 1) {
    ggml_build_forward_expand(gf, mctx_kvarn->store_kv_body_records_from_pending(
        ctx0, body_store_scratch, il, /*ih=*/0, seal_records));
} else if (layer.head_dim_k >= 512 && layer.n_head_kv > 1) {
    ... existing all-head path ...
} else {
    ... existing per-record/per-head loop ...
}
```

CUDA implementation:

- Add `ggml_cuda_kvarn_store_body_pending_records_minmax(...)` alongside `ggml_cuda_kvarn_store_body_pending_heads_minmax(...)`.
- It should reuse one `k_tile`, one `v_tile`, and one `pipeline` scratch; loop over records inside the CUDA backend call.
- Use Patch 2 event ordering in the 512d pipelined helper.
- Trace should report `n_records_batch` and `record_start` / IDs.

Acceptance:

- `-TraceStore` on Gemma pp512 should show one store op per KVarN layer for two records, not two store ops.
- Tier 2 logits pass.
- Gemma pp512 absolute KVarN t/s improves without tg64 regression.

---

## Production guardrails

Do not flip `src/llama-model.cpp` Gemma fallback until all of these are true:

- Gemma true experimental KVarN+ISWA pp512 >= 90% KVarN/normal.
- Gemma true experimental KVarN+ISWA tg64 >= 90% KVarN/normal.
- Tier 0 KVarN tests pass.
- Tier 2 logits pass.
- Qwen regression passes after CUDA changes.

Use the existing fallback unless `LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA=1` is explicitly set.

---

## Round 13 implementation update - 2026-06-12

Implemented and validated in the local working tree:

- Hardened `scripts/kvarn/run_production_gate.ps1` with explicit diagnostic-env rejection, Tier 2 Qwen arg/layer/body-record propagation, and manifest fields for Tier 2 topology.
- Hardened `scripts/kvarn/run_mainline_parity_matrix.ps1` with dirty-state/build metadata, exact command filenames, fallback-observed fields, KVarN layer set extraction, and expanded mixed-attn inner trace summary fields.
- Made diagnostic KVarN CUDA env flags dynamic instead of cached for 256d warpqk/body-mirror and dequant-cache tracing.
- Added 256d body-mirror scratch sizing across graph/runtime/test paths, but kept 256d body mirror disabled by default.
- Added first-pass Qwen 256d boundary input dumping. Captured failing forced-256d call was `head_dim=256`, `n_queries=193`, `n_sink=128`, `n_records=0`, `n_pending=0`, `n_tail=65`, `mask_type=F32`; this is a no-body sink/tail batch shape, not a body-record dequant shape.
- Tried broadening the no-body sink/tail batch kernel to 256d. Full Qwen3.6 packed-vs-split failed with NMSE `2.354e-02`, so this route was reverted and must remain disabled until a split-equivalent implementation exists.
- Added 256d all-head pending-store routing for Qwen-style `n_head_kv=2`, including larger scratch sizing and backend support for strided body/scale destination views. Store trace now shows `n_heads=2` and 20 store ops for Qwen `pp512` instead of 40 per-head store ops.
- Tuned Qwen 256d sinktail decode CTA size to 512 threads. This did not conclusively clear the `tg64` gate; keep measuring against variance.

Validation completed:

- Focused CTest passed:
  `ctest --test-dir build-kvarn-cuda-static-vs -C Release -R "test-batch-split|test-kvarn-kv|test-kvarn-cuda|test-kvarn-server-load-failure" --output-on-failure`
- Memory estimator self-test passed:
  `python scripts/kvarn/kv_memory_estimate.py --self-test`
- Qwen2.5 CUDA smoke passed for contexts `256 512`, expected layers `0-27`.
- Qwen3.6 MTP default true KVarN logits passed:
  packed repeat NMSE `0`; packed-vs-split NMSE `0`; expected layers `3-39:4`; max body records `2`; `-ncmoe 34`.

Current short parity measurements:

- Qwen3.6 MTP true KVarN, no trace, `r=3`, `-ncmoe 34`, artifact `artifacts/kvarn-mainline-parity/20260612-115825`:
  - `pp512`: mainline `206.39 t/s`, KVarN `213.05 t/s`, `103.2%` PASS in this run.
  - `tg64`: mainline `42.31 t/s`, KVarN `36.06 t/s`, `85.2%` FAIL.
- Qwen trace run, artifact `artifacts/kvarn-mainline-parity/20260612-115751`:
  - `pp512` store trace shows 20 `kv:dim256/g128` store ops with `n_heads=2`.
- Gemma 4 12B true KVarN+ISWA, forced with `LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA=1`, no trace, `r=3`, expected layers `5-47:6`, artifact `artifacts/kvarn-mainline-parity/20260612-115930`:
  - `pp512`: mainline `2313.96 t/s`, KVarN `2035.20 t/s`, `88.0%` FAIL.
  - `tg64`: mainline `66.42 t/s`, KVarN `66.32 t/s`, `99.8%` PASS.

Current blockers:

- Do not run 4k/4k production diagnostics yet. Qwen `tg64` and Gemma true KVarN+ISWA `pp512` are still below the 90% short gate.
- Do not enable 256d warpqk or 256d no-body sink/tail batch by default. Forced 256d paths are still not packed-vs-split equivalent for full Qwen3.6 MTP.
- Gemma fallback remains the production-safe default, but it is not a true KVarN+ISWA performance data point. True Gemma KVarN data requires `LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA=1`.

Recommended next patches:

- Qwen `tg64`: specialize or further tune `sinktail-decode` for the 256d, no-body, sink-only decode shape. Current traces show `sinktail-decode` only and zero body records.
- Qwen `pp512`: repeat `r=5` no-trace and warmup/no-warmup A/B to confirm the 256d all-head store batching improvement survives variance.
- Gemma true KVarN+ISWA `pp512`: continue the existing plan to batch multiple seal records for `n_head_kv == 1`, because Gemma still records two body seals per KVarN layer.

---

## Round 14 true Gemma KVarN investigation - 2026-06-12

Scope:

- Focus was corrected to true Gemma 4 12B KVarN+ISWA, not fallback. All true Gemma runs below used `LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA=1` and expected layers `5-47:6`.
- The production-safe Gemma fallback remains valid default behavior, but fallback is not counted as a true KVarN data point.

Implemented locally:

- Routed 512d, `n_head_kv == 1` single-record seals through the existing pending-record store op instead of the per-record fallback path.
  - `src/llama-graph.cpp`: both standalone KVarN and KVarN+ISWA now call `store_kv_body_records_from_pending()` for non-empty `seal_records`.
  - `ggml/src/ggml.c` and `ggml/src/ggml-cuda/ggml-cuda.cu`: `GGML_OP_KVARN_STORE_KV_BODY` pending-record mode now accepts `n_record_batch == 1`.
  - Store trace confirms Gemma pp512 now emits `kind=kv-records ... n_record_batch=1` for all 16 body stores.
- Hardened graph-side sink/tail clamping so policy experiments like `--kvarn-tail-tokens 256` no longer crash decode when the cache constructor clamps effective tail capacity.
  - Fixed the `sink_tail_k->ne[2] >= n_sink + n_tail` assert by using effective clamped KVarN params in graph active windows, seal-record decisions, reuse checks, and stable decode op params.

Rejected experiments:

- A 512d warpqk causal-mask skip path was logits-equivalent on Gemma but did not produce a useful speedup and made the default kernel shape more complex. It was removed.
- `--kvarn-tail-tokens 256` improved Gemma pp512 only slightly and did not clear the gate; `--kvarn-tail-tokens 384` removed body records for pp512 but still did not clear the gate. Larger-tail policy is not a production fix.

Validation completed after Round 14 changes:

- Focused CTest passed:
  `ctest --test-dir build-kvarn-cuda-static-vs -C Release -R "test-batch-split|test-kvarn-kv|test-kvarn-cuda|test-kvarn-server-load-failure" --output-on-failure`
- Memory estimator self-test passed.
- Qwen2.5 CUDA smoke passed for contexts `256 512`, exact expected layers `0-27`.
- Gemma true KVarN+ISWA logits passed packed repeat, packed-vs-split, and packed-vs-scratch with expected layers `5-47:6`.
- Qwen2.5 logits passed packed repeat, packed-vs-split, and packed-vs-scratch with NMSE `0`.
- Qwen3.6 MTP logits passed packed repeat and packed-vs-split with NMSE `0`, expected layers `3-39:4`, `-ncmoe 34`.
- Unsupported smoke passed.
- Tail clamp regression check passed: Gemma true KVarN `tg64` with `--kvarn-tail-tokens 256` now clamps and runs instead of asserting.

Current short parity after Round 14:

- Gemma 4 12B true KVarN+ISWA, artifact `artifacts/kvarn-mainline-parity/round13-final-gemma-true-kvarn-r3`:
  - `pp512`: mainline `2286.81 t/s`, KVarN `2015.14 t/s`, `88.1%` FAIL.
  - `tg64`: mainline `66.38 t/s`, KVarN `65.88 t/s`, `99.2%` PASS.
- Qwen3.6 MTP, artifact `artifacts/kvarn-mainline-parity/round13-final-qwen-mtp-r3`:
  - `pp512`: mainline `249.78 t/s`, KVarN `219.41 t/s`, `87.8%` FAIL.
  - `tg64`: mainline `41.60 t/s`, KVarN `35.78 t/s`, `86.0%` FAIL.

Updated diagnosis:

- Gemma pp512 is not primarily blocked by body-store dispatch overhead. Store routing changed correctly, but pp512 remained below gate.
- Tail expansion showed the same: removing body records for pp512 with `--kvarn-tail-tokens 384` still stayed around the low 80% range.
- The remaining Gemma short-prefill gap is therefore mostly in KVarN attention/topology versus mainline attention, plus the structural 384+128 ubatch split.
- The high-impact Gemma patch is a direct current-ubatch body-store path for contiguous prefill. That would allow a single q512 graph to seal records whose source tokens are in `k_cur/v_cur`, instead of forcing the safe 384+128 split that exists because record 1's source tokens are produced inside the same ubatch.

Recommended next implementation target:

- Add direct current-ubatch record sealing for contiguous prefill:
  - Detect single-sequence contiguous prompt chunks where full body records are fully contained in `k_cur/v_cur`.
  - Build K/V tiles directly from `k_cur/v_cur` for those records.
  - Use existing K/V body store kernels first; only fuse further after traces show it is still material.
  - Then relax `kvarn_tail_safe_ubatch_limit()` only for the direct-current-ubatch-safe case.
- Keep the scratch/split reference paths as correctness oracles for every change.

---

## Round 15 production patchset pass - 2026-06-12

Scope:

- Implemented the low-risk parts of `KVARN_PRODUCTION_PATCHSET_DEV_AGENT.md` without enabling any experimental 256d warpqk or 256d body-mirror path by default.
- Did not run 4k/4k benchmarks in this pass because the patchset was limited to production-gate hardening and diagnostics. Keep long-context performance work behind the short correctness and parity gates.

Implemented locally:

- Hardened `scripts/kvarn/run_production_gate.ps1` to reject boundary-dump diagnostic environment variables in production-gate runs unless `-AllowDiagnosticEnv` is explicitly used.
- Expanded `scripts/kvarn/run_mainline_parity_matrix.ps1` Markdown output so `summary.md` now includes SHAs, dirty flags, model path, GPU/runtime, run order, warmup, flash-attn, KVarN preset/iters/quantile, expected layers, fallback policy, actual layers, max body records, fallback observed, and exact command filenames. CSV still carries the full trace fields.
- Added inner-trace evidence reporting to `scripts/kvarn/compare_cuda_logits_ref.ps1` when `-TraceAttn` is enabled.
- Extended the Qwen 256d boundary-dump metadata in `ggml/src/ggml-cuda/ggml-cuda.cu` with exact-call/layer/body-active filters plus stride, scratch-capacity, mirror, QT, inferred layer, and CUDA-mode fields. The dump remains off unless `LLAMA_KVARN_ATTN_BOUNDARY_DUMP=1`.
- Added helper scripts:
  - `scripts/kvarn/capture_qwen_boundary.ps1`
  - `scripts/kvarn/replay_qwen_boundary.ps1`
- Documented the gate hardening and boundary-capture limitations in `scripts/kvarn/README.md`.

Validation completed:

- Build passed:
  `cmake --build build-kvarn-cuda-static-vs --config Release --target llama-bench llama-results test-kvarn-cuda-scratch-ref -j 8`
- `llama-cli` build passed.
- PowerShell syntax checks passed for the edited KVarN scripts.
- `git diff --check` passed.
- Focused CTest passed:
  `ctest --test-dir build-kvarn-cuda-static-vs -C Release -R "test-batch-split|test-kvarn-kv|test-kvarn-cuda|test-kvarn-server-load-failure" --output-on-failure`
- Memory estimator self-test passed:
  `python scripts/kvarn/kv_memory_estimate.py --self-test`
- Qwen2.5 CUDA smoke passed for contexts `256 512`, exact layers `0-27`.
- Qwen2.5 traced logits passed packed repeat, packed-vs-split, and packed-vs-scratch at NMSE `0`.
- Qwen3.6 boundary capture produced and validated after rebuilding `llama-results`:
  `artifacts/kvarn-boundary/round15-qwen36-default-first-body-passing/call_000000`
  - `head_dim=256`
  - `n_queries=384`
  - `n_head=16`
  - `n_head_kv=2`
  - `n_sink=128`
  - `n_records=1`
  - `n_pending=0`
  - `n_tail=128`
  - `mask_stride_query_bytes=1536`
  - `scores_nelems=512`
  - `body_records_cap=2`
  - `cuda_trace_mode=split`
  - `body_mirror_allowed=false`
  - `body_mirror_used=false`
  - `inferred_layer=3`

Transient Qwen correctness false alarm resolved:

- Before rebuilding `llama-results`, current dirty `kvarn-atx-integration` produced Qwen3.6 MTP packed-repeat failures at `Repeat=450` (`1.300e-04`) and `Repeat=384` (`1.245e-04`). No diagnostic environment variables were leaked.
- A detached same-session A/B worktree at `63fa0bbd0` passed Qwen3.6 MTP `Repeat=384` packed-repeat and packed-vs-split with NMSE `0`.
- After rebuilding current `llama-results`, current dirty `kvarn-atx-integration` passed Qwen3.6 MTP at both `Repeat=384` and `Repeat=450` with packed-repeat and packed-vs-split NMSE `0`, expected layers `3-39:4`, body-record checks enabled, and `-ncmoe 34 -fit off`.
- Conclusion: the observed Qwen failures were a stale or inconsistent local build artifact, not a Round 15 diagnostic-code regression.

Current status:

- No experimental 256d warpqk or 256d body mirror path is enabled by default.
- Qwen3.6 MTP default true KVarN correctness is clean again for the active-body short diagnostic cases tested.
- This pass did not include a new mainline parity benchmark. The last recorded production blocker remains short prefill parity, especially Qwen3.6 `pp512` and true-KVarN Gemma `pp512`.

Recommended next step:

- Run short mainline parity again with the hardened harness and no diagnostic env:
  - Qwen3.6 MTP `pp512,tg64`, `-r 3`, `-ncmoe 34`, expected layers `3-39:4`.
  - Gemma 4 12B true KVarN `pp512,tg64`, `-r 3`, expected layers `5-47:6`, force experimental ISWA only when intentionally measuring true KVarN instead of fallback.
- Use the new boundary capture only to debug correctness-equivalent 256d experiments. Do not enable 256d warpqk or 256d body mirror for production until full Qwen3.6 MTP packed-repeat and packed-vs-split are both NMSE `0`.

---

## Round 24 production parity handoff - 2026-06-13

Current local branch before this commit:

- Branch: `kvarn-atx-integration`
- Previous remote/head: `d8a9e3d6f kvarn: add exact GQA tiled prefill path`
- Mainline baseline build SHA: `263cc04a5`
- KVarN build was dirty during measurement because this Round 24 patchset was still uncommitted.

Production-short status:

- The required short production gate is now passing for both required models.
- Gemma measurements below are **true KVarN+ISWA**, not fallback. They require `LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA=1` and showed the expected KVarN layers `5,11,17,23,29,35,41,47`.
- Qwen3.6 MTP measurements showed the expected KVarN layers `3,7,11,15,19,23,27,31,35,39` with `-ncmoe 34`.

Implemented in Round 24:

- Added a direct prefill attention path for KVarN when the current ubatch already contains the complete contiguous prompt window. This lets prefill use the normal current `q_cur/k_cur/v_cur` attention path while still writing KVarN sink/tail/body cache records.
- Added direct current-ubatch body-store views from `k_cur/v_cur`, including all-head stores for Qwen-style 256d multi-KV-head layers.
- Extended the same direct prefill path to the Gemma true KVarN+ISWA non-SWA/base branch. This was the missing production path for Gemma; before this change Gemma `pp512` still went through `ggml_kvarn_attn_mixed`.
- Relaxed the KVarN ubatch splitter only for the direct-prefill-safe initial contiguous prompt case, so short prefill can run as a full `512` token graph instead of the older split topology.
- Added low-overhead CUDA timing instrumentation controlled by `LLAMA_KVARN_CUDA_TIMING=1`.
- Added `LLAMA_KVARN_PREFILL_DIRECT_TRACE=1` to prove when the direct prefill path is selected. Gemma trace confirmed `KVarN graph iswa prefill-direct trace: use=1 ... mask=[512,512]`.
- Kept the unsafe 256d warpqk path disabled by default. The production Qwen path uses the correctness-clean scalar/GQA route with the body mirror support, not the rejected warpqk threshold experiment.
- Fixed `scripts/kvarn/run_unsupported_smoke.ps1` so its Gemma KVarN+ISWA guard test explicitly sets `LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA=1`; otherwise Gemma defaults to normal fallback and the KVarN-specific invalid-env test is not exercised.

Validation completed after Round 24 changes:

- Build passed:
  `cmake --build build-kvarn-cuda-static-vs --config Release --target llama-bench llama-results test-kvarn-cuda-scratch-ref test-kvarn-kv -j 8`
- `llama-cli` and `llama-server` builds passed.
- Focused CTest passed:
  `ctest --test-dir build-kvarn-cuda-static-vs -C Release -R "test-batch-split|test-kvarn-kv|test-kvarn-cuda|test-kvarn-server-load-failure" --output-on-failure`
- Memory estimator self-test passed:
  `python scripts/kvarn/kv_memory_estimate.py --self-test`
- Gemma true KVarN+ISWA logits passed packed repeat and packed-vs-split with NMSE `0`, expected layers `5-47:6`, body-record check enabled.
- Qwen3.6 MTP logits passed packed repeat and packed-vs-split with NMSE `0`, expected layers `3-39:4`, body-record check enabled, `-ncmoe 34 -fit off`.
- Qwen2.5 FP16 CLI smoke passed for contexts `256 512`; KVarN correctly rejected that 64d-head model with the expected unsupported-dimension guard.
- Unsupported smoke passed after the Gemma true-KVarN guard fix.

Short production parity results:

- Gemma 4 12B true KVarN+ISWA, artifact `artifacts/kvarn-mainline-parity/round24-gemma-true-iswa-direct-r3`:
  - `pp512`: mainline `2271.95 t/s`, KVarN `2215.96 t/s`, `97.5%` PASS.
  - `tg64`: mainline `66.24 t/s`, KVarN `65.68 t/s`, `99.2%` PASS.
- Qwen3.6 MTP, artifact `artifacts/kvarn-mainline-parity/round24-qwen-full-direct-r3`:
  - `pp512`: mainline `244.35 t/s`, KVarN `315.93 t/s`, `129.3%` PASS.
  - `tg64`: mainline `42.76 t/s`, KVarN `42.63 t/s`, `99.7%` PASS.

Long-context diagnostic results:

- Gemma 4 12B true KVarN+ISWA, artifact `artifacts/kvarn-mainline-parity/round24-gemma-true-iswa-direct-long4096-r1`:
  - `pp4096`: mainline `2272.36 t/s`, KVarN `891.21 t/s`, `39.2%` FAIL.
  - `tg4096`: mainline `61.46 t/s`, KVarN `31.37 t/s`, `51.0%` FAIL.
- Qwen3.6 MTP, artifact `artifacts/kvarn-mainline-parity/round24-qwen-full-direct-long4096-r1`:
  - `pp4096`: mainline `187.97 t/s`, KVarN `106.35 t/s`, `56.6%` FAIL.
  - `tg4096`: mainline `49.02 t/s`, KVarN `32.13 t/s`, `65.5%` FAIL.

Current diagnosis:

- The short production path is no longer the blocker. Both required models clear `pp512` and `tg64` with true KVarN layer routing.
- The remaining production-quality gap is the long body-heavy path. At `4096`, both models allocate `30` body records per KVarN layer and fall far below mainline.
- For long prefill, the direct attention shortcut cannot avoid the cost of sealing many body records. The current direct body-store graph emits one store op per body record per KVarN layer for the current-ubatch path. That is acceptable for two records at `pp512`, but likely too expensive at thirty records.
- For long decode, the mixed body-attention path and dequant/cache behavior are the likely bottlenecks. Do not re-enable the rejected 256d warpqk experiment as a production fix; isolate body dequant/cache reuse and mixed-attention throughput first.
- Subagent exploration was attempted for body-store and mixed-attention analysis, but both subagents hit the Codex usage limit before returning results. The next engineer should continue locally from the files and artifacts listed here.

Recommended next implementation targets:

- Add a batched direct-current-ubatch body-store op for contiguous prefill records, analogous to the existing pending-record batch store. The high-value target is replacing one op per record per layer with a small number of batched record-store ops.
- Profile `GGML_OP_KVARN_STORE_KV_BODY` under `LLAMA_KVARN_CUDA_TIMING=1` on `pp4096` to split long prefill time into store, direct attention, and other graph costs before changing kernels.
- Add/verify dequant-cache hit/miss summaries for `tg4096`; if hit rates are low, fix cache keying/epoch reuse before kernel work. If hit rates are high, focus on `ggml_kvarn_attn_mixed` body-record attention throughput.
- Keep the Round 24 short gate as the regression floor: Gemma true KVarN and Qwen MTP must stay `>=90%` on `pp512` and `tg64`, with logits NMSE `0` and expected KVarN layer routing.

---

## Round 25 long-context and accuracy-gate handoff - 2026-06-13

Current local branch before this commit:

- Branch: `kvarn-atx-integration`
- Previous pushed head: `e637347ed kvarn: add direct prefill production path`
- Mainline baseline build SHA: `263cc04a5`
- KVarN build SHA in benchmark logs: `e637347ed`; build contained the dirty Round 25 working tree during measurement.

Implemented in Round 25:

- Added an end-to-end f16-vs-KVarN accuracy gate:
  - New doc: `docs/KVARN_ACCURACY_GATE.md`.
  - New script: `scripts/kvarn/run_accuracy_gate.ps1`.
  - The script runs `llama-perplexity` twice from the same binary: once with normal f16 KV and once with KVarN KV.
  - The gate forces `-np 1`, `-fit off`, and `-b <= -c` because `llama-perplexity` derives its internal sequence count from `batch/context`, and KVarN currently supports only `n_seq_max = 1`.
  - The script validates KVarN cache engagement and expected KVarN layer IDs.
- Added a direct-current-ubatch body-record batch store path for contiguous prefill, but kept it diagnostic-only:
  - Env gate: `LLAMA_KVARN_ENABLE_DIRECT_RECORD_BATCH=1`.
  - Default production behavior remains the existing per-record all-head direct store.
  - Reason: enabling the batch path regressed Qwen long prefill in local testing.
- Added an F32 dequant-cache helper for the scalar/GQA body-attention path, but kept it diagnostic-only:
  - Env gate: `LLAMA_KVARN_ENABLE_F32_DEQUANT_CACHE=1`.
  - Default production behavior remains the prior scalar/GQA dequant path.
  - Reason: enabling the cache path regressed Qwen `tg4096` in local testing.
- Preserved the Round 24 production defaults:
  - No rejected 256d warpqk production dispatch.
  - No default F32 dequant cache.
  - No default direct-record batch store.

Introductory explanation:

- Short prompts now work because KVarN only has a small amount of compressed body cache to manage.
- Long prompts fail because KVarN creates many compressed body records. At `4096` tokens, each selected KVarN layer has `30` body records.
- The current long path spends too much time either writing those body records, unpacking/dequantizing them, or running mixed attention over them.
- The Round 25 experiments tried to batch record writes and cache dequantized body data, but both experiments were slower when enabled. They are useful diagnostics, not production fixes yet.
- The Opus accuracy-gate review added an important missing check: the old gates compared KVarN path A against KVarN path B, which can miss a systematic KVarN-vs-f16 attention error. The new gate compares KVarN against normal f16 KV directly.

Validation completed after Round 25 changes:

- Build passed earlier in this pass:
  `cmake --build build-kvarn-cuda-static-vs --config Release --target llama-bench llama-results test-kvarn-kv test-kvarn-cuda-scratch-ref -- /m:1 /v:minimal /clp:ErrorsOnly`
- Focused CTest passed:
  `ctest --test-dir build-kvarn-cuda-static-vs -C Release -R "test-batch-split|test-kvarn-kv|test-kvarn-cuda|test-kvarn-server-load-failure" --output-on-failure`
- Memory estimator self-test passed:
  `python scripts/kvarn/kv_memory_estimate.py --self-test`
- Large-model logits correctness passed earlier in this pass:
  - Gemma 4 12B true KVarN+ISWA packed repeat and packed-vs-split NMSE `0`, expected layers `5-47:6`.
  - Qwen3.6 MTP packed repeat and packed-vs-split NMSE `0`, expected layers `3-39:4`, `-ncmoe 34`.
- Accuracy-gate smoke passed on local Qwen3.5 0.8B:
  - Artifact: `artifacts/kvarn-accuracy/round25-qwen35-smoke`.
  - f16 PPL `17.4150`.
  - KVarN PPL `17.4202`.
  - Increase `0.03%` with `-MaxPplIncrease 0.10`.

Current short production parity after Round 25 defaults:

- Gemma 4 12B true KVarN+ISWA, artifact `artifacts/kvarn-mainline-parity/round25-gemma-short-default-r3`:
  - `pp512`: mainline `2107.73 t/s`, KVarN `2131.22 t/s`, `101.1%` PASS.
  - `tg64`: mainline `62.52 t/s`, KVarN `62.18 t/s`, `99.5%` PASS.
- Qwen3.6 MTP, artifact `artifacts/kvarn-mainline-parity/round25-qwen-short-default-r3`:
  - `pp512`: mainline `96.75 t/s`, KVarN `96.90 t/s`, `100.2%` PASS.
  - `tg64`: mainline `27.44 t/s`, KVarN `38.18 t/s`, `139.1%` PASS.
- Note: the Qwen short numbers are noisy in this environment, but this clean default run passed all short cells and showed expected KVarN layer routing.

Current long-context diagnostics after Round 25 defaults:

- Gemma 4 12B true KVarN+ISWA, artifact `artifacts/kvarn-mainline-parity/round25-gemma-long4096-default-r1`:
  - `pp4096`: mainline `2120.02 t/s`, KVarN `842.66 t/s`, `39.7%` FAIL.
  - `tg4096`: mainline `58.17 t/s`, KVarN `30.10 t/s`, `51.7%` FAIL.
- Qwen3.6 MTP, artifact `artifacts/kvarn-mainline-parity/round25-qwen-long4096-default-r1`:
  - `pp4096`: mainline `167.98 t/s`, KVarN `103.91 t/s`, `61.9%` FAIL.
  - `tg4096`: mainline `45.28 t/s`, KVarN `29.73 t/s`, `65.7%` FAIL.

Rejected or diagnostic-only Round 25 experiments:

- `LLAMA_KVARN_ENABLE_DIRECT_RECORD_BATCH=1`:
  - Intended to reduce long prefill body-store graph overhead by storing multiple direct body records per op.
  - Regressed Qwen `pp4096` locally, so it is not enabled by default.
  - Important observation: `llama-bench pp4096` does not run as one single 4096-token ubatch; it emits multiple prompt chunks. The direct-batch path must handle real ubatch chunking, not only a synthetic single-chunk assumption.
- `LLAMA_KVARN_ENABLE_F32_DEQUANT_CACHE=1`:
  - Intended to reduce repeated scalar/GQA body dequantization.
  - Regressed Qwen `tg4096` locally, so it is not enabled by default.
  - Next review should inspect cache keying, scratch identity, active body span, and epoch invalidation before attempting to make this path production.

Recommended next implementation targets:

1. Run the new accuracy gate on the two production models with small but meaningful datasets before accepting any deeper performance change:
   - Gemma true KVarN+ISWA with `LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA=1`, expected layers `5-47:6`.
   - Qwen3.6 MTP with `-ncmoe 34`, expected layers `3-39:4`.
2. Investigate the Opus Hadamard-rotation concern as a correctness hypothesis, not yet as a proven bug:
   - Confirm whether body K/V rotation, Q rotation, sink/tail K/V rotation, and output rotation are mathematically consistent in the actual production graph.
   - If KVarN-vs-f16 accuracy fails on production models, start here before further speed work.
3. For long prefill, instrument body-store timing and graph topology without enabling the regressing direct-batch path by default:
   - Count body-store ops per layer/chunk.
   - Separate Sinkhorn/minmax cost from K/V staging and metadata sealing.
   - Rework direct-record batching only after it beats the default path on Qwen and Gemma `pp4096`.
4. For long decode, focus on body-record attention throughput:
   - First fix or disprove F32/F16 dequant-cache reuse.
   - If cache hit rates are already high, optimize `ggml_kvarn_attn_mixed` body attention itself.
5. Keep the short gate as a hard regression floor:
   - Gemma true KVarN+ISWA and Qwen MTP must remain `>=90%` on `pp512` and `tg64`.
   - Expected KVarN layer routing and logits NMSE `0` remain required.

---

## Round 26 paper-fidelity redirect - 2026-06-13

Current local branch before this commit:

- Branch: `kvarn-atx-integration`
- Previous pushed head: `aafdb00f1 kvarn: add long-context diagnostics and accuracy gate`
- Author reference clone used for comparison: `C:\Users\sjake\OneDrive\Documents\New project\KVarN-reference`, commit `a601d2a`
- Paper source used for audit: <https://arxiv.org/html/2606.03458v1>

Implemented in Round 26:

- Applied the incoming next-patches handover and added `scripts/kvarn/run_iters_sweep.ps1`.
- Hardened `run_iters_sweep.ps1`:
  - runs each accuracy gate in a child PowerShell process so one candidate cannot terminate the whole sweep;
  - stops after the first passing candidate by default, with `-ContinueAfterPass` for a full curve;
  - requires both the gate summary status and the numeric threshold to pass before marking a candidate PASS.
- Extended `scripts/kvarn/run_accuracy_gate.ps1` and the sweep with `-Chunks`, forwarded to `llama-perplexity --chunks`, so production accuracy checks can be bounded and repeatable.
- Added `docs/KVARN_PAPER_FIDELITY_AUDIT.md`.
- Added `docs/KVARN_NEXT_PATCHES_HANDOVER.md` from the incoming handover package.

Key correction:

- Do not treat lower `--kvarn-iters` as the main fix.
- Sec. 3.3, Sec. 4.2, and Appendix I of the paper imply a batched/fused systems design. The paper's low overhead claim is not evidence that a serial per-record/per-head Sinkhorn implementation is cheap.
- The branch has many correct conceptual parts, but the long path does not yet match the paper's optimized shape.

Production accuracy results:

- Qwen3.6 MTP, context 4096, batch 4096, `--chunks 2`, `-ncmoe 34`, expected layers `3-39:4`, artifact `artifacts/kvarn-iters-sweep/round26-qwen36-ctx4096-chunks2-fixed`:
  - `iters=1`: f16 PPL `4.5843`, KVarN PPL `6.6577`, increase `45.23%`.
  - `iters=2`: increase `47.21%`.
  - `iters=3`: increase `45.28%`.
  - `iters=4`: increase `46.71%`.
  - Result: no candidate passed `MaxPplIncrease=5%`.
- Gemma 4 12B true KVarN+ISWA, context 4096, batch 4096, `--chunks 2`, expected layers `5-47:6`, artifact `artifacts/kvarn-accuracy/round26-gemma-ctx4096-chunks2-iters4`:
  - f16 PPL `418.2027`.
  - KVarN PPL `137079490.9999`.
  - Increase `32778141.51%`.
  - Result: FAIL.

Interpretation:

- Long-context production accuracy is currently failing before performance tuning.
- Existing KVarN-vs-KVarN logits checks are insufficient to prove f16 faithfulness.
- Long-context throughput failures should now be interpreted as implementation-shape and correctness/fidelity failures, not proof that KVarN is inherently slow.

Immediate next patch sequence:

1. Add low-level tile-frame correctness tests for Qwen 256d and Gemma 512d:
   - store one K/V body tile;
   - dequant it;
   - compare rotated-query dot dequantized-rotated-K against f16 score;
   - compare output with dequantized V unrotated or otherwise transformed according to the production graph.
2. Fix any Hadamard/Q/V/output-frame mismatch exposed by those tests.
3. Implement true batched VarN/body-store over `records * heads` tiles:
   - per-tile scratch slices;
   - grid-strided Hadamard/Sinkhorn/quantize/finalize phases;
   - no per-tile launch loop inside `ggml_cuda_kvarn_store_body_direct_records_minmax`.
4. Implement fused dual-scale dequant/attention for long decode:
   - apply RTN scale/zp and KVarN scale inside the attention/dequant kernel;
   - avoid production F32/F16 body mirrors.
5. Improve long decode occupancy by splitting body records across CTAs per head/query and reducing partials.
6. Revisit `--kvarn-iters` only after production-model accuracy gates pass.

Stop conditions:

- Do not lower production `--kvarn-iters` from 4.
- Do not turn on `LLAMA_KVARN_ENABLE_DIRECT_RECORD_BATCH` by default until it is a true batched CUDA kernel and passes production accuracy.
- Do not optimize for throughput before bounded f16-vs-KVarN production accuracy passes for both Qwen3.6 MTP and Gemma true KVarN+ISWA.

---

## Round 27 paper-shaped body-store prototype - 2026-06-13

Current local branch before this commit:

- Branch: `kvarn-atx-integration`
- Previous pushed head: `b31b6cc1a kvarn: add paper fidelity audit and bounded accuracy sweeps`

Implemented:

- Added executable paper-frame CUDA checks in `tests/test-kvarn-cuda-dequant.cpp`:
  - K body dequant is verified in the paper frame: `H(q) . K_rot_deq == q . H(K_rot_deq)`.
  - V body dequant is verified in the paper frame: unrotating each dequantized value row before the weighted sum equals unrotating the weighted rotated-value sum.
  - These run for 128d, Qwen-shaped 256d, and Gemma-shaped 512d cases.
- Added an opt-in true batched direct-record body-store phase path:
  - Env gates: `LLAMA_KVARN_ENABLE_DIRECT_RECORD_BATCH=1` and `LLAMA_KVARN_ENABLE_DIRECT_RECORD_BATCH_PHASES=1`.
  - CUDA phases now operate over `records * heads` tiles instead of calling the one-tile pipeline in an internal record/head loop.
  - K phases run on the main stream and V phases on the existing auxiliary stream.
  - The path is currently fullrange/default-preset only: `k4/v2`, `rtn_quantile=1.0`, power-of-two head dims.
  - Added bounded trace evidence: `KVarN CUDA store-body batched-phases trace: used=1`.
- Increased body-store scratch only when `LLAMA_KVARN_ENABLE_DIRECT_RECORD_BATCH_PHASES=1`, so default memory behavior is unchanged.
- Added append-aware anchored dequant-cache invalidation:
  - Stores that know their record range call `ggml_cuda_kvarn_mark_body_store_records`.
  - Anchored F32/F16 body mirrors can now refill from the first dirty record when the cache is only one store epoch behind.
  - Caches older than the immediately previous store epoch still fall back to a conservative full refill.
  - `ggml-cuda.cu` now marks the root K-body pointer with the specific record range before dispatch instead of doing an unconditional full-body mark.

Validation:

- Build passed:
  `cmake --build build-kvarn-cuda-static-vs --config Release --target llama-cli llama-bench test-kvarn-cuda-scratch-ref -- /m:1 /v:minimal /clp:ErrorsOnly`
- Focused CTest passed:
  `ctest --test-dir build-kvarn-cuda-static-vs -C Release -R "test-batch-split|test-kvarn-kv|test-kvarn-cuda|test-kvarn-server-load-failure" --output-on-failure`
- Memory estimator passed:
  `python scripts/kvarn/kv_memory_estimate.py --self-test`
- Small Qwen3.5 smoke passed with the opt-in batched phase path and expected layers `3-23:4`.
- Small Qwen3.5 logits passed with opt-in batched phases:
  - packed repeat NMSE `0`
  - packed-vs-split NMSE `0`
  - packed-vs-scratch NMSE `0`
- Qwen3.6 MTP logits passed with opt-in batched phases, expected layers `3-39:4`, `-ncmoe 34`:
  - packed repeat NMSE `0`
  - packed-vs-split NMSE `0`
  - packed-vs-scratch NMSE `0`
- Gemma 4 12B true KVarN+ISWA logits passed with opt-in batched phases, expected layers `5-47:6`, `LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA=1`:
  - packed repeat NMSE `0`
  - packed-vs-split NMSE `0`
  - packed-vs-scratch NMSE `0`

Performance findings:

- Small Qwen3.5 pp2048:
  - default direct-record store: `6052.26 ± 138.33 t/s`
  - batched phases: `5876.72 ± 253.18 t/s`
  - result: regression/noise; not a win.
- Small Qwen3.5 pp4096:
  - default direct-record store: `4046.65 ± 62.53 t/s`
  - batched phases after K/V stream overlap: `4073.22 ± 41.14 t/s`
  - result: small `~0.7%` improvement, not decisive.
- Qwen3.6 MTP pp4096, `-ncmoe 34`, r1:
  - default direct-record store: `189.77 t/s`
  - batched phases: `185.21 t/s`
  - result: production regression. Keep batched phases opt-in only.
- After the append-aware mark fix, Qwen3.6 MTP pp4096 with `LLAMA_KVARN_ENABLE_F32_DEQUANT_CACHE=1` traced `dirty_from=0` for the direct-record prefill path and measured `189.97 t/s`.
  - Interpretation: this long-prefill graph is still storing direct-record spans from record 0, so append-only mirror reuse is not being exercised in this cell.
  - The append-aware cache remains useful for append-only store layouts, but it is not the long pp4096 production fix by itself.
- Small Qwen3.5 direct packed/no-F32-body-mirror A/B:
  - default pp4096/tg512: `4047.27 t/s`, `188.99 t/s`
  - `LLAMA_KVARN_ATTN_DISABLE_BODY_F32_MIRROR=1`: `3786.53 t/s`, `189.46 t/s`
  - result: direct packed load saves scratch memory but hurts 256d prefill and does not materially help decode. Do not change the default.

Interpretation:

- The branch now has an executable, paper-shaped batched VarN/body-store prototype, but the first implementation is not production faster on Qwen3.6.
- The long pp4096 timing still shows mixed attention growing with active body records; body-store launch count is not the only long-context bottleneck.
- The append-aware dequant-cache change removes avoidable scratch-dequant repeats only when store ranges are append-only; Qwen3.6 pp4096 currently does not hit that condition.
- This strengthens the paper-steered conclusion: the next production work should focus on fused/occupancy-improved body-record attention, not iteration reduction and not enabling the current batched-store prototype by default.

Immediate next work:

1. Keep `LLAMA_KVARN_ENABLE_DIRECT_RECORD_BATCH_PHASES` diagnostic-only.
2. Use the new paper-frame unit as a hard regression test for any further fused dequant/attention work.
3. Implement long decode/body-attention parallelism across record partitions per head/query, with reductions for max/sum/value.
4. Revisit direct-record batched store after attention is no longer dominant; possible refinements are reducing scratch writes, fusing Hadamard+VarN more tightly, and avoiding one CTA per row/column for small tiles.

---

## Round 28 paper-frame scaffold - 2026-06-14

Inputs reviewed:

- `C:\Users\sjake\Downloads\KVARN_PAPER_FRAME_SCAFFOLD.md`
- `C:\Users\sjake\Downloads\KVARN_ROUND28_DECODE_PARALLEL_BODY_ATTENTION_HANDOFF.md`
- `C:\Users\sjake\Downloads\KVARN_ROUND28_DECODE_PARALLEL_BODY_ATTENTION.patch`

Implemented from the paper-frame scaffold:

- Added `llama_kvarn_hadamard_matrix(uint32_t d)`, built by the same normalized Sylvester-Hadamard butterfly used by body store.
- Added default-off graph plumbing under `LLAMA_KVARN_ENABLE_PAPER_FRAME=1`.
- In regular KVarN and true KVarN+ISWA mixed-attention graph paths:
  - rotate `q` before `ggml_kvarn_attn_mixed`;
  - rotate sink/tail `k/v` before cache copy;
  - unrotate the KVarN attention output before reshape/output projection.
- Added paper-frame handling for pending body stores:
  - rotated sink/tail evictions copied into pending are treated as already rotated when later sealed;
  - direct raw prefill record stores are forced through the direct-record body-store op under paper-frame mode so they still receive exactly one Hadamard rotation.

Important status:

- `LLAMA_KVARN_ENABLE_PAPER_FRAME` remains default off.
- The scaffold compiles and preserves default behavior, but it does not yet pass the production 4096 accuracy gates.
- The Round 28 decode-parallel patch file is not directly applyable (`git apply --check` reports a corrupt patch at line 246). More importantly, decode speed work should stay behind accuracy until the 4096 f16-vs-KVarN gate is fixed.

Validation:

- Build passed:
  `cmake --build build-kvarn-cuda-static-vs --config Release --target llama-cli llama-bench llama-perplexity test-kvarn-cuda-scratch-ref -- /m:1 /v:minimal /clp:ErrorsOnly`
- Focused CTest passed:
  `ctest --test-dir build-kvarn-cuda-static-vs -C Release -R "test-batch-split|test-kvarn-kv|test-kvarn-cuda|test-kvarn-server-load-failure" --output-on-failure`
- Memory estimator passed:
  `python scripts/kvarn/kv_memory_estimate.py --self-test`
- Small Qwen3.5, `LLAMA_KVARN_ENABLE_PAPER_FRAME=1`, context 512:
  - packed repeat NMSE `0`
  - packed-vs-split NMSE `0`
  - packed-vs-scratch NMSE `0`
- Qwen3.6 MTP accuracy, context 512, chunks 2:
  - default: f16 PPL `3.8849`, KVarN PPL `3.8956`, increase `0.28%`, PASS.
  - paper-frame: f16 PPL `3.8849`, KVarN PPL `3.8956`, increase `0.28%`, PASS.
- Qwen3.6 MTP accuracy, context 4096, chunks 2:
  - old Round 26 baseline: increase `46.71%`.
  - paper-frame q/sink/output only: increase `46.71%`.
  - paper-frame plus pending-rotated store: increase `39.83%`.
  - paper-frame plus pending-rotated store plus forced direct-record raw prefill stores: increase `37.24%`.
  - Result: still FAIL.
- Gemma 4 12B true KVarN+ISWA accuracy, context 4096, chunks 2, `LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA=1`:
  - f16 PPL `418.2027`.
  - KVarN PPL `149227980.3447`.
  - Result: still FAIL.

Interpretation:

- Opus's frame critique was partially correct: fixing the frame changed Qwen3.6 long-context PPL in the right direction.
- The frame scaffold is not sufficient. There is still at least one long-context accuracy bug or a severe KVarN quantization/config mismatch before long-context speed numbers should be treated as production evidence.
- The next investigation should compare long-context KVarN against f16 at the layer/window boundary, not benchmark another faster kernel first.

Recommended next work:

1. Add a selected-layer boundary dump for f16 K/V, KVarN dequantized body, rotated sink/tail, pending, `q`, scores, and output under `LLAMA_KVARN_ENABLE_PAPER_FRAME=1`.
2. Run the dump at the first token where Qwen3.6 ctx4096 KVarN diverges materially from f16.
3. Decide whether the remaining error is body quantization itself, scale application, pending/direct store layout, mask/window indexing, or output-frame handling.
4. Only after Qwen3.6 and Gemma 4096 accuracy pass, resurrect the decode-parallel body-attention patch as an opt-in diagnostic path.

---

## Round 29 8-bit body ablation - 2026-06-14

Opus's proposed split was implemented as a diagnostic preset:

- Added `--kvarn-preset kvarn_k8v8_g128` to the common CLI and `llama-bench`.
- Production default remains `kvarn_k4v2_g128`.
- Backend/model/graph validation now accepts KVarN group size 128 with 1-8 bit K/V, so the 8-bit preset can exercise the same KVarN body topology.

Validation:

- Build passed:
  `cmake --build build-kvarn-cuda-static-vs --config Release --target llama-perplexity llama-bench test-kvarn-cuda-scratch-ref -- /m:1 /v:minimal /clp:ErrorsOnly`
- Focused CTest passed:
  `ctest --test-dir build-kvarn-cuda-static-vs -C Release -R "test-batch-split|test-kvarn-kv|test-kvarn-cuda|test-kvarn-server-load-failure" --output-on-failure`
- Memory estimator passed:
  `python scripts/kvarn/kv_memory_estimate.py --self-test`

Accuracy ablation results, all with `LLAMA_KVARN_ENABLE_PAPER_FRAME=1`:

- Qwen3.6 MTP ctx4096/chunks2, `kvarn_k8v8_g128`, `-ncmoe 34`:
  - f16 PPL `4.5843`
  - KVarN PPL `6.2226`
  - increase `35.74%`
  - artifact: `artifacts/kvarn-accuracy/round29-paper-frame-qwen36-ctx4096-k8v8`
- Qwen3.6 MTP ctx4096/chunks2, `kvarn_k8v8_g128`, direct prefill disabled with `LLAMA_KVARN_DISABLE_PREFILL_DIRECT_ATTN=1`:
  - f16 PPL `4.5843`
  - KVarN PPL `6.0538`
  - increase `32.06%`
  - artifact: `artifacts/kvarn-accuracy/round29-paper-frame-qwen36-ctx4096-k8v8-pendingonly`
- Gemma 4 12B true KVarN+ISWA ctx4096/chunks2, `kvarn_k8v8_g128`, `LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA=1`:
  - f16 PPL `418.2027`
  - KVarN PPL `224912419.4692`
  - increase `53780619.13%`
  - artifact: `artifacts/kvarn-accuracy/round29-paper-frame-gemma-ctx4096-k8v8`

Interpretation:

- 8-bit K/V did not rescue Qwen3.6. The remaining ctx4096 failure is therefore unlikely to be explained primarily by 2-bit V quantization quality.
- Disabling the direct prefill record path improved Qwen only slightly, so direct-record batching is not the primary corruption source.
- The next Qwen target is a boundary-level f16-vs-KVarN body-attention dump: original f16 K/V for the selected active body records, KVarN packed body/scales, dequantized K/V, `q`, mask, scores, probabilities, and output.
- Gemma true KVarN+ISWA remains a separate blocker. Since 8-bit is still catastrophic, investigate ISWA window/eviction/recycling record indexing and frame state after the shared Qwen body-attention issue is isolated.

---

## Round 30 pending-K layout fix - 2026-06-14

Inputs reviewed:

- `C:\Users\sjake\Downloads\KVARN_BITWIDTH_AND_CTX4096_DIAGNOSIS.md`
- `C:\Users\sjake\Downloads\0004kvarnvariablebitwidthpresets.patch`
- `C:\Users\sjake\Downloads\KVARN_ROUND28_PENDING_K_LAYOUT_FIX_HANDOFF.md`
- `C:\Users\sjake\Downloads\KVARN_ROUND28_PENDING_K_LAYOUT_FIX.patch`

Implemented:

- Fixed `kvarn_transpose_pending_k_head_kernel()` so pending K is gathered into the K body-store layout:
  - old, wrong: `k_tile[g*head_dim + d]`
  - new, correct: `k_tile[d*group_size + g]`
- Added an independent CUDA regression for pending-record stores at 256d and 512d, both with and without `LLAMA_KVARN_ENABLE_PAPER_FRAME=1`.
- Generalized KVarN preset parsing to `kvarn_k<K>v<V>_g128` with K,V in `[2,8]`, so `k4v4`, `k8v4`, and `k8v8` can be tested without more parser patches.

Validation:

- Build passed:
  `cmake --build build-kvarn-cuda-static-vs --config Release --target llama-perplexity llama-bench test-kvarn-cuda-scratch-ref -- /m:1 /v:minimal /clp:ErrorsOnly`
- Focused CTest passed:
  `ctest --test-dir build-kvarn-cuda-static-vs -C Release -R "test-batch-split|test-kvarn-kv|test-kvarn-cuda|test-kvarn-server-load-failure" --output-on-failure`
- Memory estimator passed:
  `python scripts/kvarn/kv_memory_estimate.py --self-test`
- Qwen3.6 MTP ctx512/chunks2 accuracy still passes:
  - f16 PPL `3.8849`
  - KVarN PPL `3.8956`
  - increase `0.28%`
- Qwen3.6 MTP ctx512 logits with paper-frame passed:
  - packed repeat NMSE `0`
  - packed-vs-split NMSE `0`
  - packed-vs-scratch NMSE `0`

Long-context accuracy results, all ctx4096/chunks2:

- Qwen3.6 MTP, paper-frame, `kvarn_k4v2_g128`, `-ncmoe 34`:
  - before fix: `37.24%` PPL increase
  - after fix: f16 PPL `4.5843`, KVarN PPL `5.1155`, increase `11.59%`
  - artifact: `artifacts/kvarn-accuracy/round30-pending-k-fix-qwen36-ctx4096-k4v2`
- Qwen3.6 MTP, paper-frame, `kvarn_k8v8_g128`, `-ncmoe 34`:
  - f16 PPL `4.5843`
  - KVarN PPL `5.1320`
  - increase `11.95%`
  - artifact: `artifacts/kvarn-accuracy/round30-pending-k-fix-qwen36-ctx4096-k8v8`
- Qwen3.6 MTP, paper-frame, `kvarn_k4v2_g128`, direct prefill disabled:
  - KVarN PPL `5.2479`
  - increase `14.48%`
  - artifact: `artifacts/kvarn-accuracy/round30-pending-k-fix-qwen36-ctx4096-k4v2-pendingonly`
- Qwen3.6 MTP, default/non-paper frame, `kvarn_k4v2_g128`:
  - KVarN PPL `6.1751`
  - increase `34.70%`
  - artifact: `artifacts/kvarn-accuracy/round30-pending-k-fix-qwen36-ctx4096-k4v2-defaultframe`
- Gemma 4 12B true KVarN+ISWA, paper-frame, `kvarn_k4v2_g128`, `LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA=1`:
  - before fix: KVarN PPL `149227980.3447`
  - after fix: f16 PPL `418.2027`, KVarN PPL `944.8251`, increase `125.93%`
  - artifact: `artifacts/kvarn-accuracy/round30-pending-k-fix-gemma-ctx4096-k4v2`
- Gemma 4 12B true KVarN+ISWA, paper-frame, `kvarn_k8v8_g128`:
  - f16 PPL `418.2027`
  - KVarN PPL `520.7935`
  - increase `24.53%`
  - artifact: `artifacts/kvarn-accuracy/round30-pending-k-fix-gemma-ctx4096-k8v8`

Interpretation:

- The pending-K layout fix is real and large. It moves Qwen3.6 ctx4096 from `37.24%` to `11.59%` PPL increase and moves Gemma true KVarN+ISWA from catastrophic to bounded-but-still-failing.
- Qwen does not improve with `k8v8`, so its remaining failure is not primarily bit-width quality.
- Disabling direct prefill is worse for Qwen, so the remaining Qwen gap is not fixed by routing everything through pending.
- Paper-frame remains necessary: default/non-paper frame is still much worse on Qwen.
- Gemma benefits from `k8v8`, but still fails by `24.53%`; true KVarN+ISWA likely has an additional sliding-window/eviction/indexing problem.

Next work:

1. Keep the pending-K fix and its independent regression.
2. Add an f16-vs-KVarN long-context boundary dump for Qwen at the first body-active divergence after the fixed pending-K store.
3. Compare f16 K/V, dequantized KVarN body, scores, probabilities, and output for the same query/head/record span.
4. For Gemma true KVarN+ISWA, inspect ISWA eviction/recycling record mapping after the shared body-store fix.
5. Do not benchmark or optimize long-context speed as production evidence until ctx4096 accuracy is below the 5% gate.

---

## Round 31 vLLM oracle and normalization diagnostics - 2026-06-14

Inputs reviewed:

- `C:\Users\sjake\Downloads\KVARN_VLLM_RECODE_HANDOFF.md`
- `C:\Users\sjake\Downloads\0001-kvarn-add-vllm-oracle.patch`
- `C:\Users\sjake\Downloads\0002-kvarn-vllm-recode-patch-plan-doc.patch`

Implemented:

- Added an independent vLLM-style oracle:
  - `scripts/kvarn/kvarn_vllm_oracle.py`
  - `scripts/kvarn/run_vllm_oracle_selftest.ps1`
- Added `docs/KVARN_VLLM_RECODE_PATCH_PLAN.md` to preserve the reference-port patch plan and stop conditions.
- Added an opt-in log/std Sinkhorn implementation behind `LLAMA_KVARN_ENABLE_LOG_STD_SINKHORN=1`.
  - CUDA store path uses log-domain std scaling when the env var is enabled.
  - CPU reference path and CUDA unit-test reference use the same env-gated math, so tests validate the selected normalization.
  - Default production behavior remains unchanged.
- Relaxed KVarN group validation from fixed `g128` to positive power-of-two groups for diagnostics such as `kvarn_k4v4_g64`.

Validation:

- Build passed:
  `cmake --build build-kvarn-cuda-static-vs --config Release --target llama-perplexity test-kvarn-cuda-scratch-ref test-kvarn-kv test-arg-parser -- /m:1 /v:minimal /clp:ErrorsOnly`
- Focused CTest passed:
  `ctest --test-dir build-kvarn-cuda-static-vs -C Release -R "test-batch-split|test-kvarn-kv|test-kvarn-cuda|test-kvarn-server-load-failure|test-arg-parser" --output-on-failure`
- Memory estimator passed:
  `python scripts/kvarn/kv_memory_estimate.py --self-test`
- vLLM oracle self-test passed for head dims 128, 256, 512 and presets `k4v2`, `k4v4`, `k8v8`.
- Opt-in log/std CUDA reference tests passed:
  `LLAMA_KVARN_ENABLE_LOG_STD_SINKHORN=1 ctest --test-dir build-kvarn-cuda-static-vs -C Release -R "test-kvarn-cuda" --output-on-failure`

Qwen3.6 MTP ctx4096/chunks2 accuracy diagnostics, all with `LLAMA_KVARN_ENABLE_PAPER_FRAME=1`, expected layers `3-39:4`, and `-ncmoe 34`:

- Baseline after Round 30 pending-K fix, `kvarn_k4v2_g128`, iters 4:
  - f16 PPL `4.5843`
  - KVarN PPL `5.1155`
  - increase `11.59%`
  - artifact: `artifacts/kvarn-accuracy/round31-vllm-oracle-qwen36-ctx4096-k4v2`
- Log/std Sinkhorn, `kvarn_k4v2_g128`, iters 8:
  - f16 PPL `4.5837`
  - KVarN PPL `5.1395`
  - increase `12.13%`
  - artifact: `artifacts/kvarn-accuracy/round31-logstd-qwen36-ctx4096-k4v2`
- Higher V bitwidth, `kvarn_k4v4_g128`, iters 4:
  - f16 PPL `4.5837`
  - KVarN PPL `5.1196`
  - increase `11.69%`
  - artifact: `artifacts/kvarn-accuracy/round31-qwen36-ctx4096-k4v4`
- Smaller group diagnostic, `kvarn_k4v4_g64`, iters 4:
  - f16 PPL `4.5837`
  - KVarN PPL `5.2641`
  - increase `14.84%`
  - artifact: `artifacts/kvarn-accuracy/round31-qwen36-ctx4096-k4v4-g64`

Interpretation:

- The vLLM-style oracle is now available and passes synthetic checks, but the production ctx4096 accuracy gate is still failing.
- Log/std Sinkhorn as currently implemented is not a production fix for Qwen3.6; it slightly worsens the ctx4096 gate and remains opt-in only.
- `k4v4` and `g64` do not rescue Qwen3.6, so the remaining failure is unlikely to be simple value-bitwidth or tile-size quality.
- Because `k8v8`, `k4v4`, and log/std do not recover accuracy, the next high-value work is a real f16-vs-KVarN body-record boundary oracle, not another speed patch.

Next work:

1. Wire the independent oracle into real body-record dumps:
   - raw source K/V tile
   - rotated source K/V tile
   - balanced/normalized K/V tile
   - packed bytes
   - scales and zero-points
   - dequantized K/V tile
   - selected query, mask, scores, softmax probabilities, and output
2. Compare production CUDA output against the independent Python oracle for one body-active Qwen ctx4096 query/head/record span.
3. If the dump matches the oracle but PPL still fails, investigate layer-routing/cumulative KVarN layer selection; if it diverges, fix the first mismatched boundary.
4. Keep long-context speed benchmarking blocked as production evidence until ctx4096 accuracy is below the 5% gate.

---

## Round 31 follow-up body-record store dump - 2026-06-14

Implemented after the Round 31 push:

- Added an explicit, opt-in CUDA body-record store dump behind:
  - `LLAMA_KVARN_DEBUG_BODY_RECORD_DUMP=1`
  - `LLAMA_KVARN_DEBUG_BODY_RECORD_DIR=<dir>`
  - `LLAMA_KVARN_DEBUG_BODY_RECORD_LIMIT=<n>`
  - optional filters `LLAMA_KVARN_DEBUG_BODY_RECORD=<record>`, `LLAMA_KVARN_DEBUG_BODY_HEAD=<head>`, and `LLAMA_KVARN_DEBUG_BODY_RECORD_CALL=<call>`
- The dump captures one store tile at the store boundary:
  - `k_tile_input.bin`
  - `v_tile_input.bin`
  - `k_rot_or_copy.bin`
  - `v_rot_or_copy.bin`
  - `k_normalized.bin`
  - `v_normalized.bin`
  - `k_body.bin`
  - `v_body.bin`
  - `scales_k.bin`
  - `scales_v.bin`
  - `body_record.json`
- Added `scripts/kvarn/replay_body_record_dump.py` to independently check:
  - the frame transform/copy into the store tile;
  - packed body/scales dequantization back into the rotated frame.

Validation:

- Build passed:
  `cmake --build build-kvarn-cuda-static-vs --config Release --target llama-perplexity test-kvarn-cuda-scratch-ref test-kvarn-kv -- /m:1 /v:minimal /clp:ErrorsOnly`
- Focused CTest passed:
  `ctest --test-dir build-kvarn-cuda-static-vs -C Release -R "test-batch-split|test-kvarn-kv|test-kvarn-cuda|test-kvarn-server-load-failure|test-arg-parser" --output-on-failure`
- Enabled-dump CUDA smoke passed:
  `LLAMA_KVARN_DEBUG_BODY_RECORD_DUMP=1 LLAMA_KVARN_DEBUG_BODY_RECORD_LIMIT=1 ctest --test-dir build-kvarn-cuda-static-vs -C Release -R "test-kvarn-cuda" --output-on-failure`
- Replay on the smoke dump passed the frame contract:
  `python scripts/kvarn/replay_body_record_dump.py --dump artifacts/kvarn-body-record/round31-smoke --max-frame-nmse 1e-10`
  - `k_frame_nmse=0`
  - `v_frame_nmse=0`
  - `k_dequant_nmse=3.209030e-03`
  - `v_dequant_nmse=1.162076e-01`
- Real Qwen3.6 MTP ctx4096/chunks2 dump run reproduced the failing long-context quality level:
  - command output artifact: `artifacts/kvarn-accuracy/round31-bodydump-qwen36-ctx4096-k4v2`
  - body-record dump: `artifacts/kvarn-body-record/round31-qwen36-ctx4096-k4v2`
  - f16 PPL `4.5837`
  - KVarN PPL `5.1165`
  - increase `11.62%`
- Replay on the real Qwen dump passed the frame contract:
  `python scripts/kvarn/replay_body_record_dump.py --dump artifacts/kvarn-body-record/round31-qwen36-ctx4096-k4v2 --max-frame-nmse 1e-10`
  - `k_frame_nmse=0`
  - `v_frame_nmse=0`
  - `k_dequant_nmse=5.827223e-03`
  - `v_dequant_nmse=2.754752e-01`

Interpretation:

- The diagnostic path is inert by default and works when enabled.
- The smoke dump proves the local frame transform/copy boundary can be checked independently.
- The real Qwen ctx4096 dump proves the selected store boundary is in the correct frame, so the remaining `11.62%` PPL gap is not explained by another K/V frame mismatch in that record.
- The dumped first record has high V dequant loss at `k4v2`. However, prior full-model `k8v8` did not recover Qwen ctx4096, so this cannot be treated as a complete explanation until the same query/head is checked at the mixed-attention boundary.
- Next step: capture the existing mixed-attention boundary dump for the same body-active Qwen run and compare per-token scores/probabilities/output against the independently replayed store/dequant data.

---

## Round 32 mixed-attention boundary replay - 2026-06-15

Inputs reviewed:

- `C:\Users\sjake\Downloads\KVARN_ROUND32_MIXED_ATTN_BOUNDARY_HANDOFF.md`
- `C:\Users\sjake\Downloads\0001-kvarn-generalize-mixed-attn-boundary-dump.patch`
- `C:\Users\sjake\Downloads\0002-kvarn-add-mixed-attn-boundary-replay.patch`
- `C:\Users\sjake\Downloads\0003-docs-round32-mixed-attn-boundary-handoff.patch`

Implemented:

- Generalized `LLAMA_KVARN_ATTN_BOUNDARY_DUMP` so it is no longer hard-coded to `head_dim == 256`.
- Added `LLAMA_KVARN_ATTN_BOUNDARY_DUMP_HEAD_DIM=<D>` for bounded 512d Gemma captures.
- Renamed new boundary dump mode to `kvarn-mixed-attn-boundary-input`.
- Added `scripts/kvarn/replay_mixed_attn_boundary.py`.
- Added `docs/KVARN_ROUND32_MIXED_ATTN_BOUNDARY_HANDOFF.md`.
- Added the new dump head-dim env var to KVarN env validation and production-gate cleanup.

Validation:

- Build passed:
  `cmake --build build-kvarn-cuda-static-vs --config Release --target llama-perplexity test-kvarn-cuda-scratch-ref test-kvarn-kv test-arg-parser -- /m:1 /v:minimal /clp:ErrorsOnly`
- Focused CTest passed:
  `ctest --test-dir build-kvarn-cuda-static-vs -C Release -R "test-batch-split|test-kvarn-kv|test-kvarn-cuda|test-kvarn-server-load-failure|test-arg-parser" --output-on-failure`
- Memory estimator passed:
  `python scripts/kvarn/kv_memory_estimate.py --self-test`
- vLLM oracle self-test passed:
  `scripts/kvarn/run_vllm_oracle_selftest.ps1`
- Replay script py-compile passed:
  `python -m py_compile scripts/kvarn/replay_mixed_attn_boundary.py`
- Existing Qwen boundary replay passed:
  - artifact: `artifacts/kvarn-boundary/round22-qwen-warpqk-qt1-layer3-fullqo/call_000000_2`
  - `out_nmse=3.287233e-14`

Qwen3.6 MTP ctx4096/chunks2 mixed-attention boundary:

- Run artifact: `artifacts/kvarn-accuracy/round32-qwen36-ctx4096-k4v2-boundary`
- Boundary artifact: `artifacts/kvarn-boundary/round32-qwen36-ctx4096-k4v2/call_000000`
- Accuracy result:
  - f16 PPL `4.5837`
  - KVarN PPL `5.1249`
  - increase `11.81%`
- Replay result:
  - layer `3`, query `0`, head `0`
  - tokens `4096`, body records `30`
  - CUDA mode `fused-batch`
  - `out_nmse=3.981132e-13`
  - `out_max_abs=4.470348e-07`

Gemma 4 12B true KVarN+ISWA ctx4096/chunks2 mixed-attention boundary:

- Run artifact: `artifacts/kvarn-accuracy/round32-gemma-ctx4096-k4v2-boundary`
- Boundary artifact: `artifacts/kvarn-boundary/round32-gemma-ctx4096-k4v2/call_000000`
- Accuracy result:
  - f16 PPL `404.2271`
  - KVarN PPL `9466.9647`
  - increase `2241.99%`
- Replay result:
  - layer `5`, query `0`, head `0`
  - tokens `4096`, body records `30`
  - CUDA mode `warpqk-f16-dequant`
  - `out_nmse=1.050904e-12`
  - `out_max_abs=3.337860e-06`

Interpretation:

- The selected Qwen and Gemma mixed-attention rows are internally consistent with the packed KVarN body/scales and token order.
- This rules out a single-row CUDA mixed-attention readout bug for the sampled rows.
- Qwen still fails ctx4096 by about `11.8%`, so the next diagnostic should compare KVarN layer outputs against the f16 baseline over multiple layers/heads/queries, not just replay KVarN against itself.
- Gemma true KVarN+ISWA still has a much larger long-context failure. Since the sampled 512d mixed-attention row replays correctly, the next Gemma target is ISWA topology/window/eviction state or cumulative layer effects, not the sampled body-record dequant order.

Next work:

1. Add f16-vs-KVarN activation boundary capture around KVarN-selected layers at ctx4096.
2. Sample several Qwen layers/heads/queries, not only layer 3/head 0/query 0.
3. For Gemma, inspect ISWA body-record span mapping, window eviction/recycling, and whether the layer-5 sampled row is representative of later KVarN layers.
4. Keep long-context speed work blocked until Qwen ctx4096 is under the 5% PPL gate and Gemma true-KVarN+ISWA is either fixed or explicitly kept out of production.

---

## Regression note - Gemma true KVarN+ISWA broke after cleanup - 2026-06-15

This is the handoff section to read before more Gemma work.

Finding:

- The severe Gemma true KVarN+ISWA ctx4096 quality regression was introduced by or before commit `94dc11c40 cleanup: remove obsolete TurboQuant experiments`.
- It was not introduced by the Round 32 mixed-attention boundary replay patch.
- It was not caused by `LLAMA_KVARN_ATTN_BOUNDARY_DUMP`; the no-dump recheck reproduced the same bad result.

Evidence:

- Round 30 artifact at commit `3af66f419`:
  - artifact: `artifacts/kvarn-accuracy/round30-pending-k-fix-gemma-ctx4096-k4v2`
  - f16 PPL `418.2027`
  - KVarN PPL `944.8251`
  - increase `125.93%`
- Recheck at commit `94dc11c40`:
  - artifact: `artifacts/kvarn-accuracy/bisect-94dc-gemma-ctx4096-k4v2`
  - f16 PPL `404.2271`
  - KVarN PPL `9210.3180`
  - increase `2178.50%`
- Current branch no-dump recheck:
  - artifact: `artifacts/kvarn-accuracy/round32-gemma-ctx4096-k4v2-nodump-recheck`
  - f16 PPL `404.2271`
  - KVarN PPL `9466.9647`
  - increase `2241.99%`

Interpretation:

- `94dc11c40` was intended as cleanup, but it touched broad runtime areas, including CUDA/backend/graph-adjacent code. Treat it as suspect for Gemma true KVarN+ISWA until surgically bisected.
- Round 32 mixed-attention replay still matters diagnostically: sampled Qwen and Gemma rows replay internally against packed KVarN cache with near-zero NMSE. That only proves KVarN is self-consistent for sampled rows; it does not prove KVarN matches f16.
- The cleanup regression must be resolved before trusting Gemma long-context speed or quality numbers.

Do not push yet:

- Local uncommitted corrective patch attempts from `KVARN_ROUND32_CORRECTIVE_PATCHES_HANDOFF.md`:
  - RTN nearest-even plus `1e-10` scale clamp.
  - pending multi-record seal guard.
- These local changes built and passed focused tests, and Qwen ctx512 logits stayed NMSE 0, but they did not fix Qwen ctx4096 and did not recover Gemma:
  - Qwen3.6 ctx4096 k4v2: `+11.57%`
  - Qwen3.6 ctx4096 k8v8: `+11.85%`
  - Gemma true KVarN+ISWA ctx4096 k4v2: `+2374.62%`
  - Gemma true KVarN+ISWA ctx4096 k8v8: `+1191.52%`

Recommended next action:

1. Revert or surgically bisect `94dc11c40` before any further Gemma optimization.
2. If reverting wholesale is too broad, split the cleanup into small chunks and require this gate after each chunk:
   - Gemma true KVarN+ISWA ctx4096/chunks2, paper-frame, expected layers `5-47:6`, `kvarn_k4v2_g128`.
3. Only after Gemma returns to the bounded Round 30 level should the team decide whether to keep the RTN/guard corrective changes.

---

## Rejected patch - lossless quality defaults - 2026-06-15

Claude-provided patch `C:\Users\sjake\Downloads\0001kvarnlosslessqualityfix.patch` was applied locally and tested, then reverted.

What the patch changed:

- Made log/std Sinkhorn normalization default-on for CPU and CUDA unless `LLAMA_KVARN_ENABLE_LOG_STD_SINKHORN=0`.
- Changed `llama_kvarn_default_params().value_bits` from `2` to `4`.
- Added `docs/KVARN_LOSSLESS_QUALITY_FIX.md`.

Local integration fixes needed for the patch:

- `tests/test-kvarn-cuda-dequant.cpp` independent CPU reference had to follow the same default-on log/std behavior, otherwise CPU/CUDA packed-byte checks failed.
- `tests/test-kvarn-kv.cpp` byte-count assertions had to be updated for V4 default.

Focused validation after those local integration fixes:

- `ctest --test-dir build-kvarn-cuda-static-vs -C Release -R "test-batch-split|test-kvarn-kv|test-kvarn-cuda|test-kvarn-server-load-failure|test-arg-parser" --output-on-failure`: PASS, 7/7.
- `python scripts/kvarn/kv_memory_estimate.py --self-test`: PASS.
- `python scripts/kvarn/kvarn_vllm_oracle.py --self-test --head-dims 128,256,512 --presets k4v2,k4v4,k8v8 --iters 16`: PASS.

Qwen3.6 MTP ctx4096 accuracy results, paper-frame enabled, expected KVarN layers `3-39:4`, `-ncmoe 34`, chunks2:

- `kvarn_k4v4_g128`, log/std default-on, iters 16:
  - artifact: `artifacts/kvarn-accuracy/round33-lossless-default-qwen36-ctx4096-k4v4-logstd16`
  - f16 PPL `4.5837`
  - KVarN PPL `5.1127`
  - increase `11.54%`
  - result: FAIL versus 5% quality gate
- `kvarn_k8v8_g128`, log/std default-on, iters 16:
  - artifact: `artifacts/kvarn-accuracy/round33-lossless-default-qwen36-ctx4096-k8v8-logstd16`
  - f16 PPL `4.5837`
  - KVarN PPL `5.1308`
  - increase `11.94%`
  - result: FAIL versus 5% quality gate

Conclusion:

- Do not push this patch as a production default change.
- V4 default and log/std default-on do not make Qwen3.6 ctx4096 lossless in this branch.
- The claim that high-bit results were only stale from before the pending-K transpose fix is not supported by this run: K8V8 still fails after the pending-K fix and after log/std default-on.
- Because even K8V8 is still around `+12%`, the next useful work is not more bit-width tuning. Continue with f16-vs-KVarN activation/logit boundary capture across layers/heads/queries and the Gemma cleanup-regression bisect.
