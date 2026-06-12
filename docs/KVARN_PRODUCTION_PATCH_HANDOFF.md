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
