# Gemma 4 KVarN+ISWA throughput failure — evidence-based diagnostic

**Audience:** Architect / CUDA implementer  
**Branch:** `kvarn-atx-integration` @ [`f5bdd5b6c`](https://github.com/JakeATX/llama.cpp/commit/f5bdd5b6c) ([compare 686356d61..f5bdd5b6c](https://github.com/JakeATX/llama.cpp/compare/686356d61...f5bdd5b6c))  
**Hardware:** RTX 5070 12 GB, `build-kvarn-cuda-static-vs`, `-fa off`, `-ngl 99`  
**Model:** `gemma-4-12b-it-UD-Q3_K_XL.gguf` (Gemma 4 12B Q3)  
**Gate:** KVarN/normal ≥ 90% on pp512 and tg64 with `LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA=1`

---

## 1. Current failure numbers (primary artifacts)

| Artifact dir | Case | Normal t/s | KVarN t/s | Ratio | Gate |
|--------------|------|----------:|----------:|------:|:----:|
| `artifacts/kvarn-bench/gemma-speed-patch-v2/` | pp512 | 2193.48 ± **510.54** | 1365.94 ± **3.76** | **62.3%** | FAIL |
| `artifacts/kvarn-bench/gemma-speed-patch-v2/` | tg64 | 66.92 ± 3.83 | 44.20 ± 3.06 | **66.0%** | FAIL |
| `artifacts/kvarn-bench/gemma-true-kvarn-post-memset-skip-iters4/` | pp512 | 2356.15 ± **629.57** | 1443.76 ± **6.29** | **61.3%** | FAIL |
| `artifacts/kvarn-bench/gemma-true-kvarn-post-memset-skip-iters4/` | tg64 | 63.99 ± 1.67 | 45.19 ± 0.04 | **70.6%** | FAIL |
| `artifacts/kvarn-bench/gemma-sinktail-decode-p0/` | pp512 | 2629.52 ± 432.56 | 1841.00 ± 9.97 | **70.0%** | FAIL |
| `artifacts/kvarn-bench/gemma-sinktail-decode-p0/` | tg64 | 64.12 ± 1.13 | 45.94 ± 0.14 | **71.6%** | FAIL |
| `artifacts/kvarn-bench/gemma-batch-store-p1/` | pp512 | 2643.18 ± — | 1872.23 ± — | **70.8%** | FAIL |
| `artifacts/kvarn-bench/gemma-batch-store-p1/` | tg64 | 66.35 ± 1.33 | 48.87 ± 0.07 | **73.7%** | FAIL |
| `artifacts/kvarn-bench/gemma-sinktail-fastpath-tg64/` | tg64 | 39.92 ± 27.99 | 43.51 ± 0.10 | **109.0%** | PASS (normal-KV variance) |
| `artifacts/kvarn-bench/gemma-true-kvarn-speed-patch/` | pp512 | 2305.01 ± 588.89 | 1357.04 ± 147.60 | **58.9%** | FAIL (aborted before tg64) |
| `artifacts/kvarn-bench/gemma-gate-push/` | pp512 | 2303.81 ± 641.63 | 1437.14 ± 42.95 | **62.4%** | FAIL |
| `artifacts/kvarn-bench/gemma-gate-push/` | tg64 | 69.45 | 47.89 | **69.0%** | FAIL |

**Source logs:** `pp512.md.txt`, `tg64.md.txt`, `summary.csv` in each directory.

**Post speed-patch (`5f3f037a4`):** marginal pp512 movement (58.9% → 62.3%); tg64 flat (~66–69%). ctest still green.

### Post-merge production fallback check

These are **not** true KVarN+ISWA runs. They validate the production policy where Gemma 4 with `--kv-cache-quant kvarn` routes to normal ISWA KV unless `LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA=1` is set.

| Artifact dir | FA | Case | Normal t/s | Fallback t/s | Ratio | Gate |
|--------------|:--:|------|----------:|-------------:|------:|:----:|
| `artifacts/kvarn-production-gate/20260608-164314/gemma-tier1-production-fallback/` | off | pp512 | 2218.50 | 2715.60 | **122.4%** | PASS |
| `artifacts/kvarn-production-gate/20260608-164314/gemma-tier1-production-fallback/` | off | tg64 | 63.93 | 65.10 | **101.8%** | PASS |
| `artifacts/kvarn-bench/gemma-tier1-fallback-post-common/` | off | pp512 | 2219.04 | 2709.42 | **122.1%** | PASS |
| `artifacts/kvarn-bench/gemma-tier1-fallback-post-common/` | off | tg64 | 64.35 | 64.77 | **100.7%** | PASS |
| `artifacts/kvarn-bench/post-merge-gemma-fa-on/` | on | pp512 | 746.13 | 764.58 | **102.5%** | informational |
| `artifacts/kvarn-bench/post-merge-gemma-fa-on/` | on | tg64 | 35.90 | 36.45 | **101.5%** | informational |

FA-on fallback pp512 moved down from the older ~111% note to **102.5%**, but remains over the production gate because the backend mode is normal ISWA fallback.

---

## 2. Root cause A — pp512 is dominated by **active body-store sealing**

### Evidence: `body records` in bench stderr

**Failing production-shaped runs** (`body records = 2` on all 8 KVarN layers):

```
artifacts/kvarn-bench/gemma-speed-patch-v2/pp512.md.txt (lines 7–15)
  body records = 2  (layers 5,11,17,23,29,35,41,47)
  estimate 4.87 MiB (sink/tail 4.00 MiB, body 0.75 MiB, scales 0.12 MiB)
  KVarN pp512 = 1365.94 t/s
```

Same pattern in `gemma-gate-push/pp512.md.txt`, `gemma-debug-ubatch512/pp512.md.txt`.

**Near-pass diagnostic runs** (`body records = 0` — tokens stayed in sink/tail, no body sealed):

```
artifacts/kvarn-bench/gemma12b-512-tail512-forced-fused/pp512.md.txt (lines 7–15)
  body records = 0
  estimate 8.00 MiB (sink/tail 8.00 MiB, body 0.00 MiB, scales 0.00 MiB)
  KVarN pp512 = 1580.96 t/s  →  ratio 87.3% vs normal 1810.54
```

Command for that run (`pp512.command.txt`):

```
--kvarn-tail-tokens 512 --kvarn-rtn-quantile 0.95 -r 1
```

Manifest: `min_kvarn_body_records=0`, `extra_args=--kvarn-tail-tokens 512`.

### Interpretation

| Config | body_records | KVarN pp512 (abs) | Notes |
|--------|:------------:|------------------:|-------|
| Default tail, iters=4, rtn=1.0 (current gate) | **2** | **~1366–1437** | Real KVarN memory path — seals 2×128-token body groups per layer |
| tail512 + no-body window | **0** | **~1581** | +15% KVarN abs vs body-active; ratio **87%** only when normal denominator ~1810 |
| body_records=2, lower normal baseline | **2** | **1261** | `gemma-debug-ubatch512/` ratio **89.1%** because normal=1416 not 2193 |

**Architect action:** pp512 gap is not “attention only”. With `body records = 2`, every prefill ubatch that seals a 128×512 K/V tile pays:

- Parallel Hadamard (cols/rows, 512 threads) — patched
- Sinkhorn iters=4 — already parallel
- Fullrange quantize (512-wide rows) — patched parallel
- **8 KVarN layers × 2 records** per pp512 bench completion

The latest code skips redundant body-buffer `cudaMemsetAsync` launches for full-range byte-overwriting K4/V2 rows. Post fast-path work (sinktail/dequant/pipelined store), Q4_XL pp512 improved to **~1872 t/s (~71% ratio)** vs **~1369 t/s (~57%)** pre-patch; tg64 KVarN **~49 t/s (~74% ratio)** vs **~45 t/s (~71%)**. Gemma KVarN layers use **`n_head_kv=1`**, so multi-head body-store batching is a graph/sched win only on models with `n_head_kv>1` (Qwen). KVarN absolute throughput is stable (low σ); ratio swings remain driven by normal-KV variance. Next code target: ISWA dual-`prepare()` on pp512 and more KVarN decode absolute headroom when normal tg64 is ~66 t/s.

---

## 3. Root cause B — **split attention kernels** are catastrophic on 512-dim

```
artifacts/kvarn-bench/gemma12b-512-tail512-split/pp512.md.txt
  body records = 0, --kvarn-tail-tokens 512, SPLIT path
  KVarN pp512 = 65.95 t/s  →  ratio 4.2% vs normal 1557.83
```

README confirms: `LLAMA_KVARN_ATTN_SPLIT_KERNELS=1` must remain **opt-in only** (Qwen pp512 regresses to ~16%).

**Architect action:** Do not route Gemma 512-dim production to split kernels. Fused / warpqk only.

---

## 4. Root cause C — **tg64 decode** (~66–69% gate) with **no body work**

```
artifacts/kvarn-bench/gemma-speed-patch-v2/tg64.md.txt (lines 7–15)
  body records = 0
  estimate 4.00 MiB (sink/tail only)
  normal tg64 = 66.92 t/s, KVarN = 44.20 t/s  →  66.0%
```

Decode does **not** hit body-store (records stay 0). Remaining gap is:

1. **512-dim fused-batch / warpqk attention** per token × 64 steps × 8 KVarN layers  
2. **ISWA dual-cache path** — only layers 5–47 step 6 use KVarN; SWA layers use normal KV but `llama_kv_cache_kvarn_iswa` still dual-prepares ubatches  
3. **warpqk V stage** — 512 sequential dim iterations × token-parallel inner loop (patched in `5f3f037a4`); tg64 barely moved

Historical tg64 (pre speed-patch): `gemma-gate-push/` **69.0%**, `decode-fix-20260607/gemma/` **64.6%**.

Post-cleanup A/Bs confirmed the current warpqk default is still best:

| Artifact dir | Variant | KVarN tg64 | Ratio |
|--------------|---------|-----------:|------:|
| `artifacts/kvarn-bench/gemma-true-kvarn-post-memset-skip-iters4/` | default warpqk | 45.19 | **70.6%** |
| `artifacts/kvarn-bench/gemma-true-kvarn-serial-fused-tg64/` | `LLAMA_KVARN_ATTN_SERIAL_FUSED=1` | 38.59 | **60.2%** |
| `artifacts/kvarn-bench/gemma-true-kvarn-decode-per-head-tg64/` | `LLAMA_KVARN_ATTN_DECODE_PER_HEAD=1` | 38.54 | **60.2%** |

**Architect action:** keep warpqk as default. Profile decode with `-TraceAttn` on tg64 and target ISWA `prepare()` overhead plus the 512-dim value stage; do not promote serial fused or per-head decode.

---

## 5. Root cause D — **ratio gate noise** from normal-KV variance

From `gemma-speed-patch-v2/summary.csv` and `pp512.md.txt`:

- **Normal pp512 σ ≈ 510–640 t/s** (run-to-run)
- **KVarN pp512 σ ≈ 4–148 t/s** (much stabler)

Same KVarN absolute throughput (~1262–1366) yields:

| normal baseline | KVarN | Ratio |
|---------------|------:|------:|
| 1416 (`gemma-debug-ubatch512`) | 1262 | **89.1%** |
| 1810 (`tail512-forced-fused`) | 1581 | **87.3%** |
| 2193 (`gemma-speed-patch-v2`) | 1366 | **62.3%** |

**Architect action:** For acceptance, report **absolute KVarN t/s** alongside ratio, or fix normal-KV bench harness warmup/graph state. Consider paired A/B in one process or more repetitions with stable denominator.

---

## 6. What the speed patch changed (and did not)

**Commit `5f3f037a4`** (`ggml/src/ggml-cuda/kvarn.cu`):

- Parallel Hadamard body-store (512-thread blocks)
- `kvarn_attn_mixed_f16_fused_batch_warpqk_kernel` when `head_dim >= 512`
- Parallel fullrange quantize rows
- `RtnQuantile` default **1.0** in bench/logits scripts

**Measured effect:** pp512 58.9% → 62.3%; tg64 ~68.7% → 66.0% (within noise). **Still ~28–34 points below gate.**

---

## 7. Production fallback contrast (policy, not experimental path)

```
artifacts/kvarn-bench/gemma-fallback-smoke2/tg64.md.txt
  No llama_kv_cache_kvarn: lines (ISWA fallback)
  normal = 23.99, kvarn row = 24.73  →  policy gate PASS (~103%)
```

Fallback avoids experimental KVarN+ISWA entirely → stable but **not** measuring true KVarN memory savings.

**Note:** tg64 normal baseline **24 vs 67 t/s** across runs suggests bench/session configuration drift; always cite `manifest.txt` + `*.command.txt` when comparing.

---

## 8. Recommended architect experiments (ordered)

1. **Decode no-body fast path** — validate the dedicated 512d sink/tail fused-batch kernel on `tg64` first. This is the production-shaped case where `body records = 0`, so any tg64 improvement can be attributed to the attention kernel rather than body-store seals.

2. **Body-store profile** — `-TraceStore` on pp512 with default tail; expect shapes like `k:dim512/g128/...` per seal. Compare seal count with vs without `--kvarn-tail-tokens 512`.

3. **Attention profile** — `-TraceAttn` on tg64; record `fused-batch` vs warpqk/sinktail shapes (`dim512`, `n_queries=1`, `n_tokens`).

4. **A/B absolute throughput** (not ratio-only) — report KVarN absolute t/s next to ratio because normal pp512 remains noisy.

5. **ISWA prepare audit** — `src/llama-kv-cache-kvarn-iswa.cpp` dual `prepare()` per ubatch; count graph nodes vs normal ISWA fallback.

6. **Do not flip** `llama-model.cpp` Gemma fallback until **both** pp512 and tg64 ≥ 90% with **body_records ≥ 2** (production-realistic tail), not tail512 no-body diagnostic alone.

---

## 9. Key file index

| Path | Contents |
|------|----------|
| `artifacts/kvarn-bench/gemma-speed-patch-v2/summary.csv` | Latest post-patch ratios |
| `artifacts/kvarn-bench/gemma-speed-patch-v2/pp512.md.txt` | body_records=2, σ values |
| `artifacts/kvarn-bench/gemma12b-512-tail512-forced-fused/` | 87.3% no-body reference |
| `artifacts/kvarn-bench/gemma12b-512-tail512-split/` | 4.2% split-kernel disaster |
| `artifacts/kvarn-bench/gemma-debug-ubatch512/` | 89.1% with body_records=2, low normal denom |
| `scripts/kvarn/README.md` § tail512 diagnostic | Narrative cross-ref |
| `src/llama-model.cpp` | Gemma production fallback (keep until true path passes) |
| `ggml/src/ggml-cuda/kvarn.cu` | Hadamard / warpqk / quantize kernels |
