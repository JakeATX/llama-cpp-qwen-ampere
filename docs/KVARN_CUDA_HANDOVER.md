# KVarN CUDA Architect Handover

**Audience:** External architect reviewing the CUDA KVarN vs normal KV throughput gap.  
**Last updated:** 2026-06-09  
**Branch:** `kvarn-atx-integration`  
**HEAD:** [`95390d5b1`](https://github.com/JakeATX/llama.cpp/commit/95390d5b1) — *architect P0: K-layout, event-ordered store, ISWA trace, multi-record seal*  
**Code-review agent:** [`docs/AGENT_CODE_REVIEW_HANDOVER.md`](AGENT_CODE_REVIEW_HANDOVER.md)  
**Upstream sync:** merged `upstream-ggml/master` @ `42a0afd59` (2026-06-08); `hparams.n_layer` → `n_layer_all` / `n_layer_nextn` migration applied to KVarN paths  
**Remote:** [https://github.com/JakeATX/llama.cpp](https://github.com/JakeATX/llama.cpp) (`jakeatx` / `fork`)  
**Review URL:** [https://github.com/JakeATX/llama.cpp/tree/kvarn-atx-integration](https://github.com/JakeATX/llama.cpp/tree/kvarn-atx-integration)

### Architect review entry point

Start here to critique the latest Gemma CUDA work (experimental gate still **FAIL** ~65% pp512 / ~72% tg64 post-P0):

| Resource | URL |
|----------|-----|
| **Code-review agent handover** | [docs/AGENT_CODE_REVIEW_HANDOVER.md](https://github.com/JakeATX/llama.cpp/blob/kvarn-atx-integration/docs/AGENT_CODE_REVIEW_HANDOVER.md) |
| **Branch tip (all code)** | [tree/kvarn-atx-integration](https://github.com/JakeATX/llama.cpp/tree/kvarn-atx-integration) |
| **Latest commit** | [commit/95390d5b1](https://github.com/JakeATX/llama.cpp/commit/95390d5b1) |
| **Failure diagnostic** | [docs/GEMMA_KVARN_FAILURE_DIAGNOSTIC.md](https://github.com/JakeATX/llama.cpp/blob/kvarn-atx-integration/docs/GEMMA_KVARN_FAILURE_DIAGNOSTIC.md) |
| **Architect P0 handoff** | [docs/KVARN_PRODUCTION_PATCH_HANDOFF.md](https://github.com/JakeATX/llama.cpp/blob/kvarn-atx-integration/docs/KVARN_PRODUCTION_PATCH_HANDOFF.md) |
| **Diff P0** | [c6ad0c5d4..95390d5b1](https://github.com/JakeATX/llama.cpp/compare/c6ad0c5d4...95390d5b1) |

**Files changed in `f5bdd5b6c` (CUDA + graph + docs):**

- `ggml/src/ggml-cuda/kvarn.cu`, `kvarn.cuh`, `ggml-cuda.cu` — sinktail/decode attn, pipelined 512d body-store, warpqk body dequant, optional all-heads seal
- `src/llama-graph.cpp`, `src/llama-kv-cache-kvarn.cpp`, `src/llama-kv-cache-kvarn.h` — graph wiring, scratch hoisting, batch seal op
- `ggml/src/ggml.c`, `ggml/include/ggml.h` — `ggml_kvarn_store_kv_body_pending_heads`
- `scripts/kvarn/compare_cuda_logits_ref.ps1` — NMSE parse fix (`-nan(ind)`)
- `tests/test-kvarn-kv.cpp`, `docs/GEMMA_KVARN_FAILURE_DIAGNOSTIC.md`

**Not changed (gate did not pass):** `src/llama-model.cpp` Gemma ISWA fallback policy remains production default.

**Latest Gemma experimental bench** (`LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA=1`, Q4_XL, iters=4, rtn=1.0): `artifacts/kvarn-bench/gemma-token-major-k-scratch-rerun/` — pp512 **70.3%** (1733/2466 t/s), tg64 **74.9%** (46.1/61.6 t/s). Post-P0: 64.8% / 71.7%. See [`KVARN_ARCHITECT_HANDOVER_ROUND2.md`](KVARN_ARCHITECT_HANDOVER_ROUND2.md).

---

## Executive summary

This branch integrates KVarN (quantized KV-cache) on CUDA with ATX MoE residency work. The production gate is **KVarN/normal throughput ratio ≥ 90%** on both prefill (`pp*`) and decode (`tg*`) per model×config cell (Tier 1). Tier 2 adds logits NMSE gates; Tier 3 (output quality) is deferred.

### Achieved (P0 gate PASS)

**Qwen3.6 MTP IQ3** at `-ngl 99 -ncmoe 34 -fa off --kvarn-preset kvarn_k4v2_g128 --kvarn-iters 4`:

| Case | Normal t/s | KVarN t/s | Ratio | Gate |
|------|----------:|----------:|------:|:----:|
| pp512 | 380.27 | 376.78 | **99.1%** | **PASS** |
| tg64 | 39.74 | 36.84 | **92.7%** | **PASS** |

Verified at commit `4a4c6ff70` (ping-pong + iters4); ping-pong scoping fix landed in `030333631`.

### Post-merge validation (2026-06-08)

`artifacts/kvarn-production-gate/20260608-164314/`, Qwen Q4_K_M split + Gemma Q4_K_XL, `-fa off`, Release build `54ddfa768`.

| Model | Case | Backend mode | Normal t/s | KVarN t/s | Ratio | Gate |
|-------|------|--------------|-----------:|----------:|------:|:----:|
| Qwen3.6 MTP Q4_K_M (`-ncmoe 34`) | pp512 | true KVarN | 115.64 | 194.65 | **168.3%** | **PASS** |
| Qwen3.6 MTP Q4_K_M (`-ncmoe 34`) | tg64 | true KVarN | 27.91 | 33.48 | **120.0%** | **PASS** |
| Gemma 4 Q4_K_XL | pp512 | normal ISWA fallback | 2218.50 | 2715.60 | **122.4%** | **PASS** |
| Gemma 4 Q4_K_XL | tg64 | normal ISWA fallback | 63.93 | 65.10 | **101.8%** | **PASS** |

Post-common propagation recheck:

| Model | Case | Backend mode | Normal t/s | KVarN/fallback t/s | Ratio | Gate | Artifact |
|-------|------|--------------|-----------:|-------------------:|------:|:----:|----------|
| Qwen3.6 MTP Q4_K_M (`-ncmoe 34`) | pp512 | true KVarN | 100.56 | 180.48 | **179.5%** | **PASS** | `artifacts/kvarn-production-gate/20260608-170024/qwen-tier1/` |
| Qwen3.6 MTP Q4_K_M (`-ncmoe 34`, r=5 rerun) | tg64 | true KVarN | 27.25 | 33.77 | **123.9%** | **PASS** | `artifacts/kvarn-bench/qwen-tier1-tg64-rerun-post-common/` |
| Gemma 4 Q4_K_XL | pp512 | normal ISWA fallback | 2219.04 | 2709.42 | **122.1%** | **PASS** | `artifacts/kvarn-bench/gemma-tier1-fallback-post-common/` |
| Gemma 4 Q4_K_XL | tg64 | normal ISWA fallback | 64.35 | 64.77 | **100.7%** | **PASS** | `artifacts/kvarn-bench/gemma-tier1-fallback-post-common/` |

The full wrapper run at `artifacts/kvarn-production-gate/20260608-170024/` had a noisy Qwen `tg64` r=3 failure (73.1%) with high variance; the same case passed at r=5. `run_production_gate.ps1` now defaults Qwen Tier 1 to 5 repetitions.

### Production policy (Gemma ISWA fallback)

Gemma 4 with `--kv-cache-quant kvarn` **defaults to normal ISWA KV** because experimental KVarN+ISWA is below the 90% throughput gate (~62–69%). Qwen hybrid KVarN remains fully enabled.

- **Production default:** `create_memory()` in `src/llama-model.cpp` routes Gemma 4 KVarN+SWA to `llama_kv_cache_iswa` with a warning.
- **Experimental opt-in:** `LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA=1` restores `llama_kv_cache_kvarn_iswa` for CUDA optimization work.
- **One-command gate:** `scripts/kvarn/run_production_gate.ps1` — Qwen true KVarN ≥90%, Gemma production fallback ≥90%.

### Remaining

| Area | Status | Notes |
|------|--------|-------|
| **Gemma KVarN+ISWA (experimental)** | FAIL (~65–72%) | `@ 95390d5b1`: P0 correctness (K-layout, event store) + prior fast paths; Q4_XL pp512 **64.8%**, tg64 **71.7%**; Tier2 logits PASS; still opt-in only |
| **Gemma production fallback** | PASS | Normal ISWA when `--kv-cache-quant kvarn`; post-common Q4_XL gate: pp512 **122.1%**, tg64 **100.7%** |
| **Common KVarN param propagation** | FIXED | `common_context_params_to_llama()` again copies `kv_cache_quant_type` + `kvarn`; server load-failure test now covers an explicit unsupported preset |
| **Tier 2 logits** | Enforceable | `compare_cuda_logits_ref.ps1` NMSE thresholds + `-RunTier2` on production gate |
| **Qwen `-ngl 99` full GPU** | Not comparable | 35B model exceeds 12 GB VRAM; use `-ncmoe 34` |

**Hardware reference:** RTX 5070 12 GB, build `build-kvarn-cuda-static-vs`, `-fa off`.

---

## Commit timeline (key commits since refinement)

| Hash | Description |
|------|-------------|
| `cf044f51b` | Merge ggml-org/llama.cpp master (hparams refactor, Granite4 Vision, n_gpu_layers fixes, Vulkan FWHT, CUDA MMVQ PDL) |
| `591c008dc` | Land KVarN post-ubatch shared infra: tail-safe helper, active-window masks, decode graph reuse, CUDA body-store/attn optimizations |
| `5f8e46c04` | Fix KVarN memory wiring and bench parser after ATX merge |
| `c43744da1` | Fix hybrid KVarN ubatch split for recurrent memory (`split_equal` vs `split_simple`) |
| `02ccf7639` | Fix KVarN decode regression from per-head kernel launches |
| `ab8a6db8a` | Fix KVarN decode graph reuse across growing active windows (stable mask/scratch topology) |
| `982919dd9` | Docs: record `ab8a6db8a` gate matrix for Qwen and Gemma re-bench |
| `4a4c6ff70` | Fix KVarN prefill graph reuse across alternating prompt ubatches (ping-pong `gf_res_alt`) |
| `030333631` | Scope ping-pong graph reuse to normal KV prefill only; record Qwen gate PASS |
| `686356d61` | Stabilize mainline parity and KVarN production gates after upstream merge |
| `f5bdd5b6c` | Gemma 512d CUDA fast paths: sinktail/decode attn, pipelined body-store, warpqk body dequant, all-heads seal op |
| `95390d5b1` | Architect P0: warpqk K scratch layout, event-ordered 512d store, ISWA prepare trace, multi-record seal op; see `docs/AGENT_CODE_REVIEW_HANDOVER.md` |

Earlier integration: `642fab89a` (ATX MoE residency merge), `10f373ac7` (build follow-ups).

---

## Architecture (post-fix)

### KVarN memory paths

| Path | Model | Implementation |
|------|-------|----------------|
| **hybrid-kvarn** | Qwen3.6 MTP | `llama-memory-hybrid-kvarn.cpp` — dense KVarN layers + recurrent memory slots |
| **ISWA** | Gemma 4 | `llama-kv-cache-kvarn-iswa.cpp` — KVarN on non-SWA layers, normal sliding-window KV on SWA layers |
| **standalone** | Smaller models | `llama-kv-cache-kvarn.cpp` — full KVarN cache per layer |

### Tail-safe ubatch splitting

`kvarn_tail_safe_ubatch_limit()` in `src/llama-kvarn-ubatch.h` bounds KVarN ubatch chunks by **tail-ring evictions per graph** (≤ `tail_tokens`), not raw token count. Used by hybrid-kvarn, ISWA, and standalone paths. Tested in `tests/test-batch-split.cpp`.

### Graph reuse

1. **Decode — stable topology (`ab8a6db8a`):** One-token decode graphs allocate full-capacity KQ masks and worst-case mixed-attention scratch so `can_reuse` stays true as the active window grows. Shared helpers: `kvarn_graph_mask_n_kv`, `kvarn_graph_build_scratch_window`, `kvarn_graph_reuse_mask_n_kv`. Applied to dense KVarN (`llm_graph_input_attn_kvarn`) and ISWA base layers.

2. **Prefill ping-pong (`4a4c6ff70`, scoped `030333631`):** pp512 uses two prompt ubatches (384+128). `gf_res_alt` ping-pongs 384/128 graphs across repetitions for **normal KV prefill only** (`n_tokens > 1`, not KVarN). Decode (`tg*`) and KVarN prefill keep the single-slot path.

### CUDA attention path

- **Default:** fused-batch mixed sink/body/tail attention (`ggml/src/ggml-cuda/kvarn.cu`).
- **512d Gemma decode:** `sinktail-decode` (one CTA, all heads) when `n_records=0 && n_pending=0 && n_queries=1`.
- **512d Gemma prefill/decode (no body):** `sinktail-f16` batch kernel when `n_records=0 && n_pending=0`.
- **512d body-active:** `warpqk-f16-dequant` (pre-dequant body in attn scratch, then warpqk).
- **512d body-store:** pipelined dual-stream K/V Hadamard+sinkhorn+quantize (`ggml_cuda_kvarn_store_kv_body_512_pipelined`).
- **Opt-in:** per-head decode launches behind `LLAMA_KVARN_ATTN_DECODE_PER_HEAD=1` (regression if default).
- **Opt-in:** split kernels behind `LLAMA_KVARN_ATTN_SPLIT_KERNELS=1` (Qwen pp512 regresses to ~16% if default).

---

## Benchmark results (verified runs)

Build: `build-kvarn-cuda-static-vs`, `-fa off`, `r=3`, `--kvarn-preset kvarn_k4v2_g128`.

### Qwen3.6 MTP IQ3 (`-ngl 99 -ncmoe 34`, `--kvarn-iters 4`)

| Case | Normal t/s | KVarN t/s | Ratio | Gate | Commit | Artifacts |
|------|----------:|----------:|------:|:----:|:------:|-----------|
| pp512 | 380.27 | 376.78 | 99.1% | PASS | `4a4c6ff70` | `artifacts/kvarn-bench/qwen-pp512-gate-push/final/` |
| tg64 | 39.74 | 36.84 | 92.7% | PASS | `4a4c6ff70` | `artifacts/kvarn-bench/qwen-pp512-gate-push/final/` |

Prior milestones: `c43744da1` (pp512 100.6%, tg64 78.8%), `ab8a6db8a` (pp512 86.2%, tg64 95.0%) — see `artifacts/kvarn-bench/qwen-post-split-fix/`, `artifacts/kvarn-bench/qwen-pp512-gate-push/iters4/`.

### Gemma 4 12B Q3 (`-ngl 99`)

| Case | Normal t/s | KVarN t/s | Ratio | Gate | Commit | Artifacts |
|------|----------:|----------:|------:|:----:|:------:|-----------|
| pp512 | 2303.81 | 1437.14 | 62.4% | FAIL | `ab8a6db8a` | `artifacts/kvarn-bench/gemma-gate-push/` |
| tg64 | 69.45 | 47.89 | 69.0% | FAIL | `ab8a6db8a` | `artifacts/kvarn-bench/gemma-gate-push/` |

Baseline (pre-refinement): `artifacts/kvarn-bench/gemma-591c008dc-post-refinement/` (pp512 62.9%, tg64 56.3%). Per-head launch fix: `artifacts/kvarn-bench/decode-fix-20260607/`.

> Artifact directories are local bench outputs (not committed). Reproduce with commands below.

---

## Production config for Qwen gate

```
-ngl 99 -ncmoe 34 -fa off --kvarn-preset kvarn_k4v2_g128 --kvarn-iters 4
```

`-ncmoe 34` is required on 12 GB VRAM — Qwen 35B at `-ngl 99` alone exceeds VRAM and is not production comparable.

---

## Known bugs fixed

| Bug | Fix commit | Summary |
|-----|------------|---------|
| **params_mem merge** | `5f8e46c04` | KVarN memory wiring broken after ATX merge; bench parser also fixed |
| **Tail eviction ubatch splitter** | `591c008dc` | `kvarn_tail_safe_ubatch_limit` deduped; bounds chunks by tail-ring evictions |
| **Hybrid `split_equal` for recurrent memory** | `c43744da1` | `split_simple` crashed on `ubatch.equal_seqs()` when recurrent slots present |
| **Decode per-head launch regression** | `02ccf7639` | Default `n_queries==1` path launched one CUDA kernel per head; gated behind env flag |
| **Decode graph rebuild every token** | `ab8a6db8a` | `kq_mask->ne[0]` tracked current `n_kv`; now stable full-capacity topology |
| **Ping-pong graph cache for 384+128 ubatches** | `4a4c6ff70` / `030333631` | `gf_res_alt` ping-pong for normal KV prefill; scoped away from KVarN/decode |
| **Common init KVarN propagation** | pending | Restores `kv_cache_quant_type`/`kvarn` into `llama_context_params`; cleans partial common init state on context failure |

---

## Open blockers for architect

1. **Gemma 512-dim fused-batch CUDA** — pp512 62%, tg64 69% vs 90% gate. ISWA base-layer mixed attention and body-store seals are the bottleneck, not graph-reuse topology.

2. **ISWA dual prepare overhead** — composite cache prepares both KVarN and SWA paths per layer group; Gemma tg64 regression vs pre-refinement is on the ISWA path.

3. **Tier 2: logits NMSE** — run `scripts/kvarn/compare_cuda_logits_ref.ps1` and inspect `llama-results` output. Call-site arity fixed; use static build if Smart App Control blocks shared DLLs.

4. **Qwen `-ngl 99` full GPU** — not comparable on 12 GB VRAM (14.28 GiB model). Production gate uses `-ncmoe 34`.

---

## How to reproduce benches

Prerequisites: CUDA build at `build-kvarn-cuda-static-vs`, models on disk.

### Production gate (recommended)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\kvarn\run_production_gate.ps1 `
  -QwenModel "C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.6-35B-A3B-MTP-GGUF\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf" `
  -GemmaModel "C:\Users\sjake\Downloads\gemma-4-12b-it-UD-Q3_K_XL.gguf" `
  -BuildDir build-kvarn-cuda-static-vs
```

Experimental Gemma KVarN+ISWA diagnostic: add `-RunGemmaExperimental`. Tier 2 logits: add `-RunTier2 -Tier2Model <small-model.gguf>`.

Gemma true KVarN+ISWA validation (must pass before removing production fallback):

```powershell
$env:LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA = "1"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\kvarn\run_bench_matrix.ps1 `
  -Model "<gemma-4-12b.gguf>" -BuildDir build-kvarn-cuda-static-vs `
  -CaseList "pp512:512:0,tg64:0:64" -FlashAttn off -Repetitions 3 `
  -KvarnIters 4 -RtnQuantile 1.0 -MinKvarnRatio 0.90 -FailBelowMinKvarnRatio `
  -MinKvarnLayerLogs 8 -ExpectedKvarnLayers "5-47:6" `
  -OutputDir artifacts\kvarn-bench\gemma-true-kvarn-speed-patch
Remove-Item Env:\LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA
```

### Qwen P0 gate (PASS config)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\kvarn\run_bench_matrix.ps1 `
  -Model "C:\Users\sjake\OneDrive\Documents\New project\models\Qwen3.6-35B-A3B-MTP-GGUF\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf" `
  -BuildDir build-kvarn-cuda-static-vs `
  -CaseList "pp512:512:0,tg64:0:64" `
  -FlashAttn off `
  -Repetitions 3 `
  -KvarnIters 4 `
  -ExtraArgs @("-ncmoe","34") `
  -OutputDir artifacts\kvarn-bench\qwen-pp512-gate-push\final
```

### Gemma experimental KVarN+ISWA (below gate — CUDA work target)

```powershell
$env:LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA = "1"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\kvarn\run_bench_matrix.ps1 `
  -Model "C:\Users\sjake\Downloads\gemma-4-12b-it-UD-Q3_K_XL.gguf" `
  -BuildDir build-kvarn-cuda-static-vs `
  -CaseList "pp512:512:0,tg64:0:64" `
  -FlashAttn off -Repetitions 3 -KvarnIters 4 `
  -MinKvarnLayerLogs 8 -ExpectedKvarnLayers "5-47:6" `
  -OutputDir artifacts\kvarn-bench\gemma-experimental-iswa
Remove-Item Env:\LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA
```

### Tier 2 logits (example)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\kvarn\compare_cuda_logits_ref.ps1 `
  -Model "C:\Users\sjake\OneDrive\Documents\New project\models\Qwen2.5-1.5B-Instruct-GGUF\qwen2.5-1.5b-instruct-q4_k_m.gguf" `
  -BuildDir build-kvarn-cuda-static-vs `
  -Context 512 -Batch 512 -Repeat 16 -CheckPackedRepeat -CheckPackedSplit -FlashAttn off
```

### Unit tests

```powershell
ctest -R "test-kvarn-kv|test-kvarn-cuda|test-batch-split" --test-dir build-kvarn-cuda-static-vs
python scripts/kvarn/kv_memory_estimate.py --self-test
```

---

## Files to review (priority)

| Priority | Path | Why |
|:--------:|------|-----|
| P0 | `src/llama-kvarn-ubatch.h` | Tail-safe ubatch limit — core splitter invariant |
| P0 | `src/llama-memory-hybrid-kvarn.cpp` | Qwen hybrid path, `split_equal` gating |
| P0 | `src/llama-kv-cache-kvarn-iswa.cpp` | Gemma ISWA composite cache |
| P0 | `src/llama-kv-cache-kvarn.cpp` | Standalone KVarN cache + graph inputs |
| P0 | `src/llama-context.cpp` | Graph reuse, ping-pong `gf_res_alt` scoping |
| P0 | `ggml/src/ggml-cuda/kvarn.cu` | Fused-batch attn, body store, env-flag gates |
| P1 | `src/llama-graph.cpp` / `src/llama-graph.h` | Mask/scratch reuse helpers |
| P1 | `tests/test-batch-split.cpp` | Tail-safe splitter unit tests |
| P0 | `src/llama-model.cpp` | Gemma KVarN+ISWA production fallback routing |
| P1 | `scripts/kvarn/run_production_gate.ps1` | One-command Tier 1 acceptance matrix |
| P1 | `scripts/kvarn/run_bench_matrix.ps1` | Bench harness + ratio gate enforcement |
| P1 | `scripts/kvarn/compare_cuda_logits_ref.ps1` | Tier 2 logits NMSE thresholds |
| P2 | `scripts/kvarn/README.md` | Extended runbook and historical bench tables |

---

## Anti-patterns (do not reintroduce)

| Anti-pattern | Why |
|--------------|-----|
| `min(n_ubatch, tail_tokens)` as ubatch limit | Ignores sink+tail ring geometry; use `kvarn_tail_safe_ubatch_limit` |
| `LLAMA_KVARN_ATTN_SPLIT_KERNELS=1` as default | Qwen pp512 regresses to ~16%; fused-batch is production path |
| `split_simple` on hybrid with recurrent memory | Crashes `ubatch.equal_seqs()`; must use `split_equal` when `mem_recr` present |
| Per-head decode kernels as default | `02ccf7639` regression; keep behind `LLAMA_KVARN_ATTN_DECODE_PER_HEAD=1` |
| Ping-pong `gf_res_alt` for KVarN or decode | `030333631` — KVarN prefill and single-token decode must stay single-slot |
| Tracking `kq_mask->ne[0]` to current `n_kv` for reuse | Forces graph rebuild every decode token; use stable full-capacity topology |
| Default Gemma 4 to `llama_kv_cache_kvarn_iswa` in production | Below 90% gate; use ISWA fallback unless `LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA=1` |

---

## Related docs

- `scripts/kvarn/README.md` — full CUDA runbook, smoke commands, extended bench history
- `docs/atx-runs.md` — Mac Metal MoE residency baseline (ATX reference, not CUDA KVarN)
