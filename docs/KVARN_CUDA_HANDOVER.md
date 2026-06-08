# KVarN CUDA Architect Handover

**Audience:** External architect reviewing the CUDA KVarN vs normal KV throughput gap.  
**Last updated:** 2026-06-07  
**Branch:** `kvarn-atx-integration`  
**HEAD:** `030333631ab364069b7f743e8f2988090c1d7f6d`  
**Remote:** [https://github.com/JakeATX/llama.cpp](https://github.com/JakeATX/llama.cpp) (`jakeatx` / `fork`)  
**Review URL:** [https://github.com/JakeATX/llama.cpp/tree/kvarn-atx-integration](https://github.com/JakeATX/llama.cpp/tree/kvarn-atx-integration)

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

### Remaining

| Area | Status | Notes |
|------|--------|-------|
| **Gemma 4 12B Q3 (ISWA)** | FAIL (~62–69%) | 512-dim fused-batch CUDA decode + ISWA dual-prepare overhead |
| **Tier 2 logits** | Pending | `llama-results` / `compare_cuda_logits_ref.ps1` NMSE gates |
| **Qwen `-ngl 99` full GPU** | Not comparable | 35B model exceeds 12 GB VRAM; use `-ncmoe 34` |

**Hardware reference:** RTX 5070 12 GB, build `build-kvarn-cuda-static-vs`, `-fa off`.

---

## Commit timeline (key commits since refinement)

| Hash | Description |
|------|-------------|
| `591c008dc` | Land KVarN post-ubatch shared infra: tail-safe helper, active-window masks, decode graph reuse, CUDA body-store/attn optimizations |
| `5f8e46c04` | Fix KVarN memory wiring and bench parser after ATX merge |
| `c43744da1` | Fix hybrid KVarN ubatch split for recurrent memory (`split_equal` vs `split_simple`) |
| `02ccf7639` | Fix KVarN decode regression from per-head kernel launches |
| `ab8a6db8a` | Fix KVarN decode graph reuse across growing active windows (stable mask/scratch topology) |
| `982919dd9` | Docs: record `ab8a6db8a` gate matrix for Qwen and Gemma re-bench |
| `4a4c6ff70` | Fix KVarN prefill graph reuse across alternating prompt ubatches (ping-pong `gf_res_alt`) |
| `030333631` | Scope ping-pong graph reuse to normal KV prefill only; record Qwen gate PASS |

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

---

## Open blockers for architect

1. **Gemma 512-dim fused-batch CUDA** — pp512 62%, tg64 69% vs 90% gate. ISWA base-layer mixed attention and body-store seals are the bottleneck, not graph-reuse topology.

2. **ISWA dual prepare overhead** — composite cache prepares both KVarN and SWA paths per layer group; Gemma tg64 regression vs pre-refinement is on the ISWA path.

3. **Tier 2: logits NMSE** — run `scripts/kvarn/compare_cuda_logits_ref.ps1` and inspect `llama-results` output. Call-site arity fixed; use static build if Smart App Control blocks shared DLLs.

4. **Qwen `-ngl 99` full GPU** — not comparable on 12 GB VRAM (14.28 GiB model). Production gate uses `-ncmoe 34`.

---

## How to reproduce benches

Prerequisites: CUDA build at `build-kvarn-cuda-static-vs`, models on disk.

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

### Gemma P0 gate (FAIL baseline — architect review target)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\kvarn\run_bench_matrix.ps1 `
  -Model "C:\Users\sjake\Downloads\gemma-4-12b-it-UD-Q3_K_XL.gguf" `
  -BuildDir build-kvarn-cuda-static-vs `
  -CaseList "pp512:512:0,tg64:0:64" `
  -FlashAttn off `
  -Repetitions 3 `
  -MinKvarnLayerLogs 8 `
  -ExpectedKvarnLayers "5-47:6" `
  -OutputDir artifacts\kvarn-bench\gemma-gate-push
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
| P1 | `scripts/kvarn/run_bench_matrix.ps1` | Bench harness + artifact layout |
| P1 | `scripts/kvarn/compare_cuda_logits_ref.ps1` | Tier 2 logits NMSE |
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

---

## Related docs

- `scripts/kvarn/README.md` — full CUDA runbook, smoke commands, extended bench history
- `docs/atx-runs.md` — Mac Metal MoE residency baseline (ATX reference, not CUDA KVarN)
