# KVarN CUDA — Architect handover (round 2, measured)

**Audience:** Architecture / code-review agent (static review + targeted patches).  
**Branch:** `kvarn-atx-integration`  
**Date:** 2026-06-09  
**Prior docs:** [`KVARN_CUDA_HANDOVER.md`](KVARN_CUDA_HANDOVER.md), [`AGENT_CODE_REVIEW_HANDOVER.md`](AGENT_CODE_REVIEW_HANDOVER.md), [`GEMMA_KVARN_FAILURE_DIAGNOSTIC.md`](GEMMA_KVARN_FAILURE_DIAGNOSTIC.md) §10

This doc is the **measured follow-up** to the token-major K scratch patch (`0001` from architect review). Do not re-litigate F1/F3 theory — focus on what benches show and what to try next.

---

## Executive summary

| Track | Status | Action |
|-------|--------|--------|
| **Gemma true KVarN+ISWA** (experimental) | **Mixed** — tg64 **PASS** (101%), pp512 **89.6%** (−0.4pp) | Round 8: scratch `src[9]` buffer after rebind; ping-pong |
| **Gemma production fallback** (default policy) | **PASS** (~111% pp512, ~101% tg64) | Do not flip `llama-model.cpp` until experimental passes |
| **Qwen3.6 MTP 128d** (`-ncmoe 34`) | **Mixed** — tg64 97.1% PASS, pp512 83.0% FAIL | Investigate MoE/memory variance; 512d patch should not touch 128d path |
| **Tier 0 CUDA tests** | **PASS** | — |
| **Tier 2 Gemma logits** | **PASS** | — |
| **Tier 2 Qwen3.6 logits** | **FAIL** repeat @ NMSE 5.29e-05 | MoE nondeterminism / threshold; split path not run to completion |

**Bottom line:** Token-major K scratch + multi-CTA decode are **correct and worth keeping** (Gemma pp512 recovered +168 t/s from post-P0). Neither Gemma nor full production gate is met.

---

## What landed (this round)

Commit (pending push): token-major K dequant scratch, warpqk q-staging, per-head `sinktail-decode`, test scratch sizing.

| ID | Files | Change |
|----|-------|--------|
| R1 | `kvarn.cu`, `kvarn.cuh` | `ggml_cuda_kvarn_dequant_body_n_k_token_major` — packed K read `d*gs+g`, scratch write `g*hd+d` |
| R2 | `kvarn.cu` | warpqk uses token-major K loads + `q_sh[head_dim]`; dispatch calls token-major dequant |
| R3 | `kvarn.cu` | `sinktail-decode`: `grid=n_head`, `block=128`, block-parallel softmax |
| R4 | `test-kvarn-cuda-dequant.cpp` | Token-major dequant test + `kvarn_test_attn_scores_floats()` for 512d scratch tail |

**Not changed:** `llama-model.cpp` policy, cross-ubatch seal batching, KVarN prefill ping-pong.

---

## Model paths (RTX 5070, 12 GB — use these)

| Model | Path | Notes |
|-------|------|-------|
| Gemma experimental | `models\gemma-4-12b-it-GGUF\gemma-4-12b-it-UD-Q4_K_XL.gguf` | `LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA=1` |
| Qwen3.6 MTP regression | `models\Qwen3.6-35B-A3B-MTP-UD-Q4_K_M-GGUF-SPLIT\Qwen3.6-35B-A3B-MTP-UD-Q4_K_M-00001-of-00005.gguf` | `-ncmoe 34`, layers `3-39:4` |
| Qwen3.6 IQ3 (docs stale) | `models\Qwen3.6-35B-A3B-MTP-GGUF\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf` | **Missing on disk**; folder has Q4_K_XL only |

---

## Measured benches (build `ee2a7f31c` + R1–R4, r=5, `--kvarn-iters 4`, rtn=1.0, `-fa off`)

### Gemma experimental (`LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA=1`)

Artifact: `artifacts/kvarn-bench/gemma-token-major-k-scratch-rerun/`

| Case | Normal t/s | KVarN t/s | Ratio | vs post-P0 `95390d5b1` | Gate |
|------|----------:|----------:|------:|------------------------|:----:|
| pp512 | 2465.5 | 1732.8 | **70.3%** | +168 t/s (+10.7%) | FAIL |
| tg64 | 61.6 | 46.1 | **74.9%** | +1.7 t/s (+3.8%) | FAIL |

Trace: pp512 `warpqk-f16-dequant` (`rec2` final ubatch); tg64 `sinktail-decode` (`rec0`).

Historical reference:

| Milestone | pp512 KVarN | pp512 % | tg64 KVarN | tg64 % |
|-----------|------------:|--------:|-----------:|-------:|
| Pre-P0 | 1872 | 70.8% | 48.9 | 73.7% |
| Post-P0 | 1565 | 64.8% | 44.4 | 71.7% |
| **This patch** | **1733** | **70.3%** | **46.1** | **74.9%** |

### Gemma production fallback (default policy, no experimental flag)

Artifact: `artifacts/kvarn-bench/gemma-production-fallback-token-major/`

| Case | Ratio |
|------|------:|
| pp512 | **110.8%** |
| tg64 | **100.9%** |

### Qwen3.6 MTP Q4_K_M (`-ncmoe 34`, 128d hybrid KVarN)

Artifact: `artifacts/kvarn-bench/qwen36-mtp-q4km-token-major-regression/`

| Case | Normal t/s | KVarN t/s | Ratio | Gate |
|------|----------:|----------:|------:|:----:|
| pp512 | 83.5 ± **36.2** | 69.3 ± 7.8 | **83.0%** | FAIL |
| tg64 | 12.58 ± 2.42 | 12.21 ± 0.96 | **97.1%** | PASS |

**Interpretation:** pp512 normal row has extreme variance (likely MoE CPU offload / VRAM fit on 12 GB). tg64 stable and near gate. 512d CUDA changes are gated on `head_dim >= 512`; Qwen pp512 gap may be environmental — **re-run with warmup** and compare to pre-patch baseline on the **same** Q4_K_M split before blaming R1–R4.

Docs that cite **IQ3_XXS** or **99% Qwen pp512** refer to a different artifact / quant — update expectations when using Q4_K_M split.

---

## Tests & logits

```text
Tier 0:  test-kvarn-kv, test-kvarn-cuda-*, test-batch-split  → PASS
Gemma:   compare_cuda_logits_ref repeat/split/scratch         → PASS (experimental ISWA)
Qwen3.6: compare_cuda_logits_ref repeat                       → FAIL (NMSE=5.29e-05, llama-results)
```

---

## Confirmed static findings (still valid)

1. **Multi-record seal op inert at Gemma pp512** — 384+128 ubatches seal one record each; `seal_records.size()>1` never fires.
2. **Cross-ubatch seal deferral** — changes `n_pending` contract; do not half-patch without new tests.
3. **KVarN prefill excluded from ping-pong** — `llama-context.cpp` ~1320–1325; high leverage once seal indices are runtime inputs.

---

## Recommended next patch order (Gemma experimental gate)

1. **Per-layer seal launch reduction** — biggest remaining pp512 cost after coalescing fix (Hadamard + 4× Sinkhorn + quantize × 8 layers per seal wave).
2. **f16 (not f32) body dequant scratch** — halve body memory traffic on warpqk path.
3. **ISWA prepare** — `LLAMA_KVARN_ISWA_PREPARE_TRACE=1`; defer SWA on `n_tokens==1` decode ubatches.
4. **Prefill graph reuse** — safe KVarN ping-pong / `can_reuse` with runtime seal indices.

**Anti-patterns (never default):** `LLAMA_KVARN_ATTN_SPLIT_KERNELS=1`, `LLAMA_KVARN_ATTN_SERIAL_FUSED=1`, `LLAMA_KVARN_ATTN_DECODE_PER_HEAD=1`.

---

## Commands to reproduce

```powershell
cd llama.cpp-kvarn-cuda
cmake --build build-kvarn-cuda-static-vs --config Release -j

ctest --test-dir build-kvarn-cuda-static-vs -C Release `
  -R "test-kvarn-kv|test-kvarn-cuda|test-batch-split" --output-on-failure

$env:LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA = "1"
powershell scripts\kvarn\run_bench_matrix.ps1 `
  -Model "...\gemma-4-12b-it-UD-Q4_K_XL.gguf" `
  -BuildDir build-kvarn-cuda-static-vs `
  -CaseList "pp512:512:0,tg64:0:64" -FlashAttn off -Repetitions 5 `
  -KvarnIters 4 -RtnQuantile 1.0 -MinKvarnRatio 0.90 `
  -MinKvarnLayerLogs 8 -ExpectedKvarnLayers "5-47:6" -TraceAttn -TraceStore `
  -OutputDir artifacts\kvarn-bench\gemma-experimental-rerun
Remove-Item Env:\LLAMA_KVARN_FORCE_EXPERIMENTAL_ISWA

$qwen = "...\Qwen3.6-35B-A3B-MTP-UD-Q4_K_M-GGUF-SPLIT\Qwen3.6-35B-A3B-MTP-UD-Q4_K_M-00001-of-00005.gguf"
powershell scripts\kvarn\run_bench_matrix.ps1 `
  -Model $qwen -BuildDir build-kvarn-cuda-static-vs `
  -CaseList "pp512:512:0,tg64:0:64" -FlashAttn off -Repetitions 5 `
  -KvarnIters 4 -MinKvarnRatio 0.90 -MinKvarnLayerLogs 10 `
  -ExpectedKvarnLayers "3-39:4" -ExtraArgs "-ncmoe","34" `
  -OutputDir artifacts\kvarn-bench\qwen36-mtp-rerun
```

---

## Copy-paste task for next architect agent

> Read `docs/KVARN_ARCHITECT_HANDOVER_ROUND2.md` on `kvarn-atx-integration`. Gemma experimental gate remains open (pp512 70.3%, tg64 74.9%). Token-major K scratch is landed and validated — do not revert without cause. Prioritize seal launch count and f16 dequant scratch for pp512; ISWA prepare + graph reuse for tg64. Qwen3.6 regression: use Q4_K_M split @ `-ncmoe 34`; tg64 passes, pp512 noisy at 83% — confirm not a 512d regression before changing 128d paths. Do not flip `llama-model.cpp` until Gemma experimental ≥90% both cases and Qwen regression stable.

---

## Round 3 patches (architect static pass, 2026-06-09)

Implements the recommended next-patch list above. All changes gated on
`head_dim>=512` or host-side KVarN-only paths; Qwen 128d untouched.

| ID | Files | Change | Targets |
|----|-------|--------|---------|
| R5 | `src/llama-context.cpp` | **KVarN prefill ping-pong** (opt-in: `LLAMA_KVARN_ENABLE_PREFILL_PINGPONG=1`; default off). Enabling on Gemma experimental ISWA aborts pp512: `kvarn_iswa_kqv_out-*` stays on a CUDA0 buffer that cannot run `KVARN_ATTN_MIXED` after the 384+128 slot swap. Scheduler reset on reuse does not rebind ISWA mixed-attn outputs — needs a dedicated fix before default-on. | pp512 |
| R6 | `kvarn.cu`, `kvarn.cuh`, dispatch | **f16 dequant scratch** (item 2): `kvarn_dequant_n_kernel` templated on output type; new `ggml_cuda_kvarn_dequant_body_n_k_token_major_f16`; warpqk body params/loads are `__half`. Halves body read traffic per CTA. f32 token-major export retained for tests. Scratch element budget unchanged (same element count in half the bytes — capacity checks still valid). | pp512 |
| R7 | `kvarn.cu` | **Seal launch trim** (item 1, safe subset): Sinkhorn scale-init folded into the first row/col iteration (`init_scale` flag); the two `kvarn_fill_f32_kernel` launches per pipeline removed (−4 launches/seal across K+V). `iters==0` still fills neutral scales. Deeper fusion (row+col per iter) needs grid-wide sync — deferred. | pp512 |
| R8 | `src/llama-context.cpp` | **Reuse attribution trace**: `LLAMA_KVARN_GRAPH_REUSE_TRACE=1` logs per-ubatch reused/rebuilt + slot. Settles the open tg64 question (decode topology is already stabilized via kv_size mask + worst-case scratch, so reuse *should* hit every token after the first — verify, don't assume). | tg64 |
| R9 | `tests/test-kvarn-cuda-dequant.cpp` | f16 token-major dequant checked against transposed CPU reference, rel-err < 2e-3. | gates |

**Numerics note (R6):** body scratch f32→f16 adds ~1e-3 relative rounding on
top of 4-bit/2-bit body quantization error — expected invisible at Tier 2
NMSE, but logits MUST be re-run (repeat/split/scratch, Gemma) before any
ratio claims.

### Round 3 measured (RTX 5070, Gemma Q4_K_XL, experimental ISWA, r=3, ping-pong **off**)

| Case | Round 2 @ `7ca5ee0cb` | Round 3 R6+R7 @ `f8507c942`+fix | Δ |
|------|----------------------:|----------------------------------:|---|
| pp512 | 70.3 % | **75.2 %** | +4.9 pp (f16 dequant + sinkhorn init fold) |
| tg64 | 74.9 % | 72.0 % | −2.9 pp (within noise; no ping-pong change) |

Tier 0: `test-kvarn-cuda-scratch-ref` PASS with 512d fused-vs-split tol 2e-3 (R6 f16 scratch).
R5 with default ping-pong-on **crashes** every pp512 run — do not enable until ISWA tensor rebind is fixed.

**Validation order for the human/CI run:**
1. ctest Tier 0; Gemma logits repeat/split/scratch.
2. Gemma experimental bench, default env (ping-pong off) → R6+R7 attribution.
3. Optional A/B: `LLAMA_KVARN_ENABLE_PREFILL_PINGPONG=1` only after R5 ISWA fix lands.
4. tg64 with `LLAMA_KVARN_GRAPH_REUSE_TRACE=1` → if any decode token after
   the first logs `reused=0`, capture which check fails in
   `llm_graph_input_attn_kv_iswa::can_reuse` — that becomes the next tg64
   patch. If all reuse and tg64 still <90%, profile ISWA dual prepare
   (`LLAMA_KVARN_ISWA_PREPARE_TRACE=1`) — it is the remaining host suspect.
5. Qwen Q4_K_M split re-run **with warmup** before attributing its pp512 gap
   to anything in this branch (normal-row σ=36.2 says environment first).

**Still deferred:** cross-ubatch seal batching (pending-window contract),
runtime seal indices (op-interface change; only needed if prefill reuse via
R5 proves insufficient), full Sinkhorn iteration fusion (grid sync).

---

## Round 4 (diagnostic safety, 2026-06-10)

| Change | Status |
|--------|--------|
| ISWA ping-pong requires `LLAMA_KVARN_ENABLE_ISWA_PREFILL_PINGPONG_UNSAFE=1` | Landed — generic `ENABLE_PREFILL_PINGPONG=1` no longer crashes Gemma pp512 |
| `can_reuse` miss attribution under `LLAMA_KVARN_GRAPH_REUSE_TRACE=1` | Landed — logs to stderr (llama-bench nulls `LLAMA_LOG_*`) |
| tg64 decode graph reuse | **Confirmed** — trace shows `reused=1` on every token after the first |

Round 4 does **not** improve throughput. Gemma experimental @ r=3: pp512 **76.1%**, tg64 **75.7%**. Gate remains open (≥90%).

---

## Round 5 patches (architect, 2026-06-09) — CUDA-graph-safe decode + rebind fix

### tg64 root cause (new, measured-mechanism)

Round 4 proved decode graph reuse works, so the gap is not rebuilds. Static
trace found it: `GGML_OP_KVARN_ATTN_MIXED` **disables CUDA graph capture for
the entire decode graph** (`ggml-cuda.cu` ~3867), and even without that hard
disable, `ggml_cuda_graph_update_required` memcmps the full node struct
**including op_params** — KVarN rewrites op_params every token
(`kvarn_graph_update_mixed_attn_params`), which would force per-token
re-capture. Net effect: every decode token pays raw launch overhead on all
~48 layers (~600+ launches × ~3-5 µs ≈ 2-4 ms/token) while the normal path
replays a captured graph. That matches the ~3.5 ms/token tg64 gap.

### R10 — device-side window indirection (the fix)

Mainline-proven pattern: dynamic values go through **input tensors**, never
op_params, so node properties stay bit-stable and the captured graph replays.

| Piece | File | Change |
|---|---|---|
| `GGML_MAX_SRC` 11→12 | `ggml/include/ggml.h` | room for src[11] |
| `ggml_kvarn_attn_mixed_set_window()` | `ggml.h/.c` | attach I32[≥5] window tensor as src[11] |
| Frozen op_params + shared `kvarn_iswa_window` input | `llama-graph.{h,cpp}` (ISWA builder) | in the pure sink/tail decode regime (`n_tokens==1`, `records==0`, `pending==0`, no seal ops) the op bakes worst-case caps (`sink_tokens`/`tail_tokens`, `tail_start=0`) and all KVarN layers share one I32[8] input carrying the live window |
| `set_input` | `llama-graph.cpp` | window-indirect graphs stream `[n_sink, n_records, n_pending, n_tail, tail_start]` into the input tensor; op_params are **never** rewritten (that's what kept invalidating capture) |
| `can_reuse` regime guard | `llama-graph.cpp` | a window-indirect graph is never reused once the live window leaves the sink/tail regime (records/pending > 0) — rebuild on regime transitions |
| CUDA glue | `ggml-cuda.cu` | capture disable now only when `src[11]==NULL`; `supports_op` validates src[11] I32[≥5]; dispatch gets `window_dev` |
| Kernel | `kvarn.cu` | `sinktail-decode` reads `n_sink/n_tail/tail_start` from device memory when `window_dev` set; host args carry frozen caps for grid/shmem sizing (runtime ≤ caps by construction) |

Scope: covers the tg64 gate regime exactly (decode from pos 0–255 stays
records=0/pending=0). Long-context decode (pending>0) falls back to today's
non-captured behavior — no regression, follow-up below.

### R11 — ping-pong rebind fix (unblocks R5)

Root cause of the `kvarn_iswa_kqv_out` abort: re-allocating a **cached** graph
on a scheduler that has since allocated a different topology leaves stale
`buffer/data` on its intermediates; `ggml_backend_sched_backend_id_from_cur`
then treats them as **pre-allocated** and aborts when the (stale) buffer/op
combination fails `supports_op`. Fix: `llm_graph_result::prepare_rebind()`
clears buffer/data on every tensor owned by the result's compute context;
called in the rebind path right after `sched_reset`. Views of external cache
tensors are re-initialized from `view_src` by the allocator; weights/KV
tensors live in other contexts and are untouched. The
`..._ISWA_PREFILL_PINGPONG_UNSAFE` gate is dropped (env still parsed);
ping-pong remains opt-in via `LLAMA_KVARN_ENABLE_PREFILL_PINGPONG=1` until
benched, then default-on.

### Validation order

1. ctest Tier 0 + Gemma logits repeat/split/scratch (R10 must be
   numerics-neutral: same kernel, values read from device instead of args).
2. tg64 with `GGML_CUDA_DISABLE_GRAPHS` **unset**: expect a one-time capture
   then pure replay; `LLAMA_KVARN_GRAPH_REUSE_TRACE=1` should still show
   reused=1. Compare tg64 vs round-4 75.7% — this is the patch under test.
3. tg64 with `GGML_CUDA_DISABLE_GRAPHS=1` to isolate R10's contribution.
4. pp512 with `LLAMA_KVARN_ENABLE_PREFILL_PINGPONG=1` (crash repro first on
   tiny run): if the abort is gone, measure; if still crashing, capture the
   abort tensor name — next suspect is input tensors flagged
   GGML_TENSOR_FLAG_INPUT needing their flag-driven backend pinning re-run.
5. Qwen Q4_K_M regression (window indirection is ISWA-builder-only; Qwen
   path untouched).

### Designed-but-not-landed (next pp512 patch, in order)

1. **Q-tiled warpqk** (QT=4): grid `ceil(n_queries/4)×n_head`, stage 4 q rows
   in shmem (`4*head_dim + 4*n_tokens + block` floats ≈ 20.5 KB at pp512 —
   fits), compute 4 score rows per K element and 4 AV outputs per V element.
   Cuts body/sink-tail read traffic 4× and quadruples FMA reuse — the warpqk
   path is scalar with zero cross-query reuse today, vs cuBLAS GEMM on the
   normal path; this is the dominant remaining kernel-side pp512 gap.
2. Extend R10 window indirection to the pending>0 decode regime (needs the
   dequant pre-pass record count host-side: bake worst-case records and make
   the dequant grid self-limiting from window_dev).
3. Sinkhorn row+col fusion via cooperative launch (only if seals still show).

### Round 5 measured (RTX 5070, Gemma Q4_XL, experimental ISWA, r=5, iters=4)

| Case | Round 4 | Round 5 @ `caba1918a`+refine | Gate (≥90%) |
|------|--------:|-----------------------------:|:-----------:|
| **tg64** | 75.7% | **100.9%** (63.1 / 62.6 t/s) | **PASS** |
| **pp512** | 76.1%* | 71.0% (1780 / 2505 t/s) | FAIL |

\*Round 4 used iters=16; gate uses iters=4. KVarN abs pp512 **improved** (+95 t/s vs
round-4 iters=16 run); ratio dipped because normal KV also got faster.

**R10 validated:** tg64 with `GGML_CUDA_DISABLE_GRAPHS=1` still ~102% — window
indirection is numerics-neutral; the win is CUDA graph replay on the default path.

**R11 partial:** `prepare_rebind()` removes the old `kvarn_iswa_kqv_out` abort, but
pp512 ping-pong still asserts in `ggml-backend.cpp` (`cur_backend_id != -1`) on the
128-token slot. Keep ping-pong **opt-in**; next fix: re-pin `GGML_TENSOR_FLAG_INPUT`
tensors after `prepare_rebind()`.

**Production fallback** (default policy, no `FORCE_EXPERIMENTAL_ISWA`): pp512
**114%**, tg64 **101%** — unchanged PASS.

**Do not flip `llama-model.cpp` yet** — experimental pp512 still ~19pp short of gate.

### Round 6 (next agent — pp512 gate)

1. **Q-tiled warpqk (QT=4)** — primary pp512 kernel gap (spec in §Designed-but-not-landed).
2. **Ping-pong rebind part 2** — INPUT tensor backend pinning after slot swap.
3. Qwen Q4_K_M split regression with `-ncmoe 34` + `-Warmup`.

---

## Round 6 patches (architect, 2026-06-10) — Q-tiled warpqk + rebind validation

### R12 — Q-tiled warpqk (QT=4), the pp512 kernel patch

`kvarn.cu`: `kvarn_attn_mixed_f16_fused_batch_warpqk_kernel` is now
`template<int QT>`. Each CTA computes **4 query rows for one head**: the query
tile is staged in shared memory (rows past the tile edge zero-filled so the
unrolled accumulation needs no bounds checks), every K element fans into 4
score accumulators, per-row block softmax (rows past the edge zeroed so AV
runs unguarded), and every V element fans into 4 output accumulators held in
registers. Effect: 4× fewer CTAs, 4× K/V read amortization, 4× FMA reuse —
attacking the scalar kernel's zero cross-query reuse, the dominant remaining
kernel gap vs the normal path's cuBLAS GEMM attention.

Dispatch selects QT=4 when `n_queries > 1` and the tiled shared layout
(`4*head_dim + 4*n_tokens + block + pad` floats ≈ 20.5 KB at pp512) fits;
otherwise the QT=1 instantiation runs, which is behavior-identical to the
previous scalar kernel (decode shapes and shmem-constrained windows are
untouched). Trace label unchanged (`warpqk-f16` / `warpqk-f16-dequant`).

### R13 — rebind part 2: validate instead of abort

Round 5 left ping-pong asserting in `ggml-backend.cpp`
(`GGML_ASSERT(cur_backend_id != -1)`) on the 128-token slot — i.e. after
`prepare_rebind()`, some node had **no backend willing to run it** at
re-split. `llama-context.cpp` now performs the rebind work inside
`can_reuse_graph`: when a candidate would need a slot rebind it (a) runs
`prepare_rebind()` and (b) walks every non-view node checking
`ggml_backend_supports_op` across all backends. Any unsupported node →
candidate rejected, normal rebuild path taken (no crash), and with
`LLAMA_KVARN_GRAPH_REUSE_TRACE=1` the **exact node name + op** is printed.
This makes pp512 ping-pong safe to enable for measurement and converts the
remaining mystery into a one-line trace for round 7 (the supports_op check
that fails for that node is then the targeted fix).

### Validation order (human/CI)

1. ctest Tier 0; Gemma logits repeat/split/scratch — R12 reorders FP
   accumulation across queries (same per-row math); expect NMSE noise-level.
2. Gemma experimental pp512 + tg64 (default env). pp512 vs 71.0% is the R12
   measurement. tg64 must hold ≥100% (QT=1 path untouched; verify).
3. pp512 with `LLAMA_KVARN_ENABLE_PREFILL_PINGPONG=1` +
   `LLAMA_KVARN_GRAPH_REUSE_TRACE=1`: either it now reuses both slots
   (measure the delta) or the trace names the unsupported node — paste that
   into round 7.
4. Qwen3.6 Q4_K_M split, `-ncmoe 34`, **with warmup**, pp512+tg64: R12 is
   inside the `head_dim>=512` branch; Qwen must be unchanged within noise.
5. If pp512 still <90% after R12: next levers are QT=8 with f16 q_sh
   (shared-memory budget doubles to ~41 KB — needs the >48 KB opt-in check
   in `kvarn_cuda_dynamic_shmem_fits`), then seal launch fusion, then
   extending R10 window indirection to the pending>0 regime.

### Round 6 measured (RTX 5070, Gemma Q4_XL, experimental ISWA, r=5, iters=4)

| Case | Round 5 | Round 6 @ `0f66be14f` | Gate (≥90%) |
|------|--------:|----------------------:|:-----------:|
| **tg64** | 100.9% | **101.0%** (61.7 / 61.1 t/s) | **PASS** |
| **pp512** | 71.0% | **84.9%** (2135 / 2517 t/s) | FAIL (−5.1 pp) |

Tier 0 + Gemma logits: **PASS**. R12 delivered **+356 KVarN t/s** on pp512 (+13.9 pp
ratio) with tg64 held.

**R13 trace (ping-pong on):** no crash; every 128-token slot logs
`graph rebind rejected: node 'kvarn_iswa_kqv_out-5' (op KVARN_ATTN_MIXED)
unsupported by every backend after rebind; rebuilding instead` — ping-pong
falls back to full rebuild (pp512 **88.6%** with ping-pong vs 84.9% off; delta
within noise until rebind actually sticks).

**Do not flip `llama-model.cpp`** — pp512 still ~5pp short of gate.

### Round 7 (next agent)

1. Fix `kvarn_iswa_kqv_out-*` `supports_op` / backend assignment after
   `prepare_rebind()` so ping-pong slot swap reuses (trace already names the node).
2. If still <90%: QT=8 f16 q-staging, seal fusion, or host prefill graph reuse.
3. Qwen Q4_K_M split `-ncmoe 34` warmup regression (not run this pass).

---

## Round 7 patches (architect, 2026-06-10) — QT=8 + rebind param re-sync

### R14 — QT=8 warpqk tier
QT=4 delivered +356 t/s on exactly the predicted mechanism, so the same lever
is turned again: dispatch now prefers QT=8 when `n_queries > 4` and the tiled
layout fits (`8*head_dim + 8*n_tokens + block + pad` ≈ 39 KB at pp512 — under
the 48 KB ceiling, no opt-in attribute needed), falling back QT=8→4→1. Halves
CTA count again and doubles K/V amortization and FMA reuse on both pp512
ubatch shapes. Decode (n_queries==1) remains QT=1, bit-identical.

### R15 — rebind validation: param re-sync + numeric failure dump
Round 6 trace localized the ping-pong blocker: only the **128-token slot**
fails, on `kvarn_iswa_kqv_out-5` / `KVARN_ATTN_MIXED`. Every supports_op
condition for that op is `op_params` vs baked-shape arithmetic — and a cached
graph's op_params reflect its **last set_input**, which is not guaranteed to
be the ubatch it is about to be reused for (e.g. a failed can_reuse probe of
the other slot earlier in the rep leaves no trace, but warmup/eval-order
effects can). Fix: `llm_graph_result::refresh_kvarn_params(ubatch)` re-syncs
mixed-attn op_params with the **current** ubatch's window before the
supports_op sweep (idempotent with the set_input that follows; window-indirect
decode graphs are explicitly skipped to preserve CUDA-graph param freezing).
If the 128 slot still rejects, the trace now dumps op_params[0..8] and every
src tensor's type/ne/buffer, making the failing conditional directly
computable from the log — no more guessing.

### Validation order
1. ctest + Gemma logits (QT=8 reorders accumulation again — noise-level only).
2. Gemma experimental pp512 (R14 measurement vs 84.9%) + tg64 hold (≥100%).
3. pp512 with `LLAMA_KVARN_ENABLE_PREFILL_PINGPONG=1` + `REUSE_TRACE=1`:
   if R15's re-sync fixed the 128 slot, both slots reuse — measure; if not,
   paste the numeric dump into round 8.
4. Qwen3.6 Q4_K_M split `-ncmoe 34` with warmup (QT change is head_dim>=512
   only).

### Round 7 measured (RTX 5070, Gemma Q4_XL, experimental ISWA, r=5, iters=4)

| Case | Round 6 | Round 7 @ `d6b3be186` | Gate (≥90%) |
|------|--------:|----------------------:|:-----------:|
| **tg64** | 101.0% | **101.1%** (63.7 / 63.0 t/s) | **PASS** |
| **pp512** | 84.9% | **89.6%** (2228 / 2486 t/s) | **−0.4 pp** (~10 t/s short) |

Tier 0 + Gemma logits: **PASS**. R14 QT=8: **+92 KVarN t/s** pp512 (+4.7 pp).

**R15 ping-pong:** re-sync did not fix 128-slot rebind. Numeric dump shows
`src[9] kvarn_iswa_attn_scores buffer=(null)` after `prepare_rebind()` while
KV tensors retain CUDA0 — `supports_op` fails on null scratch buffer, not stale
op_params alone. Round 8: preserve or re-allocate scratch sizing across slot swap.

**Gate:** pp512 needs ~10 more KVarN t/s at this normal baseline (or ping-pong
reuse once rebind works). **Do not flip `llama-model.cpp` yet** (strict ≥90%).

### Round 8 (next agent)

1. Fix `kvarn_iswa_attn_scores` / mixed-attn scratch (`src[9]`) backend assignment
   after `prepare_rebind()` on ping-pong slot swap (dump proves `buffer=null`).
2. Re-run pp512 with ping-pong — should stack host graph-reuse gain on R14.
3. Qwen Q4_K_M warmup regression.
