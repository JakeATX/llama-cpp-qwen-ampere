# KVarN production patch handoff for coding agent

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
