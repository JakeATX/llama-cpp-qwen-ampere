# KVarN Line-By-Line Audit

Date: 2026-06-15

Branch audited: `kvarn-atx-integration`

Audited head: `9286f361378fc84ec97f8102f25d280fcc43caaf`

Upstream comparison base: `upstream-ggml/master` at `e36a602ba38a26206c749ba4fb5dcf481bfd92db`

Diff command:

```powershell
git diff --stat upstream-ggml/master...HEAD -- ggml/src/ggml-cuda src scripts/kvarn tests docs/KVARN_PRODUCTION_PATCH_HANDOFF.md
git diff --name-status upstream-ggml/master...HEAD -- ggml/src/ggml-cuda src scripts/kvarn tests docs/KVARN_PRODUCTION_PATCH_HANDOFF.md
```

Scope covered: 75 KVarN/runtime/test/harness files, `30498` inserted lines and `258` deleted lines versus upstream in the scoped paths.

This audit is a static, line-by-line review of the KVarN branch delta, split by ownership area and then reconciled against local source reads. It is not a replacement for the GPU correctness gates. It identifies source defects, validation blind spots, and false positives from earlier reviews that should not drive patches.

## Coverage

CUDA/backend:

- `ggml/src/ggml-cuda/CMakeLists.txt`
- `ggml/src/ggml-cuda/fattn.cu`
- `ggml/src/ggml-cuda/gated_delta_net.cu`
- `ggml/src/ggml-cuda/ggml-cuda.cu`
- `ggml/src/ggml-cuda/kvarn.cu`
- `ggml/src/ggml-cuda/kvarn.cuh`
- `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_1-ncols2_16.cu`
- `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_2-ncols2_16.cu`
- `ggml/src/ggml-cuda/template-instances/fattn-mma-f16-instance-ncols1_4-ncols2_16.cu`
- `ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq640-dv512.cu`
- `ggml/src/ggml-cuda/vendors/hip.h`

Runtime/model/graph:

- `src/CMakeLists.txt`
- `src/llama-arch.cpp`
- `src/llama-arch.h`
- `src/llama-batch.cpp`
- `src/llama-context.cpp`
- `src/llama-context.h`
- `src/llama-cparams.h`
- `src/llama-graph.cpp`
- `src/llama-graph.h`
- `src/llama-hparams.h`
- `src/llama-kv-cache-kvarn-iswa.cpp`
- `src/llama-kv-cache-kvarn-iswa.h`
- `src/llama-kv-cache-kvarn.cpp`
- `src/llama-kv-cache-kvarn.h`
- `src/llama-kvarn-ubatch.h`
- `src/llama-memory-hybrid-kvarn.cpp`
- `src/llama-memory-hybrid-kvarn.h`
- `src/llama-memory-recurrent.cpp`
- `src/llama-memory.h`
- `src/llama-model.cpp`
- `src/llama-mtp.h`
- `src/llama.cpp`
- `src/models/delta-net-base.cpp`
- `src/models/models.h`
- `src/models/qwen35.cpp`
- `src/models/qwen35_mtp.cpp`
- `src/models/qwen35moe.cpp`
- `src/models/qwen35moe_mtp.cpp`

Harness/tests/docs:

- `scripts/kvarn/*`
- `tests/test-batch-split.cpp`
- `tests/test-kvarn-cuda-dequant.cpp`
- `tests/test-kvarn-cuda-kernels.cpp`
- `tests/test-kvarn-cuda-mixed-tail.cpp`
- `tests/test-kvarn-kv.cpp`
- `tests/test-kvarn-server-load-failure.cpp`
- KVarN-related changes in `tests/CMakeLists.txt`, `tests/test-arg-parser.cpp`, `tests/test-llama-archs.cpp`, and `tests/test-save-load-state.cpp`
- `docs/KVARN_PRODUCTION_PATCH_HANDOFF.md`

## Confirmed Findings

### P0: default non-paper-frame KVarN mixes incompatible frames once body records are active

The body store rotates K/V into the KVarN Hadamard frame before quantization, while mixed attention consumes reconstructed body K/V together with sink/tail tokens. In default mode, `LLAMA_KVARN_ENABLE_PAPER_FRAME` is off, so graph-side Q, sink/tail, and output transforms are not applied.

Relevant lines:

- `ggml/src/ggml-cuda/kvarn.cu`: body store applies K and V Hadamard transforms inside `ggml_cuda_kvarn_store_kv_body_pipelined()`.
- `ggml/src/ggml-cuda/kvarn.cu`: mixed attention reconstructs body K/V and accumulates output directly.
- `src/llama-graph.cpp`: graph-side paper frame is gated by `LLAMA_KVARN_ENABLE_PAPER_FRAME`.
- `ggml/src/ggml-cuda/kvarn.cu`: CUDA pending-store only treats pending inputs as already rotated when `LLAMA_KVARN_ENABLE_PAPER_FRAME` is set.

Impact:

- Any quality or performance measurement without `LLAMA_KVARN_ENABLE_PAPER_FRAME=1` is not measuring the paper-frame KVarN contract.
- This explains why non-paper-frame long-context data can be badly wrong and should not be used as evidence for or against KVarN.

Required fix:

- Production true KVarN must either enable paper-frame by default after the remaining gates pass, or remove/disable the default mixed-frame path for model/layer combinations where body records can be active.
- Harnesses must record and report the paper-frame flag in summaries.

### P1: multi-record pending body-store duplicates one staged pending tile into multiple records

`ggml_cuda_kvarn_store_body_pending_records_minmax()` gathers one pending K tile and one pending V tile before the record loop, then writes that same tile to every record in `records[]`.

Relevant lines:

- `ggml/src/ggml-cuda/kvarn.cu`: `kvarn_transpose_pending_k_head_kernel()` correctly builds one channel-major K tile from `pending_k`.
- `ggml/src/ggml-cuda/kvarn.cu`: `kvarn_gather_pending_v_head_kernel()` correctly builds one token-major V tile from `pending_v`.
- `ggml/src/ggml-cuda/kvarn.cu`: `ggml_cuda_kvarn_store_body_pending_records_minmax()` stages once, then loops `for (uint32_t bi = 0; bi < n_record_batch; ++bi)`.
- `ggml/src/ggml-cuda/ggml-cuda.cu`: dispatcher passes explicit `record_0..record_3` to that function.
- `src/llama-kv-cache-kvarn.cpp`: `store_kv_body_records_from_pending()` accepts multiple records.

Impact:

- If this path runs with `n_record_batch > 1`, long-context body cache content is corrupted because distinct records receive duplicate K/V.
- This can affect Gemma/ISWA or diagnostics that disable direct prefill store. It may not explain Qwen direct-prefill K8V8 results if that run used direct record store, but it is a real cache correctness bug.

Required fix:

- Either remove the multi-record pending path and emit one pending-record store op per record, or make `ggml_cuda_kvarn_store_body_pending_records_minmax()` offset `pending_k` and `pending_v` per body offset before staging each record.
- Add an independent CUDA regression that creates two pending records with different data and verifies the two packed body records differ and dequantize to their independent references.

### P1: diagnostic layer filtering allocates a subset but graph routing still assumes every KVarN layer exists

`LLAMA_KVARN_LAYER_FILTER` is parsed in memory creation, but graph code still routes scheduled KVarN layers through KVarN unconditionally and calls `get_layer_view(il)`.

Relevant lines:

- `src/llama-model.cpp`: parses `LLAMA_KVARN_LAYER_FILTER`.
- `src/llama-kv-cache-kvarn.cpp`: `layer_storage_index()` throws if a layer was not allocated.
- `src/llama-graph.cpp`: KVarN graph path calls `get_layer_view(il)` before any normal-KV fallback.

Impact:

- The layer-by-layer PPL bisection requested for Qwen `3-39:4` cannot run reliably.
- This is a tooling blocker for finding the remaining post-mixed-attention bug. A filter like `LLAMA_KVARN_LAYER_FILTER=3` can allocate layer 3, then crash on the next scheduled layer.

Required fix:

- Add a diagnostic-only mixed memory mode: included layers use KVarN; excluded full-attention layers use normal KV; recurrent/Mamba/linear-attention layers remain unchanged.
- The fix must cover both hybrid Qwen trunk graphs and standalone/MTP attention paths.
- Graph reuse should be disabled or keyed separately when the layer filter changes graph topology.

### P1: KVarN advertises cache shifting even though shift/update is not implemented

`llama_kv_cache_kvarn::get_can_shift()` returns true for normal RoPE, but `init_update()` refuses shifted cells with `FAILED_COMPUTE`.

Relevant lines:

- `src/llama-kv-cache-kvarn.cpp`: `init_update()` returns failed compute when shifted cells exist.
- `src/llama-kv-cache-kvarn.cpp`: `get_can_shift()` returns true for non-MRoPE/non-IMRoPE.
- `src/llama-kv-cache-kvarn.cpp`: `seq_add()` records shifts.
- `src/llama-context.cpp`: `memory_update(false)` logs the failure and returns false.

Impact:

- Any generation path that reaches context shifting can silently leave KVarN memory in a stale/unshifted state after the error path.
- Long-context decode tests must explicitly avoid context shifting until this is fixed.

Required fix:

- Return false from `get_can_shift()` until the KVarN shift/update graph is implemented, or hard-fail before decode continues if shifting is requested.

### P1: full state save/load is unimplemented for true KVarN memory

The KVarN state methods throw unconditionally for full state operations, while wrappers call into them.

Relevant lines:

- `src/llama-kv-cache-kvarn.cpp`: full state save/load paths throw.
- `src/llama-kv-cache-kvarn-iswa.cpp`: ISWA wrapper calls the KVarN state path for non-partial saves.
- `src/llama-memory-hybrid-kvarn.cpp`: hybrid wrapper calls the KVarN state path for non-partial saves.

Impact:

- Session save/restore is not production quality for true KVarN.
- This is separate from the ctx4096 PPL bug, but it is a production-readiness blocker.

Required fix:

- Disable full state operations explicitly with a clear diagnostic for KVarN, or implement serialization for sink/tail/body/scales/pending plus runtime metadata.

### P1: production and correctness harnesses can pass without proving the intended active KVarN path

Several validation scripts under-enforce their stated contracts.

Findings:

- `scripts/kvarn/run_production_gate.ps1` builds only `llama-bench` and `llama-results` before running `ctest`, so it can run stale or absent test binaries.
- `scripts/kvarn/run_production_gate.ps1` tier-1 parity calls do not require active body records.
- `scripts/kvarn/run_mainline_parity_matrix.ps1` defaults `MinKvarnBodyRecords = 0`.
- `scripts/kvarn/compare_cuda_logits_ref.ps1 -CheckNormalBaseline` builds normal baseline args before appending `-Batch` and `-ExtraArgs`, so normal-KV and KVarN can be compared under different runtime shapes.
- `Assert-ExpectedKvarnLayers` in multiple scripts enforces only that expected layers are present. It does not fail extra KVarN layers, despite docs saying exact routing.
- `scripts/kvarn/run_cuda_smoke.ps1` accepts "graph backend is not wired yet" as a valid guard string, which is useful for unsupported smoke but too weak for supported-path smoke.

Impact:

- The branch can report green validation while not exercising packed body records, not proving exact layer routing, or running stale test binaries.

Required fix:

- Build all relevant test targets before `ctest`.
- Require `MinKvarnBodyRecords > 0` for any production KVarN quality or parity gate.
- Add strict exact-layer mode that fails on extra KVarN layers.
- Fix normal-baseline argument construction in `compare_cuda_logits_ref.ps1`.
- Split supported CUDA smoke from unsupported-mode smoke.

### P2: `seq_rm()` accepts arbitrary removals while KVarN attention assumes a dense prefix

`seq_rm()` removes metadata for arbitrary ranges, but KVarN graph windowing and masks are derived from the current position and assume dense visible history.

Impact:

- Middle-range removals can leave stale sink/body/tail tensor content visible to attention.

Required fix:

- If KVarN only supports suffix rollback, return false for non-suffix removals and document it.
- Otherwise implement tensor compaction or visibility masks that match arbitrary removals.

### P2: no-alloc memory fitting under-reports KVarN memory

When `hparams.no_alloc` is set, the KVarN constructor creates a dummy zero-size buffer, and `memory_breakdown()` later sums backend buffer sizes.

Impact:

- Fit/probing paths can underestimate KVarN memory.
- This can cause runtime behavior differences between fit probes and real execution.

Required fix:

- Mirror upstream no-alloc sizing behavior with `ggml_backend_alloc_ctx_tensors_from_buft_size()`.
- Include pending and scratch tensors in estimates.

### P2: standalone Qwen MTP graph/memory routing has type and scaling hazards

The standalone `qwen35_mtp` / `qwen35moe_mtp` graph paths call plain KV attention builders, while architecture classification can route them through hybrid memory creation.

Additional static risks:

- The MTP hook in `llama-context.cpp` uses `t_h_pre_norm`, but model graphs populate `t_h_nextn`.
- Standalone MTP graph files omit some tensor-scale uses that the in-model Qwen MTP graph includes.

Impact:

- This may not be the same path used by the current production Qwen3.6 MTP gate, but it is a real branch-delta correctness risk.
- It must be resolved before declaring Qwen MTP support generally production-ready.

Required fix:

- Add a targeted MTP routing test that loads the standalone MTP arch path and asserts memory context type, layer id, and output-scale use.
- Align standalone MTP graph scaling with the in-model Qwen MTP graph.

## Disputed Or False-Positive Findings

### Direct prefill body-store raw view in paper-frame mode is not confirmed as a bug

A static review initially flagged that paper-frame direct body-store uses views of `k_cur` and `v_cur`, not `k_st_src` and `v_st_src`.

Local verification:

- `src/llama-graph.cpp` direct prefill store passes raw `k_cur`/`v_cur` tiles.
- `ggml/src/ggml-cuda/ggml-cuda.cu` dispatches direct-record stores through `ggml_cuda_kvarn_store_body_direct_records_minmax()` with `src_layout == 1`.
- `ggml/src/ggml-cuda/kvarn.cu` direct-record store transposes/gathers each raw tile and then calls `ggml_cuda_kvarn_store_kv_body_pipelined(..., input_already_rotated=false, ...)`.
- `input_already_rotated=false` makes CUDA apply the K/V Hadamard before normalization and quantization.

Conclusion:

- Passing raw direct-store views is expected for this path. It does not by itself prove a frame bug.
- The real paper-frame direct-store risk is whether all direct and pending store paths agree on `input_already_rotated` for their actual source tensors. Current pending path correctly uses `kvarn_paper_frame_enabled()` because pending comes from rotated sink/tail in paper-frame mode.

## Areas That Look Internally Consistent

- Pending K transpose now writes channel-major K tiles: `k_tile[d*group_size + g] = pending[d + g*pending_head_stride]`.
- Pending V gather writes token-major V tiles, which matches the V store path.
- Packed K layout is consistently dim-major: `d*group_size + g`.
- Packed V layout is consistently token-major: `g*head_dim + d`.
- K scales use `[s_col, zp, s_row]`; V scales use `[s_col, s_row, zp]`.
- GQA maps query heads to KV heads through `ikh = ih / n_gqa` consistently across the reviewed CUDA mixed-attention paths.
- Normal multi-token tail ring order uses `(tail_start + t) % n_tail`.
- Sampled full-Q/O boundary replay from Round 35 showed mixed-attention self-replay near numerical identity and f16-truth replay small for sampled layer/head cases. That narrows the remaining Qwen ctx4096 failure but does not clear graph/block-level correctness.

## Remaining Oracle Gaps

Current boundary tooling still does not prove the full transformer path:

- It does not dump pre-Hadamard and post-Hadamard Q side by side.
- It does not dump sink/tail K/V before and after graph Hadamard copy.
- It does not dump pending-to-body source tensors per record for all store paths.
- It does not dump post-unrotation attention output and compare it to an independent f16 attention result.
- It does not dump post-WO attention output or residual block output.
- It samples selected layer/head/query boundaries, not every KV head and not every scheduled KVarN layer.

Required next oracle patch:

1. Add post-unrotation and post-WO/block-output dumps.
2. Add a high-KL token replay driver that dumps the exact layer/head/query rows responsible for the worst logits/KL drift.
3. Add a full-layer option that dumps both KV heads for Qwen and all GQA query groups.
4. Add an independent reference for multi-record pending store and direct-record store that does not call the same CUDA helper being tested.

## Recommended Next Patch Order

1. Fix validation harnesses:
   - build all KVarN tests before production gate ctest,
   - enforce active body records for production KVarN gates,
   - make expected-layer checks exact by default,
   - fix normal baseline argument parity.

2. Fix the confirmed multi-record pending duplication:
   - simplest safe patch: disable batching for pending-record stores and emit one op per record,
   - then reintroduce batching only with per-record pending offsets and independent tests.

3. Make paper-frame the only allowed true-KVarN quality path:
   - require `LLAMA_KVARN_ENABLE_PAPER_FRAME=1` in true-KVarN accuracy gates,
   - record the flag in all benchmark summaries,
   - block non-paper-frame body-record runs or label them diagnostic only.

4. Implement layer-subset fallback:
   - allow included layers to use KVarN and excluded full-attention layers to use normal KV,
   - cover Qwen hybrid and MTP paths,
   - disable graph reuse for mixed diagnostic graphs until graph topology is keyed safely.

5. Add post-mixed-attention graph oracles:
   - prove `out_unrot == H*out_rot`,
   - prove post-WO and residual outputs against f16 for the worst ctx4096 token,
   - only then treat a remaining K8V8 PPL gap as quantization/model behavior.

6. Disable or implement unsupported production surfaces:
   - context shifting,
   - full state save/load,
   - arbitrary middle `seq_rm()`,
   - standalone MTP graph routing/scaling.

## Current Bottom Line

The branch is not ready to claim production quality. The line-by-line audit found at least one concrete KVarN cache corruption bug (`n_record_batch > 1` pending-store duplication), several harness holes that can produce false green gates, and multiple production-surface gaps around shifting/state/removal.

For the immediate Qwen3.6 ctx4096 K8V8 `+11%` PPL issue, the latest sampled full-Q/O replay suggests the mixed-attention CUDA kernel is probably not the primary error at sampled heads. The next highest-value work is to fix the validation harness, remove the confirmed pending-store corruption path, and add post-unrotation/post-WO/block-output oracles so the remaining error can no longer hide behind self-consistent KVarN-only comparisons.
